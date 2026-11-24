"""One formula for the grid shape and a few coefficients to go with it.

The design implemented here: next to the scale a short vector of numbers is stored, and the
grid shape is restored from it by a single formula. The coefficients must switch the mode
continuously: dense near zero (the mu-law branch: max far above most values), dense near
absmax (the reverse branch: values pressed to the maximum), dense near both scales (mass
at both edges), plus a left/right skew.

We build the levels as quantiles of a flexible magnitude density on [0, 1] — the
Kumaraswamy distribution, whose quantile function is closed-form (unlike the beta
distribution, where we would have to carry an incomplete beta):

    F(x) = 1 - (1 - x**a)**b          F⁻¹(u) = (1 - (1 - u)**(1/b))**(1/a)
    f(x) = a*b*x**(a-1) * (1 - x**a)**(b-1)

Hence the modes (this is exactly "one formula, different coefficients"):

    a < 1, b = 1   f decreasing     mass at zero       -> levels dense near zero   (mu-law)
    a > 1, b = 1   f increasing     mass at absmax     -> levels dense near edge   (reverse mu)
    a < 1, b < 1   f U-shaped       mass at both edges -> dense at zero AND at edge
    a > 1, b > 1   hump in middle   -> levels dense at mid magnitudes

The sign sides get their own pairs (a_neg, b_neg) and (a_pos, b_pos): that is the skew,
i.e. four numbers cover all the listed cases. Ground from companding theory: the optimal
level density λ(x) ∝ p(x)^(1/(q+1)), i.e. the quantile construction is not a guess but
exactly the asymptotically correct answer.

Metadata format: 4 coefficients per layer (negligible cost, divided over the whole tensor)
or 4 coefficients per group (2x the cost of a 32-bit scale). A free codebook of L levels
costs 32·L bits PER GROUP — we keep it as the upper bound of expressiveness, not as a product.
"""
from __future__ import annotations

import numpy as np

from .grids import make_grid
from .lloyd import assign, refit_frozen, weighted_lloyd

PARAMS_PER_GRID = 4
_EPS = 1e-12


def _kum_ppf(u: np.ndarray, a: float, b: float) -> np.ndarray:
    """Kumaraswamy quantile function on [0,1]."""
    return np.power(1.0 - np.power(1.0 - u, 1.0 / b), 1.0 / a)


def _check_theta(theta) -> tuple[float, float, float, float]:
    out = tuple(float(t) for t in theta)
    if len(out) != PARAMS_PER_GRID:
        raise ValueError(f"need {PARAMS_PER_GRID} coefficients, got {len(out)}")
    if not all(np.isfinite(out)) or not all(t > 0 for t in out):
        raise ValueError(f"coefficients must be positive and finite: {theta}")
    return out  # type: ignore[return-value]


def shapeq_grid(L: int, theta) -> np.ndarray:
    """Grid of L levels from four coefficients (a_neg, b_neg, a_pos, b_pos).

    Level i is the quantile of the spliced sign density at mass share (i+0.5)/L. The zero
    level appears by itself when L is odd (the quantile at exactly 0.5 is zero), so the
    uniform case stays strictly uniform, without a hammered-in step near zero.
    Each side is normalized to its own absmax, since the group scale = absmax.
    """
    a_neg, b_neg, a_pos, b_pos = _check_theta(theta)
    if L < 2:
        raise ValueError("need at least two levels")
    q = (np.arange(L) + 0.5) / L                      # mass shares, (0,1)
    neg = q < 0.5
    mag = np.empty(L, dtype=np.float64)
    # left of zero: mass accumulated downward from -1, this is the tail of the magnitude density
    mag[neg] = _kum_ppf(1.0 - 2.0 * q[neg], a_neg, b_neg)
    mag[~neg] = _kum_ppf(2.0 * q[~neg] - 1.0, a_pos, b_pos)
    if L % 2 == 1:
        mag[L // 2] = 0.0                              # quantile exactly at the mass midpoint
    grid = np.where(neg, -mag, mag)

    pos = grid >= 0.0
    for mask in (pos, ~pos):
        edge = np.max(np.abs(grid[mask]))
        if edge <= _EPS:
            raise ValueError("a grid side degenerated: increase L or use gentler coefficients")
        grid[mask] = grid[mask] / edge
    grid = np.sort(grid)
    grid[0], grid[-1] = -1.0, 1.0                      # exact bounds after normalization
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid is not strictly increasing — coefficients are too extreme")
    return grid


def cost(y: np.ndarray, w: np.ndarray, grid: np.ndarray, idx=None) -> float:
    """Quantizer objective with the scales already in place: Σ w·(y − G[level])².

    idx=None — the level is chosen by nearest (ordinary decoding). idx given — assignments
    are frozen: the case "a ready artifact exists (GSQ/RCO: indices + absmax scales), we
    change only the table". Then the error is honest precisely for that deployment, where
    the weights are not re-encoded.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    if idx is None:
        idx = assign(y, grid)
    d = y - grid[np.asarray(idx).ravel()]
    return float(np.dot(w, d * d))


def _expand(theta, two_sided: bool) -> tuple[float, float, float, float]:
    theta = tuple(float(t) for t in theta)
    if two_sided:
        return _check_theta(theta)
    if len(theta) != 2:
        raise ValueError("symmetric mode accepts 2 coefficients")
    a, b = theta[0], theta[1]
    return _check_theta((a, b, a, b))


def eqmass_grid(y: np.ndarray, w: np.ndarray, L: int,
                equal: str = "count", kappa: bool = False) -> np.ndarray:
    """The user's formulation as an algorithm: uniformity is the goal, closeness is a by-product.

    Unlike fit_warp (where uniform filling of the bins comes as a side effect of minimizing the
    weighted error), here uniformity is the FIRST and rigid goal, and exactly where the decoder
    will see it: assignment goes by the nearest level (bin boundaries are the midpoints between
    neighboring levels), so we REQUIRE the midpoints between levels to stand at the empirical
    weight quantiles q_j:

        (l_j + l_{j+1}) / 2 = q_j.                                  (*)

    System (*) is solved exactly: l_j = (−1)^j·t + s_j with the recurrence
    s_{j+1} = 2q_j − s_j, and the single free parameter t is taken at the exact
    minimum of the objective Σ w·(y − level)² (quadratic in t — closed formula, no
    iterations). Result:

      goal A — "spread the values evenly over the whole ladder": satisfied
        constructively at decode time (a bin = exactly its own share of the mass);
      goal B — "incidentally place values closer to a new nearest step": optimal
        UNDER the linked goal A. Naive centroids of quantile bins do not work for
        this: their Voronoi midpoints slide off the quantiles, and uniformity falls
        apart before decoding (measured: gaussian bins 0 / 1.7 / 14.7 / 33.7 %);
      the shift is a tool: a weight is allowed to move to another bin, the
        boundaries being cut by that same weight's distribution.

    equal="count" — bins equal by weight count (literally "uniformly dense");
    equal="mass" — equal by importance (Σw within a bin is equal).
    kappa=True — a soft version: bin boundaries are scaled by a multiplier fitted on the
    common objective ("uniformity nearly ironclad" — to honestly measure the price of
    rigidity); the default False is the rigid version, and the outer fit_scale is NOT
    applied to such a grid: it would slide the midpoints off the quantiles.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    if y.shape != w.shape:
        raise ValueError("y and w must have matching shapes")
    if L < 2:
        raise ValueError("need at least two levels")
    if equal not in ("count", "mass"):
        raise ValueError("equal is one of count | mass")
    o = np.argsort(y, kind="mergesort")
    ys, ws = y[o], np.maximum(w[o], 0.0)
    n = len(ys)
    cw = np.concatenate(([0.0], np.cumsum(ws)))
    cs = np.concatenate(([0.0], np.cumsum(ws * ys)))
    cq = np.concatenate(([0.0], np.cumsum(ws * ys * ys)))
    if equal == "mass":
        if cw[-1] <= 0:
            raise ValueError("zero importance: eqmass by mass is undefined")
        cuts = np.searchsorted(cw, np.linspace(0.0, cw[-1], L + 1)[1:-1], side="left")
    else:
        cuts = np.rint(np.linspace(0, n, L + 1, dtype=np.float64)[1:-1]).astype(np.int64)
    # Voronoi bin boundaries: midpoint between the last weight of a bin and the first of the next
    qb = (ys[np.maximum(cuts - 1, 0)] + ys[np.minimum(cuts, n - 1)]) / 2.0
    qb = np.maximum.accumulate(qb)          # with mass ties the boundary does not run backwards

    sgn = np.where(np.arange(L) % 2 == 0, 1.0, -1.0)

    def solve(kf: float) -> tuple[np.ndarray, float]:
        thr = qb * kf
        s = np.zeros(L)
        for j in range(1, L):
            s[j] = 2.0 * thr[j - 1] - s[j - 1]

        def agg(bnd):
            return (cw[bnd[1:]] - cw[bnd[:-1]], cs[bnd[1:]] - cs[bnd[:-1]],
                    cq[bnd[1:]] - cq[bnd[:-1]])

        bnd0 = np.searchsorted(ys, thr, side="right")
        bnd0 = np.concatenate(([0], bnd0, [n]))
        W, S, Q = agg(bnd0)
        t = float(np.sum(sgn * (S - W * s)) / max(float(W.sum()), 1e-300))
        # Monotonicity of the ladder = each level must stand in its own cell: that is exactly
        # the condition sgn_j·t < d_j at every joint; even j give the upper bound on t,
        # odd j the lower one. Otherwise a ladder with midpoints at quantiles does not exist.
        d = thr[:L - 1] - s[:L - 1]
        t_lo = float(np.max(-d[1::2])) if L > 2 else -np.inf
        t_hi = float(np.min(d[0::2]))
        if t_lo < t_hi:
            t = float(np.clip(t, t_lo, t_hi))     # clipping — goal A in full
        else:
            # The quantile intervals are too uneven (heavy tails): no strictly monotone
            # ladder with midpoints exactly on the quantiles exists. Take t with the
            # minimal total violation and patch pinches locally: the spread is symmetric
            # about the joint, so the joint's boundary does not slide off.
            def viol(x: float) -> float:
                return float(np.sum(np.maximum(0.0, sgn[:L - 1] * x - d) ** 2))

            c, r = 0.5 * (t_lo + t_hi), max(t_hi - t_lo, 1e-12)
            a_, b_ = c - 2.0 * r, c + 2.0 * r
            for _ in range(80):                   # golden section: piecewise-convex
                m1, m2 = b_ - 0.618 * (b_ - a_), a_ + 0.618 * (b_ - a_)
                if viol(m1) <= viol(m2):
                    b_ = m2
                else:
                    a_ = m1
            t = 0.5 * (a_ + b_)
        lev = sgn * t + s
        for _ in range(2):                        # local repair of pinches (ties)
            fixed = True
            for j in range(L - 1):
                if lev[j + 1] <= lev[j]:
                    delta = 0.5 * (lev[j] - lev[j + 1]) + 1e-12
                    lev[j] -= delta
                    lev[j + 1] += delta
                    fixed = False
            if fixed:
                break
        bnd = np.searchsorted(ys, 0.5 * (lev[:-1] + lev[1:]), side="right")
        bnd = np.concatenate(([0], bnd, [n]))     # the objective is measured the way the decoder decides
        W, S, Q = agg(bnd)
        err = float(np.sum(Q - 2.0 * lev * S + lev * lev * W))
        return lev, err

    lev, err0 = solve(1.0)
    if kappa:                               # relax: uniformity becomes soft
        best_k, best_lev, best_e = 1.0, lev, err0
        step = 0.25
        while step > 4e-3:
            moved = False
            for mult in (1.0 + step, 1.0 / (1.0 + step)):
                kf = float(np.clip(best_k * mult, 0.40, 1.25))
                cand, err = solve(kf)
                if err < best_e - 1e-18:
                    best_k, best_lev, best_e, moved = kf, cand, err, True
                    break
            if not moved:
                step *= 0.6
        lev = best_lev
    m = float(np.max(np.abs(lev)))
    if m <= _EPS:
        raise ValueError("degenerated grid: are all weights zero?")
    # There is NO normalization to max|·|=1 HERE, unlike the other grids of the family:
    # the levels are already in units of the group absmax (y from group_layout), and any
    # scaling would slide the Voronoi midpoints off the quantiles — goal A rests exactly
    # on the decoder seeing the same boundaries as the fit.
    return lev


def fit_scale(y: np.ndarray, w: np.ndarray, grid: np.ndarray,
              lo: float = 0.40, hi: float = 1.25, rel_step: float = 0.25,
              shrink: float = 0.6, min_step: float = 0.004,
              idx=None) -> tuple[float, float]:
    """Fit a step multiplier κ to a ready-made grid.

    The group scale = absmax, so at κ=1 the top grid level is nailed to the farthest weight
    and spends a level on single outliers. κ<1 means "clip the tails", κ>1 — "keep
    headroom". This is a separate lever that has nothing to do with shape: it must be given
    to all ladder steps equally, otherwise comparing shapes against a free codebook is
    unfair (the free codebook pulls its own top level inward).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    best_k, best = 1.0, cost(y, w, grid, idx)
    step = rel_step
    while step > min_step:
        improved = False
        for mult in (1.0 + step, 1.0 / (1.0 + step)):
            cand = float(np.clip(best_k * mult, lo, hi))
            value = cost(y, w, grid * cand, idx)
            if value < best - 1e-15:
                best_k, best, improved = cand, value, True
                break
        if not improved:
            step *= shrink
    return best_k, best


def _safe_cost(y: np.ndarray, w: np.ndarray, L: int, theta, two_sided: bool, idx=None) -> float:
    """Objective for a candidate: a degenerated grid (levels merging due to quantile saturation)
    is not an exception but just a very bad point. The side-by-side fitter must survive this."""
    try:
        grid = shapeq_grid(L, _expand(theta, two_sided))
    except ValueError:
        return float("inf")
    return cost(y, w, grid, idx)


def fit_warp(y: np.ndarray, w: np.ndarray, L: int, two_sided: bool = True,
             seeds=None, rel_step: float = 0.6, shrink: float = 0.55,
             min_step: float = 0.02, lo: float = 0.06, hi: float = 12.0,
             max_evals: int = 400, idx=None) -> tuple[tuple[float, float, float, float], float]:
    """Fit the shape coefficients by the same objective as everything else.

    The grid is nonlinear and non-convex in the coefficients, so we search with a swarm of
    starts + compass search in log-space: deterministic, no gradients and no scipy. Each
    point is evaluated by the exact objective, and degenerate coefficients give +inf, so the
    result is never worse than the uniform start — uniform is a=b=1.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    if y.shape != w.shape:
        raise ValueError("y and w must have matching shapes")
    if seeds is None:
        seeds = ((1.0, 1.0, 1.0, 1.0), (0.4, 1.0, 0.4, 1.0), (2.5, 1.0, 2.5, 1.0),
                 (0.4, 0.4, 0.4, 0.4), (1.7, 1.7, 1.7, 1.7))
    n_free = 4 if two_sided else 2

    init = (1.0, 1.0, 1.0, 1.0) if two_sided else (1.0, 1.0)
    best_theta, best_value = _expand(init, two_sided), float("inf")
    evals = 0
    for seed in seeds:
        theta = np.asarray(seed[:n_free], dtype=np.float64)
        value = _safe_cost(y, w, L, theta, two_sided, idx)
        evals += 1
        step = rel_step
        while step > min_step and evals < max_evals:
            improved = False
            for k in range(n_free):
                for mult in (1.0 + step, 1.0 / (1.0 + step)):
                    cand = theta.copy()
                    cand[k] = float(np.clip(cand[k] * mult, lo, hi))
                    cv = _safe_cost(y, w, L, cand, two_sided, idx)
                    evals += 1
                    if cv < value - 1e-15:
                        theta, value, improved = cand, cv, True
                        break
            if not improved:
                step *= shrink
        if value < best_value:
            best_value, best_theta = value, _expand(theta, two_sided)
    return best_theta, best_value


def group_layout(W: np.ndarray, Hdiag: np.ndarray, group: int):
    R, C = W.shape
    if C % group:
        raise ValueError(f"C={C} is not divisible by group={group}")
    Hdiag = np.asarray(Hdiag, dtype=np.float64).ravel()
    if Hdiag.shape != (C,):
        raise ValueError(f"Hdiag must have length C={C}, got {Hdiag.shape}")
    blocks = W.reshape(R, C // group, group)
    scales = np.max(np.abs(blocks), axis=2)
    scales = np.where(scales > 0, scales, 1.0)
    y = blocks / scales[:, :, None]
    w = (np.repeat(scales ** 2, group, axis=1) * Hdiag[None, :]).reshape(R, C // group, group)
    return y, w


def shape_ladder(W: np.ndarray, Hdiag: np.ndarray, group: int, L: int,
                 two_sided: bool = True, fit_rows: int | None = 1024,
                 lloyd_iters: int = 30, kappa: bool = True) -> dict[str, object]:
    """Expressiveness ladder of grid shapes under identical absmax scales.

    All steps are measured by one relative layer metric
    tr(ΔHΔᵀ)/tr(WHWᵀ), so they are directly comparable:

      uniform     — 0 coefficients (baseline)
      nf4         — 0 coefficients, but the shape is picked "by eye" for a gaussian
      kum_layer   — 4 coefficients per layer (metadata ~0 bits/weight)
      kum_group   — 4 coefficients per group (≈64 bits per group at fp16)
      lloyd_layer — free table of L levels per layer (32·L bits per layer)
      lloyd_group — free table of L levels per group (32·L bits PER GROUP, oracle)

    With kappa=True each step gets an extra row `<step>_kappa`: the same codebook, but with a
    fitted step multiplier (tail clipping). This is a separate lever, and it must be handed to
    all steps equally — otherwise the free codebook wins merely because it pulls its own top
    level inward, while a grid with a nailed absmax is denied that freedom.

    fit_rows limits the number of rows the coefficients are fitted on (every stride-th row —
    deterministically); the evaluation always uses all elements.
    """
    W = np.asarray(W, dtype=np.float64)
    y, w = group_layout(W, Hdiag, group)
    denom = float(np.sum(w * y * y))
    if denom <= 0:
        raise ValueError("zero normalization: the layer must be nonzero")

    def rel(grid: np.ndarray) -> float:
        return cost(y.ravel(), w.ravel(), grid) / denom

    yv, wv = y.ravel(), w.ravel()
    stride = int(np.ceil(W.shape[0] / fit_rows)) if fit_rows else 1
    y_fit, w_fit = (yv, wv) if stride == 1 else (y[::stride].ravel(), w[::stride].ravel())

    out: dict[str, object] = {}
    grids: dict[str, np.ndarray] = {"uniform": make_grid("uniform", L)}
    try:
        grids["nf4"] = make_grid("nf4", L)
    except ValueError:
        pass

    theta, _ = fit_warp(y_fit, w_fit, L, two_sided=two_sided)
    grids["kum_layer"] = shapeq_grid(L, theta)
    out["theta_layer"] = theta
    grids["lloyd_layer"] = weighted_lloyd(yv, wv, L, init=make_grid("uniform", L),
                                          iters=lloyd_iters, symmetrize=True)

    # The _frozen rows are the scenario "take a ready artifact and change only the table":
    # the uniform grid's indices are frozen, no re-encoding.
    idx_fit = assign(y_fit, grids["uniform"])
    idx_full = assign(yv, grids["uniform"])
    theta_f, _ = fit_warp(y_fit, w_fit, L, two_sided=two_sided, idx=idx_fit)
    out["kum_layer_frozen"] = cost(yv, wv, shapeq_grid(L, theta_f), idx_full) / denom
    out["lloyd_layer_frozen"] = cost(yv, wv, refit_frozen(yv, wv, idx_full, L), idx_full) / denom

    def add(name: str, grid: np.ndarray) -> None:
        out[name] = rel(grid)
        if kappa:
            k, value = fit_scale(yv, wv, grid)
            out[name + "_kappa"] = value / denom
            out["kappa_" + name] = k

    for name, grid in grids.items():
        add(name, grid)

    def per_group(name: str, builder) -> None:
        """Its own grid (and its own step) per group — the upper bound of expressiveness."""
        plain = tuned = 0.0
        for g in range(y.shape[1]):
            grid = builder(y[::stride, g].ravel(), w[::stride, g].ravel())
            yg, wg = y[:, g].ravel(), w[:, g].ravel()
            plain += cost(yg, wg, grid)
            tuned += fit_scale(yg, wg, grid)[1] if kappa else cost(yg, wg, grid)
        out[name] = plain / denom
        if kappa:
            out[name + "_kappa"] = tuned / denom

    per_group("kum_group", lambda yf, wf: shapeq_grid(
        L, fit_warp(yf, wf, L, two_sided=two_sided)[0]))
    per_group("lloyd_group", lambda yf, wf: weighted_lloyd(
        yf, wf, L, init=make_grid("uniform", L), iters=lloyd_iters, symmetrize=True))
    return out
