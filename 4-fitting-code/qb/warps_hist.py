"""Histogram engine for grid fitting: an objective equivalent to cost(), but over a weighted
histogram instead of a subsample of weights.

Why (see the "fitting optimizations" plan, 19.11):
  1. Speed. cost() over a 150k subsample costs ~150k·logL operations per grid fit;
     the histogram is built once (O(n)), then each fit is O(nbins·logL),
     which at nbins=8192 is tens of times faster.
  2. Determinism. fit_warp greedily turns 4 dials over a random subsample (rng from crc32
     of the tensor): a nightly fp16-GPU/fp32 cross-run gave a b4 divergence of 1.2 ppl
     exactly because of it. The histogram is an exact function of the tensor: two runs =
     bit-for-bit the same answer.
  3. Initialization instead of 5 starts: fitting (a,b) per side by direct minimization
     over a coarse grid on that same cheap objective — compass search stays as polishing.

Accuracy: the error is a piecewise-constant function of y; grid steps do not "breathe"
with the bin borders; the bias is (bin_width)²/12 per unit of mass — at nbins=8192 that
is ~5e-9, immaterial for typical y∈[−1,1] and L≤256. Checked by probes/histfit_check.py
against the exact objective.

Shares nothing with fit_warp except shapeq_grid: the legacy path is left alone.
"""
from __future__ import annotations

import numpy as np

from .warps import _EPS, _expand, _safe_cost, shapeq_grid

DEFAULT_NBINS = 8192
_LOG_LO, _LOG_HI = np.log(0.06), np.log(12.0)   # the same coefficient bounds as fit_warp


def hist_weighted(y: np.ndarray, w: np.ndarray, nbins: int = DEFAULT_NBINS):
    """Weighted histogram of y on [-1, 1] (y is already normalized by the group's absmax)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    lo = min(float(y.min()), -1.0)
    hi = max(float(y.max()), 1.0)
    edges = np.linspace(lo, hi, nbins + 1)
    mass, _ = np.histogram(y, bins=edges, weights=w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = mass > 0
    return centers[keep], mass[keep]


def cost_hist(centers: np.ndarray, mass: np.ndarray, levels: np.ndarray) -> float:
    """Σ_b mass_b·(c_b − nearest(c_b))² — exactly cost() on the histogram's quanta."""
    thr = 0.5 * (levels[:-1] + levels[1:])
    idx = np.searchsorted(thr, centers, side="right")
    d = centers - levels[idx]
    return float(np.dot(mass, d * d))


def _safe_grid_hist(L, theta, two_sided):
    try:
        return shapeq_grid(L, _expand(theta, two_sided))
    except ValueError:
        return None


def _mle_init(centers, mass, L, two_sided, grid_n=17):
    """Start (a,b) per side: minimum of the histogram objective over a coarse log-coefficient grid.

    This is the "EM wave" in its cheap form: not iterative EM over mixtures, but direct
    matching of the mass distribution's shape to Kumaraswamy quantiles (one pass,
    ~grid_n²·2 fits at μs each). The compass then refines.
    """
    a_grid = np.exp(np.linspace(_LOG_LO, _LOG_HI, grid_n))
    best = {}
    pos_m = centers >= 0.0
    for side in ("pos", "neg"):
        c_s = centers[pos_m] if side == "pos" else -centers[~pos_m]
        m_s = mass[pos_m] if side == "pos" else mass[~pos_m]
        if c_s.size == 0:
            best[side] = (1.0, 1.0)
            continue
        ba, bb, best_c = 1.0, 1.0, np.inf
        for a in a_grid:
            for b in a_grid:
                theta = (a, b, 1.0, 1.0) if side == "neg" else (1.0, 1.0, a, b)
                g = _safe_grid_hist(L, theta, True)
                if g is None:
                    continue
                half = np.sort(g[g >= 0.0] if side == "pos" else -g[g <= 0.0])
                if half.size < 2 or half[0] > c_s.min() * 1.5:
                    continue
                c = cost_hist(c_s, m_s, half)
                if c < best_c:
                    ba, bb, best_c = a, b, c
        best[side] = (ba, bb)
    a_neg, b_neg = best["neg"]
    a_pos, b_pos = best["pos"]
    return (a_neg, b_neg, a_pos, b_pos)


def _compass_hist(centers, mass, L, two_sided: bool = True,
                  rel_step: float = 0.6, shrink: float = 0.55, min_step: float = 0.02,
                  max_evals: int = 400, mle_init: bool = True):
    """Compass search over the histogram objective (does not build the histogram — takes a ready one)."""
    n_free = 4 if two_sided else 2
    objective = lambda th: (lambda g: float("inf") if g is None else cost_hist(centers, mass, g))(
        _safe_grid_hist(L, th, two_sided))

    inits = [(1.0, 1.0, 1.0, 1.0), (0.4, 1.0, 0.4, 1.0), (2.5, 1.0, 2.5, 1.0)]
    if mle_init:
        inits.insert(0, _mle_init(centers, mass, L, two_sided))

    evals = 0
    best_theta, best_value = None, np.inf
    for seed in inits:
        theta = np.asarray(seed[:n_free] if n_free == 2 else seed, dtype=np.float64)
        value = objective(theta)
        evals += 1
        step = rel_step
        while step > min_step and evals < max_evals:
            improved = False
            for k in range(n_free):
                for mult in (1.0 + step, 1.0 / (1.0 + step)):
                    cand = theta.copy()
                    cand[k] = float(np.clip(cand[k] * mult, np.exp(_LOG_LO), np.exp(_LOG_HI)))
                    cv = objective(cand)
                    evals += 1
                    if cv < value - 1e-15:
                        theta, value, improved = cand, cv, True
                        break
            if not improved:
                step *= shrink
        if value < best_value:
            best_theta, best_value = theta, value
    theta = tuple(best_theta[:n_free]) if n_free == 2 else tuple(best_theta)
    if not two_sided:
        theta = (theta[0], theta[1])
    return _expand(theta, two_sided), best_value


def _scale_compass_hist(centers, mass, grid, lo: float = 0.40, hi: float = 1.25,
                        rel_step: float = 0.25, shrink: float = 0.6, min_step: float = 0.004):
    """Scale compass over a ready histogram."""
    grid = np.asarray(grid, dtype=np.float64)
    best_k, best = 1.0, cost_hist(centers, mass, grid)
    step = rel_step
    while step > min_step:
        improved = False
        for mult in (1.0 + step, 1.0 / (1.0 + step)):
            cand = float(np.clip(best_k * mult, lo, hi))
            value = cost_hist(centers, mass, grid * cand)
            if value < best - 1e-15:
                best_k, best, improved = cand, value, True
                break
        if not improved:
            step *= shrink
    return best_k, best


def fit_warp_hist(y, w, L, two_sided: bool = True, nbins: int = DEFAULT_NBINS,
                  rel_step: float = 0.6, shrink: float = 0.55, min_step: float = 0.02,
                  max_evals: int = 400, mle_init: bool = True):
    """fit_warp over the histogram objective; the contract is the same: (theta, value).

    Starts: uniform start + (optionally) MLE start + legacy seeds (cheap on the histogram,
    we take only the two extreme ones). The compass is word-for-word like fit_warp, only the objective differs.
    """
    centers, mass = hist_weighted(y, w, nbins)
    return _compass_hist(centers, mass, L, two_sided, rel_step, shrink, min_step,
                         max_evals, mle_init)


def fit_scale_hist(y, w, grid, lo: float = 0.40, hi: float = 1.25,
                   rel_step: float = 0.25, shrink: float = 0.6, min_step: float = 0.004,
                   nbins: int = DEFAULT_NBINS):
    """fit_scale over the histogram; the contract is the same: (k, cost)."""
    centers, mass = hist_weighted(y, w, nbins)
    return _scale_compass_hist(centers, mass, grid, lo, hi, rel_step, shrink, min_step)


def fit_tensor_hist(W, Hdiag, L, group, nbins: int = DEFAULT_NBINS):
    """The histogram equivalent of fit_tensor_grid(kind=shapeq): one group_layout pass,
    ONE histogram for fitting both the shape and κ; contract: (theta, k), grid = shapeq_grid(L, theta)·k.

    Peak memory — y,w only while building the histogram (8192 bins); afterwards nothing
    is held over the tensor: important for RAM-tight runs.
    """
    from .warps import group_layout
    y, w = group_layout(np.asarray(W, dtype=np.float64),
                        np.asarray(Hdiag, dtype=np.float64), group)
    centers, mass = hist_weighted(y, w, nbins)
    del y, w
    theta, _ = _compass_hist(centers, mass, L)
    k, _ = _scale_compass_hist(centers, mass, shapeq_grid(L, theta))
    return theta, k


def lloyd_hist(y, w, L: int, iters: int = 30, nbins: int = DEFAULT_NBINS,
               refine_iters: int = 40, refine_n: int = 600_000):
    """Free Lloyd-Max codebook in the production class of weighted_lloyd: symmetric grid
    without zero-point, optimization over magnitudes. The magnitude histogram gives the
    global structure; refinement on raw |y| resolves dense clusters near zero."""
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    ay = np.abs(y)
    P = L // 2                                        # even L: magnitudes only
    edges = np.linspace(0.0, float(ay.max()), nbins + 1)
    mass, edges = np.histogram(ay, bins=edges, weights=w)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    keep = mass > 0
    ctr, m = ctr[keep], mass[keep]
    cdf = np.cumsum(m) / m.sum()
    mags = np.interp((np.arange(P) + 0.5) / P, cdf, ctr)
    prev = np.inf
    for _ in range(iters):
        thr = 0.5 * (mags[:-1] + mags[1:])
        idx = np.searchsorted(thr, ctr, side="right")
        tot = np.bincount(idx, weights=m, minlength=P)
        num = np.bincount(idx, weights=m * ctr, minlength=P)
        new = mags.copy()
        ok = tot > 0
        new[ok] = num[ok] / tot[ok]
        mags = np.sort(new)
        thr = 0.5 * (mags[:-1] + mags[1:])
        idx = np.searchsorted(thr, ctr, side="right")
        d = ctr - mags[idx]
        value = float(np.dot(m, d * d))
        if prev - value < 1e-15 * max(prev, 1.0):
            break
        prev = value
    if ay.size > refine_n:
        sel = np.linspace(0, ay.size, refine_n, endpoint=False).astype(np.int64)
        ay_s, w_s = ay[sel], w[sel]
    else:
        ay_s, w_s = ay, w
    prev = np.inf
    for _ in range(refine_iters):
        thr = 0.5 * (mags[:-1] + mags[1:])
        idx = np.searchsorted(thr, ay_s, side="right")
        tot = np.bincount(idx, weights=w_s, minlength=P)
        num = np.bincount(idx, weights=w_s * ay_s, minlength=P)
        new = mags.copy()
        ok = tot > 0
        new[ok] = num[ok] / tot[ok]
        mags = np.sort(new)
        d = ay_s - mags[np.searchsorted(0.5 * (mags[:-1] + mags[1:]), ay_s, "right")]
        value = float(np.dot(w_s, d * d))
        if prev - value < 1e-12 * max(prev, 1.0):
            break
        prev = value
    levels = np.sort(np.concatenate([-mags[::-1], mags]))
    d = y - levels[np.searchsorted(0.5 * (levels[:-1] + levels[1:]), y, "right")]
    return levels, float(np.dot(w, d * d))


def _init_quantiles(centers, mass, L):
    cdf = np.cumsum(mass) / mass.sum()
    q = (np.arange(L) + 0.5) / L
    return np.interp(q, cdf, centers)
