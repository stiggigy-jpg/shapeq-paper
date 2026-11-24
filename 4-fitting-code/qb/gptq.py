"""Quantizing a linear layer: Hessian, RTN, GPTQ, refitting the grid shape under the Hessian metric.

Conventions:
  W — (R, C): R output channels, C input channels; groups taken along dim=1 (input channels).
  X — (N, C) layer calibration inputs; H = XᵀX / N.
  The grid is normalized to a maximum of 1, group scale = absmax, symmetric quantizer
  without zero-point (as in GSQ).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grids import make_grid
from .lloyd import assign, weighted_lloyd


@dataclass(frozen=True)
class QuantResult:
    idx: np.ndarray      # (R, C) integer level indices
    scales: np.ndarray   # (R, C // group) absmax scales
    grid: np.ndarray     # (L,) normalized grid
    group: int


def hessian(x: np.ndarray, damp: float = 0.0) -> np.ndarray:
    """H = XᵀX / N. Damping is applied inside GPTQ so that this metric stays honest."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be a matrix (N, C)")
    H = x.T @ x / max(x.shape[0], 1)
    if damp:
        base = np.mean(np.diag(H))
        H = H + np.eye(H.shape[0]) * (damp * base if base > 0 else damp)
    return H


def _inverse_cholesky_upper(H: np.ndarray, percdamp: float) -> np.ndarray:
    """U upper-triangular with inv(H_damped) = UᵀU — the correction direction in sequential rounding."""
    dim = H.shape[0]
    base = float(np.mean(np.diag(H)))
    jitter = percdamp if base <= 0 else percdamp * base
    for attempt in range(8):
        try:
            Hinv = np.linalg.inv(H + np.eye(dim) * jitter)
            Hinv = 0.5 * (Hinv + Hinv.T)
            return np.linalg.cholesky(Hinv).T
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("Hessian is singular even with increased damp")


def _nearest(w: np.ndarray, scales: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest level for each value under its row's own scale. Deterministic."""
    best_d = None
    best_i = None
    for level in range(grid.size):
        d = np.abs(w - scales * grid[level])
        if best_d is None:
            best_d, best_i = d, np.zeros(d.shape, dtype=np.uint8)
        else:
            take = d < best_d
            best_d = np.where(take, d, best_d)
            best_i = np.where(take, level, best_i)
    return best_i, scales * grid[best_i]


def _check_shape(W: np.ndarray, group: int) -> tuple[int, int]:
    if W.ndim != 2:
        raise ValueError("W must be a matrix (R, C)")
    if W.shape[1] % group:
        raise ValueError(f"C={W.shape[1]} is not divisible by group={group}")
    return W.shape


def _group_absmax(W: np.ndarray, group: int) -> np.ndarray:
    R, C = W.shape
    blocks = W.reshape(R, C // group, group)
    scales = np.max(np.abs(blocks), axis=2)
    return np.where(scales > 0, scales, 1.0)


def rtn(W: np.ndarray, Hdiag: np.ndarray, grid: np.ndarray, group: int) -> QuantResult:
    """Nearest level without error compensation. Hdiag is unused here, but the signature is kept uniform."""
    W = np.asarray(W, dtype=np.float64)
    _check_shape(W, group)
    grid = np.asarray(grid, dtype=np.float64)
    scales = _group_absmax(W, group)
    y = (W.reshape(W.shape[0], -1, group) / scales[:, :, None]).ravel()
    idx = assign(y, grid).reshape(W.shape).astype(np.uint8)
    return QuantResult(idx, scales, grid, group)


def gptq(W: np.ndarray, H: np.ndarray, grid: np.ndarray, group: int,
         blocksize: int = 128, percdamp: float = 0.01) -> QuantResult:
    """Sequential rounding with correction through the Hessian factor (GPTQ/Cholesky)."""
    W = np.array(W, dtype=np.float64, copy=True)
    R, C = _check_shape(W, group)
    grid = np.asarray(grid, dtype=np.float64)
    if blocksize % group:
        # A block must consist of whole groups only (otherwise a group's scale crosses the
        # block boundary). Round up instead of erroring: 128 with group=96 becomes 192.
        blocksize = group * max(1, blocksize // group + 1)
    U = _inverse_cholesky_upper(H, percdamp)

    idx = np.zeros((R, C), dtype=np.uint8)
    scales = np.zeros((R, C // group), dtype=np.float64)

    for i1 in range(0, C, blocksize):
        i2 = min(i1 + blocksize, C)
        W1 = W[:, i1:i2].copy()
        ng = (i2 - i1) // group
        block_scales = _group_absmax(W1, group)
        scales[:, i1 // group:i2 // group] = block_scales

        U1 = U[i1:i2, i1:i2]
        Err = np.zeros_like(W1)
        for j in range(i2 - i1):
            col_scale = block_scales[:, j // group]
            picked, q = _nearest(W1[:, j], col_scale, grid)
            Err[:, j] = W1[:, j] - q
            W1[:, j] = q
            idx[:, i1 + j] = picked
            if j + 1 < i2 - i1:
                W1[:, j + 1:] -= Err[:, [j]] * (U1[j, j + 1:] / U1[j, j])
        W[:, i1:i2] = W1
        if i2 < C:
            # Same transfer law as inside the block: a column's error enters with weight
            # 1/R_jj. Without this division the cross-block transfer is scaled inconsistently,
            # error accumulates across blocks and diverges by orders of magnitude on real
            # Hessians (check — test_gptq_survives_rank_deficient_hessian).
            Err_scaled = Err / np.diag(U1)[None, :]
            W[:, i2:] -= Err_scaled @ U[i1:i2, i2:]

    return QuantResult(idx, scales, grid, group)


def reconstruct(res: QuantResult) -> np.ndarray:
    """Decode: group scale × level by index (i.e. the LUT path without the formula's arithmetic)."""
    R, C = res.idx.shape
    cols = np.arange(C) // res.group
    return res.scales[:, cols] * res.grid[res.idx]


def h_error(W: np.ndarray, What: np.ndarray, H: np.ndarray, den: float | None = None) -> float:
    """Relative layer-output error: tr(ΔHΔᵀ) / tr(WHWᵀ).

    Denominator via BLAS (D@H) rather than einsum: a naive einsum('ij,jk,ik->') on a
    576x1536 layer cost ~6 s, which made the measurement matrix outrageous. The
    denominator tr(WHWᵀ) does not depend on the quantization configuration, so it can
    be computed once per layer and passed in as den.
    """
    D = np.asarray(What, dtype=np.float64) - np.asarray(W, dtype=np.float64)
    if den is None:
        den = float(np.sum(np.asarray(W, dtype=np.float64) * (np.asarray(W, dtype=np.float64) @ H)))
    if den <= 0:
        return float("inf")
    return float(np.sum(D * (D @ H)) / den)


def refit_grid(W: np.ndarray, res: QuantResult, Hdiag: np.ndarray,
               iters: int = 20) -> np.ndarray:
    """Rebuild the grid shape to the layer's weight distribution under the Hessian metric.

    With scales fixed, the objective over levels equals
        Σ_r Σ_c H_cc (W[r,c] − s_{r,g}·G[k])² = Σ_i (s_g²·H_cc)·(y_i − G[k])²,  y = W / s_g,
    i.e. plain weighted Lloyd over the normalized values. Element (r,c) weighs as the
    square of its group's scale times the diagonal weight of ITS INPUT channel H_cc
    (not of the row: H is C-dimensional, not R-dimensional).
    """
    R, C = W.shape
    group = res.group
    Hdiag = np.asarray(Hdiag, dtype=np.float64)
    if Hdiag.shape != (C,):
        raise ValueError(f"Hdiag must have length C={C}, got {Hdiag.shape}")
    y = (W.reshape(R, C // group, group) / res.scales[:, :, None]).ravel()
    w = (np.repeat(res.scales ** 2, group, axis=1) * Hdiag[None, :]).ravel()
    grid = weighted_lloyd(y, w, L=res.grid.size, init=res.grid, iters=iters, symmetrize=True)
    # Normalization to 1 is NOT required: the scale is defined as the group's absmax and does
    # not depend on the grid, so the top level legitimately becomes the centroid of the top
    # cluster (slightly below absmax). It still must remain within [-1, 1].
    if np.max(np.abs(grid)) > 1.0:
        grid = grid / np.max(np.abs(grid))
    return grid


def refit_from_absmax(W: np.ndarray, Hdiag: np.ndarray, L: int, group: int,
                     weighted: bool = True, iters: int = 20) -> np.ndarray:
    """Rebuild the grid shape knowing nothing about any previous quantization.

    Scales are the groups' absmax, start is a uniform grid. Needed where shapes are compared
    "from a clean slate" (expressiveness ladder, lloyd/shapeq-style specs) so as not to
    fabricate a fake QuantResult just for that.
    """
    W = np.asarray(W, dtype=np.float64)
    R, C = _check_shape(W, group)
    scales = _group_absmax(W, group)
    res = QuantResult(np.zeros((R, C), dtype=np.uint8), scales, make_grid("uniform", L), group)
    weights = np.asarray(Hdiag, dtype=np.float64).ravel() if weighted else np.ones(C)
    return refit_grid(W, res, weights, iters=iters)


def improve_grid(W: np.ndarray, H: np.ndarray, grid: np.ndarray, group: int,
                 rounds: int = 3, rounder=None, den: float | None = None,
                 **round_kw) -> tuple[QuantResult, np.ndarray, list[float]]:
    """Outer loop: grid → rounding → measurement → refit. A round is accepted only if the objective did not grow.

    rounder(W, H, grid, group) -> QuantResult; GPTQ by default, RTN can be passed in
    to measure the grid shape's contribution in isolation.
    """
    Hdiag = np.diag(H)
    if rounder is None:
        def rounder(W_, H_, g_, gr_):
            return gptq(W_, H_, g_, gr_, **round_kw)

    best_res = rounder(W, H, grid, group)
    best_grid = np.asarray(grid, dtype=np.float64)
    best = h_error(W, reconstruct(best_res), H, den)
    history = [best]

    for _ in range(rounds):
        candidate_grid = refit_grid(W, best_res, Hdiag)
        candidate_res = rounder(W, H, candidate_grid, group)
        value = h_error(W, reconstruct(candidate_res), H, den)
        if value < best:
            best_res, best_grid, best = candidate_res, candidate_grid, value
        history.append(best)
    return best_res, best_grid, history
