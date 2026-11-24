"""Weighted scalar Lloyd–Max quantizer in one dimension.

This is the "level re-estimation step": with assignments fixed, the objective
    Σ_i w_i (y_i − level[k_i])²
is minimized by the clusters' weighted centroids, alternated with assignment by
midpoint boundaries. In the bench the weights are the diagonal entries of the layer's
Hessian, so levels densify where a weight error is costlier at the layer output, not
where the weights themselves are denser.
"""
from __future__ import annotations

import numpy as np


def assign(y: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Index of the nearest level; boundaries are the midpoints between neighboring levels."""
    if levels.size == 1:
        return np.zeros(y.shape, dtype=np.int64)
    thr = 0.5 * (levels[:-1] + levels[1:])
    return np.searchsorted(thr, y, side="right")


def objective(y: np.ndarray, w: np.ndarray, levels: np.ndarray) -> float:
    """Weighted mean squared error against the level grid `levels`."""
    resid = y - levels[assign(y, levels)]
    return float(np.sum(w * resid * resid) / np.sum(w))


def _default_init(y: np.ndarray, w: np.ndarray, L: int) -> np.ndarray:
    if L == 1:
        return np.array([np.average(y, weights=w)])
    return np.quantile(y, (np.arange(L) + 0.5) / L)


def _levels_from_magnitudes(mags: np.ndarray, L: int) -> np.ndarray:
    """Odd-sized grid from magnitudes m_1 <= ... <= m_P."""
    P = mags.size
    if L == 2 * P:
        return np.concatenate([-mags[::-1], mags])
    return np.concatenate([-mags[::-1], [0.0], mags])


def _symmetric_centroids(y: np.ndarray, w: np.ndarray, idx: np.ndarray,
                         L: int, mags: np.ndarray) -> np.ndarray:
    """Exact minimum of the objective over magnitudes with assignments held fixed.

    A symmetric grid is defined by magnitudes m_1..m_P; a pair of magnitudes corresponds to
    two clusters (left and right), and the optimal magnitude is the weighted centroid of |y|
    over the union of those two clusters. Averaging signed centroids is not allowed:
    on asymmetric data that is not a minimum but an arbitrary point.
    """
    P = L // 2
    sum_w = np.bincount(idx, weights=w, minlength=L)
    sum_ay = np.bincount(idx, weights=w * np.abs(y), minlength=L)
    out = mags.copy()
    odd = (L % 2 == 1)
    for r in range(1, P + 1):
        neg_pos = P - r
        pos_pos = P + r if odd else P + r - 1
        sw = sum_w[neg_pos] + sum_w[pos_pos]
        if sw > 0:
            out[r - 1] = (sum_ay[neg_pos] + sum_ay[pos_pos]) / sw
    return np.sort(out)


def refit_frozen(y: np.ndarray, w: np.ndarray, idx, L: int, init: np.ndarray | None = None) -> np.ndarray:
    """Exact minimum of the objective over levels with FROZEN assignments (one Lloyd step).

    Scenario: indices already exist (a released gguf, GSQ/RCO output) and re-encoding them
    is expensive — then only the level table changes. For a symmetric grid that is the
    weighted centroid of |y| over merged mirror clusters, i.e. the exact minimum inside
    the class of symmetric grids, not a projection.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    idx = np.asarray(idx).ravel()
    if L < 2:
        raise ValueError("need at least two levels")
    if idx.shape != y.shape or w.shape != y.shape:
        raise ValueError("y, w and idx must have matching shapes")
    if idx.min() < 0 or idx.max() >= L:
        raise ValueError(f"idx outside range [0, {L})")
    levels = (_default_init(y, w, L) if init is None
              else np.sort(np.asarray(init, dtype=np.float64).ravel()))
    mags = np.sort(np.abs(levels))[-(L // 2):]
    return _levels_from_magnitudes(_symmetric_centroids(y, w, idx, L, mags), L)


def weighted_lloyd(y: np.ndarray, w: np.ndarray, L: int, init: np.ndarray | None = None,
                   iters: int = 30, tol: float = 1e-10, symmetrize: bool = True,
                   history: list[float] | None = None) -> np.ndarray:
    """Levels minimizing objective(y, w, levels). history is appended with objective values.

    symmetrize restricts the grid class to odd-sized (symmetric quantizer without
    zero-point): optimization runs over magnitudes, exact minimum inside this class.
    Assignment by midpoint boundaries keeps the objective unchanged for points exactly on
    a boundary, so a step can fail to decrease only via a magnitude-order violation —
    just in case, a candidate is accepted only when the objective did not grow.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    if y.shape != w.shape:
        raise ValueError("y and w must have matching shapes")
    if L < 1:
        raise ValueError("need at least one level")

    levels = (_default_init(y, w, L) if init is None
              else np.sort(np.asarray(init, dtype=np.float64).ravel()))
    if levels.size != L:
        raise ValueError(f"init must contain {L} levels, got {levels.size}")

    current = objective(y, w, levels)
    if history is not None:
        history.append(current)

    for _ in range(iters):
        idx = assign(y, levels)
        if symmetrize and L > 1:
            mags = np.sort(np.abs(levels))[-(L // 2):]
            candidate = _levels_from_magnitudes(
                _symmetric_centroids(y, w, idx, L, mags), L)
        else:
            sum_w = np.bincount(idx, weights=w, minlength=L)
            sum_wy = np.bincount(idx, weights=w * y, minlength=L)
            filled = sum_w > 0
            candidate = levels.copy()
            candidate[filled] = sum_wy[filled] / sum_w[filled]
            candidate = np.sort(candidate)

        value = objective(y, w, candidate)
        if value > current:
            break
        improved = current - value
        levels, current = candidate, value
        if history is not None:
            history.append(current)
        if improved <= tol:
            break

    return levels
