"""Families of level grids (reconstructing values) and metadata accounting.

A grid is a sorted array of levels, normalized so that the maximum level equals 1.0.
The group-wise scale (absmax) multiplies the grid from outside, so normalizing to 1
removes the "grid <-> scale" degeneracy.
"""
from __future__ import annotations

import math

import numpy as np

# E2M1 magnitudes (MXFP4/NVFP4), normalized to a maximum of 6.0
_E2M1_MAGNITUDES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _uniform_positions(L: int) -> np.ndarray:
    """Uniform corridor midpoints on [-1, 1], normalized by the maximum."""
    u = (np.arange(L) + 0.5) / L - 0.5
    return u / np.max(np.abs(u))


def _normal_ppf(p: np.ndarray, lo: float = -12.0, hi: float = 12.0, iters: int = 200) -> np.ndarray:
    """Quantile function of N(0,1) by bisection over erf — no scipy dependency."""
    lo_b, hi_b = np.full(p.shape, lo), np.full(p.shape, hi)
    for _ in range(iters):
        mid = 0.5 * (lo_b + hi_b)
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(mid / math.sqrt(2.0)))
        too_small = cdf < p
        lo_b = np.where(too_small, mid, lo_b)
        hi_b = np.where(too_small, hi_b, mid)
    return 0.5 * (lo_b + hi_b)


def _symmetric_from_magnitudes(mags: tuple[float, ...]) -> np.ndarray:
    """Odd-sized grid from positive magnitudes with a duplicated zero (as in hardware formats)."""
    pos = np.sort(np.asarray(mags, dtype=np.float64))
    pos = pos / pos[-1]
    return np.concatenate([-pos[::-1], [0.0, 0.0], pos])


def make_grid(kind: str, L: int, **p) -> np.ndarray:
    """Grid levels: (L,), increasing, odd count, g[-1] == 1.0."""
    if kind == "uniform":
        return _uniform_positions(L)

    if kind == "nf4":
        if L % 2:
            raise ValueError("nf4 requires an even number of levels")
        q = _normal_ppf((np.arange(L) + 0.5) / L)
        return q / np.max(np.abs(q))

    if kind == "logexp":
        # Negative control: the "log on input / expander on output" family, whose canonical
        # form is Schuster companding (mu-law). It was designed for a huge dynamic range,
        # not for bell-shaped weights, so it should lose on them. Levels = expander: small
        # steps near zero.
        mu = float(p["mu"])
        u = _uniform_positions(L)
        if mu <= 1e-9:
            return u
        return np.sign(u) * np.expm1(np.abs(u) * math.log1p(mu)) / mu

    if kind == "asinh":
        a = float(p["a"])
        u = _uniform_positions(L)
        if a <= 1e-9:
            return u
        return np.sinh(a * u) / math.sinh(a)

    if kind == "fp4e2m1":
        if L != 2 * len(_E2M1_MAGNITUDES) + 2:
            raise ValueError(f"fp4e2m1 is defined only for L={2 * len(_E2M1_MAGNITUDES) + 2}")
        return _symmetric_from_magnitudes(_E2M1_MAGNITUDES)

    if kind == "table":
        levels = np.asarray(p["levels"], dtype=np.float64)
        return levels / np.max(np.abs(levels))

    raise ValueError(f"unknown grid family: {kind}")


def ebw(index_bits: int, group: int, numel: int, scale_bits: int = 32,
        codebook_bits: int = 0) -> float:
    """Effective bits per weight: index + group-wise scale + amortized codebook."""
    return index_bits + scale_bits / group + codebook_bits / numel
