"""Microbenchmark of weight decoding: affine reconstruction vs table lookup.

What this honestly measures: the cost of the two reconstruction paths on CPU in a naive
vectorized implementation (numpy), without fused kernels and without accounting for GEMM.
It estimates "how much table gathering costs versus arithmetic", not real inference.
"""
from __future__ import annotations

import time

import numpy as np


def decode_arithmetic(idx: np.ndarray, col_scale: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Affine path (valid only for a uniform grid): scale * (start + idx*step)."""
    step = (grid[-1] - grid[0]) / (grid.size - 1)
    out = idx.astype(np.float32) * np.float32(step)
    out += np.float32(grid[0])
    out *= col_scale
    return out


def decode_lut(idx: np.ndarray, col_scale: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Table path: gathering over levels, then scaling."""
    return grid[idx].astype(np.float32, copy=False) * col_scale


def decode_times(rows: int = 1024, cols: int = 4096, levels: int = 16, group: int = 128,
                 reps: int = 5) -> dict[str, float]:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, levels, size=(rows, cols)).astype(np.uint8)
    grid = np.linspace(-1.0, 1.0, levels).astype(np.float32)
    scales = (rng.random((rows, cols // group), dtype=np.float32) + 0.5)
    col_scale = np.repeat(scales, group, axis=1)

    measured: dict[str, float] = {}
    for name, fn in (("arithmetic", decode_arithmetic), ("lut", decode_lut)):
        fn(idx, col_scale, grid)  # warmup
        best = min(_time_once(fn, idx, col_scale, grid) for _ in range(reps))
        measured[f"{name}_ns"] = best / (rows * cols) * 1e9
    measured["weights"] = rows * cols
    return measured


def _time_once(fn, idx, col_scale, grid) -> float:
    start = time.perf_counter()
    out = fn(idx, col_scale, grid)
    elapsed = time.perf_counter() - start
    assert out.shape == idx.shape
    return elapsed
