"""Mass-balanced ternary quantization (owner's idea 20.11): E-mean + mass shares + LUT with shift.

The group is centered by its mean (E-mean), weights are assigned to levels -1/0/+1 by the
mass share split (each outer level gets split, zero gets 1-2*split); the scale s comes
from least squares. Reconstruction goes through a per-group LUT of three values
{-s+c, c, s+c} — a "corrective LUT", the shift is already baked into the table.
The bet: index log2(3)≈1.585 bits/weight + two numbers per group.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class TernResult:
    idx: np.ndarray    # (R, C) with values {-1, 0, +1} — the LUT code
    lut: np.ndarray    # (R, C//group, 3): [-s+c, c, s+c]
    group: int
    split: float       # target share of each outer level


def _as_groups(W: np.ndarray, group: int) -> tuple[np.ndarray, int, int]:
    if W.ndim != 2 or W.shape[1] % group:
        raise ValueError(f"matrix {W.shape} is not divisible by group={group}")
    R, C = W.shape
    return W.reshape(R, C // group, group), R, C


def ternary_quantize(W: np.ndarray, group: int = 96, split: float = 1.0 / 3.0) -> TernResult:
    """Assignment by mass quantiles + L2 scale. split=0 degenerates into a zero-quant."""
    wg, R, C = _as_groups(W.astype(np.float64), group)
    c = wg.mean(axis=2, keepdims=True)
    wc = wg - c
    if split <= 0.0:
        a = np.zeros_like(wc, dtype=np.int8)
    else:
        t_hi = np.quantile(wc, 1.0 - split, axis=2, keepdims=True)
        t_lo = np.quantile(wc, split, axis=2, keepdims=True)
        a = (wc > t_hi).astype(np.int8) - (wc < t_lo).astype(np.int8)
    num = (wc * a).sum(axis=2)
    den = (a.astype(np.float64) ** 2).sum(axis=2)
    s = num / np.maximum(den, EPS)
    lut = np.stack([c[:, :, 0] - s, c[:, :, 0], c[:, :, 0] + s], axis=2)
    return TernResult(idx=a.reshape(R, C), lut=lut, group=group, split=split)


def ternary_nearest(W: np.ndarray, group: int = 96, iters: int = 10) -> TernResult:
    """Comparison baseline: nearest-level with Lloyd iterations on the ±s/2 threshold and the scale."""
    wg, R, C = _as_groups(W.astype(np.float64), group)
    c = wg.mean(axis=2, keepdims=True)
    wc = wg - c
    s = np.maximum(np.abs(wc).mean(axis=2), EPS)
    for _ in range(iters):
        a = np.where(wc > s[:, :, None] / 2, 1, np.where(wc < -s[:, :, None] / 2, -1, 0)).astype(np.int8)
        num = (wc * a).sum(axis=2)
        den = (a.astype(np.float64) ** 2).sum(axis=2)
        s = np.abs(num / np.maximum(den, EPS))
    lut = np.stack([c[:, :, 0] - s, c[:, :, 0], c[:, :, 0] + s], axis=2)
    return TernResult(idx=a.reshape(R, C), lut=lut, group=group, split=float(np.abs(a).mean()))


def ternary_decode(res: TernResult) -> np.ndarray:
    """Decode strictly through the LUT: the value is taken from the table by code — no formula."""
    R, C = res.idx.shape
    g = C // res.group
    ig = (res.idx + 1).reshape(R, g, res.group)
    sel = np.take_along_axis(res.lut, ig, axis=2)  # (R, G, group) copy of the codes inside the group
    return sel.reshape(R, C)


def ternary_stats(W: np.ndarray, What: np.ndarray, idx: np.ndarray | None = None) -> dict[str, float]:
    """Metrics: relative MSE and column-sum drift in scale-stable units.

    colsum_rel: Σ_j |Σ_i d_ij| over columns, divided by Σ_j √R·σ_j — a scale that does
    not collapse from mutual cancellation of signs (with the raw Σ|colsum(W)| the
    denominator degenerates, which had produced spurious 0.5 on both arms).
    """
    d = W.astype(np.float64) - What
    R = W.shape[0]
    sigma = W.astype(np.float64).std(axis=0)
    denom = float((sigma * np.sqrt(R)).sum())
    out = {
        "nmse": float((d * d).sum() / max((W.astype(np.float64) ** 2).sum(), EPS)),
        "colsum_rel": float(np.abs(d.sum(axis=0)).sum() / max(denom, EPS)),
    }
    if idx is not None:
        out["frac_outer"] = float(np.mean(np.abs(idx) == 1))
    return out
