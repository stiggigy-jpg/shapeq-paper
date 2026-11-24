"""Tensor-level fitting machinery: grid fit, Hessian cache, 4-bit pack/unpack.

This is the public home of the code that produced the paper's fitted grids and
`.pt` packs (`fit_b4.py` and `run_ppl_grid.py` both call into it). The numeric
paths are unchanged since the shipped artifacts were built; the only knobs that
move results are `engine` (see `fit_tensor_grid`) and the Hessian-cache key.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import torch

from .data import calibration_chunks
from .grids import make_grid
from .warps import eqmass_grid, fit_scale, fit_warp, group_layout, shapeq_grid

FIT_N = 150_000  # subsample size for grid search (shape/kappa)


def fit_tensor_grid(W, H, L, group, kind, engine="hist"):
    """One grid (L levels) for one tensor. kind: uniform | uniform_k | shapeq | eqmass | eqmassk | fp4.

    engine="hist"  histogram fitter (fit_tensor_hist): shape+kappa from a single
                   histogram, no rng subsample, deterministic and several times
                   faster; exact-objective quality no worse than legacy (validators:
                   ipu_hist_parity). Used for every shapeq artifact fitted from
                   2026-11-19 on (incl. the Qwen2.5 pack).
    engine="legacy" subsample-based fitter (fit_warp/fit_scale over a tensor-seeded
                   rng). This is what the 2026-11-18 SmolLM pack was fitted with;
                   pass it to reproduce that file byte-for-byte.
    """
    if kind == "uniform":
        return make_grid("uniform", L)
    if kind in ("fp4", "fp4 e2m1"):
        return make_grid("fp4e2m1", 16)  # fixed E2M1, per-group scale = absmax, same as everyone else
    Hdiag = np.diag(H)
    if kind == "shapeq" and engine == "hist":
        from .warps_hist import fit_tensor_hist
        theta, k = fit_tensor_hist(W, Hdiag, L, group)
        return shapeq_grid(L, theta) * k
    # The grid-search subsample is seeded deterministically FROM THE TENSOR, so
    # fit quality does not depend on call order or on how many tensors were
    # fitted before (an earlier module-level RNG made grids drift between runs).
    probe = np.ascontiguousarray(W[: max(1, W.shape[0] // 4)]).tobytes()[:262144]
    rng = np.random.default_rng(zlib.crc32(probe) ^ (W.shape[0] * 1000003)
                                ^ (W.shape[1] * 9176) ^ (L * 31337))
    if kind in ("eqmass", "eqmassk"):
        y, w = group_layout(W, Hdiag, group)
        y, w = y.ravel(), w.ravel()
        if len(y) > FIT_N:
            sel = rng.choice(len(y), FIT_N, replace=False)
            y, w = y[sel], w[sel]
        # no outer fit_scale on purpose: it would move the Voronoi mid-points off
        # the quantiles, and uniformity holds exactly on the boundaries the decoder sees
        return eqmass_grid(y, w, L, kappa=(kind == "eqmassk"))
    if kind == "shapeq":
        y, w = group_layout(W, Hdiag, group)
        y, w = y.ravel(), w.ravel()
        if len(y) > FIT_N:
            sel = rng.choice(len(y), FIT_N, replace=False)
            y, w = y[sel], w[sel]
        theta, _ = fit_warp(y, w, L)
        grid = shapeq_grid(L, theta)
    else:
        grid = make_grid("uniform", L)
    y, w = group_layout(W, Hdiag, group)
    y, w = y.ravel(), w.ravel()
    # the scale objective is evaluated on a subsample: the full fp64 array
    # (e.g. 35840x448 x2) costs close to a gigabyte per tensor and swaps;
    # 1.5M weights give kappa well within the precision a grid needs
    # (same rule for every kind, so comparisons stay fair)
    if len(y) > 1_500_000:
        head = np.arange(0, len(y), len(y) // 1_500_000)[:1_500_000]
        y, w = y[head], w[head]
    if len(y) > FIT_N:
        sel = rng.choice(len(y), FIT_N, replace=False)
        k = fit_scale(y[sel], w[sel], grid)[0]
    else:
        k = fit_scale(y, w, grid)[0]
    return grid * k


def hessian_cache_dir(model_id: str, rows: int) -> Path:
    return Path(f"cache_h_{model_id.rstrip('/').split('/')[-1]}_{rows}r")


def load_hessian_cache(cache_dir, model_id: str, rows: int, seq: int, calib: int,
                       dtype: str, names: list[str]):
    """Return (offsets, mmap) if a matching cache exists, else None.

    Cache format: meta.json (key fields + per-tensor offsets) + one blob of
    row-major Hessians. dtype fp32 -> f64 blob, fp16 -> f32 blob. Written to
    disk (page-cache evictable) instead of anonymous RAM on purpose: holding
    2.6 GB of fp32/f64 per-model Hessians in RSS gets small machines OOM-killed.
    """
    hdir = Path(cache_dir)
    hdt = "f32" if dtype == "fp16" else "f64"
    meta_p, blob_p = hdir / "meta.json", hdir / f"h_{hdt}.bin"
    if not (meta_p.exists() and blob_p.exists()):
        return None
    meta = json.loads(meta_p.read_text())
    key = {"model": model_id, "rows": rows, "seq": seq, "calib": calib,
           "dtype": dtype, "names": names}
    if meta.get("offsets") is None or any(meta.get(k) != v for k, v in key.items()):
        return None
    mm = np.memmap(blob_p, dtype=np.float32 if hdt == "f32" else np.float64, mode="r")
    return meta["offsets"], mm, meta


def build_hessian_cache(cache_dir, model_id, model, names, tokenizer, rows, seq, calib, dtype):
    """H = X^T X / n per linear layer from one forward pass over the calibration corpus.

    Honest caveat (unchanged from the paper run): the Hessian is stale — it is
    collected once and not rebuilt between blocks. GSQ here is represented by
    GPTQ + grid refits with exact refit, no full Gumbel training (CPU study).
    """
    from .model import collect_inputs
    hdir = Path(cache_dir)
    hdir.mkdir(parents=True, exist_ok=True)
    hdt = "f32" if dtype == "fp16" else "f64"
    blob_p = hdir / f"h_{hdt}.bin"
    calib_chunks_arr = calibration_chunks(tokenizer, calib, seq)
    raw = collect_inputs(model, names, calib_chunks_arr, max_rows=rows)
    hcls = np.float32 if hdt == "f32" else np.float64
    offs, pos = {}, 0
    with blob_p.open("wb") as fh:
        for n in names:
            H = (raw[n].T @ raw[n] / raw[n].shape[0]).astype(hcls)
            fh.write(H.tobytes())
            offs[n] = [pos, H.shape[0]]
            pos += H.shape[0] ** 2
            del H
    key = {"model": model_id, "rows": rows, "seq": seq, "calib": calib,
           "dtype": dtype, "names": names, "offsets": offs}
    (hdir / "meta.json").write_text(json.dumps(key))
    del raw
    return offs


def hessian_get(offs, mm, name):
    pos, c = offs[name]
    return np.asarray(mm[pos:pos + c * c].reshape(c, c), dtype=np.float64)


def pack_layer(res):
    """QuantResult -> (packed uint8 (R, C/2) 4-bit codes lo|hi<<4, scales fp16 (R, C/group), lut fp16 (L,))."""
    idx = torch.as_tensor(res.idx.astype(np.uint8))
    packed = idx[:, 0::2] | (idx[:, 1::2] << 4)
    return packed, torch.as_tensor(res.scales).half(), torch.as_tensor(res.grid).half()


def unpack(packed):
    a = (packed & 15).to(torch.int64)
    b = (packed >> 4).to(torch.int64)
    return torch.stack((a, b), dim=2).reshape(packed.shape[0], packed.shape[1] * 2)


def dequant_matrix(state, name, group):
    """Exact fp64 rebuild via lut[code] * scale — what the GGML kernel reproduces."""
    packed = state[name + ".packed"].numpy()
    scales = state[name + ".scales"].numpy().astype(np.float64)   # (R,G) fp16
    lut = state[name + ".lut"].numpy().astype(np.float64)         # (L,) fp16
    idx = np.empty((packed.shape[0], 2 * packed.shape[1]), dtype=np.int64)
    idx[:, 0::2] = packed & 15
    idx[:, 1::2] = packed >> 4
    return lut[idx] * np.repeat(scales, group, axis=1)
