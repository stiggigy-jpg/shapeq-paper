# -*- coding: utf-8 -*-
"""Fit the b4 shapeq grid and pack it: HF checkpoint -> b4_<model>_shapeq.pt.

Pipeline (identical to the paper run): per tensor, fit a 16-level non-uniform
grid with the Hessian-diagonal-weighted objective (`qb.fit.fit_tensor_grid`),
round with GPTQ (`qb.bench._rounder`), then pack 4-bit codes + fp16 per-group
scale + fp16 LUT per tensor. Output `.pt` holds {state, group, names} and feeds
`pack_q4lut.py` (see the model repo) for the GGUF conversion.

Hessians come from the disk cache built here (or reused if the key matches);
the model/Hessian dtype pair follows the paper: fp32 model -> f64 Hessians,
fp16 model -> f32 Hessians.

Engine note: the shipped SmolLM pack (2026-11-18) predates the histogram
fitter and was produced with --engine legacy; the Qwen2.5 pack uses the
default histogram engine.

  python fit_b4.py --model HuggingFaceTB/SmolLM-135M --engine legacy
  python fit_b4.py --model Qwen/Qwen2.5-0.5B --dtype fp16 --group 64
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from qb.bench import _rounder
from qb.data import evaluation_chunks
from qb.fit import (build_hessian_cache, dequant_matrix, fit_tensor_grid,
                    hessian_cache_dir, hessian_get, load_hessian_cache, pack_layer)
from qb.model import linear_layers, load_model, perplexity

SEQ = 512
PROMPTS = {
    "ru": "Один раз весна, два раза весна, три",
    "en": "The capital of France is",
}


def rss_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return -1


def fakequant_install(model, modules, names, state, group, dt):
    """Replace each Linear with a plain Linear holding the LUT-decoded weights."""
    for n in names:
        parent_name, _, leaf = n.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        old = getattr(parent, leaf)
        new = nn.Linear(old.weight.shape[1], old.weight.shape[0],
                        bias=old.bias is not None, dtype=dt)
        new.weight.data = torch.tensor(dequant_matrix(state, n, group)).to(dt)
        if old.bias is not None:
            new.bias.data = old.bias.detach().clone()
        setattr(parent, leaf, new)
        del old


def evaluate(model, tok, test, batch):
    with torch.no_grad():
        ppl = perplexity(model, test, batch=batch)
    samples = {}
    for tag, prompt in PROMPTS.items():
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        samples[tag] = prompt + tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
    return ppl, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    ap.add_argument("--out", default=None, help="output .pt (default: b4_<model>_shapeq.pt)")
    ap.add_argument("--group", type=int, default=0,
                    help="weights per fp16 scale; 0 = 96 if all input dims allow, else 64")
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "fp16"),
                    help="model dtype; must match how the Hessian cache was keyed")
    ap.add_argument("--engine", default="hist", choices=("hist", "legacy"),
                    help="grid fitter; 'legacy' reproduces the 2026-11-18 SmolLM pack")
    ap.add_argument("--cache", default=None, help="Hessian cache dir "
                    "(default cache_h_<model>_<rows>r, rebuilt when the key mismatches)")
    ap.add_argument("--calib", type=int, default=32, help="calibration chunks (seq 512)")
    ap.add_argument("--rows", type=int, default=2048, help="calibration rows per layer")
    ap.add_argument("--only", default="", help="debug: comma-separated layer-name substrings")
    ap.add_argument("--test-chunks", type=int, default=96)
    ap.add_argument("--batch", type=int, default=1,
                    help="eval batch; keep 1 for 0.5B-class models (fp32 logits are ~2 GB at batch 4)")
    ap.add_argument("--no-eval", action="store_true", help="skip ppl/greedy evaluation")
    ap.add_argument("--pack-in", default=None,
                    help="skip quantization: evaluate an existing .pt only")
    args = ap.parse_args()

    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.pack_in:  # evaluation-only mode
        pk = torch.load(args.pack_in, map_location="cpu", weights_only=False)
        state, grp, names = pk["state"], int(pk["group"]), pk["names"]
        dt = torch.float16 if args.dtype == "fp16" else torch.float32
        model, _ = load_model(args.model, dtype=dt)
        modules = dict(model.named_modules())
        fakequant_install(model, modules, names, state, grp, dt)
        print(f"fake-quant from {args.pack_in}, RSS {rss_mb():.0f} MB", flush=True)
        if args.no_eval:
            return
        test = evaluation_chunks(tok, args.test_chunks, SEQ)
        ppl, samples = evaluate(model, tok, test, args.batch)
        print(f"ppl={ppl:.3f} (chunks={args.test_chunks}, batch={args.batch})", flush=True)
        out = Path(args.pack_in)
        Path(args.out or out).with_suffix(out.suffix + ".json").write_text(json.dumps({
            "model": args.model, "group": grp, "pack": str(out),
            "test_chunks": args.test_chunks, "seq": SEQ, "batch": args.batch,
            "ppl_fakequant": round(ppl, 3), "samples_greedy64": samples,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return

    model, _ = load_model(args.model, dtype=torch.float16 if args.dtype == "fp16" else torch.float32)
    names = linear_layers(model)
    modules = dict(model.named_modules())

    if args.group:
        group = args.group
    else:
        indims = [modules[n].weight.shape[1] for n in names]
        group = 96 if all(d % 96 == 0 for d in indims) else (
            64 if all(d % 64 == 0 for d in indims) else 0)
        assert group, "input dims divisible by neither 96 nor 64 — pass --group"
    for n in names:
        assert modules[n].weight.shape[1] % group == 0, \
            f"{n}: in-dim {modules[n].weight.shape[1]} not a multiple of group {group}"

    cache = Path(args.cache) if args.cache else hessian_cache_dir(args.model, args.rows)
    got = load_hessian_cache(cache, args.model, args.rows, SEQ, args.calib, args.dtype, names)
    if got is None:
        print(f"Hessians: building cache {cache} "
              f"(rows={args.rows} seq={SEQ} calib={args.calib} dtype={args.dtype}) ...", flush=True)
        offs = build_hessian_cache(cache, args.model, model, names, tok,
                                   args.rows, SEQ, args.calib, args.dtype)
        got = load_hessian_cache(cache, args.model, args.rows, SEQ, args.calib, args.dtype, names)
        assert got is not None
    offs, mm, meta = got
    print(f"Hessians: cache {cache} (dtype={meta['dtype']} rows={meta['rows']} "
          f"calib={meta['calib']}), {time.time()-t0:.0f} s", flush=True)

    filt = [s.strip() for s in args.only.split(",") if s.strip()]
    rounder = _rounder("gptq")
    state, n_w = {}, 0
    for i, n in enumerate(names):
        if filt and not any(s in n for s in filt):
            continue
        H = hessian_get(offs, mm, n)
        W = modules[n].weight.detach().numpy().astype(np.float64)
        grid = fit_tensor_grid(W, H, 16, group, "shapeq", engine=args.engine)
        res = rounder(W, H, grid, group)
        state[n + ".packed"], state[n + ".scales"], state[n + ".lut"] = pack_layer(res)
        state[n + ".bias"] = modules[n].bias.detach().clone() if modules[n].bias is not None else None
        n_w += W.size
        del W, H, res
        if i % 24 == 0:
            print(f"packed {i}/{len(names)} ({time.time()-t0:.0f} s, RSS {rss_mb():.0f} MB)", flush=True)
    packed_names = [n for n in names if n + ".packed" in state]

    out = Path(args.out) if args.out else Path(f"b4_{args.model.rstrip('/').split('/')[-1]}_shapeq.pt")
    torch.save({"state": state, "group": group, "names": packed_names}, out)
    mb = out.stat().st_size / 1e6
    bps = 8 * (mb * 1e6 - sum(state[n + ".bias"].numel() * 4 for n in packed_names
                              if state[n + ".bias"] is not None)) / n_w
    print(f"pack: {out} {mb:.0f} MB, {bps:.3f} bits/weight, {time.time()-t0:.0f} s", flush=True)

    if args.no_eval or filt:
        return
    dt = torch.float16 if args.dtype == "fp16" else torch.float32
    fakequant_install(model, modules, packed_names, state, group, dt)
    test = evaluation_chunks(tok, args.test_chunks, SEQ)
    ppl, samples = evaluate(model, tok, test, args.batch)
    print(f"ppl={ppl:.3f} (chunks={args.test_chunks}, batch={args.batch}) RSS {rss_mb():.0f} MB", flush=True)
    out.with_suffix(out.suffix + ".json").write_text(json.dumps({
        "model": args.model, "group": group, "engine": args.engine, "cache": str(cache),
        "hess_meta": {k: meta.get(k) for k in ("rows", "seq", "calib", "dtype")},
        "test_chunks": args.test_chunks, "seq": SEQ,
        "ppl_fakequant": round(ppl, 3), "bps_pack": round(bps, 4), "pack_mb": round(mb, 1),
        "samples_greedy64": samples,
    }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
