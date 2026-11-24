# -*- coding: utf-8 -*-
"""Paper Tables 2-4 driver: full quant ladder perplexity on a live model (CPU).

imatrix-free ladder: per tensor fit a grid for each bit width (qb.fit.fit_tensor_grid),
round with GPTQ, swap the weights in, measure WikiText-2 test perplexity (sliding,
96 chunks x 512 by default). Configs: fp32, fp8 e4m3, {2,3,4,6,8} bits x
{uniform, uniform+scale-step, shapeq(+3 refits)}, 4-bit fp4(E2M1), and the
Lagrangian 2/3-bit mixes for an exact average-bit budget. Appends CSV rows;
rows already present for a config are skipped (resume-friendly).

Honest caveats (same as the paper): the Hessian is stale (one pass, not rebuilt
between blocks); GSQ is represented by GPTQ + grid refits with exact refit (no
full Gumbel training on CPU); grid search uses a 150k-point subsample per tensor.

  python run_ppl_grid.py --model HuggingFaceTB/SmolLM-135M --engine legacy --out results_ppl.csv
  python run_ppl_grid.py --model Qwen/Qwen2.5-0.5B --dtype fp16 --group 64 --out results_ppl_qwen.csv
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from qb.bench import _rounder
from qb.fit import build_hessian_cache, hessian_cache_dir, load_hessian_cache
from qb.model import linear_layers, load_model, perplexity

FIELDS = ["model", "config", "ebw", "quant_s", "ppl", "ppl_loss_vs_fp32", "sec",
          "layers_2bit", "layers_3bit"]


def fp8_model(model, modules, names, group):
    for name in names:
        W = modules[name].weight.detach().numpy().astype(np.float32)
        R, C = W.shape
        Wg = W.reshape(R, C // group, group)
        s = np.maximum(np.abs(Wg).max(axis=2, keepdims=True) / np.float32(448.0), np.float32(1e-12))
        q = (torch.tensor(Wg / s).to(torch.float8_e4m3fn).to(torch.float32).numpy()) * s
        modules[name].weight.data = torch.tensor(q.reshape(R, C)).to(modules[name].weight.dtype)
    return 8.0 + 16.0 / group


def lagrangian_mix(names, numel, err2, err3, target_bits):
    """Assign 2 or 3 bits per layer minimizing expected error at an exact average budget."""
    total = sum(numel[n] for n in names)

    def bits_sum(lam):
        return sum(numel[n] * (2 if err2[n] + lam * 2 <= err3[n] + lam * 3 else 3) for n in names)

    lo, hi = 0.0, 1e9
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if bits_sum(mid) / total > target_bits:
            lo = mid
        else:
            hi = mid
    lam = hi
    choice = {n: (2 if err2[n] + lam * 2 <= err3[n] + lam * 3 else 3) for n in names}

    def cur():
        return sum(numel[n] * choice[n] for n in names) / total

    while cur() > target_bits:
        n = min((n for n in names if choice[n] == 3), key=lambda n: (err3[n] - err2[n]) / numel[n])
        choice[n] = 2
    while cur() < target_bits - 0.02:
        n = min((n for n in names if choice[n] == 2), key=lambda n: (err2[n] - err3[n]) / numel[n])
        choice[n] = 3
    return choice, cur()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    ap.add_argument("--group", type=int, default=96)
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--test-chunks", type=int, default=96, dest="test_chunks")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--mix-target", type=float, default=2.4)
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "fp16"))
    ap.add_argument("--engine", default="hist", choices=("hist", "legacy"),
                    help="grid fitter; use legacy to reproduce the 2026-11-17/18 SmolLM CSV")
    ap.add_argument("--cache", default=None, help="Hessian cache dir (default: cache_h_<model>_<rows>r)")
    ap.add_argument("--out", default="results_ppl.csv")
    ap.add_argument("--only", default="")
    ap.add_argument("--tag-suffix", default="", dest="tag_suffix",
                    help="suffix for config tags (e.g. ' hist' adds a new-engine row "
                         "next to the legacy reference without overwriting it)")
    args = ap.parse_args()

    from qb.fit import fit_tensor_grid
    out_path = Path(args.out)
    model, _ = load_model(args.model, dtype=torch.float16 if args.dtype == "fp16" else torch.float32)
    base_tag = args.dtype
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    from qb.data import evaluation_chunks
    test = evaluation_chunks(tokenizer, args.test_chunks, args.seq)
    names = linear_layers(model)
    modules = dict(model.named_modules())
    numel = {n: modules[n].weight.numel() for n in names}
    orig = {n: modules[n].weight.detach().clone() for n in names}

    def restore():
        for n in names:
            modules[n].weight.data = orig[n].clone()

    def write_row(row):
        new = not out_path.exists() or out_path.stat().st_size == 0
        with out_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
            if new:
                w.writeheader()
            w.writerow(row)

    def evaluate(tag, ebw_rate, t_quant, extra=None):
        t0 = time.time()
        p = perplexity(model, test)
        row = {"model": args.model.split("/")[-1], "config": tag, "ebw": round(ebw_rate, 3),
               "quant_s": round(t_quant, 0), "ppl": round(p, 3),
               "ppl_loss_vs_fp32": round(p / ppl_base - 1.0, 5), "sec": round(time.time() - t0, 0)}
        if extra:
            row.update(extra)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        write_row(row)

    # resume after restart: configs already in the CSV are not recomputed
    done = {}
    if out_path.exists():
        with out_path.open() as fh:
            for r in csv.DictReader(fh):
                if r.get("ppl") and r.get("config"):
                    done.setdefault(r["config"], float(r["ppl"]))

    t0 = time.time()
    if base_tag in done:
        ppl_base = done[base_tag]
        print(f"base '{base_tag}' already in CSV (ppl={ppl_base:.3f}) - skipping", flush=True)
    else:
        ppl_base = perplexity(model, test)
        print(f"ppl {base_tag} = {ppl_base:.3f}  ({time.time()-t0:.0f} s)", flush=True)
        write_row({"model": args.model.split("/")[-1], "config": base_tag, "ebw": 32.0, "quant_s": 0,
                   "ppl": round(ppl_base, 3), "ppl_loss_vs_fp32": 0.0, "sec": round(time.time() - t0, 0)})

    t0 = time.time()
    cache = Path(args.cache) if args.cache else hessian_cache_dir(args.model, args.rows)
    got = load_hessian_cache(cache, args.model, args.rows, args.seq, args.calib, args.dtype, names)
    if got is None:
        print(f"Hessians: building cache {cache} ...", flush=True)
        build_hessian_cache(cache, args.model, model, names, tokenizer,
                            args.rows, args.seq, args.calib, args.dtype)
        got = load_hessian_cache(cache, args.model, args.rows, args.seq, args.calib, args.dtype, names)
    offs, mm, _meta = got
    print(f"Hessians ready ({len(names)} layers, {time.time()-t0:.0f} s)", flush=True)

    def load_H(n):
        pos, c = offs[n]
        return np.asarray(mm[pos:pos + c * c].reshape(c, c), dtype=np.float64)

    only = {s.strip() for s in args.only.split(",") if s.strip()}

    def grid_for(W, H, bits, kind):
        return fit_tensor_grid(W, H, 2 ** bits, args.group, kind, engine=args.engine)

    rounder = _rounder("gptq")

    def run(tag, bits, kind, refit=0):
        from qb.gptq import h_error, improve_grid, reconstruct
        tag = tag + args.tag_suffix
        if only and tag not in only:
            return None
        if tag in done:
            print(f"config '{tag}' already in CSV (ppl={done[tag]:.3f}) - skipping", flush=True)
            return None
        restore()
        t0 = time.time()
        abs_err = {}
        numel_total = bits_total = 0
        for n in names:
            W = orig[n].numpy().astype(np.float64)
            H = load_H(n)
            den = float(np.sum(W * (W @ H)))
            grid = grid_for(W, H, bits, kind)
            if refit:
                best_res, best_grid, hist = improve_grid(W, H, grid, args.group,
                                                         rounds=refit, rounder=rounder, den=den)
                What = reconstruct(best_res)
                err = hist[-1]
            else:
                res = rounder(W, H, grid, args.group)
                What = reconstruct(res)
                err = h_error(W, What, H, den)
            modules[n].weight.data = torch.tensor(What, dtype=orig[n].dtype)
            abs_err[n] = err * den
            numel_total += W.size
            bits_total += W.size * bits
        rate = bits_total / numel_total + 32.0 / args.group
        evaluate(tag, rate, time.time() - t0)
        return abs_err

    err_store = {}
    for bits in (3, 2):
        for kind in ("uniform", "uniform_k", "shapeq"):
            e = run(f"b{bits} {kind}", bits, kind)
            if e:
                err_store[f"b{bits} {kind}"] = e
        run(f"b{bits} shapeq+refit3", bits, "shapeq", refit=3)
    for bits in (4,):
        for kind in ("uniform", "uniform_k", "shapeq"):
            run(f"b{bits} {kind}", bits, kind)
        run(f"b{bits} shapeq+refit3", bits, "shapeq", refit=3)
        run(f"b{bits} fp4", bits, "fp4")
    # high bit widths: does fitted shape still tie with uniform (loss near zero)?
    for bits in (6, 8):
        run(f"b{bits} uniform", bits, "uniform")
        run(f"b{bits} shapeq", bits, "shapeq")
        run(f"b{bits} shapeq+refit3", bits, "shapeq", refit=3)

    if not only or "fp8" in only:
        restore()
        t0 = time.time()
        rate = fp8_model(model, modules, names, args.group)
        evaluate("fp8 e4m3", rate, time.time() - t0)

    for base2, base3 in (("b2 shapeq", "b3 shapeq"), ("b2 uniform_k", "b3 uniform_k")):
        # tag string is CSV data; the label uses the published method name
        tag = f"mix {args.mix_target} {'shapeq' if 'shapeq' in base2 else 'uniform'}"
        if only and tag not in only:
            continue
        if base2 not in err_store or base3 not in err_store:
            print(f"skip {tag}: missing errors for {base2}/{base3}", flush=True)
            continue
        choice, avg = lagrangian_mix(names, numel, err_store[base2], err_store[base3], args.mix_target)
        restore()
        t0 = time.time()
        kind = "shapeq" if "shapeq" in base2 else "uniform_k"
        tb = tn = 0
        for n in names:
            b = choice[n]
            W = orig[n].numpy().astype(np.float64)
            H = load_H(n)
            grid = grid_for(W, H, b, kind)
            modules[n].weight.data = torch.tensor(reconstruct(rounder(W, H, grid, args.group)),
                                                  dtype=orig[n].dtype)
            tb += W.size * b
            tn += W.size
        evaluate(tag, tb / tn + 32.0 / args.group, time.time() - t0,
                 extra={"layers_2bit": sum(v == 2 for v in choice.values()),
                        "layers_3bit": sum(v == 3 for v in choice.values())})

    restore()
    print("done", flush=True)


if __name__ == "__main__":
    main()
