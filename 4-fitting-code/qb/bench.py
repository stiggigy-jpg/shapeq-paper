"""Bench orchestrator.

  cache  — one model run: save (W, H) of the selected linear layers to npz.
  sweep  — on the cache, run the "grid × rounding × refit" matrix; torch not needed.
  ppl    — quantize the whole model per config and compute perplexity on test chunks.
  report — CSV -> markdown tables.

Caveat to keep in mind: Hessians are collected by a single fp32-model run and are NOT
rebuilt after quantizing previous blocks (stale Hessian). This is the same for all
configurations, but it understates GPTQ's absolute quality relative to publications.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from qb.decode import decode_times
from qb.grids import ebw, make_grid
from qb.gptq import gptq, h_error, improve_grid, reconstruct, refit_from_absmax, rtn
from qb.warps import PARAMS_PER_GRID, fit_scale, fit_warp, group_layout, shapeq_grid

DEFAULT_GRIDS = "uniform,nf4,fp4e2m1,asinh:1.5,logexp:3,kum,shapeq_sym,shapeq,lloyd,lloydhess"


def parse_grid_spec(spec: str, L: int, W: np.ndarray, Hdiag: np.ndarray,
                    group: int = 128) -> np.ndarray:
    """Specification -> grid.

      uniform | nf4 | fp4e2m1        — 0 coefficients, "off-the-shelf" shape
      asinh:A | logexp:MU            — 1 coefficient, rigid parametric branch
      lloyd | lloydhess              — free table of L levels (MSE / in the Hessian metric)
      kum | shapeq                  — one formula + 4 coefficients fitted to the objective
      shapeq_sym                    — same, but 2 coefficients (no side-to-side skew)
      kum:a_neg,b_neg,a_pos,b_pos    — coefficients by hand, no fitting

    The first two are about expressiveness; the fitted families are about the
    working project: 4 numbers instead of a table.
    """
    name, _, param = spec.partition(":")

    if name in ("lloyd", "lloydhess"):
        return refit_from_absmax(W, Hdiag, L, group, weighted=(name == "lloydhess"))

    if name in ("kum", "shapeq", "shapeq_sym"):
        if param:
            vals = [float(v) for v in param.split(",")]
            if name == "kum" and len(vals) == PARAMS_PER_GRID:
                return shapeq_grid(L, vals)
            if name == "shapeq_sym" and len(vals) == 2:
                return shapeq_grid(L, (vals[0], vals[1]) * 2)
            raise ValueError(
                f"{spec}: kum expects {PARAMS_PER_GRID} coefficients, shapeq_sym — 2")
        y, w = group_layout(W, Hdiag if name == "shapeq" else np.ones(W.shape[1]), group)
        theta, _ = fit_warp(y.ravel(), w.ravel(), L, two_sided=(name != "shapeq_sym"))
        return shapeq_grid(L, theta)

    if name in ("asinh", "logexp"):
        key = "a" if name == "asinh" else "mu"
        return make_grid(name, L, **{key: float(param)})

    return make_grid(name, L)


def layer_selection(model, blocks: str) -> list[str]:
    from qb.model import linear_layers
    all_names = linear_layers(model)
    depth = max(int(n.split(".")[2]) for n in all_names) + 1
    wanted = {int(b) if b.isdigit() or (b.startswith("-") and b[1:].isdigit())
              else int(round(float(b) * (depth - 1))) for b in blocks.split(",")} if blocks else \
        {0, depth // 2, depth - 1}
    return [n for n in all_names if int(n.split(".")[2]) in wanted]


def _slug(name: str) -> str:
    return name.replace("/", "_").replace(".", "__")


def cmd_cache(args) -> None:
    from transformers import AutoTokenizer
    from qb.data import calibration_chunks
    from qb.model import collect_inputs, load_model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import torch
    model, config = load_model(args.model, dtype=torch.float16 if args.dtype == "fp16" else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    chunks = calibration_chunks(tokenizer, args.calib, args.seq)
    names = layer_selection(model, args.blocks)
    print(f"layers into cache: {len(names)} (blocks: {sorted({int(n.split('.')[2]) for n in names})})")
    t0 = time.time()
    inputs = collect_inputs(model, names, chunks, max_rows=args.rows)
    print(f"input collection: {time.time() - t0:.1f} s")
    for name in names:
        module = dict(model.named_modules())[name]
        W = module.weight.detach().numpy().astype(np.float32)
        X = inputs[name]
        H = (X.T @ X / X.shape[0]).astype(np.float32)
        np.savez_compressed(out / f"{_slug(name)}.npz", W=W, H=H, name=name)
        print(f"  {name:52s} W{W.shape} H{H.shape} rows={X.shape[0]}")
    (out / "meta.json").write_text(json.dumps(
        {"model": args.model, "rows": args.rows, "calib": args.calib, "seq": args.seq,
         "group": args.group, "layers": names}, indent=2))


def _rounder(rounding: str):
    if rounding == "rtn":
        return lambda W_, H_, g_, gr_: rtn(W_, np.diag(H_), g_, gr_)
    if rounding == "gptq":
        return gptq
    raise ValueError(f"unknown rounding rule: {rounding}")


def _evaluate(W, H, grid, group, rounding, refit_rounds, kappa=False):
    """One matrix cell: grid × rounding × refit rounds × step multiplier.

    κ is fitted once by the RTN objective as a cheap proxy estimate and then used for both
    RTN and GPTQ. GPTQ's optimal κ could be different (GPTQ partly compensates rounding
    error) — this limitation is recorded in the report.
    """
    kappa_value = 1.0
    if kappa:
        y, w = group_layout(W, np.diag(H), group)
        kappa_value = fit_scale(y.ravel(), w.ravel(), grid)[0]
        grid = grid * kappa_value
    rounder = _rounder(rounding)
    den = float(np.sum(W * (W @ H)))          # metric denominator: once per cell
    t0 = time.time()
    if refit_rounds:
        _, _, history = improve_grid(W, H, grid, group, rounds=refit_rounds,
                                     rounder=rounder, den=den)
        err = history[-1]
    else:
        err = h_error(W, reconstruct(rounder(W, H, grid, group)), H, den)
    return err, time.time() - t0, kappa_value


def metadata_bits(spec: str, L: int) -> int:
    """How many metadata bits the grid shape costs for the WHOLE tensor (not per group).

    This is exactly the price of the design idea: a uniform grid and one formula with four
    numbers are nearly free; a free table of L levels costs 32·L bits, which is already
    noticeable for a small tensor. Counted conservatively as fp32; fp16/BF16 is twice cheaper.
    """
    name = spec.partition(":")[0]
    if name in ("lloyd", "lloydhess"):
        return 32 * L
    if name in ("kum", "shapeq"):
        return 32 * PARAMS_PER_GRID
    if name == "shapeq_sym":
        return 32 * 2
    if ":" in spec:
        return 32            # one family coefficient (asinh, logexp, kum with explicit numbers)
    return 0


_SWEEP_FIELDS = ("layer", "bits", "grid", "rounding", "refit", "kappa", "kappa_mode",
                 "group", "ebw", "rel_err", "seconds")


def cmd_sweep(args) -> None:
    cache = Path(args.cache)
    meta = json.loads((cache / "meta.json").read_text())
    group = meta["group"]
    layers = sorted(cache.glob("*.npz"))
    grids = args.grids.split(",")
    bits = [int(b) for b in args.bits.split(",")]
    roundings = args.rounding.split(",")
    refits = [int(r) for r in args.refit.split(",")]
    kappas = [k.strip() in ("1", "true", "yes") for k in args.kappa.split(",")]

    out = Path(args.out)
    write_header = not out.exists() or out.stat().st_size == 0
    count = 0
    with out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_SWEEP_FIELDS))
        if write_header:
            writer.writeheader()

        def record(**row) -> None:
            writer.writerow(row)
            fh.flush()                      # every row on disk at once: outrun OOM
            print(f"{row['layer'][-26:]:26s} b{row['bits']} {row['grid']:9s} "
                  f"{row['rounding']:4s} refit{row['refit']} k{row['kappa']:.2f} -> "
                  f"{row['rel_err']:.5f}  ({row['seconds']:.1f} s)", flush=True)

        for path in layers:
            blob = np.load(path, allow_pickle=False)
            W, H = blob["W"].astype(np.float64), blob["H"].astype(np.float64)
            Hdiag = np.diag(H)
            name = str(blob["name"])
            for bits_n in bits:
                L = 2 ** bits_n
                for spec in grids:
                    if spec.startswith("fp4e2m1") and L != 16:
                        continue
                    try:
                        base = parse_grid_spec(spec, L, W, Hdiag, group)
                    except ValueError as exc:
                        print(f"skipping {spec} L={L}: {exc}")
                        continue
                    for rounding in roundings:
                        for refit_rounds in refits:
                            for use_kappa in kappas:
                                err, seconds, kv = _evaluate(
                                    W, H, base, group, rounding, refit_rounds, use_kappa)
                                record(layer=name, bits=bits_n, grid=spec, rounding=rounding,
                                       refit=refit_rounds, kappa=round(kv, 4),
                                       kappa_mode=("kappa" if use_kappa else "absmax"),
                                       group=group,
                                       ebw=ebw(bits_n, group, W.size,
                                               codebook_bits=metadata_bits(spec, L)),
                                       rel_err=err, seconds=round(seconds, 2))
                                count += 1
    print(f"\n{count} rows -> {out}")


def cmd_ppl(args) -> None:
    import torch
    from transformers import AutoTokenizer
    from qb.data import calibration_chunks, evaluation_chunks
    from qb.model import collect_inputs, linear_layers, load_model, perplexity

    model, config = load_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    calib = calibration_chunks(tokenizer, args.calib, args.seq)
    test = evaluation_chunks(tokenizer, args.test_chunks, args.seq)

    t0 = time.time()
    base_ppl = perplexity(model, test)
    print(f"ppl fp32: {base_ppl:.3f}  ({time.time() - t0:.0f} s)")

    names = linear_layers(model)
    t0 = time.time()
    inputs = collect_inputs(model, names, calib, max_rows=args.rows)
    print(f"Hessian collection over all {len(names)} layers: {time.time() - t0:.0f} s")

    L = 2 ** args.bits
    errs, total_time, numel = [], 0.0, 0
    kappas, modules = [], dict(model.named_modules())
    rounder = _rounder(args.rounding)
    for i, name in enumerate(names):
        W = modules[name].weight.detach().numpy().astype(np.float64)
        X = inputs[name]
        H = X.T @ X / X.shape[0]
        grid = parse_grid_spec(args.grid, L, W, np.diag(H), args.group)
        if args.kappa:
            yy, ww = group_layout(W, np.diag(H), args.group)
            kv = fit_scale(yy.ravel(), ww.ravel(), grid)[0]
            grid = grid * kv
        else:
            kv = 1.0
        kappas.append(kv)
        numel += W.size
        t1 = time.time()
        if args.refit:
            best, _, _ = improve_grid(W, H, grid, args.group, rounds=args.refit, rounder=rounder)
        else:
            best = rounder(W, H, grid, args.group)
        total_time += time.time() - t1
        What = reconstruct(best)
        errs.append(h_error(W, What, H))
        modules[name].weight.data = torch.tensor(What, dtype=torch.float32)
        if i % 20 == 0:
            print(f"  [{i + 1}/{len(names)}] {name} rel_err={errs[-1]:.4f}", flush=True)

    t0 = time.time()
    quant_ppl = perplexity(model, test)
    rate = ebw(args.bits, args.group, numel, codebook_bits=metadata_bits(args.grid, L))
    print(json.dumps({
        "model": args.model, "bits": args.bits, "grid": args.grid, "rounding": args.rounding,
        "refit": args.refit, "group": args.group, "ebw": round(rate, 4),
        "kappa_mean": round(float(np.mean(kappas)), 3),
        "kappa_min": round(float(np.min(kappas)), 3),
        "ppl_fp32": round(base_ppl, 3), "ppl_quantized": round(quant_ppl, 3),
        "ppl_time_s": round(time.time() - t0, 1),
        "quant_time_s": round(total_time, 1),
        "mean_layer_rel_err": round(float(np.mean(errs)), 5),
        "worst_layer_rel_err": round(float(np.max(errs)), 5),
    }, ensure_ascii=False, indent=2))
    if args.out:
        with Path(args.out).open("a", newline="") as fh:
            row = {"model": args.model, "bits": args.bits, "grid": args.grid,
                   "rounding": args.rounding, "refit": args.refit, "group": args.group,
                   "kappa_mean": float(np.mean(kappas)), "ebw": rate,
                   "ppl_fp32": base_ppl, "ppl_quantized": quant_ppl,
                   "mean_layer_rel_err": float(np.mean(errs)),
                   "worst_layer_rel_err": float(np.max(errs))}
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not Path(args.out).exists() or Path(args.out).stat().st_size == 0:
                writer.writeheader()
            writer.writerows([row])


def cmd_report(args) -> None:
    rows = list(csv.DictReader(open(args.csv)))
    table: dict[tuple, list[float]] = {}
    ebw_of: dict[tuple, float] = {}
    kappa_of: dict[tuple, list[float]] = {}
    legacy = "kappa_mode" not in rows[0]
    if legacy:
        print("warning: CSV has no kappa_mode column, the mode is guessed from the κ value "
              "(a layer whose optimal κ happens to be exactly 1 will land in absmax)")
    for r in rows:
        mode = (r.get("kappa_mode") or ("kappa" if float(r.get("kappa") or 1.0) < 0.999 else "absmax"))
        key = (int(float(r["bits"])), r["grid"], r["rounding"], int(float(r["refit"])), mode)
        table.setdefault(key, []).append(float(r["rel_err"]))
        ebw_of[key] = float(r["ebw"])
        kappa_of.setdefault(key, []).append(float(r.get("kappa") or 1.0))
    for bits in sorted({k[0] for k in table}):
        print(f"\n### {bits} bit — mean over the sampled layers (lower is better)\n")
        print("| grid | rounding | refit | step | mean κ | EBW | mean | median | worst layer |")
        print("|---|---|---|---|---|---|---|---|---|")
        items = sorted(((k, v) for k, v in table.items() if k[0] == bits),
                       key=lambda kv: np.mean(kv[1]))
        for (b, grid, rounding, refit, mode), errs in items:
            print(f"| {grid} | {rounding} | {refit} | {mode} | {np.mean(kappa_of[(b, grid, rounding, refit, mode)]):.3f} | "
                  f"{ebw_of[(b, grid, rounding, refit, mode)]:.3f} | "
                  f"{np.mean(errs):.5f} | {np.median(errs):.5f} | {np.max(errs):.5f} |")


def cmd_decode(args) -> None:
    print(json.dumps(decode_times(rows=args.rows, cols=args.cols), indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache")
    c.add_argument("--model", required=True)
    c.add_argument("--out", default="cache")
    c.add_argument("--blocks", default="", help="list of block indices or depth fractions; empty = 0, middle, last")
    c.add_argument("--calib", type=int, default=32)
    c.add_argument("--seq", type=int, default=512)
    c.add_argument("--rows", type=int, default=2048)
    c.add_argument("--group", type=int, default=128)
    c.add_argument("--dtype", default="fp32", choices=("fp32", "fp16"))
    c.set_defaults(fn=cmd_cache)

    s = sub.add_parser("sweep")
    s.add_argument("--cache", required=True)
    s.add_argument("--bits", default="2,3,4")
    s.add_argument("--grids", default=DEFAULT_GRIDS)
    s.add_argument("--rounding", default="rtn,gptq")
    s.add_argument("--refit", default="0,3")
    s.add_argument("--kappa", default="0,1", help="0=absmax scale, 1=fit the step multiplier")
    s.add_argument("--out", default="results_layers.csv")
    s.set_defaults(fn=cmd_sweep)

    q = sub.add_parser("ppl")
    q.add_argument("--model", required=True)
    q.add_argument("--bits", type=int, default=4)
    q.add_argument("--grid", default="uniform")
    q.add_argument("--rounding", default="gptq")
    q.add_argument("--refit", type=int, default=0)
    q.add_argument("--kappa", action="store_true", help="fit the step multiplier per layer")
    q.add_argument("--group", type=int, default=128)
    q.add_argument("--calib", type=int, default=32)
    q.add_argument("--test-chunks", type=int, default=48, dest="test_chunks")
    q.add_argument("--seq", type=int, default=512)
    q.add_argument("--rows", type=int, default=2048)
    q.add_argument("--out", default="")
    q.set_defaults(fn=cmd_ppl)

    r = sub.add_parser("report")
    r.add_argument("--csv", default="results_layers.csv")
    r.set_defaults(fn=cmd_report)

    d = sub.add_parser("decode")
    d.add_argument("--rows", type=int, default=1024)
    d.add_argument("--cols", type=int, default=4096)
    d.set_defaults(fn=cmd_decode)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
