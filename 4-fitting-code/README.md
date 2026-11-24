# SHAPEQ fitting code and paper artifacts

**Author:** Roman S. Pokora · Independent researcher.
Paper: `https://stiggigy-jpg.github.io/shapeq-paper/` · Fork: `https://github.com/stiggigy-jpg/shapeq-q4lut` · Models: `https://huggingface.co/RomanPokora/shapeq-q4lut-models`.
This repository is `https://github.com/stiggigy-jpg/shapeq-paper`.

This is the fitting and measurement half of the SHAPEQ paper. It fits a per-tensor
16-level non-uniform grid with a Hessian-diagonal-weighted objective, rounds with GPTQ,
packs the result (`fit_b4.py`), and regenerates the perplexity ladders
(`run_ppl_grid.py`). SHAPEQ rests on classical Lloyd–Max theory, is deterministic, needs
no training or gradients, and changes only the alphabet, so the same grid can be used
with other quantizers.

`pack_q4lut.py` lives in the model repository; it converts a fitted `.pt` to GGUF.
The `Q4_LUT` / `Q4_LUT64` decode kernels live in the fork.

## Layout

```
fit_b4.py          HF checkpoint -> Hessian cache -> grid fit + GPTQ -> b4_<model>_shapeq.pt
run_ppl_grid.py    paper's bit-width ladder on WikiText-2 (Tables 1-2; configs behind §6.14)
qb/                data, models, grids, warps (Kumaraswamy family + fitting),
                   warps_hist (histogram fitter), gptq, lloyd, bench, decode, ternary
results/           raw CSV/JSON behind every paper table, with MANIFEST.md mapping
                   each file to a table and a regeneration command
```

## Setup

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # CPU is enough; the study ran on an i7-10850H
```

## Reproduce the shipped packs

```sh
# SmolLM-135M (Table 5 / the GGUF in the HF repo). The shipped pack predates the
# histogram fitter; --engine legacy selects the original deterministic fitter.
python fit_b4.py --model HuggingFaceTB/SmolLM-135M --engine legacy --test-chunks 96

# Qwen2.5-0.5B: group 64 because 896/4864 are not divisible by 96; fp16 model
# with fp32 Hessians, matching the paper's Table 2 protocol.
python fit_b4.py --model Qwen/Qwen2.5-0.5B --dtype fp16 --group 64 --test-chunks 16

# then GGUF (packer from the model repository):
python pack_q4lut.py --pack b4_smollm_shapeq.pt --base smollm-f16.gguf \
    --out smollm-135m-q4lut.gguf
```

Shipped pack hashes are in `SHA256SUMS.txt` in the model repository. Compare them
after a rerun. Both runs build a Hessian cache (`cache_h_<model>_<rows>r/`, up to a few
GB; reused when the key matches). Expected runtime: about 20 min SmolLM, 15 min
Qwen2.5-0.5B on 6 CPU threads; peak RSS stays under ~2.5 GB. `--no-eval` skips
perplexity/greedy checks; `--pack-in <pt>` evaluates an existing pack.

## Reproduce the perplexity tables

```sh
# Table 1: SmolLM main ladder, 96 eval chunks, group 96, legacy engine.
python run_ppl_grid.py --model HuggingFaceTB/SmolLM-135M --engine legacy --out results_ppl.csv

# Table 2: Qwen2.5-0.5B arbitration, group 128, 1024 calibration rows, 64 eval chunks.
python run_ppl_grid.py --model Qwen/Qwen2.5-0.5B --dtype fp16 --group 128 --rows 1024 \
    --test-chunks 64 --out results_ppl_qwen.csv
```

Drivers append rows and skip configs already present, so runs are resumable. The
published copies in `results/` are kept as-is; write to your own `--out` to compare.

## Determinism and scope

* The legacy search is seeded per tensor from a CRC of the weight bytes, so fits do not
  depend on call order. The histogram fitter is deterministic by construction.
* `--engine hist` corresponds to artifacts fitted from 2026-11-19 on (Qwen pack,
  `* hist` CSV rows); `--engine legacy` corresponds to the 2026-11-17/18 SmolLM
  artifacts. Absolute perplexity may differ in the last decimal or two (BLAS reduction
  order); the grid-shape conclusions do not.
* Hessians are collected in one pass and not rebuilt between blocks (paper §8).
  Calibration uses WikiText-2 train; evaluation uses test; token sets are disjoint.
* Speeds in the paper are from one i7-10850H session, AVX2, 6 threads. The LUT kernel
  is a reference implementation, not an optimized one.

## Not included

* The one-off probe scripts behind the small figures. Their outputs are shipped in
  `results/`, with protocol in each caption and in `results/MANIFEST.md`.
* The GGUF quantizer/kernel; that lives in the fork (`https://github.com/stiggigy-jpg/shapeq-q4lut`), see `FORMAT.md`.

## License

Code: MIT, Copyright 2026 Roman S. Pokora.
Paper and tabulated results: CC BY 4.0. GGUF weights: Apache-2.0 (inherited).
