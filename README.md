# SHAPEQ — shape-adaptive quantization

**Author:** Roman S. Pokora · Independent researcher. Preprint, November 2026.

Paper (HTML): https://stiggigy-jpg.github.io/shapeq-paper/ ·
Runtime fork: https://github.com/stiggigy-jpg/shapeq-q4lut ·
Fitted GGUFs and packs: https://huggingface.co/RomanPokora/shapeq-q4lut-models

SHAPEQ fits a per-tensor 16-level non-uniform grid with a Hessian-diagonal-weighted
objective and stores 4-bit codes plus a group scale. The method rests on classical
Lloyd–Max theory, is deterministic, needs no training or gradients, and changes only
the alphabet — so the same grid composes with GPTQ, AWQ, rotations, sparse corrections,
and mixed-precision allocation. The work is a grid-shape study on two small models and
one corpus; the LUT kernel is a reference implementation, not an optimized one. Broader
validation and kernel optimization remain open.

## Contents

| path | what |
|---|---|
| `paper-en.html` | the paper, self-contained HTML (inline CSS/SVG, no assets) |
| `4-fitting-code/` | fitter (`fit_b4.py`), evaluation driver (`run_ppl_grid.py`), `qb/` library, raw CSV/JSON |
| `4-fitting-code/results/MANIFEST.md` | maps every raw artifact to a paper table and a regeneration command |
| `pack_q4lut.py` | converts a fitted `.pt` pack to GGUF (also shipped in the model repo) |
| `FORMAT.md` | the `Q4_LUT` / `Q4_LUT64` binary format spec (also in the fork) |

## Reproduce

```sh
cd 4-fitting-code
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python fit_b4.py --model HuggingFaceTB/SmolLM-135M --engine legacy --test-chunks 96
python fit_b4.py --model Qwen/Qwen2.5-0.5B --dtype fp16 --group 64 --test-chunks 16
python run_ppl_grid.py --model HuggingFaceTB/SmolLM-135M --engine legacy --out results_ppl.csv
```

Both shipped packs and both GGUFs are hashed in `SHA256SUMS.txt` in the model repo;
`b4_qwen_shapeq.pt.json` there holds perplexity and greedy-64 anchors. Absolute
perplexity can shift in the last decimal or two with BLAS reduction order. The two
engines (`legacy`, `hist`) match the dates of the shipped packs. Hessians are collected
in one pass and reused; calibration uses WikiText-2 train, evaluation uses test.

## License

Paper and tabulated results: CC BY 4.0. Code in `4-fitting-code/` and `pack_q4lut.py`:
MIT (see `4-fitting-code/LICENSE`). GGUF weights: Apache-2.0, inherited from SmolLM-135M
and Qwen2.5-0.5B (in the model repo).
