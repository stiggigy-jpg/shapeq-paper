# Artifact manifest — file -> paper table

Raw CSV/JSON behind the paper tables, as measured. Text fields are in English;
generated model samples are kept verbatim. Machine: i7-10850H, AVX2, 6 CPU threads
unless noted; corpus WikiText-2 (`wikitext-2-raw-v1`), eval chunks × 512 tokens.

| file | paper | contents | regeneration |
|---|---|---|---|
| `results_ppl.csv` | Table 1 (§6.1), rows reused in §6.14 | SmolLM-135M fp32 ladder, group 96, 96 eval chunks | `run_ppl_grid.py --model HuggingFaceTB/SmolLM-135M --engine legacy` |
| `results_ppl_qwen.csv` | Table 2 (§6.2) | Qwen2.5-0.5B fp16 ladder, group 128, 1024 calib rows, 64 eval chunks | `run_ppl_grid.py --model Qwen/Qwen2.5-0.5B --dtype fp16 --group 128 --rows 1024 --test-chunks 64` |
| `results_ppl_smollm_fp16gpu.csv` | Table 9 (§6.13) | SmolLM fp16-GPU base, full variant cross at equal bitness | same driver, `--dtype fp16`; fp16 base was run on the GPU harness, cross-checks below confirm CPU parity |
| `results_ppl_smollm_ipu.csv` | supplement to §6.13 | same RTN task on an independent iGPU trace; not used in the final tables | raw (OpenCL probe) |
| `results_gguf.json` + `.raw` | Table 7 (§6.11) | llama.cpp runtime: size/ppl/speed for stock quants (+imatrix) and q4_lut rows; `.raw` is the untouched bench log | stock quants: stock llama.cpp `llama-quantize` ladder + `llama-perplexity`/`llama-bench`; q4_lut: fork build + `fit_b4.py` + `pack_q4lut.py` |
| `results_layers.csv` | Table 4 (§6.8) | per-layer mean relative error tr(ΔHΔᵀ)/tr(WHWᵀ), 21 SmolLM linear tensors x configs | metric = `qb.gptq.h_error` per layer on the same caches; driver loop = `run_ppl_grid.run()` without the ppl step |
| `results_layers_qwen.csv` | supplement to §6.2 | same for Qwen2.5-0.5B | idem |
| `bench_b4.json` | Table 5 (§6.9) | true 4-bit pack: bits/weight, ppl16 fp32/packed/fake-quant, tok/s, RSS | `fit_b4.py --engine legacy` reproduces the pack and `ppl16_fakequant`; RSS/packed-decode tok/s came from the in-process `QLinear` runner (decode = `qb.decode`) |
| `bench_rco.json` | Tables 6 (§6.10), 10 (§7.8) | full GSQ/RCO mode: mixed-bit inference, allocation over our grid | Lagrangian allocator = `run_ppl_grid.lagrangian_mix`; the RCO outer loop is a probe on top (raw) |
| `three_param.json` / `three_param_qwen.json` | Table 3 (§6.4) | objective vs number of shape parameters (2/3/4-coefficient arms) | raw (probe over `qb.warps` fit arms) |
| `gridmap.json` / `gridmap_qwen.json` | §6.3 | fitted grid shapes per layer/bitness (the "shape map" figure) | `qb.fit.fit_tensor_grid` per tensor, kinds as in the driver |
| `peaks.json` | §6.7 | outlier share vs group scale | raw probe over `qb.warps.group_layout` |
| `eqmass.json` | §6.5 | equal-mass grids refutation summary | `qb.warps.eqmass_grid` + exact objective (raw summary) |
| `unsloth_dyn.json` | Table 8 (§6.12) | importance-ranking vs RCO ranking on a tensor-separable metric | raw (ranking probe; inputs are layer errors like `results_layers.csv`) |
| `results_tern_vm.csv` | Table 11 (§7.9) | ternary arms on a live readout (Qwen, remote CPU runner) | `qb.ternary` arms + ppl (driver was a VM one-off; raw) |
| `results_ternary_scan.csv` | §7.9 | ternary mass-balance parameter scan | `qb.ternary` scan (raw) |

Two shipped fitted packs (in the HF repo, next to the GGUFs) regenerate with:

```sh
python fit_b4.py --model HuggingFaceTB/SmolLM-135M --engine legacy   # -> b4_smollm_shapeq.pt
python fit_b4.py --model Qwen/Qwen2.5-0.5B --dtype fp16 --group 64   # -> b4_qwen_shapeq.pt

Compare the result against `SHA256SUMS.txt` in the model repository.
```

`ppl16_fakequant` anchors and `samples_greedy64` (EN/RU greedy-64 continuations) in the pack
`.json` files let you gate your rerun against the published GGUF behaviour.
