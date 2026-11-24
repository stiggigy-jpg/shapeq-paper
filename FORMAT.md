# Q4_LUT — a fitted 16-level lookup-table quantization format

`GGML_TYPE_Q4_LUT` / `GGML_TYPE_Q4_LUT64` store a weight tensor as 4-bit codes plus a
**fitted, non-uniform 16-entry value grid per tensor** (plus a fp16 scale per weight group —
96 weights in `Q4_LUT`, 64 in `Q4_LUT64`). The grid is learned from the weights and their
Hessian diagonal (see the companion paper); the runtime here only *decodes* it, it never
fits one.

The group size is part of the type because a model's input dims must be a multiple of it:
96 fits SmolLM (768/96) and costs 4.167 bits/weight; 64 covers the rest of the common
transformers (Qwen2.5's 896/4864 are not multiples of 96) and costs 4.25 bits/weight.

This is a llama.cpp fork format: `.gguf` files in these types load only in this fork. Nothing
about the existing types or code paths changes.

## On-disk layout

Two parts: the quantized payload in the GGUF tensor data block, and the grid in GGUF
key/value metadata (read by the loader, attached to the tensor at load time **through the
existing `tensor->extra` pointer** — the fork does not change `struct ggml_tensor`).

### 1. Tensor payload — array of `block_q4_lut` / `block_q4_lut64`

```c
#define QK4_LUT   96
#define QK4_LUT64 64
typedef struct {
    ggml_half d;                   //  2 B  fp16 scale for the group
    uint8_t   qs[QK4_LUT   / 2];   // 48 B = 96 codes, 4 bits each
} block_q4_lut;                    // 50 bytes per 96 weights = 4.1667 bits/weight

typedef struct {
    ggml_half d;                   //  2 B  fp16 scale for the group
    uint8_t   qs[QK4_LUT64 / 2];   // 32 B = 64 codes, 4 bits each
} block_q4_lut64;                  // 34 bytes per 64 weights = 4.25 bits/weight
```

* Type ids: `Q4_LUT = 44` (`blck_size = 96`, `type_size = 50`), `Q4_LUT64 = 45`
  (`blck_size = 64`, `type_size = 34`), both registered in `ggml_type_traits` — so every
  size/offset computation in ggml (allocation, GGUF tensor offsets, `gguf_get_tensor_size`)
  works for the new types without special cases.
* Nibble order matches Q4_0: the **low nibble is the even weight**, high nibble the odd one.
* Code value `c` in `[0,15]`. Weight value = `lut[c] * d`.
* A tensor of `N` weights holds `N / G` blocks (`G = 96` or `64`, `N` is a multiple of it).
* The scale `d` is applied once per block. The 16 grid values themselves live in metadata,
  not in the block, so blocks carry no per-group min/max.

### 2. Tensor grid — GGUF key/value metadata (`q4lut.*`)

Written by our python packer (`gguf_write_q4lut.py`), read by `llama_model::attach_q4lut`:

| Key                       | GGUF type     | meaning |
|---------------------------|---------------|---------|
| `q4lut.n`                 | `U32`         | number of tensor records |
| `q4lut.<i>.name`          | `STRING`      | final ggml tensor name (after llama name mapping) |
| `q4lut.<i>.seg_rows`      | `ARRAY(U32)`  | row offset (`ne[1]`) where each segment starts; length = nseg |
| `q4lut.<i>.seg_grids`     | `ARRAY(F32)`  | `16 * nseg` grid values, 16 per segment |

A **segment** exists because llama fuses several source tensors into one ggml tensor
(`attn_qkv`, `ffn_gate_up`) and each source weight was fitted with its **own** grid. Rows of
the fused tensor select their segment by `row_start[]`. `nseg` is 1..4. A plain (non-fused)
tensor has `nseg = 1`.

At load time each segment's `float[16]` is copied into a model-owned pool and a
`ggml_q4lut_meta` is attached to the tensor via its existing `void * extra` field
(the repack/kleidiai tensor-traits users never claim Q4_LUT tensors, so the slot is free).

## Decoding at runtime

`ggml_mul_mat_q4_lut` intercepts `mul_mat` when `src0->type` is `GGML_TYPE_Q4_LUT` or
`GGML_TYPE_Q4_LUT64` and dispatches a dedicated row-dot (one kernel pair per group size,
generated from the same macro so the group size stays a compile-time literal):

* **AVX2 + FMA + F16C** (runtime-selected when the CPU supports it): unpack 16 codes, expand
  each to its grid value via `permutevar8x32` over the two halves of the grid, multiply by the
  group scale once, FMA into the accumulator. One unpack is reused across a pack of up to 4
  activation rows, which cuts prompt-prefill cost ~45% versus one-row-at-a-time; decode is
  unaffected and greedy output stays bit-for-bit identical to the f16-dequant reference.
* **Scalar** reference fallback otherwise.
* `src1` (activations) must be F32; F16/BF16 are widened into a workspace first.

## What this fork does **not** do

* No `from_float` / `to_float`: you cannot `llama-quantize` *into* or *out of* this type.
  Packing is done offline by the python converter, which fits the grid from a real fp model
  and writes a ready GGUF. `llama-quantize` on a q4_lut model is unsupported by design.
* No generic dequant path: `to_float`/`from_float` are NULL and the type never enters the
  shared vec_dot machinery — the `mul_mat` intercept is the only consumer.
* No GPU backends. CPU AVX2/scalar only; CUDA/Metal/SYCL have no kernel and will reject
  offload, forcing these tensors to CPU.
* `nseg` is capped at 4 (one record per ggml tensor; 4 covers llama's largest fusion).

## Building

Standard llama.cpp build, from the commit the patches apply to:

```sh
git checkout <upstream-base>          # the base commit the .patch series is against
git am patches-en/000{1,2,3}-*.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

x86-64 with a C++17 compiler. The AVX2 kernel is `target`-attributed, so the file builds
without `-mavx2` and the fast path is chosen at runtime.
