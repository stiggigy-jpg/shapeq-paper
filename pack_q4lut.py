# -*- coding: utf-8 -*-
"""Build a Q4_LUT GGUF: f16 base + a fitted b4 pack -> GGUF with q4_lut/q4_lut64 (types 44/45) weights.

This is the packer from the companion project, kept here as a reference for how the
shipped `smollm-135m-q4lut.gguf` was produced. It is a self-contained line-based GGUF
reader/writer (format v2/v3):

  * KV metadata and all f16/f32 tensors are copied byte-for-byte from the base f16 GGUF;
  * the linear layers come from a fitted b4 pack (produced by the project's b4 quantizer)
    and are rewritten as blocks [fp16 d | codes] with a per-pack group (96 or 64 weights);
  * each packed tensor gets `q4lut.*` KV metadata (name, seg_rows, seg_grids) that the
    llama.cpp fork's loader reads.

Group size comes from the pack: 96 -> ggml type q4_lut (44), 64 -> q4_lut64 (45). Limitation:
the input dimension must be a multiple of the group (SmolLM-135M: 96; Qwen2.5: 64).

Requires torch+numpy and a pack `.pt` whose `state` dict holds, per HF layer name, keys
`<name>.packed` (R, C/2 uint8, low nibble = even weight), `<name>.scales` (R, G fp16) and
`<name>.lut` (16,). The pack is produced by the fitting pipeline in the paper repo and is
not bundled here.

    python pack_q4lut.py --pack b4_smollm_shapeq.pt \
        --base smollm-f16.gguf --out smollm-135m-q4lut.gguf

Loads ONLY in the companion llama.cpp q4lut fork. See FORMAT.md (repo `1-github-llamacpp`).
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import torch

# the group size comes from the pack; the ggml block type follows from it (blk is part of the type)
GGML_TYPE_F32, GGML_TYPE_F16 = 0, 1
GGML_TYPE_Q4_LUT = {96: 44, 64: 45}

def type_size(blk):
    return 2 + blk // 2

(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)
SCALAR_FMT = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
              F32: "<f", BOOL: "<B", U64: "<Q", I64: "<q", F64: "<d"}
SCALAR_SIZE = {U8: 1, I8: 1, U16: 2, I16: 2, U32: 4, I32: 4, F32: 4, BOOL: 1,
               U64: 8, I64: 8, F64: 8}

MAP = {  # ggml stem -> HF linear-layer name
    "attn_q":   "self_attn.q_proj",
    "attn_k":   "self_attn.k_proj",
    "attn_v":   "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up":   "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
DROP_KV = {"general.file_type"}


# ---------------- read ----------------

class R:
    def __init__(self, data):
        self.b, self.p = data, 0

    def raw(self, n):
        v = self.b[self.p:self.p + n]
        assert len(v) == n, "gguf trunc"
        self.p += n
        return v

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64())

    def val(self, t):
        if t == ARRAY:
            et, n = self.u32(), self.u64()
            return et, [self.val(et) for _ in range(n)]
        if t == STRING:
            return self.string()
        return self.raw(SCALAR_SIZE[t])  # raw bytes — re-emitted verbatim


def read_gguf(path):
    with open(path, "rb") as f:
        data = f.read()
    r = R(data)
    assert r.raw(4) == b"GGUF", "not GGUF"
    ver = r.u32()
    n_tensors, n_kv = r.u64(), r.u64()
    kvs = []
    for _ in range(n_kv):
        key = r.string().decode()
        t = r.u32()
        kvs.append((key, t, r.val(t)))
    tensors = []  # name, dims(list), dtype, offset
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        dt = r.u32()
        off = r.u64()
        tensors.append([name, dims, dt, off])
    align = 32
    for key, t, v in kvs:
        if key == "general.alignment":
            align = struct.unpack("<I", v)[0]
    data_start = (r.p + align - 1) // align * align
    return ver, kvs, tensors, data[data_start:], align


# ---------------- write ----------------

def enc_string(s):
    b = s if isinstance(s, bytes) else s.encode()
    return struct.pack("<Q", len(b)) + b


def enc_val(t, v):
    if t == ARRAY:
        et, items = v
        out = struct.pack("<I", et) + struct.pack("<Q", len(items))
        return out + b"".join(enc_val(et, x) for x in items)
    if t == STRING:
        return enc_string(v)
    return v if isinstance(v, bytes) else struct.pack(SCALAR_FMT[t], v)


def hf_name(ggml_name):
    s = ggml_name.decode()
    if not s.startswith("blk.") or not s.endswith(".weight"):
        return None
    rest = s[4:]
    li, rest = rest.split(".", 1)
    stem = rest[: -len(".weight")]
    return f"model.layers.{li}.{MAP[stem]}" if stem in MAP else None


def pack_tensor(st, hf, blk):
    packed = st[hf + ".packed"].numpy()                # (R, C/2) uint8, low nibble = even weight
    scales = st[hf + ".scales"].numpy()                # (R, G) fp16
    lut = st[hf + ".lut"].numpy().astype(np.float32)   # (16,)
    Rr, C2 = packed.shape
    C, G = 2 * C2, scales.shape[1]
    assert C == G * blk, f"{hf}: in-dim {C} not a multiple of group {blk}"
    out = np.empty((Rr, G, type_size(blk)), dtype=np.uint8)
    out[:, :, 0:2] = np.frombuffer(scales.astype("<f2").tobytes(), np.uint8).reshape(Rr, G, 2)
    out[:, :, 2:] = packed.reshape(Rr, G, blk // 2)
    return out.reshape(-1).tobytes(), lut, Rr, C


def detect_qk_permuted(tensors, blob, st, grp, n_head):
    """Is the f16 base storing q/k rows interleaved for RoPE (llama arch: yes, qwen2: no)?
    Decide from the first attn_q tensor: cos between the base's row 1 and the pack's row 1 vs
    mid-head row (the pack is always in HF row order)."""
    ent = next((t for t in tensors if t[0] == b"blk.0.attn_q.weight"), None)
    assert ent is not None and ent[2] == GGML_TYPE_F16, "need an f16 base with blk.0.attn_q.weight"
    ne0, ne1, off = int(ent[1][0]), int(ent[1][1]), int(ent[3])
    def brow(r):
        return np.frombuffer(blob, np.float16, count=ne0, offset=off + r*ne0*2).astype(np.float32)
    key = "model.layers.0.self_attn.q_proj"
    packed = st[key + ".packed"].numpy()
    scales = st[key + ".scales"].numpy().astype(np.float32)
    lut = st[key + ".lut"].numpy().astype(np.float32)
    def prow(r):
        lo = (packed[r] & 15).astype(np.int64); hi = (packed[r] >> 4).astype(np.int64)
        idx = np.empty(ne0, np.int64); idx[0::2] = lo; idx[1::2] = hi
        return lut[idx] * np.repeat(scales[r], grp)
    hd = ne1 // n_head
    b1 = brow(1); c = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    c_plain, c_perm = c(b1, prow(1)), c(b1, prow(hd // 2))
    assert max(c_plain, c_perm) > 0.5, f"row-order autodetect inconclusive (cos {c_plain:.3f}/{c_perm:.3f})"
    return c_perm > c_plain


def rope_row_perm(n_rows, head_dim):
    """ggml q/k tensor rows = HF rows in order [0,32,1,33,...] within each head
    (convert_hf_to_gguf reorders q/k for llama.cpp's interleaved RoPE)."""
    idx = []
    for h0 in range(0, n_rows, head_dim):
        idx += [h0 + j // 2 + (j % 2) * (head_dim // 2) for j in range(head_dim)]
    return idx[:n_rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pk = torch.load(args.pack, map_location="cpu", weights_only=False)
    st, grp = pk["state"], int(pk["group"])
    assert grp in GGML_TYPE_Q4_LUT, f"group {grp}: no ggml type registered (have 96->44, 64->45)"
    dtype_q4lut = GGML_TYPE_Q4_LUT[grp]
    ver, kvs, tensors, blob, align = read_gguf(args.base)
    kv = {k: v for k, t, v in kvs}
    arch = kv["general.architecture"].decode()
    n_head = struct.unpack("<I", kv[f"{arch}.attention.head_count"])[0]
    n_head_kv = struct.unpack("<I", kv[f"{arch}.attention.head_count_kv"])[0]

    qk_permuted = detect_qk_permuted(tensors, blob, st, grp, n_head)
    print(f'q/k rows: {"interleaved (permuted)" if qk_permuted else "HF order (no permutation)"}')

    # which tensors we can pack
    pack_map = {}
    for name, dims, dt, _ in tensors:
        hf = hf_name(name)
        if hf is None or hf + ".packed" not in st:
            continue
        C, Rr = int(dims[0]), int(dims[1])
        stv = st
        if qk_permuted and (name.endswith(b"attn_q.weight") or name.endswith(b"attn_k.weight")):
            heads = n_head if name.endswith(b"attn_q.weight") else n_head_kv
            hd = Rr // heads
            perm = rope_row_perm(Rr, hd)
            stv = dict(st)  # reorder rows into ggml RoPE order
            for suf in ("packed", "scales"):
                stv[hf + "." + suf] = st[hf + "." + suf][perm]
        raw, lut, Rp, Cp = pack_tensor(stv, hf, grp)
        assert (Rp, Cp) == (Rr, C), f"{name}: pack {(Rp, Cp)} != ggml ({Rr}, {C})"
        pack_map[name] = (raw, lut)

    # KV: base (minus file_type) + q4lut
    out_kv = [(k, t, v) for (k, t, v) in kvs if k not in DROP_KV]
    n = 0
    for name, _dims, _dt, _o in tensors:
        if name not in pack_map:
            continue
        lut = pack_map[name][1]
        out_kv.append((f"q4lut.{n}.name".encode(), STRING, name))  # name already bytes
        out_kv.append((f"q4lut.{n}.seg_rows".encode(), ARRAY, (U32, [struct.pack("<I", 0)])))
        grids = lut.astype("<f4").tobytes()
        out_kv.append((f"q4lut.{n}.seg_grids".encode(), ARRAY,
                       (F32, [grids[i * 4:i * 4 + 4] for i in range(16)])))
        n += 1
    out_kv.append((b"q4lut.n", U32, struct.pack("<I", n)))
    print(f"q4_lut: {n} tensors out of {len(tensors)}")

    # end of each source range: the next tensor by offset
    offs_sorted = sorted({t[3] for t in tensors})
    end_of = {o: (offs_sorted[i + 1] if i + 1 < len(offs_sorted) else len(blob))
              for i, o in enumerate(offs_sorted)}

    out_tensors, new_blob, cur = [], bytearray(), 0
    for name, dims, dt, off in tensors:
        if name in pack_map:
            raw, _ = pack_map[name]
            dt = dtype_q4lut
            nbytes = len(raw)
        else:
            end = end_of[off]
            if dt in (GGML_TYPE_F32, GGML_TYPE_F16):
                nel = 1
                for d in dims:
                    nel *= int(d)
                end = off + nel * (4 if dt == GGML_TYPE_F32 else 2)
            raw = blob[off:end]
            nbytes = len(raw)
        cur = (cur + align - 1) // align * align
        out_tensors.append((name, dims, dt, cur))
        new_blob += raw
        cur += nbytes

    with open(args.out, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", ver) + struct.pack("<Q", len(out_tensors))
                + struct.pack("<Q", len(out_kv)))
        for k, t, v in out_kv:
            f.write(enc_string(k) + struct.pack("<I", t) + enc_val(t, v))
        for name, dims, dt, off in out_tensors:
            f.write(enc_string(name) + struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", dt) + struct.pack("<Q", off))
        pad = (-f.tell()) % align
        f.write(b"\x00" * pad)
        f.write(new_blob)
    print(f"done: {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
