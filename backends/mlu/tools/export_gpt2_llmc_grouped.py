#!/usr/bin/env python3
"""Export a Hugging Face GPT-2 safetensors checkpoint to the llmc-style .bin
layout that `example/gpt2/checkpoint_loader.cc` (LoadFromLLMC) consumes.

IMPORTANT - the layout is NOT the llm.c interleaved order. The InfiniTrain
loader reads parameters grouped by type across layers:

    header(256 x int32)                       magic=20240326, version=3 (fp32)
    wte[padded_vocab, n_embd]                 zero-padded rows included
    wpe[block_size, n_embd]
    ln_1.weight  [n_layer, n_embd]            grouped: all layers, one type
    ln_1.bias    [n_layer, n_embd]            at a time
    c_attn.weight [n_layer, 3C, C]            transposed from HF Conv1D
    c_attn.bias   [n_layer, 3C]
    attn.c_proj.weight [n_layer, C, C]        transposed
    attn.c_proj.bias   [n_layer, C]
    ln_2.weight  [n_layer, n_embd]
    ln_2.bias    [n_layer, n_embd]
    mlp.c_fc.weight    [n_layer, 4C, C]       transposed
    mlp.c_fc.bias      [n_layer, 4C]
    mlp.c_proj.weight  [n_layer, C, 4C]       transposed
    mlp.c_proj.bias    [n_layer, C]
    ln_f.weight [n_embd]
    ln_f.bias   [n_embd]

Feeding an llm.c-interleaved file silently misaligns every parameter after
h0.ln_1.weight (observed as GPT-2 step-1 loss 18.2 instead of 4.96, on both
CPU and MLU - see docs/validation.md 2026-11-09 entry).

Usage:
    python3 export_gpt2_llmc_grouped.py <model.safetensors> <output.bin>

Requires only numpy. Verified against HF openai-community/gpt2 (124M).
"""
import json
import struct
import sys

import numpy as np


def load_safetensors(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    tensors = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        assert info['dtype'] == 'F32', f'unexpected dtype {info["dtype"]} for {name}'
        end = info['data_offsets'][1] - info['data_offsets'][0]
        with open(path, 'rb') as f:
            f.seek(8 + n + info['data_offsets'][0])
            buf = f.read(end)
        tensors[name] = np.frombuffer(buf, dtype=np.float32).reshape(info['shape'])
    return tensors


def w2d(t, name):
    """HF Conv1D weight [in, out] -> [out, in]."""
    return np.ascontiguousarray(t[name].T)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, out = sys.argv[1], sys.argv[2]
    t = load_safetensors(src)
    C = t['h.0.ln_1.weight'].shape[0]
    L = len({k.split('.')[1] for k in t if k.startswith('h.')})
    T = t['wpe.weight'].shape[0]
    V = t['wte.weight'].shape[0]
    PV = (V + 127) // 128 * 128
    NH = C // 64
    print(f'C={C} L={L} T={T} V={V} padded={PV} n_head={NH}')

    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2], header[3] = 20240326, 3, T, V
    header[4], header[5], header[6], header[7] = L, NH, C, PV

    def wr(a):
        f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())

    with open(out, 'wb') as f:
        f.write(header.tobytes())
        wr(t['wte.weight'])
        wr(np.zeros((PV - V, C), dtype=np.float32))
        wr(t['wpe.weight'])
        for i in range(L):
            wr(t[f'h.{i}.ln_1.weight'])
        for i in range(L):
            wr(t[f'h.{i}.ln_1.bias'])
        for i in range(L):
            wr(w2d(t, f'h.{i}.attn.c_attn.weight'))
        for i in range(L):
            wr(t[f'h.{i}.attn.c_attn.bias'])
        for i in range(L):
            wr(w2d(t, f'h.{i}.attn.c_proj.weight'))
        for i in range(L):
            wr(t[f'h.{i}.attn.c_proj.bias'])
        for i in range(L):
            wr(t[f'h.{i}.ln_2.weight'])
        for i in range(L):
            wr(t[f'h.{i}.ln_2.bias'])
        for i in range(L):
            wr(w2d(t, f'h.{i}.mlp.c_fc.weight'))
        for i in range(L):
            wr(t[f'h.{i}.mlp.c_fc.bias'])
        for i in range(L):
            wr(w2d(t, f'h.{i}.mlp.c_proj.weight'))
        for i in range(L):
            wr(t[f'h.{i}.mlp.c_proj.bias'])
        wr(t['ln_f.weight'])
        wr(t['ln_f.bias'])
    import os
    print('wrote', out, os.path.getsize(out), 'bytes')


if __name__ == '__main__':
    main()
