#!/usr/bin/env python3
"""Generate chain-mode golden for SoC simulation.

Problem: bin2golden.py packs per-tile input (with halo, tile-local width).
Chain mode reads from NHWC output of previous layer (full width).
This script generates golden output by running CSIM with chain inputs
and packing output as NHWC (not pre-packed tiles).

Usage:
    python3 gen_chain_golden.py model_b_int16
"""

import os
import sys
import json
import shutil
import subprocess
import numpy as np

INTERNAL = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(INTERNAL)
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)

from bin2golden import (
    read_npu1, pack_output_to_words, pack_input_to_words,
    pack_i16_to_words, pack_i8_to_words, elem_bytes,
    is_per_tile_store, n_output_words, n_input_words,
)


def run_csim_chain(model_bin, input_nchw, work_dir, bits=16):
    """Run CSIM with DUMP_LAYERS=1 to get per-layer outputs."""
    csim = os.path.join(ROOT, 'csim', 'npu_sim')
    if not os.path.exists(csim):
        print(f"ERROR: CSIM not found at {csim}")
        return None

    # Write input as raw binary
    inp_path = os.path.join(work_dir, 'input.bin')
    input_nchw.astype(np.int16 if bits == 16 else np.int8).tofile(inp_path)

    env = os.environ.copy()
    env['DUMP_LAYERS'] = '1'
    env['DUMP_DIR'] = work_dir

    cmd = [csim, model_bin, inp_path, str(bits)]
    print(f"  Running CSIM: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=work_dir)

    if result.returncode != 0:
        print(f"  CSIM failed (rc={result.returncode})")
        print(f"  stderr: {result.stderr[:500]}")
        return None

    # Load dumped layers
    outputs = {}
    for f in sorted(os.listdir(work_dir)):
        if f.startswith('csim_layer_'):
            idx = int(f[11:14])
            data = np.fromfile(os.path.join(work_dir, f), dtype=np.int16 if bits == 16 else np.int8)
            outputs[idx] = data
            print(f"  Loaded L{idx}: {len(data)} elements")

    return outputs


def main():
    if len(sys.argv) < 2:
        print("Usage: gen_chain_golden.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    model_dir = os.path.join(INTERNAL, 'models', model_name)
    model_bin = os.path.join(model_dir, 'model.bin')
    golden_dir = os.path.join(ROOT, 'rtl', 'tb', 'golden', 'golden_dma_e2e', model_name)

    if not os.path.exists(model_bin):
        print(f"ERROR: model.bin not found at {model_bin}")
        sys.exit(1)

    # Read layers
    layers, weight_blob = read_npu1(model_bin)
    print(f"Read {len(layers)} layers from {model_bin}")

    # Load input
    input_path = os.path.join(model_dir, 'input.bin')
    if not os.path.exists(input_path):
        print(f"ERROR: input.bin not found at {input_path}")
        sys.exit(1)

    # Determine input shape from L0
    l0 = layers[0]
    input_nchw = np.fromfile(input_path, dtype=np.int16 if l0.int16 else np.int8)
    input_nchw = input_nchw.reshape(l0.in_c, l0.in_h, l0.in_w)
    input_nhwc = input_nchw.transpose(1, 2, 0).copy()

    # Run CSIM to get chain outputs
    work_dir = '/tmp/chain_golden_work'
    os.makedirs(work_dir, exist_ok=True)
    csim_outputs = run_csim_chain(model_bin, input_nchw, work_dir, bits=16 if l0.int16 else 8)

    if csim_outputs is None:
        print("ERROR: CSIM failed")
        sys.exit(1)

    # Load existing metadata
    meta_path = os.path.join(golden_dir, 'metadata.json')
    with open(meta_path) as f:
        metadata = json.load(f)

    # Regenerate output .npy files using CSIM chain outputs
    # pack_output_to_words packs as NHWC (matches chain mode SoC store)
    for idx, layer in enumerate(layers):
        if idx not in csim_outputs:
            print(f"  L{idx}: no CSIM output, skipping")
            continue

        out_data = csim_outputs[idx]
        # Reshape to NHWC
        out_h = layer.out_h
        out_w = layer.out_w
        out_c = layer.out_c
        out_nhwc = out_data.reshape(out_c, out_h, out_w).transpose(1, 2, 0).copy()

        # Pack as NHWC output (matching SoC store layout)
        out_words = pack_output_to_words(out_nhwc, layer)
        np.save(os.path.join(golden_dir, f'layer_{idx:02d}_output.npy'), out_words)
        print(f"  L{idx}: saved {len(out_words)} output words (NHWC chain mode)")

    print(f"\nChain-mode golden generated in {golden_dir}")
    print(f"  Output files: layer_XX_output.npy (NHWC, matches SoC chain store)")


if __name__ == '__main__':
    main()
