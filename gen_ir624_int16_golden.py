#!/usr/bin/env python3
"""
gen_ir624_int16_golden.py — Regenerate RTL golden data for IR624 INT16 model.

Re-runs onnx_converter (with fixed tiling.py), then drives CSIM layer-by-layer
to produce bit-exact per-layer golden outputs, including Add-layer residual
inputs (input_b). Output mirrors the format consumed by
rtl/tb/e2e/test_ir624_int16_e2e.py via load_golden('ir624_int16').

Usage:
  python3 gen_ir624_int16_golden.py

Output:
  rtl/tb/golden/golden_dma_e2e/ir624_int16/
    metadata.json
    layer_XX_wgt.npy
    layer_XX_param.npy
    layer_XX_input.npy
    layer_XX_output.npy
    layer_XX_input_b.npy        (Add layers only)

SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import shutil
import subprocess
import numpy as np

# Tools path
TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

from onnx_converter import convert_model
from model_packer import pack_model, LayerConfig, OP_ELTWISE_ADD
from bin2golden import (
    read_npu1, generate_golden, pack_input_to_words, pack_output_to_words,
    n_output_words, n_full_input_words, n_input_words, elem_bytes,
)


def pack_last_tile_output(output_nhwc, layer):
    """Pack only the LAST tile's output region into uint32 words.

    The RTL stores only the last tile's output to DDR (matches
    bin2golden.n_output_words clipping). For non-tiled layers, packs the
    full output (same as pack_output_to_words).
    """
    if layer.tile_h == 0 or layer.tile_w == 0:
        return pack_output_to_words(output_nhwc, layer)

    eb = elem_bytes(layer)
    # Last tile (tile_num_h-1, tile_num_w-1) may be clipped at image border
    last_h = min(layer.tile_h, layer.out_h - (layer.tile_num_h - 1) * layer.tile_h)
    last_w = min(layer.tile_w, layer.out_w - (layer.tile_num_w - 1) * layer.tile_w)
    row_start = (layer.tile_num_h - 1) * layer.tile_h
    col_start = (layer.tile_num_w - 1) * layer.tile_w
    tile_data = output_nhwc[row_start:row_start + last_h,
                           col_start:col_start + last_w, :]

    flat = tile_data.flatten()
    if eb == 1:
        n_words = (len(flat) + 3) // 4
        padded = np.zeros(n_words * 4, dtype=np.uint8)
        padded[:len(flat)] = flat.astype(np.uint8) & 0xFF
        return padded.view('<u4')
    else:
        flat_u16 = flat.astype(np.uint16)
        n_words = (len(flat_u16) + 1) // 2
        padded = np.zeros(n_words * 2, dtype=np.uint16)
        padded[:len(flat_u16)] = flat_u16
        return padded.view('<u4')

# ─── Paths ───
MODEL_PATH = '/data/sam/ir624/WXPay_PalmIrLiveness_624_06_r20260509.onnx'
CALIB_DIR = '/data/sam/ir624/O2_for_guopeng20221209/O2_quant_datas'
INPUT_BIN = '/data/sam/ir624/debug.bin'
CSIM_PATH = os.path.join(os.path.dirname(TOOLS), 'csim', 'npu_sim')
WORK_DIR = '/tmp/ir624_int16_golden'
GOLDEN_OUT = os.path.join(os.path.dirname(TOOLS), 'rtl', 'tb', 'golden',
                          'golden_dma_e2e', 'ir624_int16')
NUM_CALIB = 100


def load_input_nchw():
    """Load debug.bin (uint8 NCHW 1x1x112x112), normalize to int16 quantized.

    The converter stores the input scale/zp in the npu1 meta; for golden
    generation we need the same int16-quantized input that CSIM consumes.
    We reuse the input.bin written by convert_model.
    """
    # The converter writes model_int16.npu1_input.bin — already quantized int16
    # NCHW. We load that.
    inp_path = os.path.join(WORK_DIR, 'model_int16.npu1_input.bin')
    raw = np.fromfile(inp_path, dtype=np.int16)
    # Shape: NCHW = 1x1x112x112
    return raw.reshape(1, 1, 112, 112)


def run_csim_full_model_dump(layers, weights_list, input_nchw, work_dir, bits=16):
    """Run CSIM on the full multi-layer model with DUMP_LAYERS env var.

    CSIM chains layer outputs internally and resolves residual_src for Add
    layers automatically. With DUMP_LAYERS=1, it writes /tmp/csim_layer_XXX.bin
    per layer (NHWC int16/int8).

    Returns: dict {layer_idx: NHWC numpy array}
    """
    # Pack full model
    model_path = os.path.join(work_dir, 'full_model.bin')
    in_path = os.path.join(work_dir, 'full_model_in.bin')
    out_path = os.path.join(work_dir, 'full_model_out.bin')
    weight_data = b''.join(weights_list)
    pack_model(layers, weight_data, model_path)

    eb = 2 if bits == 16 else 1
    dt = np.int16 if eb == 2 else np.int8
    input_nchw.astype(dt).tofile(in_path)

    # Clear any old dumps
    for f in os.listdir('/tmp'):
        if f.startswith('csim_layer_'):
            try:
                os.remove(os.path.join('/tmp', f))
            except OSError:
                pass

    env = os.environ.copy()
    env['DUMP_LAYERS'] = '1'
    result = subprocess.run(
        [CSIM_PATH, model_path, in_path, out_path],
        capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  CSIM full model failed:")
        print(f"  stdout: {result.stdout[-600:]}")
        print(f"  stderr: {result.stderr[-600:]}")
        return None

    # Load per-layer dumps
    outputs = {}
    for idx, layer in enumerate(layers):
        dump_path = f'/tmp/csim_layer_{idx:03d}.bin'
        if not os.path.exists(dump_path):
            print(f"  L{idx}: dump missing: {dump_path}")
            return None
        dt_u = np.int16 if eb == 2 else np.int8
        n_elems = layer.out_h * layer.out_w * layer.out_c
        raw = np.fromfile(dump_path, dtype=dt_u)[:n_elems]
        # CSIM dump is NHWC (matches fwrite of output.data_i16 which is NHWC)
        outputs[idx] = raw.reshape(layer.out_h, layer.out_w, layer.out_c)
    return outputs


def main():
    # Prereqs
    for path, name in [(MODEL_PATH, 'ONNX model'), (INPUT_BIN, 'debug.bin'),
                       (CALIB_DIR, 'calibration dir'), (CSIM_PATH, 'csim binary')]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    os.makedirs(WORK_DIR, exist_ok=True)
    model_bin = os.path.join(WORK_DIR, 'model_int16.npu1.bin')
    meta_npz = os.path.join(WORK_DIR, 'model_int16.npu1_meta.npz')

    # ─── Step 1: Convert model (with fixed tiling) ───
    print("=" * 60)
    print("Step 1: Convert IR624 model (INT16, fixed tiling)")
    print("=" * 60)
    convert_model(MODEL_PATH, CALIB_DIR, INPUT_BIN, model_bin,
                  input_format='int8-nchw', num_calib=NUM_CALIB, bits=16)

    # ─── Step 2: Read NPU1 binary ───
    print("\n" + "=" * 60)
    print("Step 2: Read NPU1 binary")
    print("=" * 60)
    layers, weight_blob = read_npu1(model_bin)
    print(f"  Layers: {len(layers)}")

    # Split weight_blob into per-layer weights (for single-layer CSIM)
    from bin2golden import n_wgt_words, extract_layer_weights
    # extract_layer_weights returns uint32 array; view as uint8 bytes to
    # recover the original byte representation
    weights_list = [extract_layer_weights(layers, weight_blob, i).view(np.uint8).tobytes()
                    for i in range(len(layers))]

    # ─── Step 3: Generate golden skeleton (metadata + wgt/param/placeholder) ───
    print("\n" + "=" * 60)
    print("Step 3: Generate golden skeleton")
    print("=" * 60)
    if os.path.exists(GOLDEN_OUT):
        shutil.rmtree(GOLDEN_OUT)
    input_nchw = load_input_nchw()
    # (1, 1, 112, 112) → squeeze batch → (1, 112, 112) = (C, H, W)
    input_chw = input_nchw[0]
    # NCHW (C,H,W) → NHWC (H,W,C)
    input_nhwc = input_chw.transpose(1, 2, 0)  # (112, 112, 1)

    metadata = generate_golden(layers, weight_blob, input_nhwc, GOLDEN_OUT,
                               base_ddr_addr=0x30000000, layer_offset=0x00010000)

    # ─── Step 4: Add input_b DDR addresses for Add layers ───
    # For each Add layer, allocate a region for residual input_b within
    # the Add layer's own DDR span. Layout: [input | input_b | weights | params | output]
    # We reserve space between input and weights for input_b.
    print("\n" + "=" * 60)
    print("Step 4: Patch Add-layer ddr_add_b_addr")
    print("=" * 60)
    align = 4096
    for idx, (layer, meta) in enumerate(zip(layers, metadata)):
        if layer.op_type != OP_ELTWISE_ADD:
            continue
        # input_b size = same as input size for Add layer.
        # For tiled layers, input_b is also tiled → full tiled size.
        # For non-tiled, full input size.
        if layer.tile_h > 0 and layer.tile_w > 0:
            num_tiles = layer.tile_num_h * layer.tile_num_w
            in_b_size = n_input_words(layer) * 4 * num_tiles
            in_a_size = in_b_size  # input_a same layout as input_b
        else:
            n_in_b = n_full_input_words(layer)
            in_b_size = n_in_b * 4
            in_a_size = in_b_size
        in_b_size_align = ((in_b_size + align - 1) // align) * align
        # Insert a gap for input_b after input region; shift wgt/param/out
        # addresses up by in_b_size_align
        ddr_in = meta['ddr_in_addr']
        ddr_wgt_old = meta['ddr_wgt_addr']
        ddr_param_old = meta['ddr_param_addr']
        ddr_out_old = meta['ddr_out_addr']
        in_a_size_align = ((in_a_size + align - 1) // align) * align
        ddr_add_b = ddr_in + in_a_size_align
        meta['ddr_add_b_addr'] = ddr_add_b
        meta['ddr_wgt_addr'] = ddr_wgt_old + in_b_size_align
        meta['ddr_param_addr'] = ddr_param_old + in_b_size_align
        meta['ddr_out_addr'] = ddr_out_old + in_b_size_align
        print(f"  L{idx:2d} Add: add_b_addr=0x{ddr_add_b:08X} "
              f"residual_src=L{layer.residual_src}")

    # ─── Step 5: Run CSIM full-model dump, patch input/output ───
    print("\n" + "=" * 60)
    print("Step 5: Run CSIM full-model (DUMP_LAYERS), patch input/output")
    print("=" * 60)
    csim_outputs = run_csim_full_model_dump(layers, weights_list,
                                            input_nchw[0], WORK_DIR, bits=16)
    if csim_outputs is None:
        print("  CSIM full-model failed — aborting")
        sys.exit(1)
    print(f"  CSIM dumps loaded for {len(csim_outputs)} layers")

    for idx, layer in enumerate(layers):
        # Determine this layer's input
        if layer.input_src >= 0 and layer.input_src in csim_outputs:
            inp_nhwc = csim_outputs[layer.input_src]
        elif idx > 0 and (idx - 1) in csim_outputs:
            inp_nhwc = csim_outputs[idx - 1]
        else:
            inp_nhwc = input_nhwc

        out_nhwc = csim_outputs[idx]

        # Pack and save input/output .npy
        # pack_output_to_words auto-detects per_tile_store from sched_ctrl:
        #   PTS layers → full NHWC output; non-PTS tiled → last-tile-only
        inp_words = pack_input_to_words(inp_nhwc, layer)
        out_words = pack_output_to_words(out_nhwc, layer)
        np.save(os.path.join(GOLDEN_OUT, f'layer_{idx:02d}_input.npy'), inp_words)
        np.save(os.path.join(GOLDEN_OUT, f'layer_{idx:02d}_output.npy'), out_words)

        print(f"  L{idx:2d} op={layer.op_type}: in={inp_nhwc.shape} "
              f"out={out_nhwc.shape} n_out_words_packed={len(out_words)} "
              f"meta_n_out={metadata[idx]['n_output_words']}")

        # For Add layers: save input_b (from residual_src's output)
        if layer.op_type == OP_ELTWISE_ADD and layer.residual_src >= 0:
            src_out = csim_outputs.get(layer.residual_src)
            if src_out is None:
                print(f"  L{idx:2d} Add: residual_src=L{layer.residual_src} "
                      f"output not available!")
                sys.exit(1)
            in_b_words = pack_input_to_words(src_out, layer)
            np.save(os.path.join(GOLDEN_OUT, f'layer_{idx:02d}_input_b.npy'),
                    in_b_words)
            print(f"  L{idx:2d} Add: saved input_b from L{layer.residual_src} "
                  f"({len(in_b_words)} words)")

    # ─── Step 6: Rewrite metadata.json with patched ddr_add_b_addr ───
    import json
    with open(os.path.join(GOLDEN_OUT, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # ─── Summary ───
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Golden dir: {GOLDEN_OUT}")
    print(f"  Layers: {len(layers)}")
    tiled = [(i, l) for i, l in enumerate(layers) if l.tile_h > 0]
    print(f"  Tiled layers: {len(tiled)}")
    for i, l in tiled:
        print(f"    L{i:2d} {l.tile_h}x{l.tile_w}@{l.tile_num_h}x{l.tile_num_w}")
    add_layers = [i for i, l in enumerate(layers) if l.op_type == OP_ELTWISE_ADD]
    print(f"  Add layers: {len(add_layers)} → {add_layers}")

    # Verify tiling fits bank
    from hw_config import DEFAULT_HW
    act_bank_w = DEFAULT_HW.act_bank_size // 4  # bytes → words
    print(f"\n  Bank check (act_bank={act_bank_w} words):")
    ok = True
    for i, l in enumerate(layers):
        if l.tile_h == 0:
            continue
        eb = 2 if l.data_type == 1 else 1
        inp_h = (l.tile_h - 1) * l.stride_h + l.kernel_h
        inp_w = (l.tile_w - 1) * l.stride_w + l.kernel_w
        in_w = (inp_h * inp_w * l.in_c * eb + 3) // 4
        out_w = (l.tile_h * l.tile_w * l.out_c * eb + 3) // 4
        db = l.sched_ctrl & 1
        if db and (in_w + out_w) > act_bank_w:
            print(f"    L{i:2d} OVERFLOW: in={in_w}+out={out_w}={in_w+out_w} > {act_bank_w}")
            ok = False
    if ok:
        print(f"    All tiled layers fit bank ✓")


if __name__ == '__main__':
    main()
