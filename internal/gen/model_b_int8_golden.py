#!/usr/bin/env python3
"""
gen_model_b_int8_golden.py — Regenerate RTL golden data for MODEL_B INT8 model.

Re-runs onnx_converter (with fixed tiling.py), then drives CSIM layer-by-layer
to produce bit-exact per-layer golden outputs, including Add-layer residual
inputs (input_b). Output mirrors the format consumed by
rtl/tb/e2e/test_model_b_int8_e2e.py via load_golden('model_b_int8').

Usage:
  python3 gen_model_b_int8_golden.py

Output:
  rtl/tb/golden/golden_dma_e2e/model_b_int8/
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
TOOLS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
INTERNAL = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, TOOLS)
sys.path.insert(0, INTERNAL)

from onnx_converter import convert_model
from model_packer import pack_model, LayerConfig, OP_ELTWISE_ADD
from hw_config import DEFAULT_HW
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


def pack_conv_pool_fuse_strips(input_nhwc, conv, pool_strip_h=6):
    """Pack padded L0 input strips for firmware-driven Conv→Pool fusion.

    A pool strip of ``pool_strip_h`` rows consumes ``2*h+1`` L0 rows.
    Recomputing the one-row overlap lets each L0 invocation produce one
    contiguous SRAM tensor that the existing untiled fusion path can hand
    directly to L1.
    """
    producer_h = 2 * pool_strip_h + 1
    producer_w = conv.out_w
    input_h = (producer_h - 1) * conv.stride_h + conv.kernel_h
    input_w = (producer_w - 1) * conv.stride_w + conv.kernel_w
    strip_count = (28 + pool_strip_h - 1) // pool_strip_h
    pad_value = -conv.in_zp

    # Padded coordinates make producer output (0,0) start at padded (0,0).
    # Extend the bottom for the fixed-size final producer strip; L1 clips that
    # strip to the four remaining output rows and never consumes the excess.
    strip_input_step = pool_strip_h * 2 * conv.stride_h
    padded_h = max((strip_count - 1) * strip_input_step + input_h,
                   conv.pad_top + conv.in_h)
    padded_w = max(input_w, conv.pad_left + conv.in_w)
    padded = np.full((padded_h, padded_w, conv.in_c), pad_value,
                     dtype=input_nhwc.dtype)
    padded[conv.pad_top:conv.pad_top + conv.in_h,
           conv.pad_left:conv.pad_left + conv.in_w, :] = input_nhwc

    strips = []
    for strip in range(strip_count):
        row = strip * strip_input_step
        flat = padded[row:row + input_h, :input_w, :].reshape(-1)
        packed = np.zeros((flat.size + 3) // 4 * 4, dtype=np.uint8)
        packed[:flat.size] = flat.astype(np.uint8)
        strips.append(packed.view('<u4'))
    return np.concatenate(strips)

# ─── Paths ───
MODEL_PATH = '/data/sam/model_b/WXPay_PalmIrLiveness_624_06_r20260509.onnx'
CALIB_DIR = '/data/sam/model_b/O2_for_guopeng20221209/O2_quant_datas'
INPUT_BIN = '/data/sam/model_b/debug.bin'
CSIM_PATH = os.path.join(os.path.dirname(TOOLS), 'csim', 'npu_sim')
WORK_DIR = '/tmp/model_b_int8_golden'
GOLDEN_OUT = os.path.join(os.path.dirname(TOOLS), 'rtl', 'tb', 'golden',
                          'golden_dma_e2e', 'model_b_int8')
NUM_CALIB = 100


def load_input_nchw():
    """Load debug.bin (uint8 NCHW 1x1x112x112), normalize to int8 quantized.

    The converter stores the input scale/zp in the npu1 meta; for golden
    generation we need the same int8-quantized input that CSIM consumes.
    We reuse the input.bin written by convert_model.
    """
    # The converter writes model_int8.npu1_input.bin — already quantized int8
    # NCHW. We load that.
    inp_path = os.path.join(WORK_DIR, 'model_int8.npu1_input.bin')
    raw = np.fromfile(inp_path, dtype=np.int8)
    # Shape: NCHW = 1x1x112x112
    return raw.reshape(1, 1, 112, 112)


def run_csim_full_model_dump(layers, weights_list, input_nchw, work_dir, bits=8):
    """Run CSIM on the full multi-layer model with DUMP_LAYERS env var.

    CSIM chains layer outputs internally and resolves residual_src for Add
    layers automatically. With DUMP_LAYERS=1, it writes /tmp/csim_layer_XXX.bin
    per layer (NHWC int8/int16).

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
    env['ACC_WIDTH'] = '44'  # Match SoC RTL config
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
    model_bin = os.path.join(WORK_DIR, 'model_int8.npu1.bin')
    meta_npz = os.path.join(WORK_DIR, 'model_int8.npu1_meta.npz')

    # ─── Step 1: Convert model (with fixed tiling) ───
    print("=" * 60)
    print("Step 1: Convert MODEL_B model (INT8, fixed tiling)")
    print("=" * 60)
    convert_model(MODEL_PATH, CALIB_DIR, INPUT_BIN, model_bin,
                  input_format='int8-nchw', num_calib=NUM_CALIB, bits=8)

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
    fuse_input = pack_conv_pool_fuse_strips(input_nhwc, layers[0])
    np.save(os.path.join(GOLDEN_OUT, 'layer_00_fuse_input.npy'), fuse_input)
    print(f"  L0→L1 fuse input: {len(fuse_input)} words "
          f"({len(fuse_input) // 5} words/strip × 5)")

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
                                            input_nchw[0], WORK_DIR, bits=8)
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

    # Fusion changes only RTL scheduling, not the mathematical graph.  Stamp
    # it after CSIM because the software model intentionally rejects hardware
    # SRAM-lifetime flags in a full-model run.
    fuse_pairs = ((13, 14), (16, 17), (20, 21), (23, 24))
    for start, end in fuse_pairs:
        a, b = layers[start], layers[end]
        if a.tile_h != 0 or b.tile_h != 0 or b.input_src >= 0:
            raise RuntimeError(f"L{start}->L{end} is not a safe untiled chain")
        a.sched_ctrl |= 0x02  # FUSE_START
        b.sched_ctrl |= 0x08  # FUSE_END
        metadata[start]['sched_ctrl'] = a.sched_ctrl
        metadata[end]['sched_ctrl'] = b.sched_ctrl
    print(f"  SRAM fuse pairs: {list(fuse_pairs)}")

    retiled = plan_db_tiles(layers, metadata)
    if retiled:
        for idx, th, tw, old_t, new_t in retiled:
            print(f"  L{idx:2d}: re-tiled to {th}x{tw} "
                  f"(input words {old_t} -> {new_t})")
        # tile_in_size is the halo input window of one tile, word-rounded the
        # same way the packer and DMA expect (n_input_words), not the tile
        # area: the DMA loads ((tile_h-1)*stride_h + kernel_h) rows.
        for idx, _, _, _, _ in retiled:
            m = metadata[idx]
            eb = 2 if m['data_type'] == 1 else 1
            th, tw = m['tile_h'], m['tile_w']
            if m['op_type'] == 3:
                pc = m.get('pool_cfg', 0)
                kh, kw = (pc >> 4) & 0xF, (pc >> 8) & 0xF
                sh, sw = (pc >> 12) & 0xF, (pc >> 16) & 0xF
            elif m['op_type'] == 4:
                kh = kw = sh = sw = 1
            else:
                kh, kw = m['kernel_h'], m['kernel_w']
                sh, sw = m['stride_h'], m['stride_w']
            in_h = (th - 1) * sh + kh
            in_w = (tw - 1) * sw + kw
            m['tile_in_size'] = ((in_h * in_w * m['in_c'] + 3) // 4) * 4
            m['tile_out_size'] = th * tw * m['out_c'] * eb
            m['row_cfg'] = (th << 16) | ((tw * m['out_c'] * eb + 3) // 4)
    print(f"  DB_EN layers: "
          f"{[i for i in range(len(metadata)) if metadata[i]['sched_ctrl'] & 1]}")

    # ─── Step 6: Rewrite metadata.json with patched scheduling/addresses ───
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


def plan_db_tiles(layers, metadata):
    """Pick a tile shape for each tiled layer and enable DB_EN where safe.

    Two independent wins are available here:

      1. Coverage waste. The shipped tiles are whole-bank blocks, so the grid
         overshoots the tensor badly: L11 covers 14x14 with 8x13@2x2 (52.9%
         of every tile is clipped away) and L5/L8 waste 48.4%. A shape that
         divides the output evenly fetches and stores less real data.
      2. Boundary overlap. DB_EN splits Act SRAM in half and lets the next
         tile's input prefetch while the current one computes, but the tile
         plus everything it reads must fit one half.

    Both are searched together: for every (h, w) that fits a bank half we
    score the total input words fetched across the whole layer, and take the
    shape with the fewest boundaries among the cheapest ones.
    """
    half_w = (DEFAULT_HW.act_bank_size // 2) // 4
    ofm_half_w = half_w * 2
    retiled = []
    for idx, l in enumerate(layers):
        if l.tile_h == 0 or l.tile_w == 0:
            continue
        if l.op_type not in (0, 1, 2, 3, 4):
            continue
        eb = 2 if l.data_type == 1 else 1
        m = metadata[idx]
        if l.op_type == 3:
            # Pool keeps its window in pool_cfg, not kernel/stride.
            pc = m.get('pool_cfg', 0)
            kh, kw = (pc >> 4) & 0xF, (pc >> 8) & 0xF
            sh, sw = (pc >> 12) & 0xF, (pc >> 16) & 0xF
        elif l.op_type == 4:
            kh = kw = sh = sw = 1
        else:
            kh, kw = m['kernel_h'], m['kernel_w']
            sh, sw = m['stride_h'], m['stride_w']
        if kh == 0 or kw == 0 or sh == 0 or sw == 0:
            continue
        oh, ow = m['out_h'], m['out_w']
        # Add parks both branch A and branch B in the same IFM half.
        in_cap = (half_w // 2) if l.op_type == 4 else half_w

        def cost(th, tw):
            ih = (th - 1) * sh + kh
            iw = (tw - 1) * sw + kw
            in_w = (ih * iw * m['in_c'] * eb + 3) // 4
            out_w = (th * tw * m['out_c'] * eb + 3) // 4
            nh = (oh + th - 1) // th
            nw = (ow + tw - 1) // tw
            return in_w, out_w, nh, nw

        def total_in(nh, nw, in_w):
            return nh * nw * in_w

        cur_in, cur_out, cur_nh, cur_nw = cost(l.tile_h, l.tile_w)
        cur_total = total_in(cur_nh, cur_nw, cur_in)

        # Layer 0's input is a pre-packed per-tile blob baked into the model
        # data, so its grid cannot change without repacking the input tensor.
        # Its existing 16x22 tile already fits both the IFM half (454 <= 3072)
        # and the OFM half (5632 <= 6144), so enable DB_EN in place.
        if idx == 0:
            if cur_in <= in_cap and cur_out <= ofm_half_w:
                l.sched_ctrl |= 0x01
                metadata[idx]['sched_ctrl'] = l.sched_ctrl
            continue

        best = None
        for th in range(1, oh + 1):
            for tw in range(1, ow + 1):
                in_w, out_w, nh, nw = cost(th, tw)
                if in_w > in_cap or out_w > ofm_half_w:
                    continue
                ntile = nh * nw
                if ntile < 2:        # nothing to overlap with one tile
                    continue
                # Every boundary costs a handshake plus the DMA latency that
                # overlap does not hide, and measurements show that cost is
                # per boundary rather than per byte. Rank by boundaries first
                # so the search can never land on a degenerate 1x1 grid, then
                # by fetched words.
                key = (ntile, total_in(nh, nw, in_w))
                if best is None or key < best[0]:
                    best = (key, th, tw, nh, nw)

        if best is None:
            continue
        (_, best_total), th, tw, nh, nw = best

        if (th, tw) != (l.tile_h, l.tile_w):
            l.tile_h, l.tile_w = th, tw
            l.tile_num_h, l.tile_num_w = nh, nw
            metadata[idx]['tile_h'] = th
            metadata[idx]['tile_w'] = tw
            metadata[idx]['tile_num_h'] = nh
            metadata[idx]['tile_num_w'] = nw
            retiled.append((idx, th, tw, cur_total, best_total))
        l.sched_ctrl |= 0x01
        metadata[idx]['sched_ctrl'] = l.sched_ctrl
    return retiled


if __name__ == '__main__':
    main()
