#!/usr/bin/env python3
"""
Generate RTL golden data for MODEL_A (MobileNetV2, 224x224x3, 63 layers) — UNFUSED.

Uses convert_model(no_fuse=True) so every layer gets its own independent tiling
from compute_tiling WITH the per-OC output-buffer safety clamp (design/tiling.py).
The stock model_a_int8_golden.py instead clears FUSE bits at the binary level,
which keeps the FUSED tiling (compute_fused_tiling, no clamp) — that left 4
per-OC layers (L6/L10/L14/L18, out_c=112) with tiles that overflow the DB_EN bank.
See project_md_int16_peroc_bug.md.
"""
import os, sys, glob, subprocess, numpy as np
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from bin2golden import read_npu1, generate_golden, pack_input_to_words, pack_output_to_words, n_output_words, elem_bytes

MODEL_PATH = '/data/sam/onnx_quant/MODEL_A.onnx'
CALIB_DIR = '/data/sam/onnx_quant/a3_test_images'
WORK_DIR = '/tmp/model_a_int8_golden_nofuse'
GOLDEN_OUT = '/tmp/model_a_int8_nofuse_golden'   # temp; verify then swap

def run_csim(model_bin, input_bin, work_dir, num_layers, bits=8):
    csim_exe = '/data/sam/open-npu/csim/npu_sim'
    out_path = os.path.join(work_dir, 'output.bin')
    for f in os.listdir('/tmp'):
        if f.startswith('csim_layer_'):
            os.remove(os.path.join('/tmp', f))
    env = os.environ.copy()
    env['DUMP_LAYERS'] = '1'
    env['ACC_WIDTH'] = '44'
    print(f"  CSIM: {csim_exe} {model_bin} {input_bin} {out_path}")
    result = subprocess.run([csim_exe, model_bin, input_bin, out_path],
                          capture_output=True, text=True, timeout=600, env=env)
    if result.returncode != 0:
        print(f"  CSIM stderr: {result.stderr[:500]}")
        return None
    outputs = {}
    dt_u = np.uint8 if bits == 8 else np.uint16
    for idx in range(num_layers):
        dump_path = f'/tmp/csim_layer_{idx:03d}.bin'
        if not os.path.exists(dump_path):
            print(f"  WARNING: csim_layer_{idx:03d}.bin not found")
            continue
        outputs[idx] = np.fromfile(dump_path, dtype=dt_u)
    return outputs

def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(GOLDEN_OUT, exist_ok=True)

    # Step 1: Convert with no_fuse=True (independent tiling + clamp, no FUSE bits)
    from onnx_converter import convert_model
    model_bin = os.path.join(WORK_DIR, 'model_a_int8_nofuse.npu1.bin')
    input_bin = os.path.join(WORK_DIR, 'model_a_int8_nofuse.npu1_input.bin')
    if not os.path.exists(model_bin):
        print("Step 1: Convert MODEL_A to INT8 (no_fuse=True)")
        all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))
        img = Image.open(all_imgs[0]).resize((224, 224))
        arr = np.array(img).transpose(2, 0, 1).astype(np.uint8)
        input_bin_raw = os.path.join(WORK_DIR, 'input.bin')
        arr.tofile(input_bin_raw)
        convert_model(MODEL_PATH, CALIB_DIR, input_bin_raw, model_bin,
                      input_format='int8-nchw', num_calib=50, bits=8, no_fuse=True)
    print(f"  Model (unfused): {model_bin}")

    # Step 2: Read npu1 binary
    print("Step 2: Read npu1 binary")
    layers, weight_blob = read_npu1(model_bin)
    print(f"  {len(layers)} layers, weight blob {len(weight_blob)} bytes")

    # Step 3: Prepare input
    print("Step 3: Prepare test input")
    inp_int8 = np.fromfile(input_bin, dtype=np.int8).reshape(3, 224, 224)
    input_nhwc = np.transpose(inp_int8, (1, 2, 0))
    print(f"  Input: {input_bin} ({os.path.getsize(input_bin)} bytes)")

    # Step 4: Run CSIM (unfused, with DUMP_LAYERS)
    print("Step 4: Run CSIM with DUMP_LAYERS")
    csim_outputs = run_csim(model_bin, input_bin, WORK_DIR, len(layers), bits=8)
    if csim_outputs is None:
        print("  FATAL: CSIM failed")
        return
    print(f"  CSIM dumps loaded for {len(csim_outputs)} layers")

    # Step 5: Generate golden data
    print("Step 5: Generate golden data")
    generate_golden(layers, weight_blob, input_nhwc, GOLDEN_OUT)

    # Step 5b: Patch Add-layer ddr_add_b_addr
    import json
    from bin2golden import n_input_words, OP_ELTWISE_ADD
    meta_path = os.path.join(GOLDEN_OUT, 'metadata.json')
    with open(meta_path) as f:
        md = json.load(f)
    align = 4096
    for idx, (layer, meta) in enumerate(zip(layers, md)):
        if layer.op_type != OP_ELTWISE_ADD:
            continue
        num_tiles = max(1, layer.tile_num_h) * max(1, layer.tile_num_w)
        if layer.tile_h > 0:
            in_a_size = n_input_words(layer) * 4 * num_tiles
            in_b_size = in_a_size
        else:
            in_b_size = layer.in_h * layer.in_w * layer.in_c * elem_bytes(layer)
        in_b_size_align = ((in_b_size + align - 1) // align) * align
        in_a_size_align = ((in_a_size + align - 1) // align) * align if layer.tile_h > 0 else in_b_size_align
        ddr_in = meta['ddr_in_addr']
        meta['ddr_add_b_addr'] = ddr_in + in_a_size_align
        meta['ddr_wgt_addr'] = meta['ddr_wgt_addr'] + in_b_size_align
        meta['ddr_param_addr'] = meta['ddr_param_addr'] + in_b_size_align
        meta['ddr_out_addr'] = meta['ddr_out_addr'] + in_b_size_align
        print(f"  L{idx:2d} Add: add_b_addr=0x{meta['ddr_add_b_addr']:08X}")

    # Step 5c: Patch per-tile store for DB_EN + tiled layers
    for i, m in enumerate(md):
        db_en = m['sched_ctrl'] & 1
        has_tile = m.get('tile_h', 0) > 0 and m.get('tile_w', 0) > 0
        if db_en and has_tile:
            m['sched_ctrl'] = m['sched_ctrl'] | 0x10
            m['store_mode'] = 1
            eb = 2 if m.get('post_ctrl', 0) & 0x80 else 1
            tile_h = m['tile_h']; tile_w = m['tile_w']
            row_len = (tile_w * m['out_c'] * eb + 3) // 4
            m['row_cfg'] = (tile_h << 16) | row_len
            m['tile_out_size'] = tile_h * tile_w * m['out_c'] * eb
            full_bytes = m['out_h'] * m['out_w'] * m['out_c'] * eb
            m['n_output_words'] = (full_bytes + 3) // 4
            m['dma_out_size'] = full_bytes
    with open(meta_path, 'w') as f:
        json.dump(md, f, indent=2)

    # Step 6: wgt_per_oc + pack outputs from CSIM dumps
    for i, m in enumerate(md):
        if m['op_type'] == 0 or m['op_type'] == 2:
            wgt_words = m['dma_wgt_size'] // 4
            if wgt_words > 24576:
                kd = m.get('kernel_h', 1) * m.get('kernel_w', 1) * m['in_c']
                eb = 2 if m.get('data_type', 0) == 1 else 1
                per_oc = (16 * kd * eb + 3) // 4
                if per_oc > 24576:  # 16-ch block overflows WGT SRAM (e.g. 3x3x512 int16): use 8-ch sub-groups
                    per_oc = (8 * kd * eb + 3) // 4
                m['wgt_per_oc_words'] = per_oc
                print(f"  L{i}: wgt overflow {wgt_words} > 24576, per_oc={per_oc}")
            else:
                m['wgt_per_oc_words'] = 0
        else:
            m['wgt_per_oc_words'] = 0

        if i in csim_outputs:
            out_raw = csim_outputs[i].astype(np.int8)
            out_c, out_h, out_w = m['out_c'], m['out_h'], m['out_w']
            if len(out_raw) == out_h * out_w * out_c:
                out_nhwc = out_raw.reshape(out_h, out_w, out_c)
            else:
                out_nhwc = out_raw
            if m.get('input_src', -1) >= 0 and m['input_src'] in csim_outputs:
                inp_raw = csim_outputs[m['input_src']].astype(np.int8)
                inp_c, inp_h, inp_w = m['in_c'], m['in_h'], m['in_w']
                if len(inp_raw) == inp_h * inp_w * inp_c:
                    inp_nhwc = inp_raw.reshape(inp_h, inp_w, inp_c)
                else:
                    inp_nhwc = inp_raw
            elif i > 0 and (i - 1) in csim_outputs:
                inp_raw = csim_outputs[i - 1].astype(np.int8)
                inp_c, inp_h, inp_w = m['in_c'], m['in_h'], m['in_w']
                if len(inp_raw) == inp_h * inp_w * inp_c:
                    inp_nhwc = inp_raw.reshape(inp_h, inp_w, inp_c)
                else:
                    inp_nhwc = inp_raw
            else:
                inp_nhwc = input_nhwc
            layer = layers[i]
            pts = True if m.get('store_mode', 0) & 1 else None
            inp_words = pack_input_to_words(inp_nhwc, layer)
            out_words = pack_output_to_words(out_nhwc, layer, per_tile_store=pts)
            np.save(os.path.join(GOLDEN_OUT, f'layer_{i:02d}_input.npy'), inp_words)
            np.save(os.path.join(GOLDEN_OUT, f'layer_{i:02d}_output.npy'), out_words)
            if m['op_type'] == 4 and m.get('residual_src', -1) >= 0:
                src_out = csim_outputs.get(m['residual_src'])
                if src_out is not None:
                    src_out = src_out.astype(np.int8)
                    if len(src_out) == m['in_h'] * m['in_w'] * m['in_c']:
                        in_b_words = pack_input_to_words(src_out.reshape(m['in_h'], m['in_w'], m['in_c']), layer)
                    else:
                        in_b_words = pack_input_to_words(src_out, layer)
                    np.save(os.path.join(GOLDEN_OUT, f'layer_{i:02d}_input_b.npy'), in_b_words)

    with open(meta_path, 'w') as f:
        json.dump(md, f, indent=2)
    print(f"\nDone! {len(md)} layers saved to {GOLDEN_OUT}")

if __name__ == '__main__':
    main()
