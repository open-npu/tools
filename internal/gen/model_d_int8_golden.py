#!/usr/bin/env python3
"""
Generate RTL golden data for MODEL_D (palm vein recognition, 112x112x1, 24 layers).
Uses onnx_converter + CSIM + bin2golden.

INT8 variant — same flow as INT16 but bits=8 and CSIM uses int8 dumps.
"""
import os, sys, subprocess, struct, shutil
import numpy as np

# Add tools root to path
_TOOLS_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_INTERNAL = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _TOOLS_ROOT)
sys.path.insert(0, _INTERNAL)

from bin2golden import (
    read_npu1, generate_golden, pack_input_to_words, pack_output_to_words,
    n_output_words, n_input_words, n_wgt_words, elem_bytes, OP_ELTWISE_ADD,
)
from onnx_converter import convert_model

ONNX_PATH = '/data/sam/onnx_quant/model.onnx'
CALIB_DIR = '/data/sam/model_b/O2_for_guopeng20221209/O2_quant_datas'
WORK_DIR = '/tmp/model_d_int8_golden'
GOLDEN_OUT = '/data/sam/open-npu/rtl/tb/golden/golden_dma_e2e/model_d_int8'


def run_csim(model_bin, input_bin, work_dir, num_layers, bits=8):
    """Run CSIM with DUMP_LAYERS=1 and collect per-layer outputs."""
    csim_exe = '/data/sam/open-npu/csim/npu_sim'
    out_path = os.path.join(work_dir, 'output.bin')

    # Clean old dumps
    for f in os.listdir('/tmp'):
        if f.startswith('csim_layer_'):
            os.remove(os.path.join('/tmp', f))

    print(f"  CSIM: {csim_exe} {model_bin} {input_bin} {out_path}")
    env = os.environ.copy()
    env['DUMP_LAYERS'] = '1'
    env['ACC_WIDTH'] = '44'  # Match SoC RTL config (required for INT8)
    result = subprocess.run([csim_exe, model_bin, input_bin, out_path],
                            capture_output=True, text=True, timeout=600, env=env)
    print(f"  CSIM exit code: {result.returncode}")
    if result.returncode != 0:
        print(f"  CSIM stderr: {result.stderr[-500:]}")
        return None

    # Load dumps
    dt_u = np.int8 if bits == 8 else np.int16
    outputs = {}
    for idx in range(num_layers):
        dump_path = f'/tmp/csim_layer_{idx:03d}.bin'
        if os.path.exists(dump_path):
            outputs[idx] = np.fromfile(dump_path, dtype=dt_u)
    return outputs


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(GOLDEN_OUT, exist_ok=True)

    # Step 1: Convert ONNX → NPU1
    print("=== Step 1: Convert ONNX model ===")
    model_bin = os.path.join(WORK_DIR, 'model_d_int8.npu1.bin')
    if os.environ.get('REUSE_NPU1') != '1' or not os.path.exists(model_bin):
        # Use first calibration image as input
        import glob
        calib_images = sorted(glob.glob(os.path.join(CALIB_DIR, '**', '*.jpg'), recursive=True))
        input_image = calib_images[0] if calib_images else None
        convert_model(
            model_path=ONNX_PATH,
            calib_dir=CALIB_DIR,
            input_path=input_image,
            output_path=model_bin,
            input_format='int8-nchw',
            num_calib=20,
            bits=8,
            ptq_method='mse',
        )
    else:
        print(f"  Reusing model by request: {model_bin}")

    input_bin = model_bin.replace('.npu1.bin', '.npu1_input.bin')

    # Step 2: Read NPU1 binary
    print("\n=== Step 2: Read NPU1 binary ===")
    layers, weight_blob = read_npu1(model_bin)
    print(f"  Layers: {len(layers)}")

    # Step 3: Run CSIM with layer dumps
    print("\n=== Step 3: Run CSIM ===")
    csim_outputs = run_csim(model_bin, input_bin, WORK_DIR, len(layers), bits=8)
    if csim_outputs is None:
        print("  CSIM failed!")
        sys.exit(1)
    print(f"  CSIM dumps loaded for {len(csim_outputs)} layers")

    # Step 4: Generate base golden data (weights, params, metadata)
    print("\n=== Step 4: Generate base golden data ===")
    # Parse input
    l0 = layers[0]
    input_raw = np.fromfile(input_bin, dtype=np.uint8)
    input_t = input_raw.view(np.int8)
    expected = l0.in_c * l0.in_h * l0.in_w
    input_t = input_t[:expected].reshape(l0.in_c, l0.in_h, l0.in_w)
    input_nhwc = np.transpose(input_t, (1, 2, 0))
    print(f"  Input: {input_nhwc.shape} dtype={input_nhwc.dtype}")

    generate_golden(layers, weight_blob, input_nhwc, GOLDEN_OUT)

    # Step 5: Patch metadata
    print("\n=== Step 5: Patch metadata ===")
    meta_path = os.path.join(GOLDEN_OUT, 'metadata.json')
    import json
    with open(meta_path) as f:
        md = json.load(f)

    # 5a: Add wgt_per_oc_words
    for i, m in enumerate(md):
        if m['op_type'] == 0 or m['op_type'] == 2:
            wgt_words = m['dma_wgt_size'] // 4
            if wgt_words > 20480:
                kd = m.get('kernel_h', 1) * m.get('kernel_w', 1) * m['in_c']
                eb = 2 if m.get('data_type', 0) == 1 else 1  # bytes per element
                per_oc = (16 * kd * eb + 3) // 4
                if per_oc > 20480:  # 16-ch block overflows WGT SRAM (e.g. 3x3x512 int16): use 8-ch sub-groups
                    per_oc = (8 * kd * eb + 3) // 4
                m['wgt_per_oc_words'] = per_oc
                print(f"  L{i}: wgt overflow {wgt_words} > 20480, per_oc={per_oc}")
            else:
                m['wgt_per_oc_words'] = 0
        else:
            m['wgt_per_oc_words'] = 0

    # 5b: Patch Add-layer ddr_add_b_addr
    align = 4096
    for idx, (layer, meta) in enumerate(zip(layers, md)):
        if layer.op_type != OP_ELTWISE_ADD:
            continue
        # Add (op_type=4) needs add_b
        if layer.tile_h > 0:
            num_tiles = max(1, layer.tile_num_h) * max(1, layer.tile_num_w)
            in_a_size = n_input_words(layer) * 4 * num_tiles
            in_b_size = in_a_size
        else:
            in_b_size = layer.in_h * layer.in_w * layer.in_c * elem_bytes(layer)
            in_a_size = in_b_size
        in_b_size_align = ((in_b_size + align - 1) // align) * align
        in_a_size_align = ((in_a_size + align - 1) // align) * align
        ddr_in = meta['ddr_in_addr']
        meta['ddr_add_b_addr'] = ddr_in + in_a_size_align
        meta['ddr_wgt_addr'] = meta['ddr_wgt_addr'] + in_b_size_align
        meta['ddr_param_addr'] = meta['ddr_param_addr'] + in_b_size_align
        meta['ddr_out_addr'] = meta['ddr_out_addr'] + in_b_size_align
        print(f"  L{idx} Add: add_b_addr=0x{meta['ddr_add_b_addr']:08X}")

    with open(meta_path, 'w') as f:
        json.dump(md, f, indent=2)

    # Step 6: Pack outputs and input_b from CSIM dumps
    print("\n=== Step 6: Pack outputs from CSIM dumps ===")
    for i, m in enumerate(md):
        if i not in csim_outputs:
            print(f"  L{i}: no CSIM dump (fused block intermediate), skipping")
            continue

        out_raw = csim_outputs[i].astype(np.int8)
        out_c, out_h, out_w = m['out_c'], m['out_h'], m['out_w']
        if len(out_raw) == out_h * out_w * out_c:
            out_nhwc = out_raw.reshape(out_h, out_w, out_c)
        else:
            out_nhwc = out_raw

        # Determine input source
        layer = layers[i]
        input_src = m.get('input_src', -1)

        # For Add: input_a comes from previous layer (NOT residual_src)
        # residual_src is only used for input_b (saved separately below)
        if input_src >= 0 and input_src in csim_outputs:
            inp_raw = csim_outputs[input_src].astype(np.int8)
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
        elif i == 0:
            inp_nhwc = input_nhwc
        else:
            # Fused block end: input is from previous fused block or earlier layer
            # Find nearest available layer with matching input dimensions
            inp_c, inp_h, inp_w = m['in_c'], m['in_h'], m['in_w']
            expected_elems = inp_h * inp_w * inp_c
            inp_nhwc = None
            for j in range(i - 1, -1, -1):
                if j in csim_outputs and len(csim_outputs[j]) == expected_elems:
                    inp_raw = csim_outputs[j].astype(np.int8)
                    inp_nhwc = inp_raw.reshape(inp_h, inp_w, inp_c)
                    break
            if inp_nhwc is None:
                print(f"  L{i}: WARNING — no matching input found (need {expected_elems} elems), skipping")
                continue

        pts = True if m.get('store_mode', 0) & 1 else None
        inp_words = pack_input_to_words(inp_nhwc, layer)
        out_words = pack_output_to_words(out_nhwc, layer, per_tile_store=pts)
        np.save(os.path.join(GOLDEN_OUT, f'layer_{i:02d}_input.npy'), inp_words)
        np.save(os.path.join(GOLDEN_OUT, f'layer_{i:02d}_output.npy'), out_words)

        # For Add: save input_b (residual)
        if m['op_type'] == 4 and m.get('residual_src', -1) >= 0:
            src_out = csim_outputs.get(m['residual_src'])
            if src_out is not None:
                src_out = src_out.astype(np.int8)
                if len(src_out) == m['in_h'] * m['in_w'] * m['in_c']:
                    in_b_words = pack_input_to_words(
                        src_out.reshape(m['in_h'], m['in_w'], m['in_c']), layer)
                else:
                    in_b_words = pack_input_to_words(src_out, layer)
                np.save(os.path.join(GOLDEN_OUT, f'layer_{i:02d}_input_b.npy'), in_b_words)
                print(f"  L{i}: input_b saved ({len(in_b_words)} words)")

        nw = m['n_output_words']
        status = "OK" if len(out_words) == nw else f"MISMATCH ({len(out_words)} vs {nw})"
        print(f"  L{i}: output {len(out_words)} words (expected {nw}) {status}")

    with open(meta_path, 'w') as f:
        json.dump(md, f, indent=2)

    print(f"\nDone! {len(md)} layers saved to {GOLDEN_OUT}")


if __name__ == '__main__':
    main()
