#!/usr/bin/env python3
"""
Generate RTL golden data for MODEL_A (MobileNetV2, 224x224x3, 63 layers).
Uses onnx_converter + CSIM + bin2golden.
"""
import os, sys, glob, subprocess, numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..', 'design'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bin2golden import read_npu1, generate_golden, pack_input_to_words, pack_output_to_words, n_output_words, elem_bytes

MODEL_PATH = '/data/sam/onnx_quant/MODEL_A.onnx'
CALIB_DIR = '/data/sam/onnx_quant/a3_test_images'
WORK_DIR = '/tmp/model_a_int16_golden'
GOLDEN_OUT = '/data/sam/open-npu/rtl/tb/golden/golden_dma_e2e/model_a_int16'

def run_csim(model_bin, input_bin, work_dir, num_layers, bits=16):
    """Run CSIM with DUMP_LAYERS=1 to get per-layer outputs."""
    csim_exe = '/data/sam/open-npu/csim/npu_sim'
    out_path = os.path.join(work_dir, 'output.bin')

    # Clear old dumps
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

    # Load per-layer dumps
    outputs = {}
    dt_u = np.uint16 if bits == 16 else np.uint8
    for idx in range(num_layers):
        dump_path = f'/tmp/csim_layer_{idx:03d}.bin'
        if not os.path.exists(dump_path):
            print(f"  WARNING: csim_layer_{idx:03d}.bin not found")
            continue
        raw = np.fromfile(dump_path, dtype=dt_u)
        outputs[idx] = raw
    return outputs

def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(GOLDEN_OUT, exist_ok=True)

    # Step 1: Convert model (use unfused version for per-layer dumps)
    from onnx_converter import convert_model
    model_bin_fused = os.path.join(WORK_DIR, 'model_a_int16.npu1.bin')
    model_bin = os.path.join(WORK_DIR, 'model_a_int16_unfused.npu1.bin')
    if not os.path.exists(model_bin_fused):
        print("Step 1: Convert MODEL_A to INT16")
        all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))
        img = Image.open(all_imgs[0]).resize((224, 224))
        arr = np.array(img).transpose(2, 0, 1).astype(np.uint8)
        input_bin_raw = os.path.join(WORK_DIR, 'input.bin')
        arr.tofile(input_bin_raw)
        convert_model(MODEL_PATH, CALIB_DIR, input_bin_raw, model_bin_fused,
                      input_format='int8-nchw', num_calib=50, bits=16)

    # Create unfused version if not exists
    if not os.path.exists(model_bin):
        print("Step 1b: Create unfused version (clear FUSE bits)")
        import struct, shutil
        shutil.copy2(model_bin_fused, model_bin)
        with open(model_bin, 'rb') as f:
            data = bytearray(f.read())
        magic, num_layers2, woff, wsize = struct.unpack_from('<4I', data, 0)
        off = 16  # header size
        for i in range(num_layers2):
            data[off + 48] = data[off + 48] & 0x01  # clear FUSE bits, keep DB_EN
            param_ch_count = struct.unpack_from('<H', data, off + 55)[0]
            has_lut = data[off + 57]
            has_add = data[off + 58]
            off += 62 + param_ch_count * 14
            if has_add: off += 8
            if has_lut: off += 768
        with open(model_bin, 'wb') as f:
            f.write(data)
    print(f"  Model (unfused): {model_bin}")

    # Step 2: Read npu1 binary
    print("Step 2: Read npu1 binary")
    layers, weight_blob = read_npu1(model_bin)
    print(f"  {len(layers)} layers, weight blob {len(weight_blob)} bytes")

    # Step 3: Prepare input (use converter-generated INT16 input)
    print("Step 3: Prepare test input")
    input_bin = os.path.join(WORK_DIR, 'model_a_int16.npu1_input.bin')
    if not os.path.exists(input_bin):
        # Re-run conversion to get input
        all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))
        img = Image.open(all_imgs[0]).resize((224, 224))
        arr = np.array(img).transpose(2, 0, 1).astype(np.uint8)
        input_bin_raw = os.path.join(WORK_DIR, 'input.bin')
        arr.tofile(input_bin_raw)
        convert_model(MODEL_PATH, CALIB_DIR, input_bin_raw, model_bin_fused,
                      input_format='int8-nchw', num_calib=50, bits=16)

    # Load input as NHWC for golden gen
    inp_int16 = np.fromfile(input_bin, dtype=np.int16).reshape(3, 224, 224)
    input_nhwc = np.transpose(inp_int16, (1, 2, 0))
    print(f"  Input: {input_bin} ({os.path.getsize(input_bin)} bytes)")

    # Step 4: Run CSIM (unfused, with DUMP_LAYERS)
    print("Step 4: Run CSIM with DUMP_LAYERS")
    csim_outputs = run_csim(model_bin, input_bin, WORK_DIR, len(layers), bits=16)
    if csim_outputs is None:
        print("  FATAL: CSIM failed")
        return
    print(f"  CSIM dumps loaded for {len(csim_outputs)} layers")

    # Step 5: Generate golden data (metadata, weights, params, input L0)
    print("Step 5: Generate golden data")
    generate_golden(layers, weight_blob, input_nhwc, GOLDEN_OUT)

    # Step 5b: Patch Add-layer ddr_add_b_addr
    import json
    from bin2golden import n_input_words, elem_bytes, OP_ELTWISE_ADD
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
            m['sched_ctrl'] = m['sched_ctrl'] | 0x10  # Set PTS bit
            m['store_mode'] = 1  # PER_TILE_STORE_EN
            eb = 2 if m.get('post_ctrl', 0) & 0x80 else 1  # INT16_OUT
            tile_h = m['tile_h']
            tile_w = m['tile_w']
            row_len = (tile_w * m['out_c'] * eb + 3) // 4
            m['row_cfg'] = (tile_h << 16) | row_len
            m['tile_out_size'] = tile_h * tile_w * m['out_c'] * eb
            full_bytes = m['out_h'] * m['out_w'] * m['out_c'] * eb
            m['n_output_words'] = (full_bytes + 3) // 4
            m['dma_out_size'] = full_bytes

    with open(meta_path, 'w') as f:
        json.dump(md, f, indent=2)

    # Step 6: Post-process — add wgt_per_oc_words, pack outputs from CSIM dumps

    for i, m in enumerate(md):
        # Add wgt_per_oc_words
        if m['op_type'] == 0 or m['op_type'] == 2:  # Conv2D/FC
            wgt_words = m['dma_wgt_size'] // 4
            if wgt_words > 24576:
                kd = m.get('kernel_h', 1) * m.get('kernel_w', 1) * m['in_c']
                per_oc = (16 * kd * 2 + 3) // 4
                m['wgt_per_oc_words'] = per_oc
                print(f"  L{i}: wgt overflow {wgt_words} > 24576, per_oc={per_oc}")
            else:
                m['wgt_per_oc_words'] = 0
        else:
            m['wgt_per_oc_words'] = 0

        # Update input/output from CSIM dumps
        if i in csim_outputs:
            out_raw = csim_outputs[i].astype(np.int16)
            out_c, out_h, out_w = m['out_c'], m['out_h'], m['out_w']
            if len(out_raw) == out_h * out_w * out_c:
                out_nhwc = out_raw.reshape(out_h, out_w, out_c)
            else:
                out_nhwc = out_raw

            # Input
            if m.get('input_src', -1) >= 0 and m['input_src'] in csim_outputs:
                inp_raw = csim_outputs[m['input_src']].astype(np.int16)
                inp_c, inp_h, inp_w = m['in_c'], m['in_h'], m['in_w']
                if len(inp_raw) == inp_h * inp_w * inp_c:
                    inp_nhwc = inp_raw.reshape(inp_h, inp_w, inp_c)
                else:
                    inp_nhwc = inp_raw
            elif i > 0 and (i - 1) in csim_outputs:
                inp_raw = csim_outputs[i - 1].astype(np.int16)
                inp_c, inp_h, inp_w = m['in_c'], m['in_h'], m['in_w']
                if len(inp_raw) == inp_h * inp_w * inp_c:
                    inp_nhwc = inp_raw.reshape(inp_h, inp_w, inp_c)
                else:
                    inp_nhwc = inp_raw
            else:
                inp_nhwc = input_nhwc

            # Pack and save
            layer = layers[i]
            # Use per_tile_store=True if store_mode was patched in Step 5c
            pts = True if m.get('store_mode', 0) & 1 else None
            inp_words = pack_input_to_words(inp_nhwc, layer)
            out_words = pack_output_to_words(out_nhwc, layer, per_tile_store=pts)
            np.save(os.path.join(GOLDEN_OUT, f'layer_{i:02d}_input.npy'), inp_words)
            np.save(os.path.join(GOLDEN_OUT, f'layer_{i:02d}_output.npy'), out_words)

            # Add B input for Add layers
            if m['op_type'] == 4 and m.get('residual_src', -1) >= 0:
                src_out = csim_outputs.get(m['residual_src'])
                if src_out is not None:
                    src_out = src_out.astype(np.int16)
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
