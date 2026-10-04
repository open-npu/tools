#!/usr/bin/env python3
"""
Generate RTL golden data for MODEL_A (MobileNetV2, 224x224x3, 63 layers).
Uses onnx_converter + CSIM + bin2golden.

INT8 variant — same flow as INT16 but bits=8 and CSIM uses int8 dumps.
"""
import os, sys, glob, subprocess, numpy as np
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import struct, shutil
from bin2golden import read_npu1, generate_golden, pack_input_to_words, pack_output_to_words, n_output_words, elem_bytes
from tiling import compute_tiling
from hw_config import DEFAULT_HW

OP_NAMES = {0: 'conv', 1: 'dw', 2: 'fc', 3: 'pool', 4: 'add',
            5: 'resize', 6: 'deconv', 7: 'concat'}

MODEL_PATH = '/data/sam/onnx_quant/MODEL_A.onnx'
CALIB_DIR = '/data/sam/onnx_quant/a3_test_images'
WORK_DIR = '/tmp/model_a_int8_golden'
GOLDEN_OUT = '/data/sam/open-npu/rtl/tb/golden/golden_dma_e2e/model_a_int8'

def run_csim(model_bin, input_bin, work_dir, num_layers, bits=8):
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
    dt_u = np.uint8 if bits == 8 else np.uint16
    for idx in range(num_layers):
        dump_path = f'/tmp/csim_layer_{idx:03d}.bin'
        if not os.path.exists(dump_path):
            print(f"  WARNING: csim_layer_{idx:03d}.bin not found")
            continue
        raw = np.fromfile(dump_path, dtype=dt_u)
        outputs[idx] = raw
    return outputs


def create_unfused_refit(src, dst):
    """Clear FUSE bits and restore per-layer tiles that fit one DB bank.

    Fusion writes a shared tile onto the block-start layer. After unfuse that
    tile can exceed the 6144-word bank (A8 L6 13x12 → RTL E5, npu=0).
    """
    shutil.copy2(src, dst)
    with open(dst, 'rb') as f:
        data = bytearray(f.read())
    magic, num_layers, woff, wsize = struct.unpack_from('<4I', data, 0)
    off = 16
    for i in range(num_layers):
        if off + 62 > len(data):
            break
        op = data[off]
        dt = data[off + 1]
        in_h, in_w, in_c = struct.unpack_from('<HHH', data, off + 2)
        out_h, out_w, out_c = struct.unpack_from('<HHH', data, off + 8)
        kh, kw = data[off + 14], data[off + 15]
        sh, sw = data[off + 18], data[off + 19]
        pool_h, pool_w = data[off + 25], data[off + 26]
        pool_sh, pool_sw = data[off + 27], data[off + 28]
        old_h = struct.unpack_from('<H', data, off + 39)[0]
        old_w = struct.unpack_from('<H', data, off + 41)[0]
        data[off + 48] = data[off + 48] & 0x11  # keep DB_EN + PTS, drop FUSE
        eb = 2 if dt else 1
        op_name = OP_NAMES.get(op, 'conv')
        k_h, k_w, s_h, s_w = kh, kw, sh, sw
        if op_name == 'pool':
            k_h, k_w, s_h, s_w = pool_h or 1, pool_w or 1, pool_sh or 1, pool_sw or 1
        try:
            tile = compute_tiling(
                op_type=op_name, in_h=in_h, in_w=in_w, in_c=in_c,
                out_h=out_h, out_w=out_w, out_c=out_c,
                kernel_h=k_h, kernel_w=k_w, stride_h=s_h, stride_w=s_w,
                elem_size=eb, double_buffer=True, profile=DEFAULT_HW)
        except ValueError as e:
            print(f"  L{i}: tile refit failed ({e}), keeping {old_h}x{old_w}")
            tile = None
        if tile is not None:
            new_h, new_w = tile['tile_h'], tile['tile_w']
            new_nh, new_nw = tile['tile_num_h'], tile['tile_num_w']
            if (new_h, new_w) != (old_h, old_w):
                print(f"  L{i}: tile {old_h}x{old_w} → {new_h}x{new_w} "
                      f"({new_nh}x{new_nw})")
            struct.pack_into('<HHHH', data, off + 39, new_h, new_w, new_nh, new_nw)
            if new_h == 0:
                data[off + 48] = data[off + 48] & ~0x11  # no DB/PTS if untiled
            elif tile.get('per_tile_store_en'):
                data[off + 48] = data[off + 48] | 0x11
        param_ch_count = struct.unpack_from('<H', data, off + 55)[0]
        has_lut = data[off + 57]
        has_add = data[off + 58]
        off += 62 + param_ch_count * 14
        if has_add:
            off += 8
        if has_lut:
            off += 768
    with open(dst, 'wb') as f:
        f.write(data)


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(GOLDEN_OUT, exist_ok=True)

    # Step 1: Convert model (use unfused version for per-layer dumps)
    from onnx_converter import convert_model
    model_bin_fused = os.path.join(WORK_DIR, 'model_a_int8.npu1.bin')
    model_bin = os.path.join(WORK_DIR, 'model_a_int8_unfused.npu1.bin')
    if os.environ.get('REUSE_NPU1') != '1' or not os.path.exists(model_bin_fused):
        print("Step 1: Convert MODEL_A to INT8")
        all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))
        img = Image.open(all_imgs[0]).resize((224, 224))
        arr = np.array(img).transpose(2, 0, 1).astype(np.uint8)
        input_bin_raw = os.path.join(WORK_DIR, 'input.bin')
        arr.tofile(input_bin_raw)
        convert_model(MODEL_PATH, CALIB_DIR, input_bin_raw, model_bin_fused,
                      input_format='int8-nchw', num_calib=50, bits=8)

    print("Step 1b: Unfuse + refit tiles to DB bank")
    create_unfused_refit(model_bin_fused, model_bin)
    print(f"  Model (unfused): {model_bin}")

    # Step 2: Read npu1 binary
    print("Step 2: Read npu1 binary")
    layers, weight_blob = read_npu1(model_bin)
    print(f"  {len(layers)} layers, weight blob {len(weight_blob)} bytes")

    # Step 3: Prepare input (use converter-generated INT8 input)
    print("Step 3: Prepare test input")
    input_bin = os.path.join(WORK_DIR, 'model_a_int8.npu1_input.bin')
    if not os.path.exists(input_bin):
        # Re-run conversion to get input
        all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))
        img = Image.open(all_imgs[0]).resize((224, 224))
        arr = np.array(img).transpose(2, 0, 1).astype(np.uint8)
        input_bin_raw = os.path.join(WORK_DIR, 'input.bin')
        arr.tofile(input_bin_raw)
        convert_model(MODEL_PATH, CALIB_DIR, input_bin_raw, model_bin_fused,
                      input_format='int8-nchw', num_calib=50, bits=8)

    # Load input as NHWC for golden gen
    inp_int8 = np.fromfile(input_bin, dtype=np.int8).reshape(3, 224, 224)
    input_nhwc = np.transpose(inp_int8, (1, 2, 0))
    print(f"  Input: {input_bin} ({os.path.getsize(input_bin)} bytes)")

    # Step 4: Run CSIM (unfused, with DUMP_LAYERS)
    print("Step 4: Run CSIM with DUMP_LAYERS")
    csim_outputs = run_csim(model_bin, input_bin, WORK_DIR, len(layers), bits=8)
    if csim_outputs is None:
        print("  FATAL: CSIM failed")
        sys.exit(1)
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

        # Update input/output from CSIM dumps
        if i in csim_outputs:
            out_raw = csim_outputs[i].astype(np.int8)
            out_c, out_h, out_w = m['out_c'], m['out_h'], m['out_w']
            if len(out_raw) == out_h * out_w * out_c:
                out_nhwc = out_raw.reshape(out_h, out_w, out_c)
            else:
                out_nhwc = out_raw

            # Input
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
