#!/usr/bin/env python3
"""
Generate RTL golden data for MODEL_C (YOLO-Tiny, 416x416x3, 17 layers).
Uses onnx_converter + CSIM + bin2golden.

Model_C tests detection-specific patterns:
  - Deep backbone with pooling (16x/32x downsampling)
  - Resize/Upsample for FPN-style feature fusion
  - Concat for merging multi-scale features
  - Tiled Conv + Resize + Concat
"""
import os, sys, glob, subprocess, numpy as np, struct, shutil
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..', 'design'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bin2golden import read_npu1, generate_golden, pack_input_to_words, pack_output_to_words, n_output_words, elem_bytes, OP_ELTWISE_ADD

ONNX_PATH = '/tmp/npu_yolo_output_small__uoinc2r/model_single.onnx'
WORK_DIR = '/tmp/model_c_int16_golden'
GOLDEN_OUT = '/data/sam/open-npu/rtl/tb/golden/golden_dma_e2e/model_c_int16'

def create_unfused(src, dst):
    """Create unfused version (clear FUSE bits in sched_ctrl) + enlarge tiles."""
    shutil.copy2(src, dst)
    with open(dst, 'rb') as f:
        data = bytearray(f.read())
    magic, num_layers, woff, wsize = struct.unpack_from('<4I', data, 0)
    off = 16  # header size
    ACT_DEPTH = 12288  # SPAD_KB * 64
    BANK = ACT_DEPTH // 2  # 6144
    for i in range(num_layers):
        if off + 62 > len(data):
            break
        # Clear FUSE bits
        data[off + 48] = data[off + 48] & 0x01

        # Read tile config
        tile_h = struct.unpack_from('<H', data, off + 39)[0]
        tile_w = struct.unpack_from('<H', data, off + 41)[0]
        in_c = struct.unpack_from('<H', data, off + 6)[0]
        out_c = struct.unpack_from('<H', data, off + 12)[0]
        op_type = data[off]
        eb = 2 if (data[off + 49] >> 7) & 1 else 1  # data_type

        # Enlarge tiles to reduce tile count (fits in bank)
        if tile_h > 0 and tile_w > 0:
            # Max tile area: BANK / ((in_c + out_c) * eb / 4)
            max_area = BANK * 4 // ((in_c + out_c) * eb) if (in_c + out_c) > 0 else BANK
            if max_area < 1:
                max_area = 1
            curr_area = tile_h * tile_w
            if curr_area < max_area:
                # Scale up tile keeping aspect ratio
                scale = int((max_area / curr_area) ** 0.5)
                if scale > 1:
                    new_h = min(tile_h * scale, 256)  # cap
                    new_w = min(tile_w * scale, 256)
                    # Verify fits
                    while new_h * new_w * (in_c + out_c) * eb // 4 > BANK and new_h > 1:
                        new_h -= 1
                    while new_h * new_w * (in_c + out_c) * eb // 4 > BANK and new_w > 1:
                        new_w -= 1
                    struct.pack_into('<H', data, off + 39, new_h)
                    struct.pack_into('<H', data, off + 41, new_w)
                    # Recompute tile_num
                    out_h = struct.unpack_from('<H', data, off + 8)[0]
                    out_w = struct.unpack_from('<H', data, off + 10)[0]
                    new_num_h = (out_h + new_h - 1) // new_h
                    new_num_w = (out_w + new_w - 1) // new_w
                    struct.pack_into('<H', data, off + 43, new_num_h)
                    struct.pack_into('<H', data, off + 45, new_num_w)

        # Skip variable-length data
        param_ch_count = struct.unpack_from('<H', data, off + 55)[0]
        has_lut = data[off + 57]
        has_add = data[off + 58]
        off += 62 + param_ch_count * 14
        if has_add: off += 8
        if has_lut: off += 768
    with open(dst, 'wb') as f:
        f.write(data)

def run_csim(model_bin, input_bin, work_dir, num_layers, bits=16):
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
    dt_u = np.uint16 if bits == 16 else np.uint8
    for idx in range(num_layers):
        dump_path = f'/tmp/csim_layer_{idx:03d}.bin'
        if os.path.exists(dump_path):
            outputs[idx] = np.fromfile(dump_path, dtype=dt_u)
    return outputs

def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(GOLDEN_OUT, exist_ok=True)

    # Step 1: Convert model to INT16
    from onnx_converter import convert_model
    model_bin_fused = os.path.join(WORK_DIR, 'model_c_int16.npu1.bin')
    model_bin = os.path.join(WORK_DIR, 'model_c_int16_unfused.npu1.bin')

    if not os.path.exists(model_bin_fused):
        print("Step 1: Convert YOLO-Tiny to INT16")
        # Use a test image as input
        all_imgs = sorted(glob.glob('/data/sam/onnx_quant/a3_test_images/*.jpg'))
        if all_imgs:
            img = Image.open(all_imgs[0]).resize((416, 416))
        else:
            img = Image.fromarray(np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8))
        arr = np.array(img).transpose(2, 0, 1).astype(np.uint8)
        input_bin_raw = os.path.join(WORK_DIR, 'input.bin')
        arr.tofile(input_bin_raw)
        convert_model(ONNX_PATH, '/data/sam/onnx_quant/a3_test_images', input_bin_raw,
                      model_bin_fused, input_format='int8-nchw', num_calib=20, bits=16)

    if not os.path.exists(model_bin):
        print("Step 1b: Create unfused version")
        create_unfused(model_bin_fused, model_bin)
    print(f"  Model (unfused): {model_bin}")

    # Step 2: Read npu1 binary
    print("Step 2: Read npu1 binary")
    layers, weight_blob = read_npu1(model_bin)
    print(f"  {len(layers)} layers, weight blob {len(weight_blob)} bytes")

    # Step 3: Prepare input (use converter-generated INT16 input)
    input_bin = model_bin_fused.replace('.npu1.bin', '.npu1_input.bin')
    inp_int16 = np.fromfile(input_bin, dtype=np.int16)
    # Reshape to NHWC
    inp_c, inp_h, inp_w = layers[0].in_c, layers[0].in_h, layers[0].in_w
    input_nhwc = np.transpose(inp_int16.reshape(inp_c, inp_h, inp_w), (1, 2, 0))
    print(f"  Input: {input_bin} ({os.path.getsize(input_bin)} bytes)")

    # Step 4: Run CSIM
    print("Step 4: Run CSIM with DUMP_LAYERS")
    csim_outputs = run_csim(model_bin, input_bin, WORK_DIR, len(layers), bits=16)
    if csim_outputs is None:
        print("  FATAL: CSIM failed")
        return
    print(f"  CSIM dumps loaded for {len(csim_outputs)} layers")

    # Step 5: Generate golden data
    print("Step 5: Generate golden data")
    generate_golden(layers, weight_blob, input_nhwc, GOLDEN_OUT)

    # Step 5b: Patch per-tile store for DB_EN + tiled layers,
    # and add tiling for non-tiled layers that overflow SRAM
    import json
    ACT_DEPTH = 12288  # SPAD_KB * 64
    meta_path = os.path.join(GOLDEN_OUT, 'metadata.json')
    with open(meta_path) as f:
        md = json.load(f)

    for i, m in enumerate(md):
        eb = 2 if m.get('post_ctrl', 0) & 0x80 else 1
        # Add tiling for non-tiled layers that overflow SRAM
        if m.get('tile_h', 0) == 0 and m['n_input_words'] > ACT_DEPTH:
            # Auto-tile: pick tile_h so per-tile input + output fits in SRAM
            # Input per tile = tile_h * out_w * in_c * eb / 4
            # Output per tile = tile_h * out_w * out_c * eb / 4
            # Sum must be < ACT_DEPTH
            in_c = m['in_c']
            out_c = m['out_c']
            out_w = m['out_w']
            per_row_words = (out_w * (in_c + out_c) * eb + 3) // 4
            tile_h = max(1, (ACT_DEPTH // 2) // per_row_words)  # use bank size (ACT_DEPTH/2)
            if tile_h > m['out_h']:
                tile_h = m['out_h']
            tile_num_h = (m['out_h'] + tile_h - 1) // tile_h
            tile_num_w = 1
            m['tile_h'] = tile_h
            m['tile_w'] = out_w
            m['tile_num_h'] = tile_num_h
            m['tile_num_w'] = tile_num_w
            m['sched_ctrl'] = m['sched_ctrl'] | 0x11  # DB_EN + PTS
            tile_in_words = (tile_h * out_w * in_c * eb + 3) // 4
            m['tile_in_size'] = tile_in_words * 4
            print(f"  L{i}: auto-tiled {tile_h}x{out_w}@{tile_num_h}x{tile_num_w} (input was {m['n_input_words']} > {ACT_DEPTH})")

        db_en = m['sched_ctrl'] & 1
        has_tile = m.get('tile_h', 0) > 0 and m.get('tile_w', 0) > 0
        if db_en and has_tile:
            m['sched_ctrl'] = m['sched_ctrl'] | 0x10  # Set PTS bit
            m['store_mode'] = 1
            tile_h = m['tile_h']
            tile_w = m['tile_w']
            row_len = (tile_w * m['out_c'] * eb + 3) // 4
            m['row_cfg'] = (tile_h << 16) | row_len
            m['tile_out_size'] = tile_h * tile_w * m['out_c'] * eb
            full_bytes = m['out_h'] * m['out_w'] * m['out_c'] * eb
            m['n_output_words'] = (full_bytes + 3) // 4
            m['dma_out_size'] = full_bytes
            print(f"  L{i}: PTS enabled, tile_out={m['tile_out_size']}")

    # Step 5c: Patch Add-layer ddr_add_b_addr (for Concat layers)
    align = 4096
    for idx, (layer, meta) in enumerate(zip(layers, md)):
        if layer.op_type != OP_ELTWISE_ADD:
            continue
        # Only Add (op_type=4) needs add_b; Concat (op_type=7) does not
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
        print(f"  L{idx} Concat/Add: add_b_addr=0x{meta['ddr_add_b_addr']:08X}")

    # Step 5d: Add wgt_per_oc_words
    for i, m in enumerate(md):
        if m['op_type'] == 0 or m['op_type'] == 2:
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

    with open(meta_path, 'w') as f:
        json.dump(md, f, indent=2)

    # Step 6: Pack outputs from CSIM dumps
    for i, m in enumerate(md):
        if i in csim_outputs:
            out_raw = csim_outputs[i].astype(np.int16)
            out_c, out_h, out_w = m['out_c'], m['out_h'], m['out_w']
            if len(out_raw) == out_h * out_w * out_c:
                out_nhwc = out_raw.reshape(out_h, out_w, out_c)
            else:
                out_nhwc = out_raw

            # For Concat: input comes from residual_src, not previous layer
            input_src = m.get('input_src', -1)
            if m['op_type'] == 7 and m.get('residual_src', -1) >= 0:
                input_src = m['residual_src']
            if input_src >= 0 and input_src in csim_outputs:
                inp_raw = csim_outputs[input_src].astype(np.int16)
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

            layer = layers[i]
            pts = True if m.get('store_mode', 0) & 1 else None
            inp_words = pack_input_to_words(inp_nhwc, layer)
            out_words = pack_output_to_words(out_nhwc, layer, per_tile_store=pts)
            np.save(os.path.join(GOLDEN_OUT, f'layer_{i:02d}_input.npy'), inp_words)
            np.save(os.path.join(GOLDEN_OUT, f'layer_{i:02d}_output.npy'), out_words)

            if (m['op_type'] == 4 or m['op_type'] == 7) and m.get('residual_src', -1) >= 0:
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
