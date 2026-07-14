#!/usr/bin/env python3
"""
Generate RTL golden data for ResNet-18 CIFAR (32x32x3, 10-class).
Uses test_resnet18_e2e.py's make_resnet18_model with input_size=32.
"""
import os, sys, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onnx_converter import convert_model
from bin2golden import read_npu1, generate_golden, pack_input_to_words, pack_output_to_words, n_output_words, elem_bytes
from model_packer import OP_ELTWISE_ADD
import onnx

WORK_DIR = '/tmp/resnet18_cifar_golden'
GOLDEN_OUT = '/data/sam/open-npu/rtl/tb/golden/golden_dma_e2e/resnet18_cifar_int16'

def run_csim(layers, weights_list, input_nchw, work_dir, bits=16):
    """Run CSIM with DUMP_LAYERS=1 to get per-layer outputs."""
    import subprocess
    csim_exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csim', 'npu_sim')
    
    model_bin = os.path.join(work_dir, 'resnet18_cifar.npu1.bin')
    inp_path = os.path.join(work_dir, 'resnet18_cifar.npu1_input.bin')
    out_path = os.path.join(work_dir, 'output.bin')
    
    # Clear old dumps
    for f in os.listdir('/tmp'):
        if f.startswith('csim_layer_'):
            os.remove(os.path.join('/tmp', f))
    
    env = os.environ.copy()
    env['DUMP_LAYERS'] = '1'
    
    print(f"  CSIM: {csim_exe} {model_bin} {inp_path} {out_path}")
    result = subprocess.run([csim_exe, model_bin, inp_path, out_path],
                          capture_output=True, text=True, timeout=120, env=env)
    if result.returncode != 0:
        print(f"  CSIM failed: {result.stderr[:200]}")
        return None
    
    # Load per-layer dumps
    outputs = {}
    dt_u = np.uint16 if bits == 16 else np.uint8
    for idx in range(len(layers)):
        dump_path = f'/tmp/csim_layer_{idx:03d}.bin'
        if not os.path.exists(dump_path):
            continue
        raw = np.fromfile(dump_path, dtype=dt_u)
        outputs[idx] = raw
    return outputs

def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(GOLDEN_OUT, exist_ok=True)

    # Build ResNet-18 CIFAR model
    from test_resnet18_e2e import make_resnet18_model
    print("Step 1: Build ResNet-18 CIFAR (32x32x3, 10-class)")
    model = make_resnet18_model(num_classes=10, input_size=32)
    model_path = os.path.join(WORK_DIR, 'resnet18_cifar.onnx')
    onnx.save(model, model_path)
    print(f"  Nodes: {len(model.graph.node)}")

    # Calibration + input
    calib_dir = os.path.join(WORK_DIR, 'calib')
    os.makedirs(calib_dir, exist_ok=True)
    np.random.seed(42)
    for i in range(20):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(calib_dir, f'calib_{i:04d}.jpg'))
    np.random.seed(99)
    test_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    input_bin = os.path.join(WORK_DIR, 'input.bin')
    inp_float = (test_img.astype(np.float32).transpose(2, 0, 1) - 127.5) / 255.0
    inp_int8 = np.clip(np.round(inp_float * 128), -128, 127).astype(np.int8)
    inp_int8.tofile(input_bin)

    # Convert model
    model_bin = os.path.join(WORK_DIR, 'resnet18_cifar.npu1.bin')
    print("Step 2: Convert model (INT16)")
    convert_model(model_path, calib_dir, input_bin, model_bin,
                  input_format='int8-nchw', bits=16)

    # Read NPU1
    print("Step 3: Read NPU1 binary")
    layers, weight_blob = read_npu1(model_bin)
    print(f"  Layers: {len(layers)}")

    # Generate golden (metadata + wgt/param/input/output)
    inp_path = os.path.join(WORK_DIR, 'resnet18_cifar.npu1_input.bin')
    input_nchw = np.fromfile(inp_path, dtype=np.int16).reshape(3, 32, 32)
    input_nhwc = np.transpose(input_nchw, (1, 2, 0))  # CHW → HWC
    
    print("Step 4: Generate golden metadata")
    metadata = generate_golden(layers, weight_blob, input_nhwc, GOLDEN_OUT,
                               base_ddr_addr=0x30000000, layer_offset=0x00010000)

    # Patch Add-layer input_b DDR addresses
    print("Step 5: Patch Add-layer ddr_add_b_addr")
    align = 4096
    for idx, (layer, meta) in enumerate(zip(layers, metadata)):
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

    # Run CSIM and patch input/output
    print("Step 6: Run CSIM, patch input/output")
    csim_outputs = run_csim(layers, [], input_nchw, WORK_DIR, bits=16)
    if csim_outputs is None:
        print("  CSIM failed — using generate_golden's data")
    else:
        print(f"  CSIM dumps loaded for {len(csim_outputs)} layers")
        for idx, layer in enumerate(layers):
            if idx not in csim_outputs:
                continue
            out_raw = csim_outputs[idx]
            out_c, out_h, out_w = layer.out_c, layer.out_h, layer.out_w
            out_nhwc = out_raw.reshape(out_h, out_w, out_c) if len(out_raw) == out_h*out_w*out_c else out_raw
            
            # Input
            if layer.input_src >= 0 and layer.input_src in csim_outputs:
                inp_raw = csim_outputs[layer.input_src]
                inp_c, inp_h, inp_w = layer.in_c, layer.in_h, layer.in_w
                inp_nhwc = inp_raw.reshape(inp_h, inp_w, inp_c) if len(inp_raw) == inp_h*inp_w*inp_c else inp_raw
            elif idx > 0 and (idx-1) in csim_outputs:
                inp_raw = csim_outputs[idx-1]
                inp_c, inp_h, inp_w = layer.in_c, layer.in_h, layer.in_w
                inp_nhwc = inp_raw.reshape(inp_h, inp_w, inp_c) if len(inp_raw) == inp_h*inp_w*inp_c else inp_raw
            else:
                inp_nhwc = input_nchw.transpose(1, 2, 0)  # CHW → HWC
            
            inp_words = pack_input_to_words(inp_nhwc, layer)
            out_words = pack_output_to_words(out_nhwc, layer)
            np.save(os.path.join(GOLDEN_OUT, f'layer_{idx:02d}_input.npy'), inp_words)
            np.save(os.path.join(GOLDEN_OUT, f'layer_{idx:02d}_output.npy'), out_words)
            
            if layer.op_type == OP_ELTWISE_ADD and layer.residual_src >= 0:
                src_out = csim_outputs.get(layer.residual_src)
                if src_out is not None:
                    in_b_words = pack_input_to_words(src_out.reshape(layer.in_h, layer.in_w, layer.in_c)
                                                      if len(src_out) == layer.in_h*layer.in_w*layer.in_c else src_out, layer)
                    np.save(os.path.join(GOLDEN_OUT, f'layer_{idx:02d}_input_b.npy'), in_b_words)

    # Save metadata
    import json
    with open(os.path.join(GOLDEN_OUT, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! {len(metadata)} layers saved to {GOLDEN_OUT}")
    for i, m in enumerate(metadata):
        print(f"  L{i}: op={m['op_type']} {m['in_h']}x{m['in_w']}x{m['in_c']}->{m['out_h']}x{m['out_w']}x{m['out_c']} tile={m.get('tile_h',0)}x{m.get('tile_w',0)}")

if __name__ == '__main__':
    main()
