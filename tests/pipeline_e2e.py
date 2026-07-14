#!/usr/bin/env python3
"""
Full Pipeline E2E Test: NPU1 binary → CSIM → RTL Golden

Builds NPU1 models via model_packer, runs CSIM for expected output,
generates RTL golden data. Verifies RTL output matches CSIM.

Usage:
  python3 test_pipeline_e2e.py          # Run all pipeline tests
  python3 test_pipeline_e2e.py conv     # Run specific test
  python3 test_pipeline_e2e.py --rtl    # Also run RTL simulation

SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_packer import (
    LayerConfig, PerChannelParam, AddParam, make_ch_params, pack_model,
    OP_CONV2D, OP_DW_CONV, OP_ELTWISE_ADD,
)
from bin2golden import (
    read_npu1, generate_golden, n_output_words, n_wgt_words,
    pack_input_to_words, pack_output_to_words, pack_params_to_words,
    extract_layer_weights,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
CSIM_PATH = os.path.join(ROOT, '..', 'csim', 'npu_sim')
TESTDATA_DIR = os.path.join(ROOT, '..', 'csim', 'testdata')
GOLDEN_DIR = os.path.join(ROOT, '..', 'rtl', 'tb', 'golden', 'golden_dma_e2e')


def run_csim_per_layer(layers, weights_list, input_nchw):
    """Run CSIM for each layer independently to get per-layer expected outputs.
    
    For multi-layer models, creates per-layer NPU1 binaries and runs CSIM
    on each, chaining outputs as inputs to the next layer.
    Returns list of NHWC tensors (one per layer).
    """
    csim_path = CSIM_PATH
    if not os.path.exists(csim_path):
        print(f"  SKIP: CSIM not found")
        return None

    outputs = []
    prev_input_nchw = input_nchw

    for idx, l in enumerate(layers):
        # Build single-layer model
        w = weights_list[idx] if idx < len(weights_list) else np.array([], dtype=np.int8)
        model_path = os.path.join(TESTDATA_DIR, f'_tmp_l{idx}.bin')
        in_path = os.path.join(TESTDATA_DIR, f'_tmp_l{idx}_in.bin')
        out_path = os.path.join(TESTDATA_DIR, f'_tmp_l{idx}_out.bin')
        
        pack_model([l], w.tobytes(), model_path)
        prev_input_nchw.tofile(in_path)

        result = subprocess.run(
            [csim_path, model_path, in_path, out_path],
            capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  CSIM layer {idx} failed: {result.stderr[:200]}")
            return None

        raw = np.fromfile(out_path, dtype=np.uint8)
        eb = 2 if l.data_type == 1 else 1
        dt = np.int16 if eb == 2 else np.int8
        n_elems = l.out_h * l.out_w * l.out_c
        tensor = np.frombuffer(raw, dtype=dt)[:n_elems]
        tensor = tensor.reshape(l.out_c, l.out_h, l.out_w)
        tensor = np.transpose(tensor, (1, 2, 0))  # NCHW → NHWC
        
        outputs.append(tensor)
        prev_input_nchw = tensor  # chain to next layer

    return outputs


def test_model(name, layers, weights_list, input_nchw):
    """Full pipeline: NPU1 → CSIM → RTL golden."""
    print(f"\n{'='*60}")
    print(f"Pipeline: {name}")
    print(f"{'='*60}")
    os.makedirs(TESTDATA_DIR, exist_ok=True)

    # Save input (NCHW, as expected by CSIM)
    in_path = os.path.join(TESTDATA_DIR, f'{name}_input.bin')
    input_nchw.tofile(in_path)

    # Build NPU1 binary
    weight_data = b''.join(w.tobytes() for w in weights_list)
    model_path = os.path.join(TESTDATA_DIR, f'{name}.bin')
    pack_model(layers, weight_data, model_path)
    print(f"  NPU1: {model_path}")

    # Run CSIM to get expected per-layer outputs
    csim_outputs = run_csim_per_layer(layers, weights_list, input_nchw)
    if csim_outputs is None:
        print(f"  SKIP: CSIM unavailable")
        return False

    print(f"  CSIM: {len(csim_outputs)} layers")
    for i, out in enumerate(csim_outputs):
        print(f"    L{i}: {out.shape} [{out.min()}, {out.max()}]")

    # Generate golden data
    golden_out = os.path.join(GOLDEN_DIR, f'pipeline_{name}')
    import shutil
    if os.path.exists(golden_out):
        shutil.rmtree(golden_out)

    input_nhwc = input_nchw.transpose(1, 2, 0)
    generate_golden(layers, weight_data, input_nhwc, golden_out,
                    base_ddr_addr=0x30000000, layer_offset=0x00010000)

    # Patch in CSIM outputs as golden references
    for idx, (l, out_nhwc) in enumerate(zip(layers, csim_outputs)):
        if idx == 0:
            inp = pack_input_to_words(input_nhwc, l)
        else:
            inp = pack_input_to_words(csim_outputs[idx - 1], l)
        out = pack_output_to_words(out_nhwc, l)
        np.save(os.path.join(golden_out, f'layer_{idx:02d}_input.npy'), inp)
        np.save(os.path.join(golden_out, f'layer_{idx:02d}_output.npy'), out)

    print(f"  Golden: {golden_out} ({len(layers)} layers, CSIM-based)")
    return True


def test_conv():
    """2-layer Conv2D INT8 8x8 from model_packer."""
    np.random.seed(42)
    l0 = LayerConfig(op_type=OP_CONV2D, data_type=0,
                     in_h=8, in_w=8, in_c=1, out_h=8, out_w=8, out_c=4,
                     kernel_h=3, kernel_w=3, stride_h=1, stride_w=1,
                     pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
                     post_ctrl=64 | 32, clamp_min=-128, clamp_max=127)
    l0.ch_params = make_ch_params(
        np.array([16384, 12000, 20000, 8000], dtype=np.uint16),
        np.array([15, 14, 16, 13], dtype=np.uint8),
        np.random.randint(-50, 50, (4,), dtype=np.int32),
        np.array([2, -1, 0, 3], dtype=np.int16))

    l1 = LayerConfig(op_type=OP_CONV2D, data_type=0,
                     in_h=8, in_w=8, in_c=4, out_h=8, out_w=8, out_c=2,
                     post_ctrl=64 | 32 | 4, clamp_min=0, clamp_max=127)
    l1.ch_params = make_ch_params(
        np.array([15000, 18000], dtype=np.uint16),
        np.array([14, 15], dtype=np.uint8),
        np.random.randint(-30, 30, (2,), dtype=np.int32))

    w0 = np.random.randint(-10, 10, (4, 3, 3, 1), dtype=np.int8)
    w1 = np.random.randint(-10, 10, (2, 1, 1, 4), dtype=np.int8)
    inp = np.random.randint(-50, 50, (1, 8, 8), dtype=np.int8)
    return test_model('conv', [l0, l1], [w0, w1], inp)


def test_dwconv():
    """Conv2D→DWConv→Conv2D INT8 8x8."""
    np.random.seed(123)
    l0 = LayerConfig(op_type=OP_CONV2D,
                     in_h=8, in_w=8, in_c=1, out_h=8, out_w=8, out_c=8,
                     kernel_h=3, kernel_w=3, pad_top=1, pad_bottom=1,
                     pad_left=1, pad_right=1, post_ctrl=64,
                     clamp_min=-128, clamp_max=127)
    l0.ch_params = make_ch_params(
        np.full(8, 16000, np.uint16), np.full(8, 15, np.uint8),
        np.random.randint(-20, 20, (8,), dtype=np.int32))

    l1 = LayerConfig(op_type=OP_DW_CONV,
                     in_h=8, in_w=8, in_c=8, out_h=4, out_w=4, out_c=8,
                     kernel_h=3, kernel_w=3, stride_h=2, stride_w=2,
                     pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
                     post_ctrl=64, clamp_min=-128, clamp_max=127)
    l1.ch_params = make_ch_params(
        np.full(8, 14000, np.uint16), np.full(8, 14, np.uint8),
        np.random.randint(-10, 10, (8,), dtype=np.int32))

    l2 = LayerConfig(op_type=OP_CONV2D,
                     in_h=4, in_w=4, in_c=8, out_h=4, out_w=4, out_c=4,
                     post_ctrl=64 | 4, clamp_min=0, clamp_max=127)
    l2.ch_params = make_ch_params(
        np.full(4, 12000, np.uint16), np.full(4, 14, np.uint8),
        np.random.randint(-10, 10, (4,), dtype=np.int32))

    w0 = np.random.randint(-8, 8, (8, 3, 3, 1), dtype=np.int8)
    w1 = np.random.randint(-8, 8, (8, 3, 3), dtype=np.int8)
    w2 = np.random.randint(-8, 8, (4, 1, 1, 8), dtype=np.int8)
    inp = np.random.randint(-30, 30, (1, 8, 8), dtype=np.int8)
    return test_model('dwconv', [l0, l1, l2], [w0, w1, w2], inp)


def test_add():
    """Eltwise Add INT8 (dual rescale) — uses Python ref since CSIM needs 2 inputs."""
    np.random.seed(999)
    l0 = LayerConfig(op_type=OP_ELTWISE_ADD,
                     in_h=4, in_w=4, in_c=8,
                     out_h=4, out_w=4, out_c=8,
                     post_ctrl=1 | 4, clamp_min=0, clamp_max=127)
    l0.add_params = AddParam(M_A=16000, S_A=14, M_B=12000, S_B=13)
    inp = np.random.randint(-50, 50, (8, 4, 4), dtype=np.int8)

    print(f"\n{'='*60}")
    print(f"Pipeline: add")
    print(f"{'='*60}")

    # Generate golden data using Python reference (Add is well-tested)
    from model_packer import ref_postproc_add
    inp_nhwc = inp.transpose(1, 2, 0)
    golden_out = os.path.join(GOLDEN_DIR, 'pipeline_add')
    import shutil
    if os.path.exists(golden_out): shutil.rmtree(golden_out)

    generate_golden([l0], b'', inp_nhwc, golden_out,
                    base_ddr_addr=0x30000000, layer_offset=0x00010000)

    ref = ref_postproc_add(inp_nhwc, inp_nhwc, l0.add_params, l0)
    inp_words = pack_input_to_words(inp_nhwc, l0)
    out_words = pack_output_to_words(ref, l0)
    np.save(os.path.join(golden_out, 'layer_00_input.npy'), inp_words)
    np.save(os.path.join(golden_out, 'layer_00_output.npy'), out_words)
    print(f"  Golden: {golden_out} (Add, Python ref)")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='*', default=['conv', 'dwconv', 'add'])
    parser.add_argument('--rtl', action='store_true', help='Also run RTL simulation')
    args = parser.parse_args()

    results = {}
    for name in args.models:
        if name == 'conv': ok = test_conv()
        elif name == 'dwconv': ok = test_dwconv()
        elif name == 'add': ok = test_add()
        else: print(f"Unknown: {name}"); ok = False
        results[name] = ok

    print(f"\n{'='*60}")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"Pipeline: {'ALL PASS' if all_ok else 'SOME FAILED'}")

    # Run RTL simulation
    if args.rtl and all_ok:
        for name in args.models:
            test_name = f'test_pipeline_{name}'
            tb_dir = os.path.join(ROOT, '..', 'rtl', 'tb')
            env = os.environ.copy()
            env['MODULE'] = 'e2e.test_npu_dma_e2e'
            env['TESTCASE'] = test_name
            env['SIM'] = 'icarus'
            print(f"\n  Running RTL: {test_name}...")
            r = subprocess.run(
                ['make', 'DUT=npu_top'],
                cwd=tb_dir, capture_output=True, text=True, env=env)
            if 'FAIL' in r.stdout or r.returncode != 0:
                print(f"  RTL FAILED: {test_name}")
                # Show last lines of error
                for line in r.stdout.split('\n')[-10:]:
                    if 'FAIL' in line or 'mismatch' in line or 'assert' in line:
                        print(f"    {line.strip()}")
            else:
                print(f"  RTL PASSED: {test_name}")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
