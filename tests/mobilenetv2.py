#!/usr/bin/env python3
"""
Open-NPU MobileNetV2-Tiny End-to-End Test

Implements a scaled-down MobileNetV2 architecture (16×16 input, fewer channels)
that follows the exact same structure as the real model:
  - First conv 3×3
  - Inverted residual blocks (expand 1×1 → DW 3×3 → project 1×1)
  - Global average pool
  - FC classifier

Tests the C simulator against Python reference using per-channel requantize.

SPDX-License-Identifier: Apache-2.0
"""

import sys
import os
import subprocess
import numpy as np

# Add tools to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_packer import (
    LayerConfig, PerChannelParam, pack_model, make_ch_params,
    ref_conv2d, ref_dwconv, ref_pooling, ref_postproc_perchannel,
    OP_CONV2D, OP_DW_CONV, OP_POOLING, OP_FC, OP_ELTWISE_ADD,
    POST_BIAS_EN, POST_RELU_EN, POST_RELU6_EN, POST_INT16_OUT,
    PPU_MODE_CONV_REQ, PPU_MODE_PASSTHROUGH,
)


def compute_ms(eff_scale):
    """Compute 15-bit multiplier M and 6-bit shift S such that M / 2^S ≈ eff_scale."""
    best_s, best_m = 0, max(1, int(np.round(eff_scale)))
    for s in range(64):
        m = eff_scale * (2.0 ** s)
        if 1.0 <= m <= 32767.0:
            best_s = s
            best_m = int(np.round(m))
            if best_m >= 16384:
                break
    return max(1, min(32767, best_m)), best_s


def build_mobilenetv2_tiny():
    """
    MobileNetV2-Tiny architecture (scaled down for testing):
      Input: 16×16×3 (scale_in = 1/128)

      Layer  0: Conv2D 3×3, stride=2, 3→8 ch     → 8×8×8   + ReLU6
      --- Block 1 (no expansion, t=1) ---
      Layer  1: DWConv 3×3, stride=1, 8 ch        → 8×8×8   + ReLU6
      Layer  2: Conv2D 1×1, 8→4 ch (project)      → 8×8×4   linear
      --- Block 2 (t=6, stride=2) ---
      Layer  3: Conv2D 1×1, 4→24 ch (expand)      → 8×8×24  + ReLU6
      Layer  4: DWConv 3×3, stride=2, 24 ch       → 4×4×24  + ReLU6
      Layer  5: Conv2D 1×1, 24→8 ch (project)     → 4×4×8   linear
      --- Block 3 (t=6, stride=1) ---
      Layer  6: Conv2D 1×1, 8→48 ch (expand)      → 4×4×48  + ReLU6
      Layer  7: DWConv 3×3, stride=1, 48 ch       → 4×4×48  + ReLU6
      Layer  8: Conv2D 1×1, 48→8 ch (project)     → 4×4×8   linear
      --- Head ---
      Layer  9: Conv2D 1×1, 8→32 ch               → 4×4×32  + ReLU6
      Layer 10: GlobalAvgPool                      → 1×1×32
      Layer 11: FC, 32→10 (classifier)             → 1×1×10

    Total: 12 layers.
    """
    np.random.seed(2024)

    layers = []
    weights_list = []
    # Simulate quantization scales per layer output
    # Use fixed scales that keep values alive through 12 layers
    layer_scales = []
    scale_in = 1.0 / 64.0

    def add_conv(in_h, in_w, in_c, out_c, kernel=3, stride=1, relu6=True,
                 in_scale=None):
        nonlocal scale_in
        pad = (kernel - 1) // 2
        out_h = (in_h + 2 * pad - kernel) // stride + 1
        out_w = (in_w + 2 * pad - kernel) // stride + 1

        cfg = LayerConfig()
        cfg.op_type = OP_CONV2D
        cfg.in_h, cfg.in_w, cfg.in_c = in_h, in_w, in_c
        cfg.out_h, cfg.out_w, cfg.out_c = out_h, out_w, out_c
        cfg.kernel_h, cfg.kernel_w = kernel, kernel
        cfg.dilation_h, cfg.dilation_w = 1, 1
        cfg.stride_h, cfg.stride_w = stride, stride
        cfg.pad_top = cfg.pad_bottom = cfg.pad_left = cfg.pad_right = pad
        cfg.clamp_min, cfg.clamp_max = -128, 127

        # Post-processing: per-channel requantize + optional ReLU6
        post_ctrl = POST_BIAS_EN | PPU_MODE_CONV_REQ
        if relu6:
            post_ctrl |= POST_RELU6_EN
        cfg.post_ctrl = post_ctrl

        # Weight: [out_c][kh][kw][in_c], moderate range
        w = np.random.randint(-8, 9, (out_c, kernel, kernel, in_c), dtype=np.int8)

        # Use fixed output scale (same as input) — keeps values alive
        s_in = in_scale if in_scale else scale_in
        s_w = 1.0 / 64.0
        acc_scale = s_in * s_w
        s_out = s_in  # output stays at same scale as input
        eff_scale = acc_scale / s_out  # = s_w = 1/64

        # Per-channel params: M/S such that M/2^S ≈ eff_scale
        M_arr = np.zeros(out_c, dtype=np.uint16)
        S_arr = np.zeros(out_c, dtype=np.uint8)
        bias_arr = np.random.randint(-100, 100, (out_c,), dtype=np.int64)
        for c in range(out_c):
            # Slight per-channel variation
            es = eff_scale * (0.8 + 0.4 * np.random.rand())
            M_arr[c], S_arr[c] = compute_ms(es)

        cfg.ch_params = make_ch_params(M_arr, S_arr, bias_arr)

        layers.append(cfg)
        weights_list.append(w)
        layer_scales.append(s_out)
        scale_in = s_out
        return out_h, out_w, out_c

    def add_dw(in_h, in_w, ch, stride=1, relu6=True):
        nonlocal scale_in
        pad = 1  # 3×3 with pad=1
        out_h = (in_h + 2 * pad - 3) // stride + 1
        out_w = (in_w + 2 * pad - 3) // stride + 1

        cfg = LayerConfig()
        cfg.op_type = OP_DW_CONV
        cfg.in_h, cfg.in_w, cfg.in_c = in_h, in_w, ch
        cfg.out_h, cfg.out_w, cfg.out_c = out_h, out_w, ch
        cfg.kernel_h, cfg.kernel_w = 3, 3
        cfg.dilation_h, cfg.dilation_w = 1, 1
        cfg.stride_h, cfg.stride_w = stride, stride
        cfg.pad_top = cfg.pad_bottom = cfg.pad_left = cfg.pad_right = pad
        cfg.clamp_min, cfg.clamp_max = -128, 127

        post_ctrl = POST_BIAS_EN | PPU_MODE_CONV_REQ
        if relu6:
            post_ctrl |= POST_RELU6_EN
        cfg.post_ctrl = post_ctrl

        # DW weight: [ch][3][3]
        w = np.random.randint(-8, 9, (ch, 3, 3), dtype=np.int8)

        # Per-channel params
        s_w = 1.0 / 64.0
        acc_scale = scale_in * s_w
        s_out = scale_in  # keep same output scale
        eff_scale = acc_scale / s_out

        M_arr = np.zeros(ch, dtype=np.uint16)
        S_arr = np.zeros(ch, dtype=np.uint8)
        bias_arr = np.random.randint(-100, 100, (ch,), dtype=np.int64)
        for c in range(ch):
            es = eff_scale * (0.8 + 0.4 * np.random.rand())
            M_arr[c], S_arr[c] = compute_ms(es)

        cfg.ch_params = make_ch_params(M_arr, S_arr, bias_arr)

        layers.append(cfg)
        weights_list.append(w)
        layer_scales.append(s_out)
        scale_in = s_out
        return out_h, out_w, ch

    def add_pool(in_h, in_w, ch):
        cfg = LayerConfig()
        cfg.op_type = OP_POOLING
        cfg.in_h, cfg.in_w, cfg.in_c = in_h, in_w, ch
        cfg.out_h, cfg.out_w, cfg.out_c = 1, 1, ch
        cfg.pool_mode = 1  # Avg
        cfg.pool_h, cfg.pool_w = in_h, in_w
        cfg.pool_stride_h, cfg.pool_stride_w = in_h, in_w
        cfg.global_pool = 1
        cfg.post_ctrl = PPU_MODE_PASSTHROUGH
        cfg.clamp_min, cfg.clamp_max = -128, 127
        layers.append(cfg)
        weights_list.append(None)
        layer_scales.append(scale_in)  # passthrough
        return 1, 1, ch

    def add_fc(in_c, out_c):
        nonlocal scale_in
        cfg = LayerConfig()
        cfg.op_type = OP_FC
        cfg.in_h, cfg.in_w, cfg.in_c = 1, 1, in_c
        cfg.out_h, cfg.out_w, cfg.out_c = 1, 1, out_c
        cfg.kernel_h, cfg.kernel_w = 1, 1
        cfg.dilation_h, cfg.dilation_w = 1, 1
        cfg.stride_h, cfg.stride_w = 1, 1
        cfg.post_ctrl = POST_BIAS_EN | PPU_MODE_CONV_REQ
        cfg.clamp_min, cfg.clamp_max = -128, 127

        # FC weight: [out_c][in_c] → treat as [out_c][1][1][in_c]
        w = np.random.randint(-8, 9, (out_c, 1, 1, in_c), dtype=np.int8)

        s_w = 1.0 / 64.0
        acc_scale = scale_in * s_w
        s_out = scale_in  # keep same output scale
        eff_scale = acc_scale / s_out

        M_arr = np.zeros(out_c, dtype=np.uint16)
        S_arr = np.zeros(out_c, dtype=np.uint8)
        bias_arr = np.random.randint(-100, 100, (out_c,), dtype=np.int64)
        for c in range(out_c):
            es = eff_scale * (0.8 + 0.4 * np.random.rand())
            M_arr[c], S_arr[c] = compute_ms(es)

        cfg.ch_params = make_ch_params(M_arr, S_arr, bias_arr)

        layers.append(cfg)
        weights_list.append(w)
        layer_scales.append(s_out)
        scale_in = s_out
        return 1, 1, out_c

    # Build architecture
    h, w, c = 16, 16, 3

    # Layer 0: First conv
    h, w, c = add_conv(h, w, c, 8, kernel=3, stride=2, relu6=True)
    # Block 1
    h, w, c = add_dw(h, w, c, stride=1, relu6=True)
    h, w, c = add_conv(h, w, c, 4, kernel=1, stride=1, relu6=False)
    # Block 2
    h, w, c = add_conv(h, w, c, 24, kernel=1, stride=1, relu6=True)
    h, w, c = add_dw(h, w, c, stride=2, relu6=True)
    h, w, c = add_conv(h, w, c, 8, kernel=1, stride=1, relu6=False)
    # Block 3
    h, w, c = add_conv(h, w, c, 48, kernel=1, stride=1, relu6=True)
    h, w, c = add_dw(h, w, c, stride=1, relu6=True)
    h, w, c = add_conv(h, w, c, 8, kernel=1, stride=1, relu6=False)
    # Head
    h, w, c = add_conv(h, w, c, 32, kernel=1, stride=1, relu6=True)
    h, w, c = add_pool(h, w, c)
    h, w, c = add_fc(c, 10)

    print(f"MobileNetV2-Tiny: {len(layers)} layers")
    print(f"  Input:  16×16×3")
    print(f"  Output: 1×1×10 (10-class classifier)")
    op_names = {OP_CONV2D: 'Conv2D', OP_DW_CONV: 'DWConv', OP_FC: 'FC',
                OP_POOLING: 'Pool', OP_ELTWISE_ADD: 'EltAdd'}
    for i, cfg in enumerate(layers):
        print(f"  Layer {i:2d}: {op_names.get(cfg.op_type, '?'):6s} "
              f"[{cfg.in_h}×{cfg.in_w}×{cfg.in_c}] → [{cfg.out_h}×{cfg.out_w}×{cfg.out_c}]")

    return layers, weights_list


def run_reference(layers, weights_list, input_nhwc):
    """Run Python reference inference layer by layer."""
    current = input_nhwc.copy()

    for i, (cfg, w) in enumerate(zip(layers, weights_list)):
        if cfg.op_type == OP_CONV2D:
            acc = ref_conv2d(current, w, cfg)
            current = ref_postproc_perchannel(acc, cfg.ch_params, cfg)
        elif cfg.op_type == OP_DW_CONV:
            acc = ref_dwconv(current, w, cfg)
            current = ref_postproc_perchannel(acc, cfg.ch_params, cfg)
        elif cfg.op_type == OP_POOLING:
            acc = ref_pooling(current, cfg)
            # Passthrough: just clamp
            current = np.clip(acc, cfg.clamp_min, cfg.clamp_max).astype(np.int8)
        elif cfg.op_type == OP_FC:
            # FC uses conv2d path with 1×1 kernel
            acc = ref_conv2d(current, w, cfg)
            current = ref_postproc_perchannel(acc, cfg.ch_params, cfg)
        else:
            raise ValueError(f"Unsupported op_type {cfg.op_type} in reference")

    return current


def main():
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'csim', 'testdata')
    sim_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'csim', 'npu_sim')
    os.makedirs(test_dir, exist_ok=True)

    if not os.path.exists(sim_path):
        print(f"Simulator not found: {sim_path}")
        print("Build first: cd csim && make")
        return 1

    # Build model
    print("=" * 60)
    print("MobileNetV2-Tiny End-to-End Test (per-channel requantize)")
    print("=" * 60)
    print()

    layers, weights_list = build_mobilenetv2_tiny()

    # Generate random input (16×16×3, NCHW for file)
    np.random.seed(999)
    input_nchw = np.random.randint(-60, 60, (3, 16, 16), dtype=np.int8)
    input_nhwc = input_nchw.transpose(1, 2, 0)  # [H][W][C]

    # Pack weights (no separate bias blob — bias is in ch_params)
    weight_data = b''
    for w in weights_list:
        if w is not None:
            weight_data += w.tobytes()

    # Pack model
    model_path = os.path.join(test_dir, 'test_mbv2_tiny_model.bin')
    pack_model(layers, weight_data, model_path)

    input_path = os.path.join(test_dir, 'test_mbv2_tiny_input.bin')
    input_nchw.tofile(input_path)
    print(f"\nInput: {input_path} ({input_nchw.nbytes} bytes)")

    # Run Python reference
    print("\nRunning Python reference...")
    ref_output_nhwc = run_reference(layers, weights_list, input_nhwc)
    ref_output_nchw = ref_output_nhwc.transpose(2, 0, 1)
    print(f"  Reference output (10 classes): {ref_output_nchw.flatten()}")

    # Run C simulator
    print("\nRunning C simulator...")
    output_path = os.path.join(test_dir, 'test_mbv2_tiny_output.bin')
    result = subprocess.run(
        [sim_path, model_path, input_path, output_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  Simulator FAILED (exit code {result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
        return 1

    # Compare
    sim_output = np.fromfile(output_path, dtype=np.int8)
    ref_flat = ref_output_nchw.flatten().astype(np.int8)

    print(f"  Sim output (10 classes):  {sim_output}")
    print(f"  Ref output (10 classes):  {ref_flat}")

    if len(sim_output) != len(ref_flat):
        print(f"\n  FAIL: size mismatch (sim={len(sim_output)}, ref={len(ref_flat)})")
        return 1

    if np.array_equal(sim_output, ref_flat):
        print(f"\n  BIT-EXACT match! ({len(ref_flat)} elements)")
        status = "PASS"
    else:
        diff = np.abs(sim_output.astype(np.int32) - ref_flat.astype(np.int32))
        max_diff = diff.max()
        mismatch = np.sum(diff > 0)
        print(f"\n  NOT bit-exact: {mismatch}/{len(ref_flat)} differ, max_diff={max_diff}")
        # Allow ±1 tolerance for rounding edge cases in pooling
        if max_diff <= 1:
            print(f"  PASS (within ±1 tolerance)")
            status = "PASS"
        else:
            print(f"  FAIL (max_diff={max_diff} > 1)")
            status = "FAIL"

    print()
    print("=" * 60)
    if status == "PASS":
        print("SUCCESS: MobileNetV2-Tiny 12-layer inference verified!")
        print(f"  Architecture: Conv3×3 → [DW+PW]×3blocks → Conv1×1 → GAP → FC")
        print(f"  Operators tested: Conv2D(1×1,3×3), DWConv(s1,s2), AvgPool, FC")
        print(f"  Quantization: per-channel requantize (M/S) + ReLU6")
        print(f"  Total weight params: {sum(w.size for w in weights_list if w is not None)}")
        print(f"  Classifier output (10 classes): {sim_output}")
    else:
        print("FAILED: MobileNetV2-Tiny inference mismatch")
    print("=" * 60)

    return 0 if status == "PASS" else 1


if __name__ == '__main__':
    sys.exit(main())
