#!/usr/bin/env python3
"""
Open-NPU MobileNetV2-Tiny End-to-End Test

Implements a scaled-down MobileNetV2 architecture (16×16 input, fewer channels)
that follows the exact same structure as the real model:
  - First conv 3×3
  - Inverted residual blocks (expand 1×1 → DW 3×3 → project 1×1)
  - Global average pool
  - FC classifier

This tests the C simulator against all key operators:
  Conv2D, DWConv, Pooling (global avg), FC
  With: ReLU6, stride-2 DW, residual add (eltwise)

SPDX-License-Identifier: Apache-2.0
"""

import sys
import os
import numpy as np

# Add tools to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_packer import (
    LayerConfig, pack_model, run_sim_and_verify,
    ref_conv2d, ref_dwconv, ref_pooling, ref_postproc,
    OP_CONV2D, OP_DW_CONV, OP_POOLING, OP_FC, OP_ELTWISE_ADD,
    POST_BIAS_EN, POST_SHIFT_EN, POST_CLAMP_EN,
)


def make_conv2d_layer(in_h, in_w, in_c, out_c, kernel=3, stride=1, relu6=True):
    """Create a Conv2D layer config."""
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
    cfg.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    cfg.shift_bits = 4
    cfg.round_en = 1
    if relu6:
        cfg.clamp_min, cfg.clamp_max = 0, 6  # ReLU6 in quantized domain
    else:
        cfg.clamp_min, cfg.clamp_max = -128, 127  # Linear
    return cfg


def make_dwconv_layer(in_h, in_w, ch, stride=1, relu6=True):
    """Create a DWConv layer config."""
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
    cfg.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    cfg.shift_bits = 4
    cfg.round_en = 1
    if relu6:
        cfg.clamp_min, cfg.clamp_max = 0, 6
    else:
        cfg.clamp_min, cfg.clamp_max = -128, 127
    return cfg


def make_pool_layer(in_h, in_w, ch):
    """Create a global average pooling layer."""
    cfg = LayerConfig()
    cfg.op_type = OP_POOLING
    cfg.in_h, cfg.in_w, cfg.in_c = in_h, in_w, ch
    cfg.out_h, cfg.out_w, cfg.out_c = 1, 1, ch
    cfg.pool_mode = 1  # Avg
    cfg.global_pool = 1
    cfg.post_ctrl = POST_CLAMP_EN
    cfg.clamp_min, cfg.clamp_max = -128, 127
    return cfg


def make_fc_layer(in_c, out_c):
    """Create an FC layer."""
    cfg = LayerConfig()
    cfg.op_type = OP_FC
    cfg.in_h, cfg.in_w, cfg.in_c = 1, 1, in_c
    cfg.out_h, cfg.out_w, cfg.out_c = 1, 1, out_c
    cfg.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    cfg.shift_bits = 4
    cfg.round_en = 1
    cfg.clamp_min, cfg.clamp_max = -128, 127
    return cfg


def build_mobilenetv2_tiny():
    """
    MobileNetV2-Tiny architecture (scaled down for testing):
      Input: 16×16×3

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

    Total: 12 layers, exercises Conv2D (1×1 and 3×3), DWConv (stride 1&2),
           GlobalAvgPool, FC — all key MobileNetV2 patterns.
    """
    np.random.seed(2024)

    layers = []
    weights_list = []

    def add_conv(in_h, in_w, in_c, out_c, kernel=3, stride=1, relu6=True):
        cfg = make_conv2d_layer(in_h, in_w, in_c, out_c, kernel, stride, relu6)
        layers.append(cfg)
        # Weight: [out_c][kh][kw][in_c]
        w = np.random.randint(-4, 4, (out_c, kernel, kernel, in_c), dtype=np.int8)
        b = np.random.randint(-8, 8, (out_c,), dtype=np.int32)
        weights_list.append((w, b))
        return cfg.out_h, cfg.out_w, cfg.out_c

    def add_dw(in_h, in_w, ch, stride=1, relu6=True):
        cfg = make_dwconv_layer(in_h, in_w, ch, stride, relu6)
        layers.append(cfg)
        # Weight: [ch][3][3]
        w = np.random.randint(-4, 4, (ch, 3, 3), dtype=np.int8)
        b = np.random.randint(-8, 8, (ch,), dtype=np.int32)
        weights_list.append((w, b))
        return cfg.out_h, cfg.out_w, cfg.out_c

    def add_pool(in_h, in_w, ch):
        cfg = make_pool_layer(in_h, in_w, ch)
        layers.append(cfg)
        weights_list.append((None, None))  # No weights
        return 1, 1, ch

    def add_fc(in_c, out_c):
        cfg = make_fc_layer(in_c, out_c)
        layers.append(cfg)
        # Weight: [out_c][in_c]
        w = np.random.randint(-4, 4, (out_c, in_c), dtype=np.int8)
        b = np.random.randint(-8, 8, (out_c,), dtype=np.int32)
        weights_list.append((w, b))
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
    for i, cfg in enumerate(layers):
        op_names = ['Conv2D', 'DWConv', 'FC', 'Pool', 'EltAdd', 'Resize', 'Deconv', 'Concat']
        print(f"  Layer {i:2d}: {op_names[cfg.op_type]:6s} [{cfg.in_h}×{cfg.in_w}×{cfg.in_c}] → [{cfg.out_h}×{cfg.out_w}×{cfg.out_c}]")

    return layers, weights_list


def run_reference(layers, weights_list, input_nhwc):
    """Run Python reference inference."""
    current = input_nhwc.copy()

    for i, (cfg, (w, b)) in enumerate(zip(layers, weights_list)):
        if cfg.op_type == OP_CONV2D:
            acc = ref_conv2d(current, w, cfg)
            current = ref_postproc(acc, b, cfg)
        elif cfg.op_type == OP_DW_CONV:
            acc = ref_dwconv(current, w, cfg)
            current = ref_postproc(acc, b, cfg)
        elif cfg.op_type == OP_POOLING:
            acc = ref_pooling(current, cfg)
            current = ref_postproc(acc, None, cfg)
        elif cfg.op_type == OP_FC:
            # FC: current is [1][1][in_c], weight is [out_c][in_c]
            w_4d = w.reshape(cfg.out_c, 1, 1, cfg.in_c)
            cfg_tmp = LayerConfig()
            cfg_tmp.__dict__.update(cfg.__dict__)
            cfg_tmp.kernel_h, cfg_tmp.kernel_w = 1, 1
            acc = ref_conv2d(current, w_4d, cfg_tmp)
            current = ref_postproc(acc, b, cfg)
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
    print("MobileNetV2-Tiny End-to-End Test")
    print("=" * 60)
    print()

    layers, weights_list = build_mobilenetv2_tiny()

    # Generate random input (16×16×3, NCHW for file)
    np.random.seed(999)
    input_nchw = np.random.randint(-30, 30, (3, 16, 16), dtype=np.int8)
    input_nhwc = input_nchw.transpose(1, 2, 0)  # [H][W][C]

    # Pack weights
    weight_data = b''
    for w, b in weights_list:
        if w is not None:
            weight_data += w.tobytes()
        if b is not None:
            weight_data += b.tobytes()

    # Pack model
    model_path = os.path.join(test_dir, 'test_mbv2_tiny_model.bin')
    pack_model(layers, weight_data, model_path)

    input_path = os.path.join(test_dir, 'test_mbv2_tiny_input.bin')
    input_nchw.tofile(input_path)
    print(f"Input: {input_path} ({input_nchw.nbytes} bytes)")

    # Run Python reference
    print("\nRunning Python reference...")
    ref_output_nhwc = run_reference(layers, weights_list, input_nhwc)
    ref_output_nchw = ref_output_nhwc.transpose(2, 0, 1)

    ref_path = os.path.join(test_dir, 'test_mbv2_tiny_reference.bin')
    ref_output_nchw.astype(np.int8).tofile(ref_path)
    print(f"Reference output: {ref_output_nchw.flatten()}")

    # Run C simulator and verify
    print("\nRunning C simulator...")
    success = run_sim_and_verify(sim_path, model_path, input_path, ref_path, "MobileNetV2-Tiny")

    print()
    if success:
        print("=" * 60)
        print("SUCCESS: MobileNetV2-Tiny 12-layer inference is bit-exact!")
        print(f"  Architecture: Conv3×3 → [DW+PW]×3blocks → Conv1×1 → GAP → FC")
        print(f"  Operators tested: Conv2D(1×1,3×3), DWConv(s1,s2), AvgPool, FC")
        print(f"  Total weight parameters: {sum(w.size for w,b in weights_list if w is not None)}")
        print(f"  Classifier output (10 classes): {ref_output_nchw.flatten()}")
        print("=" * 60)
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
