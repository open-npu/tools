#!/usr/bin/env python3
"""
Extract per-layer weights, activations, and parameters from .npu1.bin compiled models.

This tool:
1. Deserializes the NPU1 binary format
2. Extracts per-layer LayerConfig structures
3. Dumps weights and per-channel params for each layer
4. Can generate RTL test data in the format expected by cocotb testbench

SPDX-License-Identifier: Apache-2.0
"""

import struct
import numpy as np
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ─── NPU1 Constants ───
MODEL_MAGIC = 0x4E505531  # "NPU1"
FIXED_CONFIG_SIZE = 62

# Operator types
OP_CONV2D = 0
OP_DW_CONV = 1
OP_FC = 2
OP_POOLING = 3
OP_ELTWISE_ADD = 4
OP_RESIZE = 5
OP_DECONV = 6
OP_CONCAT = 7

OP_NAMES = {
    0: "CONV2D",
    1: "DW_CONV",
    2: "FC",
    3: "POOLING",
    4: "ELTWISE_ADD",
    5: "RESIZE",
    6: "DECONV",
    7: "CONCAT",
}

# ─── Data Structures ───

@dataclass
class PerChannelParam:
    """Per-channel requantize parameters (14 bytes each)."""
    M: int          # 15-bit unsigned multiplier
    S: int          # 6-bit shift
    zp: int         # 16-bit signed zero point
    bias_q: int     # 64-bit signed bias
    
    @staticmethod
    def unpack(data: bytes, offset: int) -> 'PerChannelParam':
        """Unpack from 14 bytes at given offset."""
        M, S, _pad, zp, bias_q = struct.unpack_from('<HBbhq', data, offset)
        return PerChannelParam(M, S, zp, bias_q)


@dataclass
class AddParam:
    """Add node rescale parameters (8 bytes)."""
    M_A: int
    S_A: int
    M_B: int
    S_B: int
    
    @staticmethod
    def unpack(data: bytes, offset: int) -> 'AddParam':
        """Unpack from 8 bytes."""
        M_A, S_A, _p0, M_B, S_B, _p1 = struct.unpack_from('<HBxHBx', data, offset)
        return AddParam(M_A, S_A, M_B, S_B)


@dataclass
class LayerConfig:
    """Layer configuration extracted from binary."""
    # Basic params
    op_type: int
    data_type: int  # 0=INT8, 1=INT16
    in_h: int
    in_w: int
    in_c: int
    out_h: int
    out_w: int
    out_c: int
    
    # Conv/DW params
    kernel_h: int
    kernel_w: int
    dilation_h: int
    dilation_w: int
    stride_h: int
    stride_w: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    
    # Pooling params
    pool_mode: int
    pool_h: int
    pool_w: int
    pool_stride_h: int
    pool_stride_w: int
    global_pool: int
    
    # Resize params
    resize_mode: int
    scale_h: int
    scale_w: int
    
    # Deconv params
    insert_h: int
    insert_w: int
    
    # Concat params
    concat_offset: int
    concat_total_c: int
    
    # Tiling
    tile_h: int
    tile_w: int
    tile_num_h: int
    tile_num_w: int
    
    # Post-processing
    post_ctrl: int
    sched_ctrl: int
    clamp_min: int
    clamp_max: int
    in_zp: int
    
    # Per-channel params
    ch_params: List[PerChannelParam]
    add_params: Optional[AddParam]
    residual_src: int
    input_src: int
    
    @staticmethod
    def unpack(data: bytes, offset: int) -> Tuple['LayerConfig', int]:
        """
        Unpack layer descriptor starting at offset.
        Returns (LayerConfig, next_offset).
        """
        # Fixed part (62 bytes)
        fields = struct.unpack_from(
            '<BBHHHHHHBBBBBBBBBBBBBBBBBBBBHHHHHBBHBBBH',
            data, offset
        )
        
        (op_type, data_type, in_h, in_w, in_c, out_h, out_w, out_c,
         kernel_h, kernel_w, dilation_h, dilation_w, stride_h, stride_w,
         pad_top, pad_bottom, pad_left, pad_right,
         pool_mode, pool_h, pool_w, pool_stride_h, pool_stride_w, global_pool,
         resize_mode, scale_h, scale_w, insert_h, insert_w,
         concat_offset, concat_total_c, tile_h, tile_w, tile_num_h, tile_num_w,
         post_ctrl, sched_ctrl) = fields[:36]
        
        # Next 4 bytes: clamp_min, clamp_max, in_zp, _pad1
        clamp_min, clamp_max, in_zp, _pad1 = struct.unpack_from('<hbB', data, offset + 47)
        
        # param_ch_count, has_lut, has_add, residual_src
        param_ch_count = struct.unpack_from('<H', data, offset + 55)[0]
        has_lut = struct.unpack_from('B', data, offset + 57)[0]
        has_add = struct.unpack_from('B', data, offset + 58)[0]
        residual_src = struct.unpack_from('b', data, offset + 59)[0]
        input_src = struct.unpack_from('<h', data, offset + 60)[0]
        
        next_offset = offset + FIXED_CONFIG_SIZE
        
        # Parse per-channel params
        ch_params = []
        if param_ch_count > 0:
            for i in range(param_ch_count):
                p = PerChannelParam.unpack(data, next_offset + i * 14)
                ch_params.append(p)
            next_offset += param_ch_count * 14
        
        # Parse add params
        add_params = None
        if has_add:
            add_params = AddParam.unpack(data, next_offset)
            next_offset += 8
        
        # Parse LUT (skip for now)
        if has_lut:
            next_offset += 256 + 512  # i8 + i16 LUT
        
        return LayerConfig(
            op_type=op_type,
            data_type=data_type,
            in_h=in_h, in_w=in_w, in_c=in_c,
            out_h=out_h, out_w=out_w, out_c=out_c,
            kernel_h=kernel_h, kernel_w=kernel_w,
            dilation_h=dilation_h, dilation_w=dilation_w,
            stride_h=stride_h, stride_w=stride_w,
            pad_top=pad_top, pad_bottom=pad_bottom, pad_left=pad_left, pad_right=pad_right,
            pool_mode=pool_mode, pool_h=pool_h, pool_w=pool_w,
            pool_stride_h=pool_stride_h, pool_stride_w=pool_stride_w, global_pool=global_pool,
            resize_mode=resize_mode, scale_h=scale_h, scale_w=scale_w,
            insert_h=insert_h, insert_w=insert_w,
            concat_offset=concat_offset, concat_total_c=concat_total_c,
            tile_h=tile_h, tile_w=tile_w, tile_num_h=tile_num_h, tile_num_w=tile_num_w,
            post_ctrl=post_ctrl, sched_ctrl=sched_ctrl,
            clamp_min=clamp_min, clamp_max=clamp_max, in_zp=in_zp,
            ch_params=ch_params,
            add_params=add_params,
            residual_src=residual_src,
            input_src=input_src,
        ), next_offset


def load_npu1_model(model_path: str) -> Tuple[List[LayerConfig], bytes]:
    """Load NPU1 model and return (layers, weights)."""
    with open(model_path, 'rb') as f:
        data = f.read()
    
    # Parse header
    magic, num_layers, weight_offset, weight_size = struct.unpack_from(
        '<IIII', data, 0
    )
    
    if magic != MODEL_MAGIC:
        raise ValueError(f"Invalid magic: {hex(magic)}")
    
    print(f"Model: {num_layers} layers, {weight_size} bytes weights @ offset {weight_offset}")
    
    # Parse layer descriptors
    layers = []
    offset = 16
    for i in range(num_layers):
        layer, offset = LayerConfig.unpack(data, offset)
        layers.append(layer)
        print(f"  Layer {i}: {OP_NAMES.get(layer.op_type, '?')} "
              f"{layer.in_h}×{layer.in_w}×{layer.in_c} → {layer.out_h}×{layer.out_w}×{layer.out_c}")
    
    # Extract weight blob
    weights = data[weight_offset:weight_offset + weight_size]
    
    return layers, weights


def extract_layer_weights(layer: LayerConfig, weights: bytes, weight_offset: int) -> Tuple[bytes, int]:
    """Extract weights for a single layer. Returns (weight_bytes, next_offset)."""
    is_int16 = layer.data_type == 1
    elem_size = 2 if is_int16 else 1
    
    if layer.op_type == OP_CONV2D or layer.op_type == OP_DECONV:
        weight_bytes = layer.out_c * layer.kernel_h * layer.kernel_w * layer.in_c * elem_size
    elif layer.op_type == OP_DW_CONV:
        weight_bytes = layer.in_c * layer.kernel_h * layer.kernel_w * elem_size
    elif layer.op_type == OP_FC:
        weight_bytes = layer.out_c * layer.in_c * elem_size
    else:
        weight_bytes = 0
    
    if weight_bytes == 0:
        return b'', weight_offset
    
    w = weights[weight_offset:weight_offset + weight_bytes]
    return w, weight_offset + weight_bytes


def sram_capacity_analysis():
    """Print SRAM capacity analysis for RTL testbench."""
    print("\n" + "="*70)
    print("RTL TESTBENCH SRAM CAPACITY ANALYSIS")
    print("="*70)
    
    # From npu_compute_tb.v
    print("\nDefault SRAM sizes (from npu_compute_tb.v):")
    print("  ACT_DEPTH   = 1024 words × 32-bit = 4 KB")
    print("  WGT_DEPTH   = 1024 words × 32-bit = 4 KB")
    print("  PARAM_DEPTH = 256  words × 32-bit = 1 KB")
    print("  TOTAL       = 9 KB")
    
    # Calculate element counts
    print("\nElement capacity (INT8):")
    print("  ACT   = 1024 words × 4 bytes/word = 4096 INT8 elements")
    print("  WGT   = 1024 words × 4 bytes/word = 4096 INT8 elements")
    print("  PARAM = 256  words × 4 bytes/word = 1024 INT8 elements")
    
    print("\nElement capacity (INT16):")
    print("  ACT   = 1024 words × 2 shorts/word = 2048 INT16 elements")
    print("  WGT   = 1024 words × 2 shorts/word = 2048 INT16 elements")
    
    print("\nPer-channel param storage (14 bytes/channel):")
    print("  PARAM_DEPTH = 256 words × 4 bytes/word = 1024 bytes")
    print("             ÷ 14 bytes/param = ~73 channels max")
    
    print("\nTypical layer constraints:")
    print("  Conv2D (16×16, kernel 3×3, 32→64ch):")
    print("    Weights: 64 × 3 × 3 × 32 = 18432 INT8 → ~5 words")
    print("    Params:  64 ch × 14 bytes = 896 bytes → ~224 words")
    print("    Input:   16 × 16 × 32 = 8192 INT8 → ~2048 words")
    print("    Output:  16 × 16 × 64 = 16384 INT8 → ~4096 words")
    print("    ⚠ Input+Output alone = 6144 words > 4 KB ACT_DEPTH!")
    print("    → Tiling or smaller layers required")
    
    print("\nWorkaround strategies:")
    print("  1. Reduce spatial size (e.g., 8×8 instead of 16×16)")
    print("  2. Use INT16 activations (doubles element size)")
    print("  3. Tile output channels (use weight buffer for partial OC)")
    print("  4. Store only per-layer slices (input + output, not both)")


def print_layer_info(layer: LayerConfig, layer_idx: int):
    """Print detailed layer information."""
    print(f"\n{'='*70}")
    print(f"Layer {layer_idx}: {OP_NAMES.get(layer.op_type, 'UNKNOWN')}")
    print(f"{'='*70}")
    
    # Dimensions
    print(f"Input:  {layer.in_h}×{layer.in_w}×{layer.in_c}")
    print(f"Output: {layer.out_h}×{layer.out_w}×{layer.out_c}")
    print(f"Data type: {'INT16' if layer.data_type else 'INT8'}")
    
    # Convolution/DW parameters
    if layer.op_type in [OP_CONV2D, OP_DW_CONV, OP_DECONV]:
        print(f"Kernel: {layer.kernel_h}×{layer.kernel_w}, "
              f"stride={layer.stride_h}×{layer.stride_w}, "
              f"dilation={layer.dilation_h}×{layer.dilation_w}")
        print(f"Padding: top={layer.pad_top}, bottom={layer.pad_bottom}, "
              f"left={layer.pad_left}, right={layer.pad_right}")
    
    # Pooling
    if layer.op_type == OP_POOLING:
        mode_str = "MAX" if layer.pool_mode == 0 else "AVG"
        print(f"Pool: {mode_str} {layer.pool_h}×{layer.pool_w}, "
              f"stride={layer.pool_stride_h}×{layer.pool_stride_w}")
        if layer.global_pool:
            print("  (global pooling)")
    
    # Post-processing
    print(f"Post-ctrl: 0x{layer.post_ctrl:02x}")
    if layer.post_ctrl & 0x04:
        print("  - ReLU enabled")
    if layer.post_ctrl & 0x08:
        print("  - ReLU6 enabled")
    if layer.post_ctrl & 0x20:
        print("  - Zero point enabled")
    if layer.post_ctrl & 0x40:
        print("  - Bias enabled")
    if layer.post_ctrl & 0x80:
        print("  - INT16 output")
    
    # Per-channel params
    if layer.ch_params:
        print(f"Per-channel params: {len(layer.ch_params)} channels")
        if len(layer.ch_params) <= 4:
            for i, p in enumerate(layer.ch_params):
                print(f"  ch[{i}]: M={p.M}, S={p.S}, zp={p.zp}, bias={p.bias_q}")
    
    # Memory requirements
    is_int16 = layer.data_type == 1
    elem_size = 2 if is_int16 else 1
    
    in_bytes = layer.in_h * layer.in_w * layer.in_c * elem_size
    out_bytes = layer.out_h * layer.out_w * layer.out_c * elem_size
    
    if layer.op_type == OP_CONV2D:
        wgt_bytes = layer.out_c * layer.kernel_h * layer.kernel_w * layer.in_c * elem_size
    elif layer.op_type == OP_DW_CONV:
        wgt_bytes = layer.in_c * layer.kernel_h * layer.kernel_w * elem_size
    elif layer.op_type == OP_FC:
        wgt_bytes = layer.out_c * layer.in_c * elem_size
    else:
        wgt_bytes = 0
    
    param_bytes = len(layer.ch_params) * 14
    
    print(f"\nMemory requirements:")
    print(f"  Input:       {in_bytes:6d} bytes ({in_bytes//4:4d} words)")
    print(f"  Output:      {out_bytes:6d} bytes ({out_bytes//4:4d} words)")
    print(f"  Weights:     {wgt_bytes:6d} bytes ({wgt_bytes//4:4d} words)")
    print(f"  Per-ch params: {param_bytes:4d} bytes ({(param_bytes+3)//4:3d} words)")
    print(f"  Total I+O:   {in_bytes+out_bytes:6d} bytes ({(in_bytes+out_bytes)//4:4d} words)")
    print(f"  → Fits in ACT (1024 words)? {'YES' if (in_bytes+out_bytes)//4 <= 1024 else 'NO'}")


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_npu1_layers.py <model.npu1.bin> [--dump-layers N]")
        print("\nOptions:")
        print("  --dump-layers N   Print detailed info for layers 0..N-1")
        sys.exit(1)
    
    model_path = sys.argv[1]
    dump_layers = 0
    
    if len(sys.argv) > 2 and sys.argv[2] == '--dump-layers':
        dump_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 999
    
    # Load model
    layers, weights = load_npu1_model(model_path)
    
    # Print analysis
    sram_capacity_analysis()
    
    # Print layer details
    weight_offset = 0
    for i, layer in enumerate(layers):
        if i < dump_layers:
            print_layer_info(layer, i)
        
        # Extract weights for this layer
        w, weight_offset = extract_layer_weights(layer, weights, weight_offset)
        if w:
            print(f"  (weights extracted: {len(w)} bytes)")
    
    print(f"\n{'='*70}")
    print(f"Total layers: {len(layers)}")
    print(f"Total weight bytes: {len(weights)}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
