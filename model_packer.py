#!/usr/bin/env python3
"""
Open-NPU Model Packer (V2: per-channel requantize)
Packs layer configs + weights into the binary format expected by npu_sim.

Binary format "NPU1":
  Header (16 bytes):
    uint32 magic = 0x4E505531
    uint32 num_layers
    uint32 weight_offset
    uint32 weight_size
  Layer descriptors (variable-length, concatenated):
    fixed_config (60 bytes) + per-channel params + [add params] + [LUT]
  Weight blob (at weight_offset)

SPDX-License-Identifier: Apache-2.0
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

MODEL_MAGIC = 0x4E505531  # "NPU1"
FIXED_CONFIG_SIZE = 60

# Operator types
OP_CONV2D = 0
OP_DW_CONV = 1
OP_FC = 2
OP_POOLING = 3
OP_ELTWISE_ADD = 4
OP_RESIZE = 5
OP_DECONV = 6
OP_CONCAT = 7

# POST_CTRL bits (matches CSR 0x180)
POST_PPU_MODE_MASK = 0x03
PPU_MODE_CONV_REQ = 0
PPU_MODE_ADD = 1
PPU_MODE_RELU_ONLY = 2
PPU_MODE_PASSTHROUGH = 3

POST_RELU_EN = (1 << 2)
POST_RELU6_EN = (1 << 3)
POST_LUT_EN = (1 << 4)
POST_ZP_EN = (1 << 5)
POST_BIAS_EN = (1 << 6)
POST_INT16_OUT = (1 << 7)


@dataclass
class PerChannelParam:
    """Per-channel requantize parameters (10 bytes)."""
    M: int = 1       # 15-bit unsigned multiplier
    S: int = 0       # 6-bit shift amount
    zp: int = 0      # 16-bit signed zero point
    bias_q: int = 0  # 32-bit signed bias

    def pack(self) -> bytes:
        """Pack to 10 bytes matching perchannel_param_t."""
        return struct.pack('<HBbhI',
                           self.M & 0x7FFF,  # uint16 M
                           self.S & 0x3F,     # uint8 S
                           0,                 # reserved
                           self.zp,           # int16 zp
                           self.bias_q & 0xFFFFFFFF)  # uint32 (store as unsigned bits)

    @staticmethod
    def pack_array(params: list) -> bytes:
        """Pack array of PerChannelParam to bytes."""
        return b''.join(p.pack() for p in params)


@dataclass
class AddParam:
    """Add node rescale parameters (8 bytes)."""
    M_A: int = 1
    S_A: int = 0
    M_B: int = 1
    S_B: int = 0

    def pack(self) -> bytes:
        """Pack to 8 bytes matching add_param_t."""
        return struct.pack('<HBxHBx',
                           self.M_A & 0x7FFF,
                           self.S_A & 0x3F,
                           self.M_B & 0x7FFF,
                           self.S_B & 0x3F)


@dataclass
class LayerConfig:
    """Layer configuration for NPU1 format."""
    op_type: int = 0
    data_type: int = 0  # 0=INT8, 1=INT16
    in_h: int = 0
    in_w: int = 0
    in_c: int = 0
    out_h: int = 0
    out_w: int = 0
    out_c: int = 0
    kernel_h: int = 1
    kernel_w: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    pool_mode: int = 0
    pool_h: int = 0
    pool_w: int = 0
    pool_stride_h: int = 0
    pool_stride_w: int = 0
    global_pool: int = 0
    resize_mode: int = 0
    scale_h: int = 0
    scale_w: int = 0
    insert_h: int = 0
    insert_w: int = 0
    concat_offset: int = 0
    concat_total_c: int = 0
    tile_h: int = 0
    tile_w: int = 0
    tile_num_h: int = 0
    tile_num_w: int = 0
    post_ctrl: int = 0
    clamp_min: int = -128
    clamp_max: int = 127
    in_zp: int = 0

    # Per-channel params (list of PerChannelParam, one per out_c)
    ch_params: List[PerChannelParam] = field(default_factory=list)
    # Add params (single AddParam or None)
    add_params: Optional[AddParam] = None
    # LUT data
    lut_i8: bytes = field(default_factory=lambda: bytes(256))
    lut_i16: bytes = field(default_factory=lambda: bytes(512))
    has_lut: bool = False

    def pack_fixed(self) -> bytes:
        """Pack the 60-byte fixed config portion."""
        buf = bytearray(FIXED_CONFIG_SIZE)
        off = 0

        def w8(v): nonlocal off; struct.pack_into('B', buf, off, v & 0xFF); off += 1
        def w8s(v): nonlocal off; struct.pack_into('b', buf, off, v); off += 1
        def w16(v): nonlocal off; struct.pack_into('<H', buf, off, v & 0xFFFF); off += 2
        def w16s(v): nonlocal off; struct.pack_into('<h', buf, off, v); off += 2

        w8(self.op_type)       # 0
        w8(self.data_type)     # 1
        w16(self.in_h)         # 2
        w16(self.in_w)         # 4
        w16(self.in_c)         # 6
        w16(self.out_h)        # 8
        w16(self.out_w)        # 10
        w16(self.out_c)        # 12
        w8(self.kernel_h)      # 14
        w8(self.kernel_w)      # 15
        w8(self.dilation_h)    # 16
        w8(self.dilation_w)    # 17
        w8(self.stride_h)      # 18
        w8(self.stride_w)      # 19
        w8(self.pad_top)       # 20
        w8(self.pad_bottom)    # 21
        w8(self.pad_left)      # 22
        w8(self.pad_right)     # 23
        w8(self.pool_mode)     # 24
        w8(self.pool_h)        # 25
        w8(self.pool_w)        # 26
        w8(self.pool_stride_h) # 27
        w8(self.pool_stride_w) # 28
        w8(self.global_pool)   # 29
        w8(self.resize_mode)   # 30
        w8(self.scale_h)       # 31
        w8(self.scale_w)       # 32
        w8(self.insert_h)      # 33
        w8(self.insert_w)      # 34
        # concat: need uint16, but off=35 is odd. Pack with no padding (packed struct)
        w16(self.concat_offset)   # 35
        w16(self.concat_total_c)  # 37
        w16(self.tile_h)       # 39
        w16(self.tile_w)       # 41
        w16(self.tile_num_h)   # 43
        w16(self.tile_num_w)   # 45
        w8(self.post_ctrl)     # 47
        w8s(0)                 # 48: _pad0
        w16s(self.clamp_min)   # 49
        w16s(self.clamp_max)   # 51
        w8s(self.in_zp)        # 53
        w8(0)                  # 54: _pad1
        w16(len(self.ch_params))  # 55: param_ch_count
        w8(1 if self.has_lut else 0)  # 57: has_lut
        w8(1 if self.add_params else 0)  # 58: has_add
        w8(0)                  # 59: _reserved[1]

        assert off == FIXED_CONFIG_SIZE, f"Expected {FIXED_CONFIG_SIZE}, got {off}"
        return bytes(buf)

    def pack_descriptor(self) -> bytes:
        """Pack full variable-length layer descriptor."""
        parts = [self.pack_fixed()]

        # Per-channel params
        if self.ch_params:
            parts.append(PerChannelParam.pack_array(self.ch_params))

        # Add params
        if self.add_params:
            parts.append(self.add_params.pack())

        # LUT
        if self.has_lut:
            parts.append(self.lut_i8[:256])
            parts.append(self.lut_i16[:512])

        return b''.join(parts)


def pack_model(layers: list, weight_data: bytes, output_path: str):
    """
    Pack a complete model to binary file (NPU1 format).

    Args:
        layers: list of LayerConfig objects
        weight_data: concatenated weights for all layers (bias now in ch_params)
        output_path: output .bin file path
    """
    num_layers = len(layers)
    weight_size = len(weight_data)

    # Serialize layer descriptors
    descriptors_bin = b''.join(layer.pack_descriptor() for layer in layers)

    # weight_offset = header(16) + descriptors
    weight_offset = 16 + len(descriptors_bin)

    # Header
    header = struct.pack('<IIII', MODEL_MAGIC, num_layers, weight_offset, weight_size)

    # Write
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(descriptors_bin)
        f.write(weight_data)

    total = len(header) + len(descriptors_bin) + len(weight_data)
    print(f"Model packed: {output_path} ({total} bytes)")
    print(f"  Layers: {num_layers}, descriptors: {len(descriptors_bin)} bytes")
    print(f"  Weights: {weight_size} bytes @ offset {weight_offset}")


# ─── Reference implementations for verification ───

def ref_conv2d(input_nhwc, weights, cfg):
    """Reference Conv2D in Python (bit-exact INT32 accumulator)."""
    oh, ow, oc = cfg.out_h, cfg.out_w, cfg.out_c
    kh, kw = cfg.kernel_h, cfg.kernel_w
    sh, sw = cfg.stride_h, cfg.stride_w
    dh, dw = cfg.dilation_h, cfg.dilation_w
    pt, pl = cfg.pad_top, cfg.pad_left
    in_h, in_w, in_c = cfg.in_h, cfg.in_w, cfg.in_c

    acc = np.zeros((oh, ow, oc), dtype=np.int64)
    for o_h in range(oh):
        for o_w in range(ow):
            for o_c in range(oc):
                s = np.int64(0)
                for fh in range(kh):
                    ih = o_h * sh - pt + fh * dh
                    if ih < 0 or ih >= in_h:
                        continue
                    for fw in range(kw):
                        iw = o_w * sw - pl + fw * dw
                        if iw < 0 or iw >= in_w:
                            continue
                        for ic in range(in_c):
                            s += np.int64(input_nhwc[ih, iw, ic]) * np.int64(
                                weights[o_c, fh, fw, ic])
                acc[o_h, o_w, o_c] = s
    return acc


def ref_dwconv(input_nhwc, weights, cfg):
    """Reference Depthwise Conv (bit-exact INT64 accumulator).
    weights shape: [channels][kh][kw]
    """
    oh, ow = cfg.out_h, cfg.out_w
    ch = cfg.in_c
    kh, kw = cfg.kernel_h, cfg.kernel_w
    sh, sw = cfg.stride_h, cfg.stride_w
    dh, dw = cfg.dilation_h, cfg.dilation_w
    pt, pl = cfg.pad_top, cfg.pad_left
    in_h, in_w = cfg.in_h, cfg.in_w

    acc = np.zeros((oh, ow, ch), dtype=np.int64)
    for o_h in range(oh):
        for o_w in range(ow):
            for c in range(ch):
                s = np.int64(0)
                for fh in range(kh):
                    ih = o_h * sh - pt + fh * dh
                    if ih < 0 or ih >= in_h:
                        continue
                    for fw in range(kw):
                        iw = o_w * sw - pl + fw * dw
                        if iw < 0 or iw >= in_w:
                            continue
                        s += np.int64(input_nhwc[ih, iw, c]) * np.int64(weights[c, fh, fw])
                acc[o_h, o_w, c] = s
    return acc


def ref_postproc_perchannel(acc, ch_params, cfg):
    """Per-channel requantize reference (bit-exact, matches PPU CONV_REQ mode)."""
    shape = acc.shape
    out_c = shape[-1]
    result = np.zeros(shape, dtype=np.int64)

    for c in range(out_c):
        p = ch_params[c]
        ch_acc = acc[..., c].astype(np.int64)

        # Step 1: bias
        if cfg.post_ctrl & POST_BIAS_EN:
            ch_acc = ch_acc + np.int64(p.bias_q)

        # Step 2: multiply by M
        M = np.int64(p.M & 0x7FFF)
        product = ch_acc * M

        # Step 3: rounding right shift by S
        S = int(p.S & 0x3F)
        if S > 0:
            result[..., c] = (product + (np.int64(1) << (S - 1))) >> S
        else:
            result[..., c] = product

        # Step 4: zero point
        if cfg.post_ctrl & POST_ZP_EN:
            result[..., c] = result[..., c] + np.int64(p.zp)

    # Step 5: clamp
    result = np.clip(result, cfg.clamp_min, cfg.clamp_max)

    # Step 6: activation
    if cfg.post_ctrl & POST_RELU_EN:
        result = np.maximum(result, 0)
    elif cfg.post_ctrl & POST_RELU6_EN:
        result = np.clip(result, 0, cfg.clamp_max)

    if cfg.post_ctrl & POST_INT16_OUT:
        return result.astype(np.int16)
    return result.astype(np.int8)


def ref_postproc_add(input_a, input_b, add_params, cfg):
    """Add mode reference: dual rescale + sum + activation."""
    a = input_a.astype(np.int64)
    b = input_b.astype(np.int64)

    # Rescale A
    M_A = np.int64(add_params.M_A & 0x7FFF)
    S_A = int(add_params.S_A & 0x3F)
    prod_a = a * M_A
    if S_A > 0:
        rescaled_a = (prod_a + (np.int64(1) << (S_A - 1))) >> S_A
    else:
        rescaled_a = prod_a

    # Rescale B
    M_B = np.int64(add_params.M_B & 0x7FFF)
    S_B = int(add_params.S_B & 0x3F)
    prod_b = b * M_B
    if S_B > 0:
        rescaled_b = (prod_b + (np.int64(1) << (S_B - 1))) >> S_B
    else:
        rescaled_b = prod_b

    # Sum + clamp + activation
    result = rescaled_a + rescaled_b
    result = np.clip(result, cfg.clamp_min, cfg.clamp_max)

    if cfg.post_ctrl & POST_RELU_EN:
        result = np.maximum(result, 0)

    if cfg.post_ctrl & POST_INT16_OUT:
        return result.astype(np.int16)
    return result.astype(np.int8)


def ref_pooling(input_nhwc, cfg):
    """Reference Pooling (bit-exact)."""
    oh, ow = cfg.out_h, cfg.out_w
    ch = cfg.in_c
    in_h, in_w = cfg.in_h, cfg.in_w

    if cfg.global_pool:
        pool_h, pool_w = in_h, in_w
        pool_sh, pool_sw = in_h, in_w
    else:
        pool_h, pool_w = cfg.pool_h, cfg.pool_w
        pool_sh, pool_sw = cfg.pool_stride_h, cfg.pool_stride_w

    pt, pl = cfg.pad_top, cfg.pad_left
    is_avg = (cfg.pool_mode == 1)

    acc = np.zeros((oh, ow, ch), dtype=np.int32)
    for o_h in range(oh):
        for o_w in range(ow):
            for c in range(ch):
                if is_avg:
                    result = np.int32(0)
                    count = 0
                else:
                    result = np.int32(-2**31)

                for ph in range(pool_h):
                    ih = o_h * pool_sh - pt + ph
                    if ih < 0 or ih >= in_h:
                        continue
                    for pw in range(pool_w):
                        iw = o_w * pool_sw - pl + pw
                        if iw < 0 or iw >= in_w:
                            continue
                        val = np.int32(input_nhwc[ih, iw, c])
                        if is_avg:
                            result += val
                            count += 1
                        else:
                            if val > result:
                                result = val

                if is_avg and count > 0:
                    if result >= 0:
                        result = (result + count // 2) // count
                    else:
                        result = (result - count // 2) // count

                acc[o_h, o_w, c] = result
    return acc


# ─── Helper: create per-channel params from numpy arrays ───

def make_ch_params(M_arr, S_arr, bias_arr, zp_arr=None):
    """Create list of PerChannelParam from arrays."""
    out_c = len(M_arr)
    if zp_arr is None:
        zp_arr = np.zeros(out_c, dtype=np.int16)
    params = []
    for c in range(out_c):
        params.append(PerChannelParam(
            M=int(M_arr[c]),
            S=int(S_arr[c]),
            zp=int(zp_arr[c]),
            bias_q=int(bias_arr[c])
        ))
    return params


# ─── Test model builders ───

def build_test_conv(output_dir: str):
    """
    Test: 2-layer Conv2D with per-channel requantize.
      Layer 0: Conv2D 3×3, 1→4 channels, 8×8, pad=1 → 8×8×4
      Layer 1: Conv2D 1×1, 4→2 channels + ReLU → 8×8×2
    """
    import os
    np.random.seed(42)

    # Layer 0: Conv2D 3×3, in_c=1, out_c=4
    layer0 = LayerConfig()
    layer0.op_type = OP_CONV2D
    layer0.in_h, layer0.in_w, layer0.in_c = 8, 8, 1
    layer0.out_h, layer0.out_w, layer0.out_c = 8, 8, 4
    layer0.kernel_h, layer0.kernel_w = 3, 3
    layer0.dilation_h, layer0.dilation_w = 1, 1
    layer0.stride_h, layer0.stride_w = 1, 1
    layer0.pad_top = layer0.pad_bottom = layer0.pad_left = layer0.pad_right = 1
    layer0.post_ctrl = POST_BIAS_EN | POST_ZP_EN | PPU_MODE_CONV_REQ
    layer0.clamp_min, layer0.clamp_max = -128, 127

    # Per-channel params for layer 0 (4 channels)
    M0 = np.array([16384, 12000, 20000, 8000], dtype=np.uint16)
    S0 = np.array([15, 14, 16, 13], dtype=np.uint8)
    bias0 = np.random.randint(-50, 50, (4,), dtype=np.int32)
    zp0 = np.array([2, -1, 0, 3], dtype=np.int16)
    layer0.ch_params = make_ch_params(M0, S0, bias0, zp0)

    # Layer 1: Conv2D 1×1, in_c=4, out_c=2 + ReLU
    layer1 = LayerConfig()
    layer1.op_type = OP_CONV2D
    layer1.in_h, layer1.in_w, layer1.in_c = 8, 8, 4
    layer1.out_h, layer1.out_w, layer1.out_c = 8, 8, 2
    layer1.kernel_h, layer1.kernel_w = 1, 1
    layer1.dilation_h, layer1.dilation_w = 1, 1
    layer1.stride_h, layer1.stride_w = 1, 1
    layer1.post_ctrl = POST_BIAS_EN | POST_RELU_EN | PPU_MODE_CONV_REQ
    layer1.clamp_min, layer1.clamp_max = -128, 127

    M1 = np.array([15000, 18000], dtype=np.uint16)
    S1 = np.array([14, 15], dtype=np.uint8)
    bias1 = np.random.randint(-30, 30, (2,), dtype=np.int32)
    zp1 = np.zeros(2, dtype=np.int16)
    layer1.ch_params = make_ch_params(M1, S1, bias1, zp1)

    # Generate weights (no separate bias — it's in ch_params)
    w0 = np.random.randint(-10, 10, (4, 3, 3, 1), dtype=np.int8)
    w1 = np.random.randint(-10, 10, (2, 1, 1, 4), dtype=np.int8)
    weight_data = w0.tobytes() + w1.tobytes()

    # Input (NCHW)
    input_nchw = np.random.randint(-50, 50, (1, 8, 8), dtype=np.int8)

    # Pack model
    model_path = os.path.join(output_dir, 'test_model.bin')
    pack_model([layer0, layer1], weight_data, model_path)

    # Save input
    input_path = os.path.join(output_dir, 'test_input.bin')
    input_nchw.tofile(input_path)

    # Compute reference
    input_nhwc = input_nchw.transpose(1, 2, 0)  # [H][W][C]
    acc0 = ref_conv2d(input_nhwc, w0, layer0)
    out0 = ref_postproc_perchannel(acc0, layer0.ch_params, layer0)
    acc1 = ref_conv2d(out0, w1, layer1)
    out1 = ref_postproc_perchannel(acc1, layer1.ch_params, layer1)

    output_nchw = out1.transpose(2, 0, 1)
    ref_path = os.path.join(output_dir, 'test_reference.bin')
    output_nchw.tofile(ref_path)
    print(f"Conv test: input={input_nchw.shape}, output={output_nchw.shape}")
    return model_path, input_path, ref_path


def build_test_dwconv(output_dir: str):
    """
    Test: Conv → DWConv → Conv (MobileNet-style)
      Layer 0: Conv2D 3×3, 1→8, 8×8, pad=1
      Layer 1: DWConv 3×3, 8ch, stride=2, pad=1 → 4×4×8
      Layer 2: Conv2D 1×1, 8→4 + ReLU
    """
    import os
    np.random.seed(123)

    layer0 = LayerConfig(op_type=OP_CONV2D,
                         in_h=8, in_w=8, in_c=1,
                         out_h=8, out_w=8, out_c=8,
                         kernel_h=3, kernel_w=3,
                         dilation_h=1, dilation_w=1,
                         stride_h=1, stride_w=1,
                         pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
                         post_ctrl=POST_BIAS_EN | PPU_MODE_CONV_REQ,
                         clamp_min=-128, clamp_max=127)
    M0 = np.full(8, 16000, dtype=np.uint16)
    S0 = np.full(8, 15, dtype=np.uint8)
    bias0 = np.random.randint(-20, 20, (8,), dtype=np.int32)
    layer0.ch_params = make_ch_params(M0, S0, bias0)

    layer1 = LayerConfig(op_type=OP_DW_CONV,
                         in_h=8, in_w=8, in_c=8,
                         out_h=4, out_w=4, out_c=8,
                         kernel_h=3, kernel_w=3,
                         dilation_h=1, dilation_w=1,
                         stride_h=2, stride_w=2,
                         pad_top=1, pad_bottom=1, pad_left=1, pad_right=1,
                         post_ctrl=POST_BIAS_EN | PPU_MODE_CONV_REQ,
                         clamp_min=-128, clamp_max=127)
    M1 = np.full(8, 14000, dtype=np.uint16)
    S1 = np.full(8, 14, dtype=np.uint8)
    bias1 = np.random.randint(-10, 10, (8,), dtype=np.int32)
    layer1.ch_params = make_ch_params(M1, S1, bias1)

    layer2 = LayerConfig(op_type=OP_CONV2D,
                         in_h=4, in_w=4, in_c=8,
                         out_h=4, out_w=4, out_c=4,
                         kernel_h=1, kernel_w=1,
                         dilation_h=1, dilation_w=1,
                         stride_h=1, stride_w=1,
                         post_ctrl=POST_BIAS_EN | POST_RELU_EN | PPU_MODE_CONV_REQ,
                         clamp_min=-128, clamp_max=127)
    M2 = np.full(4, 12000, dtype=np.uint16)
    S2 = np.full(4, 14, dtype=np.uint8)
    bias2 = np.random.randint(-10, 10, (4,), dtype=np.int32)
    layer2.ch_params = make_ch_params(M2, S2, bias2)

    # Weights
    w0 = np.random.randint(-8, 8, (8, 3, 3, 1), dtype=np.int8)
    w1 = np.random.randint(-8, 8, (8, 3, 3), dtype=np.int8)
    w2 = np.random.randint(-8, 8, (4, 1, 1, 8), dtype=np.int8)
    weight_data = w0.tobytes() + w1.tobytes() + w2.tobytes()

    input_nchw = np.random.randint(-30, 30, (1, 8, 8), dtype=np.int8)
    model_path = os.path.join(output_dir, 'test_dwconv_model.bin')
    pack_model([layer0, layer1, layer2], weight_data, model_path)
    input_path = os.path.join(output_dir, 'test_dwconv_input.bin')
    input_nchw.tofile(input_path)

    # Reference
    inp = input_nchw.transpose(1, 2, 0)
    acc0 = ref_conv2d(inp, w0, layer0)
    out0 = ref_postproc_perchannel(acc0, layer0.ch_params, layer0)
    acc1 = ref_dwconv(out0, w1, layer1)
    out1 = ref_postproc_perchannel(acc1, layer1.ch_params, layer1)
    acc2 = ref_conv2d(out1, w2, layer2)
    out2 = ref_postproc_perchannel(acc2, layer2.ch_params, layer2)

    output_nchw = out2.transpose(2, 0, 1)
    ref_path = os.path.join(output_dir, 'test_dwconv_reference.bin')
    output_nchw.tofile(ref_path)
    print(f"DWConv test: input={input_nchw.shape}, output={output_nchw.shape}")
    return model_path, input_path, ref_path


def build_test_add(output_dir: str):
    """
    Test: Eltwise Add with dual rescale (residual connection).
    Simulates two branches with different scales being added.
    """
    import os
    np.random.seed(999)

    h, w, c = 4, 4, 8

    layer = LayerConfig(op_type=OP_ELTWISE_ADD,
                        in_h=h, in_w=w, in_c=c,
                        out_h=h, out_w=h, out_c=c,
                        post_ctrl=PPU_MODE_ADD | POST_RELU_EN,
                        clamp_min=-128, clamp_max=127)
    layer.add_params = AddParam(M_A=16000, S_A=14, M_B=12000, S_B=13)

    # Inputs (two quantized tensors with different scales)
    input_a = np.random.randint(-50, 50, (h, w, c), dtype=np.int8)
    input_b = np.random.randint(-40, 40, (h, w, c), dtype=np.int8)

    # No weights for add
    weight_data = b''

    model_path = os.path.join(output_dir, 'test_add_model.bin')
    pack_model([layer], weight_data, model_path)

    # Save inputs: A is the "main" input, B is packed after in the input file
    # For testing, we'll concatenate A and B in the input file
    input_path = os.path.join(output_dir, 'test_add_input.bin')
    # NCHW format for both
    a_nchw = input_a.transpose(2, 0, 1)
    b_nchw = input_b.transpose(2, 0, 1)
    combined = np.concatenate([a_nchw.flatten(), b_nchw.flatten()])
    combined.tofile(input_path)

    # Reference
    out = ref_postproc_add(input_a, input_b, layer.add_params, layer)
    output_nchw = out.transpose(2, 0, 1)
    ref_path = os.path.join(output_dir, 'test_add_reference.bin')
    output_nchw.tofile(ref_path)
    print(f"Add test: input_a={input_a.shape}, output={output_nchw.shape}")
    return model_path, input_path, ref_path


# ─── Standalone verification (Python-only, no C sim) ───

def selftest():
    """Run Python-only verification of reference implementations."""
    print("=== Python Self-Test: per-channel requantize ===")

    # Test 1: simple per-channel requant
    acc = np.array([[[100, -200, 50, 300]]], dtype=np.int64)  # 1×1×4
    params = [
        PerChannelParam(M=16384, S=15, zp=2, bias_q=10),
        PerChannelParam(M=12000, S=14, zp=-1, bias_q=-5),
        PerChannelParam(M=20000, S=16, zp=0, bias_q=0),
        PerChannelParam(M=8000, S=13, zp=3, bias_q=20),
    ]

    class FakeCfg:
        post_ctrl = POST_BIAS_EN | POST_ZP_EN | PPU_MODE_CONV_REQ
        clamp_min = -128
        clamp_max = 127

    out = ref_postproc_perchannel(acc, params, FakeCfg())
    print(f"  Input acc: {acc.flatten()}")
    print(f"  Output:    {out.flatten()}")

    # Manual verify channel 0: (100+10)*16384 >> 15 + 2
    ch0_manual = ((100 + 10) * 16384 + (1 << 14)) >> 15
    ch0_manual += 2
    ch0_manual = max(-128, min(127, ch0_manual))
    assert out[0, 0, 0] == ch0_manual, f"ch0: expected {ch0_manual}, got {out[0,0,0]}"
    print(f"  Channel 0 verified: {ch0_manual}")

    # Test 2: Add mode
    a = np.array([[[10, -20]]], dtype=np.int8)
    b = np.array([[[30, -10]]], dtype=np.int8)
    add_p = AddParam(M_A=16000, S_A=14, M_B=12000, S_B=13)

    class FakeCfgAdd:
        post_ctrl = PPU_MODE_ADD | POST_RELU_EN
        clamp_min = -128
        clamp_max = 127

    out_add = ref_postproc_add(a, b, add_p, FakeCfgAdd())
    print(f"  Add input A: {a.flatten()}, B: {b.flatten()}")
    print(f"  Add output:  {out_add.flatten()}")

    # Manual verify channel 0:
    ra = (10 * 16000 + (1 << 13)) >> 14
    rb = (30 * 12000 + (1 << 12)) >> 13
    s = ra + rb
    s = max(0, min(127, s))  # ReLU + clamp
    assert out_add[0, 0, 0] == s, f"add ch0: expected {s}, got {out_add[0,0,0]}"
    print(f"  Add channel 0 verified: {s}")

    print("=== All self-tests PASSED ===\n")


# ─── Main entry ───

if __name__ == '__main__':
    import sys
    import os

    sim_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'csim', 'npu_sim')
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'csim', 'testdata')

    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        selftest()

    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        os.makedirs(test_dir, exist_ok=True)
        selftest()

        if not os.path.exists(sim_path):
            print(f"Simulator not found at {sim_path}")
            print("Build it first: cd csim && make")
            print("(Skipping C simulator tests, Python-only passed)")
            sys.exit(0)

        # End-to-end tests would go here
        print("End-to-end C sim tests: TODO (run after csim build)")

    elif len(sys.argv) > 1 and sys.argv[1] == 'pack':
        os.makedirs(test_dir, exist_ok=True)
        print("── Generating test data ──")
        build_test_conv(test_dir)
        build_test_dwconv(test_dir)
        build_test_add(test_dir)
        print("\nTest data generated in:", test_dir)

    else:
        print("Usage:")
        print("  python3 model_packer.py selftest  Run Python-only reference tests")
        print("  python3 model_packer.py test      Run all tests (Python + C sim)")
        print("  python3 model_packer.py pack      Generate test model binaries")
