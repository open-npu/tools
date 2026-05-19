#!/usr/bin/env python3
"""
Open-NPU Model Packer
Packs layer configs + weights into the binary format expected by npu_sim.

Binary format:
  Header (16 bytes):
    uint32 magic = 0x4E505530 ("NPU0")
    uint32 num_layers
    uint32 weight_offset  (unused, for future)
    uint32 weight_size
  Layer configs: num_layers × 828 bytes (layer_config_t)
  Weights: concatenated weight+bias data for all layers

SPDX-License-Identifier: Apache-2.0
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

MODEL_MAGIC = 0x4E505530
LAYER_CONFIG_SIZE = 828

# Operator types
OP_CONV2D = 0
OP_DW_CONV = 1
OP_FC = 2
OP_POOLING = 3
OP_ELTWISE_ADD = 4
OP_RESIZE = 5
OP_DECONV = 6
OP_CONCAT = 7

# Post-processing control bits
POST_BIAS_EN = (1 << 0)
POST_SHIFT_EN = (1 << 1)
POST_SCALE_EN = (1 << 2)
POST_CLAMP_EN = (1 << 3)
POST_LUT_EN = (1 << 4)
POST_ELTWISE_EN = (1 << 5)
POST_POOL_EN = (1 << 6)
POST_OUT_INT16 = (1 << 7)


@dataclass
class LayerConfig:
    """Layer configuration matching C struct layout."""
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
    shift_bits: int = 0
    round_en: int = 0
    scale: int = 0  # post-processing scale (int16)
    in_zp: int = 0
    weight_zp: int = 0
    out_zp: int = 0
    clamp_min: int = -128
    clamp_max: int = 127
    bias_shift: int = 0
    lut_i8: bytes = field(default_factory=lambda: bytes(256))
    lut_i16: bytes = field(default_factory=lambda: bytes(512))

    def pack(self) -> bytes:
        """Pack to binary matching C struct layout (828 bytes)."""
        buf = bytearray(LAYER_CONFIG_SIZE)

        # Use struct offsets from C analysis
        struct.pack_into('B', buf, 0, self.op_type)
        struct.pack_into('B', buf, 1, self.data_type)
        struct.pack_into('<H', buf, 2, self.in_h)
        struct.pack_into('<H', buf, 4, self.in_w)
        struct.pack_into('<H', buf, 6, self.in_c)
        struct.pack_into('<H', buf, 8, self.out_h)
        struct.pack_into('<H', buf, 10, self.out_w)
        struct.pack_into('<H', buf, 12, self.out_c)
        struct.pack_into('B', buf, 14, self.kernel_h)
        struct.pack_into('B', buf, 15, self.kernel_w)
        struct.pack_into('B', buf, 16, self.dilation_h)
        struct.pack_into('B', buf, 17, self.dilation_w)
        struct.pack_into('B', buf, 18, self.stride_h)
        struct.pack_into('B', buf, 19, self.stride_w)
        struct.pack_into('B', buf, 20, self.pad_top)
        struct.pack_into('B', buf, 21, self.pad_bottom)
        struct.pack_into('B', buf, 22, self.pad_left)
        struct.pack_into('B', buf, 23, self.pad_right)
        struct.pack_into('B', buf, 24, self.pool_mode)
        struct.pack_into('B', buf, 25, self.pool_h)
        struct.pack_into('B', buf, 26, self.pool_w)
        struct.pack_into('B', buf, 27, self.pool_stride_h)
        struct.pack_into('B', buf, 28, self.pool_stride_w)
        struct.pack_into('B', buf, 29, self.global_pool)
        struct.pack_into('B', buf, 30, self.resize_mode)
        struct.pack_into('B', buf, 31, self.scale_h)
        struct.pack_into('B', buf, 32, self.scale_w)
        struct.pack_into('B', buf, 33, self.insert_h)
        struct.pack_into('B', buf, 34, self.insert_w)
        # padding byte at 35
        struct.pack_into('<H', buf, 36, self.concat_offset)
        struct.pack_into('<H', buf, 38, self.concat_total_c)
        struct.pack_into('<H', buf, 40, self.tile_h)
        struct.pack_into('<H', buf, 42, self.tile_w)
        struct.pack_into('<H', buf, 44, self.tile_num_h)
        struct.pack_into('<H', buf, 46, self.tile_num_w)
        struct.pack_into('B', buf, 48, self.post_ctrl)
        struct.pack_into('B', buf, 49, self.shift_bits)
        struct.pack_into('B', buf, 50, self.round_en)
        # padding byte at 51
        struct.pack_into('<h', buf, 52, self.scale)  # int16_t
        struct.pack_into('b', buf, 54, self.in_zp)
        struct.pack_into('b', buf, 55, self.weight_zp)
        struct.pack_into('b', buf, 56, self.out_zp)
        struct.pack_into('b', buf, 57, self.clamp_min)
        struct.pack_into('b', buf, 58, self.clamp_max)
        struct.pack_into('B', buf, 59, self.bias_shift)

        # LUT data
        buf[60:60+256] = self.lut_i8[:256]
        buf[316:316+512] = self.lut_i16[:512]

        assert len(buf) == LAYER_CONFIG_SIZE
        return bytes(buf)


def pack_model(layers: list, weight_data: bytes, output_path: str):
    """
    Pack a complete model to binary file.

    Args:
        layers: list of LayerConfig objects
        weight_data: concatenated weights+bias for all layers
        output_path: output .bin file path
    """
    num_layers = len(layers)
    weight_size = len(weight_data)

    # Header
    header = struct.pack('<IIII', MODEL_MAGIC, num_layers, 0, weight_size)

    # Layer configs
    configs_bin = b''.join(layer.pack() for layer in layers)

    # Write
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(configs_bin)
        f.write(weight_data)

    total = len(header) + len(configs_bin) + len(weight_data)
    print(f"Model packed: {output_path} ({total} bytes)")
    print(f"  Layers: {num_layers}")
    print(f"  Weights: {weight_size} bytes")


# ─── Reference implementations for verification ───

def ref_conv2d(input_nhwc, weights, cfg):
    """Reference Conv2D in Python (bit-exact INT32 accumulator)."""
    oh, ow, oc = cfg.out_h, cfg.out_w, cfg.out_c
    kh, kw = cfg.kernel_h, cfg.kernel_w
    sh, sw = cfg.stride_h, cfg.stride_w
    dh, dw = cfg.dilation_h, cfg.dilation_w
    pt, pl = cfg.pad_top, cfg.pad_left
    in_h, in_w, in_c = cfg.in_h, cfg.in_w, cfg.in_c

    acc = np.zeros((oh, ow, oc), dtype=np.int32)
    for o_h in range(oh):
        for o_w in range(ow):
            for o_c in range(oc):
                s = np.int32(0)
                for fh in range(kh):
                    ih = o_h * sh - pt + fh * dh
                    if ih < 0 or ih >= in_h:
                        continue
                    for fw in range(kw):
                        iw = o_w * sw - pl + fw * dw
                        if iw < 0 or iw >= in_w:
                            continue
                        for ic in range(in_c):
                            s += np.int32(input_nhwc[ih, iw, ic]) * np.int32(
                                weights[o_c, fh, fw, ic])
                acc[o_h, o_w, o_c] = s
    return acc


def ref_postproc(acc, bias, cfg):
    """Reference post-processing in Python (bit-exact)."""
    result = acc.copy().astype(np.int64)

    if cfg.post_ctrl & POST_BIAS_EN:
        for c in range(result.shape[-1]):
            shifted_bias = np.int32(bias[c]) >> cfg.bias_shift
            result[..., c] += shifted_bias

    if cfg.post_ctrl & POST_SHIFT_EN:
        if cfg.round_en and cfg.shift_bits > 0:
            result += (1 << (cfg.shift_bits - 1))
        result >>= cfg.shift_bits

    if cfg.post_ctrl & POST_SCALE_EN:
        result = (result * np.int64(cfg.scale)) >> 15

    result += np.int64(cfg.out_zp)

    if cfg.post_ctrl & POST_CLAMP_EN:
        result = np.clip(result, cfg.clamp_min, cfg.clamp_max)

    return result.astype(np.int8)


def ref_dwconv(input_nhwc, weights, cfg):
    """Reference Depthwise Conv in Python (bit-exact INT32 accumulator).
    weights shape: [channels][kh][kw]
    """
    oh, ow = cfg.out_h, cfg.out_w
    ch = cfg.in_c
    kh, kw = cfg.kernel_h, cfg.kernel_w
    sh, sw = cfg.stride_h, cfg.stride_w
    dh, dw = cfg.dilation_h, cfg.dilation_w
    pt, pl = cfg.pad_top, cfg.pad_left
    in_h, in_w = cfg.in_h, cfg.in_w

    acc = np.zeros((oh, ow, ch), dtype=np.int32)
    for o_h in range(oh):
        for o_w in range(ow):
            for c in range(ch):
                s = np.int32(0)
                for fh in range(kh):
                    ih = o_h * sh - pt + fh * dh
                    if ih < 0 or ih >= in_h:
                        continue
                    for fw in range(kw):
                        iw = o_w * sw - pl + fw * dw
                        if iw < 0 or iw >= in_w:
                            continue
                        s += np.int32(input_nhwc[ih, iw, c]) * np.int32(weights[c, fh, fw])
                acc[o_h, o_w, c] = s
    return acc


def ref_pooling(input_nhwc, cfg):
    """Reference Pooling in Python (bit-exact)."""
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


# ─── Test model builders ───

def build_test_model(output_dir: str):
    """
    Build a simple 2-layer test model:
      Layer 0: Conv2D 3×3, 1→4 channels, 8×8 input, pad=1, stride=1 → 8×8×4
      Layer 1: Conv2D 1×1, 4→2 channels, 8×8 input → 8×8×2 + ReLU

    Returns input data and expected output for verification.
    """
    import os

    np.random.seed(42)

    # ─── Layer 0: Conv2D 3×3, in_c=1, out_c=4, pad=1 ───
    layer0 = LayerConfig()
    layer0.op_type = OP_CONV2D
    layer0.in_h, layer0.in_w, layer0.in_c = 8, 8, 1
    layer0.out_h, layer0.out_w, layer0.out_c = 8, 8, 4
    layer0.kernel_h, layer0.kernel_w = 3, 3
    layer0.dilation_h, layer0.dilation_w = 1, 1
    layer0.stride_h, layer0.stride_w = 1, 1
    layer0.pad_top, layer0.pad_bottom = 1, 1
    layer0.pad_left, layer0.pad_right = 1, 1
    # Post-proc: bias + shift + clamp (no scale, simple requant)
    layer0.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    layer0.shift_bits = 2
    layer0.round_en = 1
    layer0.clamp_min = -128
    layer0.clamp_max = 127

    # ─── Layer 1: Conv2D 1×1, in_c=4, out_c=2 + ReLU ───
    layer1 = LayerConfig()
    layer1.op_type = OP_CONV2D
    layer1.in_h, layer1.in_w, layer1.in_c = 8, 8, 4
    layer1.out_h, layer1.out_w, layer1.out_c = 8, 8, 2
    layer1.kernel_h, layer1.kernel_w = 1, 1
    layer1.dilation_h, layer1.dilation_w = 1, 1
    layer1.stride_h, layer1.stride_w = 1, 1
    # Post-proc: bias + shift + ReLU (clamp min=0)
    layer1.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    layer1.shift_bits = 3
    layer1.round_en = 1
    layer1.clamp_min = 0  # ReLU
    layer1.clamp_max = 127

    # ─── Generate random weights ───
    # Layer 0: [out_c=4][kh=3][kw=3][in_c=1] + bias[4]
    w0 = np.random.randint(-10, 10, (4, 3, 3, 1), dtype=np.int8)
    b0 = np.random.randint(-20, 20, (4,), dtype=np.int32)

    # Layer 1: [out_c=2][kh=1][kw=1][in_c=4] + bias[2]
    w1 = np.random.randint(-10, 10, (2, 1, 1, 4), dtype=np.int8)
    b1 = np.random.randint(-20, 20, (2,), dtype=np.int32)

    # ─── Generate random input (NCHW for file, will be converted) ───
    input_nchw = np.random.randint(-50, 50, (1, 8, 8), dtype=np.int8)

    # ─── Pack weights ───
    weight_data = b''
    weight_data += w0.tobytes()
    weight_data += b0.tobytes()
    weight_data += w1.tobytes()
    weight_data += b1.tobytes()

    # ─── Pack model ───
    model_path = os.path.join(output_dir, 'test_model.bin')
    pack_model([layer0, layer1], weight_data, model_path)

    # ─── Save input ───
    input_path = os.path.join(output_dir, 'test_input.bin')
    input_nchw.tofile(input_path)
    print(f"Input saved: {input_path} ({input_nchw.nbytes} bytes)")

    # ─── Compute reference output ───
    # Convert input to NHWC
    input_nhwc = input_nchw.transpose(1, 2, 0)  # [H][W][C]

    # Layer 0
    acc0 = ref_conv2d(input_nhwc, w0, layer0)
    out0 = ref_postproc(acc0, b0, layer0)

    # Layer 1
    acc1 = ref_conv2d(out0, w1, layer1)
    out1 = ref_postproc(acc1, b1, layer1)

    # Convert output to NCHW for comparison
    output_nchw = out1.transpose(2, 0, 1)  # [C][H][W]

    ref_path = os.path.join(output_dir, 'test_reference.bin')
    output_nchw.astype(np.int8).tofile(ref_path)
    print(f"Reference output saved: {ref_path} ({output_nchw.nbytes} bytes)")

    return model_path, input_path, ref_path


def build_test_dwconv(output_dir: str):
    """
    Test model with DWConv:
      Layer 0: Conv2D 3×3, 1→8 channels, 8×8, pad=1 → 8×8×8
      Layer 1: DWConv 3×3, 8 channels, stride=2, pad=1 → 4×4×8
      Layer 2: Conv2D 1×1, 8→4 channels + ReLU → 4×4×4
    """
    import os
    np.random.seed(123)

    # Layer 0: Conv2D 3×3
    layer0 = LayerConfig()
    layer0.op_type = OP_CONV2D
    layer0.in_h, layer0.in_w, layer0.in_c = 8, 8, 1
    layer0.out_h, layer0.out_w, layer0.out_c = 8, 8, 8
    layer0.kernel_h, layer0.kernel_w = 3, 3
    layer0.dilation_h, layer0.dilation_w = 1, 1
    layer0.stride_h, layer0.stride_w = 1, 1
    layer0.pad_top = layer0.pad_bottom = layer0.pad_left = layer0.pad_right = 1
    layer0.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    layer0.shift_bits = 2
    layer0.round_en = 1
    layer0.clamp_min, layer0.clamp_max = -128, 127

    # Layer 1: DWConv 3×3, stride=2
    layer1 = LayerConfig()
    layer1.op_type = OP_DW_CONV
    layer1.in_h, layer1.in_w, layer1.in_c = 8, 8, 8
    layer1.out_h, layer1.out_w, layer1.out_c = 4, 4, 8
    layer1.kernel_h, layer1.kernel_w = 3, 3
    layer1.dilation_h, layer1.dilation_w = 1, 1
    layer1.stride_h, layer1.stride_w = 2, 2
    layer1.pad_top = layer1.pad_bottom = layer1.pad_left = layer1.pad_right = 1
    layer1.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    layer1.shift_bits = 3
    layer1.round_en = 1
    layer1.clamp_min, layer1.clamp_max = -128, 127

    # Layer 2: Conv2D 1×1 + ReLU
    layer2 = LayerConfig()
    layer2.op_type = OP_CONV2D
    layer2.in_h, layer2.in_w, layer2.in_c = 4, 4, 8
    layer2.out_h, layer2.out_w, layer2.out_c = 4, 4, 4
    layer2.kernel_h, layer2.kernel_w = 1, 1
    layer2.dilation_h, layer2.dilation_w = 1, 1
    layer2.stride_h, layer2.stride_w = 1, 1
    layer2.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    layer2.shift_bits = 3
    layer2.round_en = 1
    layer2.clamp_min, layer2.clamp_max = 0, 127  # ReLU

    # Weights
    w0 = np.random.randint(-8, 8, (8, 3, 3, 1), dtype=np.int8)
    b0 = np.random.randint(-10, 10, (8,), dtype=np.int32)
    w1 = np.random.randint(-8, 8, (8, 3, 3), dtype=np.int8)  # DW: [ch][kh][kw]
    b1 = np.random.randint(-10, 10, (8,), dtype=np.int32)
    w2 = np.random.randint(-8, 8, (4, 1, 1, 8), dtype=np.int8)
    b2 = np.random.randint(-10, 10, (4,), dtype=np.int32)

    weight_data = w0.tobytes() + b0.tobytes() + w1.tobytes() + b1.tobytes() + w2.tobytes() + b2.tobytes()

    # Input
    input_nchw = np.random.randint(-30, 30, (1, 8, 8), dtype=np.int8)

    # Pack
    model_path = os.path.join(output_dir, 'test_dwconv_model.bin')
    pack_model([layer0, layer1, layer2], weight_data, model_path)
    input_path = os.path.join(output_dir, 'test_dwconv_input.bin')
    input_nchw.tofile(input_path)

    # Reference
    inp = input_nchw.transpose(1, 2, 0)
    acc0 = ref_conv2d(inp, w0, layer0)
    out0 = ref_postproc(acc0, b0, layer0)
    acc1 = ref_dwconv(out0, w1, layer1)
    out1 = ref_postproc(acc1, b1, layer1)
    acc2 = ref_conv2d(out1, w2, layer2)
    out2 = ref_postproc(acc2, b2, layer2)

    output_nchw = out2.transpose(2, 0, 1)
    ref_path = os.path.join(output_dir, 'test_dwconv_reference.bin')
    output_nchw.astype(np.int8).tofile(ref_path)

    print(f"DWConv test: input={input_nchw.shape}, output={output_nchw.shape}")
    return model_path, input_path, ref_path


def build_test_pooling(output_dir: str):
    """
    Test model with Pooling:
      Layer 0: Conv2D 3×3, 1→4 channels, 8×8, pad=1 → 8×8×4
      Layer 1: MaxPool 2×2, stride=2 → 4×4×4
      Layer 2: AvgPool global → 1×1×4
      Layer 3: FC 4→2 + ReLU → 1×1×2
    """
    import os
    np.random.seed(456)

    # Layer 0: Conv2D
    layer0 = LayerConfig()
    layer0.op_type = OP_CONV2D
    layer0.in_h, layer0.in_w, layer0.in_c = 8, 8, 1
    layer0.out_h, layer0.out_w, layer0.out_c = 8, 8, 4
    layer0.kernel_h, layer0.kernel_w = 3, 3
    layer0.dilation_h, layer0.dilation_w = 1, 1
    layer0.stride_h, layer0.stride_w = 1, 1
    layer0.pad_top = layer0.pad_bottom = layer0.pad_left = layer0.pad_right = 1
    layer0.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    layer0.shift_bits = 2
    layer0.round_en = 1
    layer0.clamp_min, layer0.clamp_max = -128, 127

    # Layer 1: MaxPool 2×2
    layer1 = LayerConfig()
    layer1.op_type = OP_POOLING
    layer1.in_h, layer1.in_w, layer1.in_c = 8, 8, 4
    layer1.out_h, layer1.out_w, layer1.out_c = 4, 4, 4
    layer1.pool_mode = 0  # Max
    layer1.pool_h, layer1.pool_w = 2, 2
    layer1.pool_stride_h, layer1.pool_stride_w = 2, 2
    # Pooling: no post-proc needed (output is already INT8 range for max)
    layer1.post_ctrl = POST_CLAMP_EN
    layer1.clamp_min, layer1.clamp_max = -128, 127

    # Layer 2: Global AvgPool
    layer2 = LayerConfig()
    layer2.op_type = OP_POOLING
    layer2.in_h, layer2.in_w, layer2.in_c = 4, 4, 4
    layer2.out_h, layer2.out_w, layer2.out_c = 1, 1, 4
    layer2.pool_mode = 1  # Avg
    layer2.global_pool = 1
    layer2.post_ctrl = POST_CLAMP_EN
    layer2.clamp_min, layer2.clamp_max = -128, 127

    # Layer 3: FC 4→2 + ReLU
    layer3 = LayerConfig()
    layer3.op_type = OP_FC
    layer3.in_h, layer3.in_w, layer3.in_c = 1, 1, 4
    layer3.out_h, layer3.out_w, layer3.out_c = 1, 1, 2
    layer3.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    layer3.shift_bits = 2
    layer3.round_en = 1
    layer3.clamp_min, layer3.clamp_max = 0, 127  # ReLU

    # Weights
    w0 = np.random.randint(-8, 8, (4, 3, 3, 1), dtype=np.int8)
    b0 = np.random.randint(-10, 10, (4,), dtype=np.int32)
    # Pooling layers have no weights
    w3 = np.random.randint(-8, 8, (2, 4), dtype=np.int8)  # FC: [out_c][in_c]
    b3 = np.random.randint(-10, 10, (2,), dtype=np.int32)

    weight_data = w0.tobytes() + b0.tobytes() + w3.tobytes() + b3.tobytes()

    # Input
    input_nchw = np.random.randint(-30, 30, (1, 8, 8), dtype=np.int8)

    # Pack
    model_path = os.path.join(output_dir, 'test_pooling_model.bin')
    pack_model([layer0, layer1, layer2, layer3], weight_data, model_path)
    input_path = os.path.join(output_dir, 'test_pooling_input.bin')
    input_nchw.tofile(input_path)

    # Reference
    inp = input_nchw.transpose(1, 2, 0)
    acc0 = ref_conv2d(inp, w0, layer0)
    out0 = ref_postproc(acc0, b0, layer0)

    # MaxPool (output is directly INT8 range, postproc just clamps)
    acc1 = ref_pooling(out0, layer1)
    out1 = ref_postproc(acc1, None, layer1)

    # Global AvgPool
    acc2 = ref_pooling(out1, layer2)
    out2 = ref_postproc(acc2, None, layer2)

    # FC
    # FC weight layout for C sim: [out_c][in_c]
    w3_4d = w3.reshape(2, 1, 1, 4)  # for ref_conv2d compatibility
    layer3_conv = LayerConfig()
    layer3_conv.__dict__.update(layer3.__dict__)
    layer3_conv.kernel_h, layer3_conv.kernel_w = 1, 1
    acc3 = ref_conv2d(out2, w3_4d, layer3_conv)
    out3 = ref_postproc(acc3, b3, layer3)

    output_nchw = out3.transpose(2, 0, 1)
    ref_path = os.path.join(output_dir, 'test_pooling_reference.bin')
    output_nchw.astype(np.int8).tofile(ref_path)

    print(f"Pooling test: input={input_nchw.shape}, output={output_nchw.shape}")
    return model_path, input_path, ref_path


def build_mobilenetv2_block(output_dir: str):
    """
    Test a MobileNetV2 inverted residual block (bottleneck):
      Layer 0: Conv2D 1×1, expand 4→24 (expansion ratio 6) + ReLU6
      Layer 1: DWConv 3×3, 24ch, stride=1, pad=1 + ReLU6
      Layer 2: Conv2D 1×1, 24→4 (project, no activation)

    This is the fundamental building block of MobileNetV2.
    """
    import os
    np.random.seed(789)

    in_c, expand_c, out_c = 4, 24, 4
    h, w = 8, 8

    # Layer 0: Expand 1×1
    layer0 = LayerConfig()
    layer0.op_type = OP_CONV2D
    layer0.in_h, layer0.in_w, layer0.in_c = h, w, in_c
    layer0.out_h, layer0.out_w, layer0.out_c = h, w, expand_c
    layer0.kernel_h, layer0.kernel_w = 1, 1
    layer0.dilation_h, layer0.dilation_w = 1, 1
    layer0.stride_h, layer0.stride_w = 1, 1
    layer0.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    layer0.shift_bits = 4
    layer0.round_en = 1
    layer0.clamp_min, layer0.clamp_max = 0, 6  # ReLU6 (quantized: 6 in INT8 domain)

    # Layer 1: DWConv 3×3
    layer1 = LayerConfig()
    layer1.op_type = OP_DW_CONV
    layer1.in_h, layer1.in_w, layer1.in_c = h, w, expand_c
    layer1.out_h, layer1.out_w, layer1.out_c = h, w, expand_c
    layer1.kernel_h, layer1.kernel_w = 3, 3
    layer1.dilation_h, layer1.dilation_w = 1, 1
    layer1.stride_h, layer1.stride_w = 1, 1
    layer1.pad_top = layer1.pad_bottom = layer1.pad_left = layer1.pad_right = 1
    layer1.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    layer1.shift_bits = 4
    layer1.round_en = 1
    layer1.clamp_min, layer1.clamp_max = 0, 6  # ReLU6

    # Layer 2: Project 1×1 (linear, no activation)
    layer2 = LayerConfig()
    layer2.op_type = OP_CONV2D
    layer2.in_h, layer2.in_w, layer2.in_c = h, w, expand_c
    layer2.out_h, layer2.out_w, layer2.out_c = h, w, out_c
    layer2.kernel_h, layer2.kernel_w = 1, 1
    layer2.dilation_h, layer2.dilation_w = 1, 1
    layer2.stride_h, layer2.stride_w = 1, 1
    layer2.post_ctrl = POST_BIAS_EN | POST_SHIFT_EN | POST_CLAMP_EN
    layer2.shift_bits = 4
    layer2.round_en = 1
    layer2.clamp_min, layer2.clamp_max = -128, 127  # Linear (no ReLU)

    # Weights
    w0 = np.random.randint(-5, 5, (expand_c, 1, 1, in_c), dtype=np.int8)
    b0 = np.random.randint(-8, 8, (expand_c,), dtype=np.int32)
    w1 = np.random.randint(-5, 5, (expand_c, 3, 3), dtype=np.int8)
    b1 = np.random.randint(-8, 8, (expand_c,), dtype=np.int32)
    w2 = np.random.randint(-5, 5, (out_c, 1, 1, expand_c), dtype=np.int8)
    b2 = np.random.randint(-8, 8, (out_c,), dtype=np.int32)

    weight_data = (w0.tobytes() + b0.tobytes() +
                   w1.tobytes() + b1.tobytes() +
                   w2.tobytes() + b2.tobytes())

    # Input
    input_nchw = np.random.randint(-20, 20, (in_c, h, w), dtype=np.int8)

    # Pack
    model_path = os.path.join(output_dir, 'test_mbv2_block_model.bin')
    pack_model([layer0, layer1, layer2], weight_data, model_path)
    input_path = os.path.join(output_dir, 'test_mbv2_block_input.bin')
    input_nchw.tofile(input_path)

    # Reference
    inp = input_nchw.transpose(1, 2, 0)  # [H][W][C]
    acc0 = ref_conv2d(inp, w0, layer0)
    out0 = ref_postproc(acc0, b0, layer0)
    acc1 = ref_dwconv(out0, w1, layer1)
    out1 = ref_postproc(acc1, b1, layer1)
    acc2 = ref_conv2d(out1, w2, layer2)
    out2 = ref_postproc(acc2, b2, layer2)

    output_nchw = out2.transpose(2, 0, 1)
    ref_path = os.path.join(output_dir, 'test_mbv2_block_reference.bin')
    output_nchw.astype(np.int8).tofile(ref_path)

    print(f"MBv2 block test: input={input_nchw.shape}, output={output_nchw.shape}")
    return model_path, input_path, ref_path


# ─── Run verification ───

def run_sim_and_verify(sim_path, model_path, input_path, ref_path, test_name):
    """Run simulator and compare output to reference."""
    import subprocess, os

    output_path = model_path.replace('_model.bin', '_output.bin')

    result = subprocess.run(
        [sim_path, model_path, input_path, output_path],
        capture_output=True, text=True
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"  Simulator error: {result.stderr}")
        return False

    sim_output = np.fromfile(output_path, dtype=np.int8)
    ref_output = np.fromfile(ref_path, dtype=np.int8)

    if np.array_equal(sim_output, ref_output):
        print(f"  ✓ PASS [{test_name}]: bit-exact match ({len(ref_output)} elements)")
        return True
    else:
        diff = np.where(sim_output != ref_output)[0]
        max_diff = np.max(np.abs(sim_output.astype(int) - ref_output.astype(int)))
        print(f"  ✗ FAIL [{test_name}]: {len(diff)}/{len(ref_output)} mismatches, max_diff={max_diff}")
        print(f"    First mismatch idx={diff[0]}: sim={sim_output[diff[0]]}, ref={ref_output[diff[0]]}")
        return False


if __name__ == '__main__':
    import sys
    import os

    sim_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'csim', 'npu_sim')
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'csim', 'testdata')

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        os.makedirs(test_dir, exist_ok=True)

        if not os.path.exists(sim_path):
            print(f"Simulator not found at {sim_path}")
            print("Build it first: cd csim && make")
            sys.exit(1)

        all_pass = True
        print("=" * 60)
        print("Open-NPU End-to-End Tests")
        print("=" * 60)

        # Test 1: Conv2D basic
        print("\n── Test 1: Conv2D (3×3 → 1×1) ──")
        m, i, r = build_test_model(test_dir)
        if not run_sim_and_verify(sim_path, m, i, r, "Conv2D"):
            all_pass = False

        # Test 2: DWConv
        print("\n── Test 2: Conv2D → DWConv → Conv2D ──")
        m, i, r = build_test_dwconv(test_dir)
        if not run_sim_and_verify(sim_path, m, i, r, "DWConv"):
            all_pass = False

        # Test 3: Pooling + FC
        print("\n── Test 3: Conv2D → MaxPool → GlobalAvgPool → FC ──")
        m, i, r = build_test_pooling(test_dir)
        if not run_sim_and_verify(sim_path, m, i, r, "Pooling+FC"):
            all_pass = False

        # Test 4: MobileNetV2 block
        print("\n── Test 4: MobileNetV2 Inverted Residual Block ──")
        m, i, r = build_mobilenetv2_block(test_dir)
        if not run_sim_and_verify(sim_path, m, i, r, "MBv2-Block"):
            all_pass = False

        print("\n" + "=" * 60)
        if all_pass:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
            sys.exit(1)

    elif len(sys.argv) > 1 and sys.argv[1] == 'mobilenet':
        # Full MobileNetV2 conversion (requires tflite-runtime)
        print("MobileNetV2 full model conversion requires tflite-runtime.")
        print("Install: pip install tflite-runtime")
        print("Usage: python3 model_packer.py mobilenet [model.tflite]")
        # TODO: implement full tflite → npu_sim conversion

    else:
        print("Usage:")
        print("  python3 model_packer.py test       Run all end-to-end tests")
        print("  python3 model_packer.py mobilenet  Convert MobileNetV2 (future)")
