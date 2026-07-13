#!/usr/bin/env python3
"""
bin2golden.py — Bridge: NPU1 binary → RTL golden format

Reads an NPU1 binary (from onnx_converter.py → model_packer.py) and produces
the .npy + metadata.json format consumed by RTL cocotb tests.

Usage:
  python3 bin2golden.py model.npu1.bin <output_dir> [--input input.bin]

Given an NPU1 binary with all layer weights/params embedded, generates:
  output_dir/
    metadata.json    — per-layer CSR config + DDR addresses
    layer_{i:02d}_wgt.npy    — uint32 packed weight words
    layer_{i:02d}_param.npy  — uint32 packed param words
    layer_{i:02d}_input.npy  — uint32 packed input (INT8/INT16 NHWC)
    layer_{i:02d}_output.npy — uint32 packed expected output

SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import json
import struct
import numpy as np

# Import from model_packer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_packer import (
    MODEL_MAGIC, FIXED_CONFIG_SIZE, LayerConfig, PerChannelParam, AddParam,
    OP_CONV2D, OP_DW_CONV, OP_FC, OP_POOLING, OP_ELTWISE_ADD,
    OP_RESIZE, OP_DECONV, OP_CONCAT,
    POST_BIAS_EN, POST_ZP_EN, POST_RELU_EN, POST_RELU6_EN, POST_INT16_OUT,
    PPU_MODE_CONV_REQ, PPU_MODE_ADD, PPU_MODE_PASSTHROUGH,
)

# ─── NPU1 binary reader ───

def read_npu1(path):
    """Read NPU1 binary, return (layers: list[LayerConfig], weight_blob: bytes)."""
    with open(path, 'rb') as f:
        data = f.read()

    off = 0
    magic, num_layers, weight_offset, weight_size = struct.unpack_from('<IIII', data, off)
    off += 16

    assert magic == MODEL_MAGIC, f"Bad magic: 0x{magic:08X}, expected 0x{MODEL_MAGIC:08X}"

    layers = []
    for li in range(num_layers):
        layer = LayerConfig()
        fixed = data[off:off + FIXED_CONFIG_SIZE]
        off += FIXED_CONFIG_SIZE

        def r8(): return fixed[off - FIXED_CONFIG_SIZE + idx] if False else 0
        # Use struct.unpack_from for safe reading
        pos = 0

        def read_u8(offset):
            return struct.unpack_from('B', fixed, offset)[0]

        def read_s8(offset):
            return struct.unpack_from('b', fixed, offset)[0]

        def read_u16(offset):
            return struct.unpack_from('<H', fixed, offset)[0]

        def read_s16(offset):
            return struct.unpack_from('<h', fixed, offset)[0]

        p = 0
        layer.op_type = read_u8(p); p += 1
        layer.data_type = read_u8(p); p += 1
        layer.in_h = read_u16(p); p += 2
        layer.in_w = read_u16(p); p += 2
        layer.in_c = read_u16(p); p += 2
        layer.out_h = read_u16(p); p += 2
        layer.out_w = read_u16(p); p += 2
        layer.out_c = read_u16(p); p += 2
        layer.kernel_h = read_u8(p); p += 1
        layer.kernel_w = read_u8(p); p += 1
        layer.dilation_h = read_u8(p); p += 1
        layer.dilation_w = read_u8(p); p += 1
        layer.stride_h = read_u8(p); p += 1
        layer.stride_w = read_u8(p); p += 1
        layer.pad_top = read_u8(p); p += 1
        layer.pad_bottom = read_u8(p); p += 1
        layer.pad_left = read_u8(p); p += 1
        layer.pad_right = read_u8(p); p += 1
        layer.pool_mode = read_u8(p); p += 1
        layer.pool_h = read_u8(p); p += 1
        layer.pool_w = read_u8(p); p += 1
        layer.pool_stride_h = read_u8(p); p += 1
        layer.pool_stride_w = read_u8(p); p += 1
        layer.global_pool = read_u8(p); p += 1
        layer.resize_mode = read_u8(p); p += 1
        layer.scale_h = read_u8(p); p += 1
        layer.scale_w = read_u8(p); p += 1
        layer.insert_h = read_u8(p); p += 1
        layer.insert_w = read_u8(p); p += 1
        layer.concat_offset = read_u16(p); p += 2
        layer.concat_total_c = read_u16(p); p += 2
        layer.tile_h = read_u16(p); p += 2
        layer.tile_w = read_u16(p); p += 2
        layer.tile_num_h = read_u16(p); p += 2
        layer.tile_num_w = read_u16(p); p += 2
        layer.post_ctrl = read_u8(p); p += 1
        layer.sched_ctrl = read_u8(p); p += 1
        layer.clamp_min = read_s16(p); p += 2
        layer.clamp_max = read_s16(p); p += 2
        layer.in_zp = read_s8(p); p += 1
        _pad1 = read_u8(p); p += 1  # pad
        param_ch_count = read_u16(p); p += 2
        has_lut = read_u8(p); p += 1
        has_add = read_u8(p); p += 1
        layer.residual_src = read_s8(p); p += 1
        layer.input_src = read_s16(p); p += 2

        assert p == FIXED_CONFIG_SIZE, f"Parse error: read {p}, expected {FIXED_CONFIG_SIZE}"

        # Read per-channel params
        for c in range(param_ch_count):
            M, S, _reserved, zp, bias_q = struct.unpack_from('<HBbhq', data, off)
            off += 14
            layer.ch_params.append(PerChannelParam(M=M, S=S, zp=zp, bias_q=bias_q))

        # Read add params
        if has_add:
            M_A, S_A, M_B, S_B = struct.unpack_from('<HBxHBx', data, off)
            off += 8
            layer.add_params = AddParam(M_A=M_A, S_A=S_A, M_B=M_B, S_B=S_B)

        # Read LUT (skip for now)
        if has_lut:
            off += 256 + 512  # i8 + i16 LUT

        layers.append(layer)

    # Weight blob
    weight_blob = data[weight_offset:weight_offset + weight_size]
    return layers, weight_blob


# ─── Shape helpers ───

def elem_bytes(layer):
    return 2 if layer.data_type == 1 else 1

def is_per_tile_store(layer):
    """Check if layer uses per-tile store (SCHED_CTRL bit[4])."""
    return bool(layer.sched_ctrl & 0x10)  # SCHED_CTRL_PER_TILE_STORE = bit[4]

def n_full_input_words(layer):
    """Full tensor input words (NOT tile)."""
    eb = elem_bytes(layer)
    total = layer.in_h * layer.in_w * layer.in_c * eb
    return (total + 3) // 4

def n_input_words(layer):
    """Per-layer input words. For tiled layers, use tile-sized input."""
    eb = elem_bytes(layer)
    if layer.tile_h > 0 and layer.tile_w > 0:
        # Tiled: compute input tile size (with halo for kernel overlap)
        # Pool layers use pool_h/w and pool_stride, not kernel_h/w
        if hasattr(layer, 'op_type') and layer.op_type == 3:  # OP_POOLING
            kh = layer.pool_h
            kw = layer.pool_w
            sh = layer.pool_stride_h
            sw = layer.pool_stride_w
        else:
            kh = layer.kernel_h
            kw = layer.kernel_w
            sh = layer.stride_h
            sw = layer.stride_w
        inp_h = (layer.tile_h - 1) * sh + kh
        inp_w = (layer.tile_w - 1) * sw + kw
        total = inp_h * inp_w * layer.in_c * eb
    else:
        total = layer.in_h * layer.in_w * layer.in_c * eb
    return (total + 3) // 4  # round up to word

def n_output_words(layer, per_tile_store=None):
    """Output words for DMA store / golden comparison.

    For per_tile_store=True (cascaded tiled inference): RTL stores every tile
    to NHWC DDR, so golden output is the FULL output tensor.
    For per_tile_store=False (last-tile-only): RTL stores only the LAST tile.
    For non-tiled layers: always full output tensor.
    If per_tile_store is None, auto-detect from layer.sched_ctrl bit[4].
    """
    if per_tile_store is None:
        per_tile_store = is_per_tile_store(layer)
    eb = elem_bytes(layer)
    if per_tile_store and layer.tile_h > 0 and layer.tile_w > 0:
        # Per-tile store: full output stored to NHWC DDR across all tiles
        total = layer.out_h * layer.out_w * layer.out_c * eb
    elif layer.tile_h > 0 and layer.tile_w > 0:
        # Last-tile-only store: last tile dimensions (clipped at border)
        last_h = min(layer.tile_h, layer.out_h - (layer.tile_num_h - 1) * layer.tile_h)
        last_w = min(layer.tile_w, layer.out_w - (layer.tile_num_w - 1) * layer.tile_w)
        total = last_h * last_w * layer.out_c * eb
    else:
        total = layer.out_h * layer.out_w * layer.out_c * eb
    return (total + 3) // 4

def dma_in_size(layer):
    if layer.tile_h > 0 and layer.tile_w > 0:
        num_tiles = layer.tile_num_h * layer.tile_num_w
        return n_input_words(layer) * 4 * num_tiles
    return n_full_input_words(layer) * 4

def dma_out_size(layer, per_tile_store=None):
    if per_tile_store is None:
        per_tile_store = is_per_tile_store(layer)
    return n_output_words(layer, per_tile_store) * 4

def n_wgt_words(layer):
    """Compute weight word count from layer config."""
    if layer.op_type == OP_CONV2D or layer.op_type == OP_FC:
        # weights: [out_c][kh][kw][in_c], each elem_bytes
        eb = elem_bytes(layer)
        total = layer.out_c * layer.kernel_h * layer.kernel_w * layer.in_c * eb
        return (total + 3) // 4
    elif layer.op_type == OP_DW_CONV:
        eb = elem_bytes(layer)
        total = layer.in_c * layer.kernel_h * layer.kernel_w * eb
        return (total + 3) // 4
    else:
        return 0  # no weights

def wgt_per_oc_words(layer):
    """Per-oc_group weight words (ARRAY_SIZE channels × k_depth).
    Returns 0 if all weights fit in WGT_DEPTH (no per-oc reload needed)."""
    if layer.op_type != OP_CONV2D and layer.op_type != OP_FC:
        return 0  # DW conv and others: no per-oc reload
    WGT_DEPTH = 24576  # SPAD_KB * 128, default 192KB
    total_wgt = n_wgt_words(layer)
    if total_wgt <= WGT_DEPTH:
        return 0  # all fit, no reload needed
    # Per-oc: ARRAY_SIZE × k_depth × elem_bytes / 4
    eb = elem_bytes(layer)
    per_oc_bytes = 16 * layer.kernel_h * layer.kernel_w * layer.in_c * eb
    return (per_oc_bytes + 3) // 4

def n_param_words(layer):
    """Per-channel params: 14 bytes each, packed into uint32 words."""
    n = len(layer.ch_params)
    if n == 0:
        return 0
    total = n * 14
    return (total + 3) // 4  # round up to word


# ─── Weight/param unpacking ───

def extract_layer_weights(layers, weight_blob, layer_idx):
    """Extract weight words for a specific layer from the weight blob."""
    if layer_idx >= len(layers):
        return np.array([], dtype=np.uint32)

    # Compute offset within weight_blob for this layer's weights
    offset = 0
    for i in range(layer_idx):
        offset += n_wgt_words(layers[i]) * 4  # byte offset

    n_bytes = n_wgt_words(layers[layer_idx]) * 4
    chunk = weight_blob[offset:offset + n_bytes]

    # Pad to word boundary
    if len(chunk) < n_bytes:
        chunk = chunk + b'\x00' * (n_bytes - len(chunk))

    return np.frombuffer(chunk, dtype=np.uint32)


def pack_params_to_words(layer):
    """Pack per-channel params into uint32 words.
    
    DDR format (4 words per channel, matching gen_dma_e2e_golden.py):
      Word 0: M[14:0] | (S[5:0] << 16)
      Word 1: ZP[15:0] | (bias[15:0] << 16)
      Word 2: bias[47:16]
      Word 3: bias[63:48] in [15:0]
    """
    words = []
    for p in layer.ch_params:
        m = p.M & 0x7FFF
        s = p.S & 0x3F
        w0 = m | (s << 16)
        zp = p.zp & 0xFFFF
        bias = p.bias_q
        bias_u64 = (bias + (1 << 64)) & 0xFFFFFFFFFFFFFFFF if bias < 0 else bias & 0xFFFFFFFFFFFFFFFF
        w1 = zp | (((bias_u64 >> 0) & 0xFFFF) << 16)
        w2 = (bias_u64 >> 16) & 0xFFFFFFFF
        w3 = (bias_u64 >> 48) & 0xFFFF
        words.extend([w0, w1, w2, w3])
    return np.array(words, dtype=np.uint32)


def pack_input_to_words(input_nhwc, layer, pad_value=0):
    """Pack input NHWC tensor into uint32 words (INT8 or INT16 packing).
    
    For tiled layers, packs per-tile input regions (with halo) sequentially
    in row-major tile order, matching the controller's sequential tile loading.
    For non-tiled layers, packs the full input tensor.
    """
    eb = elem_bytes(layer)
    dtype_u = np.uint16 if eb == 2 else np.uint8

    if eb == 1:
        def pack_flat(flat):
            n_words = (len(flat) + 3) // 4
            padded = np.zeros(n_words * 4, dtype=np.uint8)
            padded[:len(flat)] = flat.astype(np.uint8) & 0xFF
            return padded.view('<u4')
    else:
        def pack_flat(flat):
            flat_u16 = flat.astype(np.uint16)
            n_words = (len(flat) + 1) // 2
            padded = np.zeros(n_words * 2, dtype=np.uint16)
            padded[:len(flat_u16)] = flat_u16
            return padded.view('<u4')

    # Non-tiled: pack full input
    if layer.tile_h == 0 or layer.tile_w == 0:
        return pack_flat(input_nhwc.flatten())

    # Tiled: extract per-tile input regions with halo, pack sequentially
    # Input: NHWC tensor of shape (in_h, in_w, in_c)
    H, W, C = input_nhwc.shape

    # Compute input tile size with halo
    # Pool layers use pool_h/w and pool_stride, not kernel_h/w
    if hasattr(layer, 'op_type') and layer.op_type == 3:  # OP_POOLING
        kh = layer.pool_h
        kw = layer.pool_w
        sh = layer.pool_stride_h
        sw = layer.pool_stride_w
    else:
        kh = layer.kernel_h
        kw = layer.kernel_w
        sh = layer.stride_h
        sw = layer.stride_w
    inp_h = (layer.tile_h - 1) * sh + kh
    inp_w = (layer.tile_w - 1) * sw + kw
    
    # Compute actual padded dimensions needed for ALL tiles
    # The rightmost/bottom tiles may need more padding than pad_r/pad_b
    max_row_end = (layer.tile_num_h - 1) * sh * layer.tile_h + inp_h
    max_col_end = (layer.tile_num_w - 1) * sw * layer.tile_w + inp_w
    pad_t, pad_l = layer.pad_top, layer.pad_left
    # Oversized padding to accommodate all tile windows
    pad_b_extra = max(0, max_row_end - (H + pad_t))
    pad_r_extra = max(0, max_col_end - (W + pad_l))
    
    padded = np.full((H + pad_t + pad_b_extra, W + pad_l + pad_r_extra, C),
                     pad_value - layer.in_zp, dtype=input_nhwc.dtype)
    padded[pad_t:pad_t+H, pad_l:pad_l+W, :] = input_nhwc
    
    # Extract and pack each tile
    all_words = []
    for ty in range(layer.tile_num_h):
        for tx in range(layer.tile_num_w):
            # Input tile region in padded image
            row_start = ty * layer.tile_h * sh
            row_end = row_start + inp_h
            col_start = tx * layer.tile_w * sw
            col_end = col_start + inp_w
            tile_data = padded[row_start:row_end, col_start:col_end, :]
            all_words.append(pack_flat(tile_data.flatten()))
    
    return np.concatenate(all_words) if all_words else pack_flat(input_nhwc.flatten())


def pack_output_to_words(output_nhwc, layer, per_tile_store=None):
    """Pack output NHWC tensor into uint32 words.

    For per_tile_store=True: RTL stores each tile to NHWC DDR layout.
      Golden output is the FULL NHWC tensor (packed as contiguous NHWC).
    For per_tile_store=False (default):
      - Non-tiled: full NHWC output.
      - Tiled: LAST tile only (RTL stores only last tile to DDR).
    If per_tile_store is None, auto-detect from layer.sched_ctrl bit[4].
    """
    if per_tile_store is None:
        per_tile_store = is_per_tile_store(layer)

    eb = elem_bytes(layer)
    if eb == 1:
        def pack_flat(flat):
            n_words = (len(flat) + 3) // 4
            padded = np.zeros(n_words * 4, dtype=np.uint8)
            padded[:len(flat)] = flat.astype(np.uint8) & 0xFF
            return padded.view('<u4')
    else:
        def pack_flat(flat):
            flat_u16 = flat.astype(np.uint16)
            n_words = (len(flat_u16) + 1) // 2
            padded = np.zeros(n_words * 2, dtype=np.uint16)
            padded[:len(flat_u16)] = flat_u16
            return padded.view('<u4')

    # Per-tile store: golden is full NHWC output (RTL writes tiles to NHWC DDR)
    if per_tile_store:
        return pack_flat(output_nhwc.flatten())

    # Non-tiled: full NHWC output
    if layer.tile_h == 0 or layer.tile_w == 0:
        return pack_flat(output_nhwc.flatten())

    # Tiled, last-tile-only store: pack only the last tile's output region
    last_h = min(layer.tile_h, layer.out_h - (layer.tile_num_h - 1) * layer.tile_h)
    last_w = min(layer.tile_w, layer.out_w - (layer.tile_num_w - 1) * layer.tile_w)
    row_start = (layer.tile_num_h - 1) * layer.tile_h
    col_start = (layer.tile_num_w - 1) * layer.tile_w
    tile_data = output_nhwc[row_start:row_start + last_h,
                           col_start:col_start + last_w, :]
    return pack_flat(tile_data.flatten())


# ─── Golden generation ───

def generate_golden(layers, weight_blob, input_nhwc, output_dir,
                    base_ddr_addr=0x30000000, layer_offset=0x00010000):
    """Generate RTL golden data from model layers + input.

    Args:
        layers: list of LayerConfig
        weight_blob: raw weight bytes from NPU1 binary
        input_nhwc: input tensor in NHWC format (numpy array)
        output_dir: output directory for golden files
        base_ddr_addr: base DDR address for first layer's input
        layer_offset: DDR address increment between layers
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    ddr_addr = base_ddr_addr

    for idx, layer in enumerate(layers):
        pts = is_per_tile_store(layer)  # per-tile store flag
        n_in = n_input_words(layer)
        n_out = n_output_words(layer, pts)
        n_wgt = n_wgt_words(layer)
        n_prm = n_param_words(layer)
        # Add/Concat layers use global params (2 words) not counted in ch_params
        if layer.op_type == OP_ELTWISE_ADD and n_prm == 0:
            n_prm = 2

        # DDR input region: for tiled layers, the full tiled input is
        # tile_in_size * num_tiles (each tile's input packed sequentially).
        # For non-tiled, it's n_input_words * 4.
        if layer.tile_h > 0 and layer.tile_w > 0:
            num_tiles = layer.tile_num_h * layer.tile_num_w
            in_size = n_in * 4 * num_tiles
        else:
            in_size = n_in * 4
        wgt_size = n_wgt * 4
        param_size = n_prm * 4
        out_size = dma_out_size(layer, pts)
        align = 4096
        in_size_align = ((in_size + align - 1) // align) * align
        wgt_size_align = ((wgt_size + align - 1) // align) * align
        param_size_align = ((param_size + align - 1) // align) * align
        out_size_align = ((out_size + align - 1) // align) * align

        # DDR addresses — dynamically sized to avoid overlap
        # Note: wgt_addr must differ from param_addr even when wgt_size=0,
        # because dma_bank_sel compares ext_addr to wgt_addr/param_addr.
        # If they're equal, param loads get misrouted to weight SRAM.
        ddr_in_addr = ddr_addr
        ddr_wgt_addr = ddr_in_addr + in_size_align
        ddr_param_addr = ddr_wgt_addr + (wgt_size_align if wgt_size_align > 0 else align)
        ddr_out_addr = ddr_param_addr + param_size_align

        # Advance DDR address for next layer (after this layer's output)
        ddr_addr = ddr_out_addr + out_size_align

        # Pooling config
        pool_mode_map = {0: 0, 1: 1}  # max=0, avg=1
        pool_cfg = 0
        if layer.op_type == OP_POOLING:
            pool_cfg = (pool_mode_map.get(layer.pool_mode, 0)) \
                     | (layer.pool_h << 4) \
                     | (layer.pool_w << 8) \
                     | (layer.pool_stride_h << 12) \
                     | (layer.pool_stride_w << 16) \
                     | (layer.global_pool << 20)

        # Resize config
        resize_cfg = 0
        if layer.op_type == OP_RESIZE:
            scale_h_q44 = int(round((layer.out_h / layer.in_h) * 16.0)) & 0xFF
            scale_w_q44 = int(round((layer.out_w / layer.in_w) * 16.0)) & 0xFF
            resize_cfg = layer.resize_mode \
                       | (scale_h_q44 << 8) \
                       | (scale_w_q44 << 16)

        # Deconv config
        deconv_cfg = 0
        if layer.op_type == OP_DECONV:
            deconv_cfg = layer.insert_h | (layer.insert_w << 8)

        # Concat config
        concat_cfg = 0
        if layer.op_type == OP_CONCAT:
            concat_cfg = layer.concat_offset | (layer.concat_total_c << 16)

        # k_depth for Conv2D: kernel_h * kernel_w * in_c (per-OC-group compute passes)
        if layer.op_type in (OP_CONV2D, OP_FC):
            k_depth = layer.kernel_h * layer.kernel_w * layer.in_c
        elif layer.op_type == OP_DW_CONV:
            k_depth = layer.kernel_h * layer.kernel_w
        else:
            k_depth = 1

        # Build meta entry (matching RTL test format)
        meta = {
            'op_type': layer.op_type,
            'data_type': layer.data_type,
            'in_h': layer.in_h,
            'in_w': layer.in_w,
            'in_c': layer.in_c,
            'out_h': layer.out_h,
            'out_w': layer.out_w,
            'out_c': layer.out_c,
            'kernel_h': layer.kernel_h,
            'kernel_w': layer.kernel_w,
            'stride_h': layer.stride_h,
            'stride_w': layer.stride_w,
            'pad_top': layer.pad_top,
            'pad_left': layer.pad_left,
            'k_depth': k_depth,
            'post_ctrl': layer.post_ctrl,
            'relu6': (layer.post_ctrl & POST_RELU6_EN) != 0,
            'pool_cfg': pool_cfg,
            'resize_cfg': resize_cfg,
            'deconv_cfg': deconv_cfg,
            'concat_cfg': concat_cfg,
            'dma_in_size': dma_in_size(layer),
            'dma_wgt_size': n_wgt_words(layer) * 4,
            'wgt_per_oc_words': wgt_per_oc_words(layer),
            'dma_out_size': dma_out_size(layer, pts),
            'dma_param_count': len(layer.ch_params),
            'tile_in_size': n_input_words(layer) * 4 if layer.tile_h > 0 else 0,
            # Per-tile store (2D DMA for NHWC DDR layout)
            'tile_out_size': (layer.tile_h * layer.tile_w * layer.out_c *
                              elem_bytes(layer)) if (pts and layer.tile_h > 0) else 0,
            'store_mode': 1 if pts else 0,  # bit[0]=PER_TILE_STORE_EN
            'row_cfg': (((layer.tile_h & 0xFFFF) << 16) | ((layer.tile_w * layer.out_c *
                        elem_bytes(layer) // 4) & 0xFFFF)) if pts else 0,
            'n_input_words': n_in,
            'n_output_words': n_out,
            'tile_h': layer.tile_h,
            'tile_w': layer.tile_w,
            'tile_num_h': layer.tile_num_h,
            'tile_num_w': layer.tile_num_w,
            'sched_ctrl': layer.sched_ctrl,
            'ddr_in_addr': ddr_in_addr,
            'ddr_out_addr': ddr_out_addr,
            'ddr_wgt_addr': ddr_wgt_addr,
            'ddr_param_addr': ddr_param_addr,
            'clamp_min': layer.clamp_min,
            'clamp_max': layer.clamp_max,
            'in_zp': layer.in_zp,
            # Non-sequential input routing
            'input_src': layer.input_src,
            'residual_src': layer.residual_src,
        }

        # Add params for Add layers
        if layer.add_params:
            meta['add_M_A'] = layer.add_params.M_A
            meta['add_S_A'] = layer.add_params.S_A
            meta['add_M_B'] = layer.add_params.M_B
            meta['add_S_B'] = layer.add_params.S_B

        metadata.append(meta)

        # Save .npy files
        wgt_words = extract_layer_weights(layers, weight_blob, idx)
        prm_words = pack_params_to_words(layer)
        # For Add/Concat layers: append global Add params (2 words: M_A|S_A, M_B|S_B)
        if layer.add_params and len(prm_words) == 0:
            ap = layer.add_params
            w_a = (ap.M_A & 0x7FFF) | ((ap.S_A & 0x3F) << 16)
            w_b = (ap.M_B & 0x7FFF) | ((ap.S_B & 0x3F) << 16)
            prm_words = np.array([w_a, w_b], dtype=np.uint32)
            meta['dma_param_count'] = 2

        # Input: for layer 0 use provided input; for later layers use previous output
        # (We'll handle this in the test, just save placeholder for now)
        inp_words = pack_input_to_words(input_nhwc, layer) if idx == 0 else np.array([], dtype=np.uint32)

        # Output: placeholder (will be filled from CSIM output)
        out_words = np.array([], dtype=np.uint32)

        np.save(os.path.join(output_dir, f'layer_{idx:02d}_wgt.npy'), wgt_words)
        np.save(os.path.join(output_dir, f'layer_{idx:02d}_param.npy'), prm_words)
        np.save(os.path.join(output_dir, f'layer_{idx:02d}_input.npy'), inp_words)
        np.save(os.path.join(output_dir, f'layer_{idx:02d}_output.npy'), out_words)

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Golden data generated: {output_dir}")
    print(f"  Layers: {len(layers)}")
    print(f"  Files: metadata.json + layer_XX_wgt/param/input/output.npy")
    return metadata


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NPU1 binary → RTL golden bridge')
    parser.add_argument('model', help='NPU1 binary file (.bin)')
    parser.add_argument('output_dir', help='Output directory for golden data')
    parser.add_argument('--input', help='Input tensor binary file (raw NCHW)')
    args = parser.parse_args()

    layers, weight_blob = read_npu1(args.model)
    print(f"Read NPU1 binary: {len(layers)} layers, {len(weight_blob)} weight bytes")

    # Load input
    if args.input:
        input_raw = np.fromfile(args.input, dtype=np.uint8)
        # Determine shape from first layer
        l0 = layers[0]
        eb = elem_bytes(l0)
        if eb == 1:
            input_t = input_raw.astype(np.int8)
        else:
            input_t = input_raw.view(np.int16)
        # Reshape to NCHW then transpose to NHWC
        expected = l0.in_c * l0.in_h * l0.in_w
        input_t = input_t[:expected].reshape(l0.in_c, l0.in_h, l0.in_w)
        input_nhwc = np.transpose(input_t, (1, 2, 0))  # CHW → HWC
    else:
        # Random input for testing
        l0 = layers[0]
        np.random.seed(42)
        eb = elem_bytes(l0)
        dtype = np.int16 if eb == 2 else np.int8
        input_nhwc = np.random.randint(-30, 30,
                                        (l0.in_h, l0.in_w, l0.in_c), dtype=dtype)

    generate_golden(layers, weight_blob, input_nhwc, args.output_dir)


if __name__ == '__main__':
    main()
