#!/usr/bin/env python3
"""
Open-NPU Cycle-Level Performance Estimator

Estimates per-layer and total inference cycles based on:
  - Hardware: 16x16 systolic array (256 MACs/cycle), 32KB+32KB+64KB SRAM
  - Tiling from tiling.py
  - Layer configs from onnx_converter or manual specification

Usage:
  python3 perf_model.py --resnet18-test [--bits 8|16] [--bw 4] [--no-double-buffer]
  python3 perf_model.py --model MODEL.onnx [--bits 8|16] [--bw 4]

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import math
import sys
import os
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiling import compute_tiling
from layer_fusion import detect_fusible_blocks, compute_fused_tiling, estimate_fusion_savings
from hw_config import HWConfig as HWProfile, DEFAULT_HW, add_hw_args, hw_config_from_args


# ─── Hardware Configuration ───

@dataclass
class HWConfig:
    array_size: int = 16          # Systolic array dimension (16x16)
    macs_per_cycle: int = 256     # 16*16 = 256 MACs/cycle
    act_bank_bytes: int = 32768   # 32KB per activation bank
    weight_buf_bytes: int = 65536 # 64KB weight buffer
    ext_bw_bytes: int = 4         # 32-bit bus @NPU clock = 4 bytes/cycle
    dw_parallel_ch: int = 16      # DW conv parallel channels
    elem_size: int = 1            # 1=INT8, 2=INT16
    double_buffer: bool = True    # Enable ping-pong overlap
    adaptive_db: bool = False     # Adaptive: per-layer best of single/double buffer

    @classmethod
    def from_hw(cls, hw: HWProfile, elem_size: int = 1,
                double_buffer: bool = True, adaptive_db: bool = False):
        return cls(
            array_size=hw.array_size,
            macs_per_cycle=hw.macs_per_cycle,
            act_bank_bytes=hw.act_bank_size,
            weight_buf_bytes=hw.weight_buf_size,
            ext_bw_bytes=hw.ext_bw_bytes,
            dw_parallel_ch=hw.dw_parallel_ch,
            elem_size=elem_size,
            double_buffer=double_buffer,
            adaptive_db=adaptive_db,
        )


# ─── Layer Descriptor ───

@dataclass
class LayerDesc:
    op_type: str        # 'conv', 'dw', 'fc', 'pool', 'add', 'resize', 'concat'
    in_h: int
    in_w: int
    in_c: int
    out_h: int
    out_w: int
    out_c: int
    kernel_h: int = 1
    kernel_w: int = 1
    stride_h: int = 1
    stride_w: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    pool_h: int = 0
    pool_w: int = 0
    pool_stride_h: int = 0
    pool_stride_w: int = 0
    resize_mode: str = 'nearest'


# ─── Per-Operator Cycle Estimation ───

def _ceil_div(a, b):
    return (a + b - 1) // b


def estimate_conv(layer, tiling, hw):
    """Conv2D cycle estimation with OC groups and spatial tiling."""
    kh, kw = layer.kernel_h, layer.kernel_w
    sh, sw = layer.stride_h, layer.stride_w
    dh, dw = layer.dilation_h, layer.dilation_w
    in_c, out_c = layer.in_c, layer.out_c
    out_h, out_w = layer.out_h, layer.out_w
    es = hw.elem_size

    total_macs = out_h * out_w * out_c * in_c * kh * kw

    # Tile parameters
    tile_h = tiling['tile_h'] if tiling['tile_num_h'] > 0 else out_h
    tile_w = tiling['tile_w'] if tiling['tile_num_w'] > 0 else out_w
    num_h = tiling['tile_num_h'] if tiling['tile_num_h'] > 0 else 1
    num_w = tiling['tile_num_w'] if tiling['tile_num_w'] > 0 else 1
    num_spatial = num_h * num_w

    # OC group size from tiling result (accounts for double-buffer constraint)
    tile_oc = tiling.get('tile_oc', out_c)
    if tile_oc < 1:
        tile_oc = 1
    oc_groups = _ceil_div(out_c, tile_oc)

    # Utilization: array maps in_c → columns, tile_oc → rows
    util_ic = min(in_c, hw.array_size) / hw.array_size
    util_oc = min(tile_oc, hw.array_size) / hw.array_size
    util = util_ic * util_oc
    if util < 0.01:
        util = 0.01

    # Compute cycles
    compute_cycles = int(math.ceil(total_macs / hw.macs_per_cycle / util))

    # DMA per spatial tile
    kh_eff = (kh - 1) * dh + 1
    kw_eff = (kw - 1) * dw + 1
    input_tile_h = tile_h * sh + kh_eff - sh
    input_tile_w = tile_w * sw + kw_eff - sw
    dma_load_input_per_tile = input_tile_h * input_tile_w * in_c * es
    dma_load_weight_per_group = kh * kw * in_c * tile_oc * es
    dma_store_per_tile = tile_h * tile_w * tile_oc * es

    # Total DMA bytes
    total_dma_load_bytes = num_spatial * (
        oc_groups * dma_load_weight_per_group + dma_load_input_per_tile * oc_groups
    )
    # Note: input could be reused across OC groups, but conservative estimate
    # reloads it each time (no input double-buffer across OC groups)
    # Better model: input loaded once per spatial tile, weight reloaded per OC group
    total_dma_load_bytes = num_spatial * (
        dma_load_input_per_tile + oc_groups * dma_load_weight_per_group
    )
    total_dma_store_bytes = num_spatial * oc_groups * dma_store_per_tile

    dma_load_cycles = int(math.ceil(total_dma_load_bytes / hw.ext_bw_bytes))
    dma_store_cycles = int(math.ceil(total_dma_store_bytes / hw.ext_bw_bytes))

    # Per-tile cycles for double-buffer model
    num_tiles = num_spatial * oc_groups
    per_tile_compute = int(math.ceil(compute_cycles / num_tiles)) if num_tiles > 0 else 0
    per_tile_load = int(math.ceil(dma_load_cycles / num_tiles)) if num_tiles > 0 else 0
    per_tile_store = int(math.ceil(dma_store_cycles / num_tiles)) if num_tiles > 0 else 0

    return _build_result(total_macs, compute_cycles, dma_load_cycles, dma_store_cycles,
                         num_tiles, per_tile_compute, per_tile_load, per_tile_store, hw)


def estimate_dwconv(layer, tiling, hw):
    """Depthwise conv: uses DW_PARALLEL_CH channels in parallel, not full array."""
    kh, kw = layer.kernel_h, layer.kernel_w
    sh, sw = layer.stride_h, layer.stride_w
    dh, dw = layer.dilation_h, layer.dilation_w
    in_c = layer.in_c
    out_h, out_w = layer.out_h, layer.out_w
    es = hw.elem_size

    total_macs = out_h * out_w * in_c * kh * kw

    # DW effective throughput: DW_PARALLEL_CH per cycle
    compute_cycles = out_h * out_w * _ceil_div(in_c, hw.dw_parallel_ch) * kh * kw

    # Tiling
    tile_h = tiling['tile_h'] if tiling['tile_num_h'] > 0 else out_h
    tile_w = tiling['tile_w'] if tiling['tile_num_w'] > 0 else out_w
    num_h = tiling['tile_num_h'] if tiling['tile_num_h'] > 0 else 1
    num_w = tiling['tile_num_w'] if tiling['tile_num_w'] > 0 else 1
    num_tiles = num_h * num_w

    kh_eff = (kh - 1) * dh + 1
    kw_eff = (kw - 1) * dw + 1
    input_tile_h = tile_h * sh + kh_eff - sh
    input_tile_w = tile_w * sw + kw_eff - sw

    dma_load_bytes = num_tiles * input_tile_h * input_tile_w * in_c * es + kh * kw * in_c * es
    dma_store_bytes = num_tiles * tile_h * tile_w * in_c * es

    dma_load_cycles = int(math.ceil(dma_load_bytes / hw.ext_bw_bytes))
    dma_store_cycles = int(math.ceil(dma_store_bytes / hw.ext_bw_bytes))

    per_tile_compute = int(math.ceil(compute_cycles / num_tiles)) if num_tiles > 0 else 0
    per_tile_load = int(math.ceil(dma_load_cycles / num_tiles)) if num_tiles > 0 else 0
    per_tile_store = int(math.ceil(dma_store_cycles / num_tiles)) if num_tiles > 0 else 0

    return _build_result(total_macs, compute_cycles, dma_load_cycles, dma_store_cycles,
                         num_tiles, per_tile_compute, per_tile_load, per_tile_store, hw)


def estimate_fc(layer, tiling, hw):
    """FC layer: same as Conv with kh=kw=1, spatial=1x1."""
    in_c, out_c = layer.in_c, layer.out_c
    es = hw.elem_size

    total_macs = out_c * in_c

    # Utilization
    util_ic = min(in_c, hw.array_size) / hw.array_size
    util_oc = min(out_c, hw.array_size) / hw.array_size
    util = util_ic * util_oc
    if util < 0.01:
        util = 0.01

    compute_cycles = int(math.ceil(total_macs / hw.macs_per_cycle / util))

    # DMA
    dma_load_bytes = in_c * es + out_c * in_c * es  # input + weights
    dma_store_bytes = out_c * es

    dma_load_cycles = int(math.ceil(dma_load_bytes / hw.ext_bw_bytes))
    dma_store_cycles = int(math.ceil(dma_store_bytes / hw.ext_bw_bytes))

    num_tiles = 1
    return _build_result(total_macs, compute_cycles, dma_load_cycles, dma_store_cycles,
                         num_tiles, compute_cycles, dma_load_cycles, dma_store_cycles, hw)


def estimate_pool(layer, tiling, hw):
    """Pooling: PPU-based, ~1 element/cycle throughput."""
    out_h, out_w, out_c = layer.out_h, layer.out_w, layer.out_c
    in_c = layer.in_c
    es = hw.elem_size
    ph = layer.pool_h if layer.pool_h > 0 else layer.kernel_h
    pw = layer.pool_w if layer.pool_w > 0 else layer.kernel_w
    ps_h = layer.pool_stride_h if layer.pool_stride_h > 0 else layer.stride_h
    ps_w = layer.pool_stride_w if layer.pool_stride_w > 0 else layer.stride_w

    total_macs = 0  # Pooling does comparisons, not MACs

    # PPU throughput: 1 element/cycle (conservative)
    compute_cycles = out_h * out_w * out_c

    # Tiling
    tile_h = tiling['tile_h'] if tiling['tile_num_h'] > 0 else out_h
    tile_w = tiling['tile_w'] if tiling['tile_num_w'] > 0 else out_w
    num_h = tiling['tile_num_h'] if tiling['tile_num_h'] > 0 else 1
    num_w = tiling['tile_num_w'] if tiling['tile_num_w'] > 0 else 1
    num_tiles = num_h * num_w

    input_tile_h = tile_h * ps_h + ph - ps_h
    input_tile_w = tile_w * ps_w + pw - ps_w

    dma_load_bytes = num_tiles * input_tile_h * input_tile_w * in_c * es
    dma_store_bytes = num_tiles * tile_h * tile_w * out_c * es

    dma_load_cycles = int(math.ceil(dma_load_bytes / hw.ext_bw_bytes))
    dma_store_cycles = int(math.ceil(dma_store_bytes / hw.ext_bw_bytes))

    per_tile_compute = int(math.ceil(compute_cycles / num_tiles)) if num_tiles > 0 else 0
    per_tile_load = int(math.ceil(dma_load_cycles / num_tiles)) if num_tiles > 0 else 0
    per_tile_store = int(math.ceil(dma_store_cycles / num_tiles)) if num_tiles > 0 else 0

    return _build_result(total_macs, compute_cycles, dma_load_cycles, dma_store_cycles,
                         num_tiles, per_tile_compute, per_tile_load, per_tile_store, hw)


def estimate_add(layer, tiling, hw):
    """Eltwise Add: PPU-based, 1 element/cycle. Residual loaded from DDR."""
    out_h, out_w, out_c = layer.out_h, layer.out_w, layer.out_c
    es = hw.elem_size

    total_macs = 0
    compute_cycles = out_h * out_w * out_c  # 1 elem/cycle

    # DMA: residual input (other branch already in SRAM from previous layer)
    dma_load_bytes = out_h * out_w * out_c * es  # residual branch
    dma_store_bytes = out_h * out_w * out_c * es

    dma_load_cycles = int(math.ceil(dma_load_bytes / hw.ext_bw_bytes))
    dma_store_cycles = int(math.ceil(dma_store_bytes / hw.ext_bw_bytes))

    # Tiling for Add
    tile_h = tiling['tile_h'] if tiling['tile_num_h'] > 0 else out_h
    tile_w = tiling['tile_w'] if tiling['tile_num_w'] > 0 else out_w
    num_h = tiling['tile_num_h'] if tiling['tile_num_h'] > 0 else 1
    num_w = tiling['tile_num_w'] if tiling['tile_num_w'] > 0 else 1
    num_tiles = num_h * num_w

    per_tile_compute = int(math.ceil(compute_cycles / num_tiles)) if num_tiles > 0 else 0
    per_tile_load = int(math.ceil(dma_load_cycles / num_tiles)) if num_tiles > 0 else 0
    per_tile_store = int(math.ceil(dma_store_cycles / num_tiles)) if num_tiles > 0 else 0

    return _build_result(total_macs, compute_cycles, dma_load_cycles, dma_store_cycles,
                         num_tiles, per_tile_compute, per_tile_load, per_tile_store, hw)


def estimate_resize(layer, tiling, hw):
    """Resize: ~1 elem/cycle for nearest, ~4 for bilinear."""
    out_h, out_w, out_c = layer.out_h, layer.out_w, layer.out_c
    in_h, in_w, in_c = layer.in_h, layer.in_w, layer.in_c
    es = hw.elem_size

    total_macs = 0
    factor = 1 if layer.resize_mode == 'nearest' else 4
    compute_cycles = out_h * out_w * out_c * factor

    dma_load_bytes = in_h * in_w * in_c * es
    dma_store_bytes = out_h * out_w * out_c * es

    dma_load_cycles = int(math.ceil(dma_load_bytes / hw.ext_bw_bytes))
    dma_store_cycles = int(math.ceil(dma_store_bytes / hw.ext_bw_bytes))

    num_tiles = 1
    return _build_result(total_macs, compute_cycles, dma_load_cycles, dma_store_cycles,
                         num_tiles, compute_cycles, dma_load_cycles, dma_store_cycles, hw)


def estimate_concat(layer, tiling, hw):
    """Concat: pure data movement, no compute."""
    out_h, out_w, out_c = layer.out_h, layer.out_w, layer.out_c
    in_h, in_w, in_c = layer.in_h, layer.in_w, layer.in_c
    es = hw.elem_size

    total_macs = 0
    compute_cycles = 0

    dma_load_bytes = in_h * in_w * in_c * es
    dma_store_bytes = out_h * out_w * out_c * es

    dma_load_cycles = int(math.ceil(dma_load_bytes / hw.ext_bw_bytes))
    dma_store_cycles = int(math.ceil(dma_store_bytes / hw.ext_bw_bytes))

    num_tiles = 1
    return _build_result(total_macs, compute_cycles, dma_load_cycles, dma_store_cycles,
                         num_tiles, compute_cycles, dma_load_cycles, dma_store_cycles, hw)


def _build_result(total_macs, compute_cycles, dma_load_cycles, dma_store_cycles,
                  num_tiles, per_tile_compute, per_tile_load, per_tile_store, hw):
    """Build result dict with double-buffer and no-double-buffer estimates."""
    # No double-buffer: all sequential
    layer_cycles_no_db = dma_load_cycles + compute_cycles + dma_store_cycles

    # Double-buffer: overlap load with compute
    if num_tiles <= 1:
        layer_cycles_db = layer_cycles_no_db
    else:
        # Startup: load first tile
        # Steady state: (N-1) tiles at max(load, compute) each
        # Drain: compute last tile
        # Store: all tiles (not overlapped)
        layer_cycles_db = (per_tile_load
                           + (num_tiles - 1) * max(per_tile_load, per_tile_compute)
                           + per_tile_compute
                           + num_tiles * per_tile_store)

    if hw.double_buffer:
        layer_cycles = layer_cycles_db
    else:
        layer_cycles = layer_cycles_no_db

    bottleneck = 'compute' if per_tile_compute >= per_tile_load else 'memory'

    return {
        'total_macs': total_macs,
        'compute_cycles': compute_cycles,
        'dma_load_cycles': dma_load_cycles,
        'dma_store_cycles': dma_store_cycles,
        'layer_cycles_no_db': layer_cycles_no_db,
        'layer_cycles_db': layer_cycles_db,
        'layer_cycles': layer_cycles,
        'bottleneck': bottleneck,
        'num_tiles': num_tiles,
    }


# ─── Layer Estimation Dispatcher ───

def estimate_layer_cycles(layer, hw):
    """Estimate cycles for a single layer. Calls tiling.py and operator-specific estimator."""

    def _estimate_with_db(use_db):
        """Run estimation with specified double-buffer setting."""
        tiling = compute_tiling(
            op_type=layer.op_type,
            in_h=layer.in_h, in_w=layer.in_w, in_c=layer.in_c,
            out_h=layer.out_h, out_w=layer.out_w, out_c=layer.out_c,
            kernel_h=layer.kernel_h, kernel_w=layer.kernel_w,
            stride_h=layer.stride_h, stride_w=layer.stride_w,
            dilation_h=layer.dilation_h, dilation_w=layer.dilation_w,
            elem_size=hw.elem_size,
            double_buffer=use_db,
        )
        estimators = {
            'conv': estimate_conv,
            'dw': estimate_dwconv,
            'fc': estimate_fc,
            'pool': estimate_pool,
            'add': estimate_add,
            'resize': estimate_resize,
            'concat': estimate_concat,
        }
        fn = estimators.get(layer.op_type, estimate_conv)
        # Build a temporary hw config with the specific db setting
        hw_tmp = HWConfig(
            array_size=hw.array_size,
            macs_per_cycle=hw.macs_per_cycle,
            act_bank_bytes=hw.act_bank_bytes,
            weight_buf_bytes=hw.weight_buf_bytes,
            ext_bw_bytes=hw.ext_bw_bytes,
            dw_parallel_ch=hw.dw_parallel_ch,
            elem_size=hw.elem_size,
            double_buffer=use_db,
            adaptive_db=False,
        )
        return fn(layer, tiling, hw_tmp), use_db

    if hw.adaptive_db:
        # Try both modes, pick the one with fewer cycles
        result_sb, _ = _estimate_with_db(False)
        result_db, _ = _estimate_with_db(True)
        if result_db['layer_cycles'] <= result_sb['layer_cycles']:
            result_db['db_mode'] = 'double'
            return result_db
        else:
            result_sb['db_mode'] = 'single'
            return result_sb
    else:
        # Get tiling (use double_buffer mode to get correct tile sizes)
        tiling = compute_tiling(
            op_type=layer.op_type,
            in_h=layer.in_h, in_w=layer.in_w, in_c=layer.in_c,
            out_h=layer.out_h, out_w=layer.out_w, out_c=layer.out_c,
            kernel_h=layer.kernel_h, kernel_w=layer.kernel_w,
            stride_h=layer.stride_h, stride_w=layer.stride_w,
            dilation_h=layer.dilation_h, dilation_w=layer.dilation_w,
            elem_size=hw.elem_size,
            double_buffer=hw.double_buffer,
        )

        estimators = {
            'conv': estimate_conv,
            'dw': estimate_dwconv,
            'fc': estimate_fc,
            'pool': estimate_pool,
            'add': estimate_add,
            'resize': estimate_resize,
            'concat': estimate_concat,
        }

        fn = estimators.get(layer.op_type, estimate_conv)
        result = fn(layer, tiling, hw)
        result['db_mode'] = 'double' if hw.double_buffer else 'single'
        return result


# ─── Model-Level Runner ───

def run_perf_model(layers, hw):
    """Run performance estimation on all layers."""
    results = []
    for i, layer in enumerate(layers):
        r = estimate_layer_cycles(layer, hw)
        r['layer_idx'] = i
        r['op_type'] = layer.op_type
        r['dims'] = f"{layer.out_h}x{layer.out_w}x{layer.out_c}"
        results.append(r)
    return results


def print_report(results, hw):
    """Print formatted per-layer table and summary."""
    mode = "double-buffer" if hw.double_buffer else "no-double-buffer"
    print(f"\n{'='*100}")
    print(f"Open-NPU Performance Estimation ({mode}, BW={hw.ext_bw_bytes}B/cyc, "
          f"{'INT16' if hw.elem_size == 2 else 'INT8'})")
    print(f"{'='*100}")

    # Header
    print(f"{'Layer':>5} | {'Op':>7} | {'Output':>12} | {'MACs':>12} | "
          f"{'Compute':>9} | {'DMA_Load':>9} | {'DMA_Store':>9} | "
          f"{'Bottleneck':>10} | {'Tiles':>5} | {'Cycles':>10}")
    print("-" * 100)

    total_macs = 0
    total_cycles = 0
    total_cycles_no_db = 0
    total_cycles_db = 0
    compute_bound = 0
    memory_bound = 0

    for r in results:
        print(f"{r['layer_idx']:>5} | {r['op_type']:>7} | {r['dims']:>12} | "
              f"{r['total_macs']:>12,} | {r['compute_cycles']:>9,} | "
              f"{r['dma_load_cycles']:>9,} | {r['dma_store_cycles']:>9,} | "
              f"{r['bottleneck']:>10} | {r['num_tiles']:>5} | "
              f"{r['layer_cycles']:>10,}")
        total_macs += r['total_macs']
        total_cycles += r['layer_cycles']
        total_cycles_no_db += r['layer_cycles_no_db']
        total_cycles_db += r['layer_cycles_db']
        if r['bottleneck'] == 'compute':
            compute_bound += 1
        else:
            memory_bound += 1

    print("-" * 100)

    # Summary
    util = total_macs / (total_cycles * hw.macs_per_cycle) * 100 if total_cycles > 0 else 0
    print(f"\n{'='*60}")
    print(f"Performance Summary")
    print(f"{'='*60}")
    print(f"  Total layers:                    {len(results)}")
    print(f"  Total MACs:                      {total_macs:>14,}")
    print(f"  Total cycles (no double-buffer): {total_cycles_no_db:>14,}")
    print(f"  Total cycles (double-buffer):    {total_cycles_db:>14,}")
    print(f"  Active mode cycles:              {total_cycles:>14,}")
    print(f"  MAC utilization:                 {util:>13.1f}%")
    print(f"  Compute-bound layers:            {compute_bound:>5} / {len(results)}")
    print(f"  Memory-bound layers:             {memory_bound:>5} / {len(results)}")

    print(f"\n  Estimated inference time:")
    for freq_mhz in [100, 200, 400]:
        time_ms = total_cycles / (freq_mhz * 1e6) * 1000
        fps = 1000.0 / time_ms if time_ms > 0 else 0
        print(f"    @ {freq_mhz:>3} MHz: {time_ms:>8.2f} ms  ({fps:.1f} FPS)")

    # Roofline
    total_dma_bytes = sum(r['dma_load_cycles'] + r['dma_store_cycles']
                         for r in results) * hw.ext_bw_bytes
    arithmetic_intensity = total_macs / total_dma_bytes if total_dma_bytes > 0 else 0
    peak_gops = hw.macs_per_cycle * 200e6 / 1e9  # @200MHz
    print(f"\n  Arithmetic intensity:            {arithmetic_intensity:.2f} MACs/byte")
    print(f"  Peak throughput @200MHz:         {peak_gops:.1f} GOPS")


# ─── Fused Block Cycle Estimation ───

def estimate_fused_block_cycles(block, fused_tiling, hw):
    """Estimate cycles for a fused inverted residual block.

    Execution within one spatial tile (sequential, no DRAM for intermediates):
      1. Load input tile from DRAM
      2. Conv1×1 #1: input (tile+halo)²×c_in → (tile+halo)²×c_mid
      3. (weight reload for DW)
      4. DW 3×3: (tile+halo)²×c_mid → tile²×c_mid
      5. (weight reload for Conv1×1 #2)
      6. Conv1×1 #2: tile²×c_mid → tile²×c_out
      7. Store output tile to DRAM

    DMA only at block boundary: load input, store output.
    Intermediate DMA = 0 (stays in SRAM).
    """
    if not fused_tiling.feasible:
        return None

    tile_h = fused_tiling.tile_h
    tile_w = fused_tiling.tile_w
    halo = block.dw_kernel - 1
    th = tile_h + halo  # input tile spatial with halo

    c_in = block.c_in
    c_mid = block.c_mid
    c_out = block.c_out
    spatial = block.spatial
    es = hw.elem_size

    # ─── Compute cycles per spatial tile ───

    # Conv1×1 #1: MACs = th² × c_mid × c_in (per OC group: th² × oc1_tile × c_in)
    conv1_macs = th * th * c_mid * c_in
    util_ic1 = min(c_in, hw.array_size) / hw.array_size
    util_oc1 = min(fused_tiling.oc1_tile, hw.array_size) / hw.array_size
    util1 = max(util_ic1 * util_oc1, 0.01)
    conv1_cycles = int(math.ceil(conv1_macs / hw.macs_per_cycle / util1))

    # DW 3×3: c_mid channels, output tile²
    dw_k = block.dw_kernel
    dw_macs = tile_h * tile_w * c_mid * dw_k * dw_k
    dw_cycles = tile_h * tile_w * _ceil_div(c_mid, hw.dw_parallel_ch) * dw_k * dw_k

    # Conv1×1 #2: MACs = tile² × c_out × c_mid
    conv2_macs = tile_h * tile_w * c_out * c_mid
    util_ic2 = min(c_mid, hw.array_size) / hw.array_size
    util_oc2 = min(fused_tiling.oc2_tile, hw.array_size) / hw.array_size
    util2 = max(util_ic2 * util_oc2, 0.01)
    conv2_cycles = int(math.ceil(conv2_macs / hw.macs_per_cycle / util2))

    per_tile_compute = conv1_cycles + dw_cycles + conv2_cycles

    # ─── DMA cycles per spatial tile ───

    # Load: input tile (tile+halo)² × c_in
    load_bytes = th * th * c_in * es
    # Store: output tile tile² × c_out
    store_bytes = tile_h * tile_w * c_out * es

    # Weight reloads within fused tile:
    #   Conv1×1 #1: c_in × c_mid (all OC groups per tile)
    #   DW: dw_k² × c_mid
    #   Conv1×1 #2: c_mid × c_out (all OC groups per tile)
    w1_bytes = c_in * c_mid * es
    w2_bytes = dw_k * dw_k * c_mid * es
    w3_bytes = c_mid * c_out * es
    weight_load_bytes = w1_bytes + w2_bytes + w3_bytes

    per_tile_load = int(math.ceil((load_bytes + weight_load_bytes) / hw.ext_bw_bytes))
    per_tile_store = int(math.ceil(store_bytes / hw.ext_bw_bytes))

    # ─── Total cycles ───

    num_tiles = fused_tiling.spatial_tiles

    # With double-buffer overlap (tile-level):
    if num_tiles <= 1:
        total_cycles = per_tile_load + per_tile_compute + per_tile_store
    else:
        # Pipeline: load[0], (N-1)×max(load,compute), compute[last], all stores
        total_cycles = (per_tile_load
                        + (num_tiles - 1) * max(per_tile_load, per_tile_compute)
                        + per_tile_compute
                        + num_tiles * per_tile_store)

    total_macs = conv1_macs + dw_macs + conv2_macs
    bottleneck = 'compute' if per_tile_compute >= per_tile_load else 'memory'

    return {
        'total_macs': total_macs,
        'compute_cycles': per_tile_compute * num_tiles,
        'dma_load_cycles': per_tile_load * num_tiles,
        'dma_store_cycles': per_tile_store * num_tiles,
        'layer_cycles': total_cycles,
        'num_tiles': num_tiles,
        'bottleneck': bottleneck,
        'fused': True,
        'block_layers': f"{block.start_idx}-{block.end_idx}",
    }


def run_perf_model_with_fusion(layers, hw):
    """Run performance estimation with super-layer fusion.

    Detects fusible blocks, estimates them as single fused units,
    and estimates remaining layers individually.
    """
    blocks = detect_fusible_blocks(layers)

    # Mark which layers are part of fused blocks
    fused_layer_set = set()
    for block in blocks:
        for idx in range(block.start_idx, block.end_idx + 1):
            fused_layer_set.add(idx)

    results = []
    block_iter = iter(blocks)
    next_block = next(block_iter, None)

    i = 0
    while i < len(layers):
        if next_block and i == next_block.start_idx:
            # Estimate as fused block
            ft = compute_fused_tiling(next_block, elem_size=hw.elem_size)
            r = estimate_fused_block_cycles(next_block, ft, hw)
            if r:
                r['layer_idx'] = i
                r['op_type'] = 'FUSED'
                r['dims'] = (f"{next_block.spatial}x{next_block.spatial}x"
                             f"{next_block.c_in}→{next_block.c_mid}→{next_block.c_out}")
                results.append(r)
            i = next_block.end_idx + 1
            next_block = next(block_iter, None)
        else:
            # Estimate individual layer
            r = estimate_layer_cycles(layers[i], hw)
            r['layer_idx'] = i
            r['op_type'] = layers[i].op_type
            r['dims'] = f"{layers[i].out_h}x{layers[i].out_w}x{layers[i].out_c}"
            r['fused'] = False
            results.append(r)
            i += 1

    return results, blocks


# ─── Built-in M110 MobileNetV2 ───

def m110_layers():
    """M110 MobileNetV2-style face embedding (63 layers)."""
    L = LayerDesc
    layers = [
        L('conv', 224, 224, 3, 112, 112, 56, kernel_h=3, kernel_w=3, stride_h=2, stride_w=2),
        L('dw', 112, 112, 56, 112, 112, 56, kernel_h=3, kernel_w=3),
        L('conv', 112, 112, 56, 112, 112, 56, kernel_h=1, kernel_w=1),
        L('conv', 112, 112, 56, 112, 112, 112, kernel_h=1, kernel_w=1),
        L('dw', 112, 112, 112, 56, 56, 112, kernel_h=3, kernel_w=3, stride_h=2, stride_w=2),
        L('conv', 56, 56, 112, 56, 56, 56, kernel_h=1, kernel_w=1),
    ]
    # 4 inverted residual blocks at 56×56
    for _ in range(4):
        layers.append(L('conv', 56, 56, 56, 56, 56, 112, kernel_h=1, kernel_w=1))
        layers.append(L('dw', 56, 56, 112, 56, 56, 112, kernel_h=3, kernel_w=3))
        layers.append(L('conv', 56, 56, 112, 56, 56, 56, kernel_h=1, kernel_w=1))
        layers.append(L('add', 56, 56, 56, 56, 56, 56))
    # Stride-2 transition
    layers.append(L('conv', 56, 56, 56, 56, 56, 224, kernel_h=1, kernel_w=1))
    layers.append(L('dw', 56, 56, 224, 28, 28, 224, kernel_h=3, kernel_w=3, stride_h=2, stride_w=2))
    layers.append(L('conv', 28, 28, 224, 28, 28, 96, kernel_h=1, kernel_w=1))
    # 6 inverted residual blocks at 28×28
    for _ in range(6):
        layers.append(L('conv', 28, 28, 96, 28, 28, 192, kernel_h=1, kernel_w=1))
        layers.append(L('dw', 28, 28, 192, 28, 28, 192, kernel_h=3, kernel_w=3))
        layers.append(L('conv', 28, 28, 192, 28, 28, 96, kernel_h=1, kernel_w=1))
        layers.append(L('add', 28, 28, 96, 28, 28, 96))
    # Stride-2 transition
    layers.append(L('conv', 28, 28, 96, 28, 28, 384, kernel_h=1, kernel_w=1))
    layers.append(L('dw', 28, 28, 384, 14, 14, 384, kernel_h=3, kernel_w=3, stride_h=2, stride_w=2))
    layers.append(L('conv', 14, 14, 384, 14, 14, 96, kernel_h=1, kernel_w=1))
    # 2 inverted residual blocks at 14×14
    for _ in range(2):
        layers.append(L('conv', 14, 14, 96, 14, 14, 192, kernel_h=1, kernel_w=1))
        layers.append(L('dw', 14, 14, 192, 14, 14, 192, kernel_h=3, kernel_w=3))
        layers.append(L('conv', 14, 14, 192, 14, 14, 96, kernel_h=1, kernel_w=1))
        layers.append(L('add', 14, 14, 96, 14, 14, 96))
    # Head
    layers.append(L('conv', 14, 14, 96, 14, 14, 512, kernel_h=1, kernel_w=1))
    layers.append(L('dw', 14, 14, 512, 1, 1, 512, kernel_h=14, kernel_w=14, stride_h=14, stride_w=14))
    layers.append(L('conv', 1, 1, 512, 1, 1, 512, kernel_h=1, kernel_w=1))
    return layers


# ─── Built-in ResNet-18 Half-Width ───

def resnet18_layers():
    """ResNet-18 half-width (32/64/128/256 channels), 31 NPU layers."""
    L = LayerDesc
    layers = [
        # Stem
        L('conv', 224, 224, 3, 112, 112, 32, kernel_h=7, kernel_w=7, stride_h=2, stride_w=2),
        L('pool', 112, 112, 32, 56, 56, 32, pool_h=3, pool_w=3, pool_stride_h=2, pool_stride_w=2),
        # Layer1 Block0
        L('conv', 56, 56, 32, 56, 56, 32, kernel_h=3, kernel_w=3),
        L('conv', 56, 56, 32, 56, 56, 32, kernel_h=3, kernel_w=3),
        L('add', 56, 56, 32, 56, 56, 32),
        # Layer1 Block1
        L('conv', 56, 56, 32, 56, 56, 32, kernel_h=3, kernel_w=3),
        L('conv', 56, 56, 32, 56, 56, 32, kernel_h=3, kernel_w=3),
        L('add', 56, 56, 32, 56, 56, 32),
        # Layer2 Block0 (stride=2, with 1x1 shortcut)
        L('conv', 56, 56, 32, 28, 28, 64, kernel_h=3, kernel_w=3, stride_h=2, stride_w=2),
        L('conv', 28, 28, 64, 28, 28, 64, kernel_h=3, kernel_w=3),
        L('conv', 56, 56, 32, 28, 28, 64, kernel_h=1, kernel_w=1, stride_h=2, stride_w=2),
        L('add', 28, 28, 64, 28, 28, 64),
        # Layer2 Block1
        L('conv', 28, 28, 64, 28, 28, 64, kernel_h=3, kernel_w=3),
        L('conv', 28, 28, 64, 28, 28, 64, kernel_h=3, kernel_w=3),
        L('add', 28, 28, 64, 28, 28, 64),
        # Layer3 Block0 (stride=2, with 1x1 shortcut)
        L('conv', 28, 28, 64, 14, 14, 128, kernel_h=3, kernel_w=3, stride_h=2, stride_w=2),
        L('conv', 14, 14, 128, 14, 14, 128, kernel_h=3, kernel_w=3),
        L('conv', 28, 28, 64, 14, 14, 128, kernel_h=1, kernel_w=1, stride_h=2, stride_w=2),
        L('add', 14, 14, 128, 14, 14, 128),
        # Layer3 Block1
        L('conv', 14, 14, 128, 14, 14, 128, kernel_h=3, kernel_w=3),
        L('conv', 14, 14, 128, 14, 14, 128, kernel_h=3, kernel_w=3),
        L('add', 14, 14, 128, 14, 14, 128),
        # Layer4 Block0 (stride=2, with 1x1 shortcut)
        L('conv', 14, 14, 128, 7, 7, 256, kernel_h=3, kernel_w=3, stride_h=2, stride_w=2),
        L('conv', 7, 7, 256, 7, 7, 256, kernel_h=3, kernel_w=3),
        L('conv', 14, 14, 128, 7, 7, 256, kernel_h=1, kernel_w=1, stride_h=2, stride_w=2),
        L('add', 7, 7, 256, 7, 7, 256),
        # Layer4 Block1
        L('conv', 7, 7, 256, 7, 7, 256, kernel_h=3, kernel_w=3),
        L('conv', 7, 7, 256, 7, 7, 256, kernel_h=3, kernel_w=3),
        L('add', 7, 7, 256, 7, 7, 256),
        # Global Average Pool + FC
        L('pool', 7, 7, 256, 1, 1, 256, pool_h=7, pool_w=7, pool_stride_h=7, pool_stride_w=7),
        L('fc', 1, 1, 256, 1, 1, 10),
    ]
    return layers


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description='Open-NPU Performance Estimator')
    parser.add_argument('--resnet18-test', action='store_true',
                        help='Run built-in ResNet-18 half-width test')
    parser.add_argument('--m110-test', action='store_true',
                        help='Run built-in M110 MobileNetV2 test')
    parser.add_argument('--bits', type=int, default=8, choices=[8, 16],
                        help='Data type: 8=INT8, 16=INT16 (default: 8)')
    parser.add_argument('--no-double-buffer', action='store_true',
                        help='Disable double-buffer (serial DMA+compute, full buffer)')
    parser.add_argument('--adaptive', action='store_true',
                        help='Adaptive mode: per-layer best of single/double buffer')
    parser.add_argument('--compare', action='store_true',
                        help='Compare single-buffer vs double-buffer vs adaptive performance')
    parser.add_argument('--fusion', action='store_true',
                        help='Enable super-layer fusion (Conv1x1→DW→Conv1x1 blocks)')
    add_hw_args(parser)
    args = parser.parse_args()

    hw_cfg = hw_config_from_args(args)

    if not args.resnet18_test and not args.m110_test:
        print("Usage: python3 perf_model.py --resnet18-test|--m110-test [--bits 8|16] [--bw 4] [--compare] [--fusion]")
        return

    if args.m110_test:
        layers = m110_layers()
        model_name = "M110 MobileNetV2"
    else:
        layers = resnet18_layers()
        model_name = "ResNet-18 Half-Width"

    if args.fusion:
        # Fusion comparison mode
        hw = HWConfig.from_hw(
            hw_cfg,
            elem_size=2 if args.bits == 16 else 1,
            double_buffer=True,
            adaptive_db=True,
        )

        # Without fusion (adaptive baseline)
        results_baseline = run_perf_model(layers, hw)
        total_baseline = sum(r['layer_cycles'] for r in results_baseline)
        total_macs_baseline = sum(r['total_macs'] for r in results_baseline)

        # With fusion
        results_fused, blocks = run_perf_model_with_fusion(layers, hw)
        total_fused = sum(r['layer_cycles'] for r in results_fused)
        total_macs_fused = sum(r['total_macs'] for r in results_fused)

        print(f"\n{'='*100}")
        print(f"{model_name} — Super-Layer Fusion Analysis "
              f"({'INT16' if hw.elem_size == 2 else 'INT8'}, BW={hw.ext_bw_bytes}B/cyc)")
        print(f"{'='*100}")

        print(f"\n  Detected fusible blocks: {len(blocks)}")
        if blocks:
            print(f"\n  {'#':>3} | {'Layers':>8} | {'Spatial':>7} | {'Channels':>14} | "
                  f"{'Tile':>6} | {'Tiles':>5}")
            print("  " + "-" * 60)
            for idx, blk in enumerate(blocks):
                ft = compute_fused_tiling(blk, elem_size=hw.elem_size)
                ch = f"{blk.c_in}→{blk.c_mid}→{blk.c_out}"
                print(f"  {idx:>3} | {blk.start_idx:>2}-{blk.end_idx:<4} | "
                      f"{blk.spatial:>3}×{blk.spatial:<3} | {ch:>14} | "
                      f"{ft.tile_h:>2}×{ft.tile_w:<2} | {ft.spatial_tiles:>5}")

        print(f"\n  {'Entry':>5} | {'Type':>5} | {'Dims':>24} | {'Cycles':>10} | {'Bottleneck':>10}")
        print("  " + "-" * 70)
        for r in results_fused:
            fused_marker = " *" if r.get('fused') else ""
            print(f"  {r['layer_idx']:>5} | {r['op_type']:>5} | {r['dims']:>24} | "
                  f"{r['layer_cycles']:>10,} | {r.get('bottleneck', '-'):>10}{fused_marker}")

        speedup = total_baseline / total_fused if total_fused > 0 else 0
        print(f"\n  {'='*60}")
        print(f"  Fusion Performance Summary")
        print(f"  {'='*60}")
        print(f"    Baseline (adaptive, no fusion):  {total_baseline:>12,} cycles")
        print(f"    With super-layer fusion:         {total_fused:>12,} cycles")
        print(f"    Speedup:                         {speedup:>11.2f}x")
        print(f"    Fused blocks:                    {len(blocks)}")

        # DMA savings
        baseline_dma = sum(r['dma_load_cycles'] + r['dma_store_cycles']
                           for r in results_baseline) * hw.ext_bw_bytes
        fused_dma = sum(r['dma_load_cycles'] + r['dma_store_cycles']
                        for r in results_fused) * hw.ext_bw_bytes
        dma_saving = (baseline_dma - fused_dma) / baseline_dma * 100 if baseline_dma > 0 else 0
        print(f"    DMA bytes (baseline):            {baseline_dma:>12,}")
        print(f"    DMA bytes (fused):               {fused_dma:>12,}")
        print(f"    DMA reduction:                   {dma_saving:>11.1f}%")

        print(f"\n  Estimated inference time:")
        for freq_mhz in [100, 200, 400]:
            t_b = total_baseline / (freq_mhz * 1e6) * 1000
            t_f = total_fused / (freq_mhz * 1e6) * 1000
            print(f"    @ {freq_mhz:>3} MHz: baseline {t_b:>7.2f} ms → "
                  f"fused {t_f:>7.2f} ms ({1000/t_f:.1f} FPS)")

        return

    if args.compare:
        # Run all three modes and show comparison
        print(f"\n{model_name} ({len(layers)} layers, "
              f"{'INT16' if args.bits == 16 else 'INT8'}) — Single vs Double vs Adaptive\n")
        print("=" * 110)

        hw_single = HWConfig.from_hw(
            hw_cfg,
            elem_size=2 if args.bits == 16 else 1,
            double_buffer=False,
        )
        hw_double = HWConfig.from_hw(
            hw_cfg,
            elem_size=2 if args.bits == 16 else 1,
            double_buffer=True,
        )
        hw_adaptive = HWConfig.from_hw(
            hw_cfg,
            elem_size=2 if args.bits == 16 else 1,
            double_buffer=True,
            adaptive_db=True,
        )

        results_single = run_perf_model(layers, hw_single)
        results_double = run_perf_model(layers, hw_double)
        results_adaptive = run_perf_model(layers, hw_adaptive)

        # Per-layer comparison table
        print(f"\n{'Layer':>5} | {'Op':>5} | {'Output':>10} | "
              f"{'SB Cycles':>10} | {'DB Cycles':>10} | "
              f"{'Adapt Cyc':>10} | {'Mode':>6} | {'vs SB':>6}")
        print("-" * 90)

        total_single = 0
        total_double = 0
        total_adaptive = 0
        db_layers = 0
        sb_layers = 0

        for rs, rd, ra in zip(results_single, results_double, results_adaptive):
            cyc_s = rs['layer_cycles']
            cyc_d = rd['layer_cycles']
            cyc_a = ra['layer_cycles']
            total_single += cyc_s
            total_double += cyc_d
            total_adaptive += cyc_a
            mode = ra.get('db_mode', '?')
            if mode == 'double':
                db_layers += 1
            else:
                sb_layers += 1
            speedup = cyc_s / cyc_a if cyc_a > 0 else 0
            print(f"{rs['layer_idx']:>5} | {rs['op_type']:>5} | {rs['dims']:>10} | "
                  f"{cyc_s:>10,} | {cyc_d:>10,} | "
                  f"{cyc_a:>10,} | {'DB' if mode == 'double' else 'SB':>6} | {speedup:>5.2f}x")

        print("-" * 90)

        sp_db = total_single / total_double if total_double > 0 else 0
        sp_ad = total_single / total_adaptive if total_adaptive > 0 else 0
        print(f"\n{'Summary':>20}")
        print(f"  Single-buffer total:    {total_single:>12,} cycles")
        print(f"  Double-buffer total:    {total_double:>12,} cycles  ({sp_db:.2f}x vs SB)")
        print(f"  Adaptive total:         {total_adaptive:>12,} cycles  ({sp_ad:.2f}x vs SB)")
        print(f"  Adaptive choices:       {db_layers} layers DB, {sb_layers} layers SB")

        for freq_mhz in [100, 200, 400]:
            t_s = total_single / (freq_mhz * 1e6) * 1000
            t_a = total_adaptive / (freq_mhz * 1e6) * 1000
            print(f"\n  @ {freq_mhz:>3} MHz:")
            print(f"    Single-buffer: {t_s:>8.2f} ms ({1000/t_s:.1f} FPS)")
            print(f"    Adaptive:      {t_a:>8.2f} ms ({1000/t_a:.1f} FPS)")

    else:
        hw = HWConfig.from_hw(
            hw_cfg,
            elem_size=2 if args.bits == 16 else 1,
            double_buffer=not args.no_double_buffer,
            adaptive_db=args.adaptive,
        )
        print(f"{model_name} ({len(layers)} layers, "
              f"{'INT16' if hw.elem_size == 2 else 'INT8'})")
        results = run_perf_model(layers, hw)
        print_report(results, hw)


if __name__ == '__main__':
    main()
