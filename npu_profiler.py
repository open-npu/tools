#!/usr/bin/env python3
"""
Open-NPU Performance, Roofline & On-Chip SRAM Profiler (Nsight-style)

Analyzes quantized NPU models (NPU1 binary or metadata.json) and optional
RTL/SoC hardware PMU logs to provide:
  1. Speed-of-Light (SOL%) Compute & External DMA Bandwidth summary
  2. Roofline Arithmetic Intensity & Bus-Width Scaling (32 / 64 / 128-bit)
  3. 4-Bank On-Chip SRAM (IFM / OFM / WGT / PARAM) occupancy, tiling halo
     amplification, weight reload amplification, and fusion traffic savings
  4. Cycle-accurate Hardware PMU Stall Attribution (from SoC UART logs)
  5. Actionable Compiler / Tiling / Fusion / Microarchitecture Advisor

Usage examples:
  # Offline static + roofline analysis on a packed NPU1 binary
  python3 npu_profiler.py --model model.npu1.bin --dma-bus-width 128

  # Offline analysis on converter/golden metadata.json
  python3 npu_profiler.py --meta path/to/metadata.json --dma-bus-width 128

  # Full Nsight-style HW PMU stall + SRAM + Roofline analysis with SoC log
  python3 npu_profiler.py --meta path/to/metadata.json --soc-log sim.log --dma-bus-width 128

  # Run built-in self-test
  python3 npu_profiler.py selftest

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from hw_config import HWConfig, add_hw_args, hw_config_from_args
from model_packer import (
    FIXED_CONFIG_SIZE,
    MODEL_MAGIC,
    OP_CONCAT,
    OP_CONV2D,
    OP_DECONV,
    OP_DW_CONV,
    OP_ELTWISE_ADD,
    OP_FC,
    OP_POOLING,
    OP_RESIZE,
    SCHED_CTRL_DB_EN,
    SCHED_CTRL_FUSE_END,
    SCHED_CTRL_FUSE_MID,
    SCHED_CTRL_FUSE_START,
    SCHED_CTRL_FUSE_STRIP,
    SCHED_CTRL_PER_TILE_STORE,
    SCHED_CTRL_SUPER_BLOCK,
)

OP_NAMES = {
    OP_CONV2D: "Conv2D",
    OP_DW_CONV: "DWConv",
    OP_FC: "FC",
    OP_POOLING: "Pooling",
    OP_ELTWISE_ADD: "EltwiseAdd",
    OP_RESIZE: "Resize",
    OP_DECONV: "Deconv",
    OP_CONCAT: "Concat",
}

PPU_LANES = 8
PHYSICAL_PARAM_BYTES_PER_CH = 16  # 4 x 32-bit words per channel in RTL PARAM SRAM
BINARY_PARAM_BYTES_PER_CH = 14    # packed NPU1 descriptor bytes per channel


@dataclass
class LayerDesc:
    idx: int
    op_type: int
    data_type: int  # 0=INT8, 1=INT16
    in_h: int
    in_w: int
    in_c: int
    out_h: int
    out_w: int
    out_c: int
    kernel_h: int = 1
    kernel_w: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_top: int = 0
    pad_left: int = 0
    pool_mode: int = 0
    pool_h: int = 1
    pool_w: int = 1
    pool_stride_h: int = 1
    pool_stride_w: int = 1
    global_pool: int = 0
    tile_h: int = 0
    tile_w: int = 0
    tile_num_h: int = 0
    tile_num_w: int = 0
    post_ctrl: int = 0
    sched_ctrl: int = 0
    wgt_layout: int = 1
    param_ch_count: int = 0
    has_lut: bool = False
    has_add: bool = False
    residual_src: int = -1
    input_src: int = -1
    dma_in_size: int = 0
    dma_wgt_size: int = 0
    dma_out_size: int = 0
    tile_in_size: int = 0
    tile_out_size: int = 0


@dataclass
class HWLayerPMU:
    idx: int
    npu_cycles: int = 0
    hw_macs: int = 0
    dma_cycles: int = 0
    comp_cycles: int = 0
    ovlp_cycles: int = 0
    st_mac: int = 0
    st_wait: int = 0
    st_wb: int = 0
    st_pool: int = 0
    st_add: int = 0
    st_rsz: int = 0
    st_cfg: int = 0
    st_oth: int = 0
    wt_act: int = 0
    wt_wgt: int = 0
    wt_db: int = 0
    mac_idle: int = 0
    wt_store: int = 0
    sram_fused_skip_store: bool = False


@dataclass
class SoCLogSummary:
    hw_config_hex: Optional[int] = None
    dma_width_bits: Optional[int] = None
    dma_proto: Optional[str] = None
    spad_kb: Optional[int] = None
    wall_cycles: Optional[int] = None
    layers: Dict[int, HWLayerPMU] = field(default_factory=dict)
    super_ranges: List[Tuple[int, int]] = field(default_factory=list)
    untiled_layers: List[int] = field(default_factory=list)


@dataclass
class LayerProfile:
    idx: int
    op_name: str
    op_family: str
    shape_str: str
    elem_bytes: int
    macs: int
    ops: int
    # Raw vs actual external bytes
    raw_in_bytes: int
    raw_wgt_bytes: int
    raw_out_bytes: int
    raw_param_bytes: int
    ext_in_bytes: int
    ext_wgt_bytes: int
    ext_out_bytes: int
    ext_param_bytes: int
    total_ext_bytes: int
    halo_extra_bytes: int
    wgt_reload_extra_bytes: int
    fusion_saved_bytes: int
    # SRAM footprint
    ifm_used_bytes: int
    ofm_used_bytes: int
    wgt_used_bytes: int
    prm_used_bytes: int
    ifm_occ_pct: float
    ofm_occ_pct: float
    wgt_occ_pct: float
    prm_occ_pct: float
    # Tiling / schedule flags
    tiled: bool
    num_tiles: int
    oc_groups: int
    db_en: bool
    fuse_tag: str
    lane_occ_pct: float
    # Analytical roofline
    arith_intensity: float
    ideal_mac_cycles: int
    ideal_ppu_cycles: int
    ideal_comp_cycles: int
    ideal_dma_cycles: int
    est_cycles: int
    bound_regime: str
    # Optional HW PMU
    hw_cycles: Optional[int] = None
    hw_dma_cycles: Optional[int] = None
    hw_comp_cycles: Optional[int] = None
    hw_ovlp_cycles: Optional[int] = None
    pmu: Optional[HWLayerPMU] = None


# ─────────────────────────────────────────────────────────────────────
#  Input Parsers
# ─────────────────────────────────────────────────────────────────────

def parse_npu1_binary(path: str) -> List[LayerDesc]:
    with open(path, "rb") as f:
        blob = f.read()
    if len(blob) < 16:
        raise ValueError(f"File too small to be NPU1 binary: {path}")
    magic, num_layers, weight_offset, _ = struct.unpack_from("<IIII", blob, 0)
    if magic != MODEL_MAGIC:
        raise ValueError(f"Invalid NPU1 magic 0x{magic:08X} in {path}")

    off = 16
    layers: List[LayerDesc] = []
    for idx in range(num_layers):
        if off + FIXED_CONFIG_SIZE > weight_offset:
            raise ValueError(f"Truncated descriptor at layer {idx}")
        b = blob[off:off + FIXED_CONFIG_SIZE]
        op_type, data_type = struct.unpack_from("<BB", b, 0)
        in_h, in_w, in_c, out_h, out_w, out_c = struct.unpack_from("<6H", b, 2)
        kh, kw, dh, dw, sh, sw, pt, pb, pl, pr = struct.unpack_from("<10B", b, 14)
        pm, ph, pw, psh, psw, gpool = struct.unpack_from("<6B", b, 24)
        concat_off, concat_tot = struct.unpack_from("<2H", b, 35)
        th, tw, tnh, tnw = struct.unpack_from("<4H", b, 39)
        post_ctrl, sched_ctrl = struct.unpack_from("<2B", b, 47)
        wgt_layout = b[54]
        param_ch_count = struct.unpack_from("<H", b, 55)[0]
        has_lut = bool(b[57])
        has_add = bool(b[58])
        residual_src = struct.unpack_from("<b", b, 59)[0]
        input_src = struct.unpack_from("<h", b, 60)[0]

        var_bytes = param_ch_count * BINARY_PARAM_BYTES_PER_CH
        if has_add:
            var_bytes += 8
        if has_lut:
            var_bytes += 256 + 512
        off += FIXED_CONFIG_SIZE + var_bytes

        layers.append(LayerDesc(
            idx=idx,
            op_type=op_type,
            data_type=data_type,
            in_h=in_h, in_w=in_w, in_c=in_c,
            out_h=out_h, out_w=out_w, out_c=out_c,
            kernel_h=max(1, kh), kernel_w=max(1, kw),
            dilation_h=max(1, dh), dilation_w=max(1, dw),
            stride_h=max(1, sh), stride_w=max(1, sw),
            pad_top=pt, pad_left=pl,
            pool_mode=pm, pool_h=max(1, ph), pool_w=max(1, pw),
            pool_stride_h=max(1, psh), pool_stride_w=max(1, psw),
            global_pool=gpool,
            tile_h=th, tile_w=tw, tile_num_h=tnh, tile_num_w=tnw,
            post_ctrl=post_ctrl, sched_ctrl=sched_ctrl,
            wgt_layout=wgt_layout, param_ch_count=param_ch_count,
            has_lut=has_lut, has_add=has_add,
            residual_src=residual_src, input_src=input_src,
        ))
    return layers


def parse_metadata_json(path: str) -> List[LayerDesc]:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    layers: List[LayerDesc] = []
    for idx, m in enumerate(meta):
        pool_cfg = m.get("pool_cfg", 0)
        pm = pool_cfg & 0xF
        ph = max(1, (pool_cfg >> 4) & 0xF)
        pw = max(1, (pool_cfg >> 8) & 0xF)
        psh = max(1, (pool_cfg >> 12) & 0xF)
        psw = max(1, (pool_cfg >> 16) & 0xF)
        gpool = (pool_cfg >> 20) & 0x1
        op_type = m.get("op_type", 0)
        layers.append(LayerDesc(
            idx=idx,
            op_type=op_type,
            data_type=m.get("data_type", 0),
            in_h=m.get("in_h", 1),
            in_w=m.get("in_w", 1),
            in_c=m.get("in_c", 1),
            out_h=m.get("out_h", 1),
            out_w=m.get("out_w", 1),
            out_c=m.get("out_c", 1),
            kernel_h=max(1, m.get("kernel_h", 1)),
            kernel_w=max(1, m.get("kernel_w", 1)),
            dilation_h=max(1, m.get("dilation_h", 1)),
            dilation_w=max(1, m.get("dilation_w", 1)),
            stride_h=max(1, m.get("stride_h", 1)),
            stride_w=max(1, m.get("stride_w", 1)),
            pad_top=m.get("pad_top", 0),
            pad_left=m.get("pad_left", 0),
            pool_mode=pm, pool_h=ph, pool_w=pw,
            pool_stride_h=psh, pool_stride_w=psw,
            global_pool=gpool,
            tile_h=m.get("tile_h", 0),
            tile_w=m.get("tile_w", 0),
            tile_num_h=m.get("tile_num_h", 0),
            tile_num_w=m.get("tile_num_w", 0),
            post_ctrl=m.get("post_ctrl", 0),
            sched_ctrl=m.get("sched_ctrl", 0),
            wgt_layout=m.get("wgt_layout", 1),
            param_ch_count=m.get("dma_param_count", m.get("out_c", 0)),
            has_lut=bool(m.get("post_ctrl", 0) & (1 << 4)),
            has_add=(op_type == OP_ELTWISE_ADD),
            residual_src=m.get("residual_src", -1),
            input_src=m.get("input_src", -1),
            dma_in_size=m.get("dma_in_size", 0),
            dma_wgt_size=m.get("dma_wgt_size", 0),
            dma_out_size=m.get("dma_out_size", 0),
            tile_in_size=m.get("tile_in_size", 0),
            tile_out_size=m.get("tile_out_size", 0),
        ))
    return layers


_RE_HW_CFG = re.compile(
    r"SPAD\s+(\d+)KB,\s+DMA\s+(\d+)b\s+(WB|AXI4),\s+HW config:\s+0x([0-9a-fA-F]+)"
)
_RE_PERF_L = re.compile(
    r"PERF_L(\d+)\s+npu=(\d+)\s+mac=(\d+)\s+dma=(\d+)\s+comp=(\d+)\s+ovlp=(\d+)"
)
_RE_STL = re.compile(
    r"STL(\d+)\s+mac=(\d+)\s+wait=(\d+)\s+wb=(\d+)\s+pool=(\d+)\s+add=(\d+)\s+"
    r"rsz=(\d+)\s+cfg=(\d+)\s+oth=(\d+)\s+wact=(\d+)\s+wwgt=(\d+)\s+wdb=(\d+)\s+"
    r"midle=(\d+)(?:\s+dma=(\d+))?\s+wst=(\d+)"
)
_RE_FUSE_SKIP = re.compile(r"L(\d+):\s+PASS\s+\(SRAM fuse")
_RE_SUPER_PASS = re.compile(r"SUPER PASS L(\d+)-L(\d+)")
_RE_STRIP_PASS = re.compile(r"L(\d+)->L(\d+)\s+STRIP\s+PASS")
_RE_PERF_BEGIN = re.compile(r"\[PERF\]\s+cyc=(\d+)\s+\|\s+PERF_BEGIN")
_RE_PERF_END = re.compile(r"\[PERF\]\s+cyc=(\d+)\s+\|\s+PERF_END")
_RE_WALL_CYC = re.compile(r"(?:total_cycles|Inference cycles|PERF_CYCLES_TOTAL)[=:\s]+(\d+)")


def parse_soc_log(path: str) -> SoCLogSummary:
    summary = SoCLogSummary()
    t_begin: Optional[int] = None
    t_end: Optional[int] = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_b = _RE_PERF_BEGIN.search(line)
            if m_b:
                t_begin = int(m_b.group(1))
                continue
            m_e = _RE_PERF_END.search(line)
            if m_e:
                t_end = int(m_e.group(1))
                if t_begin is not None and t_end >= t_begin:
                    summary.wall_cycles = t_end - t_begin
                continue
            m_hw = _RE_HW_CFG.search(line)
            if m_hw:
                summary.spad_kb = int(m_hw.group(1))
                summary.dma_width_bits = int(m_hw.group(2))
                summary.dma_proto = m_hw.group(3).lower()
                summary.hw_config_hex = int(m_hw.group(4), 16)
                continue

            m_pl = _RE_PERF_L.search(line)
            if m_pl:
                idx = int(m_pl.group(1))
                rec = summary.layers.setdefault(idx, HWLayerPMU(idx=idx))
                rec.npu_cycles = int(m_pl.group(2))
                rec.hw_macs = int(m_pl.group(3))
                rec.dma_cycles = int(m_pl.group(4))
                rec.comp_cycles = int(m_pl.group(5))
                rec.ovlp_cycles = int(m_pl.group(6))
                continue

            m_stl = _RE_STL.search(line)
            if m_stl:
                idx = int(m_stl.group(1))
                rec = summary.layers.setdefault(idx, HWLayerPMU(idx=idx))
                rec.st_mac = int(m_stl.group(2))
                rec.st_wait = int(m_stl.group(3))
                rec.st_wb = int(m_stl.group(4))
                rec.st_pool = int(m_stl.group(5))
                rec.st_add = int(m_stl.group(6))
                rec.st_rsz = int(m_stl.group(7))
                rec.st_cfg = int(m_stl.group(8))
                rec.st_oth = int(m_stl.group(9))
                rec.wt_act = int(m_stl.group(10))
                rec.wt_wgt = int(m_stl.group(11))
                rec.wt_db = int(m_stl.group(12))
                rec.mac_idle = int(m_stl.group(13))
                if m_stl.group(14) is not None:
                    rec.dma_cycles = int(m_stl.group(14))
                rec.wt_store = int(m_stl.group(15))
                if rec.npu_cycles == 0:
                    rec.npu_cycles = (
                        rec.st_mac + rec.st_wait + rec.st_wb + rec.st_pool
                        + rec.st_add + rec.st_rsz + rec.st_cfg + rec.st_oth
                    )
                continue

            m_fs = _RE_FUSE_SKIP.search(line)
            if m_fs:
                idx = int(m_fs.group(1))
                rec = summary.layers.setdefault(idx, HWLayerPMU(idx=idx))
                rec.sram_fused_skip_store = True
                continue

            m_sp = _RE_SUPER_PASS.search(line) or _RE_STRIP_PASS.search(line)
            if m_sp:
                summary.super_ranges.append((int(m_sp.group(1)), int(m_sp.group(2))))
                continue

            m_wc = _RE_WALL_CYC.search(line)
            if m_wc:
                summary.wall_cycles = int(m_wc.group(1))
    return summary


# ─────────────────────────────────────────────────────────────────────
#  Analytical Roofline & SRAM Model
# ─────────────────────────────────────────────────────────────────────

def classify_op_family(d: LayerDesc) -> str:
    if d.op_type == OP_CONV2D:
        if d.kernel_h == 1 and d.kernel_w == 1:
            return "Conv1x1"
        return f"Conv{d.kernel_h}x{d.kernel_w}"
    if d.op_type == OP_DW_CONV:
        if d.global_pool or (d.out_h == 1 and d.out_w == 1 and d.kernel_h == d.in_h):
            return "GlobalDW"
        return f"DW{d.kernel_h}x{d.kernel_w}"
    return OP_NAMES.get(d.op_type, f"Op{d.op_type}")


def detect_super_blocks(layers: List[LayerDesc]) -> Dict[int, str]:
    """Identify eligible or tagged Conv1x1->DW3x3(s1)->Conv1x1[->Add] super-blocks."""
    tags: Dict[int, str] = {}
    i = 0
    n = len(layers)
    while i < n:
        d0 = layers[i]
        if d0.sched_ctrl & SCHED_CTRL_SUPER_BLOCK:
            tags[i] = "SUPER_HEAD"
            j = i + 1
            while j < n and (layers[j].sched_ctrl & SCHED_CTRL_SUPER_BLOCK):
                tags[j] = "SUPER_SUB"
                j += 1
            i = j
            continue
        # Also detect structural eligibility matching super_tile.py (56, 28, 14 INT8)
        if (i + 2 < n
                and d0.op_type == OP_CONV2D and d0.kernel_h == 1 and d0.kernel_w == 1
                and layers[i + 1].op_type == OP_DW_CONV
                and layers[i + 1].kernel_h == 3 and layers[i + 1].kernel_w == 3
                and layers[i + 1].stride_h == 1 and layers[i + 1].stride_w == 1
                and layers[i + 2].op_type == OP_CONV2D
                and layers[i + 2].kernel_h == 1 and layers[i + 2].kernel_w == 1
                and d0.data_type == 0
                and d0.out_h == d0.out_w and d0.out_h in (56, 28, 14)
                and d0.out_c == layers[i + 1].in_c == layers[i + 2].in_c):
            w_sum = (d0.in_c * d0.out_c + 9 * layers[i + 1].in_c
                     + layers[i + 2].in_c * layers[i + 2].out_c)
            p_sum = (d0.out_c + layers[i + 1].out_c + layers[i + 2].out_c) * 16
            if w_sum <= 80 * 1024 and p_sum <= 12 * 1024:
                tags[i] = "SUPER_ELIGIBLE"
        i += 1
    return tags


def analyze_model(
    layers: List[LayerDesc],
    hw: HWConfig,
    soc_log: Optional[SoCLogSummary] = None,
) -> List[LayerProfile]:
    profiles: List[LayerProfile] = []
    super_tags = detect_super_blocks(layers)
    soc_super_map: Dict[int, Tuple[int, int]] = {}
    if soc_log:
        for s_lo, s_hi in soc_log.super_ranges:
            for k in range(s_lo, s_hi + 1):
                soc_super_map[k] = (s_lo, s_hi)

    for i, d in enumerate(layers):
        eb = 2 if d.data_type == 1 else 1
        op_name = OP_NAMES.get(d.op_type, f"Op{d.op_type}")
        op_family = classify_op_family(d)
        out_px = d.out_h * d.out_w

        # 1. Exact MACs & Ops
        if d.op_type in (OP_CONV2D, OP_FC):
            k_depth = d.kernel_h * d.kernel_w * d.in_c
            macs = out_px * d.out_c * k_depth
            ops = macs * 2
        elif d.op_type == OP_DW_CONV:
            k_area = d.kernel_h * d.kernel_w
            macs = out_px * d.out_c * k_area
            ops = macs * 2
        elif d.op_type == OP_POOLING:
            macs = 0
            k_area = (d.in_h * d.in_w) if d.global_pool else (d.pool_h * d.pool_w)
            ops = out_px * d.out_c * k_area
        else:
            macs = 0
            ops = out_px * d.out_c

        # 2. Raw Tensor Bytes
        in_mult = 2 if d.op_type == OP_ELTWISE_ADD else 1
        raw_in = in_mult * d.in_h * d.in_w * d.in_c * eb
        raw_out = d.out_h * d.out_w * d.out_c * eb
        if d.op_type in (OP_CONV2D, OP_FC):
            raw_wgt = d.out_c * d.in_c * d.kernel_h * d.kernel_w * eb
        elif d.op_type == OP_DW_CONV:
            raw_wgt = d.in_c * d.kernel_h * d.kernel_w * eb
        else:
            raw_wgt = 0
        param_ch = d.param_ch_count if d.param_ch_count > 0 else (
            d.out_c if d.op_type in (OP_CONV2D, OP_DW_CONV, OP_FC) else 0
        )
        raw_param = param_ch * PHYSICAL_PARAM_BYTES_PER_CH + (8 if d.has_add else 0)

        # 3. Tiling & Schedule Analysis
        # Replicate SoC untile optimization check if untiled in SRAM fits
        tiled = (d.tile_h > 0 and d.tile_w > 0 and (d.tile_num_h * d.tile_num_w) > 1)
        num_tiles = (d.tile_num_h * d.tile_num_w) if tiled else 1
        db_en = bool(d.sched_ctrl & SCHED_CTRL_DB_EN)

        sc = d.sched_ctrl
        fuse_flags = []
        if i in soc_super_map:
            s_lo, s_hi = soc_super_map[i]
            fuse_flags.append(f"SUPER({s_lo}-{s_hi})")
        elif sc & SCHED_CTRL_SUPER_BLOCK:
            fuse_flags.append("SUPER")
        elif i in super_tags and super_tags[i] == "SUPER_ELIGIBLE":
            fuse_flags.append("SUPER_CAND")
        if sc & SCHED_CTRL_FUSE_STRIP:
            fuse_flags.append("STRIP")
        if sc & SCHED_CTRL_FUSE_START:
            fuse_flags.append("FUSE_S")
        if sc & SCHED_CTRL_FUSE_MID:
            fuse_flags.append("FUSE_M")
        if sc & SCHED_CTRL_FUSE_END:
            fuse_flags.append("FUSE_E")
        fuse_tag = "+".join(fuse_flags) if fuse_flags else "-"

        in_from_sram = bool(sc & (SCHED_CTRL_FUSE_MID | SCHED_CTRL_FUSE_END))
        out_kept_in_sram = bool(sc & (SCHED_CTRL_FUSE_START | SCHED_CTRL_FUSE_MID))
        if i in soc_super_map:
            s_lo, s_hi = soc_super_map[i]
            if i > s_lo:
                in_from_sram = True
            if i < s_hi:
                out_kept_in_sram = True
        if soc_log and i in soc_log.layers and soc_log.layers[i].sram_fused_skip_store:
            out_kept_in_sram = True

        # Tile input/output footprints in SRAM
        if tiled:
            kh_eff = (d.kernel_h - 1) * d.dilation_h + 1
            kw_eff = (d.kernel_w - 1) * d.dilation_w + 1
            if d.op_type == OP_POOLING:
                kh_eff, kw_eff = d.pool_h, d.pool_w
                sh, sw = d.pool_stride_h, d.pool_stride_w
            else:
                sh, sw = d.stride_h, d.stride_w
            tin_h = min(d.in_h, (d.tile_h - 1) * sh + kh_eff)
            tin_w = min(d.in_w, (d.tile_w - 1) * sw + kw_eff)
            calc_tile_in = in_mult * tin_h * tin_w * d.in_c * eb
            tile_in_b = d.tile_in_size if d.tile_in_size > 0 else calc_tile_in
            tile_out_b = d.tile_out_size if d.tile_out_size > 0 else (d.tile_h * d.tile_w * d.out_c * eb)
        else:
            # Global DW / Global Pool stream slices rather than buffering whole HxW
            if (d.op_type in (OP_DW_CONV, OP_POOLING)
                    and d.out_h == 1 and d.out_w == 1
                    and d.kernel_h == d.in_h and d.kernel_w == d.in_w):
                tile_in_b = min(raw_in, hw.act_bank_size)
            else:
                tile_in_b = raw_in
            tile_out_b = raw_out

        # Weight OC groups & reload check
        if d.op_type in (OP_CONV2D, OP_FC):
            oc_groups = math.ceil(d.out_c / hw.mac_lanes)
            lane_occ_pct = (d.out_c / (oc_groups * hw.mac_lanes)) * 100.0 if oc_groups else 100.0
            wgt_per_oc = d.in_c * d.kernel_h * d.kernel_w * eb
            if raw_wgt <= hw.weight_buf_size:
                wgt_used_b = raw_wgt
                wgt_reloads = 1
            else:
                # Reload per OC group; if spatially tiled AND wgt > 80KB, reloads per spatial tile!
                grp_ch = hw.mac_lanes
                while grp_ch > 1 and grp_ch * wgt_per_oc > hw.weight_buf_size:
                    grp_ch //= 2
                wgt_used_b = min(hw.weight_buf_size, grp_ch * wgt_per_oc)
                wgt_reloads = num_tiles if tiled else 1
        elif d.op_type == OP_DW_CONV:
            oc_groups = math.ceil(d.out_c / hw.dw_parallel_ch)
            lane_occ_pct = (d.out_c / (oc_groups * hw.dw_parallel_ch)) * 100.0 if oc_groups else 100.0
            wgt_used_b = min(raw_wgt, hw.weight_buf_size)
            wgt_reloads = 1
        else:
            oc_groups = math.ceil(d.out_c / PPU_LANES)
            lane_occ_pct = (d.out_c / (oc_groups * PPU_LANES)) * 100.0 if oc_groups else 100.0
            wgt_used_b = 0
            wgt_reloads = 0

        ifm_used_b = tile_in_b * (2 if (db_en and tiled and d.op_type != OP_ELTWISE_ADD) else 1)
        ofm_used_b = tile_out_b * (2 if (db_en and tiled) else 1)
        if d.op_type == OP_CONCAT:
            ifm_used_b = min(hw.act_bank_size, tile_in_b + tile_out_b)
        prm_used_b = min(hw.param_sram_size, raw_param)

        # Actual external DDR traffic
        if in_from_sram:
            ext_in = 0
        elif d.dma_in_size > 0:
            ext_in = d.dma_in_size
        else:
            ext_in = (tile_in_b * num_tiles) if tiled else raw_in

        if out_kept_in_sram:
            ext_out = 0
        elif d.dma_out_size > 0:
            ext_out = d.dma_out_size
        else:
            ext_out = raw_out

        ext_wgt = raw_wgt * max(1, wgt_reloads) if raw_wgt > 0 else 0
        ext_param = raw_param
        total_ext = ext_in + ext_wgt + ext_out + ext_param

        halo_extra = max(0, ext_in - raw_in) if not in_from_sram else 0
        wgt_reload_extra = max(0, ext_wgt - raw_wgt)
        fusion_saved = (raw_in if in_from_sram else 0) + (raw_out if out_kept_in_sram else 0)

        # 4. Analytical Cycle Model (Micro-Sequencer + Roofline)
        if d.op_type in (OP_CONV2D, OP_FC):
            k_depth = d.kernel_h * d.kernel_w * d.in_c
            ideal_mac_cyc = out_px * oc_groups * k_depth
            # 8-lane PPU drains up to 64 lanes in 8 cycles per output pixel/OC-group
            ppu_cyc_per_px = sum(
                math.ceil(min(hw.mac_lanes, d.out_c - g * hw.mac_lanes) / PPU_LANES)
                for g in range(oc_groups)
            )
            ideal_ppu_cyc = out_px * ppu_cyc_per_px
            # Row engine sequences K-loop then WB per pixel/group
            ideal_comp_cyc = ideal_mac_cyc + ideal_ppu_cyc
        elif d.op_type == OP_DW_CONV:
            k_area = d.kernel_h * d.kernel_w
            ideal_mac_cyc = out_px * oc_groups * k_area
            ideal_ppu_cyc = out_px * oc_groups  # 8 DW lanes match 8 PPU lanes (1 cyc WB)
            ideal_comp_cyc = ideal_mac_cyc + ideal_ppu_cyc
        elif d.op_type == OP_POOLING:
            k_area = (d.in_h * d.in_w) if d.global_pool else (d.pool_h * d.pool_w)
            ideal_mac_cyc = 0
            ideal_ppu_cyc = 0
            div_init = 40 if d.pool_mode == 1 else 0
            ideal_comp_cyc = div_init + math.ceil((out_px * d.out_c * max(1, k_area)) / PPU_LANES)
        else:
            ideal_mac_cyc = 0
            ideal_ppu_cyc = math.ceil((out_px * d.out_c) / PPU_LANES)
            ideal_comp_cyc = ideal_ppu_cyc

        ideal_dma_cyc = math.ceil(total_ext / hw.ext_bw_bytes) if hw.ext_bw_bytes > 0 else 0
        # With DB_EN or internal overlap, part of DMA overlaps compute; single-buffer serializes
        if db_en and tiled:
            est_cyc = max(ideal_comp_cyc, ideal_dma_cyc) + math.ceil(0.15 * min(ideal_comp_cyc, ideal_dma_cyc))
        else:
            # Untiled layer loads IFM+WGT, computes, then stores OFM
            est_cyc = ideal_comp_cyc + ideal_dma_cyc

        ai = (macs / total_ext) if total_ext > 0 else float("inf")

        # Classify bottleneck regime
        pmu_rec = soc_log.layers.get(i) if soc_log else None
        has_full_soc_pmu = bool(soc_log and len(soc_log.layers) == len(layers))
        if pmu_rec is not None and (pmu_rec.npu_cycles > 0 or has_full_soc_pmu):
            hw_cyc = pmu_rec.npu_cycles
        else:
            hw_cyc = None

        if hw_cyc == 0 and has_full_soc_pmu:
            bound = "FUSED-SUBLAYER"
            if fuse_tag == "-":
                fuse_tag = "HW_FUSED"
        elif in_from_sram and out_kept_in_sram:
            bound = "SRAM-FUSED"
        elif ideal_dma_cyc > ideal_comp_cyc * 1.05:
            bound = "BW-BOUND"
        elif d.op_type in (OP_CONV2D, OP_FC) and ideal_ppu_cyc >= ideal_mac_cyc * 0.35:
            bound = "PPU/WB-HEAVY"
        elif d.op_type == OP_DW_CONV:
            bound = "DW-8LANE-BOUND"
        elif d.op_type in (OP_POOLING, OP_ELTWISE_ADD, OP_RESIZE, OP_CONCAT):
            bound = "VEC-8LANE-BOUND"
        else:
            bound = "COMPUTE-BOUND"

        if pmu_rec and pmu_rec.npu_cycles > 0:
            mem_stall = pmu_rec.wt_act + pmu_rec.wt_wgt + pmu_rec.wt_store + pmu_rec.wt_db
            if mem_stall > pmu_rec.npu_cycles * 0.40:
                bound = "BW-BOUND (PMU)"
            elif pmu_rec.st_wb > pmu_rec.st_mac * 0.40 and pmu_rec.st_mac > 0:
                bound = "PPU/WB-HEAVY (PMU)"
            elif pmu_rec.st_mac > pmu_rec.npu_cycles * 0.65:
                bound = "COMPUTE-BOUND (PMU)"

        shape_str = f"{d.in_h}x{d.in_w}x{d.in_c} -> {d.out_h}x{d.out_w}x{d.out_c}"
        profiles.append(LayerProfile(
            idx=i,
            op_name=op_name,
            op_family=op_family,
            shape_str=shape_str,
            elem_bytes=eb,
            macs=macs,
            ops=ops,
            raw_in_bytes=raw_in,
            raw_wgt_bytes=raw_wgt,
            raw_out_bytes=raw_out,
            raw_param_bytes=raw_param,
            ext_in_bytes=ext_in,
            ext_wgt_bytes=ext_wgt,
            ext_out_bytes=ext_out,
            ext_param_bytes=ext_param,
            total_ext_bytes=total_ext,
            halo_extra_bytes=halo_extra,
            wgt_reload_extra_bytes=wgt_reload_extra,
            fusion_saved_bytes=fusion_saved,
            ifm_used_bytes=ifm_used_b,
            ofm_used_bytes=ofm_used_b,
            wgt_used_bytes=wgt_used_b,
            prm_used_bytes=prm_used_b,
            ifm_occ_pct=min(100.0, (ifm_used_b / hw.act_bank_size) * 100.0),
            ofm_occ_pct=min(100.0, (ofm_used_b / hw.ofm_bank_size) * 100.0),
            wgt_occ_pct=min(100.0, (wgt_used_b / hw.weight_buf_size) * 100.0),
            prm_occ_pct=min(100.0, (prm_used_b / hw.param_sram_size) * 100.0),
            tiled=tiled,
            num_tiles=num_tiles,
            oc_groups=oc_groups,
            db_en=db_en,
            fuse_tag=fuse_tag,
            lane_occ_pct=lane_occ_pct,
            arith_intensity=ai,
            ideal_mac_cycles=ideal_mac_cyc,
            ideal_ppu_cycles=ideal_ppu_cyc,
            ideal_comp_cycles=ideal_comp_cyc,
            ideal_dma_cycles=ideal_dma_cyc,
            est_cycles=est_cyc,
            bound_regime=bound,
            hw_cycles=hw_cyc,
            hw_dma_cycles=pmu_rec.dma_cycles if pmu_rec else None,
            hw_comp_cycles=pmu_rec.comp_cycles if pmu_rec else None,
            hw_ovlp_cycles=pmu_rec.ovlp_cycles if pmu_rec else None,
            pmu=pmu_rec,
        ))
    return profiles


# ─────────────────────────────────────────────────────────────────────
#  Nsight-Style Optimization Advisor
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    severity: str       # HIGH / MEDIUM / LOW
    category: str       # SRAM-TILING / FUSION / COMPUTE-PIPE / BUS-BW
    title: str
    detail: str
    est_cycle_saving: int


def generate_optimization_findings(
    layers: List[LayerDesc],
    profiles: List[LayerProfile],
    hw: HWConfig,
) -> List[Finding]:
    findings: List[Finding] = []
    total_cyc = sum((p.hw_cycles if p.hw_cycles is not None else p.est_cycles) for p in profiles)
    if total_cyc <= 0:
        return findings

    # 1. Tile Halo Amplification Check
    halo_layers = [p for p in profiles if p.halo_extra_bytes > 16384 and p.raw_in_bytes > 0
                   and (p.halo_extra_bytes / p.raw_in_bytes) > 0.15]
    if halo_layers:
        tot_halo_b = sum(p.halo_extra_bytes for p in halo_layers)
        saved_cyc = tot_halo_b // max(1, hw.ext_bw_bytes)
        top_ids = ", ".join(
            f"L{p.idx}({p.op_family}, +{p.halo_extra_bytes / 1024:.1f}KB / "
            f"+{100.0 * p.halo_extra_bytes / p.raw_in_bytes:.0f}%)"
            for p in sorted(halo_layers, key=lambda x: x.halo_extra_bytes, reverse=True)[:4]
        )
        findings.append(Finding(
            severity="HIGH" if saved_cyc > total_cyc * 0.04 else "MEDIUM",
            category="SRAM-TILING",
            title=f"Spatial Tiling Halo Re-Read Overhead ({tot_halo_b / 1024:.1f} KB across {len(halo_layers)} layers)",
            detail=(f"2D square/small tiles with DB_EN halve IFM to {hw.act_bank_size // 2048}KB, "
                    f"causing halo overlap re-fetches on {top_ids}. Full-width strip tiling "
                    f"(tile_w = in_w) eliminates horizontal halo and turns 2D strided DMA into "
                    f"long 1D contiguous bursts."),
            est_cycle_saving=saved_cyc,
        ))

    # 2. Spatially Tiled + Oversized Weight Reload Check
    wgt_reload_layers = [p for p in profiles if p.wgt_reload_extra_bytes > 0]
    if wgt_reload_layers:
        tot_wgt_extra = sum(p.wgt_reload_extra_bytes for p in wgt_reload_layers)
        saved_cyc = tot_wgt_extra // max(1, hw.ext_bw_bytes)
        top_ids = ", ".join(
            f"L{p.idx}({p.num_tiles} tiles, +{p.wgt_reload_extra_bytes / 1024:.0f}KB)"
            for p in sorted(wgt_reload_layers, key=lambda x: x.wgt_reload_extra_bytes, reverse=True)[:4]
        )
        findings.append(Finding(
            severity="HIGH" if saved_cyc > total_cyc * 0.05 else "MEDIUM",
            category="SRAM-CACHE",
            title=f"Weight Reload Across Spatial Tiles ({tot_wgt_extra / 1024:.1f} KB extra DDR reads)",
            detail=(f"Layers {top_ids} exceed {hw.weight_buf_size // 1024}KB WGT SRAM while also "
                    f"spatially tiled. Reordering loop to OC-group-outer / spatial-tile-inner "
                    f"loads each weight group once instead of {wgt_reload_layers[0].num_tiles}x."),
            est_cycle_saving=saved_cyc,
        ))

    # 3. Unfused SRAM-Resident Chain / Super-Block Opportunities
    unfused_pairs = []
    unfused_bytes = 0
    for i in range(len(layers) - 1):
        d0, d1 = layers[i], layers[i + 1]
        p0, p1 = profiles[i], profiles[i + 1]
        if p0.ext_out_bytes == 0 or p1.ext_in_bytes == 0:
            continue
        if d1.input_src not in (-1, i) or d1.op_type in (OP_ELTWISE_ADD, OP_CONCAT):
            continue
        # Check if intermediate activation fits cleanly in IFM (32KB) and OFM (48KB)
        if p0.raw_out_bytes <= hw.act_bank_size and p1.raw_out_bytes <= hw.ofm_bank_size:
            roundtrip = p0.raw_out_bytes + p1.raw_in_bytes
            unfused_pairs.append((i, i + 1, roundtrip))
            unfused_bytes += roundtrip
    if unfused_pairs:
        saved_cyc = unfused_bytes // max(1, hw.ext_bw_bytes)
        pair_str = ", ".join(
            f"L{a}->L{b}({b_sz / 1024:.1f}KB)"
            for a, b, b_sz in sorted(unfused_pairs, key=lambda x: x[2], reverse=True)[:5]
        )
        findings.append(Finding(
            severity="HIGH" if saved_cyc > total_cyc * 0.05 else "MEDIUM",
            category="FUSION",
            title=f"Unfused On-Chip SRAM-Resident Transitions ({len(unfused_pairs)} pairs, {unfused_bytes / 1024:.1f} KB DDR round-trip)",
            detail=(f"Adjacent layers {pair_str} fit wholly inside 32KB IFM / 48KB OFM SRAM "
                    f"but still flush OFM to DDR and re-read IFM. Enabling SRAM ping-pong "
                    f"chaining (FUSE_START/MID/END) or widening super-block eligibility removes "
                    f"both Store and Load DMA phases."),
            est_cycle_saving=saved_cyc,
        ))

    # 4. PPU 8-Lane Writeback Serialization on Shallow-K Pointwise Conv1x1
    wb_heavy = []
    wb_extra_cyc = 0
    for d, p in zip(layers, profiles):
        if d.op_type == OP_CONV2D and d.kernel_h == 1 and d.kernel_w == 1:
            wb_cyc = p.pmu.st_wb if (p.pmu and p.pmu.st_wb > 0) else p.ideal_ppu_cycles
            mac_cyc = p.pmu.st_mac if (p.pmu and p.pmu.st_mac > 0) else p.ideal_mac_cycles
            if wb_cyc > 0 and mac_cyc > 0 and wb_cyc >= 0.25 * mac_cyc:
                wb_heavy.append((p.idx, d.in_c, d.out_c, mac_cyc, wb_cyc))
                # Overlapping PPU drain with next pixel's K-loop saves ~85% of ST_WB
                wb_extra_cyc += int(wb_cyc * 0.85)
    if wb_heavy:
        ex_str = ", ".join(
            f"L{idx}(IC={ic}->OC={oc}, WB={wb / 1e3:.1f}K vs MAC={mc / 1e3:.1f}K)"
            for idx, ic, oc, mc, wb in sorted(wb_heavy, key=lambda x: x[4], reverse=True)[:4]
        )
        findings.append(Finding(
            severity="HIGH" if wb_extra_cyc > total_cyc * 0.06 else "MEDIUM",
            category="COMPUTE-PIPE",
            title=f"PPU 8-Lane Drain Serialized After 64-Lane Row MAC ({wb_extra_cyc:,} cycles across {len(wb_heavy)} PW1x1 layers)",
            detail=(f"On shallow-IC 1x1 convolutions ({ex_str}), the 64-lane MAC row finishes "
                    f"in IC cycles ({wb_heavy[0][1]} cyc) and then waits 8+ cycles for the "
                    f"8-lane PPU to drain 64 accumulators (ST_WB). Shadowing `racc` (ping-pong "
                    f"accumulator latch) or widening PPU from 8 to 16 lanes overlaps PPU drain "
                    f"with the next pixel's MAC loop."),
            est_cycle_saving=wb_extra_cyc,
        ))

    # 5. Dense 64-Lane OC Under-Occupancy (Small Channel Count Layers)
    low_occ = [p for p in profiles if p.op_name in ("Conv2D", "FC") and p.lane_occ_pct < 75.0]
    if low_occ:
        wasted_cyc = sum(
            int(((p.pmu.st_mac if p.pmu and p.pmu.st_mac > 0 else p.ideal_mac_cycles)
                 * (1.0 - p.lane_occ_pct / 100.0)) * 0.5)
            for p in low_occ
        )
        ex_str = ", ".join(
            f"L{p.idx}({p.shape_str.split('->')[1].strip()}, occ={p.lane_occ_pct:.0f}%)"
            for p in low_occ[:4]
        )
        findings.append(Finding(
            severity="MEDIUM" if wasted_cyc > total_cyc * 0.04 else "LOW",
            category="COMPUTE-PACKING",
            title=f"Sub-64 Output Channel Under-Occupancy ({len(low_occ)} layers < 75% lane fill)",
            detail=(f"Early stem/narrow layers ({ex_str}) have OC < 64 (e.g. 16, 24, 32), leaving "
                    f"half of the 64 MAC lanes idle. Spatial pixel-packing (2 pixels x 32 OC or "
                    f"4 pixels x 16 OC in the 64-lane row) doubles throughput on stem layers."),
            est_cycle_saving=wasted_cyc,
        ))

    # 6. External DMA Bus Width Check
    if hw.dma_bus_width < 128:
        cur_dma = sum(math.ceil(p.total_ext_bytes / hw.ext_bw_bytes) for p in profiles)
        w128_dma = sum(math.ceil(p.total_ext_bytes / 16) for p in profiles)
        saved_bw = max(0, cur_dma - w128_dma)
        if saved_bw > total_cyc * 0.05:
            findings.append(Finding(
                severity="HIGH",
                category="BUS-BW",
                title=f"External DMA Bus Width Bottleneck ({hw.dma_bus_width}-bit -> 128-bit saves ~{saved_bw:,} DMA cycles)",
                detail=(f"Current profile uses {hw.dma_bus_width}-bit DMA ({hw.ext_bw_bytes} B/cyc, "
                        f"ridge={hw.mac_lanes / hw.ext_bw_bytes:.1f} MAC/B). Enabling `-DDMA_BUS_WIDTH=128` "
                        f"(16 B/cyc, ridge=4.0 MAC/B) cuts external transfer cycles by "
                        f"{16 // hw.ext_bw_bytes}x with <1.9% logic area overhead."),
                est_cycle_saving=saved_bw,
            ))

    findings.sort(key=lambda f: f.est_cycle_saving, reverse=True)
    return findings


# ─────────────────────────────────────────────────────────────────────
#  Formatting & Reporting
# ─────────────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def print_report(
    model_label: str,
    layers: List[LayerDesc],
    profiles: List[LayerProfile],
    hw: HWConfig,
    freq_mhz: float,
    top_n: int,
    soc_log: Optional[SoCLogSummary] = None,
) -> None:
    has_hw = any(p.hw_cycles is not None for p in profiles)
    cyc_fn = lambda p: (p.hw_cycles if p.hw_cycles is not None else p.est_cycles)

    total_cycles = sum(cyc_fn(p) for p in profiles)
    wall_cycles = soc_log.wall_cycles if (soc_log and soc_log.wall_cycles) else total_cycles
    total_macs = sum(p.macs for p in profiles)
    total_ops = sum(p.ops for p in profiles)
    total_ext_bytes = sum(p.total_ext_bytes for p in profiles)
    total_raw_bytes = sum(p.raw_in_bytes + p.raw_wgt_bytes + p.raw_out_bytes + p.raw_param_bytes for p in profiles)
    total_halo = sum(p.halo_extra_bytes for p in profiles)
    total_wgt_reload = sum(p.wgt_reload_extra_bytes for p in profiles)
    total_fuse_saved = sum(p.fusion_saved_bytes for p in profiles)

    # Ideal architectural compute floor (64-lane dense + 8-lane DW + 8-lane Vec)
    arch_ideal_comp = sum(p.ideal_comp_cycles for p in profiles)
    dense_ideal_cyc = math.ceil(total_macs / hw.mac_lanes) if total_macs > 0 else 1

    peak_gmacs = (hw.mac_lanes * freq_mhz) / 1000.0
    peak_bw_gbs = (hw.ext_bw_bytes * freq_mhz) / 1000.0
    latency_ms = total_cycles / (freq_mhz * 1000.0) if freq_mhz > 0 else 0.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    achieved_gmacs = (total_macs / max(1, total_cycles)) * (freq_mhz / 1000.0)
    achieved_bw_gbs = (total_ext_bytes / max(1, total_cycles)) * (freq_mhz / 1000.0)

    compute_sol_pct = (dense_ideal_cyc / max(1, total_cycles)) * 100.0
    arch_sol_pct = min(100.0, (arch_ideal_comp / max(1, total_cycles)) * 100.0)
    dma_ideal_cyc = math.ceil(total_ext_bytes / hw.ext_bw_bytes)
    bw_sol_pct = min(100.0, (dma_ideal_cyc / max(1, total_cycles)) * 100.0)
    ridge_mac_per_b = hw.mac_lanes / hw.ext_bw_bytes
    overall_ai = total_macs / max(1, total_ext_bytes)

    mode_str = "RTL/SoC Hardware PMU + Analytical" if has_hw else "Offline Analytical + Roofline"

    print("=" * 94)
    print(f"  OPEN-NPU NSIGHT-STYLE PROFILER  |  Model: {model_label}  |  Mode: {mode_str}")
    print("=" * 94)
    print(f"  HW Config : {hw.mac_lanes} Dense MACs/cyc | {hw.dw_parallel_ch} DW lanes | "
          f"{PPU_LANES} PPU lanes | {hw.spad_total_kb} KB 4-Bank SRAM "
          f"(IFM {hw.act_bank_size // 1024}K / OFM {hw.ofm_bank_size // 1024}K / "
          f"WGT {hw.weight_buf_size // 1024}K / PRM {hw.param_sram_size // 1024}K)")
    print(f"  Bus & Clk : {hw.dma_bus_width}-bit {hw.dma_bus_proto.upper()} ({hw.ext_bw_bytes} B/cyc) "
          f"@ {freq_mhz:.0f} MHz  |  Peak: {peak_gmacs:.2f} GMAC/s ({peak_gmacs * 2:.2f} GOPS), "
          f"{peak_bw_gbs:.2f} GB/s  |  HW_CONFIG=0x{hw.hw_config_reg:08X}")
    print("-" * 94)

    # ── Section 1: Speed-of-Light (SOL) ──
    print("\n[1] SPEED-OF-LIGHT (SOL) & ROOFLINE SUMMARY")
    print(f"  Layers / Workload       : {len(profiles)} layers  |  {total_macs / 1e6:.2f} M MACs  "
          f"({total_ops / 1e6:.2f} M Ops)")
    print(f"  NPU Active Cycles       : {total_cycles:,} cycles  ->  {latency_ms:.2f} ms / frame  "
          f"({fps:.1f} FPS @ {freq_mhz:.0f} MHz)")
    if wall_cycles > total_cycles:
        cpu_gap = wall_cycles - total_cycles
        print(f"  End-to-End Wall Cycles  : {wall_cycles:,} cycles  "
              f"(CPU/Firmware dispatch gap: {cpu_gap:,} cyc = {100.0 * cpu_gap / wall_cycles:.1f}%)")
    print(f"  Compute Throughput      : {achieved_gmacs:.2f} / {peak_gmacs:.2f} GMAC/s  "
          f"|  Dense 64-MAC SOL: {compute_sol_pct:5.1f}%  |  Arch-Pipe SOL: {arch_sol_pct:5.1f}%")
    print(f"  External Memory Traffic : {_fmt_bytes(total_ext_bytes)}  "
          f"(Sustained: {achieved_bw_gbs:.2f} / {peak_bw_gbs:.2f} GB/s  |  DMA Bus SOL: {bw_sol_pct:5.1f}%)")
    print(f"  Roofline Intensity      : {overall_ai:.2f} MAC/Byte  vs  "
          f"Dense Ridge Point = {ridge_mac_per_b:.1f} MAC/Byte "
          f"({'COMPUTE-SIDE' if overall_ai >= ridge_mac_per_b else 'MEMORY-BW-SIDE'})")

    # Bus scaling quick comparison (analytical)
    w_est = {}
    for bits in (32, 64, 128):
        bw = bits // 8
        c_sum = 0
        for p in profiles:
            d_cyc = math.ceil(p.total_ext_bytes / bw)
            if p.db_en and p.tiled:
                c_sum += max(p.ideal_comp_cycles, d_cyc) + math.ceil(0.15 * min(p.ideal_comp_cycles, d_cyc))
            else:
                c_sum += p.ideal_comp_cycles + d_cyc
        w_est[bits] = c_sum
    print(f"  Analytical Bus Scaling  : 32b={w_est[32]:,} cyc (1.00x)  ->  "
          f"64b={w_est[64]:,} cyc ({w_est[32] / max(1, w_est[64]):.2f}x)  ->  "
          f"128b={w_est[128]:,} cyc ({w_est[32] / max(1, w_est[128]):.2f}x)")

    # ── Section 2: Operator Family Breakdown ──
    print("\n[2] OPERATOR FAMILY BREAKDOWN")
    fam_map: Dict[str, List[LayerProfile]] = {}
    for p in profiles:
        fam_map.setdefault(p.op_family, []).append(p)
    hdr = (f"  {'Family':<12} {'Cnt':>4} {'Cycles':>12} {'Cyc%':>6} "
           f"{'MACs(M)':>9} {'DDR Traffic':>11} {'AI(M/B)':>8} {'DenseSOL%':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for fam, plist in sorted(fam_map.items(), key=lambda kv: sum(cyc_fn(x) for x in kv[1]), reverse=True):
        fc = sum(cyc_fn(x) for x in plist)
        fm = sum(x.macs for x in plist)
        fb = sum(x.total_ext_bytes for x in plist)
        fai = fm / max(1, fb)
        fsol = ((fm / hw.mac_lanes) / max(1, fc)) * 100.0 if fm > 0 else 0.0
        print(f"  {fam:<12} {len(plist):>4} {fc:>12,} {100.0 * fc / max(1, total_cycles):>5.1f}% "
              f"{fm / 1e6:>9.2f} {_fmt_bytes(fb):>11} {fai:>8.2f} {fsol:>8.1f}%")

    # ── Section 3: 4-Bank On-Chip SRAM ("Cache") & Traffic Audit ──
    print("\n[3] ON-CHIP 4-BANK SRAM ('CACHE') OCCUPANCY & DATAFLOW EFFICIENCY")
    for bname, cap, occ_fn, used_fn in [
        ("IFM Bank", hw.act_bank_size, lambda p: p.ifm_occ_pct, lambda p: p.ifm_used_bytes),
        ("OFM Bank", hw.ofm_bank_size, lambda p: p.ofm_occ_pct, lambda p: p.ofm_used_bytes),
        ("WGT Bank", hw.weight_buf_size, lambda p: p.wgt_occ_pct, lambda p: p.wgt_used_bytes),
        ("PRM Bank", hw.param_sram_size, lambda p: p.prm_occ_pct, lambda p: p.prm_used_bytes),
    ]:
        active = [p for p in profiles if used_fn(p) > 0]
        peak_u = max((used_fn(p) for p in active), default=0)
        avg_occ = (sum(occ_fn(p) for p in active) / len(active)) if active else 0.0
        full_cnt = sum(1 for p in active if occ_fn(p) >= 85.0)
        print(f"  {bname:<9} ({cap // 1024:>2} KB) : Peak = {_fmt_bytes(peak_u):>8} "
              f"({min(100.0, 100.0 * peak_u / cap):5.1f}%)  |  Avg Active Occupancy = {avg_occ:5.1f}%  "
              f"|  High-Pressure Layers (>=85%): {full_cnt}/{len(active)}")

    tiled_cnt = sum(1 for p in profiles if p.tiled)
    db_cnt = sum(1 for p in profiles if p.db_en)
    fused_cnt = sum(1 for p in profiles if p.fuse_tag != "-")
    print(f"  Schedule Coverage  : Spatially Tiled = {tiled_cnt}/{len(profiles)}  |  "
          f"DB Ping-Pong = {db_cnt}/{len(profiles)}  |  Fused/Super = {fused_cnt}/{len(profiles)}")
    print(f"  DDR Amplification  : Raw Tensors = {_fmt_bytes(total_raw_bytes)}  |  "
          f"Tile Halo Extra = +{_fmt_bytes(total_halo)}  |  "
          f"WGT Reload Extra = +{_fmt_bytes(total_wgt_reload)}  |  "
          f"On-Chip Fusion Saved = -{_fmt_bytes(total_fuse_saved)}")

    # ── Section 4: Hardware PMU Stall Attribution (if SoC log provided) ──
    pmu_layers = [p.pmu for p in profiles if p.pmu and p.pmu.npu_cycles > 0]
    if pmu_layers:
        print("\n[4] HARDWARE PMU PIPELINE STATE & STALL BREAKDOWN (RTL CYCLE-EXACT)")
        tot_npu = sum(m.npu_cycles for m in pmu_layers)
        st_mac = sum(m.st_mac for m in pmu_layers)
        st_wb = sum(m.st_wb for m in pmu_layers)
        st_vec = sum(m.st_pool + m.st_add + m.st_rsz for m in pmu_layers)
        st_wait = sum(m.st_wait for m in pmu_layers)
        st_cfg_oth = sum(m.st_cfg + m.st_oth for m in pmu_layers)
        wt_act = sum(m.wt_act for m in pmu_layers)
        wt_wgt = sum(m.wt_wgt for m in pmu_layers)
        wt_store = sum(m.wt_store for m in pmu_layers)
        wt_db = sum(m.wt_db for m in pmu_layers)
        tot_dma = sum(m.dma_cycles for m in pmu_layers)
        tot_ovlp = sum(m.ovlp_cycles for m in pmu_layers)

        print(f"  Active MAC Compute (ST_MAC)       : {st_mac:>11,} cyc ({100.0 * st_mac / tot_npu:5.1f}%)")
        print(f"  PPU Writeback Drain (ST_WB)       : {st_wb:>11,} cyc ({100.0 * st_wb / tot_npu:5.1f}%)")
        print(f"  Pool / Add / Resize (ST_VEC)      : {st_vec:>11,} cyc ({100.0 * st_vec / tot_npu:5.1f}%)")
        print(f"  Memory / DMA Wait States (ST_WAIT): {st_wait:>11,} cyc ({100.0 * st_wait / tot_npu:5.1f}%)")
        print(f"    ├─ Wait IFM Load  (WT_ACT)      : {wt_act:>11,} cyc ({100.0 * wt_act / tot_npu:5.1f}%)")
        print(f"    ├─ Wait WGT Load  (WT_WGT)      : {wt_wgt:>11,} cyc ({100.0 * wt_wgt / tot_npu:5.1f}%)")
        print(f"    ├─ Wait OFM Store (WT_STORE)    : {wt_store:>11,} cyc ({100.0 * wt_store / tot_npu:5.1f}%)")
        print(f"    └─ Wait DB Sync   (WT_DB)       : {wt_db:>11,} cyc ({100.0 * wt_db / tot_npu:5.1f}%)")
        print(f"  Control / Setup / Other (CFG+OTH) : {st_cfg_oth:>11,} cyc ({100.0 * st_cfg_oth / tot_npu:5.1f}%)")
        if tot_dma > 0:
            print(f"  DMA BusyCycles / Compute Overlap  : DMA={tot_dma:,} cyc  |  "
                  f"Overlap={tot_ovlp:,} cyc ({100.0 * tot_ovlp / tot_dma:4.1f}% of DMA hidden behind compute)")

    # Precompute fused-group MACs for tail layers whose HW PMU aggregates preceding 0-cycle sublayers
    group_macs: Dict[int, int] = {p.idx: p.macs for p in profiles}
    group_label_map: Dict[int, str] = {}
    if has_hw:
        pending_macs = 0
        pending_start: Optional[int] = None
        for p in profiles:
            if p.hw_cycles == 0:
                if pending_start is None:
                    pending_start = p.idx
                pending_macs += p.macs
            else:
                if pending_start is not None:
                    group_macs[p.idx] = pending_macs + p.macs
                    group_label_map[p.idx] = f"L{pending_start}..L{p.idx}"
                    pending_start = None
                    pending_macs = 0

    # ── Section 5: Top Hotspot Layers ──
    sec_num = 5 if pmu_layers else 4
    print(f"\n[{sec_num}] TOP {min(top_n, len(profiles))} HOTSPOT LAYERS / SUPER-BLOCKS (BY CYCLES)")
    thdr = (f"  {'ID':<9} {'Op':<10} {'Shape (In -> Out)':<27} {'Cycles':>10} {'Cyc%':>5} "
            f"{'Dense%':>6} {'DDR':>8} {'Tiles':>5} {'Fuse':<10} {'Regime'}")
    print(thdr)
    print("  " + "-" * (len(thdr) - 2))
    ranked = sorted(profiles, key=cyc_fn, reverse=True)[:top_n]
    for p in ranked:
        c = cyc_fn(p)
        if c == 0:
            continue
        eff_macs = group_macs.get(p.idx, p.macs)
        lid = group_label_map.get(p.idx, f"L{p.idx}")
        dsol = ((eff_macs / hw.mac_lanes) / c) * 100.0 if eff_macs > 0 else 0.0
        ftag = "SUPER/FUSE" if p.idx in group_label_map and p.fuse_tag == "-" else p.fuse_tag
        print(f"  {lid:<9} {p.op_family:<10} {p.shape_str:<27} {c:>10,} "
              f"{100.0 * c / max(1, total_cycles):>4.1f}% {dsol:>5.1f}% "
              f"{_fmt_bytes(p.total_ext_bytes):>8} {p.num_tiles:>5} {ftag:<10} {p.bound_regime}")

    # ── Section 6: Actionable Optimization Advisor ──
    findings = generate_optimization_findings(layers, profiles, hw)
    adv_num = sec_num + 1
    print(f"\n[{adv_num}] ACTIONABLE OPTIMIZATION ADVISOR (COMPUTE & SRAM CACHE)")
    if not findings:
        print("  No major bottlenecks detected — model is running near architectural roofline.")
    else:
        for idx, f in enumerate(findings, 1):
            pct = 100.0 * f.est_cycle_saving / max(1, total_cycles)
            print(f"  ({idx}) [{f.severity}] [{f.category}] {f.title}")
            print(f"      Est. Upper-Bound Saving: ~{f.est_cycle_saving:,} cycles (~{pct:.1f}% of NPU runtime)")
            print(f"      Insight: {f.detail}")
    print("=" * 94)


# ─────────────────────────────────────────────────────────────────────
#  Visual Dashboard Renderers (PNG Poster + Interactive HTML)
# ─────────────────────────────────────────────────────────────────────

def _load_fonts():
    from PIL import ImageFont
    bold_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    reg_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    mono_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]

    def pick(paths, size):
        for p in paths:
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
        return ImageFont.load_default()

    return {
        "h1": pick(bold_paths, 30),
        "h2": pick(bold_paths, 20),
        "h3": pick(bold_paths, 16),
        "kpi": pick(bold_paths, 28),
        "body": pick(reg_paths, 15),
        "small": pick(reg_paths, 13),
        "mono": pick(mono_paths, 13),
        "tiny": pick(reg_paths, 11),
    }


def render_png_dashboard(
    out_path: str,
    model_label: str,
    layers: List[LayerDesc],
    profiles: List[LayerProfile],
    hw: HWConfig,
    freq_mhz: float,
    soc_log: Optional[SoCLogSummary] = None,
) -> bool:
    """Render an NVIDIA Nsight-style 2200x1560 dark-theme PNG dashboard."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("WARNING: Pillow (PIL) not installed; skipping PNG generation.", file=sys.stderr)
        return False

    W, H = 2200, 1560
    BG = (13, 17, 23)
    CARD = (22, 27, 34)
    BORDER = (48, 54, 61)
    TXT = (230, 237, 243)
    MUTED = (139, 148, 158)
    NV_GREEN = (118, 185, 0)
    CYAN = (56, 189, 248)
    AMBER = (245, 158, 11)
    CORAL = (251, 113, 133)
    VIOLET = (167, 139, 250)
    TEAL = (45, 212, 191)

    img = Image.new("RGB", (W, H), BG)
    drw = ImageDraw.Draw(img)
    fnt = _load_fonts()

    has_hw = any(p.hw_cycles is not None for p in profiles)
    cyc_fn = lambda p: (p.hw_cycles if p.hw_cycles is not None else p.est_cycles)
    total_cycles = max(1, sum(cyc_fn(p) for p in profiles))
    total_macs = sum(p.macs for p in profiles)
    total_ext = sum(p.total_ext_bytes for p in profiles)
    total_raw = sum(p.raw_in_bytes + p.raw_wgt_bytes + p.raw_out_bytes + p.raw_param_bytes for p in profiles)
    total_halo = sum(p.halo_extra_bytes for p in profiles)
    total_wrel = sum(p.wgt_reload_extra_bytes for p in profiles)
    total_fsav = sum(p.fusion_saved_bytes for p in profiles)

    peak_gmacs = (hw.mac_lanes * freq_mhz) / 1000.0
    peak_bw = (hw.ext_bw_bytes * freq_mhz) / 1000.0
    ach_gmacs = (total_macs / total_cycles) * (freq_mhz / 1000.0)
    lat_ms = total_cycles / (freq_mhz * 1000.0)
    fps = 1000.0 / lat_ms if lat_ms > 0 else 0.0
    dense_sol = min(100.0, ((total_macs / hw.mac_lanes) / total_cycles) * 100.0)
    arch_sol = min(100.0, (sum(p.ideal_comp_cycles for p in profiles) / total_cycles) * 100.0)
    bw_sol = min(100.0, ((total_ext / hw.ext_bw_bytes) / total_cycles) * 100.0)

    def card(x0, y0, x1, y1, title):
        drw.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=CARD, outline=BORDER, width=2)
        drw.rectangle([x0 + 16, y0 + 16, x0 + 22, y0 + 38], fill=NV_GREEN)
        drw.text((x0 + 32, y0 + 15), title, font=fnt["h2"], fill=TXT)

    # ── Top Header Banner ──
    drw.rounded_rectangle([24, 20, W - 24, 132], radius=12, fill=(18, 24, 32), outline=BORDER, width=2)
    drw.rectangle([24, 20, 34, 132], fill=NV_GREEN)
    mode_tag = "RTL/SoC Hardware PMU + Roofline" if has_hw else "Offline Analytical + Roofline"
    drw.text((52, 34), f"OPEN-NPU NSIGHT PROFILER  —  {model_label}", font=fnt["h1"], fill=TXT)
    drw.text(
        (52, 76),
        f"Mode: {mode_tag}   |   HW: {hw.mac_lanes} Dense MACs, {hw.dw_parallel_ch} DW, {PPU_LANES} PPU   |   "
        f"SRAM: {hw.spad_total_kb} KB (32/48/80/12 KB)   |   Bus: {hw.dma_bus_width}-bit {hw.dma_bus_proto.upper()} @ {freq_mhz:.0f} MHz",
        font=fnt["body"], fill=MUTED,
    )

    kpis = [
        ("LATENCY / FPS", f"{lat_ms:.2f} ms  ({fps:.1f} FPS)", NV_GREEN),
        ("THROUGHPUT", f"{ach_gmacs:.2f} / {peak_gmacs:.1f} GMAC/s", CYAN),
        ("DENSE / ARCH SOL", f"{dense_sol:.1f}%  /  {arch_sol:.1f}%", AMBER),
        ("EXT DDR TRAFFIC", f"{_fmt_bytes(total_ext)}  ({total_macs / 1e6:.0f}M MAC)", VIOLET),
    ]
    kx = W - 1060
    for lbl, val, col in kpis:
        drw.rounded_rectangle([kx, 32, kx + 248, 120], radius=8, fill=(13, 17, 23), outline=BORDER, width=1)
        drw.text((kx + 14, 42), lbl, font=fnt["tiny"], fill=MUTED)
        drw.text((kx + 14, 68), val, font=fnt["h3"], fill=col)
        kx += 260

    # ── Panel 1: Speed-of-Light & Bus Width Scaling (Left Top) ──
    p1 = (24, 152, 1088, 590)
    card(*p1, "1. SPEED-OF-LIGHT (SOL%) & DMA BUS SCALING")
    pmu_layers = [p.pmu for p in profiles if p.pmu and p.pmu.npu_cycles > 0]
    ovlp_pct = 0.0
    if pmu_layers and sum(m.dma_cycles for m in pmu_layers) > 0:
        ovlp_pct = 100.0 * sum(m.ovlp_cycles for m in pmu_layers) / sum(m.dma_cycles for m in pmu_layers)
    else:
        ovlp_pct = 100.0 * sum(1 for p in profiles if p.db_en) / max(1, len(profiles))

    gauges = [
        ("Arch-Adjusted Compute SOL (64-MAC + 8-DW + 8-Vec)", arch_sol, NV_GREEN),
        ("Absolute 64-Lane Dense MAC Utilization", dense_sol, CYAN),
        ("External Memory Bandwidth SOL", bw_sol, AMBER),
        ("DMA / Compute Overlap Efficiency" if pmu_layers else "Double-Buffer Layer Coverage", ovlp_pct, TEAL),
    ]
    gy = p1[1] + 62
    for glbl, gval, gcol in gauges:
        drw.text((p1[0] + 28, gy), glbl, font=fnt["body"], fill=TXT)
        drw.text((p1[2] - 100, gy), f"{gval:5.1f}%", font=fnt["h3"], fill=gcol)
        bx0, by0, bx1, by1 = p1[0] + 28, gy + 24, p1[2] - 28, gy + 42
        drw.rounded_rectangle([bx0, by0, bx1, by1], radius=6, fill=(33, 38, 45))
        fill_w = int((bx1 - bx0) * max(0.0, min(100.0, gval)) / 100.0)
        if fill_w > 4:
            drw.rounded_rectangle([bx0, by0, bx0 + fill_w, by1], radius=6, fill=gcol)
        gy += 58

    # Bus scaling sub-chart
    drw.text((p1[0] + 28, gy + 8), "Analytical External Bus Scaling (Cycles & Speedup):", font=fnt["h3"], fill=TXT)
    w_est = {}
    for bits in (32, 64, 128):
        bw = bits // 8
        c_sum = 0
        for p in profiles:
            d_cyc = math.ceil(p.total_ext_bytes / bw)
            c_sum += (max(p.ideal_comp_cycles, d_cyc) + math.ceil(0.15 * min(p.ideal_comp_cycles, d_cyc))) if (p.db_en and p.tiled) else (p.ideal_comp_cycles + d_cyc)
        w_est[bits] = max(1, c_sum)
    max_w = w_est[32]
    by = gy + 38
    for bits, bcol in [(32, CORAL), (64, AMBER), (128, NV_GREEN)]:
        drw.text((p1[0] + 28, by + 3), f"{bits:>3}-bit DMA", font=fnt["mono"], fill=TXT)
        bx0, bx1 = p1[0] + 150, p1[2] - 250
        bw_px = int((bx1 - bx0) * (w_est[bits] / max_w))
        drw.rounded_rectangle([bx0, by, bx1, by + 22], radius=5, fill=(33, 38, 45))
        drw.rounded_rectangle([bx0, by, bx0 + max(8, bw_px), by + 22], radius=5, fill=bcol)
        sp = max_w / w_est[bits]
        drw.text((bx1 + 14, by + 2), f"{w_est[bits]:>11,} cyc  ({sp:.2f}x)", font=fnt["mono"], fill=bcol)
        by += 32

    # ── Panel 2: Log-Log Roofline Model (Right Top) ──
    p2 = (1112, 152, W - 24, 590)
    card(*p2, f"2. ROOFLINE MODEL (GMAC/s vs MAC/Byte @ {freq_mhz:.0f} MHz)")
    rx0, ry0, rx1, ry1 = p2[0] + 82, p2[1] + 64, p2[2] - 36, p2[3] - 54
    drw.rectangle([rx0, ry0, rx1, ry1], fill=(15, 20, 27), outline=BORDER, width=1)

    # Log10 axes: X in [0.1, 250] MAC/Byte -> log10 [-1.0, 2.4]; Y in [0.05, 20] GMAC/s -> log10 [-1.3, 1.3]
    lx_min, lx_max = -1.0, 2.4
    ly_min, ly_max = -1.3, 1.35

    def map_rf(ai_val, gmac_val):
        lx = max(lx_min, min(lx_max, math.log10(max(1e-3, ai_val))))
        ly = max(ly_min, min(ly_max, math.log10(max(1e-3, gmac_val))))
        px = rx0 + int((lx - lx_min) / (lx_max - lx_min) * (rx1 - rx0))
        py = ry1 - int((ly - ly_min) / (ly_max - ly_min) * (ry1 - ry0))
        return px, py

    # Grid lines
    for xv in [0.1, 0.5, 1, 2, 4, 10, 25, 100]:
        gx, _ = map_rf(xv, 1.0)
        drw.line([gx, ry0, gx, ry1], fill=(30, 36, 44), width=1)
        drw.text((gx - 14, ry1 + 8), f"{xv:g}", font=fnt["tiny"], fill=MUTED)
    drw.text(((rx0 + rx1) // 2 - 90, ry1 + 26), "Arithmetic Intensity (MAC / Byte)", font=fnt["small"], fill=MUTED)

    for yv in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.8]:
        _, gy_p = map_rf(1.0, yv)
        drw.line([rx0, gy_p, rx1, gy_p], fill=(30, 36, 44), width=1)
        drw.text((rx0 - 52, gy_p - 7), f"{yv:g}", font=fnt["tiny"], fill=MUTED)

    # Draw Roofs: 32b (0.8 GB/s), 64b (1.6 GB/s), 128b (3.2 GB/s) + Dense peak + DW peak
    dw_peak = (hw.dw_parallel_ch * freq_mhz) / 1000.0
    _, y_dw = map_rf(1.0, dw_peak)
    drw.line([rx0, y_dw, rx1, y_dw], fill=(120, 100, 50), width=2)
    drw.text((rx1 - 190, y_dw + 4), f"8-DW Roof ({dw_peak:.1f} GMAC/s)", font=fnt["tiny"], fill=AMBER)

    for bw_b, rcol, rlbl in [(4, CORAL, "32b"), (8, AMBER, "64b"), (16, NV_GREEN, "128b")]:
        bw_gbs = (bw_b * freq_mhz) / 1000.0
        ridge = peak_gmacs / bw_gbs
        p_start = map_rf(0.1, 0.1 * bw_gbs)
        p_knee = map_rf(ridge, peak_gmacs)
        p_end = map_rf(250.0, peak_gmacs)
        w_line = 3 if bw_b == hw.ext_bw_bytes else 1
        drw.line([p_start, p_knee, p_end], fill=rcol, width=w_line)
        lx_lbl, ly_lbl = map_rf(ridge * 0.48, ridge * 0.48 * bw_gbs)
        drw.text((lx_lbl - 28, ly_lbl - 16), f"{rlbl} ({bw_gbs:.1f}GB/s)", font=fnt["tiny"], fill=rcol)

    # Plot layer bubbles
    fam_colors = {
        "Conv1x1": CYAN,
        "DW3x3": AMBER,
        "GlobalDW": AMBER,
        "FC": VIOLET,
    }
    top_cyc_ids = {p.idx for p in sorted(profiles, key=cyc_fn, reverse=True)[:6]}
    for p in profiles:
        c = cyc_fn(p)
        if p.macs <= 0 or c <= 0:
            continue
        ai_v = p.macs / max(1, p.total_ext_bytes)
        gm_v = (p.macs / c) * (freq_mhz / 1000.0)
        px, py = map_rf(ai_v, gm_v)
        rad = max(5, min(16, int(4 + 28 * math.sqrt(c / total_cycles))))
        bcol = fam_colors.get(p.op_family, NV_GREEN)
        drw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=bcol, outline=TXT, width=1)
        if p.idx in top_cyc_ids:
            drw.text((px + rad + 4, py - 7), f"L{p.idx}", font=fnt["tiny"], fill=TXT)

    # Legend inside Roofline
    leg_x = rx0 + 14
    for l_txt, l_col in [("Conv3x3/7x7", NV_GREEN), ("Conv1x1", CYAN), ("DWConv", AMBER), ("FC", VIOLET)]:
        drw.ellipse([leg_x, ry0 + 10, leg_x + 10, ry0 + 20], fill=l_col)
        drw.text((leg_x + 15, ry0 + 8), l_txt, font=fnt["tiny"], fill=TXT)
        leg_x += 108

    # ── Panel 3: 4-Bank On-Chip SRAM ("Cache") & Traffic Waterfall (Middle Left) ──
    p3 = (24, 610, 1088, 1030)
    card(*p3, "3. 4-BANK ON-CHIP SRAM ('CACHE') OCCUPANCY & DDR AMPLIFICATION")
    sy = p3[1] + 62
    banks = [
        ("IFM Bank (32 KB)", hw.act_bank_size, lambda p: p.ifm_used_bytes, lambda p: p.ifm_occ_pct, CYAN),
        ("OFM Bank (48 KB)", hw.ofm_bank_size, lambda p: p.ofm_used_bytes, lambda p: p.ofm_occ_pct, NV_GREEN),
        ("WGT Bank (80 KB)", hw.weight_buf_size, lambda p: p.wgt_used_bytes, lambda p: p.wgt_occ_pct, AMBER),
        ("PRM Bank (12 KB)", hw.param_sram_size, lambda p: p.prm_used_bytes, lambda p: p.prm_occ_pct, VIOLET),
    ]
    for bname, bcap, ufn, ofn, bcol in banks:
        act = [p for p in profiles if ufn(p) > 0]
        pk_pct = min(100.0, 100.0 * max((ufn(p) for p in act), default=0) / bcap)
        av_pct = (sum(ofn(p) for p in act) / len(act)) if act else 0.0
        drw.text((p3[0] + 28, sy), bname, font=fnt["body"], fill=TXT)
        bx0, bx1 = p3[0] + 215, p3[2] - 255
        drw.rounded_rectangle([bx0, sy + 2, bx1, sy + 22], radius=5, fill=(33, 38, 45))
        pk_w = int((bx1 - bx0) * pk_pct / 100.0)
        av_w = int((bx1 - bx0) * av_pct / 100.0)
        if pk_w > 4:
            drw.rounded_rectangle([bx0, sy + 2, bx0 + pk_w, sy + 22], radius=5, fill=(55, 65, 81))
        if av_w > 4:
            drw.rounded_rectangle([bx0, sy + 2, bx0 + av_w, sy + 22], radius=5, fill=bcol)
        drw.text((bx1 + 14, sy + 2), f"Avg {av_pct:4.1f}% | Peak {pk_pct:5.1f}%", font=fnt["mono"], fill=bcol)
        sy += 44

    # DDR Amplification Bridge
    drw.line([p3[0] + 24, sy + 8, p3[2] - 24, sy + 8], fill=BORDER, width=1)
    drw.text((p3[0] + 28, sy + 20), "DDR Traffic Accounting (Raw vs Cache Amplification vs Fusion Savings):", font=fnt["h3"], fill=TXT)
    bridge = [
        ("Raw Tensors", _fmt_bytes(total_raw), CYAN),
        ("+ Tile Halo", f"+{_fmt_bytes(total_halo)}", AMBER),
        ("+ WGT Reload", f"+{_fmt_bytes(total_wrel)}", CORAL),
        ("- SRAM Fusion", f"-{_fmt_bytes(total_fsav)}", NV_GREEN),
        ("= Net Ext DDR", _fmt_bytes(total_ext), TXT),
    ]
    bx = p3[0] + 28
    for b_lbl, b_val, b_col in bridge:
        drw.rounded_rectangle([bx, sy + 52, bx + 192, sy + 118], radius=8, fill=(15, 20, 27), outline=BORDER, width=1)
        drw.text((bx + 12, sy + 60), b_lbl, font=fnt["tiny"], fill=MUTED)
        drw.text((bx + 12, sy + 84), b_val, font=fnt["h3"], fill=b_col)
        bx += 202

    # ── Panel 4: Pipeline State / Operator Breakdown (Middle Right) ──
    p4 = (1112, 610, W - 24, 1030)
    if pmu_layers:
        card(*p4, "4. RTL HARDWARE PMU PIPELINE & STALL ATTRIBUTION")
        tot_npu = max(1, sum(m.npu_cycles for m in pmu_layers))
        items = [
            ("ST_MAC   (64-Lane Row MAC Active)", sum(m.st_mac for m in pmu_layers), NV_GREEN),
            ("ST_WB    (8-Lane PPU Drain/Requant)", sum(m.st_wb for m in pmu_layers), CYAN),
            ("WT_WGT   (Stall Waiting on WGT DMA)", sum(m.wt_wgt for m in pmu_layers), CORAL),
            ("WT_DB    (Double-Buffer Sync Wait)", sum(m.wt_db for m in pmu_layers), AMBER),
            ("WT_ACT   (Stall Waiting on IFM DMA)", sum(m.wt_act for m in pmu_layers), VIOLET),
            ("ST_VEC   (Pooling / Add / Resize)", sum(m.st_pool + m.st_add + m.st_rsz for m in pmu_layers), TEAL),
            ("WT_STORE (Stall Waiting on OFM DMA)", sum(m.wt_store for m in pmu_layers), (244, 114, 182)),
        ]
        denom = tot_npu
    else:
        card(*p4, "4. OPERATOR FAMILY CYCLE & COMPUTE BREAKDOWN")
        fam_agg: Dict[str, int] = {}
        for p in profiles:
            fam_agg[p.op_family] = fam_agg.get(p.op_family, 0) + cyc_fn(p)
        palette = [NV_GREEN, CYAN, AMBER, VIOLET, TEAL, CORAL, (244, 114, 182)]
        items = [
            (f"{fam:<12} ({sum(1 for x in profiles if x.op_family == fam):>2} layers)", cyc_v, palette[idx % len(palette)])
            for idx, (fam, cyc_v) in enumerate(sorted(fam_agg.items(), key=lambda kv: kv[1], reverse=True)[:7])
        ]
        denom = total_cycles

    # Stacked horizontal bar at top of Panel 4
    sx0, sx1 = p4[0] + 28, p4[2] - 28
    sy0 = p4[1] + 62
    drw.rounded_rectangle([sx0, sy0, sx1, sy0 + 28], radius=6, fill=(33, 38, 45))
    cur_x = sx0
    for _, iv, icol in items:
        w_seg = int((sx1 - sx0) * (iv / denom))
        if w_seg > 2:
            drw.rectangle([cur_x, sy0, min(sx1, cur_x + w_seg), sy0 + 28], fill=icol)
            cur_x += w_seg

    iy = sy0 + 48
    for ilbl, iv, icol in items:
        pct = 100.0 * iv / denom
        drw.rectangle([sx0, iy + 4, sx0 + 14, iy + 18], fill=icol)
        drw.text((sx0 + 24, iy), ilbl, font=fnt["mono"], fill=TXT)
        bx0, bx1 = sx0 + 365, sx1 - 210
        drw.rounded_rectangle([bx0, iy + 3, bx1, iy + 19], radius=4, fill=(33, 38, 45))
        bw_i = int((bx1 - bx0) * min(1.0, pct / 100.0))
        if bw_i > 3:
            drw.rounded_rectangle([bx0, iy + 3, bx0 + bw_i, iy + 19], radius=4, fill=icol)
        drw.text((bx1 + 12, iy), f"{iv:>10,} cyc ({pct:4.1f}%)", font=fnt["mono"], fill=icol)
        iy += 42

    # ── Panel 5: Per-Layer Cycle Waterfall + Optimization Advisor (Bottom) ──
    p5_l = (24, 1050, 1340, H - 24)
    card(*p5_l, f"5. PER-LAYER CYCLE WATERFALL ({len(profiles)} LAYERS, COLOR = BOTTLENECK REGIME)")
    wx0, wy0, wx1, wy1 = p5_l[0] + 68, p5_l[1] + 66, p5_l[2] - 28, p5_l[3] - 52
    drw.rectangle([wx0, wy0, wx1, wy1], fill=(15, 20, 27), outline=BORDER, width=1)

    max_l_cyc = max((cyc_fn(p) for p in profiles), default=1)
    n_l = max(1, len(profiles))
    slot_w = (wx1 - wx0) / n_l
    bar_w = max(3.0, slot_w * 0.76)

    for idx_l, p in enumerate(profiles):
        c = cyc_fn(p)
        bh = int((wy1 - wy0 - 18) * (c / max(1, max_l_cyc)))
        bx_l = int(wx0 + idx_l * slot_w + (slot_w - bar_w) * 0.5)
        bx_r = max(bx_l + 2, int(bx_l + bar_w))
        if "BW-BOUND" in p.bound_regime:
            bcol = CORAL
        elif "PPU/WB" in p.bound_regime:
            bcol = CYAN
        elif "DW-" in p.bound_regime:
            bcol = AMBER
        elif "FUSED" in p.bound_regime:
            bcol = TEAL
        else:
            bcol = NV_GREEN
        if bh > 1:
            drw.rectangle([bx_l, wy1 - bh, bx_r, wy1 - 1], fill=bcol)
        step = max(1, n_l // 16)
        if idx_l % step == 0:
            drw.text((bx_l - 4, wy1 + 6), f"L{p.idx}", font=fnt["tiny"], fill=MUTED)

    # Annotate top 4 tallest bars
    for p in sorted(profiles, key=cyc_fn, reverse=True)[:4]:
        c = cyc_fn(p)
        bh = int((wy1 - wy0 - 18) * (c / max(1, max_l_cyc)))
        bx_c = int(wx0 + p.idx * slot_w)
        drw.text((max(wx0 + 4, min(wx1 - 95, bx_c - 10)), max(wy0 + 4, wy1 - bh - 16)),
                 f"L{p.idx}:{p.op_family}", font=fnt["tiny"], fill=TXT)

    # Regime legend
    rlx = wx0 + 12
    for rtxt, rcol in [("Compute-Bound", NV_GREEN), ("PPU/WB-Heavy", CYAN), ("DW-8Lane", AMBER), ("BW-Bound", CORAL), ("SRAM-Fused", TEAL)]:
        drw.rectangle([rlx, wy0 + 8, rlx + 12, wy0 + 20], fill=rcol)
        drw.text((rlx + 17, wy0 + 6), rtxt, font=fnt["tiny"], fill=TXT)
        rlx += 150

    # Panel 6: Optimization Advisor (Bottom Right)
    p6 = (1364, 1050, W - 24, H - 24)
    card(*p6, "6. OPTIMIZATION ADVISOR (RANKED SAVINGS)")
    findings = generate_optimization_findings(layers, profiles, hw)[:4]
    fy = p6[1] + 58
    if not findings:
        drw.text((p6[0] + 28, fy + 20), "Near architectural roofline — no major bottlenecks.", font=fnt["body"], fill=NV_GREEN)
    else:
        for f in findings:
            pct = 100.0 * f.est_cycle_saving / total_cycles
            scol = CORAL if f.severity == "HIGH" else (AMBER if f.severity == "MEDIUM" else CYAN)
            drw.rounded_rectangle([p6[0] + 20, fy, p6[2] - 20, fy + 94], radius=8, fill=(15, 20, 27), outline=BORDER, width=1)
            drw.rectangle([p6[0] + 20, fy, p6[0] + 26, fy + 94], fill=scol)
            title_short = f.title if len(f.title) <= 64 else f.title[:61] + "..."
            drw.text((p6[0] + 36, fy + 8), f"[{f.category}] {title_short}", font=fnt["h3"], fill=scol)
            drw.text((p6[0] + 36, fy + 34), f"Est. Upper-Bound Saving: ~{f.est_cycle_saving:,} cycles (~{pct:.1f}%)", font=fnt["mono"], fill=NV_GREEN)
            det_short = f.detail if len(f.detail) <= 92 else f.detail[:89] + "..."
            drw.text((p6[0] + 36, fy + 58), det_short, font=fnt["tiny"], fill=MUTED)
            fy += 104

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)
    print(f"Exported PNG visual profile to {out_path}")
    return True


def render_html_dashboard(
    out_path: str,
    model_label: str,
    layers: List[LayerDesc],
    profiles: List[LayerProfile],
    hw: HWConfig,
    freq_mhz: float,
    soc_log: Optional[SoCLogSummary] = None,
) -> None:
    """Export a zero-dependency interactive HTML dashboard (hover Roofline + full layer table)."""
    import html as _html

    has_hw = any(p.hw_cycles is not None for p in profiles)
    cyc_fn = lambda p: (p.hw_cycles if p.hw_cycles is not None else p.est_cycles)
    total_cycles = max(1, sum(cyc_fn(p) for p in profiles))
    total_macs = sum(p.macs for p in profiles)
    total_ext = sum(p.total_ext_bytes for p in profiles)
    peak_gmacs = (hw.mac_lanes * freq_mhz) / 1000.0
    ach_gmacs = (total_macs / total_cycles) * (freq_mhz / 1000.0)
    lat_ms = total_cycles / (freq_mhz * 1000.0)
    dense_sol = min(100.0, ((total_macs / hw.mac_lanes) / total_cycles) * 100.0)
    arch_sol = min(100.0, (sum(p.ideal_comp_cycles for p in profiles) / total_cycles) * 100.0)
    findings = generate_optimization_findings(layers, profiles, hw)

    rows_html = []
    pts_js = []
    for p in profiles:
        c = cyc_fn(p)
        dsol = ((p.macs / hw.mac_lanes) / c) * 100.0 if (p.macs > 0 and c > 0) else 0.0
        gmacs_l = (p.macs / c) * (freq_mhz / 1000.0) if c > 0 else 0.0
        ai_l = p.macs / max(1, p.total_ext_bytes)
        if p.macs > 0 and c > 0:
            pts_js.append({
                "idx": p.idx, "op": p.op_family, "shape": p.shape_str,
                "ai": round(ai_l, 2), "gmacs": round(gmacs_l, 2),
                "cyc": c, "pct": round(100.0 * c / total_cycles, 2),
                "regime": p.bound_regime,
            })
        rows_html.append(
            f"<tr><td>L{p.idx}</td><td>{_html.escape(p.op_family)}</td>"
            f"<td>{_html.escape(p.shape_str)}</td><td class='num'>{c:,}</td>"
            f"<td class='num'>{100.0 * c / total_cycles:.1f}%</td>"
            f"<td class='num'>{dsol:.1f}%</td>"
            f"<td class='num'>{_fmt_bytes(p.total_ext_bytes)}</td>"
            f"<td class='num'>{p.ifm_occ_pct:.0f}% / {p.ofm_occ_pct:.0f}% / {p.wgt_occ_pct:.0f}%</td>"
            f"<td class='num'>{p.num_tiles}</td>"
            f"<td>{_html.escape(p.fuse_tag)}</td>"
            f"<td><span class='tag'>{_html.escape(p.bound_regime)}</span></td></tr>"
        )

    adv_html = "".join(
        f"<div class='adv'><b>[{_html.escape(f.severity)}] [{_html.escape(f.category)}] {_html.escape(f.title)}</b>"
        f"<div class='sav'>Est. Upper-Bound Saving: ~{f.est_cycle_saving:,} cycles (~{100.0 * f.est_cycle_saving / total_cycles:.1f}%)</div>"
        f"<p>{_html.escape(f.detail)}</p></div>"
        for f in findings
    ) or "<div class='adv'>Near architectural roofline.</div>"

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Open-NPU Profile — {_html.escape(model_label)}</title>
<style>
  body {{ background:#0d1117; color:#e6edf3; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; margin:24px; }}
  h1 {{ margin:0 0 6px 0; font-size:24px; color:#76b900; }}
  .sub {{ color:#8b949e; font-size:13px; margin-bottom:18px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:18px; }}
  .kpi {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px 16px; }}
  .kpi .k {{ color:#8b949e; font-size:11px; text-transform:uppercase; }}
  .kpi .v {{ font-size:20px; font-weight:700; margin-top:4px; color:#58a6ff; }}
  .grid2 {{ display:grid; grid-template-columns:1.2fr 1fr; gap:16px; margin-bottom:18px; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px; }}
  .card h2 {{ margin:0 0 12px 0; font-size:15px; color:#e6edf3; border-left:4px solid #76b900; padding-left:10px; }}
  .adv {{ background:#0d1117; border:1px solid #30363d; border-left:4px solid #f59e0b; border-radius:6px; padding:10px 12px; margin-bottom:10px; font-size:13px; }}
  .adv .sav {{ color:#76b900; font-family:monospace; margin:4px 0; }}
  .adv p {{ margin:4px 0 0 0; color:#8b949e; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th, td {{ border-bottom:1px solid #21262d; padding:7px 8px; text-align:left; }}
  th {{ background:#0d1117; color:#8b949e; position:sticky; top:0; }}
  td.num {{ text-align:right; font-family:monospace; }}
  .tag {{ background:#21262d; padding:2px 6px; border-radius:4px; font-size:11px; }}
  #tip {{ position:fixed; display:none; background:#1f2937; border:1px solid #76b900; padding:8px 10px; border-radius:6px; font-size:12px; pointer-events:none; z-index:99; }}
</style></head><body>
<h1>Open-NPU Nsight-Style Interactive Profiler — {_html.escape(model_label)}</h1>
<div class="sub">HW: {hw.mac_lanes} Dense MACs | {hw.dw_parallel_ch} DW | {PPU_LANES} PPU | {hw.spad_total_kb} KB 4-Bank SRAM | {hw.dma_bus_width}-bit {hw.dma_bus_proto.upper()} @ {freq_mhz:.0f} MHz | Mode: {'RTL HW PMU' if has_hw else 'Analytical Roofline'}</div>
<div class="kpis">
  <div class="kpi"><div class="k">Active Cycles</div><div class="v">{total_cycles:,}</div></div>
  <div class="kpi"><div class="k">Latency / FPS</div><div class="v" style="color:#76b900">{lat_ms:.2f} ms ({1000.0 / max(1e-6, lat_ms):.1f} FPS)</div></div>
  <div class="kpi"><div class="k">Throughput</div><div class="v">{ach_gmacs:.2f} / {peak_gmacs:.1f} GMAC/s</div></div>
  <div class="kpi"><div class="k">Dense / Arch SOL%</div><div class="v" style="color:#f59e0b">{dense_sol:.1f}% / {arch_sol:.1f}%</div></div>
  <div class="kpi"><div class="k">Ext DDR Traffic</div><div class="v" style="color:#a78bfa">{_fmt_bytes(total_ext)}</div></div>
</div>
<div class="grid2">
  <div class="card"><h2>Interactive Log-Log Roofline (Hover Bubbles for Layer Details)</h2>
    <svg id="rf" viewBox="0 0 760 360" style="width:100%;height:auto;background:#0d1117;border-radius:6px;"></svg>
  </div>
  <div class="card"><h2>Actionable Optimization Advisor</h2>{adv_html}</div>
</div>
<div class="card"><h2>Per-Layer Breakdown ({len(profiles)} Layers — SRAM Occupancy = IFM / OFM / WGT)</h2>
  <div style="max-height:520px;overflow:auto;">
    <table><thead><tr><th>ID</th><th>Op</th><th>Shape</th><th>Cycles</th><th>Cyc%</th><th>Dense%</th><th>DDR</th><th>SRAM Occ%</th><th>Tiles</th><th>Fuse</th><th>Regime</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody></table>
  </div>
</div>
<div id="tip"></div>
<script>
const pts = {json.dumps(pts_js)};
const peak = {peak_gmacs}, freq = {freq_mhz};
const svg = document.getElementById('rf'), tip = document.getElementById('tip');
const x0=58, y0=22, x1=735, y1=322, lxMin=-1.0, lxMax=2.4, lyMin=-1.3, lyMax=1.35;
const mx = ai => x0 + (Math.min(lxMax,Math.max(lxMin,Math.log10(Math.max(1e-3,ai))))-lxMin)/(lxMax-lxMin)*(x1-x0);
const my = gm => y1 - (Math.min(lyMax,Math.max(lyMin,Math.log10(Math.max(1e-3,gm))))-lyMin)/(lyMax-lyMin)*(y1-y0);
let h = '';
[0.1,0.5,1,2,4,10,25,100].forEach(v=>{{ let x=mx(v); h+=`<line x1="${{x}}" y1="${{y0}}" x2="${{x}}" y2="${{y1}}" stroke="#21262d"/><text x="${{x-10}}" y="${{y1+16}}" fill="#8b949e" font-size="10">${{v}}</text>`; }});
[0.1,0.5,1,2,4,8,12.8].forEach(v=>{{ let y=my(v); h+=`<line x1="${{x0}}" y1="${{y}}" x2="${{x1}}" y2="${{y}}" stroke="#21262d"/><text x="${{x0-42}}" y="${{y+4}}" fill="#8b949e" font-size="10">${{v}}</text>`; }});
[[4,'#fb7185','32b'],[8,'#f59e0b','64b'],[16,'#76b900','128b']].forEach(([b,c,lbl])=>{{
  let bw = b*freq/1000, r = peak/bw;
  h+=`<polyline fill="none" stroke="${{c}}" stroke-width="${{b==={hw.ext_bw_bytes}?2.8:1.3}}" points="${{mx(0.1)}},${{my(0.1*bw)}} ${{mx(r)}},${{my(peak)}} ${{mx(250)}},${{my(peak)}}"/>`;
}});
pts.forEach((p,i)=>{{
  let cx=mx(p.ai), cy=my(p.gmacs), r=Math.max(4,Math.min(14,4+Math.sqrt(p.pct)*3.2));
  let col = p.op==='Conv1x1'?'#38bdf8':(p.op.startsWith('DW')?'#f59e0b':(p.op==='FC'?'#a78bfa':'#76b900'));
  h+=`<circle cx="${{cx}}" cy="${{cy}}" r="${{r}}" fill="${{col}}" fill-opacity="0.82" stroke="#e6edf3" stroke-width="1" data-i="${{i}}"/>`;
}});
svg.innerHTML = h;
svg.addEventListener('mousemove', e=>{{
  const idx = e.target.getAttribute('data-i');
  if(idx===null){{ tip.style.display='none'; return; }}
  const p = pts[+idx];
  tip.style.display='block'; tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY+14)+'px';
  tip.innerHTML = `<b>L${{p.idx}} (${{p.op}})</b><br>${{p.shape}}<br>Cycles: ${{p.cyc.toLocaleString()}} (${{p.pct}}%)<br>Throughput: ${{p.gmacs}} GMAC/s | AI: ${{p.ai}} MAC/B<br>Regime: ${{p.regime}}`;
}});
</script></body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"Exported interactive HTML profile to {out_path}")


# ─────────────────────────────────────────────────────────────────────
#  Self-Test
# ─────────────────────────────────────────────────────────────────────

def selftest() -> None:
    import tempfile
    from model_packer import LayerConfig, PerChannelParam, pack_model

    print("=== Running npu_profiler self-test ===")
    l0 = LayerConfig(
        op_type=OP_CONV2D, data_type=0,
        in_h=56, in_w=56, in_c=24, out_h=56, out_w=56, out_c=64,
        kernel_h=1, kernel_w=1, stride_h=1, stride_w=1,
        tile_h=14, tile_w=14, tile_num_h=4, tile_num_w=4,
        sched_ctrl=SCHED_CTRL_DB_EN, wgt_layout=1,
        ch_params=[PerChannelParam() for _ in range(64)],
    )
    l1 = LayerConfig(
        op_type=OP_DW_CONV, data_type=0,
        in_h=56, in_w=56, in_c=64, out_h=56, out_w=56, out_c=64,
        kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, pad_top=1, pad_left=1,
        tile_h=14, tile_w=14, tile_num_h=4, tile_num_w=4,
        sched_ctrl=SCHED_CTRL_DB_EN, wgt_layout=1,
        ch_params=[PerChannelParam() for _ in range(64)],
    )
    wgt = bytes(24 * 64 + 64 * 9)
    with tempfile.TemporaryDirectory() as td:
        bin_path = os.path.join(td, "toy.npu1.bin")
        pack_model([l0, l1], wgt, bin_path)
        descs = parse_npu1_binary(bin_path)
        assert len(descs) == 2
        hw = HWConfig(ext_bw_bytes=16, dma_bus_proto="axi4")
        profs = analyze_model(descs, hw)
        assert profs[0].macs == 56 * 56 * 64 * 24
        assert profs[1].macs == 56 * 56 * 64 * 9
        print_report("toy_selftest", descs, profs, hw, freq_mhz=200.0, top_n=5)
    print("=== npu_profiler self-test PASSED ===")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
        return

    parser = argparse.ArgumentParser(
        description="Open-NPU Roofline, On-Chip SRAM & Hardware PMU Profiler"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to NPU1 binary (.npu1.bin / .bin)")
    parser.add_argument("--meta", type=str, default=None,
                        help="Path to layer metadata.json")
    parser.add_argument("--soc-log", type=str, default=None,
                        help="Optional RTL/SoC UART simulation log containing PERF_L / STL counters")
    parser.add_argument("--label", type=str, default=None,
                        help="Display label for the profiled model")
    parser.add_argument("--freq-mhz", type=float, default=200.0,
                        help="Target clock frequency in MHz for throughput/FPS projection (default: 200)")
    parser.add_argument("--top", type=int, default=12,
                        help="Number of top hotspot layers to display (default: 12)")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Optional path to export structured JSON report")
    parser.add_argument("--png-out", type=str, default=None,
                        help="Optional path to export Nsight-style multi-panel PNG chart")
    parser.add_argument("--html-out", type=str, default=None,
                        help="Optional path to export interactive HTML Roofline & SRAM dashboard")
    add_hw_args(parser)
    args = parser.parse_args()

    if not args.model and not args.meta:
        parser.error("Provide either --model <file.npu1.bin> or --meta <metadata.json> (or run 'selftest').")

    soc_log = parse_soc_log(args.soc_log) if args.soc_log else None
    # Auto-sync DMA bus width/proto from SoC log if user didn't explicitly pass --dma-bus-width / --ext-bw
    if soc_log and args.dma_bus_width is None and args.ext_bw is None and soc_log.dma_width_bits:
        args.dma_bus_width = soc_log.dma_width_bits
    if soc_log and args.dma_bus_proto is None and soc_log.dma_proto:
        args.dma_bus_proto = soc_log.dma_proto

    hw = hw_config_from_args(args)
    layers = parse_npu1_binary(args.model) if args.model else parse_metadata_json(args.meta)
    label = args.label or os.path.basename(args.model or os.path.dirname(os.path.abspath(args.meta)))

    profiles = analyze_model(layers, hw, soc_log=soc_log)
    print_report(label, layers, profiles, hw, freq_mhz=args.freq_mhz, top_n=args.top, soc_log=soc_log)

    if args.png_out:
        render_png_dashboard(args.png_out, label, layers, profiles, hw, freq_mhz=args.freq_mhz, soc_log=soc_log)
    if args.html_out:
        render_html_dashboard(args.html_out, label, layers, profiles, hw, freq_mhz=args.freq_mhz, soc_log=soc_log)

    if args.json_out:
        findings = generate_optimization_findings(layers, profiles, hw)
        payload = {
            "label": label,
            "hw_config": asdict(hw),
            "layers": [asdict(p) for p in profiles],
            "findings": [asdict(f) for f in findings],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Exported JSON profile to {args.json_out}")


if __name__ == "__main__":
    main()
