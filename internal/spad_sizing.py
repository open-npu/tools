#!/usr/bin/env python3
"""Size the scratchpad for a candidate network.

Runs the real tiling calculator over a layer list for several SPAD_KB values
and reports, per configuration, whether every layer is feasible and how much
DMA traffic the resulting tiling costs.

SPAD_KB is the RTL knob in rtl/include/npu_defines.vh. npu_top.v derives:
    ACT_DEPTH   = SPAD_KB * 64   words  -> act  bytes = SPAD_KB * 256
    WGT_DEPTH   = SPAD_KB * 128  words  -> wgt  bytes = SPAD_KB * 512
    PARAM_DEPTH = SPAD_KB * 16   words  -> param bytes = SPAD_KB * 64
so the physical SRAM is SPAD_KB * 832 bytes, not SPAD_KB * 1024.

Usage:
    python3 tools/internal/spad_sizing.py
    python3 tools/internal/spad_sizing.py --net catdog32 --bits 8
"""

import argparse
import os
import sys

INTERNAL = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(INTERNAL)
sys.path.insert(0, TOOLS)

from hw_config import HWConfig
from tiling import compute_tiling


def spad_profile(spad_kb):
    """HWConfig matching what npu_top.v derives from SPAD_KB."""
    return HWConfig(
        mac_lanes=64,
        act_bank_size=spad_kb * 256,
        weight_buf_size=spad_kb * 512,
        param_sram_max_ch=(spad_kb * 64) // 14,
    )


def conv(name, ih, iw, ic, oc, k=3, s=1):
    oh = (ih + (k // 2) * 2 - k) // s + 1
    ow = (iw + (k // 2) * 2 - k) // s + 1
    return dict(name=name, op='conv', in_h=ih, in_w=iw, in_c=ic,
                out_h=oh, out_w=ow, out_c=oc, kh=k, kw=k, sh=s, sw=s)


def dw(name, ih, iw, c, k=3, s=1):
    oh = (ih + (k // 2) * 2 - k) // s + 1
    ow = (iw + (k // 2) * 2 - k) // s + 1
    return dict(name=name, op='dw', in_h=ih, in_w=iw, in_c=c,
                out_h=oh, out_w=ow, out_c=c, kh=k, kw=k, sh=s, sw=s)


def pool(name, ih, iw, c, k=2, s=2):
    return dict(name=name, op='pool', in_h=ih, in_w=iw, in_c=c,
                out_h=ih // s, out_w=iw // s, out_c=c, kh=k, kw=k, sh=s, sw=s)


def gpool(name, ih, iw, c):
    return dict(name=name, op='pool', in_h=ih, in_w=iw, in_c=c,
                out_h=1, out_w=1, out_c=c, kh=ih, kw=iw, sh=1, sw=1)


def fc(name, ic, oc):
    return dict(name=name, op='fc', in_h=1, in_w=1, in_c=ic,
                out_h=1, out_w=1, out_c=oc, kh=1, kw=1, sh=1, sw=1)


# ── Candidate networks ──
# catdog32 is MODEL_G: CIFAR-scale cat/dog classifier, plain Conv/Pool.
# Channel counts are multiples of 16.
NETS = {
    'catdog32': [
        conv('c0', 32, 32, 3, 16),
        conv('c1', 32, 32, 16, 16),
        pool('p0', 32, 32, 16),
        conv('c2', 16, 16, 16, 32),
        conv('c3', 16, 16, 32, 32),
        pool('p1', 16, 16, 32),
        conv('c4', 8, 8, 32, 64),
        gpool('gp', 8, 8, 64),
        fc('fc', 64, 2),
    ],
    # Same shape but depthwise-separable, which is what a real edge net uses.
    'catdog32_dw': [
        conv('c0', 32, 32, 3, 16),
        dw('d0', 32, 32, 16),
        conv('pw0', 32, 32, 16, 32, k=1),
        pool('p0', 32, 32, 32),
        dw('d1', 16, 16, 32),
        conv('pw1', 16, 16, 32, 64, k=1),
        pool('p1', 16, 16, 64),
        dw('d2', 8, 8, 64),
        conv('pw2', 8, 8, 64, 64, k=1),
        gpool('gp', 8, 8, 64),
        fc('fc', 64, 2),
    ],
    'catdog64': [
        conv('c0', 64, 64, 3, 16, s=2),
        conv('c1', 32, 32, 16, 32),
        pool('p0', 32, 32, 32),
        conv('c2', 16, 16, 32, 64),
        pool('p1', 16, 16, 64),
        conv('c3', 8, 8, 64, 64),
        gpool('gp', 8, 8, 64),
        fc('fc', 64, 2),
    ],
}


def layer_report(layer, elem, profile, double_buffer):
    t = compute_tiling(
        op_type=layer['op'],
        in_h=layer['in_h'], in_w=layer['in_w'], in_c=layer['in_c'],
        out_h=layer['out_h'], out_w=layer['out_w'], out_c=layer['out_c'],
        kernel_h=layer['kh'], kernel_w=layer['kw'],
        stride_h=layer['sh'], stride_w=layer['sw'],
        elem_size=elem, double_buffer=double_buffer, profile=profile,
    )
    in_bytes = layer['in_h'] * layer['in_w'] * layer['in_c'] * elem
    out_bytes = layer['out_h'] * layer['out_w'] * layer['out_c'] * elem
    if layer['op'] == 'conv':
        wgt = layer['kh'] * layer['kw'] * layer['in_c'] * layer['out_c'] * elem
    elif layer['op'] == 'dw':
        wgt = layer['kh'] * layer['kw'] * layer['in_c'] * elem
    elif layer['op'] == 'fc':
        wgt = layer['in_c'] * layer['out_c'] * elem
    else:
        wgt = 0
    tiles = max(1, t['tile_num_h']) * max(1, t['tile_num_w'])
    oc_groups = 1
    if t.get('tile_oc', layer['out_c']) > 0:
        oc_groups = -(-layer['out_c'] // t['tile_oc'])
    # Weights are re-fetched once per spatial tile when OC is grouped.
    dma = in_bytes * tiles + out_bytes + wgt * (tiles if oc_groups > 1 else 1)
    return t, tiles, oc_groups, wgt, in_bytes, out_bytes, dma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', default='catdog32', choices=sorted(NETS))
    ap.add_argument('--bits', type=int, default=8, choices=(8, 16))
    ap.add_argument('--spad', type=int, nargs='*',
                    default=[16, 24, 32, 48, 64, 96, 192])
    ap.add_argument('--verbose', action='store_true',
                    help='print the per-layer tiling for each SPAD_KB')
    args = ap.parse_args()

    elem = args.bits // 8
    layers = NETS[args.net]
    peak_wgt = 0
    for l in layers:
        if l['op'] == 'conv':
            peak_wgt = max(peak_wgt, l['kh'] * l['kw'] * l['in_c'] * l['out_c'] * elem)

    print(f"net={args.net} bits={args.bits} layers={len(layers)} "
          f"largest_weight={peak_wgt}B")
    print()
    print(f"{'SPAD_KB':>7} {'phys':>7} {'act':>7} {'wgt':>7} {'param':>6} "
          f"{'tiled':>6} {'maxtiles':>8} {'DMA/inf':>10}")
    print('-' * 66)

    for spad in args.spad:
        profile = spad_profile(spad)
        total_dma = 0
        tiled = 0
        max_tiles = 1
        details = []
        for l in layers:
            # Mirror onnx_converter: single-buffer if it fits, else DB tiling.
            t_sb, tiles_sb = layer_report(l, elem, profile, False)[:2]
            if t_sb['tile_num_h'] == 0:
                t, tiles, ocg, wgt, ib, ob, dma = layer_report(l, elem, profile, False)
            else:
                t, tiles, ocg, wgt, ib, ob, dma = layer_report(l, elem, profile, True)
            if t['tile_num_h'] != 0:
                tiled += 1
            max_tiles = max(max_tiles, tiles * ocg)
            total_dma += dma
            details.append((l['name'], t['tile_h'], t['tile_w'], tiles, ocg, dma))

        print(f"{spad:>7} {spad*832//1024:>6}K {spad*256//1024:>6}K "
              f"{spad*512//1024:>6}K {spad*64//1024:>5}K "
              f"{tiled:>6} {max_tiles:>8} {total_dma/1024:>9.1f}K")

        if args.verbose:
            for name, th, tw, tiles, ocg, dma in details:
                print(f"        {name:>5}: tile={th}x{tw} tiles={tiles} "
                      f"oc_groups={ocg} dma={dma/1024:.1f}K")

    print()
    print("phys = act+wgt+param actually instantiated (SPAD_KB*832B).")
    print("DMA/inf = bytes moved across the Wishbone bus for one inference.")


if __name__ == '__main__':
    main()
