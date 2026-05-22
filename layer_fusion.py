"""
Open-NPU Layer Fusion: Block Detection and Fused Tiling Calculator

Detects fusible MobileNet-style Inverted Residual Blocks (Conv1×1 → DW3×3 → Conv1×1)
and computes optimal tiling parameters for fused execution where intermediate tensors
stay in SRAM (no DRAM round-trip).

SRAM execution model (2-bank alternating):
  Phase 1: Bank_A = input (tile+halo)²×c_in   → Conv1×1 → Bank_B = (tile+halo)²×c_mid
  Phase 2: Bank_B = DW input (already there)   → DW 3×3  → Bank_A = tile²×c_mid
  Phase 3: Bank_A = Conv1×1#2 input (already)  → Conv1×1 → Bank_B = tile²×c_out

Result: Only block input is loaded from DRAM, only block output is stored to DRAM.
Intermediate tensors (c_mid channels) never leave SRAM → 80%+ DMA saving.

SPDX-License-Identifier: Apache-2.0
"""

import math
from dataclasses import dataclass
from typing import List, Optional

# Hardware constants (match tiling.py)
ACT_BANK_SIZE = 32 * 1024       # 32 KB per activation bank
WEIGHT_BUF_SIZE = 64 * 1024     # 64 KB weight buffer
PARAM_SRAM_MAX_CH = 512


@dataclass
class FusedBlock:
    """A detected fusible inverted residual block."""
    start_idx: int          # Starting layer index in the model
    end_idx: int            # Ending layer index (inclusive)
    out_h: int              # Output spatial height
    out_w: int              # Output spatial width
    c_in: int               # Block input channels
    c_mid: int              # Expanded intermediate channels
    c_out: int              # Block output channels
    dw_kernel: int          # DW kernel size (typically 3)
    dw_stride: int          # DW stride (must be 1 for fusion)


@dataclass
class FusedTiling:
    """Tiling result for a fused block."""
    tile_h: int
    tile_w: int
    spatial_tiles: int      # Total number of spatial tiles
    oc1_tile: int           # Conv1×1 #1 output channel tile
    oc1_groups: int         # Conv1×1 #1 OC groups
    oc2_tile: int           # Conv1×1 #2 output channel tile
    oc2_groups: int         # Conv1×1 #2 OC groups
    bank_a_bytes: int       # Peak Bank A usage
    bank_b_bytes: int       # Peak Bank B usage
    feasible: bool          # Whether fusion is feasible


def detect_fusible_blocks(layers) -> List[FusedBlock]:
    """Detect fusible inverted residual blocks in a layer list.

    Pattern: Conv1×1(expand) → DW_kxk(stride=1) → Conv1×1(project)

    Args:
        layers: List of LayerDesc (from perf_model) or dicts with keys:
                op_type, in_c, out_c, kernel_h, kernel_w, stride_h,
                out_h, out_w

    Returns:
        List of FusedBlock instances (non-overlapping, greedy match)
    """
    blocks = []
    n = len(layers)
    i = 0

    while i <= n - 3:
        la, lb, lc = layers[i], layers[i + 1], layers[i + 2]

        # Get attributes (support both dataclass and dict)
        def _get(layer, key, default=None):
            if hasattr(layer, key):
                return getattr(layer, key)
            elif isinstance(layer, dict):
                return layer.get(key, default)
            return default

        # Check pattern: Conv1×1 → DW → Conv1×1
        a_op = _get(la, 'op_type')
        b_op = _get(lb, 'op_type')
        c_op = _get(lc, 'op_type')

        if a_op != 'conv' or b_op != 'dw' or c_op != 'conv':
            i += 1
            continue

        # Conv1×1 #1: kernel must be 1×1
        a_kh = _get(la, 'kernel_h', 1)
        a_kw = _get(la, 'kernel_w', 1)
        if a_kh != 1 or a_kw != 1:
            i += 1
            continue

        # DW: stride must be 1 (stride-2 DW breaks spatial alignment)
        b_sh = _get(lb, 'stride_h', 1)
        b_sw = _get(lb, 'stride_w', 1)
        if b_sh != 1 or b_sw != 1:
            i += 1
            continue

        # Conv1×1 #2: kernel must be 1×1
        c_kh = _get(lc, 'kernel_h', 1)
        c_kw = _get(lc, 'kernel_w', 1)
        if c_kh != 1 or c_kw != 1:
            i += 1
            continue

        # Channel alignment: Conv1#1.out_c == DW.in_c == DW.out_c == Conv1#2.in_c
        a_out_c = _get(la, 'out_c')
        b_in_c = _get(lb, 'in_c')
        b_out_c = _get(lb, 'out_c')
        c_in_c = _get(lc, 'in_c')
        if not (a_out_c == b_in_c == b_out_c == c_in_c):
            i += 1
            continue

        # Spatial alignment: all same spatial dims (since DW stride=1)
        a_out_h = _get(la, 'out_h')
        b_out_h = _get(lb, 'out_h')
        c_out_h = _get(lc, 'out_h')
        a_out_w = _get(la, 'out_w')
        b_out_w = _get(lb, 'out_w')
        c_out_w = _get(lc, 'out_w')
        if not (a_out_h == b_out_h == c_out_h):
            i += 1
            continue
        if not (a_out_w == b_out_w == c_out_w):
            i += 1
            continue

        # Valid block detected
        block = FusedBlock(
            start_idx=i,
            end_idx=i + 2,
            out_h=c_out_h,
            out_w=c_out_w,
            c_in=_get(la, 'in_c'),
            c_mid=a_out_c,
            c_out=_get(lc, 'out_c'),
            dw_kernel=_get(lb, 'kernel_h', 3),
            dw_stride=1,
        )
        blocks.append(block)
        i += 3  # Skip past this block (non-overlapping)

    return blocks


def compute_fused_tiling(block: FusedBlock, elem_size: int = 1) -> FusedTiling:
    """Compute optimal tile size for fused block execution.

    SRAM constraint model (2-bank alternating):
      Phase 1: Bank_A = (th+halo)×(tw+halo) × c_in    [input from DRAM]
               Bank_B = (th+halo)×(tw+halo) × c_mid   [Conv1×1 #1 output]
      Phase 2: Bank_B = DW input (same data)
               Bank_A = th×tw × c_mid                  [DW output]
      Phase 3: Bank_A = Conv1×1 #2 input (same data)
               Bank_B = th×tw × c_out                  [final output to DRAM]

    Bank constraint:
      Bank_A_max = max((th+halo)×(tw+halo) × c_in, th×tw × c_mid) × elem ≤ ACT_BANK_SIZE
      Bank_B_max = max((th+halo)×(tw+halo) × c_mid, th×tw × c_out) × elem ≤ ACT_BANK_SIZE

    Where halo = dw_kernel - 1 (typically 2 for DW 3×3 with pad=1)

    Strategy: find the largest square tile that fits, then try expanding H or W
    independently if the feature map is non-square.
    """
    halo = block.dw_kernel - 1  # pad on each side for DW
    c_in = block.c_in
    c_mid = block.c_mid
    c_out = block.c_out
    out_h = block.out_h
    out_w = block.out_w

    def _fits(th, tw):
        """Check if tile th×tw fits in SRAM banks."""
        ih = th + halo
        iw = tw + halo
        bank_a = max(ih * iw * c_in, th * tw * c_mid) * elem_size
        bank_b = max(ih * iw * c_mid, th * tw * c_out) * elem_size
        return bank_a <= ACT_BANK_SIZE and bank_b <= ACT_BANK_SIZE

    # Find largest square tile that fits
    best_t = 0
    max_dim = max(out_h, out_w)
    for t in range(1, max_dim + 1):
        if _fits(t, t):
            best_t = t
        else:
            break

    if best_t == 0:
        # Fusion not feasible with current SRAM
        return FusedTiling(
            tile_h=0, tile_w=0, spatial_tiles=0,
            oc1_tile=0, oc1_groups=0,
            oc2_tile=0, oc2_groups=0,
            bank_a_bytes=0, bank_b_bytes=0,
            feasible=False,
        )

    # Start from square tile, try to expand along the longer dimension
    best_th = best_t
    best_tw = best_t

    # Try expanding tile_h if out_h > out_w (fewer H tiles preferred)
    for th in range(best_t + 1, out_h + 1):
        if _fits(th, best_tw):
            best_th = th
        else:
            break

    # Try expanding tile_w if out_w > out_h
    for tw in range(best_t + 1, out_w + 1):
        if _fits(best_th, tw):
            best_tw = tw
        else:
            break

    # OC tiling for Conv1×1 #1 (weight = c_in × tile_oc1 bytes)
    weight_per_oc1 = c_in * elem_size  # 1×1 conv: weight_per_oc = in_c
    oc1_tile = min(c_mid, WEIGHT_BUF_SIZE // weight_per_oc1)
    oc1_tile = min(oc1_tile, PARAM_SRAM_MAX_CH)
    oc1_groups = math.ceil(c_mid / oc1_tile)

    # OC tiling for Conv1×1 #2 (weight = c_mid × tile_oc2 bytes)
    weight_per_oc2 = c_mid * elem_size
    oc2_tile = min(c_out, WEIGHT_BUF_SIZE // weight_per_oc2)
    oc2_tile = min(oc2_tile, PARAM_SRAM_MAX_CH)
    oc2_groups = math.ceil(c_out / oc2_tile)

    # Spatial tile count (H and W independently)
    tiles_h = math.ceil(out_h / best_th)
    tiles_w = math.ceil(out_w / best_tw)
    spatial_tiles = tiles_h * tiles_w

    # Peak bank usage at best tile
    ih = best_th + halo
    iw = best_tw + halo
    bank_a_bytes = max(ih * iw * c_in, best_th * best_tw * c_mid) * elem_size
    bank_b_bytes = max(ih * iw * c_mid, best_th * best_tw * c_out) * elem_size

    return FusedTiling(
        tile_h=best_th,
        tile_w=best_tw,
        spatial_tiles=spatial_tiles,
        oc1_tile=oc1_tile,
        oc1_groups=oc1_groups,
        oc2_tile=oc2_tile,
        oc2_groups=oc2_groups,
        bank_a_bytes=bank_a_bytes,
        bank_b_bytes=bank_b_bytes,
        feasible=True,
    )


def estimate_fusion_savings(block: FusedBlock, tiling: FusedTiling,
                            elem_size: int = 1) -> dict:
    """Estimate DMA byte savings from fusing a block.

    Without fusion (3 independent layers):
      Load:  input + mid1_read + mid2_read = S²×(c_in + c_mid + c_mid)
      Store: mid1_write + mid2_write + output = S²×(c_mid + c_mid + c_out)
      Total: S²×(c_in + 4×c_mid + c_out)

    With fusion:
      Load:  input only = S²×c_in + weight reloads (per spatial tile)
      Store: output only = S²×c_out
      Total: S²×(c_in + c_out) + weight_reload_overhead

    Note: Weight reloads happen per spatial tile for Conv1×1 (if OC groups > 1),
    but are small compared to activation tensors.
    """
    if not tiling.feasible:
        return {'feasible': False}

    s2 = block.out_h * block.out_w
    c_in, c_mid, c_out = block.c_in, block.c_mid, block.c_out

    # Activation DMA without fusion
    unfused_act_bytes = s2 * (c_in + 4 * c_mid + c_out) * elem_size

    # Activation DMA with fusion (only block boundary)
    fused_act_bytes = s2 * (c_in + c_out) * elem_size

    # Weight reload overhead with fusion:
    # Per spatial tile, we reload weights for all 3 sub-layers
    # Conv1×1 #1: c_in × c_mid (once per spatial tile, or per OC group)
    # DW 3×3:     dw_kernel² × c_mid (once per spatial tile)
    # Conv1×1 #2: c_mid × c_out (once per spatial tile, or per OC group)
    dw_k = block.dw_kernel
    w1_bytes = c_in * c_mid * elem_size
    w2_bytes = dw_k * dw_k * c_mid * elem_size
    w3_bytes = c_mid * c_out * elem_size

    # Without fusion: weights loaded once per layer execution (same total)
    unfused_weight_bytes = w1_bytes + w2_bytes + w3_bytes

    # With fusion: weights reloaded per spatial tile (3 sub-layers per tile)
    # But Conv1×1 weights may need OC-group passes within each spatial tile
    fused_weight_bytes = tiling.spatial_tiles * (w1_bytes + w2_bytes + w3_bytes)

    # Total comparison
    unfused_total = unfused_act_bytes + unfused_weight_bytes
    fused_total = fused_act_bytes + fused_weight_bytes

    act_saving_bytes = unfused_act_bytes - fused_act_bytes
    act_saving_pct = act_saving_bytes / unfused_act_bytes * 100 if unfused_act_bytes > 0 else 0

    # Net saving (activation saving - weight reload overhead)
    net_saving_bytes = unfused_total - fused_total
    net_saving_pct = net_saving_bytes / unfused_total * 100 if unfused_total > 0 else 0

    return {
        'feasible': True,
        'unfused_act_bytes': unfused_act_bytes,
        'fused_act_bytes': fused_act_bytes,
        'act_saving_bytes': act_saving_bytes,
        'act_saving_pct': act_saving_pct,
        'unfused_weight_bytes': unfused_weight_bytes,
        'fused_weight_bytes': fused_weight_bytes,
        'unfused_total_bytes': unfused_total,
        'fused_total_bytes': fused_total,
        'net_saving_bytes': net_saving_bytes,
        'net_saving_pct': net_saving_pct,
    }


# ─── Self-Test ───

def _m110_layers():
    """M110 MobileNetV2-style layer list for testing (simplified)."""

    @dataclass
    class L:
        op_type: str
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

    layers = [
        # Stem
        L('conv', 224, 224, 3, 112, 112, 56, 3, 3, 2, 2),
        L('dw', 112, 112, 56, 112, 112, 56, 3, 3, 1, 1),
        L('conv', 112, 112, 56, 112, 112, 56, 1, 1),
        # Block: expand → DW → project
        L('conv', 112, 112, 56, 112, 112, 112, 1, 1),
        L('dw', 112, 112, 112, 56, 56, 112, 3, 3, 2, 2),  # stride-2, NOT fusible
        L('conv', 56, 56, 112, 56, 56, 56, 1, 1),
    ]

    # 4 fusible blocks at 56×56
    for _ in range(4):
        layers.append(L('conv', 56, 56, 56, 56, 56, 112, 1, 1))   # expand
        layers.append(L('dw', 56, 56, 112, 56, 56, 112, 3, 3))    # DW s1
        layers.append(L('conv', 56, 56, 112, 56, 56, 56, 1, 1))   # project
        layers.append(L('add', 56, 56, 56, 56, 56, 56))           # residual

    # Stride-2 transition
    layers.append(L('conv', 56, 56, 56, 56, 56, 224, 1, 1))
    layers.append(L('dw', 56, 56, 224, 28, 28, 224, 3, 3, 2, 2))  # stride-2, NOT fusible
    layers.append(L('conv', 28, 28, 224, 28, 28, 96, 1, 1))

    # 6 fusible blocks at 28×28
    for _ in range(6):
        layers.append(L('conv', 28, 28, 96, 28, 28, 192, 1, 1))
        layers.append(L('dw', 28, 28, 192, 28, 28, 192, 3, 3))
        layers.append(L('conv', 28, 28, 192, 28, 28, 96, 1, 1))
        layers.append(L('add', 28, 28, 96, 28, 28, 96))

    # Stride-2 transition
    layers.append(L('conv', 28, 28, 96, 28, 28, 384, 1, 1))
    layers.append(L('dw', 28, 28, 384, 14, 14, 384, 3, 3, 2, 2))  # NOT fusible
    layers.append(L('conv', 14, 14, 384, 14, 14, 96, 1, 1))

    # 2 fusible blocks at 14×14
    for _ in range(2):
        layers.append(L('conv', 14, 14, 96, 14, 14, 192, 1, 1))
        layers.append(L('dw', 14, 14, 192, 14, 14, 192, 3, 3))
        layers.append(L('conv', 14, 14, 192, 14, 14, 96, 1, 1))
        layers.append(L('add', 14, 14, 96, 14, 14, 96))

    # Head
    layers.append(L('conv', 14, 14, 96, 14, 14, 512, 1, 1))
    layers.append(L('dw', 14, 14, 512, 1, 1, 512, 14, 14, 14, 14))
    layers.append(L('conv', 1, 1, 512, 1, 1, 512, 1, 1))

    return layers


def _resnet18_layers():
    """ResNet-18 half-width layers for testing (no DW, should find 0 blocks)."""

    @dataclass
    class L:
        op_type: str
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

    return [
        L('conv', 224, 224, 3, 112, 112, 32, 7, 7, 2, 2),
        L('pool', 112, 112, 32, 56, 56, 32, 3, 3, 2, 2),
        L('conv', 56, 56, 32, 56, 56, 32, 3, 3),
        L('conv', 56, 56, 32, 56, 56, 32, 3, 3),
        L('add', 56, 56, 32, 56, 56, 32),
        L('conv', 56, 56, 32, 56, 56, 32, 3, 3),
        L('conv', 56, 56, 32, 56, 56, 32, 3, 3),
        L('add', 56, 56, 32, 56, 56, 32),
        L('conv', 56, 56, 32, 28, 28, 64, 3, 3, 2, 2),
        L('conv', 28, 28, 64, 28, 28, 64, 3, 3),
        L('add', 28, 28, 64, 28, 28, 64),
        L('conv', 28, 28, 64, 28, 28, 64, 3, 3),
        L('conv', 28, 28, 64, 28, 28, 64, 3, 3),
        L('add', 28, 28, 64, 28, 28, 64),
    ]


if __name__ == '__main__':
    print("=" * 70)
    print("Open-NPU Layer Fusion: Block Detection & Fused Tiling Calculator")
    print("=" * 70)

    # ─── Test 1: M110 MobileNetV2 ───
    print("\n--- M110 (MobileNetV2-style, 63 layers) ---")
    m110 = _m110_layers()
    blocks = detect_fusible_blocks(m110)
    print(f"  Total layers: {len(m110)}")
    print(f"  Detected fusible blocks: {len(blocks)}")

    total_unfused_dma = 0
    total_fused_dma = 0

    print(f"\n  {'Block':>5} | {'Layers':>8} | {'Spatial':>7} | {'Channels':>14} | "
          f"{'Tile':>6} | {'SpatTiles':>9} | {'BankA':>6} | {'BankB':>6} | "
          f"{'DMA Save':>8}")
    print("  " + "-" * 95)

    for idx, block in enumerate(blocks):
        tiling = compute_fused_tiling(block)
        savings = estimate_fusion_savings(block, tiling)

        if tiling.feasible:
            total_unfused_dma += savings['unfused_total_bytes']
            total_fused_dma += savings['fused_total_bytes']
            ch_str = f"{block.c_in}→{block.c_mid}→{block.c_out}"
            print(f"  {idx:>5} | {block.start_idx:>2}-{block.end_idx:<3} | "
                  f"{block.out_h:>3}×{block.out_w:<3} | {ch_str:>14} | "
                  f"{tiling.tile_h:>2}×{tiling.tile_w:<2} | "
                  f"{tiling.spatial_tiles:>9} | "
                  f"{tiling.bank_a_bytes//1024:>4}KB | "
                  f"{tiling.bank_b_bytes//1024:>4}KB | "
                  f"{savings['act_saving_pct']:>6.0f}%")

    if total_unfused_dma > 0:
        overall_pct = (total_unfused_dma - total_fused_dma) / total_unfused_dma * 100
        print(f"\n  Total DMA (unfused): {total_unfused_dma / 1024:.0f} KB")
        print(f"  Total DMA (fused):   {total_fused_dma / 1024:.0f} KB")
        print(f"  Overall DMA saving:  {overall_pct:.1f}%")

    # ─── Test 2: ResNet-18 (no DW blocks, should detect 0) ───
    print("\n--- ResNet-18 (no DW blocks) ---")
    resnet = _resnet18_layers()
    blocks_r = detect_fusible_blocks(resnet)
    print(f"  Total layers: {len(resnet)}")
    print(f"  Detected fusible blocks: {len(blocks_r)} (expected: 0)")
    assert len(blocks_r) == 0, "False positive! ResNet-18 has no inverted residual blocks"
    print("  PASS: No false positives")

    # ─── Test 3: Non-square feature map (H != W) ───
    print("\n--- Non-square (40×80) fusible block ---")

    @dataclass
    class LNS:
        op_type: str
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

    nonsquare = [
        LNS('conv', 40, 80, 32, 40, 80, 64, 1, 1),   # expand
        LNS('dw', 40, 80, 64, 40, 80, 64, 3, 3),     # DW s1
        LNS('conv', 40, 80, 64, 40, 80, 32, 1, 1),   # project
    ]
    blocks_ns = detect_fusible_blocks(nonsquare)
    assert len(blocks_ns) == 1, f"Expected 1 non-square block, got {len(blocks_ns)}"
    b = blocks_ns[0]
    assert b.out_h == 40 and b.out_w == 80, f"Expected 40×80, got {b.out_h}×{b.out_w}"
    t = compute_fused_tiling(b)
    assert t.feasible, "Non-square block should be feasible"
    tiles_h = math.ceil(40 / t.tile_h)
    tiles_w = math.ceil(80 / t.tile_w)
    assert t.spatial_tiles == tiles_h * tiles_w, "Tile count must use H and W independently"
    print(f"  Block: 40×80, c_in=32, c_mid=64, c_out=32")
    print(f"  Tile: {t.tile_h}×{t.tile_w}, spatial_tiles={t.spatial_tiles} ({tiles_h}H × {tiles_w}W)")
    print(f"  PASS: Non-square handled correctly")

    print("\n" + "=" * 70)
    print("Self-test PASSED")
    print("=" * 70)
