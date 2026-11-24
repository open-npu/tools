"""
Open-NPU Layer Fusion: Block Detection and Fused Tiling Calculator

Detects fusible synth_dw-style Inverted Residual Blocks (Conv1×1 → DW3×3 → Conv1×1)
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
from hw_config import HWConfig, DEFAULT_HW


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


def compute_fused_tiling(block: FusedBlock, elem_size: int = 1,
                         profile: HWConfig = None) -> FusedTiling:
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

    if profile is None:
        profile = DEFAULT_HW
    act_bank_size = profile.act_bank_size
    weight_buf_size = profile.weight_buf_size
    param_sram_max_ch = profile.param_sram_max_ch

    def _fits(th, tw):
        """Check if tile th×tw fits in SRAM banks."""
        ih = th + halo
        iw = tw + halo
        bank_a = max(ih * iw * c_in, th * tw * c_mid) * elem_size
        bank_b = max(ih * iw * c_mid, th * tw * c_out) * elem_size
        return bank_a <= act_bank_size and bank_b <= act_bank_size

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
    oc1_tile = min(c_mid, weight_buf_size // weight_per_oc1)
    oc1_tile = min(oc1_tile, param_sram_max_ch)
    oc1_groups = math.ceil(c_mid / oc1_tile)

    # OC tiling for Conv1×1 #2 (weight = c_mid × tile_oc2 bytes)
    weight_per_oc2 = c_mid * elem_size
    oc2_tile = min(c_out, weight_buf_size // weight_per_oc2)
    oc2_tile = min(oc2_tile, param_sram_max_ch)
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


# sched_ctrl bits (must match model_packer.py)
_FUSE_START = 1 << 1
_FUSE_MID = 1 << 2
_FUSE_END = 1 << 3
_DB_EN = 1 << 0
_PTS = 1 << 4
_FUSE_STRIP = 1 << 5
_SUPER_BLOCK = 1 << 7


def stamp_fused_tiling(layers, block: FusedBlock, tiling: FusedTiling,
                       include_add: bool = True, allow_tiled: bool = False) -> int:
    """Write one tile grid onto every layer of an IR block.

    START / MID / END (and a following residual Add with the same spatial
    size) share tile_h/w/num. DMA double-buffer is cleared so fusion can
    use the two native act regions. If the whole map fits in one tile,
    tiles are cleared (tile_h=0) so the existing untiled skip-DMA path
    can fire. Tiled fusion is rejected by default because the RTL executes
    every tile of START before MID; it cannot keep one tile resident through
    START→MID→END yet.

    Returns the number of layers stamped.
    """
    if not tiling.feasible or (tiling.spatial_tiles > 1 and not allow_tiled):
        return 0

    idxs = [block.start_idx, block.start_idx + 1, block.end_idx]
    fuse_bits = [_FUSE_START, _FUSE_MID, _FUSE_END]

    add_idx = block.end_idx + 1
    if include_add and add_idx < len(layers):
        add = layers[add_idx]
        add_op = add.op_type if hasattr(add, 'op_type') else add.get('op_type')
        add_oh = add.out_h if hasattr(add, 'out_h') else add.get('out_h')
        add_ow = add.out_w if hasattr(add, 'out_w') else add.get('out_w')
        add_oc = add.out_c if hasattr(add, 'out_c') else add.get('out_c')
        if add_op in ('add', 4) and add_oh == block.out_h and add_ow == block.out_w \
                and add_oc == block.c_out:
            # The project layer becomes another middle stage so it skips its
            # DDR store; Add is the block end and stores the final result.
            fuse_bits[-1] = _FUSE_MID
            idxs.append(add_idx)
            fuse_bits.append(_FUSE_END)

    # One spatial tile → present as untiled so RTL skip_act_load (tile_h==0) works.
    if tiling.spatial_tiles <= 1:
        th = tw = nh = nw = 0
    else:
        th = tiling.tile_h
        tw = tiling.tile_w
        nh = math.ceil(block.out_h / th)
        nw = math.ceil(block.out_w / tw)

    def _set(obj, key, val):
        if hasattr(obj, key):
            setattr(obj, key, val)
        else:
            obj[key] = val

    def _get(obj, key, default=0):
        if hasattr(obj, key):
            return getattr(obj, key)
        return obj.get(key, default)

    for idx, fbit in zip(idxs, fuse_bits):
        lyr = layers[idx]
        sched = int(_get(lyr, 'sched_ctrl', 0))
        sched &= ~(_DB_EN | _FUSE_START | _FUSE_MID | _FUSE_END | _PTS)
        sched |= fbit
        # END (and Add) still write the block output; PTS only if spatially tiled.
        if th > 0 and fbit in (0, _FUSE_END):
            sched |= _PTS
        _set(lyr, 'sched_ctrl', sched)
        _set(lyr, 'tile_h', th)
        _set(lyr, 'tile_w', tw)
        _set(lyr, 'tile_num_h', nh)
        _set(lyr, 'tile_num_w', nw)

    return len(idxs)


def _lf_get(obj, key, default=None):
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _lf_set(obj, key, val):
    if hasattr(obj, key):
        setattr(obj, key, val)
    else:
        obj[key] = val


def _collect_referenced(layers, must_store_indices=None) -> set:
    ref = set(must_store_indices or ())
    for lyr in layers:
        for key in ('input_src', 'residual_src'):
            v = _lf_get(lyr, key, -1)
            if v is not None and int(v) >= 0:
                ref.add(int(v))
    return ref


def apply_super_block_fusion(layers, profile: HWConfig = None,
                             must_store_indices=None) -> List[tuple]:
    """Detect and mark general INT8 stride-1 Conv1x1 -> DW3x3 -> Conv1x1 super-blocks.

    Works identically on LayerConfig instances (in onnx_converter) and metadata
    dicts (in gen_soc_test). Skips blocks that already fit completely untiled in
    SRAM (since untiled FUSE_START/MID/END executes with zero halo overhead).
    """
    if profile is None:
        profile = DEFAULT_HW
    ifm_bytes = profile.act_bank_size
    ofm_bytes = profile.ofm_bank_size
    wgt_words_cap = profile.weight_buf_size // 4
    param_words_cap = profile.param_sram_size // 4
    untiled_cap = min(ifm_bytes, ofm_bytes)

    referenced = _collect_referenced(layers, must_store_indices)
    marked = []
    n = len(layers)
    i = 1  # i >= 1 guarantees NHWC input from a preceding NPU layer in DDR
    while i + 2 < n:
        a, b, c = layers[i], layers[i + 1], layers[i + 2]
        if i in referenced or (i + 1) in referenced:
            i += 1
            continue
        b_src = int(_lf_get(b, 'input_src', -1))
        c_src = int(_lf_get(c, 'input_src', -1))
        if b_src not in (-1, i) or c_src not in (-1, i + 1):
            i += 1
            continue

        # Must be INT8 Conv1x1(s1,p0) -> DW3x3(s1,p1) -> Conv1x1(s1,p0)
        if (_lf_get(a, 'op_type') != 0 or _lf_get(b, 'op_type') != 1 or _lf_get(c, 'op_type') != 0):
            i += 1
            continue
        if any(int(_lf_get(x, 'data_type', 0)) != 0 for x in (a, b, c)):
            i += 1
            continue
        if any(int(_lf_get(x, 'dilation_h', 1)) != 1 or int(_lf_get(x, 'dilation_w', 1)) != 1
               for x in (a, b, c)):
            i += 1
            continue
        if any(int(_lf_get(x, 'wgt_layout', 1)) != 1 for x in (a, b, c)):
            i += 1
            continue

        h = int(_lf_get(a, 'in_h', 0))
        w = int(_lf_get(a, 'in_w', 0))
        in_c = int(_lf_get(a, 'in_c', 0))
        mid_c = int(_lf_get(a, 'out_c', 0))
        out_c = int(_lf_get(c, 'out_c', 0))

        b_pad_ok = (int(_lf_get(b, 'pad_top', 0)) == 1
                    and int(_lf_get(b, 'pad_left', 0)) == 1
                    and int(_lf_get(b, 'pad_bottom', 1)) == 1
                    and int(_lf_get(b, 'pad_right', 1)) == 1)
        a_pad_ok = (int(_lf_get(a, 'pad_top', 0)) == 0
                    and int(_lf_get(a, 'pad_left', 0)) == 0
                    and int(_lf_get(a, 'pad_bottom', 0)) == 0
                    and int(_lf_get(a, 'pad_right', 0)) == 0)
        c_pad_ok = (int(_lf_get(c, 'pad_top', 0)) == 0
                    and int(_lf_get(c, 'pad_left', 0)) == 0
                    and int(_lf_get(c, 'pad_bottom', 0)) == 0
                    and int(_lf_get(c, 'pad_right', 0)) == 0)

        geom_ok = (
            h > 0 and w > 0
            and in_c >= 4 and (in_c % 4) == 0
            and mid_c >= 4 and (mid_c % 4) == 0
            and out_c >= 4 and (out_c % 4) == 0
            and (_lf_get(a, 'out_h'), _lf_get(a, 'out_w')) == (h, w)
            and (_lf_get(b, 'in_h'), _lf_get(b, 'in_w'), _lf_get(b, 'out_h'), _lf_get(b, 'out_w')) == (h, w, h, w)
            and (_lf_get(c, 'in_h'), _lf_get(c, 'in_w'), _lf_get(c, 'out_h'), _lf_get(c, 'out_w')) == (h, w, h, w)
            and int(_lf_get(b, 'in_c')) == mid_c and int(_lf_get(b, 'out_c')) == mid_c
            and int(_lf_get(c, 'in_c')) == mid_c
            and (_lf_get(a, 'kernel_h'), _lf_get(a, 'kernel_w'), _lf_get(a, 'stride_h'), _lf_get(a, 'stride_w')) == (1, 1, 1, 1)
            and (_lf_get(b, 'kernel_h'), _lf_get(b, 'kernel_w'), _lf_get(b, 'stride_h'), _lf_get(b, 'stride_w')) == (3, 3, 1, 1)
            and (_lf_get(c, 'kernel_h'), _lf_get(c, 'kernel_w'), _lf_get(c, 'stride_h'), _lf_get(c, 'stride_w')) == (1, 1, 1, 1)
            and a_pad_ok and b_pad_ok and c_pad_ok
        )
        if not geom_ok:
            i += 1
            continue

        # Skip blocks that fit completely untiled in SRAM (handled with zero halo
        # by apply_untiled_sram_fusion).
        all_untiled = (
            int(_lf_get(a, 'tile_h', 0)) == 0
            and int(_lf_get(b, 'tile_h', 0)) == 0
            and int(_lf_get(c, 'tile_h', 0)) == 0
            and max(h * w * in_c, h * w * mid_c, h * w * out_c) <= untiled_cap
        )
        if all_untiled:
            i += 1
            continue

        # Resident weight + param SRAM capacity across all 3 slots
        wgt_words = ((in_c * mid_c + 3) // 4
                     + (9 * mid_c + 3) // 4
                     + (mid_c * out_c + 3) // 4)
        param_words = (mid_c + mid_c + out_c) * 4
        if wgt_words > wgt_words_cap or param_words > param_words_cap:
            i += 1
            continue

        # Search optimal (th, tw) satisfying S_SUPER_VALIDATE IFM/OFM caps
        best_cost = None
        best_hw = None
        for th in range(1, h + 1):
            tiles_h = (h + th - 1) // th
            if (tiles_h - 1) * th >= h:
                continue
            for tw in range(1, w + 1):
                tiles_w = (w + tw - 1) // tw
                if (tiles_w - 1) * tw >= w:
                    continue
                ifm_peak = max((th + 2) * (tw + 2) * in_c, th * tw * mid_c)
                ofm_peak = max((th + 2) * (tw + 2) * mid_c, th * tw * out_c)
                if ifm_peak > ifm_bytes or ofm_peak > ofm_bytes:
                    break  # both peaks grow monotonically with tw
                ntiles = tiles_h * tiles_w
                if ntiles <= 1:
                    # Whole map fits super-block SRAM in 1 tile, or search keeps going
                    pass
                halo_pixels = ntiles * (th + 2) * (tw + 2)
                dma_rows = ntiles * (th + 2)
                cost = (ntiles, halo_pixels, dma_rows, abs(th - tw))
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_hw = (th, tw, tiles_h, tiles_w)

        if best_hw is None:
            i += 1
            continue

        th, tw, nh, nw = best_hw
        for idx_sub, x in enumerate((a, b, c)):
            _lf_set(x, 'tile_h', th)
            _lf_set(x, 'tile_w', tw)
            _lf_set(x, 'tile_num_h', nh)
            _lf_set(x, 'tile_num_w', nw)
            sched = int(_lf_get(x, 'sched_ctrl', 0)) & ~0xFF
            sched |= _PTS
            if idx_sub == 0:
                sched |= _SUPER_BLOCK
            _lf_set(x, 'sched_ctrl', sched)
            if isinstance(x, dict):
                x['store_mode'] = 1
                in_th = (th - 1) * int(x['stride_h']) + int(x['kernel_h'])
                in_tw = (tw - 1) * int(x['stride_w']) + int(x['kernel_w'])
                x['tile_in_size'] = ((in_th * in_tw * int(x['in_c']) + 3) // 4) * 4
                x['tile_out_size'] = ((th * tw * int(x['out_c']) + 3) // 4) * 4
                x['row_cfg'] = (th << 16) | ((tw * int(x['out_c']) + 3) // 4)
        marked.append((i, th, tw, nh, nw))
        i += 3

    return marked


def compute_conv_pool_strip_params(conv, pool, profile: HWConfig = None):
    """Return (pool_strip_h, producer_h, strip_count) if Conv->Pool strip fusion
    is legal and bandwidth-profitable, else None.
    """
    if profile is None:
        profile = DEFAULT_HW
    ifm_bytes = profile.act_bank_size
    ofm_bytes = profile.ofm_bank_size
    wgt_bytes_cap = profile.weight_buf_size
    param_max_ch = profile.param_sram_max_ch
    untiled_cap = min(ifm_bytes, ofm_bytes)

    if int(_lf_get(conv, 'op_type', -1)) != 0 or int(_lf_get(pool, 'op_type', -1)) != 3:
        return None
    if int(_lf_get(conv, 'data_type', 0)) != 0 or int(_lf_get(pool, 'data_type', 0)) != 0:
        return None
    if int(_lf_get(conv, 'wgt_layout', 1)) != 1:
        return None

    in_h = int(_lf_get(conv, 'in_h', 0))
    in_w = int(_lf_get(conv, 'in_w', 0))
    in_c = int(_lf_get(conv, 'in_c', 0))
    out_h = int(_lf_get(conv, 'out_h', 0))
    out_w = int(_lf_get(conv, 'out_w', 0))
    out_c = int(_lf_get(conv, 'out_c', 0))
    pool_out_h = int(_lf_get(pool, 'out_h', 0))
    pool_out_w = int(_lf_get(pool, 'out_w', 0))
    pool_out_c = int(_lf_get(pool, 'out_c', 0))

    if (out_h, out_w, out_c) != (int(_lf_get(pool, 'in_h', 0)),
                                 int(_lf_get(pool, 'in_w', 0)),
                                 int(_lf_get(pool, 'in_c', 0))):
        return None

    # Parse pool kernel/stride/global from either LayerConfig or metadata dict
    if hasattr(pool, 'pool_h'):
        pool_h = int(pool.pool_h)
        pool_w = int(pool.pool_w)
        pool_sh = int(pool.pool_stride_h)
        global_pool = int(pool.global_pool)
    else:
        pcfg = int(pool.get('pool_cfg', 0))
        pool_h = (pcfg >> 4) & 0xF
        pool_w = (pcfg >> 8) & 0xF
        pool_sh = (pcfg >> 12) & 0xF
        global_pool = (pcfg >> 20) & 0x1

    if global_pool or pool_h == 0 or pool_w == 0 or pool_sh == 0 or pool_out_h == 0:
        return None

    # Direct NHWC row slicing requires word-aligned input rows and pool output rows
    if (in_w * in_c) % 4 != 0 or (pool_out_w * pool_out_c) % 4 != 0:
        return None

    # Skip pairs that already fit completely untiled in SRAM
    if (int(_lf_get(conv, 'tile_h', 0)) == 0
            and int(_lf_get(pool, 'tile_h', 0)) == 0
            and max(in_h * in_w * in_c, out_h * out_w * out_c,
                    pool_out_h * pool_out_w * pool_out_c) <= untiled_cap):
        return None

    kh = int(_lf_get(conv, 'kernel_h', 1))
    kw = int(_lf_get(conv, 'kernel_w', 1))
    dh = max(1, int(_lf_get(conv, 'dilation_h', 1)))
    sh = int(_lf_get(conv, 'stride_h', 1))
    kh_eff = (kh - 1) * dh + 1

    conv_wgt_bytes = out_c * kh * kw * in_c
    if conv_wgt_bytes > wgt_bytes_cap or out_c > param_max_ch:
        return None
    if int(_lf_get(conv, 'wgt_per_oc_words', 0)) != 0:
        return None

    pool_strip_h = 0
    for cand in range(1, pool_out_h + 1):
        prod_h = min(out_h, (cand - 1) * pool_sh + pool_h)
        conv_in_h = min(in_h, (prod_h - 1) * sh + kh_eff)
        conv_in_bytes = conv_in_h * in_w * in_c
        prod_bytes = prod_h * out_w * out_c
        pool_out_bytes = cand * pool_out_w * pool_out_c
        if (conv_in_bytes > ifm_bytes
                or prod_bytes > ofm_bytes
                or pool_out_bytes > ifm_bytes):
            break
        pool_strip_h = cand

    if pool_strip_h == 0:
        return None

    producer_h = min(out_h, (pool_strip_h - 1) * pool_sh + pool_h)
    strip_count = (pool_out_h + pool_strip_h - 1) // pool_strip_h
    conv_in_strip_h = min(in_h, (producer_h - 1) * sh + kh_eff)

    # Bandwidth cost model: ensure reloading Conv wgt/param per strip + halo rows
    # saves at least 25% of the Conv<->Pool intermediate DDR round-trip.
    inter_bytes = out_h * out_w * out_c
    reload_cost = (strip_count - 1) * (conv_wgt_bytes + out_c * 16)
    halo_rows = max(0, strip_count * conv_in_strip_h - in_h)
    halo_cost = halo_rows * in_w * in_c
    if reload_cost + halo_cost >= int(1.5 * inter_bytes):
        return None

    return (pool_strip_h, producer_h, strip_count)


def apply_conv_pool_strip_fusion(layers, profile: HWConfig = None,
                                 must_store_indices=None,
                                 stamp_firmware_fields: bool = False) -> List[tuple]:
    """Mark eligible Conv -> Pool pairs with SCHED_CTRL_FUSE_STRIP.

    When stamp_firmware_fields=False (in onnx_converter), only sets bit 5 of
    sched_ctrl while preserving valid NPU1 tile grids for CSIM.
    When stamp_firmware_fields=True (in gen_soc_test), writes the firmware
    strip parameters into tile_h/w/num_h/num_w and tile_in_size=0.
    """
    if profile is None:
        profile = DEFAULT_HW
    referenced = _collect_referenced(layers, must_store_indices)
    marked = []
    n = len(layers)
    i = 0
    while i + 1 < n:
        conv, pool = layers[i], layers[i + 1]
        c_sched = int(_lf_get(conv, 'sched_ctrl', 0))
        p_sched = int(_lf_get(pool, 'sched_ctrl', 0))
        if (c_sched & (_SUPER_BLOCK | _FUSE_START | _FUSE_MID | _FUSE_END)) or \
           (p_sched & (_SUPER_BLOCK | _FUSE_START | _FUSE_MID | _FUSE_END)):
            i += 1
            continue
        # If L0 already received legacy fuse_input slot_bytes > 0, keep it untouched
        if (c_sched & _FUSE_STRIP) and int(_lf_get(conv, 'tile_in_size', 0)) > 0:
            i += 2
            continue
        if i in referenced:
            i += 1
            continue
        p_src = int(_lf_get(pool, 'input_src', -1))
        if p_src not in (-1, i):
            i += 1
            continue

        params = compute_conv_pool_strip_params(conv, pool, profile=profile)
        if params is None:
            i += 1
            continue

        pool_strip_h, producer_h, strip_count = params
        _lf_set(conv, 'sched_ctrl', c_sched | _FUSE_STRIP)
        if stamp_firmware_fields:
            _lf_set(conv, 'tile_h', producer_h)
            _lf_set(conv, 'tile_w', int(_lf_get(conv, 'out_w')))
            _lf_set(conv, 'tile_num_h', strip_count)
            _lf_set(conv, 'tile_num_w', pool_strip_h)
            _lf_set(conv, 'tile_in_size', 0)
        marked.append((i, i + 1, strip_count, producer_h, pool_strip_h))
        i += 2

    return marked


def apply_untiled_sram_fusion(layers, profile: HWConfig = None,
                              must_store_indices=None) -> List[tuple]:
    """Fuse maximal chains of untiled layers through IFM/OFM SRAM role swap."""
    if profile is None:
        profile = DEFAULT_HW
    cap_bytes = min(profile.act_bank_size, profile.ofm_bank_size)
    referenced = _collect_referenced(layers, must_store_indices)

    def _can_link(k: int) -> bool:
        prod, cons = layers[k], layers[k + 1]
        if k in referenced:
            return False
        if k > 0 and (int(_lf_get(layers[k - 1], 'sched_ctrl', 0)) & _FUSE_STRIP):
            return False
        if k >= 2 and (int(_lf_get(layers[k - 2], 'sched_ctrl', 0)) & _SUPER_BLOCK):
            return False
        if k >= 1 and (int(_lf_get(layers[k - 1], 'sched_ctrl', 0)) & _SUPER_BLOCK):
            return False
        ps = int(_lf_get(prod, 'sched_ctrl', 0))
        cs = int(_lf_get(cons, 'sched_ctrl', 0))
        if (ps & (_SUPER_BLOCK | _FUSE_STRIP | _DB_EN)) or \
           (cs & (_SUPER_BLOCK | _FUSE_STRIP | _DB_EN)):
            return False
        if int(_lf_get(prod, 'tile_h', 0)) != 0 or int(_lf_get(cons, 'tile_h', 0)) != 0:
            return False
        p_op = int(_lf_get(prod, 'op_type', -1))
        c_op = int(_lf_get(cons, 'op_type', -1))
        if p_op not in (0, 1, 2, 3, 5) or c_op not in (0, 1, 2, 3, 5):
            return False
        c_src = int(_lf_get(cons, 'input_src', -1))
        if c_src not in (-1, k):
            return False
        p_dt = int(_lf_get(prod, 'data_type', 0))
        c_dt = int(_lf_get(cons, 'data_type', 0))
        if p_dt != c_dt:
            return False
        eb = 2 if p_dt else 1
        p_oh, p_ow, p_oc = int(_lf_get(prod, 'out_h')), int(_lf_get(prod, 'out_w')), int(_lf_get(prod, 'out_c'))
        c_ih, c_iw, c_ic = int(_lf_get(cons, 'in_h')), int(_lf_get(cons, 'in_w')), int(_lf_get(cons, 'in_c'))
        shape_match = ((p_oh, p_ow, p_oc) == (c_ih, c_iw, c_ic)
                       or (c_op == 2 and c_ih == 1 and c_iw == 1 and p_oh * p_ow * p_oc == c_ic))
        if not shape_match:
            return False
        p_in_b = int(_lf_get(prod, 'in_h')) * int(_lf_get(prod, 'in_w')) * int(_lf_get(prod, 'in_c')) * eb
        p_out_b = p_oh * p_ow * p_oc * eb
        c_in_b = c_ih * c_iw * c_ic * eb
        c_out_b = int(_lf_get(cons, 'out_h')) * int(_lf_get(cons, 'out_w')) * int(_lf_get(cons, 'out_c')) * eb
        if max(p_in_b, p_out_b, c_in_b, c_out_b) > cap_bytes:
            return False
        return True

    chains = []
    n = len(layers)
    i = 0
    while i + 1 < n:
        if not _can_link(i):
            i += 1
            continue
        end = i + 1
        while end + 1 < n and _can_link(end):
            end += 1
        for k in range(i, end + 1):
            sched = int(_lf_get(layers[k], 'sched_ctrl', 0)) & ~(_FUSE_START | _FUSE_MID | _FUSE_END)
            if k == i:
                sched |= _FUSE_START
            elif k == end:
                sched |= _FUSE_END
            else:
                sched |= _FUSE_MID
            _lf_set(layers[k], 'sched_ctrl', sched)
        chains.append(tuple(range(i, end + 1)))
        i = end + 1

    return chains


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

def _model_x():
    """model_x synth_dw-style layer list for testing (simplified)."""

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


def _synth_res_layers():
    """synth_res half-width layers for testing (no DW, should find 0 blocks)."""

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

    # ─── Test 1: model_x synth_dw ───
    print("\n--- model_x (synth_dw-style, 63 layers) ---")
    model_x = _model_x()
    blocks = detect_fusible_blocks(model_x)
    print(f"  Total layers: {len(model_x)}")
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

    # ─── Test 2: synth_res (no DW blocks, should detect 0) ───
    print("\n--- synth_res (no DW blocks) ---")
    synth_res = _synth_res_layers()
    blocks_r = detect_fusible_blocks(synth_res)
    print(f"  Total layers: {len(synth_res)}")
    print(f"  Detected fusible blocks: {len(blocks_r)} (expected: 0)")
    assert len(blocks_r) == 0, "False positive! synth_res has no inverted residual blocks"
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

    # ─── Test 4: stamp writes the same grid on START/MID/END + Add ───
    print("\n--- stamp_fused_tiling unifies the IR grid ---")
    stamped = [
        {'op_type': 'conv', 'out_h': 56, 'out_w': 56, 'out_c': 112,
         'tile_h': 1, 'tile_w': 109, 'tile_num_h': 112, 'tile_num_w': 2,
         'sched_ctrl': 1},
        {'op_type': 'dw', 'out_h': 56, 'out_w': 56, 'out_c': 112,
         'tile_h': 6, 'tile_w': 6, 'tile_num_h': 10, 'tile_num_w': 10,
         'sched_ctrl': 1},
        {'op_type': 'conv', 'out_h': 56, 'out_w': 56, 'out_c': 56,
         'tile_h': 8, 'tile_w': 9, 'tile_num_h': 7, 'tile_num_w': 7,
         'sched_ctrl': 1},
        {'op_type': 'add', 'out_h': 56, 'out_w': 56, 'out_c': 56,
         'tile_h': 9, 'tile_w': 9, 'tile_num_h': 7, 'tile_num_w': 7,
         'sched_ctrl': 1},
    ]
    blk = FusedBlock(start_idx=0, end_idx=2, out_h=56, out_w=56,
                     c_in=56, c_mid=112, c_out=56, dw_kernel=3, dw_stride=1)
    ft = compute_fused_tiling(blk)
    assert ft.feasible
    assert stamp_fused_tiling(stamped, blk, ft) == 0, \
        "tiled fusion must stay disabled until RTL executes tile-major blocks"
    n = stamp_fused_tiling(stamped, blk, ft, allow_tiled=True)
    assert n == 4, f"expected 4 stamped layers, got {n}"
    grids = [(s['tile_h'], s['tile_w'], s['tile_num_h'], s['tile_num_w']) for s in stamped]
    assert len(set(grids)) == 1, f"grids diverged: {grids}"
    assert all((s['sched_ctrl'] & _DB_EN) == 0 for s in stamped), "DB_EN must be off"
    assert stamped[0]['sched_ctrl'] & _FUSE_START
    assert stamped[1]['sched_ctrl'] & _FUSE_MID
    assert stamped[2]['sched_ctrl'] & _FUSE_MID
    assert stamped[3]['sched_ctrl'] & _FUSE_END
    print(f"  Shared grid: {grids[0][0]}×{grids[0][1]}  {grids[0][2]}×{grids[0][3]} tiles")
    print("  PASS: START/MID/END/Add share one grid, DB_EN cleared")

    print("\n" + "=" * 70)
    print("Self-test PASSED")
    print("=" * 70)
