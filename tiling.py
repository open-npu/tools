"""
Open-NPU Tiling Calculator

Computes optimal tile dimensions for each layer based on hardware SRAM constraints.

Hardware resources (0.2T variant):
  - Activation Bank A (input tile):  32 KB
  - Activation Bank B (output tile): 32 KB
  - Weight Buffer:                    64 KB
  - Parameter SRAM:                    8 KB (max 512 channels/tile)

SPDX-License-Identifier: Apache-2.0
"""

import math

# Hardware constants (match npu_config.h)
ACT_BANK_SIZE = 32 * 1024       # 32 KB per activation bank
WEIGHT_BUF_SIZE = 64 * 1024     # 64 KB weight buffer
PARAM_SRAM_MAX_CH = 512         # Parameter SRAM supports max 512 channels


def compute_tiling(op_type, in_h, in_w, in_c, out_h, out_w, out_c,
                   kernel_h=1, kernel_w=1, stride_h=1, stride_w=1,
                   dilation_h=1, dilation_w=1, elem_size=2,
                   double_buffer=False):
    """Compute optimal tile dimensions for a layer.

    Args:
        op_type: 'conv', 'dw', 'pool', 'fc', 'add', 'resize', 'concat'
        in_h, in_w, in_c: input dimensions
        out_h, out_w, out_c: output dimensions
        kernel_h, kernel_w: convolution kernel size
        stride_h, stride_w: stride
        dilation_h, dilation_w: dilation
        elem_size: bytes per element (1=INT8, 2=INT16)
        double_buffer: if True, use half-buffer sizes (ping/pong mode)

    Returns:
        dict with keys: tile_h, tile_w, tile_num_h, tile_num_w, tile_oc
              (tile_h/w = 0 and tile_num_h/w = 0 means no tiling needed)
    """
    # Effective buffer sizes (halved in double-buffer mode)
    act_bank = ACT_BANK_SIZE // 2 if double_buffer else ACT_BANK_SIZE
    weight_buf = WEIGHT_BUF_SIZE // 2 if double_buffer else WEIGHT_BUF_SIZE

    # FC and Concat don't do spatial tiling
    if op_type in ('fc', 'concat'):
        return {'tile_h': 0, 'tile_w': 0, 'tile_num_h': 0, 'tile_num_w': 0, 'tile_oc': out_c}

    # Effective kernel size (with dilation)
    kh_eff = (kernel_h - 1) * dilation_h + 1
    kw_eff = (kernel_w - 1) * dilation_w + 1

    # Compute tile_oc (output channel tiling due to weight buffer constraint)
    if op_type in ('conv',):
        weight_per_oc = kernel_h * kernel_w * in_c * elem_size
        if weight_per_oc > 0:
            tile_oc = min(out_c, weight_buf // weight_per_oc)
            tile_oc = min(tile_oc, PARAM_SRAM_MAX_CH)
            tile_oc = max(tile_oc, 1)
        else:
            tile_oc = out_c
    elif op_type == 'dw':
        # DW conv: weight = in_c * kh * kw, out_c = in_c, no OC tiling
        tile_oc = out_c
    else:
        # Pool, Add, Resize: no weight, tile_oc = out_c
        tile_oc = out_c

    # Check if the entire layer fits without tiling
    full_input_size = in_h * in_w * in_c * elem_size
    full_output_size = out_h * out_w * out_c * elem_size
    full_weight_size = kernel_h * kernel_w * in_c * out_c * elem_size if op_type == 'conv' else 0

    if (full_input_size <= act_bank and
        full_output_size <= act_bank and
        full_weight_size <= weight_buf):
        # No tiling needed
        return {'tile_h': 0, 'tile_w': 0, 'tile_num_h': 0, 'tile_num_w': 0, 'tile_oc': out_c}

    # Search for optimal output tile size
    # Strategy: maximize tile area while preferring square-ish tiles
    # (square tiles minimize halo overhead ratio)

    best_tile_h = 1
    best_tile_w = 1
    best_area = 1

    # Search: try all reasonable tile sizes, pick largest area with smallest perimeter
    # Limit search space for efficiency
    max_tile_h = min(out_h, act_bank // (tile_oc * elem_size))
    max_tile_w = min(out_w, act_bank // (tile_oc * elem_size))

    for tile_h in range(1, max_tile_h + 1):
        for tile_w in range(1, max_tile_w + 1):
            # Output tile constraint: tile_h * tile_w * tile_oc * elem_size <= act_bank
            out_tile_size = tile_h * tile_w * tile_oc * elem_size
            if out_tile_size > act_bank:
                break  # tile_w only grows, so break inner loop

            # Input tile dimensions (including halo for convolution)
            in_tile_h = tile_h * stride_h + kh_eff - stride_h
            in_tile_w = tile_w * stride_w + kw_eff - stride_w

            # Input tile constraint: in_tile_h * in_tile_w * in_c * elem_size <= act_bank
            in_tile_size = in_tile_h * in_tile_w * in_c * elem_size
            if in_tile_size > act_bank:
                break  # tile_w only grows, so break inner loop

            # This tile is valid. Prefer larger area; tie-break by squareness.
            area = tile_h * tile_w
            if area > best_area or (area == best_area and
                                    abs(tile_h - tile_w) < abs(best_tile_h - best_tile_w)):
                best_tile_h = tile_h
                best_tile_w = tile_w
                best_area = area

    # If search found nothing better than 1x1, use 1x1
    tile_h = best_tile_h
    tile_w = best_tile_w

    # Compute tile counts
    tile_num_h = math.ceil(out_h / tile_h)
    tile_num_w = math.ceil(out_w / tile_w)

    # If only 1 tile in each direction and it covers the full output,
    # no tiling needed (just direct compute)
    if tile_num_h == 1 and tile_num_w == 1 and tile_oc == out_c:
        return {'tile_h': 0, 'tile_w': 0, 'tile_num_h': 0, 'tile_num_w': 0, 'tile_oc': out_c}

    return {
        'tile_h': tile_h,
        'tile_w': tile_w,
        'tile_num_h': tile_num_h,
        'tile_num_w': tile_num_w,
        'tile_oc': tile_oc,
    }


def print_tiling_summary(layers_info, label=""):
    """Print a summary of tiling decisions for all layers."""
    if label:
        print(f"\n  [{label}]")
    print(f"\n{'Layer':>6} {'Op':>5} {'InHxWxC':>14} {'OutHxWxC':>14} "
          f"{'Tile HxW':>10} {'TileOC':>6} {'Tiles':>8} {'InTile':>8} {'OutTile':>8}")
    print("-" * 100)

    for i, info in enumerate(layers_info):
        tile = info['tiling']
        if tile['tile_num_h'] == 0:
            tile_str = "none"
            tiles_str = "1x1"
            in_t_str = "-"
            out_t_str = "-"
            oc_str = str(tile.get('tile_oc', info['out_c']))
        else:
            tile_str = f"{tile['tile_h']}x{tile['tile_w']}"
            tiles_str = f"{tile['tile_num_h']}x{tile['tile_num_w']}"
            oc_str = str(tile.get('tile_oc', info['out_c']))
            # Compute actual sizes for display
            kh_eff = (info.get('kernel_h', 1) - 1) * info.get('dilation_h', 1) + 1
            kw_eff = (info.get('kernel_w', 1) - 1) * info.get('dilation_w', 1) + 1
            in_tile_h = tile['tile_h'] * info.get('stride_h', 1) + kh_eff - info.get('stride_h', 1)
            in_tile_w = tile['tile_w'] * info.get('stride_w', 1) + kw_eff - info.get('stride_w', 1)
            in_t_str = f"{in_tile_h * in_tile_w * info['in_c'] * info['elem_size'] // 1024}KB"
            t_oc = tile.get('tile_oc', info['out_c'])
            out_t_str = f"{tile['tile_h'] * tile['tile_w'] * t_oc * info['elem_size'] // 1024}KB"

        print(f"{i:>6} {info['op_type']:>5} "
              f"{info['in_h']:>3}x{info['in_w']:>3}x{info['in_c']:>4} "
              f"{info['out_h']:>3}x{info['out_w']:>3}x{info['out_c']:>4} "
              f"{tile_str:>10} {oc_str:>6} {tiles_str:>8} {in_t_str:>8} {out_t_str:>8}")


if __name__ == '__main__':
    # Self-test: verify tiling for ResNet-18 INT8 layers (single vs double buffer)
    print("=== Tiling Calculator: Single-Buffer vs Double-Buffer (ResNet-18 INT8) ===\n")

    test_layers = [
        # Layer 0: Conv 7x7/2, 3→32, 224→112
        {'op_type': 'conv', 'in_h': 224, 'in_w': 224, 'in_c': 3,
         'out_h': 112, 'out_w': 112, 'out_c': 32,
         'kernel_h': 7, 'kernel_w': 7, 'stride_h': 2, 'stride_w': 2},
        # Layer 1: MaxPool 3x3/2, 32→32, 112→56
        {'op_type': 'pool', 'in_h': 112, 'in_w': 112, 'in_c': 32,
         'out_h': 56, 'out_w': 56, 'out_c': 32,
         'kernel_h': 3, 'kernel_w': 3, 'stride_h': 2, 'stride_w': 2},
        # Layer 2: Conv 3x3, 32→32, 56→56
        {'op_type': 'conv', 'in_h': 56, 'in_w': 56, 'in_c': 32,
         'out_h': 56, 'out_w': 56, 'out_c': 32,
         'kernel_h': 3, 'kernel_w': 3, 'stride_h': 1, 'stride_w': 1},
        # Layer 4: Add, 56x56x32
        {'op_type': 'add', 'in_h': 56, 'in_w': 56, 'in_c': 32,
         'out_h': 56, 'out_w': 56, 'out_c': 32,
         'kernel_h': 1, 'kernel_w': 1, 'stride_h': 1, 'stride_w': 1},
        # Layer 8: Conv 3x3/2, 32→64, 56→28
        {'op_type': 'conv', 'in_h': 56, 'in_w': 56, 'in_c': 32,
         'out_h': 28, 'out_w': 28, 'out_c': 64,
         'kernel_h': 3, 'kernel_w': 3, 'stride_h': 2, 'stride_w': 2},
        # Layer 12: Conv 3x3, 64→64, 28→28
        {'op_type': 'conv', 'in_h': 28, 'in_w': 28, 'in_c': 64,
         'out_h': 28, 'out_w': 28, 'out_c': 64,
         'kernel_h': 3, 'kernel_w': 3, 'stride_h': 1, 'stride_w': 1},
        # Layer 15: Conv 3x3/2, 64→128, 28→14
        {'op_type': 'conv', 'in_h': 28, 'in_w': 28, 'in_c': 64,
         'out_h': 14, 'out_w': 14, 'out_c': 128,
         'kernel_h': 3, 'kernel_w': 3, 'stride_h': 2, 'stride_w': 2},
        # Layer 19: Conv 3x3, 128→128, 14→14
        {'op_type': 'conv', 'in_h': 14, 'in_w': 14, 'in_c': 128,
         'out_h': 14, 'out_w': 14, 'out_c': 128,
         'kernel_h': 3, 'kernel_w': 3, 'stride_h': 1, 'stride_w': 1},
        # Layer 22: Conv 3x3/2, 128→256, 14→7
        {'op_type': 'conv', 'in_h': 14, 'in_w': 14, 'in_c': 128,
         'out_h': 7, 'out_w': 7, 'out_c': 256,
         'kernel_h': 3, 'kernel_w': 3, 'stride_h': 2, 'stride_w': 2},
        # Layer 26: Conv 3x3, 256→256, 7→7
        {'op_type': 'conv', 'in_h': 7, 'in_w': 7, 'in_c': 256,
         'out_h': 7, 'out_w': 7, 'out_c': 256,
         'kernel_h': 3, 'kernel_w': 3, 'stride_h': 1, 'stride_w': 1},
        # Layer 29: GlobalAvgPool, 256, 7→1
        {'op_type': 'pool', 'in_h': 7, 'in_w': 7, 'in_c': 256,
         'out_h': 1, 'out_w': 1, 'out_c': 256,
         'kernel_h': 7, 'kernel_w': 7, 'stride_h': 7, 'stride_w': 7},
        # Layer 30: FC, 256→10
        {'op_type': 'fc', 'in_h': 1, 'in_w': 1, 'in_c': 256,
         'out_h': 1, 'out_w': 1, 'out_c': 10,
         'kernel_h': 1, 'kernel_w': 1, 'stride_h': 1, 'stride_w': 1},
    ]

    elem_size = 1  # INT8

    for db_mode, label in [(False, "Single-Buffer (32KB act, 64KB wgt)"),
                           (True,  "Double-Buffer (16KB act, 32KB wgt)")]:
        layers_info = []
        for layer in test_layers:
            tile = compute_tiling(elem_size=elem_size,
                                  dilation_h=layer.get('dilation_h', 1),
                                  dilation_w=layer.get('dilation_w', 1),
                                  double_buffer=db_mode,
                                  **{k: v for k, v in layer.items()
                                     if k not in ('dilation_h', 'dilation_w')})
            info = {**layer, 'tiling': tile, 'elem_size': elem_size}
            layers_info.append(info)

        print_tiling_summary(layers_info, label)

    # Constraint verification for double-buffer mode
    print("\n\n=== Constraint Verification (Double-Buffer Mode) ===")
    act_bank = ACT_BANK_SIZE // 2
    weight_buf = WEIGHT_BUF_SIZE // 2
    print(f"  Effective act_bank = {act_bank // 1024}KB, weight_buf = {weight_buf // 1024}KB\n")

    layers_info_db = []
    for layer in test_layers:
        tile = compute_tiling(elem_size=elem_size,
                              dilation_h=layer.get('dilation_h', 1),
                              dilation_w=layer.get('dilation_w', 1),
                              double_buffer=True,
                              **{k: v for k, v in layer.items()
                                 if k not in ('dilation_h', 'dilation_w')})
        info = {**layer, 'tiling': tile, 'elem_size': elem_size}
        layers_info_db.append(info)

    for i, info in enumerate(layers_info_db):
        tile = info['tiling']
        if tile['tile_num_h'] == 0:
            continue

        kh_eff = (info['kernel_h'] - 1) * info.get('dilation_h', 1) + 1
        kw_eff = (info['kernel_w'] - 1) * info.get('dilation_w', 1) + 1
        in_tile_h = tile['tile_h'] * info['stride_h'] + kh_eff - info['stride_h']
        in_tile_w = tile['tile_w'] * info['stride_w'] + kw_eff - info['stride_w']

        in_tile_bytes = in_tile_h * in_tile_w * info['in_c'] * elem_size
        out_tile_bytes = tile['tile_h'] * tile['tile_w'] * tile['tile_oc'] * elem_size

        # Weight constraint for conv
        if info['op_type'] == 'conv':
            w_per_oc = info['kernel_h'] * info['kernel_w'] * info['in_c'] * elem_size
            w_tile_bytes = w_per_oc * tile['tile_oc']
            oc_groups = math.ceil(info['out_c'] / tile['tile_oc'])
        else:
            w_tile_bytes = 0
            oc_groups = 1

        in_ok = "ok" if in_tile_bytes <= act_bank else "FAIL"
        out_ok = "ok" if out_tile_bytes <= act_bank else "FAIL"
        w_ok = "ok" if w_tile_bytes <= weight_buf else "FAIL"

        print(f"  Layer {i}: in={in_tile_bytes//1024}KB [{in_ok}]  "
              f"out={out_tile_bytes//1024}KB [{out_ok}]  "
              f"wgt={w_tile_bytes//1024}KB [{w_ok}]  "
              f"tile_oc={tile['tile_oc']}  oc_groups={oc_groups}")
