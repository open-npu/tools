"""Eligibility and capacity checks for the controller-owned super-tile ABI.

The strict RTL path supports Model-A INT8 stride-1 Conv1x1→DW3x3→PW1x1
blocks at 56², 28², and 14².  Geometry is descriptor-owned and may contain
multiple tile rows and columns; every tile includes a one-pixel Expand halo.
"""

from dataclasses import dataclass
from math import ceil
from typing import Sequence


IFM_BYTES = 32 * 1024
OFM_BYTES = 48 * 1024
WGT_BYTES = 80 * 1024
PARAM_BYTES = 12 * 1024
PARAM_BYTES_PER_CHANNEL = 16  # physical RTL table: four 32-bit words/channel
MODEL_A_TILE = {56: (28, 10), 28: (28, 6), 14: (14, 12)}


@dataclass(frozen=True)
class SuperLayer:
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
    elem_bytes: int = 1
    db_en: bool = False
    group_major: bool = False


@dataclass(frozen=True)
class SuperTilePlan:
    tile_h: int
    tile_w: int
    tile_rows: int
    tile_cols: int
    weight_bytes: int
    param_bytes: int
    ifm_peak_bytes: int
    ofm_peak_bytes: int


def model_a_tile_for(spatial: int) -> tuple[int, int]:
    """Return the strict RTL tile for a supported square Model-A block."""
    try:
        return MODEL_A_TILE[spatial]
    except KeyError as exc:
        raise ValueError(f"unsupported Model-A super spatial size: {spatial}") from exc


def plan_super_tile(layers: Sequence[SuperLayer], tile_h: int = 14,
                    tile_w: int = 12) -> SuperTilePlan:
    """Validate a PW→DW(s1)→PW INT8 block and return physical SRAM peaks.

    The first pointwise layer computes the DW halo, so its spatial extent is
    ``tile + kernel - 1``.  Out-of-image halo cells are filled with that
    layer's output zero-point by the block FSM/compute halo path.
    """
    if len(layers) != 3:
        raise ValueError("super-tile requires exactly three descriptors")
    pw0, dw, pw1 = layers
    if (pw0.op_type, dw.op_type, pw1.op_type) != ("conv", "dw", "conv"):
        raise ValueError("super-tile mode requires Conv1x1 -> DW -> Conv1x1")
    if (pw0.kernel_h, pw0.kernel_w, pw1.kernel_h, pw1.kernel_w) != (1, 1, 1, 1):
        raise ValueError("super-tile pointwise layers must use 1x1 kernels")
    if dw.kernel_h != 3 or dw.kernel_w != 3:
        raise ValueError("first prototype supports only DW 3x3")
    if dw.stride_h != 1 or dw.stride_w != 1:
        raise ValueError("first prototype supports only DW stride 1")
    if any(layer.elem_bytes != 1 for layer in layers):
        raise ValueError("first prototype supports INT8 only")
    if any(layer.db_en or layer.group_major for layer in layers):
        raise ValueError("super-tile requires single-buffer tile-major mode")
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError("super-tile dimensions must be positive")
    if not (pw0.out_c == dw.in_c == dw.out_c == pw1.in_c):
        raise ValueError("super-tile intermediate channels do not match")
    if not (pw0.out_h == dw.in_h == dw.out_h == pw1.in_h == pw1.out_h
            and pw0.out_w == dw.in_w == dw.out_w == pw1.in_w == pw1.out_w):
        raise ValueError("super-tile spatial dimensions do not match")

    halo_h = tile_h + dw.kernel_h - 1
    halo_w = tile_w + dw.kernel_w - 1
    # Bank roles alternate: IFM holds block input then DW output; OFM holds
    # expanded PW output then final block output.
    ifm_peak = max(halo_h * halo_w * pw0.in_c,
                   tile_h * tile_w * dw.out_c)
    ofm_peak = max(halo_h * halo_w * pw0.out_c,
                   tile_h * tile_w * pw1.out_c)
    weight_bytes = (pw0.in_c * pw0.out_c
                    + dw.kernel_h * dw.kernel_w * dw.in_c
                    + pw1.in_c * pw1.out_c)
    param_bytes = (pw0.out_c + dw.out_c + pw1.out_c) * PARAM_BYTES_PER_CHANNEL

    errors = []
    if ifm_peak > IFM_BYTES:
        errors.append(f"IFM {ifm_peak} > {IFM_BYTES}")
    if ofm_peak > OFM_BYTES:
        errors.append(f"OFM {ofm_peak} > {OFM_BYTES}")
    if weight_bytes > WGT_BYTES:
        errors.append(f"WGT {weight_bytes} > {WGT_BYTES}")
    if param_bytes > PARAM_BYTES:
        errors.append(f"PARAM {param_bytes} > {PARAM_BYTES}")
    if errors:
        raise ValueError("super-tile capacity exceeded: " + ", ".join(errors))

    return SuperTilePlan(
        tile_h=tile_h,
        tile_w=tile_w,
        tile_rows=ceil(pw1.out_h / tile_h),
        tile_cols=ceil(pw1.out_w / tile_w),
        weight_bytes=weight_bytes,
        param_bytes=param_bytes,
        ifm_peak_bytes=ifm_peak,
        ofm_peak_bytes=ofm_peak,
    )
