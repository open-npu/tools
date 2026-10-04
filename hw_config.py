"""
Open-NPU Hardware Configuration

Parameterizable hardware config for offline compilation (tiling, fusion, perf model).
All parameters are free — no predefined profiles. Users explore configurations freely
and decide which combinations work best after FPGA/silicon validation.

SPDX-License-Identifier: Apache-2.0
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HWConfig:
    """NPU hardware configuration (all dimensions freely configurable)."""
    array_size: int = 8          # Systolic array dimension (N×N)
    act_bank_size: int = 24576   # IFM bytes (24KB). DB half is 12KB. OFM is 48KB.
    weight_buf_size: int = 81920 # Weight SRAM bytes (80KB)
    param_sram_max_ch: int = 512 # Max output channels per param tile (= param_sram / 14)
    dw_parallel_ch: int = 8      # DW conv parallel channels (typically = array_size)
    has_int16: bool = True       # INT16 data type support
    has_lut: bool = True         # LUT activation support
    ext_bw_bytes: int = 4        # External bus bandwidth (bytes/cycle)

    @property
    def macs_per_cycle(self) -> int:
        return self.array_size * self.array_size

    @property
    def spad_total_kb(self) -> int:
        # IFM is act_bank_size. OFM is a separate 48KB bank; param is 12KB.
        return (self.act_bank_size + 48 * 1024
                + self.weight_buf_size + 12 * 1024) // 1024


# Default config: 8×8 array for A/D (1×1 + DW). 16×16 via --array-size.
DEFAULT_HW = HWConfig()


def hw_config_from_args(args) -> HWConfig:
    """Build HWConfig from argparse namespace.

    Expected args attributes (all optional, falls back to defaults):
        --array-size, --act-bank-kb, --weight-buf-kb,
        --param-max-ch, --no-int16, --ext-bw
    """
    act_bank_kb = getattr(args, 'act_bank_kb', None)
    weight_buf_kb = getattr(args, 'weight_buf_kb', None)

    return HWConfig(
        array_size=getattr(args, 'array_size', None) or DEFAULT_HW.array_size,
        act_bank_size=(act_bank_kb * 1024) if act_bank_kb else DEFAULT_HW.act_bank_size,
        weight_buf_size=(weight_buf_kb * 1024) if weight_buf_kb else DEFAULT_HW.weight_buf_size,
        param_sram_max_ch=getattr(args, 'param_max_ch', None) or DEFAULT_HW.param_sram_max_ch,
        dw_parallel_ch=getattr(args, 'array_size', None) or DEFAULT_HW.dw_parallel_ch,
        has_int16=not getattr(args, 'no_int16', False),
        has_lut=DEFAULT_HW.has_lut,
        ext_bw_bytes=getattr(args, 'ext_bw', None) or DEFAULT_HW.ext_bw_bytes,
    )


def add_hw_args(parser):
    """Add hardware configuration arguments to an argparse parser."""
    grp = parser.add_argument_group('Hardware configuration',
        'Override NPU hardware parameters (default: 8×8 array, 164KB scratchpad)')
    grp.add_argument('--array-size', type=int, default=None,
                     help='Systolic array dimension N (NxN MACs/cycle, default: 8)')
    grp.add_argument('--act-bank-kb', type=int, default=None,
                     help='IFM bank size in KB (default: 24)')
    grp.add_argument('--weight-buf-kb', type=int, default=None,
                     help='Weight buffer size in KB (default: 80)')
    grp.add_argument('--param-max-ch', type=int, default=None,
                     help='Max output channels per param tile (default: 512)')
    grp.add_argument('--no-int16', action='store_true',
                     help='Disable INT16 support (INT8 only)')
    grp.add_argument('--ext-bw', type=int, default=None,
                     help='External bus bandwidth in bytes/cycle (default: 4)')
