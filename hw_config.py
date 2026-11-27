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
    mac_lanes: int = 64          # Dense row MACs issued per cycle
    act_bank_size: int = 32768   # IFM bytes (32KB). DB half is 16KB.
    ofm_bank_size: int = 49152   # OFM bytes (48KB). DB half is 24KB.
    weight_buf_size: int = 81920 # Weight SRAM bytes (80KB)
    param_sram_size: int = 12288 # Parameter SRAM bytes (12KB)
    param_sram_max_ch: int = 512 # Max output channels per param tile (= param_sram / 14)
    dw_parallel_ch: int = 8      # Depthwise channels processed in parallel
    has_int16: bool = True       # INT16 data type support
    has_lut: bool = True         # LUT activation support
    ext_bw_bytes: int = 4        # External DMA bus bandwidth (bytes/cycle: 4=32b, 8=64b, 16=128b)
    dma_bus_proto: str = 'wb'    # External DMA bus protocol ('wb' or 'axi4')

    @property
    def dma_bus_width(self) -> int:
        return self.ext_bw_bytes * 8

    @property
    def hw_config_reg(self) -> int:
        """Expected 32-bit REG_HW_CONFIG (0x018) value matching npu_csr.v."""
        dw_enc = {32: 0, 64: 1, 128: 2}.get(self.dma_bus_width, 0)
        proto_enc = 1 if self.dma_bus_proto == 'axi4' else 0
        return (
            (proto_enc << 29)
            | (dw_enc << 27)
            | ((1 if self.has_lut else 0) << 25)
            | ((1 if self.has_int16 else 0) << 24)
            | ((self.spad_total_kb & 0xFF) << 16)
            | (3 << 12)  # acc_enc: 44-bit
            | (1 << 8)   # ppu_enc: 8 lanes
            | (self.mac_lanes & 0xFF)
        )

    @property
    def macs_per_cycle(self) -> int:
        return self.mac_lanes

    @property
    def spad_total_kb(self) -> int:
        return (self.act_bank_size + self.ofm_bank_size
                + self.weight_buf_size + self.param_sram_size) // 1024


# Default config: one 64-MAC dense row plus an 8-lane depthwise bank.
DEFAULT_HW = HWConfig()


def hw_config_from_args(args) -> HWConfig:
    """Build HWConfig from argparse namespace.

    Expected args attributes (all optional, falls back to defaults):
        --mac-lanes, --dw-lanes, --act-bank-kb, --weight-buf-kb,
        --param-max-ch, --no-int16, --ext-bw
    """
    act_bank_kb = getattr(args, 'act_bank_kb', None)
    ofm_bank_kb = getattr(args, 'ofm_bank_kb', None)
    weight_buf_kb = getattr(args, 'weight_buf_kb', None)
    param_kb = getattr(args, 'param_kb', None)

    legacy_array = getattr(args, 'array_size', None)
    return HWConfig(
        mac_lanes=(getattr(args, 'mac_lanes', None)
                   or (legacy_array * legacy_array if legacy_array else None)
                   or DEFAULT_HW.mac_lanes),
        act_bank_size=(act_bank_kb * 1024) if act_bank_kb else DEFAULT_HW.act_bank_size,
        ofm_bank_size=(ofm_bank_kb * 1024) if ofm_bank_kb else DEFAULT_HW.ofm_bank_size,
        weight_buf_size=(weight_buf_kb * 1024) if weight_buf_kb else DEFAULT_HW.weight_buf_size,
        param_sram_size=(param_kb * 1024) if param_kb else DEFAULT_HW.param_sram_size,
        param_sram_max_ch=(getattr(args, 'param_max_ch', None)
                           or ((param_kb * 1024) // 16 if param_kb
                               else DEFAULT_HW.param_sram_max_ch)),
        dw_parallel_ch=(getattr(args, 'dw_lanes', None)
                        or legacy_array or DEFAULT_HW.dw_parallel_ch),
        has_int16=not getattr(args, 'no_int16', False),
        has_lut=DEFAULT_HW.has_lut,
        ext_bw_bytes=(
            (getattr(args, 'dma_bus_width', None) // 8)
            if getattr(args, 'dma_bus_width', None)
            else (getattr(args, 'ext_bw', None) or DEFAULT_HW.ext_bw_bytes)
        ),
        dma_bus_proto=(getattr(args, 'dma_bus_proto', None) or DEFAULT_HW.dma_bus_proto),
    )


def add_hw_args(parser):
    """Add hardware configuration arguments to an argparse parser."""
    grp = parser.add_argument_group('Hardware configuration',
        'Override NPU hardware parameters (default: 64-MAC row, 8 DW lanes)')
    grp.add_argument('--mac-lanes', type=int, default=None,
                     help='Dense MAC row width (default: 64)')
    grp.add_argument('--dw-lanes', type=int, default=None,
                     help='Depthwise parallel channels (default: 8)')
    grp.add_argument('--array-size', type=int, default=None,
                     help='Deprecated compatibility alias: maps N to N*N MAC lanes and N DW lanes')
    grp.add_argument('--act-bank-kb', type=int, default=None,
                     help='IFM bank size in KB (default: 32)')
    grp.add_argument('--ofm-bank-kb', type=int, default=None,
                     help='OFM bank size in KB (default: 48)')
    grp.add_argument('--weight-buf-kb', type=int, default=None,
                     help='Weight buffer size in KB (default: 80)')
    grp.add_argument('--param-max-ch', type=int, default=None,
                     help='Max output channels per param tile (default: 512)')
    grp.add_argument('--param-kb', type=int, default=None,
                     help='Parameter SRAM size in KB (default: 12)')
    grp.add_argument('--no-int16', action='store_true',
                     help='Disable INT16 support (INT8 only)')
    grp.add_argument('--ext-bw', type=int, default=None,
                     help='External bus bandwidth in bytes/cycle (default: 4)')
    grp.add_argument('--dma-bus-width', type=int, choices=[32, 64, 128], default=None,
                     help='External DMA data bus width in bits (32, 64, or 128; default: 32)')
    grp.add_argument('--dma-bus-proto', type=str, choices=['wb', 'axi4'], default=None,
                     help='External DMA master protocol (wb or axi4; default: wb)')
