import os, sys, cocotb, numpy as np
from cocotb.clock import Clock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_npu_dma_e2e import WbSlave, WbMasterMem, reset, load_golden, program_layer, run_layer_and_wait
@cocotb.test()
async def test_md_l16_only(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    wb = WbSlave(dut, dut.clk); mem = WbMasterMem(dut, dut.clk)
    cocotb.start_soon(mem.run())
    md, ld = load_golden('model_d_int16')
    i = 16; m = md[i]; d = ld[i]
    mem.populate(m['ddr_in_addr'], d['input'])
    if len(d['wgt']) > 0: mem.populate(m['ddr_wgt_addr'], d['wgt'])
    if len(d['param']) > 0: mem.populate(m['ddr_param_addr'], d['param'])
    if 'input_b' in d and len(d['input_b']) > 0 and 'ddr_add_b_addr' in m:
        mem.populate(m['ddr_add_b_addr'], d['input_b'])
    await program_layer(wb, m)
    dut._log.info(f"L16 {m['in_h']}x{m['in_w']}x{m['in_c']} -> {m['out_h']}x{m['out_w']}x{m['out_c']} k={m['kernel_h']}x{m['kernel_w']} tile={m.get('tile_h',0)}x{m.get('tile_w',0)}")
    done = await run_layer_and_wait(wb, dut, timeout=20000000000)
    nw=m['n_output_words']; ref=d['output']; oa=m['ddr_out_addr']
    got=np.array([mem.mem.get(oa+j*4,0) for j in range(nw)],dtype=np.uint32)
    mm=np.where(got!=ref)[0] if len(ref)==nw else np.arange(nw)
    if len(ref)!=nw: dut._log.error(f"FAIL L16: size {len(ref)} vs {nw}")
    elif len(mm)==0: dut._log.info(f"PASS L16: {nw}/{nw}")
    else: dut._log.error(f"FAIL L16: {len(mm)}/{nw} first w[{mm[0]}] exp={ref[mm[0]]:08X} got={got[mm[0]]:08X}")
