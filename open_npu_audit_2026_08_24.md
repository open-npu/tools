# Open-NPU 整体审计报告（2026-08-24）

范围：

- Phase-1 可验收 / 可发布
- ZC706 FPGA 可部署
- `design`、`csim`、`rtl`、`soc`、`driver` 全链路

结论：

| 目标 | 结论 |
|---|---|
| 当前量化模型的 SoC Verilator 功能验证 | 十格在本次工作过程中均跑到 RTL↔CSIM bit-exact；D8 最后一轮 24 层全部 PASS |
| Phase-1 可发布 | **NO-GO** |
| ZC706 M0 单层 bring-up | **NO-GO** |
| ZC706 A–E 全模型 | **NO-GO**，需要 M1 DDR |

这里的 NO-GO 不等于“RTL 什么都没做对”。主要问题是：

1. D8 INT8 对浮点已用 MSE clip 过门（0.952442 ≥ 0.95）。旧记录 0.947 是 percentile PTQ。
2. 正式验收入口没有执行十格两道门。
3. 当前十格 SoC 路径关闭了 fusion，且输出区预填 golden，证据强度不够。
4. driver 和 RTL 的 CSR ABI 不一致。
5. ZC706 target 还没有完成最基本的总线、IRQ、引脚和内存集成。

---

## 1. 审计基线

### 1.1 当前仓库

`/data/sam/open-npu` 本身不是一个 git 仓库；子目录各自独立。

| 仓库 | HEAD |
|---|---|
| design | `705fff63a4bd91f0938f6e1ef082af3d49701820` |
| csim | `a932bd89f18a318b59c9843e69c4f2a68262613d` |
| rtl | `196d48a522e5f1fe3c88b060b713a5406c3938c3` |
| soc | `c052d5750dc194372499d55342ff87460042a26c` |
| driver | `0f9ec33da24ed474cfdecaee013b1f2c828879dc` |

`design`、`csim`、`rtl`、`soc` 都有大量未提交修改；`rtl` 还包含大量改写后的 golden `.npy`。因此现有结果不能用一个 commit 或 tag 唯一重建。

### 1.2 工具

| 工具 | 版本 |
|---|---|
| Python | 3.10.12 |
| Verilator | 5.036 |
| Icarus | 11.0 |
| Yosys | 0.65 |
| RISC-V GCC | 10.2.0 |
| Vivado | **当前机器未安装** |

### 1.3 本次执行的短检查

- `tools/tests/test_phase1_guards.py`：8/8 PASS。
- bare-metal driver host compile：PASS。
- FreeRTOS stub compile：PASS。
- 十套 golden metadata 都存在：
  - A：63 层
  - B：25 层
  - C：17 层
  - D：24 层
  - E：31 层

没有在本次审计中重新启动十次长 SoC 仿真、全量 cocotb、综合或 Vivado。

---

## 2. P0：发布 / 部署阻断项

### P0-1 正式 Phase-1 入口没有执行十格两道门

证据：

- `tools/PHASE1_ACCEPTANCE.md` 规定唯一入口是：

  ```sh
  python3 tools/phase1_regression.py
  ```

- `tools/phase1_regression.py:88-168` 运行 design、CSIM、RTL cocotb 和 generic synthesis。
- 它没有调用：
  - `tools/phase1_model_matrix.py`
  - `soc/run_model_e2e.sh`
  - A–E × INT8/INT16 的十次 SoC Verilator

影响：

`phase1_regression.py` 可以在 D8 精度仍失败、十格没有在当前代码上生成 golden / 运行 RTL 时打印：

```text
PHASE-1 REGRESSION FULL: PASS
```

修复：

让正式入口强制执行十格，每格必须同时满足：

1. 当前 converter+CSIM ↔ ONNX float 达到阈值
2. 当前 RTL ↔ 同一次生成的 CSIM golden bit-exact

不允许 `recorded` 值、缓存模型、缺格或 SKIP。

### P0-2 D8 INT8 精度 — 已关闭（MSE clip）

当场对比（同图 `neg/000000.jpg`、50 张标定）：

| PTQ | 余弦 |
|---|---|
| default percentile | 0.947490 |
| **mse** | **0.952442** |
| admm | 0.947425 |
| mse_admm | 0.952066 |

正式 D8 路径（`phase1_model_matrix.py`、`model_d_int8_golden.py`）使用 `ptq_method=mse`。门槛 0.95 已过，此条不再阻断。

后续可选：D 专用掌静脉集、多图复测、非负 ReLU6 码字。不是当前验收阻断项。

### P0-3 SoC 输出工作区预填 golden，可产生假 PASS

证据：

- `soc/firmware/build_main_ram.py:175-179` 把 golden 输出写入运行时 `ddr_out`。
- `soc/firmware/main.c:422-520` 运行后直接把同一地址与 blob 中 golden 比较。
- 之前已经实际出现：
  - NPU 没有覆盖的 word 保持 golden，因此显示 PASS。
  - 只写最后一块到 offset 0，表现成“小范围 mismatch”。
  - `PERF_Lx npu=0` 但层仍 PASS。
- `soc/sim/sim_main.cpp:142-144` 在没有识别到 `RESULT` 时，只要收到过 UART，也可能返回 0。

影响：

`PASS` 不能单独证明 NPU 写过全部输出；卡死、提前 IRQ、漏 tile 都可能被掩盖。

修复：

- 工作区输出初始化为 poison，而不是 golden。
- golden 只留在只读比对区。
- 每层至少验证：
  - NPU busy cycle > 0（对真正硬件执行层）
  - 输出全区已被覆盖
  - 最终必须出现 `RESULT: PASS`
- simulator 未见 `RESULT`、达到总周期或 UART timeout 时必须非零退出。

### P0-4 ZC706 target 当前无法完成有效集成

证据：

- `soc/fpga/zc706_npu.py:55` 使用 `self.npu.wb_mst`。
- wrapper 实际成员是 `wb_master`：`soc/litex/npu_core.py:23`。
- FPGA target 没有像仿真 target `soc/litex/npu_soc.py:72-80` 那样：
  - 把 `wb_slave` 映射到 `0x80000000`
  - 注册 NPU IRQ
- `soc/fpga/xilinx_zc706.py` 的 clock、UART、reset、LED 全是 `TODO-VERIFY`。

影响：

当前脚本在 LiteX elaboration 阶段就可能失败；即使修正属性名，CPU 也访问不到 NPU CSR，IRQ 也未接入。

### P0-5 M0 512 KiB BRAM 无法运行 A–E 全模型

证据：

- `soc/fpga/zc706_npu.py:51`：main RAM = 512 KiB。
- 当前 D8 `BLOB_MODEL_SIZE = 2,450,396` bytes。
- 模型 DDR 区还需要权重、参数和每层非重叠工作区，实际范围远大于 blob。
- M0 没有 DDR3。

结论：

- M0 只能用于专门缩小的单层 / 小网络 bring-up。
- A–E 全模型必须先完成 M1 LiteDRAM / DDR3。

---

## 3. P1：应在下一阶段优先关闭

### P1-1 十格 PASS 没有可发布的可追溯证据

本次工作过程中十格 SoC 均曾跑到 bit-exact；但仓库内现存日志不能独立证明这一点：

- `soc/sim/sim_*.log` 时间跨 7 月和 8 月。
- 日志不含 tools/csim/rtl/soc commit、模型 hash、golden hash、firmware hash。
- `sim_ma_int8.log` 横幅写的是 INT16。
- `sim_mc_int16.log` 没有 `RESULT`，且只到 L7。
- 当前源码和这些日志产生时的源码不同。

修复：

每格保存一个 manifest：

```text
model SHA-256
calibration-list SHA-256
evaluation-input SHA-256
tools/csim/rtl/soc commit + dirty diff hash
NPU1 SHA-256
golden metadata/npy aggregate SHA-256
NPU2 blob SHA-256
firmware + sim binary SHA-256
完整命令、工具版本、最终 RESULT
```

### P1-2 golden 生成器允许复用缓存，并可能失败后成功退出

例子：

- `tools/gen/model_d_int8_golden.py:64-80`：`/tmp` 中 NPU1 已存在就直接复用。
- `:91-94`：CSIM 失败后 `return`，Python 进程仍可能退出 0。
- 输出 golden 目录没有先清理或原子替换。

影响：

`soc/run_model_e2e.sh` 虽然写着 fresh golden，底层生成器仍可复用旧 converter 产物和残留 `.npy`。

修复：

- 每次使用新临时目录。
- 禁止 `if exists: reuse`。
- 转换 / CSIM / dump 缺失必须非零退出。
- 校验 dump 数量、shape、dtype。
- 完成后原子替换 golden 目录。

### P1-3 当前 SoC 十格固件关闭全部 fusion

证据：

`soc/firmware/main.c:270-273`：

```c
NPU_REG(REG_DMA_CTRL) = sched & ~0x0E;
```

并且：

- 每层软复位
- `act_base=0`
- 每层 DDR round-trip

影响：

当前十格 PASS 证明的是 unfused 运行方式；不能作为 fusion 功能或带宽收益证据。

三天卡死的问题虽然通过 `db_prefetch_done` 修复避免了死等，但当前整网路径并没有把 FUSE bits 送入 RTL。

必须做出明确产品决定：

1. Phase-1 正式冻结 **unfused DDR 模式**，从模型元数据、文档和验收中移除 fusion 能力声明；或
2. 恢复 fusion，单独覆盖：
   - non-tiled SRAM fusion
   - tiled + DB_EN + PTS
   - mixed tiled→untiled
   - 不同 tile grid

在这些测试通过前，不应宣称 fusion 已验收。

### P1-4 正式回归要求零 SKIP，但 RTL E2E 含 SKIP / expected-fail

`rtl/tb/e2e/test_npu_dma_e2e.py` 当前有：

- 6 个 `skip=True`
- 2 个 `expect_fail=True`
- 其中一个是支持范围内的 INT16 + DB_EN divergence
- 另一个是 fused + DB_EN

而 `phase1_regression.py` 对全套 cocotb 使用 `reject_skip=True`。

影响：

规范、回归脚本和测试现状不一致。要么正式删除退休测试，要么修复 / 改成有效测试；不能一边要求零 SKIP，一边把已知功能错误藏在 expected-fail。

### P1-5 firmware Makefile 会复用错误模型的 blob

`soc/firmware/Makefile:53-55`：

```make
$(TEST_DATA_H) $(TEST_DATA_BIN):
	python3 gen_soc_test.py --model $(MODEL)
```

目标不依赖：

- `MODEL`
- metadata / npy
- `gen_soc_test.py`
- golden manifest

只修改 `MODEL=` 时，旧 `test_data.bin` 可以继续使用。

修复：

- 用模型名 / manifest hash 进入目标文件名。
- 生成 `.stamp`，内容包含完整输入 hash。
- firmware 链接前校验 header 中 model name / bits / layer count。

### P1-6 driver CSR ABI 与 RTL 不一致

#### H/W 颠倒

- driver：`DIM_HW(h,w) = h | (w<<16)`
- RTL：`H=reg[31:16]`，`W=reg[15:0]`

当前 A–E 空间尺寸多为正方形，因此被隐藏。非方形 tensor 会交换 H/W。

#### padding 不一致

- driver：`top, bottom, left, right`
- RTL 实际只解码 `top=bits[7:0]`、`left=bits[15:8]`

driver 写到 bits[15:8] 的是 bottom，RTL 会当成 left。

#### clamp 不一致

- driver：low16=min，high16=max
- RTL PPU：只取 low16 作为 `clamp_max`

#### in_zp 丢失

- NPU2 固件把 `in_zp` 写入 `LAYER_MODE[23:8]`
- bare-metal driver 的 `LAYER_MODE(op,dtype)` 没有写 `in_zp`

修复：

以 RTL CSR decode 为唯一规范，建立 C/Python/RTL 共用的寄存器打包向量测试。先统一 ABI，再修实现。

### P1-7 driver 不能运行当前十格模型

`driver/baremetal/npu_driver.c`：

- 强制 `DMA_CTRL=0`
- 不写 tile input/output size、store mode、row config、weight-per-OC、2D stride
- residual Add 把 B 分支错误地指向 `cur_in`
- 不维护历史层输出地址
- I/O ping-pong 固定 512 KiB，API 不接收 buffer size

因此 driver README 中的 `npu_run_model()` 不能代表当前 SoC 固件已经验证的模型能力。

### P1-8 completion / bus fault 语义不可靠

- `STATUS_DONE` / `STATUS_ERROR` 由单周期 `hw_done/hw_error` 直接组合。
- bare-metal driver 轮询 `STATUS_DONE`，可能错过脉冲。
- IRQ_STATUS 是粘滞的，但 polling 路径没以它作为完成依据。
- DMA Wishbone 只有 ACK，没有 ERR/RTY，也没有硬件 watchdog。
- 总线永久 stall 会让 DMA 永久 BUSY，只能靠软件外部 reset。

修复：

- polling 使用 sticky IRQ_STATUS，或设计 sticky STATUS。
- 明确 DONE 与 ERROR 同时出现时 ERROR 优先。
- Wishbone wrapper 接 ERR；controller 映射成 error code。
- 增加可配置 DMA watchdog。

### P1-9 driver 没有 cache coherency 契约

driver 中没有 clean、invalidate、fence 或平台 callback。

在带 D-cache 的 FPGA CPU 上：

- NPU 可能看不到 CPU 尚未写回的输入 / 权重 / 参数。
- CPU 可能读到 DMA 写回前缓存的旧输出。

修复：

driver config 增加平台 cache hooks 和 memory barrier，并在所有 DMA 边界调用。

### P1-10 FPGA 没有真实综合 / 实现 / 时序证据

现有 `rtl/synth/npu_synth.ys` 是 generic Yosys sanity：

- `synth -noabc`
- 无 Xilinx techmap
- 无 XDC
- 无 BRAM/DSP 映射
- 无 place/route / STA

`stat_passB.txt` 仍把参数化 SRAM 展开成约百万级 FF/MUX，说明所谓 SRAM blackbox 没有覆盖 `$paramod` 实例。

因此它不能回答：

- XC7Z045 是否放得下
- DSP48 / BRAM36 使用量
- 100 MHz 是否收敛
- 是否存在未约束路径

---

## 4. P2：普通功能 / 维护缺口

### P2-1 NPU1 / NPU2 缺少完整边界校验

NPU2 固件只检查 magic：

- 不检查 blob version / size
- 不检查 header 加法溢出
- 不检查 DDR 地址范围
- `layer_out_addr` 只有 80 项，但第二遍按 `num_layers` 全部循环

NPU1 driver 同样没有接收 model buffer size，无法验证 `weight_offset`、descriptor 和 variable data 是否越界。

### P2-2 RTOS async API 不是真异步

`npu_rtos_run_model_async()` 只：

- 获取 mutex
- 记录请求
- 返回

NPU 真正启动发生在 `npu_rtos_wait()`。

这和 README “提交后立即运行，调用方做其他工作”的语义不符。

超时后也没有 abort/reset；硬件可能继续运行，随后 mutex 被释放，引起下一个请求和迟到 IRQ 串扰。

### P2-3 Resize 接受了硬件没有精确实现的 ONNX 语义

converter 只看 `mode=nearest/linear`，忽略：

- `coordinate_transformation_mode`
- `nearest_mode`
- 其他 Resize 坐标语义

CSIM / RTL 使用固定坐标公式。对于其他合法 ONNX Resize，converter 可能静默改变语义。

修复：

只接受已经 bit-exact 支持的属性组合；其他组合显式拒绝。

### P2-4 2D DMA 行尾要求未被硬件保护

多个 row length 使用 `bytes >> 2`。如果一行不是 4-byte 对齐，会静默丢掉 1–3 bytes。

当前 converter 尝试生成 word-aligned tile，但：

- CSR 是公开接口
- driver 可以写任意形状
- border tile / odd channels 仍需要明确约束

修复：

converter、driver、controller 三层都做 4-byte 对齐 guard，并加入最后一行 / 最后一列测试。

### P2-5 PTS fallback 长度存在重复乘高度风险

`npu_ctrl.v:811/903` 在 `tile_out_w_actual==0` 时：

```verilog
r_tile_row_len <= tile_row_len * tile_row_count;
```

`S_TILE_STORE` 又做：

```verilog
dma_xfer_len <= r_tile_row_len * r_tile_row_count;
```

fallback 路径可能按高度平方放大长度。当前模型通常提供 actual width，因此被隐藏。

### P2-6 HW_CONFIG scratchpad 单位不一致

- RTL 把 `SPAD_KB_P=192` 直接写入名为 `SPAD_SIZE_4KB` 的字段。
- driver 再乘 4，报告为 768 KiB。
- 实际冻结值是 192 KiB。

应明确字段是 KiB 还是 4 KiB 单位，并同步 RTL、driver、文档。

### P2-7 DB_EN 有双 bank，但当前不是计算/DMA真正重叠

计算核在 tile 完成后才发 `tile_done`，controller 随后才发下一 tile DMA。功能上是 ping-pong，性能上不是“算 N 时预取 N+1”。

因此：

- 不应直接宣称已有完整 overlap 收益。
- 需要用 PERF 和真实 DDR 测量。

### P2-8 DMA burst 尚未实现

`npu_dma.v` 注释声明 4/8/16/32-word burst，但实现是 single-beat Wishbone。

M1 DDR 性能规划不能基于现有 burst 宣称。

---

## 5. P3：文档和清理

1. `npu_top.v` 文件头仍写 “no ping-pong, no tiling loop”，已过期。
2. D8 UART 横幅仍显示 INT16。
3. `main.c` 保留 L15–L19 定向 DBG 输出。
4. `tools/README*.md` 的转换 / 测试示例与当前命令行不一致。
5. `soc/fpga/zc706_npu.py` 文档写 `--load --flash`，实现没有 programmer / load / flash 路径。
6. `firmware.hex` 和 `main_ram.init` 的时间早于当前 `main.c`；当前磁盘镜像已经落后于源码。
7. `soc` 中有 core dump、生成目录和 `__pycache__`，发布前要清理。

---

## 6. 已确认的正向项

以下不是完整验收，但属于有效基础：

- converter guard：8/8 PASS。
- NPU1 固定描述符 62 字节，Python 与 CSIM packed struct 基本对齐。
- INT8 已统一为 signed symmetric，反量化 `q * scale`。
- RTL 和 CSIM 使用 `ACC_WIDTH=44`。
- 软复位 / abort 会复位 DMA 和 compute engine。
- 固件每次 START 前清 sticky IRQ。
- DMA ACK backpressure、reset/abort、IRQ W1C 有专门测试。
- 十套 A–E golden metadata 都存在。
- D8 最后一轮 SoC 24 层全部 bit-exact。
- 本次 bare-metal / RTOS stub compile 均通过。

这些正向项不能抵消 P0/P1 的发布证据和 FPGA 集成缺口。

---

## 7. 修复路线

### 阶段 A：Phase-1 结果可信

目标：先让 PASS 可信，不急着加功能。

1. 修复 simulator：未见 `RESULT` 必须失败。
2. 运行时输出区 poison，禁止 golden 预填工作区。
3. 重写 golden 流程：
   - 无缓存
   - 临时目录
   - 非零失败退出
   - manifest + SHA-256
   - 原子替换
4. `phase1_regression.py` 纳入十格两道门。
5. 修复 / 删除所有 SKIP 和 expected-fail，使规范与测试一致。
6. 固化 commit 后重新跑十格并归档日志。

退出条件：

- 一条正式命令执行所有门。
- 十格结果均来自同一组 commit。
- 没有 recorded、SKIP、xfail、旧 golden 或未见 RESULT 的成功。

### 阶段 B：关闭 D8 精度和 fusion 定义

1. 用 D 专用掌静脉 calibration/evaluation。
2. 尝试非负 activation 量化、敏感层策略或混合精度。
3. 决定 Phase-1 是否承诺 fusion：
   - 不承诺：正式冻结 unfused DDR 模式。
   - 承诺：恢复 FUSE bits 并补四类组合测试。

退出条件：

- D8 固定多图集达到 ≥0.95。
- 文档、metadata、固件和 RTL 对 fusion 能力描述一致。

### 阶段 C：统一软件 ABI

1. 生成单一 CSR 规范。
2. 修 DIM/padding/clamp/in_zp。
3. driver 补 tile/PTS/stride/OC reload/残差。
4. 增加 buffer size、overflow 和 blob boundary 检查。
5. 加 cache hooks。
6. completion 改为 sticky IRQ 语义。
7. 修 RTOS async 与 timeout abort。

退出条件：

- C/Python/RTL 寄存器向量测试 bit-exact。
- driver 能运行至少 A/B/D/E 的代表模型，并对同一 CSIM golden。

### 阶段 D：ZC706 M0

1. 修 `wb_master` 属性。
2. 映射 NPU CSR slave 到 `0x80000000`。
3. 注册 IRQ。
4. 导入官方 ZC706 Master XDC，核对 clock/reset/UART/LED。
5. 建立 reset tree 和 MMCM lock 处理。
6. 强制 NPU 源按 SystemVerilog 读取。
7. 实现 firmware ROM/RAM 加载和 JTAG bitstream load。
8. 生成小于 512 KiB 且地址在 BRAM 窗口内的单层镜像。
9. Vivado synth/place/route，记录 utilization、WNS/TNS、DRC。
10. 上板执行 CSR ID、DMA、IRQ、单层 Conv bit-exact。

退出条件：

- 100 MHz 时序收敛。
- 板上单层输出与 CSIM bit-exact。
- bitstream、firmware、test blob 均有 hash。

### 阶段 E：ZC706 M1

1. 实现 LiteDRAM DDR3。
2. 完成 DDR calibration / memtest。
3. 实现总线 ERR / timeout。
4. 测 single-beat DMA 基线，再实现真正 burst。
5. A–E 十格板上回归。
6. IRQ / FreeRTOS 实跑。

退出条件：

- A–E 十格板上结果与同版本 CSIM bit-exact。
- 记录实际 DDR 性能、Fmax 和功耗/资源。

---

## 8. GO / NO-GO 清单

### Phase-1 GO 条件

- [x] D8 ≥ 0.95（MSE clip，0.952442）
- [ ] 正式入口包含十格两道门
- [ ] fresh golden 可机检
- [ ] 输出区不预填 golden
- [ ] simulator 未见 RESULT 必失败
- [ ] 零 SKIP / xfail
- [ ] fusion 能力声明与实际执行一致
- [ ] 十格同 commit、同工具链、日志有 hash
- [ ] 三个冻结仓库提交并打同名 annotated acceptance tag

当前：**NO-GO**

### ZC706 M0 GO 条件

- [ ] CSR slave / DMA master / IRQ 都接入
- [ ] 官方 XDC 和 reset tree
- [ ] 可生成、可加载 bitstream
- [ ] 小于 512 KiB 的 BRAM 单层镜像
- [ ] Vivado implementation + 100 MHz STA
- [ ] 板上 CSR / DMA / IRQ / 单层 bit-exact

当前：**NO-GO**

### ZC706 全模型 GO 条件

- [ ] 完成 M0
- [ ] DDR3 / LiteDRAM
- [ ] memtest 和 DMA fault handling
- [ ] 足够的模型/工作区内存
- [ ] A–E 十格板上回归

当前：**NO-GO**

---

## 9. 建议的下一步

不要先继续调 D8，也不要先跑 Vivado。第一步应修“结果可信度”：

1. simulator 无 RESULT 时失败
2. 移除 `ddr_out` golden 预填
3. fresh golden + manifest
4. 正式回归纳入十格

这四项完成后，再跑一次十格。否则继续修功能时，PASS 仍然可能不能作为发布证据。
