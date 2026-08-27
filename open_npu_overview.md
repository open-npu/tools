# Open-NPU 整机说明

这份文档是整颗 NPU 的地图。先讲它是什么、一次推理经过哪些人，再讲硬件、量化、切块、软件链、验收。切块 / 双缓冲 / 融合的信号级细节在 `tools/model_d_fused_tiling_explained.md`。

读完后应能回答：一张图从 ONNX 到 RTL 比对，中间每一步谁在干什么、数据在哪、怎样才算过。

---

## 0. 一句话

Open-NPU 是一颗 **INT8 / INT16 定点卷积加速器**，挂在 RISC-V SoC 上。CPU 不跑卷积；它只写寄存器、发启动、等中断，然后把 DDR 里的输出和参考结果比。

Phase-1 冻结配置：

| 项 | 值 |
|---|---|
| 脉动阵列 | 16×16，每拍最多 256 次乘加 |
| 片上便签本（scratchpad） | 192 KiB |
| 累加位宽 | 44 bit |
| 数据类型 | 有符号 INT8、有符号 INT16 |
| 模型描述符 | NPU1，每层固定头 62 字节 |

验收不是「某个算子的 cocotb 过了」，而是 **5 个整网 × 两种位宽 = 10 格**，每格都要：

1. CSIM 反量化结果 vs ONNX 浮点：余弦够高（INT8 ≥ 0.95，INT16 ≥ 0.99）
2. SoC RTL 链式推理 vs **当场** 由当前 converter+CSIM 生成的 golden：**逐 word 相同**

第二步过、第一步过，才说 RTL 和浮点对齐。没有单独的「RTL vs ONNX Runtime」门。

---

## 1. 仓库里有几层，各干什么

可以想成四层楼，从上到下：

```
ONNX 浮点模型
    │  converter（量化、切块、融合、打包）
    ▼
NPU1 二进制  ──────────►  CSIM（C 功能模拟，算 golden）
    │
    │  gen/*_golden.py + gen_soc_test.py
    ▼
NPU2 固件 blob + RISC-V 程序
    │
    ▼
SoC 仿真 / FPGA：VexRiscv 写 CSR → NPU RTL 算 → UART 打 PASS/FAIL
```

| 目录 | 角色 |
|---|---|
| `tools/` | 转换器、切块、打包、golden 生成、验收脚本、本说明 |
| `design/` | 架构 / 接口 / 寄存器规格（公开文档仓 `open-npu/design`） |
| `csim/` | C 模拟器 `npu_sim`。数值必须和 RTL 的 PPU / 累加位宽一致 |
| `rtl/` | 可综合 Verilog：控制器、DMA、计算核、脉动阵列、深度卷积、后处理 |
| `soc/` | LiteX + VexRiscv + 固件 + Verilator 整机仿真 |
| `driver/` | 独立 HAL / 驱动，和当前 SoC 固件不是同一条验收路径 |

`rtl/` 里还有 Icarus + cocotb 的模块测试。那是算子级。**整网对错以 `soc/sim` 的 Verilator 为准。** Icarus 跑 fused 大层可以卡几天，不代表模型本身要几天。

---

## 2. 一次推理里有几个人

三条线叠在一起：

```
CPU 固件 (soc/firmware/main.c)
    写 CSR → 发 START → 空转等 IRQ_DONE → 读 DDR 和 golden 逐 word 比

NPU 控制器 (rtl/src/npu_ctrl.v)
    搬权重 / 搬输入 / 搬量化参数 / 叫计算 / 把输出搬回 DDR
    切块、双缓冲、融合时，它还负责「下一块预取好了没」

NPU 计算核 (rtl/src/npu_compute.v)
    只认已经在片上 SRAM 里的一块数据
    按 tile → 输出通道组 循环
    真正乘加在脉动阵列或深度卷积引擎里，再进后处理单元（PPU）
```

名词：

- **DDR / 主存**：仿真里是 LiteX 的 `main_ram`，地址从 `0x40000000` 起。大，慢。权重、整图激活、golden 都在这。
- **SRAM**：NPU 旁边的小便签本。卷积不在 DDR 上乘加，必须先 DMA 搬进来。
- **CSR**：CPU 能读写的控制寄存器，基址 `0x80000000`。Wishbone slave。
- **DMA**：NPU 自己当 Wishbone master，在主存和三块 SRAM 之间搬 32-bit word。
- **层（layer）**：图里的一个算子，例如「14×14×256 的 1×1 卷积」。固件按 0、1、2… 各 START 一次。
- **链式推理**：第 N 层的 DDR 输出就是第 N+1 层的输入。不是每层都从打包好的独立输入重跑。

固件默认是 **整网链式**，不是单层 standalone。`STANDALONE_LAYER` 只在调试 FC 一类特例时打开。

---

## 3. 硬件长什么样

顶层是 `rtl/src/npu_top.v`。对外两根 Wishbone、一根中断：

```
                    Wishbone slave          Wishbone master
     CPU ─────────────────────────────────► CSR
                                              │
                                              ▼
                                           npu_ctrl
                                          /    |    \
                                       DMA   compute  IRQ(hw_done)
                                        │       │
                              ┌─────────┼───────┼─────────┐
                              ▼         ▼       ▼         ▼
                          权重SRAM   激活SRAM  参数SRAM   （外部主存）
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                    脉动阵列 16×16   深度卷积引擎      后处理 PPU
                    (Conv / FC)      (DW ≤7×7)
```

### 3.1 三块片上 SRAM

`SPAD_KB=192` 时，`npu_top.v` 按 word（4 字节）切：

| SRAM | 深度（word） | 大约容量 | 放什么 |
|---|---|---|---|
| 激活 | `192×64 = 12288` | 48 KiB | 输入块、输出块；双缓冲时从中间切开成两个银行 |
| 权重 | `192×128 = 24576` | 96 KiB | 当前层 / 当前输出通道组的权重 |
| 参数 | `192×16 = 3072` | 12 KiB | 每通道量化参数（14 字节/通道） |

激活 SRAM 装不下整张特征图时，就要 **切块（tile）**。转换器里的 `tools/tiling.py` 按这块预算算 `tile_h / tile_w`。

双缓冲打开时，激活 SRAM 对半：

```
bank0 = 前半    bank1 = 后半
```

一边算当前块，一边 DMA 预取下一块。细节见融合说明第 4–5 节。

### 3.2 脉动阵列（`npu_systolic.v` + `npu_pe.v`）

- **权重静止（weight-stationary）**：先把一小块权重装进 16×16 个 PE，再把激活从第 0 列流进去。
- 每个 PE：16-bit 数据口 × 16-bit，累加到 44-bit。INT8 进阵列前符号扩展。
- 输入通道比 16 深时，计算核做多趟部分和（partial sum），在累加器里加完再送 PPU。
- 输出通道比 16 多时，按 **OC 组** 切。一组算完，ctrl 可以再 DMA 下一组权重（`wgt_per_oc`）。

### 3.3 深度卷积（`npu_dw_conv.v`）

Depthwise：每个通道自己和自己的小核卷积，不做通道混合。硬件核最大 7×7，带 padding。并行通道数等于 `ARRAY_SIZE`（16）。

整图被一个核盖住、输出 1×1 的全局 DW（例如 MobileNet 末尾那种），走专用的 **按 16 通道切片流式** 路径，不要求整张激活同时住在 SRAM 里。

### 3.4 后处理 PPU（`npu_ppu.v`）

卷积在整数累加器里，还不是能存回 INT8/INT16 的激活。PPU 每通道做：

```
acc  → +bias_q  → ×M  → >>S（带四舍五入） → +zp  → clamp → ReLU/ReLU6 → 写出
```

4 级流水，每拍一个结果。`ARRAY_SIZE` 条 lane 并行。模式：

| `POST_CTRL[1:0]` | 名字 | 用途 |
|---|---|---|
| 0 | CONV_REQ | 卷积 / DW / FC 之后的重量化 |
| 1 | ADD | 残差两边先按各自 M/S 缩放再加 |
| 2 | RELU_ONLY | 只做激活 |
| 3 | PASSTHROUGH | 累加器截断后直通 |

CSIM 的 `csim/src/postproc.c` 按同一条流水实现。两边 `ACC_WIDTH` 必须都是 44，否则 INT16 大层会对不齐。

### 3.5 DMA（`npu_dma.v`）

只搬 32-bit word。

- **1D**：连续搬 `xfer_len` 个 word。权重、参数、不切块的整层激活走这条。
- **2D**：`row_len × row_count`，行与行之间跳 `stride` 字节。切块时从 NHWC 大图里抠一块、或把一块写回大图，走这条。

历史上出过：1D 权重搬运误用了激活的 `in_stride`，后面的字全搬错。规则现在是：**1D 永远连续；stride 只给 2D 激活。**

### 3.6 控制器（`npu_ctrl.v`）

一层不切块、不融合时的固定套路：

1. DMA：DDR 权重 → 权重 SRAM
2. DMA：DDR 输入 → 激活 SRAM
3. DMA：DDR 量化参数 → 参数 SRAM
4. 拉高 `compute_start`，等 `compute_done`
5. DMA：激活 SRAM 输出 → DDR
6. 脉冲 `hw_done`；CSR 锁成 `IRQ_DONE`

切块 / 双缓冲 / 融合会在第 2、5 步插入循环或跳过。见第 7 节和融合专文。

软复位（`CTRL_SOFT_RST`）让控制器回到空闲，取消进行中的 DMA/计算。SRAM 内容和已写的 CSR **故意保留**。`ABORT` 取消在飞事务，且不会自动再开一层。

---

## 4. 数据在内存里长什么样

硬件和 CSIM 的激活都是 **NHWC**：同一像素的全部通道挨在一起，然后下一列、下一行。

```
地址增大 →
[h0 w0 c0][h0 w0 c1]…[h0 w0 cC-1][h0 w1 c0]…[h1 w0 c0]…
```

ONNX / ONNX Runtime 一般是 NCHW。转换器负责：

- 标定、跑浮点：NCHW
- 量化后的输入 bin、CSIM、RTL：NHWC
- 权重：按输出通道排，再按核空间和输入通道，和阵列装载顺序一致

DMA 和 CPU 都按 **32-bit word** 看内存：

- INT8：一 word 装 4 个有符号字节
- INT16：一 word 装 2 个有符号半字

比对也是逐 word。所以「1568/1568 mismatches」的意思是 1568 个 32-bit 字全不对，不是 1568 个像素。

切块写回时，一块在 SRAM 里是 **紧凑矩形**（没有整图行间距）。要写回 DDR 里的大 NHWC 图，必须开 **PTS（per-tile store）**：DMA 按行写，行距 = `out_w × out_c × 每元素字节数`。否则下一块会覆盖上一块，或只在地址 0 留下最后一块。

---

## 5. 量化：浮点怎么变成整数

转换器 `tools/onnx_converter.py` 做训练后量化（PTQ）：

1. 用标定图跑 ONNX Runtime，记每层激活范围，得到 **每张量** `scale_in` / `scale_out`
2. 权重 **按输出通道** 量化，得到 `scale_w[ch]`
3. 把 `scale_in × scale_w[ch] / scale_out` 收成硬件能乘的 `M[ch] × 2^(-S[ch])`
4. bias 收成 `bias_q = round(bias / (scale_in × scale_w[ch]))`，存在 64-bit，送进 44-bit 累加器时截断

推理时 **没有浮点乘法**。硬件只做上面那条整数流水。要对回浮点（算余弦）时：

```
float ≈ q × scale_out
```

本项目的 INT8 激活按 **有符号、零点 0** 存。转换器里若出现 `output_zp=128`，那是历史包袱，读端 **不要** 再减 128。错减一次，INT8 余弦会从 ~1.0 掉到 0.4 这种假失败。

INT16 同理：有符号，反量化仍是 `q × scale`。

加法（残差）两边尺度往往不同。PPU 的 ADD 模式用两套 `(M_A, S_A)` / `(M_B, S_B)`，先对齐再加，再按输出尺度写出。

---

## 6. 支持哪些算子

编码在 CSR `LAYER_MODE` 低 4 位，和 NPU1 / CSIM 一致：

| 编码 | 算子 | 谁算 | 备注 |
|---|---|---|---|
| 0 | Conv2D | 脉动阵列 + PPU | 普通卷积；分组卷积只接受「真 depthwise」 |
| 1 | DW Conv | 深度卷积引擎 + PPU | 核 ≤ 7×7 |
| 2 | FC | 脉动阵列（空间 1×1） | 不切空间块 |
| 3 | Pool | 计算核 | Max / Avg / Global |
| 4 | Eltwise Add | 计算核 + PPU ADD | 残差；另一路地址在 `DMA_ADD_B_ADDR` |
| 5 | Resize | 计算核 | 最近邻；坐标按 **整图** 算，切块时要带 origin |
| 6 | Deconv | 计算核 | 转置卷积 |
| 7 | Concat | 计算核 | 按通道拼；固件按 phase 只比对本 phase 拥有的通道 |

图级还会先做一层 **算子融合（不是 SRAM 融合）**：Conv+BN+ReLU、Clip(0,6) 收成 ReLU6。转换器拒绝：

- 任意没实现的 ONNX 算子（不会默默删掉）
- 非 depthwise 的 grouped conv
- 不是 ReLU / Clip(0,6) 的 Clip
- 把非零 mean/std 折进带 padding 的第一层卷积（边界像素和中心像素偏置不同，一个通道 bias 表达不了）
- 算不出既装得进 SRAM、又 word 对齐的切块方案的模型

---

## 7. 一层怎么切、怎么融、怎么和下一层接

三件独立的事，容易混：

### 7.1 切块

激活 SRAM 只有约 48 KiB。14×14×256 的 INT16 已经约 100 KiB。于是把输出平面切成矩形：

- `tile_h` / `tile_w`：一块标称高宽
- `tile_num_h` / `tile_num_w`：高、宽方向各几块
- 最右 / 最下一块会裁短（余数 tile）

计算核按 `tile_y → tile_x → oc_group` 循环。控制器自己另记 `tile_x_seq / tile_y_seq`，用来算写回 DDR 的偏移。**两套坐标必须同步**，否则最后一块会写到错误地址。

`tile_h == 0` 表示这层不切块，整张图一次算完。

### 7.2 双缓冲 `DB_EN`（`sched_ctrl` bit0）

算块 N 的同时预取块 N+1。靠一对握手：

- `tile_done`：计算核「非最后一块算完了」（一拍脉冲）
- `db_prefetch_done`：控制器「下一块输入已经在对面银行」

`db_prefetch_done` **不是** 层完成。计算核在非最后一块之后会进入 `S_TILE_WAIT_DB` 死等它。控制器如果既不预取、又不重新拉高这个信号，仿真就会在某一块上永远转，Icarus 看起来像「卡了三天」。

### 7.3 SRAM 融合 `FUSE_*`（bit1/2/3）

目标：MobileNet 式 `1×1 → DW 3×3 → 1×1` 中间张量不回 DDR。

| 位 | 名字 | 本意 |
|---|---|---|
| bit1 | FUSE_START | 融合块第一层：要 DMA 输入，算完 **不** 写回 DDR |
| bit2 | FUSE_MID | 中间层：不搬输入（SRAM 里已有），不写回 |
| bit3 | FUSE_END | 最后一层：不搬输入，算完写回 DDR |
| bit4 | PTS | 每块按 NHWC 写回 DDR（级联推理需要） |

**只有前后两层切块网格相同、中间结果能就地复用时，跳过 DMA 才安全。** 网格不同（例如 6×4 再接 2×2）时，中间结果必须进 DDR，再按下一层的网格重新 2D 加载。

控制器里现在的条件是：

```
skip_act_load = (fuse_mid | fuse_end) & (tile_h == 0)
skip_store    = (fuse_start | fuse_mid) & (tile_h == 0)
```

也就是：**切块的融合不再靠「跳过 DMA」省带宽**，改走 DDR + PTS。不切块的融合仍可 SRAM 直通。

D16 的 L9–L11 切块网格不同，修完这条之后整网 bit-exact。D8 的 L16–L18 是「切块 START + 不切块 MID/END」，仍是当前缺口。信号级时间线见 `tools/model_d_fused_tiling_explained.md`。

### 7.4 固件怎么把层串起来

`main.c` 对每一层：

1. 解析输入从哪来：第 0 层用打包输入；否则用上一层 `ddr_out`；skip / 残差用 `input_src` / `residual_src`
2. Add / Concat 再解析 branch B 的 DDR 地址
3. 不是 FUSE_MID/END 则软复位（清掉上一层状态）
4. **每次 START 前 W1C 清 `IRQ_DONE`**。不清的话，`npu_wait_done()` 会立刻看到上一层的锁存，以为这层做完了，实际 NPU 没干活。
5. 写完全部 CSR，`CTRL_START`，等中断
6. 冲 cache，逐 word 比 DDR 输出和 blob 里的 golden。运行时 `ddr_out` 预填的是 poison（`0xDEADBEEF`），不是 golden；`PERF_Lx npu=0` 直接判 FAIL。

`sched_ctrl` 写进 `REG_DMA_CTRL`。当前固件曾把 FUSE 三位清掉，强迫每层都 DMA，用来隔离 D8 第二段融合；那是调试态，不是架构终态。

`sched_ctrl` 写进 `REG_DMA_CTRL`。当前固件曾把 FUSE 三位清掉，强迫每层都 DMA，用来隔离 D8 第二段融合；那是调试态，不是架构终态。

---

## 8. 软件链：从 ONNX 到「L11 PASS」

完整、干净的一条命令：

```bash
soc/run_model_e2e.sh model_d_int16
```

它做三步，**禁止复用旧 npy / 旧 blob**：

### 8.1 转换 + CSIM dump

`tools/gen/model_*_golden.py`：

1. `convert_model()`：ONNX → NPU1（`0x4E505531`）
2. 跑 `/data/sam/open-npu/csim/npu_sim`，`DUMP_LAYERS=1`，`ACC_WIDTH=44`
3. 把每层 `/tmp/csim_layer_XXX.bin` 和权重、参数打成  
   `rtl/tb/golden/golden_dma_e2e/<model>/layer_XX_{wgt,param,input,output}.npy`  
   以及 `metadata.json`（尺寸、地址、`sched_ctrl`、切块）

中间 fused 层在 CSIM 里可能没有独立 dump（没写回）。生成脚本若用「同形状的更早一层」去填 L11 输入，得到的 packed 单层测试是 **无效** 的。整网链式才是真的。

### 8.2 打固件

`soc/firmware/gen_soc_test.py` 读上述 golden，写出：

- `test_data.bin`：魔数 `NPU2`（`0x4E505532`），每层 36 个 word 的头，后面跟打包的权重 / 参数 / 输入 / golden 输出
- `soc_test_data.h`：C 里的基址和布局

再 `make MODEL=model_d_int16` 编 RISC-V 程序，链进 `main_ram.init`。

**`make MODEL=...` 在 `test_data.bin` 已存在时可能是空操作。** 换模型或换 golden 必须 `make clean && make MODEL=...`。否则 D8 会默默跑着 D16 的数据。

主存布局（仿真）：

| 地址 | 内容 |
|---|---|
| `0x40000000` | 程序 / 数据 |
| `0x40010000` | NPU2 blob（层表 + 部分 inline） |
| `0x40200000` 一带 | 权重、参数、各层 DDR 输入输出（从 metadata 的 `0x30xxxxxx` 重映射过来） |
| `0x80000000` | NPU CSR |
| `0xF0001800` | LiteX UART |

注意：生成脚本会把 **golden 预填进 `ddr_out`**。NPU 没写到的字，CPU 比对仍会 PASS。所以「只写了最后一块、却整层几乎全过」是假过。看 `PERF_Lx npu=`：为 0 就是没干活。

### 8.3 SoC Verilator

`soc/sim/obj_dir/Vsim`：VexRiscv + NPU + 主存。UART 打每层 PASS/FAIL，最后 `RESULT: PASS/FAIL`。

这才是 Phase-1 的 RTL 门。不要用 `rtl/tb/e2e/test_npu_dma_e2e.py` 或归档在 `debug_archive/` 里的旧单层测试代替整网。

---

## 9. 两种二进制，不要混

| | NPU1 | NPU2 |
|---|---|---|
| 魔数 | `0x4E505531` | `0x4E505532` |
| 谁读 | CSIM `npu_sim` | SoC 固件 |
| 层描述 | 62 字节 fixed_config + 变长参数 | 36×4 字节 CSR 镜像头 |
| 产出 | `*.npu1.bin` | `soc/firmware/test_data.bin` |

转换器只保证 NPU1。NPU2 是 golden 脚本 + `gen_soc_test.py` 的二次打包，为的是固件能直接 `NPU_REG(...) = e[k]`。

`driver/` 里的 `npu_run_model()` 走 NPU1 映射，和当前 Verilator 验收固件不是同一条路径。

---

## 10. Phase-1 十格模型

| 格 | 模型 | 输入 | 大约层数 | 余弦门 | 角色 |
|---|---|---|---|---|---|
| A8 / A16 | MobileNetV2 | 3×224×224 | 63 | 0.95 / 0.99 | 深度卷积、全局池、长链 |
| B8 / B16 | 掌静脉活体 | 1×112×112 | 25 | 同上 | 切块 + stride DMA |
| C8 / C16 | YOLO-Tiny | 3×416×416 | 17 | 同上 | 大图、Concat、很多 tile |
| D8 / D16 | 掌静脉识别 | 1×112×112 | 24 | 同上 | SRAM 融合 + 不同网格切块 |
| E8 / E16 | ResNet-18 | 3×224×224 | 31 | 同上 | 残差 Add、常规卷积 |

脚本：

- 余弦：`tools/phase1_model_matrix.py`（只做门 1；验收必须加 `--rtl`）
- 整网回归入口：`tools/phase1_regression.py`（正式门，full 会跑十格两道门）
- 硬件冻结项：`tools/PHASE1_ACCEPTANCE.md`

---

## 11. 现在十格到哪了

以 **当前 converter + 当场 CSIM golden + SoC Verilator** 为准。旧日志、七月的 npy 作废。

| 格 | CSIM ↔ 浮点 | RTL ↔ 当前 CSIM |
|---|---|---|
| A8 | ~0.975 | PASS（63 层） |
| A16 | ~1.000 | PASS |
| B8 | ~1.000 | PASS（25 层） |
| B16 | ~1.000 | PASS |
| C8 | ~0.999 | PASS |
| C16 | 1.000 | PASS（17 层，曾经超时，现已跑完） |
| D8 | **0.952（MSE clip，过 0.95）** | **PASS（24 层，对 MSE golden）** |
| D16 | ~1.000 | PASS（含 fused L11、L19） |
| E8 | ~0.989 | PASS（31 层） |
| E16 | 1.000 | PASS |

十格两道门都过了：余弦（D8 正式 PTQ 是 MSE 激活截断，0.952442）和 RTL↔CSIM bit-exact。

D8 第二段融合的根因：L16 切块写出后，固件仍让 L17/L18 从上一层 SRAM `out_base` 读，而强迫走 DDR 时 DMA 把输入放在 SRAM[0]。两边对不上。现已改为每层软复位、`act_base=0`、FUSE 位不进 `DMA_CTRL`，L16–L23 当场 PASS。

---

## 12. CPU 视角：要写哪些寄存器

固件 `npu_program_layer()` 和 cocotb `program_layer()` 顺序一致。分组：

| 区 | 偏移 | 内容 |
|---|---|---|
| 控制 | `0x000–0x01C` | START / 软复位 / 状态 / IRQ / 版本 / 性能计数 |
| 层几何 | `0x040–0x078` | 算子、INT8/16、H/W/C、核、步长、pad、池化/resize/deconv/concat、切块、SRAM 基址 |
| DMA | `0x100–0x148` | 入/出/权重/参数/AddB 地址与长度、stride、`sched_ctrl`、PTS、每 OC 权重 |
| 后处理 | `0x180–` | `POST_CTRL`、通道数、clamp |

`LAYER_MODE` 打包：`op[3:0] | (int16<<4) | (in_zp<<8)`。

`SRAM_BASE`：低 16 位输入基址（word），高 16 位输出基址。不切块融合的 MID/END，输入基址必须是 **上一层写出的位置**，不是 0（0 是上一层的输入）。

启动：写 `CTRL_START`。结束：`IRQ_STATUS` bit0，写 1 清除。固件在等的时候也会看 `STATUS_ERROR` / 超时。

---

## 13. 怎样跑、怎样读日志

```bash
# 干净的一整格（转换 + 新 golden + 固件 + 仿真）
soc/run_model_e2e.sh model_a_int8

# 只重跑仿真（golden/固件已是当前的）
cd soc/sim && ./obj_dir/Vsim

# 换模型：必须 clean
cd soc/firmware && make clean && make MODEL=model_d_int8
cd ../sim && ./obj_dir/Vsim

# 只看浮点余弦（不是 RTL 门）
python3 tools/phase1_model_matrix.py
```

UART 里一层正常结束像：

```
  L11: PASS (12544 words)
  PERF_L11 npu=1234567 mac=89012
```

要警惕的：

- `PASS` 且 `npu=0`：现在应直接 FAIL。若仍出现，是固件检查没生效。
- `FAIL k/k` 且 `got=0xDEADBEEF`：NPU 没覆盖工作区（poison 还在）
- `FAIL` 只在缓冲开头的一小截：只写了最后一块、偏移没往前走
- 横幅仍写「Palm Vein INT16」：字符串写死了，D8 也会显示这句，以 `MODEL=` 和 L0 word 数（INT8 的 L0 是 50176 word）为准

Icarus 单层 fused 诊断（`rtl/tb/e2e/test_md_l11_diag.py`）不要当整网进度。它慢、且曾用错输入。

---

## 14. 常见误判（比代码本身更容易让人迷路）

1. **复用旧 golden。** 改过 converter / CSIM / 量化公式之后，磁盘上的 npy 全部作废。
2. **INT8 减 128。** 有符号、zp=0，反量化是 `q * scale`。
3. **算子 cocotb 过了就当 Phase-1 过了。** 十格整网才算。
4. **Icarus 卡死 = 模型要算三天。** 多半是 `db_prefetch_done` 握手没完成；同一条网在 Verilator SoC 上几分钟。
5. **`make MODEL=` 没 clean。** 固件仍链着上一格的 `test_data.bin`。
6. **PASS + npu=0。** 见第 7.4 节。
7. **融合层没有 CSIM dump，却拿别的层同形状数据当输入。** 单层 packed 测试无效。
8. **把 `db_prefetch_done` 当成层完成。** 它只表示「下一块预取好了」。

---

## 15. 关键文件

硬件：

- `rtl/include/npu_defines.vh` — 阵列、SRAM、位宽、CSR 基址
- `rtl/src/npu_top.v` — 顶层互连
- `rtl/src/npu_ctrl.v` — 一层的 DMA/计算调度
- `rtl/src/npu_compute.v` — tile / OC 循环，派发各算子
- `rtl/src/npu_dma.v` — 1D/2D 搬运
- `rtl/src/npu_systolic.v` / `npu_pe.v` / `npu_dw_conv.v` / `npu_ppu.v` / `npu_csr.v`

软件：

- `tools/onnx_converter.py` — ONNX → NPU1
- `tools/model_packer.py` — NPU1 布局、`sched_ctrl` 位
- `tools/tiling.py` / `layer_fusion.py` / `hw_config.py`
- `tools/gen/model_*_golden.py` — 当场 CSIM → npy
- `csim/npu_sim` — 功能 golden
- `soc/firmware/main.c` — 链式调度和比对
- `soc/firmware/gen_soc_test.py` — npy → NPU2 blob
- `soc/run_model_e2e.sh` — 一格的干净入口

验收与说明：

- `tools/PHASE1_ACCEPTANCE.md` — 正式门（偏仓库回归）
- `tools/phase1_model_matrix.py` — 十格余弦
- `tools/model_d_fused_tiling_explained.md` — 融合 / 双缓冲信号
- 本文 — 整机地图

---

## 16. 想接着查时从哪抠

先问自己卡在哪一层楼：

| 现象 | 先看 |
|---|---|
| 余弦低、RTL 已 bit-exact | 标定、scale、zp；不要改 timeout |
| 某层 FAIL、上一层 PASS | 这一层的输入地址、stride、是否真的跑了（`PERF`） |
| fused 中间层无 golden | 看融合块的 **END** 层和它后面的 Add |
| 仿真永远不打下一层 | ctrl 是否在 `tile_done` 后重新拉高 `db_prefetch_done` |
| 刚换模型结果离谱 | `make clean`、L0 尺寸是否对应当前位宽 |

D8 余弦门已过（MSE clip，0.952）。换 PTQ 后必须重新出 CSIM golden 再比 RTL。
