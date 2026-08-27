# MODEL_D 分层、切块、双缓冲、融合：从一次启动说到这次的 bug

整机地图见 `design/open_npu_overview.md`。本文只讲切块、双缓冲、融合，以及 MODEL_D 上踩过的坑。

这份说明按「固件点一次 START 之后硬件在干什么」来写。
先把名词说清楚，再讲 MODEL_D 的 L9–L11，最后才是这次修了什么。

涉及的主要文件：

- 固件：`soc/firmware/main.c`（排层、等完成、比对）
- 控制器：`rtl/src/npu_ctrl.v`（DMA 和计算谁先谁后）
- 计算核：`rtl/src/npu_compute.v`（真正做卷积，按块往前走）
- 调度位：`design/model_packer.py` 里的 `sched_ctrl`

---

## 1. 一次推理里有几个人

可以想成三条流水线叠在一起：

```
CPU 固件          写 CSR → 发 START → 空转等 IRQ_DONE → 读 DDR 比对
                    │
NPU 控制器 npu_ctrl  搬权重 / 搬输入 / 叫计算 / 把输出搬回 DDR
                    │
NPU 计算核 compute   只认 SRAM 里已经放好的一块数据，算完脉冲通知 ctrl
```

DDR 是大内存，SRAM 是 NPU 旁边的小便签本。卷积不能直接在 DDR 上乘加，必须先 DMA 搬进 SRAM。

一层（layer）= 图里的一个算子，比如「14×14×256 的 1×1 卷积」。
整网 MODEL_D 有 24 层。固件按 0、1、2… 顺序各 START 一次。

---

## 2. 一层不切块时，ctrl 的固定套路

`npu_ctrl.v` 文件头写的就是这个顺序：

1. DMA：DDR 权重 → 权重 SRAM
2. DMA：DDR 输入 → 激活 SRAM（activation SRAM）
3. DMA：DDR 量化参数 → 参数 SRAM
4. 拉高 `compute_start`，等 `compute_done`
5. DMA：激活 SRAM 里的输出 → DDR
6. 给 CSR 一个 `hw_done` 脉冲；CSR 把它锁成 `IRQ_DONE`

固件 `npu_wait_done()` 看到 `IRQ_DONE` 就认为这层结束，然后用 CPU 把 DDR 里的输出和 golden 逐 word 比。

---

## 3. 为什么要切块（tile）

激活 SRAM 只有一百多 KB。14×14×256 的 INT16 已经约 100 KB，再加重叠的 pad、输出，一张图常常塞不进去。

于是把空间拆成矩形块。例如 D16 的 L11：

- 整图输出 14×14×128
- 每块标称 3×6
- 排成 5 行 × 3 列 = 15 块
- 最右/最下一块会裁短（余数 tile）

CSR 里对应：

| 名字 | 含义 |
|---|---|
| `tile_h` / `tile_w` | 一块有多高、多宽 |
| `tile_num_h` / `tile_num_w` | 高、宽方向各几块 |
| `tile_x` / `tile_y`（compute 里） | 当前算到第几块 |
| `tile_x_seq` / `tile_y_seq`（ctrl 里） | ctrl 自己记的块坐标，用来算写回 DDR 的偏移 |

两套坐标必须同步。不同步的话，最后一块的结果会被写到错误的 DDR 地址。

---

## 4. 双缓冲 DB_EN：一边算、一边搬下一块

`sched_ctrl` bit0 = `DB_EN`（double-buffer enable）。

激活 SRAM 从中间切开，变成两个银行（bank）：

```
bank0 = SRAM 前半    bank1 = SRAM 后半
```

`ping_pong_flag`：compute 现在读哪一半。0 读前半，1 读后半。

设计意图：

```
时间 →
compute:  [ 算 tile0 @ bank0 ] [ 算 tile1 @ bank1 ] [ 算 tile2 @ bank0 ] ...
DMA:      [ 搬 tile1 → bank1 ] [ 搬 tile2 → bank0 ] [ 搬 tile3 → bank1 ] ...
```

算 tile N 的同时，把 tile N+1 搬进来。这叫 prefetch（预取）。

---

## 5. `tile_done` 和 `db_prefetch_done` 在干什么

这是这次卡住时在等的那对握手。它们**不是**「整层做完」，只是「这一块做完，下一块输入好了没」。

### compute 侧（`npu_compute.v`）

每块卷积结束，如果后面还有块：

1. 脉冲 `tile_done`（只亮一个时钟）
2. 自己进入状态 `S_TILE_WAIT_DB`，停住
3. 看输入信号 `db_prefetch_done`
   - 为 1：下一块已经在对面银行里，进入 `S_TILE_SETUP` 继续
   - 为 0：认为预取还没完，每个时钟再看一次

最后一块不发 `tile_done`，直接 `compute_done`，整层结束。

### ctrl 侧（`npu_ctrl.v`）

看到 `tile_done` 且开了 `DB_EN` 时，本来应该：

1. 把 `db_prefetch_done` **拉低**（告诉 compute：先别动）
2. 如开了 PTS（见下一节），先把当前块输出写到 DDR
3. 再 DMA 预取下一块输入
4. DMA 完成：翻转 `ping_pong_flag`，把 `db_prefetch_done` **拉高**
5. compute 被放开，算下一块

所以：`db_prefetch_done = 0` 的意思是「下一块输入还没就绪」。
compute 等它，是为了避免读到空银行或半成品。

---

## 6. PTS：按块写回 NHWC 的 DDR

`sched_ctrl` bit4，以及 CSR `store_mode` bit0，叫 PTS（per-tile store）。

不切块时，整张输出在 SRAM 里是连续的，最后一次 DMA 整段写回即可。

切块 + 双缓冲时，SRAM 里永远只有**当前这一块**的输出。必须每块算完就写回 DDR，而且要写到这张图在 NHWC 布局里该在的位置：

```
DDR 偏移 = (tile_y * tile_h * out_w + tile_x * tile_w) * out_c * 每元素字节数
```

这就是 ctrl 里的 `r_tile_ddr_offset`，用的是 **ctrl 的** `tile_x_seq / tile_y_seq`。
如果 `tile_done` 被忽略，这两个计数器不加，最后一块会被写到 offset 0（图的左上角）。

NHWC = 内存顺序是「一行里的像素，每个像素先排完所有通道」。

---

## 7. Fusion：本想少搬几次 SRAM

转换器会把常见的小块

```
Conv 1×1  →  Depthwise 3×3  →  Conv 1×1
```

标成一个 fused block（融合块）。`sched_ctrl` 的 bit1–3：

| 位 | 名字 | 这层在块里的角色 |
|---|---|---|
| bit1 | `FUSE_START` | 第一层：从 DDR 读入，输出希望留在 SRAM |
| bit2 | `FUSE_MID` | 中间层：输入已在 SRAM，输出也留在 SRAM |
| bit3 | `FUSE_END` | 最后一层：输入已在 SRAM，输出写回 DDR |

ctrl 里对应两个开关（**不切块时才成立**）：

```
skip_act_load = 中层或末层     →  不要再从 DDR 搬输入
skip_store    = 首层或中层     →  不要把输出写回 DDR
```

不切块、整张图能放进 SRAM 时，这是对的：三层共用同一本便签，少两次大 DMA。

---

## 8. MODEL_D 的 L9–L11 为什么不能这么融合

D16 这三层都切了块，而且**切法不一样**：

| 层 | 角色 | 算子 | 输入 | 输出 | 切块 |
|---|---|---|---|---|---|
| L8 | 融合块之前 | 1×1 | 14×14×256 | 14×14×128 | 5×3，块 3×6 |
| L9 | FUSE_START | 1×1 | 14×14×128 | 14×14×256 | 3×4，块 6×4 |
| L10 | FUSE_MID | DW 3×3 pad=1 | 14×14×256 | 14×14×256 | 7×7，块 2×2 |
| L11 | FUSE_END | 1×1 | 14×14×256 | 14×14×128 | 5×3，块 3×6 |

L9 按 6×4 写 SRAM，L10 按 2×2 去读，格子对不上。
双缓冲下 SRAM 里还只有「当前一块」，不是整张 14×14。

所以「输出留在 SRAM 给下一层用」只对「不切块、三层同一张全图」成立。
L9–L11 这种**切块网格不同**的融合，必须退回普通链式：每层从上一层的 DDR 输出做二维 DMA 读入，算完 PTS 写回自己的 DDR。

---

## 9. 这次实际踩到的三件事

### 9.1 固件把融合位掩码写错，L9/L10 根本没算

`IRQ_DONE` 是粘滞的：硬件做完拉高，CPU 必须写 1 才能清掉（W1C）。

`npu_wait_done()` 一看到这位是 1 就返回。

旧代码：

```c
if (l > 0 && !(sched & 0x06)) {  /* 注释写着「不是 MID/END」 */
    软复位;
    清 IRQ;
}
```

`0x06` 是 bit1|bit2 = **START|MID**，不是 MID|END。

结果：

- L8 算完，`IRQ_DONE` 仍是 1
- L9 是 FUSE_START，不进 if，不清 IRQ
- `wait_done()` 立刻返回（约 5k cycle）
- L10 同样秒退
- L11 是 FUSE_END（bit3），`0x06` 配不上，反而会复位、会真的跑
- L11 读到的是从未写过的 L10 输出缓冲

所以日志里 L9/L10「skip verification」而且几乎 0 时间，L11 全错。

已改：用 `0x0C`（MID|END）决定要不要软复位；**每次 START 前都清 IRQ**。

### 9.2 切块融合仍走 `skip_act_load`，写回只发生一次且写错位置

旧 RTL：

```
skip_act_load = fuse_mid | fuse_end;   // 不管切不切块
tile_done 处理: if (DB_EN && tile_done && !skip_act_load)
```

L11 切了块却 `skip_act_load=1`：

- 中间各块的 `tile_done` 被丢掉
- `tile_x_seq` 不加
- PTS 不跑
- 只有最后一块在 `compute_done` 时写 DDR，偏移按 (0,0) 算 → 写到缓冲开头

### 9.3 工作区 DDR 预先填了 golden，把 9.2 伪装成「只错一小块」

`build_main_ram.py` 会把每层 golden **同时**写到比对用的 blob，以及该层的 `ddr_out` 工作区。

没被 DMA 覆盖的 word，比对时等于 golden，显示 PASS。

因此旧 D16 L11 看起来是 **256/12544**：

- 256 word = 最后一块余数（2×2×128 INT16）
- 写在 offset 0，所以从 `w[0]` 开始错
- 其余 12288 word 仍是预填的 golden

并不是卷积只算错了 256 个点。
打开「每块都写回」之后，L11 变成 12544/12544，这才是真实的「L10 没产出、L11 在读垃圾」。

L13 `npu=0` 却 PASS，也是同一类：NPU 没干活，比对的是预填值。看一层是否真跑了，要看 `PERF_Lx npu=` / `mac=`，不能只看 PASS。

---

## 10. Icarus 为什么在 `db_prefetch_done` 上空等三天

隔离测试跑的是 **融合 + 切块 + DB_EN** 的 L10。

compute 每块结束都进 `S_TILE_WAIT_DB` 等 `db_prefetch_done`。
这是为普通双缓冲预取准备的，compute **不知道**自己在 fused 中层。

ctrl 在 `skip_act_load` 时把整个 `tile_done` 分支跳过：

- 不做预取（这点符合融合）
- 也不再拉高 `db_prefetch_done`（这点漏了）

只要该信号是 0（上一层 PTS 拉低过、或某次边界拉低后没人拉回），compute 就每个时钟空等。
Icarus 仍在仿真这些空转，CPU 100%，日志停在 L10 `tile(6,2)`。

这和「模型要算三天」无关。SoC Verilator 全程 D16 大约 5 分钟。
超时按仿真时间计，挂死后仿真时钟几乎不走，墙钟可以空转很多天。

正确行为：fused 中层不预取时，ctrl 收到 `tile_done` 也必须：

- 推进 `tile_x_seq`
- 把 `db_prefetch_done` 保持/拉回 1

这样 compute 才会进下一块。

---

## 11. 现在代码里的约定

`npu_ctrl.v`：

```
skip_act_load = (中层或末层) 且 不切块
skip_store    = (首层或中层) 且 不切块
```

切了块的融合层，按普通链式做：二维 DMA 读上一层 DDR，PTS 写自己的 DDR。

`gen_soc_test.py`：即使某层没有 golden npy（L9/L10），只要 `dma_out_size > 0`，也要分配 `ddr_out`，否则下一层没有可读的地址。

固件：START 前必清 `IRQ_DONE`；只有真正要复用 SRAM 的不切块 MID/END 才跳过软复位。

---

## 12. 对照日志时怎么读

| 日志 | 含义 |
|---|---|
| `L11: PASS (6272 words)` | CPU 比对过，长度对得上。D8 是 6272，D16 是 12544 |
| `FAIL — 256/12544` 且从 `w[0]` 起 | 很像「只写了最后一块到开头 + 其余是预填 golden」 |
| `no golden output, skipped verification` | 这层没有可比的 npy，不代表算对了 |
| `PERF_Lx npu=5k` 而同尺寸层通常是几百万 | 这层几乎没进 NPU（常见于 IRQ 秒退） |
| `PERF_Lx npu=0 mac=0` 却 PASS | 优先怀疑预填 golden，NPU 没覆盖工作区 |
| `[CMP_TILE] tile(y,x)` | compute 开始一块；长时间停在同一块且无下一块，就是握手或核内死循环 |

验收标准仍是：CSIM 对 ONNX 浮点看余弦；RTL 对**当场** CSIM 要 bit-exact。预填 golden 只是仿真加载技巧，不能当成「硬件算过了」。
