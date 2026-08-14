# 权重复用 1b + 树形归约 详细设计

日期：2026-08-13（设计）/ 2026-08-14（完成）
状态：✅ 已完成，全矩阵 10/10 bit-exact 验证通过
前置：性能基线（hw_perf_cnt/hw_mac_cnt 全矩阵采集）

## 1. 问题定量

卷积发射利用率 2-2.7%。成本模型（72 个卷积层实测一致）：

- int8：~37-41 周期/元素广播；int16：~50-55
- 每 (像素, k_pass) 开销 ~592 周期，其中权重 16 列重流 ~192、激活+flush+drain+PPU/WB ~400
- 根因：`S_PIXEL_NEXT → S_WGT_CMD`（npu_compute.v:1596-1608），像素在外 pass 在内，权重块每像素重流，而同 (group, pass) 权重对全 tile 像素相同

阵列本身是 weight-stationary（PE.weight_reg 驻留，MODE_COMPUTE 不写），调度器未利用。

## 2. 核心机制

**16 像素分块**：把同一 (oc_group, k_pass) 的权重装载复用到 16 个连续像素上；部分和跨 pass 存 acc_buf。

新循环嵌套（仅 Conv/FC 路径，op 0/2）：

```
for oc_group:                       # 不变
  for blk in 0..nblocks-1:          # 每 tile 像素按 16 分块
    for k_pass in 0..k_pass_max:    # pass 提到像素内层循环之外
      S_WGT_CMD..EMIT               # ← 每 (group, blk, pass) 一次，被 16 px 复用
      for px in block (≤16):
        S_SPATIAL_SETUP → S_ACT_CMD..EMIT ×k_pass_remain → S_ACT_FLUSH
        drain 16 cols → acc_buf[px][col]  (pass==0 覆写, 否则 +=)
    # 本块所有 pass 完成：
    for px in block:
      S_PARAM_LOAD → S_PPU_FEED ×col_last+1 → S_WRITEBACK   # 现有 PPU/WB 路径
```

权重装载次数：nblocks × passes（mb8 L2：49×36 = 1764，原 28224，÷16）。

## 3. acc_buf 规格

```verilog
reg signed [ACC_W-1:0] acc_buf [0:255];   // 16 px × 16 ch × 44b ≈ 11K FF
// 索引：{px_in_blk[3:0], col[3:0]}
```

- pass==0 覆写、pass>0 加法累加
- **位精确性论证**：PE.acc_reg 是纯 wraparound 加法（无饱和，npu_pe.v:68），mod 2^44 加法满足结合律/交换律 → 按 pass 分组求和与连续求和逐位相同；PPU 之前的值不变 → **golden 无需重生成**
- PE drain 自带 acc 清零（npu_pe.v:76-81），每 (px,pass) 自动从新起点累加，无需额外清零逻辑
- 综合注记：FF 阵列约 11K FF；若后端在意可换 256×44 单口 SRAM macro，接口已隔离（读/写各一处）

## 4. FSM 改动清单（仅 npu_compute.v，预估 +200 行）

| 位置 | 改动 |
|---|---|
| 新增寄存器 | `blk_idx[15:0]`, `px_in_blk[3:0]`, `blk_px_cnt[4:0]`, `ppu_px[3:0]`（块末 PPU 循环用） |
| S_TILE_SETUP | 算 `nblocks = ceil(out_tile_h*out_tile_w / 16)`；blk_idx=0 |
| S_OC_SETUP | 不变（wgt_base/param_base 逻辑沿用） |
| S_PIXEL_NEXT | 改为三层递进：px_in_blk+1 < blk_px_cnt → 下一像素（**不再跳 S_WGT_CMD**，直接 S_SPATIAL_SETUP）；块内像素完 → k_pass+1 ≤ k_pass_max → S_WGT_CMD（重置 px 回块首）；pass 完 → 进块末 PPU 循环 |
| S_DRAIN 完成路径 | 新增分叉：drain 值不再直接进 PPU，写 acc_buf（pass==0 覆写 / 否则 +=），col 递增；16 列完 → 下一像素（S_SPATIAL_SETUP） |
| 块末 PPU 循环 | 复用 S_PARAM_LOAD/S_PPU_FEED/S_PPU_WAIT/S_WRITEBACK；ppu_acc_in 源加 mux：块模式取 acc_buf[{ppu_px, ch}][31:0] 扩符（现有 drain 直通源保留给非块模式…见 §7 降级）；每像素 PPU+WB 完 → ppu_px+1；块完 → blk_idx+1 → pass=0 → S_WGT_CMD |
| S_OC_NEXT | blk 循环结束条件挂到这里：blk_idx+1 ≥ nblocks → 原 group 收尾逻辑 |
| S_ACT_FLUSH | 不变（每 px-pass 一次，代价已计入预算） |

新增状态只需 1 个：`S_ACC_WR`（drain 值写 acc_buf，可与 drain 完成拍合并则 0 个）。

**不改**：npu_systolic / npu_pe / npu_ctrl / npu_ppu / CSR / 固件 / golden。

## 5. 周期预算（int8，每 px-pass）

| 项 | 现状 | 1b | 1b+激活预取 |
|---|---|---|---|
| 权重装载 | 192 | 192/16px = 12 | 12 |
| 激活读+feed+flush | ~280 | ~280 | ~140（预取下一词） |
| drain 16 列 | ~32 | 32+16(acc写) | 48 |
| PPU/WB 摊销 | ~88 | ~88 | ~88 |
| **合计/px-pass** | **~592** | **~180-200** | **~120** |
| **每元素** | **37** | **~12** | **~7.5** |
| **卷积加速比** | 1× | **~3.2×** | **~5×** |

## 5b. 实测结果（2026-08-14，全矩阵 SoC E2E bit-exact 全过）

| 模型 | 旧 NPU 周期 | 1b | 加速比 |
|---|---|---|---|
| ma8 | 1,782.9M | 1,299.5M | 1.37× |
| ma16 | 2,330.0M | 1,423.2M | 1.64× |
| mb8 | 155.2M | 106.6M | 1.46× |
| mb16 | 205.0M | 114.6M | 1.79× |
| mc8 | 2,219.0M | 1,517.2M | 1.46× |
| mc16 | 2,990.7M | 1,672.7M | 1.79× |
| md8 | 277.2M | 201.5M | 1.38× |
| md16 | 369.2M | 226.5M | 1.63× |
| me8 | 4,341.6M | 3,009.2M | 1.44× |
| me16 | 7,805.2M | 5,765.7M | 1.35× |

规律：int8 ~1.4×、int16 ~1.6-1.8×（int16 权重块字节翻倍，重流占比更高故收益更大）。普通卷积层单体 1.41-1.48×（与成本模型 1.43× 吻合）。me16 被 L23/26/27 拖低（1×1 tile 退化，见下）。

**实测低于 §5 预估的原因**：预估只算了权重装载一项的摊销；1×1 tile 层（me16 L23/26/27，输入 SRAM 限制）块=1 像素完全无复用。

## 5c. 第二阶段：树形归约（2026-08-14，rtl 见提交日志）

精读激活路径后发现 1b 之后的真瓶颈不是激活读（仅 ~11%），而是 **drain 后每列 16 拍串行行归约（S_REDUCE，占 ~74%）**。改为单周期加法树（i < k_pass_remain 掩零无效行；wraparound 加法结合律 → bit-exact）。激活预取被数据证伪，未实施。

最终全矩阵（1b + 树形归约，全部 bit-exact PASS）：

| 模型 | 旧 | 最终 | 加速比 |
|---|---|---|---|
| ma8 | 1782.9M | 714.6M | 2.49× |
| ma16 | 2330.0M | 838.3M | 2.78× |
| mb8 | 155.2M | 49.2M | 3.15× |
| mb16 | 205.0M | 57.2M | 3.58× |
| mc8 | 2219.0M | 673.1M | 3.30× |
| mc16 | 2990.7M | 828.7M | 3.61× |
| md8 | 277.2M | 106.3M | 2.61× |
| md16 | 369.2M | 131.3M | 2.81× |
| me8 | 4341.6M | 1309.1M | 3.32× |
| me16 | 7805.2M | 3740.5M | 2.09× |

连 1px-tile 的 grp8 层也获益（L23 980M→764M=1.28×，drain/归约与块大小无关）。

## 6. 特殊路径兼容

| 场景 | 处理 |
|---|---|
| int16 | act 解包/wgt 装载路径不变；acc_buf 位宽不变 |
| grp8（8 通道子组） | col_last=7，drain/PPU 只走 8 列，acc_buf 只用一半 ✓ |
| per-oc 权重重载 | ctrl 每组装 SRAM[0] 不变；阵列按 (blk,pass) 重读同一块 ✓ 无新增 DDR 流量 |
| DB_EN 分 tile | 块循环在 tile 内；nblocks 按 out_tile_h×out_tile_w 算 |
| 块尾不齐 (px%16) | blk_px_cnt = min(16, 剩余)；循环上界用它 |
| 末 pass 不齐 (kd%16) | 现有 k_pass_remain 逻辑不变 |
| padding 像素 | 激活路径发 0/in_zp，不变 |
| FC (op 2) | 单像素无复用维度，**走原路径**（块=1 像素自然退化，无需特判） |
| DW/Pool/Add/Concat/Resize/stream | 完全不经此路径，零影响 |

## 7. 降级与回退

- 加一个 `wgt_reuse_en` 内部 wire（默认 1；op 非 0/2 或 FC 退化时走原循环）。不出 CSR，纯内部。
- 回退 = 把 S_PIXEL_NEXT 的分叉条件常量化回原跳转（一处 if）。

## 8. 验证计划

1. cocotb 单测：现有卷积用例（test_npu_dma_e2e 内 Conv 系列）+ test_md_l12_l13（per-oc+Add 邻接）+ test_me_l23_chain2d（grp8 邻接）——全部要求 bit-exact
2. SoC E2E：md16（最快全模型）→ me16（grp8+大 kd）→ ma8（dw_stream 不受影响确认）→ 全矩阵 10/10
3. 性能断言：md8 总 NPU 周期应 ≤ 现状 40%（277M → ≤110M），在报告里对账

## 9. 面积评估

acc_buf 11K FF + 块控制逻辑 ~1-2K 门 ≈ 相当于 act SRAM 的 ~3%。无新 SRAM 宏、无新 CSR、无固件改动。

## 10. 风险

| 风险 | 缓解 |
|---|---|
| acc_buf 累加时序（drain 同拍读 PE 值+读旧 acc_buf+加+写） | 写成两拍（读/加、写），多 1 周期/列可接受；或 ACC_W 拆高低 32b 存储 |
| 块末 PPU 循环与现有 per-pixel PPU 的 WB 地址逻辑（RMW 跨像素字合并） | WB 地址公式不变（按全局像素坐标），RMW 读改写本来就是按字地址独立 |
| 长 kd 层 acc_buf 溢出 | 每 pass 部分和 ≤ 2^14×16 = 2^18，44b 绝无溢出 ✓ |
