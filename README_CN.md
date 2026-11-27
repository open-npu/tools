# Open-NPU 工具链

[English](README.md)

用于 Open-NPU 加速器的神经网络模型转换、量化和打包的 Python 工具链。

整机地图：[open-npu/design — open_npu_overview.md](https://github.com/open-npu/design/blob/main/open_npu_overview.md)。

## 组件

| 脚本 | 说明 |
|------|------|
| `onnx_converter.py` | ONNX → NPU 二进制转换器（含 PTQ） |
| `model_packer.py` | NPU 执行用二进制打包 |
| `tiling.py` | 空间 tiling 策略计算 |
| `layer_fusion.py` | 层融合（Conv+BN+ReLU、Conv->Pool strip、SRAM chain） |
| `super_tile.py` | 空间 Super-Block 融合规划 |
| `hw_config.py` | 参数化硬件配置（MAC/DW 通道数、四库 SRAM 容量、`32/64/128-bit` WB/AXI4 DMA 总线） |
| `npu_profiler.py` | Nsight 风格 Roofline、四库 SRAM 占用率、硬件 PMU 停顿分解与优化建议分析器 |
| `ptq_mse_admm.py` | 转换器使用的 MSE clip / ADMM |

## 使用方法

```bash
# 将 ONNX 模型转换为 NPU 二进制
python3 onnx_converter.py --model MODEL.onnx --calib CALIB_DIR --input sample.bin --output model.npu1.bin

# 分析模型算力利用率 (SOL%)、Roofline、片上四库 SRAM 占用与优化空间
python3 npu_profiler.py --model model.npu1.bin --dma-bus-width 128

# 结合 RTL/SoC 硬件 PMU 仿真日志做周期级停顿归因分析
python3 npu_profiler.py --meta path/to/metadata.json --soc-log sim.log --dma-bus-width 128

# 运行内置打包器与 CSIM 参考自检
python3 model_packer.py test
```

## 依赖

- Python 3.8+
- numpy, onnx, onnxruntime

## 相关仓库

- [open-npu/rtl](https://github.com/open-npu/rtl) — 可综合 Verilog 实现
- [open-npu/csim](https://github.com/open-npu/csim) — C 周期近似模拟器
- [open-npu/design](https://github.com/open-npu/design) — 架构设计文档

## 许可证

Apache-2.0
