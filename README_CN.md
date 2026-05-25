# Open-NPU 工具链

[English](README.md)

用于 Open-NPU 加速器的神经网络模型转换、量化和验证的 Python 工具链。

## 组件

| 脚本 | 说明 |
|------|------|
| `onnx_converter.py` | ONNX → NPU 二进制转换器（含 PTQ） |
| `quantize_perchannel.py` | Per-channel INT8 量化 |
| `quantize_int16.py` | INT16 量化（用于敏感层） |
| `quantize_mixed.py` | 混合 INT8/INT16 精度 |
| `quantize_fixedpoint.py` | 定点量化工具 |
| `layer_fusion.py` | 层融合（Conv+BN+ReLU、DW super-layer） |
| `tiling.py` | 空间 tiling 策略计算 |
| `model_packer.py` | NPU 执行用二进制打包 |
| `perf_model.py` | 性能建模与周期估算 |
| `hw_config.py` | 硬件配置参数 |
| `compare.py` | 输出比对工具 |
| `batch_accuracy.py` | 批量精度评估 |

## 使用方法

```bash
# 将 ONNX 模型转换为 NPU 二进制
python onnx_converter.py model.onnx --output model_npu/

# 运行端到端验证
python test_mobilenetv2.py
python test_resnet18_e2e.py
python test_yolo_tiny_e2e.py
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
