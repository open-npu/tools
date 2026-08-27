# Open-NPU 工具链

[English](README.md)

用于 Open-NPU 加速器的神经网络模型转换、量化和打包的 Python 工具链。

产品脚本在 `tools/` 根目录。测试、golden 生成、Phase-1 回归在
[`internal/`](internal/README.md)。发布时整棵删掉即可（`rm -rf internal/`）。

整机地图：[open-npu/design — open_npu_overview.md](https://github.com/open-npu/design/blob/main/open_npu_overview.md)。融合 / 双缓冲信号：[internal/model_d_fused_tiling_explained.md](internal/model_d_fused_tiling_explained.md)。

## 组件

| 脚本 | 说明 |
|------|------|
| `onnx_converter.py` | ONNX → NPU 二进制转换器（含 PTQ） |
| `model_packer.py` | NPU 执行用二进制打包 |
| `tiling.py` | 空间 tiling 策略计算 |
| `layer_fusion.py` | 层融合（Conv+BN+ReLU、DW super-layer） |
| `hw_config.py` | 硬件配置参数 |
| `ptq_mse_admm.py` | 转换器使用的 MSE clip / ADMM |

## 使用方法

```bash
# 将 ONNX 模型转换为 NPU 二进制
python onnx_converter.py --model MODEL.onnx --calib CALIB_DIR --output model.npu1.bin

# 内部（发布时去掉）
python3 internal/phase1_regression.py
python3 internal/tests/mobilenetv2.py
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
