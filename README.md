# Open-NPU Tools

[中文版](README_CN.md)

Python toolchain for converting, quantizing, and validating neural network models for the Open-NPU accelerator.

## Components

| Script | Description |
|--------|-------------|
| `onnx_converter.py` | ONNX → NPU binary converter with PTQ |
| `quantize_perchannel.py` | Per-channel INT8 quantization |
| `quantize_int16.py` | INT16 quantization for sensitive layers |
| `quantize_mixed.py` | Mixed INT8/INT16 precision |
| `quantize_fixedpoint.py` | Fixed-point quantization utilities |
| `layer_fusion.py` | Layer fusion (Conv+BN+ReLU, DW super-layer) |
| `tiling.py` | Spatial tiling strategy computation |
| `model_packer.py` | Binary packing for NPU execution |
| `perf_model.py` | Performance modeling & cycle estimation |
| `hw_config.py` | Hardware configuration parameters |
| `compare.py` | Output comparison utilities |
| `batch_accuracy.py` | Batch accuracy evaluation |

## Usage

```bash
# Convert ONNX model to NPU binary
python onnx_converter.py model.onnx --output model_npu/

# Run end-to-end validation
python test_mobilenetv2.py
python test_resnet18_e2e.py
python test_yolo_tiny_e2e.py
```

## Requirements

- Python 3.8+
- numpy, onnx, onnxruntime

## Related Repositories

- [open-npu/rtl](https://github.com/open-npu/rtl) — Synthesizable Verilog implementation
- [open-npu/csim](https://github.com/open-npu/csim) — C cycle-approximate simulator
- [open-npu/design](https://github.com/open-npu/design) — Architecture specifications

## License

Apache-2.0
