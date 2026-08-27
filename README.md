# Open-NPU Tools

[中文版](README_CN.md)

Python toolchain for converting, quantizing, and packing neural network models for the Open-NPU accelerator.

Product scripts live at the `tools/` root. Tests, golden generators, and
Phase-1 regression live in [`internal/`](internal/README.md). Drop that
directory as a unit at release (`rm -rf internal/`).

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
| `ptq_mse_admm.py` | MSE clip / ADMM helpers used by the converter |

## Usage

```bash
# Convert ONNX model to NPU binary
python onnx_converter.py --model MODEL.onnx --calib CALIB_DIR --output model.npu1.bin

# Internal (not shipped at release)
python3 internal/phase1_regression.py
python3 internal/tests/mobilenetv2.py
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
