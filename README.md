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
| `model_packer.py` | Binary packing for NPU execution |
| `tiling.py` | Spatial tiling strategy computation |
| `layer_fusion.py` | Layer fusion (Conv+BN+ReLU, DW super-layer) |
| `hw_config.py` | Hardware configuration parameters |
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
