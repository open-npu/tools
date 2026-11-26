# Open-NPU Tools

[中文版](README_CN.md)

Python toolchain for converting, quantizing, and packing neural network models for the Open-NPU accelerator.

## Components

| Script | Description |
|--------|-------------|
| `onnx_converter.py` | ONNX → NPU binary converter with PTQ |
| `model_packer.py` | Binary packing for NPU execution |
| `tiling.py` | Spatial tiling strategy computation |
| `layer_fusion.py` | Layer fusion (Conv+BN+ReLU, Conv->Pool strip, SRAM chain) |
| `super_tile.py` | Spatial super-block fusion planner |
| `hw_config.py` | Hardware configuration parameters |
| `ptq_mse_admm.py` | MSE clip / ADMM helpers used by the converter |

## Usage

```bash
# Convert ONNX model to NPU binary
python3 onnx_converter.py --model MODEL.onnx --calib CALIB_DIR --input sample.bin --output model.npu1.bin

# Run built-in packer + CSIM reference self-tests
python3 model_packer.py test
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
