# tools/internal — not product

This directory is the converter team's tests, golden generators, accuracy
helpers, and Phase-1 regression. It is **not** part of the published toolchain.

| Path | What |
|------|------|
| `tests/` | Converter / CSIM E2E scripts |
| `gen/` | Per-model CSIM → RTL golden generators |
| `quantize/` | Standalone PTQ experiments (not used by the converter) |
| `bin2golden.py` | NPU1 → RTL `.npy` golden packer |
| `compare.py` / `batch_accuracy.py` | Cosine vs ONNX Runtime |
| `perf_model.py` | Cycle estimator |
| `extract_npu1_layers.py` | Debug dump of NPU1 binaries |
| `phase1_regression.py` | Official Phase-1 gate |
| `phase1_model_matrix.py` | 10-cell cosine + SoC RTL |
| `PHASE1_ACCEPTANCE.md` | Frozen gates |

Official command:

```sh
python3 tools/internal/phase1_regression.py
```

At release, drop this tree as a unit (`rm -rf internal/`). Do not scatter
new tests, generators, or VCD dumps at the `tools/` root.
