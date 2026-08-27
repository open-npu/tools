# Phase 1 Acceptance Baseline

Phase 1 is accepted only when this command exits with status 0:

```sh
python3 design/phase1_regression.py
```

`--quick` and `--section` are diagnostic subsets. Their success does not qualify
a revision for a Phase-1 tag.

## Frozen hardware profile

- Array: 16×16
- Scratchpad: 192 KiB
- Accumulator: 44 bits
- Activation/weight modes: signed INT8 and signed INT16
- NPU1 fixed descriptor size: 62 bytes

## Mandatory gates

1. Converter guard tests reject unsupported operators, unsupported grouped
   convolution, non-ReLU/ReLU6 Clip bounds, invalid padded mean/std folding,
   and infeasible SRAM tiles.
2. Model packer tests execute through the C simulator; Python-only reference
   checks are not sufficient.
3. All active design E2E tests execute without `SKIP`, including MODEL_A INT8
   and INT16, MODEL_B, synthetic ResNet-18, MobileNetV2, Resize, Concat,
   pooling, FC, and pipeline tests.
4. CSIM build, operator tests, post-processing tests, Add E2E, and tiled Resize
   tests all pass.
5. RTL lint and synthesis exit successfully.
6. All active RTL unit, integration, robustness, and E2E cocotb suites pass
   with zero failures and zero skipped tests.
7. Soft reset makes the controller idle and cancels DMA/compute activity.
   Abort cancels in-flight datapath activity and prevents auto-restart.
8. INT16 MODEL_A average cosine is at least 0.999. An 8×8 global AvgPool is
   bit-exact between the reference, CSIM, and RTL paths.
9. The 10-cell matrix (A–E × INT8/INT16) runs on the current converter + CSIM
   + RTL. Each cell must meet both gates:
   - CSIM dequant vs ONNX float cosine ≥ 0.95 (INT8) or ≥ 0.99 (INT16)
   - SoC Verilator vs the CSIM golden generated in that same run, bit-exact
   Recorded cosine values, cached npu1 binaries, missing cells, and SKIP are
   not acceptance. `python3 design/phase1_model_matrix.py` without `--rtl` is
   diagnostic cosine only and does not qualify.
10. SoC runtime DDR outputs are initialized to poison, not golden. Golden lives
    only in the blob compare region. A layer with `PERF_Lx npu=0` is a failure.
    The simulator exits non-zero unless UART contains `RESULT: PASS`.

## Explicitly unsupported

- Arbitrary ONNX operators are never silently removed.
- Grouped Conv is unsupported unless it is true depthwise Conv.
- Clip is supported only as ReLU or Clip(0, 6).
- Folding a non-zero mean/std offset into a padded first Conv is rejected
  because one channel bias cannot represent position-dependent border terms.
- Models for which no SRAM-safe and word-aligned tile exists are rejected.

## Versioning rule

After a full pass, commit the reviewed changes in `design`, `csim`, and `rtl`,
then create the same annotated tag on those three commits. Never move or reuse
the historical `v1.0-phase1` tag. The acceptance tag is:

```text
v1.0-phase1-accepted
```

Record the three tagged commit IDs and the full regression log together.
