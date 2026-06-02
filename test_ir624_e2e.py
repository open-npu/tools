#!/usr/bin/env python3
"""
IR624 Real Model End-to-End Validation

Validates the full NPU toolchain on a real-world IR palm liveness model:
  - Input: 112x112x1 grayscale IR image
  - Output: 100-class logits
  - Flow: ORT float32 reference → INT8 convert+csim → cos similarity
          if cos < 0.99, retry with INT16

Model: WXPay_PalmIrLiveness_624_06
ONNX: /data/sam/ir624/WXPay_PalmIrLiveness_624_06_r20260509.onnx
Calibration: /data/sam/ir624/O2_for_guopeng20221209/O2_quant_datas/ (1002 JPGs)
Test input: /data/sam/ir624/debug.bin (uint8, 112x112x1)

Success criteria: cosine similarity >= 0.99

SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import subprocess
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_converter import convert_model

MODEL_PATH = '/data/sam/ir624/WXPay_PalmIrLiveness_624_06_r20260509.onnx'
CALIB_DIR = '/data/sam/ir624/O2_for_guopeng20221209/O2_quant_datas'
INPUT_BIN = '/data/sam/ir624/debug.bin'
CSIM_PATH = '/data/sam/open-npu/csim/npu_sim'
OUTPUT_DIR = '/tmp/ir624_e2e'
COS_THRESHOLD = 0.99
NUM_CALIB = 100  # use 100 images for calibration


def cosine_sim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)


def get_ort_reference():
    """Run ORT float32 inference on debug.bin, return output vector."""
    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape  # e.g. [1, 1, 112, 112]

    # Load debug.bin as uint8, convert to float32 with (pixel-127.5)/255
    raw = np.fromfile(INPUT_BIN, dtype=np.uint8).reshape(input_shape)
    inp = (raw.astype(np.float32) - 127.5) / 255.0

    out = sess.run(None, {input_name: inp})[0]
    print(f"  ORT output shape: {out.shape}")
    print(f"  ORT output range: [{out.min():.4f}, {out.max():.4f}]")
    return out.flatten()


def run_npu_pipeline(bits):
    """Convert model and run csim, return dequantized output vector."""
    tag = f"int{bits}"
    model_bin = os.path.join(OUTPUT_DIR, f'model_{tag}.npu1.bin')
    npu_input = os.path.join(OUTPUT_DIR, f'model_{tag}.npu1_input.bin')
    npu_output = os.path.join(OUTPUT_DIR, f'output_{tag}.bin')
    meta_path = os.path.join(OUTPUT_DIR, f'model_{tag}.npu1_meta.npz')

    # Step 1: Convert
    print(f"\n--- Converting model (INT{bits}) ---")
    convert_model(MODEL_PATH, CALIB_DIR, INPUT_BIN, model_bin,
                  input_format='int8-nchw', num_calib=NUM_CALIB, bits=bits)

    # Step 2: Run csim
    print(f"\n--- Running csim (INT{bits}) ---")
    if not os.path.exists(npu_input):
        raise FileNotFoundError(f"NPU input not found: {npu_input}")

    result = subprocess.run(
        [CSIM_PATH, model_bin, npu_input, npu_output],
        capture_output=True, text=True)

    if result.returncode != 0:
        print(f"csim stdout: {result.stdout[-500:]}")
        print(f"csim stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"csim failed with exit code {result.returncode}")

    print(f"  csim completed successfully")

    # Step 3: Load and dequantize output
    meta = np.load(meta_path)
    out_scale = float(meta['output_scale'])
    out_zp = int(meta['output_zp']) if 'output_zp' in meta else 0
    output_elements = int(meta['output_elements'])

    if bits == 16:
        csim_q = np.fromfile(npu_output, dtype=np.int16)[:output_elements]
    else:
        csim_q = np.fromfile(npu_output, dtype=np.int8)[:output_elements]

    csim_float = (csim_q.astype(np.float32) - out_zp) * out_scale

    print(f"  csim output elements: {len(csim_q)}")
    print(f"  csim quantized range: [{csim_q.min()}, {csim_q.max()}]")
    print(f"  csim dequant range: [{csim_float.min():.4f}, {csim_float.max():.4f}]")
    print(f"  output scale={out_scale:.8f}, zp={out_zp}")

    return csim_float


def main():
    # Check prerequisites
    for path, name in [(MODEL_PATH, 'ONNX model'), (INPUT_BIN, 'debug.bin'),
                       (CALIB_DIR, 'calibration dir'), (CSIM_PATH, 'csim binary')]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: ORT float32 reference
    print("=" * 60)
    print("Step 1: ORT Float32 Reference")
    print("=" * 60)
    ort_ref = get_ort_reference()

    # Step 2: INT8 pipeline
    print("\n" + "=" * 60)
    print("Step 2: INT8 NPU Pipeline")
    print("=" * 60)
    try:
        int8_out = run_npu_pipeline(bits=8)
        int8_cos = cosine_sim(ort_ref, int8_out)
        print(f"\n  INT8 cosine similarity: {int8_cos:.6f}")
    except Exception as e:
        print(f"\n  INT8 pipeline FAILED: {e}")
        int8_cos = 0.0
        int8_out = None

    # Step 3: If INT8 < threshold, try INT16
    int16_cos = None
    if int8_cos < COS_THRESHOLD:
        print(f"\n  INT8 cos ({int8_cos:.4f}) < {COS_THRESHOLD}, trying INT16...")
        print("\n" + "=" * 60)
        print("Step 3: INT16 NPU Pipeline")
        print("=" * 60)
        try:
            int16_out = run_npu_pipeline(bits=16)
            int16_cos = cosine_sim(ort_ref, int16_out)
            print(f"\n  INT16 cosine similarity: {int16_cos:.6f}")
        except Exception as e:
            print(f"\n  INT16 pipeline FAILED: {e}")
            int16_cos = 0.0
    else:
        print(f"\n  INT8 cos ({int8_cos:.4f}) >= {COS_THRESHOLD}, INT16 not needed")

    # Summary
    print("\n" + "=" * 60)
    print("IR624 E2E VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Model: {os.path.basename(MODEL_PATH)}")
    print(f"  Input: 112x112x1 grayscale (debug.bin)")
    print(f"  Output: 100-class logits")
    print(f"  Calibration images: {NUM_CALIB}")
    print(f"  Threshold: cos >= {COS_THRESHOLD}")
    print()
    print(f"  INT8  cosine: {int8_cos:.6f}  {'PASS' if int8_cos >= COS_THRESHOLD else 'FAIL'}")
    if int16_cos is not None:
        print(f"  INT16 cosine: {int16_cos:.6f}  {'PASS' if int16_cos >= COS_THRESHOLD else 'FAIL'}")

    best_cos = max(int8_cos, int16_cos or 0.0)
    best_bits = 8 if int8_cos >= (int16_cos or 0.0) else 16

    if best_cos >= COS_THRESHOLD:
        print(f"\n  PASS — Best result: INT{best_bits} cos={best_cos:.6f}")
        return True
    else:
        print(f"\n  FAIL — Best result: INT{best_bits} cos={best_cos:.6f} < {COS_THRESHOLD}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
