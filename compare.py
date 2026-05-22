#!/usr/bin/env python3
"""
Open-NPU Accuracy Comparison Tool

Compares ONNX Runtime float32 inference output with NPU simulator INT8 output.

Usage:
  python3 compare.py --model MODEL.onnx --input debug.bin --npu-output output.bin \
      --meta model_meta.npz

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import numpy as np
import onnxruntime as ort


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def compare_outputs(model_path, input_path, npu_output_path, meta_path):
    """Run comparison between ONNX Runtime and NPU sim outputs."""

    # Load metadata
    meta = np.load(meta_path)
    output_scale = float(meta['output_scale'])
    output_zp = int(meta['output_zp'])
    input_shape = list(meta['input_shape'].astype(int))
    bits = int(meta['bits']) if 'bits' in meta else 8

    print("=== NPU Accuracy Comparison ===")
    print(f"  Model: {model_path}")
    print(f"  Input: {input_path}")
    print(f"  Quantization: INT{bits} per-channel")
    print(f"  Output scale: {output_scale:.8f}, zp: {output_zp}")
    print()

    # 1. Run ONNX Runtime float inference
    print("── ONNX Runtime (float32) ──")
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Load input and preprocess: (uint8 - 127.5) / 255
    raw = np.fromfile(input_path, dtype=np.uint8).reshape(input_shape)
    float_input = (raw.astype(np.float32) - 127.5) / 255.0
    ort_output = sess.run([out_name], {inp_name: float_input})[0]
    print(f"  Output shape: {ort_output.shape}")
    print(f"  Output range: [{ort_output.min():.4f}, {ort_output.max():.4f}]")
    print(f"  Output L2 norm: {np.linalg.norm(ort_output.flatten()):.4f}")
    print()

    # 2. Load NPU sim output and dequantize
    print("── NPU Sim (dequant) ──")
    if bits == 16:
        npu_raw = np.fromfile(npu_output_path, dtype=np.int16)
    else:
        npu_raw = np.fromfile(npu_output_path, dtype=np.int8)
    print(f"  Raw elements: {npu_raw.size}")
    print(f"  Raw range: [{npu_raw.min()}, {npu_raw.max()}]")

    # Dequantize: float = (int8_val - zp) * scale
    npu_float = (npu_raw.astype(np.float64) - output_zp) * output_scale
    # Reshape to match ORT output
    npu_float = npu_float.reshape(ort_output.shape)
    print(f"  Dequant range: [{npu_float.min():.4f}, {npu_float.max():.4f}]")
    print(f"  Dequant L2 norm: {np.linalg.norm(npu_float.flatten()):.4f}")
    print()

    # 3. Compute metrics
    print("── Accuracy Metrics ──")
    ort_flat = ort_output.flatten().astype(np.float64)
    npu_flat = npu_float.flatten()

    cos_sim = cosine_similarity(ort_flat, npu_flat)
    mse = np.mean((ort_flat - npu_flat) ** 2)
    mae = np.mean(np.abs(ort_flat - npu_flat))
    max_err = np.max(np.abs(ort_flat - npu_flat))
    snr = 10 * np.log10(np.mean(ort_flat**2) / max(mse, 1e-20))

    print(f"  Cosine Similarity:  {cos_sim:.6f}")
    print(f"  MSE:                {mse:.8f}")
    print(f"  MAE:                {mae:.6f}")
    print(f"  Max Abs Error:      {max_err:.6f}")
    print(f"  SNR (dB):           {snr:.2f}")
    print()

    # 4. Per-element distribution
    abs_errors = np.abs(ort_flat - npu_flat)
    print("── Error Distribution ──")
    percentiles = [50, 90, 95, 99, 100]
    for p in percentiles:
        print(f"  P{p:3d}: {np.percentile(abs_errors, p):.6f}")
    print()

    # 5. Feature vector comparison (if output is embedding)
    if ort_output.size <= 2048:
        # Likely a feature embedding, compare element-wise
        print("── Feature Vector Comparison (first 20 dims) ──")
        print(f"  {'Dim':>4s} {'ORT':>10s} {'NPU':>10s} {'Diff':>10s}")
        for d in range(min(20, ort_output.size)):
            print(f"  {d:4d} {ort_flat[d]:10.4f} {npu_flat[d]:10.4f} "
                  f"{ort_flat[d]-npu_flat[d]:10.4f}")

    # Summary
    print()
    print("═══════════════════════════════")
    if cos_sim > 0.99:
        print(f"  RESULT: EXCELLENT (cosine={cos_sim:.4f})")
    elif cos_sim > 0.95:
        print(f"  RESULT: GOOD (cosine={cos_sim:.4f})")
    elif cos_sim > 0.90:
        print(f"  RESULT: ACCEPTABLE (cosine={cos_sim:.4f})")
    else:
        print(f"  RESULT: POOR (cosine={cos_sim:.4f}) — check quantization")
    print("═══════════════════════════════")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NPU Accuracy Comparison')
    parser.add_argument('--model', required=True, help='ONNX float32 model')
    parser.add_argument('--input', required=True, help='Original input file (debug.bin)')
    parser.add_argument('--npu-output', required=True, help='NPU sim output file')
    parser.add_argument('--meta', required=True, help='Quantization metadata (.npz)')
    args = parser.parse_args()

    compare_outputs(args.model, args.input, args.npu_output, args.meta)
