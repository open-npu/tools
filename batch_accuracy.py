#!/usr/bin/env python3
"""
Open-NPU Batch Accuracy Test: INT8 vs INT16 multi-image comparison.

Converts the model once per bit-width, then runs N test images through
both ONNX Runtime (float32 reference) and NPU C-sim (quantized).
Reports per-image and average cosine similarity.

Usage:
  python3 batch_accuracy.py --model MODEL_A.onnx --calib CALIB_DIR \
      --images TEST_DIR --num-images 20

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import os
import sys
import glob
import subprocess
import tempfile
import numpy as np
import onnxruntime as ort
from PIL import Image

CSIM_BIN = os.path.join(os.path.dirname(__file__), '..', 'csim', 'npu_sim')
CONVERTER = os.path.join(os.path.dirname(__file__), 'onnx_converter.py')


def preprocess_image(img_path, input_shape):
    """Load image, resize to model input, return float32 NCHW.

    Preprocessing matches calibration: float = (pixel_uint8 - 127.5) / 255
    This gives range [-0.5, +0.5] for pixel values [0, 255].
    """
    _, c, h, w = input_shape
    img = Image.open(img_path).convert('RGB')
    img = img.resize((w, h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)   # [H,W,3], 0-255 as float
    arr = arr.transpose(2, 0, 1)            # [3,H,W]

    # Same preprocessing as calibration
    float_arr = (arr - 127.5) / 255.0       # range [-0.5, +0.5]
    float_arr = float_arr[np.newaxis, ...]  # [1,C,H,W]

    return float_arr


def cosine_similarity(a, b):
    """Compute cosine similarity between two flat vectors."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def convert_model(model_path, calib_dir, bits, work_dir, num_calib=50):
    """Convert ONNX model to NPU1 format. Returns (npu1_path, meta_path)."""
    prefix = f"model_int{bits}"
    output = os.path.join(work_dir, f"{prefix}.npu1.bin")
    # Use a dummy input for conversion (the first calib image)
    dummy_input = os.path.join(work_dir, "dummy_input.bin")

    # Create a dummy int8 input
    dummy_data = np.zeros((3, 224, 224), dtype=np.int8)
    dummy_data.tofile(dummy_input)

    cmd = [
        sys.executable, CONVERTER,
        '--model', model_path,
        '--calib', calib_dir,
        '--input', dummy_input,
        '--output', output,
        '--num-calib', str(num_calib),
        '--bits', str(bits),
    ]
    print(f"  Converting model (INT{bits})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Conversion failed:\n{result.stderr}")
        sys.exit(1)

    meta_path = output.replace('.npu1.bin', '.npu1_meta.npz')
    if not os.path.exists(meta_path):
        # Try alternate naming
        meta_path = output + '_meta.npz'
    return output, meta_path


def run_csim(npu1_path, input_bin_path, output_bin_path):
    """Run NPU C-sim on a single input."""
    cmd = [CSIM_BIN, npu1_path, input_bin_path, output_bin_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  CSIM ERROR: {result.stderr}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Batch INT8 vs INT16 accuracy')
    parser.add_argument('--model', required=True, help='ONNX float32 model')
    parser.add_argument('--calib', required=True, help='Calibration image directory')
    parser.add_argument('--images', required=True, help='Test image directory')
    parser.add_argument('--num-images', type=int, default=20, help='Number of test images')
    parser.add_argument('--num-calib', type=int, default=50, help='Calibration images')
    args = parser.parse_args()

    # Discover test images
    image_files = sorted(
        glob.glob(os.path.join(args.images, '*.jpg')) +
        glob.glob(os.path.join(args.images, '*.png'))
    )
    if not image_files:
        print(f"ERROR: No images found in {args.images}")
        sys.exit(1)
    image_files = image_files[:args.num_images]
    num = len(image_files)
    print(f"=== Batch Accuracy Test: {num} images ===\n")

    # Load ONNX model for float reference
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    input_shape = list(sess.get_inputs()[0].shape)
    print(f"Model: {args.model}")
    print(f"Input shape: {input_shape}")
    print()

    # Convert model for both bit-widths
    work_dir = tempfile.mkdtemp(prefix='npu_batch_')
    print(f"Working directory: {work_dir}\n")

    print("── Step 1: Model Conversion ──")
    npu8_path, meta8_path = convert_model(
        args.model, args.calib, 8, work_dir, args.num_calib)
    npu16_path, meta16_path = convert_model(
        args.model, args.calib, 16, work_dir, args.num_calib)

    meta8 = np.load(meta8_path)
    meta16 = np.load(meta16_path)
    out_scale_8 = float(meta8['output_scale'])
    out_zp_8 = int(meta8['output_zp'])
    in_scale_8 = float(meta8['input_scale'])
    out_scale_16 = float(meta16['output_scale'])
    out_zp_16 = int(meta16['output_zp'])
    in_scale_16 = float(meta16['input_scale'])
    print(f"  INT8  input:  scale={in_scale_8:.8f}")
    print(f"  INT8  output: scale={out_scale_8:.8f}, zp={out_zp_8}")
    print(f"  INT16 input:  scale={in_scale_16:.10f}")
    print(f"  INT16 output: scale={out_scale_16:.10f}, zp={out_zp_16}")
    print()

    # Run inference on each image
    print("── Step 2: Inference ──")
    print(f"{'#':>3s}  {'INT8 cos':>10s}  {'INT16 cos':>10s}  Image")
    print("─" * 70)

    cos_int8_list = []
    cos_int16_list = []

    for i, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)[:40]

        # Preprocess: get float input matching calibration preprocessing
        float_input = preprocess_image(img_path, input_shape)

        # Float reference
        ort_output = sess.run([out_name], {inp_name: float_input})[0]
        ort_flat = ort_output.flatten().astype(np.float64)

        # Quantize input for INT8 csim: int8_q = round(float / in_scale_8)
        float_flat = float_input.flatten()
        q8 = np.clip(np.round(float_flat / in_scale_8), -128, 127).astype(np.int8)
        input8_bin = os.path.join(work_dir, 'input8.bin')
        q8.tofile(input8_bin)

        # INT8 csim
        out8_bin = os.path.join(work_dir, 'out8.bin')
        if run_csim(npu8_path, input8_bin, out8_bin):
            raw8 = np.fromfile(out8_bin, dtype=np.int8)
            dequant8 = (raw8.astype(np.float64) - out_zp_8) * out_scale_8
            cos8 = cosine_similarity(ort_flat, dequant8)
        else:
            cos8 = float('nan')

        # Quantize input for INT16 csim: int16_q = round(float / in_scale_16)
        q16 = np.clip(np.round(float_flat / in_scale_16), -32768, 32767).astype(np.int16)
        input16_bin = os.path.join(work_dir, 'input16.bin')
        q16.tofile(input16_bin)

        # INT16 csim
        out16_bin = os.path.join(work_dir, 'out16.bin')
        if run_csim(npu16_path, input16_bin, out16_bin):
            raw16 = np.fromfile(out16_bin, dtype=np.int16)
            dequant16 = (raw16.astype(np.float64) - out_zp_16) * out_scale_16
            cos16 = cosine_similarity(ort_flat, dequant16)
        else:
            cos16 = float('nan')

        cos_int8_list.append(cos8)
        cos_int16_list.append(cos16)
        print(f"{i+1:3d}  {cos8:10.6f}  {cos16:10.6f}  {img_name}")

    # Summary
    print("─" * 70)
    print()
    print("══════════════════════════════════════════════")
    print("              SUMMARY")
    print("══════════════════════════════════════════════")

    arr8 = np.array(cos_int8_list)
    arr16 = np.array(cos_int16_list)

    valid8 = arr8[~np.isnan(arr8)]
    valid16 = arr16[~np.isnan(arr16)]

    print(f"  Images tested:    {num}")
    print()
    print(f"  {'Metric':<20s} {'INT8':>10s}  {'INT16':>10s}")
    print(f"  {'─'*20} {'─'*10}  {'─'*10}")
    print(f"  {'Mean cosine':<20s} {valid8.mean():10.6f}  {valid16.mean():10.6f}")
    print(f"  {'Median cosine':<20s} {np.median(valid8):10.6f}  {np.median(valid16):10.6f}")
    print(f"  {'Min cosine':<20s} {valid8.min():10.6f}  {valid16.min():10.6f}")
    print(f"  {'Max cosine':<20s} {valid8.max():10.6f}  {valid16.max():10.6f}")
    print(f"  {'Std dev':<20s} {valid8.std():10.6f}  {valid16.std():10.6f}")
    print()
    print(f"  INT16 improvement: +{(valid16.mean() - valid8.mean()):.6f} mean cosine")
    print("══════════════════════════════════════════════")


if __name__ == '__main__':
    main()
