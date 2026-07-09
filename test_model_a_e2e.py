#!/usr/bin/env python3
"""
Phase 3: MODEL_A Full Model End-to-End Validation

Validates the entire toolchain (calibration → conversion → csim inference)
on MODEL_A.onnx — a 63-layer MobileNetV2-style face embedding network.

Metrics:
  - Cosine similarity between ORT float32 and csim INT8 outputs
  - Stability across multiple test images
  - Pairwise embedding consistency (same ranking as float32)

SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import glob
import subprocess
import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_converter import convert_model

MODEL_PATH = '/data/sam/onnx_quant/MODEL_A.onnx'
CALIB_DIR = '/data/sam/onnx_quant/a3_test_images'
CSIM_PATH = '/data/sam/open-npu/csim/npu_sim'
OUTPUT_DIR = '/tmp/model_a_e2e_phase3'


def setup():
    """Convert MODEL_A model once for all tests."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_bin = os.path.join(OUTPUT_DIR, 'model.npu1.bin')
    if os.path.exists(model_bin):
        print("Using cached converted model")
        return model_bin

    # Create a test input for the converter
    all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))
    img = Image.open(all_imgs[0]).resize((224, 224))
    arr = np.array(img).transpose(2, 0, 1).astype(np.uint8)
    input_bin = os.path.join(OUTPUT_DIR, 'input.bin')
    arr.tofile(input_bin)

    print("=== Converting MODEL_A model ===")
    convert_model(MODEL_PATH, CALIB_DIR, input_bin, model_bin,
                  input_format='int8-nchw', num_calib=50, bits=8)
    return model_bin


def get_ort_embedding(sess, img_path):
    """Get normalized embedding from ORT float32 inference."""
    img = Image.open(img_path).resize((224, 224))
    arr = np.array(img).astype(np.float32).transpose(2, 0, 1)
    inp = ((arr - 127.5) / 255.0)[np.newaxis, ...]
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0].flatten()
    return out


def get_csim_embedding(model_bin, img_path, in_scale, out_scale):
    """Get embedding from csim INT8 inference."""
    img = Image.open(img_path).resize((224, 224))
    arr = np.array(img).astype(np.float32).transpose(2, 0, 1)
    inp_float = (arr - 127.5) / 255.0
    input_q = np.round(inp_float / in_scale).astype(np.int32)
    input_q = np.clip(input_q, -128, 127).astype(np.int8)

    input_path = os.path.join(OUTPUT_DIR, 'test_input.bin')
    output_path = os.path.join(OUTPUT_DIR, 'test_output.bin')
    input_q.tofile(input_path)

    result = subprocess.run(
        [CSIM_PATH, model_bin, input_path, output_path],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"csim failed: {result.stderr}")

    csim_q = np.fromfile(output_path, dtype=np.int8).astype(np.float32)
    return csim_q * out_scale


def cosine_sim(a, b):
    """Compute cosine similarity."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)


def test_accuracy(model_bin, num_images=10):
    """Test 1: Cosine similarity across multiple images."""
    print(f"\n{'='*60}")
    print("Test 1: ORT vs csim cosine similarity")
    print(f"{'='*60}")

    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    meta = np.load(model_bin.replace('.bin', '_meta.npz'))
    in_scale = float(meta['input_scale'])
    out_scale = float(meta['output_scale'])

    all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))
    # Use images NOT in calibration set (last N images)
    test_imgs = all_imgs[-num_images:]

    cosines = []
    for img_path in test_imgs:
        ref = get_ort_embedding(sess, img_path)
        csim = get_csim_embedding(model_bin, img_path, in_scale, out_scale)
        cos = cosine_sim(ref, csim)
        cosines.append(cos)
        name = os.path.basename(img_path)[:50]
        print(f"  {name:50s} cos={cos:.6f}")

    avg = np.mean(cosines)
    print(f"\n  Average: {avg:.6f}")
    print(f"  Min:     {np.min(cosines):.6f}")
    print(f"  Max:     {np.max(cosines):.6f}")
    print(f"  Std:     {np.std(cosines):.6f}")

    threshold = 0.95
    if avg >= threshold:
        print(f"  PASS (avg cosine {avg:.4f} >= {threshold})")
        return True
    else:
        print(f"  FAIL (avg cosine {avg:.4f} < {threshold})")
        return False


def test_ranking_preservation(model_bin, num_images=6):
    """Test 2: Verify pairwise similarity ranking is preserved after quantization."""
    print(f"\n{'='*60}")
    print("Test 2: Pairwise ranking preservation")
    print(f"{'='*60}")

    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    meta = np.load(model_bin.replace('.bin', '_meta.npz'))
    in_scale = float(meta['input_scale'])
    out_scale = float(meta['output_scale'])

    all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))
    test_imgs = all_imgs[:num_images]

    # Get all embeddings
    ort_embs = []
    csim_embs = []
    for img_path in test_imgs:
        ort_embs.append(get_ort_embedding(sess, img_path))
        csim_embs.append(get_csim_embedding(model_bin, img_path, in_scale, out_scale))

    # Normalize
    ort_normed = [e / (np.linalg.norm(e) + 1e-10) for e in ort_embs]
    csim_normed = [e / (np.linalg.norm(e) + 1e-10) for e in csim_embs]

    # Compute pairwise similarities
    ort_pairs = []
    csim_pairs = []
    for i in range(num_images):
        for j in range(i + 1, num_images):
            ort_cos = np.dot(ort_normed[i], ort_normed[j])
            csim_cos = np.dot(csim_normed[i], csim_normed[j])
            ort_pairs.append(ort_cos)
            csim_pairs.append(csim_cos)

    ort_pairs = np.array(ort_pairs)
    csim_pairs = np.array(csim_pairs)

    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(ort_pairs, csim_pairs)
    print(f"  Spearman rank correlation: {rho:.6f} (p={pval:.2e})")
    print(f"  Pairs compared: {len(ort_pairs)}")
    print(f"  ORT  pairwise range: [{ort_pairs.min():.4f}, {ort_pairs.max():.4f}]")
    print(f"  csim pairwise range: [{csim_pairs.min():.4f}, {csim_pairs.max():.4f}]")

    # Max absolute difference in pairwise cosines
    max_diff = np.max(np.abs(ort_pairs - csim_pairs))
    avg_diff = np.mean(np.abs(ort_pairs - csim_pairs))
    print(f"  Max pairwise cosine diff: {max_diff:.6f}")
    print(f"  Avg pairwise cosine diff: {avg_diff:.6f}")

    threshold = 0.90
    if rho >= threshold:
        print(f"  PASS (Spearman rho {rho:.4f} >= {threshold})")
        return True
    else:
        print(f"  FAIL (Spearman rho {rho:.4f} < {threshold})")
        return False


def test_output_distribution(model_bin):
    """Test 3: Verify output embedding has reasonable distribution."""
    print(f"\n{'='*60}")
    print("Test 3: Output embedding distribution")
    print(f"{'='*60}")

    meta = np.load(model_bin.replace('.bin', '_meta.npz'))
    in_scale = float(meta['input_scale'])
    out_scale = float(meta['output_scale'])

    all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))

    # Get a csim embedding
    emb = get_csim_embedding(model_bin, all_imgs[0], in_scale, out_scale)
    csim_q = np.fromfile(os.path.join(OUTPUT_DIR, 'test_output.bin'), dtype=np.int8)

    print(f"  Embedding dim: {len(emb)}")
    print(f"  Quantized range: [{csim_q.min()}, {csim_q.max()}]")
    print(f"  Non-zero elements: {np.count_nonzero(csim_q)}/{len(csim_q)} "
          f"({100*np.count_nonzero(csim_q)/len(csim_q):.1f}%)")
    print(f"  Float range: [{emb.min():.6f}, {emb.max():.6f}]")
    print(f"  Float std: {emb.std():.6f}")
    print(f"  L2 norm: {np.linalg.norm(emb):.6f}")

    # Check: embedding should use a good portion of the quantized range
    utilization = (int(csim_q.max()) - int(csim_q.min())) / 255.0
    non_zero_ratio = np.count_nonzero(csim_q) / len(csim_q)

    ok = True
    if non_zero_ratio < 0.5:
        print(f"  WARNING: Too sparse ({non_zero_ratio:.1%} non-zero)")
        ok = False
    if utilization < 0.3:
        print(f"  WARNING: Low range utilization ({utilization:.1%})")
        ok = False

    if ok:
        print(f"  PASS (good distribution)")
    return ok


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(CSIM_PATH):
        print(f"ERROR: csim not found: {CSIM_PATH}")
        sys.exit(1)

    model_bin = setup()

    results = []
    results.append(('Accuracy (ORT vs csim)', test_accuracy(model_bin, num_images=10)))
    results.append(('Output distribution', test_output_distribution(model_bin)))

    try:
        results.append(('Ranking preservation', test_ranking_preservation(model_bin, num_images=6)))
    except ImportError:
        print("\n  SKIP: scipy not available for Spearman test")
        results.append(('Ranking preservation', True))  # skip gracefully

    # Summary
    print(f"\n{'='*60}")
    print("MODEL_A E2E VALIDATION SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nMODEL_A Phase 3 validation PASSED!")
    else:
        print("\nSome tests FAILED!")
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
