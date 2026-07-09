#!/usr/bin/env python3
"""
INT16 End-to-End Precision Validation

Validates the INT16 quantization path through the full toolchain:
  ONNX float32 → INT16 PTQ conversion → csim inference → cosine comparison

Tests both MODEL_A (63-layer face embedding) and ResNet-18 (31-layer classification)
to verify INT16 provides near-lossless precision compared to FP32.

Expected results:
  - MODEL_A INT16 cosine vs FP32: >= 0.999 (vs INT8 ~0.975)
  - ResNet-18 INT16 cosine vs FP32: >= 0.998 (vs INT8 ~0.998)

SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import glob
import subprocess
import tempfile
import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_converter import convert_model

MODEL_PATH = '/data/sam/onnx_quant/MODEL_A.onnx'
CALIB_DIR = '/data/sam/onnx_quant/a3_test_images'
CSIM_PATH = '/data/sam/open-npu/csim/npu_sim'
OUTPUT_DIR = '/tmp/int16_e2e_test'


def cosine_sim(a, b):
    """Compute cosine similarity."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_csim(model_bin, input_data, bits=16):
    """Run csim and return dequantized output."""
    input_path = os.path.join(OUTPUT_DIR, f'tmp_input_{bits}.bin')
    output_path = os.path.join(OUTPUT_DIR, f'tmp_output_{bits}.bin')
    input_data.tofile(input_path)

    result = subprocess.run(
        [CSIM_PATH, model_bin, input_path, output_path],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"csim failed: {result.stderr}")

    meta = np.load(model_bin.replace('.bin', '_meta.npz'))
    out_scale = float(meta['output_scale'])

    if bits == 16:
        raw = np.fromfile(output_path, dtype=np.int16).astype(np.float32)
    else:
        raw = np.fromfile(output_path, dtype=np.int8).astype(np.float32)
    return raw * out_scale


def quantize_input(img_path, in_scale, bits=16):
    """Load image and quantize to INT8 or INT16."""
    img = Image.open(img_path).resize((224, 224))
    arr = np.array(img).astype(np.float32).transpose(2, 0, 1)
    inp_float = (arr - 127.5) / 255.0

    if bits == 16:
        input_q = np.clip(np.round(inp_float / in_scale), -32768, 32767).astype(np.int16)
    else:
        input_q = np.clip(np.round(inp_float / in_scale), -128, 127).astype(np.int8)
    return input_q


def setup_model_a():
    """Convert MODEL_A model in both INT8 and INT16 modes."""
    model_a_dir = os.path.join(OUTPUT_DIR, 'model_a')
    os.makedirs(model_a_dir, exist_ok=True)

    all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))
    img = Image.open(all_imgs[0]).resize((224, 224))
    arr = np.array(img).transpose(2, 0, 1).astype(np.uint8)
    input_bin = os.path.join(model_a_dir, 'input.bin')
    arr.tofile(input_bin)

    models = {}
    for bits in (8, 16):
        model_bin = os.path.join(model_a_dir, f'model_int{bits}.npu1.bin')
        if not os.path.exists(model_bin):
            print(f"  Converting MODEL_A INT{bits}...")
            convert_model(MODEL_PATH, CALIB_DIR, input_bin, model_bin,
                          input_format='int8-nchw', num_calib=50, bits=bits)
        models[bits] = model_bin

    return models


def setup_resnet18():
    """Build and convert synthetic ResNet-18 in both INT8 and INT16 modes."""
    import onnx
    from test_resnet18_e2e import make_resnet18_model, run_e2e

    resnet_dir = os.path.join(OUTPUT_DIR, 'resnet18')
    os.makedirs(resnet_dir, exist_ok=True)

    model_path = os.path.join(resnet_dir, 'resnet18.onnx')
    if not os.path.exists(model_path):
        print("  Building synthetic ResNet-18...")
        model = make_resnet18_model(num_classes=10, input_size=224)
        onnx.save(model, model_path)

    # Calibration images
    calib_dir = os.path.join(resnet_dir, 'calib')
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)
        for i in range(20):
            np.random.seed(i)
            img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(calib_dir, f'calib_{i:03d}.jpg'))

    # Test image
    test_path = os.path.join(resnet_dir, 'test.jpg')
    if not os.path.exists(test_path):
        np.random.seed(999)
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(img).save(test_path)

    return model_path, calib_dir, test_path


def test_model_a_int16(num_images=10):
    """Test MODEL_A INT16 precision across multiple images."""
    print(f"\n{'='*60}")
    print("Test 1: MODEL_A INT16 vs FP32 (face embedding, 63 layers)")
    print(f"{'='*60}")

    if not os.path.exists(MODEL_PATH):
        print("  SKIP: MODEL_A model not available")
        return None

    models = setup_model_a()

    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    meta16 = np.load(models[16].replace('.bin', '_meta.npz'))
    meta8 = np.load(models[8].replace('.bin', '_meta.npz'))
    in_scale_16 = float(meta16['input_scale'])
    in_scale_8 = float(meta8['input_scale'])

    all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))
    test_imgs = all_imgs[-num_images:]

    cos8_list = []
    cos16_list = []

    print(f"\n  {'Image':<50} {'INT8':>8} {'INT16':>8}")
    print(f"  {'-'*50} {'-'*8} {'-'*8}")

    for img_path in test_imgs:
        # FP32 reference
        img = Image.open(img_path).resize((224, 224))
        arr = np.array(img).astype(np.float32).transpose(2, 0, 1)
        inp_float = (arr - 127.5) / 255.0
        ref = sess.run(None, {input_name: inp_float[np.newaxis, ...]})[0].flatten()

        # INT8
        q8 = quantize_input(img_path, in_scale_8, bits=8)
        out8 = run_csim(models[8], q8, bits=8)
        cos8 = cosine_sim(ref, out8)
        cos8_list.append(cos8)

        # INT16
        q16 = quantize_input(img_path, in_scale_16, bits=16)
        out16 = run_csim(models[16], q16, bits=16)
        cos16 = cosine_sim(ref, out16)
        cos16_list.append(cos16)

        name = os.path.basename(img_path)[:48]
        print(f"  {name:<50} {cos8:.6f} {cos16:.6f}")

    avg8 = np.mean(cos8_list)
    avg16 = np.mean(cos16_list)
    print(f"  {'-'*50} {'-'*8} {'-'*8}")
    print(f"  {'AVERAGE':<50} {avg8:.6f} {avg16:.6f}")
    print(f"\n  INT16 improvement over INT8: +{avg16-avg8:.6f}")
    print(f"  INT16 range: [{min(cos16_list):.6f}, {max(cos16_list):.6f}]")

    threshold = 0.999
    if avg16 >= threshold:
        print(f"  PASS (INT16 avg cosine {avg16:.6f} >= {threshold})")
        return True
    else:
        print(f"  FAIL (INT16 avg cosine {avg16:.6f} < {threshold})")
        return False


def test_resnet18_int16():
    """Test ResNet-18 INT16 precision."""
    print(f"\n{'='*60}")
    print("Test 2: ResNet-18 INT16 vs FP32 (classification, 31 layers)")
    print(f"{'='*60}")

    try:
        from test_resnet18_e2e import make_resnet18_model, run_e2e
    except ImportError:
        print("  SKIP: test_resnet18_e2e not available")
        return None

    model_path, calib_dir, test_path = setup_resnet18()

    # FP32 reference
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    test_img = np.array(Image.open(test_path))
    inp_f = ((test_img.astype(np.float32).transpose(2, 0, 1) - 127.5) / 255.0)[np.newaxis, ...]
    ref = sess.run(None, {sess.get_inputs()[0].name: inp_f})[0].flatten()

    # INT8
    print("  Running INT8...")
    out8, _ = run_e2e(model_path, calib_dir, test_path, bits=8)
    cos8 = cosine_sim(ref, out8)

    # INT16
    print("  Running INT16...")
    out16, _ = run_e2e(model_path, calib_dir, test_path, bits=16)
    cos16 = cosine_sim(ref, out16)

    print(f"\n  INT8  cosine: {cos8:.6f}  top-1: {'match' if ref.argmax() == out8.argmax() else 'MISMATCH'}")
    print(f"  INT16 cosine: {cos16:.6f}  top-1: {'match' if ref.argmax() == out16.argmax() else 'MISMATCH'}")
    print(f"  Improvement: +{cos16-cos8:.6f}")

    # Multi-image stability
    print("\n  Multi-image (5 random images):")
    cos16_list = []
    for i in range(5):
        np.random.seed(100 + i)
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img_path = os.path.join(OUTPUT_DIR, 'resnet18', f'multi_{i}.jpg')
        Image.fromarray(img).save(img_path)

        inp_f = ((img.astype(np.float32).transpose(2, 0, 1) - 127.5) / 255.0)[np.newaxis, ...]
        r = sess.run(None, {sess.get_inputs()[0].name: inp_f})[0].flatten()

        o16, _ = run_e2e(model_path, calib_dir, img_path, bits=16)
        c16 = cosine_sim(r, o16)
        cos16_list.append(c16)
        print(f"    Image {i}: cos={c16:.6f} top-1={'match' if r.argmax()==o16.argmax() else 'MISS'}")

    avg16 = np.mean(cos16_list)
    print(f"  Average: {avg16:.6f}")

    threshold = 0.995
    if cos16 >= threshold and avg16 >= threshold:
        print(f"  PASS (INT16 cosine {cos16:.6f} >= {threshold})")
        return True
    else:
        print(f"  FAIL (INT16 cosine {cos16:.6f} < {threshold})")
        return False


def test_int16_output_distribution():
    """Test 3: Verify INT16 output uses the full dynamic range well."""
    print(f"\n{'='*60}")
    print("Test 3: INT16 output distribution (MODEL_A)")
    print(f"{'='*60}")

    if not os.path.exists(MODEL_PATH):
        print("  SKIP: MODEL_A model not available")
        return None

    models = setup_model_a()
    meta16 = np.load(models[16].replace('.bin', '_meta.npz'))
    in_scale_16 = float(meta16['input_scale'])

    all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, '*.jpg')))
    q16 = quantize_input(all_imgs[0], in_scale_16, bits=16)

    input_path = os.path.join(OUTPUT_DIR, 'tmp_input_16.bin')
    output_path = os.path.join(OUTPUT_DIR, 'tmp_output_16.bin')
    q16.tofile(input_path)
    subprocess.run([CSIM_PATH, models[16], input_path, output_path],
                   capture_output=True)

    raw = np.fromfile(output_path, dtype=np.int16)
    out_scale = float(meta16['output_scale'])
    emb = raw.astype(np.float32) * out_scale

    print(f"  Embedding dim: {len(raw)}")
    print(f"  Quantized range: [{raw.min()}, {raw.max()}]")
    print(f"  Non-zero elements: {np.count_nonzero(raw)}/{len(raw)} "
          f"({100*np.count_nonzero(raw)/len(raw):.1f}%)")
    print(f"  Float range: [{emb.min():.8f}, {emb.max():.8f}]")
    print(f"  Float std: {emb.std():.8f}")
    print(f"  L2 norm: {np.linalg.norm(emb):.6f}")

    # INT16 should have much better range utilization
    utilization = (int(raw.max()) - int(raw.min())) / 65535.0
    non_zero_ratio = np.count_nonzero(raw) / len(raw)
    print(f"  Range utilization: {utilization:.1%}")
    print(f"  Non-zero ratio: {non_zero_ratio:.1%}")

    ok = True
    if non_zero_ratio < 0.5:
        print(f"  WARNING: Too sparse ({non_zero_ratio:.1%} non-zero)")
        ok = False
    if utilization < 0.1:
        print(f"  WARNING: Low range utilization ({utilization:.1%})")
        ok = False

    if ok:
        print(f"  PASS (good distribution)")
    return ok


def test_model_size():
    """Test 4: Verify INT16 model is ~2x the INT8 model (weights doubled)."""
    print(f"\n{'='*60}")
    print("Test 4: Model size comparison")
    print(f"{'='*60}")

    if not os.path.exists(MODEL_PATH):
        print("  SKIP: MODEL_A model not available")
        return None

    models = setup_model_a()
    size8 = os.path.getsize(models[8])
    size16 = os.path.getsize(models[16])
    ratio = size16 / size8

    print(f"  INT8  model: {size8:>10,} bytes ({size8/1024:.1f} KB)")
    print(f"  INT16 model: {size16:>10,} bytes ({size16/1024:.1f} KB)")
    print(f"  Ratio (INT16/INT8): {ratio:.2f}x")
    print(f"  Weight overhead: +{(ratio-1)*100:.0f}%")

    # INT16 should be roughly 2x (weights double, descriptors stay same)
    if 1.5 <= ratio <= 2.5:
        print(f"  PASS (ratio {ratio:.2f}x in expected range [1.5, 2.5])")
        return True
    else:
        print(f"  FAIL (ratio {ratio:.2f}x outside expected range)")
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Open-NPU INT16 End-to-End Precision Validation")
    print("=" * 60)
    print(f"  csim: {CSIM_PATH}")
    print(f"  MODEL_A: {MODEL_PATH}")
    print(f"  Output: {OUTPUT_DIR}")

    if not os.path.exists(CSIM_PATH):
        print(f"\nERROR: csim not found at {CSIM_PATH}")
        print("  Run: cd /data/sam/open-npu/csim && make")
        sys.exit(1)

    results = []

    # Test 1: MODEL_A multi-image precision
    r = test_model_a_int16(num_images=10)
    if r is not None:
        results.append(('MODEL_A INT16 cosine >= 0.999', r))

    # Test 2: ResNet-18 INT16 precision
    r = test_resnet18_int16()
    if r is not None:
        results.append(('ResNet-18 INT16 cosine >= 0.995', r))

    # Test 3: Output distribution
    r = test_int16_output_distribution()
    if r is not None:
        results.append(('INT16 output distribution', r))

    # Test 4: Model size sanity
    r = test_model_size()
    if r is not None:
        results.append(('Model size ratio', r))

    # Summary
    print(f"\n{'='*60}")
    print("INT16 E2E VALIDATION SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\nAll {len(results)} tests PASSED!")
        print("INT16 quantization provides near-lossless precision.")
    else:
        print("\nSome tests FAILED!")

    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
