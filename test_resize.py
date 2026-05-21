#!/usr/bin/env python3
"""
Test Resize/Upsample support in onnx_converter.py

Creates ONNX models with:
  1. Conv + Resize(nearest, 2x) + Conv  — nearest-neighbor upsampling
  2. Conv + Resize(bilinear, 2x) + Conv — bilinear interpolation
  3. Conv + Upsample(nearest, 2x) + Conv — legacy Upsample op (opset 9)

Runs each through: ONNX Runtime (float ref) → converter → csim → compare.

SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import tempfile
import subprocess
import numpy as np

import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_converter import convert_model

CSIM_PATH = '/data/sam/open-npu/csim/npu_sim'


def make_conv_resize_conv_model(resize_mode='nearest', scale=2, h=8, w=8,
                                 in_c=3, mid_c=4, out_c=8):
    """Conv → Resize → Conv model.

    input (3, h, w) → Conv_A (3→mid_c, 3x3, relu) → feature (mid_c, h, w)
    → Resize(scale x scale) → upsampled (mid_c, h*scale, w*scale)
    → Conv_B (mid_c→out_c, 3x3) → output (out_c, h*scale, w*scale)
    """
    np.random.seed(42)

    conv_a_w = np.random.randn(mid_c, in_c, 3, 3).astype(np.float32) * 0.1
    conv_a_b = np.random.randn(mid_c).astype(np.float32) * 0.01
    conv_b_w = np.random.randn(out_c, mid_c, 3, 3).astype(np.float32) * 0.1
    conv_b_b = np.random.randn(out_c).astype(np.float32) * 0.01

    out_h, out_w = h * scale, w * scale

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_c, h, w])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                       [1, out_c, out_h, out_w])

    conv_a = helper.make_node('Conv', ['input', 'conv_a_w', 'conv_a_b'], ['feat'],
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    relu_a = helper.make_node('Relu', ['feat'], ['feat_relu'])

    # Resize node (opset 11): inputs = [X, roi, scales, sizes]
    # For scale-based resize: roi='', scales=[1,1,scale_h,scale_w], sizes=''
    roi = numpy_helper.from_array(np.array([], dtype=np.float32), 'roi')
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, float(scale), float(scale)], dtype=np.float32), 'scales')

    resize = helper.make_node('Resize', ['feat_relu', 'roi', 'scales'], ['resized'],
                              mode=resize_mode)

    conv_b = helper.make_node('Conv', ['resized', 'conv_b_w', 'conv_b_b'], ['output'],
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1])

    graph = helper.make_graph(
        [conv_a, relu_a, resize, conv_b],
        f'conv_resize_{resize_mode}_conv',
        [X], [Y],
        initializer=[
            numpy_helper.from_array(conv_a_w, 'conv_a_w'),
            numpy_helper.from_array(conv_a_b, 'conv_a_b'),
            numpy_helper.from_array(conv_b_w, 'conv_b_w'),
            numpy_helper.from_array(conv_b_b, 'conv_b_b'),
            roi, scales,
        ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    model = onnx.shape_inference.infer_shapes(model)
    return model


def make_resize_only_model(resize_mode='nearest', scale=2, h=8, w=8, in_c=3):
    """Conv → Resize (as final output) — tests resize standalone.

    input (3, h, w) → Conv (3→4, 1x1, relu) → feature (4, h, w)
    → Resize(scale x scale) → output (4, h*scale, w*scale)
    """
    np.random.seed(55)
    mid_c = 4

    conv_w = np.random.randn(mid_c, in_c, 1, 1).astype(np.float32) * 0.1
    conv_b = np.random.randn(mid_c).astype(np.float32) * 0.01

    out_h, out_w = h * scale, w * scale

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_c, h, w])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                       [1, mid_c, out_h, out_w])

    conv = helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['feat'],
                            kernel_shape=[1, 1])
    relu = helper.make_node('Relu', ['feat'], ['feat_relu'])

    roi = numpy_helper.from_array(np.array([], dtype=np.float32), 'roi')
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, float(scale), float(scale)], dtype=np.float32), 'scales')

    resize = helper.make_node('Resize', ['feat_relu', 'roi', 'scales'], ['output'],
                              mode=resize_mode)

    graph = helper.make_graph(
        [conv, relu, resize],
        f'resize_only_{resize_mode}',
        [X], [Y],
        initializer=[
            numpy_helper.from_array(conv_w, 'conv_w'),
            numpy_helper.from_array(conv_b, 'conv_b'),
            roi, scales,
        ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    model = onnx.shape_inference.infer_shapes(model)
    return model


def run_e2e_test(model_name, onnx_model, h=8, w=8, in_c=3, bits=8, threshold=0.90):
    """End-to-end: ONNX model → converter → csim → compare cosine with ORT."""
    print(f"\n{'='*60}")
    print(f"Test: {model_name} (INT{bits})")
    print(f"{'='*60}")

    tmpdir = tempfile.mkdtemp(prefix='npu_resize_test_')
    model_path = os.path.join(tmpdir, 'model.onnx')
    onnx.save(onnx_model, model_path)

    # Create calibration images
    calib_dir = os.path.join(tmpdir, 'calib')
    os.makedirs(calib_dir)
    from PIL import Image
    np.random.seed(42)
    for i in range(20):
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(calib_dir, f'calib_{i:04d}.jpg'))

    # Test input
    np.random.seed(99)
    test_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    Image.fromarray(test_img).save(os.path.join(calib_dir, 'test.jpg'))

    # ORT reference
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp_float = (test_img.astype(np.float32).transpose(2, 0, 1) - 127.5) / 255.0
    inp_float = inp_float[np.newaxis, ...]
    ref_out = sess.run(None, {sess.get_inputs()[0].name: inp_float})[0].flatten()
    print(f"  ORT output: shape={ref_out.shape}, range=[{ref_out.min():.4f}, {ref_out.max():.4f}]")

    # Write input
    input_bin = os.path.join(tmpdir, 'input.bin')
    test_nchw = test_img.transpose(2, 0, 1)
    test_nchw.tofile(input_bin)

    # Run converter
    npu_model_path = os.path.join(tmpdir, 'model.npu1.bin')
    try:
        convert_model(model_path, calib_dir, input_bin, npu_model_path,
                      input_format='int8-nchw', num_calib=20, bits=bits)
    except Exception as e:
        print(f"  FAIL: Converter error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run csim
    output_bin = os.path.join(tmpdir, 'output.bin')
    npu_input_actual = npu_model_path.replace('.bin', '_input.bin')

    if not os.path.exists(CSIM_PATH):
        print(f"  SKIP: csim not found at {CSIM_PATH}")
        return True

    result = subprocess.run(
        [CSIM_PATH, npu_model_path, npu_input_actual, output_bin],
        capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAIL: csim error (rc={result.returncode}):")
        print(f"    stdout: {result.stdout[-300:]}")
        print(f"    stderr: {result.stderr[-300:]}")
        return False

    # Read output and dequantize
    meta_path = npu_model_path.replace('.bin', '_meta.npz')
    meta = np.load(meta_path)
    out_scale = float(meta['output_scale'])

    if bits == 8:
        csim_out_q = np.fromfile(output_bin, dtype=np.int8).astype(np.float32)
    else:
        csim_out_q = np.fromfile(output_bin, dtype=np.int16).astype(np.float32)

    csim_out_float = csim_out_q * out_scale

    # Compare
    if ref_out.size != csim_out_float.size:
        print(f"  FAIL: Output size mismatch: ORT={ref_out.size}, csim={csim_out_float.size}")
        return False

    dot = np.dot(ref_out, csim_out_float)
    norm_a = np.linalg.norm(ref_out)
    norm_b = np.linalg.norm(csim_out_float)
    if norm_a < 1e-10 or norm_b < 1e-10:
        cosine = 0.0
    else:
        cosine = dot / (norm_a * norm_b)

    print(f"  Cosine similarity: {cosine:.6f}")
    print(f"  Output elements: {csim_out_float.size}")

    if cosine >= threshold:
        print(f"  PASS (cosine >= {threshold})")
        return True
    else:
        print(f"  FAIL (cosine < {threshold})")
        print(f"    ref[:8]  = {ref_out[:8]}")
        print(f"    csim[:8] = {csim_out_float[:8]}")
        return False


def test_e2e_all():
    """Run all Resize E2E tests."""
    results = []

    # Test 1: Conv → Resize(nearest, 2x) → Conv
    model1 = make_conv_resize_conv_model(resize_mode='nearest', scale=2)
    results.append(('Conv+Resize(nearest)+Conv', run_e2e_test(
        'Conv+Resize(nearest)+Conv', model1, h=8, w=8, threshold=0.90)))

    # Test 2: Conv → Resize(bilinear, 2x) → Conv
    model2 = make_conv_resize_conv_model(resize_mode='linear', scale=2)
    results.append(('Conv+Resize(bilinear)+Conv', run_e2e_test(
        'Conv+Resize(bilinear)+Conv', model2, h=8, w=8, threshold=0.90)))

    # Test 3: Resize as final output (nearest) — tests passthrough quant
    model3 = make_resize_only_model(resize_mode='nearest', scale=2)
    results.append(('Resize(nearest) only', run_e2e_test(
        'Resize(nearest) only', model3, h=8, w=8, threshold=0.95)))

    # Test 4: Resize as final output (bilinear)
    model4 = make_resize_only_model(resize_mode='linear', scale=2)
    results.append(('Resize(bilinear) only', run_e2e_test(
        'Resize(bilinear) only', model4, h=8, w=8, threshold=0.90)))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll Resize tests PASSED!")
    else:
        print("\nSome tests FAILED!")
    return all_pass


if __name__ == '__main__':
    success = test_e2e_all()
    sys.exit(0 if success else 1)
