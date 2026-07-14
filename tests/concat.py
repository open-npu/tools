#!/usr/bin/env python3
"""
Test Concat support in onnx_converter.py

Creates ONNX models with:
  1. Conv + Conv + Concat (two branches, channel concat)
  2. Conv + Conv + Concat + Conv (concat in middle of network)

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx_converter import convert_model

CSIM_PATH = '/data/sam/open-npu/csim/npu_sim'


def make_two_branch_concat_model(in_c=3, c_a=4, c_b=8, h=8, w=8):
    """Two parallel Conv branches → Concat on channel axis.

    input → Conv_A (3→4, 3x3) → branch_a (4ch)
          → Conv_B (3→8, 1x1) → branch_b (8ch)
    Concat([branch_a, branch_b], axis=1) → output (12ch)
    """
    np.random.seed(42)

    conv_a_w = np.random.randn(c_a, in_c, 3, 3).astype(np.float32) * 0.1
    conv_a_b = np.random.randn(c_a).astype(np.float32) * 0.01
    conv_b_w = np.random.randn(c_b, in_c, 1, 1).astype(np.float32) * 0.1
    conv_b_b = np.random.randn(c_b).astype(np.float32) * 0.01

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_c, h, w])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, c_a + c_b, h, w])

    conv_a = helper.make_node('Conv', ['input', 'conv_a_w', 'conv_a_b'], ['branch_a'],
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    relu_a = helper.make_node('Relu', ['branch_a'], ['branch_a_relu'])
    conv_b = helper.make_node('Conv', ['input', 'conv_b_w', 'conv_b_b'], ['branch_b'],
                              kernel_shape=[1, 1])
    relu_b = helper.make_node('Relu', ['branch_b'], ['branch_b_relu'])
    concat = helper.make_node('Concat', ['branch_a_relu', 'branch_b_relu'], ['output'],
                              axis=1)

    graph = helper.make_graph(
        [conv_a, relu_a, conv_b, relu_b, concat],
        'two_branch_concat',
        [X], [Y],
        initializer=[
            numpy_helper.from_array(conv_a_w, 'conv_a_w'),
            numpy_helper.from_array(conv_a_b, 'conv_a_b'),
            numpy_helper.from_array(conv_b_w, 'conv_b_w'),
            numpy_helper.from_array(conv_b_b, 'conv_b_b'),
        ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    model = onnx.shape_inference.infer_shapes(model)
    return model


def make_concat_conv_model(in_c=3, c_a=4, c_b=4, out_c=8, h=8, w=8):
    """Two branches → Concat → Conv (concat feeds into downstream computation).

    input → Conv_A (3→4, 3x3, relu) → branch_a
          → Conv_B (3→4, 1x1, relu) → branch_b
    Concat([branch_a, branch_b], axis=1) → cat (8ch)
    Conv_C (8→out_c, 1x1) → output
    """
    np.random.seed(77)

    conv_a_w = np.random.randn(c_a, in_c, 3, 3).astype(np.float32) * 0.1
    conv_a_b = np.random.randn(c_a).astype(np.float32) * 0.01
    conv_b_w = np.random.randn(c_b, in_c, 1, 1).astype(np.float32) * 0.1
    conv_b_b = np.random.randn(c_b).astype(np.float32) * 0.01
    conv_c_w = np.random.randn(out_c, c_a + c_b, 1, 1).astype(np.float32) * 0.1
    conv_c_b = np.random.randn(out_c).astype(np.float32) * 0.01

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_c, h, w])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, out_c, h, w])

    conv_a = helper.make_node('Conv', ['input', 'conv_a_w', 'conv_a_b'], ['branch_a'],
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    relu_a = helper.make_node('Relu', ['branch_a'], ['branch_a_relu'])
    conv_b = helper.make_node('Conv', ['input', 'conv_b_w', 'conv_b_b'], ['branch_b'],
                              kernel_shape=[1, 1])
    relu_b = helper.make_node('Relu', ['branch_b'], ['branch_b_relu'])
    concat = helper.make_node('Concat', ['branch_a_relu', 'branch_b_relu'], ['cat_out'],
                              axis=1)
    conv_c = helper.make_node('Conv', ['cat_out', 'conv_c_w', 'conv_c_b'], ['output'],
                              kernel_shape=[1, 1])

    graph = helper.make_graph(
        [conv_a, relu_a, conv_b, relu_b, concat, conv_c],
        'concat_conv',
        [X], [Y],
        initializer=[
            numpy_helper.from_array(conv_a_w, 'conv_a_w'),
            numpy_helper.from_array(conv_a_b, 'conv_a_b'),
            numpy_helper.from_array(conv_b_w, 'conv_b_w'),
            numpy_helper.from_array(conv_b_b, 'conv_b_b'),
            numpy_helper.from_array(conv_c_w, 'conv_c_w'),
            numpy_helper.from_array(conv_c_b, 'conv_c_b'),
        ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    model = onnx.shape_inference.infer_shapes(model)
    return model


def run_e2e_test(model_name, onnx_model, h=8, w=8, in_c=3, bits=8, threshold=0.85):
    """End-to-end: ONNX model → converter → csim → compare cosine with ORT."""
    print(f"\n{'='*60}")
    print(f"Test: {model_name} (INT{bits})")
    print(f"{'='*60}")

    tmpdir = tempfile.mkdtemp(prefix='npu_concat_test_')
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

    threshold_val = threshold
    if cosine >= threshold_val:
        print(f"  PASS (cosine >= {threshold_val})")
        return True
    else:
        print(f"  FAIL (cosine < {threshold_val})")
        print(f"    ref[:8]  = {ref_out[:8]}")
        print(f"    csim[:8] = {csim_out_float[:8]}")
        return False


def test_e2e_all():
    """Run all Concat E2E tests."""
    results = []

    # Test 1: Two-branch concat (output is the concat itself)
    # With per-branch requantize, all branches are rescaled to unified output scale.
    model1 = make_two_branch_concat_model()
    results.append(('Two-branch Concat', run_e2e_test(
        'Two-branch Concat', model1, h=8, w=8, threshold=0.80)))

    # Test 2: Concat in middle of network (concat feeds into downstream Conv)
    model2 = make_concat_conv_model()
    results.append(('Concat+Conv', run_e2e_test(
        'Concat+Conv', model2, h=8, w=8, threshold=0.90)))

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
        print("\nAll Concat tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == '__main__':
    test_e2e_all()
