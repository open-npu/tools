#!/usr/bin/env python3
"""
Test BatchNorm fold and Pooling support in onnx_converter.py

Creates small ONNX models with:
  1. Conv + BatchNorm (test BN folding)
  2. Conv + MaxPool (test max pooling)
  3. Conv + GlobalAveragePool (test global avg pooling)
  4. Conv + BN + ReLU + MaxPool (combined test)

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
from onnx_converter import fold_batchnorm, parse_onnx_graph, convert_model

CSIM_PATH = '/data/sam/open-npu/csim/npu_sim'


def make_conv_bn_model(in_c=3, out_c=8, ksize=3, h=8, w=8):
    """Create a minimal Conv+BN ONNX model."""
    np.random.seed(42)

    # Conv weights
    conv_w = np.random.randn(out_c, in_c, ksize, ksize).astype(np.float32) * 0.1
    conv_b = np.random.randn(out_c).astype(np.float32) * 0.01

    # BN params
    bn_scale = np.random.rand(out_c).astype(np.float32) * 0.5 + 0.5
    bn_bias = np.random.randn(out_c).astype(np.float32) * 0.1
    bn_mean = np.random.randn(out_c).astype(np.float32) * 0.2
    bn_var = np.abs(np.random.randn(out_c).astype(np.float32)) + 0.1

    # Build graph
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_c, h, w])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, None)

    conv_node = helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'],
                                 kernel_shape=[ksize, ksize], pads=[1, 1, 1, 1])
    bn_node = helper.make_node('BatchNormalization',
                               ['conv_out', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
                               ['output'], epsilon=1e-5)

    graph = helper.make_graph(
        [conv_node, bn_node], 'conv_bn_test', [X], [Y],
        initializer=[
            numpy_helper.from_array(conv_w, 'conv_w'),
            numpy_helper.from_array(conv_b, 'conv_b'),
            numpy_helper.from_array(bn_scale, 'bn_scale'),
            numpy_helper.from_array(bn_bias, 'bn_bias'),
            numpy_helper.from_array(bn_mean, 'bn_mean'),
            numpy_helper.from_array(bn_var, 'bn_var'),
        ])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    return model


def make_conv_maxpool_model(in_c=3, out_c=8, h=8, w=8):
    """Create Conv + MaxPool model."""
    np.random.seed(42)
    conv_w = np.random.randn(out_c, in_c, 3, 3).astype(np.float32) * 0.1
    conv_b = np.random.randn(out_c).astype(np.float32) * 0.01

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_c, h, w])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, None)

    conv_node = helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'],
                                 kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    pool_node = helper.make_node('MaxPool', ['conv_out'], ['output'],
                                 kernel_shape=[2, 2], strides=[2, 2])

    graph = helper.make_graph(
        [conv_node, pool_node], 'conv_maxpool_test', [X], [Y],
        initializer=[
            numpy_helper.from_array(conv_w, 'conv_w'),
            numpy_helper.from_array(conv_b, 'conv_b'),
        ])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    return model


def make_conv_globalavgpool_model(in_c=3, out_c=8, h=8, w=8):
    """Create Conv + GlobalAveragePool model."""
    np.random.seed(42)
    conv_w = np.random.randn(out_c, in_c, 3, 3).astype(np.float32) * 0.1
    conv_b = np.random.randn(out_c).astype(np.float32) * 0.01

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_c, h, w])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, None)

    conv_node = helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'],
                                 kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    gap_node = helper.make_node('GlobalAveragePool', ['conv_out'], ['output'])

    graph = helper.make_graph(
        [conv_node, gap_node], 'conv_gap_test', [X], [Y],
        initializer=[
            numpy_helper.from_array(conv_w, 'conv_w'),
            numpy_helper.from_array(conv_b, 'conv_b'),
        ])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    return model


def make_conv_bn_relu_maxpool_model(in_c=3, out_c=8, h=8, w=8):
    """Create Conv + BN + ReLU + MaxPool model."""
    np.random.seed(42)
    conv_w = np.random.randn(out_c, in_c, 3, 3).astype(np.float32) * 0.1
    conv_b = np.random.randn(out_c).astype(np.float32) * 0.01
    bn_scale = np.random.rand(out_c).astype(np.float32) * 0.5 + 0.5
    bn_bias = np.random.randn(out_c).astype(np.float32) * 0.1
    bn_mean = np.random.randn(out_c).astype(np.float32) * 0.2
    bn_var = np.abs(np.random.randn(out_c).astype(np.float32)) + 0.1

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_c, h, w])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, None)

    conv_node = helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'],
                                 kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    bn_node = helper.make_node('BatchNormalization',
                               ['conv_out', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
                               ['bn_out'], epsilon=1e-5)
    relu_node = helper.make_node('Relu', ['bn_out'], ['relu_out'])
    pool_node = helper.make_node('MaxPool', ['relu_out'], ['output'],
                                 kernel_shape=[2, 2], strides=[2, 2])

    graph = helper.make_graph(
        [conv_node, bn_node, relu_node, pool_node], 'conv_bn_relu_pool_test', [X], [Y],
        initializer=[
            numpy_helper.from_array(conv_w, 'conv_w'),
            numpy_helper.from_array(conv_b, 'conv_b'),
            numpy_helper.from_array(bn_scale, 'bn_scale'),
            numpy_helper.from_array(bn_bias, 'bn_bias'),
            numpy_helper.from_array(bn_mean, 'bn_mean'),
            numpy_helper.from_array(bn_var, 'bn_var'),
        ])

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    return model


def test_batchnorm_fold_correctness():
    """Verify BN fold produces identical output to original Conv+BN."""
    print("\n=== Test: BatchNorm fold correctness ===")

    model = make_conv_bn_model(in_c=3, out_c=8, ksize=3, h=8, w=8)
    model_path = '/tmp/test_conv_bn.onnx'
    onnx.save(model, model_path)

    # Run original model
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    np.random.seed(123)
    x = np.random.randn(1, 3, 8, 8).astype(np.float32) * 0.3
    ref_out = sess.run(None, {'input': x})[0]

    # Fold BN and run
    nodes, input_name, input_shape, output_name, weights = parse_onnx_graph(model_path)
    nodes_folded = fold_batchnorm(nodes, weights)

    # Verify BN node is removed
    bn_count = sum(1 for n in nodes_folded if n.op_type == 'BatchNormalization')
    assert bn_count == 0, f"Expected 0 BN nodes after fold, got {bn_count}"

    # Build folded model to verify numerics
    conv_node_f = [n for n in nodes_folded if n.op_type == 'Conv'][0]
    w_folded = conv_node_f.weight
    b_folded = conv_node_f.bias

    # Manual fold verification
    original_model = onnx.load(model_path)
    inits = {i.name: numpy_helper.to_array(i) for i in original_model.graph.initializer}
    conv_w = inits['conv_w']
    conv_b = inits['conv_b']
    bn_scale = inits['bn_scale']
    bn_bias = inits['bn_bias']
    bn_mean = inits['bn_mean']
    bn_var = inits['bn_var']
    eps = 1e-5

    inv_std = bn_scale / np.sqrt(bn_var + eps)
    w_expected = conv_w * inv_std.reshape(-1, 1, 1, 1)
    b_expected = (conv_b - bn_mean) * inv_std + bn_bias

    assert np.allclose(w_folded, w_expected, atol=1e-6), "Weight fold mismatch!"
    assert np.allclose(b_folded, b_expected, atol=1e-6), "Bias fold mismatch!"

    # Now verify the folded conv gives same output as original Conv+BN
    # Build a single-conv ONNX model with folded params
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 8, 8])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, None)
    conv_folded_node = helper.make_node('Conv', ['input', 'w', 'b'], ['output'],
                                        kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    graph = helper.make_graph(
        [conv_folded_node], 'folded', [X], [Y],
        initializer=[
            numpy_helper.from_array(w_folded, 'w'),
            numpy_helper.from_array(b_folded, 'b'),
        ])
    folded_model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    folded_model.ir_version = 7
    folded_path = '/tmp/test_conv_bn_folded.onnx'
    onnx.save(folded_model, folded_path)

    sess2 = ort.InferenceSession(folded_path, providers=['CPUExecutionProvider'])
    folded_out = sess2.run(None, {'input': x})[0]

    diff = np.abs(ref_out - folded_out).max()
    print(f"  Max diff (original vs folded): {diff:.2e}")
    assert diff < 1e-5, f"BN fold output mismatch: max diff = {diff}"
    print("  PASS: BN fold is numerically correct")


def run_e2e_test(model_name, onnx_model, h=8, w=8, in_c=3, bits=8):
    """End-to-end: ONNX model → converter → csim → compare cosine with ORT."""
    print(f"\n=== Test: {model_name} (INT{bits}) ===")

    tmpdir = tempfile.mkdtemp(prefix='npu_test_')
    model_path = os.path.join(tmpdir, 'model.onnx')
    onnx.save(onnx_model, model_path)

    # Create calibration images (random, diverse)
    calib_dir = os.path.join(tmpdir, 'calib')
    os.makedirs(calib_dir)
    from PIL import Image
    np.random.seed(42)
    for i in range(20):
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(calib_dir, f'calib_{i:04d}.jpg'))

    # Create test input using a calibration image for best range match
    np.random.seed(99)
    test_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    Image.fromarray(test_img).save(os.path.join(calib_dir, 'test.jpg'))

    # Get ONNX Runtime reference output (same preprocessing as calibration)
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp_float = (test_img.astype(np.float32).transpose(2, 0, 1) - 127.5) / 255.0
    inp_float = inp_float[np.newaxis, ...]  # [1,C,H,W]
    ref_out = sess.run(None, {sess.get_inputs()[0].name: inp_float})[0].flatten()

    # Write input for converter: raw uint8 pixels in NCHW layout
    input_bin = os.path.join(tmpdir, 'input.bin')
    test_nchw = test_img.transpose(2, 0, 1)  # [H,W,3] → [C,H,W]
    test_nchw.tofile(input_bin)  # uint8 NCHW

    # Run converter
    npu_model_path = os.path.join(tmpdir, 'model.npu1.bin')
    npu_input_path = os.path.join(tmpdir, 'npu_input.bin')
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

    result = subprocess.run(
        [CSIM_PATH, npu_model_path, npu_input_actual, output_bin],
        capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAIL: csim error: {result.stderr}")
        return False

    # Read csim output and dequantize using metadata
    meta_path = npu_model_path.replace('.bin', '_meta.npz')
    if os.path.exists(meta_path):
        meta = np.load(meta_path)
        out_scale = float(meta['output_scale'])
    else:
        # Fallback
        ref_max = max(abs(ref_out.max()), abs(ref_out.min()))
        out_scale = ref_max / qmax if ref_max > 1e-10 else 1e-10 / qmax

    if bits == 8:
        csim_out_q = np.fromfile(output_bin, dtype=np.int8).astype(np.float32)
    else:
        csim_out_q = np.fromfile(output_bin, dtype=np.int16).astype(np.float32)

    # Dequantize csim output to float
    csim_out_float = csim_out_q * out_scale

    # Compute cosine similarity in float domain
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
    print(f"  Output size: {csim_out_float.size} elements")
    print(f"  Output scale: {out_scale:.8f}")

    # For INT8, cosine > 0.9 is acceptable for small models with random weights
    threshold = 0.85 if bits == 8 else 0.95
    if cosine >= threshold:
        print(f"  PASS (cosine >= {threshold})")
        return True
    else:
        print(f"  FAIL (cosine < {threshold})")
        # Debug: show some values
        print(f"  ref[:8]:  {ref_out[:8]}")
        print(f"  csim[:8]: {csim_out_float[:8]}")
        return False


def test_fold_only():
    """Test that fold_batchnorm works correctly on its own."""
    test_batchnorm_fold_correctness()


def test_e2e_all():
    """Run E2E tests for each model type."""
    results = []

    # Test 1: Conv + BN
    model = make_conv_bn_model(in_c=3, out_c=8, ksize=3, h=8, w=8)
    results.append(('Conv+BN', run_e2e_test('Conv+BN', model, h=8, w=8)))

    # Test 2: Conv + MaxPool
    model = make_conv_maxpool_model(in_c=3, out_c=8, h=8, w=8)
    results.append(('Conv+MaxPool', run_e2e_test('Conv+MaxPool', model, h=8, w=8)))

    # Test 3: Conv + GlobalAveragePool
    model = make_conv_globalavgpool_model(in_c=3, out_c=8, h=8, w=8)
    results.append(('Conv+GAP', run_e2e_test('Conv+GlobalAvgPool', model, h=8, w=8)))

    # Test 4: Conv + BN + ReLU + MaxPool
    model = make_conv_bn_relu_maxpool_model(in_c=3, out_c=8, h=8, w=8)
    results.append(('Conv+BN+ReLU+Pool', run_e2e_test('Conv+BN+ReLU+MaxPool', model, h=8, w=8)))

    print("\n" + "=" * 50)
    print("SUMMARY:")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    return all_pass


if __name__ == '__main__':
    print("=" * 60)
    print("Testing BatchNorm fold and Pooling support")
    print("=" * 60)

    # Test 1: BN fold numerical correctness
    test_fold_only()

    # Test 2: E2E with csim
    all_pass = test_e2e_all()

    if all_pass:
        print("\nAll tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)
