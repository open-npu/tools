#!/usr/bin/env python3
"""
Test Flatten/Reshape passthrough and Gemm→FC conversion in onnx_converter.py

Creates ONNX models with:
  1. Conv + GlobalAvgPool + Flatten + Gemm (typical classification head)
  2. Conv + Reshape + Gemm (reshape as passthrough)
  3. Gemm + ReLU (FC with fused activation)

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


def make_conv_gap_flatten_gemm_model(in_c=3, mid_c=16, out_c=10, h=8, w=8):
    """Conv 3x3 → GlobalAvgPool → Flatten → Gemm (classification head)."""
    np.random.seed(42)

    conv_w = np.random.randn(mid_c, in_c, 3, 3).astype(np.float32) * 0.1
    conv_b = np.random.randn(mid_c).astype(np.float32) * 0.01
    gemm_w = np.random.randn(out_c, mid_c).astype(np.float32) * 0.1
    gemm_b = np.random.randn(out_c).astype(np.float32) * 0.01

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_c, h, w])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, out_c])

    conv_node = helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'],
                                  kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    relu_node = helper.make_node('Relu', ['conv_out'], ['relu_out'])
    gap_node = helper.make_node('GlobalAveragePool', ['relu_out'], ['gap_out'])
    flatten_node = helper.make_node('Flatten', ['gap_out'], ['flat_out'], axis=1)
    gemm_node = helper.make_node('Gemm', ['flat_out', 'gemm_w', 'gemm_b'], ['output'],
                                  alpha=1.0, beta=1.0, transB=1)

    graph = helper.make_graph(
        [conv_node, relu_node, gap_node, flatten_node, gemm_node],
        'conv_gap_flatten_gemm',
        [X], [Y],
        initializer=[
            numpy_helper.from_array(conv_w, 'conv_w'),
            numpy_helper.from_array(conv_b, 'conv_b'),
            numpy_helper.from_array(gemm_w, 'gemm_w'),
            numpy_helper.from_array(gemm_b, 'gemm_b'),
        ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    model = onnx.shape_inference.infer_shapes(model)
    return model


def make_conv_reshape_gemm_model(in_c=3, mid_c=8, out_c=4, h=4, w=4):
    """Conv 1x1 → Reshape → Gemm (reshape flattens spatial dims)."""
    np.random.seed(7)

    conv_w = np.random.randn(mid_c, in_c, 1, 1).astype(np.float32) * 0.1
    conv_b = np.random.randn(mid_c).astype(np.float32) * 0.01
    # After conv: [1, mid_c, h, w] → reshape to [1, mid_c * h * w] → gemm
    flat_dim = mid_c * h * w
    gemm_w = np.random.randn(out_c, flat_dim).astype(np.float32) * 0.05
    gemm_b = np.random.randn(out_c).astype(np.float32) * 0.01
    reshape_shape = np.array([1, flat_dim], dtype=np.int64)

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_c, h, w])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, out_c])

    conv_node = helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'],
                                  kernel_shape=[1, 1])
    reshape_node = helper.make_node('Reshape', ['conv_out', 'shape'], ['reshape_out'])
    gemm_node = helper.make_node('Gemm', ['reshape_out', 'gemm_w', 'gemm_b'], ['output'],
                                  alpha=1.0, beta=1.0, transB=1)

    graph = helper.make_graph(
        [conv_node, reshape_node, gemm_node],
        'conv_reshape_gemm',
        [X], [Y],
        initializer=[
            numpy_helper.from_array(conv_w, 'conv_w'),
            numpy_helper.from_array(conv_b, 'conv_b'),
            numpy_helper.from_array(gemm_w, 'gemm_w'),
            numpy_helper.from_array(gemm_b, 'gemm_b'),
            numpy_helper.from_array(reshape_shape, 'shape'),
        ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    model = onnx.shape_inference.infer_shapes(model)
    return model


def make_gemm_relu_model(in_features=48, out_features=10, h=4, w=4, in_c=3):
    """Conv 1x1 → Flatten → Gemm + ReLU (FC with fused activation)."""
    np.random.seed(88)

    mid_c = in_features // (h * w)  # e.g. 48/(4*4) = 3
    conv_w = np.random.randn(mid_c, in_c, 1, 1).astype(np.float32) * 0.1
    conv_b = np.random.randn(mid_c).astype(np.float32) * 0.01
    gemm_w = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
    gemm_b = np.random.randn(out_features).astype(np.float32) * 0.01

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, in_c, h, w])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, None)

    conv_node = helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'],
                                  kernel_shape=[1, 1])
    flatten_node = helper.make_node('Flatten', ['conv_out'], ['flat_out'], axis=1)
    gemm_node = helper.make_node('Gemm', ['flat_out', 'gemm_w', 'gemm_b'], ['gemm_out'],
                                  alpha=1.0, beta=1.0, transB=1)
    relu_node = helper.make_node('Relu', ['gemm_out'], ['output'])

    graph = helper.make_graph(
        [conv_node, flatten_node, gemm_node, relu_node],
        'conv_gemm_relu',
        [X], [Y],
        initializer=[
            numpy_helper.from_array(conv_w, 'conv_w'),
            numpy_helper.from_array(conv_b, 'conv_b'),
            numpy_helper.from_array(gemm_w, 'gemm_w'),
            numpy_helper.from_array(gemm_b, 'gemm_b'),
        ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    model = onnx.shape_inference.infer_shapes(model)
    return model


def run_e2e_test(model_name, onnx_model, h=8, w=8, in_c=3, bits=8):
    """End-to-end: ONNX model → converter → csim → compare cosine with ORT."""
    print(f"\n{'='*60}")
    print(f"Test: {model_name} (INT{bits})")
    print(f"{'='*60}")

    tmpdir = tempfile.mkdtemp(prefix='npu_fc_test_')
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

    # Get ORT reference output
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp_float = (test_img.astype(np.float32).transpose(2, 0, 1) - 127.5) / 255.0
    inp_float = inp_float[np.newaxis, ...]  # [1,C,H,W]
    ref_out = sess.run(None, {sess.get_inputs()[0].name: inp_float})[0].flatten()
    print(f"  ORT output: {ref_out.shape}, range=[{ref_out.min():.4f}, {ref_out.max():.4f}]")

    # Write input for converter
    input_bin = os.path.join(tmpdir, 'input.bin')
    test_nchw = test_img.transpose(2, 0, 1)  # [C,H,W]
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
        print(f"  (Converter succeeded, csim test skipped)")
        return True

    result = subprocess.run(
        [CSIM_PATH, npu_model_path, npu_input_actual, output_bin],
        capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAIL: csim error (rc={result.returncode}):")
        print(f"    stdout: {result.stdout[:200]}")
        print(f"    stderr: {result.stderr[:200]}")
        return False

    # Read csim output and dequantize
    meta_path = npu_model_path.replace('.bin', '_meta.npz')
    meta = np.load(meta_path)
    out_scale = float(meta['output_scale'])

    qmax_val = 127 if bits == 8 else 32767
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
    print(f"  Output scale: {out_scale:.8f}")

    threshold = 0.85 if bits == 8 else 0.95
    if cosine >= threshold:
        print(f"  PASS (cosine >= {threshold})")
        return True
    else:
        print(f"  FAIL (cosine < {threshold})")
        print(f"    ref[:8]  = {ref_out[:8]}")
        print(f"    csim[:8] = {csim_out_float[:8]}")
        return False


def test_flatten_passthrough():
    """Test that Flatten/Reshape are correctly treated as passthrough ops."""
    print("\n=== Test: Flatten/Reshape Passthrough Logic ===")

    from onnx_converter import parse_onnx_graph, fuse_graph, fold_batchnorm

    # Use the conv_gap_flatten_gemm model
    model = make_conv_gap_flatten_gemm_model()
    tmpdir = tempfile.mkdtemp(prefix='npu_pt_test_')
    model_path = os.path.join(tmpdir, 'model.onnx')
    onnx.save(model, model_path)

    nodes, input_name, input_shape, output_name, weights = parse_onnx_graph(model_path)
    nodes = fold_batchnorm(nodes, weights)
    fused_ops, passthrough_map = fuse_graph(nodes, weights)

    # Check passthrough map has Flatten entry
    assert len(passthrough_map) > 0, "Expected passthrough_map to have Flatten entry"
    print(f"  Passthrough map: {len(passthrough_map)} entries")
    for out_name, in_name in passthrough_map.items():
        print(f"    {out_name} → {in_name}")

    # Check fused ops: should have Conv, Pool, FC (no Flatten)
    op_types = [op['type'] for op in fused_ops]
    print(f"  Fused op types: {op_types}")
    assert 'fc' in op_types, f"Expected 'fc' in fused ops, got: {op_types}"
    assert 'pool' in op_types, f"Expected 'pool' in fused ops, got: {op_types}"

    # Verify no Flatten/Reshape appears in fused ops
    for op in fused_ops:
        assert op['node'].op_type not in ('Flatten', 'Reshape', 'Squeeze', 'Unsqueeze'), \
            f"Shape-only op {op['node'].op_type} should not appear in fused ops"

    print("  PASS: Flatten correctly skipped, FC detected")
    return True


def test_e2e_all():
    """Run all E2E tests."""
    results = []

    # Test 0: passthrough logic
    results.append(('Flatten passthrough', test_flatten_passthrough()))

    # Test 1: Conv + GAP + Flatten + Gemm (main test case)
    model1 = make_conv_gap_flatten_gemm_model()
    results.append(('Conv+GAP+Flatten+Gemm', run_e2e_test(
        'Conv+GAP+Flatten+Gemm', model1, h=8, w=8, in_c=3)))

    # Test 2: Conv + Reshape + Gemm (reshape as passthrough)
    model2 = make_conv_reshape_gemm_model()
    results.append(('Conv+Reshape+Gemm', run_e2e_test(
        'Conv+Reshape+Gemm', model2, h=4, w=4, in_c=3)))

    # Test 3: Conv + Flatten + Gemm + ReLU (FC with activation)
    model3 = make_gemm_relu_model(in_features=48, out_features=10, h=4, w=4, in_c=3)
    results.append(('Conv+Flatten+Gemm+ReLU', run_e2e_test(
        'Conv+Flatten+Gemm+ReLU', model3, h=4, w=4, in_c=3)))

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
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == '__main__':
    test_e2e_all()
