#!/usr/bin/env python3
"""
Phase 3: YOLO-Tiny-style Detection Model E2E Validation

Builds a YOLOv3-Tiny-inspired detection network using ONNX helpers:
  - Backbone: 6x (Conv+BN→ReLU + MaxPool) for progressive downsampling
  - Neck: Upsample(2x) + Concat for multi-scale feature fusion
  - Head: 1x1 Conv producing detection outputs at 2 scales

This tests detection-specific patterns:
  - Deep backbone with pooling (16x/32x downsampling)
  - Resize/Upsample for FPN-style feature fusion
  - Concat for merging multi-scale features
  - Multiple output scales (large objects + small objects)

Input: [1, 3, 416, 416] (classic YOLO input)
Outputs: [1, 18, 13, 13] + [1, 18, 26, 26] (2 scales, 3 anchors * (4+1+1) = 18)

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
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_converter import convert_model

CSIM_PATH = '/data/sam/open-npu/csim/npu_sim'


def make_conv_bn_relu(name, in_c, out_c, kernel=3, pad=1, stride=1):
    """Create Conv+BN+ReLU nodes and initializers.
    
    Returns (nodes, initializers, output_name)
    For simplicity, we pre-fold BN into Conv (just use Conv+ReLU with scaled weights).
    """
    np.random.seed(hash(name) % (2**31))
    
    # Conv weight and bias (BN already folded conceptually)
    w = (np.random.randn(out_c, in_c, kernel, kernel) * np.sqrt(2.0 / (in_c * kernel * kernel))).astype(np.float32)
    b = (np.random.randn(out_c) * 0.01).astype(np.float32)
    
    w_name = f'{name}_w'
    b_name = f'{name}_b'
    conv_out = f'{name}_conv'
    relu_out = f'{name}_out'
    
    conv = helper.make_node('Conv', [f'{name}_in', w_name, b_name], [conv_out],
                            kernel_shape=[kernel, kernel],
                            pads=[pad, pad, pad, pad],
                            strides=[stride, stride])
    relu = helper.make_node('Relu', [conv_out], [relu_out])
    
    inits = [
        numpy_helper.from_array(w, w_name),
        numpy_helper.from_array(b, b_name),
    ]
    return [conv, relu], inits, relu_out


def make_yolo_tiny_model(num_classes=1):
    """Build YOLOv3-Tiny-style detection network.
    
    Architecture (simplified for MCU):
      Input: [1, 3, 416, 416]
      
      Backbone:
        Conv1: 3→16, 3x3, relu → 416x416
        Pool1: 2x2 → 208x208
        Conv2: 16→32, 3x3, relu → 208x208
        Pool2: 2x2 → 104x104
        Conv3: 32→64, 3x3, relu → 104x104
        Pool3: 2x2 → 52x52
        Conv4: 64→128, 3x3, relu → 52x52
        Pool4: 2x2 → 26x26
        Conv5: 128→128, 3x3, relu → 26x26  ← branch point for FPN
        Pool5: 2x2 → 13x13
        Conv6: 128→256, 3x3, relu → 13x13
      
      Head (large scale, 13x13):
        Conv7: 256→18, 1x1 → 13x13 (output_large)
      
      Neck (upsample + concat):
        Conv8: 256→64, 1x1, relu → 13x13
        Resize: 2x nearest → 26x26
        Concat: [upsampled(64ch), Conv5_out(128ch)] → 192ch, 26x26
      
      Head (small scale, 26x26):
        Conv9: 192→64, 3x3, relu → 26x26
        Conv10: 64→18, 1x1 → 26x26 (output_small)
      
    Output channels: 3 anchors * (4 bbox + 1 obj + num_classes) = 3*(5+1) = 18
    """
    np.random.seed(2024)
    
    out_ch = 3 * (5 + num_classes)  # 18
    
    nodes = []
    inits = []
    
    # Input
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 416, 416])
    
    # --- Backbone ---
    # Conv1: 3→16
    n, ini, out = make_conv_bn_relu('conv1', 3, 16)
    # Patch input name
    n[0].input[0] = 'input'
    nodes.extend(n); inits.extend(ini)
    
    # Pool1
    pool1 = helper.make_node('MaxPool', [out], ['pool1_out'],
                             kernel_shape=[2, 2], strides=[2, 2])
    nodes.append(pool1)
    
    # Conv2: 16→32
    n, ini, out = make_conv_bn_relu('conv2', 16, 32)
    n[0].input[0] = 'pool1_out'
    nodes.extend(n); inits.extend(ini)
    
    # Pool2
    pool2 = helper.make_node('MaxPool', [out], ['pool2_out'],
                             kernel_shape=[2, 2], strides=[2, 2])
    nodes.append(pool2)
    
    # Conv3: 32→64
    n, ini, out = make_conv_bn_relu('conv3', 32, 64)
    n[0].input[0] = 'pool2_out'
    nodes.extend(n); inits.extend(ini)
    
    # Pool3
    pool3 = helper.make_node('MaxPool', [out], ['pool3_out'],
                             kernel_shape=[2, 2], strides=[2, 2])
    nodes.append(pool3)
    
    # Conv4: 64→128
    n, ini, out = make_conv_bn_relu('conv4', 64, 128)
    n[0].input[0] = 'pool3_out'
    nodes.extend(n); inits.extend(ini)
    
    # Pool4
    pool4 = helper.make_node('MaxPool', [out], ['pool4_out'],
                             kernel_shape=[2, 2], strides=[2, 2])
    nodes.append(pool4)
    
    # Conv5: 128→128 (branch point — this feeds into both scales)
    n, ini, out = make_conv_bn_relu('conv5', 128, 128)
    n[0].input[0] = 'pool4_out'
    nodes.extend(n); inits.extend(ini)
    conv5_out = out  # 26x26x128, used by FPN concat
    
    # Pool5
    pool5 = helper.make_node('MaxPool', [conv5_out], ['pool5_out'],
                             kernel_shape=[2, 2], strides=[2, 2])
    nodes.append(pool5)
    
    # Conv6: 128→256
    n, ini, out = make_conv_bn_relu('conv6', 128, 256)
    n[0].input[0] = 'pool5_out'
    nodes.extend(n); inits.extend(ini)
    conv6_out = out  # 13x13x256
    
    # --- Head: large scale (13x13) ---
    # Conv7: 256→18, 1x1 (no relu — raw detection output)
    conv7_w = np.random.randn(out_ch, 256, 1, 1).astype(np.float32) * 0.1
    conv7_b = np.random.randn(out_ch).astype(np.float32) * 0.01
    inits.append(numpy_helper.from_array(conv7_w, 'conv7_w'))
    inits.append(numpy_helper.from_array(conv7_b, 'conv7_b'))
    conv7 = helper.make_node('Conv', [conv6_out, 'conv7_w', 'conv7_b'], ['output_large'],
                             kernel_shape=[1, 1])
    nodes.append(conv7)
    
    # --- Neck: upsample path ---
    # Conv8: 256→64, 1x1, relu (channel reduction before upsample)
    n, ini, out = make_conv_bn_relu('conv8', 256, 64, kernel=1, pad=0)
    n[0].input[0] = conv6_out
    nodes.extend(n); inits.extend(ini)
    
    # Resize 2x nearest: 13x13 → 26x26
    roi = numpy_helper.from_array(np.array([], dtype=np.float32), 'resize_roi')
    scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), 'resize_scales')
    inits.extend([roi, scales])
    resize = helper.make_node('Resize', [out, 'resize_roi', 'resize_scales'], ['upsampled'],
                              mode='nearest')
    nodes.append(resize)
    
    # Concat: upsampled(64ch) + conv5_out(128ch) → 192ch at 26x26
    concat = helper.make_node('Concat', ['upsampled', conv5_out], ['concat_out'], axis=1)
    nodes.append(concat)
    
    # --- Head: small scale (26x26) ---
    # Conv9: 192→64, 3x3, relu
    n, ini, out = make_conv_bn_relu('conv9', 192, 64)
    n[0].input[0] = 'concat_out'
    nodes.extend(n); inits.extend(ini)
    
    # Conv10: 64→18, 1x1 (no relu — raw detection output)
    conv10_w = np.random.randn(out_ch, 64, 1, 1).astype(np.float32) * 0.1
    conv10_b = np.random.randn(out_ch).astype(np.float32) * 0.01
    inits.append(numpy_helper.from_array(conv10_w, 'conv10_w'))
    inits.append(numpy_helper.from_array(conv10_b, 'conv10_b'))
    conv10 = helper.make_node('Conv', [out, 'conv10_w', 'conv10_b'], ['output_small'],
                              kernel_shape=[1, 1])
    nodes.append(conv10)
    
    # Outputs
    Y_large = helper.make_tensor_value_info('output_large', TensorProto.FLOAT,
                                             [1, out_ch, 13, 13])
    Y_small = helper.make_tensor_value_info('output_small', TensorProto.FLOAT,
                                             [1, out_ch, 26, 26])
    
    graph = helper.make_graph(nodes, 'yolo_tiny', [X], [Y_large, Y_small],
                              initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    model = onnx.shape_inference.infer_shapes(model)
    return model


def run_single_output_e2e(model_name, model_path, calib_dir, test_img_path,
                          output_name, h=416, w=416, bits=8):
    """Convert and run csim for a single-output model variant."""
    tmpdir = tempfile.mkdtemp(prefix=f'npu_yolo_{output_name}_')
    
    # Extract sub-graph that produces only the target output
    model = onnx.load(model_path)
    extracted = onnx.utils.Extractor(model).extract_model(
        input_names=['input'],
        output_names=[output_name])
    
    single_model_path = os.path.join(tmpdir, 'model_single.onnx')
    onnx.save(extracted, single_model_path)
    
    # Prepare input
    test_img = Image.open(test_img_path).resize((w, h))
    arr = np.array(test_img).transpose(2, 0, 1).astype(np.uint8)
    input_bin = os.path.join(tmpdir, 'input.bin')
    arr.tofile(input_bin)
    
    # Convert
    npu_model = os.path.join(tmpdir, 'model.npu1.bin')
    try:
        convert_model(single_model_path, calib_dir, input_bin, npu_model,
                      input_format='int8-nchw', num_calib=20, bits=bits)
    except Exception as e:
        print(f"  FAIL: Converter error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Run csim
    output_bin = os.path.join(tmpdir, 'output.bin')
    npu_input = npu_model.replace('.bin', '_input.bin')
    
    if not os.path.exists(CSIM_PATH):
        print(f"  SKIP: csim not found")
        return None
    
    result = subprocess.run(
        [CSIM_PATH, npu_model, npu_input, output_bin],
        capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  FAIL: csim error (rc={result.returncode}):")
        print(f"    {result.stdout[-200:]}")
        print(f"    {result.stderr[-200:]}")
        return None
    
    # Read and dequantize
    meta = np.load(npu_model.replace('.bin', '_meta.npz'))
    out_scale = float(meta['output_scale'])
    
    if bits == 16:
        csim_q = np.fromfile(output_bin, dtype=np.int16).astype(np.float32)
    else:
        csim_q = np.fromfile(output_bin, dtype=np.int8).astype(np.float32)
    csim_float = csim_q * out_scale
    
    return csim_float


def main():
    print("=" * 60)
    print("Phase 3: YOLO-Tiny Detection Model E2E Validation")
    print("=" * 60)
    
    # Build model
    print("\n--- Building YOLO-Tiny model ---")
    model = make_yolo_tiny_model(num_classes=1)
    
    tmpdir = tempfile.mkdtemp(prefix='npu_yolo_test_')
    model_path = os.path.join(tmpdir, 'yolo_tiny.onnx')
    onnx.save(model, model_path)
    
    # Check model
    onnx.checker.check_model(model)
    print(f"  Model saved: {model_path}")
    print(f"  Nodes: {len(model.graph.node)}")
    
    # Create calibration images (416x416)
    calib_dir = os.path.join(tmpdir, 'calib')
    os.makedirs(calib_dir)
    np.random.seed(42)
    for i in range(20):
        img = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(calib_dir, f'calib_{i:04d}.jpg'))
    
    # Test image
    np.random.seed(99)
    test_img = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
    test_img_path = os.path.join(calib_dir, 'test.jpg')
    Image.fromarray(test_img).save(test_img_path)
    
    # ORT reference (both outputs)
    print("\n--- Running ORT reference ---")
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp_float = (test_img.astype(np.float32).transpose(2, 0, 1) - 127.5) / 255.0
    inp_float = inp_float[np.newaxis, ...]
    ort_outs = sess.run(None, {sess.get_inputs()[0].name: inp_float})
    ref_large = ort_outs[0].flatten()  # [1, 18, 13, 13]
    ref_small = ort_outs[1].flatten()  # [1, 18, 26, 26]
    print(f"  output_large: shape={ort_outs[0].shape}, range=[{ref_large.min():.4f}, {ref_large.max():.4f}]")
    print(f"  output_small: shape={ort_outs[1].shape}, range=[{ref_small.min():.4f}, {ref_small.max():.4f}]")
    
    # Run E2E for each output head separately
    # (Our converter supports single-output models)
    results = []
    
    print("\n--- Testing output_large (13x13, 32x downsample) ---")
    csim_large = run_single_output_e2e(
        'yolo_large', model_path, calib_dir, test_img_path, 'output_large')
    if csim_large is not None:
        if ref_large.size != csim_large.size:
            print(f"  Size mismatch: ORT={ref_large.size}, csim={csim_large.size}")
            results.append(('output_large (13x13)', False))
        else:
            dot = np.dot(ref_large, csim_large)
            na, nb = np.linalg.norm(ref_large), np.linalg.norm(csim_large)
            cos = dot / (na * nb) if na > 1e-10 and nb > 1e-10 else 0.0
            print(f"  Cosine similarity: {cos:.6f}")
            passed = cos >= 0.90
            print(f"  {'PASS' if passed else 'FAIL'} (threshold 0.90)")
            results.append(('output_large (13x13)', passed))
    else:
        results.append(('output_large (13x13)', False))
    
    print("\n--- Testing output_small (26x26, with Resize+Concat) ---")
    csim_small = run_single_output_e2e(
        'yolo_small', model_path, calib_dir, test_img_path, 'output_small')
    if csim_small is not None:
        if ref_small.size != csim_small.size:
            print(f"  Size mismatch: ORT={ref_small.size}, csim={csim_small.size}")
            results.append(('output_small (26x26, Resize+Concat)', False))
        else:
            dot = np.dot(ref_small, csim_small)
            na, nb = np.linalg.norm(ref_small), np.linalg.norm(csim_small)
            cos = dot / (na * nb) if na > 1e-10 and nb > 1e-10 else 0.0
            print(f"  Cosine similarity: {cos:.6f}")
            passed = cos >= 0.90
            print(f"  {'PASS' if passed else 'FAIL'} (threshold 0.90)")
            results.append(('output_small (26x26, Resize+Concat)', passed))
    else:
        results.append(('output_small (26x26, Resize+Concat)', False))
    
    # Summary
    print(f"\n{'='*60}")
    print("YOLO-Tiny E2E SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\nYOLO-Tiny Phase 3 validation PASSED!")
    else:
        print("\nSome tests FAILED!")
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
