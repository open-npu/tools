#!/usr/bin/env python3
"""
Open-NPU Mixed Precision (INT16act × INT8wgt) Quantization Verification

Simulates mixed precision inference: activations quantized to INT16,
weights quantized to INT8, with int32 accumulator.

Usage:
    python3 quantize_mixed.py

SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'm216p_snap_rgb.onnx')
INPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'debug.bin')
CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'a3_train_images')


def load_calibration_images(calib_dir, max_images=400):
    """Load calibration images as float32 NCHW [0, 1]."""
    files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.jpg')])[:max_images]
    images = []
    for f in files:
        img = np.array(Image.open(os.path.join(calib_dir, f))).astype(np.float32) / 255.0
        img_nchw = img.transpose(2, 0, 1)[np.newaxis, ...]
        images.append(img_nchw)
    return images


def collect_activation_ranges(model_path, images):
    """Run inference and collect per-tensor max-abs values."""
    import onnx
    from onnx import helper

    model = onnx.load(model_path)
    graph = model.graph

    shape_info = onnx.shape_inference.infer_shapes(model)

    existing_outputs = {o.name for o in graph.output}
    added = []
    for node in graph.node:
        if node.op_type in ('Conv', 'Relu', 'Add'):
            for out in node.output:
                if out not in existing_outputs:
                    for vi in shape_info.graph.value_info:
                        if vi.name == out:
                            graph.output.append(vi)
                            added.append(out)
                            break

    import tempfile
    tmp_fd = tempfile.NamedTemporaryFile(suffix='.onnx', prefix='npu_qmix_', delete=False)
    tmp_path = tmp_fd.name
    tmp_fd.close()
    onnx.save(model, tmp_path)

    sess = ort.InferenceSession(tmp_path)
    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    max_abs = {name: 0.0 for name in out_names}

    print(f"Collecting activation ranges from {len(images)} images...")
    for i, img in enumerate(images):
        results = sess.run(out_names, {input_name: img})
        for name, val in zip(out_names, results):
            ma = float(np.max(np.abs(val)))
            if ma > max_abs[name]:
                max_abs[name] = ma
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(images)}")

    os.unlink(tmp_path)

    input_max = max(float(np.max(np.abs(img))) for img in images)
    max_abs[input_name] = input_max

    print(f"  Done. Collected {len(max_abs)} tensor ranges.")
    return max_abs


def conv2d_int64(x, w, strides, pads, dilations, group):
    """Vectorized convolution using im2col with int64 accumulation."""
    n, ic, ih, iw = x.shape
    oc, ic_g, kh, kw = w.shape
    sh, sw = strides
    pt, pl, pb, pr = pads
    dh, dw = dilations

    if pt + pb + pl + pr > 0:
        x_pad = np.zeros((n, ic, ih + pt + pb, iw + pl + pr), dtype=np.int64)
        x_pad[:, :, pt:pt+ih, pl:pl+iw] = x
    else:
        x_pad = x

    ih_pad, iw_pad = x_pad.shape[2], x_pad.shape[3]
    oh = (ih_pad - dh * (kh - 1) - 1) // sh + 1
    ow = (iw_pad - dw * (kw - 1) - 1) // sw + 1

    oc_per_group = oc // group

    output = np.zeros((n, oc, oh, ow), dtype=np.int64)

    for g in range(group):
        x_g = x_pad[:, g*ic_g:(g+1)*ic_g, :, :]

        cols = np.zeros((n, ic_g * kh * kw, oh * ow), dtype=np.int64)
        col_idx = 0
        for c in range(ic_g):
            for fh in range(kh):
                for fw in range(kw):
                    h_start = fh * dh
                    w_start = fw * dw
                    h_indices = np.arange(oh) * sh + h_start
                    w_indices = np.arange(ow) * sw + w_start
                    patch = x_g[:, c, h_indices[:, None], w_indices[None, :]]
                    cols[:, col_idx, :] = patch.reshape(n, -1)
                    col_idx += 1

        w_g = w[g*oc_per_group:(g+1)*oc_per_group].reshape(oc_per_group, -1)

        for ni in range(n):
            out_g = w_g @ cols[ni]
            output[ni, g*oc_per_group:(g+1)*oc_per_group, :, :] = out_g.reshape(oc_per_group, oh, ow)

    return output


def simulate_mixed_precision(model_path, input_data, max_abs, act_bits=16, wgt_bits=8):
    """Simulate mixed precision: activations at act_bits, weights at wgt_bits.
    
    Accumulator uses int64 (simulating actual hardware behavior).
    Reports what happens with int32 accumulator for overflow analysis.
    """
    import onnx
    from onnx import numpy_helper

    act_qmax = (1 << (act_bits - 1)) - 1  # 32767 for INT16
    wgt_qmax = (1 << (wgt_bits - 1)) - 1  # 127 for INT8

    model = onnx.load(model_path)
    graph = model.graph

    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    input_name = graph.input[0].name

    # Quantize input with act_bits
    input_max = max_abs.get(input_name, np.max(np.abs(input_data)))
    input_scale = act_qmax / (input_max + 1e-10)
    input_q = np.clip(np.round(input_data * input_scale), -act_qmax, act_qmax).astype(np.int64)

    tensor_q = {input_name: (input_q, input_scale)}

    stats = {'overflow_count': 0, 'max_acc': 0, 'layers_processed': 0}
    int32_max = 2**31 - 1
    # Also check 34-bit accumulator
    int34_max = 2**33 - 1

    for node_idx, node in enumerate(graph.node):
        if node.op_type == 'Conv':
            x_name = node.input[0]
            w_name = node.input[1]
            b_name = node.input[2] if len(node.input) > 2 else None
            out_name = node.output[0]

            x_q, x_scale = tensor_q[x_name]
            w_fp = initializers[w_name]
            b_fp = initializers[b_name] if b_name else None

            # Quantize weights with wgt_bits (INT8)
            w_max = np.max(np.abs(w_fp))
            w_scale = wgt_qmax / (w_max + 1e-10)
            w_q = np.clip(np.round(w_fp * w_scale), -wgt_qmax, wgt_qmax).astype(np.int64)

            # Get attributes
            attrs = {a.name: a for a in node.attribute}
            group = attrs['group'].i if 'group' in attrs else 1
            strides = list(attrs['strides'].ints) if 'strides' in attrs else [1, 1]
            pads = list(attrs['pads'].ints) if 'pads' in attrs else [0, 0, 0, 0]
            dilations = list(attrs['dilations'].ints) if 'dilations' in attrs else [1, 1]

            # Compute conv with int64 accumulator
            output_q = conv2d_int64(x_q, w_q, strides, pads, dilations, group)

            # Check overflow stats
            max_val = int(np.max(np.abs(output_q)))
            if max_val > int32_max:
                stats['overflow_count'] += 1
            if max_val > stats['max_acc']:
                stats['max_acc'] = max_val

            # Dequantize accumulator to float
            combined_scale = x_scale * w_scale
            output_fp = output_q.astype(np.float64) / combined_scale

            # Add bias
            if b_fp is not None:
                output_fp += b_fp.reshape(1, -1, 1, 1)

            # Requantize output with act_bits (INT16 activation)
            if out_name in max_abs and max_abs[out_name] > 0:
                out_scale = act_qmax / max_abs[out_name]
            else:
                out_max = np.max(np.abs(output_fp))
                out_scale = act_qmax / (out_max + 1e-10)

            out_q = np.clip(np.round(output_fp * out_scale), -act_qmax, act_qmax).astype(np.int64)
            tensor_q[out_name] = (out_q, out_scale)

            stats['layers_processed'] += 1
            if stats['layers_processed'] % 10 == 0:
                print(f"  Layer {stats['layers_processed']}/51 ({node.name})")

        elif node.op_type == 'Relu':
            x_name = node.input[0]
            out_name = node.output[0]
            if x_name in tensor_q:
                x_q, x_scale = tensor_q[x_name]
                out_q = np.maximum(x_q, 0)
                tensor_q[out_name] = (out_q, x_scale)

        elif node.op_type == 'Add':
            a_name = node.input[0]
            b_name = node.input[1]
            out_name = node.output[0]

            a_q, a_scale = tensor_q[a_name]
            b_q, b_scale = tensor_q[b_name]

            a_fp = a_q.astype(np.float64) / a_scale
            b_fp = b_q.astype(np.float64) / b_scale
            out_fp = a_fp + b_fp

            if out_name in max_abs and max_abs[out_name] > 0:
                out_scale = act_qmax / max_abs[out_name]
            else:
                out_max = np.max(np.abs(out_fp))
                out_scale = act_qmax / (out_max + 1e-10)

            out_q = np.clip(np.round(out_fp * out_scale), -act_qmax, act_qmax).astype(np.int64)
            tensor_q[out_name] = (out_q, out_scale)

        elif node.op_type == 'Reshape':
            x_name = node.input[0]
            shape_name = node.input[1]
            out_name = node.output[0]
            if x_name in tensor_q:
                x_q, x_scale = tensor_q[x_name]
                if shape_name in initializers:
                    target_shape = tuple(initializers[shape_name].astype(int).tolist())
                else:
                    target_shape = (-1,)
                tensor_q[out_name] = (x_q.reshape(target_shape), x_scale)

        elif node.op_type == 'Constant':
            pass

    # Get final output
    final_name = graph.output[0].name
    if final_name in tensor_q:
        final_q, final_scale = tensor_q[final_name]
        final_fp = final_q.astype(np.float64) / final_scale
        return final_fp, stats
    else:
        print(f"ERROR: output '{final_name}' not found in quantized tensors")
        return None, stats


def main():
    print("=" * 60)
    print("Open-NPU Mixed Precision Quantization Verification")
    print("  Activation: INT16,  Weight: INT8")
    print("=" * 60)

    # Load calibration images
    images = load_calibration_images(CALIB_DIR, max_images=400)
    print(f"Loaded {len(images)} calibration images")

    # Collect activation ranges
    max_abs = collect_activation_ranges(MODEL_PATH, images)

    # Load test input
    data = np.fromfile(INPUT_PATH, dtype=np.uint8).reshape(1, 3, 224, 224)
    input_nchw = (data.astype(np.float32) - 127.5) / 255.0

    # FP32 reference
    print("\nRunning FP32 reference...")
    sess = ort.InferenceSession(MODEL_PATH)
    result_fp32 = sess.run(['output'], {'input': input_nchw})[0].flatten()
    print(f"  FP32 output: shape={result_fp32.shape}, norm={np.linalg.norm(result_fp32):.6f}")

    # Mixed precision: INT16act × INT8wgt
    print("\nRunning Mixed Precision (INT16act × INT8wgt)...")
    result_mixed, stats_mixed = simulate_mixed_precision(
        MODEL_PATH, input_nchw, max_abs, act_bits=16, wgt_bits=8)
    
    cos_mixed = l2_mixed = None
    if result_mixed is not None:
        result_mixed = result_mixed.flatten().astype(np.float32)
        cos_mixed = float(np.dot(result_mixed, result_fp32) / (
            np.linalg.norm(result_mixed) * np.linalg.norm(result_fp32) + 1e-10))
        l2_mixed = float(np.linalg.norm(result_mixed - result_fp32) / (np.linalg.norm(result_fp32) + 1e-10))
        print(f"  Cosine vs FP32: {cos_mixed:.6f}")
        print(f"  L2 rel error:   {l2_mixed:.6f}")
        print(f"  output[0:5]:    {result_mixed[:5]}")
        print(f"  INT32 overflow layers: {stats_mixed['overflow_count']}")
        print(f"  Max accumulator: {stats_mixed['max_acc']}")
        print(f"  Max acc / INT32_MAX: {stats_mixed['max_acc'] / (2**31-1):.2f}x")

    # Also run pure INT8 for comparison
    print("\nRunning Pure INT8 (INT8act × INT8wgt)...")
    result_int8, stats_int8 = simulate_mixed_precision(
        MODEL_PATH, input_nchw, max_abs, act_bits=8, wgt_bits=8)
    
    cos_int8 = l2_int8 = None
    if result_int8 is not None:
        result_int8 = result_int8.flatten().astype(np.float32)
        cos_int8 = float(np.dot(result_int8, result_fp32) / (
            np.linalg.norm(result_int8) * np.linalg.norm(result_fp32) + 1e-10))
        l2_int8 = float(np.linalg.norm(result_int8 - result_fp32) / (np.linalg.norm(result_fp32) + 1e-10))
        print(f"  Cosine vs FP32: {cos_int8:.6f}")
        print(f"  L2 rel error:   {l2_int8:.6f}")
        print(f"  INT32 overflow layers: {stats_int8['overflow_count']}")
        print(f"  Max accumulator: {stats_int8['max_acc']}")

    # Also run pure INT16 for comparison
    print("\nRunning Pure INT16 (INT16act × INT16wgt)...")
    result_int16, stats_int16 = simulate_mixed_precision(
        MODEL_PATH, input_nchw, max_abs, act_bits=16, wgt_bits=16)
    
    cos_int16 = l2_int16 = None
    if result_int16 is not None:
        result_int16 = result_int16.flatten().astype(np.float32)
        cos_int16 = float(np.dot(result_int16, result_fp32) / (
            np.linalg.norm(result_int16) * np.linalg.norm(result_fp32) + 1e-10))
        l2_int16 = float(np.linalg.norm(result_int16 - result_fp32) / (np.linalg.norm(result_fp32) + 1e-10))
        print(f"  Cosine vs FP32: {cos_int16:.6f}")
        print(f"  L2 rel error:   {l2_int16:.6f}")
        print(f"  INT32 overflow layers: {stats_int16['overflow_count']}")
        print(f"  Max accumulator: {stats_int16['max_acc']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Scheme':<28} {'Cosine':>10} {'L2_rel':>10} {'OVF_layers':>10} {'Max_acc':>14} {'Acc bits':>9}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*14} {'-'*9}")

    if cos_int8 is not None:
        acc_bits_8 = int(np.ceil(np.log2(stats_int8['max_acc'] + 1))) + 1
        print(f"  {'INT8act × INT8wgt':<28} {cos_int8:>10.6f} {l2_int8:>10.6f} {stats_int8['overflow_count']:>10} {stats_int8['max_acc']:>14} {acc_bits_8:>9}")

    if cos_mixed is not None:
        acc_bits_m = int(np.ceil(np.log2(stats_mixed['max_acc'] + 1))) + 1
        print(f"  {'INT16act × INT8wgt':<28} {cos_mixed:>10.6f} {l2_mixed:>10.6f} {stats_mixed['overflow_count']:>10} {stats_mixed['max_acc']:>14} {acc_bits_m:>9}")

    if cos_int16 is not None:
        acc_bits_16 = int(np.ceil(np.log2(stats_int16['max_acc'] + 1))) + 1
        print(f"  {'INT16act × INT16wgt':<28} {cos_int16:>10.6f} {l2_int16:>10.6f} {stats_int16['overflow_count']:>10} {stats_int16['max_acc']:>14} {acc_bits_16:>9}")

    print(f"  {'FP32 (reference)':<28} {'1.000000':>10} {'0.000000':>10} {'N/A':>10} {'N/A':>14} {'N/A':>9}")

    # Hardware recommendation
    print("\n" + "-" * 60)
    print("HARDWARE IMPLICATIONS:")
    if cos_mixed is not None:
        print(f"  INT16act×INT8wgt achieves cosine={cos_mixed:.6f}")
        print(f"  Required accumulator width: {acc_bits_m}-bit (signed)")
        if stats_mixed['overflow_count'] == 0:
            print(f"  32-bit accumulator: SAFE (no overflow)")
        else:
            print(f"  32-bit accumulator: OVERFLOW in {stats_mixed['overflow_count']} layers")
            print(f"  Recommended: {acc_bits_m}-bit accumulator")


if __name__ == '__main__':
    main()
