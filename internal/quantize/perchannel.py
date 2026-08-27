#!/usr/bin/env python3
"""
Open-NPU Per-Channel Quantization Precision Verification

Compares per-tensor vs per-channel weight quantization for INT8 and INT16.
Activations always use per-tensor quantization (hardware-friendly).

Usage:
    python3 quantize_perchannel.py

SPDX-License-Identifier: Apache-2.0
"""

import os
import tempfile
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'm216p_snap_rgb.onnx')
INPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'debug.bin')
CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'a3_train_images')


def preprocess_uint8_nchw(raw_u8):
    """Canonical input preprocessing: (uint8 - 127.5) / 255.0."""
    return (raw_u8.astype(np.float32) - 127.5) / 255.0


def load_calibration_images(calib_dir, max_images=400):
    """Load calibration images as float32 NCHW using canonical preprocessing."""
    files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.jpg')])[:max_images]
    images = []
    for f in files:
        img = np.array(Image.open(os.path.join(calib_dir, f)).convert('RGB'), dtype=np.uint8)
        img_nchw_u8 = img.transpose(2, 0, 1)[np.newaxis, ...]
        img_nchw = preprocess_uint8_nchw(img_nchw_u8)
        images.append(img_nchw)
    return images


def collect_activation_ranges(model_path, images):
    """Run inference and collect per-tensor max-abs values."""
    import onnx

    model = onnx.load(model_path)
    graph = model.graph

    shape_info = onnx.shape_inference.infer_shapes(model)

    existing_outputs = {o.name for o in graph.output}
    for node in graph.node:
        if node.op_type in ('Conv', 'Relu', 'Add'):
            for out in node.output:
                if out not in existing_outputs:
                    for vi in shape_info.graph.value_info:
                        if vi.name == out:
                            graph.output.append(vi)
                            break

    tmp_fd = tempfile.NamedTemporaryFile(suffix='.onnx', prefix='npu_qpc_', delete=False)
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


def simulate_quantized(model_path, input_data, max_abs, bits=16, per_channel_weight=False):
    """Simulate quantized inference.
    
    Args:
        bits: quantization bit width (8 or 16)
        per_channel_weight: if True, each output channel has its own weight scale
    """
    import onnx
    from onnx import numpy_helper

    qmax = (1 << (bits - 1)) - 1

    model = onnx.load(model_path)
    graph = model.graph

    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    input_name = graph.input[0].name

    # Quantize input (always per-tensor)
    input_max = max_abs.get(input_name, np.max(np.abs(input_data)))
    input_scale = qmax / (input_max + 1e-10)
    input_q = np.clip(np.round(input_data * input_scale), -qmax, qmax).astype(np.int64)

    tensor_q = {input_name: (input_q, input_scale)}

    stats = {'overflow_count': 0, 'max_acc': 0, 'layers_processed': 0}
    int32_max = 2**31 - 1

    for node in graph.node:
        if node.op_type == 'Conv':
            x_name = node.input[0]
            w_name = node.input[1]
            b_name = node.input[2] if len(node.input) > 2 else None
            out_name = node.output[0]

            x_q, x_scale = tensor_q[x_name]
            w_fp = initializers[w_name]
            b_fp = initializers[b_name] if b_name else None

            # Get conv attributes
            attrs = {a.name: a for a in node.attribute}
            group = attrs['group'].i if 'group' in attrs else 1
            strides = list(attrs['strides'].ints) if 'strides' in attrs else [1, 1]
            pads = list(attrs['pads'].ints) if 'pads' in attrs else [0, 0, 0, 0]
            dilations = list(attrs['dilations'].ints) if 'dilations' in attrs else [1, 1]

            oc = w_fp.shape[0]

            if per_channel_weight:
                # Per-channel: each output channel has its own scale
                # w_fp shape: [OC, IC/group, KH, KW]
                w_q = np.zeros_like(w_fp, dtype=np.int64)
                w_scales = np.zeros(oc, dtype=np.float64)
                for oi in range(oc):
                    ch_max = np.max(np.abs(w_fp[oi]))
                    if ch_max < 1e-10:
                        w_scales[oi] = 1.0
                    else:
                        w_scales[oi] = qmax / ch_max
                    w_q[oi] = np.clip(np.round(w_fp[oi] * w_scales[oi]), -qmax, qmax).astype(np.int64)
            else:
                # Per-tensor: all channels share one scale
                w_max = np.max(np.abs(w_fp))
                w_scale_single = qmax / (w_max + 1e-10)
                w_q = np.clip(np.round(w_fp * w_scale_single), -qmax, qmax).astype(np.int64)
                w_scales = np.full(oc, w_scale_single)

            # Compute conv with int64 accumulator
            output_q = conv2d_int64(x_q, w_q, strides, pads, dilations, group)

            # Check overflow
            max_val = int(np.max(np.abs(output_q)))
            if max_val > int32_max:
                stats['overflow_count'] += 1
            if max_val > stats['max_acc']:
                stats['max_acc'] = max_val

            # Dequantize per-channel
            # output_q[n, oc, oh, ow], each oc has combined_scale = x_scale * w_scales[oc]
            output_fp = np.zeros_like(output_q, dtype=np.float64)
            for oi in range(oc):
                combined_scale = x_scale * w_scales[oi]
                output_fp[:, oi, :, :] = output_q[:, oi, :, :].astype(np.float64) / combined_scale

            # Add bias
            if b_fp is not None:
                output_fp += b_fp.reshape(1, -1, 1, 1)

            # Requantize output (per-tensor activation)
            if out_name in max_abs and max_abs[out_name] > 0:
                out_scale = qmax / max_abs[out_name]
            else:
                out_max = np.max(np.abs(output_fp))
                out_scale = qmax / (out_max + 1e-10)

            out_q = np.clip(np.round(output_fp * out_scale), -qmax, qmax).astype(np.int64)
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
                out_scale = qmax / max_abs[out_name]
            else:
                out_max = np.max(np.abs(out_fp))
                out_scale = qmax / (out_max + 1e-10)

            out_q = np.clip(np.round(out_fp * out_scale), -qmax, qmax).astype(np.int64)
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
        print(f"ERROR: output '{final_name}' not found")
        return None, stats


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def l2_rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 60)
    print("Open-NPU Per-Channel vs Per-Tensor Quantization")
    print("=" * 60)

    images = load_calibration_images(CALIB_DIR, max_images=400)
    print(f"Loaded {len(images)} calibration images")

    max_abs = collect_activation_ranges(MODEL_PATH, images)

    # Load test input (canonical format: uint8 NCHW)
    data = np.fromfile(INPUT_PATH, dtype=np.uint8).reshape(1, 3, 224, 224)
    input_nchw = preprocess_uint8_nchw(data)

    # FP32 reference
    print("\nRunning FP32 reference...")
    sess = ort.InferenceSession(MODEL_PATH)
    fp32 = sess.run(['output'], {'input': input_nchw})[0].flatten()
    print(f"  FP32 norm={np.linalg.norm(fp32):.6f}")

    results = {}

    # INT8 per-tensor weight
    print("\n[1/4] INT8 per-tensor weight...")
    r, s = simulate_quantized(MODEL_PATH, input_nchw, max_abs, bits=8, per_channel_weight=False)
    r = r.flatten().astype(np.float32)
    results['INT8 per-tensor'] = (cosine_sim(r, fp32), l2_rel(r, fp32), s)

    # INT8 per-channel weight
    print("\n[2/4] INT8 per-channel weight...")
    r, s = simulate_quantized(MODEL_PATH, input_nchw, max_abs, bits=8, per_channel_weight=True)
    r = r.flatten().astype(np.float32)
    results['INT8 per-channel'] = (cosine_sim(r, fp32), l2_rel(r, fp32), s)

    # INT16 per-tensor weight
    print("\n[3/4] INT16 per-tensor weight...")
    r, s = simulate_quantized(MODEL_PATH, input_nchw, max_abs, bits=16, per_channel_weight=False)
    r = r.flatten().astype(np.float32)
    results['INT16 per-tensor'] = (cosine_sim(r, fp32), l2_rel(r, fp32), s)

    # INT16 per-channel weight
    print("\n[4/4] INT16 per-channel weight...")
    r, s = simulate_quantized(MODEL_PATH, input_nchw, max_abs, bits=16, per_channel_weight=True)
    r = r.flatten().astype(np.float32)
    results['INT16 per-channel'] = (cosine_sim(r, fp32), l2_rel(r, fp32), s)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Scheme':<25} {'Cosine':>10} {'L2_rel':>10} {'OVF':>5} {'Max_acc':>14}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*5} {'-'*14}")

    for name, (cos, l2, st) in results.items():
        print(f"  {name:<25} {cos:>10.6f} {l2:>10.6f} {st['overflow_count']:>5} {st['max_acc']:>14}")

    print(f"  {'FP32 (reference)':<25} {'1.000000':>10} {'0.000000':>10} {'—':>5} {'—':>14}")

    # Per-channel improvement
    print("\n" + "-" * 60)
    print("PER-CHANNEL IMPROVEMENT:")
    cos_8t = results['INT8 per-tensor'][0]
    cos_8c = results['INT8 per-channel'][0]
    cos_16t = results['INT16 per-tensor'][0]
    cos_16c = results['INT16 per-channel'][0]
    print(f"  INT8:  per-tensor={cos_8t:.6f} → per-channel={cos_8c:.6f}  (+{cos_8c-cos_8t:.6f})")
    print(f"  INT16: per-tensor={cos_16t:.6f} → per-channel={cos_16c:.6f}  (+{cos_16c-cos_16t:.6f})")


if __name__ == '__main__':
    main()
