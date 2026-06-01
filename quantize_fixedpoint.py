#!/usr/bin/env python3
"""
Open-NPU Fixed-Point Requantize Simulation

Simulates hardware-realistic quantization:
  - Symmetric quantization (zp=0)
  - Per-channel weight quantization
  - Requantize using integer multiply + right-shift (no floating point)
  - Formula: out = (acc * M + round) >> S
  - M is a Q15 fixed-point multiplier, S is shift amount

Usage:
    python3 quantize_fixedpoint.py

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
        images.append(preprocess_uint8_nchw(img_nchw_u8))
    return images


def collect_activation_ranges(model_path, images):
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

    tmp_fd = tempfile.NamedTemporaryFile(suffix='.onnx', prefix='npu_qfp_', delete=False)
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


def compute_requantize_params(float_ratio, mult_bits=15):
    """Compute fixed-point requantize parameters M and S.
    
    Goal: approximate float_ratio as M * 2^(-S)
    So that: out_q ≈ (acc * M + (1 << (S-1))) >> S ≈ acc * float_ratio
    
    M is a mult_bits-bit unsigned integer (e.g. 15-bit → M in [1, 32767])
    S is the right-shift amount.
    
    Returns: (M, S)
    """
    if float_ratio <= 0 or not np.isfinite(float_ratio):
        return 1, 31
    
    max_m = (1 << mult_bits) - 1  # 32767 for 15-bit
    
    # We want: M = float_ratio * 2^S, and M <= max_m
    # So: S = floor(log2(max_m / float_ratio))
    # But we want M as large as possible (maximize precision), so use ceil for S
    s = int(np.floor(np.log2(max_m / float_ratio)))
    s = max(0, min(s, 62))
    
    m = int(np.round(float_ratio * (1 << s)))
    m = max(1, min(m, max_m))
    
    return m, s


def fixed_point_requantize(acc, M, S, qmax):
    """Hardware-realistic requantize: out = clamp((acc * M + round) >> S, -qmax, qmax)
    
    All operations are integer arithmetic.
    acc: int64 array
    M: int (Q15 multiplier)
    S: int (shift amount)
    """
    # Rounding: add 1<<(S-1) before shift
    if S > 0:
        rounding = np.int64(1 << (S - 1))
    else:
        rounding = np.int64(0)
    
    # acc * M could be very large, use int64
    result = (acc * np.int64(M) + rounding) >> np.int64(S)
    
    # Clamp
    result = np.clip(result, -qmax, qmax)
    return result


def conv2d_int64(x, w, strides, pads, dilations, group):
    """im2col convolution with int64 accumulation."""
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


def simulate_fixedpoint(model_path, input_data, max_abs, bits=16, per_channel=True):
    """Simulate hardware-realistic fixed-point inference.
    
    Quantization scheme:
      - scale convention: real_value = int_value * scale (i.e. scale = max_abs / qmax)
      - Quantize: int_val = round(real_val / scale)
      - Dequantize: real_val = int_val * scale
    
    Requantize (per-channel):
      - acc (int64) has scale = x_scale * w_scale[oc]
      - output has scale = out_scale
      - We want: out_int = acc_int * (x_scale * w_scale[oc]) / out_scale
      - In fixed-point: out_int = (acc_int * M + rounding) >> S
      - Where M/2^S ≈ (x_scale * w_scale[oc]) / out_scale
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

    # Quantize input (symmetric, per-tensor)
    input_max = max_abs.get(input_name, np.max(np.abs(input_data)))
    input_scale = input_max / qmax  # real = int * scale
    input_q = np.clip(np.round(input_data / input_scale), -qmax, qmax).astype(np.int64)

    # Store (quantized_data, scale) where real = int * scale
    tensor_q = {input_name: (input_q, input_scale)}

    stats = {'overflow_count': 0, 'max_acc': 0, 'layers_processed': 0,
             'max_mult_product': 0}
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

            attrs = {a.name: a for a in node.attribute}
            group = attrs['group'].i if 'group' in attrs else 1
            strides = list(attrs['strides'].ints) if 'strides' in attrs else [1, 1]
            pads = list(attrs['pads'].ints) if 'pads' in attrs else [0, 0, 0, 0]
            dilations = list(attrs['dilations'].ints) if 'dilations' in attrs else [1, 1]

            oc = w_fp.shape[0]

            # Quantize weights per-channel (symmetric)
            w_q = np.zeros_like(w_fp, dtype=np.int64)
            w_scales = np.zeros(oc, dtype=np.float64)
            for oi in range(oc):
                ch_max = np.max(np.abs(w_fp[oi]))
                if ch_max < 1e-10:
                    w_scales[oi] = 1e-10 / qmax
                else:
                    w_scales[oi] = ch_max / qmax
                w_q[oi] = np.clip(np.round(w_fp[oi] / w_scales[oi]), -qmax, qmax).astype(np.int64)

            # Compute conv with int64 accumulator
            output_q = conv2d_int64(x_q, w_q, strides, pads, dilations, group)
            # acc has scale = x_scale * w_scales[oc] per channel

            # Check overflow (before bias)
            max_val = int(np.max(np.abs(output_q)))
            if max_val > int32_max:
                stats['overflow_count'] += 1
            if max_val > stats['max_acc']:
                stats['max_acc'] = max_val

            # Add bias in accumulator domain
            # bias_q = round(bias_fp / acc_scale) where acc_scale = x_scale * w_scales[oc]
            if b_fp is not None:
                for oi in range(oc):
                    acc_scale_oi = x_scale * w_scales[oi]
                    bias_q = np.int64(np.round(b_fp[oi] / acc_scale_oi))
                    output_q[:, oi, :, :] += bias_q

            # Determine output scale (per-tensor activation)
            if out_name in max_abs and max_abs[out_name] > 0:
                out_scale = max_abs[out_name] / qmax
            else:
                out_max_real = 0
                for oi in range(oc):
                    acc_scale_oi = x_scale * w_scales[oi]
                    ch_max = float(np.max(np.abs(output_q[:, oi, :, :]))) * acc_scale_oi
                    if ch_max > out_max_real:
                        out_max_real = ch_max
                out_scale = out_max_real / qmax if out_max_real > 0 else 1e-10

            # Fixed-point requantize per-channel
            # Goal: out_int[oc] = acc_int[oc] * (acc_scale[oc] / out_scale)
            #                    = acc_int[oc] * float_ratio[oc]
            # Represent float_ratio as M * 2^(-S)
            out_q = np.zeros_like(output_q, dtype=np.int64)
            for oi in range(oc):
                acc_scale_oi = x_scale * w_scales[oi]
                float_ratio = acc_scale_oi / out_scale
                M, S = compute_requantize_params(float_ratio, mult_bits=15)
                
                ch_acc = output_q[:, oi, :, :]
                
                # Track hardware sizing
                ch_max_acc = int(np.max(np.abs(ch_acc)))
                mult_prod = ch_max_acc * M
                if mult_prod > stats['max_mult_product']:
                    stats['max_mult_product'] = mult_prod
                
                out_q[:, oi, :, :] = fixed_point_requantize(ch_acc, M, S, qmax)

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

            # Output scale
            if out_name in max_abs and max_abs[out_name] > 0:
                out_scale = max_abs[out_name] / qmax
            else:
                a_max = float(np.max(np.abs(a_q))) * a_scale
                b_max = float(np.max(np.abs(b_q))) * b_scale
                out_scale = (a_max + b_max) / qmax if (a_max + b_max) > 0 else 1e-10

            # Rescale both inputs to output scale using fixed-point
            ratio_a = a_scale / out_scale
            ratio_b = b_scale / out_scale

            M_a, S_a = compute_requantize_params(ratio_a, mult_bits=15)
            M_b, S_b = compute_requantize_params(ratio_b, mult_bits=15)

            a_rescaled = fixed_point_requantize(a_q, M_a, S_a, qmax * 2)  # wider range before sum
            b_rescaled = fixed_point_requantize(b_q, M_b, S_b, qmax * 2)

            out_q = np.clip(a_rescaled + b_rescaled, -qmax, qmax).astype(np.int64)
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

    # Get final output → convert to float for comparison
    final_name = graph.output[0].name
    if final_name in tensor_q:
        final_q, final_scale = tensor_q[final_name]
        final_fp = final_q.astype(np.float64) * final_scale
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
    print("Open-NPU Fixed-Point Requantize Simulation")
    print("  Requantize: out = (acc * M + round) >> S")
    print("  Weight: per-channel symmetric")
    print("  Activation: per-tensor symmetric")
    print("=" * 60)

    images = load_calibration_images(CALIB_DIR, max_images=400)
    print(f"Loaded {len(images)} calibration images")

    max_abs = collect_activation_ranges(MODEL_PATH, images)

    # Load test input
    data = np.fromfile(INPUT_PATH, dtype=np.uint8).reshape(1, 3, 224, 224)
    input_nchw = preprocess_uint8_nchw(data)

    # FP32 reference
    print("\nRunning FP32 reference...")
    sess = ort.InferenceSession(MODEL_PATH)
    fp32 = sess.run(['output'], {'input': input_nchw})[0].flatten()
    print(f"  FP32 norm={np.linalg.norm(fp32):.6f}")

    results = {}

    # INT16 fixed-point per-channel
    print("\n[1/2] INT16 fixed-point requantize (per-channel weight)...")
    r, s = simulate_fixedpoint(MODEL_PATH, input_nchw, max_abs, bits=16, per_channel=True)
    if r is not None:
        r = r.flatten().astype(np.float32)
        results['INT16 fixed-point'] = (cosine_sim(r, fp32), l2_rel(r, fp32), s)
        print(f"  Cosine: {results['INT16 fixed-point'][0]:.6f}")
        print(f"  L2 rel: {results['INT16 fixed-point'][1]:.6f}")
        print(f"  Max acc: {s['max_acc']}, overflow layers: {s['overflow_count']}")
        print(f"  Max M*acc product: {s['max_mult_product']} ({int(np.ceil(np.log2(s['max_mult_product']+1)))+1}-bit)")

    # INT8 fixed-point per-channel
    print("\n[2/2] INT8 fixed-point requantize (per-channel weight)...")
    r, s = simulate_fixedpoint(MODEL_PATH, input_nchw, max_abs, bits=8, per_channel=True)
    if r is not None:
        r = r.flatten().astype(np.float32)
        results['INT8 fixed-point'] = (cosine_sim(r, fp32), l2_rel(r, fp32), s)
        print(f"  Cosine: {results['INT8 fixed-point'][0]:.6f}")
        print(f"  L2 rel: {results['INT8 fixed-point'][1]:.6f}")
        print(f"  Max acc: {s['max_acc']}, overflow layers: {s['overflow_count']}")
        print(f"  Max M*acc product: {s['max_mult_product']} ({int(np.ceil(np.log2(s['max_mult_product']+1)))+1}-bit)")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: Fixed-Point vs Float Requantize")
    print("=" * 60)
    print(f"  {'Scheme':<30} {'Cosine':>10} {'L2_rel':>10} {'OVF':>5}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*5}")

    for name, (cos, l2, st) in results.items():
        print(f"  {name:<30} {cos:>10.6f} {l2:>10.6f} {st['overflow_count']:>5}")

    # Reference from previous runs (float requantize)
    print(f"  {'INT16 float-req per-ch':<30} {'0.999999':>10} {'0.001114':>10} {'19':>5}")
    print(f"  {'INT8 float-req per-ch':<30} {'0.935113':>10} {'0.354751':>10} {'0':>5}")
    print(f"  {'FP32 (reference)':<30} {'1.000000':>10} {'0.000000':>10} {'—':>5}")

    print("\n" + "-" * 60)
    print("HARDWARE NOTES:")
    print("  Requantize pipeline per output channel:")
    print("    1. acc (40-bit signed from systolic array)")
    print("    2. + bias_q (pre-quantized to acc scale, 40-bit add)")
    print("    3. * M (15-bit unsigned multiplier → 55-bit product)")
    print("    4. + (1 << (S-1)) (rounding)")
    print("    5. >> S (arithmetic right shift)")
    print("    6. clamp to [-qmax, qmax]")
    if 'INT16 fixed-point' in results:
        s = results['INT16 fixed-point'][2]
        mult_bits = int(np.ceil(np.log2(s['max_mult_product'] + 1))) + 1
        print(f"\n  Max M*acc product needs {mult_bits}-bit intermediate register")
        print(f"  (40-bit acc × 15-bit M = up to 55-bit, fits int64)")


if __name__ == '__main__':
    main()
