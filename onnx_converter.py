#!/usr/bin/env python3
"""
Open-NPU ONNX → NPU1 Converter with PTQ (Post-Training Quantization)

Converts a float32 ONNX model to NPU1 binary format using per-channel
weight quantization and per-tensor activation quantization with calibration data.

Usage:
  python3 onnx_converter.py --model MODEL.onnx --calib CALIB_DIR --output model.npu1.bin \
      --input INPUT.bin --input-format int8-nchw

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import os
import sys
import glob
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
import onnxruntime as ort
from PIL import Image

# Import our model_packer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_packer import (
    LayerConfig, PerChannelParam, AddParam, pack_model,
    OP_CONV2D, OP_DW_CONV, OP_FC, OP_POOLING, OP_ELTWISE_ADD, OP_CONCAT,
    POST_BIAS_EN, POST_ZP_EN, POST_RELU_EN, POST_RELU6_EN, POST_INT16_OUT,
    PPU_MODE_CONV_REQ, PPU_MODE_ADD, PPU_MODE_RELU_ONLY, PPU_MODE_PASSTHROUGH,
)


# ─── Utility ───

def preprocess_image_from_file(path, input_shape):
    """Load and preprocess calibration image (resize to HxW, normalize to float32)."""
    _, c, h, w = input_shape
    img = Image.open(path).convert('RGB')
    img = img.resize((w, h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)  # [H,W,3]
    arr = arr.transpose(2, 0, 1)  # [3,H,W]
    # Normalize: (pixel - 127.5) / 255  → range ~ [-0.5, 0.5]
    arr = (arr - 127.5) / 255.0
    return arr[np.newaxis, ...]  # [1,C,H,W]


def preprocess_int8_input(path, input_shape):
    """Load debug.bin (INT8 NCHW) and convert to float32 for ONNX Runtime.
    Transform: (int8_value - 127.5) / 255
    """
    _, c, h, w = input_shape
    data = np.fromfile(path, dtype=np.int8).reshape(1, c, h, w)
    # User specified: float = (int8_val - 127.5) / 255
    return (data.astype(np.float32) - 127.5) / 255.0


# ─── ONNX Graph Parser ───

class OnnxNode:
    """Parsed ONNX node with resolved attributes."""
    def __init__(self, node, weights, shape_map):
        self.op_type = node.op_type
        self.name = node.name or node.output[0]
        self.inputs = list(node.input)
        self.outputs = list(node.output)
        self.attrs = {}

        for a in node.attribute:
            if a.type == onnx.AttributeProto.INT:
                self.attrs[a.name] = a.i
            elif a.type == onnx.AttributeProto.INTS:
                self.attrs[a.name] = list(a.ints)
            elif a.type == onnx.AttributeProto.FLOAT:
                self.attrs[a.name] = a.f
            elif a.type == onnx.AttributeProto.FLOATS:
                self.attrs[a.name] = list(a.floats)

        # Resolve weight and bias
        self.weight = weights.get(node.input[1]) if len(node.input) > 1 else None
        self.bias = weights.get(node.input[2]) if len(node.input) > 2 else None

        # Shapes
        self.input_shape = shape_map.get(node.input[0]) if len(node.input) > 0 else None
        self.output_shape = shape_map.get(node.output[0]) if len(node.output) > 0 else None


def parse_onnx_graph(model_path):
    """Parse ONNX model into list of OnnxNodes."""
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)

    # Build weight map
    weights = {}
    for init in model.graph.initializer:
        weights[init.name] = numpy_helper.to_array(init)

    # Build shape map
    shape_map = {}
    for vi in model.graph.value_info:
        shape_map[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    for vi in model.graph.input:
        shape_map[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    for vi in model.graph.output:
        shape_map[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]

    # Parse nodes
    nodes = []
    for node in model.graph.node:
        nodes.append(OnnxNode(node, weights, shape_map))

    input_name = model.graph.input[0].name
    input_shape = shape_map[input_name]
    output_name = model.graph.output[0].name

    return nodes, input_name, input_shape, output_name, weights


# ─── Calibration: collect activation ranges ───

def calibrate_model(model_path, calib_dir, input_shape, num_images=50):
    """Run calibration images through ONNX Runtime to collect per-tensor activation ranges.

    Returns:
        activation_ranges: dict { tensor_name: (min, max) }
        activation_histograms: dict { tensor_name: (percentile_min, percentile_max) }
    """
    print(f"Calibrating with images from: {calib_dir}")

    # Get all intermediate tensor names
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)

    tensor_names = set()
    for vi in model.graph.value_info:
        tensor_names.add(vi.name)
    for inp in model.graph.input:
        tensor_names.add(inp.name)

    # Create session with all intermediate outputs
    for name in tensor_names:
        if not any(o.name == name for o in model.graph.output):
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))

    # Save temp model
    temp_path = '/tmp/calib_model.onnx'
    onnx.save(model, temp_path)

    sess = ort.InferenceSession(temp_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    # Collect images
    image_files = sorted(glob.glob(os.path.join(calib_dir, '*.jpg')) +
                         glob.glob(os.path.join(calib_dir, '*.png')))
    if len(image_files) == 0:
        raise ValueError(f"No images found in {calib_dir}")
    image_files = image_files[:num_images]
    print(f"  Using {len(image_files)} calibration images")

    # Collect min/max AND percentile data
    ranges = {}  # tensor_name -> (running_min, running_max)
    # For percentile: collect per-image percentile values
    percentile_data = {}  # tensor_name -> list of (p_low, p_high)

    PERCENTILE_LOW = 0.1   # 0.1th percentile
    PERCENTILE_HIGH = 99.9  # 99.9th percentile

    for i, img_path in enumerate(image_files):
        inp = preprocess_image_from_file(img_path, input_shape)
        outputs = sess.run(None, {input_name: inp})
        output_names = [o.name for o in sess.get_outputs()]

        for name, val in zip(output_names, outputs):
            val = val.astype(np.float64).flatten()
            vmin, vmax = float(val.min()), float(val.max())
            p_low = float(np.percentile(val, PERCENTILE_LOW))
            p_high = float(np.percentile(val, PERCENTILE_HIGH))

            if name in ranges:
                old_min, old_max = ranges[name]
                ranges[name] = (min(old_min, vmin), max(old_max, vmax))
                old_pl, old_ph = percentile_data[name]
                percentile_data[name] = (min(old_pl, p_low), max(old_ph, p_high))
            else:
                ranges[name] = (vmin, vmax)
                percentile_data[name] = (p_low, p_high)

        if (i + 1) % 10 == 0:
            print(f"  Calibrated {i+1}/{len(image_files)} images")

    # Add input range (fixed for our normalization: [-0.5, 0.5])
    ranges[input_name] = (-0.5, 0.5)
    percentile_data[input_name] = (-0.5, 0.5)

    print(f"  Collected ranges for {len(ranges)} tensors")

    # Cleanup
    os.remove(temp_path)
    return ranges, percentile_data


# ─── PTQ Quantization ───

def compute_scale_zp_symmetric(vmin, vmax, bits=8):
    """Compute symmetric quantization scale (zp=0).
    INT8:  [-127, 127], scale = max_abs / 127
    INT16: [-32767, 32767], scale = max_abs / 32767
    """
    qmax = (1 << (bits - 1)) - 1  # 127 for 8-bit, 32767 for 16-bit
    max_abs = max(abs(vmin), abs(vmax))
    if max_abs < 1e-10:
        max_abs = 1e-10
    scale = max_abs / qmax
    return scale, 0


def compute_scale_zp_asymmetric(vmin, vmax, bits=8):
    """Compute asymmetric quantization scale and zero point.
    Range: [vmin, vmax] → [-128, 127] (int8) or [-32768, 32767] (int16)
    """
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    qrange = qmax - qmin  # 255 for 8-bit, 65535 for 16-bit
    if vmax - vmin < 1e-10:
        vmax = vmin + 1e-10
    scale = (vmax - vmin) / qrange
    zp = int(np.round(qmin - vmin / scale))
    zp = max(qmin, min(qmax, zp))
    return scale, zp


def quantize_weight_perchannel(weight, out_axis=0, bits=8):
    """Per-channel symmetric weight quantization.
    Returns: weight_q (int8 or int16), scale_w (float per channel)
    """
    qmax = (1 << (bits - 1)) - 1  # 127 or 32767
    qmin = -qmax
    out_c = weight.shape[out_axis]
    scale_w = np.zeros(out_c, dtype=np.float64)
    dtype = np.int8 if bits == 8 else np.int16
    weight_q = np.zeros_like(weight, dtype=dtype)

    for c in range(out_c):
        if out_axis == 0:
            w_ch = weight[c]
        else:
            raise ValueError("Unsupported out_axis")
        max_abs = np.abs(w_ch).max()
        if max_abs < 1e-10:
            max_abs = 1e-10
        scale_w[c] = max_abs / qmax
        w_ch_q = np.round(w_ch / scale_w[c]).astype(np.int32)
        w_ch_q = np.clip(w_ch_q, qmin, qmax).astype(dtype)
        if out_axis == 0:
            weight_q[c] = w_ch_q

    return weight_q, scale_w


def compute_requant_params(scale_in, scale_w_perchannel, scale_out, bias_float=None):
    """Compute per-channel requantize parameters M, S, bias_q.

    The requantize equation:
        output_q = round(acc * (scale_in * scale_w[ch] / scale_out) ) + zp_out

    We represent scale_in * scale_w[ch] / scale_out as M[ch] * 2^(-S[ch]):
        M[ch] = round(effective_scale * 2^S)
        Choose S such that M fits in 15 bits (0 < M <= 32767)

    Bias quantization:
        bias_q = round(bias / (scale_in * scale_w[ch]))
        Stored as int64 to handle INT16 where scales are very small.
    """
    out_c = len(scale_w_perchannel)
    M_arr = np.zeros(out_c, dtype=np.int32)
    S_arr = np.zeros(out_c, dtype=np.int32)
    bias_q_arr = np.zeros(out_c, dtype=np.int64)

    for c in range(out_c):
        # Effective scale
        eff_scale = (scale_in * scale_w_perchannel[c]) / scale_out

        # Find S such that M = round(eff_scale * 2^S) fits in [1, 32767]
        # S is a 6-bit field (0..63). Search until M is in [16384, 32767] for
        # maximum precision, or until S reaches the hardware limit.
        best_s = 0
        best_m = max(1, int(np.round(eff_scale)))
        for s in range(64):  # S field is 6-bit, supports 0..63
            m = eff_scale * (2.0 ** s)  # use float pow to avoid int overflow
            if m >= 1.0 and m <= 32767.0:
                best_s = s
                best_m = int(np.round(m))
                # Try to get M as large as possible for precision
                if best_m >= 16384:  # good enough precision
                    break

        # Ensure M is in valid range
        best_m = max(1, min(32767, best_m))
        M_arr[c] = best_m
        S_arr[c] = best_s

        # Quantize bias: bias_q = round(bias / (scale_in * scale_w[ch]))
        if bias_float is not None:
            denom = scale_in * scale_w_perchannel[c]
            if denom < 1e-20:
                # Near-zero weight channel: bias has no meaningful effect
                bias_q_arr[c] = 0
            else:
                # Compute bias_q at full precision (stored as int64)
                bq_f64 = np.float64(bias_float[c]) / denom
                bias_q_arr[c] = int(np.round(bq_f64))
        else:
            bias_q_arr[c] = 0

    return M_arr, S_arr, bias_q_arr


# ─── Graph Fusion ───

def fold_batchnorm(nodes, weights):
    """Fold BatchNormalization into preceding Conv weight/bias (float32 level).

    For each BN node immediately following a Conv:
        w_new = w * (gamma / sqrt(var + eps))
        b_new = (b - mean) * (gamma / sqrt(var + eps)) + beta

    Modifies weights dict in-place and removes BN nodes from graph.
    Returns filtered nodes list.
    """
    # Map: output_tensor_name → node (to find Conv preceding BN)
    output_to_node = {}
    for node in nodes:
        for out in node.outputs:
            output_to_node[out] = node

    bn_outputs_to_conv_output = {}  # BN output → Conv output (for redirect)
    bn_nodes_to_remove = set()

    for node in nodes:
        if node.op_type != 'BatchNormalization':
            continue

        # BN inputs: [x, gamma, beta, mean, var]
        if len(node.inputs) < 5:
            continue

        bn_input = node.inputs[0]
        conv_node = output_to_node.get(bn_input)
        if conv_node is None or conv_node.op_type != 'Conv':
            continue

        # Extract BN parameters from weights
        gamma = weights.get(node.inputs[1])
        beta = weights.get(node.inputs[2])
        mean = weights.get(node.inputs[3])
        var = weights.get(node.inputs[4])
        if gamma is None or beta is None or mean is None or var is None:
            continue

        eps = node.attrs.get('epsilon', 1e-5)

        # Compute fold factors
        inv_std = gamma / np.sqrt(var + eps)  # shape [C]
        C = inv_std.shape[0]

        # Fold into Conv weight
        conv_weight_name = conv_node.inputs[1]
        w = weights[conv_weight_name]  # [OC, IC/g, KH, KW] or [OC, 1, KH, KW] for DW
        w_new = w * inv_std.reshape(C, *([1] * (w.ndim - 1)))
        weights[conv_weight_name] = w_new
        conv_node.weight = w_new

        # Fold into Conv bias (create if absent)
        if len(conv_node.inputs) > 2 and conv_node.inputs[2] in weights:
            bias_name = conv_node.inputs[2]
            b = weights[bias_name]
        else:
            # No bias — create zero bias and add to Conv inputs
            b = np.zeros(C, dtype=np.float32)
            bias_name = conv_weight_name + '_folded_bias'
            if len(conv_node.inputs) <= 2:
                conv_node.inputs.append(bias_name)
            else:
                conv_node.inputs[2] = bias_name

        b_new = (b - mean) * inv_std + beta
        weights[bias_name] = b_new
        conv_node.bias = b_new

        # Redirect: BN output → Conv output (downstream nodes use Conv output)
        bn_output = node.outputs[0]
        conv_output = conv_node.outputs[0]
        bn_outputs_to_conv_output[bn_output] = conv_output
        bn_nodes_to_remove.add(id(node))

    if not bn_nodes_to_remove:
        return nodes

    # Redirect all downstream references from BN outputs to Conv outputs
    for node in nodes:
        for i, inp in enumerate(node.inputs):
            if inp in bn_outputs_to_conv_output:
                node.inputs[i] = bn_outputs_to_conv_output[inp]

    # Remove BN nodes
    filtered = [n for n in nodes if id(n) not in bn_nodes_to_remove]
    print(f"  Folded {len(bn_nodes_to_remove)} BatchNorm nodes into Conv")
    return filtered


def fuse_graph(nodes, weights):
    """Fuse Relu/Clip into preceding Conv/Add nodes, skip shape-only ops.

    Returns:
        fused: list of fused operation descriptors
            {type: 'conv'/'dw'/'add'/'pool'/'fc', node: OnnxNode, relu: bool, relu6: bool, ...}
        passthrough_map: dict { output_tensor → input_tensor } for shape-only ops
            (Reshape, Flatten, Squeeze, Unsqueeze) — used to propagate quantization params
    """
    fused = []
    output_to_op = {}  # tensor_name → index in fused list

    skip_outputs = set()  # outputs that are consumed by fused relu/clip

    # Build passthrough map: shape-only ops redirect output → input
    # (these ops don't change values, just tensor shape)
    passthrough_map = {}  # output_name → input_name (the data source)
    for node in nodes:
        if node.op_type in ('Reshape', 'Flatten', 'Squeeze', 'Unsqueeze'):
            passthrough_map[node.outputs[0]] = node.inputs[0]

    # Resolve transitive passthrough (e.g. Reshape → Flatten chain)
    def resolve_passthrough(tensor_name):
        """Trace through passthrough ops to find the original data tensor."""
        visited = set()
        while tensor_name in passthrough_map and tensor_name not in visited:
            visited.add(tensor_name)
            tensor_name = passthrough_map[tensor_name]
        return tensor_name

    # First pass: mark relu/clip nodes that can be fused
    relu_inputs = {}   # input_tensor → output_tensor (for Relu)
    relu6_inputs = {}  # input_tensor → output_tensor (for Clip(0,6) aka ReLU6)
    for node in nodes:
        if node.op_type == 'Relu':
            relu_inputs[node.inputs[0]] = node.outputs[0]
        elif node.op_type == 'Clip':
            # Detect Clip(min=0, max=6) as ReLU6
            # ONNX opset 11+: inputs = [input, min, max]
            # ONNX opset 6:  attributes min/max
            clip_min, clip_max = None, None
            if len(node.inputs) >= 3:
                # opset 11+: min/max are constant inputs
                min_name = node.inputs[1] if len(node.inputs) > 1 else ''
                max_name = node.inputs[2] if len(node.inputs) > 2 else ''
                if min_name and min_name in weights:
                    clip_min = float(weights[min_name].flatten()[0])
                elif min_name == '':
                    clip_min = None  # unbounded
                if max_name and max_name in weights:
                    clip_max = float(weights[max_name].flatten()[0])
                elif max_name == '':
                    clip_max = None  # unbounded
            else:
                # opset 6: attributes
                clip_min = node.attrs.get('min', None)
                clip_max = node.attrs.get('max', None)

            if clip_min is not None and clip_min == 0.0 and clip_max is not None and clip_max == 6.0:
                # ReLU6
                relu6_inputs[node.inputs[0]] = node.outputs[0]
            elif clip_min is not None and clip_min == 0.0 and clip_max is None:
                # Just ReLU
                relu_inputs[node.inputs[0]] = node.outputs[0]
            else:
                # General Clip — treat as ReLU6 if min=0 (use max as clamp)
                if clip_min is not None and clip_min == 0.0 and clip_max is not None:
                    relu6_inputs[node.inputs[0]] = node.outputs[0]

    # Second pass: build fused ops
    for node in nodes:
        if node.op_type in ('Relu', 'Clip'):
            continue  # handled by fusion
        if node.op_type in ('Constant', 'Reshape', 'Flatten', 'Squeeze',
                            'Unsqueeze', 'BatchNormalization'):
            continue  # skip shape-only / utility ops (BN should already be folded)

        if node.op_type == 'Conv':
            group = node.attrs.get('group', 1)
            is_dw = (group > 1 and node.weight is not None and
                     group == node.weight.shape[0])
            out_tensor = node.outputs[0]
            op = {
                'type': 'dw' if is_dw else 'conv',
                'node': node,
                'relu': out_tensor in relu_inputs,
                'relu6': out_tensor in relu6_inputs,
                'relu_output': relu_inputs.get(out_tensor) or relu6_inputs.get(out_tensor),
            }
            fused.append(op)

        elif node.op_type == 'Add':
            out_tensor = node.outputs[0]
            op = {
                'type': 'add',
                'node': node,
                'relu': out_tensor in relu_inputs,
                'relu6': out_tensor in relu6_inputs,
                'relu_output': relu_inputs.get(out_tensor) or relu6_inputs.get(out_tensor),
            }
            fused.append(op)

        elif node.op_type in ('MaxPool', 'AveragePool', 'GlobalAveragePool'):
            out_tensor = node.outputs[0]
            op = {
                'type': 'pool',
                'node': node,
                'relu': out_tensor in relu_inputs,
                'relu6': out_tensor in relu6_inputs,
                'relu_output': relu_inputs.get(out_tensor) or relu6_inputs.get(out_tensor),
            }
            fused.append(op)

        elif node.op_type in ('Gemm', 'MatMul'):
            out_tensor = node.outputs[0]
            op = {
                'type': 'fc',
                'node': node,
                'relu': out_tensor in relu_inputs,
                'relu6': out_tensor in relu6_inputs,
                'relu_output': relu_inputs.get(out_tensor) or relu6_inputs.get(out_tensor),
            }
            fused.append(op)

        elif node.op_type == 'Concat':
            out_tensor = node.outputs[0]
            op = {
                'type': 'concat',
                'node': node,
                'relu': out_tensor in relu_inputs,
                'relu6': out_tensor in relu6_inputs,
                'relu_output': relu_inputs.get(out_tensor) or relu6_inputs.get(out_tensor),
            }
            fused.append(op)

    return fused, passthrough_map


# ─── Main Conversion ───

def convert_model(model_path, calib_dir, input_path, output_path,
                  input_format='int8-nchw', num_calib=50, bits=8):
    """Full conversion pipeline: ONNX float32 → NPU1 quantized."""

    # 1. Parse graph
    print(f"=== Parsing ONNX model (INT{bits} mode) ===")
    nodes, input_name, input_shape, output_name, all_weights = parse_onnx_graph(model_path)
    print(f"  Input: {input_name} {input_shape}")
    print(f"  Output: {output_name}")
    print(f"  Nodes: {len(nodes)}")

    # 2. Calibrate
    print("\n=== Running PTQ Calibration ===")
    act_ranges, act_percentiles = calibrate_model(model_path, calib_dir, input_shape, num_calib)

    # 2.5. Fold BatchNorm into Conv (graph optimization, float32 level)
    print("\n=== Folding BatchNorm ===")
    nodes = fold_batchnorm(nodes, all_weights)

    # 3. Fuse graph
    print("\n=== Fusing graph ===")
    fused_ops, passthrough_map = fuse_graph(nodes, all_weights)

    # Helper: resolve passthrough tensors (Reshape/Flatten/Squeeze don't change data)
    def resolve_passthrough(tensor_name):
        visited = set()
        while tensor_name in passthrough_map and tensor_name not in visited:
            visited.add(tensor_name)
            tensor_name = passthrough_map[tensor_name]
        return tensor_name

    print(f"  Fused ops: {len(fused_ops)}")
    if passthrough_map:
        print(f"  Passthrough (shape-only) ops: {len(passthrough_map)}")
    for i, op in enumerate(fused_ops):
        act_str = '+ReLU6' if op.get('relu6') else ('+ReLU' if op['relu'] else '')
        if op['type'] in ('conv', 'dw'):
            n = op['node']
            ks = n.attrs.get('kernel_shape', [1, 1])
            print(f"    [{i:2d}] {op['type'].upper():4s} {ks} {n.input_shape} → {n.output_shape} {act_str}")
        elif op['type'] == 'pool':
            n = op['node']
            print(f"    [{i:2d}] POOL {n.op_type} {n.input_shape} → {n.output_shape} {act_str}")
        elif op['type'] == 'fc':
            n = op['node']
            print(f"    [{i:2d}] FC   {n.op_type} {n.input_shape} → {n.output_shape} {act_str}")
        elif op['type'] == 'concat':
            n = op['node']
            n_inputs = len([inp for inp in n.inputs if inp not in all_weights])
            print(f"    [{i:2d}] CAT  {n_inputs} inputs → {n.output_shape} {act_str}")
        else:
            n = op['node']
            print(f"    [{i:2d}] ADD  {n.input_shape} {act_str}")

    # 4. Quantize each layer
    print(f"\n=== Quantizing layers (INT{bits} per-channel) ===")
    npu_layers = []
    weight_blobs = []

    qmin = -(1 << (bits - 1))      # -128 or -32768
    qmax = (1 << (bits - 1)) - 1   # 127 or 32767

    # Map: tensor_name → quantization params (scale, zp)
    tensor_quant = {}
    # Map: tensor_name → fused layer index (for residual tracking)
    tensor_to_layer_idx = {}

    def get_tensor_quant(tensor_name):
        """Look up quantization params, resolving passthrough ops."""
        if tensor_name in tensor_quant:
            return tensor_quant[tensor_name]
        resolved = resolve_passthrough(tensor_name)
        if resolved in tensor_quant:
            # Cache for future lookups
            tensor_quant[tensor_name] = tensor_quant[resolved]
            return tensor_quant[tensor_name]
        return None

    # Helper: get calibrated range
    def get_range(tensor_name, is_relu_output=False):
        """Get calibration range.
        
        For INT8: use 99.9th percentile (more robust to outliers, since INT8
        has coarse step size where clipping a few outliers hurts less than
        widening the range for all values).
        
        For INT16: use full min/max (INT16 has fine step size, so widening
        the range costs very little precision, but clipping hurts significantly
        because the clipping error dominates the tiny quantization noise).
        """
        # Resolve passthrough (Reshape/Flatten output → original data tensor)
        resolved = resolve_passthrough(tensor_name)
        if bits >= 16:
            # INT16: use full min/max range to avoid clipping
            r = act_ranges.get(resolved) or act_ranges.get(tensor_name)
            if r is None:
                r = act_percentiles.get(resolved) or act_percentiles.get(tensor_name, (-1.0, 1.0))
        else:
            # INT8: use percentile for robustness
            r = act_percentiles.get(resolved) or act_percentiles.get(tensor_name)
            if r is None:
                r = act_ranges.get(resolved) or act_ranges.get(tensor_name, (-1.0, 1.0))
        vmin, vmax = r
        if is_relu_output:
            vmin = 0.0  # ReLU outputs are non-negative
        return vmin, vmax

    # Input quantization
    in_min, in_max = get_range(input_name)
    in_scale, in_zp = compute_scale_zp_symmetric(in_min, in_max, bits)
    tensor_quant[input_name] = (in_scale, in_zp)
    print(f"  Input scale={in_scale:.8f}, zp={in_zp}")

    for i, op in enumerate(fused_ops):
        node = op['node']
        # Determine the effective output tensor name (after relu/clip fusion)
        has_act = op['relu'] or op.get('relu6', False)
        if has_act and op['relu_output']:
            eff_output = op['relu_output']
        else:
            eff_output = node.outputs[0]

        if op['type'] in ('conv', 'dw'):
            # Get input scale
            input_tensor = node.inputs[0]
            if get_tensor_quant(input_tensor) is None:
                r = get_range(input_tensor)
                s, z = compute_scale_zp_symmetric(r[0], r[1], bits)
                tensor_quant[input_tensor] = (s, z)
            scale_in, zp_in = tensor_quant[input_tensor]

            # Quantize weights per-channel
            weight = node.weight
            weight_q, scale_w = quantize_weight_perchannel(weight, out_axis=0, bits=bits)
            bias = node.bias

            # Determine output scale from activation ranges
            out_range = get_range(eff_output, is_relu_output=has_act)
            scale_out, zp_out = compute_scale_zp_symmetric(out_range[0], out_range[1], bits)
            tensor_quant[eff_output] = (scale_out, zp_out)

            # Compute requantize params
            M_arr, S_arr, bias_q_arr = compute_requant_params(
                scale_in, scale_w, scale_out, bias)

            # Build NPU LayerConfig
            _, _, ih, iw = node.input_shape if node.input_shape else [1, 1, 1, 1]
            _, oc, oh, ow = node.output_shape if node.output_shape else [1, 1, 1, 1]
            ic = weight.shape[1] if op['type'] == 'conv' else weight.shape[0]  # DW: groups=ic
            if op['type'] == 'dw':
                ic = weight.shape[0]

            ks = node.attrs.get('kernel_shape', [1, 1])
            strides = node.attrs.get('strides', [1, 1])
            pads = node.attrs.get('pads', [0, 0, 0, 0])
            dilations = node.attrs.get('dilations', [1, 1])

            cfg = LayerConfig(
                op_type=OP_DW_CONV if op['type'] == 'dw' else OP_CONV2D,
                data_type=1 if bits == 16 else 0,
                in_h=ih, in_w=iw, in_c=ic,
                out_h=oh, out_w=ow, out_c=oc,
                kernel_h=ks[0], kernel_w=ks[1],
                stride_h=strides[0], stride_w=strides[1],
                dilation_h=dilations[0], dilation_w=dilations[1],
                pad_top=pads[0], pad_bottom=pads[2],
                pad_left=pads[1], pad_right=pads[3],
                post_ctrl=POST_BIAS_EN | PPU_MODE_CONV_REQ,
                clamp_min=qmin, clamp_max=qmax,
                in_zp=int(zp_in),
            )
            if bits == 16:
                cfg.post_ctrl |= POST_INT16_OUT
            if op.get('relu6'):
                cfg.post_ctrl |= POST_RELU6_EN
                # ReLU6: clamp_max = min(qmax, round(6.0 / scale_out))
                relu6_qmax = int(np.round(6.0 / scale_out))
                cfg.clamp_max = min(qmax, relu6_qmax)
            elif op['relu']:
                cfg.post_ctrl |= POST_RELU_EN
            if zp_out != 0:
                cfg.post_ctrl |= POST_ZP_EN

            # Per-channel params
            ch_params = []
            for c in range(oc):
                ch_params.append(PerChannelParam(
                    M=int(M_arr[c]),
                    S=int(S_arr[c]),
                    zp=int(zp_out),
                    bias_q=int(bias_q_arr[c]),
                ))
            cfg.ch_params = ch_params

            npu_layers.append(cfg)
            # Track this layer's output tensor
            tensor_to_layer_idx[eff_output] = len(npu_layers) - 1
            # Also map the raw conv output (before relu) to same layer
            tensor_to_layer_idx[node.outputs[0]] = len(npu_layers) - 1

            # Weight blob
            if op['type'] == 'dw':
                # DW weight: ONNX [C,1,KH,KW] → NPU [C,KH,KW]
                w_npu = weight_q.reshape(oc, ks[0], ks[1])
            else:
                # Conv weight: ONNX [OC,IC,KH,KW] → NPU [OC,KH,KW,IC]
                w_npu = weight_q.transpose(0, 2, 3, 1)

            weight_blobs.append(w_npu.tobytes())

        elif op['type'] == 'add':
            # Add node: both inputs should already be quantized
            input_a_name = node.inputs[0]
            input_b_name = node.inputs[1]

            # Get scales for both inputs
            if get_tensor_quant(input_a_name) is None:
                r = get_range(input_a_name)
                tensor_quant[input_a_name] = compute_scale_zp_symmetric(r[0], r[1], bits)
            if get_tensor_quant(input_b_name) is None:
                r = get_range(input_b_name)
                tensor_quant[input_b_name] = compute_scale_zp_symmetric(r[0], r[1], bits)

            scale_a, _ = tensor_quant[input_a_name]
            scale_b, _ = tensor_quant[input_b_name]

            # Output scale
            out_range = get_range(eff_output, is_relu_output=has_act)
            scale_out, zp_out = compute_scale_zp_symmetric(out_range[0], out_range[1], bits)
            tensor_quant[eff_output] = (scale_out, zp_out)

            # Compute M_A, S_A, M_B, S_B
            def compute_ms(eff_scale):
                best_s, best_m = 0, max(1, int(np.round(eff_scale)))
                for s in range(64):  # S field is 6-bit, supports 0..63
                    m = eff_scale * (2.0 ** s)
                    if 1.0 <= m <= 32767.0:
                        best_s = s
                        best_m = int(np.round(m))
                        if best_m >= 16384:
                            break
                return max(1, min(32767, best_m)), best_s

            eff_a = scale_a / scale_out
            eff_b = scale_b / scale_out
            M_A, S_A = compute_ms(eff_a)
            M_B, S_B = compute_ms(eff_b)

            _, _, ih, iw = node.input_shape if node.input_shape else [1, 1, 1, 1]
            ic = node.input_shape[1] if node.input_shape else 1
            _, oc, oh, ow = node.output_shape if node.output_shape else [1, 1, 1, 1]

            cfg = LayerConfig(
                op_type=OP_ELTWISE_ADD,
                data_type=1 if bits == 16 else 0,
                in_h=ih, in_w=iw, in_c=ic,
                out_h=oh, out_w=ow, out_c=oc,
                post_ctrl=PPU_MODE_ADD,
                clamp_min=qmin, clamp_max=qmax,
            )
            if bits == 16:
                cfg.post_ctrl |= POST_INT16_OUT
            if op.get('relu6'):
                cfg.post_ctrl |= POST_RELU6_EN
                relu6_qmax = int(np.round(6.0 / scale_out))
                cfg.clamp_max = min(qmax, relu6_qmax)
            elif op['relu']:
                cfg.post_ctrl |= POST_RELU_EN

            cfg.add_params = AddParam(M_A=M_A, S_A=S_A, M_B=M_B, S_B=S_B)

            # Determine residual source:
            # In the sequential execution model, current tensor = input_a (latest output).
            # input_b = the residual/shortcut from an earlier layer.
            # The "current" flowing into Add is input_a (the one just produced by previous layer).
            # input_b is the skip connection from an older layer.
            # We need to figure out which is which:
            #   - input_a_name is one of the Add's inputs
            #   - input_b_name is the other
            # The "current" tensor is the output of the immediately preceding fused layer.
            prev_layer_idx = len(npu_layers) - 1
            prev_output_tensors = set()
            # Identify which input comes from the immediately previous layer (= "current")
            # and which comes from an earlier skip layer (= "residual_src")
            idx_a = tensor_to_layer_idx.get(input_a_name,
                    tensor_to_layer_idx.get(resolve_passthrough(input_a_name), -1))
            idx_b = tensor_to_layer_idx.get(input_b_name,
                    tensor_to_layer_idx.get(resolve_passthrough(input_b_name), -1))

            if idx_a == prev_layer_idx:
                # input_a is current, input_b is residual
                cfg.residual_src = idx_b
            elif idx_b == prev_layer_idx:
                # input_b is current, input_a is residual
                # Swap M_A/S_A and M_B/S_B since A=current, B=residual in PPU
                cfg.add_params = AddParam(M_A=M_B, S_A=S_B, M_B=M_A, S_B=S_A)
                cfg.residual_src = idx_a
            else:
                # Fallback: pick the one with higher index as "current"
                if idx_a > idx_b:
                    cfg.residual_src = idx_b
                else:
                    cfg.add_params = AddParam(M_A=M_B, S_A=S_B, M_B=M_A, S_B=S_A)
                    cfg.residual_src = idx_a

            print(f"    Add[{len(npu_layers)}]: input_a=L{idx_a}, input_b=L{idx_b}, "
                  f"residual_src=L{cfg.residual_src}")

            npu_layers.append(cfg)
            tensor_to_layer_idx[eff_output] = len(npu_layers) - 1
            tensor_to_layer_idx[node.outputs[0]] = len(npu_layers) - 1

        elif op['type'] == 'pool':
            # Pooling: no weight, passthrough quantization
            input_tensor = node.inputs[0]
            if get_tensor_quant(input_tensor) is None:
                r = get_range(input_tensor)
                s, z = compute_scale_zp_symmetric(r[0], r[1], bits)
                tensor_quant[input_tensor] = (s, z)
            scale_in, zp_in = tensor_quant[input_tensor]

            # Output scale from calibration (pooling doesn't change scale much)
            out_range = get_range(eff_output, is_relu_output=has_act)
            scale_out, zp_out = compute_scale_zp_symmetric(out_range[0], out_range[1], bits)
            tensor_quant[eff_output] = (scale_out, zp_out)

            # Dimensions
            _, _, ih, iw = node.input_shape if node.input_shape else [1, 1, 1, 1]
            ic = node.input_shape[1] if node.input_shape else 1
            _, oc, oh, ow = node.output_shape if node.output_shape else [1, 1, 1, 1]

            # Parse pooling attributes
            is_global = (node.op_type == 'GlobalAveragePool')
            pool_mode = 0 if node.op_type == 'MaxPool' else 1  # 0=Max, 1=Avg

            if is_global:
                kernel_shape = [ih, iw]
                pool_strides = [ih, iw]
                pads = [0, 0, 0, 0]
            else:
                kernel_shape = node.attrs.get('kernel_shape', [2, 2])
                pool_strides = node.attrs.get('strides', [2, 2])
                pads = node.attrs.get('pads', [0, 0, 0, 0])

            # Post-processing: pooling uses PASSTHROUGH (values stay in range)
            # If followed by activation, use RELU_ONLY for clamp+relu
            post_ctrl = PPU_MODE_PASSTHROUGH
            if bits == 16:
                post_ctrl |= POST_INT16_OUT
            if op.get('relu6'):
                post_ctrl |= POST_RELU6_EN
                post_ctrl = (post_ctrl & ~0x03) | PPU_MODE_RELU_ONLY
            elif op['relu']:
                post_ctrl |= POST_RELU_EN
                post_ctrl = (post_ctrl & ~0x03) | PPU_MODE_RELU_ONLY

            cfg = LayerConfig(
                op_type=OP_POOLING,
                data_type=1 if bits == 16 else 0,
                in_h=ih, in_w=iw, in_c=ic,
                out_h=oh, out_w=ow, out_c=oc,
                pool_mode=pool_mode,
                pool_h=kernel_shape[0], pool_w=kernel_shape[1],
                pool_stride_h=pool_strides[0], pool_stride_w=pool_strides[1],
                global_pool=1 if is_global else 0,
                pad_top=pads[0], pad_bottom=pads[2] if len(pads) > 2 else pads[0],
                pad_left=pads[1] if len(pads) > 1 else 0,
                pad_right=pads[3] if len(pads) > 3 else (pads[1] if len(pads) > 1 else 0),
                post_ctrl=post_ctrl,
                clamp_min=qmin, clamp_max=qmax,
            )

            print(f"    Pool[{len(npu_layers)}]: {node.op_type} "
                  f"k={kernel_shape} s={pool_strides} "
                  f"{ih}x{iw}x{ic} → {oh}x{ow}x{oc}")

            npu_layers.append(cfg)
            tensor_to_layer_idx[eff_output] = len(npu_layers) - 1
            tensor_to_layer_idx[node.outputs[0]] = len(npu_layers) - 1
            # No weight blob for pooling

        elif op['type'] == 'fc':
            # Fully-connected: Gemm or MatMul with constant weight
            # Gemm: Y = alpha * A * B + beta * C (typical: alpha=1, beta=1, transB=1)
            # Input may come through Flatten/Reshape — resolve passthrough
            input_tensor = node.inputs[0]
            if get_tensor_quant(input_tensor) is None:
                r = get_range(input_tensor)
                s, z = compute_scale_zp_symmetric(r[0], r[1], bits)
                tensor_quant[input_tensor] = (s, z)
            scale_in, zp_in = tensor_quant[input_tensor]

            # Get weight matrix
            weight_name = node.inputs[1]
            weight = all_weights.get(weight_name)
            if weight is None:
                print(f"  WARNING: FC weight '{weight_name}' not found, skipping")
                continue

            # Handle Gemm transB attribute
            # ONNX Gemm: Y = alpha * A * B' + beta * C
            #   transB=1: B' = B^T, so B stored as [out_c, in_c] (already correct)
            #   transB=0: B stored as [in_c, out_c], need transpose to [out_c, in_c]
            transB = node.attrs.get('transB', 0) if node.op_type == 'Gemm' else 0
            if not transB:
                # transB=0: weight is [in_c, out_c], transpose to [out_c, in_c]
                weight = weight.T.copy()
            # After this: weight shape = [out_c, in_c]

            # For MatMul: A=[1, in_c] * B=[in_c, out_c] → [1, out_c]
            # Weight B needs transpose to become [out_c, in_c]
            if node.op_type == 'MatMul' and weight.ndim == 2:
                # MatMul: A * B, B is [in_c, out_c]
                # Need [out_c, in_c] for our format
                in_features = node.input_shape[-1] if node.input_shape else weight.shape[0]
                if weight.shape[0] == in_features:
                    weight = weight.T.copy()

            out_c, in_c = weight.shape[0], weight.shape[1]

            # Reorder FC weight columns from NCHW to NHWC layout
            # csim stores activation tensors in NHWC order, but ONNX Flatten produces
            # NCHW order. We need to permute the weight's input dimension accordingly.
            # Only needed when FC input comes from a spatial tensor (H>1 or W>1).
            fc_input_tensor = resolve_passthrough(node.inputs[0])
            # Find the 4D shape of the tensor before flatten/reshape
            fc_in_4d = None
            for n in nodes:
                if n.outputs and n.outputs[0] == fc_input_tensor and n.output_shape:
                    fc_in_4d = n.output_shape
                    break
            if fc_in_4d is None:
                # Try node.input_shape (for Gemm directly on graph input)
                fc_in_4d = node.input_shape

            if fc_in_4d is not None and len(fc_in_4d) == 4:
                _, fc_c, fc_h, fc_w = fc_in_4d
                if fc_h > 1 or fc_w > 1:
                    # Need to reorder: NCHW flatten order → NHWC flatten order
                    # NCHW index: c*H*W + h*W + w
                    # NHWC index: h*W*C + w*C + c
                    assert in_c == fc_c * fc_h * fc_w, \
                        f"FC in_c={in_c} != {fc_c}*{fc_h}*{fc_w}={fc_c*fc_h*fc_w}"
                    # Build permutation: for each NHWC position, find NCHW source
                    perm = np.zeros(in_c, dtype=np.int32)
                    for c in range(fc_c):
                        for h in range(fc_h):
                            for w in range(fc_w):
                                nchw_idx = c * fc_h * fc_w + h * fc_w + w
                                nhwc_idx = h * fc_w * fc_c + w * fc_c + c
                                perm[nhwc_idx] = nchw_idx
                    weight = weight[:, perm]

            # Get bias (Gemm input[2] or None)
            bias = None
            if len(node.inputs) > 2 and node.inputs[2] in all_weights:
                bias = all_weights[node.inputs[2]]

            # Quantize weight per-channel
            weight_q, scale_w = quantize_weight_perchannel(
                weight.reshape(out_c, 1, 1, in_c), out_axis=0, bits=bits)
            weight_q = weight_q.reshape(out_c, in_c)

            # Output scale
            out_range = get_range(eff_output, is_relu_output=has_act)
            scale_out, zp_out = compute_scale_zp_symmetric(out_range[0], out_range[1], bits)
            tensor_quant[eff_output] = (scale_out, zp_out)

            # Compute requantize params
            M_arr, S_arr, bias_q_arr = compute_requant_params(
                scale_in, scale_w, scale_out, bias)

            # Build LayerConfig for FC
            # csim FC expects: in_h=1, in_w=1, in_c=features, out_h=1, out_w=1, out_c=outputs
            cfg = LayerConfig(
                op_type=OP_FC,
                data_type=1 if bits == 16 else 0,
                in_h=1, in_w=1, in_c=in_c,
                out_h=1, out_w=1, out_c=out_c,
                post_ctrl=POST_BIAS_EN | PPU_MODE_CONV_REQ,
                clamp_min=qmin, clamp_max=qmax,
                in_zp=int(zp_in),
            )
            if bits == 16:
                cfg.post_ctrl |= POST_INT16_OUT
            if op.get('relu6'):
                cfg.post_ctrl |= POST_RELU6_EN
                relu6_qmax = int(np.round(6.0 / scale_out))
                cfg.clamp_max = min(qmax, relu6_qmax)
            elif op['relu']:
                cfg.post_ctrl |= POST_RELU_EN
            if zp_out != 0:
                cfg.post_ctrl |= POST_ZP_EN

            # Per-channel params
            ch_params = []
            for c in range(out_c):
                ch_params.append(PerChannelParam(
                    M=int(M_arr[c]),
                    S=int(S_arr[c]),
                    zp=int(zp_out),
                    bias_q=int(bias_q_arr[c]),
                ))
            cfg.ch_params = ch_params

            npu_layers.append(cfg)
            tensor_to_layer_idx[eff_output] = len(npu_layers) - 1
            tensor_to_layer_idx[node.outputs[0]] = len(npu_layers) - 1

            # Weight blob: FC weight layout for csim is [out_c][in_c] (row-major)
            weight_blobs.append(weight_q.tobytes())

            print(f"    FC[{len(npu_layers)-1}]: {node.op_type} "
                  f"{in_c} → {out_c}"
                  f"{' +ReLU6' if op.get('relu6') else ' +ReLU' if op['relu'] else ''}")

        elif op['type'] == 'concat':
            # Concat: emit one sub-layer per input branch
            # Each sub-layer copies its input channels into the output at concat_offset
            # No requantization — passthrough (values stay in same quantized domain)
            axis = node.attrs.get('axis', 1)  # typically axis=1 (channel dim in NCHW)

            # Get all non-weight inputs (the data inputs to concat)
            concat_inputs = [inp for inp in node.inputs if inp not in all_weights]

            # Determine total output channels and per-input channel counts
            total_c = 0
            input_channels = []
            for inp_name in concat_inputs:
                resolved = resolve_passthrough(inp_name)
                # Find shape: check shape_map via nodes or calibration data
                inp_shape = None
                for n in nodes:
                    if n.outputs and n.outputs[0] == resolved and n.output_shape:
                        inp_shape = n.output_shape
                        break
                    if n.outputs and n.outputs[0] == inp_name and n.output_shape:
                        inp_shape = n.output_shape
                        break
                if inp_shape is None and node.output_shape:
                    # Fallback: can't determine individual input shape
                    # This shouldn't happen with shape inference
                    print(f"  WARNING: Can't determine shape for concat input '{inp_name}'")
                    continue
                # Channel dim depends on axis (NCHW: axis=1 → dim[1])
                c = inp_shape[axis] if inp_shape else 0
                input_channels.append(c)
                total_c += c

            # Verify against output shape
            if node.output_shape:
                expected_total_c = node.output_shape[axis]
                if total_c != expected_total_c:
                    # Use output shape as ground truth
                    total_c = expected_total_c

            # Spatial dimensions from output
            if node.output_shape and len(node.output_shape) == 4:
                _, _, oh, ow = node.output_shape
            else:
                oh, ow = 1, 1

            # Emit one Concat sub-layer per input
            offset = 0
            for branch_idx, (inp_name, ch_count) in enumerate(
                    zip(concat_inputs, input_channels)):
                # Resolve quantization for this input
                if get_tensor_quant(inp_name) is None:
                    r = get_range(inp_name)
                    s, z = compute_scale_zp_symmetric(r[0], r[1], bits)
                    tensor_quant[inp_name] = (s, z)

                # Determine source layer for residual_src
                resolved_inp = resolve_passthrough(inp_name)
                src_layer_idx = tensor_to_layer_idx.get(inp_name,
                    tensor_to_layer_idx.get(resolved_inp, -1))

                # For the first sub-layer: if source is immediately preceding,
                # use -1 (current); otherwise use residual_src
                prev_layer_idx = len(npu_layers) - 1
                if branch_idx == 0 and src_layer_idx == prev_layer_idx:
                    residual = -1  # read from current
                else:
                    residual = src_layer_idx

                cfg = LayerConfig(
                    op_type=OP_CONCAT,
                    data_type=1 if bits == 16 else 0,
                    in_h=oh, in_w=ow, in_c=ch_count,
                    out_h=oh, out_w=ow, out_c=total_c,
                    concat_offset=offset,
                    concat_total_c=total_c,
                    post_ctrl=PPU_MODE_PASSTHROUGH,
                    clamp_min=qmin, clamp_max=qmax,
                    residual_src=residual,
                )
                if bits == 16:
                    cfg.post_ctrl |= POST_INT16_OUT

                npu_layers.append(cfg)
                offset += ch_count

            # Track the concat output tensor
            # Use calibrated range for the concat output (covers both branches)
            out_range = get_range(eff_output)
            scale_out, zp_out = compute_scale_zp_symmetric(out_range[0], out_range[1], bits)
            tensor_quant[eff_output] = (scale_out, zp_out)
            tensor_to_layer_idx[eff_output] = len(npu_layers) - 1
            tensor_to_layer_idx[node.outputs[0]] = len(npu_layers) - 1

            print(f"    Concat[{len(npu_layers)-len(concat_inputs)}..{len(npu_layers)-1}]: "
                  f"{len(concat_inputs)} branches, channels={input_channels}, total={total_c}")

    # 5. Pack model
    print(f"\n=== Packing NPU1 model ({len(npu_layers)} layers) ===")
    all_weights_bin = b''.join(weight_blobs)
    pack_model(npu_layers, all_weights_bin, output_path)

    # 6. Prepare input tensor
    print(f"\n=== Preparing input tensor ===")
    if input_format == 'int8-nchw':
        # Read raw pixel bytes (uint8 NCHW), quantize to target bit-width for NPU
        raw = np.fromfile(input_path, dtype=np.uint8).reshape(input_shape)
        # Input preprocessing matches calibration: float = (pixel_uint8 - 127.5) / 255
        # input_q = round(float_val / in_scale)
        float_val = (raw.astype(np.float32) - 127.5) / 255.0
        input_q = np.round(float_val / in_scale).astype(np.int32)
        input_q = np.clip(input_q, qmin, qmax)
        if bits == 16:
            input_q = input_q.astype(np.int16)
        else:
            input_q = input_q.astype(np.int8)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    # Save as NCHW (npu_sim expects this)
    npu_input_path = output_path.replace('.bin', '_input.bin')
    input_q.reshape(-1).tofile(npu_input_path)
    elem_bytes = 2 if bits == 16 else 1
    print(f"  NPU input saved: {npu_input_path} ({input_q.size * elem_bytes} bytes)")
    print(f"  Input quant range: [{input_q.min()}, {input_q.max()}]")

    # Save quantization metadata for comparison script
    meta_path = output_path.replace('.bin', '_meta.npz')
    np.savez(meta_path,
             input_scale=in_scale,
             input_zp=in_zp,
             output_scale=scale_out,
             output_zp=zp_out,
             input_shape=np.array(input_shape),
             bits=bits,
             output_elements=npu_layers[-1].out_h * npu_layers[-1].out_w * npu_layers[-1].out_c)

    print(f"  Metadata saved: {meta_path}")
    print(f"\n=== Conversion complete ===")
    print(f"  Model: {output_path}")
    print(f"  Input: {npu_input_path}")
    print(f"  Output scale: {scale_out:.8f} (for dequantization)")

    return npu_input_path, meta_path


# ─── Main ───

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Open-NPU ONNX→NPU1 Converter')
    parser.add_argument('--model', required=True, help='ONNX float32 model path')
    parser.add_argument('--calib', required=True, help='Calibration image directory')
    parser.add_argument('--input', required=True, help='Test input file (debug.bin)')
    parser.add_argument('--output', default='model.npu1.bin', help='Output NPU1 model')
    parser.add_argument('--input-format', default='int8-nchw',
                        choices=['int8-nchw'], help='Input data format')
    parser.add_argument('--num-calib', type=int, default=50,
                        help='Number of calibration images to use')
    parser.add_argument('--bits', type=int, default=8, choices=[8, 16],
                        help='Quantization bit-width (8 or 16)')
    args = parser.parse_args()

    convert_model(args.model, args.calib, args.input, args.output,
                  args.input_format, args.num_calib, args.bits)
