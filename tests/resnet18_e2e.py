#!/usr/bin/env python3
"""
Phase 3: ResNet-18 Classification Model E2E Validation

Builds a ResNet-18-style classification network using ONNX helpers:
  - Conv7x7 stem + MaxPool
  - 4 stages of residual blocks (BasicBlock: Conv3x3+BN+ReLU+Conv3x3+BN + skip)
  - Global Average Pooling + FC (1000-class or 10-class)

This tests classification-specific patterns:
  - Residual/skip connections (Add with different source layers)
  - Deep network (18 layers)
  - Large spatial early layers (112x112, 56x56)
  - Global Average Pooling → FC at the end

Input: [1, 3, 224, 224]
Output: [1, num_classes]

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx_converter import convert_model

CSIM_PATH = '/data/sam/open-npu/csim/npu_sim'


class ResNetBuilder:
    """Builds ResNet-18 graph incrementally."""
    
    def __init__(self, seed=42):
        self.nodes = []
        self.inits = []
        self.idx = 0
        self.rng = np.random.RandomState(seed)
    
    def _make_weight(self, shape, name):
        """Kaiming-init weight."""
        fan_in = np.prod(shape[1:])
        w = (self.rng.randn(*shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.inits.append(numpy_helper.from_array(w, name))
        return name
    
    def _make_bias(self, out_c, name):
        b = (self.rng.randn(out_c) * 0.01).astype(np.float32)
        self.inits.append(numpy_helper.from_array(b, name))
        return name
    
    def conv_bn_relu(self, input_name, in_c, out_c, kernel=3, stride=1, pad=1,
                     relu=True, name=None):
        """Conv + BN(folded) + optional ReLU. Returns output tensor name."""
        if name is None:
            name = f'layer{self.idx}'
            self.idx += 1
        
        w_name = self._make_weight([out_c, in_c, kernel, kernel], f'{name}_w')
        b_name = self._make_bias(out_c, f'{name}_b')
        conv_out = f'{name}_conv'
        
        conv = helper.make_node('Conv', [input_name, w_name, b_name], [conv_out],
                                kernel_shape=[kernel, kernel],
                                strides=[stride, stride],
                                pads=[pad, pad, pad, pad])
        self.nodes.append(conv)
        
        if relu:
            relu_out = f'{name}_relu'
            self.nodes.append(helper.make_node('Relu', [conv_out], [relu_out]))
            return relu_out
        else:
            return conv_out
    
    def add(self, a_name, b_name, name=None):
        """Element-wise Add. Returns output name."""
        if name is None:
            name = f'add{self.idx}'
            self.idx += 1
        out = f'{name}_out'
        self.nodes.append(helper.make_node('Add', [a_name, b_name], [out]))
        return out
    
    def relu(self, input_name, name=None):
        if name is None:
            name = f'relu{self.idx}'
            self.idx += 1
        out = f'{name}_out'
        self.nodes.append(helper.make_node('Relu', [input_name], [out]))
        return out
    
    def maxpool(self, input_name, kernel=3, stride=2, pad=1, name=None):
        if name is None:
            name = f'pool{self.idx}'
            self.idx += 1
        out = f'{name}_out'
        self.nodes.append(helper.make_node('MaxPool', [input_name], [out],
                                           kernel_shape=[kernel, kernel],
                                           strides=[stride, stride],
                                           pads=[pad, pad, pad, pad]))
        return out
    
    def gap(self, input_name, name=None):
        """Global Average Pooling."""
        if name is None:
            name = f'gap{self.idx}'
            self.idx += 1
        out = f'{name}_out'
        self.nodes.append(helper.make_node('GlobalAveragePool', [input_name], [out]))
        return out
    
    def basic_block(self, input_name, in_c, out_c, stride=1, block_name='blk'):
        """ResNet BasicBlock: conv3x3+relu + conv3x3 + skip + relu.
        
        If stride>1 or in_c != out_c, uses 1x1 conv for shortcut.
        """
        # Main path
        x = self.conv_bn_relu(input_name, in_c, out_c, kernel=3, stride=stride, pad=1,
                              relu=True, name=f'{block_name}_conv1')
        x = self.conv_bn_relu(x, out_c, out_c, kernel=3, stride=1, pad=1,
                              relu=False, name=f'{block_name}_conv2')
        
        # Shortcut
        if stride != 1 or in_c != out_c:
            shortcut = self.conv_bn_relu(input_name, in_c, out_c, kernel=1,
                                         stride=stride, pad=0, relu=False,
                                         name=f'{block_name}_downsample')
        else:
            shortcut = input_name
        
        # Add + ReLU
        out = self.add(x, shortcut, name=f'{block_name}_add')
        out = self.relu(out, name=f'{block_name}_relu')
        return out


def make_resnet18_model(num_classes=10, input_size=224):
    """Build ResNet-18 (with smaller channel counts for faster testing).
    
    Architecture:
      stem: Conv7x7(3→64, s=2) + MaxPool(3x3, s=2)  → 56x56x64
      layer1: 2x BasicBlock(64→64)                    → 56x56x64
      layer2: 2x BasicBlock(64→128, s=2 for first)    → 28x28x128
      layer3: 2x BasicBlock(128→256, s=2 for first)   → 14x14x256
      layer4: 2x BasicBlock(256→512, s=2 for first)   → 7x7x512
      GAP → Flatten → FC(512→num_classes)
      
    For MCU-friendly size, we use half channels:
      64→32, 128→64, 256→128, 512→256
    """
    # Use half-width for reasonable test size
    c1, c2, c3, c4 = 32, 64, 128, 256
    
    builder = ResNetBuilder(seed=2024)
    
    # Stem: Conv7x7 + MaxPool
    x = builder.conv_bn_relu('input', 3, c1, kernel=7, stride=2, pad=3,
                             relu=True, name='stem')
    x = builder.maxpool(x, kernel=3, stride=2, pad=1, name='stem_pool')
    
    # Layer 1: 2x BasicBlock, no downsample
    x = builder.basic_block(x, c1, c1, stride=1, block_name='layer1_0')
    x = builder.basic_block(x, c1, c1, stride=1, block_name='layer1_1')
    
    # Layer 2: first block with stride=2
    x = builder.basic_block(x, c1, c2, stride=2, block_name='layer2_0')
    x = builder.basic_block(x, c2, c2, stride=1, block_name='layer2_1')
    
    # Layer 3: first block with stride=2
    x = builder.basic_block(x, c2, c3, stride=2, block_name='layer3_0')
    x = builder.basic_block(x, c3, c3, stride=1, block_name='layer3_1')
    
    # Layer 4: first block with stride=2
    x = builder.basic_block(x, c3, c4, stride=2, block_name='layer4_0')
    x = builder.basic_block(x, c4, c4, stride=1, block_name='layer4_1')
    
    # GAP
    x = builder.gap(x, name='avgpool')
    
    # Reshape [1, 256, 1, 1] → [1, 256] for Gemm
    reshape_shape = numpy_helper.from_array(
        np.array([1, c4], dtype=np.int64), 'fc_shape')
    builder.inits.append(reshape_shape)
    builder.nodes.append(helper.make_node('Reshape', [x, 'fc_shape'], ['fc_in']))
    
    # FC: Gemm(256 → num_classes)
    fc_w = (builder.rng.randn(num_classes, c4) * np.sqrt(2.0 / c4)).astype(np.float32)
    fc_b = (builder.rng.randn(num_classes) * 0.01).astype(np.float32)
    builder.inits.append(numpy_helper.from_array(fc_w, 'fc_w'))
    builder.inits.append(numpy_helper.from_array(fc_b, 'fc_b'))
    builder.nodes.append(helper.make_node('Gemm', ['fc_in', 'fc_w', 'fc_b'], ['output'],
                                           transB=1))
    
    # I/O
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT,
                                       [1, 3, input_size, input_size])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, num_classes])
    
    graph = helper.make_graph(builder.nodes, 'resnet18_half', [X], [Y],
                              initializer=builder.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    model = onnx.shape_inference.infer_shapes(model)
    return model


def run_e2e(model_path, calib_dir, test_img_path, h=224, w=224, bits=8):
    """Full E2E: convert + csim + compare."""
    tmpdir = tempfile.mkdtemp(prefix='npu_resnet_')
    
    # Prepare input
    test_img = Image.open(test_img_path).resize((w, h))
    arr = np.array(test_img).transpose(2, 0, 1).astype(np.uint8)
    input_bin = os.path.join(tmpdir, 'input.bin')
    arr.tofile(input_bin)
    
    # Convert
    npu_model = os.path.join(tmpdir, 'model.npu1.bin')
    try:
        convert_model(model_path, calib_dir, input_bin, npu_model,
                      input_format='int8-nchw', num_calib=20, bits=bits)
    except Exception as e:
        print(f"  FAIL: Converter error: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # Run csim
    output_bin = os.path.join(tmpdir, 'output.bin')
    npu_input = npu_model.replace('.bin', '_input.bin')
    
    if not os.path.exists(CSIM_PATH):
        print("  SKIP: csim not found")
        return None, None
    
    result = subprocess.run(
        [CSIM_PATH, npu_model, npu_input, output_bin],
        capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  FAIL: csim error (rc={result.returncode}):")
        print(f"    {result.stdout[-300:]}")
        print(f"    {result.stderr[-300:]}")
        return None, None
    
    # Read and dequantize
    meta = np.load(npu_model.replace('.bin', '_meta.npz'))
    out_scale = float(meta['output_scale'])
    
    if bits == 16:
        csim_q = np.fromfile(output_bin, dtype=np.int16).astype(np.float32)
    else:
        csim_q = np.fromfile(output_bin, dtype=np.int8).astype(np.float32)
    csim_float = csim_q * out_scale
    
    return csim_float, npu_model


def main():
    print("=" * 60)
    print("Phase 3: ResNet-18 Classification E2E Validation")
    print("=" * 60)
    
    # Build model
    num_classes = 10
    print(f"\n--- Building ResNet-18 (half-width, {num_classes} classes) ---")
    model = make_resnet18_model(num_classes=num_classes, input_size=224)
    
    tmpdir = tempfile.mkdtemp(prefix='npu_resnet_test_')
    model_path = os.path.join(tmpdir, 'resnet18.onnx')
    onnx.save(model, model_path)
    onnx.checker.check_model(model)
    
    print(f"  Model saved: {model_path}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Initializers: {len(model.graph.initializer)}")
    
    # Create calibration images
    calib_dir = os.path.join(tmpdir, 'calib')
    os.makedirs(calib_dir)
    np.random.seed(42)
    for i in range(20):
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(calib_dir, f'calib_{i:04d}.jpg'))
    
    # Test image
    np.random.seed(99)
    test_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    test_img_path = os.path.join(calib_dir, 'test.jpg')
    Image.fromarray(test_img).save(test_img_path)
    
    # ORT reference
    print("\n--- ORT reference ---")
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp_float = (test_img.astype(np.float32).transpose(2, 0, 1) - 127.5) / 255.0
    inp_float = inp_float[np.newaxis, ...]
    ref_out = sess.run(None, {sess.get_inputs()[0].name: inp_float})[0].flatten()
    print(f"  Output shape: {ref_out.shape}")
    print(f"  Output range: [{ref_out.min():.4f}, {ref_out.max():.4f}]")
    print(f"  Argmax (predicted class): {ref_out.argmax()}")
    
    # E2E test
    print("\n--- INT8 E2E ---")
    csim_out, npu_model = run_e2e(model_path, calib_dir, test_img_path, bits=8)
    
    results = []
    
    if csim_out is not None:
        if ref_out.size != csim_out.size:
            print(f"  Size mismatch: ORT={ref_out.size}, csim={csim_out.size}")
            results.append(('INT8 cosine', False))
        else:
            # Cosine
            dot = np.dot(ref_out, csim_out)
            na, nb = np.linalg.norm(ref_out), np.linalg.norm(csim_out)
            cos = dot / (na * nb) if na > 1e-10 and nb > 1e-10 else 0.0
            print(f"  Cosine similarity: {cos:.6f}")
            
            # Top-1 match
            ort_class = ref_out.argmax()
            csim_class = csim_out.argmax()
            top1_match = ort_class == csim_class
            print(f"  ORT  predicted class: {ort_class} (score={ref_out[ort_class]:.4f})")
            print(f"  csim predicted class: {csim_class} (score={csim_out[csim_class]:.4f})")
            print(f"  Top-1 match: {'YES' if top1_match else 'NO'}")
            
            cos_pass = cos >= 0.90
            print(f"  {'PASS' if cos_pass else 'FAIL'} (cosine threshold 0.90)")
            results.append(('INT8 cosine >= 0.90', cos_pass))
            results.append(('INT8 top-1 match', top1_match))
    else:
        results.append(('INT8 cosine', False))
    
    # Multi-image stability test
    print("\n--- Multi-image stability (5 images) ---")
    cosines = []
    top1_matches = 0
    for i in range(5):
        np.random.seed(100 + i)
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img_path = os.path.join(tmpdir, f'test_{i}.jpg')
        Image.fromarray(img).save(img_path)
        
        # ORT
        inp_f = ((img.astype(np.float32).transpose(2, 0, 1) - 127.5) / 255.0)[np.newaxis, ...]
        ref = sess.run(None, {sess.get_inputs()[0].name: inp_f})[0].flatten()
        
        # csim
        csim, _ = run_e2e(model_path, calib_dir, img_path, bits=8)
        if csim is not None and ref.size == csim.size:
            dot = np.dot(ref, csim)
            na, nb = np.linalg.norm(ref), np.linalg.norm(csim)
            cos = dot / (na * nb) if na > 1e-10 and nb > 1e-10 else 0.0
            cosines.append(cos)
            if ref.argmax() == csim.argmax():
                top1_matches += 1
            print(f"  Image {i}: cos={cos:.6f}, top1={'match' if ref.argmax()==csim.argmax() else 'MISMATCH'}")
    
    if cosines:
        avg_cos = np.mean(cosines)
        print(f"\n  Average cosine: {avg_cos:.6f}")
        print(f"  Top-1 accuracy: {top1_matches}/{len(cosines)} ({100*top1_matches/len(cosines):.0f}%)")
        results.append(('Multi-image avg cosine >= 0.90', avg_cos >= 0.90))
        results.append((f'Top-1 accuracy >= 80%', top1_matches >= 4))
    
    # Summary
    print(f"\n{'='*60}")
    print("ResNet-18 E2E SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\nResNet-18 Phase 3 validation PASSED!")
    else:
        print("\nSome tests FAILED!")
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
