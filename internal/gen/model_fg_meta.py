#!/usr/bin/env python3
"""Emit tiling metadata for the public nets used in the engine comparison.

    H  ResNet-20 CIFAR            32x32x3    He et al. CIFAR series
    I  MobileNetV2 1.0            224x224x3  Arm ML-Zoo image classification
    J  ResNet-8 CIFAR             32x32x3    MLPerf Tiny image classification
    K  MobileNetV1 0.25x          96x96x3    MLPerf Tiny visual wake words
    L  DS-CNN                     49x10x1    MLPerf Tiny keyword spotting

Each net is built locally with random weights and pushed through the real
converter, so the layer list and tile grid are whatever the scheduler would
actually emit. Only geometry is written out; the util benchmark drives its own
activations, so trained weights are not needed.

    python3 tools/internal/gen/model_fg_meta.py
"""

import json
import os
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, '..', '..'))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, 'internal'))

from onnx_converter import convert_model          # noqa: E402
from bin2golden import read_npu1                  # noqa: E402

WORK = '/tmp/model_fg_meta'
GOLDEN = os.path.join(ROOT, 'rtl', 'tb', 'golden', 'golden_dma_e2e')

META_KEYS = (
    'op_type', 'data_type', 'in_h', 'in_w', 'in_c', 'out_h', 'out_w', 'out_c',
    'kernel_h', 'kernel_w', 'stride_h', 'stride_w', 'pad_top', 'pad_left',
    'tile_h', 'tile_w', 'tile_num_h', 'tile_num_w', 'post_ctrl', 'sched_ctrl',
    'in_zp', 'input_src', 'residual_src', 'wgt_layout',
)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(y + idt)


class ResNet20(nn.Module):
    """He et al. CIFAR ResNet, n=3 -> 20 layers."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        blocks = []
        cin = 16
        for cout, stride in ((16, 1), (32, 2), (64, 2)):
            for b in range(3):
                blocks.append(BasicBlock(cin, cout, stride if b == 0 else 1))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


def constants_to_initializers(model):
    """The converter looks up Clip bounds in initializers, but the exporter
    emits them as Constant nodes."""
    graph = model.graph
    for node in list(graph.node):
        if node.op_type != 'Constant':
            continue
        tensor = None
        for attr in node.attribute:
            if attr.name == 'value':
                tensor = attr.t
        if tensor is None:
            continue
        tensor.name = node.output[0]
        graph.initializer.append(tensor)
        graph.node.remove(node)
    return model


def drop_identity(model):
    """Rewire and delete Identity nodes the exporter leaves behind."""
    graph = model.graph
    outputs = {o.name for o in graph.output}
    changed = True
    while changed:
        changed = False
        for node in list(graph.node):
            if node.op_type != 'Identity' or node.output[0] in outputs:
                continue
            src, dst = node.input[0], node.output[0]
            for other in graph.node:
                for i, name in enumerate(other.input):
                    if name == dst:
                        other.input[i] = src
            graph.node.remove(node)
            changed = True
    return model


class ResNet8(nn.Module):
    """MLPerf Tiny image classification reference (resnet_v1_eembc)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.s1 = BasicBlock(16, 16, 1)
        self.s2 = BasicBlock(16, 32, 2)
        self.s3 = BasicBlock(32, 64, 2)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.s3(self.s2(self.s1(self.stem(x))))
        return self.fc(torch.flatten(self.pool(x), 1))


def _dw_sep(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU6(),
        nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU6())


class MobileNetV1(nn.Module):
    """MLPerf Tiny visual-wake-words reference: alpha 0.25, 96x96, 2 classes."""

    CFG = ((64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
           (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2),
           (1024, 1))

    def __init__(self, alpha=0.25, num_classes=2):
        super().__init__()
        def ch(c):
            return max(8, int(c * alpha))
        layers = [nn.Conv2d(3, ch(32), 3, 2, 1, bias=False),
                  nn.BatchNorm2d(ch(32)), nn.ReLU6()]
        cin = ch(32)
        for cout, stride in self.CFG:
            layers.append(_dw_sep(cin, ch(cout), stride))
            cin = ch(cout)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.fc(torch.flatten(x, 1))


class DSCNN(nn.Module):
    """MLPerf Tiny keyword-spotting reference: 49x10x1 MFCC, 12 classes."""

    def __init__(self, num_classes=12, ch=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, ch, (10, 4), (2, 2), (5, 1), bias=False),
            nn.BatchNorm2d(ch), nn.ReLU())
        blocks = []
        for _ in range(4):
            blocks += [
                nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(),
                nn.Conv2d(ch, ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU()]
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.pool(self.blocks(self.stem(x)))
        return self.fc(torch.flatten(x, 1))


def build_onnx(model, size, path):
    model.eval()
    h, w, c = size if isinstance(size, tuple) else (size, size, 3)
    dummy = torch.randn(1, c, h, w)
    torch.onnx.export(model, dummy, path, opset_version=13,
                      input_names=['input'], output_names=['output'],
                      dynamo=False)
    m = drop_identity(constants_to_initializers(onnx.load(path)))
    onnx.save(m, path)
    return m


def make_calib(size, n=12):
    h, w, c = size
    tag = f'{h}x{w}x{c}'
    calib = os.path.join(WORK, f'calib{tag}')
    os.makedirs(calib, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n):
        img = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(calib, f'c{i:03d}.jpg'))
    inp = os.path.join(WORK, f'input{tag}.bin')
    rng.randint(-128, 128, (c, h, w)).astype(np.int8).tofile(inp)
    return calib, inp


def emit(tag, model, size):
    size = size if isinstance(size, tuple) else (size, size, 3)
    os.makedirs(WORK, exist_ok=True)
    onnx_path = os.path.join(WORK, f'{tag}.onnx')
    build_onnx(model, size, onnx_path)
    calib, inp = make_calib(size)
    out_bin = os.path.join(WORK, f'{tag}.npu1.bin')
    convert_model(onnx_path, calib, inp, out_bin,
                  input_format='int8-nchw', num_calib=8, bits=8)
    layers, _ = read_npu1(out_bin)
    meta = [{k: getattr(lay, k) for k in META_KEYS} for lay in layers]
    dest = os.path.join(GOLDEN, f'model_{tag}_int8')
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'{tag}: {len(meta)} layers -> {dest}/metadata.json')


def main():
    torch.manual_seed(0)
    only = set(sys.argv[1:])
    from torchvision.models import mobilenet_v2
    todo = [
        ('h', lambda: ResNet20(), (32, 32, 3)),
        ('i', lambda: mobilenet_v2(weights=None), (224, 224, 3)),
        ('j', lambda: ResNet8(), (32, 32, 3)),
        ('k', lambda: MobileNetV1(), (96, 96, 3)),
        ('l', lambda: DSCNN(), (49, 10, 1)),
    ]
    for tag, build, size in todo:
        if only and tag not in only:
            continue
        emit(tag, build(), size)


if __name__ == '__main__':
    main()
