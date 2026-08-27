#!/usr/bin/env python3
"""Compare default / MSE / ADMM / MSE+ADMM PTQ on Model D INT8 cosine."""

import os
import shutil
import subprocess
import sys

import numpy as np
import onnxruntime as ort

INTERNAL = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(INTERNAL)
sys.path.insert(0, TOOLS)
sys.path.insert(0, INTERNAL)
from onnx_converter import convert_model
from phase1_model_matrix import cosine, make_raw_input

ONNX = '/data/sam/onnx_quant/model.onnx'
CALIB = '/data/sam/model_b/O2_for_guopeng20221209/O2_quant_datas'
CSIM = '/data/sam/open-npu/csim/npu_sim'
SHAPE = (1, 112, 112)
NCALIB = 50
WORK = '/tmp/ptq_d8_compare'


def run_one(method):
    out_dir = os.path.join(WORK, method)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, 'input.bin')
    npu_bin = os.path.join(out_dir, 'model.npu1.bin')
    npu_in = os.path.join(out_dir, 'model.npu1_input.bin')
    npu_out = os.path.join(out_dir, 'model.npu1_output.bin')
    meta_p = os.path.join(out_dir, 'model.npu1_meta.npz')

    float_in = make_raw_input(SHAPE, CALIB, raw)
    print(f'\n======== D8 ptq={method} ========', flush=True)
    convert_model(ONNX, CALIB, raw, npu_bin,
                  input_format='int8-nchw', num_calib=NCALIB, bits=8,
                  ptq_method=method)

    env = os.environ.copy()
    env['ACC_WIDTH'] = '44'
    proc = subprocess.run([CSIM, npu_bin, npu_in, npu_out],
                          capture_output=True, text=True, env=env)
    if proc.returncode:
        print(proc.stdout[-600:] if proc.stdout else '')
        print(proc.stderr[-600:] if proc.stderr else '')
        raise RuntimeError(f'{method} csim failed')

    meta = np.load(meta_p)
    scale = float(meta['output_scale'])
    n = int(meta['output_elements'])
    q = np.fromfile(npu_out, dtype=np.int8)[:n]
    csim = q.astype(np.float64) * scale
    sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
    ref = sess.run(None, {sess.get_inputs()[0].name: float_in})[0].ravel().astype(np.float64)
    n = min(len(ref), len(csim))
    cos = cosine(ref[:n], csim[:n])
    mae = float(np.mean(np.abs(ref[:n] - csim[:n])))
    sat = int(np.sum(np.abs(q) >= 127))
    print(f'  {method}: cosine={cos:.6f}  mae={mae:.4f}  '
          f'sat={sat}/{n}  out_scale={scale:.6f}', flush=True)
    return cos


def main():
    methods = ['default', 'mse', 'admm', 'mse_admm']
    if len(sys.argv) > 1:
        methods = sys.argv[1].split(',')
    rows = []
    for m in methods:
        rows.append((m, run_one(m)))
    print('\nMETHOD      COSINE    vs default')
    base = dict(rows).get('default')
    for name, cos in rows:
        delta = '' if base is None else f'  {cos - base:+.6f}'
        print(f' {name:<10} {cos:.6f}{delta}')
    return 0 if all(c >= 0.95 for _, c in rows if _ != 'default') or (
        max(c for _, c in rows) >= 0.95) else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        sys.exit(1)
