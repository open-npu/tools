#!/usr/bin/env python3
"""Phase-1 10-cell model matrix.

A cell passes only when both are true on the current converter + CSIM + RTL:
  1. CSIM dequant vs ONNX float cosine is high
  2. SoC/RTL chained inference is bit-exact against a freshly generated CSIM golden

Default is gate (1) only. Pass --rtl (or run via phase1_regression.py) for both.
Recorded values, cached npu1 binaries, missing cells, and SKIP are not acceptance.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_converter import convert_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSIM = os.path.join(ROOT, "csim", "npu_sim")
E2E = os.path.join(ROOT, "soc", "run_model_e2e.sh")
WORK = "/tmp/phase1_model_matrix"

CELLS = (
    ("A", 8, "/data/sam/onnx_quant/MODEL_A.onnx",
     "/data/sam/onnx_quant/a3_test_images", (3, 224, 224), 50),
    ("A", 16, "/data/sam/onnx_quant/MODEL_A.onnx",
     "/data/sam/onnx_quant/a3_test_images", (3, 224, 224), 50),
    ("B", 8, "/data/sam/model_b/WXPay_PalmIrLiveness_624_06_r20260509.onnx",
     "/data/sam/model_b/O2_for_guopeng20221209/O2_quant_datas", (1, 112, 112), 100),
    ("B", 16, "/data/sam/model_b/WXPay_PalmIrLiveness_624_06_r20260509.onnx",
     "/data/sam/model_b/O2_for_guopeng20221209/O2_quant_datas", (1, 112, 112), 100),
    ("C", 8, "/tmp/npu_yolo_output_small__uoinc2r/model_single.onnx",
     "/data/sam/onnx_quant/a3_test_images", (3, 416, 416), 20),
    ("C", 16, "/tmp/npu_yolo_output_small__uoinc2r/model_single.onnx",
     "/data/sam/onnx_quant/a3_test_images", (3, 416, 416), 20),
    ("D", 8, "/data/sam/onnx_quant/model.onnx",
     "/data/sam/model_b/O2_for_guopeng20221209/O2_quant_datas", (1, 112, 112), 50),
    ("D", 16, "/data/sam/onnx_quant/model.onnx",
     "/data/sam/model_b/O2_for_guopeng20221209/O2_quant_datas", (1, 112, 112), 50),
    ("E", 8, "/data/sam/onnx_quant/resnet18_imagenet.onnx",
     "/data/sam/onnx_quant/imagenet_calib", (3, 224, 224), 50),
    ("E", 16, "/data/sam/onnx_quant/resnet18_imagenet.onnx",
     "/data/sam/onnx_quant/imagenet_calib", (3, 224, 224), 50),
)

SOC_NAME = {
    ("A", 8): "model_a_int8",
    ("A", 16): "model_a_int16",
    ("B", 8): "model_b_int8",
    ("B", 16): "model_b_int16",
    ("C", 8): "model_c_int8",
    ("C", 16): "model_c_int16",
    ("D", 8): "model_d_int8",
    ("D", 16): "model_d_int16",
    ("E", 8): "model_e_int8",
    ("E", 16): "model_e_int16",
}

THRESH = {8: 0.95, 16: 0.99}


def cosine(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def make_raw_input(shape, calib_dir, dest):
    c, h, w = shape
    imgs = sorted(
        glob.glob(os.path.join(calib_dir, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(calib_dir, "**", "*.jpeg"), recursive=True)
        + glob.glob(os.path.join(calib_dir, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(calib_dir, "**", "*.bmp"), recursive=True)
    )
    imgs = [p for p in imgs if os.path.isfile(p)]
    if not imgs:
        raise RuntimeError(f"no images in {calib_dir}")
    img = Image.open(imgs[0])
    if c == 1:
        img = img.convert("L").resize((w, h))
        arr = np.array(img, dtype=np.uint8)[None]
    else:
        img = img.convert("RGB").resize((w, h))
        arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
    arr.tofile(dest)
    # NCHW float in [-0.5, 0.5], matching converter default
    return (arr.astype(np.float32) / 255.0 - 0.5)[None]


def run_cosine(model, bits, onnx_path, calib, shape, ncalib):
    tag = f"model_{model.lower()}_int{bits}"
    out_dir = os.path.join(WORK, tag)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, "input.bin")
    npu_bin = os.path.join(out_dir, "model.npu1.bin")
    npu_in = os.path.join(out_dir, "model.npu1_input.bin")
    npu_out = os.path.join(out_dir, "model.npu1_output.bin")
    meta_p = os.path.join(out_dir, "model.npu1_meta.npz")

    float_in = make_raw_input(shape, calib, raw)
    # D8 accepted PTQ: MSE activation clip (0.952). Default percentile is ~0.947.
    ptq = "mse" if model == "D" and bits == 8 else "default"
    convert_model(onnx_path, calib, raw, npu_bin,
                  input_format="int8-nchw", num_calib=ncalib, bits=bits,
                  ptq_method=ptq)

    env = os.environ.copy()
    env["ACC_WIDTH"] = "44"
    proc = subprocess.run(
        [CSIM, npu_bin, npu_in, npu_out],
        capture_output=True, text=True, env=env)
    if proc.returncode:
        print(proc.stdout[-800:] if proc.stdout else "")
        print(proc.stderr[-800:] if proc.stderr else "")
        raise RuntimeError(f"{tag} csim failed")

    meta = np.load(meta_p)
    scale = float(meta["output_scale"])
    n = int(meta["output_elements"])
    dt = np.int16 if bits == 16 else np.int8
    # CSIM writes signed integers. Converter stores output_zp=128 for INT8 as a
    # hint for uint8 readers; subtracting it from signed int8 is wrong and
    # collapses cosine. Symmetric quant is dequant = q * scale.
    csim = np.fromfile(npu_out, dtype=dt)[:n].astype(np.float32) * scale

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: float_in})[0].ravel()
    n = min(len(ref), len(csim))
    return cosine(ref[:n], csim[:n])


def run_rtl(soc_name):
    if not os.path.isfile(E2E):
        raise RuntimeError(f"missing {E2E}")
    print(f"=== SoC RTL {soc_name} ===", flush=True)
    proc = subprocess.run(["bash", E2E, soc_name])
    if proc.returncode:
        raise RuntimeError(f"{soc_name} SoC RTL failed with exit code {proc.returncode}")
    return True


def selected_cells(only):
    rows = []
    for cell in CELLS:
        key = f"{cell[0]}{cell[1]}"
        if only and key not in only:
            continue
        rows.append(cell)
    if only and len(rows) != len(only):
        have = {f"{c[0]}{c[1]}" for c in rows}
        missing = sorted(only - have)
        raise RuntimeError(f"unknown cell(s): {', '.join(missing)}")
    if not only and len(rows) != 10:
        raise RuntimeError(f"expected 10 cells, got {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rtl", action="store_true",
        help="also run SoC Verilator bit-exact vs fresh CSIM golden",
    )
    parser.add_argument("--only", default="",
                        help="comma list like C8,D16,E8")
    args = parser.parse_args()
    only = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    cells = selected_cells(only)

    if not os.path.isfile(CSIM):
        raise RuntimeError(f"CSIM binary missing: {CSIM}")

    print("Phase-1 model matrix")
    print("  gate 1: CSIM vs ONNX cosine (this run, no recorded values)")
    if args.rtl:
        print("  gate 2: SoC RTL vs freshly generated CSIM golden")
    else:
        print("  gate 2: omitted (--rtl not set; not an acceptance run)")
    print(flush=True)

    rows = []
    failed = False
    for model, bits, onnx_path, calib, shape, ncalib in cells:
        key = f"{model}{bits}"
        thr = THRESH[bits]
        print(f"=== {model} INT{bits} cosine ===", flush=True)
        try:
            cos = run_cosine(model, bits, onnx_path, calib, shape, ncalib)
            cos_ok = cos >= thr
            cos_src = "this-run"
        except Exception as exc:
            print(f"  {key} cosine raised: {exc}", flush=True)
            cos, cos_ok, cos_src = 0.0, False, "error"
        print(f"  {model} INT{bits}: cosine={cos:.6f}  "
              f"thr={thr}  {'PASS' if cos_ok else 'FAIL'}  ({cos_src})",
              flush=True)

        rtl_ok, rtl_src = None, "omitted"
        if args.rtl:
            soc_name = SOC_NAME[(model, bits)]
            try:
                run_rtl(soc_name)
                rtl_ok, rtl_src = True, "this-run"
            except Exception as exc:
                print(f"  {key} RTL raised: {exc}", flush=True)
                rtl_ok, rtl_src = False, "error"
            print(f"  {model} INT{bits}: RTL↔CSIM  "
                  f"{'PASS' if rtl_ok else 'FAIL'}  ({rtl_src})",
                  flush=True)

        cell_ok = cos_ok and (rtl_ok is not False)
        if args.rtl:
            cell_ok = cos_ok and bool(rtl_ok)
        if not cell_ok:
            failed = True
        rows.append((model, bits, cos, thr, cos_ok, rtl_ok, cos_src, rtl_src))

    print("\nCELL  BITS  COSINE    THR    CSIM↔ONNX  RTL↔CSIM")
    for model, bits, cos, thr, cos_ok, rtl_ok, cos_src, rtl_src in rows:
        rtl_s = "omitted" if rtl_ok is None else ("PASS" if rtl_ok else "FAIL")
        print(f" {model:>2}    {bits:<4} {cos:.6f}  {thr:.2f}  "
              f"{'PASS' if cos_ok else 'FAIL':<4}      {rtl_s:<8}  "
              f"{cos_src}/{rtl_src}")
    if len(rows) != len(cells):
        raise RuntimeError("missing cell in results")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
