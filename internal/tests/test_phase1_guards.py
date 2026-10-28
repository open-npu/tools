#!/usr/bin/env python3
"""Phase-1 converter and capacity guard regressions."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from onnx_converter import compute_requant_params, fold_mean_std, fuse_graph
from model_packer import (
    LayerConfig as PackedLayerConfig,
    WGT_LAYOUT_K_MAJOR,
    ref_pooling,
)
from tiling import compute_tiling
from extract_npu1_layers import LayerConfig as ExtractedLayerConfig
from bin2golden import wgt_per_oc_words
from perf_model import (
    HWConfig as PerfHWConfig,
    model_a_layers,
    run_perf_model,
    run_perf_model_with_fusion,
)


def node(op_type, inputs=None, outputs=None, attrs=None, weight=None):
    return SimpleNamespace(
        op_type=op_type,
        name=f"test_{op_type}",
        inputs=inputs or ["x"],
        outputs=outputs or ["y"],
        attrs=attrs or {},
        weight=weight,
        bias=None,
    )


class Phase1GuardTests(unittest.TestCase):
    def test_unsupported_operator_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "Unsupported ONNX operator"):
            fuse_graph([node("Sigmoid")], {})

    def test_general_clip_is_not_miscompiled_as_relu6(self):
        clip = node("Clip", ["x", "min", "max"])
        with self.assertRaisesRegex(ValueError, "Unsupported Clip bounds"):
            fuse_graph([clip], {
                "min": np.array([0.0], dtype=np.float32),
                "max": np.array([5.0], dtype=np.float32),
            })

    def test_non_depthwise_grouped_conv_is_rejected(self):
        conv = node(
            "Conv", ["x", "w"], attrs={"group": 2},
            weight=np.ones((8, 2, 3, 3), dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported grouped Conv"):
            fuse_graph([conv], {"w": conv.weight})

    def test_padded_mean_std_offset_fold_is_rejected(self):
        conv = node(
            "Conv", ["input", "w", "b"], attrs={"pads": [1, 1, 1, 1]},
            weight=np.ones((2, 3, 3, 3), dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "position-dependent"):
            fold_mean_std(
                [conv],
                {"w": conv.weight, "b": np.zeros(2, dtype=np.float32)},
                "input",
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

    def test_int16_tiny_scale_product_keeps_bias(self):
        _, _, bias_q = compute_requant_params(
            1.5e-5, np.array([5.0e-6]), 1.4e-4,
            np.array([0.4], dtype=np.float64),
        )
        self.assertNotEqual(int(bias_q[0]), 0)

    def test_impossible_tile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No feasible"):
            compute_tiling(
                "conv", 1, 1, 30000, 1, 1, 1,
                kernel_h=1, kernel_w=1, elem_size=1,
            )

    def test_activation_double_buffer_does_not_halve_weight_capacity(self):
        common = dict(
            op_type="conv",
            in_h=28, in_w=28, in_c=128,
            out_h=28, out_w=28, out_c=128,
            kernel_h=3, kernel_w=3, elem_size=1,
        )
        single = compute_tiling(double_buffer=False, **common)
        double = compute_tiling(double_buffer=True, **common)
        self.assertEqual(double["tile_oc"], single["tile_oc"])

    def test_global_depthwise_uses_streaming_exception(self):
        result = compute_tiling(
            "dw", 14, 14, 512, 1, 1, 512,
            kernel_h=14, kernel_w=14, elem_size=1,
        )
        self.assertEqual(result["tile_num_h"], 0)
        self.assertEqual(result["tile_num_w"], 0)

    def test_global_pool_uses_streaming_exception(self):
        result = compute_tiling(
            "pool", 7, 7, 256, 1, 1, 256,
            kernel_h=7, kernel_w=7, stride_h=7, stride_w=7,
            elem_size=1,
        )
        self.assertEqual(result["tile_num_h"], 0)
        self.assertEqual(result["tile_num_w"], 0)

    def test_avg_pool_two_elements_matches_rtl_rounding(self):
        cfg = PackedLayerConfig(
            op_type=3, in_h=1, in_w=2, in_c=1,
            out_h=1, out_w=1, out_c=1,
            pool_mode=1, pool_h=1, pool_w=2,
            pool_stride_h=1, pool_stride_w=2,
        )
        positive = ref_pooling(np.array([[[2], [4]]], dtype=np.int8), cfg)
        negative = ref_pooling(np.array([[[-2], [-4]]], dtype=np.int8), cfg)
        self.assertEqual(int(positive[0, 0, 0]), 3)
        self.assertEqual(int(negative[0, 0, 0]), -4)

    def test_npu1_extractor_matches_fixed_descriptor_layout(self):
        packed = PackedLayerConfig(
            op_type=1, data_type=1,
            in_h=5, in_w=10, in_c=8,
            out_h=3, out_w=7, out_c=8,
            pad_top=1, pad_left=2,
            post_ctrl=0xA5, sched_ctrl=0x0B,
            clamp_min=-1234, clamp_max=2345,
            in_zp=-7, wgt_layout=WGT_LAYOUT_K_MAJOR,
        ).pack_fixed()
        decoded, next_offset = ExtractedLayerConfig.unpack(packed, 0)
        self.assertEqual(next_offset, 62)
        self.assertEqual((decoded.in_h, decoded.in_w), (5, 10))
        self.assertEqual((decoded.post_ctrl, decoded.sched_ctrl), (0xA5, 0x0B))
        self.assertEqual((decoded.clamp_min, decoded.clamp_max), (-1234, 2345))
        self.assertEqual(decoded.in_zp, -7)
        self.assertEqual(decoded.wgt_layout, WGT_LAYOUT_K_MAJOR)

    def test_fc_reload_stride_uses_64_lane_row_group(self):
        cfg = PackedLayerConfig(
            op_type=2, data_type=0,
            in_h=1, in_w=1, in_c=512,
            out_h=1, out_w=1, out_c=1000,
            wgt_layout=WGT_LAYOUT_K_MAJOR,
        )
        self.assertEqual(wgt_per_oc_words(cfg), 8192)

    def test_fusion_estimator_rejects_slow_blocks(self):
        layers = model_a_layers()
        hw = PerfHWConfig(elem_size=1, double_buffer=True, adaptive_db=True)
        baseline = run_perf_model(layers, hw)
        optimized, selected = run_perf_model_with_fusion(layers, hw)
        self.assertEqual(selected, [])
        self.assertEqual(
            sum(result["layer_cycles"] for result in optimized),
            sum(result["layer_cycles"] for result in baseline),
        )


if __name__ == "__main__":
    unittest.main()
