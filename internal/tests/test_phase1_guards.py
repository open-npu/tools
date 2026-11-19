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
from super_tile import SuperLayer, model_a_tile_for, plan_super_tile
from extract_npu1_layers import LayerConfig as ExtractedLayerConfig
from bin2golden import use_group_major, wgt_per_oc_words
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

    def test_single_buffer_spatial_tile_requires_per_tile_store(self):
        result = compute_tiling(
            "conv", 28, 28, 512, 28, 28, 512,
            kernel_h=3, kernel_w=3, elem_size=1,
            double_buffer=False,
        )
        self.assertGreater(result["tile_num_h"] * result["tile_num_w"], 1)
        self.assertTrue(result["per_tile_store_en"])

    def test_int16_pool_avoids_partial_tail_tiles(self):
        result = compute_tiling(
            "pool", 26, 26, 128, 13, 13, 128,
            kernel_h=2, kernel_w=2, stride_h=2, stride_w=2,
            elem_size=2,
        )
        self.assertEqual(13 % result["tile_h"], 0)
        self.assertEqual(13 % result["tile_w"], 0)
        self.assertTrue(result["per_tile_store_en"])

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

    def test_tiled_dense_conv_enables_cross_tile_weight_reuse(self):
        cfg = PackedLayerConfig(
            op_type=0, data_type=0,
            in_h=28, in_w=28, in_c=256,
            out_h=28, out_w=28, out_c=256,
            kernel_h=3, kernel_w=3,
            tile_h=5, tile_w=6, tile_num_h=6, tile_num_w=5,
            sched_ctrl=0x10, wgt_layout=WGT_LAYOUT_K_MAJOR,
        )
        self.assertTrue(use_group_major(cfg, per_tile_store=True))

    def test_unaligned_partial_channel_store_keeps_tile_major(self):
        cfg = PackedLayerConfig(
            op_type=0, data_type=0,
            in_h=8, in_w=8, in_c=20000,
            out_h=8, out_w=8, out_c=6,
            kernel_h=1, kernel_w=1,
            tile_h=1, tile_w=1, tile_num_h=8, tile_num_w=8,
            sched_ctrl=0x10, wgt_layout=WGT_LAYOUT_K_MAJOR,
        )
        # Only two INT8 channels fit in one resident group; a group slice is
        # not a whole DMA word and would require per-pixel read-modify-write.
        self.assertFalse(use_group_major(cfg, per_tile_store=True))

    def test_double_buffer_keeps_tile_major_until_group_reuse_is_validated(self):
        cfg = PackedLayerConfig(
            op_type=0, data_type=0,
            in_h=28, in_w=28, in_c=256,
            out_h=28, out_w=28, out_c=256,
            kernel_h=3, kernel_w=3,
            tile_h=5, tile_w=6, tile_num_h=6, tile_num_w=5,
            sched_ctrl=0x11, wgt_layout=WGT_LAYOUT_K_MAJOR,
        )
        self.assertFalse(use_group_major(cfg, per_tile_store=True))

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

    def test_model_a_l52_l54_super_tile_capacity(self):
        layers = [
            SuperLayer("conv", 14, 14, 96, 14, 14, 192),
            SuperLayer("dw", 14, 14, 192, 14, 14, 192,
                       kernel_h=3, kernel_w=3),
            SuperLayer("conv", 14, 14, 192, 14, 14, 96),
        ]
        plan = plan_super_tile(layers, tile_h=14, tile_w=12)
        self.assertEqual((plan.tile_rows, plan.tile_cols), (1, 2))
        self.assertEqual(plan.weight_bytes, 38592)
        self.assertEqual(plan.param_bytes, 7680)
        self.assertLessEqual(plan.ifm_peak_bytes, 32 * 1024)
        self.assertLessEqual(plan.ofm_peak_bytes, 48 * 1024)

    def test_all_model_a_strict_super_geometries_fit(self):
        for spatial, in_c, mid_c, expected_grid in (
                (56, 56, 112, (2, 6)),
                (28, 96, 192, (1, 5)),
                (14, 96, 192, (1, 2))):
            th, tw = model_a_tile_for(spatial)
            layers = [
                SuperLayer("conv", spatial, spatial, in_c,
                           spatial, spatial, mid_c),
                SuperLayer("dw", spatial, spatial, mid_c,
                           spatial, spatial, mid_c, kernel_h=3, kernel_w=3),
                SuperLayer("conv", spatial, spatial, mid_c,
                           spatial, spatial, in_c),
            ]
            plan = plan_super_tile(layers, tile_h=th, tile_w=tw)
            self.assertEqual((plan.tile_rows, plan.tile_cols), expected_grid)

    def test_super_tile_rejects_db_and_capacity_overflow(self):
        db_layers = [
            SuperLayer("conv", 14, 14, 96, 14, 14, 192, db_en=True),
            SuperLayer("dw", 14, 14, 192, 14, 14, 192,
                       kernel_h=3, kernel_w=3),
            SuperLayer("conv", 14, 14, 192, 14, 14, 96),
        ]
        with self.assertRaisesRegex(ValueError, "single-buffer"):
            plan_super_tile(db_layers)

        too_wide = [
            SuperLayer("conv", 14, 14, 1024, 14, 14, 1024),
            SuperLayer("dw", 14, 14, 1024, 14, 14, 1024,
                       kernel_h=3, kernel_w=3),
            SuperLayer("conv", 14, 14, 1024, 14, 14, 1024),
        ]
        with self.assertRaisesRegex(ValueError, "capacity exceeded"):
            plan_super_tile(too_wide)


if __name__ == "__main__":
    unittest.main()
