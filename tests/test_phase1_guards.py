#!/usr/bin/env python3
"""Phase-1 converter and capacity guard regressions."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onnx_converter import compute_requant_params, fold_mean_std, fuse_graph
from tiling import compute_tiling


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


if __name__ == "__main__":
    unittest.main()
