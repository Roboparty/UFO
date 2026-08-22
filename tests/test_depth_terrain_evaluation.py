import unittest

import torch

from humanoidverse.depth_terrain_evaluation import (
    MetricAccumulator,
    region_metrics,
    stair_edge_mask,
    validate_geometry_sample,
)


class DepthTerrainEvaluationTest(unittest.TestCase):
    def test_stair_edge_marks_both_sides_of_discontinuity(self):
        gt = torch.ones((1, 21, 13))
        gt[:, 10:, :] = 0.82
        edge = stair_edge_mask(gt).reshape(1, 21, 13)
        self.assertTrue(edge[:, 9, :].all())
        self.assertTrue(edge[:, 10, :].all())
        self.assertFalse(edge[:, 8, :].any())
        self.assertFalse(edge[:, 11, :].any())

    def test_metric_accumulator_ignores_invisible_and_nonfinite_gt(self):
        predicted = torch.ones((2, 273))
        gt = torch.ones((2, 273))
        predicted[0, 58] = 1.2
        visible = torch.ones((2, 273), dtype=torch.bool)
        visible[1, 0] = False
        gt[1, 1] = float("nan")
        accumulator = MetricAccumulator(torch.device("cpu"))
        accumulator.update(predicted, visible, gt)
        summary = accumulator.summary()
        self.assertEqual(summary["samples"], 2)
        self.assertAlmostEqual(summary["center_visibility"], 1.0)
        self.assertAlmostEqual(summary["center_mae_m"], 0.1, places=6)
        self.assertLess(summary["visible_fraction"], 1.0)

    def test_region_metrics_separate_rear_underfoot_and_forward(self):
        predicted = torch.ones((1, 273))
        gt = torch.ones((1, 273))
        visible = torch.ones((1, 273), dtype=torch.bool)
        metrics = region_metrics(predicted, visible, gt)
        self.assertEqual(set(metrics), {"rear", "underfoot", "forward"})
        for values in metrics.values():
            self.assertEqual(values["visible_fraction"], 1.0)
            self.assertEqual(values["mae_m"], 0.0)

    def test_geometry_contract_rejects_mask_mismatch(self):
        class Frame:
            valid = torch.ones((1, 1, 1), dtype=torch.bool)
            depth_z = torch.ones((1, 1, 1))
            camera_pos_w = torch.zeros((1, 3))
            camera_optical_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

        predicted = torch.ones((1, 273))
        gt = torch.ones((1, 273))
        visible = torch.ones((1, 273), dtype=torch.bool)
        visible[0, 0] = False
        with self.assertRaises(RuntimeError):
            validate_geometry_sample(
                frame=Frame(),
                predicted=predicted,
                visible=visible,
                gt=gt,
                root_state=torch.zeros((1, 13)),
            )


if __name__ == "__main__":
    unittest.main()
