import unittest
from types import SimpleNamespace

import torch

from humanoidverse.depth_terrain_evaluation import (
    MetricAccumulator,
    benchmark_environment_steps,
    camera_frame_diagnostics,
    elevated_platform_probe_xy,
    region_metrics,
    stair_edge_mask,
    validate_geometry_sample,
)
from humanoidverse.perception.depth_camera import DepthCameraConfig


class DepthTerrainEvaluationTest(unittest.TestCase):
    def test_camera_diagnostics_prove_downward_axis_and_range_to_optical_z(self):
        camera = DepthCameraConfig(width=3, height=3, down_pitch_deg=48.0)
        intrinsic = camera.intrinsics().float()
        pixel = torch.tensor([1.0, 1.0, 1.0])
        optical_unit = torch.linalg.solve(intrinsic, pixel)
        optical_unit /= torch.linalg.vector_norm(optical_unit)
        ray_range = torch.tensor([[[2.0] * 3] * 3])
        frame = SimpleNamespace(
            range_image=ray_range,
            depth_z=ray_range * optical_unit[2],
        )

        diagnostics = camera_frame_diagnostics(frame, camera)

        self.assertTrue(diagnostics["sample_valid"])
        self.assertGreater(diagnostics["optical_axis_torso"][0], 0.0)
        self.assertLess(diagnostics["optical_axis_torso"][2], 0.0)
        self.assertAlmostEqual(diagnostics["range_to_optical_z_residual_m"], 0.0)

    def test_benchmark_uses_two_warmup_steps_and_requested_timed_steps(self):
        class WrappedEnv:
            def __init__(self):
                self.calls = 0

            def step(self, actions, *, to_numpy):
                self.calls += 1
                self.last_actions = actions
                self.last_to_numpy = to_numpy

        wrapped = WrappedEnv()
        result = benchmark_environment_steps(
            wrapped,
            SimpleNamespace(num_dof=29),
            num_envs=4,
            num_steps=3,
            device="cpu",
        )

        self.assertEqual(wrapped.calls, 5)
        self.assertEqual(tuple(wrapped.last_actions.shape), (4, 29))
        self.assertFalse(wrapped.last_to_numpy)
        self.assertEqual(result["num_steps"], 3)
        self.assertGreater(result["policy_steps_per_second"], 0.0)

    def test_elevated_platform_probe_is_inside_first_raised_band(self):
        core = SimpleNamespace(
            _terrain_patch_size=torch.tensor([14.0, 14.0]),
            _terrain_grid_rows=10,
            _terrain_grid_cols=5,
            terrain_component_names=("flat", "slope", "stairs", "rough", "platforms"),
            config=SimpleNamespace(terrain=SimpleNamespace(platforms=SimpleNamespace(center_width=1.5, band_width=0.8))),
        )

        x, y = elevated_platform_probe_xy(core)

        self.assertAlmostEqual(x, 8.15)
        self.assertAlmostEqual(y, 28.0)

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
