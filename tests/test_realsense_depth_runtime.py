import unittest

import torch

from humanoidverse.perception.realsense_depth_runtime import (
    RealSenseCalibration,
    RealSenseDepthRuntime,
    depth_to_meters,
    resize_full_fov_depth,
)


class RealSenseDepthRuntimeTest(unittest.TestCase):
    def test_calibration_scales_full_fov_intrinsics(self):
        calibration = RealSenseCalibration(
            native_width=4,
            native_height=2,
            target_width=2,
            target_height=1,
            intrinsic_matrix=(4.0, 0.2, 1.5, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0),
        )
        target = calibration.target_intrinsics()
        torch.testing.assert_close(
            target,
            torch.tensor([[2.0, 0.1, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
        )

    def test_invalid_depth_remains_nan_after_full_fov_resize(self):
        depth = torch.tensor([[1.0, float("nan")], [float("nan"), 2.0]])
        resized = resize_full_fov_depth(depth, target_height=1, target_width=1)
        self.assertAlmostEqual(float(resized), 1.5)
        invalid = resize_full_fov_depth(torch.full((2, 2), float("nan")), target_height=1, target_width=1)
        self.assertTrue(torch.isnan(invalid))

    def test_native_depth_scale(self):
        depth = torch.tensor([[0, 1000, 2000]], dtype=torch.uint16)
        meters = depth_to_meters(depth, depth_scale_m=0.001)
        self.assertTrue(torch.isnan(meters[0, 0]))
        torch.testing.assert_close(meters[0, 1:], torch.tensor([1.0, 2.0]))

    def test_runtime_preserves_partial_map_visibility_contract(self):
        calibration = RealSenseCalibration(
            native_width=4,
            native_height=2,
            target_width=4,
            target_height=2,
            intrinsic_matrix=(4.0, 0.0, 1.5, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0),
            depth_scale_m=1.0,
        )
        runtime = RealSenseDepthRuntime(calibration=calibration, perception_checkpoint=None, device="cpu")
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        output = runtime.step(
            torch.ones((1, 2, 4)),
            torso_pos_w=torch.zeros((1, 3)),
            torso_quat_w=identity,
            pelvis_pos_w=torch.tensor([[0.0, 0.0, 1.0]]),
            pelvis_heading_quat_w=identity,
            timestamp_s=torch.zeros(1),
            reset_mask=torch.ones(1, dtype=torch.bool),
        )
        self.assertEqual(tuple(output.partial_map.shape), (1, 273))
        self.assertEqual(tuple(output.visible_mask.shape), (1, 273))
        self.assertEqual(tuple(output.terrain_actor.shape), (1, 273))
        self.assertTrue(torch.equal(output.visible_mask, torch.isfinite(output.partial_map)))
        runtime.reset()


if __name__ == "__main__":
    unittest.main()
