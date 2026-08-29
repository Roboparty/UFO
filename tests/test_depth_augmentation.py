import unittest

import torch

from humanoidverse.perception.depth_augmentation import (
    CameraFrameScheduler,
    DepthTimingAugmentationConfig,
    DepthCalibrationAugmentationConfig,
    LocalCalibrationAugmentation,
    MetricDepthAugmentation,
    MetricDepthAugmentationConfig,
    phase2i_v1_depth_augmentation_config,
    phase2i_v1_timing_augmentation_config,
)


class MetricDepthAugmentationTest(unittest.TestCase):
    def test_range_gate_invalidates_out_of_range_without_saturation(self):
        augmentation = MetricDepthAugmentation(
            MetricDepthAugmentationConfig(max_depth_m=2.0, blur_probability=0.0),
            seed=1,
        )
        depth = torch.tensor([[[0.0, 0.5, 2.0, 2.1, float("nan")]]])
        output, valid, _sigma = augmentation(depth)
        torch.testing.assert_close(output[0, 0, 1:3], torch.tensor([0.5, 2.0]))
        self.assertTrue(torch.equal(valid, torch.tensor([[[False, True, True, False, False]]])))
        self.assertTrue(torch.isnan(output[0, 0, 0]))
        self.assertTrue(torch.isnan(output[0, 0, 3]))

    def test_blur_does_not_expand_visibility(self):
        augmentation = MetricDepthAugmentation(
            MetricDepthAugmentationConfig(max_depth_m=2.0, blur_probability=1.0, sigma_min_px=1.0, sigma_max_px=1.0),
            seed=2,
        )
        depth = torch.full((1, 9, 9), float("nan"))
        depth[:, 3:6, 3:6] = 1.0
        output, valid, sigma = augmentation(depth)
        self.assertTrue(torch.all(sigma > 0.0))
        self.assertTrue(torch.equal(torch.isfinite(output), valid))
        self.assertEqual(int(torch.isfinite(output).sum()), 9)

    def test_blur_is_validity_aware(self):
        augmentation = MetricDepthAugmentation(
            MetricDepthAugmentationConfig(max_depth_m=2.0, blur_probability=1.0, sigma_min_px=1.0, sigma_max_px=1.0),
            seed=3,
        )
        depth = torch.full((1, 7, 7), 1.0)
        depth[:, :, 3:] = float("nan")
        output, valid, _sigma = augmentation(depth)
        self.assertTrue(torch.equal(torch.isfinite(output), valid))
        self.assertTrue(torch.isnan(output[:, :, 3:]).all())
        self.assertTrue(torch.isfinite(output[:, :, :3]).all())

    def test_blur_supports_multiple_environments(self):
        augmentation = MetricDepthAugmentation(
            MetricDepthAugmentationConfig(max_depth_m=2.0, blur_probability=1.0, sigma_min_px=1.0, sigma_max_px=2.0),
            seed=4,
        )
        depth = torch.ones((4, 8, 8))
        depth[1, :, :2] = float("nan")
        output, valid, sigma = augmentation(depth)
        self.assertEqual(tuple(output.shape), tuple(depth.shape))
        self.assertEqual(tuple(valid.shape), tuple(depth.shape))
        self.assertEqual(tuple(sigma.shape), (4,))
        self.assertTrue(torch.equal(torch.isfinite(output), valid))

    def test_depth_dependent_measurement_noise_is_seeded(self):
        config = MetricDepthAugmentationConfig(
            max_depth_m=3.0,
            blur_probability=0.0,
            measurement_base_std_m=0.001,
            measurement_quadratic_std_m_per_m2=0.003,
        )
        depth = torch.tensor([[[0.5, 1.0, 2.0]]]).expand(2, 5, 3).clone()
        first, first_valid, _ = MetricDepthAugmentation(config, seed=9)(depth)
        second, second_valid, _ = MetricDepthAugmentation(config, seed=9)(depth)
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.equal(first_valid, second_valid))
        self.assertFalse(torch.equal(first, depth))

    def test_pixel_dropout_never_expands_visibility(self):
        augmentation = MetricDepthAugmentation(
            MetricDepthAugmentationConfig(
                max_depth_m=2.0,
                blur_probability=0.0,
                pixel_dropout_probability=1.0,
            ),
            seed=10,
        )
        output, valid, _ = augmentation(torch.ones((2, 6, 7)))
        self.assertFalse(valid.any())
        self.assertTrue(torch.isnan(output).all())

    def test_region_dropout_removes_a_contiguous_patch(self):
        augmentation = MetricDepthAugmentation(
            MetricDepthAugmentationConfig(
                max_depth_m=2.0,
                blur_probability=0.0,
                region_dropout_probability=1.0,
                region_dropout_min_height_fraction=0.25,
                region_dropout_max_height_fraction=0.25,
                region_dropout_min_width_fraction=0.25,
                region_dropout_max_width_fraction=0.25,
            ),
            seed=11,
        )
        output, valid, _ = augmentation(torch.ones((1, 20, 20)))
        self.assertEqual(int((~valid).sum()), 25)
        self.assertTrue(torch.equal(torch.isfinite(output), valid))

    def test_30hz_camera_scheduler_runs_on_50hz_control_clock(self):
        scheduler = CameraFrameScheduler(
            DepthTimingAugmentationConfig(camera_frequency_hz=30.0, control_frequency_hz=50.0),
            batch_size=1,
            device="cpu",
            seed=12,
        )
        valid = []
        timestamps = []
        for step in range(50):
            fresh, timestamp, _ = scheduler.step(torch.tensor([step / 50.0]))
            valid.append(bool(fresh))
            timestamps.append(float(timestamp))
        self.assertIn(sum(valid), (30, 31))
        self.assertTrue(all(right > left for left, right in zip(timestamps, timestamps[1:])))

    def test_phase2i_v1_preset_is_synchronous_blur_only(self):
        depth = phase2i_v1_depth_augmentation_config()
        timing = phase2i_v1_timing_augmentation_config()
        self.assertEqual((depth.sigma_min_px, depth.sigma_max_px), (0.0, 3.0))
        self.assertEqual(depth.blur_probability, 1.0)
        self.assertEqual(depth.max_depth_m, 2.0)
        self.assertEqual(depth.measurement_base_std_m, 0.0)
        self.assertEqual(depth.edge_corruption_probability, 0.0)
        self.assertEqual(depth.pixel_dropout_probability, 0.0)
        self.assertEqual(depth.region_dropout_probability, 0.0)
        self.assertEqual(timing.camera_frequency_hz, 50.0)
        self.assertEqual(timing.control_frequency_hz, 50.0)
        self.assertEqual(timing.frame_drop_probability, 0.0)
        self.assertEqual(timing.duplicate_frame_probability, 0.0)
        self.assertEqual(timing.timestamp_jitter_s, 0.0)

    def test_dropped_and_duplicate_packets_are_not_valid_history_frames(self):
        for field in ("frame_drop_probability", "duplicate_frame_probability"):
            config = DepthTimingAugmentationConfig(
                **{field: 1.0},
            )
            scheduler = CameraFrameScheduler(config, batch_size=2, device="cpu", seed=13)
            fresh, _timestamp, diagnostics = scheduler.step(torch.zeros(2))
            self.assertFalse(fresh.any())
            self.assertTrue(diagnostics["dropped" if field.startswith("frame_drop") else "duplicated"].all())

    def test_calibration_dr_is_episode_static_and_reset_local(self):
        augmentation = LocalCalibrationAugmentation(
            DepthCalibrationAugmentationConfig(
                focal_scale_bound=0.01,
                principal_point_bound_px=(0.5, 0.5),
                translation_bound_m=(0.005, 0.005, 0.005),
                rotation_bound_deg=(0.5, 0.5, 0.5),
            ),
            intrinsic_matrix=torch.tensor([[50.0, 0.0, 31.5], [0.0, 50.0, 17.5], [0.0, 0.0, 1.0]]),
            camera_pos_torso=(0.0, 0.0, 0.4),
            camera_optical_quat_torso_xyzw=(0.0, 0.0, 0.0, 1.0),
            batch_size=2,
            device="cpu",
            seed=14,
        )
        first_intrinsics = augmentation.intrinsics.clone()
        first_position = augmentation.camera_pos_torso.clone()
        augmentation.reset(torch.tensor([True, False]))
        self.assertFalse(torch.equal(augmentation.intrinsics[0], first_intrinsics[0]))
        self.assertFalse(torch.equal(augmentation.camera_pos_torso[0], first_position[0]))
        torch.testing.assert_close(augmentation.intrinsics[1], first_intrinsics[1])
        torch.testing.assert_close(augmentation.camera_pos_torso[1], first_position[1])


if __name__ == "__main__":
    unittest.main()
