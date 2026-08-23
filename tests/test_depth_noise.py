import unittest

import torch

from humanoidverse.perception.depth_noise import (
    DepthNoiseConfig,
    DepthNoisePipeline,
    DropoutNoiseConfig,
    EdgeNoiseConfig,
    ExtrinsicNoiseConfig,
    LatencyNoiseConfig,
    MeasurementNoiseConfig,
    depth_noise_preset,
)


def _inputs(batch_size: int = 2, *, value: float = 1.0, timestamp: float = 0.0):
    depth = torch.full((batch_size, 4, 5), value)
    position = torch.stack((torch.arange(batch_size, dtype=torch.float32), torch.zeros(batch_size), torch.ones(batch_size)), dim=-1)
    quaternion = torch.zeros((batch_size, 4))
    quaternion[:, 3] = 1.0
    timestamps = torch.full((batch_size,), timestamp)
    env_ids = torch.arange(batch_size, dtype=torch.int64)
    return depth, position, quaternion, timestamps, env_ids


def _run(pipeline: DepthNoisePipeline, inputs, reset_mask: torch.Tensor):
    depth, position, quaternion, timestamps, env_ids = inputs
    return pipeline(
        depth_z=depth,
        camera_pos_w=position,
        camera_optical_quat_w=quaternion,
        timestamp_s=timestamps,
        env_ids=env_ids,
        reset_mask=reset_mask,
    )


class DepthNoisePipelineTests(unittest.TestCase):
    def _pipeline(self, config: DepthNoiseConfig, *, seed: int = 123, batch_size: int = 2):
        return DepthNoisePipeline(
            config,
            batch_size=batch_size,
            image_height=4,
            image_width=5,
            device="cpu",
            noise_seed=seed,
        )

    def test_clean_path_preserves_synchronized_tuple(self):
        pipeline = self._pipeline(depth_noise_preset("clean"))
        inputs = _inputs()
        result = _run(pipeline, inputs, torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(result.depth_z, inputs[0], atol=0.0, rtol=0.0)
        torch.testing.assert_close(result.camera_pos_w, inputs[1], atol=0.0, rtol=0.0)
        torch.testing.assert_close(result.camera_optical_quat_w, inputs[2], atol=0.0, rtol=0.0)
        torch.testing.assert_close(result.timestamp_s, inputs[3], atol=0.0, rtol=0.0)
        self.assertEqual(float(result.diagnostics["latency_frames"].max()), 0.0)

    def test_latency_delays_depth_pose_and_timestamp_together(self):
        config = DepthNoiseConfig(latency=LatencyNoiseConfig(min_frames=1, max_frames=1))
        pipeline = self._pipeline(config)
        first = _inputs(value=1.0, timestamp=0.0)
        initial = _run(pipeline, first, torch.ones(2, dtype=torch.bool))
        second = list(_inputs(value=2.0, timestamp=0.02))
        second[1] = second[1] + 10.0
        delayed = _run(pipeline, tuple(second), torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(initial.depth_z, first[0])
        torch.testing.assert_close(delayed.depth_z, first[0])
        torch.testing.assert_close(delayed.camera_pos_w, first[1])
        torch.testing.assert_close(delayed.camera_optical_quat_w, first[2])
        torch.testing.assert_close(delayed.timestamp_s, first[3])

    def test_reset_repeats_current_frame_and_clears_previous_episode(self):
        config = DepthNoiseConfig(latency=LatencyNoiseConfig(min_frames=2, max_frames=2))
        pipeline = self._pipeline(config)
        _run(pipeline, _inputs(value=1.0), torch.ones(2, dtype=torch.bool))
        _run(pipeline, _inputs(value=2.0, timestamp=0.02), torch.zeros(2, dtype=torch.bool))
        reset_inputs = _inputs(value=2.4, timestamp=0.0)
        result = _run(pipeline, reset_inputs, torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(result.depth_z, reset_inputs[0])
        torch.testing.assert_close(result.timestamp_s, reset_inputs[3])

    def test_episode_static_extrinsic_changes_only_after_reset(self):
        config = DepthNoiseConfig(
            extrinsic=ExtrinsicNoiseConfig(
                translation_bound_m=(0.01, 0.01, 0.01),
                rotation_bound_deg=(1.0, 1.0, 1.0),
            )
        )
        pipeline = self._pipeline(config)
        inputs = _inputs()
        first = _run(pipeline, inputs, torch.ones(2, dtype=torch.bool))
        second = _run(pipeline, inputs, torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(first.camera_pos_w, second.camera_pos_w, atol=0.0, rtol=0.0)
        torch.testing.assert_close(first.camera_optical_quat_w, second.camera_optical_quat_w, atol=0.0, rtol=0.0)
        third = _run(pipeline, inputs, torch.ones(2, dtype=torch.bool))
        self.assertFalse(torch.equal(first.camera_pos_w, third.camera_pos_w))
        torch.testing.assert_close(first.depth_z, inputs[0], atol=0.0, rtol=0.0)

    def test_edge_mask_uses_clean_depth_not_measurement_noise(self):
        config = DepthNoiseConfig(
            measurement=MeasurementNoiseConfig(base_std_m=0.2),
            edge=EdgeNoiseConfig(
                depth_threshold_m=0.05,
                corruption_probability=1.0,
                invalid_probability=1.0,
            ),
        )
        pipeline = self._pipeline(config)
        flat = _run(pipeline, _inputs(), torch.ones(2, dtype=torch.bool))
        self.assertTrue(torch.isfinite(flat.depth_z).all())

        pipeline = self._pipeline(config)
        step_inputs = list(_inputs())
        step_inputs[0][:, :, 3:] = 1.2
        step = _run(pipeline, tuple(step_inputs), torch.ones(2, dtype=torch.bool))
        self.assertTrue(torch.isnan(step.depth_z).any())
        self.assertGreater(float(step.diagnostics["clean_edge_fraction"].mean()), 0.0)

    def test_dropout_outputs_nan_instead_of_zero(self):
        config = DepthNoiseConfig(dropout=DropoutNoiseConfig(probability=1.0))
        result = _run(self._pipeline(config), _inputs(), torch.ones(2, dtype=torch.bool))
        self.assertTrue(torch.isnan(result.depth_z).all())

    def test_noise_is_reproducible_and_other_env_reset_does_not_change_env_zero(self):
        config = depth_noise_preset("combined", "nominal")
        left = self._pipeline(config, seed=456)
        right = self._pipeline(config, seed=456)
        initial = _inputs()
        left_first = _run(left, initial, torch.ones(2, dtype=torch.bool))
        right_first = _run(right, initial, torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(left_first.depth_z, right_first.depth_z, equal_nan=True)

        next_inputs = _inputs(value=1.1, timestamp=0.02)
        left_second = _run(left, next_inputs, torch.tensor([False, True]))
        right_second = _run(right, next_inputs, torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(left_second.depth_z[0], right_second.depth_z[0], equal_nan=True)
        torch.testing.assert_close(left_second.camera_pos_w[0], right_second.camera_pos_w[0])

    def test_presets_are_monotonic_and_hash_is_stable(self):
        mild = depth_noise_preset("combined", "mild")
        nominal = depth_noise_preset("combined", "nominal")
        strong = depth_noise_preset("combined", "strong")
        self.assertLess(mild.dropout.probability, nominal.dropout.probability)
        self.assertLess(nominal.dropout.probability, strong.dropout.probability)
        self.assertLess(mild.latency.max_frames, strong.latency.max_frames)
        self.assertEqual(nominal.hash(), depth_noise_preset("combined", "nominal").hash())


if __name__ == "__main__":
    unittest.main()
