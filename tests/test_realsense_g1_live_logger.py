import math
import unittest

import numpy as np

from humanoidverse.tools.realsense_g1_live_logger import (
    CAMERA_DEPTH_LINK_FROM_OPTICAL_QUAT_XYZW,
    TORSO_ORIGIN_IN_WAIST_YAW,
    RealSenseCapture,
    build_capture_payload,
    instinct_nominal_torso_from_optical,
    interpolate_series,
    map_camera_timestamps_to_monotonic,
    quaternion_distance_degrees,
    quaternion_slerp_xyzw,
    rotate_vector_xyzw,
    runtime_frame_schedule,
    torso_pose_from_pelvis,
)


class RealSenseG1LiveLoggerTest(unittest.TestCase):
    def test_instinct_nominal_mount_composes_standard_optical_axes(self):
        np.testing.assert_allclose(
            rotate_vector_xyzw(CAMERA_DEPTH_LINK_FROM_OPTICAL_QUAT_XYZW, [1.0, 0.0, 0.0]),
            [0.0, -1.0, 0.0],
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            rotate_vector_xyzw(CAMERA_DEPTH_LINK_FROM_OPTICAL_QUAT_XYZW, [0.0, 1.0, 0.0]),
            [0.0, 0.0, -1.0],
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            rotate_vector_xyzw(CAMERA_DEPTH_LINK_FROM_OPTICAL_QUAT_XYZW, [0.0, 0.0, 1.0]),
            [1.0, 0.0, 0.0],
            atol=1.0e-8,
        )
        nominal = instinct_nominal_torso_from_optical()
        self.assertEqual(nominal["transform_name"], "T_torso_link_from_camera_optical")
        self.assertEqual(nominal["reference_kind"], "nominal_reference")
        np.testing.assert_allclose(
            nominal["torso_from_camera_optical_translation_m"],
            [0.0487988662332928, 0.015, 0.4378029937970051],
            atol=1.0e-12,
        )
        optical_forward_torso = rotate_vector_xyzw(
            nominal["torso_from_camera_optical_quaternion_xyzw"],
            [0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            optical_forward_torso,
            [0.66913165, 0.00354942, -0.74313541],
            atol=1.0e-8,
        )

    def test_slerp_uses_short_quaternion_arc(self):
        identity = np.asarray([0.0, 0.0, 0.0, 1.0])
        yaw_180 = np.asarray([0.0, 0.0, 1.0, 0.0])
        midpoint = quaternion_slerp_xyzw(identity, -yaw_180, 0.5)
        rotated = rotate_vector_xyzw(midpoint, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotated, [0.0, -1.0, 0.0], atol=1.0e-7)

    def test_interpolation_reports_signed_nearest_dt_and_bracket_span(self):
        values, nearest_dt, bracket_span = interpolate_series(
            np.asarray([100, 200], dtype=np.int64),
            np.asarray([[0.0], [10.0]]),
            np.asarray([125, 175], dtype=np.int64),
        )
        np.testing.assert_allclose(values[:, 0], [2.5, 7.5])
        np.testing.assert_array_equal(nearest_dt, [-25, 25])
        np.testing.assert_array_equal(bracket_span, [100, 100])

    def test_runtime_schedule_selects_latest_60hz_frame_at_50hz(self):
        camera_timestamps = np.arange(121, dtype=np.float64) / 60.0
        indices, runtime_ticks, camera_age = runtime_frame_schedule(camera_timestamps, 50.0)
        self.assertEqual(indices.shape, (101,))
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 120)
        self.assertTrue(np.all(np.diff(indices) >= 1))
        np.testing.assert_allclose(runtime_ticks[[0, -1]], [0.0, 2.0], atol=1.0e-12)
        self.assertGreaterEqual(float(camera_age.min()), -1.0e-12)
        self.assertLessEqual(float(camera_age.max()), 1.0 / 60.0 + 1.0e-12)

    def test_zero_waist_fk_matches_g1_model_origin(self):
        position, quaternion = torso_pose_from_pelvis(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            np.zeros(3),
        )
        np.testing.assert_allclose(position, np.asarray([1.0, 2.0, 3.0]) + TORSO_ORIGIN_IN_WAIST_YAW)
        np.testing.assert_allclose(quaternion, [0.0, 0.0, 0.0, 1.0])

    def test_waist_yaw_rotates_torso_origin_and_orientation(self):
        position, quaternion = torso_pose_from_pelvis(
            np.zeros(3),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            np.asarray([math.pi / 2.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(position, [0.0, -0.0039635, 0.044], atol=1.0e-8)
        rotated = rotate_vector_xyzw(quaternion, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1.0e-8)

    def test_global_camera_clock_maps_exposure_not_receive_time(self):
        targets, metadata = map_camera_timestamps_to_monotonic(
            np.asarray([1000.0, 1100.0]),
            ["timestamp_domain.global_time", "timestamp_domain.global_time"],
            np.asarray([6020000000, 6120000000], dtype=np.int64),
            np.asarray([1020000000, 1120000000], dtype=np.int64),
        )
        np.testing.assert_array_equal(targets, [6000000000, 6100000000])
        self.assertEqual(metadata["mode"], "camera_realtime_plus_host_monotonic_offset")
        self.assertEqual(metadata["receive_residual_median_ns"], 20000000)

    def test_quaternion_distance_is_sign_invariant(self):
        quaternion = np.asarray([0.1, -0.2, 0.3, 0.9])
        self.assertAlmostEqual(quaternion_distance_degrees(quaternion, -quaternion), 0.0)

    def test_capture_payload_keeps_raw_depth_and_replay_pose_contract(self):
        class FakeCamera:
            width = 4
            height = 2
            fps = 10
            intrinsic_matrix = np.asarray([[4.0, 0.0, 1.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
            distortion_coefficients = np.zeros(5)
            distortion_model = "distortion.brown_conrady"
            depth_scale_m = 0.001

            @staticmethod
            def describe():
                return {"serial": "test", "width": 4, "height": 2, "fps": 10}

        targets_mono_ns = [6000000000, 6100000000]
        camera_wall_ns = [1000000000, 1100000000]
        frames = []
        for index, (target_ns, wall_ns) in enumerate(zip(targets_mono_ns, camera_wall_ns)):
            frames.append(
                {
                    "depth": np.full((2, 4), 1000 + index, dtype=np.uint16),
                    "frame_number": index,
                    "camera_timestamp_ms": wall_ns * 1.0e-6,
                    "timestamp_domain": "timestamp_domain.global_time",
                    "wait_start_monotonic_ns": target_ns + 10000000,
                    "receive_monotonic_ns": target_ns + 20000000,
                    "receive_realtime_ns": wall_ns + 20000000,
                    "metadata": [-1] * len(RealSenseCapture.METADATA_NAMES),
                }
            )

        state_times_ns = [5990000000, 6010000000, 6090000000, 6110000000]
        odom_samples = []
        lowstate_samples = []
        torso_imu_samples = []
        for index, monotonic_ns in enumerate(state_times_ns):
            realtime_ns = monotonic_ns - 5000000000
            odom_samples.append(
                {
                    "host_monotonic_ns": monotonic_ns,
                    "host_realtime_ns": realtime_ns,
                    "header_stamp_ns": realtime_ns,
                    "position": [0.01 * index, 0.0, 0.8],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "linear_velocity": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                    "frame_id": "odom",
                    "child_frame_id": "robot_center",
                }
            )
            imu = {
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "gyroscope": [0.0, 0.0, 0.0],
                "accelerometer": [0.0, 0.0, 9.81],
                "rpy": [0.0, 0.0, 0.0],
                "temperature": 30,
            }
            lowstate_samples.append(
                {
                    "host_monotonic_ns": monotonic_ns,
                    "host_realtime_ns": realtime_ns,
                    "tick": index,
                    "mode_pr": 0,
                    "mode_machine": 5,
                    "imu": imu,
                    "joint_position": [0.0] * 29,
                    "joint_velocity": [0.0] * 29,
                    "joint_tau_est": [0.0] * 29,
                }
            )
            torso_imu_samples.append(
                {"host_monotonic_ns": monotonic_ns, "host_realtime_ns": realtime_ns, "imu": imu}
            )

        payload, metadata, calibration = build_capture_payload(
            frames,
            FakeCamera(),
            odom_samples,
            lowstate_samples,
            torso_imu_samples,
            {
                "mount_pos_torso": [0.1, 0.0, 0.2],
                "optical_quat_torso_xyzw": [0.0, 0.0, 0.0, 1.0],
                "source_path": "/measured.json",
            },
            "flat",
            50.0,
        )
        self.assertEqual(payload["depth"].dtype, np.uint16)
        self.assertEqual(payload["depth"].shape, (2, 2, 4))
        self.assertEqual(payload["torso_pos_w"].shape, (2, 3))
        np.testing.assert_array_equal(payload["pelvis_pos_w"], payload["egomotion_pos_w"])
        self.assertEqual(payload["pelvis_heading_quat_w_xyzw"].shape, (2, 4))
        np.testing.assert_allclose(payload["timestamp_s"], [0.0, 0.1])
        np.testing.assert_array_equal(payload["runtime_frame_index"], [0, 0, 0, 0, 0, 1])
        np.testing.assert_allclose(payload["runtime_tick_timestamp_s"], np.arange(6) * 0.02)
        self.assertEqual(metadata["raw_stream_counts"]["lowstate"], 4)
        self.assertIsNone(metadata["odom_frame_contract"]["robot_center_equals_model_pelvis"])
        self.assertEqual(metadata["odom_frame_contract"]["usage"], "relative egomotion proxy")
        self.assertEqual(calibration["intrinsic_matrix"], FakeCamera.intrinsic_matrix.reshape(-1).tolist())


if __name__ == "__main__":
    unittest.main()
