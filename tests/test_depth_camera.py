import math
import unittest

import torch

from humanoidverse.perception.depth_camera import (
    DepthCameraConfig,
    FullIntrinsicsDepthPatternCfg,
    InstallationRandomizedRayCastSensor,
    convert_camera_quaternion_to_optical_xyzw,
    depth_frame_from_raycast,
    intrinsic_matrix_from_fov,
    make_depth_camera_sensor_cfg,
    rotate_xyzw,
    source_from_optical_rotation,
    torso_from_optical_rotation,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class DepthCameraConventionTest(unittest.TestCase):
    def test_quaternion_order_round_trip(self):
        wxyz = torch.tensor([[0.5, -0.5, 0.5, -0.5]])
        torch.testing.assert_close(xyzw_to_wxyz(wxyz_to_xyzw(wxyz)), wxyz)

    def test_opengl_and_mujoco_to_optical(self):
        for convention in ("opengl", "mujoco"):
            rotation = source_from_optical_rotation(convention)
            torch.testing.assert_close(rotation[:, 0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
            torch.testing.assert_close(rotation[:, 1], torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64))
            torch.testing.assert_close(rotation[:, 2], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64))

    def test_ros_is_optical(self):
        torch.testing.assert_close(source_from_optical_rotation("ros"), torch.eye(3, dtype=torch.float64))

    def test_world_flu_to_optical(self):
        rotation = source_from_optical_rotation("world")
        torch.testing.assert_close(rotation[:, 0], torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64))
        torch.testing.assert_close(rotation[:, 1], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64))
        torch.testing.assert_close(rotation[:, 2], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    def test_identity_flu_camera_points_forward_and_down(self):
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        optical_quat = convert_camera_quaternion_to_optical_xyzw(identity, convention="world")
        axes = rotate_xyzw(optical_quat, torch.eye(3, dtype=torch.float64).unsqueeze(0))[0]
        torch.testing.assert_close(axes[2], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    def test_downward_mount_basis_is_orthonormal(self):
        rotation = torso_from_optical_rotation(48.0)
        torch.testing.assert_close(rotation.T @ rotation, torch.eye(3, dtype=torch.float64), atol=1e-12, rtol=0.0)
        self.assertLess(float(rotation[2, 2]), 0.0)
        self.assertGreater(float(rotation[0, 2]), 0.0)

    def test_full_k_center_ray_and_range_to_optical_z(self):
        camera = DepthCameraConfig(
            width=3,
            height=3,
            intrinsic_matrix=(2.0, 0.2, 1.5, 0.0, 2.0, 1.5, 0.0, 0.0, 1.0),
            down_pitch_deg=0.0,
        )
        _offsets, directions = FullIntrinsicsDepthPatternCfg(camera).generate_rays(None, "cpu")
        center = directions[4].double()
        torch.testing.assert_close(center, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        intrinsic = camera.intrinsics()
        corner = torch.linalg.solve(intrinsic, torch.tensor([0.5, 0.5, 1.0], dtype=torch.float64))
        corner = corner / torch.linalg.vector_norm(corner)
        range_value = 2.0
        optical_z = range_value * corner[2]
        self.assertLess(float(optical_z), range_value)
        self.assertGreater(float(optical_z), 0.0)

    def test_default_intrinsics_match_requested_fov(self):
        matrix = intrinsic_matrix_from_fov(width=64, height=36, horizontal_fov_deg=89.0, vertical_fov_deg=58.0)
        self.assertEqual(float(matrix[0, 2]), 32.0)
        self.assertEqual(float(matrix[1, 2]), 18.0)
        horizontal = 2.0 * math.degrees(math.atan(32.0 / float(matrix[0, 0])))
        vertical = 2.0 * math.degrees(math.atan(18.0 / float(matrix[1, 1])))
        self.assertAlmostEqual(horizontal, 89.0)
        self.assertAlmostEqual(vertical, 58.0)

    def test_every_ray_matches_upstream_intrinsics_pixel_center_formula(self):
        camera = DepthCameraConfig(
            width=64,
            height=36,
            horizontal_fov_deg=89.51,
            vertical_fov_deg=58.29,
            down_pitch_deg=0.0,
            mount_pos_torso=(0.1, -0.2, 0.3),
        )
        offsets, actual = FullIntrinsicsDepthPatternCfg(camera).generate_rays(None, "cpu")
        grid = torch.meshgrid(
            torch.arange(camera.width, dtype=torch.int32),
            torch.arange(camera.height, dtype=torch.int32),
            indexing="xy",
        )
        pixels = torch.vstack([axis.ravel() for axis in grid]).T
        pixels = torch.hstack((pixels, torch.ones((len(pixels), 1)))).float()
        pixels += torch.tensor([[0.5, 0.5, 0.0]])
        optical = (torch.linalg.inv(camera.intrinsics().float()) @ pixels.T).T
        # InstinctMJ converts optical (+x right, +y down, +z forward) to
        # world-camera/FLU (+x forward, +y left, +z up).
        expected = optical[:, [2, 0, 1]] * torch.tensor([1.0, -1.0, -1.0])
        expected /= expected.norm(dim=1, keepdim=True)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-6)
        torch.testing.assert_close(
            offsets,
            torch.tensor(camera.mount_pos_torso).expand(camera.width * camera.height, 3),
            rtol=0.0,
            atol=0.0,
        )

    def test_direct_camera_base_alignment_retains_full_torso_rotation(self):
        camera = DepthCameraConfig()
        sensor_cfg = make_depth_camera_sensor_cfg(camera, ray_alignment="base")
        assert sensor_cfg.ray_alignment == "base"
        sensor = sensor_cfg.build()
        roll, pitch = math.radians(21.0), math.radians(-17.0)
        rotation_x = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, math.cos(roll), -math.sin(roll)], [0.0, math.sin(roll), math.cos(roll)]]
        )
        rotation_y = torch.tensor(
            [[math.cos(pitch), 0.0, math.sin(pitch)], [0.0, 1.0, 0.0], [-math.sin(pitch), 0.0, math.cos(pitch)]]
        )
        frame_rotation = (rotation_x @ rotation_y).unsqueeze(0)
        actual = sensor._compute_alignment_rotation(frame_rotation)
        torch.testing.assert_close(actual, frame_rotation, rtol=0.0, atol=0.0)

    def test_invalid_range_and_intrinsics_fail_fast(self):
        with self.assertRaises(ValueError):
            DepthCameraConfig(min_range=2.0, max_range=1.0).validate()
        with self.assertRaises(ValueError):
            DepthCameraConfig(intrinsic_matrix=(1.0,) * 8).validate()

    def test_per_episode_installation_randomization_matches_rp1_bounds(self):
        position_error = 0.05
        angle_error = math.radians(5.0)
        sensor_cfg = make_depth_camera_sensor_cfg(
            DepthCameraConfig(),
            ray_alignment="base",
            position_error=position_error,
            angle_error_rad=angle_error,
        )
        sensor = sensor_cfg.build()
        self.assertIsInstance(sensor, InstallationRandomizedRayCastSensor)
        sensor._position_delta = torch.zeros((256, 3))
        sensor._rotation_delta = torch.eye(3).expand(256, 3, 3).clone()

        torch.manual_seed(19)
        sensor.randomize_installation(torch.arange(256))
        self.assertLessEqual(float(sensor._position_delta.abs().max()), position_error)
        self.assertGreater(float(sensor._position_delta.abs().max()), 0.04)
        identity = torch.eye(3).expand(256, 3, 3)
        torch.testing.assert_close(
            sensor._rotation_delta.transpose(-1, -2) @ sensor._rotation_delta,
            identity,
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            torch.linalg.det(sensor._rotation_delta),
            torch.ones(256),
            rtol=0.0,
            atol=1.0e-6,
        )

        first_env = sensor._position_delta[0].clone()
        sensor.randomize_installation(torch.tensor([1]))
        torch.testing.assert_close(sensor._position_delta[0], first_env)

    def test_configurable_mount_orientation(self):
        identity = DepthCameraConfig(optical_quat_torso_xyzw=(0.0, 0.0, 0.0, 1.0))
        torch.testing.assert_close(identity.torso_from_optical(), torch.eye(3, dtype=torch.float64))

    def test_raycast_miss_and_outside_range_become_nan(self):
        class Data:
            distances = torch.tensor([[-1.0, 0.05, 1.0, 3.0]])
            frame_pos_w = torch.zeros((1, 1, 3))
            frame_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])

        class Sensor:
            data = Data()

        camera = DepthCameraConfig(
            width=2,
            height=2,
            intrinsic_matrix=(2.0, 0.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0),
            min_range=0.1,
            max_range=2.5,
        )
        frame = depth_frame_from_raycast(Sensor(), camera)
        self.assertEqual(int(frame.valid.sum()), 1)
        self.assertTrue(torch.isnan(frame.depth_z[0, 0, 0]))
        self.assertTrue(torch.isnan(frame.depth_z[0, 0, 1]))
        self.assertTrue(torch.isfinite(frame.depth_z[0, 1, 0]))
        self.assertTrue(torch.isnan(frame.depth_z[0, 1, 1]))


if __name__ == "__main__":
    unittest.main()
