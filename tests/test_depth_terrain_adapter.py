import math
import unittest

import torch

from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter
from humanoidverse.terrains.terrain_observation import RobotCentricGridPatternCfg


def quat_axis_angle(axis, angle):
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    return torch.cat((axis * math.sin(angle / 2.0), torch.tensor([math.cos(angle / 2.0)], dtype=torch.float64)))


def quat_mul(left, right):
    lx, ly, lz, lw = left.unbind()
    rx, ry, rz, rw = right.unbind()
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def rotate(quaternion, vectors):
    quaternion = quaternion / torch.linalg.vector_norm(quaternion)
    xyz = quaternion[:3].expand_as(vectors)
    cross = 2.0 * torch.cross(xyz, vectors, dim=-1)
    return vectors + quaternion[3] * cross + torch.cross(xyz, cross, dim=-1)


class DepthTerrainAdapterTest(unittest.TestCase):
    def setUp(self):
        self.height = 13
        self.width = 21
        self.K = torch.tensor([[10.0, 0.0, 4.0], [0.0, 10.0, 6.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        self.adapter = DepthTerrainAdapter(self.K, self.height, self.width).double()
        self.camera_down = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        self.identity = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)

    def render_plane(self, camera_pos, camera_quat, a=0.0, b=0.0, c=0.0):
        rays_camera = self.adapter.pixel_ray_lut.double().reshape(-1, 3)
        directions = rotate(camera_quat, rays_camera)
        numerator = a * camera_pos[0] + b * camera_pos[1] + c - camera_pos[2]
        denominator = directions[:, 2] - a * directions[:, 0] - b * directions[:, 1]
        depth = numerator / denominator
        depth[(depth <= 0.0) | ~torch.isfinite(depth)] = float("nan")
        return depth.reshape(1, self.height, self.width)

    def project(self, depth, camera_pos=None, camera_quat=None, pelvis_pos=None, heading=None):
        camera_pos = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) if camera_pos is None else camera_pos
        camera_quat = self.camera_down if camera_quat is None else camera_quat
        pelvis_pos = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) if pelvis_pos is None else pelvis_pos
        heading = self.identity if heading is None else heading
        return self.adapter(
            depth,
            camera_pos.unsqueeze(0),
            camera_quat.unsqueeze(0),
            pelvis_pos.unsqueeze(0),
            heading.unsqueeze(0),
        )

    def test_grid_shape_center_and_flatten_order_match_pbfm(self):
        expected, _ = RobotCentricGridPatternCfg().generate_rays(None, "cpu")
        self.assertEqual(self.adapter.GRID_SHAPE, (21, 13))
        self.assertEqual(self.adapter.GRID_DIMENSION, 273)
        self.assertEqual(self.adapter.CENTER_INDEX, 4 * 13 + 6)
        self.assertEqual(self.adapter.CENTER_INDEX, 58)
        x = torch.linspace(-0.4, 1.6, 21)
        y = torch.linspace(-0.6, 0.6, 13)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        actual = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros(273)), dim=-1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[58], torch.zeros(3))

    def test_flat_and_elevated_flat(self):
        for ground_z, expected_clearance in ((0.0, 1.0), (0.35, 0.65)):
            depth = self.render_plane(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), self.camera_down, c=ground_z)
            clearance, visible = self.project(depth)
            self.assertGreater(int(visible.sum()), 100)
            torch.testing.assert_close(clearance[visible], torch.full_like(clearance[visible], expected_clearance), atol=1e-10, rtol=1e-10)

    def test_positive_and_negative_slope(self):
        offsets, _ = RobotCentricGridPatternCfg().generate_rays(None, "cpu")
        for slope in (-0.1, 0.1):
            depth = self.render_plane(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), self.camera_down, a=slope)
            clearance, visible = self.project(depth)
            expected = 1.0 - slope * offsets[:, 0].double()
            torch.testing.assert_close(clearance[0, visible[0]], expected[visible[0]], atol=0.012, rtol=0.0)

    def test_single_stair_edge(self):
        depth = self.render_plane(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), self.camera_down)
        columns = torch.arange(self.width).view(1, 1, -1)
        depth = torch.where(columns >= 10, depth - 0.2, depth)
        clearance, visible = self.project(depth)
        self.assertTrue(visible.any())
        values = clearance[visible]
        self.assertTrue(torch.any(torch.isclose(values, torch.tensor(1.0, dtype=values.dtype))))
        self.assertTrue(torch.any(torch.isclose(values, torch.tensor(0.8, dtype=values.dtype))))
        low_surface_cells = torch.where(visible[0] & torch.isclose(clearance[0], torch.tensor(1.0, dtype=clearance.dtype)))[0]
        high_surface_cells = torch.where(visible[0] & torch.isclose(clearance[0], torch.tensor(0.8, dtype=clearance.dtype)))[0]
        self.assertLess(float(low_surface_cells.float().median()), float(high_surface_cells.float().median()))

    def test_yaw_invariance(self):
        depth = self.render_plane(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), self.camera_down)
        baseline, baseline_visible = self.project(depth)
        yaw = quat_axis_angle([0.0, 0.0, 1.0], 0.73)
        transformed, transformed_visible = self.project(
            depth,
            camera_quat=quat_mul(yaw, self.camera_down),
            heading=yaw,
        )
        self.assertTrue(torch.equal(baseline_visible, transformed_visible))
        torch.testing.assert_close(baseline[baseline_visible], transformed[transformed_visible], atol=1e-10, rtol=1e-10)

    def test_world_z_translation_invariance(self):
        camera_pos = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        depth = self.render_plane(camera_pos, self.camera_down)
        baseline, baseline_visible = self.project(depth)
        delta_z = 4.0
        transformed, transformed_visible = self.project(
            depth,
            camera_pos=camera_pos + torch.tensor([0.0, 0.0, delta_z], dtype=torch.float64),
            pelvis_pos=torch.tensor([0.0, 0.0, 1.0 + delta_z], dtype=torch.float64),
        )
        self.assertTrue(torch.equal(baseline_visible, transformed_visible))
        torch.testing.assert_close(baseline[baseline_visible], transformed[transformed_visible], atol=1e-10, rtol=1e-10)

    def test_camera_pitch(self):
        pitch = quat_axis_angle([0.0, 1.0, 0.0], 0.22)
        camera_quat = quat_mul(pitch, self.camera_down)
        camera_pos = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        depth = self.render_plane(camera_pos, camera_quat)
        clearance, visible = self.project(depth, camera_pos=camera_pos, camera_quat=camera_quat)
        self.assertGreater(int(visible.sum()), 150)
        torch.testing.assert_close(clearance[visible], torch.ones_like(clearance[visible]), atol=1e-10, rtol=1e-10)

    def test_off_center_principal_point_and_non_square_focal_lengths(self):
        K = torch.tensor([[13.0, 0.0, 7.3], [0.0, 8.0, 4.2], [0.0, 0.0, 1.0]], dtype=torch.float64)
        adapter = DepthTerrainAdapter(K, 11, 17).double()
        self.assertAlmostEqual(float(adapter.pixel_ray_lut[4, 7, 0]), (7.0 - 7.3) / 13.0)
        self.assertAlmostEqual(float(adapter.pixel_ray_lut[4, 7, 1]), (4.0 - 4.2) / 8.0)
        depth = torch.ones((1, 11, 17), dtype=torch.float64)
        clearance, visible = adapter(
            depth,
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
            self.camera_down.unsqueeze(0),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
            self.identity.unsqueeze(0),
        )
        torch.testing.assert_close(clearance[visible], torch.ones_like(clearance[visible]))

    def test_skew_intrinsics_use_full_inverse_matrix(self):
        K = torch.tensor([[13.0, 2.5, 7.3], [0.0, 8.0, 4.2], [0.0, 0.0, 1.0]], dtype=torch.float64)
        adapter = DepthTerrainAdapter(K, 11, 17)
        pixel = torch.tensor([7.0, 4.0, 1.0], dtype=torch.float64)
        expected = torch.linalg.solve(K, pixel)
        expected = expected / expected[2]
        torch.testing.assert_close(adapter.pixel_ray_lut[4, 7], expected)
        naive_x = (pixel[0] - K[0, 2]) / K[0, 0]
        self.assertNotAlmostEqual(float(expected[0]), float(naive_x), places=6)

    def test_surface_above_pelvis_clamps_clearance_to_zero(self):
        adapter = DepthTerrainAdapter(torch.eye(3), 1, 1).double()
        clearance, visible = adapter(
            torch.tensor([[[0.5]]], dtype=torch.float64),
            torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64),
            self.camera_down.unsqueeze(0),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
            self.identity.unsqueeze(0),
        )
        self.assertTrue(visible[0, 58])
        self.assertEqual(float(clearance[0, 58]), 0.0)

    def test_invalid_live_poses_fail_fast(self):
        depth = torch.ones((1, self.height, self.width), dtype=torch.float64)
        valid_position = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        valid_quaternion = self.identity.unsqueeze(0)
        cases = {
            "zero optical quaternion": (
                valid_position,
                torch.zeros((1, 4), dtype=torch.float64),
                valid_position,
                valid_quaternion,
            ),
            "non-finite heading quaternion": (
                valid_position,
                valid_quaternion,
                valid_position,
                torch.tensor([[0.0, 0.0, float("nan"), 1.0]], dtype=torch.float64),
            ),
            "non-finite camera position": (
                torch.tensor([[0.0, float("inf"), 1.0]], dtype=torch.float64),
                valid_quaternion,
                valid_position,
                valid_quaternion,
            ),
        }
        for name, poses in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.adapter(depth, *poses)

    def test_out_of_grid_is_rejected_and_invisible_is_nan(self):
        adapter = DepthTerrainAdapter(torch.eye(3), 1, 1).double()
        clearance, visible = adapter(
            torch.ones((1, 1, 1), dtype=torch.float64),
            torch.tensor([[3.0, 0.0, 1.0]], dtype=torch.float64),
            self.camera_down.unsqueeze(0),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
            self.identity.unsqueeze(0),
        )
        self.assertFalse(visible.any())
        self.assertTrue(torch.isnan(clearance).all())

    def test_multiple_points_in_cell_choose_highest_surface(self):
        K = torch.tensor([[100.0, 0.0, 0.5], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
        adapter = DepthTerrainAdapter(K, 1, 2).double()
        clearance, visible = adapter(
            torch.tensor([[[0.8, 0.6]]], dtype=torch.float64),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
            self.camera_down.unsqueeze(0),
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
            self.identity.unsqueeze(0),
        )
        self.assertEqual(int(visible.sum()), 1)
        self.assertTrue(visible[0, 58])
        self.assertAlmostEqual(float(clearance[0, 58]), 0.6, places=10)
        self.assertTrue(torch.isnan(clearance[~visible]).all())


if __name__ == "__main__":
    unittest.main()
