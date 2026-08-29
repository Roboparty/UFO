import unittest

import torch

from humanoidverse.perception.depth_camera import (
    DepthCameraConfig,
    quaternion_multiply_xyzw,
    rotate_xyzw,
    rotation_matrix_to_xyzw,
)
from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter
from humanoidverse.perception.local_depth_terrain_adapter import (
    LocalDepthTerrainAdapter,
    g1_torso_pose_in_pelvis,
    gravity_aligned_basis_in_pelvis,
)
from humanoidverse.utils.torch_utils import calc_heading_quat, quat_from_euler_xyz


class LocalDepthTerrainAdapterTest(unittest.TestCase):
    def setUp(self):
        self.camera = DepthCameraConfig(width=21, height=13)
        self.intrinsics = self.camera.intrinsics()
        self.camera_quat_torso = rotation_matrix_to_xyzw(self.camera.torso_from_optical())
        self.local = LocalDepthTerrainAdapter(
            self.intrinsics,
            self.camera.height,
            self.camera.width,
            camera_pos_torso=self.camera.mount_pos_torso,
            camera_optical_quat_torso_xyzw=tuple(float(value) for value in self.camera_quat_torso),
        ).double()
        self.world = DepthTerrainAdapter(
            self.intrinsics,
            self.camera.height,
            self.camera.width,
        ).double()

    def test_g1_waist_fk_zero_and_yaw(self):
        waist = torch.tensor([[0.0, 0.0, 0.0], [torch.pi / 2.0, 0.0, 0.0]], dtype=torch.float64)
        position, orientation = g1_torso_pose_in_pelvis(waist)

        torch.testing.assert_close(position[0], torch.tensor([-0.0039635, 0.0, 0.044], dtype=torch.float64))
        torch.testing.assert_close(position[1], torch.tensor([0.0, -0.0039635, 0.044], dtype=torch.float64))
        torch.testing.assert_close(orientation[0], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64))

    def test_gravity_basis_is_orthonormal_and_upright_identity(self):
        gravity = torch.tensor([[0.0, 0.0, -1.0], [0.2, -0.3, -0.9327379]], dtype=torch.float64)
        basis = gravity_aligned_basis_in_pelvis(gravity)

        torch.testing.assert_close(basis[0], torch.eye(3, dtype=torch.float64))
        identity = torch.eye(3, dtype=torch.float64).expand(2, -1, -1)
        torch.testing.assert_close(basis.transpose(-1, -2) @ basis, identity, atol=1.0e-7, rtol=0.0)
        torch.testing.assert_close(basis[:, :, 2], -gravity / gravity.norm(dim=-1, keepdim=True))

    def test_local_projection_matches_world_formulation(self):
        dtype = torch.float64
        roll = torch.tensor([0.0, 0.18, -0.12], dtype=dtype)
        pitch = torch.tensor([0.0, -0.14, 0.09], dtype=dtype)
        yaw = torch.tensor([0.0, 1.1, -0.7], dtype=dtype)
        base_quat_w = quat_from_euler_xyz(roll, pitch, yaw)
        pelvis_pos_w = torch.tensor(
            [[0.0, 0.0, 0.85], [2.0, -3.0, 1.1], [-1.2, 0.4, 0.72]],
            dtype=dtype,
        )
        waist = torch.tensor(
            [[0.0, 0.0, 0.0], [0.22, -0.11, 0.16], [-0.31, 0.13, -0.08]],
            dtype=dtype,
        )
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=dtype).expand(3, -1)
        projected_gravity = DepthTerrainAdapter._rotate_inverse_xyzw(base_quat_w, gravity_w)
        camera_pos_p, camera_quat_p = self.local.camera_pose_in_pelvis(waist)
        camera_pos_w = pelvis_pos_w + rotate_xyzw(base_quat_w, camera_pos_p)
        camera_quat_w = quaternion_multiply_xyzw(base_quat_w, camera_quat_p)
        heading_quat_w = calc_heading_quat(base_quat_w, w_last=True)

        generator = torch.Generator().manual_seed(11)
        depth = 0.25 + 1.5 * torch.rand(
            (3, self.camera.height, self.camera.width),
            generator=generator,
            dtype=dtype,
        )
        depth[0, 0, 0] = float("nan")
        depth[1, 2, 3] = float("nan")

        local_map, local_visible = self.local(depth, projected_gravity, waist)
        world_map, world_visible = self.world(
            depth,
            camera_pos_w,
            camera_quat_w,
            pelvis_pos_w,
            heading_quat_w,
        )

        self.assertTrue(torch.equal(local_visible, world_visible))
        torch.testing.assert_close(
            local_map[local_visible],
            world_map[world_visible],
            atol=2.0e-6,
            rtol=0.0,
        )

    def test_batched_calibration_override_matches_fixed_calibration(self):
        batch_size = 2
        depth = torch.ones((batch_size, self.camera.height, self.camera.width), dtype=torch.float64)
        gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64).expand(batch_size, -1)
        waist = torch.zeros((batch_size, 3), dtype=torch.float64)
        fixed_map, fixed_visible = self.local(depth, gravity, waist)
        overridden_map, overridden_visible = self.local(
            depth,
            gravity,
            waist,
            intrinsic_matrix=self.intrinsics.double().expand(batch_size, -1, -1),
            camera_pos_torso=self.local.camera_pos_torso.expand(batch_size, -1),
            camera_optical_quat_torso_xyzw=self.local.camera_optical_quat_torso_xyzw.expand(batch_size, -1),
        )
        self.assertTrue(torch.equal(fixed_visible, overridden_visible))
        torch.testing.assert_close(fixed_map[fixed_visible], overridden_map[overridden_visible])


if __name__ == "__main__":
    unittest.main()
