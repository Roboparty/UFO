import math
import unittest

import torch

from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter
from humanoidverse.perception.temporal_terrain import (
    TerrainHistoryBuffer,
    TemporalTerrainCompletion,
    WarpedTerrainHistory,
    terrain_completion_loss,
    terrain_completion_metrics,
    warp_terrain_history_to_current,
)


def grid_index(x: float, y: float) -> int:
    ix = round((x - DepthTerrainAdapter.X_MIN) / DepthTerrainAdapter.RESOLUTION)
    iy = round((y - DepthTerrainAdapter.Y_MIN) / DepthTerrainAdapter.RESOLUTION)
    return ix * DepthTerrainAdapter.GRID_SHAPE[1] + iy


class TemporalTerrainWarpTest(unittest.TestCase):
    def test_history_reset_removes_all_previous_episode_frames(self):
        history = TerrainHistoryBuffer(
            batch_size=2,
            time_steps=3,
            proprio_dim=4,
            device="cpu",
        )
        for timestamp in (0.0, 0.02):
            history.append(
                partial_map=torch.full((2, 273), 0.8),
                visible_mask=torch.ones((2, 273), dtype=torch.bool),
                pelvis_pos_w=torch.zeros((2, 3)),
                heading_yaw_w=torch.zeros(2),
                timestamp_s=torch.full((2,), timestamp),
                proprio=torch.ones((2, 4)),
            )

        history.reset(torch.tensor([False, True]))
        history.append(
            partial_map=torch.full((2, 273), 0.7),
            visible_mask=torch.ones((2, 273), dtype=torch.bool),
            pelvis_pos_w=torch.ones((2, 3)),
            heading_yaw_w=torch.ones(2),
            timestamp_s=torch.tensor([0.04, 0.0]),
            proprio=torch.full((2, 4), 2.0),
        )

        self.assertEqual(int(history.frame_valid[0].sum()), 3)
        self.assertEqual(int(history.frame_valid[1].sum()), 1)
        self.assertFalse(history.visible_masks[1, :-1].any())
        self.assertTrue(torch.isnan(history.partial_maps[1, :-1]).all())
        self.assertTrue(torch.all(history.proprio[1, :-1] == 0.0))

    def test_single_frame_view_invalidates_all_past_frames(self):
        history = TerrainHistoryBuffer(
            batch_size=1,
            time_steps=3,
            proprio_dim=2,
            device="cpu",
        )
        for timestamp in (0.0, 0.02, 0.04):
            history.append(
                partial_map=torch.full((1, 273), 0.8),
                visible_mask=torch.ones((1, 273), dtype=torch.bool),
                pelvis_pos_w=torch.zeros((1, 3)),
                heading_yaw_w=torch.zeros(1),
                timestamp_s=torch.tensor([timestamp]),
                proprio=torch.ones((1, 2)),
            )

        single = history.single_frame_view()

        self.assertEqual(int(single.frame_valid.sum()), 1)
        self.assertFalse(single.visible_masks[:, :-1].any())
        self.assertTrue(single.visible_masks[:, -1].all())
        torch.testing.assert_close(single.partial_maps[:, -1], history.partial_maps[:, -1])

    def test_identity_warp_preserves_map_and_mask(self):
        values = torch.linspace(0.2, 1.0, 273).reshape(1, 1, 273)
        mask = torch.ones_like(values, dtype=torch.bool)
        pose = torch.tensor([[[1.0, -2.0, 0.8]]])
        yaw = torch.tensor([[0.7]])

        warped = warp_terrain_history_to_current(values, mask, pose, yaw)

        torch.testing.assert_close(warped.clearances, values, atol=2.0e-6, rtol=0.0)
        self.assertTrue(torch.equal(warped.visible_masks, mask))
        torch.testing.assert_close(
            warped.motion_features[0, 0],
            torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_forward_translation_moves_old_point_rearward(self):
        values = torch.full((1, 2, 273), float("nan"))
        mask = torch.zeros_like(values, dtype=torch.bool)
        values[0, 0, grid_index(0.5, 0.0)] = 0.7
        mask[0, 0, grid_index(0.5, 0.0)] = True
        pose = torch.tensor([[[0.0, 0.0, 0.8], [0.1, 0.0, 0.8]]])
        yaw = torch.zeros((1, 2))

        warped = warp_terrain_history_to_current(values, mask, pose, yaw)

        target = grid_index(0.4, 0.0)
        self.assertTrue(warped.visible_masks[0, 0, target])
        self.assertAlmostEqual(float(warped.clearances[0, 0, target]), 0.7, places=5)

    def test_heading_rotation_moves_world_forward_point_to_current_right(self):
        values = torch.full((1, 2, 273), float("nan"))
        mask = torch.zeros_like(values, dtype=torch.bool)
        values[0, 0, grid_index(0.5, 0.0)] = 0.7
        mask[0, 0, grid_index(0.5, 0.0)] = True
        pose = torch.tensor([[[0.0, 0.0, 0.8], [0.0, 0.0, 0.8]]])
        yaw = torch.tensor([[0.0, math.pi / 2.0]])

        warped = warp_terrain_history_to_current(values, mask, pose, yaw)

        target = grid_index(0.0, -0.5)
        self.assertTrue(warped.visible_masks[0, 0, target])
        self.assertAlmostEqual(float(warped.clearances[0, 0, target]), 0.7, places=5)

    def test_vertical_translation_updates_clearance(self):
        values = torch.full((1, 2, 273), float("nan"))
        mask = torch.zeros_like(values, dtype=torch.bool)
        center = grid_index(0.0, 0.0)
        values[0, 0, center] = 0.7
        mask[0, 0, center] = True
        pose = torch.tensor([[[0.0, 0.0, 0.8], [0.0, 0.0, 1.0]]])
        yaw = torch.zeros((1, 2))

        warped = warp_terrain_history_to_current(values, mask, pose, yaw)

        self.assertAlmostEqual(float(warped.clearances[0, 0, center]), 0.9, places=5)

    def test_age_and_reset_validity_remove_old_frame(self):
        values = torch.ones((1, 3, 273))
        mask = torch.ones_like(values, dtype=torch.bool)
        pose = torch.zeros((1, 3, 3))
        yaw = torch.zeros((1, 3))
        timestamps = torch.tensor([[0.0, 0.5, 0.7]])
        frame_valid = torch.tensor([[True, False, True]])

        warped = warp_terrain_history_to_current(
            values,
            mask,
            pose,
            yaw,
            timestamps_s=timestamps,
            frame_valid=frame_valid,
            history_seconds=0.6,
        )

        self.assertFalse(warped.visible_masks[0, 0].any())
        self.assertFalse(warped.visible_masks[0, 1].any())
        self.assertTrue(warped.visible_masks[0, 2].all())


class TemporalTerrainCompletionTest(unittest.TestCase):
    def test_completion_preserves_current_visible_cells_and_backpropagates(self):
        batch_size, time_steps = 2, 4
        values = torch.full((batch_size, time_steps, 273), float("nan"))
        masks = torch.zeros_like(values, dtype=torch.bool)
        values[:, :, 100:130] = 0.7
        masks[:, :, 100:130] = True
        values[:, -1, 58] = torch.tensor([0.81, 0.62])
        masks[:, -1, 58] = True
        history = WarpedTerrainHistory(
            clearances=values,
            visible_masks=masks,
            motion_features=torch.zeros((batch_size, time_steps, 6)),
        )
        model = TemporalTerrainCompletion(hidden_channels=8, proprio_dim=5, proprio_channels=4)
        proprio = torch.randn((batch_size, time_steps, 5))

        output = model(history, proprio=proprio)

        self.assertEqual(tuple(output.predicted_clearance.shape), (batch_size, 273))
        self.assertEqual(tuple(output.hidden.shape), (batch_size, 8, 13, 21))
        self.assertTrue(torch.isfinite(output.predicted_clearance).all())
        self.assertTrue(torch.all(output.predicted_clearance >= 0.0))
        torch.testing.assert_close(output.completed_clearance[:, 58], values[:, -1, 58])
        target = torch.full((batch_size, 273), 0.75)
        loss = terrain_completion_loss(output.predicted_clearance, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_completion_loss_ignores_invalid_teacher_cells(self):
        prediction = torch.zeros((1, 273), requires_grad=True)
        target = torch.ones((1, 273))
        target[:, 10:] = float("nan")

        loss = terrain_completion_loss(prediction, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.all(prediction.grad[:, 10:] == 0.0))

    def test_metrics_separate_missing_underfoot_and_edges(self):
        target = torch.full((1, 273), 0.8)
        target[:, 13 * 10 :] = 0.6
        prediction = target.clone()
        prediction[:, 58] += 0.1
        current_visible = torch.ones_like(target, dtype=torch.bool)
        current_visible[:, 58] = False

        metrics = terrain_completion_metrics(
            prediction,
            target,
            current_visible=current_visible,
            history_visible=current_visible,
        )

        self.assertGreater(float(metrics["missing_mae"]), 0.09)
        self.assertEqual(float(metrics["visible_mae"]), 0.0)
        self.assertGreater(float(metrics["underfoot_mae"]), 0.0)
        self.assertIn("edge_mae", metrics)
        self.assertEqual(float(metrics["history_visible_fraction"]), float(metrics["current_visible_fraction"]))
        self.assertEqual(float(metrics["history_coverage_gain"]), 0.0)

    def test_metrics_report_temporal_coverage_gain(self):
        target = torch.zeros((1, 273))
        prediction = torch.ones_like(target)
        current_visible = torch.zeros_like(target, dtype=torch.bool)
        current_visible[:, :10] = True
        history_visible = current_visible.clone()
        history_visible[:, :30] = True

        metrics = terrain_completion_metrics(
            prediction,
            target,
            current_visible=current_visible,
            history_visible=history_visible,
        )

        self.assertAlmostEqual(float(metrics["current_visible_fraction"]), 10.0 / 273.0)
        self.assertAlmostEqual(float(metrics["history_visible_fraction"]), 30.0 / 273.0)
        self.assertAlmostEqual(float(metrics["history_coverage_gain"]), 20.0 / 273.0)
        self.assertEqual(float(metrics["history_observed_missing_mae"]), 1.0)
        self.assertEqual(float(metrics["never_observed_mae"]), 1.0)


if __name__ == "__main__":
    unittest.main()
