from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from humanoidverse.mjlab_reward_relabel import (
    TERRAIN_REFERENCE_RAY_INDEX,
    RewardWrapperHV,
    canonicalize_terrain_relabel_qpos,
)


def _terrain_observation(clearance: np.ndarray) -> dict[str, torch.Tensor]:
    batch_size = clearance.shape[0]
    terrain_actor = np.zeros((batch_size, 273), dtype=np.float32)
    terrain_actor[:, TERRAIN_REFERENCE_RAY_INDEX] = clearance
    privileged_state = np.zeros((batch_size, 8), dtype=np.float32)
    privileged_state[:, 0] = clearance
    return {
        "privileged_state": torch.from_numpy(privileged_state),
        "terrain_actor": torch.from_numpy(terrain_actor),
        "terrain_priv": torch.zeros(batch_size, 273),
    }


class TerrainRewardRelabelCanonicalizationTest(unittest.TestCase):
    def test_terrain_qpos_uses_ground_relative_root_height_without_mutation(self) -> None:
        qpos = np.zeros((3, 36), dtype=np.float32)
        qpos[:, 2] = np.array([0.82, -0.75, 2.8], dtype=np.float32)
        original = qpos.copy()
        clearance = np.array([0.82, 0.65, 0.8], dtype=np.float32)

        canonical = canonicalize_terrain_relabel_qpos(qpos, _terrain_observation(clearance))

        np.testing.assert_array_equal(qpos, original)
        np.testing.assert_allclose(canonical[:, 2], clearance, atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(canonical[:, :2], original[:, :2])
        np.testing.assert_array_equal(canonical[:, 3:], original[:, 3:])

    def test_vertical_translation_produces_identical_canonical_qpos(self) -> None:
        lower = np.zeros((1, 36), dtype=np.float32)
        elevated = lower.copy()
        lower[:, 2] = 0.8
        elevated[:, 2] = 2.8
        observation = _terrain_observation(np.array([0.8], dtype=np.float32))

        lower_canonical = canonicalize_terrain_relabel_qpos(lower, observation)
        elevated_canonical = canonicalize_terrain_relabel_qpos(elevated, observation)

        np.testing.assert_array_equal(lower_canonical, elevated_canonical)

    def test_original_flat_fb_qpos_is_unchanged(self) -> None:
        qpos = np.random.default_rng(7).normal(size=(4, 36)).astype(np.float32)
        observation = {"privileged_state": torch.zeros(4, 8)}
        output = canonicalize_terrain_relabel_qpos(qpos, observation)
        self.assertIs(output, qpos)

    def test_inconsistent_center_clearance_fails(self) -> None:
        qpos = np.zeros((1, 36), dtype=np.float32)
        observation = _terrain_observation(np.array([0.8], dtype=np.float32))
        observation["terrain_actor"][0, TERRAIN_REFERENCE_RAY_INDEX] = 0.7
        with self.assertRaisesRegex(RuntimeError, "inconsistent root clearances"):
            canonicalize_terrain_relabel_qpos(qpos, observation)

    def test_nonfinite_clearance_fails(self) -> None:
        qpos = np.zeros((1, 36), dtype=np.float32)
        observation = _terrain_observation(np.array([np.nan], dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            canonicalize_terrain_relabel_qpos(qpos, observation)

    def test_reward_wrapper_relabels_canonical_next_qpos(self) -> None:
        clearance = np.array([0.65, 0.8], dtype=np.float32)
        next_observation = _terrain_observation(clearance)
        qpos = torch.zeros(2, 36)
        qpos[:, 2] = torch.tensor([-0.75, 2.8])
        sample = {
            "action": torch.zeros(2, 29),
            "next": {
                "qpos": qpos,
                "qvel": torch.zeros(2, 35),
                "observation": next_observation,
            },
        }

        class Dataset:
            def size(self):
                return 2

            def sample(self, _size):
                return sample

        model = SimpleNamespace(
            device="cpu",
            reward_wr_inference=lambda **kwargs: torch.ones(1, 256),
        )
        wrapper = RewardWrapperHV(
            model=model,
            inference_dataset=Dataset(),
            num_samples_per_inference=1,
            inference_function="reward_wr_inference",
            max_workers=1,
            env_model=object(),
        )

        with patch("humanoidverse.mjlab_reward_relabel.relabel", return_value=np.ones((2, 1))) as relabel_mock:
            wrapper.reward_inference("move-ego-0-0.7")

        relabeled_qpos = relabel_mock.call_args.args[1]
        np.testing.assert_allclose(relabeled_qpos[:, 2], clearance, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
