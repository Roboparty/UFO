from types import SimpleNamespace

import pytest
import torch

from humanoidverse.agents.envs.expert_motion_loader import source_mixed_expert_priorities


def test_expert_priorities_preserve_source_level_mass_not_motion_count() -> None:
    # Two LaFAN motions share 0.8 mass; six 100STYLE motions share 0.2.
    motion_lib = SimpleNamespace(
        _num_unique_motions=8,
        _sampling_prob=torch.tensor(
            [0.4, 0.4, 0.2 / 6, 0.2 / 6, 0.2 / 6, 0.2 / 6, 0.2 / 6, 0.2 / 6]
        ),
    )

    priorities = source_mixed_expert_priorities(
        motion_lib,
        list(range(8)),
        device="cpu",
    )

    assert torch.isclose(priorities[:2].sum(), torch.tensor(0.8))
    assert torch.isclose(priorities[2:].sum(), torch.tensor(0.2))
    assert torch.isclose(priorities.sum(), torch.tensor(1.0))


def test_expert_priorities_follow_motion_id_order() -> None:
    motion_lib = SimpleNamespace(
        _num_unique_motions=3,
        _sampling_prob=torch.tensor([0.6, 0.3, 0.1]),
    )

    priorities = source_mixed_expert_priorities(
        motion_lib,
        [2, 0, 1],
        device="cpu",
    )

    assert torch.allclose(priorities, torch.tensor([0.1, 0.6, 0.3]))


@pytest.mark.parametrize("motion_ids", ([0, 1], [0, 1, 1]))
def test_expert_priorities_reject_incomplete_or_duplicate_motion_ids(motion_ids) -> None:
    motion_lib = SimpleNamespace(
        _num_unique_motions=3,
        _sampling_prob=torch.tensor([0.6, 0.3, 0.1]),
    )

    with pytest.raises(ValueError, match="every MotionLib motion exactly once"):
        source_mixed_expert_priorities(motion_lib, motion_ids, device="cpu")
