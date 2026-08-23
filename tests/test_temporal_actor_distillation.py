from __future__ import annotations

import copy

import torch
from torch import nn

from humanoidverse.distill_temporal_terrain_actor import (
    DistillationConfig,
    actor_distillation_loss,
    configure_actor_only_training,
    module_checksum,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._actor = nn.Linear(3, 2)
        self._obs_normalizer = nn.BatchNorm1d(3)
        self._backward_map = nn.Linear(3, 2)


def test_config_rejects_invalid_distillation_mix() -> None:
    config = DistillationConfig(high_stairs_envs=0)
    try:
        config.validate()
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid environment count was accepted")


def test_config_rejects_unsorted_or_duplicate_milestones() -> None:
    for milestones in ((1000, 500), (500, 500), (0, 500)):
        try:
            DistillationConfig(milestone_steps=milestones).validate()
        except ValueError as error:
            assert "milestone_steps" in str(error)
        else:
            raise AssertionError(f"invalid milestones were accepted: {milestones}")


def test_configure_actor_only_training_freezes_everything_else() -> None:
    model = _TinyModel()
    original_actor = copy.deepcopy(model._actor.state_dict())
    teacher = configure_actor_only_training(model)

    assert teacher.training is False
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.requires_grad for parameter in model._actor.parameters())
    assert not any(parameter.requires_grad for parameter in model._obs_normalizer.parameters())
    assert not any(parameter.requires_grad for parameter in model._backward_map.parameters())
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, original_actor[name])


def test_actor_distillation_updates_only_student_actor() -> None:
    model = _TinyModel()
    teacher = configure_actor_only_training(model)
    frozen_checksum = module_checksum(model._backward_map)
    teacher_checksum = module_checksum(teacher)
    actor_checksum = module_checksum(model._actor)
    optimizer = torch.optim.Adam(model._actor.parameters(), lr=0.1)

    inputs = torch.randn(8, 3)
    with torch.no_grad():
        targets = teacher(inputs) + 0.25
    student_temporal = model._actor(inputs)
    student_gt = model._actor(inputs + 0.1)
    loss, deploy, anchor = actor_distillation_loss(
        student_temporal,
        targets,
        student_gt,
        anchor_weight=0.5,
    )
    assert torch.isclose(loss, deploy + 0.5 * anchor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert module_checksum(model._actor) != actor_checksum
    assert module_checksum(model._backward_map) == frozen_checksum
    assert module_checksum(teacher) == teacher_checksum


def test_distillation_loss_is_zero_for_identical_actions() -> None:
    actions = torch.randn(4, 29)
    loss, deploy, anchor = actor_distillation_loss(actions, actions, actions, anchor_weight=1.0)
    assert loss.item() == 0.0
    assert deploy.item() == 0.0
    assert anchor.item() == 0.0
