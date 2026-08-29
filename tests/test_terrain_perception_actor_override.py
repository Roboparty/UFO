from __future__ import annotations

import torch
from torch import nn

from humanoidverse.actor_override import load_actor_module_override
from humanoidverse.terrain_perception_closed_loop import _load_actor_override, _state_dict_checksum


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._actor = nn.Linear(3, 2)
        self.other = nn.Linear(3, 2)


def test_actor_override_changes_only_actor_and_records_identity(tmp_path) -> None:
    model = _Model()
    other_before = {name: value.clone() for name, value in model.other.state_dict().items()}
    actor_state = {name: value + 1.0 for name, value in model._actor.state_dict().items()}
    path = tmp_path / "actor_step_000500.pt"
    torch.save({"step": 500, "actor": actor_state}, path)

    identity = _load_actor_override(model, path)

    assert identity == {"path": str(path.resolve()), "step": 500, "checksum": _state_dict_checksum(actor_state)}
    for name, value in model._actor.state_dict().items():
        torch.testing.assert_close(value, actor_state[name])
    for name, value in model.other.state_dict().items():
        torch.testing.assert_close(value, other_before[name])
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_actor_override_rejects_nonfinite_state(tmp_path) -> None:
    model = _Model()
    state = model._actor.state_dict()
    state["bias"] = torch.full_like(state["bias"], float("nan"))
    path = tmp_path / "bad.pt"
    torch.save({"step": 1, "actor": state}, path)
    try:
        _load_actor_override(model, path)
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite Actor checkpoint was accepted")


def test_actor_module_override_does_not_mutate_sibling_module(tmp_path) -> None:
    model = _Model()
    sibling_before = {name: value.clone() for name, value in model.other.state_dict().items()}
    actor_state = {name: value - 0.5 for name, value in model._actor.state_dict().items()}
    path = tmp_path / "actor_step_002000.pt"
    torch.save({"step": 2000, "actor": actor_state}, path)

    identity = load_actor_module_override(model._actor, path)

    assert identity["step"] == 2000
    assert identity["checksum"] == _state_dict_checksum(actor_state)
    for name, value in model._actor.state_dict().items():
        torch.testing.assert_close(value, actor_state[name])
    for name, value in model.other.state_dict().items():
        torch.testing.assert_close(value, sibling_before[name])
    assert not any(parameter.requires_grad for parameter in model._actor.parameters())
    assert all(parameter.requires_grad for parameter in model.other.parameters())
