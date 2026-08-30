import math

import torch

from humanoidverse.aux_reward_gradient_diagnostics import (
    _expanded_component_state,
    ensemble_mean_uncertainty,
    project_residual_head,
    validate_reconstruction,
)


class _ToyParallelCritic(torch.nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 3, output_dim))
        self.bias = torch.nn.Parameter(torch.randn(2, 1, output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,eio->ebo", value, self.weight) + self.bias


def test_component_output_projection_exactly_recovers_scalar_critic() -> None:
    torch.manual_seed(8)
    scalar = _ToyParallelCritic(1)
    components = _ToyParallelCritic(6)
    fractions = torch.tensor([0.50, 0.20, 0.15, 0.10, 0.05])
    state, head_names = _expanded_component_state(scalar, components, fractions)
    components.load_state_dict(state)
    with torch.no_grad():
        components.weight[..., :5].add_(torch.randn_like(components.weight[..., :5]))
        components.bias[..., :5].add_(torch.randn_like(components.bias[..., :5]))
    project_residual_head(components, scalar, head_names)

    value = torch.randn(7, 3)
    expected = scalar(value)
    actual = components(value).sum(dim=-1, keepdim=True)
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_ensemble_mean_uncertainty_matches_two_member_absolute_gap() -> None:
    predictions = torch.tensor([[[1.0], [4.0]], [[3.0], [1.0]]])
    mean, uncertainty = ensemble_mean_uncertainty(predictions)

    assert torch.equal(mean, torch.tensor([[2.0], [2.5]]))
    assert torch.equal(uncertainty, torch.tensor([[2.0], [3.0]]))


def test_reconstruction_validation_fails_closed() -> None:
    report = {
        "batch_index": 0,
        "value_sum_error": {"relative_l2": 0.0},
        "gradients": {
            group: {
                "reconstruction": {
                    "cosine": 0.9 if group == "depth_encoder" else 1.0,
                    "norm_ratio": 1.0,
                    "relative_error": 0.0,
                }
            }
            for group in ("all_actor", "depth_encoder", "policy_head")
        },
    }

    result = validate_reconstruction([report])

    assert result["valid"] is False
    assert result["reward_change_authorized_by_diagnostic"] is False
    assert any("depth_encoder" in failure for failure in result["failures"])


def test_uncertainty_is_zero_only_when_ensemble_agrees() -> None:
    predictions = torch.ones((2, 4, 1)) * 2.5
    _mean, uncertainty = ensemble_mean_uncertainty(predictions)
    assert math.isclose(float(uncertainty.max()), 0.0)
