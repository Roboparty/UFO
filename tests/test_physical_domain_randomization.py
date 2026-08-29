from __future__ import annotations

import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from humanoidverse.agents.envs.humanoidverse_mjlab import (
    _mass_scale_range_to_pseudo_inertia_alpha,
)


def test_link_mass_range_is_physics_consistent_ten_percent_scaling() -> None:
    config_path = Path(__file__).parents[1] / "humanoidverse/config/domain_rand/domain_rand.yaml"
    config = OmegaConf.load(config_path).domain_rand
    assert config.randomize_link_mass is True
    assert list(config.link_mass_range) == [0.9, 1.1]

    alpha_lower, alpha_upper = _mass_scale_range_to_pseudo_inertia_alpha(
        config.link_mass_range
    )
    assert math.exp(2.0 * alpha_lower) == pytest.approx(0.9)
    assert math.exp(2.0 * alpha_upper) == pytest.approx(1.1)


@pytest.mark.parametrize("invalid", ([1.0], [0.0, 1.0], [1.1, 0.9]))
def test_invalid_link_mass_scale_range_is_rejected(invalid: list[float]) -> None:
    with pytest.raises(ValueError, match="link_mass_range"):
        _mass_scale_range_to_pseudo_inertia_alpha(invalid)
