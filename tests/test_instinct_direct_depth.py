from __future__ import annotations

import math

import gymnasium
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F

from humanoidverse.agents.base_model import save_model
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.nn_models import DepthEncoder, DirectDepthActorArchiConfig
from humanoidverse.agents.presets.fb_depth import build_fb_depth_agent
from humanoidverse.perception.instinct_direct_depth import (
    BASELINE_TYPE,
    REFERENCE_COMMIT,
    REFERENCE_PROJECT,
    RP1DirectDepthConfig,
    RP1DirectDepthRuntime,
    preprocess_rp1_depth,
    raycast_ranges_to_image_plane,
)


def test_model_checkpoint_uses_atomic_cpu_safetensors_snapshot(tmp_path) -> None:
    class _Config:
        @staticmethod
        def model_dump() -> dict[str, str]:
            return {"name": "shared_test_model"}

    class _SharedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.primary = torch.nn.Linear(3, 2, bias=False)
            self.alias = self.primary
            self.cfg = _Config()

    model = _SharedModel()
    expected = model.primary.weight.detach().clone()
    output = tmp_path / "model"

    save_model(str(output), model)

    checkpoint = safetensors.torch.load_file(output / "model.safetensors")
    assert set(checkpoint) == {"primary.weight", "alias.weight"}
    for tensor in checkpoint.values():
        torch.testing.assert_close(tensor, expected)
    assert not list(output.glob(".model.safetensors.tmp-*"))


def test_rp1_direct_depth_contract() -> None:
    cfg = RP1DirectDepthConfig()
    cfg.validate()
    assert REFERENCE_PROJECT == "UFO-rp1"
    assert REFERENCE_COMMIT == "8c364e1001734097aac58e5033a1b5076925d3c5"
    assert BASELINE_TYPE == "rp1_direct_depth"
    assert cfg.sampled_ages == (35, 30, 25, 20, 15, 10, 5, 0)
    assert (cfg.output_height, cfg.output_width) == (36, 32)
    assert cfg.delayed_frame_ranges == (0, 1)
    assert cfg.position_error == 0.05
    assert cfg.angle_error_rad == math.radians(5.0)
    assert cfg.include_geom_groups == (0, 2, 5)
    assert not cfg.exclude_parent_body


def test_flip_crop_gaussian_quantization_matches_rp1_stage_by_stage() -> None:
    cfg = RP1DirectDepthConfig()
    raw = torch.linspace(-0.25, 3.0, 2 * 36 * 64, dtype=torch.float32).reshape(2, 36, 64)
    raw[0, 1, 16] = torch.nan
    raw[0, 1, 17] = torch.inf
    raw[0, 1, 18] = -torch.inf
    sanitized = torch.nan_to_num(raw, nan=0.0, posinf=2.5, neginf=0.0)
    cropped = sanitized.flip(1)[:, :, 16:-16]
    gaussian = torch.tensor([1.0, 2.0, 1.0])
    kernel = (gaussian[:, None] * gaussian[None, :]).reshape(1, 1, 3, 3) / 16.0
    blurred = F.conv2d(F.pad(cropped[:, None], (1, 1, 1, 1), mode="reflect"), kernel)[:, 0]
    expected = blurred.clamp(0.0, 2.5).div(2.5).mul(255.0).round().to(torch.uint8)
    actual = preprocess_rp1_depth(raw, cfg, enable_noise=False)
    assert actual.dtype == torch.uint8
    assert actual.shape == (2, 36, 32)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_invalid_ranges_become_zero_and_valid_ranges_use_image_plane_depth() -> None:
    cfg = RP1DirectDepthConfig()
    camera = cfg.camera_config()
    ranges = torch.ones((1, cfg.height * cfg.width), dtype=torch.float32)
    ranges[0, 0] = -1.0
    depth = raycast_ranges_to_image_plane(ranges, camera)
    assert depth.shape == (1, 36, 64)
    assert depth[0, 0, 0] == 0.0
    row, column = 17, 31
    intrinsic = camera.intrinsics().float()
    pixel = torch.tensor([column + 0.5, row + 0.5, 1.0])
    ray = torch.linalg.solve(intrinsic, pixel)
    expected = ray[2] / ray.norm()
    torch.testing.assert_close(depth[0, row, column], expected, rtol=0.0, atol=1.0e-6)


def test_history_is_oldest_to_newest_and_supports_real_zero_or_one_frame_delay() -> None:
    cfg = RP1DirectDepthConfig()
    runtime = RP1DirectDepthRuntime(2, "cpu", cfg, enable_noise=False)
    value = torch.zeros((), dtype=torch.uint8)
    runtime.current_frame = lambda _sensor: torch.full(  # type: ignore[method-assign]
        (2, cfg.output_height, cfg.output_width), int(value), dtype=torch.uint8
    )
    runtime.reset_from_sensor(object(), torch.arange(2))
    runtime.delay_frames.zero_()
    for frame_value in range(1, 37):
        value.fill_(frame_value)
        runtime.append_from_sensor(object())
    expected_zero_delay = torch.tensor([1, 6, 11, 16, 21, 26, 31, 36], dtype=torch.uint8)
    torch.testing.assert_close(runtime.observation()[:, :, 0, 0], expected_zero_delay.expand(2, -1))

    runtime.delay_frames.fill_(1)
    runtime._latest = runtime._sample_history(35)
    expected_one_delay = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35], dtype=torch.uint8)
    torch.testing.assert_close(runtime.observation()[:, :, 0, 0], expected_one_delay.expand(2, -1))


def test_rp1_noise_pipeline_is_seed_reproducible() -> None:
    cfg = RP1DirectDepthConfig()
    raw = torch.linspace(0.0, 2.5, 4 * 36 * 64).reshape(4, 36, 64)
    torch.manual_seed(123)
    first = preprocess_rp1_depth(raw, cfg, enable_noise=True)
    torch.manual_seed(123)
    second = preprocess_rp1_depth(raw, cfg, enable_noise=True)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_rp1_encoder_actor_and_preset_schema() -> None:
    encoder = DepthEncoder()
    convolutions = [module for module in encoder.frame_cnn if isinstance(module, torch.nn.Conv2d)]
    assert [(module.in_channels, module.out_channels, module.stride) for module in convolutions] == [
        (1, 16, (2, 2)),
        (16, 32, (2, 2)),
        (32, 64, (1, 1)),
    ]
    assert isinstance(encoder.gru, torch.nn.GRU)
    assert encoder(torch.zeros(3, 8, 36, 32, dtype=torch.uint8)).shape == (3, 256)
    assert encoder(torch.zeros(3, 8, 36, 32)).shape == (3, 256)

    space = gymnasium.spaces.Dict(
        {
            "state": gymnasium.spaces.Box(-np.inf, np.inf, (87,), dtype=np.float32),
            "privileged_state": gymnasium.spaces.Box(-np.inf, np.inf, (221,), dtype=np.float32),
            "last_action": gymnasium.spaces.Box(-np.inf, np.inf, (29,), dtype=np.float32),
            "history_actor": gymnasium.spaces.Box(-np.inf, np.inf, (203,), dtype=np.float32),
            "terrain_priv": gymnasium.spaces.Box(-0.5, 0.5, (273,), dtype=np.float32),
            "depth_image": gymnasium.spaces.Box(0, 255, (8, 36, 32), dtype=np.uint8),
        }
    )
    actor_cfg = DirectDepthActorArchiConfig(
        name="direct_depth",
        model="residual",
        hidden_dim=64,
        hidden_layers=2,
        embedding_layers=2,
        input_filter=DictInputFilterConfig(
            name="DictInputFilterConfig",
            key=["state", "last_action", "history_actor"],
        ),
    )
    actor = actor_cfg.build(space, 256, 29)
    obs = {
        key: torch.zeros((2, *value.shape), dtype=torch.uint8 if key == "depth_image" else torch.float32)
        for key, value in space.spaces.items()
    }
    assert actor(obs, torch.zeros(2, 256), 0.05).mean.shape == (2, 29)

    preset = build_fb_depth_agent(device="cpu", compile=False)
    assert preset.model.archi.actor.depth_key == "depth_image"
    assert preset.model.archi.actor.depth_latent_dim == 256
    assert preset.model.archi.f.input_filter.key[-1] == "terrain_priv"
    assert preset.model.archi.critic.input_filter.key[-1] == "terrain_priv"
    assert "terrain_actor" not in preset.model.obs_normalizer.normalizers
    assert "depth_image" in preset.model.obs_normalizer.normalizers


def test_compact_replay_stores_one_uint8_frame_and_reconstructs_history() -> None:
    buffer = TrajectoryDictBufferMultiDim(
        capacity=10,
        device="cpu",
        n_dim=2,
        end_key="truncated",
        output_key_t=["observation"],
        output_key_tp1=["observation"],
        compact_depth_history=True,
    )
    time_steps = 6
    newest = torch.arange(time_steps, dtype=torch.uint8).reshape(time_steps, 1, 1, 1, 1)
    depth = newest.expand(time_steps, 1, 8, 36, 32).clone()
    state = torch.arange(time_steps, dtype=torch.float32).reshape(time_steps, 1, 1)
    truncated = torch.zeros(time_steps, 1, 1, dtype=torch.bool)
    truncated[-1] = True
    data = {
        "observation": {"depth_image": depth, "state": state},
        "terminated": torch.zeros_like(truncated),
        "truncated": truncated,
    }
    buffer.extend(data)
    assert data["observation"]["depth_image"].shape == (time_steps, 1, 8, 36, 32)
    assert buffer.storage["observation"]["depth_image"].shape == (10, 1, 36, 32)
    assert buffer.storage["observation"]["depth_image"].dtype == torch.uint8

    restored = buffer.get_full_buffer()["observation"]["depth_image"]
    assert restored.shape == (time_steps - 1, 8, 36, 32)
    torch.testing.assert_close(
        restored[-1, :, 0, 0],
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 4], dtype=torch.uint8),
    )
