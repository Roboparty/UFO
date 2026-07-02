#!/usr/bin/env python3
"""Export the UFO FB backward latent encoder as a standalone ONNX model.

The exported graph matches runtime logic:
  obs (state, last_action, privileged_state)
    -> obs normalizer
    -> backward map
    -> project_z
    -> z
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir


def _require_1d_box(space: gym.spaces.Space, key: str) -> int:
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"obs_space['{key}'] must be gym.spaces.Box, got {type(space)}")
    if len(space.shape) != 1:
        raise ValueError(f"obs_space['{key}'] must be 1D, got shape={space.shape}")
    return int(space.shape[0])


class BackwardEncoderWrapper(nn.Module):
    """ONNX-exportable wrapper for backward latent inference."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.obs_normalizer = model._obs_normalizer
        self.backward_map = model._backward_map
        self.use_project_z = bool(model.cfg.archi.norm_z)

    def forward(
        self,
        state: torch.Tensor,
        last_action: torch.Tensor,
        privileged_state: torch.Tensor,
    ) -> torch.Tensor:
        obs = {
            "state": state,
            "last_action": last_action,
            "privileged_state": privileged_state,
        }
        normalized_obs = self.obs_normalizer(obs)
        z = self.backward_map(normalized_obs)
        if self.use_project_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z


def main(
    model_folder: Path,
    output: Path | None = None,
    checkpoint_subdir: str = "checkpoint",
    device: str = "cuda",
    batch_size: int = 1,
    opset: int = 13,
    verify: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    model_folder = Path(model_folder).resolve()
    checkpoint_dir = model_folder / checkpoint_subdir
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")

    output_path = (model_folder / "exported" / "backward_encoder.onnx") if output is None else Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint_dir(checkpoint_dir, device=device)
    model.eval()
    model.requires_grad_(False)

    # Export on CPU to avoid runtime/provider constraints during conversion.
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    wrapper = BackwardEncoderWrapper(model_cpu).to("cpu").eval()

    if not isinstance(model_cpu.obs_space, gym.spaces.Dict):
        raise TypeError(f"Expected dict obs space, got {type(model_cpu.obs_space)}")
    required_keys = ("state", "last_action", "privileged_state")
    missing = [k for k in required_keys if k not in model_cpu.obs_space.spaces]
    if missing:
        raise KeyError(f"Missing required obs keys for backward encoder export: {missing}")

    state_dim = _require_1d_box(model_cpu.obs_space["state"], "state")
    action_dim = _require_1d_box(model_cpu.obs_space["last_action"], "last_action")
    priv_dim = _require_1d_box(model_cpu.obs_space["privileged_state"], "privileged_state")

    batch = int(batch_size)
    if batch <= 0:
        raise ValueError(f"batch_size must be >=1, got {batch}")

    example_state = torch.randn(batch, state_dim, dtype=torch.float32)
    example_last_action = torch.randn(batch, action_dim, dtype=torch.float32)
    example_privileged_state = torch.randn(batch, priv_dim, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (example_state, example_last_action, example_privileged_state),
        str(output_path),
        input_names=["state", "last_action", "privileged_state"],
        output_names=["z"],
        dynamic_axes={
            "state": {0: "batch"},
            "last_action": {0: "batch"},
            "privileged_state": {0: "batch"},
            "z": {0: "batch"},
        },
        opset_version=int(opset),
        do_constant_folding=True,
        verbose=False,
    )

    print(f"Exported backward encoder ONNX: {output_path}")
    print(
        "Input dims: "
        f"state={state_dim}, last_action={action_dim}, privileged_state={priv_dim}; "
        f"z_dim={model_cpu.cfg.archi.z_dim}"
    )

    if not verify:
        return

    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        print("Skip verify: onnxruntime is not installed.")
        return

    with torch.no_grad():
        torch_z = (
            wrapper(
                example_state,
                example_last_action,
                example_privileged_state,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    ort_session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_z = ort_session.run(
        ["z"],
        {
            "state": example_state.numpy().astype(np.float32),
            "last_action": example_last_action.numpy().astype(np.float32),
            "privileged_state": example_privileged_state.numpy().astype(np.float32),
        },
    )[0]

    max_abs = float(np.max(np.abs(torch_z - ort_z)))
    if not np.allclose(torch_z, ort_z, atol=float(atol), rtol=float(rtol)):
        raise RuntimeError(
            f"ONNX verify failed: max_abs={max_abs:.6e}, "
            f"atol={atol}, rtol={rtol}"
        )
    print(f"ONNX verify passed: max_abs={max_abs:.6e}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
