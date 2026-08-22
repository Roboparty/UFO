"""Chunked supervision data for temporal terrain completion.

The dataset deliberately stores projected partial maps rather than raw depth.
This keeps camera geometry outside the learned model and outside RL replay.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from humanoidverse.perception.depth_terrain_adapter import DepthTerrainAdapter


@dataclass
class TerrainPerceptionFrameBatch:
    """One synchronized observation from a vectorized environment."""

    partial_map: torch.Tensor
    visible_mask: torch.Tensor
    pelvis_pos_w: torch.Tensor
    heading_yaw_w: torch.Tensor
    timestamp_s: torch.Tensor
    proprio: torch.Tensor
    gt_terrain_actor: torch.Tensor
    episode_id: torch.Tensor
    env_id: torch.Tensor
    terrain_type: torch.Tensor

    def validate(self) -> int:
        if self.partial_map.ndim != 2 or self.partial_map.shape[1] != DepthTerrainAdapter.GRID_DIMENSION:
            raise ValueError("partial_map must have shape [B, 273]")
        batch_size = self.partial_map.shape[0]
        expected = (batch_size, DepthTerrainAdapter.GRID_DIMENSION)
        if self.visible_mask.shape != expected or self.visible_mask.dtype != torch.bool:
            raise ValueError("visible_mask must be bool with shape [B, 273]")
        if self.gt_terrain_actor.shape != expected:
            raise ValueError("gt_terrain_actor must have shape [B, 273]")
        if self.pelvis_pos_w.shape != (batch_size, 3):
            raise ValueError("pelvis_pos_w must have shape [B, 3]")
        if self.proprio.ndim != 2 or self.proprio.shape[0] != batch_size:
            raise ValueError("proprio must have shape [B, P]")
        for name in ("heading_yaw_w", "timestamp_s", "episode_id", "env_id", "terrain_type"):
            if getattr(self, name).shape != (batch_size,):
                raise ValueError(f"{name} must have shape [B]")
        if torch.any(self.visible_mask & ~torch.isfinite(self.partial_map)):
            raise ValueError("visible partial-map cells must be finite")
        for name in ("pelvis_pos_w", "heading_yaw_w", "timestamp_s", "proprio"):
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must be finite")
        if not torch.isfinite(self.gt_terrain_actor).all():
            raise ValueError("GT terrain_actor must be finite")
        return batch_size

    def cpu(self) -> TerrainPerceptionFrameBatch:
        return TerrainPerceptionFrameBatch(**{field.name: getattr(self, field.name).detach().cpu() for field in fields(self)})


class TerrainPerceptionChunkWriter:
    """Append synchronized frames and atomically flush fixed-size chunks."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        output_dir: str | Path,
        *,
        chunk_steps: int = 128,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if chunk_steps <= 0:
            raise ValueError("chunk_steps must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / "manifest.json").exists() or any(self.output_dir.glob("chunk_*.pt")):
            raise FileExistsError(f"terrain perception output directory already contains a dataset: {self.output_dir}")
        self.chunk_steps = chunk_steps
        self.metadata = dict(metadata or {})
        self._frames: list[TerrainPerceptionFrameBatch] = []
        self._num_envs: int | None = None
        self._proprio_dim: int | None = None
        self._chunk_index = 0
        self._chunks: list[dict[str, Any]] = []
        self._closed = False

    def append(self, frame: TerrainPerceptionFrameBatch) -> Path | None:
        if self._closed:
            raise RuntimeError("cannot append to a closed terrain dataset writer")
        num_envs = frame.validate()
        if self._num_envs is None:
            self._num_envs = num_envs
            self._proprio_dim = frame.proprio.shape[1]
        if num_envs != self._num_envs or frame.proprio.shape[1] != self._proprio_dim:
            raise ValueError("num_envs and proprio dimension must remain constant")
        self._frames.append(frame.cpu())
        if len(self._frames) >= self.chunk_steps:
            return self.flush()
        return None

    def flush(self) -> Path | None:
        if not self._frames:
            return None
        payload = {
            field.name: torch.stack([getattr(frame, field.name) for frame in self._frames], dim=0)
            for field in fields(TerrainPerceptionFrameBatch)
        }
        filename = f"chunk_{self._chunk_index:06d}.pt"
        output_path = self.output_dir / filename
        temporary_path = self.output_dir / f".{filename}.tmp"
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
        self._chunks.append({"file": filename, "steps": len(self._frames)})
        self._chunk_index += 1
        self._frames.clear()
        self._write_manifest()
        return output_path

    def _write_manifest(self) -> None:
        manifest = {
            "format": "pbfm_temporal_terrain",
            "version": self.FORMAT_VERSION,
            "grid_dimension": DepthTerrainAdapter.GRID_DIMENSION,
            "num_envs": self._num_envs,
            "proprio_dim": self._proprio_dim,
            "chunks": self._chunks,
            "metadata": self.metadata,
        }
        path = self.output_dir / "manifest.json"
        temporary_path = self.output_dir / ".manifest.json.tmp"
        temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_path, path)

    def close(self) -> None:
        if not self._closed:
            self.flush()
            self._closed = True

    def __enter__(self) -> TerrainPerceptionChunkWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class TerrainPerceptionSequenceDataset(Dataset):
    """Read fixed-length histories without crossing episode/reset boundaries."""

    def __init__(
        self,
        root: str | Path,
        *,
        sequence_steps: int,
        history_seconds: float = 0.6,
    ) -> None:
        if sequence_steps <= 0 or history_seconds <= 0.0:
            raise ValueError("sequence_steps and history_seconds must be positive")
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("format") != "pbfm_temporal_terrain" or manifest.get("version") != 1:
            raise ValueError("unsupported temporal terrain dataset format")
        self.sequence_steps = sequence_steps
        self.history_seconds = history_seconds
        self.proprio_dim = int(manifest["proprio_dim"])
        self.metadata = dict(manifest.get("metadata", {}))
        self._chunk_files: list[Path] = []
        self._samples: list[tuple[int, int, int]] = []
        self._cached_chunk_index: int | None = None
        self._cached_chunk: dict[str, torch.Tensor] | None = None
        for chunk_index, item in enumerate(manifest["chunks"]):
            chunk_path = self.root / item["file"]
            payload = torch.load(chunk_path, map_location="cpu", weights_only=True)
            self._validate_chunk(payload, manifest)
            self._chunk_files.append(chunk_path)
            steps, num_envs = payload["episode_id"].shape
            for env_id in range(num_envs):
                for end in range(sequence_steps - 1, steps):
                    start = end - sequence_steps + 1
                    episodes = payload["episode_id"][start : end + 1, env_id]
                    timestamps = payload["timestamp_s"][start : end + 1, env_id]
                    same_episode = torch.all(episodes == episodes[-1])
                    chronological = torch.all(timestamps[1:] >= timestamps[:-1])
                    within_window = timestamps[-1] - timestamps[0] <= history_seconds + 1.0e-6
                    if same_episode and chronological and within_window:
                        self._samples.append((chunk_index, env_id, end))

    @staticmethod
    def _validate_chunk(payload: dict[str, torch.Tensor], manifest: dict[str, Any]) -> None:
        expected_keys = {field.name for field in fields(TerrainPerceptionFrameBatch)}
        if set(payload) != expected_keys:
            raise ValueError(f"terrain dataset chunk keys do not match schema: {set(payload)}")
        if payload["partial_map"].ndim != 3 or payload["partial_map"].shape[-1] != 273:
            raise ValueError("terrain dataset partial_map must have shape [S, B, 273]")
        steps, num_envs = payload["partial_map"].shape[:2]
        if num_envs != manifest["num_envs"] or payload["proprio"].shape != (
            steps,
            num_envs,
            manifest["proprio_dim"],
        ):
            raise ValueError("terrain dataset chunk dimensions do not match manifest")

    def __len__(self) -> int:
        return len(self._samples)

    def chunk_index_for_sample(self, index: int) -> int:
        return self._samples[index][0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        chunk_index, env_id, end = self._samples[index]
        start = end - self.sequence_steps + 1
        if self._cached_chunk_index != chunk_index or self._cached_chunk is None:
            self._cached_chunk = torch.load(
                self._chunk_files[chunk_index],
                map_location="cpu",
                weights_only=True,
            )
            self._cached_chunk_index = chunk_index
        chunk = self._cached_chunk
        sequence_keys = (
            "partial_map",
            "visible_mask",
            "pelvis_pos_w",
            "heading_yaw_w",
            "timestamp_s",
            "proprio",
        )
        sample = {key: chunk[key][start : end + 1, env_id] for key in sequence_keys}
        sample.update(
            {
                "gt_terrain_actor": chunk["gt_terrain_actor"][end, env_id],
                "episode_id": chunk["episode_id"][end, env_id],
                "env_id": chunk["env_id"][end, env_id],
                "terrain_type": chunk["terrain_type"][end, env_id],
            }
        )
        return sample
