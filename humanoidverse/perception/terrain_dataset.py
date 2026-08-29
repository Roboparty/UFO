"""Chunked supervision data for temporal terrain completion.

The dataset deliberately stores projected partial maps rather than raw depth.
This keeps camera geometry outside the learned model and outside RL replay.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict, deque
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


@dataclass
class OdometryFreeTerrainPerceptionFrameBatch:
    """One synchronized local observation with no world/odometry fields."""

    partial_map: torch.Tensor
    visible_mask: torch.Tensor
    timestamp_s: torch.Tensor
    proprio: torch.Tensor
    gt_terrain_actor: torch.Tensor
    episode_id: torch.Tensor
    env_id: torch.Tensor
    terrain_type: torch.Tensor
    frame_valid: torch.Tensor

    def validate(self) -> int:
        if self.partial_map.ndim != 2 or self.partial_map.shape[1] != DepthTerrainAdapter.GRID_DIMENSION:
            raise ValueError("partial_map must have shape [B, 273]")
        batch_size = self.partial_map.shape[0]
        expected = (batch_size, DepthTerrainAdapter.GRID_DIMENSION)
        if self.visible_mask.shape != expected or self.visible_mask.dtype != torch.bool:
            raise ValueError("visible_mask must be bool with shape [B, 273]")
        if self.gt_terrain_actor.shape != expected:
            raise ValueError("gt_terrain_actor must have shape [B, 273]")
        if self.proprio.ndim != 2 or self.proprio.shape[0] != batch_size:
            raise ValueError("proprio must have shape [B, P]")
        for name in ("timestamp_s", "episode_id", "env_id", "terrain_type"):
            if getattr(self, name).shape != (batch_size,):
                raise ValueError(f"{name} must have shape [B]")
        if self.frame_valid.shape != (batch_size,) or self.frame_valid.dtype != torch.bool:
            raise ValueError("frame_valid must be bool with shape [B]")
        if torch.any(~self.frame_valid & self.visible_mask.any(dim=1)):
            raise ValueError("invalid camera frames cannot expose visible terrain cells")
        if torch.any(self.visible_mask & ~torch.isfinite(self.partial_map)):
            raise ValueError("visible partial-map cells must be finite")
        if not torch.isfinite(self.timestamp_s).all() or not torch.isfinite(self.proprio).all():
            raise ValueError("timestamps and proprio must be finite")
        if not torch.isfinite(self.gt_terrain_actor).all():
            raise ValueError("GT terrain_actor must be finite")
        return batch_size

    def cpu(self) -> OdometryFreeTerrainPerceptionFrameBatch:
        return OdometryFreeTerrainPerceptionFrameBatch(**{field.name: getattr(self, field.name).detach().cpu() for field in fields(self)})


class TerrainPerceptionChunkWriter:
    """Append synchronized frames and atomically flush fixed-size chunks."""

    FORMAT_VERSION = 1
    ODOMETRY_FREE_FORMAT_VERSION = 3

    def __init__(
        self,
        output_dir: str | Path,
        *,
        chunk_steps: int = 128,
        metadata: dict[str, Any] | None = None,
        odometry_free: bool = False,
    ) -> None:
        if chunk_steps <= 0:
            raise ValueError("chunk_steps must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / "manifest.json").exists() or any(self.output_dir.glob("chunk_*.pt")):
            raise FileExistsError(f"terrain perception output directory already contains a dataset: {self.output_dir}")
        self.chunk_steps = chunk_steps
        self.metadata = dict(metadata or {})
        self.odometry_free = bool(odometry_free)
        self._frame_type = OdometryFreeTerrainPerceptionFrameBatch if self.odometry_free else TerrainPerceptionFrameBatch
        self._frames: list[TerrainPerceptionFrameBatch | OdometryFreeTerrainPerceptionFrameBatch] = []
        self._num_envs: int | None = None
        self._proprio_dim: int | None = None
        self._chunk_index = 0
        self._chunks: list[dict[str, Any]] = []
        self._closed = False

    def append(
        self,
        frame: TerrainPerceptionFrameBatch | OdometryFreeTerrainPerceptionFrameBatch,
    ) -> Path | None:
        if self._closed:
            raise RuntimeError("cannot append to a closed terrain dataset writer")
        if not isinstance(frame, self._frame_type):
            raise TypeError(f"writer expects {self._frame_type.__name__}, got {type(frame).__name__}")
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
            field.name: torch.stack([getattr(frame, field.name) for frame in self._frames], dim=0) for field in fields(self._frame_type)
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
            "version": (self.ODOMETRY_FREE_FORMAT_VERSION if self.odometry_free else self.FORMAT_VERSION),
            "schema": "odometry_free_local" if self.odometry_free else "world_pose",
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
        if manifest.get("format") != "pbfm_temporal_terrain" or manifest.get("version") not in (1, 2, 3):
            raise ValueError("unsupported temporal terrain dataset format")
        self.odometry_free = manifest["version"] in (2, 3)
        self.has_frame_valid = manifest["version"] >= 3
        expected_schema = "odometry_free_local" if self.odometry_free else "world_pose"
        if manifest.get("schema", "world_pose") != expected_schema:
            raise ValueError("terrain dataset version and schema disagree")
        self.sequence_steps = sequence_steps
        self.history_seconds = history_seconds
        self.proprio_dim = int(manifest["proprio_dim"])
        self.metadata = dict(manifest.get("metadata", {}))
        self._chunk_files: list[Path] = []
        # Legacy samples store (end_chunk, env, end_step).  V3 samples add
        # (start_chunk, start_valid_offset, end_valid_offset), which makes
        # cross-chunk history lookup a direct slice instead of many tiny
        # torch.nonzero calls in the hot __getitem__ path.
        self._samples: list[tuple[int, ...]] = []
        self._sample_terrain_types: list[int] = []
        self._valid_steps: list[tuple[torch.Tensor, ...]] = []
        # A v3 sequence near a chunk boundary may need the current and previous
        # payload.  Keeping two entries avoids alternating multi-hundred-MB
        # disk reads while the chunk-grouped sampler is active.
        self._cached_chunks: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        previous_episode: torch.Tensor | None = None
        valid_history: list[deque[tuple[int, int]]] | None = None
        for chunk_index, item in enumerate(manifest["chunks"]):
            chunk_path = self.root / item["file"]
            payload = torch.load(chunk_path, map_location="cpu", weights_only=True)
            self._validate_chunk(payload, manifest)
            self._chunk_files.append(chunk_path)
            steps, num_envs = payload["episode_id"].shape
            if self.has_frame_valid:
                if previous_episode is None:
                    previous_episode = payload["episode_id"][0].clone()
                    valid_history = [deque(maxlen=sequence_steps) for _ in range(num_envs)]
                elif previous_episode.shape != (num_envs,):
                    raise ValueError("terrain dataset changes num_envs between chunks")
                assert valid_history is not None
                chunk_valid_steps: list[list[int]] = [[] for _ in range(num_envs)]
                for end in range(steps):
                    episode = payload["episode_id"][end]
                    changed = episode != previous_episode
                    for env_id in torch.nonzero(changed, as_tuple=False).flatten().tolist():
                        valid_history[env_id].clear()
                    previous_episode = episode.clone()
                    fresh = payload["frame_valid"][end]
                    for env_id in torch.nonzero(fresh, as_tuple=False).flatten().tolist():
                        end_offset = len(chunk_valid_steps[env_id])
                        chunk_valid_steps[env_id].append(end)
                        history = valid_history[env_id]
                        history.append((chunk_index, end_offset))
                        if len(history) == sequence_steps:
                            start_chunk, start_offset = history[0]
                            self._samples.append(
                                (
                                    chunk_index,
                                    env_id,
                                    end,
                                    start_chunk,
                                    start_offset,
                                    end_offset,
                                )
                            )
                            self._sample_terrain_types.append(
                                int(payload["terrain_type"][end, env_id].item())
                            )
                self._valid_steps.append(
                    tuple(torch.tensor(indices, dtype=torch.long) for indices in chunk_valid_steps)
                )
                continue
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
                        self._sample_terrain_types.append(
                            int(payload["terrain_type"][end, env_id].item())
                        )

        if len(self._sample_terrain_types) != len(self._samples):
            raise RuntimeError("terrain sample metadata lost alignment with sequence indices")

    @staticmethod
    def _validate_chunk(payload: dict[str, torch.Tensor], manifest: dict[str, Any]) -> None:
        frame_type = OdometryFreeTerrainPerceptionFrameBatch if manifest["version"] == 2 else TerrainPerceptionFrameBatch
        if manifest["version"] == 3:
            frame_type = OdometryFreeTerrainPerceptionFrameBatch
        expected_keys = {field.name for field in fields(frame_type)}
        if manifest["version"] == 2:
            expected_keys.remove("frame_valid")
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

    def sample_indices_for_terrain_ids(self, terrain_ids: tuple[int, ...]) -> list[int]:
        """Return sequence indices whose endpoint belongs to an approved terrain set."""
        selected_ids = {int(value) for value in terrain_ids}
        if not selected_ids or any(value < 0 for value in selected_ids):
            raise ValueError("terrain_ids must contain non-negative terrain IDs")
        return [
            index
            for index, terrain_type in enumerate(self._sample_terrain_types)
            if terrain_type in selected_ids
        ]

    def _load_chunk(self, chunk_index: int) -> dict[str, torch.Tensor]:
        cached = self._cached_chunks.pop(chunk_index, None)
        if cached is None:
            cached = torch.load(
                self._chunk_files[chunk_index],
                map_location="cpu",
                weights_only=True,
            )
        self._cached_chunks[chunk_index] = cached
        while len(self._cached_chunks) > 2:
            self._cached_chunks.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_location = self._samples[index]
        chunk_index, env_id, end = sample_location[:3]
        chunk = self._load_chunk(chunk_index)
        if self.has_frame_valid:
            _chunk, _env, _end, start_chunk, start_offset, end_offset = sample_location
            locations: list[tuple[int, torch.Tensor]] = []
            for source_index in range(start_chunk, chunk_index + 1):
                valid_steps = self._valid_steps[source_index][env_id]
                lower = start_offset if source_index == start_chunk else 0
                upper = end_offset + 1 if source_index == chunk_index else len(valid_steps)
                if upper > lower:
                    locations.append((source_index, valid_steps[lower:upper]))
            if sum(int(indices.numel()) for _source, indices in locations) != self.sequence_steps:
                raise RuntimeError("v3 terrain sample lost its indexed camera-frame history")
        else:
            start = end - self.sequence_steps + 1
            indices = slice(start, end + 1)
        sequence_keys = [
            "partial_map",
            "visible_mask",
            "timestamp_s",
            "proprio",
        ]
        if self.has_frame_valid:
            sequence_keys.append("frame_valid")
        if not self.odometry_free:
            sequence_keys.extend(("pelvis_pos_w", "heading_yaw_w"))
        if self.has_frame_valid:
            sample = {
                key: torch.cat(
                    [self._load_chunk(source)[key][indices, env_id] for source, indices in locations],
                    dim=0,
                )
                for key in sequence_keys
            }
        else:
            sample = {key: chunk[key][indices, env_id] for key in sequence_keys}
        if not self.has_frame_valid:
            sample["frame_valid"] = torch.ones(self.sequence_steps, dtype=torch.bool)
        sample.update(
            {
                "gt_terrain_actor": chunk["gt_terrain_actor"][end, env_id],
                "episode_id": chunk["episode_id"][end, env_id],
                "env_id": chunk["env_id"][end, env_id],
                "terrain_type": chunk["terrain_type"][end, env_id],
            }
        )
        return sample
