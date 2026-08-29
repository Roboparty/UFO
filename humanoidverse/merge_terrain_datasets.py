"""Merge equal-length terrain perception shards along the environment axis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def merge_terrain_datasets(inputs: list[Path], output: Path) -> dict[str, object]:
    if len(inputs) < 2:
        raise ValueError("at least two input shards are required")
    manifests = [json.loads((path / "manifest.json").read_text()) for path in inputs]
    for manifest in manifests:
        if manifest.get("format") != "pbfm_temporal_terrain" or manifest.get("version") != 1:
            raise ValueError("all shards must use pbfm_temporal_terrain version 1")
    chunk_counts = [len(manifest["chunks"]) for manifest in manifests]
    if len(set(chunk_counts)) != 1:
        raise ValueError(f"all shards must have the same chunk count, got {chunk_counts}")
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output dataset is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    offsets: list[int] = []
    offset = 0
    for manifest in manifests:
        offsets.append(offset)
        offset += int(manifest["num_envs"])
    merged_chunks: list[dict[str, object]] = []
    for chunk_index in range(chunk_counts[0]):
        payloads = []
        steps: list[int] = []
        for shard_index, (root, manifest) in enumerate(zip(inputs, manifests, strict=True)):
            filename = manifest["chunks"][chunk_index]["file"]
            payload = torch.load(root / filename, map_location="cpu", weights_only=True)
            payloads.append(payload)
            steps.append(int(payload["partial_map"].shape[0]))
            if int(payload["partial_map"].shape[1]) != int(manifest["num_envs"]):
                raise ValueError(f"shard {root} chunk {chunk_index} has inconsistent env count")
            payload["env_id"] = payload["env_id"] + offsets[shard_index]
        if len(set(steps)) != 1:
            raise ValueError(f"chunk {chunk_index} has mismatched step counts: {steps}")
        keys = set(payloads[0])
        if any(set(payload) != keys for payload in payloads[1:]):
            raise ValueError(f"chunk {chunk_index} keys differ between shards")
        merged = {key: torch.cat([payload[key] for payload in payloads], dim=1) for key in keys}
        filename = f"chunk_{chunk_index:06d}.pt"
        temporary = output / f".{filename}.tmp"
        torch.save(merged, temporary)
        os.replace(temporary, output / filename)
        merged_chunks.append({"file": filename, "steps": steps[0]})

    metadata = dict(manifests[0].get("metadata", {}))
    metadata["merged_shards"] = [str(path.expanduser().resolve()) for path in inputs]
    metadata["merged_num_shards"] = len(inputs)
    manifest = {
        "format": "pbfm_temporal_terrain",
        "version": 1,
        "grid_dimension": 273,
        "num_envs": offset,
        "proprio_dim": manifests[0]["proprio_dim"],
        "chunks": merged_chunks,
        "metadata": metadata,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = merge_terrain_datasets(args.input, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "num_envs": result["num_envs"]}, indent=2))


if __name__ == "__main__":
    main()
