"""Build a UFO motion data manifest from robot_state CSV/NPZ sources."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from humanoidverse.tools.data_inspect import inspect_data_source
from humanoidverse.utils.motion_data import prepare_motion_manifest


def build_manifest_config(
    *,
    robot: str,
    source: str | list[str],
    fmt: str,
    name: str,
    weight: float = 1.0,
    fps: float | None = None,
    clip_seconds: float = 10.0,
    stride_seconds: float | None = None,
    keep_short: bool = True,
    min_clip_seconds: float = 1.0,
) -> dict[str, Any]:
    result = inspect_data_source(
        robot_config=robot,
        source=source,
        fmt=fmt,
        fps=fps,
        clip_seconds=clip_seconds,
        stride_seconds=stride_seconds,
        keep_short=keep_short,
        min_clip_seconds=min_clip_seconds,
        dataset_name=name,
        weight=weight,
    )
    return result.suggested_manifest


def write_manifest(config: dict[str, Any], out_path: str | Path, *, force: bool = False) -> Path:
    path = Path(out_path).expanduser()
    if path.exists() and not force:
        raise FileExistsError(f"Data manifest already exists: {path}. Use --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, help="Robot YAML path")
    parser.add_argument("--source", required=True, nargs="+", help="CSV/NPZ source path, directory, or glob")
    parser.add_argument("--format", default="auto", choices=["auto", "robot_state_csv", "robot_state_npz"])
    parser.add_argument("--name", required=True, help="Dataset name")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=None)
    parser.add_argument("--keep-short", dest="keep_short", action="store_true", default=True)
    parser.add_argument("--drop-short", dest="keep_short", action="store_false")
    parser.add_argument("--min-clip-seconds", type=float, default=1.0)
    parser.add_argument("--out", required=True, help="Output data manifest path")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force cache rebuild when building")
    parser.add_argument("--no-build", action="store_true", help="Only write the manifest")
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifest")
    args = parser.parse_args(argv)

    source: str | list[str] = args.source[0] if len(args.source) == 1 else args.source
    config = build_manifest_config(
        robot=args.robot,
        source=source,
        fmt=args.format,
        name=args.name,
        weight=args.weight,
        fps=args.fps,
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        keep_short=args.keep_short,
        min_clip_seconds=args.min_clip_seconds,
    )
    dataset = config["datasets"][0]
    print(f"Detected RobotState schema via {dataset['format']} reader")
    manifest_path = write_manifest(config, args.out, force=args.force)
    print(f"Wrote data manifest: {manifest_path}")

    if args.no_build:
        return
    result = prepare_motion_manifest(manifest_path, rebuild_cache=args.rebuild_cache)
    print(f"manifest path: {result.manifest_path}")
    print(f"train_data_paths: {result.train_data_paths}")
    print(f"train_data_weights: {result.train_data_weights}")
    print(f"inference_paths: {result.inference_paths}")


if __name__ == "__main__":
    main()
