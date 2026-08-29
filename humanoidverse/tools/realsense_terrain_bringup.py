"""Replay synchronized D435i frames through the hardware terrain pipeline.

The input ``.npz`` is intentionally explicit so calibration and pose timing
can be validated before a robot is allowed to move. Required arrays are:
``depth`` (T,H,W), ``torso_pos_w`` (T,3), ``torso_quat_w_xyzw`` (T,4),
``pelvis_pos_w`` (T,3), ``pelvis_heading_quat_w_xyzw`` (T,4), and
``timestamp_s`` (T,). ``depth`` is native uint16 unless ``--depth-in-meters``
is supplied. All poses use world-frame xyzw quaternions. Live-logger NPZs also
contain ``runtime_frame_index``; replay applies it so a 60 Hz raw stream is
consumed on the timestamp-derived 50 Hz runtime cadence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from humanoidverse.perception.depth_augmentation import MetricDepthAugmentationConfig
from humanoidverse.perception.realsense_depth_runtime import RealSenseCalibration, RealSenseDepthRuntime


def _array(payload: np.lib.npyio.NpzFile, name: str, shape_tail: tuple[int, ...]) -> torch.Tensor:
    if name not in payload:
        raise ValueError(f"input npz is missing {name!r}")
    value = np.asarray(payload[name])
    if value.ndim != len(shape_tail) + 1 or tuple(value.shape[1:]) != shape_tail:
        raise ValueError(f"{name} must have shape [T, {', '.join(map(str, shape_tail))}], got {value.shape}")
    return torch.from_numpy(value)


def run_replay(
    *,
    input_path: Path,
    output_path: Path,
    calibration_path: Path,
    perception_checkpoint: Path | None,
    device: str,
    depth_in_meters: bool,
    depth_augmentation: MetricDepthAugmentationConfig | None = None,
) -> dict[str, object]:
    calibration = RealSenseCalibration.from_json(calibration_path)
    with np.load(input_path, allow_pickle=False) as payload:
        depth = _array(payload, "depth", (calibration.native_height, calibration.native_width))
        torso_pos = _array(payload, "torso_pos_w", (3,)).float()
        torso_quat = _array(payload, "torso_quat_w_xyzw", (4,)).float()
        pelvis_pos = _array(payload, "pelvis_pos_w", (3,)).float()
        heading_quat = _array(payload, "pelvis_heading_quat_w_xyzw", (4,)).float()
        timestamps = _array(payload, "timestamp_s", ()).float().reshape(-1)
        proprio = torch.from_numpy(np.asarray(payload["proprio"])).float() if "proprio" in payload else None
        reset = torch.from_numpy(np.asarray(payload["reset_mask"]).astype(bool)) if "reset_mask" in payload else None
        source_frame_count = int(depth.shape[0])
        runtime_selection: dict[str, object] = {"mode": "all_input_frames"}
        if "runtime_frame_index" in payload:
            raw_indices = np.asarray(payload["runtime_frame_index"])
            if raw_indices.ndim != 1 or raw_indices.size == 0 or not np.issubdtype(raw_indices.dtype, np.integer):
                raise ValueError("runtime_frame_index must be a non-empty integer vector")
            indices = raw_indices.astype(np.int64, copy=False)
            if np.any(indices < 0) or np.any(indices >= source_frame_count) or np.any(np.diff(indices) < 0):
                raise ValueError("runtime_frame_index must be ordered and within the raw frame range")
            index = torch.from_numpy(indices)
            # PyTorch has no CPU advanced-index kernel for uint16. Int32 keeps
            # native Z16 values exact and depth_to_meters handles it normally.
            depth = depth.to(torch.int32)[index]
            torso_pos = torso_pos[index]
            torso_quat = torso_quat[index]
            pelvis_pos = pelvis_pos[index]
            heading_quat = heading_quat[index]
            timestamps = timestamps[index]
            if proprio is not None:
                proprio = proprio[index]
            if reset is not None:
                reset = reset[index]
                if reset.numel() > 1:
                    reset[1:] &= index[1:] != index[:-1]
            runtime_selection = {
                "mode": "npz_runtime_frame_index",
                "runtime_ticks": int(indices.size),
                "unique_camera_frames": int(np.unique(indices).size),
            }
            if "runtime_camera_age_s" in payload:
                camera_age_s = np.asarray(payload["runtime_camera_age_s"], dtype=np.float64)
                if camera_age_s.shape != indices.shape or not np.isfinite(camera_age_s).all():
                    raise ValueError("runtime_camera_age_s must be finite and match runtime_frame_index")
                runtime_selection["camera_age_p95_ms"] = float(np.percentile(camera_age_s, 95) * 1.0e3)
                runtime_selection["camera_age_max_ms"] = float(np.max(camera_age_s) * 1.0e3)

    frame_count = int(depth.shape[0])
    if any(int(value.shape[0]) != frame_count for value in (torso_pos, torso_quat, pelvis_pos, heading_quat, timestamps)):
        raise ValueError("all input arrays must have the same frame count")
    if proprio is not None and int(proprio.shape[0]) != frame_count:
        raise ValueError("proprio must have the same frame count as depth")
    if reset is not None and reset.shape != (frame_count,):
        raise ValueError("reset_mask must have shape [T]")
    if depth_in_meters:
        depth = depth / float(calibration.depth_scale_m)

    runtime = RealSenseDepthRuntime(
        calibration=calibration,
        perception_checkpoint=perception_checkpoint,
        device=device,
        batch_size=1,
        depth_augmentation=depth_augmentation,
    )
    partial_maps: list[np.ndarray] = []
    visible_masks: list[np.ndarray] = []
    completed_maps: list[np.ndarray] = []
    visible_fractions: list[float] = []
    try:
        for index in range(frame_count):
            output = runtime.step(
                depth[index].unsqueeze(0),
                torso_pos_w=torso_pos[index].unsqueeze(0),
                torso_quat_w=torso_quat[index].unsqueeze(0),
                pelvis_pos_w=pelvis_pos[index].unsqueeze(0),
                pelvis_heading_quat_w=heading_quat[index].unsqueeze(0),
                timestamp_s=timestamps[index].reshape(1),
                proprio=None if proprio is None else proprio[index].unsqueeze(0),
                reset_mask=torch.tensor([bool(reset[index]) if reset is not None else index == 0]),
            )
            partial_maps.append(output.partial_map[0].cpu().numpy())
            visible_masks.append(output.visible_mask[0].cpu().numpy())
            completed_maps.append(output.terrain_actor[0].cpu().numpy())
            visible_fractions.append(float(output.visible_mask.float().mean().item()))
    finally:
        del runtime

    partial = np.stack(partial_maps)
    visible = np.stack(visible_masks)
    completed = np.stack(completed_maps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path.with_suffix(".npz"), partial_map=partial, visible_mask=visible, terrain_actor=completed)
    summary = {
        "input": str(input_path.expanduser().resolve()),
        "output_npz": str(output_path.with_suffix(".npz").resolve()),
        "calibration": str(calibration_path.expanduser().resolve()),
        "perception_checkpoint": None if perception_checkpoint is None else str(perception_checkpoint.expanduser().resolve()),
        "source_frames": source_frame_count,
        "frames": frame_count,
        "runtime_frame_selection": runtime_selection,
        "native_depth_shape": list(depth.shape[1:]),
        "target_depth_shape": [calibration.target_height, calibration.target_width],
        "full_fov_downsample": True,
        "depth_input_unit": "meters" if depth_in_meters else "native_depth_units",
        "visible_fraction_mean": float(np.mean(visible_fractions)),
        "partial_map_finite_fraction": float(np.isfinite(partial).mean()),
        "completed_map_finite_fraction": float(np.isfinite(completed).mean()),
        "completed_map_clearance_range_m": [
            float(np.nanmin(completed)),
            float(np.nanmax(completed)),
        ]
        if np.isfinite(completed).any()
        else None,
        "pose_contract": "world-frame positions and xyzw quaternions; camera optical axes are +x right, +y down, +z forward",
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output summary JSON; maps are saved beside it as .npz")
    parser.add_argument("--perception-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--depth-in-meters", action="store_true")
    parser.add_argument("--depth-gate-max", type=float, default=None)
    parser.add_argument("--blur-probability", type=float, default=0.0)
    parser.add_argument("--blur-sigma-min-px", type=float, default=0.0)
    parser.add_argument("--blur-sigma-max-px", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depth_augmentation = None
    if args.depth_gate_max is not None or args.blur_probability > 0.0 or args.blur_sigma_max_px > 0.0:
        depth_augmentation = MetricDepthAugmentationConfig(
            max_depth_m=2.0 if args.depth_gate_max is None else args.depth_gate_max,
            blur_probability=args.blur_probability,
            sigma_min_px=args.blur_sigma_min_px,
            sigma_max_px=args.blur_sigma_max_px,
        )
    summary = run_replay(
        input_path=args.input_npz,
        output_path=args.output,
        calibration_path=args.calibration,
        perception_checkpoint=args.perception_checkpoint,
        device=args.device,
        depth_in_meters=args.depth_in_meters,
        depth_augmentation=depth_augmentation,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
