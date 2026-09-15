"""Trim the leading T-pose calibration segment from retargeted motion NPZ files.

LAFAN1 captures start with the subject holding a static T-pose for a few seconds before the
actual motion begins. The cut is found where the arms leave the pose held in frame 0, which
survives the +-pi wrapping that makes absolute shoulder angles unreliable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

TRIM_MARKER_KEY = "tpose_trimmed_frames"

ARM_JOINT_KEYWORDS = ("shoulder", "elbow")
DEFAULT_ARM_DEVIATION = 0.45
DEFAULT_STATIC_SPEED = 0.004
DEFAULT_MIN_STATIC_SECONDS = 0.3
DEFAULT_MARGIN_SECONDS = 0.2
DEFAULT_MAX_TRIM_SECONDS = 20.0

FPS_KEYS = ("framerate", "fps")


@dataclass(frozen=True)
class TrimPlan:
    path: Path
    frames: int
    fps: float
    start_frame: int
    already_trimmed: bool = False

    @property
    def kept_frames(self) -> int:
        return self.frames - self.start_frame

    @property
    def trimmed_seconds(self) -> float:
        return self.start_frame / self.fps


@dataclass(frozen=True)
class VideoTrimPlan:
    npz_path: Path
    video_path: Path
    start_frame: int
    video_frames: int
    npz_frames: int
    fps: float
    already_trimmed: bool = False

    @property
    def kept_frames(self) -> int:
        return self.npz_frames


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _arm_joint_indices(joint_names: list[str]) -> list[int]:
    indices = [idx for idx, name in enumerate(joint_names) if any(key in name.lower() for key in ARM_JOINT_KEYWORDS)]
    return indices if indices else list(range(len(joint_names)))


def detect_tpose_end(
    joint_pos: np.ndarray,
    joint_names: list[str],
    fps: float,
    *,
    arm_deviation: float = DEFAULT_ARM_DEVIATION,
    static_speed: float = DEFAULT_STATIC_SPEED,
    min_static_seconds: float = DEFAULT_MIN_STATIC_SECONDS,
    margin_seconds: float = DEFAULT_MARGIN_SECONDS,
    max_trim_seconds: float = DEFAULT_MAX_TRIM_SECONDS,
) -> int:
    """Return the first frame to keep, or 0 when the motion does not start in a T-pose."""
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    limit = min(joint_pos.shape[0] - 1, int(max_trim_seconds * fps))
    if limit <= 1:
        return 0

    # A calibration pose is held still; without that the motion starts right away and nothing is dropped.
    speed = np.abs(np.diff(joint_pos[: limit + 1], axis=0)).mean(axis=1)
    active = np.flatnonzero(speed > static_speed)
    static_end = int(active[0]) if active.size else limit
    if static_end < min_static_seconds * fps:
        return 0

    # The held pose only ends once the arms swing away from it.
    arm_idx = _arm_joint_indices(joint_names)
    deviation = np.abs(_wrap_angle(joint_pos[:limit, arm_idx] - joint_pos[0, arm_idx])).mean(axis=1)
    moved = np.flatnonzero(deviation > arm_deviation)
    arm_end = int(moved[0]) if moved.size else 0

    return min(max(arm_end, static_end) + int(round(margin_seconds * fps)), limit)


def _fps_from_npz(data: Any, path: Path) -> float:
    for key in FPS_KEYS:
        if key in data:
            fps = float(np.asarray(data[key]).reshape(-1)[0])
            if not np.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"NPZ file={path} has invalid {key}={fps}")
            return fps
    raise ValueError(f"NPZ file={path} has no {' or '.join(FPS_KEYS)} field")


def plan_trim(path: Path, **detect_kwargs: float) -> TrimPlan:
    with np.load(path, allow_pickle=True) as data:
        if TRIM_MARKER_KEY in data:
            frames = int(np.asarray(data["joint_pos"]).shape[0])
            return TrimPlan(path=path, frames=frames, fps=_fps_from_npz(data, path), start_frame=0, already_trimmed=True)
        joint_pos = np.asarray(data["joint_pos"])
        joint_names = [str(name) for name in np.asarray(data["joint_names"]).reshape(-1)]
        fps = _fps_from_npz(data, path)
    start_frame = detect_tpose_end(joint_pos, joint_names, fps, **detect_kwargs)  # type: ignore[arg-type]
    return TrimPlan(path=path, frames=int(joint_pos.shape[0]), fps=fps, start_frame=start_frame)


def trim_motion_npz(path: Path, plan: TrimPlan) -> bool:
    """Rewrite the NPZ in place with the leading T-pose removed. Returns True when the file changed."""
    if plan.already_trimmed or plan.start_frame <= 0:
        return False

    with np.load(path, allow_pickle=True) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    trimmed = {
        key: value[plan.start_frame :] if value.ndim >= 1 and value.shape[0] == plan.frames else value for key, value in arrays.items()
    }
    trimmed[TRIM_MARKER_KEY] = np.array(plan.start_frame)

    # Write next to the target so the replace stays atomic on the same filesystem.
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".npz.tmp")
    os.close(handle)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("wb") as f:
            np.savez(f, **trimmed)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return True


def sibling_video_path(npz_path: Path) -> Path | None:
    candidate = npz_path.with_suffix(".mp4")
    return candidate if candidate.exists() else None


def probe_video_frames(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    raw = result.stdout.strip()
    if raw.isdigit():
        return int(raw)
    raise ValueError(f"ffprobe could not read frame count from {path}: {raw!r}")


def ffmpeg_trim_command(src: Path, dst: Path, start_frame: int) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"trim=start_frame={start_frame},setpts=PTS-STARTPTS",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(dst),
    ]


def read_trim_marker(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        if TRIM_MARKER_KEY not in data:
            return 0
        return int(np.asarray(data[TRIM_MARKER_KEY]).reshape(-1)[0])


def plan_video_trim(npz_path: Path) -> VideoTrimPlan | None:
    video_path = sibling_video_path(npz_path)
    if video_path is None:
        return None
    with np.load(npz_path, allow_pickle=True) as data:
        npz_frames = int(np.asarray(data["joint_pos"]).shape[0])
        fps = _fps_from_npz(data, npz_path)
        dropped = read_trim_marker(npz_path) if TRIM_MARKER_KEY in data else 0
    video_frames = probe_video_frames(video_path)
    if dropped <= 0:
        return VideoTrimPlan(
            npz_path=npz_path,
            video_path=video_path,
            start_frame=0,
            video_frames=video_frames,
            npz_frames=npz_frames,
            fps=fps,
            already_trimmed=video_frames == npz_frames,
        )
    if video_frames == npz_frames:
        return VideoTrimPlan(
            npz_path=npz_path,
            video_path=video_path,
            start_frame=0,
            video_frames=video_frames,
            npz_frames=npz_frames,
            fps=fps,
            already_trimmed=True,
        )
    if video_frames != npz_frames + dropped:
        raise ValueError(
            f"{video_path.name}: expected {npz_frames + dropped} frames before trim or {npz_frames} after trim, got {video_frames}"
        )
    return VideoTrimPlan(
        npz_path=npz_path,
        video_path=video_path,
        start_frame=dropped,
        video_frames=video_frames,
        npz_frames=npz_frames,
        fps=fps,
    )


def trim_motion_video(plan: VideoTrimPlan) -> bool:
    """Rewrite the sibling MP4 in place so it starts at the same frame as the trimmed NPZ."""
    if plan.already_trimmed or plan.start_frame <= 0:
        return False
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to trim motion videos")

    handle, tmp_name = tempfile.mkstemp(dir=str(plan.video_path.parent), suffix=".tmp.mp4")
    os.close(handle)
    tmp_path = Path(tmp_name)
    try:
        subprocess.run(ffmpeg_trim_command(plan.video_path, tmp_path, plan.start_frame), check=True)
        os.replace(tmp_path, plan.video_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return True


def find_motion_files(roots: list[str], pattern: str) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        candidate = Path(root)
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(sorted(candidate.rglob(pattern)))
        else:
            paths.extend(sorted(Path().glob(root)))
    return sorted({path.resolve() for path in paths})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, nargs="+", help="NPZ file, directory, or glob to trim in place")
    parser.add_argument("--pattern", default="*.npz", help="Glob used when --source points at a directory")
    parser.add_argument("--arm-deviation", type=float, default=DEFAULT_ARM_DEVIATION, help="rad, mean arm-joint distance from frame 0")
    parser.add_argument("--static-speed", type=float, default=DEFAULT_STATIC_SPEED, help="rad/frame, mean joint speed of a held pose")
    parser.add_argument("--min-static-seconds", type=float, default=DEFAULT_MIN_STATIC_SECONDS, help="Shortest held pose worth trimming")
    parser.add_argument("--margin-seconds", type=float, default=DEFAULT_MARGIN_SECONDS, help="Extra time dropped after the transition")
    parser.add_argument("--max-trim-seconds", type=float, default=DEFAULT_MAX_TRIM_SECONDS, help="Upper bound on what may be dropped")
    parser.add_argument("--dry-run", action="store_true", help="Report the detected cuts without touching the files")
    parser.add_argument("--skip-npz", action="store_true", help="Leave NPZ files unchanged")
    parser.add_argument("--skip-video", action="store_true", help="Leave sibling MP4 files unchanged")
    parser.add_argument("--video-jobs", type=int, default=4, help="Parallel ffmpeg encodes for sibling videos")
    args = parser.parse_args(argv)

    paths = find_motion_files(args.source, args.pattern)
    if not paths:
        raise SystemExit(f"No NPZ files matched source={args.source} pattern={args.pattern!r}")

    trimmed_count = 0
    skipped: list[Path] = []
    if not args.skip_npz:
        for path in paths:
            plan = plan_trim(
                path,
                arm_deviation=args.arm_deviation,
                static_speed=args.static_speed,
                min_static_seconds=args.min_static_seconds,
                margin_seconds=args.margin_seconds,
                max_trim_seconds=args.max_trim_seconds,
            )
            if plan.already_trimmed:
                logger.info(f"{path.name}: already trimmed, skipping")
                skipped.append(path)
                continue
            if plan.start_frame == 0:
                logger.warning(f"{path.name}: no leading T-pose detected, skipping")
                skipped.append(path)
                continue
            logger.info(
                f"{path.name}: drop {plan.start_frame} frames ({plan.trimmed_seconds:.2f}s), {plan.frames} -> {plan.kept_frames} frames"
            )
            if not args.dry_run:
                trim_motion_npz(path, plan)
            trimmed_count += 1
        action = "would trim" if args.dry_run else "trimmed"
        logger.info(f"{action} {trimmed_count}/{len(paths)} NPZ files, skipped {len(skipped)}")

    if args.skip_video:
        return

    video_plans = [plan_video_trim(path) for path in paths]
    pending = [plan for plan in video_plans if plan is not None and not plan.already_trimmed and plan.start_frame > 0]
    aligned = sum(1 for plan in video_plans if plan is not None and plan.already_trimmed)
    missing = sum(1 for plan in video_plans if plan is None)
    for plan in pending:
        logger.info(
            f"{plan.video_path.name}: drop {plan.start_frame} frames "
            f"({plan.start_frame / plan.fps:.2f}s), {plan.video_frames} -> {plan.kept_frames} frames"
        )
    if args.dry_run:
        logger.info(f"would trim {len(pending)}/{len(paths)} MP4 files, already aligned {aligned}, missing {missing}")
        return

    trimmed_videos = 0
    workers = max(1, args.video_jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(trim_motion_video, plan): plan for plan in pending}
        for future in as_completed(futures):
            plan = futures[future]
            future.result()
            trimmed_videos += 1
            logger.info(f"{plan.video_path.name}: wrote {plan.kept_frames} frames")
    logger.info(f"trimmed {trimmed_videos}/{len(paths)} MP4 files, already aligned {aligned}, missing {missing}")


if __name__ == "__main__":
    main()
