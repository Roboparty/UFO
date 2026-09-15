from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoidverse.tools.motion_trim_tpose import (
    TRIM_MARKER_KEY,
    detect_tpose_end,
    ffmpeg_trim_command,
    plan_trim,
    plan_video_trim,
    probe_video_frames,
    sibling_video_path,
    trim_motion_npz,
    trim_motion_video,
)

JOINT_NAMES = [
    "left_hip_y",
    "left_knee",
    "right_hip_y",
    "right_knee",
    "left_shoulder_y",
    "left_elbow",
    "right_shoulder_y",
    "right_elbow",
]

FPS = 30.0


def _synthetic_motion(tpose_frames: int = 90, total_frames: int = 400) -> dict[str, np.ndarray]:
    """Static T-pose (arms out, elbows straight), a short transition, then a walking-like motion."""
    rng = np.random.default_rng(0)
    joint_pos = np.zeros((total_frames, len(JOINT_NAMES)), dtype=np.float64)
    joint_pos[:, 4] = 1.57
    joint_pos[:, 6] = -1.57
    joint_pos += rng.normal(scale=1e-4, size=joint_pos.shape)

    transition = 15
    ramp = np.linspace(0.0, 1.0, transition)[:, None]
    lowered = np.array([0.0, 0.5, 0.0, 0.5, 0.0, -1.5, 0.0, -1.5])
    tpose = joint_pos[tpose_frames].copy()
    joint_pos[tpose_frames : tpose_frames + transition] = tpose + ramp * (lowered - tpose)

    after = total_frames - tpose_frames - transition
    swing = 0.6 * np.sin(np.linspace(0.0, 12.0, after))[:, None]
    joint_pos[tpose_frames + transition :] = lowered + swing

    return {
        "framerate": np.array(FPS),
        "joint_names": np.array(JOINT_NAMES),
        "joint_pos": joint_pos,
        "base_pos_w": np.cumsum(np.full((total_frames, 3), 0.001), axis=0),
        "base_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (total_frames, 1)),
    }


def _write_npz(directory: Path, name: str = "motion.npz", **overrides) -> Path:
    data = _synthetic_motion(**overrides)
    path = directory / name
    np.savez(path, **data)
    return path


class TestDetectTposeEnd(unittest.TestCase):
    def test_cut_lands_after_the_arms_come_down(self) -> None:
        data = _synthetic_motion(tpose_frames=90)
        start = detect_tpose_end(data["joint_pos"], [str(n) for n in data["joint_names"]], FPS)
        self.assertGreater(start, 90)
        self.assertLess(start, 130)

    def test_longer_tpose_gives_a_later_cut(self) -> None:
        short = _synthetic_motion(tpose_frames=60)
        long = _synthetic_motion(tpose_frames=200)
        short_start = detect_tpose_end(short["joint_pos"], JOINT_NAMES, FPS)
        long_start = detect_tpose_end(long["joint_pos"], JOINT_NAMES, FPS)
        self.assertGreater(long_start, short_start + 100)

    def test_motion_without_leading_tpose_is_not_trimmed(self) -> None:
        rng = np.random.default_rng(1)
        joint_pos = 0.6 * np.sin(np.linspace(0.0, 30.0, 300))[:, None] + rng.normal(scale=0.01, size=(300, len(JOINT_NAMES)))
        self.assertEqual(detect_tpose_end(joint_pos, JOINT_NAMES, FPS), 0)

    def test_trim_is_capped_by_max_trim_seconds(self) -> None:
        data = _synthetic_motion(tpose_frames=350, total_frames=600)
        start = detect_tpose_end(data["joint_pos"], JOINT_NAMES, FPS, max_trim_seconds=2.0)
        self.assertLessEqual(start, 60)


class TestTrimMotionNpz(unittest.TestCase):
    def test_all_time_series_are_trimmed_consistently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_npz(Path(tmp), tpose_frames=90, total_frames=400)
            with np.load(path) as original:
                base_pos_before = np.array(original["base_pos_w"])

            plan = plan_trim(path)
            trim_motion_npz(path, plan)

            with np.load(path) as trimmed:
                self.assertEqual(trimmed["joint_pos"].shape[0], 400 - plan.start_frame)
                self.assertEqual(trimmed["base_pos_w"].shape[0], 400 - plan.start_frame)
                self.assertEqual(trimmed["base_quat_w"].shape[0], 400 - plan.start_frame)
                self.assertEqual(trimmed["joint_names"].shape[0], len(JOINT_NAMES))
                self.assertEqual(float(trimmed["framerate"]), FPS)
                self.assertEqual(int(trimmed[TRIM_MARKER_KEY]), plan.start_frame)
                np.testing.assert_allclose(trimmed["base_pos_w"][0], base_pos_before[plan.start_frame])

    def test_already_trimmed_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_npz(Path(tmp), tpose_frames=90, total_frames=400)
            first = plan_trim(path)
            trim_motion_npz(path, first)

            second = plan_trim(path)
            self.assertTrue(second.already_trimmed)
            self.assertEqual(second.start_frame, 0)

            with np.load(path) as before:
                frames_before = before["joint_pos"].shape[0]
            trim_motion_npz(path, second)
            with np.load(path) as after:
                self.assertEqual(after["joint_pos"].shape[0], frames_before)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe not installed")
class TestTrimMotionVideo(unittest.TestCase):
    def test_ffmpeg_command_keeps_the_same_cut_as_the_npz(self) -> None:
        cmd = ffmpeg_trim_command(Path("a.mp4"), Path("b.mp4"), 99)
        self.assertEqual(cmd[cmd.index("-vf") + 1], "trim=start_frame=99,setpts=PTS-STARTPTS")

    def test_sibling_video_is_trimmed_to_the_npz_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            npz_path = _write_npz(directory, name="clip_retargeted.npz", tpose_frames=20, total_frames=90)
            plan = plan_trim(npz_path)
            trim_motion_npz(npz_path, plan)

            video_path = directory / "clip_retargeted.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=red:s=16x16:r=30:d=3",
                    "-frames:v",
                    "90",
                    "-pix_fmt",
                    "yuv420p",
                    str(video_path),
                ],
                check=True,
                stdin=subprocess.DEVNULL,
            )
            self.assertEqual(sibling_video_path(npz_path), video_path)
            self.assertEqual(probe_video_frames(video_path), 90)

            video_plan = plan_video_trim(npz_path)
            self.assertIsNotNone(video_plan)
            assert video_plan is not None
            self.assertEqual(video_plan.start_frame, plan.start_frame)
            self.assertTrue(trim_motion_video(video_plan))
            self.assertEqual(probe_video_frames(video_path), 90 - plan.start_frame)

            second = plan_video_trim(npz_path)
            self.assertIsNotNone(second)
            assert second is not None
            self.assertTrue(second.already_trimmed)
            self.assertFalse(trim_motion_video(second))


if __name__ == "__main__":
    unittest.main()
