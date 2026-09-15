from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from humanoidverse.train import build_ufo_mjlab_config
from humanoidverse.train import parse_args as parse_train_args
from humanoidverse.training.workspace import build_video_preview_command


class VideoPreviewConfigTest(unittest.TestCase):
    def test_defaults_disable_previews(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit_video",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
        )
        self.assertEqual(cfg.video_every_checkpoints, 0)
        self.assertEqual(cfg.tags["robot"], "g1_29dof")

    def test_preview_command_uses_checkpoint_robot_and_data(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cuda:0",
            work_dir="/tmp/ufo_unit_video",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            data_path="humanoidverse/data/lafan_29dof_10s-clipped.pkl",
            video_every_checkpoints=2,
            video_motion_ids=[3, 7],
            video_max_steps=123,
            video_render_size=256,
            video_data_path="/data/full_motions.pkl",
        )
        cmd = build_video_preview_command(model_folder=Path("/runs/exp"), cfg=cfg)
        self.assertEqual(cmd[:3], [sys.executable, "-m", "humanoidverse.tracking_inference"])
        self.assertIn("/runs/exp", cmd)
        self.assertEqual(cmd[cmd.index("--device") + 1], "cuda:0")
        self.assertEqual(cmd[cmd.index("--export-onnx") + 1], "false")
        self.assertEqual(cmd[cmd.index("--save-mp4") + 1], "true")
        motion_start = cmd.index("--motion-list") + 1
        self.assertEqual(cmd[motion_start : motion_start + 2], ["3", "7"])
        self.assertEqual(cmd[cmd.index("--max-steps") + 1], "123")
        self.assertEqual(cmd[cmd.index("--render-size") + 1], "256")
        self.assertEqual(cmd[cmd.index("--data-path") + 1], "/data/full_motions.pkl")
        self.assertTrue(cmd[cmd.index("--robot-config") + 1].endswith("configs/robots/g1_29dof.yaml"))

    def test_preview_command_falls_back_to_training_data_path(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit_video",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            data_path=["/data/a.pkl", "/data/b.pkl"],
            data_mix_weights=[0.5, 0.5],
            video_every_checkpoints=1,
        )
        cmd = build_video_preview_command(model_folder=Path("/runs/exp"), cfg=cfg)
        self.assertEqual(cmd[cmd.index("--data-path") + 1], "/data/a.pkl")

    def test_cli_flags_and_manifest_inference_path(self) -> None:
        argv = [
            "train.py",
            "--agent",
            "fb",
            "--data-manifest",
            "configs/data/example_mix.yaml",
            "--gpu-ids",
            "single",
            "--smoke",
            "--video-every-checkpoints",
            "3",
            "--video-motion-ids",
            "1",
            "2",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_train_args()
        self.assertEqual(args.video_every_checkpoints, 3)
        self.assertEqual(args.video_motion_ids, [1, 2])
        # example_mix.yaml declares no inference_path, so the preview falls back to the training data.
        self.assertIsNone(args.video_data_path)

    def test_cli_rejects_negative_video_interval(self) -> None:
        argv = ["train.py", "--agent", "fb", "--gpu-ids", "single", "--smoke", "--video-every-checkpoints", "-1"]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "video-every-checkpoints"):
            parse_train_args()


if __name__ == "__main__":
    unittest.main()
