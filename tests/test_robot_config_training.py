from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabCore
from humanoidverse.train import _resolve_training_robot_config, build_ufo_mjlab_config, parse_args as parse_train_args
from humanoidverse.tracking_inference import (
    _expert_qpos_from_obs,
    _resolve_tracking_robot_config,
    _target_states_from_obs,
    parse_args as parse_tracking_args,
)
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _write_tiny_robot_with_training(root: Path, *, missing_actuator_joint: bool = False) -> Path:
    xml_path = root / "tiny_train.xml"
    xml_path.write_text(
        """
<mujoco model="tiny_train">
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2 2"/>
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="joint1_motor" joint="joint1"/>
    <motor name="joint2_motor" joint="joint2"/>
  </actuator>
</mujoco>
""".strip()
    )
    joint2_block = (
        []
        if missing_actuator_joint
        else [
            "        joint2:",
            "          effort_limit: 2.0",
            "          velocity_limit: 20.0",
            "          armature: 0.02",
            "          friction: 0.002",
        ]
    )
    robot_config = root / "tiny_train.yaml"
    robot_config.write_text(
        "\n".join(
            [
                "name: tiny_train",
                "xml_path: tiny_train.xml",
                "base_body: base",
                "root_quat_order: xyzw",
                "coordinate_system: z_up",
                "dof_unit: rad",
                "control_joints:",
                "  mode: all_actuated",
                "feet: [link2]",
                "hands: []",
                "key_bodies: [base, link1, link2]",
                "default_dof_pos: {}",
                "training:",
                "  hydra_robot: g1/g1_29dof_hard_waist",
                "  hydra_overrides: []",
                "  semantics:",
                "    contact_bodies: [link2]",
                "    undesired_contact_bodies: [base]",
                "    torso_name: base",
                "    left_ankle_dof_names: []",
                "    right_ankle_dof_names: []",
                "  init_state:",
                "    pos: [0.0, 0.0, 1.0]",
                "    rot: [0.0, 0.0, 0.0, 1.0]",
                "    lin_vel: [0.0, 0.0, 0.0]",
                "    ang_vel: [0.0, 0.0, 0.0]",
                "    default_joint_angles:",
                "      joint1: 0.0",
                "      joint2: 0.0",
                "  control:",
                "    action_scale: 0.25",
                "    action_clip_value: 5.0",
                "    normalize_action_to: 5.0",
                "    effort_limit: [1.0, 2.0]",
                "    velocity_limit: [10.0, 20.0]",
                "    stiffness: {joint1: 1.0, joint2: 2.0}",
                "    damping: {joint1: 0.1, joint2: 0.2}",
                "  actuator:",
                "    source: yaml",
                "    joints:",
                "      joint1:",
                "        effort_limit: 1.0",
                "        velocity_limit: 10.0",
                "        armature: 0.01",
                "        friction: 0.001",
                *joint2_block,
            ]
        )
    )
    return robot_config


class RobotConfigTrainingTest(unittest.TestCase):
    def test_old_g1_default_builds_cfg(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
        )
        self.assertTrue(str(cfg.env.robot_config_path).endswith("configs/robots/g1_29dof.yaml"))
        self.assertTrue(str(cfg.env.mjcf_path).endswith("humanoidverse/data/robots/g1_mjlab/g1_29dof.xml"))

    def test_explicit_g1_robot_config_builds_cfg(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            robot_config="configs/robots/g1_29dof.yaml",
        )
        self.assertTrue(str(cfg.env.robot_config_path).endswith("configs/robots/g1_29dof.yaml"))

    def test_manifest_robot_config_is_used_when_cli_missing(self) -> None:
        argv = [
            "train.py",
            "--agent",
            "fb",
            "--data-manifest",
            "configs/data/example_mix.yaml",
            "--gpu-ids",
            "single",
            "--smoke",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_train_args()
        self.assertTrue(str(args.robot_config).endswith("configs/robots/g1_29dof.yaml"))

    def test_cli_manifest_robot_config_mismatch_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_with_training(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "does not match data manifest robot_config"):
                _resolve_training_robot_config(tiny_robot, "configs/robots/g1_29dof.yaml")

    def test_tracking_manifest_robot_config_is_used_when_cli_missing(self) -> None:
        argv = [
            "tracking_inference.py",
            "--model-folder",
            "/tmp/ufo_unit_model",
            "--data-manifest",
            "configs/data/example_robot_state_auto_build.yaml",
            "--dataset",
            "g1_robot_state_sample",
            "--export-onnx",
            "false",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_tracking_args()
        self.assertTrue(str(args.robot_config).endswith("configs/robots/g1_29dof.yaml"))

    def test_tracking_cli_manifest_robot_config_mismatch_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_with_training(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "does not match data manifest robot_config"):
                _resolve_tracking_robot_config(tiny_robot, "configs/robots/g1_29dof.yaml")

    def test_aux_foot_rewards_require_contact_bodies(self) -> None:
        core = object.__new__(HumanoidVerseMjlabCore)
        core.reward_scales = {"penalty_feet_ori": -1.0}
        cfg = OmegaConf.create(
            {
                "robot": {
                    "contact_bodies": ["left_foot"],
                    "left_ankle_dof_names": ["left_ankle_pitch_joint", "left_ankle_roll_joint"],
                    "right_ankle_dof_names": ["right_ankle_pitch_joint", "right_ankle_roll_joint"],
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "robot.contact_bodies.*penalty_feet_ori"):
            core._validate_aux_reward_semantics(cfg)

    def test_aux_ankle_reward_requires_both_ankle_fields(self) -> None:
        core = object.__new__(HumanoidVerseMjlabCore)
        core.reward_scales = {"penalty_ankle_roll": -1.0}
        cfg = OmegaConf.create(
            {
                "robot": {
                    "contact_bodies": ["left_foot", "right_foot"],
                    "left_ankle_dof_names": ["left_ankle_pitch_joint"],
                    "right_ankle_dof_names": [],
                }
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "robot.left_ankle_dof_names, robot.right_ankle_dof_names.*penalty_ankle_roll",
        ):
            core._validate_aux_reward_semantics(cfg)

    def test_yaml_actuator_missing_joint_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_with_training(Path(tmpdir), missing_actuator_joint=True)
            with self.assertRaisesRegex(ValueError, "missing parameters for joint 'joint2'"):
                load_robot_training_spec(tiny_robot)

    def test_tracking_shapes_follow_num_dof(self) -> None:
        obs = {
            "ref_body_pos": torch.zeros(4, 1, 3),
            "ref_body_rots": torch.zeros(4, 1, 4),
            "ref_body_vels": torch.zeros(4, 1, 3),
            "ref_body_angular_vels": torch.zeros(4, 1, 3),
            "dof_pos": torch.zeros(4, 2),
            "ref_dof_vel": torch.ones(4, 2),
        }
        obs["ref_body_rots"][..., 3] = 1.0
        qpos = _expert_qpos_from_obs(obs, num_dof=2)
        self.assertEqual(qpos.shape, (4, 9))
        target = _target_states_from_obs(obs, device="cpu", num_dof=2)
        self.assertEqual(tuple(target["dof_states"].shape), (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
