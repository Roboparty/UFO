from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import torch

from humanoidverse.goal_inference import (
    _find_goal_json,
    _target_states_from_obs,
    load_and_validate_goal_json,
    parse_args as parse_goal_args,
)
from humanoidverse.mjlab_inference_utils import write_mjlab_relabel_xml
from humanoidverse.reward_inference import (
    DEFAULT_TASKS,
    _is_non_g1_locomotion_task,
    _resolve_reward_tasks,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_tiny_robot_config(root: Path) -> Path:
    xml_path = root / "tiny_robot.xml"
    xml_path.write_text(
        """
<mujoco model="tiny_robot">
  <worldbody>
    <body name="base" pos="0 0 0.8">
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
    <motor name="old_joint1_motor" joint="joint1"/>
  </actuator>
</mujoco>
""".strip()
    )
    robot_config = root / "tiny_robot.yaml"
    robot_config.write_text(
        "\n".join(
            [
                "name: tiny_robot",
                "xml_path: tiny_robot.xml",
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
            ]
        )
    )
    return robot_config


def _fake_robot_training(name: str, num_dof: int) -> SimpleNamespace:
    return SimpleNamespace(
        robot=SimpleNamespace(
            name=name,
            control_joint_names=[f"joint_{idx}" for idx in range(num_dof)],
        )
    )


class RobotConfigGoalRewardInferenceTest(unittest.TestCase):
    def test_goal_target_states_follow_num_dof(self) -> None:
        obs = {
            "ref_body_pos": torch.zeros(4, 1, 3),
            "ref_body_rots": torch.zeros(4, 1, 4),
            "ref_body_vels": torch.zeros(4, 1, 3),
            "ref_body_angular_vels": torch.zeros(4, 1, 3),
            "dof_pos": torch.arange(27, dtype=torch.float32).repeat(4, 1),
            "ref_dof_vel": torch.ones(4, 27),
        }
        obs["ref_body_rots"][..., 3] = 1.0
        target = _target_states_from_obs(obs, device="cpu", num_dof=27)
        self.assertEqual(tuple(target["dof_states"].shape), (1, 27, 2))
        self.assertTrue(torch.equal(target["dof_states"][0, :, 0], torch.arange(27, dtype=torch.float32)))

    def test_non_g1_goal_json_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "Non-G1 goal inference requires --goal-json"):
            _find_goal_json(None, num_dof=27, robot_name="tiny_robot")

    def test_goal_json_dimension_mismatch_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = [
                ("dof_pos", [0.0] * 26, "expected dimension 27"),
                ("qpos", [0.0] * 33, "expected dimension 34"),
            ]
            for key, value, pattern in cases:
                goal_json = root / f"{key}.json"
                goal_json.write_text(json.dumps([{"motion_id": 0, "frames": [0], key: value}]))
                with self.subTest(key=key):
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_and_validate_goal_json(goal_json, num_dof=27)

    def test_goal_cli_manifest_robot_config_mismatch_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_config(Path(tmpdir))
            argv = [
                "goal_inference.py",
                "--model-folder",
                "/tmp/ufo_unit_model",
                "--data-manifest",
                "configs/data/example_robot_state_auto_build.yaml",
                "--dataset",
                "g1_robot_state_sample",
                "--robot-config",
                str(tiny_robot),
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "does not match data manifest robot_config"):
                    parse_goal_args()

    def test_g1_default_goal_json_fallback_is_preserved(self) -> None:
        goal_json = _find_goal_json(None, num_dof=29, robot_name="g1_29dof")
        self.assertTrue(goal_json.name.endswith("goal_frames_lafan29dof.json"))
        goals = load_and_validate_goal_json(goal_json, num_dof=29)
        self.assertTrue(goals)

    def test_generic_relabel_xml_has_one_motor_per_control_joint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            robot_config = _write_tiny_robot_config(root)
            robot_xml = root / "tiny_robot.xml"
            output = write_mjlab_relabel_xml(
                robot_xml,
                root / "out",
                ["joint1", "joint2"],
                "tiny_robot",
                root_body_name="base",
            )
            model = mujoco.MjModel.from_xml_path(str(output))
            data = mujoco.MjData(model)
            self.assertEqual(model.nu, 2)
            self.assertEqual(data.ctrl.size, 2)
            text = output.read_text()
            self.assertIn('body="pelvis"', text)
            self.assertEqual(text.count("<motor "), 2)
            self.assertTrue(robot_config.exists())

    def test_non_g1_default_tasks_are_locomotion_only(self) -> None:
        tasks, mode = _resolve_reward_tasks(None, _fake_robot_training("tiny_robot", 27))
        self.assertEqual(mode, "non-G1 locomotion-only tasks")
        self.assertTrue(tasks)
        self.assertTrue(all(_is_non_g1_locomotion_task(task) for task in tasks))

    def test_non_g1_rejects_g1_semantics_tasks(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Task raisearms-l-l requires G1-style arm/body semantics and is not enabled for robot tiny_robot",
        ):
            _resolve_reward_tasks(["move-ego-0-0.3", "raisearms-l-l"], _fake_robot_training("tiny_robot", 27))
        with self.assertRaisesRegex(ValueError, "requires G1-style arm/body semantics"):
            _resolve_reward_tasks(["move-ego-low-0.7-0.3-l-l"], _fake_robot_training("tiny_robot", 27))

    def test_g1_default_tasks_are_preserved(self) -> None:
        self.assertIn("raisearms-l-l", DEFAULT_TASKS)
        self.assertIn("sitonground", DEFAULT_TASKS)
        tasks, mode = _resolve_reward_tasks(None, _fake_robot_training("g1_29dof", 29))
        self.assertEqual(mode, "G1 full tasks")
        self.assertEqual(tasks, DEFAULT_TASKS)

    def test_goal_and_reward_export_calls_use_current_signature(self) -> None:
        for relative_path in ("humanoidverse/goal_inference.py", "humanoidverse/reward_inference.py"):
            source = REPO_ROOT / relative_path
            tree = ast.parse(source.read_text())
            export_model = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "_export_model"
            )
            export_call = next(
                node
                for node in ast.walk(export_model)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "export_meta_policy_as_onnx"
            )
            self.assertEqual(len(export_call.args), 3, relative_path)
            self.assertEqual([keyword.arg for keyword in export_call.keywords], ["z_dim"], relative_path)

    def test_goal_backward_observation_uses_goal_motion_ids(self) -> None:
        source = REPO_ROOT / "humanoidverse/goal_inference.py"
        tree = ast.parse(source.read_text())
        run_goal = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run_goal_inference"
        )
        calls = [
            node
            for node in ast.walk(run_goal)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_backward_observation"
        ]
        motion_arg_names = [
            call.args[1].id
            for call in calls
            if len(call.args) >= 2 and isinstance(call.args[1], ast.Name)
        ]
        self.assertIn("motion_id", motion_arg_names)
        self.assertIn("first_motion_id", motion_arg_names)
        self.assertFalse(
            any(len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) and call.args[1].value == 0 for call in calls)
        )


if __name__ == "__main__":
    unittest.main()
