from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import mujoco
from omegaconf import OmegaConf

from humanoidverse.tools.robot_inspect import infer_hydra_robot_group, infer_robot_semantics, inspect_mujoco_xml, write_robot_yaml
from humanoidverse.utils.robot_spec import load_robot_spec, load_robot_training_spec
from humanoidverse.utils.robot_spec.xml_training_infer import infer_control_joints_from_xml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_xml(root: Path, *, actuators: bool = True, freejoint: bool = True) -> Path:
    freejoint_xml = '<freejoint name="root"/>' if freejoint else ""
    qpos = "0 0 0.9 1 0 0 0 0.1 -0.2 0.05 0.1 -0.2 0.05" if freejoint else "0.1 -0.2 0.05 0.1 -0.2 0.05"
    actuator_xml = (
        """
  <actuator>
    <motor name="left_hip_motor" joint="left_hip_joint" forcerange="-120 120"/>
    <motor name="left_knee_motor" joint="left_knee_joint" ctrlrange="-90 90"/>
    <motor name="left_ankle_motor" joint="left_ankle_joint"/>
    <motor name="right_hip_motor" joint="right_hip_joint" forcerange="-120 120"/>
    <motor name="right_knee_motor" joint="right_knee_joint" ctrlrange="-90 90"/>
    <motor name="right_ankle_motor" joint="right_ankle_joint"/>
  </actuator>
"""
        if actuators
        else ""
    )
    xml = root / "tiny_hydra.xml"
    xml.write_text(
        f"""
<mujoco model="tiny_hydra">
  <compiler autolimits="true"/>
  <worldbody>
    <body name="pelvis" pos="0 0 0.8">
      {freejoint_xml}
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="torso_link" pos="0 0 0.2">
        <geom type="sphere" size="0.05" mass="0.3"/>
      </body>
      <body name="left_hip_link" pos="0 0.1 -0.1">
        <joint name="left_hip_joint" type="hinge" axis="0 1 0" range="-1 1" armature="0.02" frictionloss="0.001"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
        <body name="left_knee_link" pos="0 0 -0.2">
          <joint name="left_knee_joint" type="hinge" axis="0 1 0" range="-2 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
          <body name="left_ankle_roll_link" pos="0 0 -0.2">
            <joint name="left_ankle_joint" type="hinge" axis="1 0 0"/>
            <geom type="sphere" size="0.03" mass="0.1"/>
          </body>
        </body>
      </body>
      <body name="right_hip_link" pos="0 -0.1 -0.1">
        <joint name="right_hip_joint" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
        <body name="right_knee_link" pos="0 0 -0.2">
          <joint name="right_knee_joint" type="hinge" axis="0 1 0" range="-2 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
          <body name="right_ankle_roll_link" pos="0 0 -0.2">
            <joint name="right_ankle_joint" type="hinge" axis="1 0 0"/>
            <geom type="sphere" size="0.03" mass="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key name="stand" qpos="{qpos}"/>
  </keyframe>
  {actuator_xml}
</mujoco>
""".strip()
    )
    return xml


def _run_robot_inspect(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "python", "-m", "humanoidverse.tools.robot_inspect", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class RobotXmlHydraGenerationTest(unittest.TestCase):
    def test_legacy_robot_inspect_behavior_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml = _write_xml(root)
            out = root / "legacy.yaml"
            inspection = inspect_mujoco_xml(xml, name="tiny_hydra")
            write_robot_yaml(out, inspection, infer_robot_semantics(inspection))
            config = OmegaConf.to_container(OmegaConf.load(out), resolve=True)
            self.assertNotIn("training", config)
            spec = load_robot_spec(out)
            self.assertEqual(spec.name, "tiny_hydra")

    def test_with_training_and_hydra_out_generates_loadable_drafts(self) -> None:
        hydra_dir = REPO_ROOT / "humanoidverse" / "config" / "robot" / "__tmp_xml_hydra_test__"
        if hydra_dir.exists():
            shutil.rmtree(hydra_dir)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                xml = _write_xml(root)
                robot_yaml = root / "tiny_hydra.yaml"
                hydra_out = hydra_dir / "tiny_hydra_auto.yaml"
                result = _run_robot_inspect(
                    [
                        "--xml",
                        str(xml),
                        "--name",
                        "tiny_hydra",
                        "--out",
                        str(robot_yaml),
                        "--hydra-out",
                        str(hydra_out),
                        "--force",
                    ]
                )
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertTrue(robot_yaml.exists())
                self.assertTrue(hydra_out.exists())
                self.assertTrue(hydra_out.read_text().startswith("# AUTO-GENERATED DRAFT. REVIEW BEFORE TRAINING.\n# @package _global_"))
                robot_config = OmegaConf.to_container(OmegaConf.load(robot_yaml), resolve=True)
                hydra_config = OmegaConf.to_container(OmegaConf.load(hydra_out), resolve=True)
                self.assertEqual(robot_config["training"]["hydra_robot"], "__tmp_xml_hydra_test__/tiny_hydra_auto")
                self.assertEqual(hydra_config["robot"]["actions_dim"], len(robot_config["control_joints"]["names"]))
                spec = load_robot_training_spec(robot_yaml)
                model = mujoco.MjModel.from_xml_path(str(xml))
                actuator_joints, _warnings = infer_control_joints_from_xml(model)
                self.assertEqual(len(spec.robot.control_joint_names), len(actuator_joints))
                self.assertEqual(len(spec.effort_limits), len(spec.robot.control_joint_names))
                self.assertEqual(len(spec.velocity_limits), len(spec.robot.control_joint_names))
                self.assertEqual(set(spec.default_joint_angles), set(spec.robot.control_joint_names))
                self.assertEqual(spec.actuator["source"], "yaml")
                for joint in spec.robot.control_joint_names:
                    params = spec.actuator["joints"][joint]
                    for key in ("effort_limit", "velocity_limit", "armature", "friction"):
                        self.assertIn(key, params)
                self.assertEqual(robot_config["metadata"]["review_status"], "draft")
                self.assertTrue(robot_config["metadata"]["warnings"])
        finally:
            if hydra_dir.exists():
                shutil.rmtree(hydra_dir)

    def test_hydra_out_without_force_errors_when_file_exists(self) -> None:
        hydra_dir = REPO_ROOT / "humanoidverse" / "config" / "robot" / "__tmp_xml_hydra_exists__"
        if hydra_dir.exists():
            shutil.rmtree(hydra_dir)
        try:
            hydra_dir.mkdir(parents=True)
            hydra_out = hydra_dir / "tiny_hydra_auto.yaml"
            hydra_out.write_text("already: here\n")
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                xml = _write_xml(root)
                result = _run_robot_inspect(
                    [
                        "--xml",
                        str(xml),
                        "--name",
                        "tiny_hydra",
                        "--out",
                        str(root / "tiny_hydra.yaml"),
                        "--hydra-out",
                        str(hydra_out),
                    ]
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("already exists", result.stderr + result.stdout)
        finally:
            if hydra_dir.exists():
                shutil.rmtree(hydra_dir)

    def test_missing_actuator_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml = _write_xml(root, actuators=False)
            result = _run_robot_inspect(
                [
                    "--xml",
                    str(xml),
                    "--name",
                    "tiny_hydra",
                    "--out",
                    str(root / "tiny_hydra.yaml"),
                    "--with-training",
                    "--hydra-robot",
                    "tiny/tiny_auto",
                ]
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no joint-transmission actuators", result.stderr + result.stdout)

    def test_missing_freejoint_records_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml = _write_xml(root, freejoint=False)
            robot_yaml = root / "tiny_no_freejoint.yaml"
            result = _run_robot_inspect(
                [
                    "--xml",
                    str(xml),
                    "--name",
                    "tiny_no_freejoint",
                    "--out",
                    str(robot_yaml),
                    "--with-training",
                    "--hydra-robot",
                    "tiny/tiny_auto",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            robot_config = OmegaConf.to_container(OmegaConf.load(robot_yaml), resolve=True)
            self.assertTrue(any("No freejoint" in item for item in robot_config["metadata"]["warnings"]))

    def test_with_training_requires_hydra_robot_or_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml = _write_xml(root)
            result = _run_robot_inspect(
                ["--xml", str(xml), "--name", "tiny_hydra", "--out", str(root / "tiny.yaml"), "--with-training"]
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--with-training requires either --hydra-out or --hydra-robot", result.stderr + result.stdout)

    def test_hydra_out_group_path_inference_rejects_other_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "must be under humanoidverse/config/robot"):
                infer_hydra_robot_group(Path(tmpdir) / "robot.yaml")


if __name__ == "__main__":
    unittest.main()
