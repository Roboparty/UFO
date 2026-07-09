from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoidverse.tools.data_build import build_manifest_config, write_manifest
from humanoidverse.tools.data_inspect import inspect_data_source
from humanoidverse.tools.robot_inspect import infer_robot_semantics, inspect_mujoco_xml, write_robot_yaml
from humanoidverse.utils.motion_data.manifest import prepare_motion_manifest
from humanoidverse.utils.robot_spec import load_robot_spec


def _write_import_robot(root: Path) -> Path:
    xml_path = root / "tiny_import.xml"
    xml_path.write_text(
        """
<mujoco model="tiny_import">
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="torso" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.1"/>
        <body name="left_foot" pos="0 0.1 -0.2">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2 2"/>
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
        <body name="right_foot" pos="0 -0.1 -0.2">
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
        <body name="left_hand" pos="0 0.2 0.1">
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
        <body name="right_hand" pos="0 -0.2 0.1">
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
    return xml_path


def _write_robot_state_csv(path: Path, *, named_joints: bool, frames: int) -> None:
    if named_joints:
        fieldnames = [
            "time",
            "root_pos_x",
            "root_pos_y",
            "root_pos_z",
            "root_quat_x",
            "root_quat_y",
            "root_quat_z",
            "root_quat_w",
            "joint1",
            "joint2",
        ]
    else:
        fieldnames = [
            "time",
            "root_pos_x",
            "root_pos_y",
            "root_pos_z",
            "root_quat_x",
            "root_quat_y",
            "root_quat_z",
            "root_quat_w",
            "dof_0",
            "dof_1",
        ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(frames):
            row = {
                "time": idx * 0.02,
                "root_pos_x": 0.0,
                "root_pos_y": 0.0,
                "root_pos_z": 1.0,
                "root_quat_x": 0.0,
                "root_quat_y": 0.0,
                "root_quat_z": 0.0,
                "root_quat_w": 1.0,
            }
            if named_joints:
                row.update({"joint1": 0.1 * idx, "joint2": 0.2 * idx})
            else:
                row.update({"dof_0": 0.1 * idx, "dof_1": 0.2 * idx})
            writer.writerow(row)


def _write_robot_state_npz(path: Path, *, frames: int) -> None:
    idx = np.arange(frames, dtype=np.float32)
    np.savez(
        path,
        root_pos=np.zeros((frames, 3), dtype=np.float32),
        root_quat=np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (frames, 1)),
        dof_pos=np.stack([0.1 * idx, 0.2 * idx], axis=1).astype(np.float32),
        joint_names=np.asarray(["joint1", "joint2"]),
        fps=np.asarray(50),
    )


class ImportToolsTest(unittest.TestCase):
    def test_robot_inspect_writes_valid_robot_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = _write_import_robot(root)
            out_path = root / "tiny_import.yaml"

            inspection = inspect_mujoco_xml(xml_path, name="tiny_import")
            semantics = infer_robot_semantics(inspection)
            write_robot_yaml(out_path, inspection, semantics)
            spec = load_robot_spec(out_path)

            self.assertEqual(spec.base_body, "base")
            self.assertEqual(spec.free_joint, "root")
            self.assertEqual(spec.control_joint_names, ["joint1", "joint2"])
            self.assertIn("base", spec.key_bodies)
            self.assertIn("torso", spec.key_bodies)
            self.assertIn("left_foot", spec.feet)
            self.assertIn("right_foot", spec.feet)

    def test_data_inspect_csv_xml_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = _write_import_robot(root)
            robot_yaml = root / "tiny_import.yaml"
            inspection = inspect_mujoco_xml(xml_path, name="tiny_import")
            write_robot_yaml(robot_yaml, inspection, infer_robot_semantics(inspection))
            csv_path = root / "state_xml_order.csv"
            _write_robot_state_csv(csv_path, named_joints=False, frames=1250)

            result = inspect_data_source(
                robot_config=robot_yaml,
                source=str(csv_path),
                fmt="robot_state_csv",
                fps=50,
                clip_seconds=10,
            )
            self.assertEqual(result.root_pos_columns, ["root_pos_x", "root_pos_y", "root_pos_z"])
            self.assertEqual(result.root_quat_columns, ["root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"])
            self.assertEqual(result.dof_pos_mode, "xml_order")
            self.assertEqual(result.estimated_clip_count, 3)

    def test_data_inspect_named_joint_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = _write_import_robot(root)
            robot_yaml = root / "tiny_import.yaml"
            inspection = inspect_mujoco_xml(xml_path, name="tiny_import")
            write_robot_yaml(robot_yaml, inspection, infer_robot_semantics(inspection))
            csv_path = root / "state_named.csv"
            _write_robot_state_csv(csv_path, named_joints=True, frames=60)

            result = inspect_data_source(robot_config=robot_yaml, source=str(csv_path), fmt="robot_state_csv")
            self.assertEqual(result.dof_pos_mode, "auto_by_joint_name")
            self.assertEqual(result.manifest_columns["dof_pos"], "auto_by_joint_name")

    def test_data_build_manifest_builds_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = _write_import_robot(root)
            robot_yaml = root / "tiny_import.yaml"
            inspection = inspect_mujoco_xml(xml_path, name="tiny_import")
            write_robot_yaml(robot_yaml, inspection, infer_robot_semantics(inspection))
            csv_path = root / "state.csv"
            _write_robot_state_csv(csv_path, named_joints=False, frames=60)

            manifest_config = build_manifest_config(
                robot=str(robot_yaml),
                source=str(csv_path),
                fmt="robot_state_csv",
                name="tiny_motion",
                weight=1.0,
                fps=50,
                clip_seconds=10,
            )
            manifest_path = write_manifest(manifest_config, root / "tiny_motion.yaml")
            result = prepare_motion_manifest(manifest_path, cache_root=root / "cache", rebuild_cache=True)

            self.assertEqual(len(result.train_data_paths), 1)
            train_path = Path(result.train_data_paths[0])
            full_path = Path(result.inference_paths["tiny_motion"])
            self.assertTrue(train_path.exists())
            self.assertTrue(full_path.exists())
            self.assertTrue(train_path.name.endswith("_train_near10s_ufo.pkl"))
            self.assertTrue(full_path.name.endswith("_full_ufo.pkl"))

    def test_data_build_manifest_builds_npz_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = _write_import_robot(root)
            robot_yaml = root / "tiny_import.yaml"
            inspection = inspect_mujoco_xml(xml_path, name="tiny_import")
            write_robot_yaml(robot_yaml, inspection, infer_robot_semantics(inspection))
            npz_path = root / "state.npz"
            _write_robot_state_npz(npz_path, frames=60)

            manifest_config = build_manifest_config(
                robot=str(robot_yaml),
                source=str(npz_path),
                fmt="robot_state_npz",
                name="tiny_motion_npz",
                weight=1.0,
                fps=50,
                clip_seconds=10,
            )
            manifest_path = write_manifest(manifest_config, root / "tiny_motion_npz.yaml")
            result = prepare_motion_manifest(manifest_path, cache_root=root / "cache", rebuild_cache=True)

            self.assertEqual(len(result.train_data_paths), 1)
            self.assertTrue(Path(result.train_data_paths[0]).exists())
            self.assertTrue(Path(result.inference_paths["tiny_motion_npz"]).exists())


if __name__ == "__main__":
    unittest.main()
