from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from humanoidverse.utils.motion_data.adapters import load_csv_ufo, load_robot_state_csv, load_robot_state_npz, load_ufo_npz
from humanoidverse.utils.motion_data.clip import clip_ufo_motion_dict
from humanoidverse.utils.motion_data.manifest import prepare_manifest_dataset_path, prepare_motion_manifest
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.robot_spec import load_robot_spec


def _motion_dict(fps: float = 50.0) -> dict[str, dict]:
    return {
        "tiny": {
            "root_trans_offset": np.zeros((3, 3), dtype=np.float32),
            "pose_aa": np.zeros((3, 2, 3), dtype=np.float32),
            "fps": fps,
        }
    }


def _long_motion_dict(seconds: float, fps: float = 50.0) -> dict[str, dict]:
    frames = int(seconds * fps)
    return {
        "long": {
            "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
            "pose_aa": np.zeros((frames, 3, 3), dtype=np.float32),
            "dof_pos": np.zeros((frames, 2), dtype=np.float32),
            "root_quat": np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (frames, 1)),
            "fps": fps,
        }
    }


def _write_tiny_robot(root: Path) -> tuple[Path, Path]:
    xml_path = root / "tiny.xml"
    xml_path.write_text(
        """
<mujoco model="tiny">
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
    robot_config = root / "tiny_robot.yaml"
    robot_config.write_text(
        "\n".join(
            [
                "name: tiny",
                "xml_path: tiny.xml",
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
    return xml_path, robot_config


def _write_robot_state_csv(path: Path, *, named_joints: bool = True, frames: int = 3) -> None:
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


class MotionDataAdapterTest(unittest.TestCase):
    def test_validate_ufo_motion_dict(self) -> None:
        data = validate_ufo_motion_dict(_motion_dict(), "unit")
        self.assertEqual(list(data), ["tiny"])

    def test_validate_rejects_invalid_fps(self) -> None:
        with self.assertRaisesRegex(ValueError, "fps must be > 0"):
            validate_ufo_motion_dict(_motion_dict(fps=0.0), "unit")

    def test_ufo_npz_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.npz"
            np.savez(
                path,
                root_trans_offset=np.zeros((4, 3), dtype=np.float32),
                pose_aa=np.zeros((4, 2, 3), dtype=np.float32),
                fps=np.asarray(60),
            )
            data = load_ufo_npz(str(path), source_name="npz")
            self.assertIn("sample", data)
            self.assertEqual(data["sample"]["pose_aa"].shape, (4, 2, 3))

    def test_csv_ufo_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            fieldnames = [
                "time",
                "root_trans_offset_x",
                "root_trans_offset_y",
                "root_trans_offset_z",
                "pose_aa_flat_0",
                "pose_aa_flat_1",
                "pose_aa_flat_2",
            ]
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for idx in range(3):
                    writer.writerow(
                        {
                            "time": idx * 0.02,
                            "root_trans_offset_x": 0.0,
                            "root_trans_offset_y": 0.0,
                            "root_trans_offset_z": 0.0,
                            "pose_aa_flat_0": 0.0,
                            "pose_aa_flat_1": 0.0,
                            "pose_aa_flat_2": 0.0,
                        }
                    )
            data = load_csv_ufo(str(path), source_name="csv")
            self.assertAlmostEqual(float(data["sample"]["fps"]), 50.0)
            self.assertEqual(data["sample"]["pose_aa"].shape, (3, 1, 3))

    def test_csv_ufo_rejects_raw_joint_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw_joint.csv"
            fieldnames = [
                "time",
                "root_trans_offset_x",
                "root_trans_offset_y",
                "root_trans_offset_z",
                "left_hip_pitch",
            ]
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "time": 0.0,
                        "root_trans_offset_x": 0.0,
                        "root_trans_offset_y": 0.0,
                        "root_trans_offset_z": 0.0,
                        "left_hip_pitch": 0.0,
                    }
                )
            with self.assertRaisesRegex(ValueError, "Raw G1 joint CSV requires"):
                load_csv_ufo(str(path), source_name="csv")

    def test_csv_ufo_named_columns_reject_missing_joint_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing_zero.csv"
            fieldnames = [
                "time",
                "root_trans_offset_x",
                "root_trans_offset_y",
                "root_trans_offset_z",
                "pose_aa_1_x",
                "pose_aa_1_y",
                "pose_aa_1_z",
            ]
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for idx in range(2):
                    writer.writerow(
                        {
                            "time": idx * 0.02,
                            "root_trans_offset_x": 0.0,
                            "root_trans_offset_y": 0.0,
                            "root_trans_offset_z": 0.0,
                            "pose_aa_1_x": 0.0,
                            "pose_aa_1_y": 0.0,
                            "pose_aa_1_z": 0.0,
                        }
                    )
            with self.assertRaisesRegex(ValueError, "pose_aa joint indices must be contiguous from 0"):
                load_csv_ufo(str(path), source_name="csv")

    def test_robot_spec_parses_minimal_mujoco_xml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, robot_config = _write_tiny_robot(root)
            spec = load_robot_spec(robot_config)
            self.assertEqual(spec.name, "tiny")
            self.assertEqual(spec.base_body, "base")
            self.assertEqual(spec.free_joint, "root")
            self.assertEqual(spec.body_names, ["base", "link1", "link2"])
            self.assertEqual(spec.control_joint_names, ["joint1", "joint2"])
            self.assertEqual(spec.joint_axes["joint1"], [0.0, 0.0, 1.0])
            self.assertGreaterEqual(spec.joint_qpos_addr["joint1"], 7)

    def test_robot_state_csv_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, robot_config = _write_tiny_robot(root)
            spec = load_robot_spec(robot_config)
            path = root / "state.csv"
            _write_robot_state_csv(path, named_joints=True, frames=4)
            data = load_robot_state_csv(
                str(path),
                source_name="robot_csv",
                robot_spec=spec,
                columns={
                    "root_pos": ["root_pos_x", "root_pos_y", "root_pos_z"],
                    "root_quat": ["root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"],
                    "dof_pos": "auto_by_joint_name",
                },
            )
            motion = data["state"]
            self.assertEqual(motion["pose_aa"].shape, (4, 3, 3))
            self.assertAlmostEqual(float(motion["fps"]), 50.0)
            self.assertAlmostEqual(float(motion["pose_aa"][2, 1, 2]), 0.2)

    def test_robot_state_npz_adapter_reorders_joint_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, robot_config = _write_tiny_robot(root)
            spec = load_robot_spec(robot_config)
            path = root / "state.npz"
            np.savez(
                path,
                root_pos=np.zeros((4, 3), dtype=np.float32),
                root_quat=np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (4, 1)),
                dof_pos=np.asarray([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4]], dtype=np.float32),
                joint_names=np.asarray(["joint2", "joint1"]),
                fps=np.asarray(50),
            )
            data = load_robot_state_npz(str(path), source_name="robot_npz", robot_spec=spec)
            motion = data["state"]
            self.assertEqual(motion["pose_aa"].shape, (4, 3, 3))
            self.assertAlmostEqual(float(motion["dof_pos"][1, 0]), 0.2)
            self.assertAlmostEqual(float(motion["pose_aa"][1, 1, 2]), 0.2)
            self.assertAlmostEqual(float(motion["pose_aa"][1, 2, 1]), 2.0)

    def test_clip_25_seconds_keeps_tail(self) -> None:
        clipped = clip_ufo_motion_dict(
            _long_motion_dict(seconds=25.0, fps=50.0),
            clip_seconds=10.0,
            stride_seconds=10.0,
            keep_short=True,
            min_clip_seconds=1.0,
            source_name="clip",
        )
        self.assertEqual(list(clipped), ["long__clip000", "long__clip001", "long__clip002"])
        self.assertEqual(clipped["long__clip000"]["root_trans_offset"].shape[0], 500)
        self.assertEqual(clipped["long__clip002"]["root_trans_offset"].shape[0], 250)

    def test_clip_short_motion_keep_short_true(self) -> None:
        clipped = clip_ufo_motion_dict(
            _long_motion_dict(seconds=8.0, fps=50.0),
            clip_seconds=10.0,
            keep_short=True,
            min_clip_seconds=1.0,
            source_name="clip",
        )
        self.assertEqual(list(clipped), ["long__clip000"])
        self.assertEqual(clipped["long__clip000"]["root_trans_offset"].shape[0], 400)

    def test_clip_short_motion_keep_short_false_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "No motion clips were generated"):
            clip_ufo_motion_dict(
                _long_motion_dict(seconds=8.0, fps=50.0),
                clip_seconds=10.0,
                keep_short=False,
                min_clip_seconds=1.0,
                source_name="clip",
            )

    def test_manifest_paths_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkl_path = root / "a.pkl"
            npz_path = root / "b.npz"
            manifest_path = root / "mix.yaml"
            joblib.dump(_motion_dict(fps=30), pkl_path)
            np.savez(
                npz_path,
                root_trans_offset=np.zeros((2, 3), dtype=np.float32),
                pose_aa=np.zeros((2, 2, 3), dtype=np.float32),
                fps=np.asarray(60),
            )
            manifest_path.write_text(
                "\n".join(
                    [
                        "datasets:",
                        "  - name: pkl",
                        "    format: ufo_pkl",
                        "    train_path: a.pkl",
                        "    weight: 2",
                        "  - name: npz",
                        "    format: ufo_npz",
                        "    train_path: b.npz",
                        "    weight: 1",
                    ]
                )
            )
            result = prepare_motion_manifest(manifest_path, cache_root=root / "cache")
            self.assertEqual(len(result.train_data_paths), 2)
            self.assertAlmostEqual(result.train_data_weights[0], 2 / 3)
            self.assertAlmostEqual(result.train_data_weights[1], 1 / 3)
            self.assertTrue(Path(result.train_data_paths[1]).exists())

    def test_manifest_dataset_path_uses_inference_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.pkl"
            inference_path = root / "inference.pkl"
            manifest_path = root / "mix.yaml"
            joblib.dump(_motion_dict(fps=30), train_path)
            joblib.dump(_motion_dict(fps=60), inference_path)
            manifest_path.write_text(
                "\n".join(
                    [
                        "datasets:",
                        "  - name: pkl",
                        "    format: ufo_pkl",
                        "    train_path: train.pkl",
                        "    inference_path: inference.pkl",
                        "    weight: 1",
                    ]
                )
            )
            path = prepare_manifest_dataset_path(manifest_path, "pkl", split="inference", cache_root=root / "cache")
            self.assertEqual(Path(path), inference_path.resolve())

    def test_manifest_dataset_path_falls_back_to_train_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.pkl"
            manifest_path = root / "mix.yaml"
            joblib.dump(_motion_dict(fps=30), train_path)
            manifest_path.write_text(
                "\n".join(
                    [
                        "datasets:",
                        "  - name: pkl",
                        "    format: ufo_pkl",
                        "    train_path: train.pkl",
                        "    weight: 1",
                    ]
                )
            )
            path = prepare_manifest_dataset_path(manifest_path, "pkl", split="inference", cache_root=root / "cache")
            self.assertEqual(Path(path), train_path.resolve())

    def test_manifest_auto_build_robot_state_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, robot_config = _write_tiny_robot(root)
            csv_path = root / "state.csv"
            _write_robot_state_csv(csv_path, named_joints=False, frames=60)
            manifest_path = root / "robot_state.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        f"robot_config: {robot_config.name}",
                        "datasets:",
                        "  - name: robot",
                        "    format: robot_state_csv",
                        "    source_path: state.csv",
                        "    weight: 1",
                        "    fps: 50",
                        "    columns:",
                        "      root_pos: [root_pos_x, root_pos_y, root_pos_z]",
                        "      root_quat: [root_quat_x, root_quat_y, root_quat_z, root_quat_w]",
                        "      dof_pos: xml_order",
                        "    auto_build:",
                        "      train_clip_seconds: 10.0",
                        "      clip_stride_seconds: 10.0",
                        "      keep_short: true",
                        "      min_clip_seconds: 1.0",
                    ]
                )
            )
            result = prepare_motion_manifest(manifest_path, cache_root=root / "cache")
            self.assertEqual(len(result.train_data_paths), 1)
            self.assertTrue(result.train_data_paths[0].endswith("robot_train_near10s_ufo.pkl"))
            self.assertTrue(Path(result.train_data_paths[0]).exists())

            inference_path = prepare_manifest_dataset_path(manifest_path, "robot", split="inference", cache_root=root / "cache")
            self.assertTrue(inference_path.endswith("robot_full_ufo.pkl"))
            self.assertTrue(Path(inference_path).exists())


if __name__ == "__main__":
    unittest.main()
