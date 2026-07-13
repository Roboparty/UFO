from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch

from humanoidverse.tools.data_inspect import inspect_data_source
from humanoidverse.utils.motion_data.adapters import (
    SUPPORTED_FORMATS,
    load_motion_data_by_format,
    load_robot_state_csv,
    load_robot_state_npz,
)
from humanoidverse.utils.motion_data.clip import clip_ufo_motion_dict
from humanoidverse.utils.motion_data.manifest import prepare_manifest_dataset_path, prepare_motion_manifest
from humanoidverse.utils.motion_data.robot_state import RobotStateMotion, reorder_dof_by_joint_names, validate_robot_state_motion
from humanoidverse.utils.motion_data.robot_state_convert import robot_state_dict_to_ufo_motion_dict, robot_state_to_ufo_motion
from humanoidverse.utils.motion_data.robot_state_readers import read_robot_state_csv, read_robot_state_npz
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.motion_lib.motion_lib_base import _dof_vel_from_dof_pos, _raw_dof_pos_from_motion_file
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


def _robot_state_motion(
    *,
    frames: int = 4,
    dof_pos: np.ndarray | None = None,
    joint_names: list[str] | None = None,
    zero_quat: bool = False,
) -> RobotStateMotion:
    root_quat = np.zeros((frames, 4), dtype=np.float32) if zero_quat else np.tile(
        np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (frames, 1)
    )
    if dof_pos is None:
        dof_pos = np.zeros((frames, 2), dtype=np.float32)
    return RobotStateMotion(
        motion_key="state",
        root_pos=np.zeros((frames, 3), dtype=np.float32),
        root_quat=root_quat,
        dof_pos=dof_pos,
        fps=50.0,
        joint_names=joint_names,
        source="unit",
        metadata={"test": True},
    )


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


def _write_headerless_robot_state_csv(path: Path, *, frames: int = 3, include_time: bool = False) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for idx in range(frames):
            row: list[float] = []
            if include_time:
                row.append(idx * 0.02)
            row.extend([0.0, 0.0, 1.0])
            row.extend([0.0, 0.0, 0.0, 1.0])
            row.extend([0.1 * idx, 0.2 * idx])
            writer.writerow(row)


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


def _write_robot_state_npz(path: Path, *, frames: int = 4, swapped_joint_names: bool = True) -> None:
    idx = np.arange(frames, dtype=np.float32)
    if swapped_joint_names:
        dof_pos = np.stack([idx + 1.0, 0.1 * (idx + 1.0)], axis=1).astype(np.float32)
        joint_names = np.asarray(["joint2", "joint1"])
    else:
        dof_pos = np.stack([0.1 * (idx + 1.0), idx + 1.0], axis=1).astype(np.float32)
        joint_names = np.asarray(["joint1", "joint2"])
    np.savez(
        path,
        root_pos=np.zeros((frames, 3), dtype=np.float32),
        root_quat=np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (frames, 1)),
        dof_pos=dof_pos,
        joint_names=joint_names,
        fps=np.asarray(50),
    )


class MotionDataAdapterTest(unittest.TestCase):
    def test_validate_ufo_motion_dict(self) -> None:
        data = validate_ufo_motion_dict(_motion_dict(), "unit")
        self.assertEqual(list(data), ["tiny"])

    def test_validate_rejects_invalid_fps(self) -> None:
        with self.assertRaisesRegex(ValueError, "fps must be > 0"):
            validate_ufo_motion_dict(_motion_dict(fps=0.0), "unit")

    def test_public_supported_formats_are_minimal(self) -> None:
        self.assertEqual(SUPPORTED_FORMATS, {"ufo_pkl", "robot_state_csv", "robot_state_npz"})

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

    def test_robot_state_motion_validate_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, robot_config = _write_tiny_robot(Path(tmpdir))
            spec = load_robot_spec(robot_config)
            motion = validate_robot_state_motion(_robot_state_motion(), spec, "unit")
            self.assertEqual(motion.root_pos.shape, (4, 3))
            self.assertEqual(motion.dof_pos.shape, (4, 2))
            self.assertAlmostEqual(motion.fps, 50.0)

    def test_robot_state_motion_rejects_mismatched_time_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, robot_config = _write_tiny_robot(Path(tmpdir))
            spec = load_robot_spec(robot_config)
            motion = RobotStateMotion(
                motion_key="bad",
                root_pos=np.zeros((3, 3), dtype=np.float32),
                root_quat=np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (4, 1)),
                dof_pos=np.zeros((4, 2), dtype=np.float32),
                fps=50.0,
            )
            with self.assertRaisesRegex(ValueError, "must share T"):
                validate_robot_state_motion(motion, spec, "unit")

    def test_robot_state_motion_rejects_zero_root_quat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, robot_config = _write_tiny_robot(Path(tmpdir))
            spec = load_robot_spec(robot_config)
            with self.assertRaisesRegex(ValueError, "root_quat contains zero"):
                validate_robot_state_motion(_robot_state_motion(zero_quat=True), spec, "unit")

    def test_robot_state_joint_names_reorder_correctly(self) -> None:
        dof = np.asarray([[10.0, 1.0], [20.0, 2.0]], dtype=np.float32)
        reordered = reorder_dof_by_joint_names(dof, ["joint2", "joint1"], ["joint1", "joint2"], "unit")
        np.testing.assert_allclose(reordered, np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32))

    def test_robot_state_joint_names_rejects_unexpected_entries(self) -> None:
        dof = np.asarray([[10.0, 1.0, 99.0], [20.0, 2.0, 88.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "unexpected joint_names entries"):
            reorder_dof_by_joint_names(dof, ["joint2", "joint1", "joint3"], ["joint1", "joint2"], "unit")

    def test_robot_state_validation_reorders_joint_named_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, robot_config = _write_tiny_robot(Path(tmpdir))
            spec = load_robot_spec(robot_config)
            dof = np.asarray([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4]], dtype=np.float32)
            motion = validate_robot_state_motion(_robot_state_motion(dof_pos=dof, joint_names=["joint2", "joint1"]), spec, "unit")
            self.assertEqual(motion.joint_names, ["joint1", "joint2"])
            self.assertAlmostEqual(float(motion.dof_pos[1, 0]), 0.2)
            self.assertAlmostEqual(float(motion.dof_pos[1, 1]), 2.0)

    def test_robot_state_csv_reader_returns_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, robot_config = _write_tiny_robot(root)
            spec = load_robot_spec(robot_config)
            path = root / "state.csv"
            _write_robot_state_csv(path, named_joints=False, frames=4)
            data = read_robot_state_csv(
                str(path),
                source_name="robot_csv",
                robot_spec=spec,
                columns={
                    "root_pos": ["root_pos_x", "root_pos_y", "root_pos_z"],
                    "root_quat": ["root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"],
                    "dof_pos": "xml_order",
                },
            )
            self.assertIsInstance(data["state"], RobotStateMotion)
            self.assertEqual(data["state"].dof_pos.shape, (4, 2))
            self.assertAlmostEqual(data["state"].fps, 50.0)

    def test_headerless_robot_state_csv_reader_defaults_to_xml_order_without_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, robot_config = _write_tiny_robot(root)
            spec = load_robot_spec(robot_config)
            path = root / "state.csv"
            _write_headerless_robot_state_csv(path, frames=4, include_time=False)
            data = read_robot_state_csv(str(path), source_name="robot_csv", robot_spec=spec, fps=50)
            motion = data["state"]
            self.assertEqual(motion.dof_pos.shape, (4, 2))
            self.assertAlmostEqual(motion.fps, 50.0)
            self.assertAlmostEqual(float(motion.dof_pos[2, 0]), 0.2)
            self.assertAlmostEqual(float(motion.dof_pos[2, 1]), 0.4)
            self.assertEqual(motion.metadata["columns"]["dof_pos"], "xml_order")

    def test_headerless_robot_state_csv_reader_accepts_leading_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, robot_config = _write_tiny_robot(root)
            spec = load_robot_spec(robot_config)
            path = root / "state.csv"
            _write_headerless_robot_state_csv(path, frames=4, include_time=True)
            data = read_robot_state_csv(str(path), source_name="robot_csv", robot_spec=spec)
            motion = data["state"]
            self.assertEqual(motion.dof_pos.shape, (4, 2))
            self.assertAlmostEqual(motion.fps, 50.0)
            self.assertAlmostEqual(float(motion.root_pos[0, 2]), 1.0)

    def test_data_inspect_accepts_headerless_robot_state_csv_without_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, robot_config = _write_tiny_robot(root)
            path = root / "state.csv"
            _write_headerless_robot_state_csv(path, frames=60, include_time=False)
            result = inspect_data_source(
                robot_config=robot_config,
                source=str(path),
                fmt="robot_state_csv",
                fps=50,
                dataset_name="robot",
            )
            self.assertEqual(result.dof_pos_mode, "xml_order")
            self.assertEqual(result.root_pos_columns, ["root_pos_x", "root_pos_y", "root_pos_z"])
            dataset = result.suggested_manifest["datasets"][0]
            self.assertEqual(dataset["columns"]["dof_pos"], "xml_order")
            self.assertEqual(dataset["fps"], 50.0)

    def test_robot_state_npz_reader_returns_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, robot_config = _write_tiny_robot(root)
            spec = load_robot_spec(robot_config)
            path = root / "state.npz"
            _write_robot_state_npz(path, swapped_joint_names=True)
            data = read_robot_state_npz(str(path), source_name="robot_npz", robot_spec=spec)
            self.assertIsInstance(data["state"], RobotStateMotion)
            self.assertEqual(data["state"].joint_names, ["joint1", "joint2"])
            self.assertAlmostEqual(float(data["state"].dof_pos[1, 0]), 0.2)

    def test_robot_state_converter_outputs_ufo_motion_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, robot_config = _write_tiny_robot(Path(tmpdir))
            spec = load_robot_spec(robot_config)
            dof = np.asarray([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0]], dtype=np.float32)
            motion = _robot_state_motion(dof_pos=dof)
            converted = robot_state_to_ufo_motion(motion, spec, "unit")
            self.assertEqual(converted["root_trans_offset"].shape, (4, 3))
            self.assertEqual(converted["pose_aa"].shape, (4, 3, 3))
            self.assertAlmostEqual(float(converted["pose_aa"][1, 1, 2]), 0.2)
            self.assertAlmostEqual(float(converted["pose_aa"][1, 2, 1]), 2.0)
            as_dict = robot_state_dict_to_ufo_motion_dict({"state": motion}, spec, "unit")
            self.assertIn("state", as_dict)

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
            _write_robot_state_npz(path, swapped_joint_names=True)
            data = load_robot_state_npz(str(path), source_name="robot_npz", robot_spec=spec)
            motion = data["state"]
            self.assertEqual(motion["pose_aa"].shape, (4, 3, 3))
            self.assertAlmostEqual(float(motion["dof_pos"][1, 0]), 0.2)
            self.assertAlmostEqual(float(motion["pose_aa"][1, 1, 2]), 0.2)
            self.assertAlmostEqual(float(motion["pose_aa"][1, 2, 1]), 2.0)

    def test_ufo_pkl_old_path_continues_to_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.pkl"
            joblib.dump(_motion_dict(fps=50), path)
            data = load_motion_data_by_format("ufo_pkl", str(path), source_name="legacy")
            self.assertEqual(list(data), ["tiny"])

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
            second_pkl_path = root / "b.pkl"
            manifest_path = root / "mix.yaml"
            joblib.dump(_motion_dict(fps=30), pkl_path)
            joblib.dump(_motion_dict(fps=60), second_pkl_path)
            manifest_path.write_text(
                "\n".join(
                    [
                        "datasets:",
                        "  - name: pkl_a",
                        "    format: ufo_pkl",
                        "    train_path: a.pkl",
                        "    weight: 2",
                        "  - name: pkl_b",
                        "    format: ufo_pkl",
                        "    train_path: b.pkl",
                        "    weight: 1",
                    ]
                )
            )
            result = prepare_motion_manifest(manifest_path, cache_root=root / "cache")
            self.assertEqual(len(result.train_data_paths), 2)
            self.assertAlmostEqual(result.train_data_weights[0], 2 / 3)
            self.assertAlmostEqual(result.train_data_weights[1], 1 / 3)
            self.assertTrue(Path(result.train_data_paths[0]).exists())
            self.assertTrue(Path(result.train_data_paths[1]).exists())

    def test_manifest_rejects_duplicate_dataset_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "duplicate.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        "datasets:",
                        "  - name: pkl",
                        "    format: ufo_pkl",
                        "    train_path: a.pkl",
                        "    weight: 1",
                        "  - name: pkl",
                        "    format: ufo_pkl",
                        "    train_path: b.pkl",
                        "    weight: 1",
                    ]
                )
            )
            with self.assertRaisesRegex(ValueError, "dataset name must be unique"):
                prepare_motion_manifest(manifest_path, cache_root=root / "cache")

    def test_manifest_rejects_invalid_weights(self) -> None:
        cases = [
            ("infinite.yaml", "    weight: .inf", "weight must be finite"),
            ("negative.yaml", "    weight: -1", "weight must be non-negative"),
            ("zero.yaml", "    weight: 0", "sum to a positive value"),
        ]
        for filename, weight_line, error_pattern in cases:
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    manifest_path = root / filename
                    manifest_path.write_text(
                        "\n".join(
                            [
                                "datasets:",
                                "  - name: pkl",
                                "    format: ufo_pkl",
                                "    train_path: a.pkl",
                                weight_line,
                            ]
                        )
                    )
                    with self.assertRaisesRegex(ValueError, error_pattern):
                        prepare_motion_manifest(manifest_path, cache_root=root / "cache")

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


    def test_motion_lib_prefers_raw_dof_pos_when_present(self) -> None:
        raw = np.asarray(
            [
                [0.0, 0.0],
                [0.1, -0.2],
                [0.4, -0.8],
                [0.9, -1.8],
            ],
            dtype=np.float32,
        )
        dof_pos = _raw_dof_pos_from_motion_file({"dof_pos": raw}, 1, 4, torch.float32)
        self.assertIsNotNone(dof_pos)
        np.testing.assert_allclose(dof_pos.numpy(), raw[1:4])

        dof_vel = _dof_vel_from_dof_pos(dof_pos, 0.02)
        expected = np.asarray([[15.0, -30.0], [25.0, -50.0], [25.0, -50.0]], dtype=np.float32)
        np.testing.assert_allclose(dof_vel.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_manifest_auto_build_robot_state_csv_source(self) -> None:
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

    def test_manifest_auto_build_robot_state_npz_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, robot_config = _write_tiny_robot(root)
            npz_path = root / "state.npz"
            _write_robot_state_npz(npz_path, frames=60, swapped_joint_names=True)
            manifest_path = root / "robot_state_npz.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        f"robot_config: {robot_config.name}",
                        "datasets:",
                        "  - name: robot_npz",
                        "    format: robot_state_npz",
                        "    source_path: state.npz",
                        "    weight: 1",
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
            full_path = prepare_manifest_dataset_path(manifest_path, "robot_npz", split="inference", cache_root=root / "cache")
            full_data = joblib.load(full_path)
            self.assertAlmostEqual(float(full_data["state"]["dof_pos"][1, 0]), 0.2)


if __name__ == "__main__":
    unittest.main()
