from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from humanoidverse.utils.motion_data.adapters import load_csv_ufo, load_ufo_npz
from humanoidverse.utils.motion_data.manifest import prepare_manifest_dataset_path, prepare_motion_manifest
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict


def _motion_dict(fps: float = 50.0) -> dict[str, dict]:
    return {
        "tiny": {
            "root_trans_offset": np.zeros((3, 3), dtype=np.float32),
            "pose_aa": np.zeros((3, 2, 3), dtype=np.float32),
            "fps": fps,
        }
    }


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


if __name__ == "__main__":
    unittest.main()
