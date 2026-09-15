from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from humanoidverse.utils.motion_data import prepare_motion_manifest
from humanoidverse.utils.motion_data.ground_align import (
    GROUND_PENETRATION_FIELD,
    GroundAlignmentConfig,
    apply_ground_alignment,
    compute_ground_penetration,
    filter_clips_by_ground_penetration,
    strip_ground_penetration_field,
)
from humanoidverse.utils.motion_data.robot_state import RobotStateMotion
from humanoidverse.utils.robot_spec import load_robot_spec

TINY_XML = """
<mujoco model="tiny_ground">
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="base_sphere" type="sphere" size="0.1" mass="1"/>
      <body name="leg" pos="0 0 -0.1">
        <joint name="hip" type="hinge" axis="0 1 0" range="-3 3"/>
        <geom name="leg_capsule" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="0.5"/>
        <body name="foot" pos="0 0 -0.5">
          <geom name="foot_box" type="box" size="0.1 0.05 0.02" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_motor" joint="hip"/>
  </actuator>
</mujoco>
""".strip()


def _write_tiny_robot(root: Path) -> Path:
    xml_path = root / "tiny_ground.xml"
    xml_path.write_text(TINY_XML)
    robot_config = root / "tiny_ground.yaml"
    robot_config.write_text(
        "\n".join(
            [
                "name: tiny_ground",
                "xml_path: tiny_ground.xml",
                "base_body: base",
                "root_quat_order: xyzw",
                "control_joints:",
                "  mode: all_actuated",
                "feet: [foot]",
                "hands: []",
                "key_bodies: [base, foot]",
            ]
        )
    )
    return robot_config


def _identity_quat(n: int) -> np.ndarray:
    quat = np.zeros((n, 4), dtype=np.float32)
    quat[:, 3] = 1.0
    return quat


class GroundPenetrationTest(unittest.TestCase):
    def test_penetration_matches_analytic_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_robot_spec(_write_tiny_robot(Path(tmpdir)))
            # Straight leg: lowest point is the foot box bottom at base_z - 0.1 - 0.5 - 0.02.
            root_pos = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.3]], dtype=np.float32)
            dof = np.zeros((2, 1), dtype=np.float32)
            pen = compute_ground_penetration(spec, root_pos, _identity_quat(2), dof)
            np.testing.assert_allclose(pen, [0.0, 0.32], atol=1e-6)

            # Leg rotated 90 deg about y: the box half-extent 0.1 points down from z = base_z - 0.1.
            root_pos = np.array([[0.0, 0.0, 0.05]], dtype=np.float32)
            dof = np.array([[np.pi / 2]], dtype=np.float32)
            pen = compute_ground_penetration(spec, root_pos, _identity_quat(1), dof)
            np.testing.assert_allclose(pen, [0.15], atol=1e-6)

    def test_per_frame_lift_removes_sustained_penetration_and_keeps_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_robot_spec(_write_tiny_robot(Path(tmpdir)))
            frames = 60
            root_pos = np.zeros((frames, 3), dtype=np.float32)
            root_pos[:, 2] = 0.62  # foot bottom exactly at z = 0
            root_pos[20:40, 2] = 0.42  # sustained 0.2 m penetration
            motion = RobotStateMotion(
                motion_key="m",
                root_pos=root_pos,
                root_quat=_identity_quat(frames),
                dof_pos=np.zeros((frames, 1), dtype=np.float32),
                fps=30.0,
            )
            config = GroundAlignmentConfig(mode="per_frame_lift", tolerance_m=0.02, smooth_seconds=0.0)
            aligned, residual, stats = apply_ground_alignment(motion, spec, config)
            self.assertEqual(stats["frames_penetrating"], 20)
            self.assertAlmostEqual(stats["max_penetration_m"], 0.2, places=5)
            np.testing.assert_allclose(aligned.root_pos[20:40, 2], 0.42 + 0.18, atol=1e-5)
            np.testing.assert_allclose(aligned.root_pos[:20, 2], 0.62, atol=1e-6)
            np.testing.assert_allclose(residual[20:40], 0.02, atol=1e-5)
            # The original motion object is untouched.
            np.testing.assert_allclose(motion.root_pos[20:40, 2], 0.42, atol=1e-6)

    def test_smoothing_spreads_lift_and_mode_none_only_measures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_robot_spec(_write_tiny_robot(Path(tmpdir)))
            frames = 90
            root_pos = np.zeros((frames, 3), dtype=np.float32)
            root_pos[:, 2] = 0.62
            root_pos[45, 2] = 0.42  # one-frame spike
            motion = RobotStateMotion(
                motion_key="m",
                root_pos=root_pos,
                root_quat=_identity_quat(frames),
                dof_pos=np.zeros((frames, 1), dtype=np.float32),
                fps=30.0,
            )
            smoothed, residual, stats = apply_ground_alignment(
                motion, spec, GroundAlignmentConfig(tolerance_m=0.02, smooth_seconds=0.5)
            )
            lift = smoothed.root_pos[:, 2] - root_pos[:, 2]
            self.assertLess(lift[45], 0.18)  # spike is only partially lifted ...
            self.assertGreater(lift[45], 0.0)
            self.assertGreater(lift[40], 0.0)  # ... and neighbours are lifted a little
            self.assertAlmostEqual(float(lift.sum()), 0.18, places=4)  # moving average preserves total lift
            self.assertGreater(float(residual[45]), 0.02)

            measured, residual_none, stats_none = apply_ground_alignment(motion, spec, GroundAlignmentConfig(mode="none"))
            self.assertIs(measured, motion)
            self.assertAlmostEqual(float(residual_none[45]), 0.2, places=5)
            self.assertEqual(stats_none["max_lift_m"], 0.0)

    def test_filter_and_strip_helpers(self) -> None:
        data = {
            "good": {GROUND_PENETRATION_FIELD: np.array([0.0, 0.05]), "fps": 30},
            "bad": {GROUND_PENETRATION_FIELD: np.array([0.0, 0.5]), "fps": 30},
            "legacy": {"fps": 30},
        }
        kept, dropped = filter_clips_by_ground_penetration(data, 0.1, source_name="unit")
        self.assertEqual(sorted(kept), ["good", "legacy"])
        self.assertEqual(dropped, ["bad"])
        stripped = strip_ground_penetration_field(kept)
        self.assertNotIn(GROUND_PENETRATION_FIELD, stripped["good"])
        self.assertIn(GROUND_PENETRATION_FIELD, kept["good"])  # input untouched
        with self.assertRaisesRegex(ValueError, "dropped every clip"):
            filter_clips_by_ground_penetration({"bad": data["bad"]}, 0.1, source_name="unit")

    def test_manifest_parsing(self) -> None:
        self.assertIsNone(GroundAlignmentConfig.from_manifest(None))
        self.assertIsNone(GroundAlignmentConfig.from_manifest(False))
        self.assertEqual(GroundAlignmentConfig.from_manifest(True), GroundAlignmentConfig())
        cfg = GroundAlignmentConfig.from_manifest({"mode": "none", "max_residual_penetration_m": None})
        self.assertEqual(cfg.mode, "none")
        self.assertIsNone(cfg.max_residual_penetration_m)
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            GroundAlignmentConfig.from_manifest({"tolerance": 0.1})
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            GroundAlignmentConfig(mode="lift")

    def test_auto_build_manifest_aligns_and_filters_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            robot_config = _write_tiny_robot(root)
            fps = 10
            frames = 40  # 4 s -> 4 one-second clips
            root_pos = np.zeros((frames, 3), dtype=np.float32)
            root_pos[:, 2] = 0.62
            root_pos[10:20, 2] = 0.42  # clip 1: sustained penetration, fully lifted
            root_pos[35, 2] = 0.02  # clip 3: 0.6 m one-frame spike -> residual stays deep -> dropped
            rows = np.concatenate([root_pos, _identity_quat(frames), np.zeros((frames, 1), dtype=np.float32)], axis=1)
            csv_path = root / "motion_a.csv"
            np.savetxt(csv_path, rows, delimiter=",")
            manifest = root / "manifest.yaml"
            manifest.write_text(
                "\n".join(
                    [
                        f"robot_config: {robot_config}",
                        "datasets:",
                        "  - name: tiny",
                        "    format: robot_state_csv",
                        f"    source_path: {csv_path}",
                        "    weight: 1.0",
                        f"    fps: {fps}",
                        "    ground_alignment:",
                        "      mode: per_frame_lift",
                        "      tolerance_m: 0.02",
                        "      smooth_seconds: 0.3",
                        "      max_residual_penetration_m: 0.1",
                        "    auto_build:",
                        "      train_clip_seconds: 1.0",
                        "      clip_stride_seconds: 1.0",
                        "      keep_short: true",
                        "      min_clip_seconds: 0.5",
                    ]
                )
            )
            result = prepare_motion_manifest(manifest, rebuild_cache=True, cache_root=root / "cache")
            train = joblib.load(result.train_data_paths[0])
            self.assertEqual(sorted(train), ["motion_a__clip000", "motion_a__clip001", "motion_a__clip002"])
            for record in train.values():
                self.assertNotIn(GROUND_PENETRATION_FIELD, record)
                self.assertIn("ground_alignment", record["metadata"])
            lifted = train["motion_a__clip001"]["root_trans_offset"][:, 2]
            self.assertTrue(np.all(lifted > 0.42 + 0.1))
            full = joblib.load(result.inference_paths["tiny"])
            self.assertEqual(list(full), ["motion_a"])
            self.assertNotIn(GROUND_PENETRATION_FIELD, full["motion_a"])
            self.assertEqual(full["motion_a"]["metadata"]["ground_alignment"]["frames_penetrating"], 11)


if __name__ == "__main__":
    unittest.main()
