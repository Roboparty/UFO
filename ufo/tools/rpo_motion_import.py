"""Import the ufo_rpo_v3 RPO motion dataset into UFO without resampling.

The v3 converter (codebase/scripts/motion/build_rpo_dataset.py) stores 50 Hz
MuJoCo FK of the RPO model. Its velocities are finite differences of link-origin
poses in the world frame, which already match UFO's motion-library convention,
so they are preserved exactly instead of being recomputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from easydict import EasyDict

from ufo.motion.formats import RobotStateMotion, clip_ufo_motion_dict, robot_state_to_ufo_motion, validate_ufo_motion_dict
from ufo.motion.kinematics import HumanoidKinematics
from ufo.tools.robot_config import load_robot_spec
from ufo.tools.rp1_motion_import import quaternion_error, require, sha256

KEYS = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def convert(manifest_path, config_path, output_path, train_output_path, report_path):
    for path in (output_path, train_output_path, report_path):
        require(not Path(path).exists(), f"Refusing to overwrite {path}")
    require(len({Path(path).resolve() for path in (output_path, train_output_path, report_path)}) == 3, "Output paths must be distinct")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("robot") == "rpo", "Expected a ufo_rpo_v3 RPO motion manifest")
    meta, records = manifest["npz"], manifest["clips"]
    dataset = manifest_path.parent.name
    spec = load_robot_spec(config_path)
    require(meta["body_quaternion_order"] == spec.root_quat_order == "xyzw", "Expected XYZW quaternions")
    require(spec.coordinate_system == "z_up" and spec.dof_unit == "rad", "Expected Z-up and radians")
    joints, bodies = meta["joint_order"], meta["body_order"]
    require(len(joints) == len(set(joints)) and set(joints) == set(spec.control_joint_names), "Joint names do not match robot")
    require(len(bodies) == len(set(bodies)) and set(bodies) == set(spec.body_names), "Body names do not match robot")
    require(spec.body_names[0] == spec.base_body, "UFO requires the root body at index zero")
    parser = HumanoidKinematics(
        EasyDict(asset=EasyDict(assetRoot=str(spec.xml_file.parent), assetFileName=spec.xml_file.name), extend_config=[])
    )
    require(parser.body_names == spec.body_names, "UFO FK body order differs from RobotSpec")
    root_id = bodies.index(spec.base_body)
    body_order = [bodies.index(name) for name in spec.body_names]
    joint_order = [joints.index(name) for name in spec.control_joint_names]
    motions, audits = {}, []
    torch.set_num_threads(1)
    for record in records:
        name = record["name"]
        require(name not in motions, f"Duplicate motion name: {name}")
        path = (manifest_path.parent / record["npz"]).resolve()
        require(path.is_relative_to(manifest_path.parent), f"NPZ path escapes dataset: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            require(set(loaded.files) == set(KEYS), f"{name}: unexpected NPZ fields {sorted(loaded.files)}")
            fields = {key: np.asarray(loaded[key]) for key in KEYS}
        fps = float(fields["fps"].item())
        require(fps == float(meta["output_fps"]), f"{name}: invalid fps")
        frames = record["num_frames"]
        require(frames >= 3, f"{name}: at least three frames required")
        shapes = {
            "joint_pos": (frames, len(joints)),
            "joint_vel": (frames, len(joints)),
            "body_pos_w": (frames, len(bodies), 3),
            "body_quat_w": (frames, len(bodies), 4),
            "body_lin_vel_w": (frames, len(bodies), 3),
            "body_ang_vel_w": (frames, len(bodies), 3),
        }
        for key, shape in shapes.items():
            require(fields[key].shape == shape and np.isfinite(fields[key]).all(), f"{name}: invalid {key} shape/values")
        quaternions = fields["body_quat_w"]
        require(np.max(np.abs(np.linalg.norm(quaternions, axis=-1) - 1)) < 1e-5, f"{name}: non-unit quaternion")
        # Velocities must follow the v3 converter's definitions (float32 storage tolerance).
        joint_gradient_error = float(
            np.max(np.abs(np.gradient(fields["joint_pos"].astype(np.float64), axis=0) * fps - fields["joint_vel"]))
        )
        body_gradient_error = float(
            np.max(np.abs(np.gradient(fields["body_pos_w"].astype(np.float64), axis=0) * fps - fields["body_lin_vel_w"]))
        )
        require(joint_gradient_error < 1e-3, f"{name}: joint velocity is not the documented central difference")
        require(body_gradient_error < 1e-3, f"{name}: body velocity is not the documented link-origin central difference")
        motion = robot_state_to_ufo_motion(
            RobotStateMotion(
                name,
                fields["body_pos_w"][:, root_id],
                quaternions[:, root_id],
                fields["joint_pos"],
                fps,
                joint_names=joints,
                source=dataset,
            ),
            spec,
        )
        motion.update(
            dof_vel=fields["joint_vel"][:, joint_order].copy(),
            body_vel=fields["body_lin_vel_w"][:, body_order].copy(),
            body_ang_vel=fields["body_ang_vel_w"][:, body_order].copy(),
        )
        npz_sha256 = sha256(path)
        motion["metadata"].update(
            source_npz=record["npz"],
            source_npz_sha256=npz_sha256,
            source_csv_sha256=record["source_sha256"],
            source_manifest_sha256=sha256(manifest_path),
            body_velocity_reference="link_origin",
            body_velocity_frame="world",
            joint_velocity_definition="numpy.gradient(joint_pos, axis=0) * fps, one-sided endpoints",
            body_velocity_definition="numpy.gradient(body_pos_w, axis=0) * fps, one-sided endpoints",
            body_angular_velocity_definition="averaged world-frame rotation-vector differences (ufo_rpo_v3)",
            frames_preserved=frames,
        )
        sampled = np.unique(np.linspace(0, frames - 1, min(128, frames), dtype=np.int64))
        with torch.no_grad():
            fk = parser.fk_batch(
                torch.from_numpy(motion["pose_aa"][sampled][None]), torch.from_numpy(motion["root_trans_offset"][sampled][None]), dt=1 / fps
            )
        errors = dict(
            ufo_position_m=float(np.max(np.abs(fk.global_translation[0].numpy() - fields["body_pos_w"][sampled][:, body_order]))),
            ufo_rotation_rad=quaternion_error(fk.global_rotation[0].numpy(), quaternions[sampled][:, body_order]),
        )
        for key, error in errors.items():
            require(error < 2e-5, f"{name}: FK {key} error {error} exceeds tolerance")
        require(np.array_equal(motion["dof_pos"], fields["joint_pos"][:, joint_order]), f"{name}: joint positions changed")
        audits.append(
            dict(
                name=name,
                frames=frames,
                fps=fps,
                npz_sha256=npz_sha256,
                fk_sampled_frames=len(sampled),
                errors=errors,
                joint_gradient_max_error=joint_gradient_error,
                body_gradient_max_error=body_gradient_error,
            )
        )
        motions[name] = motion
        print(json.dumps(audits[-1]), flush=True)
    validate_ufo_motion_dict(motions, dataset)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    joblib.dump(motions, temporary)
    temporary.replace(output_path)
    train_output_path = Path(train_output_path)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    training = clip_ufo_motion_dict(motions, clip_seconds=10, stride_seconds=10, min_clip_seconds=1)
    temporary = train_output_path.with_suffix(train_output_path.suffix + ".tmp")
    joblib.dump(training, temporary)
    temporary.replace(train_output_path)
    report = dict(
        dataset=dataset,
        source_manifest=str(manifest_path),
        source_manifest_sha256=sha256(manifest_path),
        robot_config=str(Path(config_path).resolve()),
        robot_config_sha256=sha256(config_path),
        robot_xml_sha256=sha256(spec.xml_path),
        output=str(output_path.resolve()),
        output_sha256=sha256(output_path),
        train_output=str(train_output_path.resolve()),
        train_output_sha256=sha256(train_output_path),
        train_clips=len(training),
        train_frames=sum(len(motion["dof_pos"]) for motion in training.values()),
        training_windows={"clip_seconds": 10, "stride_seconds": 10, "min_clip_seconds": 1, "keep_short": True},
        clips=len(motions),
        frames=sum(item["frames"] for item in audits),
        joint_order=spec.control_joint_names,
        body_order=spec.body_names,
        velocity_semantics=motions[records[0]["name"]]["metadata"],
        audit=audits,
    )
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest", required=True, type=Path, help="ufo_rpo_v3 codebase/data/motions/<dataset>/manifest.json")
    cli.add_argument("--robot-config", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--train-output", required=True, type=Path)
    cli.add_argument("--report", required=True, type=Path)
    args = cli.parse_args()
    convert(args.manifest, args.robot_config, args.output, args.train_output, args.report)
