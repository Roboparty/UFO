"""Inspect a MuJoCo XML and write a UFO RobotSpec YAML draft."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mujoco
import yaml
from loguru import logger
from omegaconf import OmegaConf

from humanoidverse.utils.robot_spec import load_robot_spec, load_robot_training_spec
from humanoidverse.utils.robot_spec.xml_training_infer import build_hydra_robot_config_draft, build_robot_training_yaml_draft


class _IndentedSafeDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False):  # noqa: ANN201 - PyYAML hook signature.
        return super().increase_indent(flow, False)


def _mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return str(name) if name else f"{obj_type.name.lower()}_{obj_id}"


def _body_names(model: mujoco.MjModel) -> list[str]:
    return [_mj_name(model, mujoco.mjtObj.mjOBJ_BODY, idx) for idx in range(1, model.nbody)]


def _joint_names(model: mujoco.MjModel) -> list[str]:
    return [_mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, idx) for idx in range(model.njnt)]


def _actuated_joint_names(model: mujoco.MjModel) -> list[str]:
    joints: list[str] = []
    for idx in range(model.nu):
        if int(model.actuator_trntype[idx]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        joint_id = int(model.actuator_trnid[idx, 0])
        if joint_id >= 0:
            joints.append(_mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
    return joints


def _freejoint_body(model: mujoco.MjModel) -> tuple[str | None, str | None]:
    for idx in range(model.njnt):
        if int(model.jnt_type[idx]) == int(mujoco.mjtJoint.mjJNT_FREE):
            joint_name = _mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)
            body_name = _mj_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[idx]))
            return joint_name, body_name
    return None, None


def _body_parent_map(model: mujoco.MjModel) -> dict[str, str | None]:
    names = [_mj_name(model, mujoco.mjtObj.mjOBJ_BODY, idx) for idx in range(model.nbody)]
    parents: dict[str, str | None] = {}
    for idx in range(1, model.nbody):
        parent_id = int(model.body_parentid[idx])
        parents[names[idx]] = None if parent_id == 0 else names[parent_id]
    return parents


def _body_depths(model: mujoco.MjModel) -> dict[str, int]:
    names = [_mj_name(model, mujoco.mjtObj.mjOBJ_BODY, idx) for idx in range(model.nbody)]
    depths: dict[str, int] = {}
    for idx in range(1, model.nbody):
        depth = 0
        parent_id = int(model.body_parentid[idx])
        while parent_id != 0:
            depth += 1
            parent_id = int(model.body_parentid[parent_id])
        depths[names[idx]] = depth
    return depths


def _leaf_bodies(model: mujoco.MjModel) -> set[str]:
    names = [_mj_name(model, mujoco.mjtObj.mjOBJ_BODY, idx) for idx in range(model.nbody)]
    parents = {int(model.body_parentid[idx]) for idx in range(1, model.nbody)}
    return {names[idx] for idx in range(1, model.nbody) if idx not in parents}


def _ordered_unique(values: list[str], xml_order: list[str]) -> list[str]:
    seen = set()
    allowed = set(values)
    result: list[str] = []
    for name in xml_order:
        if name in allowed and name not in seen:
            result.append(name)
            seen.add(name)
    return result


def _side_matches(lower_name: str, side_tokens: tuple[str, ...]) -> bool:
    if side_tokens == ("",):
        return True
    return any(token in lower_name for token in side_tokens)


def _select_side_candidate(
    body_names: list[str],
    depths: dict[str, int],
    leaves: set[str],
    *,
    side_tokens: tuple[str, ...],
    keywords: tuple[str, ...],
    preferred: tuple[str, ...],
) -> str | None:
    best: tuple[float, int, str] | None = None
    for idx, name in enumerate(body_names):
        lower = name.lower()
        if not _side_matches(lower, side_tokens):
            continue
        if not any(keyword in lower for keyword in keywords):
            continue
        keyword_score = max((len(keyword) for keyword in keywords if keyword in lower), default=0)
        preferred_score = 5 if any(token in lower for token in preferred) else 0
        leaf_score = 3 if name in leaves else 0
        score = preferred_score + keyword_score + leaf_score + depths.get(name, 0) * 0.1
        candidate = (score, -idx, name)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best is not None else None


def _guess_pair(body_names: list[str], depths: dict[str, int], leaves: set[str], *, kind: str) -> list[str]:
    if kind == "feet":
        keywords = ("foot", "toe", "ankle")
        preferred = ("foot", "toe", "ankle_roll")
    elif kind == "hands":
        keywords = ("hand", "wrist", "gripper", "palm")
        preferred = ("hand", "palm", "gripper", "wrist_yaw")
    else:
        raise ValueError(f"Unsupported semantic pair kind={kind}")
    left = _select_side_candidate(body_names, depths, leaves, side_tokens=("left", "l_"), keywords=keywords, preferred=preferred)
    right = _select_side_candidate(body_names, depths, leaves, side_tokens=("right", "r_"), keywords=keywords, preferred=preferred)
    guessed = [name for name in (left, right) if name is not None]
    if guessed:
        return _ordered_unique(guessed, body_names)
    fallback = _select_side_candidate(body_names, depths, leaves, side_tokens=("",), keywords=keywords, preferred=preferred)
    return [fallback] if fallback is not None else []


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def infer_hydra_robot_group(hydra_out: str | Path) -> str:
    path = Path(hydra_out).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    hydra_root = (_repo_root() / "humanoidverse" / "config" / "robot").resolve()
    try:
        rel_path = path.resolve().relative_to(hydra_root)
    except ValueError as exc:
        raise ValueError(
            "--hydra-out must be under humanoidverse/config/robot/ so the Hydra robot group path can be inferred. "
            "Use --with-training --hydra-robot if you want to provide an existing Hydra group instead."
        ) from exc
    if rel_path.suffix not in {".yaml", ".yml"}:
        raise ValueError("--hydra-out must point to a .yaml or .yml file")
    return str(rel_path.with_suffix("")).replace("\\", "/")


def _save_yaml(path: Path, config: dict[str, Any], *, force: bool, header: str | None = None) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output file already exists: {path}. Use --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(config, Dumper=_IndentedSafeDumper, sort_keys=False, default_flow_style=False)
    if header:
        text = header.rstrip() + "\n" + text
    path.write_text(text)


def inspect_mujoco_xml(xml_path: str | Path, *, name: str | None = None) -> dict[str, Any]:
    path = Path(xml_path).expanduser()
    model = mujoco.MjModel.from_xml_path(str(path))
    body_names = _body_names(model)
    free_joint, free_body = _freejoint_body(model)
    return {
        "name": name or path.stem,
        "xml_path": str(xml_path),
        "model": model,
        "body_names": body_names,
        "joint_names": _joint_names(model),
        "actuated_joint_names": _actuated_joint_names(model),
        "free_joint": free_joint,
        "free_body": free_body,
        "body_parent": _body_parent_map(model),
        "body_depths": _body_depths(model),
        "leaf_bodies": _leaf_bodies(model),
    }


def infer_robot_semantics(
    inspection: dict[str, Any],
    *,
    base_body: str | None = None,
    feet: list[str] | None = None,
    hands: list[str] | None = None,
    key_bodies: list[str] | None = None,
) -> dict[str, Any]:
    body_names = list(inspection["body_names"])
    depths = dict(inspection["body_depths"])
    leaves = set(inspection["leaf_bodies"])
    inferred_base = base_body or inspection.get("free_body")
    if inferred_base is None:
        inferred_base = body_names[0] if body_names else ""
        logger.warning(f"No freejoint found; using first non-world body as base_body={inferred_base!r}")
    inferred_feet = feet if feet is not None else _guess_pair(body_names, depths, leaves, kind="feet")
    inferred_hands = hands if hands is not None else _guess_pair(body_names, depths, leaves, kind="hands")
    if not inferred_feet:
        logger.warning("Could not guess feet bodies; leaving feet empty. Please edit the robot YAML if needed.")
    if not inferred_hands:
        logger.warning("Could not guess hands bodies; leaving hands empty. Please edit the robot YAML if needed.")
    if key_bodies is not None:
        inferred_key_bodies = key_bodies
    else:
        torso_like = [name for name in body_names if any(token in name.lower() for token in ("torso", "chest", "trunk", "waist"))]
        inferred_key_bodies = _ordered_unique([inferred_base, *torso_like, *inferred_feet, *inferred_hands], body_names)
    return {"base_body": inferred_base, "feet": inferred_feet, "hands": inferred_hands, "key_bodies": inferred_key_bodies}


def write_robot_yaml(
    out_path: str | Path,
    inspection: dict[str, Any],
    semantics: dict[str, Any],
    *,
    root_quat_order: str = "xyzw",
    coordinate_system: str = "z_up",
    dof_unit: str = "rad",
    force: bool = False,
) -> Path:
    path = Path(out_path).expanduser()
    if path.exists() and not force:
        raise FileExistsError(f"Robot YAML already exists: {path}. Use --force to overwrite it.")
    config = {
        "name": inspection["name"],
        "xml_path": inspection["xml_path"],
        "base_body": semantics["base_body"],
        "root_quat_order": root_quat_order,
        "coordinate_system": coordinate_system,
        "dof_unit": dof_unit,
        "control_joints": {"mode": "all_actuated"},
        "feet": list(semantics["feet"]),
        "hands": list(semantics["hands"]),
        "key_bodies": list(semantics["key_bodies"]),
        "default_dof_pos": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), path)
    load_robot_spec(path)
    return path


def write_robot_training_yaml(
    out_path: str | Path,
    inspection: dict[str, Any],
    semantics: dict[str, Any],
    *,
    hydra_robot: str,
    args: argparse.Namespace,
) -> Path:
    path = Path(out_path).expanduser()
    config = build_robot_training_yaml_draft(
        model=inspection["model"],
        name=str(inspection["name"]),
        xml_path=str(inspection["xml_path"]),
        hydra_robot=hydra_robot,
        semantics=semantics,
        default_keyframe=args.default_keyframe,
        actuator_source=args.actuator_source,
        action_scale=args.action_scale,
        action_clip_value=args.action_clip_value,
        normalize_action_to=args.normalize_action_to,
        effort_limit_default=args.effort_limit_default,
        velocity_limit_default=args.velocity_limit_default,
        armature_default=args.armature_default,
        friction_default=args.friction_default,
        pd_template=args.pd_template,
        review_status=args.review_status,
    )
    _save_yaml(path, config, force=bool(args.force))
    load_robot_training_spec(path)
    return path


def _print_summary(inspection: dict[str, Any], semantics: dict[str, Any]) -> None:
    model = inspection["model"]
    print("Robot XML summary")
    print(f"  name: {inspection['name']}")
    print(f"  nq/nv/nu: {model.nq}/{model.nv}/{model.nu}")
    print(f"  bodies: {len(inspection['body_names'])}")
    print(f"  joints: {len(inspection['joint_names'])}")
    print(f"  actuators: {model.nu}")
    print(f"  freejoint: {inspection.get('free_joint')} on body {inspection.get('free_body')}")
    print(f"  base_body: {semantics['base_body']}")
    print(f"  actuated_joints: {inspection['actuated_joint_names']}")
    print(f"  guessed feet: {semantics['feet']}")
    print(f"  guessed hands: {semantics['hands']}")
    print(f"  guessed key_bodies: {semantics['key_bodies']}")
    print("  Please inspect feet/hands/key_bodies before large-scale training.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True, help="MuJoCo XML path")
    parser.add_argument("--name", required=True, help="Robot name for the generated YAML")
    parser.add_argument("--out", required=True, help="Output robot YAML path")
    parser.add_argument("--base-body", default=None, help="Override base_body")
    parser.add_argument("--feet", default=None, help="Comma-separated feet body override")
    parser.add_argument("--hands", default=None, help="Comma-separated hands body override")
    parser.add_argument("--key-bodies", default=None, help="Comma-separated key_bodies override")
    parser.add_argument("--root-quat-order", default="xyzw", choices=["xyzw", "wxyz"])
    parser.add_argument("--coordinate-system", default="z_up")
    parser.add_argument("--dof-unit", default="rad")
    parser.add_argument("--with-training", action="store_true", help="Also write a draft training section into the Robot YAML.")
    parser.add_argument("--hydra-out", default=None, help="Output path for an auto-generated Hydra robot config draft.")
    parser.add_argument("--hydra-robot", default=None, help="Existing Hydra robot group, required with --with-training when --hydra-out is omitted.")
    parser.add_argument("--default-keyframe", default="stand", help="Preferred MuJoCo keyframe name for default pose inference.")
    parser.add_argument("--actuator-source", default="yaml", choices=["yaml", "g1_mode15", "g1-mode_15"])
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--action-clip-value", type=float, default=5.0)
    parser.add_argument("--normalize-action-to", type=float, default=5.0)
    parser.add_argument("--effort-limit-default", type=float, default=80.0)
    parser.add_argument("--velocity-limit-default", type=float, default=20.0)
    parser.add_argument("--armature-default", type=float, default=0.01)
    parser.add_argument("--friction-default", type=float, default=0.0)
    parser.add_argument("--pd-template", default="humanoid", choices=["humanoid"])
    parser.add_argument("--review-status", default="draft")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args(argv)
    if args.hydra_out is not None:
        args.with_training = True
        if args.hydra_robot is None:
            args.hydra_robot = infer_hydra_robot_group(args.hydra_out)
    if args.with_training and args.hydra_robot is None:
        parser.error("--with-training requires either --hydra-out or --hydra-robot.")
    if args.actuator_source != "yaml":
        logger.warning("Non-yaml actuator sources are legacy/G1-specific. Use --actuator-source yaml for new robots.")

    inspection = inspect_mujoco_xml(args.xml, name=args.name)
    semantics = infer_robot_semantics(
        inspection,
        base_body=args.base_body,
        feet=_parse_csv_list(args.feet),
        hands=_parse_csv_list(args.hands),
        key_bodies=_parse_csv_list(args.key_bodies),
    )
    if args.with_training:
        out_path = write_robot_training_yaml(args.out, inspection, semantics, hydra_robot=args.hydra_robot, args=args)
    else:
        out_path = write_robot_yaml(
            args.out,
            inspection,
            semantics,
            root_quat_order=args.root_quat_order,
            coordinate_system=args.coordinate_system,
            dof_unit=args.dof_unit,
            force=args.force,
        )
    if args.hydra_out is not None:
        hydra_path = Path(args.hydra_out).expanduser()
        robot_config = OmegaConf.to_container(OmegaConf.load(out_path), resolve=True)
        hydra_config = build_hydra_robot_config_draft(
            model=inspection["model"],
            robot_config=robot_config,
            pd_template=args.pd_template,
            review_status=args.review_status,
        )
        _save_yaml(
            hydra_path,
            hydra_config,
            force=bool(args.force),
            header="# AUTO-GENERATED DRAFT. REVIEW BEFORE TRAINING.\n# @package _global_",
        )
        print(f"Wrote Hydra robot config draft: {hydra_path}")
    _print_summary(inspection, semantics)
    print(f"Wrote robot YAML: {out_path}")
    if args.with_training:
        robot_config = OmegaConf.to_container(OmegaConf.load(out_path), resolve=True)
        warnings = list((robot_config.get("metadata") or {}).get("warnings") or [])
        print("Robot training draft generated. Review semantics, default pose, PD gains, actuator parameters, contact bodies, and reward/termination-related fields before formal training.")
        for warning in warnings:
            print(f"  warning: {warning}")


if __name__ == "__main__":
    main()
