#!/usr/bin/env python3
"""Validate deployment model artifacts without committing model binaries."""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class Reporter:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")


def load_manifest(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not load as a mapping")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("manifest must contain an artifacts list")
    return data


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _shape_matches(actual: list[Any], expected: list[Any]) -> bool:
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected, strict=True):
        if isinstance(e, str):
            continue
        if a != e:
            return False
    return True


def _check_onnx(path: Path, artifact: dict[str, Any], reporter: Reporter) -> None:
    import onnxruntime as ort

    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        reporter.fail(f"ONNX load failed for {path}: {exc.__class__.__name__}: {exc}")
        return

    reporter.ok(f"ONNX loads with CPUExecutionProvider: {path}")
    reporter.info(f"providers: {sess.get_providers()}")
    inputs = {x.name: list(x.shape) for x in sess.get_inputs()}
    outputs = {x.name: list(x.shape) for x in sess.get_outputs()}
    reporter.info(f"inputs: {inputs}")
    reporter.info(f"outputs: {outputs}")

    expected_inputs = artifact.get("expected_inputs") or {}
    expected_outputs = artifact.get("expected_outputs") or {}
    for name, expected in expected_inputs.items():
        if name not in inputs:
            reporter.fail(f"{path} missing expected input {name!r}")
        elif not _shape_matches(inputs[name], list(expected)):
            reporter.fail(f"{path} input {name!r} shape {inputs[name]} != expected {expected}")
        else:
            reporter.ok(f"{path} input {name!r} shape matches {expected}")
    for name, expected in expected_outputs.items():
        if name not in outputs:
            reporter.fail(f"{path} missing expected output {name!r}")
        elif not _shape_matches(outputs[name], list(expected)):
            reporter.fail(f"{path} output {name!r} shape {outputs[name]} != expected {expected}")
        else:
            reporter.ok(f"{path} output {name!r} shape matches {expected}")


def _load_context(path: Path) -> np.ndarray:
    try:
        import joblib

        obj = joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            obj = pickle.load(f)
    return np.asarray(obj)


def _check_context(path: Path, artifact: dict[str, Any], reporter: Reporter) -> int | None:
    try:
        arr = _load_context(path)
    except Exception as exc:
        reporter.fail(f"context load failed for {path}: {exc.__class__.__name__}: {exc}")
        return None
    if arr.ndim < 2 or not np.all(np.isfinite(arr[:1])):
        reporter.fail(f"context has invalid shape/value: {arr.shape}")
        return None
    expected_last_dim = artifact.get("expected_last_dim")
    if expected_last_dim is not None and int(arr.shape[-1]) != int(expected_last_dim):
        reporter.fail(f"context last dim {arr.shape[-1]} != expected {expected_last_dim}")
    else:
        reporter.ok(f"context loads: {path}")
        reporter.info(f"context shape: {arr.shape}")
    return int(arr.shape[-1])


def _check_obs_plus_context(policy_onnx: Path, context_dim: int | None, reporter: Reporter) -> None:
    if context_dim is None:
        return
    sys.path.insert(0, str(ROOT))
    try:
        from scripts.onboard.check_policy_preflight import _compute_obs_dim, _read_yaml
    except Exception as exc:
        reporter.warn(f"could not import policy preflight helpers: {exc.__class__.__name__}: {exc}")
        return
    policy_config = _read_yaml(ROOT / "config/policy/g1_policy.yaml")
    obs_dim = _compute_obs_dim(policy_config, reporter)
    if obs_dim is None:
        return
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(policy_onnx), providers=["CPUExecutionProvider"])
        actor_obs = sess.get_inputs()[0].shape[-1]
    except Exception as exc:
        reporter.warn(f"could not re-open policy ONNX for obs/context check: {exc.__class__.__name__}: {exc}")
        return
    if int(obs_dim) + int(context_dim) == int(actor_obs):
        reporter.ok(f"policy obs + context matches ONNX input: {obs_dim}+{context_dim}={actor_obs}")
    else:
        reporter.fail(f"policy obs + context {obs_dim}+{context_dim} does not match ONNX input {actor_obs}")


def check_manifest(manifest_path: Path, root: Path, reporter: Reporter) -> None:
    manifest = load_manifest(manifest_path)
    policy_onnx: Path | None = None
    context_dim: int | None = None
    for artifact in manifest["artifacts"]:
        if not isinstance(artifact, dict):
            reporter.fail(f"manifest artifact is not a mapping: {artifact!r}")
            continue
        rel = Path(str(artifact.get("path", "")))
        path = rel if rel.is_absolute() else root / rel
        required = bool(artifact.get("required", True))
        artifact_type = str(artifact.get("type", "file"))
        if not path.is_file():
            if required:
                reporter.fail(f"required artifact missing: {path}")
            else:
                reporter.warn(f"optional artifact missing: {path}")
            continue
        reporter.ok(f"artifact exists: {path}")

        expected_hash = artifact.get("sha256")
        if expected_hash:
            actual_hash = sha256_file(path)
            if actual_hash == str(expected_hash):
                reporter.ok(f"sha256 matches: {rel}")
            else:
                reporter.fail(f"sha256 mismatch for {rel}: {actual_hash} != {expected_hash}")
        else:
            reporter.warn(f"no authoritative sha256 in manifest for {rel}")

        if artifact_type.endswith("onnx"):
            _check_onnx(path, artifact, reporter)
            if artifact_type == "policy_onnx":
                policy_onnx = path
        elif artifact_type == "tracking_context":
            context_dim = _check_context(path, artifact, reporter)

    if policy_onnx is not None:
        _check_obs_plus_context(policy_onnx, context_dim, reporter)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate UFO deployment model artifacts")
    parser.add_argument("--manifest", default="model/g1_policy/artifact_manifest.yaml")
    args = parser.parse_args()

    reporter = Reporter()
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    reporter.info(f"repo: {ROOT}")
    reporter.info(f"manifest: {manifest_path}")
    try:
        check_manifest(manifest_path, ROOT, reporter)
    except Exception as exc:
        reporter.fail(f"artifact check failed: {exc.__class__.__name__}: {exc}")

    if reporter.failures:
        reporter.info(f"summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
        return 1
    reporter.info(f"summary: all checks passed, {reporter.warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
