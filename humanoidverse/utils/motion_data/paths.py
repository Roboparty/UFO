"""Path expansion helpers for motion data files."""

from __future__ import annotations

import glob
import os
from pathlib import Path


def _as_path_list(path_spec: str | os.PathLike[str] | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(path_spec, (list, tuple)):
        return [str(item) for item in path_spec]
    return [str(path_spec)]


def _candidate_patterns(raw_path: str, base_dir: Path | None) -> list[Path]:
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    path = Path(expanded)
    if path.is_absolute():
        return [path]
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(base_dir / path)
    candidates.append(Path.cwd() / path)
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def expand_motion_paths(
    path_spec: str | os.PathLike[str] | list[str] | tuple[str, ...],
    *,
    base_dir: Path | None = None,
    suffix: str,
) -> list[Path]:
    """Resolve a path, directory, glob, or list of those into concrete files."""

    resolved: list[Path] = []
    missing_patterns: list[str] = []
    for raw in _as_path_list(path_spec):
        found_for_raw: list[Path] = []
        for candidate in _candidate_patterns(raw, base_dir):
            candidate_str = str(candidate)
            if glob.has_magic(candidate_str):
                found_for_raw = [Path(item) for item in sorted(glob.glob(candidate_str))]
            elif candidate.is_dir():
                found_for_raw = sorted(candidate.glob(f"*{suffix}"))
            elif candidate.exists():
                found_for_raw = [candidate]
            if found_for_raw:
                break
        if not found_for_raw:
            missing_patterns.append(raw)
        resolved.extend(found_for_raw)

    filtered = [path.expanduser().resolve() for path in resolved if path.suffix == suffix]
    if not filtered:
        raise FileNotFoundError(f"No {suffix} motion files matched: {missing_patterns or path_spec}")
    return filtered
