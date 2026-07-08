"""Validation helpers for robot YAML data."""

from __future__ import annotations

from collections.abc import Iterable


def ensure_known_names(kind: str, names: Iterable[str], known_names: set[str]) -> list[str]:
    values = [str(name) for name in names]
    missing = [name for name in values if name not in known_names]
    if missing:
        raise ValueError(f"Unknown {kind} names in robot config: {missing}. Known names include: {sorted(known_names)[:12]}")
    return values


def ensure_unique(kind: str, names: Iterable[str]) -> list[str]:
    values = [str(name) for name in names]
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in values:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"Duplicate {kind} names in robot config: {duplicates}")
    return values
