"""Pure, local-only profile inheritance resolution."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def _merge(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(parent))
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            # Lists and scalars deliberately replace the parent value.
            result[key] = copy.deepcopy(value)
    return result


def resolve_manifest_inheritance(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve local ``extends`` references without executing external data."""

    result = copy.deepcopy(dict(payload))
    raw_profiles = result.get("profiles")
    if not isinstance(raw_profiles, list):
        return result
    by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_profiles:
        if not isinstance(raw, Mapping):
            continue
        profile_id = str(raw.get("id", "")).strip()
        if profile_id:
            if profile_id in by_id:
                raise ValueError(f"duplicate profile id: {profile_id}")
            by_id[profile_id] = dict(raw)

    resolved: dict[str, dict[str, Any]] = {}
    visiting: list[str] = []

    def resolve(profile_id: str) -> dict[str, Any]:
        if profile_id in resolved:
            return copy.deepcopy(resolved[profile_id])
        if profile_id in visiting:
            cycle = " -> ".join([*visiting, profile_id])
            raise ValueError(f"profile inheritance cycle: {cycle}")
        raw = by_id.get(profile_id)
        if raw is None:
            raise ValueError(f"unknown profile parent: {profile_id}")
        visiting.append(profile_id)
        parent_ref = raw.get("extends")
        if parent_ref is None:
            merged = _merge({}, raw)
        else:
            if not isinstance(parent_ref, str) or not parent_ref.strip():
                raise ValueError(f"extends must be a local profile id for {profile_id!r}")
            parent_name = parent_ref.strip()
            if "://" in parent_name or "/" in parent_name or "\\" in parent_name:
                raise ValueError(f"remote or path profile parent is forbidden: {parent_name!r}")
            merged = _merge(resolve(parent_name), raw)
        visiting.pop()
        resolved[profile_id] = merged
        return copy.deepcopy(merged)

    output: list[Any] = []
    for raw in raw_profiles:
        if not isinstance(raw, Mapping):
            output.append(copy.deepcopy(raw))
            continue
        profile_id = str(raw.get("id", "")).strip()
        if profile_id:
            output.append(resolve(profile_id))
        else:
            output.append(copy.deepcopy(dict(raw)))
    result["profiles"] = output
    return result
