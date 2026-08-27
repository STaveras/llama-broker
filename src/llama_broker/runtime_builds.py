"""Broker-owned catalog of concrete runtime builds."""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .runtimes import get_runtime_info


SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RuntimeBuildError(ValueError):
    """A runtime build catalog or selection is invalid."""


@dataclass(frozen=True)
class RuntimeBuild:
    id: str
    runtime: str
    display_name: str
    server_binary: Path
    server_sha256: str


def parse_runtime_builds(
    raw: object, *, allow_missing_artifacts: bool = False
) -> tuple[RuntimeBuild, ...]:
    if not isinstance(raw, list):
        raise RuntimeBuildError("runtime_builds must be a list")

    builds: list[RuntimeBuild] = []
    seen: set[str] = set()
    fields = {"id", "runtime", "display_name", "server_binary", "server_sha256"}
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeBuildError("every runtime build must be a JSON object")
        unknown = sorted(set(item) - fields)
        if unknown:
            raise RuntimeBuildError(
                "unknown runtime build fields: " + ", ".join(unknown)
            )

        build_id = item.get("id")
        if not isinstance(build_id, str) or not SAFE_ID_RE.fullmatch(build_id.strip()):
            raise RuntimeBuildError(f"invalid runtime build id: {build_id!r}")
        build_id = build_id.strip()
        if build_id in seen:
            raise RuntimeBuildError(f"duplicate runtime build id: {build_id}")
        seen.add(build_id)

        runtime = item.get("runtime")
        if not isinstance(runtime, str):
            raise RuntimeBuildError(f"runtime must be a string for build {build_id!r}")
        runtime = runtime.strip().lower()
        try:
            get_runtime_info(runtime)
        except ValueError as error:
            raise RuntimeBuildError(str(error)) from error

        display_name = item.get("display_name")
        if not isinstance(display_name, str) or not display_name.strip():
            raise RuntimeBuildError(
                f"display_name must be a non-empty string for build {build_id!r}"
            )

        server_binary = item.get("server_binary")
        if not isinstance(server_binary, str):
            raise RuntimeBuildError(
                f"server_binary must be a string for build {build_id!r}"
            )
        binary_path = Path(server_binary).expanduser()
        if not binary_path.is_absolute() or (
            not allow_missing_artifacts and not binary_path.is_file()
        ):
            raise RuntimeBuildError(
                f"server_binary must be an absolute existing executable for build {build_id!r}"
            )
        if not allow_missing_artifacts and not os.access(binary_path, os.X_OK):
            raise RuntimeBuildError(
                f"server_binary must be executable for build {build_id!r}: {binary_path}"
            )

        server_sha256 = item.get("server_sha256")
        if not isinstance(server_sha256, str) or not SHA256_RE.fullmatch(server_sha256.strip()):
            raise RuntimeBuildError(
                f"server_sha256 must be a lowercase 64-character SHA-256 for build {build_id!r}"
            )

        builds.append(
            RuntimeBuild(
                id=build_id,
                runtime=runtime,
                display_name=display_name.strip(),
                server_binary=binary_path.resolve(),
                server_sha256=server_sha256.strip(),
            )
        )
    return tuple(builds)


def catalog_from_manifest(
    manifest: Mapping[str, Any], *, allow_missing_artifacts: bool = False
) -> tuple[RuntimeBuild, ...]:
    return parse_runtime_builds(
        manifest.get("runtime_builds", []),
        allow_missing_artifacts=allow_missing_artifacts,
    )


def runtime_build_label(build: RuntimeBuild) -> str:
    return f"{build.display_name} ({build.id})"


def serving_backend_label(build: RuntimeBuild) -> str:
    return f"{build.display_name} ({build.runtime})"


def resolve_runtime_build(
    manifest: Mapping[str, Any],
    build_id: str,
    *,
    allow_missing_artifacts: bool = False,
) -> RuntimeBuild:
    wanted = str(build_id).strip().casefold()
    for build in catalog_from_manifest(
        manifest, allow_missing_artifacts=allow_missing_artifacts
    ):
        if wanted in {
            build.id.casefold(),
            build.display_name.casefold(),
            runtime_build_label(build).casefold(),
            serving_backend_label(build).casefold(),
        }:
            return build
    raise RuntimeBuildError(f"unknown runtime build: {build_id}")


def _profile_value(profile: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(profile, Mapping):
        return profile.get(key, default)
    return getattr(profile, key, default)


def infer_runtime_build(
    manifest: Mapping[str, Any],
    profile: Mapping[str, Any] | object,
    *,
    allow_missing_artifacts: bool = False,
) -> RuntimeBuild | None:
    runtime = str(_profile_value(profile, "runtime", "llamacpp")).strip().lower()
    server_binary = _profile_value(profile, "server_binary")
    server_sha256 = _profile_value(profile, "server_sha256")
    if not server_binary or not server_sha256:
        return None
    binary_path = Path(str(server_binary)).expanduser()
    if not binary_path.is_absolute():
        return None
    matches = [
        build
        for build in catalog_from_manifest(
            manifest, allow_missing_artifacts=allow_missing_artifacts
        )
        if build.runtime == runtime
        and build.server_binary == binary_path.resolve()
        and build.server_sha256 == str(server_sha256).strip()
    ]
    return matches[0] if len(matches) == 1 else None


def apply_runtime_build(
    manifest: Mapping[str, Any],
    profile: Mapping[str, Any],
    build_id: str,
    *,
    runtime: str | None = None,
) -> dict[str, Any]:
    build = resolve_runtime_build(manifest, build_id)
    selected_runtime = str(
        runtime if runtime is not None else profile.get("runtime", "llamacpp")
    ).strip().lower()
    if build.runtime != selected_runtime:
        raise RuntimeBuildError(
            f"runtime build {build.id!r} uses runtime {build.runtime!r}, "
            f"not {selected_runtime!r}"
        )
    updated = copy.deepcopy(dict(profile))
    updated.update(
        {
            "runtime": build.runtime,
            "runtime_build": build.id,
            "server_binary": str(build.server_binary),
            "server_sha256": build.server_sha256,
        }
    )
    return updated
