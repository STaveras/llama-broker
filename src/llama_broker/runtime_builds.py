"""External catalog of concrete runtime builds.

Schema-v1 manifests may keep their inline ``runtime_builds`` list. Schema-v2
manifests resolve runtime IDs from the external catalog selected by
``LLAMA_BROKER_RUNTIME_CATALOG``.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .paths import RUNTIME_CATALOG_PATH
from .runtimes import get_runtime_info


SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CATALOG_SCHEMA_VERSION = 2


class RuntimeBuildError(ValueError):
    """A runtime build catalog or selection is invalid."""


@dataclass(frozen=True)
class RuntimeBuild:
    id: str
    runtime: str
    display_name: str
    server_binary: Path
    server_sha256: str
    bundle_root: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    source_repository: Path | None = None
    provenance_file: Path | None = None
    source_branch: str | None = None
    source_commit: str | None = None
    dirty: bool | None = None
    selected_profile_ids: tuple[str, ...] = ()
    build_options: Mapping[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    deprecated: bool = False


@dataclass(frozen=True)
class RuntimeCatalog:
    schema_version: int
    deployment_revision: str
    builds: tuple[RuntimeBuild, ...]
    temporary_aliases: Mapping[str, str]
    path: Path

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "deployment_revision": self.deployment_revision,
            "temporary_aliases": dict(self.temporary_aliases),
            "runtimes": [_runtime_build_mapping(build) for build in self.builds],
        }


def _absolute_path(
    value: object,
    field_name: str,
    *,
    allow_missing: bool,
    directory: bool = False,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeBuildError(f"{field_name} must be an absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeBuildError(f"{field_name} must be an absolute path: {value!r}")
    if not allow_missing:
        valid = path.is_dir() if directory else path.is_file()
        if not valid:
            suffix = "directory" if directory else "existing executable"
            raise RuntimeBuildError(
                f"{field_name} must be an absolute existing {suffix}: {path}"
            )
    return path.resolve()


def _string_mapping(raw: object, field_name: str) -> Mapping[str, str]:
    if raw is None:
        return MappingProxyType({})
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
    ):
        raise RuntimeBuildError(f"{field_name} must be a string mapping")
    return MappingProxyType(dict(raw))


def _optional_text(item: Mapping[str, Any], field_name: str) -> str | None:
    value = item.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeBuildError(f"{field_name} must be a non-empty string when present")
    return value.strip()


def _runtime_build_mapping(build: RuntimeBuild) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": build.id,
        "runtime": build.runtime,
        "display_name": build.display_name,
        "bundle_root": str(build.bundle_root) if build.bundle_root else None,
        "server_binary": str(build.server_binary),
        "server_sha256": build.server_sha256,
        "environment": dict(build.environment),
        "source_repository": str(build.source_repository)
        if build.source_repository
        else None,
        "provenance_file": str(build.provenance_file)
        if build.provenance_file
        else None,
        "source_branch": build.source_branch,
        "source_commit": build.source_commit,
        "dirty": build.dirty,
        "selected_profile_ids": list(build.selected_profile_ids),
        "build_options": copy.deepcopy(dict(build.build_options)),
        "aliases": list(build.aliases),
    }
    if build.deprecated:
        result["deprecated"] = True
    return {key: value for key, value in result.items() if value is not None}


def parse_runtime_builds(
    raw: object,
    *,
    allow_missing_artifacts: bool = False,
) -> tuple[RuntimeBuild, ...]:
    """Parse inline or external runtime records without mutating their source."""
    if not isinstance(raw, list):
        raise RuntimeBuildError("runtime builds must be a list")

    builds: list[RuntimeBuild] = []
    seen: set[str] = set()
    fields = {
        "id",
        "runtime",
        "display_name",
        "bundle_root",
        "server_binary",
        "server_sha256",
        "environment",
        "launch_environment",
        "source_repository",
        "provenance_file",
        "source_branch",
        "source_commit",
        "dirty",
        "selected_profile_ids",
        "build_options",
        "aliases",
        "deprecated",
    }
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

        server_binary = _absolute_path(
            item.get("server_binary"),
            "server_binary",
            allow_missing=allow_missing_artifacts,
        )
        server_sha256 = item.get("server_sha256")
        if not isinstance(server_sha256, str) or not SHA256_RE.fullmatch(
            server_sha256.strip().lower()
        ):
            raise RuntimeBuildError(
                f"server_sha256 must be a lowercase 64-character SHA-256 for build {build_id!r}"
            )
        server_sha256 = server_sha256.strip().lower()
        if not allow_missing_artifacts and not os.access(server_binary, os.X_OK):
            raise RuntimeBuildError(
                f"server_binary must be executable for build {build_id!r}: {server_binary}"
            )

        bundle_root: Path | None = None
        if item.get("bundle_root") is not None:
            bundle_root = _absolute_path(
                item["bundle_root"],
                "bundle_root",
                allow_missing=allow_missing_artifacts,
                directory=True,
            )
            if bundle_root.exists() and bundle_root not in server_binary.parents:
                raise RuntimeBuildError(
                    f"server_binary for build {build_id!r} is outside bundle_root"
                )

        raw_environment = item.get("environment")
        if "launch_environment" in item:
            launch_environment = item.get("launch_environment")
            if raw_environment is not None and raw_environment != launch_environment:
                raise RuntimeBuildError(
                    f"environment and launch_environment disagree for build {build_id!r}"
                )
            raw_environment = launch_environment
        environment = _string_mapping(raw_environment, "environment")

        source_repository = (
            _absolute_path(
                item["source_repository"],
                "source_repository",
                allow_missing=True,
                directory=True,
            )
            if item.get("source_repository") is not None
            else None
        )
        provenance_file = (
            _absolute_path(
                item["provenance_file"],
                "provenance_file",
                allow_missing=True,
            )
            if item.get("provenance_file") is not None
            else None
        )

        source_branch = _optional_text(item, "source_branch")
        source_commit = _optional_text(item, "source_commit")
        dirty = item.get("dirty")
        if dirty is not None and not isinstance(dirty, bool):
            raise RuntimeBuildError(f"dirty must be boolean for build {build_id!r}")

        selected = item.get("selected_profile_ids", [])
        if not isinstance(selected, list) or not all(
            isinstance(value, str) and value.strip() for value in selected
        ):
            raise RuntimeBuildError(
                f"selected_profile_ids must be a string list for build {build_id!r}"
            )
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list) or not all(
            isinstance(value, str) and SAFE_ID_RE.fullmatch(value.strip())
            for value in aliases
        ):
            raise RuntimeBuildError(
                f"aliases must be a safe string list for build {build_id!r}"
            )
        build_options = item.get("build_options", {})
        if not isinstance(build_options, dict):
            raise RuntimeBuildError(
                f"build_options must be an object for build {build_id!r}"
            )
        deprecated = item.get("deprecated", False)
        if not isinstance(deprecated, bool):
            raise RuntimeBuildError(
                f"deprecated must be boolean for build {build_id!r}"
            )

        builds.append(
            RuntimeBuild(
                id=build_id,
                runtime=runtime,
                display_name=display_name.strip(),
                server_binary=server_binary,
                server_sha256=server_sha256,
                bundle_root=bundle_root,
                environment=environment,
                source_repository=source_repository,
                provenance_file=provenance_file,
                source_branch=source_branch,
                source_commit=source_commit,
                dirty=dirty,
                selected_profile_ids=tuple(dict.fromkeys(value.strip() for value in selected)),
                build_options=MappingProxyType(copy.deepcopy(build_options)),
                aliases=tuple(dict.fromkeys(value.strip() for value in aliases)),
                deprecated=deprecated,
            )
        )
    return tuple(builds)


def _catalog_path(path: Path | str | None) -> Path:
    if path is None:
        value = os.environ.get("LLAMA_BROKER_RUNTIME_CATALOG")
        catalog_path = Path(value).expanduser() if value else RUNTIME_CATALOG_PATH
    else:
        catalog_path = Path(path).expanduser()
    if not catalog_path.is_absolute():
        raise RuntimeBuildError(f"runtime catalog must be an absolute path: {catalog_path}")
    return catalog_path.resolve()


def load_runtime_catalog(
    path: Path | str | None = None,
    *,
    allow_missing_artifacts: bool = False,
) -> RuntimeCatalog:
    """Load the external runtime authority and validate its revision metadata."""
    catalog_path = _catalog_path(path)
    try:
        decoded = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeBuildError(
            f"could not read runtime catalog {catalog_path}: {error}"
        ) from error
    if not isinstance(decoded, dict):
        raise RuntimeBuildError("runtime catalog must contain a JSON object")

    schema = decoded.get("schema_version", 1)
    if isinstance(schema, bool) or not isinstance(schema, int) or schema not in {1, 2}:
        raise RuntimeBuildError(f"unsupported runtime catalog schema_version: {schema!r}")
    revision = decoded.get("deployment_revision", "")
    if schema >= 2 and (not isinstance(revision, str) or not revision.strip()):
        raise RuntimeBuildError("runtime catalog requires deployment_revision")
    if not isinstance(revision, str):
        raise RuntimeBuildError("deployment_revision must be a string")

    raw_builds = decoded.get("runtimes")
    if raw_builds is None:
        raw_builds = decoded.get("runtime_builds")
    builds = parse_runtime_builds(
        raw_builds,
        allow_missing_artifacts=allow_missing_artifacts,
    )
    by_id = {build.id.casefold(): build.id for build in builds}

    raw_aliases = decoded.get("temporary_aliases", {})
    if not isinstance(raw_aliases, dict) or not all(
        isinstance(old, str) and isinstance(new, str)
        for old, new in raw_aliases.items()
    ):
        raise RuntimeBuildError("temporary_aliases must map strings to runtime IDs")
    aliases: dict[str, str] = {}
    for old, new in raw_aliases.items():
        old_key = old.strip()
        new_key = new.strip().casefold()
        if not old_key or not SAFE_ID_RE.fullmatch(old_key):
            raise RuntimeBuildError(f"invalid temporary runtime alias: {old!r}")
        if new_key not in by_id:
            raise RuntimeBuildError(
                f"temporary runtime alias points to unknown build: {new!r}"
            )
        aliases[old_key] = by_id[new_key]

    return RuntimeCatalog(
        schema_version=schema,
        deployment_revision=revision.strip(),
        builds=builds,
        temporary_aliases=MappingProxyType(aliases),
        path=catalog_path,
    )


def external_catalog_for_manifest(
    manifest: Mapping[str, Any],
    *,
    catalog_path: Path | str | None = None,
    allow_missing_artifacts: bool = False,
) -> RuntimeCatalog | None:
    """Return the external catalog for v2, or ``None`` for a v1 manifest."""
    schema = manifest.get("schema_version")
    if schema != 2:
        return None
    revision = manifest.get("deployment_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise RuntimeBuildError("schema-v2 profile manifest requires deployment_revision")
    catalog = load_runtime_catalog(
        catalog_path,
        allow_missing_artifacts=allow_missing_artifacts,
    )
    if catalog.deployment_revision != revision.strip():
        raise RuntimeBuildError(
            "deployment_revision mismatch between profile manifest and runtime catalog"
        )
    return catalog


def catalog_from_manifest(
    manifest: Mapping[str, Any],
    *,
    allow_missing_artifacts: bool = False,
    catalog_path: Path | str | None = None,
) -> tuple[RuntimeBuild, ...]:
    catalog = external_catalog_for_manifest(
        manifest,
        catalog_path=catalog_path,
        allow_missing_artifacts=allow_missing_artifacts,
    )
    if catalog is not None:
        return catalog.builds
    return parse_runtime_builds(
        manifest.get("runtime_builds", []),
        allow_missing_artifacts=allow_missing_artifacts,
    )


def runtime_build_label(build: RuntimeBuild) -> str:
    return f"{build.display_name} ({build.id})"


def serving_backend_label(build: RuntimeBuild) -> str:
    return f"{build.display_name} ({build.runtime})"


def _build_matches(build: RuntimeBuild, wanted: str) -> bool:
    return wanted in {
        build.id.casefold(),
        build.display_name.casefold(),
        runtime_build_label(build).casefold(),
        serving_backend_label(build).casefold(),
        *(alias.casefold() for alias in build.aliases),
    }


def resolve_runtime_build(
    manifest: Mapping[str, Any],
    build_id: str,
    *,
    allow_missing_artifacts: bool = False,
    catalog_path: Path | str | None = None,
) -> RuntimeBuild:
    wanted = str(build_id).strip().casefold()
    catalog = external_catalog_for_manifest(
        manifest,
        catalog_path=catalog_path,
        allow_missing_artifacts=allow_missing_artifacts,
    )
    builds = catalog.builds if catalog is not None else catalog_from_manifest(
        manifest,
        allow_missing_artifacts=allow_missing_artifacts,
    )
    if catalog is not None:
        alias_target = next(
            (
                target
                for old, target in catalog.temporary_aliases.items()
                if old.casefold() == wanted
            ),
            None,
        )
        if alias_target is not None:
            wanted = alias_target.casefold()
    for build in builds:
        if _build_matches(build, wanted):
            return build
    raise RuntimeBuildError(f"unknown runtime build: {build_id}")


def resolve_profile_runtime(
    manifest: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    allow_missing_artifacts: bool = False,
    catalog_path: Path | str | None = None,
) -> tuple[dict[str, Any], RuntimeBuild | None]:
    """Derive v2 runtime launch fields without changing the source manifest."""
    updated = copy.deepcopy(dict(profile))
    requested = updated.get("runtime_build")
    if manifest.get("schema_version") == 2 and (
        not isinstance(requested, str) or not requested.strip()
    ):
        raise RuntimeBuildError("schema-v2 profile requires runtime_build")
    if not isinstance(requested, str) or not requested.strip():
        return updated, None

    build = resolve_runtime_build(
        manifest,
        requested,
        allow_missing_artifacts=allow_missing_artifacts,
        catalog_path=catalog_path,
    )
    explicit_runtime = updated.get("runtime")
    if explicit_runtime is not None and str(explicit_runtime).strip().lower() != build.runtime:
        raise RuntimeBuildError(
            f"runtime field does not match runtime build {build.id!r}"
        )
    explicit_binary = updated.get("server_binary")
    if explicit_binary is not None:
        try:
            explicit_path = Path(str(explicit_binary)).expanduser().resolve()
        except OSError as error:
            raise RuntimeBuildError("invalid explicit server_binary") from error
        if explicit_path != build.server_binary:
            raise RuntimeBuildError(
                f"server_binary does not match runtime build {build.id!r}"
            )
    explicit_hash = updated.get("server_sha256")
    if explicit_hash is not None and str(explicit_hash).strip().lower() != build.server_sha256:
        raise RuntimeBuildError(
            f"server_sha256 does not match runtime build {build.id!r}"
        )

    merged_environment = dict(build.environment)
    raw_profile_environment = updated.get("environment", {})
    if not isinstance(raw_profile_environment, dict):
        raise RuntimeBuildError("profile environment must be an object")
    merged_environment.update(raw_profile_environment)
    updated.update(
        {
            "runtime": build.runtime,
            "runtime_build": build.id,
            "server_binary": str(build.server_binary),
            "server_sha256": build.server_sha256,
            "environment": merged_environment,
        }
    )
    return updated, build


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
    raw_runtime = _profile_value(profile, "runtime")
    runtime = str(raw_runtime).strip().lower() if raw_runtime else None
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
        if (runtime is None or build.runtime == runtime)
        and build.server_binary == binary_path.resolve()
        and build.server_sha256 == str(server_sha256).strip().lower()
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
        runtime if runtime is not None else profile.get("runtime", build.runtime)
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
