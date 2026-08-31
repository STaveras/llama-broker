#!/usr/bin/env python3
"""Shared, fail-closed profile configuration primitives.

The broker manifest remains the only execution authority.  This module keeps
the terminal wizard and the curses UI on the same projection, validation, and
atomic-apply path so they cannot silently grow different profile semantics.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .broker_profiles import ProfileError, ProfileRegistry
from .manifest_resolution import resolve_manifest_inheritance
from .runtimes import get_runtime_info


SIMPLE_FIELDS: tuple[str, ...] = (
    "runtime",
    "display_name",
    "alias",
    "role",
    "capabilities",
    "context_size",
    "parallel",
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "kv_cache_dtype",
    "tool_call_parser",
    "tool_parser_plugin",
    "chat_template",
    "limit_mm_per_prompt",
    "allowed_local_media_path",
    "batch_size",
    "ubatch_size",
    "cache_type_k",
    "cache_type_v",
    "flash_attention",
    "gpu_layers",
    "mmproj_path",
)

ADVANCED_FIELDS: tuple[str, ...] = (
    "server_binary",
    "model_path",
    "model_sha256",
    "server_sha256",
    "mmproj_sha256",
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "kv_cache_dtype",
    "tool_call_parser",
    "tool_parser_plugin",
    "chat_template",
    "limit_mm_per_prompt",
    "allowed_local_media_path",
    "fit",
    "fit_target_mib",
    "fit_ctx",
    "ctx_checkpoints",
    "cache_ram_mib",
    "startup_timeout",
    "safety",
    "environment",
    "extra_args",
    "model_draft",
)

_GGUF_TYPES = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}


class ApplyError(RuntimeError):
    """A manifest update could not be committed safely."""


@dataclass(frozen=True)
class ApplyResult:
    changed: bool
    restarted: bool
    rolled_back: bool = False


def _legacy_manifest_candidate(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        return None
    ids: set[str] = set()
    for record in profiles:
        if not isinstance(record, dict):
            return None
        profile_id = record.get("id")
        if not isinstance(profile_id, str) or not profile_id.strip() or profile_id in ids:
            return None
        ids.add(profile_id)
        # Requiring both artifact identities avoids guessing how an old
        # manifest should be pinned during automatic migration.
        for field in ("alias", "server_binary", "model_path", "server_sha256", "model_sha256"):
            if not record.get(field):
                return None
    migrated = copy.deepcopy(dict(payload))
    migrated["schema_version"] = 1
    migrated.pop("default_profile", None)
    return migrated


def _create_migration_backup(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    suffix = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.bak-{stamp}-{suffix}")
        suffix += 1
    shutil.copy2(path, backup)
    with backup.open("rb") as handle:
        os.fsync(handle.fileno())
    return backup


def _load_manifest_payload(path: Path | str) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"cannot read profile manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ProfileError("profile manifest must be a JSON object")
    schema = payload.get("schema_version")
    if schema != 1:
        if schema not in (None, 0):
            raise ProfileError(f"unsupported profile manifest schema_version: {schema!r}")
        migrated = _legacy_manifest_candidate(payload)
        if migrated is None:
            raise ProfileError("profile manifest is ambiguous or unsupported for automatic migration")
        _create_migration_backup(manifest_path)
        try:
            write_manifest_atomically(manifest_path, migrated)
        except Exception as error:
            raise ProfileError(f"profile manifest migration failed: {error}") from error
        payload = migrated
    payload = copy.deepcopy(payload)
    payload.pop("default_profile", None)
    return manifest_path, payload


def load_manifest_source(
    path: Path | str, *, allow_missing_artifacts: bool = False
) -> dict[str, Any]:
    """Load the persisted manifest while preserving explicit inheritance links."""

    manifest_path, payload = _load_manifest_payload(path)
    try:
        resolve_manifest_inheritance(payload)
        validate_manifest(
            payload,
            directory=manifest_path.parent,
            allow_missing_artifacts=allow_missing_artifacts,
        )
    except ValueError as error:
        raise ProfileError(str(error)) from error
    return payload


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Load the effective manifest with local inheritance fully resolved."""

    _manifest_path, payload = _load_manifest_payload(path)
    try:
        return resolve_manifest_inheritance(payload)
    except ValueError as error:
        raise ProfileError(str(error)) from error


def find_profile_record(
    manifest: Mapping[str, Any], selector: str
) -> tuple[int, dict[str, Any]] | None:
    """Find a profile by id, alias, or alias entry (case-insensitive)."""

    wanted = str(selector).strip().casefold()
    if not wanted:
        return None
    raw_profiles = manifest.get("profiles", [])
    if not isinstance(raw_profiles, list):
        return None
    for index, raw in enumerate(raw_profiles):
        if not isinstance(raw, dict):
            continue
        names = [raw.get("id"), raw.get("alias"), *(raw.get("aliases") or [])]
        if any(str(name).strip().casefold() == wanted for name in names if name):
            return index, raw
    return None


def clone_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(record))


def simple_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the user-facing serving fields without exposing safety internals."""

    return {field: copy.deepcopy(record[field]) for field in SIMPLE_FIELDS if field in record}


def apply_simple_fields(record: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Project typed simple-screen values onto a profile draft."""

    selected_runtime = str(
        values.get("runtime", record.get("runtime", "llamacpp"))
    ).strip().lower()
    runtime_info = get_runtime_info(selected_runtime)
    unsupported = sorted(
        field
        for field in values
        if field not in runtime_info.editable_fields
    )
    if unsupported:
        raise ProfileError(
            f"runtime {selected_runtime!r} does not support edit fields: "
            + ", ".join(unsupported)
        )

    draft = clone_record(record)
    for field in SIMPLE_FIELDS:
        if field not in values:
            continue
        value = values[field]
        if field == "capabilities":
            if isinstance(value, str):
                value = [item.strip() for item in value.split(",") if item.strip()]
            value = list(value)
        elif field == "mmproj_path":
            value = str(value).strip() or None
        elif field in {
            "context_size",
            "parallel",
            "batch_size",
            "ubatch_size",
            "tensor_parallel_size",
        }:
            value = int(value)
        elif field == "gpu_memory_utilization":
            value = float(value)
        draft[field] = copy.deepcopy(value)
    if str(draft.get("runtime", "llamacpp")).strip().lower() == "vllm":
        draft.pop("mmproj_path", None)
        draft.pop("mmproj_sha256", None)
    elif draft.get("mmproj_path"):
        capabilities = set(draft.get("capabilities", ["text"]))
        capabilities.add("vision")
        draft["capabilities"] = sorted(capabilities)
        projector = Path(str(draft["mmproj_path"])).expanduser()
        if projector.exists() and projector.is_file():
            draft["mmproj_path"] = str(projector.resolve())
            draft["mmproj_sha256"] = sha256_file(projector)
    else:
        draft.pop("mmproj_path", None)
        draft.pop("mmproj_sha256", None)
        if not draft.get("native_vision"):
            capabilities = set(draft.get("capabilities", ["text"]))
            capabilities.discard("vision")
            capabilities.add("text")
            draft["capabilities"] = sorted(capabilities)
    return draft


def semantic_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """Describe top-level manifest changes in stable, human-readable order."""

    changes: list[str] = []
    keys = sorted(set(before) | set(after))
    for key in keys:
        if before.get(key) != after.get(key):
            changes.append(key)
    return changes


def merge_manifest_drafts(
    base: Mapping[str, Any],
    disk: Mapping[str, Any],
    draft: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Three-way merge a draft with changes made on disk since it was opened.

    Draft edits win true conflicts.  The returned paths are intended for a
    confirmation prompt; callers must still run normal manifest validation.
    """

    conflicts: list[str] = []

    def merge_value(old: Any, current: Any, wanted: Any, path: str) -> Any:
        if current == old:
            return copy.deepcopy(wanted)
        if wanted == old or current == wanted:
            return copy.deepcopy(current)
        if isinstance(old, Mapping) and isinstance(current, Mapping) and isinstance(wanted, Mapping):
            result: dict[str, Any] = {}
            for key in sorted(set(old) | set(current) | set(wanted)):
                child_path = f"{path}.{key}" if path else str(key)
                if key not in wanted:
                    if key in old and key in current and current[key] != old[key]:
                        result[key] = copy.deepcopy(current[key])
                    continue
                if key not in current:
                    if key in old and wanted[key] != old[key]:
                        conflicts.append(child_path)
                        result[key] = copy.deepcopy(wanted[key])
                    continue
                result[key] = merge_value(
                    old.get(key), current[key], wanted[key], child_path
                )
            return result
        conflicts.append(path or "manifest")
        return copy.deepcopy(wanted)

    def merge_records(key: str) -> list[Any] | None:
        old_items = base.get(key)
        current_items = disk.get(key)
        wanted_items = draft.get(key)
        if not isinstance(old_items, list) or not isinstance(current_items, list) or not isinstance(wanted_items, list):
            return None
        old_by_id = {item.get("id"): item for item in old_items if isinstance(item, Mapping) and item.get("id")}
        current_by_id = {item.get("id"): item for item in current_items if isinstance(item, Mapping) and item.get("id")}
        wanted_by_id = {item.get("id"): item for item in wanted_items if isinstance(item, Mapping) and item.get("id")}
        ordered_ids = list(current_by_id)
        ordered_ids.extend(item_id for item_id in wanted_by_id if item_id not in current_by_id)
        result: list[Any] = []
        for item_id in ordered_ids:
            child_path = f"{key}[{item_id}]"
            if item_id not in wanted_by_id:
                if item_id not in old_by_id:
                    result.append(copy.deepcopy(current_by_id[item_id]))
                elif current_by_id.get(item_id) == old_by_id[item_id]:
                    continue
                else:
                    conflicts.append(child_path)
                    result.append(copy.deepcopy(current_by_id[item_id]))
                continue
            if item_id not in current_by_id:
                if item_id not in old_by_id:
                    result.append(copy.deepcopy(wanted_by_id[item_id]))
                else:
                    conflicts.append(child_path)
                    result.append(copy.deepcopy(wanted_by_id[item_id]))
                continue
            result.append(
                merge_value(
                    old_by_id.get(item_id),
                    current_by_id[item_id],
                    wanted_by_id[item_id],
                    child_path,
                )
            )
        return result

    merged: dict[str, Any] = {}
    for key in sorted(set(base) | set(disk) | set(draft)):
        if key in {"profiles", "runtime_builds"}:
            records = merge_records(key)
            if records is not None:
                merged[key] = records
                continue
        merged[key] = merge_value(base.get(key), disk.get(key), draft.get(key), key)
    return merged, conflicts


def changed_profile_ids(before: Mapping[str, Any], after: Mapping[str, Any]) -> set[str]:
    def by_id(payload: Mapping[str, Any]) -> dict[str, Any]:
        raw = payload.get("profiles", [])
        return {
            str(item.get("id")): item
            for item in raw
            if isinstance(item, dict) and item.get("id")
        }

    old = by_id(before)
    new = by_id(after)
    return {profile_id for profile_id in set(old) | set(new) if old.get(profile_id) != new.get(profile_id)}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    target = Path(path).expanduser()
    if target.is_file():
        with target.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    if target.is_dir():
        for child in sorted(item for item in target.rglob("*") if item.is_file()):
            digest.update(child.relative_to(target).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()
    raise FileNotFoundError(target)


def _read_gguf_string(handle) -> str:
    raw_length = handle.read(8)
    if len(raw_length) != 8:
        raise ValueError("truncated GGUF string length")
    length = struct.unpack("<Q", raw_length)[0]
    if length > 1 << 20:
        raise ValueError("unreasonable GGUF string length")
    raw = handle.read(length)
    if len(raw) != length:
        raise ValueError("truncated GGUF string")
    return raw.decode("utf-8", errors="replace")


def _read_gguf_value(handle, value_type: int) -> object | None:
    if value_type == 8:
        return _read_gguf_string(handle)
    if value_type == 9:
        type_raw = handle.read(4)
        count_raw = handle.read(8)
        if len(type_raw) != 4 or len(count_raw) != 8:
            raise ValueError("truncated GGUF array")
        element_type = struct.unpack("<I", type_raw)[0]
        count = struct.unpack("<Q", count_raw)[0]
        if count > 1 << 24:
            raise ValueError("unreasonable GGUF array length")
        for _ in range(count):
            _read_gguf_value(handle, element_type)
        return None
    spec = _GGUF_TYPES.get(value_type)
    if spec is None:
        raise ValueError(f"unsupported GGUF metadata type {value_type}")
    fmt, size = spec
    raw = handle.read(size)
    if len(raw) != size:
        raise ValueError("truncated GGUF metadata value")
    return struct.unpack("<" + fmt, raw)[0]


def read_gguf_metadata(path: Path | str) -> dict[str, object]:
    """Read the small GGUF metadata header, never tensor data."""

    with Path(path).expanduser().open("rb") as handle:
        header = handle.read(24)
        if len(header) != 24 or header[:4] != b"GGUF":
            return {}
        _, version, _, metadata_count = struct.unpack("<4sIQQ", header)
        if version not in {1, 2, 3} or metadata_count > 1 << 20:
            return {}
        metadata: dict[str, object] = {}
        for _ in range(metadata_count):
            key = _read_gguf_string(handle)
            raw_type = handle.read(4)
            if len(raw_type) != 4:
                return {}
            value = _read_gguf_value(handle, struct.unpack("<I", raw_type)[0])
            if key in {
                "general.name",
                "general.basename",
                "general.architecture",
                "general.context_length",
                "llama.context_length",
            } or key.endswith(".context_length"):
                metadata[key] = value
        return metadata


def gguf_display_name(path: Path | str) -> tuple[str, dict[str, object]]:
    model_path = Path(path).expanduser()
    try:
        metadata = read_gguf_metadata(model_path)
    except (OSError, ValueError, struct.error):
        metadata = {}
    name = str(
        metadata.get("general.name")
        or metadata.get("general.basename")
        or model_path.stem
    ).strip()
    return name, metadata


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "model"


def discover_local_ggufs(
    roots: Sequence[Path | str] | None = None,
    *,
    max_results: int = 256,
) -> list[Path]:
    """Find local GGUF files for the manual-import branch of the wizard."""

    if max_results <= 0:
        return []
    configured = roots or (
        os.environ.get("MODELS_DIR", str(Path.home() / ".local/share/models")),
        str(Path.home() / "models"),
    )
    found: list[Path] = []
    seen: set[Path] = set()
    for root_value in configured:
        root = Path(root_value).expanduser()
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*.gguf")
        try:
            for candidate in candidates:
                path = candidate.resolve()
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                found.append(path)
                if len(found) >= max_results:
                    return sorted(found)
        except OSError:
            continue
    return sorted(found)


def _template_record(manifest: Mapping[str, Any], selector: str | None = None) -> dict[str, Any]:
    if selector:
        found = find_profile_record(manifest, selector)
        if found:
            return clone_record(found[1])
    raw_profiles = manifest.get("profiles", [])
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ProfileError("cannot create a profile without a template profile")
    template = next(
        (item for item in raw_profiles if isinstance(item, dict)),
        None,
    )
    if template is None:
        raise ProfileError("profile template is not an object")
    return clone_record(template)


def _unique_id(manifest: Mapping[str, Any], proposed: str) -> str:
    existing = {
        str(item.get("id"))
        for item in manifest.get("profiles", [])
        if isinstance(item, dict)
    }
    base = slugify(proposed)
    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _prepare_record(
    manifest: Mapping[str, Any],
    model_path: Path,
    *,
    template_selector: str | None = None,
    profile_id: str | None = None,
    alias: str | None = None,
    context_size: int | None = None,
    parallel: int | None = None,
    server_binary: Path | None = None,
    mmproj_path: Path | None = None,
) -> dict[str, Any]:
    template = _template_record(manifest, template_selector)
    name, metadata = gguf_display_name(model_path)
    generated_alias = slugify(alias or name)
    generated_id = _unique_id(manifest, profile_id or generated_alias)
    if profile_id and find_profile_record(manifest, profile_id) is None:
        generated_id = slugify(profile_id)
    runtime = server_binary or Path(str(template["server_binary"])).expanduser()
    record = template
    record.update(
        {
            "id": generated_id,
            "alias": generated_alias,
            "aliases": [generated_alias],
            "display_name": name,
            "role": "persistent",
            "capabilities": ["text"],
            "native_vision": False,
            "server_binary": str(runtime.resolve()),
            "server_sha256": sha256_file(runtime),
            "model_path": str(model_path.resolve()),
            "model_sha256": sha256_file(model_path),
            "context_size": int(
                context_size
                or metadata.get("general.context_length")
                or template.get("context_size", 32768)
            ),
            "parallel": int(parallel or 1),
        }
    )
    record.pop("restore_profile", None)
    for key in ("fit", "fit_target_mib", "fit_ctx", "ctx_checkpoints", "cache_ram_mib", "model_draft"):
        record.pop(key, None)
    if mmproj_path:
        record["capabilities"] = ["text", "vision"]
        record["mmproj_path"] = str(mmproj_path.resolve())
        record["mmproj_sha256"] = sha256_file(mmproj_path)
    else:
        record.pop("mmproj_path", None)
        record.pop("mmproj_sha256", None)
    environment = dict(record.get("environment", {}))
    if record["parallel"] > 1 and (
        str(record.get("cache_type_k", "")).startswith("kvarn")
        or str(record.get("cache_type_v", "")).startswith("kvarn")
    ):
        environment["LLAMA_KVARN_MULTI_STREAM"] = "1"
    record["environment"] = environment
    return record


def draft_from_gguf(
    manifest: Mapping[str, Any],
    model_path: Path | str,
    **kwargs: Any,
) -> dict[str, Any]:
    return _prepare_record(manifest, Path(model_path).expanduser(), **kwargs)


def draft_from_endpoint(
    manifest: Mapping[str, Any],
    endpoint: Any,
    model: Any,
    *,
    model_path: Path | str | None = None,
    server_binary: Path | str | None = None,
    mmproj_path: Path | str | None = None,
    template_selector: str | None = None,
) -> dict[str, Any]:
    """Turn discovery hints into a broker-owned draft.

    Discovery never becomes a passthrough backend: the caller must supply a
    local GGUF and (unless the template is retained) a broker-owned runtime.
    """

    if model_path is None:
        raise ProfileError(
            "endpoint adoption requires a local GGUF path; discovery metadata cannot be execution authority"
        )
    context = getattr(model, "context_length", None) or getattr(endpoint, "n_ctx", None)
    parallel = getattr(model, "parallel", None) or getattr(endpoint, "total_slots", None) or 1
    return _prepare_record(
        manifest,
        Path(model_path).expanduser(),
        template_selector=template_selector,
        profile_id=str(getattr(model, "id", "model")),
        alias=str(getattr(model, "id", "model")),
        context_size=context,
        parallel=parallel,
        server_binary=Path(server_binary).expanduser() if server_binary else None,
        mmproj_path=Path(mmproj_path).expanduser() if mmproj_path else None,
    )


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    directory: Path | None = None,
    allow_missing_artifacts: bool = False,
) -> None:
    """Run the exact production registry validation against an in-memory draft."""

    parent = directory or Path.cwd()
    parent.mkdir(parents=True, exist_ok=True)
    try:
        effective_manifest = resolve_manifest_inheritance(manifest)
    except ValueError as error:
        raise ProfileError(str(error)) from error
    descriptor, temporary_name = tempfile.mkstemp(prefix=".profiles.validate.", suffix=".json", dir=parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(effective_manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        ProfileRegistry.from_path(
            temporary_path, allow_missing_artifacts=allow_missing_artifacts
        )
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def write_manifest_atomically(path: Path | str, manifest: Mapping[str, Any]) -> None:
    """Validate, fsync, and atomically publish a manifest replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_manifest(manifest, directory=destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary_path, destination.stat().st_mode & 0o777)
        except FileNotFoundError:
            os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def apply_manifest_transaction(
    profiles_path: Path | str,
    new_manifest: Mapping[str, Any],
    *,
    active_profile: str | None,
    install_units: Callable[[], Any],
    restart_broker: Callable[[], Any],
    writer: Callable[[Path, Mapping[str, Any]], None] = write_manifest_atomically,
    reload_broker: Callable[[], Any] | None = None,
) -> ApplyResult:
    """Publish a profile edit and refresh the broker registry without restarting the backend."""

    path = Path(profiles_path)
    old_manifest = load_manifest_source(path)
    if old_manifest == new_manifest:
        return ApplyResult(changed=False, restarted=False)
    validate_manifest(new_manifest, directory=path.parent)
    writer(path, new_manifest)
    reload_attempted = False
    try:
        install_units()
        if reload_broker is not None:
            reload_attempted = True
            reload_broker()
        return ApplyResult(changed=True, restarted=False)
    except Exception as error:
        rollback_errors: list[str] = []
        try:
            writer(path, old_manifest)
            install_units()
            if reload_broker is not None and reload_attempted:
                reload_broker()
        except Exception as rollback_error:
            rollback_errors.append(str(rollback_error))
        detail = f"profile apply failed: {error}"
        if rollback_errors:
            detail += "; rollback also failed: " + "; ".join(rollback_errors)
        raise ApplyError(detail) from error
