#!/usr/bin/env python3
"""Safe, resumable migration of broker-served model and runtime artifacts."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import ssl
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from llama_broker.paths import (  # noqa: E402
    CACHE_DIR as DEFAULT_CACHE_DIR,
    MODELS_DIR as DEFAULT_MODELS_DIR,
    RUNTIMES_DIR as DEFAULT_RUNTIMES_DIR,
    RUNTIME_CATALOG_PATH as DEFAULT_CATALOG_PATH,
    RUNTIME_PROVENANCE_INDEX_PATH as DEFAULT_PROVENANCE_INDEX_PATH,
)
from llama_broker.profile_config import read_gguf_metadata  # noqa: E402


PLAN_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin", ".pt", ".pth"})


class MigrationError(RuntimeError):
    """A migration safety gate failed."""


# Deployment exceptions are supplied explicitly, never built into public defaults.
MODEL_TARGET_OVERRIDES: dict[str, str] = {}

RUNTIME_PREFIXES = {
    "beellama": "beellama-kvarn",
    "atomic": "atomic-qat",
    "thecodacus-perf-d927e7d": "thecodacus-perf",
    "llamacpp-54ec2d12ae3a": "adaptive-kv",
    "llamacpp-8ed7c9a37a7b": "upstream-adaptive",
    "llamacpp-e7dc9e7d5809": "adaptive-kv-windows",
}

RUNTIME_REPOSITORY_OVERRIDES = {}

CANONICAL_RUNTIME_REPOSITORIES = {}


# ---------------------------------------------------------------------------
# Small durable primitives


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    target = Path(path).expanduser()
    if not target.is_file():
        raise MigrationError(f"artifact is not a regular file: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MigrationError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(decoded, dict):
        raise MigrationError(f"JSON file must contain an object: {path}")
    return decoded


def _resolve_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise MigrationError(f"source artifact is unavailable: {candidate}: {error}") from error
    if not resolved.is_file():
        raise MigrationError(f"source artifact is not a regular file: {resolved}")
    return resolved


def _safe_relative(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise MigrationError(f"path escapes migration root: {path} (root {root})") from error
    return relative.as_posix()


def _copy_verified(source: Path, target: Path, expected_sha256: str) -> bool:
    """Atomically copy one file; return True when a write was needed."""
    source_sha = sha256_file(source)
    if source_sha != expected_sha256:
        raise MigrationError(
            f"source hash changed before copy: {source} expected {expected_sha256} got {source_sha}"
        )
    if target.is_file() and not target.is_symlink():
        if sha256_file(target) == expected_sha256:
            return False
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".stage", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination, source.open("rb") as origin:
            while chunk := origin.read(16 * 1024 * 1024):
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(temporary, stat.S_IMODE(source.stat().st_mode) | 0o600)
        if sha256_file(temporary) != expected_sha256:
            raise MigrationError(f"staged hash mismatch: {temporary}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _plan_digest(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("plan_sha256", None)
    return _sha256_bytes(_json_bytes(body))


def _load_plan(path: Path | str) -> dict[str, Any]:
    plan_path = Path(path).expanduser().resolve()
    payload = _load_json(plan_path)
    expected = payload.get("plan_sha256")
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        raise MigrationError("migration plan has no valid plan_sha256")
    actual = _plan_digest(payload)
    if actual != expected:
        raise MigrationError(
            f"plan hash mismatch: expected {expected}, got {actual}"
        )
    payload["_plan_path"] = str(plan_path)
    return payload


def _write_plan(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(payload)
    output["plan_sha256"] = _plan_digest(output)
    _atomic_write(path, output)
    return output


def _mode(path: Path) -> str:
    return oct(stat.S_IMODE(path.lstat().st_mode))


def _git_text(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _compact_file_fingerprint(path: Path) -> bytes:
    size = path.stat().st_size
    if size <= 64 * 1024 * 1024:
        return bytes.fromhex(sha256_file(path))
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if size > 1024 * 1024:
            handle.seek(max(0, size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    return digest.digest()


def _repository_snapshot(repo: Path | None) -> dict[str, Any] | None:
    if repo is None or not ((repo / ".git").exists() or (repo / ".git").is_file()):
        return None
    changed = [
        path
        for path in _git_text(repo, "diff", "--name-only").splitlines()
        if path != ".runtime-provenance.json"
    ]
    untracked = [
        path
        for path in _git_text(repo, "ls-files", "--others", "--exclude-standard").splitlines()
        if path != ".runtime-provenance.json"
    ]
    dirty_paths = list(dict.fromkeys([*changed, *untracked]))
    digest = hashlib.sha256()
    for relative in sorted(dirty_paths):
        path = repo / relative
        digest.update(relative.encode("utf-8", "surrogateescape"))
        digest.update(b"\\0")
        try:
            if path.is_file():
                digest.update(_compact_file_fingerprint(path))
            else:
                digest.update(str(path.lstat().st_size).encode("ascii"))
        except OSError:
            digest.update(b"<unreadable>")
        digest.update(b"\\0")
    return {
        "path": str(repo),
        "branch": _git_text(repo, "branch", "--show-current"),
        "head": _git_text(repo, "rev-parse", "HEAD"),
        "dirty": bool(dirty_paths),
        "changed_paths": changed,
        "untracked_paths": untracked,
        "dirty_content_hash": digest.hexdigest(),
    }


# ---------------------------------------------------------------------------
# Model naming and inventory


def _family_and_parameters(name: str) -> tuple[str, str] | None:
    lowered = name.casefold()
    family_match = re.search(r"(qwen[-_]?3[._-]?[568]|gemma[-_]?4)", lowered)
    if not family_match:
        return None
    raw_family = family_match.group(1).replace("_", ".").replace("-", ".")
    raw_family = re.sub(r"\.+", ".", raw_family)
    if raw_family.startswith("qwen"):
        family = "qwen-" + raw_family[4:]
    else:
        family = "gemma-4"
    parameter_match = re.search(
        r"(?<![a-z0-9])([0-9]+(?:[._][0-9]+)?b)(?:[-_]([a-z][0-9]+b))?",
        lowered,
    )
    if not parameter_match:
        return None
    parameters = parameter_match.group(1).replace("_", ".")
    if parameter_match.group(2):
        parameters += "-" + parameter_match.group(2)
    return family, parameters


def _quantization(name: str) -> str | None:
    lowered = name.casefold()
    match = re.search(
        r"(?:^|[-_])(ud[-_]q[0-9]+(?:[-_][a-z0-9]+)*|q[0-9]+(?:[-_][a-z0-9]+)*|iq[0-9]+(?:[-_][a-z0-9]+)*|bf16|f16|f32)(?:[-_.]|$)",
        lowered,
    )
    if not match:
        return None
    return match.group(1).replace("_", "-")


def canonical_model_target(
    source: Path | str,
    *,
    kind: str = "model",
    overrides: Mapping[str, str] | None = None,
    related_model: Path | str | None = None,
) -> str:
    """Return a deterministic family/parameter/variant target or fail closed."""
    literal = str(Path(source).expanduser())
    selected_overrides = dict(MODEL_TARGET_OVERRIDES)
    selected_overrides.update(overrides or {})
    if literal in selected_overrides:
        target = selected_overrides[literal]
        if kind == "projector" and "/mmproj-" not in target:
            target = target.rsplit("/", 1)[0] + "/mmproj-f16.gguf"
        return target

    source_path = Path(source)
    name = source_path.name
    basis = Path(related_model).name if related_model else name
    family_params = _family_and_parameters(basis)
    if family_params is None:
        family_params = _family_and_parameters(name)
    if family_params is None:
        raise MigrationError(f"ambiguous model identity; add an explicit override: {source}")
    family, parameters = family_params
    quant = _quantization(name) or _quantization(basis)
    if quant is None:
        raise MigrationError(f"ambiguous model quantization; add an explicit override: {source}")
    if kind == "projector":
        quant = "mmproj-" + quant.removeprefix("ud-")
        return f"{family}/{parameters}/{quant}.gguf"

    lowered = name.casefold()
    variant = ""
    if "antidoom" in lowered:
        variant = "-antidoom"
        version = re.search(r"antidoom[-_]?v([0-9]+)", lowered)
        if version:
            variant += f"-v{version.group(1)}"
    if "mtp" in lowered and "-mtp" not in variant:
        variant += "-mtp"
    return f"{family}/{parameters}/{quant}{variant}.gguf"


def _inventory_root(root: Path) -> list[dict[str, object]]:
    if not root.exists() and not root.is_symlink():
        return []
    candidates: list[dict[str, object]] = []
    if root.is_file():
        paths: Iterable[Path] = (root,)
    elif root.is_dir():
        paths = root.rglob("*")
    else:
        return []
    try:
        for path in paths:
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if ".git" in relative.parts or any(part.startswith(".") for part in relative.parts):
                continue
            if not path.is_file():
                continue
            if path.suffix.casefold() not in MODEL_SUFFIXES and path.name not in {
                "llama-server",
                "llama-server.exe",
            }:
                continue
            try:
                resolved = path.resolve(strict=True)
                result = resolved.stat()
            except OSError:
                continue
            basename = path.name.casefold()
            kind = (
                "runtime"
                if "llama-server" in basename
                else (
                    "deferred"
                    if basename.startswith(("ggml-vocab", "mmproj-", "mtp-"))
                    else ("serving-candidate" if path.suffix.casefold() == ".gguf" else "deferred")
                )
            )
            candidates.append(
                {
                    "path": str(resolved),
                    "kind": kind,
                    "size": result.st_size,
                    "mtime_ns": result.st_mtime_ns,
                }
            )
    except OSError:
        return candidates
    unique = {str(item["path"]): item for item in candidates}
    return [unique[key] for key in sorted(unique)]


def _profile_path_entries(
    manifest: Mapping[str, Any],
    *,
    models_dir: Path,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    profiles = manifest.get("profiles", [])
    if not isinstance(profiles, list):
        raise MigrationError("profile manifest must contain a profiles list")
    mappings: dict[tuple[str, str], dict[str, Any]] = {}
    cleanup: list[str] = []
    for raw_profile in profiles:
        if not isinstance(raw_profile, dict):
            raise MigrationError("every profile must be an object")
        model_path = raw_profile.get("model_path")
        if not isinstance(model_path, str):
            continue
        related = Path(model_path)
        try:
            related_basis = related.resolve(strict=False)
        except OSError:
            related_basis = related
        fields: list[tuple[str, str, bool]] = [("model_path", "model", True)]
        if isinstance(raw_profile.get("mmproj_path"), str):
            fields.append(("mmproj_path", "projector", True))
        if isinstance(raw_profile.get("model_draft"), str):
            fields.append(("model_draft", "draft", False))
        for field_name, kind, required in fields:
            source_literal = str(Path(str(raw_profile[field_name])).expanduser())
            source = Path(source_literal)
            try:
                resolved = source.resolve(strict=True)
                expected_hash = raw_profile.get(
                    "model_sha256" if field_name == "model_path" else (
                        "mmproj_sha256" if field_name == "mmproj_path" else "model_draft_sha256"
                    )
                )
                if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash.lower()):
                    expected_hash = sha256_file(resolved)
                else:
                    expected_hash = expected_hash.lower()
                actual_hash = sha256_file(resolved)
                if actual_hash != expected_hash and required:
                    raise MigrationError(
                        f"profile artifact hash mismatch before migration: {source_literal}"
                    )
            except MigrationError:
                if required:
                    raise
                resolved = source
                expected_hash = None
                actual_hash = None
            except OSError as error:
                if required:
                    raise MigrationError(f"required profile artifact unavailable: {source_literal}: {error}") from error
                resolved = source
                expected_hash = None
                actual_hash = None

            if kind == "draft":
                target_relative = canonical_model_target(
                    source_literal,
                    kind="model",
                    related_model=related_basis,
                )
            else:
                target_relative = canonical_model_target(
                    source_literal,
                    kind=kind,
                    related_model=related_basis,
                )
            target = models_dir / target_relative
            key = (source_literal, target_relative)
            mapping = mappings.get(key)
            if mapping is None:
                mapping = {
                    "source": source_literal,
                    "resolved_source": str(resolved),
                    "target": str(target),
                    "kind": kind,
                    "required": required,
                    "sha256": actual_hash or expected_hash,
                    "size": resolved.stat().st_size if resolved.is_file() else None,
                    "profiles": [],
                }
                mappings[key] = mapping
            mapping["profiles"].append(str(raw_profile.get("id", "<unknown>")))
            cleanup.append(source_literal)
            if str(resolved) != source_literal:
                cleanup.append(str(resolved))

        # Path-valued optional MoE support is intentionally discovered from the
        # literal profile argv and relocated into the generated cache root.
        extra_args = raw_profile.get("extra_args", [])
        if isinstance(extra_args, list):
            for index, token in enumerate(extra_args):
                if token == "--moe-cache-profile" and index + 1 < len(extra_args):
                    cache_source_literal = str(Path(str(extra_args[index + 1])).expanduser())
                    cache_source = Path(cache_source_literal)
                    try:
                        resolved_cache = cache_source.resolve(strict=True)
                        cache_hash = sha256_file(resolved_cache)
                    except OSError:
                        resolved_cache = cache_source
                        cache_hash = None
                    family_params = _family_and_parameters(related_basis.name)
                    if family_params is None:
                        raise MigrationError(
                            f"ambiguous MoE cache family; add an explicit override: {related_basis}"
                        )
                    family, parameters = family_params
                    cache_target = cache_dir / family / parameters / "moe-cache-profile.csv"
                    key = (cache_source_literal, str(cache_target))
                    mappings.setdefault(
                        key,
                        {
                            "source": cache_source_literal,
                            "resolved_source": str(resolved_cache),
                            "target": str(cache_target),
                            "kind": "moe-cache",
                            "required": False,
                            "sha256": cache_hash,
                            "size": resolved_cache.stat().st_size if resolved_cache.is_file() else None,
                            "profiles": [str(raw_profile.get("id", "<unknown>"))],
                        },
                    )
                    cleanup.append(cache_source_literal)
    return list(mappings.values()), list(dict.fromkeys(cleanup))


# ---------------------------------------------------------------------------
# Runtime bundle preparation


def _wrapper_target(path: Path) -> tuple[Path, dict[str, str]]:
    if not path.is_file():
        return path, {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return path, {}
    match = re.search(r"exec\s+([^\s]+/llama-server)\s+--n-cpu-moe\s+([0-9]+)", text)
    if not match:
        return path, {}
    return Path(match.group(1)).expanduser(), {"LLAMA_ARG_N_CPU_MOE": match.group(2)}


def _runtime_prefix(old_id: str) -> str:
    if old_id in RUNTIME_PREFIXES:
        return RUNTIME_PREFIXES[old_id]
    return re.sub(r"[^a-z0-9]+", "-", old_id.casefold()).strip("-") or "runtime"


def _bundle_file_sources(binary: Path) -> tuple[Path, list[tuple[Path, str]]]:
    real_binary, _wrapper_env = _wrapper_target(binary)
    if not real_binary.is_file():
        raise MigrationError(f"runtime executable is unavailable: {real_binary}")
    parent = real_binary.parent
    files: list[tuple[Path, str]] = [(real_binary, "bin/llama-server")]
    if parent.name == "bin" and parent.parent.name == "runtime" and (parent.parent / "lib").is_dir():
        library_root = parent.parent / "lib"
    else:
        library_root = parent
    if library_root.is_dir():
        for candidate in sorted(library_root.iterdir(), key=lambda item: item.name):
            if candidate.name.startswith("lib") and (candidate.is_file() or candidate.is_symlink()):
                try:
                    candidate.resolve(strict=True)
                except OSError:
                    continue
                files.append((candidate.resolve(), f"lib/{candidate.name}"))
    return real_binary, files


def _runtime_mapping(
    build: Mapping[str, Any],
    *,
    runtimes_dir: Path,
    profile_ids: Sequence[str],
) -> dict[str, Any]:
    old_id = str(build.get("id", "")).strip()
    old_binary = Path(str(build.get("server_binary", ""))).expanduser()
    if not old_id or not old_binary.is_absolute():
        raise MigrationError("runtime build has no absolute server_binary")
    real_binary, wrapper_env = _wrapper_target(old_binary)
    required = bool(profile_ids)
    try:
        real_binary, bundle_sources = _bundle_file_sources(old_binary)
        real_sha = sha256_file(real_binary)
    except MigrationError as error:
        if required:
            raise
        return {
            "old_id": old_id,
            "valid": False,
            "required": False,
            "warning": str(error),
        }

    declared_sha = str(build.get("server_sha256", "")).strip().lower()
    if real_binary == old_binary and declared_sha and declared_sha != real_sha:
        error = MigrationError(
            f"runtime hash mismatch for {old_id}: expected {declared_sha}, got {real_sha}"
        )
        if required:
            raise error
        return {
            "old_id": old_id,
            "valid": False,
            "required": False,
            "warning": str(error),
        }

    environment: dict[str, str] = {}
    raw_build_environment = build.get("environment", {})
    if isinstance(raw_build_environment, dict):
        environment.update(
            {
                str(key): str(value)
                for key, value in raw_build_environment.items()
                if isinstance(key, str) and isinstance(value, str)
            }
        )
    environment.update(wrapper_env)
    prefix = _runtime_prefix(old_id)
    has_libraries = any(
        relative.startswith("lib/") for _source, relative in bundle_sources
    )
    raw_source_repository = build.get("source_repository") or RUNTIME_REPOSITORY_OVERRIDES.get(old_id)
    source_repository = None
    canonical_source_repository = None
    source_snapshot = None
    if isinstance(raw_source_repository, str) and raw_source_repository.strip():
        source_repository = Path(raw_source_repository).expanduser().resolve()
        canonical_source_repository = CANONICAL_RUNTIME_REPOSITORIES.get(
            str(source_repository), str(source_repository)
        )
        source_snapshot = _repository_snapshot(source_repository)
    digest_environment = dict(environment)
    if has_libraries:
        digest_environment["LD_LIBRARY_PATH"] = "$BUNDLE_ROOT/lib"
    # The digest excludes absolute paths and timestamps. It includes normalized
    # relative contents, launch environment, and the old build identity.
    digest = hashlib.sha256()
    digest.update(old_id.encode("utf-8"))
    digest.update(b"\0")
    for source, relative in bundle_sources:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(source)))
    digest.update(
        _json_bytes(
            {
                "environment": digest_environment,
                "source_snapshot": source_snapshot,
                "build_options": build.get("build_options", {}),
            }
        )
    )
    suffix = digest.hexdigest()[:12]
    new_id = f"{prefix}-{suffix}"
    bundle_root = runtimes_dir / new_id
    if has_libraries:
        environment["LD_LIBRARY_PATH"] = str(bundle_root / "lib")
    bundle_files = [
        {
            "source": str(source),
            "target": str(bundle_root / relative),
            "relative": relative,
            "sha256": sha256_file(source),
            "size": source.stat().st_size,
        }
        for source, relative in bundle_sources
    ]
    return {
        "old_id": old_id,
        "new_id": new_id,
        "runtime": str(build.get("runtime", "llamacpp")),
        "display_name": str(build.get("display_name", old_id)),
        "source_binary": str(old_binary),
        "resolved_source_binary": str(real_binary),
        "server_sha256": real_sha,
        "old_server_sha256": str(build.get("server_sha256", "")),
        "bundle_root": str(bundle_root),
        "environment": environment,
        "bundle_files": bundle_files,
        "source_repository": str(source_repository) if source_repository else None,
        "canonical_source_repository": canonical_source_repository,
        "source_snapshot": source_snapshot,
        "required": required,
        "valid": True,
        "selected_profile_ids": list(profile_ids),
        "aliases": [],
        "build_options": copy.deepcopy(build.get("build_options", {})),
    }


def _runtime_profiles(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    raw_profiles = manifest.get("profiles", [])
    if not isinstance(raw_profiles, list):
        return result
    for profile in raw_profiles:
        if not isinstance(profile, dict):
            continue
        build_id = profile.get("runtime_build")
        profile_id = profile.get("id")
        if isinstance(build_id, str) and isinstance(profile_id, str):
            result.setdefault(build_id, []).append(profile_id)
    return result


def _read_runtime_builds(manifest: Mapping[str, Any], catalog_path: Path) -> list[dict[str, Any]]:
    raw = manifest.get("runtime_builds")
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict)]
    if catalog_path.is_file():
        catalog = _load_json(catalog_path)
        raw_catalog = catalog.get("runtimes", catalog.get("runtime_builds", []))
        if isinstance(raw_catalog, list):
            return [dict(item) for item in raw_catalog if isinstance(item, dict)]
    return []


def _write_provenance_sidecars(
    plan: Mapping[str, Any],
    *,
    post_cleanup: bool = False,
    index_path: Path | str = DEFAULT_PROVENANCE_INDEX_PATH,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for mapping in plan.get("runtime_mappings", []):
        if not isinstance(mapping, dict) or not mapping.get("valid"):
            continue
        raw_repo = mapping.get("source_repository")
        if not isinstance(raw_repo, str) or not raw_repo:
            continue
        repo = Path(raw_repo).expanduser()
        if not repo.is_dir():
            continue
        snapshot = _repository_snapshot(repo)
        if snapshot is None:
            continue
        canonical = mapping.get("canonical_source_repository") or str(repo)
        payload = {
            "schema_version": 1,
            "repository_path": str(repo),
            "canonical_relocated_path": str(canonical),
            "repository_identity": {
                "branch": snapshot.get("branch"),
                "head": snapshot.get("head"),
            },
            "dirty": snapshot.get("dirty"),
            "changed_paths": snapshot.get("changed_paths", []),
            "untracked_paths": snapshot.get("untracked_paths", []),
            "dirty_content_hash": snapshot.get("dirty_content_hash"),
            "source_binary": mapping.get("resolved_source_binary"),
            "source_binary_sha256": mapping.get("server_sha256"),
            "canonical_binary": str(Path(str(mapping["bundle_root"])) / "bin" / "llama-server"),
            "runtime_ids": [mapping.get("new_id")],
            "profile_ids": list(mapping.get("selected_profile_ids", [])),
            "source_snapshot": mapping.get("source_snapshot"),
            "post_cleanup_state": snapshot if post_cleanup else None,
        }
        sidecar = repo / ".runtime-provenance.json"
        _atomic_write(sidecar, payload)
        entries.append(payload)
    index = Path(index_path).expanduser()
    previous: dict[str, Any] = {}
    if index.is_file():
        try:
            previous = _load_json(index)
        except MigrationError:
            previous = {}
    previous_entries = previous.get("repositories", [])
    if not isinstance(previous_entries, list):
        previous_entries = []
    by_path = {
        str(item.get("repository_path")): item
        for item in previous_entries
        if isinstance(item, dict) and item.get("repository_path")
    }
    for entry in entries:
        by_path[str(entry["repository_path"])] = entry
    result = {"schema_version": 1, "repositories": list(by_path.values())}
    _atomic_write(index, result)
    return result


def build_migration_plan(
    manifest_path: Path | str,
    *,
    catalog_path: Path | str,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    runtimes_dir: Path | str = DEFAULT_RUNTIMES_DIR,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    output_path: Path | str | None = None,
    inventory_roots: Iterable[Path | str] = (),
    inventory_path: Path | str | None = None,
    validate_loaders: bool = True,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    catalog_file = Path(catalog_path).expanduser().resolve()
    models_root = Path(models_dir).expanduser().resolve()
    runtimes_root = Path(runtimes_dir).expanduser().resolve()
    cache_root = Path(cache_dir).expanduser().resolve()
    manifest = _load_json(manifest_file)
    manifest_bytes = manifest_file.read_bytes()
    if manifest.get("schema_version") not in {1, 2}:
        raise MigrationError("unsupported profile manifest schema_version")
    model_mappings, cleanup_paths = _profile_path_entries(
        manifest,
        models_dir=models_root,
        cache_dir=cache_root,
    )
    profiles_by_build = _runtime_profiles(manifest)
    runtime_mappings: list[dict[str, Any]] = []
    invalid_runtime_warnings: list[str] = []
    for build in _read_runtime_builds(manifest, catalog_file):
        mapping = _runtime_mapping(
            build,
            runtimes_dir=runtimes_root,
            profile_ids=profiles_by_build.get(str(build.get("id", "")), []),
        )
        if not mapping.get("valid"):
            invalid_runtime_warnings.append(str(mapping.get("warning", "invalid runtime")))
            continue
        runtime_mappings.append(mapping)
        cleanup_paths.append(str(build.get("server_binary", "")))
        if mapping.get("resolved_source_binary") != mapping.get("source_binary"):
            cleanup_paths.append(str(mapping["resolved_source_binary"]))

    current_runtime_ids = {
        str(item.get("new_id"))
        for item in runtime_mappings
        if isinstance(item, dict) and item.get("new_id")
    }
    orphaned_staged_paths: list[str] = []
    if runtimes_root.is_dir():
        for candidate in sorted(runtimes_root.iterdir(), key=lambda item: item.name):
            if not candidate.is_dir() or (candidate / ".git").exists():
                continue
            if candidate.name in current_runtime_ids:
                continue
            if (candidate / "bin" / "llama-server").is_file():
                orphaned_staged_paths.append(str(candidate))
                cleanup_paths.append(str(candidate))

    repository_relocations: list[dict[str, Any]] = []
    for mapping in runtime_mappings:
        source_repository = mapping.get("source_repository")
        canonical_repository = mapping.get("canonical_source_repository")
        if (
            isinstance(source_repository, str)
            and isinstance(canonical_repository, str)
            and source_repository
            and canonical_repository
            and source_repository != canonical_repository
        ):
            repository_relocations.append(
                {
                    "source": source_repository,
                    "target": canonical_repository,
                    "snapshot": mapping.get("source_snapshot"),
                }
            )
            cleanup_paths.append(source_repository)

    if inventory_path is not None and Path(inventory_path).is_file():
        inventory_payload = _load_json(Path(inventory_path))
        deferred_inventory = inventory_payload.get("inventory", [])
        if not isinstance(deferred_inventory, list):
            deferred_inventory = []
    else:
        deferred_inventory = []
        for raw_root in inventory_roots:
            deferred_inventory.extend(_inventory_root(Path(raw_root).expanduser()))
        unique_inventory = {
            str(item.get("path")): item
            for item in deferred_inventory
            if isinstance(item, dict) and item.get("path")
        }
        deferred_inventory = [unique_inventory[key] for key in sorted(unique_inventory)]

    normalized_inventory: list[dict[str, Any]] = []
    for item in deferred_inventory:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        basename = Path(str(normalized.get("path", ""))).name.casefold()
        if basename.startswith(("ggml-vocab", "mmproj-", "mtp-")):
            normalized["kind"] = "deferred"
        normalized_inventory.append(normalized)
    deferred_inventory = normalized_inventory

    old_catalog_sha = sha256_file(catalog_file) if catalog_file.is_file() else None
    runtime_ids = {item["old_id"]: item["new_id"] for item in runtime_mappings}
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_file),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "catalog_path": str(catalog_file),
        "catalog_sha256": old_catalog_sha,
        "models_dir": str(models_root),
        "runtimes_dir": str(runtimes_root),
        "cache_dir": str(cache_root),
        "deployment_revision": "model-store-" + _sha256_bytes(manifest_bytes)[:16],
        "model_mappings": model_mappings,
        "runtime_mappings": runtime_mappings,
        "repository_relocations": repository_relocations,
        "orphaned_staged_paths": orphaned_staged_paths,
        "runtime_id_map": runtime_ids,
        "deferred_inventory": deferred_inventory,
        "cleanup_paths": list(dict.fromkeys(path for path in cleanup_paths if path)),
        "warnings": invalid_runtime_warnings,
        "validate_loaders": bool(validate_loaders),
        "migration": {
            "staged": False,
            "applied": False,
            "cleaned": False,
        },
    }
    if any(
        isinstance(item, dict) and item.get("source_repository")
        for item in runtime_mappings
    ):
        _write_provenance_sidecars(plan)
        plan["provenance_index"] = str(DEFAULT_PROVENANCE_INDEX_PATH)
    if output_path is not None:
        return _write_plan(Path(output_path).expanduser().resolve(), plan)
    plan["plan_sha256"] = _plan_digest(plan)
    return plan


# ---------------------------------------------------------------------------
# Phase operations


def _stage_runtime(mapping: Mapping[str, Any]) -> list[str]:
    changed: list[str] = []
    for file_entry in mapping.get("bundle_files", []):
        if not isinstance(file_entry, dict):
            raise MigrationError("runtime bundle file entry is invalid")
        source = _resolve_file(str(file_entry["source"]))
        target = Path(str(file_entry["target"]))
        expected = str(file_entry["sha256"])
        if _copy_verified(source, target, expected):
            changed.append(str(target))
    binary = Path(str(mapping["bundle_root"])) / "bin" / "llama-server"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise MigrationError(f"staged runtime is not executable: {binary}")
    return changed


def stage_migration(plan_path: Path | str) -> dict[str, Any]:
    plan = _load_plan(plan_path)
    changed: list[str] = []
    warnings: list[str] = list(plan.get("warnings", []))
    for mapping in plan.get("model_mappings", []):
        if not isinstance(mapping, dict):
            raise MigrationError("model mapping is invalid")
        source_literal = str(mapping["resolved_source"])
        source = Path(source_literal)
        if not source.is_file():
            if mapping.get("required", True):
                raise MigrationError(f"required model source is unavailable: {source}")
            warnings.append(f"optional artifact unavailable: {mapping['source']}")
            continue
        expected = mapping.get("sha256")
        if not isinstance(expected, str):
            expected = sha256_file(source)
        target = Path(str(mapping["target"]))
        if _copy_verified(source, target, expected):
            changed.append(str(target))
    for mapping in plan.get("runtime_mappings", []):
        if not isinstance(mapping, dict):
            raise MigrationError("runtime mapping is invalid")
        changed.extend(_stage_runtime(mapping))
    report = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "staged": True,
        "changed": changed,
        "warnings": warnings,
    }
    report_path = Path(str(plan["_plan_path"])).with_suffix(".stage.json")
    _atomic_write(report_path, report)
    return report


def _run_loader(binary: Path, environment: Mapping[str, str]) -> str:
    try:
        header = binary.read_bytes()[:4]
    except OSError as error:
        raise MigrationError(f"cannot inspect runtime executable {binary}: {error}") from error
    env = dict(os.environ)
    env.update({str(key): str(value) for key, value in environment.items()})
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MigrationError(f"runtime loader check failed for {binary}: {error}") from error
    if result.returncode != 0:
        raise MigrationError(
            f"runtime --version failed for {binary}: {(result.stderr or result.stdout).strip()}"
        )
    return (result.stdout or result.stderr).splitlines()[0] if (result.stdout or result.stderr) else ""


def verify_migration(
    plan_path: Path | str,
    *,
    validate_loaders: bool | None = None,
) -> dict[str, Any]:
    plan = _load_plan(plan_path)
    manifest_file = Path(str(plan["manifest_path"]))
    current_manifest_sha = sha256_file(manifest_file)
    if current_manifest_sha != plan["manifest_sha256"]:
        raise MigrationError(
            f"manifest drift: expected {plan['manifest_sha256']}, got {current_manifest_sha}"
        )
    warnings: list[str] = list(plan.get("warnings", []))
    verified: list[str] = []
    for mapping in plan.get("model_mappings", []):
        if not isinstance(mapping, dict):
            raise MigrationError("model mapping is invalid")
        source = Path(str(mapping["resolved_source"]))
        if not source.is_file():
            if mapping.get("required", True):
                raise MigrationError(f"required source disappeared: {source}")
            warnings.append(f"optional source disappeared: {mapping['source']}")
            continue
        expected = mapping.get("sha256")
        if not isinstance(expected, str):
            raise MigrationError(f"mapping has no hash: {mapping['source']}")
        actual = sha256_file(source)
        if actual != expected:
            if mapping.get("required", True):
                raise MigrationError(f"source hash drift: {source}")
            warnings.append(f"optional source hash drift: {mapping['source']}")
            continue
        target = Path(str(mapping["target"]))
        if not target.is_file() or sha256_file(target) != expected:
            if mapping.get("required", True):
                raise MigrationError(f"staged target is missing or mismatched: {target}")
            warnings.append(f"optional staged target is missing: {target}")
            continue
        verified.append(str(target))
    loader_checks = plan.get("validate_loaders", True) if validate_loaders is None else validate_loaders
    versions: dict[str, str] = {}
    for mapping in plan.get("runtime_mappings", []):
        if not isinstance(mapping, dict) or not mapping.get("valid"):
            continue
        for file_entry in mapping.get("bundle_files", []):
            if not isinstance(file_entry, dict):
                raise MigrationError("runtime bundle file entry is invalid")
            target = Path(str(file_entry["target"]))
            expected = str(file_entry["sha256"])
            if not target.is_file() or sha256_file(target) != expected:
                raise MigrationError(f"runtime bundle target is missing or mismatched: {target}")
        binary = Path(str(mapping["bundle_root"])) / "bin" / "llama-server"
        if not binary.is_file() or sha256_file(binary) != str(mapping["server_sha256"]):
            raise MigrationError(f"runtime executable is missing or mismatched: {binary}")
        if loader_checks:
            versions[str(mapping["new_id"])] = _run_loader(binary, mapping.get("environment", {}))
    result = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "verified": True,
        "verified_artifacts": verified,
        "runtime_versions": versions,
        "warnings": list(dict.fromkeys(warnings)),
    }
    report_path = Path(str(plan["_plan_path"])).with_suffix(".verify.json")
    _atomic_write(report_path, result)
    return result


def _paired_backups(manifest_path: Path, catalog_path: Path, backup_dir: Path) -> dict[str, str | None]:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_backup = backup_dir / f"profiles.json.bak-{stamp}"
    shutil.copy2(manifest_path, manifest_backup)
    os.chmod(manifest_backup, 0o600)
    catalog_backup: Path | None = None
    catalog_absent: Path | None = None
    if catalog_path.is_file():
        catalog_backup = backup_dir / f"catalog.json.bak-{stamp}"
        shutil.copy2(catalog_path, catalog_backup)
        os.chmod(catalog_backup, 0o600)
    else:
        catalog_absent = backup_dir / "catalog.json.absent"
        catalog_absent.write_text("catalog.json was absent before apply\n", encoding="utf-8")
        os.chmod(catalog_absent, 0o600)
    return {
        "profiles": str(manifest_backup),
        "catalog": str(catalog_backup) if catalog_backup else None,
        "catalog_absent": str(catalog_absent) if catalog_absent else None,
    }


def _catalog_records(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for mapping in plan.get("runtime_mappings", []):
        if not isinstance(mapping, dict) or not mapping.get("valid"):
            continue
        records.append(
            {
                "id": mapping["new_id"],
                "runtime": mapping["runtime"],
                "display_name": mapping["display_name"],
                "bundle_root": mapping["bundle_root"],
                "server_binary": str(Path(str(mapping["bundle_root"])) / "bin" / "llama-server"),
                "server_sha256": mapping["server_sha256"],
                "environment": dict(mapping.get("environment", {})),
                "source_repository": (
                    mapping.get("canonical_source_repository")
                    or mapping.get("source_repository")
                ),
                "provenance_file": (
                    str(Path(str(mapping["canonical_source_repository"])) / ".runtime-provenance.json")
                    if mapping.get("canonical_source_repository")
                    else None
                ),
                "source_branch": (
                    mapping.get("source_snapshot", {}).get("branch")
                    if isinstance(mapping.get("source_snapshot"), dict)
                    else None
                ),
                "source_commit": (
                    mapping.get("source_snapshot", {}).get("head")
                    if isinstance(mapping.get("source_snapshot"), dict)
                    else None
                ),
                "dirty": (
                    mapping.get("source_snapshot", {}).get("dirty")
                    if isinstance(mapping.get("source_snapshot"), dict)
                    else None
                ),
                "selected_profile_ids": list(mapping.get("selected_profile_ids", [])),
                "build_options": copy.deepcopy(mapping.get("build_options", {})),
                "aliases": [],
            }
        )
    return records


def _rewrite_manifest(plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(manifest))
    result["schema_version"] = 2
    result["deployment_revision"] = plan["deployment_revision"]
    result.pop("runtime_builds", None)
    model_by_source = {
        str(item["source"]): item
        for item in plan.get("model_mappings", [])
        if isinstance(item, dict)
    }
    runtime_map = dict(plan.get("runtime_id_map", {}))
    profiles = result.get("profiles", [])
    if not isinstance(profiles, list):
        raise MigrationError("profile manifest must contain a profiles list")
    for profile in profiles:
        if not isinstance(profile, dict):
            raise MigrationError("every profile must be an object")
        for field_name, hash_field in (
            ("model_path", "model_sha256"),
            ("mmproj_path", "mmproj_sha256"),
            ("model_draft", "model_draft_sha256"),
        ):
            source = profile.get(field_name)
            if not isinstance(source, str):
                continue
            mapping = model_by_source.get(str(Path(source).expanduser()))
            if mapping is None:
                continue
            profile[field_name] = mapping["target"]
            if mapping.get("sha256"):
                profile[hash_field] = mapping["sha256"]
        old_build = profile.get("runtime_build")
        if isinstance(old_build, str) and old_build in runtime_map:
            profile["runtime_build"] = runtime_map[old_build]
            profile.pop("runtime", None)
            profile.pop("server_binary", None)
            profile.pop("server_sha256", None)
            environment = profile.get("environment", {})
            if not isinstance(environment, dict):
                raise MigrationError(f"profile environment is invalid: {profile.get('id')}")
            mapping = next(
                item for item in plan["runtime_mappings"] if item.get("new_id") == runtime_map[old_build]
            )
            for key, value in mapping.get("environment", {}).items():
                if key == "LD_LIBRARY_PATH" or key not in environment:
                    environment[key] = value
            profile["environment"] = environment
        extra_args = profile.get("extra_args")
        if isinstance(extra_args, list):
            rewritten: list[str] = []
            index = 0
            while index < len(extra_args):
                token = extra_args[index]
                if token == "--moe-cache-profile" and index + 1 < len(extra_args):
                    source = str(Path(str(extra_args[index + 1])).expanduser())
                    mapping = model_by_source.get(source)
                    if mapping is not None:
                        rewritten.extend((token, str(mapping["target"])))
                        index += 2
                        continue
                rewritten.append(token)
                index += 1
            profile["extra_args"] = rewritten
    return result


def _migration_lock_path(plan: Mapping[str, Any]) -> Path:
    return Path(str(plan["manifest_path"])).parent / ".model-store-migration.lock"


def _assert_quiescent() -> None:
    if os.environ.get("LLAMA_BROKER_MIGRATION_SKIP_LIVE") == "1":
        return
    url = os.environ.get("LLAMA_BROKER_ADMIN_URL", "https://127.0.0.1:11434")
    endpoint = url.rstrip("/") + "/admin/status"
    context = ssl._create_unverified_context()
    request = urllib.request.Request(endpoint)
    try:
        with urllib.request.urlopen(request, context=context, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise MigrationError(f"live broker status is unavailable: {error}") from error
    leases = payload.get("leases") if isinstance(payload, dict) else None
    if not isinstance(leases, dict):
        raise MigrationError("live broker status has no leases snapshot")
    active = leases.get("active_requests", 0)
    transitioning = leases.get("transitioning", False)
    queued = leases.get("queued_transitions", 0)
    if active != 0 or transitioning or queued:
        raise MigrationError(
            f"broker is not quiescent: active_requests={active}, "
            f"transitioning={transitioning}, queued_transitions={queued}"
        )


def _apply_migration_unlocked(
    plan_path: Path | str,
    *,
    apply: bool = False,
    validate_loaders: bool | None = None,
    require_quiescent: bool = False,
) -> dict[str, Any]:
    if not apply:
        raise MigrationError("apply requires explicit --apply")
    plan = _load_plan(plan_path)
    if require_quiescent:
        _assert_quiescent()
    verify_migration(plan_path, validate_loaders=validate_loaders)
    manifest_path = Path(str(plan["manifest_path"]))
    catalog_path = Path(str(plan["catalog_path"]))
    backup_dir = manifest_path.parent / "migrations" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backups = _paired_backups(manifest_path, catalog_path, backup_dir)
    old_manifest = _load_json(manifest_path)
    new_manifest = _rewrite_manifest(plan, old_manifest)
    revision = str(plan["deployment_revision"])
    catalog = {
        "schema_version": 2,
        "deployment_revision": revision,
        "temporary_aliases": {
            str(item["old_id"]): str(item["new_id"])
            for item in plan.get("runtime_mappings", [])
            if isinstance(item, dict) and item.get("valid")
        },
        "runtimes": _catalog_records(plan),
        "inventory": copy.deepcopy(plan.get("deferred_inventory", [])),
    }
    journal = backup_dir / "journal.json"
    _atomic_write(
        journal,
        {
            "schema_version": 1,
            "deployment_revision": revision,
            "plan_sha256": plan["plan_sha256"],
            "backups": backups,
            "status": "writing-paired-state",
        },
    )
    catalog_written = False
    try:
        _atomic_write(catalog_path, catalog)
        catalog_written = True
        _atomic_write(manifest_path, new_manifest, mode=stat.S_IMODE(manifest_path.stat().st_mode))
    except BaseException as error:
        try:
            if backups["catalog"]:
                shutil.copy2(str(backups["catalog"]), catalog_path)
            elif catalog_written:
                catalog_path.unlink(missing_ok=True)
            shutil.copy2(str(backups["profiles"]), manifest_path)
        except BaseException as rollback_error:
            raise MigrationError(f"paired apply failed and rollback failed: {rollback_error}") from error
        raise MigrationError(f"paired apply rolled back: {error}") from error
    _atomic_write(
        journal,
        {
            "schema_version": 1,
            "deployment_revision": revision,
            "plan_sha256": plan["plan_sha256"],
            "backups": backups,
            "status": "applied",
        },
    )
    applied_marker = Path(str(plan["_plan_path"])).with_suffix(".applied.json")
    _atomic_write(applied_marker, {"plan_sha256": plan["plan_sha256"], "deployment_revision": revision})
    runtime_suffix = ""
    for mapping in plan.get("runtime_mappings", []):
        if isinstance(mapping, dict) and mapping.get("valid"):
            runtime_suffix = str(mapping["new_id"]).rsplit("-", 1)[-1]
            break
    return {
        "schema_version": 1,
        "applied": True,
        "deployment_revision": revision,
        "backups": backups,
        "runtime_id_suffix": runtime_suffix,
    }


@contextmanager
def _migration_file_lock(plan: Mapping[str, Any]):
    lock_path = _migration_lock_path(plan)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise MigrationError(f"could not acquire migration lock: {error}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def apply_migration(
    plan_path: Path | str,
    *,
    apply: bool = False,
    validate_loaders: bool | None = None,
    require_quiescent: bool = False,
) -> dict[str, Any]:
    if not apply:
        raise MigrationError("apply requires explicit --apply")
    plan = _load_plan(plan_path)
    with _migration_file_lock(plan):
        return _apply_migration_unlocked(
            plan_path,
            apply=True,
            validate_loaders=validate_loaders,
            require_quiescent=require_quiescent,
        )


def _copy_repository_relocation(relocation: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(str(relocation["source"])).expanduser()
    target = Path(str(relocation["target"])).expanduser()
    if not source.is_dir():
        raise MigrationError(f"runtime source repository is unavailable: {source}")
    if source.resolve() == target.resolve():
        snapshot = _repository_snapshot(source)
        return {"source": str(source), "target": str(target), "snapshot": snapshot}
    if target.exists():
        if not target.is_dir():
            raise MigrationError(f"repository relocation target is not a directory: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, symlinks=True, copy_function=shutil.copy2)
    source_snapshot = relocation.get("snapshot")
    target_snapshot = _repository_snapshot(target)
    if target_snapshot is None:
        raise MigrationError(f"relocated repository is not a Git checkout: {target}")
    if isinstance(source_snapshot, dict):
        if target_snapshot.get("branch") != source_snapshot.get("branch"):
            raise MigrationError(f"relocated branch mismatch: {target}")
        if target_snapshot.get("head") != source_snapshot.get("head"):
            raise MigrationError(f"relocated HEAD mismatch: {target}")
        expected_changed = set(source_snapshot.get("changed_paths", []))
        if not expected_changed.issubset(set(target_snapshot.get("changed_paths", []))):
            raise MigrationError(f"relocated dirty tracked paths mismatch: {target}")
    source_sidecar = source / ".runtime-provenance.json"
    target_sidecar = target / ".runtime-provenance.json"
    if source_sidecar.is_file():
        try:
            sidecar = _load_json(source_sidecar)
            sidecar["repository_path"] = str(target)
            sidecar["canonical_relocated_path"] = str(target)
            sidecar["post_cleanup_state"] = target_snapshot
            _atomic_write(target_sidecar, sidecar)
        except MigrationError:
            raise
    return {
        "source": str(source),
        "target": str(target),
        "snapshot": target_snapshot,
    }


def _remove_exact_tree(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        _remove_exact(path)
        return
    for current, directories, files in os.walk(path, topdown=False, followlinks=False):
        current_path = Path(current)
        for filename in files:
            (current_path / filename).unlink()
        for dirname in directories:
            child = current_path / dirname
            if child.is_symlink():
                child.unlink()
            else:
                child.rmdir()
    path.rmdir()


def _finalize_paired_state(plan: Mapping[str, Any], relocations: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    manifest_path = Path(str(plan["manifest_path"]))
    catalog_path = Path(str(plan["catalog_path"]))
    if not manifest_path.is_file() or not catalog_path.is_file():
        return None
    manifest = _load_json(manifest_path)
    catalog = _load_json(catalog_path)
    if manifest.get("schema_version") != 2 or catalog.get("schema_version") != 2:
        raise MigrationError("final cleanup requires schema-v2 paired state")
    final_revision = "model-store-final-" + str(plan["plan_sha256"])[:16]
    backup_dir = manifest_path.parent / "migrations" / (
        "final-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    backups = _paired_backups(manifest_path, catalog_path, backup_dir)
    catalog["deployment_revision"] = final_revision
    catalog["temporary_aliases"] = {}
    relocation_by_source = {item["source"]: item for item in relocations}
    for runtime in catalog.get("runtimes", []):
        if not isinstance(runtime, dict):
            continue
        old_source = runtime.get("source_repository")
        relocation = relocation_by_source.get(old_source)
        if relocation is not None:
            runtime["source_repository"] = relocation["target"]
            runtime["provenance_file"] = str(
                Path(str(relocation["target"])) / ".runtime-provenance.json"
            )
            snapshot = relocation.get("snapshot")
            if isinstance(snapshot, dict):
                runtime["source_branch"] = snapshot.get("branch")
                runtime["source_commit"] = snapshot.get("head")
                runtime["dirty"] = snapshot.get("dirty")
    manifest["deployment_revision"] = final_revision
    try:
        _atomic_write(catalog_path, catalog)
        _atomic_write(manifest_path, manifest, mode=stat.S_IMODE(manifest_path.stat().st_mode))
    except BaseException as error:
        try:
            shutil.copy2(str(backups["catalog"]), catalog_path)
            shutil.copy2(str(backups["profiles"]), manifest_path)
        except BaseException as rollback_error:
            raise MigrationError(f"final paired update failed and rollback failed: {rollback_error}") from error
        raise MigrationError(f"final paired update rolled back: {error}") from error
    index_path = DEFAULT_PROVENANCE_INDEX_PATH
    if index_path.is_file():
        index = _load_json(index_path)
        entries = index.get("repositories", [])
        if isinstance(entries, list):
            relocation_by_source = {item["source"]: item for item in relocations}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                relocation = relocation_by_source.get(entry.get("repository_path"))
                if relocation is not None:
                    entry["repository_path"] = relocation["target"]
                    entry["canonical_relocated_path"] = relocation["target"]
                    entry["post_cleanup_state"] = relocation.get("snapshot")
            _atomic_write(index_path, index)
    return {"deployment_revision": final_revision, "backups": backups}


def _remove_exact(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        # A directory is removed only when every entry was explicitly named by
        # the plan; this function never performs recursive deletion.
        entries = list(path.iterdir())
        if entries:
            raise MigrationError(f"refusing to remove non-empty unplanned directory: {path}")
        path.rmdir()
        return
    raise MigrationError(f"cleanup path does not exist: {path}")


def cleanup_migration(
    plan_path: Path | str,
    *,
    cleanup: bool = False,
    confirm: bool = False,
    extra_paths: Sequence[Path | str] = (),
) -> dict[str, Any]:
    if not cleanup or not confirm:
        raise MigrationError("cleanup requires separate --cleanup --confirm")
    plan = _load_plan(plan_path)
    marker = Path(str(plan["_plan_path"])).with_suffix(".applied.json")
    if not marker.is_file():
        raise MigrationError("cleanup requires a successfully applied migration")
    planned = {str(Path(path).expanduser()) for path in plan.get("cleanup_paths", [])}
    for raw in extra_paths:
        candidate = str(Path(raw).expanduser())
        if candidate not in planned:
            raise MigrationError(f"unplanned cleanup path: {candidate}")
    lock_path = _migration_lock_path(plan)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise MigrationError(f"could not acquire migration cleanup lock: {error}") from error
        relocations: list[dict[str, Any]] = []
        for relocation in plan.get("repository_relocations", []):
            if not isinstance(relocation, dict):
                raise MigrationError("repository relocation entry is invalid")
            relocations.append(_copy_repository_relocation(relocation))
        final_state = _finalize_paired_state(plan, relocations)
        repo_sources = {str(item["source"]) for item in plan.get("repository_relocations", []) if isinstance(item, dict)}
        orphaned_staged = {str(item) for item in plan.get("orphaned_staged_paths", [])}
        removed: list[str] = []
        for raw in sorted(planned):
            path = Path(raw)
            if not path.exists() and not path.is_symlink():
                continue
            if raw in repo_sources or raw in orphaned_staged:
                _remove_exact_tree(path)
            else:
                _remove_exact(path)
            removed.append(raw)
        report = {
            "schema_version": 1,
            "plan_sha256": plan["plan_sha256"],
            "cleaned": True,
            "removed": removed,
            "repository_relocations": relocations,
            "final_state": final_state,
        }
        _atomic_write(Path(str(plan["_plan_path"])).with_suffix(".cleanup.json"), report)
        return report


# ---------------------------------------------------------------------------
# CLI


def _common_phase_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--no-loader-check", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--profiles", type=Path, required=True)
    plan.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    plan.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    plan.add_argument("--runtimes-dir", type=Path, default=DEFAULT_RUNTIMES_DIR)
    plan.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--inventory-root", action="append", default=[], type=Path)
    plan.add_argument("--inventory", type=Path)
    plan.add_argument("--no-loader-check", action="store_true")
    for name in ("stage", "verify", "apply", "cleanup"):
        phase = sub.add_parser(name)
        _common_phase_options(phase)
        if name == "apply":
            phase.add_argument("--apply", action="store_true")
        if name == "cleanup":
            phase.add_argument("--cleanup", action="store_true")
            phase.add_argument("--confirm", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "plan":
            result = build_migration_plan(
                args.profiles,
                catalog_path=args.catalog,
                models_dir=args.models_dir,
                runtimes_dir=args.runtimes_dir,
                cache_dir=args.cache_dir,
                output_path=args.output,
                inventory_roots=args.inventory_root,
                inventory_path=args.inventory,
                validate_loaders=not args.no_loader_check,
            )
        elif args.phase == "stage":
            result = stage_migration(args.plan)
        elif args.phase == "verify":
            result = verify_migration(args.plan, validate_loaders=not args.no_loader_check)
        elif args.phase == "apply":
            result = apply_migration(
                args.plan,
                apply=args.apply,
                validate_loaders=not args.no_loader_check,
                require_quiescent=True,
            )
        else:
            result = cleanup_migration(
                args.plan,
                cleanup=args.cleanup,
                confirm=args.confirm,
            )
    except (MigrationError, OSError, ValueError) as error:
        print(f"model-store-migrate: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
