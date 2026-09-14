"""Discovery-driven setup planning for Llama Broker.

This module catalogs existing artifacts only.  It never downloads, compiles,
loads, activates, or overwrites a runtime or model.  The CLI owns prompts and
confirmation; these functions are deterministic and safe to exercise offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .broker_profiles import ProfileError
from .profile_config import ApplyError, load_manifest_source, sha256_file, validate_manifest
from .runtime_builds import RuntimeBuild, RuntimeBuildError, infer_runtime_build


MODEL_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin", ".pt", ".pth"})
PROFILE_ID_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class RuntimeCandidate:
    runtime: str
    path: Path
    qualified: bool
    verified: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "runtime": self.runtime,
            "path": str(self.path),
            "qualified": self.qualified,
            "status": "verified" if self.verified else "unverified",
        }


@dataclass(frozen=True)
class ModelCandidate:
    path: Path
    sha256: str | None
    verified: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "status": "verified" if self.verified else "unverified",
        }


@dataclass(frozen=True)
class DiscoveryReport:
    runtimes: tuple[RuntimeCandidate, ...] = ()
    models: tuple[ModelCandidate, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "runtimes": [item.as_dict() for item in self.runtimes],
            "models": [item.as_dict() for item in self.models],
        }


@dataclass(frozen=True)
class SetupPlan:
    report: DiscoveryReport
    selected_runtime: Path | None = None
    selected_model: Path | None = None
    profile_record: dict[str, object] | None = None
    runtime_build_record: dict[str, object] | None = None
    operations: tuple[dict[str, object], ...] = ()
    errors: tuple[str, ...] = ()
    enable_startup: bool = False
    offer_integrate: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "discovery": self.report.as_dict(),
            "selected_runtime": str(self.selected_runtime) if self.selected_runtime else None,
            "selected_model": str(self.selected_model) if self.selected_model else None,
            "profile": self.profile_record,
            "runtime_build": self.runtime_build_record,
            "operations": list(self.operations),
            "errors": list(self.errors),
            "enable_startup": self.enable_startup,
            "offer_integrate": self.offer_integrate,
        }


def _runtime_for_path(path: Path) -> tuple[str, bool] | None:
    name = path.name.casefold()
    if name in {"llama-server", "llama-server.exe"} or "llama-server" in name:
        return "llamacpp", True
    if name in {"vllm", "vllm.exe"}:
        return "vllm", False
    return None


def _iter_candidates(paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        if resolved.is_file():
            yield resolved
            continue
        if not resolved.is_dir():
            continue
        try:
            children = sorted(
                item
                for item in resolved.rglob("*")
                if not any(part.startswith(".") for part in item.relative_to(resolved).parts)
                and ".git" not in item.parts
            )
        except OSError:
            continue
        for child in children:
            if child.is_file():
                yield child


def _inventory_metadata(path: Path, *, kind: str) -> dict[str, object]:
    stat_result = path.stat()
    return {
        "path": str(path.resolve()),
        "kind": kind,
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }


def _initial_inventory(roots: Iterable[Path]) -> list[dict[str, object]]:
    candidates: dict[Path, dict[str, object]] = {}
    for raw in roots:
        root = Path(raw).expanduser()
        if not root.exists() and not root.is_symlink():
            continue
        paths: Iterable[Path]
        if root.is_file():
            paths = (root,)
        elif root.is_dir():
            try:
                paths = root.rglob("*")
            except OSError:
                continue
        else:
            continue
        try:
            for candidate in paths:
                if any(part.startswith(".") for part in candidate.relative_to(root).parts):
                    continue
                if ".git" in candidate.parts or not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                runtime = _runtime_for_path(resolved)
                if runtime is not None and os.access(resolved, os.X_OK):
                    candidates[resolved] = _inventory_metadata(resolved, kind="runtime")
                elif resolved.suffix.casefold() in MODEL_SUFFIXES and os.access(resolved, os.R_OK):
                    basename = resolved.name.casefold()
                    kind = (
                        "deferred"
                        if basename.startswith(("ggml-vocab", "mmproj-", "mtp-"))
                        else ("serving-candidate" if resolved.suffix.casefold() == ".gguf" else "deferred")
                    )
                    candidates[resolved] = _inventory_metadata(resolved, kind=kind)
        except OSError:
            continue
    return [candidates[path] for path in sorted(candidates, key=lambda item: str(item))]


def _write_catalog_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_runtime_catalog(
    catalog_path: Path | str,
    *,
    inventory_roots: Iterable[Path],
    referenced_paths: Iterable[Path] = (),
    explicit_paths: Iterable[Path] = (),
) -> dict[str, object]:
    """Create the one-time inventory, then use the catalog as setup authority."""
    path = Path(catalog_path).expanduser()
    if not path.is_absolute():
        raise ValueError(f"runtime catalog must be an absolute path: {path}")
    path = path.resolve()
    if not path.exists():
        inventory = _initial_inventory(inventory_roots)
        digest = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        payload: dict[str, object] = {
            "schema_version": 2,
            "deployment_revision": "inventory-" + hashlib.sha256(digest).hexdigest()[:16],
            "temporary_aliases": {},
            "runtimes": [],
            "inventory": inventory,
        }
        _write_catalog_atomically(path, payload)
        return {
            "mode": "initial-inventory",
            "broad_scan": True,
            "inventory_count": len(inventory),
            "inventory_added": inventory,
            "inspected_paths": [],
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read runtime catalog {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("runtimes", []), list):
        raise ValueError(f"runtime catalog is invalid: {path}")
    inspected: list[str] = []
    for raw in (*referenced_paths, *explicit_paths):
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            inspected.append(str(candidate.resolve()))
        else:
            inspected.append(str(candidate))
    inventory = payload.get("inventory", [])
    inventory_count = len(inventory) if isinstance(inventory, list) else 0
    return {
        "mode": "catalog-first",
        "broad_scan": False,
        "inventory_count": inventory_count,
        "inventory_added": [],
        "inspected_paths": list(dict.fromkeys(inspected)),
    }


def discover_candidates(paths: Iterable[Path]) -> DiscoveryReport:
    """Discover executable runtimes and model files under explicit paths."""

    runtimes: dict[tuple[str, Path], RuntimeCandidate] = {}
    models: dict[Path, ModelCandidate] = {}
    for path in _iter_candidates(paths):
        runtime = _runtime_for_path(path)
        if runtime is not None and os.access(path, os.X_OK):
            runtime_name, qualified = runtime
            runtimes[(runtime_name, path)] = RuntimeCandidate(
                runtime=runtime_name,
                path=path,
                qualified=qualified,
                verified=True,
            )
        if path.suffix.casefold() in MODEL_SUFFIXES and os.access(path, os.R_OK):
            try:
                digest = sha256_file(path)
            except (OSError, ValueError):
                digest = None
            models[path] = ModelCandidate(path=path, sha256=digest, verified=digest is not None)
    return DiscoveryReport(
        runtimes=tuple(sorted(runtimes.values(), key=lambda item: (not item.qualified, item.runtime, str(item.path)))),
        models=tuple(sorted(models.values(), key=lambda item: str(item.path))),
    )


def _select_path(value: Path | str | None, paths: Iterable[Path]) -> Path | None:
    if value is None:
        return None
    target = Path(value).expanduser()
    try:
        target = target.resolve()
    except OSError:
        return None
    for path in paths:
        if path == target or path.name == str(value):
            return path
    return target if target.exists() else None


def _profile_id(model: Path) -> str:
    slug = PROFILE_ID_RE.sub("-", model.stem.casefold()).strip("-") or "model"
    return slug[:48]


def _runtime_build_record(runtime: RuntimeCandidate) -> dict[str, object]:
    digest = sha256_file(runtime.path)
    build_id = f"{runtime.runtime}-{digest[:12]}"
    display_name = f"{runtime.runtime} build ({digest[:12]})"
    return {
        "id": build_id,
        "runtime": runtime.runtime,
        "display_name": display_name,
        "server_binary": str(runtime.path),
        "server_sha256": digest,
    }


def _profile_record(runtime: RuntimeCandidate, model: ModelCandidate) -> dict[str, object]:
    profile_id = _profile_id(model.path)
    if runtime.runtime != "llamacpp":
        raise ValueError("experimental runtimes require an explicit qualified profile implementation")
    if not runtime.verified or not model.verified or model.sha256 is None:
        raise ValueError("selected runtime and model must be verified")
    return {
        "id": profile_id,
        "alias": profile_id,
        "aliases": [profile_id],
        "display_name": model.path.stem,
        "role": "persistent",
        "capabilities": ["text"],
        "runtime": runtime.runtime,
        "server_binary": str(runtime.path),
        "server_sha256": sha256_file(runtime.path),
        "model_path": str(model.path),
        "model_sha256": model.sha256,
        "context_size": 32768,
        "parallel": 1,
        "batch_size": 512,
        "ubatch_size": 128,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "flash_attention": "on",
        "gpu_layers": "auto",
        "startup_timeout": 300,
        "safety": {
            "service": {
                "type": "exec",
                "memory_max_mib": 20480,
                "memory_swap_max_mib": 1024,
                "tasks_max": 512,
                "kill_signal": "SIGINT",
                "timeout_stop_sec": 20,
            },
            "admission": {
                "min_ram_available_mib": 8192,
                "max_swap_used_mib": 1024,
                "max_swap_growth_mib": 128,
                "min_total_vram_mib": 12000,
                "min_free_vram_before_start_mib": 8192,
                "driver_vram_reserve_mib": 512,
                "vision_vram_reserve_mib": 1536,
                "safety_vram_reserve_mib": 1024,
            },
        },
        "environment": {
            "GGML_CUDA_PDL": "0",
        },
        "extra_args": ["--no-warmup", "--reasoning", "off"],
    }


def build_setup_plan(
    report: DiscoveryReport,
    *,
    runtime_path: Path | str | None = None,
    model_path: Path | str | None = None,
    allow_experimental: bool = False,
    create_model_link: bool = False,
    enable_startup: bool = False,
    offer_integrate: bool = False,
) -> SetupPlan:
    """Build a plan; no changes occur until the caller applies it explicitly."""

    runtimes = report.runtimes
    models = report.models
    selected_runtime = _select_path(runtime_path, (item.path for item in runtimes))
    selected_model = _select_path(model_path, (item.path for item in models))
    errors: list[str] = []
    operations: list[dict[str, object]] = []
    record = None
    runtime_build_record = None
    if runtime_path is not None and selected_runtime is None:
        errors.append(f"runtime was not discovered or does not exist: {runtime_path}")
    if model_path is not None and selected_model is None:
        errors.append(f"model was not discovered or does not exist: {model_path}")
    if selected_runtime is not None and selected_model is not None:
        runtime = next(item for item in runtimes if item.path == selected_runtime)
        model = next(item for item in models if item.path == selected_model)
        if not runtime.qualified and not allow_experimental:
            errors.append("experimental runtime requires explicit opt-in")
        else:
            try:
                runtime_build_record = _runtime_build_record(runtime)
                record = _profile_record(runtime, model)
            except (OSError, ValueError) as error:
                errors.append(str(error))
            if record is not None:
                operations.append({"action": "create_profile", "profile_id": record["id"]})
                if create_model_link:
                    operations.append({"action": "create_model_symlink", "source": str(model.path)})
    if enable_startup:
        operations.append({"action": "enable_broker_startup"})
    if offer_integrate:
        operations.append({"action": "offer_integrate"})
    return SetupPlan(
        report=report,
        selected_runtime=selected_runtime,
        selected_model=selected_model,
        profile_record=record,
        runtime_build_record=runtime_build_record,
        operations=tuple(operations),
        errors=tuple(errors),
        enable_startup=enable_startup,
        offer_integrate=offer_integrate,
    )


def render_report(report: DiscoveryReport, *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(report.as_dict(), indent=2, sort_keys=True)
    lines = ["Discovered runtimes:"]
    for item in report.runtimes:
        qualification = "qualified" if item.qualified else "experimental"
        lines.append(f"  [{qualification}] {item.runtime}: {item.path}")
    lines.append("Discovered models:")
    for item in report.models:
        lines.append(f"  [{'verified' if item.verified else 'unverified'}] {item.path}")
    if not report.runtimes:
        lines.append("  (none)")
    if not report.models:
        lines.append("  (none)")
    return "\n".join(lines)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _setup_identity_path(value: object, description: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{description} is missing from the setup plan")
    try:
        return Path(value).expanduser().resolve()
    except OSError as error:
        raise ProfileError(f"{description} has an invalid path: {value!r}") from error


def _setup_identity_hash(value: object, description: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value.strip()):
        raise ProfileError(f"{description} is missing a valid SHA-256 pin")
    return value.strip().lower()


def _check_setup_artifact_hash(path: Path, expected: str, description: str) -> None:
    try:
        actual = sha256_file(path)
    except (OSError, ValueError) as error:
        raise ProfileError(
            f"{description} is unavailable while applying setup: {path}: {error}"
        ) from error
    if actual != expected:
        raise ProfileError(
            f"{description} changed since setup planning: {path} "
            f"(expected SHA-256 {expected}, got {actual})"
        )


def _validate_setup_artifacts(
    plan: SetupPlan,
    profile_record: dict[str, object],
    *,
    source_schema: object,
    catalog_build: RuntimeBuild | None,
) -> None:
    """Re-check the immutable identities selected when the plan was built."""
    if plan.selected_runtime is None or plan.selected_model is None:
        raise ProfileError("setup plan is missing a selected runtime or model")

    selected_runtime = plan.selected_runtime.expanduser().resolve()
    pinned_runtime = _setup_identity_path(
        profile_record.get("server_binary"), "selected runtime binary"
    )
    if pinned_runtime != selected_runtime:
        raise ProfileError(
            "selected runtime identity changed since setup planning: "
            f"planned {selected_runtime}, pinned {pinned_runtime}"
        )
    expected_runtime = _setup_identity_hash(
        profile_record.get("server_sha256"), "selected runtime binary"
    )

    if source_schema == 2:
        if catalog_build is None:
            raise ProfileError("selected runtime is not registered in the runtime catalog")
        if pinned_runtime != catalog_build.server_binary:
            raise ProfileError(
                "selected runtime identity does not match catalog build "
                f"{catalog_build.id!r}: {pinned_runtime}"
            )
        catalog_hash = _setup_identity_hash(
            catalog_build.server_sha256,
            f"catalog build {catalog_build.id!r} runtime binary",
        )
        if expected_runtime != catalog_hash:
            raise ProfileError(
                "selected runtime identity does not match catalog build "
                f"{catalog_build.id!r}: SHA-256 pin differs"
            )
        expected_runtime = catalog_hash
    elif plan.runtime_build_record is not None:
        planned_build_path = _setup_identity_path(
            plan.runtime_build_record.get("server_binary"),
            "planned runtime build binary",
        )
        planned_build_hash = _setup_identity_hash(
            plan.runtime_build_record.get("server_sha256"),
            "planned runtime build binary",
        )
        if pinned_runtime != planned_build_path or expected_runtime != planned_build_hash:
            raise ProfileError(
                "selected runtime identity does not match the planned runtime build"
            )
        expected_runtime = planned_build_hash

    _check_setup_artifact_hash(selected_runtime, expected_runtime, "selected runtime artifact")

    selected_model = plan.selected_model.expanduser().resolve()
    pinned_model = _setup_identity_path(
        profile_record.get("model_path"), "selected model"
    )
    if pinned_model != selected_model:
        raise ProfileError(
            "selected model identity changed since setup planning: "
            f"planned {selected_model}, pinned {pinned_model}"
        )
    expected_model = _setup_identity_hash(
        profile_record.get("model_sha256"), "selected model"
    )
    planned_model = next(
        (candidate for candidate in plan.report.models if candidate.path == selected_model),
        None,
    )
    if planned_model is not None and planned_model.sha256 is not None:
        planned_model_hash = _setup_identity_hash(
            planned_model.sha256, "planned model"
        )
        if expected_model != planned_model_hash:
            raise ProfileError("selected model identity does not match the setup plan")
        expected_model = planned_model_hash
    _check_setup_artifact_hash(selected_model, expected_model, "selected model artifact")


def apply_profile_plan(
    plan: SetupPlan,
    profiles_path: Path,
    *,
    models_dir: Path | None = None,
    create_model_link: bool = False,
) -> dict[str, object]:
    """Apply only a reviewed profile/symlink plan using atomic writes."""

    if plan.errors:
        raise ValueError("; ".join(plan.errors))
    if plan.profile_record is None:
        return {
            "schema_version": 1,
            "applied": [],
            "skipped": ["no explicit selection"],
        }

    original_manifest = profiles_path.read_bytes() if profiles_path.exists() else None
    original_mode = profiles_path.stat().st_mode & 0o777 if profiles_path.exists() else 0o600
    existing_migration_backups = {
        backup.name
        for backup in profiles_path.parent.glob(f"{profiles_path.name}.bak-*")
    }
    migration_attempted = False
    owned_manifest_bytes: bytes | None = None
    owned_migration_backups: dict[Path, bytes] = {}
    temporary_path: Path | None = None
    destination: Path | None = None
    created_link = False
    missing_manifest = object()
    unreadable_manifest = object()

    def _current_manifest_bytes() -> bytes | object:
        try:
            return profiles_path.read_bytes()
        except FileNotFoundError:
            return missing_manifest
        except OSError:
            return unreadable_manifest

    def _capture_migration_state() -> None:
        nonlocal owned_manifest_bytes
        if not migration_attempted:
            return
        current = _current_manifest_bytes()
        if isinstance(current, bytes) and current != original_manifest:
            owned_manifest_bytes = current
        try:
            backups = profiles_path.parent.glob(f"{profiles_path.name}.bak-*")
            for backup in backups:
                if backup.name in existing_migration_backups or backup in owned_migration_backups:
                    continue
                try:
                    contents = backup.read_bytes()
                except OSError:
                    continue
                # Migration backups are exact copies of the pre-migration
                # manifest. Anything else is not provably broker-owned.
                if contents == original_manifest:
                    owned_migration_backups[backup] = contents
        except OSError:
            return

    def _restore_original_manifest() -> None:
        if original_manifest is None:
            profiles_path.unlink(missing_ok=True)
            _fsync_directory(profiles_path.parent)
            return
        profiles_path.parent.mkdir(parents=True, exist_ok=True)
        restore_fd, restore_name = tempfile.mkstemp(
            prefix=f".{profiles_path.name}.rollback.",
            dir=profiles_path.parent,
        )
        restore_path = Path(restore_name)
        try:
            with os.fdopen(restore_fd, "wb") as handle:
                handle.write(original_manifest)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(restore_path, original_mode)
            os.replace(restore_path, profiles_path)
            _fsync_directory(profiles_path.parent)
        finally:
            restore_path.unlink(missing_ok=True)

    def _rollback_state() -> list[str]:
        rollback_errors: list[str] = []
        if created_link and destination is not None:
            try:
                destination.unlink(missing_ok=True)
            except BaseException as error:
                rollback_errors.append(str(error))
        if owned_manifest_bytes is not None:
            current = _current_manifest_bytes()
            # The manifest is restored only while its bytes still prove that
            # this operation owns the current publication. A concurrent editor
            # therefore keeps its replacement intact.
            if current == owned_manifest_bytes:
                try:
                    _restore_original_manifest()
                except BaseException as error:
                    rollback_errors.append(str(error))
        for backup, owned_bytes in owned_migration_backups.items():
            try:
                if _current_manifest_bytes() is unreadable_manifest:
                    continue
                if backup.read_bytes() != owned_bytes:
                    continue
                backup.unlink()
            except FileNotFoundError:
                continue
            except BaseException as error:
                rollback_errors.append(str(error))
        return rollback_errors

    try:
        if profiles_path.exists():
            try:
                source_payload = json.loads(original_manifest or b"")
            except (TypeError, ValueError):
                source_payload = None
            migration_attempted = (
                isinstance(source_payload, dict)
                and source_payload.get("schema_version") not in (1, 2)
            )
            try:
                payload = load_manifest_source(profiles_path, validate=False)
            finally:
                _capture_migration_state()
        else:
            payload = {
                "schema_version": 1,
                "profiles": [],
            }
        source_schema = payload.get("schema_version", 1)
        if source_schema == 2 and "runtime_builds" in payload:
            raise ProfileError(
                "schema 2 uses the external catalog; remove the inline "
                "runtime_builds key and repair the manifest separately"
            )

        records = payload.get("profiles")
        if not isinstance(records, list):
            raise ValueError("profile manifest must contain a profiles list")

        profile_record = dict(plan.profile_record)
        artifact_record = dict(profile_record)
        catalog_build: RuntimeBuild | None = None
        if source_schema == 2:
            try:
                catalog_build = infer_runtime_build(payload, profile_record)
            except RuntimeBuildError as error:
                raise ProfileError(str(error)) from error
            if catalog_build is None:
                raise ProfileError(
                    "unknown runtime build: selected runtime is not registered in "
                    "the external runtime catalog"
                )
            profile_record["runtime_build"] = catalog_build.id
            for field_name in ("runtime", "server_binary", "server_sha256"):
                profile_record.pop(field_name, None)
        else:
            runtime_builds = payload.setdefault("runtime_builds", [])
            if not isinstance(runtime_builds, list):
                raise ValueError("profile manifest runtime_builds must be a list")
            if plan.runtime_build_record is not None:
                build_id = plan.runtime_build_record["id"]
                existing_build = next(
                    (
                        item
                        for item in runtime_builds
                        if isinstance(item, dict) and item.get("id") == build_id
                    ),
                    None,
                )
                if existing_build is None:
                    runtime_builds.append(plan.runtime_build_record)
                elif existing_build != plan.runtime_build_record:
                    raise ValueError(
                        f"runtime build already exists with different data: {build_id}"
                    )
            if plan.runtime_build_record is not None:
                profile_record["runtime_build"] = plan.runtime_build_record["id"]

        profile_id = profile_record["id"]
        if any(isinstance(item, dict) and item.get("id") == profile_id for item in records):
            raise ValueError(f"profile already exists: {profile_id}")
        records.append(profile_record)
        validate_manifest(payload, directory=profiles_path.parent)
        _validate_setup_artifacts(
            plan,
            artifact_record,
            source_schema=source_schema,
            catalog_build=catalog_build,
        )

        profiles_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{profiles_path.name}.", dir=profiles_path.parent
        )
        temporary_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, original_mode)

        if create_model_link:
            if models_dir is None or plan.selected_model is None:
                raise ValueError("model link requested without a models directory")
            models_dir.mkdir(parents=True, exist_ok=True)
            destination = models_dir / plan.selected_model.name
            if destination.exists() or destination.is_symlink():
                if not destination.is_symlink() or destination.resolve() != plan.selected_model:
                    raise FileExistsError(f"model link target is not broker-owned: {destination}")
            else:
                destination.symlink_to(plan.selected_model)
                created_link = True

        _validate_setup_artifacts(
            plan,
            artifact_record,
            source_schema=source_schema,
            catalog_build=catalog_build,
        )
        published_manifest = temporary_path.read_bytes()
        os.replace(temporary_path, profiles_path)
        owned_manifest_bytes = published_manifest
        _fsync_directory(profiles_path.parent)
    except BaseException as error:
        try:
            rollback_errors = _rollback_state()
        except BaseException as rollback_error:
            rollback_errors = [str(rollback_error)]
        if rollback_errors:
            raise ApplyError(
                f"setup apply failed: {error}; rollback also failed: "
                + "; ".join(rollback_errors)
            ) from error
        raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    applied = ["profile"]
    if create_model_link:
        applied.append("model_symlink")
    return {"schema_version": source_schema, "applied": applied, "profile_id": profile_id}
