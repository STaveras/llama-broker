"""Discovery-driven setup planning for Llama Broker.

This module catalogs existing artifacts only.  It never downloads, compiles,
loads, activates, or overwrites a runtime or model.  The CLI owns prompts and
confirmation; these functions are deterministic and safe to exercise offline.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .profile_config import load_manifest_source, sha256_file


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
            "GGML_CUDA_DISABLE_GRAPHS": "1",
            "GGML_CUDA_GRAPH_OPT": "0",
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
        return {"schema_version": 1, "applied": [], "skipped": ["no explicit selection"]}

    original_manifest = profiles_path.read_bytes() if profiles_path.exists() else None
    original_mode = profiles_path.stat().st_mode & 0o777 if profiles_path.exists() else 0o600
    migration_published = False
    if profiles_path.exists():
        payload = load_manifest_source(profiles_path)
        migration_published = profiles_path.read_bytes() != original_manifest
    else:
        payload = {
            "schema_version": 1,
            "profiles": [],
        }
    records = payload.get("profiles")
    if not isinstance(records, list):
        raise ValueError("profile manifest must contain a profiles list")
    profile_id = plan.profile_record["id"]
    if any(isinstance(item, dict) and item.get("id") == profile_id for item in records):
        raise ValueError(f"profile already exists: {profile_id}")
    records.append(plan.profile_record)
    payload["schema_version"] = 1

    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{profiles_path.name}.", dir=profiles_path.parent
    )
    temporary_path = Path(temp_name)
    destination: Path | None = None
    created_link = False
    manifest_replaced = False
    try:
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

        os.replace(temporary_path, profiles_path)
        manifest_replaced = True
    except BaseException as error:
        rollback_error: BaseException | None = None
        try:
            if created_link and destination is not None:
                destination.unlink(missing_ok=True)
            if manifest_replaced or migration_published:
                if original_manifest is None:
                    profiles_path.unlink(missing_ok=True)
                else:
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
                    finally:
                        restore_path.unlink(missing_ok=True)
        except BaseException as caught:
            rollback_error = caught
        if rollback_error is not None:
            raise RuntimeError(f"setup rollback failed: {rollback_error}") from error
        raise
    finally:
        temporary_path.unlink(missing_ok=True)

    applied = ["profile"]
    if create_model_link:
        applied.append("model_symlink")
    return {"schema_version": 1, "applied": applied, "profile_id": profile_id}
