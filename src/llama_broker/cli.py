#!/usr/bin/env python3
"""Administrative CLI for broker-owned immutable backend profiles.

Usage:
    brokerctl model list                   List client-facing model references
    brokerctl status                       Show broker state
    brokerctl profile render MODEL         Render a profile unit
    brokerctl load MODEL                   Load model profile
    brokerctl unload                       Unload the backend
    brokerctl verify [MODEL]               Verify SHA-256 hashes
    brokerctl setup [--clean]              Regenerate units and optionally remove stale units
    brokerctl security init --ip IP        Create local CA and TLS certificates
    brokerctl integrate install             Install broker-owned units
    brokerctl integrate configure [TOOLS]   Configure AI coding tools
    brokerctl security rotate               Rotate native API keys
    brokerctl security revoke KEY           Revoke a native API key
    brokerctl profile list                 List profiles with status
    brokerctl profile show ID              Show one complete profile as JSON
    brokerctl profile add ID --model PATH  Add a complete profile
    brokerctl profile discover             Discover backends via port sweep
    brokerctl profile edit [MODEL]         Open the field-oriented profile editor
    brokerctl profile add MODEL            Inspect a candidate (non-mutating)
    brokerctl runtime list                 List runtimes and their profiles
    brokerctl clean [CATEGORY ...]         List unused models/runtimes/backups; --apply cleans
"""

from __future__ import annotations

import argparse
import base64
import copy
import http.client
import json
import os
import re
import socket
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from . import daemon as broker_runtime
from .backend_controller import BackendControllerError, _atomic_write
from .broker_security import (
    ensure_security_material,
    initialize_security_material,
    load_security_config,
)
from .broker_profiles import (
    ALLOWED_CACHE_TYPES,
    ALLOWED_ROLES,
    PROFILE_ID_RE,
    Profile,
    ProfileError,
    ProfileRegistry,
    launch_environment,
)
from .discovery import discover_backends
from .profile_config import (
    ApplyError,
    apply_manifest_transaction,
    load_manifest_source,
    sha256_file,
    write_manifest_atomically,
)
from .runtime_builds import (
    apply_runtime_build,
    infer_runtime_build,
    resolve_runtime_build,
    runtime_build_label,
)
from .runtimes import get_runtime_info, list_runtimes, runtime_names
from .runtime_control import (RuntimeControlError, project_add, project_list, project_fetch,
                              project_commits, bundle_plan, bundle_build, bundle_list, bundle_verify)
from .integration import filtered_manifest, inspect_manifest, report_for
from .hygiene import (
    CATEGORIES as CLEAN_CATEGORIES,
    DEFAULT_CATEGORIES as DEFAULT_CLEAN_CATEGORIES,
    ApplyResult,
    CleanOptions,
    CleanPlan,
    CleanRoots,
    HygieneError,
    apply_plan,
    build_plan,
    still_unreferenced,
)
from .paths import (
    CONFIG_DIR,
    MODELS_DIR,
    RUNTIME_CATALOG_PATH,
    RUNTIME_PROVENANCE_INDEX_PATH,
    RUNTIMES_DIR,
    XDG_CACHE_HOME,
)
from .setup_wizard import (
    apply_profile_plan,
    build_setup_plan,
    discover_candidates,
    ensure_runtime_catalog,
    render_report,
)
from .autotune import (
    ADAPTERS,
    AutotuneEngine,
    BenchmarkHarness,
    AutotuneTransaction,
    BenchmarkResult,
    Candidate,
    HttpCandidateSession,
    MaintenanceLease,
    MaintenanceLeaseExpired,
    SystemdCandidateLifecycle,
    atomic_write_json,
    discover_schema,
    generate_candidates,
    profile_revision,
)
from .tools.broker_backend import BackendClient
from .daemon import (
    LOCK_PATH,
    PROFILES_PATH,
    _operation_lock,
    backend_endpoint_ready,
    create_production_controller,
    effective_backend_url,
    SYSTEMD_USER_DIR,
)


BROKER_ADMIN_URL = "http://127.0.0.1:11434"
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
# Source checkouts keep templates and example manifests at the repository root.
# Installed wheels fall back to package data, which later helpers may resolve
# through importlib.resources.
BROKER_DIR = PROJECT_ROOT if (PROJECT_ROOT / "docs").is_dir() else PACKAGE_DIR
BIN_DIR = Path.home() / "bin"


def _broker_admin_url() -> str:
    explicit = os.environ.get("LLAMA_BROKER_ADMIN_URL")
    if explicit:
        return explicit.rstrip("/")
    client_url = os.environ.get("LLAMA_BROKER_CLIENT_URL")
    if client_url:
        return client_url.rstrip("/").removesuffix("/v1")
    security = load_security_config()
    scheme = "https" if security.broker_tls_cert_file else "http"
    return f"{scheme}://127.0.0.1:11434"


def _broker_admin(path: str, payload: dict[str, object]) -> dict[str, object]:
    """Use the live broker's drain coordinator; never bypass its lease gate."""
    parts = urlsplit(_broker_admin_url())
    security = load_security_config()
    if parts.scheme == "https":
        connection = http.client.HTTPSConnection(
            parts.hostname,
            parts.port or 443,
            timeout=3600,
            context=security.broker_client_context(),
        )
    else:
        connection = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=3600)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json", "Connection": "close"},
        )
        response = connection.getresponse()
        raw = response.read()
    except OSError as error:
        raise BackendControllerError(
            "live Llama-Broker is unavailable; direct lifecycle fallback is disabled "
            "because it could bypass active request leases"
        ) from error
    finally:
        connection.close()
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BackendControllerError("live broker returned invalid admin JSON") from error
    if response.status not in {200, 202}:
        raise BackendControllerError(
            f"live broker admin request failed with HTTP {response.status}: {decoded}"
        )
    return decoded


def _broker_admin_get(path: str) -> dict[str, object]:
    """Read live broker context without taking a lifecycle lock or lease."""
    parts = urlsplit(_broker_admin_url())
    security = load_security_config()
    if parts.scheme == "https":
        connection = http.client.HTTPSConnection(
            parts.hostname,
            parts.port or 443,
            timeout=2,
            context=security.broker_client_context(),
        )
    else:
        connection = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=2)
    try:
        connection.request("GET", path, headers={"Connection": "close"})
        response = connection.getresponse()
        raw = response.read()
    except (OSError, http.client.HTTPException) as error:
        raise BackendControllerError("live Llama-Broker status is unavailable") from error
    finally:
        connection.close()
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise BackendControllerError("live broker returned invalid status JSON") from error
    if response.status != 200 or not isinstance(decoded, dict):
        raise BackendControllerError(
            f"live broker status failed with HTTP {response.status}: {decoded}"
        )
    return decoded


def _hash_file(path: Path) -> str:
    return sha256_file(path)


def _service_template_path() -> Path:
    source_template = PROJECT_ROOT / "docs/examples/llama-broker.service.in"
    if source_template.exists():
        return source_template
    return PACKAGE_DIR / "data/llama-broker.service.in"


def _service_exec_start() -> str:
    installed = shutil.which("llama-broker")
    if installed:
        return installed
    return f"{sys.executable} {BROKER_DIR / 'llama_broker.py'}"


def _profile_verification_result(
    profile: Profile, *, verify_artifacts: bool = True
) -> dict[str, object]:
    failures = _verify_profile(profile) if verify_artifacts else []
    return {
        "id": profile.id,
        "status": "ok" if not failures else "unverified",
        "errors": failures,
        "runtime": profile.runtime,
        "runtime_build": profile.runtime_build,
        "deployment_revision": profile.deployment_revision,
        "bundle_root": str(profile.bundle_root) if profile.bundle_root else None,
        "source_repository": (
            str(profile.source_repository) if profile.source_repository else None
        ),
        "provenance_file": (
            str(profile.provenance_file) if profile.provenance_file else None
        ),
        "source_branch": profile.source_branch,
        "source_commit": profile.source_commit,
        "dirty": profile.dirty,
        "selected_profile_ids": list(profile.selected_profile_ids),
        "warnings": list(profile.warnings),
        "degraded_features": list(profile.degraded_features),
    }


def _verify_profile(profile: Profile) -> list[str]:
    failures: list[str] = []
    checks = [
        (profile.server_binary, profile.server_sha256, "server_binary"),
        (profile.model_path, profile.model_sha256, "model"),
    ]
    if profile.mmproj_path is not None and profile.mmproj_sha256 is not None:
        checks.append((profile.mmproj_path, profile.mmproj_sha256, "mmproj"))
    for path_field, hash_field, label in (
        ("strata_artifacts_manifest_path", "strata_artifacts_manifest_sha256", "Strata manifest"),
        ("strata_launch_config_path", "strata_launch_config_sha256", "Strata launch config"),
    ):
        path = getattr(profile, path_field, None)
        digest = getattr(profile, hash_field, None)
        if path is not None and isinstance(digest, str):
            checks.append((Path(path), digest, label))
    checks.extend(
        (Path(path), digest, "Strata artifact")
        for path, digest in getattr(profile, "strata_artifact_files", ())
    )
    for path, expected, kind in checks:
        try:
            actual = _hash_file(path)
        except OSError as error:
            failures.append(f"{profile.id} {kind} unavailable: {error}")
            continue
        if actual != expected:
            failures.append(
                f"{profile.id} {kind} SHA mismatch: expected {expected}, got {actual} ({path})"
            )
    return failures


CLEAN_KEEP_FILENAME = "clean-keep"


def _format_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _clean_roots(profiles_path: Path) -> CleanRoots:
    return CleanRoots(
        profiles_path=profiles_path,
        catalog_path=RUNTIME_CATALOG_PATH,
        provenance_path=RUNTIME_PROVENANCE_INDEX_PATH,
        models_dir=MODELS_DIR,
        runtimes_dir=RUNTIMES_DIR,
        config_dir=CONFIG_DIR,
        cache_home=XDG_CACHE_HOME,
        home=Path.home(),
    )


def _clean_keep_paths(extra: Sequence[str]) -> tuple[Path, ...]:
    """Paths `clean` must never touch: --keep plus CONFIG_DIR/clean-keep."""
    lines = list(extra)
    keep_file = CONFIG_DIR / CLEAN_KEEP_FILENAME
    if keep_file.is_file():
        lines.extend(
            line.strip()
            for line in keep_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    paths = tuple(Path(line).expanduser() for line in lines)
    for path in paths:
        if not path.is_absolute():
            raise HygieneError(f"keep path must be absolute: {path}")
    return paths


def _interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _print_clean_plan(plan: CleanPlan, categories: Sequence[str], *, applying: bool) -> None:
    print("brokerctl clean")
    for category in CLEAN_CATEGORIES:
        if category not in categories:
            if category == "caches":
                print("\ncaches: not checked (add `caches` to include package caches)")
            continue
        items = [c for c in plan.candidates if c.category == category]
        action = items[0].action if items else ""
        print(f"\n{category}" + (f" ({action})" if action else ": nothing to clean"))
        for item in items:
            print(f"  {_format_bytes(item.size_bytes):>10}  {item.path}")
            print(f"  {'':>10}  {item.reason}")
    if plan.skipped:
        print("\nleft alone:")
        for skipped in plan.skipped:
            print(f"  {skipped.category:<9} {skipped.path}  ({skipped.reason})")
    if not applying and plan.candidates:
        print(
            f"\nWould archive {_format_bytes(plan.total_bytes('archive'))} "
            f"and delete {_format_bytes(plan.total_bytes('delete'))}."
        )
        if any(c.category == "models" and c.action == "delete" for c in plan.candidates):
            print("Model files are deleted permanently; add --archive-dir DIR to keep a copy.")


def _print_clean_result(result: ApplyResult) -> None:
    archived = sum(int(entry["size_bytes"]) for entry in result.archived)
    deleted = sum(int(entry["size_bytes"]) for entry in result.deleted)
    print(f"\nArchived {len(result.archived)} item(s), {_format_bytes(archived)}"
          + (f" to {result.run_dir} (see MANIFEST.json)" if result.run_dir else ""))
    print(f"Deleted {len(result.deleted)} item(s), {_format_bytes(deleted)}")
    for entry in result.skipped:
        print(f"  skipped {entry['path']}: {entry['reason']}")
    for entry in result.failed:
        print(f"  FAILED {entry['path']}: {entry['error']} (original kept)", file=sys.stderr)


def _confirm_clean(plan: CleanPlan) -> bool:
    """Ask before acting on a dry run's plan; anything but yes declines."""
    steps = []
    for action, verb in (("archive", "archive"), ("delete", "permanently delete")):
        items = [c for c in plan.candidates if c.action == action]
        if items:
            steps.append(f"{verb} {len(items)} item(s), {_format_bytes(plan.total_bytes(action))}")
    question = " and ".join(steps)
    try:
        answer = input(f"\n{question[0].upper()}{question[1:]}? [y/N] ").strip().casefold()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _cmd_clean(args: argparse.Namespace) -> int:
    categories = tuple(dict.fromkeys(args.categories)) or DEFAULT_CLEAN_CATEGORIES
    unknown = [category for category in categories if category not in CLEAN_CATEGORIES]
    if unknown:
        raise HygieneError(
            f"unknown clean category {', '.join(unknown)}; choose from {', '.join(CLEAN_CATEGORIES)}"
        )
    if args.min_size_mib < 0 or args.min_age_days < 0 or args.keep_backups < 0:
        raise HygieneError("--min-size-mib, --min-age-days and --keep-backups cannot be negative")
    # Deleting is the default; an archive directory opts models and backups
    # into being copied off first.
    archive_value = args.archive_dir or os.environ.get("LLAMA_BROKER_ARCHIVE_DIR")
    archive_dir = Path(archive_value).expanduser() if archive_value else None
    options = CleanOptions(
        categories=categories,
        min_size_bytes=args.min_size_mib * 1024 * 1024,
        min_age_seconds=args.min_age_days * 86400.0,
        keep_backups=args.keep_backups,
        keep_paths=_clean_keep_paths(args.keep),
        archive=archive_dir is not None,
    )
    roots = _clean_roots(args.profiles)
    plan = build_plan(roots, options)

    def apply() -> ApplyResult:
        return apply_plan(
            plan,
            archive_dir=archive_dir,
            still_unreferenced=still_unreferenced(roots, options),
            log=(lambda _message: None) if args.json else (lambda message: print(message, file=sys.stderr)),
        )

    result = apply() if args.apply else None
    if args.json:
        print(json.dumps({
            "schema_version": 1,
            "dry_run": result is None,
            "categories": list(categories),
            "candidates": [candidate.as_mapping() for candidate in plan.candidates],
            "skipped": [skipped.as_mapping() for skipped in plan.skipped],
            "totals": {
                "archive_bytes": plan.total_bytes("archive"),
                "delete_bytes": plan.total_bytes("delete"),
            },
            "result": result.as_mapping() if result is not None else None,
        }, indent=2, sort_keys=True))
        return 1 if result is not None and result.failed else 0

    _print_clean_plan(plan, categories, applying=args.apply)
    if result is not None:
        _print_clean_result(result)
    if result is None and plan.candidates:
        # A dry run at a terminal offers to go ahead; scripts get the plan only.
        if not _interactive_terminal():
            print("\nRun the same command with --apply to do this.")
        elif _confirm_clean(plan):
            result = apply()
            _print_clean_result(result)
        else:
            print("Nothing changed.")
    return 1 if result is not None and result.failed else 0


PROFILE_MUTATION_REMINDER = (
    "Reminder: profile edits do not install units or change the running broker; "
    "run `brokerctl integrate install` or `brokerctl setup` when ready."
)
CLI_CACHE_TYPES = ("f16", "kvarn4", "turbo4")
KNOWN_CACHE_TYPES = ALLOWED_CACHE_TYPES
ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REQUIRED_LLAMACPP_ENVIRONMENT = frozenset({"GGML_CUDA_PDL"})


def _load_editable_manifest(
    profiles_path: Path, *, allow_missing_artifacts: bool = False
) -> tuple[ProfileRegistry, dict[str, object]]:
    """Load a manifest for diagnostics or mutation.

    Missing artifacts are allowed only for read-only diagnostics and the TUI
    inspection surface. Any mutation still validates strictly before publish.
    """

    registry = ProfileRegistry.from_path(
        profiles_path, allow_missing_artifacts=allow_missing_artifacts
    )
    return registry, load_manifest_source(
        profiles_path, allow_missing_artifacts=allow_missing_artifacts
    )


def _write_validated_manifest(profiles_path: Path, data: dict[str, object]) -> None:
    """Validate and durably publish a replacement manifest."""

    write_manifest_atomically(profiles_path, data)


def _record_for_selector(
    registry: ProfileRegistry,
    data: dict[str, object],
    selector: str,
) -> tuple[Profile, dict[str, object]]:
    try:
        profile = registry.get(selector)
    except ProfileError:
        profile = registry.resolve(selector)
    records = data.get("profiles")
    if not isinstance(records, list):
        raise ProfileError("profile manifest must contain a profiles list")
    for record in records:
        if isinstance(record, dict) and record.get("id") == profile.id:
            return profile, record
    raise ProfileError(f"profile record not found: {profile.id}")


def _is_authoritative_manifest(profiles_path: Path) -> bool:
    return profiles_path.resolve() == PROFILES_PATH.resolve()


def _refresh_live_profile_registry(profiles_path: Path) -> None:
    if not _is_authoritative_manifest(profiles_path):
        return
    try:
        submitted = _broker_admin("/admin/autotune/reload", {"wait": False})
        operation = submitted.get("operation")
        request_id = operation.get("request_id") if isinstance(operation, Mapping) else None
        if not isinstance(request_id, str) or not request_id:
            raise BackendControllerError(
                "live broker did not return a profile reload request ID"
            )
        while True:
            result = _broker_admin_get(f"/admin/operations/{request_id}")
            operation = result.get("operation")
            if not isinstance(operation, Mapping):
                raise BackendControllerError(
                    f"profile reload status is unavailable for request {request_id}"
                )
            status = operation.get("status")
            if status in {"succeeded", "failed", "cancelled"}:
                if status != "succeeded":
                    detail = operation.get("error") or operation.get("error_type") or status
                    raise BackendControllerError(
                        f"profile reload request {request_id} {status}: {detail}"
                    )
                return
            time.sleep(0.5)
    except Exception as error:
        raise ApplyError(f"live broker profile registry refresh failed: {error}") from error


def _publish_profile_change(
    profiles_path: Path,
    data: dict[str, object],
    description: str,
) -> int:
    old_manifest = json.loads(profiles_path.read_text(encoding="utf-8"))
    if not isinstance(old_manifest, dict):
        raise ApplyError("profile manifest must contain a JSON object")
    live_manifest = _is_authoritative_manifest(profiles_path)
    _write_validated_manifest(profiles_path, data)
    try:
        _refresh_live_profile_registry(profiles_path)
    except Exception:
        try:
            _write_validated_manifest(profiles_path, old_manifest)
        except Exception as rollback_error:
            raise ApplyError(
                "profile mutation failed and manifest rollback also failed: "
                f"{rollback_error}"
            ) from rollback_error
        raise
    print(f"{description} in {profiles_path}")
    if live_manifest:
        print("Live broker profile registry refreshed.")
    else:
        print(PROFILE_MUTATION_REMINDER)
    return 0


def _parse_env_assignment(raw: str) -> tuple[str, str]:
    key, separator, value = raw.partition("=")
    key = key.strip()
    if not separator or not ENVIRONMENT_NAME_RE.fullmatch(key):
        raise ProfileError(f"environment assignment must be KEY=VALUE: {raw!r}")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise ProfileError(f"environment value contains a forbidden control character: {key}")
    return key, value


def _normalise_existing_path(raw: str, field: str) -> Path:
    value = Path(raw).expanduser()
    if not value.is_absolute():
        raise ProfileError(f"{field} must be an absolute path: {raw!r}")
    if not value.exists():
        raise ProfileError(f"{field} does not exist: {value}")
    return value.resolve()


def _validate_cache_type(value: str, field: str) -> str:
    normalised = value.strip().lower()
    if normalised not in KNOWN_CACHE_TYPES:
        supported = ", ".join(sorted(KNOWN_CACHE_TYPES))
        raise ProfileError(f"invalid {field} {value!r}; supported cache types: {supported}")
    return normalised


def _validate_gpu_layers(value: str) -> str:
    normalised = value.strip().lower()
    if normalised in {"all", "auto"} or normalised.isdigit():
        return normalised
    raise ProfileError("gpu-layers must be 'all', 'auto', or a non-negative integer")


def _set_scalar_field(record: dict[str, object], field: str, raw_value: str) -> None:
    field_map = {
        "display-name": "display_name",
        "role": "role",
        "alias": "alias",
        "server-binary": "server_binary",
        "strata-artifacts-manifest": "strata_artifacts_manifest_path",
        "strata-launch-config": "strata_launch_config_path",
        "model": "model_path",
        "model-sha256": "model_sha256",
        "context": "context_size",
        "parallel": "parallel",
        "batch": "batch_size",
        "ubatch": "ubatch_size",
        "cache-k": "cache_type_k",
        "cache-v": "cache_type_v",
        "moe-cache-slots-limit": "moe_cache_slots_limit",
        "flash": "flash_attention",
        "gpu-layers": "gpu_layers",
        "mmproj": "mmproj_path",
        "restore": "restore_profile",
        "startup-timeout": "startup_timeout",
    }
    if field not in field_map:
        supported = ", ".join(sorted(field_map))
        raise ProfileError(f"unknown profile field {field!r}; supported fields: {supported}")

    key = field_map[field]
    if field == "role":
        if raw_value not in ALLOWED_ROLES:
            raise ProfileError(
                f"invalid role {raw_value!r}; expected one of: "
                + ", ".join(sorted(ALLOWED_ROLES))
            )
        record[key] = raw_value
    elif field == "alias":
        value = raw_value.strip()
        if not value:
            raise ProfileError("alias must not be empty")
        record[key] = value
    elif field in {
        "context",
        "parallel",
        "batch",
        "ubatch",
        "startup-timeout",
        "moe-cache-slots-limit",
    }:
        try:
            value = int(raw_value, 10)
        except ValueError as error:
            raise ProfileError(f"{field} must be an integer: {raw_value!r}") from error
        record[key] = value
    elif field in {"cache-k", "cache-v"}:
        record[key] = _validate_cache_type(raw_value, field)
    elif field == "flash":
        value = raw_value.strip().lower()
        if value not in {"auto", "on", "off"}:
            raise ProfileError("flash must be one of: auto, on, off")
        record[key] = value
    elif field == "gpu-layers":
        record[key] = _validate_gpu_layers(raw_value)
    elif field in {
        "server-binary", "model", "mmproj",
        "strata-artifacts-manifest", "strata-launch-config",
    }:
        value = raw_value.strip()
        if field == "mmproj" and not value:
            record.pop("mmproj_path", None)
            record.pop("mmproj_sha256", None)
            if not record.get("native_vision"):
                capabilities = [
                    item for item in record.get("capabilities", []) if item != "vision"
                ]
                record["capabilities"] = capabilities or ["text"]
            return
        path = _normalise_existing_path(value, key)
        record[key] = str(path)
        if field == "server-binary":
            record["server_sha256"] = sha256_file(path)
        elif field == "model":
            record["model_sha256"] = sha256_file(path)
        elif field == "strata-artifacts-manifest":
            record["strata_artifacts_manifest_sha256"] = sha256_file(path)
        elif field == "strata-launch-config":
            record["strata_launch_config_sha256"] = sha256_file(path)
        else:
            record["mmproj_sha256"] = sha256_file(path)
            capabilities = set(record.get("capabilities", ["text"]))
            capabilities.add("vision")
            record["capabilities"] = sorted(capabilities)
    elif field == "model-sha256":
        value = raw_value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ProfileError("model-sha256 must be a lowercase 64-character SHA-256")
        record[key] = value
    elif field == "restore":
        if raw_value.strip():
            record[key] = raw_value.strip()
        else:
            record.pop(key, None)
    else:
        record[key] = raw_value


def _remove_optional_field(record: dict[str, object], field: str) -> None:
    if field == "mmproj":
        record.pop("mmproj_path", None)
        record.pop("mmproj_sha256", None)
        if not record.get("native_vision"):
            capabilities = [
                item for item in record.get("capabilities", []) if item != "vision"
            ]
            record["capabilities"] = capabilities or ["text"]
        return
    if field == "restore":
        record.pop("restore_profile", None)
        return
    if field == "model-draft":
        record.pop("model_draft", None)
        return
    if field == "strata-artifacts-manifest":
        record.pop("strata_artifacts_manifest_path", None)
        record.pop("strata_artifacts_manifest_sha256", None)
        return
    if field == "strata-launch-config":
        record.pop("strata_launch_config_path", None)
        record.pop("strata_launch_config_sha256", None)
        return
    if field in {"environment", "env"}:
        environment = dict(record.get("environment", {}))
        if str(record.get("runtime", "llamacpp")) == "llamacpp":
            environment = {
                key: value
                for key, value in environment.items()
                if key in REQUIRED_LLAMACPP_ENVIRONMENT
            }
        else:
            environment.clear()
        record["environment"] = environment
        return
    if field in {"extra-args", "args"}:
        record["extra_args"] = []
        return
    if field == "aliases":
        record.pop("aliases", None)
        return
    if field == "fit":
        for key in ("fit", "fit_target_mib", "fit_ctx", "ctx_checkpoints", "cache_ram_mib"):
            record.pop(key, None)
        return
    if field == "autotune":
        record.pop("autotune", None)
        return
    supported = "mmproj, restore, model-draft, environment, extra-args, aliases, fit, autotune"
    raise ProfileError(f"unknown optional field {field!r}; supported fields: {supported}")


def _template_profile_record(data: dict[str, object]) -> dict[str, object]:
    records = data.get("profiles")
    if not isinstance(records, list) or not records:
        raise ProfileError("cannot add a profile without an existing template profile")
    template = next(
        (record for record in records if isinstance(record, dict)),
        None,
    )
    if template is None:
        raise ProfileError("profile template is not an object")
    return copy.deepcopy(template)


def _build_added_profile(args: argparse.Namespace, data: dict[str, object]) -> dict[str, object]:
    profile_id = args.profile_id.strip()
    if not PROFILE_ID_RE.fullmatch(profile_id):
        raise ProfileError(f"invalid profile id: {profile_id!r}")
    record = _template_profile_record(data)

    template_build = None
    raw_template_build = record.get("runtime_build")
    if isinstance(raw_template_build, str) and raw_template_build.strip():
        try:
            template_build = resolve_runtime_build(data, raw_template_build)
        except ValueError as error:
            raise ProfileError(str(error)) from error

    requested_server_path = None
    requested_server_sha256 = None
    if args.server_binary:
        requested_server_path = _normalise_existing_path(
            args.server_binary, "server-binary"
        )
        requested_server_sha256 = sha256_file(requested_server_path)

    selected_build = None
    requested_runtime_build = getattr(args, "runtime_build", None)
    if requested_runtime_build is not None:
        try:
            selected_build = resolve_runtime_build(data, requested_runtime_build)
        except ValueError as error:
            raise ProfileError(str(error)) from error
    elif requested_server_path is not None:
        selected_build = infer_runtime_build(
            data,
            {
                "runtime": record.get("runtime"),
                "server_binary": str(requested_server_path),
                "server_sha256": requested_server_sha256,
            },
        )
    if selected_build is None:
        selected_build = template_build

    strata_manifest_path: Path | None = None
    strata_launch_path: Path | None = None
    strata_launch_config: dict[str, Any] | None = None
    strata_context_size: int | None = None
    strata_default_alias: str | None = None
    if selected_build is not None and selected_build.runtime == "strata":
        raw_manifest_path = getattr(args, "strata_artifacts_manifest", None)
        raw_launch_path = getattr(args, "strata_launch_config", None)
        if not raw_manifest_path or not raw_launch_path:
            raise ProfileError(
                "Strata profile add requires --strata-artifacts-manifest and --strata-launch-config"
            )
        strata_manifest_path = _normalise_existing_path(
            str(raw_manifest_path), "strata-artifacts-manifest"
        )
        strata_launch_path = _normalise_existing_path(
            str(raw_launch_path), "strata-launch-config"
        )
        try:
            strata_launch_config = json.loads(strata_launch_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProfileError(f"cannot read Strata launch config: {error}") from error
        if not isinstance(strata_launch_config, dict):
            raise ProfileError("Strata launch config must be a JSON object")
        strata_default_alias = strata_launch_config.get("model_name")
        engine_args = strata_launch_config.get("engine_args")
        if not isinstance(strata_default_alias, str) or not strata_default_alias.strip():
            raise ProfileError("Strata launch config requires a non-empty model_name")
        if not isinstance(engine_args, list) or not all(isinstance(value, str) for value in engine_args):
            raise ProfileError("Strata launch config engine_args must be a string list")
        try:
            context_index = engine_args.index("--max-context")
            strata_context_size = int(engine_args[context_index + 1])
        except (ValueError, IndexError) as error:
            raise ProfileError("Strata launch config requires --max-context") from error
        if engine_args.count("--max-context") != 1 or strata_context_size <= 0:
            raise ProfileError("Strata launch config has invalid --max-context")
        manifest = json.loads(strata_manifest_path.read_text(encoding="utf-8"))
        try:
            first_shard = manifest["model"]["shards"][0]["path"]
            expected_model = (strata_manifest_path.parent / first_shard).resolve()
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise ProfileError("Strata artifact manifest has no first model shard") from error
        selected_model = _normalise_existing_path(str(args.model), "model")
        if selected_model != expected_model:
            raise ProfileError("--model must be the first shard pinned by the Strata artifact manifest")
        requested_aliases = [item.strip() for item in (args.alias or []) if item.strip()]
        if requested_aliases and requested_aliases[0] != strata_default_alias:
            raise ProfileError("Strata profile alias must match launch config model_name")
        if args.context is not None and args.context != strata_context_size:
            raise ProfileError("--context must match Strata launch config --max-context")
        if args.parallel not in (None, 1):
            raise ProfileError("Strata profile parallel must be 1")
        if args.role not in (None, "persistent"):
            raise ProfileError("Strata profiles must be persistent")
        if any(getattr(args, field, None) is not None for field in (
            "batch", "ubatch", "cache_k", "cache_v", "flash", "gpu_layers", "restore"
        )):
            raise ProfileError("Strata profiles do not accept llama.cpp-only options")
        if args.mmproj or args.env or args.extra_args:
            raise ProfileError("Strata profiles do not accept vision, profile environment, or extra args")

    if selected_build is not None:
        if requested_server_path is not None and (
            requested_server_path != selected_build.server_binary
            or requested_server_sha256 != selected_build.server_sha256
        ):
            raise ProfileError(
                f"server_binary {requested_server_path} does not match runtime build "
                f"{selected_build.id!r}"
            )
        template_runtime = (
            template_build.runtime
            if template_build is not None
            else str(record.get("runtime") or "llamacpp").strip().lower()
        )
        runtime_changed = template_runtime != selected_build.runtime
        if runtime_changed:
            editable_fields = get_runtime_info(selected_build.runtime).editable_fields
            for field in tuple(record):
                if field not in editable_fields:
                    record.pop(field, None)
            if selected_build.runtime == "strata":
                record.pop("extra_args", None)
            else:
                record["extra_args"] = []
        record.update(
            {
                "runtime": selected_build.runtime,
                "runtime_build": selected_build.id,
                "server_binary": str(selected_build.server_binary),
                "server_sha256": selected_build.server_sha256,
            }
        )
        raw_environment = record.get("environment", {})
        if not isinstance(raw_environment, dict):
            raise ProfileError("environment must be a mapping for the new profile")
        inherited_environment = {} if runtime_changed else dict(raw_environment)
        if template_build is not None and not runtime_changed:
            for key, value in template_build.environment.items():
                if inherited_environment.get(key) == value:
                    inherited_environment.pop(key)
        if selected_build.runtime == "vllm":
            inherited_environment = {
                key: value
                for key, value in inherited_environment.items()
                if not key.startswith(("GGML_", "LLAMA_"))
            }
        environment = dict(selected_build.environment)
        environment.update(inherited_environment)
        if selected_build.runtime == "llamacpp":
            environment.setdefault("GGML_CUDA_PDL", "0")
        record["environment"] = environment
    else:
        server_path = requested_server_path or _normalise_existing_path(
            str(record.get("server_binary", "")), "server-binary"
        )
        record["server_binary"] = str(server_path)
        record["server_sha256"] = (
            requested_server_sha256
            if requested_server_sha256 is not None
            else sha256_file(server_path)
        )

    model_path = _normalise_existing_path(args.model, "model")
    default_alias = strata_default_alias if strata_default_alias is not None else profile_id
    aliases = [item.strip() for item in (args.alias or [default_alias]) if item.strip()]
    if not aliases:
        raise ProfileError("at least one non-empty alias is required")

    record["id"] = profile_id
    record["alias"] = aliases[0]
    record["aliases"] = aliases
    record["display_name"] = args.display_name or profile_id
    record["role"] = args.role or "persistent"
    record["capabilities"] = ["text"]
    record["native_vision"] = False
    record.pop("restore_profile", None)
    record.pop("model_draft", None)
    record.pop("fit", None)
    for key in ("fit_target_mib", "fit_ctx", "ctx_checkpoints", "cache_ram_mib"):
        record.pop(key, None)
    record["model_path"] = str(model_path)
    record["model_sha256"] = sha256_file(model_path)
    if selected_build is not None and selected_build.runtime == "strata":
        assert strata_manifest_path is not None and strata_launch_path is not None
        assert strata_context_size is not None
        record["strata_artifacts_manifest_path"] = str(strata_manifest_path)
        record["strata_artifacts_manifest_sha256"] = sha256_file(strata_manifest_path)
        record["strata_launch_config_path"] = str(strata_launch_path)
        record["strata_launch_config_sha256"] = sha256_file(strata_launch_path)
        record["context_size"] = strata_context_size
        record["parallel"] = 1
        record["capabilities"] = ["text"]
        record["native_vision"] = False
        record.pop("extra_args", None)
        record["safety"] = {
            "service": {
                "type": "exec",
                "memory_max_mib": 20_480,
                "memory_swap_max_mib": 1_024,
                "tasks_max": 512,
                "kill_signal": "SIGINT",
                "timeout_stop_sec": 20,
            },
            "admission": {
                "min_ram_available_mib": 6_144,
                "max_swap_used_mib": 1_024,
                "max_swap_growth_mib": 128,
                "min_total_vram_mib": 12_000,
                "min_free_vram_before_start_mib": 8_192,
                "driver_vram_reserve_mib": 512,
                "vision_vram_reserve_mib": 0,
                "safety_vram_reserve_mib": 1_024,
            },
        }

    if args.context is not None:
        record["context_size"] = args.context
    if args.parallel is not None:
        record["parallel"] = args.parallel
    if args.batch is not None:
        record["batch_size"] = args.batch
    if args.ubatch is not None:
        record["ubatch_size"] = args.ubatch
    if args.cache_k is not None:
        record["cache_type_k"] = args.cache_k
    if args.cache_v is not None:
        record["cache_type_v"] = args.cache_v
    if args.flash is not None:
        record["flash_attention"] = args.flash
    if args.gpu_layers is not None:
        record["gpu_layers"] = _validate_gpu_layers(args.gpu_layers)
    if args.mmproj:
        mmproj_path = _normalise_existing_path(args.mmproj, "mmproj")
        record["mmproj_path"] = str(mmproj_path)
        record["mmproj_sha256"] = sha256_file(mmproj_path)
        record["capabilities"] = ["text", "vision"]
    else:
        record.pop("mmproj_path", None)
        record.pop("mmproj_sha256", None)
    if args.restore:
        record["restore_profile"] = args.restore
    if args.env:
        environment = dict(record.get("environment", {}))
        for assignment in args.env:
            key, value = _parse_env_assignment(assignment)
            environment[key] = value
        record["environment"] = environment
    if args.extra_args:
        record["extra_args"] = list(args.extra_args)
    return record


# ── Interactive profile editing ──────────────────────────────────────────────

EDITABLE_FIELDS = [
    ("runtime", "Serving runtime", str),
    ("context_size", "Context window size", int),
    ("parallel", "Parallel requests (slots)", int),
    ("strata_artifacts_manifest_path", "Strata artifact manifest path", str),
    ("strata_launch_config_path", "Strata launch config path", str),
    ("tensor_parallel_size", "vLLM tensor parallel size", int),
    ("gpu_memory_utilization", "vLLM GPU memory utilization", float),
    ("kv_cache_dtype", "vLLM KV cache dtype", str),
    ("tool_call_parser", "vLLM tool-call parser", str),
    ("tool_parser_plugin", "vLLM tool parser plugin", str),
    ("chat_template", "vLLM chat template", str),
    ("batch_size", "Batch size", int),
    ("ubatch_size", "Micro-batch size", int),
    ("cache_type_k", "KV cache type (k)", str),
    ("cache_type_v", "KV cache type (v)", str),
    ("fit", "Auto-fit to VRAM", bool),
    ("fit_target_mib", "Fit target MiB", int),
    ("fit_ctx", "Fit context size", int),
]

STRATA_HASHED_PATH_FIELDS = {
    "strata_artifacts_manifest_path": "strata_artifacts_manifest_sha256",
    "strata_launch_config_path": "strata_launch_config_sha256",
}

FIT_ONLY_FIELDS = (
    "fit_target_mib",
    "fit_ctx",
    "ctx_checkpoints",
    "cache_ram_mib",
)


def _interactive_edit(
    profile_id: str,
    profiles_path: Path,
    flags: dict[str, str | None],
    *,
    flag_mode: bool = False,
) -> int:
    """Interactively edit a profile's parameters."""
    data = json.loads(profiles_path.read_text(encoding="utf-8"))
    profile_idx = None
    profile_data = None
    for i, p in enumerate(data.get("profiles", [])):
        if p.get("id") == profile_id or p.get("alias") == profile_id:
            profile_idx = i
            profile_data = p
            break

    if profile_idx is None:
        print(f"Profile not found: {profile_id}", file=sys.stderr)
        return 1

    selected_runtime = str(
        flags.get("runtime") or profile_data.get("runtime", "llamacpp")
    ).strip().lower()
    runtime_info = get_runtime_info(selected_runtime)
    unsupported = sorted(
        field
        for field, value in flags.items()
        if value is not None and field not in runtime_info.editable_fields
    )
    if unsupported:
        raise ProfileError(
            f"runtime {selected_runtime!r} does not support edit fields: "
            + ", ".join(unsupported)
        )

    display_name = profile_data.get("display_name", profile_data.get("alias", profile_id))
    print(f"\nEditing: {profile_data['id']} ({display_name})\n")
    inferred_build = infer_runtime_build(data, profile_data)
    print(
        f"Runtime build: {runtime_build_label(inferred_build) if inferred_build is not None else '(custom/manual)'}"
    )

    changed = False
    runtime_build = flags.get("runtime_build")
    if runtime_build is not None:
        # Resolve the build strictly, but defer the runtime/build relationship
        # check to the final manifest validation. This lets a flag-based edit
        # hold a temporary mismatch while still refusing to publish one.
        build = resolve_runtime_build(data, runtime_build)
        updated = copy.deepcopy(profile_data)
        updated.update(
            {
                "runtime": selected_runtime,
                "runtime_build": build.id,
                "server_binary": str(build.server_binary),
                "server_sha256": build.server_sha256,
            }
        )
        if updated != profile_data:
            profile_data = updated
            data["profiles"][profile_idx] = profile_data
            changed = True
            print(f"  Runtime build: {runtime_build}")
    for field_key, field_label, field_type in EDITABLE_FIELDS:
        current = profile_data.get(field_key)
        flag_value = flags.get(field_key)

        if flag_mode and flag_value is None:
            continue

        if field_key in STRATA_HASHED_PATH_FIELDS:
            raw_path = flag_value
            if raw_path is None:
                raw_path = input(f"  {field_label} [{current or ''}]: ").strip()
            raw_path = raw_path.strip()
            hash_key = STRATA_HASHED_PATH_FIELDS[field_key]
            if not raw_path:
                profile_data.pop(field_key, None)
                profile_data.pop(hash_key, None)
                changed = True
                continue
            path = _normalise_existing_path(raw_path, field_key)
            digest = sha256_file(path)
            if profile_data.get(field_key) != str(path) or profile_data.get(hash_key) != digest:
                profile_data[field_key] = str(path)
                profile_data[hash_key] = digest
                changed = True
                print(f"  {field_label}: pinned {path}")
            continue

        if flag_value is not None:
            # Flag-based: convert and apply
            if field_key == "fit" and flag_value.strip().lower() in {
                "off",
                "false",
                "0",
                "no",
                "unset",
                "native",
                "omit",
            }:
                removed = [key for key in ("fit", *FIT_ONLY_FIELDS) if key in profile_data]
                for key in removed:
                    profile_data.pop(key, None)
                if removed:
                    changed = True
                    print(f"  {field_label}: removed ({', '.join(removed)})")
                continue
            new_value = _coerce_value(flag_value, field_type)
            if new_value != current:
                profile_data[field_key] = new_value
                changed = True
                print(f"  {field_label}: {current} -> {new_value}")
            continue

        # Interactive mode
        if field_type == bool:
            prompt = f"  {field_label} [{'on' if current else 'off'}]: "
            raw = input(prompt).strip().lower()
            if raw in ("on", "true", "1", "yes"):
                new_value = True
            elif raw in ("off", "false", "0", "no", "unset", "native", "omit"):
                removed = [key for key in ("fit", *FIT_ONLY_FIELDS) if key in profile_data]
                for key in removed:
                    profile_data.pop(key, None)
                if removed:
                    changed = True
                    print(f"  {field_label}: removed ({', '.join(removed)})")
                continue
            elif not raw:
                continue
            else:
                print(f"    (skipped, invalid boolean)")
                continue
        else:
            current_str = str(current) if current is not None else ""
            prompt = f"  {field_label} [{current_str}]: "
            raw = input(prompt).strip()
            if not raw:
                continue
            new_value = _coerce_value(raw, field_type)

        if new_value != current:
            profile_data[field_key] = new_value
            changed = True
            print(f"  {field_label}: {current} -> {new_value}")

    if changed:
        return _publish_profile_change(
            profiles_path,
            data,
            f"Updated {profile_id}",
        )
    else:
        print("\nNo changes.")

    return 0


def _coerce_value(raw: str, target_type: type) -> Any:
    """Coerce a string value to the target type."""
    if target_type == bool:
        return raw.lower() in ("on", "true", "1", "yes")
    if target_type == int:
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"expected integer, got: {raw!r}")
    if target_type == float:
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"expected number, got: {raw!r}")
    return raw


# ── Subcommand: setup ───────────────────────────────────────────────────────

def _cmd_integrate_configure(args: argparse.Namespace) -> int:
    """Configure coding tools through the established provider adapter."""

    from .configure import main as configure_main

    configure_args: list[str] = []
    if getattr(args, "tool_names", None):
        configure_args.extend(["--tools", *args.tool_names])
    if getattr(args, "dry_run", False):
        configure_args.append("--dry-run")
    if getattr(args, "model", None):
        configure_args.extend(["--model", args.model])
    configure_args.extend(["--profiles", str(args.profiles)])
    return configure_main(configure_args)


def _cmd_integrate(args: argparse.Namespace) -> int:
    """Apply selected unit/provider integration independently for each profile."""

    if getattr(args, "dry_run", False) and getattr(args, "json", False):
        print("integrate --dry-run is interactive and cannot be combined with --json", file=sys.stderr)
        return 2
    mode = getattr(args, "integrate_command", None)
    clean_requested = bool(getattr(args, "clean", False))
    if mode == "configure":
        if clean_requested:
            print("integrate --clean requires the install mode", file=sys.stderr)
            return 2
        return _cmd_integrate_configure(args)
    install_requested = mode in (None, "install")
    configure_requested = mode is None
    if clean_requested and not install_requested:
        print("integrate --clean requires the install mode", file=sys.stderr)
        return 2
    if clean_requested and not getattr(args, "install_units", True):
        print(
            "integrate --clean requires unit installation; remove --no-install-units",
            file=sys.stderr,
        )
        return 2
    try:
        payload, inspections = inspect_manifest(
            args.profiles,
            verify_artifacts=bool(getattr(args, "full_verify", False)),
        )
    except (OSError, ProfileError, ValueError) as error:
        print(f"ERROR: integration cannot read profiles: {error}", file=sys.stderr)
        return 1

    manifest_profile_ids: frozenset[str] | None = None
    if clean_requested:
        try:
            manifest_profile_ids = _safe_manifest_profile_ids(payload)
        except (ProfileError, ValueError) as error:
            print(f"ERROR: integration cannot establish safe manifest ownership: {error}", file=sys.stderr)
            return 1

    ready = [item for item in inspections if item.status == "ready" and item.profile is not None]
    if not ready:
        result = report_for(inspections)
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            for item in inspections:
                print(f"{item.profile_id}: {item.status} ({item.reason})")
        return 1

    # A filtered registry lets valid profiles render independently.  Restore
    # targets that point at skipped profiles are removed rather than guessed.
    valid_ids = {item.profile.id for item in ready if item.profile is not None}

    def registry_for(profile_ids: set[str]) -> ProfileRegistry:
        profiles = tuple(
            replace(
                item.profile,
                restore_profile=(
                    item.profile.restore_profile
                    if item.profile.restore_profile in profile_ids
                    else None
                ),
            )
            for item in ready
            if item.profile is not None and item.profile.id in profile_ids
        )
        return ProfileRegistry(profiles)

    planned = list(inspections)
    controller = None
    if install_requested:
        registry = registry_for(valid_ids)
        try:
            controller = create_production_controller(registry)
        except (BackendControllerError, ProfileError, ValueError, OSError) as error:
            print(f"ERROR: integration controller unavailable: {error}", file=sys.stderr)
            return 1
        assert controller is not None

        for item in ready:
            path = controller.unit_path(item.profile.id)
            try:
                rendered = controller.render_unit(item.profile.id)
                if path.is_file() and path.read_text(encoding="utf-8") == rendered:
                    status = "unchanged"
                elif path.exists() or path.is_symlink():
                    status = "updated"
                else:
                    status = "installed"
            except (BackendControllerError, OSError, ValueError) as error:
                status = "broken"
                item = replace(item, reason=f"cannot render unit: {error}")
            for index, original in enumerate(planned):
                if original.profile_id == item.profile_id:
                    planned[index] = replace(item, status=status)
                    break

        renderable_ids = {
            item.profile_id
            for item in planned
            if item.status in {"installed", "updated", "unchanged"}
        }
        if renderable_ids != valid_ids:
            valid_ids = renderable_ids
            if valid_ids:
                try:
                    controller = create_production_controller(registry_for(valid_ids))
                except (BackendControllerError, ProfileError, ValueError, OSError) as error:
                    print(f"ERROR: integration controller unavailable: {error}", file=sys.stderr)
                    return 1
            else:
                controller = None

    provider_inspections = tuple(
        replace(item, status="ready")
        for item in planned
        if item.status in (
            {"installed", "updated", "unchanged"}
            if install_requested
            else {"ready"}
        )
    )
    systemd_ready = install_requested and bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS"))
    if install_requested and not systemd_ready:
        planned = [
            replace(item, status="skipped", reason="systemd user session unavailable")
            if item.status in {"installed", "updated", "unchanged"}
            else item
            for item in planned
        ]

    plan_errors: list[str] = []
    broker_unit_path = broker_runtime.SYSTEMD_USER_DIR / "llama-broker.service"
    broker_unit_text: str | None = None
    broker_unit_status = "skipped"
    if install_requested:
        try:
            template = _service_template_path()
            if not template.is_file():
                raise FileNotFoundError(f"service template not found: {template}")
            broker_unit_text = template.read_text(encoding="utf-8")
            broker_unit_text = broker_unit_text.replace("{{REPO_DIR}}", str(BROKER_DIR))
            broker_unit_text = broker_unit_text.replace("{{USER_HOME}}", str(Path.home()))
            broker_unit_text = broker_unit_text.replace("{{EXEC_START}}", _service_exec_start())
            if broker_unit_path.is_file() and broker_unit_path.read_text(encoding="utf-8") == broker_unit_text:
                broker_unit_status = "unchanged"
            elif broker_unit_path.exists() or broker_unit_path.is_symlink():
                broker_unit_status = "updated"
            else:
                broker_unit_status = "installed"
        except OSError as error:
            plan_errors.append(f"broker unit integration failed: {error}")

    provider_manifest = (
        filtered_manifest(payload, provider_inspections) if configure_requested else None
    )
    provider_status = "planned" if provider_manifest is not None else "skipped"
    operations: list[dict[str, object]] = [
        {
            "action": "profile_unit",
            "profile_id": item.profile_id,
            "status": item.status,
            "reason": item.reason,
        }
        for item in planned
    ]
    cleanup_report: dict[str, object] | None = None
    if clean_requested:
        cleanup_report = {
            "action": "stale_unit_cleanup",
            "requested": True,
            "status": "planned",
            "removed": [],
        }
        operations.append(cleanup_report)
    if not install_requested:
        operations = []
    if install_requested:
        operations.append(
            {
                "action": "broker_unit",
                "status": broker_unit_status,
                "path": str(broker_unit_path),
            }
        )
    if configure_requested:
        operations.append(
            {
                "action": "client/provider",
                "status": provider_status,
                "profiles": sorted(valid_ids),
            }
        )
    plan_report = report_for(tuple(planned))
    plan_report["systemd"] = (
        "available" if systemd_ready else "unavailable"
    ) if install_requested else "not requested"
    plan_report["operations"] = operations
    plan_report["broker_unit"] = broker_unit_status
    plan_report["client_provider"] = provider_status
    if plan_errors:
        plan_report["errors"] = list(plan_errors)
    apply_confirmed = not getattr(args, "dry_run", False)
    if getattr(args, "dry_run", False):
        print("Proposed integration plan:")
        for operation in operations:
            print(f"  - {operation['action']}: {operation}")
        try:
            answer = input("Apply this integration plan? [y/N] ").strip().casefold()
        except EOFError:
            answer = ""
        if answer not in {"y", "yes"}:
            print("Integration cancelled; no changes applied.")
            return 0
        apply_confirmed = True

    applied_errors: list[str] = list(plan_errors)
    unit_install_succeeded = False
    if install_requested:
        if systemd_ready and getattr(args, "install_units", True) and controller is not None:
            try:
                controller.install_units()
                unit_install_succeeded = True
            except (BackendControllerError, OSError, ValueError) as error:
                applied_errors.append(f"systemd unit integration failed: {error}")
        elif not systemd_ready:
            applied_errors.append("systemd user session unavailable; unit files were left untouched")
        elif controller is None:
            applied_errors.append("no profile rendered successfully; profile units were left untouched")

    if clean_requested:
        assert cleanup_report is not None
        cleanup_report["status"] = "skipped"
        if not systemd_ready:
            cleanup_report["reason"] = "systemd user session unavailable"
            applied_errors.append("stale unit cleanup skipped: systemd user session unavailable")
        elif not unit_install_succeeded:
            cleanup_report["reason"] = "profile unit installation did not complete"
            applied_errors.append("stale unit cleanup skipped: profile unit installation did not complete")
        elif controller is None:
            cleanup_report["reason"] = "no profile controller available"
            applied_errors.append("stale unit cleanup skipped: no profile controller available")
        else:
            try:
                assert manifest_profile_ids is not None
                removed_paths = tuple(
                    controller.cleanup_stale_units(
                        manifest_profile_ids=manifest_profile_ids,
                    )
                )
                cleanup_report["status"] = "removed" if removed_paths else "unchanged"
                cleanup_report["removed"] = [Path(path).name for path in removed_paths]
                cleanup_report["paths"] = [str(path) for path in removed_paths]
            except (BackendControllerError, OSError, ValueError) as error:
                cleanup_report["status"] = "failed"
                cleanup_report["error"] = str(error)
                applied_errors.append(f"stale unit cleanup failed: {error}")

    # The broker unit is lightweight control-plane integration.  It is written
    # independently of profile units and never starts a backend.
    if install_requested and broker_unit_text is not None:
        try:
            broker_unit_path.parent.mkdir(parents=True, exist_ok=True)
            if broker_unit_status != "unchanged":
                _atomic_write(broker_unit_path, broker_unit_text)
            plan_report["broker_unit"] = broker_unit_status
        except OSError as error:
            applied_errors.append(f"broker unit integration failed: {error}")

    # Provider configuration receives a temporary filtered manifest so broken
    # profiles never block valid providers and the temporary file is removed
    # after the adapter run.
    if configure_requested and provider_manifest is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="llama-broker-integrate-") as temp:
                temp_profiles = Path(temp) / "profiles.json"
                temp_profiles.write_text(json.dumps(provider_manifest), encoding="utf-8")
                from .configure import main as configure_main

                configure_args = ["--profiles", str(temp_profiles)]
                tool_names = list(getattr(args, "tool_names", ()))
                if tool_names:
                    configure_args[0:0] = ["--tools", *tool_names]
                if getattr(args, "model", None):
                    configure_args.extend(["--model", args.model])
                if apply_confirmed:
                    provider_exit = configure_main(configure_args, indent=True)
                    if provider_exit != 0:
                        applied_errors.append(f"provider integration returned exit code {provider_exit}")
                plan_report["client_provider"] = "applied" if not applied_errors else "partial"
        except (OSError, ValueError, RuntimeError) as error:
            applied_errors.append(f"provider integration failed: {error}")
    elif configure_requested:
        applied_errors.append("no verified profiles available for provider integration")

    if applied_errors:
        plan_report["errors"] = applied_errors
    partial = bool(applied_errors) or any(
        item.status in {"broken", "unverified", "skipped"} for item in planned
    )
    if getattr(args, "json", False):
        print(json.dumps(plan_report, indent=2, sort_keys=True))
    else:
        print("Integration summary:")
        for key, value in plan_report.get("counts", {}).items():
            if value:
                print(f"  {key}: {value}")
        if plan_report.get("broker_unit"):
            print(f"  broker unit: {plan_report['broker_unit']}")
        if plan_report.get("client_provider"):
            print(f"  client/provider: {plan_report['client_provider']}")
        if cleanup_report is not None:
            removed = cleanup_report.get("removed", [])
            removed_count = len(removed) if isinstance(removed, list) else 0
            print(
                f"  stale unit cleanup: {removed_count} removed "
                f"({cleanup_report['status']})"
            )
        for error in applied_errors:
            print(f"  ERROR: {error}", file=sys.stderr)
    return 3 if partial else 0


def _enable_broker_startup() -> tuple[bool, str]:
    """Enable only the lightweight control-plane unit, never a model unit."""

    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        return False, "systemd user session unavailable (DBUS_SESSION_BUS_ADDRESS is unset)"
    try:
        result = subprocess.run(
            ["systemctl", "--user", "enable", "llama-broker.service"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"could not enable broker startup: {error}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "systemctl failed").strip()
        return False, f"could not enable broker startup: {detail}"
    return True, "broker startup enabled at user login"


def _render_setup_plan(plan: object) -> str:
    """Render every setup mutation that may be applied after confirmation."""

    lines = ["Proposed setup plan:"]
    operations = getattr(plan, "operations", ())
    if not operations:
        lines.append("  (no changes; select a runtime and model explicitly)")
    for operation in operations:
        action = operation.get("action", "unknown")
        details = ", ".join(
            f"{key}={value}"
            for key, value in operation.items()
            if key != "action"
        )
        lines.append(f"  - {action}" + (f" ({details})" if details else ""))
    return "\n".join(lines)


def _confirm_setup_plan(plan: object) -> bool:
    """Ask for the final human confirmation required by setup."""

    print("\n" + _render_setup_plan(plan))
    try:
        answer = input("\nApply this plan? [y/N] ").strip().casefold()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _safe_manifest_profile_ids(payload: Mapping[str, Any]) -> frozenset[str]:
    """Return the complete manifest ID set or fail closed for cleanup."""
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raise ProfileError("unsafe manifest profile id set: profiles must be a list")
    profile_ids: list[str] = []
    for index, raw_profile in enumerate(raw_profiles):
        if not isinstance(raw_profile, dict):
            raise ProfileError(f"unsafe manifest profile id at index {index}: profile is not an object")
        profile_id = raw_profile.get("id")
        if not isinstance(profile_id, str) or not PROFILE_ID_RE.fullmatch(profile_id):
            raise ProfileError(f"unsafe manifest profile id at index {index}: {profile_id!r}")
        if profile_id in profile_ids:
            raise ProfileError(f"ambiguous manifest profile id: {profile_id!r} is duplicated")
        profile_ids.append(profile_id)
    return frozenset(profile_ids)


def _setup_referenced_paths(profiles_path: Path) -> list[Path]:
    references: list[Path] = []
    if profiles_path.is_file():
        payload: object = {}
        try:
            payload = json.loads(profiles_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict) and isinstance(payload.get("profiles"), list):
            for profile in payload["profiles"]:
                if not isinstance(profile, dict):
                    continue
                for field_name in ("server_binary", "model_path", "mmproj_path", "model_draft"):
                    value = profile.get(field_name)
                    if isinstance(value, str):
                        references.append(Path(value).expanduser())
                extra_args = profile.get("extra_args", [])
                if isinstance(extra_args, list):
                    for index, value in enumerate(extra_args):
                        if value == "--moe-cache-profile" and index + 1 < len(extra_args):
                            references.append(Path(str(extra_args[index + 1])).expanduser())
    raw_schema = payload.get("schema_version") if isinstance(payload, dict) else None
    strict_schema = (
        isinstance(raw_schema, int)
        and not isinstance(raw_schema, bool)
        and raw_schema in {1, 2}
    )
    try:
        if RUNTIME_CATALOG_PATH.is_file() and profiles_path.is_file() and strict_schema:
            registry = ProfileRegistry.from_path(profiles_path, allow_missing_artifacts=True)
            for profile in registry.profiles:
                references.extend(
                    path
                    for path in (
                        profile.server_binary,
                        profile.model_path,
                        profile.mmproj_path,
                        profile.model_draft,
                    )
                    if path is not None
                )
    except (ProfileError, OSError, ValueError):
        pass
    unique: dict[str, Path] = {}
    for path in references:
        unique[str(path)] = path
    return list(unique.values())


def _cmd_setup_wizard(args: argparse.Namespace) -> int:
    """Discover existing artifacts and apply only an explicitly reviewed plan."""

    catalog_was_missing = not RUNTIME_CATALOG_PATH.is_file()
    referenced_paths = _setup_referenced_paths(args.profiles)
    explicit_paths: list[Path] = []
    if getattr(args, "runtime_path", None) is not None:
        explicit_paths.append(Path(args.runtime_path).expanduser())
    if getattr(args, "model_path", None) is not None:
        explicit_paths.append(Path(args.model_path).expanduser())
    try:
        catalog_report = ensure_runtime_catalog(
            RUNTIME_CATALOG_PATH,
            inventory_roots=[RUNTIMES_DIR, MODELS_DIR],
            referenced_paths=referenced_paths,
            explicit_paths=explicit_paths,
        )
        print(
            f"Runtime catalog: {catalog_report['mode']} "
            f"(broad_scan={catalog_report['broad_scan']})"
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: runtime catalog setup failed: {error}", file=sys.stderr)
        return 1

    scan_paths = [Path(item).expanduser() for item in getattr(args, "scan_path", [])]
    runtime_path = getattr(args, "runtime_path", None)
    model_path = getattr(args, "model_path", None)
    if runtime_path is not None:
        runtime_candidate = Path(runtime_path).expanduser()
        scan_paths.append(runtime_candidate.parent if catalog_was_missing else runtime_candidate)
    if model_path is not None:
        model_candidate = Path(model_path).expanduser()
        scan_paths.append(model_candidate.parent if catalog_was_missing else model_candidate)
    if not scan_paths:
        if catalog_was_missing:
            scan_paths = [
                Path.home() / "bin",
                Path.home() / ".local" / "bin",
                RUNTIMES_DIR,
                MODELS_DIR,
            ]
        else:
            scan_paths = [path for path in referenced_paths if path.exists()]
        if catalog_was_missing and sys.stdin.isatty() and sys.stdout.isatty():
            print("Enter additional scan paths one at a time; press Enter when finished.")
            while True:
                try:
                    value = input("Scan path (blank to continue): ").strip()
                except EOFError:
                    break
                if not value:
                    break
                scan_paths.append(Path(value).expanduser())

    report = discover_candidates(scan_paths)
    if getattr(args, "check", False):
        print(render_report(report, json_output=bool(getattr(args, "json", False))))
        return 0

    if runtime_path is None and model_path is None and sys.stdin.isatty() and sys.stdout.isatty():
        print(render_report(report))
        qualified = [item for item in report.runtimes if item.qualified]
        choices = qualified or list(report.runtimes)
        if choices:
            for index, item in enumerate(choices, start=1):
                label = "qualified" if item.qualified else "experimental"
                print(f"  {index}. [{label}] {item.runtime}: {item.path}")
            selected = input("Select runtime number (blank to cancel): ").strip()
            if selected.isdigit() and 1 <= int(selected) <= len(choices):
                runtime_path = choices[int(selected) - 1].path
                if not choices[int(selected) - 1].qualified:
                    allow = input("Use experimental runtime? [y/N] ").strip().casefold()
                    if allow not in {"y", "yes"}:
                        runtime_path = None
        if report.models and runtime_path is not None:
            for index, item in enumerate(report.models, start=1):
                print(f"  {index}. {item.path}")
            selected = input("Select model number (blank to cancel): ").strip()
            if selected.isdigit() and 1 <= int(selected) <= len(report.models):
                model_path = report.models[int(selected) - 1].path

    startup = getattr(args, "enable_startup", None)
    if startup is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            startup = input("Enable broker startup at login? [Y/n] ").strip().casefold() not in {"n", "no"}
        else:
            startup = True

    plan = build_setup_plan(
        report,
        runtime_path=runtime_path,
        model_path=model_path,
        allow_experimental=bool(getattr(args, "allow_experimental", False)),
        create_model_link=bool(getattr(args, "link_model", False)),
        enable_startup=bool(startup),
        offer_integrate=bool(getattr(args, "integrate", False)),
    )
    if getattr(args, "dry_run", False):
        print(render_report(report))
        print("\n" + _render_setup_plan(plan))
        return 1 if plan.errors else 0
    if plan.errors:
        for error in plan.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if plan.profile_record is None:
        print(render_report(report))
        print("\n" + _render_setup_plan(plan))
        print("No profile was selected. Setup never chooses a model implicitly.", file=sys.stderr)
        return 2

    if not _confirm_setup_plan(plan):
        print("Setup cancelled; no changes applied.")
        return 0

    try:
        applied = apply_profile_plan(
            plan,
            args.profiles,
            models_dir=MODELS_DIR,
            create_model_link=bool(getattr(args, "link_model", False)),
        )
    except (ApplyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: setup apply failed: {error}", file=sys.stderr)
        return 1

    partial = False
    startup_result = "not requested"
    if plan.enable_startup:
        ok, startup_result = _enable_broker_startup()
        partial = not ok
        print(("OK: " if ok else "PARTIAL: ") + startup_result)
    print("Setup applied: " + ", ".join(applied.get("applied", [])))
    if not getattr(args, "integrate", False):
        print("Run `brokerctl integrate` to generate service/provider integration.")
    if getattr(args, "integrate", False):
        integration_result = _cmd_integrate(args)
        if partial or integration_result == 3:
            return 3
        return integration_result
    return 3 if partial else 0


def _cmd_setup(args: argparse.Namespace) -> int:
    """Regenerate units and configure providers without selecting a tool model."""
    clean = bool(getattr(args, "clean", False))
    if clean and not getattr(args, "install_units", True):
        print("brokerctl setup --clean requires unit installation; remove --no-install-units", file=sys.stderr)
        return 2
    print("Running idempotent setup...\n")
    errors = 0
    registry: ProfileRegistry | None = None

    # 0. Seed the default profiles path from the checked-in example when the
    # user has not created one yet (default path only; explicit --profiles
    # pointing at a missing file is an error).
    if not args.profiles.exists() and args.profiles == PROFILES_PATH:
        example = BROKER_DIR / "profiles.example.json"
        if example.is_file():
            try:
                args.profiles.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(example, args.profiles)
                print(
                    f"  Seeded {args.profiles} from profiles.example.json "
                    "(edit paths before loading profiles)"
                )
            except OSError as error:
                print(f"  ERROR: could not seed profiles: {error}", file=sys.stderr)
                errors += 1
        else:
            print(
                "  WARN: profiles.example.json not found; skipping seed",
                file=sys.stderr,
            )

    # 0. Ensure native llama-server credentials exist before providers are
    # configured. The key itself is never printed; providers read it from the
    # path-only security configuration.
    print("[0/4] Preparing native llama-server API key...")
    try:
        security, _api_key = ensure_security_material()
        _apply_setup_network_defaults(security)
        print(f"  API-key file: {security.api_key_file}")
        print("  Native key ready (value redacted)")
    except (OSError, ValueError) as error:
        print(f"  ERROR: {error}", file=sys.stderr)
        errors += 1

    # 1. Read the catalog without touching the currently selected backend.
    print("[1/4] Reading available models (catalog-only)...")
    try:
        registry = ProfileRegistry.from_path(args.profiles)
        print(f"  Catalog profiles: {len(registry.profiles)} (none selected implicitly)")
    except (ProfileError, OSError, ValueError) as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        errors += 1

    if getattr(args, "install_units", True):
        print("  Regenerating broker-owned units...")
        if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
            print("  SKIPPED: DBUS_SESSION_BUS_ADDRESS not set")
            print("  (run with DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus)")
            if clean:
                print("  ERROR: --clean requires a user systemd session", file=sys.stderr)
                errors += 1
        else:
            try:
                if registry is None:
                    registry = ProfileRegistry.from_path(args.profiles)
                controller = create_production_controller(registry)
                installed = controller.install_units()
                for profile_id, path in installed.items():
                    print(f"  {profile_id} -> {path}")
                if not installed:
                    print("  (already up to date)")
                if clean:
                    removed = controller.cleanup_stale_units()
                    if removed:
                        for path in removed:
                            print(f"  removed stale unit {path}")
                    else:
                        print("  no stale broker-owned units")
            except (ProfileError, BackendControllerError, OSError, ValueError) as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                errors += 1
            except Exception as e:
                print(f"  WARN: systemd units skipped ({e})")
    else:
        print("  Broker-owned units unchanged (--no-install-units was supplied).")

    # 1.5 Render + install the broker daemon unit from the checked-in template.
    if getattr(args, "install_units", True):
        print("  Installing broker daemon unit...")
        template = _service_template_path()
        unit_text = template.read_text(encoding="utf-8")
        unit_text = unit_text.replace("{{REPO_DIR}}", str(BROKER_DIR))
        unit_text = unit_text.replace("{{USER_HOME}}", str(Path.home()))
        unit_text = unit_text.replace("{{EXEC_START}}", _service_exec_start())
        unit_path = broker_runtime.SYSTEMD_USER_DIR / "llama-broker.service"
        try:
            unit_path.parent.mkdir(parents=True, exist_ok=True)
            unit_path.write_text(unit_text, encoding="utf-8")
            print(f"  {unit_path} <- {template.name}")
        except OSError as error:
            print(f"  ERROR: could not write unit: {error}", file=sys.stderr)
            errors += 1

    # 2. Configure tools
    print("\n[2/4] Configuring tools...")
    try:
        from .configure import main as configure_main
        configure_args = ["--profiles", str(args.profiles)]
        configure_exit = configure_main(configure_args, indent=True)
        if configure_exit != 0:
            errors += 1
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        errors += 1

    # 3. Verify profiles
    if args.verify:
        print("\n[3/4] Verifying profiles...")
        try:
            if registry is None:
                registry = ProfileRegistry.from_path(args.profiles)
            all_failures = []
            for profile in registry.profiles:
                failures = _verify_profile(profile)
                if failures:
                    all_failures.extend(failures)
                else:
                    print(f"  {profile.id}: OK")
            if all_failures:
                for f in all_failures:
                    print(f"  FAIL: {f}", file=sys.stderr)
                errors += 1
        except (ProfileError, OSError) as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            errors += 1
    else:
        print("\n[3/4] Verifying profiles... SKIPPED (use --verify to check)")

    # 4. Create symlinks
    print("\n[4/4] Creating symlinks...")
    brokerctl_path = BROKER_DIR / "brokerctl.py"
    target = BIN_DIR / "brokerctl"
    try:
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(brokerctl_path)
        target.chmod(0o755)
        print(f"  {target} -> {brokerctl_path}")
    except OSError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        errors += 1

    print(f"\n{'Setup complete.' if errors == 0 else f'Setup finished with {errors} error(s).'}")
    return errors


def _apply_setup_network_defaults(security: Any) -> None:
    """Make offline unit generation match the TLS-enabled service template."""

    if getattr(security, "broker_tls_cert_file", None) is not None and not any(
        name in os.environ for name in ("LLAMA_BROKER_HOST", "LLM_BROKER_HOST")
    ):
        broker_runtime.BROKER_HOST = "0.0.0.0"
    if (
        getattr(security, "backend_tls_cert_file", None) is not None
        and not any(
            name in os.environ
            for name in (
                "LLAMA_BROKER_BACKEND_LISTEN_HOST",
                "LLM_BROKER_BACKEND_LISTEN_HOST",
            )
        )
    ):
        broker_runtime.BACKEND_LISTEN_HOST = "0.0.0.0"


def _cmd_security_init(args: argparse.Namespace) -> int:
    """Create local TLS material and the path-only security configuration."""

    result = initialize_security_material(
        ip_addresses=args.ip,
        hostnames=args.hostname,
        security_path=args.security_file,
        api_key_path=args.api_key_file,
        days=args.days,
        force=args.force,
    )
    action = "created" if result.created else "already present"
    print(f"Security material {action}.")
    print(f"  Security config: {result.config.source_path}")
    print(f"  API-key file:    {result.config.api_key_file}")
    print(f"  CA certificate:  {result.config.broker_tls_ca_file}")
    print(f"  Broker cert:     {result.config.broker_tls_cert_file}")
    print(f"  Backend cert:    {result.config.backend_tls_cert_file}")
    print(f"  SANs:            {', '.join(result.sans)}")
    print(
        "  API-key value is redacted; brokerctl setup refreshes broker-owned "
        "provider credentials."
    )
    print("Next: brokerctl setup; use --no-install-units only to skip unit regeneration.")
    return 0


# ── Subcommands: model list and profiles list ───────────────────────────────

_DEFAULT_TABLE_WIDTH = 80
_MIN_TABLE_WIDTH = 40
_MIN_COLUMN_WIDTH = 8
_COLUMN_GUTTER = 2


def _terminal_width(explicit: int | None = None) -> int:
    """Best available terminal width in columns, with a sane floor.

    *explicit* wins when given; otherwise the controlling terminal is probed
    and a conservative default is used for pipes and non-terminals.
    """

    columns = explicit if explicit is not None else shutil.get_terminal_size().columns
    return max(int(columns), _MIN_TABLE_WIDTH)


def _wrap_cell(text: str, width: int) -> list[str]:
    """Word-wrap one cell so every line fits in *width* columns."""

    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = ""
        for word in words:
            if len(word) > width:
                # One token longer than the whole column: hard-break it so the
                # table keeps its shape instead of silently overflowing.
                for chunk in range(0, len(word), width):
                    piece = word[chunk : chunk + width]
                    if not line:
                        line = piece
                    else:
                        lines.append(line)
                        line = piece
                continue
            if not line:
                line = word
                continue
            if len(line) + 1 + len(word) > width:
                lines.append(line)
                line = word
            else:
                line += " " + word
        if line:
            lines.append(line)
    return lines


def _cell_width(text: str) -> int:
    """Natural width of one cell: its longest raw line, ignoring wrapping.

    Word-wrapping turns long text into *rows*, not *columns*, so a cell only
    contributes as much width as its widest single line needs.
    """

    return max((len(line) for line in text.split("\n")), default=0)


def _word_width(text: str) -> int:
    """Widest single word in a cell; the floor below which wrapping hard-breaks."""

    return max((len(word) for word in text.split()), default=0)


def _allocate_widths(
    natural: list[int], word_widths: list[int], budget: int
) -> list[int]:
    """Fit *natural* column widths into *budget* columns.

    Phase 1 shrinks the widest columns' slack toward their word floor (no
    hard-breaking, just more rows); phase 2, only when a word is wider than
    the terminal, shrinks below the floor so wrapping hard-breaks it. Either
    way the table fits the budget instead of overflowing.
    """

    widths = list(natural)
    gutters = _COLUMN_GUTTER * (len(widths) - 1)
    floors = [max(w, _MIN_COLUMN_WIDTH) for w in word_widths]

    while sum(widths) + gutters > budget:
        # Phase 1: give up slack above the word floor, widest first.
        candidates = [i for i, w in enumerate(widths) if w > floors[i]]
        if not candidates:
            break
        widest = max(candidates, key=lambda i: widths[i] - floors[i])
        widths[widest] -= 1
    # Phase 2: a word is wider than the terminal; hard-break below the floor.
    while sum(widths) + gutters > budget:
        candidates = [i for i, w in enumerate(widths) if w > _MIN_COLUMN_WIDTH]
        if not candidates:
            break
        widest = max(candidates, key=lambda i: widths[i])
        widths[widest] -= 1
    return widths


def _render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    width: int = _DEFAULT_TABLE_WIDTH,
) -> str:
    """Render a small dependency-free table for human-facing CLI output.

    Columns size to their content when the table fits within *width* columns;
    otherwise long cells wrap vertically so the output always fits the terminal
    instead of scrolling sideways.
    """

    rendered_rows = [[str(value) for value in row] for row in rows]
    rendered_headers = [str(header) for header in headers]
    columns = len(rendered_headers)

    def column_natural(index: int) -> int:
        # As wide as the widest single line in the column.
        return max(
            _cell_width(rendered_headers[index]),
            max((_cell_width(row[index]) for row in rendered_rows), default=0),
        )

    def column_word(index: int) -> int:
        # As wide as the widest single word in the column.
        return max(
            _word_width(rendered_headers[index]),
            max((_word_width(row[index]) for row in rendered_rows), default=0),
        )

    # Natural width of each column: as wide as its widest single line.
    natural = [column_natural(index) for index in range(columns)]
    word_widths = [column_word(index) for index in range(columns)]
    if sum(natural) + _COLUMN_GUTTER * (columns - 1) > width:
        natural = _allocate_widths(natural, word_widths, width)
    widths = natural
    header_cells = [_wrap_cell(header, widths[index]) for index, header in enumerate(rendered_headers)]
    row_cells = [
        [_wrap_cell(cell, widths[index]) for index, cell in enumerate(row)] for row in rendered_rows
    ]

    def render_block(cells: list[list[str]]) -> list[str]:
        height = max(len(cell) for cell in cells)
        lines: list[str] = []
        for line_index in range(height):
            parts = [
                cell[line_index] if line_index < len(cell) else ""
                for cell in cells
            ]
            line = (
                _COLUMN_GUTTER * " "
            ).join(
                part.ljust(widths[index]) if index < columns - 1 else part
                for index, part in enumerate(parts)
            )
            lines.append(line.rstrip())
        return lines

    parts: list[str] = []
    parts.extend(render_block(header_cells))
    parts.extend(render_block([["-" * widths[index]] for index in range(columns)]))
    for row in row_cells:
        parts.extend(render_block(row))
    return "\n".join(parts)


def _model_payload(profile: Profile) -> dict[str, object]:
    """Return one canonical, client-facing model record for CLI consumers."""

    return {
        "id": profile.alias,
        "display_name": profile.display_name,
        "runtime": profile.runtime,
        "capabilities": sorted(profile.capabilities),
        "context_size": profile.context_size,
        "parallel": profile.parallel,
    }


def _cmd_model_list(args: argparse.Namespace) -> int:
    """List client-facing model references backed by canonical profiles."""

    registry = ProfileRegistry.from_path(args.profiles)
    models = [_model_payload(profile) for profile in registry.profiles]
    if getattr(args, "json", False):
        print(
            json.dumps(
                {"schema_version": 1, "object": "model_list", "data": models},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not models:
        print("No models configured.")
        return 0

    rows = [
        (
            profile.alias,
            profile.display_name,
            profile.runtime,
            ", ".join(sorted(profile.capabilities)),
            f"{profile.context_size:,}",
            profile.parallel,
        )
        for profile in registry.profiles
    ]
    print(
        _render_table(
            (
                "PROFILE",
                "MODEL",
                "RUNTIME",
                "CAPABILITIES",
                "CONTEXT",
                "SLOTS",
            ),
            rows,
            width=_terminal_width(getattr(args, "width", None)),
        )
    )
    return 0

def _cmd_profiles_list(args: argparse.Namespace) -> int:
    """List profiles with status."""
    registry = ProfileRegistry.from_path(args.profiles)
    for profile in registry.profiles:
        caps = ",".join(sorted(profile.capabilities))
        slot_ctx = profile.slot_context_window
        print(
            f"{profile.alias:<30} role={profile.role:<12} "
            f"ctx={profile.context_size:<8} parallel={profile.parallel} "
            f"slot_ctx={slot_ctx:<8} caps=[{caps}]"
        )
    return 0


def _autotune_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _autotune_unit_directives(profile: Profile) -> Sequence[str]:
    service = profile.safety.service
    lines = [
        f"MemoryMax={service.memory_max_mib}M",
        f"MemorySwapMax={service.memory_swap_max_mib}M",
        f"TasksMax={service.tasks_max}",
        "KillMode=control-group",
        "SendSIGKILL=no",
    ]
    import shlex

    environment = launch_environment(
        profile.environment,
        runtime=profile.runtime,
        autotune=True,
    )
    lines.extend(
        f"Environment={shlex.quote(f'{key}={value}')}"
        for key, value in sorted(environment.items())
    )
    return lines


def _run_live_autotune(
    profile: Profile,
    schema: Any,
    candidates: Sequence[Candidate],
    *,
    time_budget: float,
    lease: MaintenanceLease | None = None,
    progress: Callable[[str, Candidate, int, int, BenchmarkResult | None], None] | None = None,
) -> Any:
    """Run one profile's candidates through disposable local systemd units."""

    security = load_security_config()
    security.validate(require_files=True)
    backend_url = effective_backend_url(security)
    api_keys = security.authentication_keys()
    api_key = api_keys[-1] if api_keys else None
    endpoint: dict[str, Any] = {
        "port": None,
        "alias": None,
        "max_output_tokens": profile.default_max_output_tokens,
    }
    last_heartbeat = [0.0]

    def keep_lease_alive() -> None:
        if lease is None:
            return
        now = time.monotonic()
        if now - last_heartbeat[0] >= 10.0:
            lease.heartbeat()
            last_heartbeat[0] = now

    base_parts = urlsplit(backend_url)
    fixture_path: Path | None = None
    if "vision" in profile.capabilities:
        fixture_fd, fixture_name = tempfile.mkstemp(prefix="broker-autotune-", suffix=".png")
        os.close(fixture_fd)
        fixture_path = Path(fixture_name)
        fixture_path.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            )
        )

    def candidate_url() -> str:
        port = endpoint.get("port")
        if not isinstance(port, int):
            raise RuntimeError("candidate endpoint was not allocated")
        return base_parts._replace(
            netloc=f"127.0.0.1:{port}", path="", query="", fragment=""
        ).geturl()

    def command_for(candidate: Candidate) -> Sequence[str]:
        port = _autotune_loopback_port()
        alias = f"autotune-{profile.id}-{candidate.index}"
        endpoint.update(
            {
                "port": port,
                "alias": alias,
                "max_output_tokens": candidate.changes.get(
                    "max_output_tokens", profile.default_max_output_tokens
                ),
            }
        )
        ephemeral = replace(
            profile,
            alias=alias,
            aliases=(alias,),
            **dict(candidate.changes),
        )
        managed_args = security.backend_server_args(
            listen_host="127.0.0.1",
            listen_port=port,
            runtime=profile.runtime,
        )
        return ephemeral.command(
            host="127.0.0.1", port=port, managed_args=managed_args
        )

    def request_factory(_unit: str, mode: str, streaming: bool, concurrent: bool) -> Any:
        keep_lease_alive()
        alias = endpoint.get("alias")
        if not isinstance(alias, str):
            raise RuntimeError("candidate alias was not allocated")
        return HttpCandidateSession(
            candidate_url(),
            alias,
            image_fixture=fixture_path,
            timeout=min(30.0, max(5.0, time_budget)),
            ca_file=security.backend_tls_ca_file,
            ssl_context=security.backend_client_context(),
            api_key=api_key,
            max_output_tokens=int(endpoint["max_output_tokens"]),
        ).request(mode, streaming=streaming, concurrent=concurrent)

    def readiness_probe(_candidate: Candidate) -> bool:
        alias = endpoint.get("alias")
        if not isinstance(alias, str):
            return False
        deadline = time.monotonic() + max(10.0, min(float(profile.startup_timeout), 300.0))
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        while time.monotonic() < deadline:
            keep_lease_alive()
            try:
                with BackendClient(
                    candidate_url(),
                    timeout=1.0,
                    ca_file=security.backend_tls_ca_file,
                    ssl_context=security.backend_client_context(),
                    default_headers=headers,
                ) as client:
                    payload = client.list_models()
                models = payload.get("data", payload.get("models", []))
                if isinstance(models, list) and any(
                    isinstance(item, dict) and item.get("id") == alias for item in models
                ):
                    return True
            except (OSError, TypeError, ValueError, http.client.HTTPException):
                pass
            time.sleep(0.25)
        return False

    def unit_directives(_candidate: Candidate) -> Sequence[str]:
        return _autotune_unit_directives(profile)

    lifecycle = SystemdCandidateLifecycle(
        command_for,
        request_factory,
        unit_dir=SYSTEMD_USER_DIR,
        readiness_probe=readiness_probe,
        unit_directives_for=unit_directives,
    )
    try:
        harness = BenchmarkHarness(lifecycle, image_fixture=fixture_path)

        def benchmark(candidate: Candidate, baseline: bool) -> BenchmarkResult:
            return harness.run(
                candidate,
                capabilities=profile.capabilities,
                parallelism=profile.parallel,
                baseline=baseline,
            )

        return AutotuneEngine(
            schema=schema,
            candidates=candidates,
            benchmark=benchmark,
            max_seconds=time_budget,
            progress=progress,
        ).run()
    finally:
        if fixture_path is not None:
            fixture_path.unlink(missing_ok=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _candidate_changes(profile: Profile, candidate: Candidate) -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}
    for field_name, after in candidate.changes.items():
        before = getattr(profile, field_name, None)
        if before != after:
            changes[field_name] = {
                "before": _json_value(before),
                "after": _json_value(after),
            }
    before_extra = tuple(str(value) for value in getattr(profile, "extra_args", ()))
    after_extra = tuple(str(value) for value in candidate.extra_args)
    if before_extra != after_extra:
        changes["extra_args"] = {
            "before": _json_value(before_extra),
            "after": _json_value(after_extra),
        }
    return changes


def _autotune_metrics(
    result: BenchmarkResult, profile: Profile | None = None
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "candidate": result.candidate.index,
        "strategy": result.candidate.strategy,
        "valid": result.valid,
        "aggregate_tokens_per_second": result.aggregate_tokens_per_second,
        "p95_latency_seconds": result.p95_latency_seconds,
        "best_latency_seconds": result.best_latency_seconds,
        "operational_errors": list(result.operational_errors),
        "modes": [
            {
                "mode": mode.mode,
                "scored_requests": mode.scored_requests,
                "aggregate_tokens_per_second": mode.aggregate_tokens_per_second,
                "p95_latency_seconds": mode.p95_latency_seconds,
                "p95_ttft_seconds": mode.p95_ttft_seconds,
                "latency_mean_seconds": mode.latency_mean_seconds,
                "latency_stddev_seconds": mode.latency_stddev_seconds,
                "latency_ci95_seconds": (
                    list(mode.latency_ci95_seconds)
                    if mode.latency_ci95_seconds is not None
                    else None
                ),
                "throughput_stddev_tokens_per_second": mode.throughput_stddev_tokens_per_second,
                "throughput_ci95_tokens_per_second": (
                    list(mode.throughput_ci95_tokens_per_second)
                    if mode.throughput_ci95_tokens_per_second is not None
                    else None
                ),
                "streaming_decode_tokens_per_second": mode.streaming_decode_tokens_per_second,
                "operational_errors": list(mode.operational_errors),
                "valid": mode.valid,
                "repeats": mode.repeats,
            }
            for mode in result.modes
        ],
    }
    if profile is not None:
        metrics["changes"] = _candidate_changes(profile, result.candidate)
    return metrics


def _autotune_report(profile: Profile, search: Any, *, status: str = "complete", error: str | None = None) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": 1,
        "profile_id": profile.id,
        "runtime": profile.runtime,
        "accuracy": "not_measured",
        "status": status,
    }
    if error is not None:
        report["operational_errors"] = [error]
        return report
    report.update(
        {
            "stopped_reason": search.stopped_reason,
            "experiment_count": len(search.evaluated),
            "baseline": _autotune_metrics(search.baseline, profile),
            "winner": _autotune_metrics(search.winner, profile),
            "best_latency": _autotune_metrics(search.best_latency, profile),
            "winner_changes": _candidate_changes(profile, search.winner.candidate),
            "experiments": [_autotune_metrics(item, profile) for item in search.evaluated],
        }
    )
    return report


def _cmd_profiles_autotune(args: argparse.Namespace) -> int:
    if args.max_experiments <= 0 or args.max_experiments > 64:
        raise ProfileError("--max-experiments must be in 1..64")
    if args.time_budget <= 0:
        raise ProfileError("--time-budget must be positive")
    manifest = load_manifest_source(args.profiles, allow_missing_artifacts=True)
    registry = ProfileRegistry.from_path(args.profiles, allow_missing_artifacts=True)
    profiles = [registry.resolve(args.profile_id)] if args.profile_id else list(registry.profiles)
    records = manifest.get("profiles")
    if not isinstance(records, list):
        raise ProfileError("profile manifest must contain a profiles list")
    record_by_id = {
        item.get("id"): item
        for item in records
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }

    eligible: list[Profile] = []
    reports: list[dict[str, object]] = []
    for profile in profiles:
        record = record_by_id.get(profile.id)
        metadata = profile.autotune if isinstance(profile.autotune, Mapping) else {}
        complete = (
            metadata.get("status") == "complete"
            and metadata.get("profile_revision") == profile_revision(record or {})
        )
        if complete and not args.force:
            reports.append({
                "schema_version": 1,
                "profile_id": profile.id,
                "runtime": profile.runtime,
                "status": "skipped",
                "reason": "complete; use --force to retune",
            })
            continue
        if profile.runtime not in ADAPTERS:
            reports.append({
                "schema_version": 1,
                "profile_id": profile.id,
                "runtime": profile.runtime,
                "status": "skipped",
                "reason": f"unsupported runtime: {profile.runtime}",
            })
            continue
        eligible.append(profile)

    active_state = _broker_admin_get("/admin/status")
    backend_state = active_state.get("backend", {})
    previous_active = backend_state.get("profile") if isinstance(backend_state, dict) else None
    active_to_stop = previous_active if isinstance(previous_active, str) and eligible else None
    if active_to_stop is not None and not args.yes:
        estimated = sum(
            len(generate_candidates(
                profile,
                discover_schema(profile),
                max_candidates=args.max_experiments,
            ))
            for profile in eligible
        )
        message = (
            f"active profile {active_to_stop!r} must be stopped for autotune; "
            f"{len(eligible)} profile(s), up to {estimated} candidate experiment(s). "
            "Re-run with --yes to proceed."
        )
        if args.json:
            print(json.dumps({"schema_version": 1, "status": "confirmation_required", "message": message}, sort_keys=True))
        else:
            print(message, file=sys.stderr)
        return 2

    winners: dict[str, tuple[Candidate, Any]] = {}
    winner_metadata: dict[str, Mapping[str, Any]] = {}
    live_failures: set[str] = set()
    active_succeeded = False
    aborted = False
    report_dir = broker_runtime.CONFIG_DIR / "autotune"

    def restore_active() -> None:
        if not isinstance(previous_active, str):
            return
        if active_to_stop is not None and active_to_stop not in live_failures:
            _broker_admin("/admin/switch", {"model": previous_active, "wait": True})

    try:
        with MaintenanceLease(broker_runtime.AUTOTUNE_LEASE_PATH, ttl_seconds=60.0) as lease:
            if active_to_stop is not None:
                _broker_admin("/admin/stop", {})
            for profile in eligible:
                schema = discover_schema(profile, include_fit=False)
                candidates = generate_candidates(
                    profile, schema, max_candidates=args.max_experiments
                )
                if not args.json:
                    print(f"Autotuning {profile.id}: {len(candidates)} candidate(s)")

                def progress(
                    phase: str,
                    candidate: Candidate,
                    completed: int,
                    total: int,
                    result: BenchmarkResult | None,
                ) -> None:
                    if args.json:
                        return
                    ordinal = completed + 1 if phase == "started" else completed
                    if phase == "started":
                        print(
                            f"[{ordinal}/{total}] candidate {candidate.index} "
                            f"({candidate.strategy}) starting...",
                            flush=True,
                        )
                        return
                    if result is None:
                        print(
                            f"[{ordinal}/{total}] candidate {candidate.index} finished",
                            flush=True,
                        )
                        return
                    state = "complete" if result.valid else "failed"
                    detail = (
                        f"throughput={result.aggregate_tokens_per_second:.2f} tok/s "
                        f"p95={result.p95_latency_seconds:.3f}s"
                        if result.valid
                        else f"errors={','.join(result.operational_errors) or 'invalid'}"
                    )
                    print(
                        f"[{ordinal}/{total}] candidate {candidate.index} {state}; {detail}",
                        flush=True,
                    )
                try:
                    search = _run_live_autotune(
                        profile,
                        schema,
                        candidates,
                        time_budget=args.time_budget,
                        lease=lease,
                        progress=progress,
                    )
                    report = _autotune_report(profile, search)
                    report_path = report_dir / f"{profile.id}-{time.time_ns()}.json"
                    atomic_write_json(report_path, report)
                    record = record_by_id.get(profile.id, {})
                    tuned_record = ADAPTERS[profile.runtime].apply(
                        record, search.winner.candidate
                    )
                    metadata = {
                        "status": "complete",
                        "winner_candidate": search.winner.candidate.index,
                        "strategy": search.winner.candidate.strategy,
                        "confidence": search.winner.candidate.metadata.get("confidence", "high"),
                        "report": str(report_path),
                        "last_success_report": str(report_path),
                        "profile_revision": profile_revision(tuned_record),
                        "experiment_count": len(search.evaluated),
                        "updated_at": time.time(),
                        "accuracy": "not_measured",
                    }
                    winners[profile.id] = (search.winner.candidate, ADAPTERS[profile.runtime])
                    winner_metadata[profile.id] = metadata
                    reports.append({
                        "profile_id": profile.id,
                        "runtime": profile.runtime,
                        "status": "complete" if not args.dry_run else "dry_run",
                        "report": str(report_path),
                        "winner": _autotune_metrics(search.winner, profile),
                        "winner_changes": _candidate_changes(profile, search.winner.candidate),
                        "experiment_count": len(search.evaluated),
                    })
                    if profile.id == active_to_stop:
                        active_succeeded = True
                except MaintenanceLeaseExpired:
                    raise
                except Exception as error:
                    live_failures.add(profile.id)
                    error_name = type(error).__name__
                    report = _autotune_report(profile, None, status="failed", error=error_name)
                    report_path = report_dir / f"{profile.id}-{time.time_ns()}-failed.json"
                    atomic_write_json(report_path, report)
                    reports.append({
                        "profile_id": profile.id,
                        "runtime": profile.runtime,
                        "status": "failed",
                        "error": error_name,
                        "report": str(report_path),
                    })
                    if profile.id == active_to_stop and not args.json:
                        print(f"{profile.id}: failed; leaving broker stopped", file=sys.stderr)
    except (KeyboardInterrupt, MaintenanceLeaseExpired) as error:
        aborted = True
        live_failures.update(profile.id for profile in eligible)
        if not args.json:
            print(f"autotune aborted ({type(error).__name__}); no winners were committed", file=sys.stderr)
        if active_to_stop is not None and isinstance(previous_active, str):
            try:
                _broker_admin("/admin/switch", {"model": previous_active, "wait": True})
            except Exception as restore_error:
                if not args.json:
                    print(f"failed to restore active profile: {type(restore_error).__name__}", file=sys.stderr)
    finally:
        if not aborted and active_to_stop is not None and active_succeeded and (args.dry_run or not winners):
            restore_active()

    if winners and not args.dry_run and not aborted:
        def reload_registry() -> None:
            _broker_admin("/admin/autotune/reload", {})

        def restore_after_reload(active: str | None) -> None:
            if active is not None and active_to_stop not in live_failures:
                _broker_admin("/admin/switch", {"model": active, "wait": True})

        def quiescent() -> bool:
            status = _broker_admin_get("/admin/status")
            leases = status.get("leases")
            return isinstance(leases, dict) and leases.get("active_requests", 0) == 0

        def mark_degraded(reason: str) -> None:
            _broker_admin("/admin/autotune/degraded", {"reason": reason})

        with MaintenanceLease(broker_runtime.AUTOTUNE_LEASE_PATH, ttl_seconds=60.0):
            transaction = AutotuneTransaction(
                Path(args.profiles),
                reload_registry=reload_registry,
                restore_active=restore_after_reload,
                quiescent=quiescent,
                mark_degraded=mark_degraded,
            )
            committed = transaction.commit_many(
                manifest,
                winners=winners,
                previous_active=previous_active if active_to_stop not in live_failures else None,
                metadata=winner_metadata,
            )
        output: dict[str, object] = {"schema_version": 1, "profiles": reports, "commit": committed}
    else:
        output = {"schema_version": 1, "profiles": reports}

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        for item in reports:
            winner = item.get("winner")
            if isinstance(winner, dict):
                print(
                    f"{item.get('profile_id')}: {item.get('status')} "
                    f"candidate={winner.get('candidate')} "
                    f"throughput={winner.get('aggregate_tokens_per_second', 0):.2f} tok/s "
                    f"p95={winner.get('p95_latency_seconds', 0):.3f}s "
                    f"report={item.get('report')}"
                )
                changes = item.get("winner_changes")
                if isinstance(changes, Mapping):
                    if changes:
                        print("  winner changes:")
                        for field_name, change in changes.items():
                            if isinstance(change, Mapping):
                                print(
                                    f"    {field_name}: {change.get('before')!r} "
                                    f"-> {change.get('after')!r}"
                                )
                    else:
                        print("  winner changes: none (baseline retained)")
            else:
                print(f"{item.get('profile_id')}: {item.get('status')}")
    return 1 if live_failures else 0


def _cmd_profiles_show(args: argparse.Namespace) -> int:
    registry, data = _load_editable_manifest(
        args.profiles, allow_missing_artifacts=True
    )
    _profile, record = _record_for_selector(registry, data, args.profile_id)
    shown = copy.deepcopy(record)
    inferred = infer_runtime_build(
        data, record, allow_missing_artifacts=True
    )
    if "runtime_build" not in shown and inferred is not None:
        shown["runtime_build"] = inferred.id
    print(json.dumps(shown, indent=2, ensure_ascii=False))
    return 0


def _cmd_profiles_add_entry(args: argparse.Namespace) -> int:
    registry, data = _load_editable_manifest(args.profiles)
    records = data.get("profiles")
    if not isinstance(records, list):
        raise ProfileError("profile manifest must contain a profiles list")
    if any(isinstance(record, dict) and record.get("id") == args.profile_id for record in records):
        raise ProfileError(f"duplicate profile id: {args.profile_id}")
    records.append(_build_added_profile(args, data))
    del registry
    return _publish_profile_change(args.profiles, data, f"Added profile {args.profile_id}")


def _cmd_profiles_remove(args: argparse.Namespace) -> int:
    registry, data = _load_editable_manifest(args.profiles)
    profile, record = _record_for_selector(registry, data, args.profile_id)
    records = data.get("profiles")
    if not isinstance(records, list) or len(records) <= 1:
        raise ProfileError("cannot remove the only profile in a manifest")
    records.remove(record)
    for other in records:
        if isinstance(other, dict) and other.get("restore_profile") == profile.id:
            other.pop("restore_profile", None)
    return _publish_profile_change(args.profiles, data, f"Removed profile {profile.id}")


def _cmd_profiles_rename(args: argparse.Namespace) -> int:
    registry, data = _load_editable_manifest(args.profiles)
    profile, record = _record_for_selector(registry, data, args.profile_id)
    new_id = args.new_id.strip()
    if not PROFILE_ID_RE.fullmatch(new_id):
        raise ProfileError(f"invalid profile id: {new_id!r}")
    records = data.get("profiles")
    if not isinstance(records, list):
        raise ProfileError("profile manifest must contain a profiles list")
    if new_id != profile.id and any(
        isinstance(item, dict) and item.get("id") == new_id for item in records
    ):
        raise ProfileError(f"duplicate profile id: {new_id}")
    if new_id == profile.id:
        print("No changes.")
        return 0
    record["id"] = new_id
    for item in records:
        if isinstance(item, dict) and item.get("restore_profile") == profile.id:
            item["restore_profile"] = new_id
    return _publish_profile_change(args.profiles, data, f"Renamed profile {profile.id} to {new_id}")


def _cmd_profiles_set(args: argparse.Namespace) -> int:
    registry, data = _load_editable_manifest(args.profiles)
    _profile, record = _record_for_selector(registry, data, args.profile_id)
    before = copy.deepcopy(record)
    _set_scalar_field(record, args.field, args.value)
    if record == before:
        print("No changes.")
        return 0
    return _publish_profile_change(
        args.profiles, data, f"Updated {args.field} for profile {args.profile_id}"
    )


def _cmd_profiles_unset(args: argparse.Namespace) -> int:
    registry, data = _load_editable_manifest(args.profiles)
    _profile, record = _record_for_selector(registry, data, args.profile_id)
    before = copy.deepcopy(record)
    _remove_optional_field(record, args.field)
    if record == before:
        print("No changes.")
        return 0
    return _publish_profile_change(
        args.profiles, data, f"Cleared {args.field} for profile {args.profile_id}"
    )


def _cmd_profiles_alias(args: argparse.Namespace) -> int:
    registry, data = _load_editable_manifest(args.profiles)
    profile, record = _record_for_selector(registry, data, args.profile_id)
    aliases = record.setdefault("aliases", [])
    if not isinstance(aliases, list):
        raise ProfileError(f"aliases must be a list for {profile.id!r}")
    alias = args.alias.strip()
    if not alias:
        raise ProfileError("alias must not be empty")
    if args.alias_command == "add":
        if any(str(item).casefold() == alias.casefold() for item in [record.get("alias"), *aliases]):
            print("No changes.")
            return 0
        aliases.append(alias)
        return _publish_profile_change(
            args.profiles, data, f"Added alias {alias!r} to profile {profile.id}"
        )
    if alias.casefold() == str(record.get("alias", "")).casefold():
        raise ProfileError("cannot remove the primary alias; set alias first")
    kept = [item for item in aliases if str(item).casefold() != alias.casefold()]
    if len(kept) == len(aliases):
        raise ProfileError(f"alias not found for profile {profile.id}: {alias}")
    record["aliases"] = kept
    return _publish_profile_change(
        args.profiles, data, f"Removed alias {alias!r} from profile {profile.id}"
    )


def _cmd_profiles_env(args: argparse.Namespace) -> int:
    registry, data = _load_editable_manifest(args.profiles)
    profile, record = _record_for_selector(registry, data, args.profile_id)
    environment = record.setdefault("environment", {})
    if not isinstance(environment, dict):
        raise ProfileError(f"environment must be a mapping for {profile.id!r}")
    if args.env_command == "set":
        key, value = _parse_env_assignment(args.assignment)
        if environment.get(key) == value:
            print("No changes.")
            return 0
        environment[key] = value
        return _publish_profile_change(
            args.profiles, data, f"Set environment {key} for profile {profile.id}"
        )
    key = args.key.strip()
    if not ENVIRONMENT_NAME_RE.fullmatch(key):
        raise ProfileError(f"invalid environment name: {key!r}")
    if key in REQUIRED_LLAMACPP_ENVIRONMENT and profile.runtime == "llamacpp":
        raise ProfileError(f"cannot unset required safety environment variable: {key}")
    if key not in environment:
        raise ProfileError(f"environment variable not found for profile {profile.id}: {key}")
    del environment[key]
    return _publish_profile_change(
        args.profiles, data, f"Unset environment {key} for profile {profile.id}"
    )


def _cmd_profiles_args(args: argparse.Namespace) -> int:
    registry, data = _load_editable_manifest(args.profiles)
    profile, record = _record_for_selector(registry, data, args.profile_id)
    extra_args = record.setdefault("extra_args", [])
    if not isinstance(extra_args, list):
        raise ProfileError(f"extra_args must be a list for {profile.id!r}")
    if args.args_command == "add":
        values = list(args.flags)
        if values and values[0] == "--":
            values = values[1:]
        if not values or any(not value for value in values):
            raise ProfileError("args add requires at least one non-empty flag")
        extra_args.extend(values)
        return _publish_profile_change(
            args.profiles, data, f"Added arguments to profile {profile.id}"
        )
    flag = args.flag
    try:
        index = extra_args.index(flag)
    except ValueError as error:
        raise ProfileError(f"argument not found for profile {profile.id}: {flag}") from error
    del extra_args[index]
    if index < len(extra_args) and not str(extra_args[index]).startswith("--"):
        del extra_args[index]
    return _publish_profile_change(
        args.profiles, data, f"Removed argument {flag!r} from profile {profile.id}"
    )


def _cmd_profiles_template(args: argparse.Namespace) -> int:
    ProfileRegistry.from_path(args.profiles)
    source = args.profiles.resolve()
    destination = (BROKER_DIR / "profiles.json").resolve()
    if source == destination:
        raise ProfileError("authoritative manifest and repository template are the same file")
    _atomic_write(destination, args.profiles.read_text(encoding="utf-8"))
    print(f"Synced repository template {destination} from {source}")
    return 0


def _cmd_runtime_list(args: argparse.Namespace) -> int:
    """List registered runtimes and the profile-backed models selecting each."""

    registry = ProfileRegistry.from_path(args.profiles)
    catalog_builds = registry.runtime_builds
    runtime_rows: list[dict[str, object]] = []
    for runtime in list_runtimes():
        profiles = [
            profile
            for profile in registry.profiles
            if profile.runtime == runtime.name
        ]
        builds: list[dict[str, object]] = []
        seen_builds: set[str] = set()
        for build in catalog_builds:
            if build.runtime != runtime.name or build.id in seen_builds:
                continue
            seen_builds.add(build.id)
            builds.append(
                {
                    "id": build.id,
                    "display_name": build.display_name,
                    "runtime": build.runtime,
                    "bundle_root": str(build.bundle_root) if build.bundle_root else None,
                    "server_binary": str(build.server_binary),
                    "server_sha256": build.server_sha256,
                    "environment": dict(build.environment),
                    "source_repository": (
                        str(build.source_repository) if build.source_repository else None
                    ),
                    "provenance_file": (
                        str(build.provenance_file) if build.provenance_file else None
                    ),
                    "source_branch": build.source_branch,
                    "source_commit": build.source_commit,
                    "dirty": build.dirty,
                    "selected_profile_ids": list(build.selected_profile_ids),
                    "aliases": list(build.aliases),
                    "deprecated": build.deprecated,
                }
            )
        runtime_rows.append(
            {
                "name": runtime.name,
                "capabilities": list(runtime.capabilities),
                "profiles": [
                    {"id": profile.id, "model": profile.alias}
                    for profile in profiles
                ],
                "builds": builds,
            }
        )

    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "schema_version": 2,
                    "object": "runtime_list",
                    "catalog_path": (
                        str(registry.runtime_catalog_path)
                        if registry.runtime_catalog_path
                        else None
                    ),
                    "deployment_revision": registry.deployment_revision,
                    "data": runtime_rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not runtime_rows:
        print("No runtimes registered.")
        return 0

    # One row per (family, bundle) pair so the table reads:
    # family -> bundle -> the profiles using that bundle. The family cell
    # spans its bundle rows (blanked on continuation rows).
    rows: list[tuple[str, str, str]] = []
    for row in runtime_rows:
        family = str(row["name"])
        family_profiles = [
            profile
            for profile in registry.profiles
            if profile.runtime == family
        ]
        # Catalog build ids for this family, in catalog order.
        catalog_builds = [
            str(build["id"]) for build in row["builds"]  # type: ignore[index]
        ]
        # Ordered, deduplicated bundle list: catalog order first, then any
        # profile references not in the catalog.
        ordered: list[str] = []
        for build_id in catalog_builds:
            if build_id not in ordered:
                ordered.append(build_id)
        for profile in family_profiles:
            bundle_id = profile.runtime_build
            if bundle_id is not None and bundle_id not in ordered:
                ordered.append(bundle_id)
        bundles = ordered if ordered else ["(none)"]
        # Profiles grouped under each bundle, in registry order.
        profiles_by_bundle: dict[str, list[str]] = {}
        for profile in family_profiles:
            profiles_by_bundle.setdefault(
                profile.runtime_build or "(none)", []
            ).append(profile.alias)
        for bundle in bundles:
            aliases = profiles_by_bundle.get(bundle)
            rows.append(
                (
                    family,
                    bundle,
                    ", ".join(aliases) if aliases else "(none)",
                )
            )
    # Blank the family cell on continuation rows (one family per visual block).
    rendered: list[tuple[str, str, str]] = []
    last_family: str | None = None
    for family, bundle, profiles in rows:
        rendered.append(
            ("" if family == last_family else family, bundle, profiles)
        )
        last_family = family
    print(
        _render_table(
            ("RUNTIME", "BUNDLE", "PROFILES"),
            rendered,
            width=_terminal_width(getattr(args, "width", None)),
        )
    )
    return 0


def _cmd_runtime_control(args: argparse.Namespace) -> int:
    root = Path(args.control_root).expanduser().resolve()
    if args.runtime_command == "project":
        if args.project_command == "add": result = project_add(root, args.name, args.url, default_branch=args.default_branch)
        elif args.project_command == "list": result = project_list(root)
        elif args.project_command == "fetch": result = project_fetch(root, args.name)
        else: result = project_commits(root, args.name, args.ref)
    else:
        if args.bundle_command == "plan": result = bundle_plan(root, args.name, branch=args.branch, commit=args.commit, allow_network=args.allow_network)
        elif args.bundle_command == "build": result = bundle_build(root, args.name, branch=args.branch, commit=args.commit, allow_network=args.allow_network, jobs=args.jobs)
        elif args.bundle_command == "list": result = bundle_list(root)
        else: result = bundle_verify(root, args.name)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _configure_after_credential_change(args: argparse.Namespace) -> None:
    """Refresh configured tools after the native key file has changed."""

    from .configure import main as configure_main

    result = configure_main(["--profiles", str(args.profiles)])
    if result != 0:
        raise BackendControllerError("tool configuration failed after credential change")


def _cmd_security_rotate(args: argparse.Namespace) -> int:
    result = _broker_admin(
        "/admin/credentials/rotate",
        {"replace": bool(args.replace)},
    )
    _configure_after_credential_change(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_security_revoke(args: argparse.Namespace) -> int:
    result = _broker_admin(
        "/admin/credentials/revoke",
        {"key": args.key},
    )
    _configure_after_credential_change(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


# ── Subcommand: profiles discover ───────────────────────────────────────────

def _cmd_profiles_discover(args: argparse.Namespace) -> int:
    """Discover backends via port sweep."""
    result = discover_backends(
        sweep_range=args.sweep,
        dirty_only=args.dirty_only,
        timeout=args.timeout,
    )

    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "object": "discovery",
                    "data": asdict(result),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result.endpoints else 1

    if not result.endpoints:
        print("No backends discovered.")
        if result.failed_ports:
            print(f"Failed ports: {', '.join(map(str, result.failed_ports))}")
        return 1

    print(f"Discovered {len(result.endpoints)} backend(s) in {result.duration_ms:.0f}ms:\n")
    for ep in result.endpoints:
        print(f"  {ep.label}")
        for model in ep.models:
            ctx = f"ctx={model.context_length}" if model.context_length else ""
            par = f"parallel={model.parallel}" if model.parallel else ""
            slot = f"slot_ctx={model.slot_context_window}" if model.slot_context_window else ""
            parts = " ".join(filter(None, [ctx, par, slot]))
            print(f"    {model.id}  {parts}")

    if args.save:
        print(f"\nSaved state to port_state.json")

    return 0


# ── Subcommand: profiles edit ───────────────────────────────────────────────

def _select_profile_for_edit(profiles_path: Path) -> str | None:
    """Prompt for an explicit profile selection before opening the editor."""
    manifest = load_manifest_source(profiles_path, allow_missing_artifacts=True)
    profiles = [
        profile
        for profile in manifest.get("profiles", [])
        if isinstance(profile, dict) and isinstance(profile.get("id"), str)
    ]
    if not profiles:
        print("No profiles configured.", file=sys.stderr)
        return None

    print("Select a profile to edit:")
    for index, profile in enumerate(profiles, start=1):
        profile_id = str(profile["id"])
        alias = str(profile.get("alias", ""))
        display_name = str(profile.get("display_name", alias or profile_id))
        runtime_build = str(profile.get("runtime_build", profile.get("runtime", "")))
        labels = [display_name, profile_id]
        if alias and alias not in labels:
            labels.append(alias)
        if runtime_build:
            labels.append(f"runtime={runtime_build}")
        print(f"  {index}. {' | '.join(labels)}")

    by_selector = {
        selector: str(profile["id"])
        for profile in profiles
        for selector in (profile["id"], profile.get("alias"))
        if isinstance(selector, str) and selector
    }
    while True:
        try:
            choice = input("Profile number or ID (blank/q to cancel): ").strip()
        except EOFError:
            print("\nEdit cancelled.")
            return None
        if not choice or choice.casefold() in {"q", "quit", "cancel"}:
            print("Edit cancelled.")
            return None
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(profiles):
                return str(profiles[index - 1]["id"])
        elif choice in by_selector:
            return by_selector[choice]
        print("Invalid selection; choose a listed number, profile ID, or alias.", file=sys.stderr)


def _cmd_profiles_edit(args: argparse.Namespace) -> int:
    """Open the field-oriented editor or apply a compatibility flag edit."""
    flags = {
        "runtime": args.runtime,
        "runtime_build": args.runtime_build,
        "strata_artifacts_manifest_path": args.strata_artifacts_manifest_path,
        "strata_launch_config_path": args.strata_launch_config_path,
        "context_size": args.context_size,
        "parallel": args.parallel,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "tool_call_parser": args.tool_call_parser,
        "tool_parser_plugin": args.tool_parser_plugin,
        "chat_template": args.chat_template,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "fit": args.fit,
        "fit_target_mib": args.fit_target_mib,
        "fit_ctx": args.fit_ctx,
    }
    # If any flags were provided, skip interactive mode
    has_flags = any(v is not None for v in flags.values())
    if has_flags:
        if not args.model:
            print("profiles edit flags require a profile id or alias", file=sys.stderr)
            return 2
        return _interactive_edit(args.model, args.profiles, flags, flag_mode=True)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            "profiles edit requires an interactive terminal; use the field editor from a TTY "
            "or provide explicit edit flags",
            file=sys.stderr,
        )
        return 2

    from .broker_tui import run_tui

    selected_model = args.model
    if selected_model is None:
        selected_model = _select_profile_for_edit(args.profiles)
        if selected_model is None:
            return 0

    def apply_callback(updated: dict[str, object], changed_ids: set[str]) -> object:
        return _apply_profile_configuration(args.profiles, updated, changed_ids)

    return run_tui(args.profiles, selected_model, apply_callback=apply_callback)


# ── Subcommand: profiles add ────────────────────────────────────────────────

def _cmd_profiles_add(args: argparse.Namespace) -> int:
    """Add a complete profile, retaining the legacy discovery-only form."""
    if args.model is not None:
        return _cmd_profiles_add_entry(args)
    if sys.stdin.isatty() and sys.stdout.isatty():
        from .broker_tui import run_guided_wizard

        def apply_callback(updated: dict[str, object], changed_ids: set[str]) -> object:
            return _apply_profile_configuration(args.profiles, updated, changed_ids)

        return run_guided_wizard(args.profiles, args.profile_id, apply_callback=apply_callback)
    # Discover backends to find the model
    result = discover_backends(timeout=args.timeout)
    if not result.endpoints:
        print("No backends discovered. Start a backend first.", file=sys.stderr)
        return 1

    # Find matching model
    target = args.profile_id
    found = None
    for ep in result.endpoints:
        for model in ep.models:
            if model.id == target or target in model.id:
                found = (ep, model)
                break
        if found:
            break

    if not found:
        print(f"Model '{target}' not found on any discovered backend.")
        print("Available models:")
        for ep in result.endpoints:
            for model in ep.models:
                print(f"  {model.id} @ {ep.label}")
        return 1

    ep, model = found

    # Check if already in profiles
    data = json.loads(args.profiles.read_text(encoding="utf-8"))
    for p in data.get("profiles", []):
        if p.get("id") == model.id or p.get("alias") == model.id:
            print(f"Profile already exists: {model.id}")
            return 0

    print(
        "Refusing to auto-add an execution profile: discovery cannot prove the "
        "executable/model paths, SHA-256 identities, cgroup limits, CUDA safety "
        "environment, or native device-placement policy.",
        file=sys.stderr,
    )
    print("Discovery hints (not a valid execution profile):", file=sys.stderr)
    print(
        json.dumps(
            {
                "id": model.id,
                "alias": model.id,
                "context_size": model.context_length,
                "parallel": model.parallel,
                "source": ep.label,
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    print("Review profiles.json and add a complete pinned profile manually.", file=sys.stderr)
    return 2


def _restart_broker_control_plane() -> None:
    """Restart the broker service so it reloads the authoritative manifest.

    Profile units are broker-owned; restarting a profile unit directly would
    bypass the coordinator's request leases and can strand a live request.
    """

    try:
        result = subprocess.run(
            ["systemctl", "--user", "restart", "llama-broker.service"],
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
            env=os.environ.copy(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BackendControllerError(f"failed to restart Llama-Broker control plane: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "systemctl returned a failure").strip()
        raise BackendControllerError(f"Llama-Broker restart failed: {detail}")


def _apply_profile_configuration(
    profiles_path: Path,
    updated_manifest: dict[str, object],
    changed_ids: set[str],
) -> object:
    """Apply a wizard/TUI draft under the same lifecycle lock as model switches."""

    before_registry = ProfileRegistry.from_path(profiles_path)
    active_profile: str | None = None
    try:
        state = create_production_controller(before_registry).state()
        if state.get("phase") == "active":
            raw_active = state.get("profile")
            active_profile = str(raw_active) if raw_active else None
    except (BackendControllerError, OSError, KeyError):
        # A stopped broker has no active lease; edits remain safe and units
        # can be installed for the next controlled start.
        active_profile = None

    def install_units() -> object:
        if _is_authoritative_manifest(profiles_path):
            # The live broker reload endpoint installs its own generated
            # units after replacing the in-memory registry.
            return None
        registry = ProfileRegistry.from_path(profiles_path)
        return create_production_controller(registry).install_units()

    def apply_transaction() -> object:
        try:
            return apply_manifest_transaction(
                profiles_path,
                updated_manifest,
                active_profile=active_profile,
                install_units=install_units,
                restart_broker=_restart_broker_control_plane,
                reload_broker=lambda: _refresh_live_profile_registry(profiles_path),
            )
        except ApplyError:
            raise

    if _is_authoritative_manifest(profiles_path):
        return apply_transaction()
    with _operation_lock(LOCK_PATH):
        return apply_transaction()


# ── Argument parser ─────────────────────────────────────────────────────────

def _add_integrate_options(
    parser: argparse.ArgumentParser,
    *,
    nested: bool = False,
    allow_clean: bool = True,
    allow_install_units: bool = True,
) -> None:
    """Add integration options to the combined and explicit integration modes."""

    default = argparse.SUPPRESS if nested else False
    parser.add_argument("--dry-run", action="store_true", default=default, help="show the plan and ask before applying")
    parser.add_argument("--json", action="store_true", default=default, help="emit a versioned noninteractive report")
    parser.add_argument(
        "--full-verify",
        "--verify",
        dest="full_verify",
        action="store_true",
        default=default,
        help="verify artifact paths and hashes before applying (slow)",
    )
    if allow_clean:
        parser.add_argument("--clean", action="store_true", default=default, help="remove only stale broker-owned units")
    if allow_install_units:
        parser.add_argument(
            "--no-install-units",
            dest="install_units",
            action="store_false",
            default=argparse.SUPPRESS if nested else True,
            help=argparse.SUPPRESS,
        )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brokerctl",
        description="Administrative CLI for broker-owned immutable backend profiles.",
    )
    parser.add_argument("--profiles", type=Path, default=PROFILES_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    model = subparsers.add_parser(
        "model", help="inspect profile-backed client-facing models"
    )
    model_sub = model.add_subparsers(dest="model_command", required=True)
    model_list = model_sub.add_parser(
        "list", help="list canonical model references from the profile manifest"
    )
    model_list.add_argument("--json", action="store_true", help="emit a versioned machine-readable report")
    model_list.add_argument(
        "--width",
        type=int,
        help="target output width in columns; the table wraps to fit instead of overflowing the terminal (default: detected terminal width)",
    )

    status = subparsers.add_parser("status", help="show durable backend state")
    status.add_argument("-v", "--verbose", action="store_true", help="show full JSON state")
    status.add_argument("--json", action="store_true", help="emit a versioned machine-readable report")

    load = subparsers.add_parser("load", help="queue a model profile load")
    load.add_argument("model")
    load.add_argument("--raw", action="store_true", help="output raw JSON")
    start = subparsers.add_parser("start", help="alias for load (queue a profile load)")
    start.add_argument("model")
    start.add_argument("--raw", action="store_true", help="output raw JSON")
    restart = subparsers.add_parser("restart", help="reload active model or start a new one")
    restart.add_argument("model", nargs="?")
    restart.add_argument("--json", action="store_true", help="emit a versioned machine-readable report")

    unload = subparsers.add_parser("unload", help="unload the active model (stop the backend)")
    unload.add_argument("--raw", action="store_true", help="output raw JSON")

    operation = subparsers.add_parser(
        "operation", aliases=["operations"], help="inspect, wait for, or cancel a transition request"
    )
    operation_sub = operation.add_subparsers(dest="operation_command", required=True)
    operation_status = operation_sub.add_parser("status", help="show one transition request")
    operation_status.add_argument("request_id")
    operation_sub.add_parser("list", help="list pending and recently completed operations")
    operation_sub.add_parser("tui", help="open the interactive operation monitor")
    operation_wait = operation_sub.add_parser("wait", help="wait for one transition request to finish")
    operation_wait.add_argument("request_id")
    operation_wait.add_argument("--timeout", type=float, default=3600.0)
    operation_wait.add_argument("--interval", type=float, default=0.5)
    operation_cancel = operation_sub.add_parser("cancel", help="cancel one transition request")
    operation_cancel.add_argument("request_id")

    verify = subparsers.add_parser("verify", help="verify pinned runtime/model/projector SHA-256")
    verify.add_argument("model", nargs="?")
    verify.add_argument("--json", action="store_true", help="emit a versioned machine-readable report")

    clean = subparsers.add_parser(
        "clean",
        help="list unused models, orphaned runtime bundles and old backups; --apply cleans them",
        description=(
            "Lists unused models, orphaned runtime bundles and old manifest backups, "
            "then asks before cleaning them (scripts and --json only get the list). "
            "--apply cleans without asking. Cleaning deletes; with --archive-dir, models "
            "and backups are copied there and verified first. Package caches are "
            "included only when `caches` is named."
        ),
    )
    clean.add_argument(
        "categories",
        nargs="*",
        metavar="CATEGORY",
        help=f"any of {', '.join(CLEAN_CATEGORIES)} (default: {', '.join(DEFAULT_CLEAN_CATEGORIES)})",
    )
    clean.add_argument("--apply", action="store_true", help="clean the listed items without asking")
    clean.add_argument(
        "--archive-dir",
        type=Path,
        help="copy models and backups here before removing them, instead of deleting; "
        "must be on another disk (or set LLAMA_BROKER_ARCHIVE_DIR)",
    )
    clean.add_argument("--min-size-mib", type=int, default=256, help="ignore smaller unused model files (default 256)")
    clean.add_argument("--min-age-days", type=float, default=7.0, help="leave anything modified more recently (default 7)")
    clean.add_argument("--keep-backups", type=int, default=10, help="newest backups kept per manifest (default 10)")
    clean.add_argument(
        "--keep",
        action="append",
        default=[],
        metavar="PATH",
        help=f"never touch PATH (repeatable; also read from {CONFIG_DIR / CLEAN_KEEP_FILENAME})",
    )
    clean.add_argument("--json", action="store_true", help="emit a versioned machine-readable report")

    # Public setup is the discovery wizard. It never selects a runtime profile
    # implicitly.
    setup = subparsers.add_parser(
        "setup",
        help="discover existing runtimes/models and create explicitly selected profiles",
    )
    setup.add_argument("--scan-path", action="append", default=[], type=Path)
    setup.add_argument("--runtime-path", "--select-runtime", dest="runtime_path", type=Path)
    setup.add_argument("--model-path", "--select-model", dest="model_path", type=Path)
    setup.add_argument("--allow-experimental", action="store_true")
    setup.add_argument("--link-model", action="store_true", help="create a reversible model symlink")
    setup.add_argument("--check", action="store_true", help="discover and report without mutating")
    setup.add_argument("--dry-run", action="store_true", help="show the discovery/apply plan without mutating")
    startup_group = setup.add_mutually_exclusive_group()
    startup_group.add_argument("--enable-startup", dest="enable_startup", action="store_true")
    startup_group.add_argument("--no-enable-startup", dest="enable_startup", action="store_false")
    setup.set_defaults(enable_startup=None)
    setup.add_argument("--integrate", action="store_true", help="run integrate after applying setup")
    setup.add_argument("--verify", action="store_true", help=argparse.SUPPRESS)
    setup.add_argument("--clean", action="store_true", help=argparse.SUPPRESS)
    setup.add_argument("--no-install-units", dest="install_units", action="store_false", default=True, help=argparse.SUPPRESS)


    integrate = subparsers.add_parser(
        "integrate",
        help="install broker units and configure external providers",
    )
    _add_integrate_options(integrate)
    integrate.add_argument(
        "--model",
        help="explicitly select this broker profile alias for configured tools",
    )
    integrate_sub = integrate.add_subparsers(dest="integrate_command")

    integrate_install = integrate_sub.add_parser(
        "install", help="install broker-owned units without configuring providers"
    )
    _add_integrate_options(
        integrate_install,
        nested=True,
    )

    integrate_configure = integrate_sub.add_parser(
        "configure", help="configure coding tools without installing units"
    )
    integrate_configure.add_argument(
        "tool_names",
        nargs="*",
        help="tools to configure (default: all except Codex; name codex explicitly to opt in)",
    )
    integrate_configure.add_argument("--dry-run", action="store_true", help="preview without writing")
    integrate_configure.add_argument(
        "--model",
        help="explicitly select this broker profile alias for configured tools",
    )

    runtime = subparsers.add_parser("runtime", help="inspect registered serving runtimes")
    runtime_sub = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_list = runtime_sub.add_parser(
        "list", help="list runtimes and the profile-backed models selecting each"
    )
    runtime_list.add_argument("--json", action="store_true", help="emit a versioned machine-readable report")
    runtime_list.add_argument(
        "--width",
        type=int,
        help="target output width in columns; the table wraps to fit instead of overflowing the terminal (default: detected terminal width)",
    )
    runtime.add_argument("--control-root", type=Path, default=Path.home() / ".local/share/llama-broker/runtime-control")
    project = runtime_sub.add_parser("project", help="register and inspect reviewed source projects")
    project_sub = project.add_subparsers(dest="project_command", required=True)
    pa = project_sub.add_parser("add"); pa.add_argument("name"); pa.add_argument("url"); pa.add_argument("--default-branch")
    project_sub.add_parser("list")
    pf = project_sub.add_parser("fetch"); pf.add_argument("name")
    pc = project_sub.add_parser("commits"); pc.add_argument("name"); pc.add_argument("--ref")
    bundle = runtime_sub.add_parser("bundle", help="plan, build, list, or verify immutable CPU bundles")
    bundle_sub = bundle.add_subparsers(dest="bundle_command", required=True)
    for command in ("plan", "build"):
        bp = bundle_sub.add_parser(command); bp.add_argument("name"); bp.add_argument("--branch"); bp.add_argument("--commit"); bp.add_argument("--allow-network", action="store_true")
        if command == "build": bp.add_argument("--jobs", type=int, default=2)
    bundle_sub.add_parser("list")
    bv = bundle_sub.add_parser("verify"); bv.add_argument("name")

    security = subparsers.add_parser(
        "security", help="provision and rotate deployment-owned API-key and TLS material"
    )
    security_sub = security.add_subparsers(dest="security_command", required=True)
    security_init = security_sub.add_parser(
        "init", help="create a local CA and broker/backend TLS certificates"
    )
    security_init.add_argument(
        "--ip",
        "--lan-ip",
        dest="ip",
        action="append",
        required=True,
        help="LAN IP to include in the certificate SAN; repeat for multiple addresses",
    )
    security_init.add_argument(
        "--hostname",
        action="append",
        default=[],
        help="DNS name to include in the certificate SAN; repeat as needed",
    )
    security_init.add_argument(
        "--security-file",
        type=Path,
        default=None,
        help="security.json path (default: configured/default user path)",
    )
    security_init.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="native API-key path (default: configured/default user path)",
    )
    security_init.add_argument(
        "--days",
        type=int,
        default=825,
        help="certificate lifetime in days (default: 825)",
    )
    security_init.add_argument(
        "--force",
        action="store_true",
        help="replace existing generated TLS material",
    )

    rotate_security = security_sub.add_parser(
        "rotate", help="generate a new key and restart the active backend"
    )
    rotate_security.add_argument(
        "--replace",
        action="store_true",
        help="revoke existing keys immediately instead of retaining overlap",
    )
    revoke_security = security_sub.add_parser(
        "revoke", help="revoke one native backend API key"
    )
    revoke_security.add_argument("key")

    # Profiles remain the authoritative model-to-runtime binding manifest.
    profiles = subparsers.add_parser("profile", help="profile manifest management")
    profiles_sub = profiles.add_subparsers(dest="profiles_command", required=True)

    profiles_sub.add_parser("list", help="list profiles with status")
    render = profiles_sub.add_parser("render", help="render a profile unit without installing it")
    render.add_argument("model")
    show = profiles_sub.add_parser("show", help="show one complete profile as JSON")
    show.add_argument("profile_id")

    autotune = profiles_sub.add_parser(
        "autotune", help="benchmark bounded profile candidates and persist a winner"
    )
    autotune.add_argument("profile_id", nargs="?", help="optional profile id; omit to tune all profiles")
    autotune.add_argument("--force", action="store_true", help="retune profiles already marked complete")
    autotune.add_argument("--dry-run", action="store_true", help="run the real benchmark without persisting winners")
    autotune.add_argument("--yes", action="store_true", help="allow graceful stop of the active profile")
    autotune.add_argument("--max-experiments", type=int, default=64, help="maximum candidates per profile")
    autotune.add_argument("--time-budget", type=float, default=1800.0, help="maximum seconds per profile")
    autotune.add_argument("--json", action="store_true", help="emit clean machine-readable output")

    discover = profiles_sub.add_parser("discover", help="discover backends via port sweep")
    discover.add_argument("--dirty-only", action="store_true", help="only recheck stale ports")
    discover.add_argument("--sweep", help="port range to scan (e.g. 8000-12000)")
    discover.add_argument("--timeout", type=float, default=2.0, help="probe timeout in seconds")
    discover.add_argument("--save", action="store_true", help="persist results")
    discover.add_argument("--json", action="store_true", help="emit a versioned machine-readable report")

    edit = profiles_sub.add_parser("edit", help="open the field-oriented profile editor")
    edit.add_argument("model", nargs="?", help="profile id or alias (default: first profile)")
    edit.add_argument("--runtime", choices=runtime_names(), help="set serving runtime")
    edit.add_argument("--runtime-build", help="select a concrete cataloged runtime build")
    edit.add_argument("--strata-artifacts-manifest-path", help="set Strata artifact manifest path")
    edit.add_argument("--strata-launch-config-path", help="set Strata launch config path")
    edit.add_argument("--context-size", help="set context_size")
    edit.add_argument("--parallel", help="set parallel requests")
    edit.add_argument("--tensor-parallel-size", help="set vLLM tensor parallel size")
    edit.add_argument("--gpu-memory-utilization", help="set vLLM GPU memory utilization")
    edit.add_argument("--kv-cache-dtype", help="set vLLM KV cache dtype")
    edit.add_argument("--tool-call-parser", help="set vLLM tool-call parser")
    edit.add_argument("--tool-parser-plugin", help="set vLLM tool parser plugin")
    edit.add_argument("--chat-template", help="set vLLM chat template")
    edit.add_argument("--batch-size", help="set batch_size")
    edit.add_argument("--ubatch-size", help="set micro-batch size")
    edit.add_argument("--cache-type-k", help="set KV cache type (k)")
    edit.add_argument("--cache-type-v", help="set KV cache type (v)")
    edit.add_argument(
        "--fit",
        help="set auto-fit on, or remove all fit-family arguments with off/unset/native",
    )
    edit.add_argument("--fit-target-mib", help="set fit target MiB")
    edit.add_argument("--fit-ctx", help="set fit context size")

    add = profiles_sub.add_parser(
        "add",
        help="add a complete profile, or inspect a legacy discovered candidate",
    )
    add.add_argument("profile_id", help="new profile id (or legacy discovered model id)")
    add.add_argument("--model", help="absolute local model path")
    add.add_argument("--alias", action="append", help="profile alias; repeat for more aliases")
    add.add_argument("--display-name", help="human-readable display name")
    add.add_argument("--role", choices=sorted(ALLOWED_ROLES), default=None)
    add.add_argument("--runtime-build", help="select a concrete cataloged runtime build")
    add.add_argument("--server-binary", help="absolute pinned runtime server binary path")
    add.add_argument("--strata-artifacts-manifest", help="pinned Strata artifact manifest")
    add.add_argument("--strata-launch-config", help="pinned Strata launch config")
    add.add_argument("--context", type=int, help="context size")
    add.add_argument("--parallel", type=int, help="parallel slots")
    add.add_argument("--batch", type=int, help="batch size")
    add.add_argument("--ubatch", type=int, help="micro-batch size")
    add.add_argument("--cache-k", choices=CLI_CACHE_TYPES, help="K cache type")
    add.add_argument("--cache-v", choices=CLI_CACHE_TYPES, help="V cache type")
    add.add_argument("--flash", choices=("on", "off"), help="flash attention")
    add.add_argument("--gpu-layers", help="all, auto, or a non-negative layer count")
    add.add_argument("--mmproj", help="absolute vision projector path")
    add.add_argument("--restore", help="profile id to restore after this profile")
    add.add_argument("--env", action="append", help="environment assignment KEY=VALUE")
    add.add_argument("--arg", dest="extra_args", action="append", help="one extra server argument")
    add.add_argument("--timeout", type=float, default=2.0, help="probe timeout in seconds")

    remove = profiles_sub.add_parser("remove", help="remove a profile by id or alias")
    remove.add_argument("profile_id")

    rename = profiles_sub.add_parser("rename", help="rename a profile id")
    rename.add_argument("profile_id")
    rename.add_argument("new_id")

    set_field = profiles_sub.add_parser("set", help="set one scalar profile field")
    set_field.add_argument("profile_id")
    set_field.add_argument("field")
    set_field.add_argument("value")

    unset = profiles_sub.add_parser("unset", help="clear one optional profile field")
    unset.add_argument("profile_id")
    unset.add_argument("field")

    alias = profiles_sub.add_parser("alias", help="manage additional aliases")
    alias_sub = alias.add_subparsers(dest="alias_command", required=True)
    alias_add = alias_sub.add_parser("add", help="add an alias")
    alias_add.add_argument("profile_id")
    alias_add.add_argument("alias")
    alias_remove = alias_sub.add_parser("remove", help="remove an alias")
    alias_remove.add_argument("profile_id")
    alias_remove.add_argument("alias")

    environment = profiles_sub.add_parser("env", help="manage profile environment variables")
    env_sub = environment.add_subparsers(dest="env_command", required=True)
    env_set = env_sub.add_parser("set", help="set KEY=VALUE")
    env_set.add_argument("profile_id")
    env_set.add_argument("assignment")
    env_unset = env_sub.add_parser("unset", help="remove KEY")
    env_unset.add_argument("profile_id")
    env_unset.add_argument("key")

    profile_args = profiles_sub.add_parser("args", help="manage extra server arguments")
    args_sub = profile_args.add_subparsers(dest="args_command", required=True)
    args_add = args_sub.add_parser("add", help="append one or more arguments")
    args_add.add_argument("profile_id")
    args_add.add_argument("flags", nargs=argparse.REMAINDER)
    args_remove = args_sub.add_parser("remove", help="remove one argument and its value")
    args_remove.add_argument("profile_id")
    args_remove.add_argument("flag")

    profiles_sub.add_parser("template", help="sync the repository profiles.json template")

    return parser



def _print_admin_state(state: dict[str, object], *, raw: bool) -> None:
    """Print an admin response as JSON or a concise human-readable summary."""
    if raw:
        print(json.dumps(state, indent=2, sort_keys=True))
        return

    operation = state.get("operation")
    if isinstance(operation, dict):
        print(f"Request:   {operation.get('status', 'unknown')}")
        request_id = operation.get("request_id")
        execution_id = operation.get("execution_id")
        if isinstance(request_id, str):
            print(f"Request ID: {request_id}")
        if isinstance(execution_id, str):
            print(f"Execution ID: {execution_id}")
        if operation.get("target") is not None:
            print(f"Target:    {operation['target']}")
    print(f"Broker:    {state.get('status', 'unknown')}")
    backend = state.get("backend")
    if not isinstance(backend, dict):
        print(f"Backend:   {backend if backend is not None else '(unknown)'}")
        return

    profile = backend.get("profile")
    phase = backend.get("phase", "unknown")
    accepted = isinstance(operation, dict) and operation.get("status") in {
        "accepted",
        "queued",
        "preflight",
        "committed",
    }
    if accepted:
        print(
            f"Backend:   transition pending (current: {profile if profile else 'none'})"
        )
        phase = "transition_pending"
    else:
        print(f"Backend:   {profile if profile else phase}")
    print(f"Phase:     {phase}")
    if "ready" in backend:
        print(f"Ready:     {'no' if accepted else ('yes' if backend['ready'] else 'no')}")
    if "generation" in backend:
        print(f"Generation: {backend['generation']}")
    if "host" in backend and "port" in backend:
        print(f"Endpoint:  {backend['host']}:{backend['port']}")


def _crash_loop_lines(live_state: dict[str, object]) -> list[str]:
    """Describe profiles the broker's crash-loop guard is refusing or watching."""
    crash_loops = live_state.get("crash_loops")
    if not isinstance(crash_loops, dict):
        return []
    lines = []
    for profile_id, entry in sorted(crash_loops.items()):
        if not isinstance(entry, dict):
            continue
        cause = entry.get("last_cause") or "unknown cause"
        crashes = entry.get("crashes", 0)
        if entry.get("tripped"):
            lines.append(
                f"Crash loop: {profile_id} refused for {entry.get('retry_after', 0)}s "
                f"after {crashes} crashes (`brokerctl restart` retries now); last: {cause}"
            )
        elif entry.get("probation"):
            lines.append(f"Crash loop: {profile_id} on probation; last: {cause}")
    return lines


def _human_restart_summary(state: dict[str, object]) -> str:
    """One human-readable line per aspect of a restart request response."""
    operation = state.get("operation")
    target = None
    if isinstance(operation, dict):
        candidate = operation.get("target")
        if isinstance(candidate, str) and candidate:
            target = candidate
    backend = state.get("backend")
    if not isinstance(backend, dict):
        backend = {}
    profile = backend.get("profile")
    current = profile if isinstance(profile, str) else None
    if target is None and current is not None:
        target = current
    target_text = target if isinstance(target, str) else "(none)"
    phase = backend.get("phase")
    phase_text = phase if isinstance(phase, str) else "unknown"
    return (
        f"Requested: restart {target_text}\n"
        f"Target:    {target_text}\n"
        f"Phase:     {phase_text}"
    )


def _cmd_operation(args: argparse.Namespace) -> int:
    """Inspect, cancel, or explicitly wait for broker operation requests."""
    if args.operation_command == "list":
        result = _broker_admin_get("/admin/operations")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.operation_command == "tui":
        from .broker_tui import OperationsTUIModel, run_operations_tui

        registry = ProfileRegistry.from_path(args.profiles, allow_missing_artifacts=True)

        def wait_for_tui_operation(request_id: str) -> Mapping[str, Any]:
            deadline = time.monotonic() + 3600.0
            while True:
                result = _broker_admin_get(f"/admin/operations/{request_id}")
                operation = result.get("operation")
                if not isinstance(operation, Mapping):
                    return result
                if operation.get("status") in {"succeeded", "failed", "cancelled"}:
                    return result
                if not operation.get("waiting", True):
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return result
                time.sleep(min(0.5, remaining))

        model = OperationsTUIModel(
            profiles=(profile.id for profile in registry.profiles),
            list_operations=lambda: _broker_admin_get("/admin/operations"),
            refresh_profiles=lambda: (
                profile.id
                for profile in ProfileRegistry.from_path(
                    args.profiles, allow_missing_artifacts=True
                ).profiles
            ),
            submit_load=lambda profile_id: _broker_admin(
                "/admin/switch", {"model": profile_id, "wait": False}
            ),
            submit_restart=lambda: _broker_admin("/admin/restart", {"wait": False}),
            submit_stop=lambda: _broker_admin("/admin/stop", {"wait": False}),
            submit_reload=lambda: _broker_admin(
                "/admin/autotune/reload", {"wait": False}
            ),
            status_operation=lambda request_id: _broker_admin_get(
                f"/admin/operations/{request_id}"
            ),
            cancel_operation=lambda request_id: _broker_admin(
                f"/admin/operations/{request_id}/cancel", {}
            ),
            wait_operation=wait_for_tui_operation,
        )
        return run_operations_tui(model)
    request_id = args.request_id
    endpoint = f"/admin/operations/{request_id}"
    if args.operation_command == "status":
        result = _broker_admin_get(endpoint)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.operation_command == "cancel":
        result = _broker_admin(f"{endpoint}/cancel", {})
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.operation_command != "wait":
        raise BackendControllerError("unknown operation command")
    if args.timeout <= 0 or args.interval <= 0:
        print("operation wait timeout and interval must be positive", file=sys.stderr)
        return 2

    deadline = time.monotonic() + args.timeout
    while True:
        result = _broker_admin_get(endpoint)
        operation = result.get("operation")
        if not isinstance(operation, dict):
            raise BackendControllerError("broker returned malformed operation status")
        status = operation.get("status")
        if status in {"succeeded", "failed", "cancelled"} or not operation.get("waiting", True):
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if status == "succeeded" else (1 if status in {"failed", "cancelled"} else 0)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 124
        time.sleep(min(args.interval, remaining))


def _normalise_cli_argv(argv: list[str] | None) -> list[str] | None:
    """Allow option-like values after repeated ``profiles add --arg`` flags."""

    raw = list(sys.argv[1:] if argv is None else argv)
    normalised: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if token == "--arg" and index + 1 < len(raw):
            normalised.append(f"--arg={raw[index + 1]}")
            index += 2
            continue
        normalised.append(token)
        index += 1
    return normalised


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(_normalise_cli_argv(argv))

    try:
        # New commands that don't need registry
        if args.command == "setup":
            return _cmd_setup_wizard(args)
        if args.command == "integrate":
            if args.dry_run and args.json:
                print("integrate --dry-run is interactive and cannot be combined with --json", file=sys.stderr)
                return 2
            return _cmd_integrate(args)
        if args.command == "security":
            if args.security_command == "init":
                return _cmd_security_init(args)
            if args.security_command == "rotate":
                return _cmd_security_rotate(args)
            if args.security_command == "revoke":
                return _cmd_security_revoke(args)
        if args.command == "model" and args.model_command == "list":
            return _cmd_model_list(args)
        if args.command == "runtime":
            if args.runtime_command == "list":
                return _cmd_runtime_list(args)
            return _cmd_runtime_control(args)
        if args.command == "profile":
            if args.profiles_command == "list":
                return _cmd_profiles_list(args)
            if args.profiles_command == "show":
                return _cmd_profiles_show(args)
            if args.profiles_command == "autotune":
                return _cmd_profiles_autotune(args)
            if args.profiles_command == "discover":
                return _cmd_profiles_discover(args)
            if args.profiles_command == "edit":
                return _cmd_profiles_edit(args)
            if args.profiles_command == "add":
                return _cmd_profiles_add(args)
            if args.profiles_command == "remove":
                return _cmd_profiles_remove(args)
            if args.profiles_command == "rename":
                return _cmd_profiles_rename(args)
            if args.profiles_command == "set":
                return _cmd_profiles_set(args)
            if args.profiles_command == "unset":
                return _cmd_profiles_unset(args)
            if args.profiles_command == "alias":
                return _cmd_profiles_alias(args)
            if args.profiles_command == "env":
                return _cmd_profiles_env(args)
            if args.profiles_command == "args":
                return _cmd_profiles_args(args)
            if args.profiles_command == "template":
                return _cmd_profiles_template(args)

        if args.command in {"operation", "operations"}:
            return _cmd_operation(args)
        if args.command == "clean":
            return _cmd_clean(args)

        # Original commands
        registry = ProfileRegistry.from_path(
            args.profiles, allow_missing_artifacts=args.command == "status"
        )
        controller = create_production_controller(registry)

        if args.command == "status":
            state = controller.state()
            phase = state.get("phase", "unknown")
            profile_id = state.get("profile")
            # Durable state is useful when the broker is stopped, but the live
            # health snapshot is the source of truth for leases and transitions.
            backend_parts = urlsplit(effective_backend_url(load_security_config()))
            backend_host = backend_parts.hostname or "127.0.0.1"
            backend_port = backend_parts.port or 10000
            backend_ready = backend_endpoint_ready(
                backend_host, backend_port, timeout=0.25
            )
            live_state: dict[str, object] = {}
            try:
                live_state = _broker_admin_get("/admin/status")
            except BackendControllerError:
                pass
            live_backend = live_state.get("backend")
            live_leases = live_state.get("leases")
            if isinstance(live_leases, dict) and live_leases.get("transitioning"):
                # Older daemons cache the last completed backend phase while
                # loading; an open TCP port does not make that phase current.
                phase = "switching"
                backend_ready = False
            elif isinstance(live_backend, dict) and backend_ready:
                profile_id = live_backend.get("profile", profile_id)
                phase = live_backend.get("phase", phase)
                if "ready" in live_backend:
                    backend_ready = bool(live_backend["ready"])
            stale_active = phase == "active" and not backend_ready
            configured_identity: tuple[str, str] | None = None
            running_identity: tuple[str, str] | None = None
            if profile_id and phase == "active":
                configured_profile = registry.try_resolve(str(profile_id))
                configured_build = getattr(configured_profile, "runtime_build", None)
                configured_sha = getattr(configured_profile, "server_sha256", None)
                running_build = state.get("runtime_build")
                running_sha = state.get("server_sha256")
                if isinstance(configured_build, str) and configured_build and isinstance(configured_sha, str) and configured_sha:
                    configured_identity = (configured_build, configured_sha)
                if isinstance(running_build, str) and running_build and isinstance(running_sha, str) and running_sha:
                    running_identity = (running_build, running_sha)
            runtime_configuration_pending = (
                configured_identity is not None
                and running_identity is not None
                and configured_identity != running_identity
            )

            if getattr(args, "json", False):
                detail = dict(state)
                detail["schema_version"] = 1
                detail["observed_phase"] = "stale" if stale_active else phase
                detail["backend_ready"] = backend_ready
                detail["runtime_configuration_pending"] = runtime_configuration_pending
                if live_state:
                    detail["live"] = live_state
                print(json.dumps(detail, indent=2, sort_keys=True))
                return 1 if stale_active or phase not in {"active", "inactive"} else 0

            # Broker status
            observed_phase = "stale" if stale_active else phase
            broker_health = live_state.get("status", "unavailable")
            print(f"Broker:    {observed_phase} (port 11434; health: {broker_health})")
            for line in _crash_loop_lines(live_state):
                print(line)

            # Backend status
            if stale_active:
                print(
                    f"Backend:   unavailable (persisted profile: {profile_id}; "
                    f"{backend_host}:{backend_port} is not accepting connections)"
                )
                recovery = state.get("recovery")
                if isinstance(recovery, dict):
                    attempts = recovery.get("attempts", 0)
                    action = recovery.get("action", "unknown")
                    error = str(recovery.get("error", "unknown failure"))
                    print(f"Recovery:  {action}; attempts={attempts}; {error}")
            elif profile_id and phase == "active":
                profile = registry.try_resolve(str(profile_id))
                display = profile.display_name if profile else profile_id
                print(f"Model:     {profile_id} ({display})")
                if profile is not None:
                    runtime_info = get_runtime_info(profile.runtime)
                    binary = profile.server_binary or runtime_info.binaries[0]
                    print(f"Backend:   {runtime_info.name} ({binary})")
                    print(f"Context:   {profile.context_size:,} tokens total; {profile.context_size // profile.parallel:,} per slot")
                    leases = live_state.get("leases")
                    leases = leases if isinstance(leases, dict) else {}
                    # Leases count in-flight broker requests, not backend slots:
                    # /slots and /props probes hold a lease, and requests beyond
                    # the profile's parallel limit wait in the backend's queue.
                    active = int(leases.get("inference_requests", leases.get("active_requests", 0)) or 0)
                    pending = int(leases.get("pending_requests", 0) or 0)
                    pending_transitions = int(leases.get("queued_transitions", 0) or 0)
                    slots = max(int(profile.parallel or 0), 0)
                    in_use = min(active, slots)
                    pending += active - in_use
                    utilisation = in_use / slots if slots else 0.0
                    print(f"Requests:  active={in_use} pending={pending} (pending transitions={pending_transitions})")
                    print(f"Slots:     {in_use}/{slots} in use ({utilisation:.0%})")
                    if runtime_configuration_pending:
                        print("Runtime configuration pending")
                else:
                    print("Backend:   (unknown)")
                print(f"Port:      {backend_port}")

                # Check alias symlink
                alias_path = Path.home() / ".config/systemd/user/llama-server.service"
                if alias_path.is_symlink():
                    target = alias_path.readlink().name
                    print(f"Alias:     llama-server.service -> {target}")
            elif phase == "inactive":
                print("Backend:   (none)")
            else:
                print(f"Backend:   {phase}")

            # JSON detail if requested. Include the live snapshot when one was
            # available so scripts can consume the same context shown above.
            if args.verbose:
                detail = dict(state)
                if live_state:
                    detail["live"] = live_state
                print("\n" + json.dumps(detail, indent=2, sort_keys=True))
            return 1 if stale_active or phase not in {"active", "inactive"} else 0
        if args.command == "profile" and args.profiles_command == "render":
            profile = registry.resolve(args.model)
            print(controller.render_unit(profile.id), end="")
            return 0
        if args.command in {"load", "start", "restart"}:
            if args.command == "restart":
                payload: dict[str, object] = {"wait": False}
                if args.model is not None:
                    payload["model"] = registry.resolve(args.model).id
                state = _broker_admin("/admin/restart", payload)
                if args.json:
                    print(json.dumps(state, indent=2, sort_keys=True))
                else:
                    print(_human_restart_summary(state))
                return 0
            profile = registry.resolve(args.model)
            state = _broker_admin(
                "/admin/switch",
                {"model": profile.id, "wait": False},
            )
            _print_admin_state(state, raw=args.raw)
            return 0
        if args.command == "unload":
            state = _broker_admin("/admin/stop", {"wait": False})
            _print_admin_state(state, raw=args.raw)
            return 0
        if args.command == "verify":
            profiles = (
                [registry.resolve(args.model)] if args.model else list(registry.profiles)
            )
            failures: list[str] = []
            results: list[dict[str, object]] = []
            for profile in profiles:
                result = _profile_verification_result(profile)
                profile_failures = result["errors"]
                assert isinstance(profile_failures, list)
                failures.extend(profile_failures)
                results.append(result)
                if not profile_failures and not getattr(args, "json", False):
                    print(f"verified: {profile.id}")
            if getattr(args, "json", False):
                print(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "catalog_path": (
                                str(registry.runtime_catalog_path)
                                if registry.runtime_catalog_path
                                else None
                            ),
                            "deployment_revision": registry.deployment_revision,
                            "profiles": results,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            if failures:
                if not getattr(args, "json", False):
                    print("\n".join(failures), file=sys.stderr)
                return 1
            return 0

    except (ProfileError, ApplyError, BackendControllerError, OSError, ValueError) as error:
        print(f"brokerctl: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
