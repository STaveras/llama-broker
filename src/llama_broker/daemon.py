#!/usr/bin/env python3
"""Bounded OpenAI-compatible model broker for the local Hermes runtime.

All model traffic enters on the public broker port.  The broker owns a single
private llama-server backend, drains active leases before cross-profile
switches, and restores transient profiles (such as vision) after their last
request completes.
"""

from __future__ import annotations

import fcntl
import http.client
import ipaddress
import json
import math
import os
import queue
import re
import secrets
import select
import signal
import socket
import ssl
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol
from urllib.parse import unquote, urlsplit

from .backend_controller import (
    BackendController,
    BackendControllerError,
    BackendExit,
    LifecycleCancelledError,
    SwitchError,
    subprocess_command_runner,
    systemd_backend_exit_probe,
    systemd_service_inventory_probe,
    systemd_service_state_probe,
)
from .crash_loop import CrashLoopGuard
from .connection_reliability import AdmissionError, BackendPipe, InferenceAdmission
from .gpu_memory import GpuTelemetryError, probe_gpu_memory
from .broker_security import (
    SecurityConfig,
    SecurityError,
    load_security_config,
    read_api_keys,
    revoke_api_key,
    rotate_api_keys,
    write_api_keys,
)
from .state_store import StateFileError, StateFileStore
from .transition_operations import (
    ExecutionView,
    OperationCancelledError,
    OperationStateError,
    OperationSupersededError,
    PersistentTransitionQueue,
    QueueCapacityError,
    ReconciliationRequiredError,
    RequestView,
    TransitionQueue,
)
from .tools.broker_backend import BackendClient
from .tools.broker_backend import make_connection
from .broker_profiles import Profile, ProfileError, ProfileRegistry
from . import decisions
from .decisions import DecisionBackendError, DecisionRequestError
from .system1 import (
    System1ConfigError,
    System1Error,
    System1Service,
    System1Sidecar,
    load_system1_config,
)
from .autotune import MaintenanceLease
from .profile_config import sha256_file
from .paths import (
    CONFIG_DIR as XDG_CONFIG_DIR,
    LOCK_PATH as XDG_LOCK_PATH,
    PROCESS_LOCK_PATH as XDG_PROCESS_LOCK_PATH,
    PROFILES_PATH as XDG_PROFILES_PATH,
    STATE_PATH as XDG_STATE_PATH,
    SYSTEMD_USER_DIR as XDG_SYSTEMD_USER_DIR,
    migrate_legacy_state,
)


ROOT = Path(
    os.environ.get("LLAMA_BROKER_DIR", Path(__file__).resolve().parents[2])
).expanduser()
CONFIG_DIR = XDG_CONFIG_DIR
PROFILES_PATH = XDG_PROFILES_PATH
STATE_PATH = XDG_STATE_PATH
LOCK_PATH = XDG_LOCK_PATH
PROCESS_LOCK_PATH = XDG_PROCESS_LOCK_PATH
SYSTEMD_USER_DIR = XDG_SYSTEMD_USER_DIR
AUTOTUNE_LEASE_PATH = CONFIG_DIR / "autotune-maintenance.json"
SERVICE_PREFIX = os.environ.get("LLAMA_BROKER_SERVICE_PREFIX", "llama-broker-profile")
LEGACY_SERVICE_UNIT = "llama-server.service"

_DEFAULT_BACKEND_URL = "http://127.0.0.1:10000"

_ARTIFACT_VERIFICATION_CACHE: dict[tuple[object, ...], None] = {}
_ARTIFACT_VERIFICATION_CACHE_LOCK = threading.Lock()
_ARTIFACT_VERIFICATION_CACHE_MAX = 1024


def _artifact_verification_key(path: Path, expected: str) -> tuple[object, ...] | None:
    """Return a process-local identity for a file and its expected digest."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (
        str(path),
        expected,
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _artifact_verification_cached(key: tuple[object, ...] | None) -> bool:
    if key is None:
        return False
    with _ARTIFACT_VERIFICATION_CACHE_LOCK:
        return key in _ARTIFACT_VERIFICATION_CACHE


def _remember_artifact_verification(key: tuple[object, ...] | None) -> None:
    if key is None:
        return
    with _ARTIFACT_VERIFICATION_CACHE_LOCK:
        _ARTIFACT_VERIFICATION_CACHE[key] = None
        while len(_ARTIFACT_VERIFICATION_CACHE) > _ARTIFACT_VERIFICATION_CACHE_MAX:
            _ARTIFACT_VERIFICATION_CACHE.pop(next(iter(_ARTIFACT_VERIFICATION_CACHE)))


def _env_value(canonical: str, legacy: str, default: str) -> str:
    return os.environ.get(canonical, os.environ.get(legacy, default))



def _optional_env_int(canonical: str, legacy: str) -> int | None:
    value = os.environ.get(canonical)
    if value is None:
        value = os.environ.get(legacy)
    return int(value) if value is not None else None


_BACKEND_URL_EXPLICIT = any(
    name in os.environ for name in ("LLAMA_BROKER_BACKEND_URL", "LLM_BROKER_BACKEND_URL")
)
_BACKEND_LISTEN_HOST_EXPLICIT = any(
    name in os.environ
    for name in ("LLAMA_BROKER_BACKEND_LISTEN_HOST", "LLM_BROKER_BACKEND_LISTEN_HOST")
)
BACKEND_URL = _env_value(
    "LLAMA_BROKER_BACKEND_URL", "LLM_BROKER_BACKEND_URL", _DEFAULT_BACKEND_URL
).rstrip("/")
BACKEND_LISTEN_HOST = _env_value(
    "LLAMA_BROKER_BACKEND_LISTEN_HOST", "LLM_BROKER_BACKEND_LISTEN_HOST", "127.0.0.1"
)
BACKEND_LISTEN_PORT = int(
    _env_value("LLAMA_BROKER_BACKEND_LISTEN_PORT", "LLM_BROKER_BACKEND_LISTEN_PORT", "10000")
)
BACKEND_TIMEOUT = float(
    _env_value("LLAMA_BROKER_BACKEND_TIMEOUT", "LLM_BROKER_BACKEND_TIMEOUT", "1800")
)
BROKER_HOST = _env_value("LLAMA_BROKER_HOST", "LLM_BROKER_HOST", "127.0.0.1")
BROKER_PORT = int(_env_value("LLAMA_BROKER_PORT", "LLM_BROKER_PORT", "11434"))
MAX_WORKERS = int(_env_value("LLAMA_BROKER_MAX_WORKERS", "LLM_BROKER_MAX_WORKERS", "2"))
CONTROL_WORKERS = int(
    _env_value("LLAMA_BROKER_CONTROL_WORKERS", "LLM_BROKER_CONTROL_WORKERS", "2")
)
CONTROL_CLIENT_TIMEOUT = float(
    _env_value(
        "LLAMA_BROKER_CONTROL_CLIENT_TIMEOUT",
        "LLM_BROKER_CONTROL_CLIENT_TIMEOUT",
        "2",
    )
)
INFERENCE_QUEUE_MAX_SOCKETS = int(
    _env_value("LLAMA_BROKER_INFERENCE_QUEUE_MAX_SOCKETS", "LLM_BROKER_INFERENCE_QUEUE_MAX_SOCKETS", "16")
)
INFERENCE_QUEUE_MAX_BODY_BYTES = int(
    _env_value("LLAMA_BROKER_INFERENCE_QUEUE_MAX_BODY_BYTES", "LLM_BROKER_INFERENCE_QUEUE_MAX_BODY_BYTES", "67108864")
)
INFERENCE_QUEUE_WAIT_TIMEOUT = float(
    _env_value("LLAMA_BROKER_INFERENCE_QUEUE_WAIT_TIMEOUT", "LLM_BROKER_INFERENCE_QUEUE_WAIT_TIMEOUT", "360")
)
SSE_HEARTBEAT_INTERVAL = float(
    _env_value("LLAMA_BROKER_SSE_HEARTBEAT_INTERVAL", "LLM_BROKER_SSE_HEARTBEAT_INTERVAL", "15")
)
# Bounds for draining request bodies the broker rejected without reading.
REJECTED_BODY_DRAIN_TIMEOUT = 2.0
HEADER_WORKERS = 16
DRAIN_TIMEOUT = float(
    _env_value("LLAMA_BROKER_DRAIN_TIMEOUT", "LLM_BROKER_DRAIN_TIMEOUT", "10")
)
TRANSITION_TIMEOUT = float(
    _env_value(
        "LLAMA_BROKER_TRANSITION_TIMEOUT", "LLM_BROKER_TRANSITION_TIMEOUT", "360"
    )
)
# A stop may spend the command-runner's bounded grace period waiting for
# llama-server to unwind CUDA/KV-cache state before the controller escalates to
# a cgroup kill.  Keep the HTTP transition watchdog above that grace period so
# callers receive the completed inactive state instead of a quarantined 504.
STOP_TIMEOUT = float(
    _env_value("LLAMA_BROKER_STOP_TIMEOUT", "LLM_BROKER_STOP_TIMEOUT", "90")
)
RECOVERY_INTERVAL = float(
    _env_value("LLAMA_BROKER_RECOVERY_INTERVAL", "LLM_BROKER_RECOVERY_INTERVAL", "2")
)
MANIFEST_RELOAD_INTERVAL = float(
    _env_value(
        "LLAMA_BROKER_MANIFEST_RELOAD_INTERVAL",
        "LLM_BROKER_MANIFEST_RELOAD_INTERVAL",
        "2",
    )
)


def _manifest_identity(profiles_path: Path | str) -> str:
    """Fingerprint the raw manifest file so a change can be detected cheaply.

    The watcher polls this hash instead of fully parsing the registry on every
    tick: hashing the small manifest file is O(bytes) and catches edits to ANY
    profile type (resident or switchable), whereas re-parsing + re-fingerprinting
    the registry would re-verify pinned artifacts (re-hashing multi-GB model files
    and Strata source checkouts) every interval. A same-content rewrite is
    still coalesced by the reload path's own snapshot dedup.
    """
    return sha256(Path(profiles_path).read_bytes()).hexdigest()
RECOVERY_MAX_BACKOFF = float(
    _env_value("LLAMA_BROKER_RECOVERY_MAX_BACKOFF", "LLM_BROKER_RECOVERY_MAX_BACKOFF", "60")
)
# Request-time backend crashes tolerated per profile inside the window before
# its requests are refused for the cooldown instead of restarting it; 0 disables.
CRASH_LOOP_THRESHOLD = int(
    _env_value("LLAMA_BROKER_CRASH_LOOP_THRESHOLD", "LLM_BROKER_CRASH_LOOP_THRESHOLD", "3")
)
CRASH_LOOP_WINDOW = float(
    _env_value("LLAMA_BROKER_CRASH_LOOP_WINDOW", "LLM_BROKER_CRASH_LOOP_WINDOW", "600")
)
CRASH_LOOP_COOLDOWN = float(
    _env_value("LLAMA_BROKER_CRASH_LOOP_COOLDOWN", "LLM_BROKER_CRASH_LOOP_COOLDOWN", "300")
)
MAX_BODY_BYTES = int(
    _env_value(
        "LLAMA_BROKER_MAX_BODY_BYTES",
        "LLM_BROKER_MAX_BODY_BYTES",
        str(32 * 1024 * 1024),
    )
)
MAX_OUTPUT_TOKENS = _optional_env_int(
    "LLAMA_BROKER_MAX_OUTPUT_TOKENS", "LLM_BROKER_MAX_OUTPUT_TOKENS"
)
PREEMPT_IDLE_LOADS = os.environ.get("LLAMA_BROKER_PREEMPT_IDLE_LOADS", "1") != "0"
HANDOFF_MIN_VRAM_RISE_MIB = int(_env_value("LLAMA_BROKER_HANDOFF_MIN_VRAM_RISE_MIB", "LLM_BROKER_HANDOFF_MIN_VRAM_RISE_MIB", "256"))
HANDOFF_WINDOW_SAMPLES = int(_env_value("LLAMA_BROKER_HANDOFF_WINDOW_SAMPLES", "LLM_BROKER_HANDOFF_WINDOW_SAMPLES", "3"))


HOP_BY_HOP_REQUEST_HEADERS = {
    "accept-encoding",
    "connection",
    "content-length",
    "expect",
    "host",
    "keep-alive",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

HOP_BY_HOP_RESPONSE_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def effective_backend_url(security: SecurityConfig | None = None) -> str:
    """Use HTTPS automatically when setup configured native backend TLS."""

    security = security or load_security_config()
    if not _BACKEND_URL_EXPLICIT and security.backend_tls_cert_file is not None:
        return urlsplit(BACKEND_URL)._replace(scheme="https").geturl()
    return BACKEND_URL


def backend_url_for_runtime(backend_url: str, runtime: str) -> str:
    """Return the configured secured transport without silently downgrading it."""
    if str(runtime).strip().lower() == "basalt":
        parts = urlsplit(backend_url)
        if parts.scheme.lower() != "https":
            raise SecurityError("basalt backend transport requires HTTPS")
        return backend_url
    if str(runtime).strip().lower() != "strata":
        return backend_url
    parts = urlsplit(backend_url)
    host = (parts.hostname or "").strip().lower()
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        raise SecurityError("strata backend transport requires a loopback endpoint")
    return parts._replace(scheme="http").geturl()


def effective_backend_listen_host(security: SecurityConfig | None = None) -> str:
    """Expose a TLS-protected backend on the LAN unless explicitly narrowed."""

    security = security or load_security_config()
    if not _BACKEND_LISTEN_HOST_EXPLICIT and security.backend_tls_cert_file is not None:
        return "0.0.0.0"
    return BACKEND_LISTEN_HOST


_AUTHORIZATION_LOG_PATTERN = re.compile(
    r"(?i)(authorization\s*[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^,\s}]+(?:\s+[^,\s:=]+)?)"
)
_SECRET_LOG_PATTERN = re.compile(
    r"(?i)(?P<prefix>\"?(?:api[-_]?key|token|password|secret)\"?\s*[=:]\s*\"?)(?P<value>[^\"\s,}]+)"
)


def _redact_log_message(message: str) -> str:
    message = _AUTHORIZATION_LOG_PATTERN.sub(r"\1[REDACTED]", message)
    return _SECRET_LOG_PATTERN.sub(r"\g<prefix>[REDACTED]", message)


def _log(message: str) -> None:
    print(_redact_log_message(message), file=sys.stderr, flush=True)


def _controller_event_logger(event: dict[str, object]) -> None:
    _log(
        "[broker-event] "
        + json.dumps(event, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    )


def backend_endpoint_ready(host: str, port: int, timeout: float = 0.25) -> bool:
    """Return whether the private backend endpoint accepts a bounded TCP connect."""

    if timeout <= 0:
        raise ValueError("backend readiness timeout must be positive")
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _backend_auth_headers(security: SecurityConfig) -> dict[str, str]:
    keys = read_api_keys(security.api_key_file) if security.api_key_file else []
    if not keys and security.api_key:
        keys = [security.api_key]
    return {"Authorization": f"Bearer {keys[-1]}"} if keys else {}


def backend_serves_alias(
    host: str,
    port: int,
    alias: str,
    timeout: float = 0.5,
    *,
    scheme: str = "http",
    ca_file: str | Path | None = None,
    headers: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the private backend advertises *alias* exactly."""

    if timeout <= 0:
        raise ValueError("backend alias readiness timeout must be positive")
    host_for_url = f"[{host}]" if ":" in host and not host.startswith("[") else host
    client = BackendClient(
        f"{scheme}://{host_for_url}:{port}",
        timeout=timeout,
        ca_file=ca_file,
        default_headers=headers,
    )
    try:
        payload = client.list_models()
        if not isinstance(payload, Mapping):
            return False
        models = payload.get("data", payload.get("models", []))
        return isinstance(models, list) and any(
            isinstance(model, Mapping) and model.get("id") == alias
            for model in models
        )
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        http.client.HTTPException,
    ):
        return False
    finally:
        client.close()


class UnknownModelError(LookupError):
    pass


class BrokerCoordinationError(RuntimeError):
    error_type = "broker_coordination_error"
    http_status = HTTPStatus.SERVICE_UNAVAILABLE


class NoActiveProfileError(BrokerCoordinationError):
    error_type = "no_active_profile"


class BrokerQuiescingError(BrokerCoordinationError):
    error_type = "broker_quiescing"


class BrokerDrainTimeoutError(BrokerCoordinationError):
    error_type = "broker_drain_timeout"
    http_status = HTTPStatus.GATEWAY_TIMEOUT


class BrokerTransitionTimeoutError(BrokerCoordinationError):
    error_type = "broker_transition_timeout"
    http_status = HTTPStatus.GATEWAY_TIMEOUT


class BrokerPreemptedError(BrokerQuiescingError):
    pass


class BrokerLifecycleError(BrokerCoordinationError):
    error_type = "broker_lifecycle_failed"


class BrokerStateUnknownError(BrokerCoordinationError):
    error_type = "broker_state_unknown"


class BrokerRequestStaleError(BrokerCoordinationError):
    error_type = "broker_request_stale"
    http_status = HTTPStatus.CONFLICT


class StaleOperationSnapshotError(LifecycleCancelledError):
    """A queued operation no longer matches the profile facts it captured."""


class BrokerAlreadyRunningError(RuntimeError):
    pass


class Controller(Protocol):
    def state(self) -> dict[str, Any]: ...

    def switch(self, profile_id: str) -> dict[str, Any]: ...

    def restart(self, profile_id: str) -> dict[str, Any]: ...

    def stop_all(self) -> dict[str, Any]: ...

    def record_recovery_failure(
        self, profile_id: str, *, attempts: int, error: str
    ) -> dict[str, Any]: ...


def _verify_profile_artifacts(profile: Any, *, force: bool = False) -> None:
    """Hash pinned artifacts for explicit checks or backend failure diagnostics."""

    checks: list[tuple[Path, object, str]] = [
        (
            Path(profile.server_binary),
            getattr(profile, "server_sha256", None),
            "server_binary",
        ),
        (Path(profile.model_path), getattr(profile, "model_sha256", None), "model"),
    ]
    mmproj_path = getattr(profile, "mmproj_path", None)
    if mmproj_path is not None:
        checks.append(
            (Path(mmproj_path), getattr(profile, "mmproj_sha256", None), "mmproj")
        )
    for path_field, hash_field, label in (
        ("strata_artifacts_manifest_path", "strata_artifacts_manifest_sha256", "Strata manifest"),
        ("strata_launch_config_path", "strata_launch_config_sha256", "Strata launch config"),
    ):
        path = getattr(profile, path_field, None)
        if path is not None:
            checks.append((Path(path), getattr(profile, hash_field, None), label))
    checks.extend(
        (Path(path), digest, "Strata artifact")
        for path, digest in getattr(profile, "strata_artifact_files", ())
    )

    for path, expected, kind in checks:
        if not isinstance(expected, str) or not expected:
            raise BackendControllerError(
                f"{profile.id} {kind} has no configured SHA-256 digest"
            )
        if path.is_symlink() or not path.is_file():
            raise BackendControllerError(f"{profile.id} {kind} is not a regular file: {path}")
        key = _artifact_verification_key(path, str(expected))
        if not force and _artifact_verification_cached(key):
            continue
        try:
            actual = sha256_file(path)
        except OSError as error:
            raise BackendControllerError(
                f"{profile.id} {kind} is unavailable: {error}"
            ) from error
        if path.is_symlink() or not path.is_file():
            raise BackendControllerError(f"{profile.id} {kind} changed to a non-regular file during verification: {path}")
        after_key = _artifact_verification_key(path, str(expected))
        if key is not None and after_key != key:
            raise BackendControllerError(
                f"{profile.id} {kind} changed during verification: {path}"
            )
        if actual != expected:
            raise BackendControllerError(
                f"{profile.id} {kind} SHA mismatch: expected {expected}, "
                f"got {actual} ({path})"
            )
        _remember_artifact_verification(key)


def apply_reasoning_policy(
    payload: Mapping[str, Any],
    *,
    runtime: str,
    max_output_tokens: int | None = None,
    request_max_output_tokens: int | None = None,
    batch_size: int = 1,
    ubatch_size: int = 1,
) -> dict[str, Any]:
    """Apply the experimental client reasoning policy for llama.cpp templates."""

    rewritten = dict(payload)
    if runtime != "llamacpp":
        return rewritten
    if max_output_tokens is None or max_output_tokens <= 0:
        raise ValueError("llamacpp reasoning policy requires a positive output limit")
    if request_max_output_tokens is not None and (
        isinstance(request_max_output_tokens, bool)
        or not isinstance(request_max_output_tokens, int)
        or request_max_output_tokens <= 0
    ):
        raise ValueError("request output limit must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(ubatch_size, bool) or not isinstance(ubatch_size, int) or ubatch_size <= 0:
        raise ValueError("ubatch_size must be a positive integer")

    budget_limit = request_max_output_tokens or max_output_tokens
    raw_kwargs = rewritten.get("chat_template_kwargs")
    kwargs = dict(raw_kwargs) if isinstance(raw_kwargs, Mapping) else {}
    explicit_template = "enable_thinking" in kwargs

    effort: str | None = None
    reasoning = rewritten.get("reasoning")
    if isinstance(reasoning, Mapping):
        raw_effort = reasoning.get("effort")
        if isinstance(raw_effort, str):
            effort = raw_effort
    raw_effort = rewritten.get("reasoning_effort")
    if effort is None and isinstance(raw_effort, str):
        effort = raw_effort
    if effort is not None:
        effort = {
            "min": "minimal",
            "med": "medium",
            "xhigh": "extra high",
        }.get(effort.strip().casefold(), effort.strip().casefold())

    explicit_budget = (
        "reasoning_budget_tokens" in rewritten
        or "thinking_budget_tokens" in rewritten
    )
    zero_budget_requested = any(
        isinstance(rewritten.get(key), int)
        and not isinstance(rewritten.get(key), bool)
        and rewritten.get(key) == 0
        for key in ("reasoning_budget_tokens", "thinking_budget_tokens")
    )
    # llama.cpp forks do not agree on the legacy spelling. BeeLlama consumes
    # thinking_budget_tokens; Atomic accepts that spelling as a fallback.
    if "reasoning_budget_tokens" in rewritten:
        if "thinking_budget_tokens" not in rewritten:
            rewritten["thinking_budget_tokens"] = rewritten["reasoning_budget_tokens"]
        rewritten.pop("reasoning_budget_tokens", None)

    zero_budget = (
        isinstance(rewritten.get("thinking_budget_tokens"), int)
        and not isinstance(rewritten.get("thinking_budget_tokens"), bool)
        and rewritten.get("thinking_budget_tokens") == 0
    )
    if zero_budget_requested or zero_budget:
        kwargs["enable_thinking"] = False
        rewritten["reasoning_effort"] = "none"
        rewritten["thinking_budget_tokens"] = 0
        rewritten.pop("reasoning_budget_tokens", None)
        rewritten["chat_template_kwargs"] = kwargs
        return rewritten

    if effort is None:
        if explicit_budget:
            if not explicit_template:
                kwargs["enable_thinking"] = True
        elif not explicit_template:
            kwargs["enable_thinking"] = False
            rewritten["reasoning_effort"] = "none"
            rewritten["thinking_budget_tokens"] = 0
    elif effort == "none":
        if not explicit_template:
            kwargs["enable_thinking"] = False
            rewritten["reasoning_effort"] = "none"
            if "reasoning_budget_tokens" not in rewritten and "thinking_budget_tokens" not in rewritten:
                rewritten["thinking_budget_tokens"] = 0
    elif effort in {"minimal", "low", "medium", "high", "extra high", "max"}:
        if not explicit_template:
            kwargs["enable_thinking"] = True
        template_disabled = explicit_template and kwargs.get("enable_thinking") is False
        if (
            not template_disabled
            and "reasoning_budget_tokens" not in rewritten
            and "thinking_budget_tokens" not in rewritten
        ):
            levels = ("minimal", "low", "medium", "high", "extra high", "max")
            index = levels.index(effort)
            if effort == "max":
                budget = budget_limit
            else:
                steps = len(levels) - 1
                quantum = ubatch_size if budget_limit >= len(levels) * ubatch_size else 1
                # 3072 covers the observed 2561 thinking / 2608 total-token turns.
                minimum = min(
                    max(3072, batch_size, len(levels) * ubatch_size),
                    max(1, budget_limit // len(levels)),
                )
                minimum = ((minimum + quantum - 1) // quantum) * quantum
                target = minimum ** (steps - index) * budget_limit**index
                lower = minimum // quantum
                upper = (budget_limit + quantum - 1) // quantum
                # First whole quantum above the geometric target, without float rounding.
                while lower < upper:
                    middle = (lower + upper) // 2
                    if (middle * quantum) ** steps >= target:
                        upper = middle
                    else:
                        lower = middle + 1
                budget = min(budget_limit, lower * quantum)
            rewritten["thinking_budget_tokens"] = max(1, budget)
        rewritten["reasoning_effort"] = effort
    else:
        # Be tolerant of client-specific effort enums. Unsupported values
        # do not reach llama.cpp; absent an explicit budget, default to off.
        rewritten.pop("reasoning_effort", None)
        if not explicit_budget and not explicit_template:
            kwargs["enable_thinking"] = False
            rewritten["thinking_budget_tokens"] = 0

    if kwargs:
        rewritten["chat_template_kwargs"] = kwargs
    return rewritten


def reasoning_policy_metadata(
    payload: Mapping[str, Any], rewritten: Mapping[str, Any], *, runtime: str
) -> dict[str, object] | None:
    """Return safe reasoning metadata without exposing request content."""

    if runtime != "llamacpp":
        return None

    effort = rewritten.get("reasoning_effort")
    if not isinstance(effort, str):
        reasoning = rewritten.get("reasoning")
        effort = reasoning.get("effort") if isinstance(reasoning, Mapping) else None
    if not isinstance(effort, str):
        effort = payload.get("reasoning_effort")
    if not isinstance(effort, str):
        effort = "unspecified"
    explicit_budget = any(
        key in payload for key in ("reasoning_budget_tokens", "thinking_budget_tokens")
    )
    effective_budget = rewritten.get("reasoning_budget_tokens")
    if effective_budget is None:
        effective_budget = rewritten.get("thinking_budget_tokens")
    if not isinstance(effective_budget, int) or isinstance(effective_budget, bool):
        effective_budget = None
    return {
        "runtime": runtime,
        "effort": effort,
        "budget_present": explicit_budget,
        "budget_tokens": effective_budget,
    }


def route_payload(
    registry: ProfileRegistry,
    payload: Mapping[str, Any],
    *,
    active_profile: str | None,
) -> tuple[Profile, dict[str, Any]]:
    """Resolve an explicit alias and rewrite it to the backend's exact alias.

    A missing ``model`` uses the currently loaded profile.  A present but
    unknown or empty value fails closed; it never falls through to a catalog
    entry or starts an arbitrary profile.
    """

    if "model" in payload:
        requested = payload.get("model")
        profile = registry.try_resolve(requested)
        if profile is None:
            raise UnknownModelError(f"unknown model: {requested}")
    else:
        if active_profile is None:
            raise NoActiveProfileError("no active model loaded")
        try:
            profile = registry.get(active_profile)
        except ProfileError as error:
            raise NoActiveProfileError(
                f"active model is unavailable: {active_profile}"
            ) from error
    rewritten = dict(payload)
    rewritten["model"] = getattr(profile, "backend_model_name", profile.alias)
    return profile, rewritten


def _requested_output_tokens(payload: Mapping[str, Any]) -> int | None:
    """Return a valid client output budget, if one was supplied."""

    for key in ("max_tokens", "max_completion_tokens"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def enforce_output_limit(
    payload: Mapping[str, Any], maximum: int, *, default: int | None = None
) -> dict[str, Any]:
    """Require a bounded completion budget before forwarding to llama-server."""

    if maximum <= 0:
        raise ValueError("broker maximum output tokens must be positive")
    if default is None:
        default = maximum
    if default <= 0:
        raise ValueError("broker default output tokens must be positive")
    if default > maximum:
        raise ValueError("broker default output tokens cannot exceed maximum")
    rewritten = dict(payload)
    keys = ("max_tokens", "max_completion_tokens")
    present = [key for key in keys if key in rewritten]
    if not present:
        rewritten["max_tokens"] = default
        return rewritten
    for key in present:
        value = rewritten[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        if value > maximum:
            raise ValueError(
                f"{key} exceeds Llama-Broker safety limit {maximum}"
            )
    return rewritten


def ensure_single_system_message(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge duplicate system messages and ensure exactly one system message at position 0.

    Some llama-server chat templates (e.g. Qwen3.5) raise a Jinja exception
    when the system message is not the first entry or when there are multiple
    system messages.  This is common when lean-ctx injects a compacted-context
    summary as a second ``role=system`` message.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return dict(payload)

    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(system_msgs) <= 1:
        if system_msgs and non_system and messages[0].get("role") != "system":
            rewritten = dict(payload)
            rewritten["messages"] = system_msgs + non_system
            return rewritten
        return dict(payload)

    merged_content = "\n\n".join(
        str(m.get("content", "")) for m in system_msgs if m.get("content")
    )
    merged = dict(system_msgs[0])
    merged["content"] = merged_content

    rewritten = dict(payload)
    rewritten["messages"] = [merged] + non_system
    return rewritten


class ProfileCoordinator:
    """Writer-priority leases with bounded drains and quarantined transitions."""

    def __init__(
        self,
        initial_profile: str | None,
        switch_profile: Callable[[str], Any],
        stop_backend: Callable[[], Any] | None = None,
        restart_profile: Callable[[str], Any] | None = None,
        *,
        drain_timeout: float = DRAIN_TIMEOUT,
        transition_timeout: float = TRANSITION_TIMEOUT,
        stop_timeout: float = STOP_TIMEOUT,
        switch_profile_with_commit: Callable[[str, Callable[[], None]], Any] | None = None,
        restart_profile_with_commit: Callable[[str, Callable[[], None]], Any] | None = None,
        stop_backend_with_commit: Callable[[Callable[[], None]], Any] | None = None,
        preempt_activation: Callable[[], Any] | None = None,
    ):
        for name, value in (
            ("drain_timeout", drain_timeout),
            ("transition_timeout", transition_timeout),
            ("stop_timeout", stop_timeout),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self._active_profile = initial_profile
        self._switch_profile = switch_profile
        self._stop_backend = stop_backend
        self._restart_profile = restart_profile or switch_profile
        self._switch_profile_with_commit = switch_profile_with_commit
        self._restart_profile_with_commit = restart_profile_with_commit
        self._stop_backend_with_commit = stop_backend_with_commit
        self._drain_timeout = drain_timeout
        self._transition_timeout = transition_timeout
        self._stop_timeout = stop_timeout
        self._active_requests = 0
        # Leases held by /slots and /props probes. They block transitions like
        # any other lease but never occupy a backend slot.
        self._probe_requests = 0
        self._queued_transitions = 0
        self._external_queued_transitions = 0
        self._transitioning = False
        self._transition_name: str | None = None
        self._transition_target: str | None = None
        self._pending_restore: tuple[str, str] | None = None
        self._restore_generation = object()
        self._temporary_restore: tuple[object, str, str | None] | None = None
        self._shutdown_requested = False
        self._preempt_activation = preempt_activation
        self._transition_preemptible = False
        self._transition_preempted = False
        self._slot_reservation: object | None = None
        self._condition = threading.Condition()

    def _quiescing_unlocked(self) -> bool:
        return (
            self._shutdown_requested
            or self._transitioning
            or self._queued_transitions > 0
            or self._external_queued_transitions > 0
        )

    @property
    def active_profile(self) -> str | None:
        with self._condition:
            return self._active_profile

    @property
    def active_requests(self) -> int:
        with self._condition:
            return self._active_requests

    @property
    def shutting_down(self) -> bool:
        with self._condition:
            return self._shutdown_requested

    @property
    def quiescing(self) -> bool:
        with self._condition:
            return self._quiescing_unlocked()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "profile": self._active_profile,
                "active_requests": self._active_requests,
                "inference_requests": self._active_requests - self._probe_requests,
                "quiescing": self._quiescing_unlocked(),
                "transitioning": self._transitioning,
                "queued_transitions": self._queued_transitions + self._external_queued_transitions,
                "external_queued_transitions": self._external_queued_transitions,
                "transition": self._transition_name,
                "transition_target": self._transition_target,
                "transition_preempted": self._transition_preempted,
                "pending_restore": self._pending_restore,
                "shutting_down": self._shutdown_requested,
            }

    def _queue_transition_unlocked(self) -> None:
        if self._shutdown_requested:
            raise BrokerQuiescingError("Llama-Broker is shutting down")
        self._queued_transitions += 1
        self._condition.notify_all()

    def _queue_transition(self) -> None:
        with self._condition:
            self._queue_transition_unlocked()

    def set_external_queued_transitions(self, count: int) -> None:
        """Reflect the durable queue in lease admission without losing reservations."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("external queued transition count must be a non-negative integer")
        with self._condition:
            if self._shutdown_requested and count:
                raise BrokerQuiescingError("Llama-Broker is shutting down")
            self._external_queued_transitions = count
            self._condition.notify_all()

    @contextmanager
    def transition_admission_guard(self) -> Iterator[None]:
        """Serialize durable acceptance with inference leases and transient restore."""
        with self._condition:
            if self._shutdown_requested:
                raise BrokerQuiescingError("Llama-Broker is shutting down")
            yield

    def begin_external_transition(self) -> None:
        """Atomically promote one durable queue entry to an active coordinator reservation."""
        with self._condition:
            if self._shutdown_requested:
                raise BrokerQuiescingError("Llama-Broker is shutting down")
            if self._external_queued_transitions <= 0:
                raise RuntimeError("no externally queued transition is available")
            self._external_queued_transitions -= 1
            self._queued_transitions += 1
            self._condition.notify_all()

    def cancel_temporary_restore_for_queued_operation(self) -> None:
        """An explicit queued operation supersedes any pending automatic restore."""
        with self._condition:
            self._cancel_temporary_restore_unlocked()
            self._condition.notify_all()

    def run_reserved_switch(
        self,
        target_profile: str,
        *,
        commit_callback: Callable[[], None] | None = None,
    ) -> Any:
        """Execute a switch for a reservation obtained via begin_external_transition."""
        with self._condition:
            if self._queued_transitions <= 0:
                raise RuntimeError("no transition reservation is available")
        return self._run_switch_transition(
            target_profile,
            commit_callback=commit_callback,
        )

    def _wait_for_transition_slot(
        self,
        operation_name: str,
        *,
        force_after_drain_timeout: bool = False,
        force_immediately: bool = False,
        preemptible: bool = False,
        reservation: object | None = None,
    ) -> bool:
        forced = False
        with self._condition:
            try:
                if self._shutdown_requested:
                    raise BrokerQuiescingError("Llama-Broker is shutting down")

                transition_deadline = time.monotonic() + self._transition_timeout
                while self._transitioning or (
                    self._slot_reservation is not None
                    and self._slot_reservation is not reservation
                ):
                    if self._shutdown_requested:
                        raise BrokerQuiescingError("Llama-Broker is shutting down")
                    remaining = transition_deadline - time.monotonic()
                    if remaining <= 0:
                        transition = self._transition_name or "an existing transition"
                        raise BrokerTransitionTimeoutError(
                            f"{operation_name} waited for {transition} beyond "
                            f"{self._transition_timeout:g}s; transition remains in progress"
                        )
                    self._condition.wait(timeout=remaining)

                if force_immediately:
                    forced = True
                else:
                    drain_deadline = time.monotonic() + self._drain_timeout
                    while self._active_requests > 0:
                        if self._shutdown_requested:
                            raise BrokerQuiescingError("Llama-Broker is shutting down")
                        remaining = drain_deadline - time.monotonic()
                        if remaining <= 0:
                            if force_after_drain_timeout:
                                forced = True
                                break
                            raise BrokerDrainTimeoutError(
                                f"{operation_name} could not drain "
                                f"{self._active_requests} active request(s) within "
                                f"{self._drain_timeout:g}s"
                            )
                        self._condition.wait(timeout=remaining)

                if self._shutdown_requested:
                    raise BrokerQuiescingError("Llama-Broker is shutting down")
                self._queued_transitions -= 1
                self._transitioning = True
                self._transition_name = operation_name
                self._transition_preemptible = preemptible
                self._transition_preempted = False
                if reservation is not None and self._slot_reservation is reservation:
                    self._slot_reservation = None
                self._condition.notify_all()
                return forced
            except BaseException:
                self._queued_transitions -= 1
                if reservation is not None and self._slot_reservation is reservation:
                    self._slot_reservation = None
                self._condition.notify_all()
                raise

    def _finish_empty_transition(self) -> None:
        with self._condition:
            failed_target = self._transition_target
            self._transitioning = False
            self._transition_name = None
            self._transition_target = None
            self._transition_preemptible = False
            self._transition_preempted = False
            if (
                self._pending_restore is not None
                and self._pending_restore[0] == failed_target
            ):
                self._pending_restore = None
                self._restore_generation = object()
            self._condition.notify_all()

    def _run_transition(
        self,
        operation_name: str,
        operation: Callable[[], Any],
        *,
        target_profile: str | None,
        timeout: float,
        acquire_lease: bool = False,
    ) -> Any:
        with self._condition:
            self._transition_target = target_profile
        finished = threading.Event()
        outcome: dict[str, Any] = {
            "abandoned": False,
            "done": False,
            "error": None,
            "preempted": False,
            "result": None,
        }

        def worker() -> None:
            try:
                result = operation()
                error: BaseException | None = None
            except BaseException as caught:
                result = None
                error = caught

            deferred_restore: tuple[str, str, object] | None = None
            with self._condition:
                outcome["result"] = result
                outcome["error"] = error
                outcome["preempted"] = self._transition_preempted
                abandoned = bool(outcome["abandoned"])
                if error is None:
                    self._active_profile = target_profile
                    if acquire_lease and not abandoned:
                        if self._shutdown_requested:
                            outcome["error"] = BrokerQuiescingError(
                                "Llama-Broker is shutting down"
                            )
                        else:
                            self._active_requests += 1
                if outcome["error"] is not None and (
                    outcome["preempted"]
                    or not isinstance(outcome["error"], LifecycleCancelledError)
                ):
                    # A failed lifecycle operation leaves ownership unknown. A
                    # pre-commit cancellation preserves the prior backend, but a
                    # preempted load has already released that backend.
                    self._active_profile = None
                self._transitioning = False
                self._transition_name = None
                self._transition_target = None
                self._transition_preemptible = False
                self._transition_preempted = False
                pending_restore = self._pending_restore
                if pending_restore is not None and pending_restore[0] == target_profile:
                    self._pending_restore = None
                    if error is None and not self._shutdown_requested:
                        deferred_restore = (
                            pending_restore[0],
                            pending_restore[1],
                            self._restore_generation,
                        )
                    else:
                        self._restore_generation = object()
                outcome["done"] = True
                finished.set()
                self._condition.notify_all()

            if abandoned:
                suffix = f": {error}" if error is not None else ""
                _log(f"[broker] {operation_name} completed after caller deadline{suffix}")
            if deferred_restore is not None:
                source_profile, restore_profile, restore_generation = deferred_restore
                with self._condition:
                    restore_is_current = (
                        restore_generation == self._restore_generation
                        and not self._shutdown_requested
                    )
                if restore_is_current:
                    try:
                        self.restore_when_idle(source_profile, restore_profile)
                    except (BackendControllerError, BrokerCoordinationError) as restore_error:
                        _log(
                            f"[broker] deferred restore of {source_profile!r} to "
                            f"{restore_profile!r} did not complete: {restore_error}"
                        )

        thread = threading.Thread(
            target=worker,
            name="llama-broker-transition",
            daemon=True,
        )
        try:
            thread.start()
        except BaseException:
            self._finish_empty_transition()
            raise
        if not finished.wait(timeout=timeout):
            with self._condition:
                if not outcome["done"]:
                    outcome["abandoned"] = True
                    timed_out = True
                else:
                    timed_out = False
            if timed_out:
                raise BrokerTransitionTimeoutError(
                    f"{operation_name} exceeded {timeout:g}s; operation remains "
                    "quarantined until its worker exits"
                )

        error = outcome["error"]
        if error is not None:
            if outcome["preempted"]:
                raise BrokerPreemptedError(
                    f"{operation_name} was preempted by a request for another profile"
                ) from error
            raise error
        return outcome["result"]

    def _quiescing_error_unlocked(self) -> BrokerQuiescingError:
        reason = self._transition_name or "a queued transition"
        return BrokerQuiescingError(
            f"broker is quiescing for {reason}; retry after the transition settles"
        )

    def _can_preempt_unlocked(self, profile_id: str) -> bool:
        return (
            self._preempt_activation is not None
            and not self._shutdown_requested
            and self._transitioning
            and self._transition_preemptible
            and not self._transition_preempted
            and self._transition_target is not None
            and self._transition_target != profile_id
            and self._active_requests == 0
            and self._queued_transitions == 0
            and self._external_queued_transitions == 0
            and self._slot_reservation is None
        )

    def _request_preemption(self, preempted: str | None, profile_id: str) -> None:
        try:
            accepted = bool(self._preempt_activation())  # type: ignore[misc]
        except BaseException as error:
            accepted = False
            _log(f"[broker] could not preempt {preempted}: {error}")
        if accepted:
            _log(
                f"[broker] preempting idle {preempted} for a request "
                f"for {profile_id!r}"
            )

    def _acquire(
        self,
        profile_id: str,
        *,
        transient: bool = False,
        admission_only: bool = False,
    ) -> tuple[object, str, str | None] | None:
        may_preempt = not transient
        while True:
            try:
                return self._acquire_once(
                    profile_id,
                    transient=transient,
                    may_preempt=may_preempt,
                    admission_only=admission_only,
                )
            except BrokerPreemptedError:
                if transient:
                    raise
                _log(
                    f"[broker] request for {profile_id!r} was preempted; "
                    "waiting to load it after the newer request"
                )
                may_preempt = False

    def _acquire_once(
        self,
        profile_id: str,
        *,
        transient: bool,
        may_preempt: bool,
        admission_only: bool = False,
    ) -> tuple[object, str, str | None] | None:
        temporary_restore: tuple[object, str, str | None] | None = None
        reservation: object | None = None
        preempted: str | None = None
        with self._condition:
            if admission_only and not self._inference_ready_unlocked(profile_id):
                raise self._quiescing_error_unlocked()
            if self._active_profile == profile_id:
                if self._quiescing_unlocked():
                    raise self._quiescing_error_unlocked()
                self._active_requests += 1
                return None
            if transient and self._temporary_restore is not None:
                raise BrokerQuiescingError(
                    "another temporary profile transition is already active"
                )
            if transient and self._quiescing_unlocked():
                raise self._quiescing_error_unlocked()
            if transient:
                temporary_restore = (object(), profile_id, self._active_profile)
            if may_preempt and self._can_preempt_unlocked(profile_id):
                reservation = object()
                preempted = self._transition_name
            self._queue_transition_unlocked()
            if reservation is not None:
                self._slot_reservation = reservation
                self._transition_preempted = True
            if temporary_restore is not None:
                self._temporary_restore = temporary_restore

        if reservation is not None:
            self._request_preemption(preempted, profile_id)
        operation_name = f"request switch to {profile_id!r}"
        try:
            self._wait_for_transition_slot(
                operation_name,
                preemptible=not transient,
                reservation=reservation,
            )
            if temporary_restore is not None:
                with self._condition:
                    temporary_restore_is_current = self._temporary_restore is temporary_restore
                if not temporary_restore_is_current:
                    self._finish_empty_transition()
                    raise BrokerQuiescingError(
                        "temporary profile transition was cancelled"
                    )
            self._run_transition(
                operation_name,
                lambda: self._switch_profile(profile_id),
                target_profile=profile_id,
                timeout=self._transition_timeout,
                acquire_lease=True,
            )
        except BaseException:
            if temporary_restore is not None:
                with self._condition:
                    if self._temporary_restore is temporary_restore:
                        self._temporary_restore = None
                        self._condition.notify_all()
            raise
        return temporary_restore

    def _acquire_active(self, *, probe: bool = False) -> str:
        with self._condition:
            if self._quiescing_unlocked():
                raise self._quiescing_error_unlocked()
            if self._active_profile is None:
                raise BrokerQuiescingError("private backend has no active profile")
            self._active_requests += 1
            if probe:
                self._probe_requests += 1
            return self._active_profile

    def _release(self, *, probe: bool = False) -> None:
        with self._condition:
            if self._active_requests <= 0:
                raise RuntimeError("profile lease underflow")
            self._active_requests -= 1
            if probe:
                self._probe_requests -= 1
            if self._active_requests == 0:
                self._condition.notify_all()

    @contextmanager
    def lease(
        self, profile_id: str, transient: bool = False, *, admission_only: bool = False
    ) -> Iterator[None]:
        temporary_restore = self._acquire(
            profile_id, transient=transient, admission_only=admission_only
        )
        try:
            yield
        finally:
            self._release()
            if temporary_restore is not None:
                self._complete_temporary_restore(temporary_restore)

    def _inference_ready_unlocked(self, profile_id: str) -> bool:
        return not self._quiescing_unlocked() and (
            self._temporary_restore is None or self._temporary_restore[1] == profile_id
        ) and (
            self._active_profile == profile_id or self._active_requests == 0
        )

    def inference_ready(self, profile_id: str) -> bool:
        with self._condition:
            return self._inference_ready_unlocked(profile_id)

    def request_shutdown(self) -> None:
        with self._condition:
            self._shutdown_requested = True
            self._cancel_temporary_restore_unlocked()
            self._condition.notify_all()

    def _complete_temporary_restore(
        self,
        temporary_restore: tuple[object, str, str | None],
    ) -> None:
        _, source_profile, restore_profile = temporary_restore
        with self._condition:
            if self._temporary_restore is not temporary_restore:
                return
        if restore_profile is not None:
            try:
                self.restore_when_idle(source_profile, restore_profile)
            except BaseException as error:
                _log(
                    f"[broker] failed to restore temporary profile "
                    f"{source_profile!r} to {restore_profile!r}: {error}"
                )
        with self._condition:
            if self._temporary_restore is temporary_restore:
                self._temporary_restore = None
                self._condition.notify_all()

    @contextmanager
    def lease_active(self, *, probe: bool = False) -> Iterator[str]:
        profile_id = self._acquire_active(probe=probe)
        try:
            yield profile_id
        finally:
            self._release(probe=probe)

    def restore_when_idle(self, source_profile: str, target_profile: str) -> None:
        """Restore a transient source with the same bounded writer priority."""

        with self._condition:
            if self._active_profile != source_profile:
                if (
                    not self._shutdown_requested
                    and self._transitioning
                    and self._transition_target == source_profile
                ):
                    self._pending_restore = (source_profile, target_profile)
                    self._restore_generation = object()
                    self._condition.notify_all()
                return
            self._queue_transition_unlocked()
        operation_name = f"restore {source_profile!r} to {target_profile!r}"
        self._wait_for_transition_slot(operation_name)
        with self._condition:
            source_is_still_active = self._active_profile == source_profile
        if not source_is_still_active:
            self._finish_empty_transition()
            return
        self._run_transition(
            operation_name,
            lambda: self._switch_profile(target_profile),
            target_profile=target_profile,
            timeout=self._transition_timeout,
        )

    def switch_when_idle(self, target_profile: str) -> Any:
        """Bound the drain and transition phases of an administrative switch."""

        with self._condition:
            self._cancel_temporary_restore_unlocked()
            self._condition.notify_all()
        self._queue_transition()
        return self._run_switch_transition(target_profile)

    def _cancel_temporary_restore_unlocked(self) -> None:
        self._pending_restore = None
        self._temporary_restore = None
        self._restore_generation = object()

    def run_reserved_restart(
        self,
        profile_id: str,
        *,
        commit_callback: Callable[[], None] | None = None,
    ) -> Any:
        """Execute a priority restart for a previously reserved operation."""
        with self._condition:
            if self._queued_transitions <= 0:
                raise RuntimeError("no transition reservation is available")
            if self._active_profile != profile_id:
                raise BrokerCoordinationError(
                    f"cannot restart inactive profile {profile_id!r}"
                )
            self._cancel_temporary_restore_unlocked()
        operation_name = f"restart active profile {profile_id!r}"
        forced = self._wait_for_transition_slot(
            operation_name,
            force_immediately=True,
        )
        if forced:
            with self._condition:
                active_requests = self._active_requests
            _log(
                f"[broker] {operation_name} interrupting "
                f"{active_requests} active request(s); restarting immediately"
            )
        restart_with_commit = self._restart_profile_with_commit
        if commit_callback is None:
            operation = lambda: self._restart_profile(profile_id)
        elif restart_with_commit is None:
            operation = lambda: (commit_callback(), self._restart_profile(profile_id))[1]
        else:
            operation = lambda: restart_with_commit(profile_id, commit_callback)
        return self._run_transition(
            operation_name,
            operation,
            target_profile=profile_id,
            timeout=self._transition_timeout,
        )

    def run_reserved_reload(
        self,
        action: Callable[[Callable[[], None]], Any],
        *,
        commit_callback: Callable[[], None],
    ) -> Any:
        """Run a profile/catalog reload in the shared normal FIFO lane."""
        with self._condition:
            if self._queued_transitions <= 0:
                raise RuntimeError("no transition reservation is available")
            self._cancel_temporary_restore_unlocked()
            target_profile = self._active_profile
            self._condition.notify_all()
        operation_name = "profile/catalog reload"
        # Reload swaps profile metadata and generated units without replacing
        # the running backend, so in-flight requests can safely keep their
        # existing profile snapshot while the new registry becomes active.
        self._wait_for_transition_slot(operation_name, force_immediately=True)
        return self._run_transition(
            operation_name,
            lambda: action(commit_callback),
            target_profile=target_profile,
            timeout=self._transition_timeout,
        )

    def run_reserved_stop(
        self,
        *,
        commit_callback: Callable[[], None] | None = None,
    ) -> Any:
        """Execute a priority unload/stop for a previously reserved operation."""
        if self._stop_backend is None:
            raise RuntimeError("backend stop is not configured")
        if self._queued_transitions <= 0:
            raise RuntimeError("no transition reservation is available")
        with self._condition:
            self._cancel_temporary_restore_unlocked()
            self._condition.notify_all()
        operation_name = "admin stop"
        forced = self._wait_for_transition_slot(
            operation_name,
            force_after_drain_timeout=True,
        )
        if forced:
            with self._condition:
                active_requests = self._active_requests
            _log(
                f"[broker] admin stop drain deadline expired with "
                f"{active_requests} active request(s); invoking the configured stop"
            )
        stop_with_commit = self._stop_backend_with_commit
        if commit_callback is None:
            operation = self._stop_backend
        elif stop_with_commit is None:
            operation = lambda: (commit_callback(), self._stop_backend())[1]
        else:
            operation = lambda: stop_with_commit(commit_callback)
        return self._run_transition(
            operation_name,
            operation,
            target_profile=None,
            timeout=self._stop_timeout,
        )

    def _run_switch_transition(
        self,
        target_profile: str,
        *,
        commit_callback: Callable[[], None] | None = None,
    ) -> Any:
        operation_name = f"admin switch to {target_profile!r}"
        self._wait_for_transition_slot(operation_name)
        switch_with_commit = self._switch_profile_with_commit
        if commit_callback is None:
            operation = lambda: self._switch_profile(target_profile)
        elif switch_with_commit is None:
            operation = lambda: (
                commit_callback(),
                self._switch_profile(target_profile),
            )[1]
        else:
            operation = lambda: switch_with_commit(target_profile, commit_callback)
        return self._run_transition(
            operation_name,
            operation,
            target_profile=target_profile,
            timeout=self._transition_timeout,
        )

    def switch_when_idle_async(self, target_profile: str) -> None:
        """Queue an administrative switch and return before its backend work completes."""

        with self._condition:
            self._cancel_temporary_restore_unlocked()
            self._condition.notify_all()
        self._queue_transition()

        def run_transition() -> None:
            try:
                self._run_switch_transition(target_profile)
            except BaseException as error:
                _log(
                    f"[broker] asynchronous admin switch to {target_profile!r} "
                    f"did not complete: {error}"
                )

        thread = threading.Thread(
            target=run_transition,
            name="llama-broker-admin-transition",
            daemon=True,
        )
        try:
            thread.start()
        except BaseException:
            with self._condition:
                self._queued_transitions -= 1
                self._condition.notify_all()
            raise

    def restart_when_idle(self, profile_id: str | None = None) -> Any:
        """Immediately interrupt active requests, then restart the active profile."""

        with self._condition:
            target_profile = profile_id or self._active_profile
            if target_profile is None:
                raise BrokerQuiescingError("private backend has no active profile")
            if self._active_profile != target_profile:
                raise BrokerCoordinationError(
                    f"cannot restart inactive profile {target_profile!r}"
                )
            self._cancel_temporary_restore_unlocked()
            self._condition.notify_all()
        self._queue_transition()
        operation_name = f"restart active profile {target_profile!r}"
        forced = self._wait_for_transition_slot(
            operation_name,
            force_immediately=True,
        )
        if forced:
            with self._condition:
                active_requests = self._active_requests
            _log(
                f"[broker] {operation_name} interrupting "
                f"{active_requests} active request(s); restarting immediately"
            )
        return self._run_transition(
            operation_name,
            lambda: self._restart_profile(target_profile),
            target_profile=target_profile,
            timeout=self._transition_timeout,
        )

    def stop_when_idle(self) -> Any:
        """Drain, then escalate to the configured graceful stop at the deadline."""

        if self._stop_backend is None:
            raise RuntimeError("backend stop is not configured")
        with self._condition:
            self._cancel_temporary_restore_unlocked()
            self._condition.notify_all()
        self._queue_transition()
        operation_name = "admin stop"
        forced = self._wait_for_transition_slot(
            operation_name,
            force_after_drain_timeout=True,
        )
        if forced:
            with self._condition:
                active_requests = self._active_requests
            _log(
                f"[broker] admin stop drain deadline expired with "
                f"{active_requests} active request(s); invoking the configured "
                "stop; the controller will force-kill only if its bounded "
                "stop command times out"
            )
        return self._run_transition(
            operation_name,
            self._stop_backend,
            target_profile=None,
            timeout=self._stop_timeout,
        )

    def stop_for_process_shutdown(self) -> Any:
        """Explicitly stop runtimes during destructive process teardown.

        Normal broker restarts must use :meth:`handoff_for_process_shutdown`.
        This method remains available only for callers that explicitly require
        all runtime services to be unloaded.
        """

        if self._stop_backend is None:
            return None
        operation_name = "destructive broker shutdown"
        forced = False
        with self._condition:
            self._shutdown_requested = True
            self._cancel_temporary_restore_unlocked()
            self._condition.notify_all()
            while self._transitioning:
                self._condition.wait(timeout=0.25)
            deadline = time.monotonic() + self._drain_timeout
            while self._active_requests > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    forced = True
                    break
                self._condition.wait(timeout=remaining)
            self._transitioning = True
            self._transition_name = operation_name
            self._condition.notify_all()
        if forced:
            _log(
                f"[broker] {operation_name} drain deadline expired with "
                f"{self.active_requests} active request(s); stopping runtimes"
            )
        try:
            result = self._stop_backend()
        except BaseException:
            with self._condition:
                self._transitioning = False
                self._transition_name = None
                self._transition_target = None
                self._condition.notify_all()
            raise
        with self._condition:
            self._active_profile = None
            self._transitioning = False
            self._transition_name = None
            self._transition_target = None
            self._condition.notify_all()
        return result

    def handoff_for_process_shutdown(self) -> None:
        """Quiesce broker ownership without stopping the runtime service.

        The broker control plane and the broker-owned profile service have
        independent lifecycles. A graceful broker restart therefore rejects new
        work and waits for in-flight requests/transitions, but leaves the active
        profile unit, alias, and durable controller state untouched.
        """

        operation_name = "broker control-plane handoff"
        deadline = time.monotonic() + self._drain_timeout + self._transition_timeout
        with self._condition:
            self._shutdown_requested = True
            self._cancel_temporary_restore_unlocked()
            self._condition.notify_all()
            while self._transitioning or self._active_requests > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _log(
                        f"[broker] {operation_name} deadline expired with "
                        f"{self._active_requests} active request(s) and "
                        f"transitioning={self._transitioning}; preserving runtime state"
                    )
                    return
                self._condition.wait(timeout=min(remaining, 0.25))


    def invalidate_when_idle(self, expected_profile: str) -> bool:
        """Forget a dead backend so the next request performs a real switch."""

        with self._condition:
            if self._quiescing_unlocked() or self._active_requests > 0:
                return False
            if self._active_profile != expected_profile:
                return False
            self._active_profile = None
            self._condition.notify_all()
            return True

    def synchronize_after_controller_failure(
        self, active_profile: str | None
    ) -> None:
        """Publish the controller's last proven profile after lifecycle failure."""

        if active_profile is not None and not isinstance(active_profile, str):
            raise TypeError("active_profile must be a string or None")
        with self._condition:
            self._active_profile = active_profile
            self._condition.notify_all()


class BrokerApplication:
    def __init__(
        self,
        *,
        registry: ProfileRegistry,
        controller: Controller,
        backend_host: str,
        backend_port: int,
        backend_timeout: float,
        max_body_bytes: int,
        max_output_tokens: int | None = None,
        backend_url: str | None = None,
        security: SecurityConfig | None = None,
        backend_ready: Callable[[], bool] | None = None,
        backend_alias_ready: Callable[[str], bool] | None = None,
        drain_timeout: float = DRAIN_TIMEOUT,
        transition_timeout: float = TRANSITION_TIMEOUT,
        stop_timeout: float = STOP_TIMEOUT,
        start_operation_worker: bool = True,
        profiles_path: Path | str | None = None,
        preempt_idle_loads: bool = PREEMPT_IDLE_LOADS,
        system1: System1Service | None = None,
    ) -> None:
        if not math.isfinite(backend_timeout) or backend_timeout <= 0:
            raise ValueError("backend_timeout must be finite and positive")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.registry = registry
        self._profiles_path = Path(profiles_path) if profiles_path is not None else None
        self.system1 = system1
        self.controller = controller
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.backend_timeout = backend_timeout
        self.max_body_bytes = max_body_bytes
        self.max_output_tokens = max_output_tokens
        self.admin_operation_timeout = drain_timeout + transition_timeout + 1.0
        self.security = security or SecurityConfig()
        self._state_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = False
        self._shutdown_error: BaseException | None = None
        self._recovery_stop = threading.Event()
        self._recovery_thread: threading.Thread | None = None
        self._recovery_attempts = 0
        self._recovery_next_at = 0.0
        self._recovery_profile: str | None = None
        self._autotune_lease = MaintenanceLease(AUTOTUNE_LEASE_PATH)
        self._autotune_degraded_reason: str | None = None
        self.crash_loops = CrashLoopGuard(
            threshold=CRASH_LOOP_THRESHOLD,
            window=CRASH_LOOP_WINDOW,
            cooldown=CRASH_LOOP_COOLDOWN,
        )
        state = dict(controller.state())
        self._controller_state = state

        self._backend_url = (backend_url or f"http://{backend_host}:{backend_port}").rstrip("/")
        self._backend_ca_file = self.security.backend_tls_ca_file
        probe_timeout = min(max(backend_timeout, 0.05), 0.5)
        self._backend_ready = backend_ready or (
            lambda: backend_endpoint_ready(backend_host, backend_port, probe_timeout)
        )

        def readiness_for_alias(alias: str) -> bool:
            profile = self.registry.resolve(alias)
            endpoint = backend_url_for_runtime(self._backend_url, profile.runtime)
            endpoint_scheme = urlsplit(endpoint).scheme or "http"
            return backend_serves_alias(
                backend_host,
                backend_port,
                getattr(profile, "backend_model_name", profile.alias),
                probe_timeout,
                scheme=endpoint_scheme,
                ca_file=self._backend_ca_file,
                headers=_backend_auth_headers(self.security),
            )

        self._backend_alias_ready = backend_alias_ready or readiness_for_alias

        initial = state.get("profile") if state.get("phase") == "active" else None
        if initial is not None and not isinstance(initial, str):
            initial = None
        if initial is not None:
            try:
                initial_alias = self.registry.get(initial).alias
                if not self._backend_alias_ready(initial_alias):
                    initial = None
            except Exception:
                initial = None
        self.coordinator = ProfileCoordinator(
            initial,
            self._switch_profile,
            self._stop_backend,
            self._restart_profile,
            drain_timeout=drain_timeout,
            transition_timeout=transition_timeout,
            stop_timeout=stop_timeout,
            switch_profile_with_commit=self._switch_profile_with_commit,
            restart_profile_with_commit=self._restart_profile_with_commit,
            stop_backend_with_commit=self._stop_backend_with_commit,
            preempt_activation=(
                getattr(controller, "preempt_activation", None)
                if preempt_idle_loads
                else None
            ),
        )
        self._operation_admission_lock = threading.RLock()
        self._operation_condition = threading.Condition()
        self._operation_worker_stop = threading.Event()
        self._operation_worker_wake = threading.Event()
        self._operation_worker_thread: threading.Thread | None = None
        self._manifest_watcher_stop = threading.Event()
        self._manifest_watcher_thread: threading.Thread | None = None
        self._last_manifest_identity = getattr(self.registry, "manifest_identity", None)
        self._pending_manifest_reload: RequestView | None = None
        state_store = getattr(controller, "state_store", None)
        if state_store is None:
            self.operation_queue: TransitionQueue | PersistentTransitionQueue = TransitionQueue()
        else:
            self.operation_queue = PersistentTransitionQueue(state_store)
        self.coordinator.set_external_queued_transitions(
            self.operation_queue.pending_transition_count
        )
        if self.operation_queue.needs_reconciliation:
            reconcile = getattr(controller, "reconcile", None)
            if not callable(reconcile):
                raise BrokerStateUnknownError(
                    "operation recovery requires backend reconciliation"
                )
            recovered_state = dict(reconcile(active_probe_timeout=0.5))
            self._remember_controller_state(recovered_state)
            recovered_profile = (
                recovered_state.get("profile")
                if recovered_state.get("phase") == "active"
                else None
            )
            self.coordinator.synchronize_after_controller_failure(
                recovered_profile if isinstance(recovered_profile, str) else None
            )
            self.operation_queue.acknowledge_reconciliation()
        if start_operation_worker:
            self.start_operation_worker()

    def start_operation_worker(self) -> None:
        """Start the single durable transition worker, if not already running."""
        thread = self._operation_worker_thread
        if thread is not None and thread.is_alive():
            return
        self._operation_worker_stop.clear()
        thread = threading.Thread(
            target=self._operation_worker_loop,
            name="llama-broker-operation-worker",
            daemon=True,
        )
        self._operation_worker_thread = thread
        thread.start()
        if self.operation_queue.pending_transition_count:
            self._operation_worker_wake.set()

    def _validate_execution_snapshot(self, execution: ExecutionView) -> None:
        captured = self.operation_queue.get_profile_snapshot(execution.execution_id)
        if execution.operation_type == "reload":
            if captured is None or self._profiles_path is None:
                raise StaleOperationSnapshotError(
                    "profile reload has no captured catalog revision; resubmit it"
                )
            try:
                current_registry = ProfileRegistry.from_path(self._profiles_path)
                current_revision = current_registry.snapshot_revision_identity()
            except ProfileError as error:
                raise StaleOperationSnapshotError(
                    f"profile/catalog revision is no longer valid: {error}"
                ) from error
            if current_revision != captured:
                raise StaleOperationSnapshotError(
                    "profile/catalog files changed after reload submission; resubmit"
                )
            return
        if execution.operation_type not in {"switch", "restart"}:
            return
        target = execution.target_profile_id
        if not isinstance(target, str):
            raise StaleOperationSnapshotError("transition snapshot has no target profile")
        if captured is None:
            raise StaleOperationSnapshotError(
                "operation has no immutable profile snapshot; resubmit it"
            )
        try:
            current = self.registry.profile_snapshot_identity(target)
        except ProfileError as error:
            raise StaleOperationSnapshotError(
                f"captured profile {target!r} is no longer available; resubmit"
            ) from error
        if current != captured:
            raise StaleOperationSnapshotError(
                f"profile/catalog facts changed after request submission for {target!r}; resubmit"
            )
        if getattr(self.controller, "state_store", None) is not None:
            profile = self.registry.get(target)
            if not self._profile_artifacts_available(profile):
                raise StaleOperationSnapshotError(
                    f"required artifacts are missing or unsafe for {target!r}"
                )

    def _operation_worker_loop(self) -> None:
        while not self._operation_worker_stop.is_set():
            self._operation_worker_wake.wait()
            self._operation_worker_wake.clear()
            if self._operation_worker_stop.is_set():
                return
            while not self._operation_worker_stop.is_set():
                with self._operation_admission_lock:
                    if self.coordinator.shutting_down:
                        return
                    execution = self.operation_queue.next_execution()
                    if execution is None:
                        self.coordinator.set_external_queued_transitions(
                            self.operation_queue.pending_transition_count
                        )
                        break
                    try:
                        if execution.operation_type in {"switch", "restart"}:
                            self._validate_execution_snapshot(execution)
                    except StaleOperationSnapshotError as snapshot_error:
                        self.operation_queue.finish(
                            execution.execution_id,
                            error=str(snapshot_error),
                            error_type=type(snapshot_error).__name__,
                        )
                        self.coordinator.set_external_queued_transitions(
                            self.operation_queue.pending_transition_count
                        )
                        self._notify_operation_waiters()
                        continue
                    try:
                        self.coordinator.begin_external_transition()
                        self.coordinator.set_external_queued_transitions(
                            self.operation_queue.pending_transition_count
                        )
                    except BaseException as error:
                        _log(f"[broker] could not reserve queued operation: {error}")
                        return
                    current_execution = execution
                self._notify_operation_waiters()

                def mark_committed(
                    execution_id: str = current_execution.execution_id,
                ) -> None:
                    self._validate_execution_snapshot(current_execution)
                    self.operation_queue.mark_committed(execution_id)

                try:
                    if execution.operation_type == "switch":
                        if execution.target_profile_id is None:
                            raise OperationStateError("switch execution has no target profile")
                        self.coordinator.run_reserved_switch(
                            execution.target_profile_id,
                            commit_callback=mark_committed,
                        )
                    elif execution.operation_type == "restart":
                        if execution.target_profile_id is None:
                            raise OperationStateError("restart execution has no target profile")
                        self.coordinator.run_reserved_restart(
                            execution.target_profile_id,
                            commit_callback=mark_committed,
                        )
                    elif execution.operation_type == "stop":
                        self.coordinator.run_reserved_stop(commit_callback=mark_committed)
                    elif execution.operation_type == "reload":
                        self.coordinator.run_reserved_reload(
                            self.reload_profiles_for_operation,
                            commit_callback=mark_committed,
                        )
                    else:
                        raise OperationStateError(
                            f"unsupported operation type: {execution.operation_type}"
                        )
                except BaseException as caught_error:
                    error: BaseException = caught_error
                    if execution.operation_type in {"switch", "restart"}:
                        try:
                            self._validate_execution_snapshot(current_execution)
                        except StaleOperationSnapshotError as snapshot_error:
                            error = snapshot_error
                            self._refresh_after_lifecycle(raise_on_error=False)
                    try:
                        self._finish_operation(
                            execution, error=str(error), error_type=type(error).__name__,
                        )
                    except BaseException as persistence_error:
                        _log(
                            f"[broker] failed to persist operation failure "
                            f"{execution.execution_id}: {persistence_error}"
                        )
                        with self._operation_admission_lock:
                            self.coordinator.set_external_queued_transitions(
                                max(1, self.operation_queue.pending_transition_count)
                            )
                        self._notify_operation_waiters()
                        return
                else:
                    try:
                        self._finish_operation(execution)
                    except BaseException as persistence_error:
                        _log(
                            f"[broker] failed to persist operation completion "
                            f"{execution.execution_id}: {persistence_error}"
                        )
                        with self._operation_admission_lock:
                            self.coordinator.set_external_queued_transitions(
                                max(1, self.operation_queue.pending_transition_count)
                            )
                        self._notify_operation_waiters()
                        return
                with self._operation_admission_lock:
                    self.coordinator.set_external_queued_transitions(
                        self.operation_queue.pending_transition_count
                    )
                self._notify_operation_waiters()

    def _finish_operation(
        self, execution: ExecutionView, *, error: str | None = None,
        error_type: str | None = None,
    ) -> None:
        """Publish terminal state and clear matching watcher bookkeeping.

        Successful queue completion also reconciles the loaded baseline; actual
        application publishes it at the registry swap even after a queue timeout.
        Admission serialization prevents completion racing pending registration.
        """
        with self._operation_admission_lock:
            finished = self.operation_queue.finish(
                execution.execution_id, error=error, error_type=error_type,
            )
            if execution.operation_type == "reload":
                if finished.status == "succeeded":
                    self._last_manifest_identity = getattr(self.registry, "manifest_identity", None)
                pending = self._pending_manifest_reload
                if pending is not None and pending.execution_id == execution.execution_id:
                    self._pending_manifest_reload = None

    def _notify_operation_waiters(self) -> None:
        with self._operation_condition:
            self._operation_condition.notify_all()

    def _registry_for_submission_unlocked(self) -> ProfileRegistry:
        pending_reload = any(
            execution.operation_type == "reload"
            and execution.status in {"queued", "preflight", "committed"}
            for execution in self.operation_queue.list_executions()
        )
        if pending_reload and self._profiles_path is not None:
            return ProfileRegistry.from_path(self._profiles_path)
        return self.registry

    def submit_switch_operation(self, profile_id: str) -> RequestView:
        """Persist a request against the registry revision at its FIFO position."""
        with self._operation_admission_lock, self.coordinator.transition_admission_guard():
            submission_registry = self._registry_for_submission_unlocked()
            profile = submission_registry.resolve(profile_id)
            profile_snapshot = submission_registry.profile_snapshot_identity(profile.id)
            previous_depth = self.operation_queue.pending_transition_count
            self.coordinator.set_external_queued_transitions(previous_depth + 1)
            try:
                request = self.operation_queue.submit_load(
                    profile.id,
                    profile_snapshot=profile_snapshot,
                )
            except BaseException:
                self.coordinator.set_external_queued_transitions(previous_depth)
                raise
            self.crash_loops.clear(profile.id)
            self.coordinator.cancel_temporary_restore_for_queued_operation()
            self.coordinator.set_external_queued_transitions(
                self.operation_queue.pending_transition_count
            )
            self._operation_worker_wake.set()
        self._notify_operation_waiters()
        return request

    def submit_restart_operation(self, profile_id: str) -> RequestView:
        """Queue a priority restart, or a normal load if target is inactive."""
        with self._operation_admission_lock, self.coordinator.transition_admission_guard():
            submission_registry = self._registry_for_submission_unlocked()
            profile = submission_registry.resolve(profile_id)
            profile_snapshot = submission_registry.profile_snapshot_identity(profile.id)
            previous = self.operation_queue.pending_transition_count
            self.coordinator.set_external_queued_transitions(previous + 1)
            try:
                if self.coordinator.active_profile != profile.id:
                    request = self.operation_queue.submit_load(
                        profile.id,
                        profile_snapshot=profile_snapshot,
                    )
                else:
                    request = self.operation_queue.submit_restart(
                        profile.id,
                        profile_snapshot=profile_snapshot,
                    )
            except BaseException:
                self.coordinator.set_external_queued_transitions(previous)
                raise
            self.crash_loops.clear(profile.id)
            self.coordinator.cancel_temporary_restore_for_queued_operation()
            self.coordinator.set_external_queued_transitions(
                self.operation_queue.pending_transition_count
            )
            self._operation_worker_wake.set()
        self._notify_operation_waiters()
        return request

    def submit_stop_operation(self) -> RequestView:
        with self._operation_admission_lock, self.coordinator.transition_admission_guard():
            previous = self.operation_queue.pending_transition_count
            self.coordinator.set_external_queued_transitions(previous + 1)
            try:
                request = self.operation_queue.submit_stop()
            except BaseException:
                self.coordinator.set_external_queued_transitions(previous)
                raise
            self.crash_loops.clear()
            self.coordinator.cancel_temporary_restore_for_queued_operation()
            self.coordinator.set_external_queued_transitions(
                self.operation_queue.pending_transition_count
            )
            self._operation_worker_wake.set()
        self._notify_operation_waiters()
        return request

    def submit_profile_reload_operation(self) -> RequestView:
        if self._profiles_path is None:
            raise BrokerLifecycleError("profile reload source path is not configured")
        with self._operation_admission_lock, self.coordinator.transition_admission_guard():
            reload_registry = ProfileRegistry.from_path(self._profiles_path)
            profile_snapshot = reload_registry.snapshot_revision_identity()
            previous = self.operation_queue.pending_transition_count
            self.coordinator.set_external_queued_transitions(previous + 1)
            try:
                request = self.operation_queue.submit_reload(
                    profile_snapshot=profile_snapshot
                )
            except BaseException:
                self.coordinator.set_external_queued_transitions(previous)
                raise
            # A reload may change the crashing profile's configuration.
            self.crash_loops.clear()
            self.coordinator.cancel_temporary_restore_for_queued_operation()
            self.coordinator.set_external_queued_transitions(
                self.operation_queue.pending_transition_count
            )
            self._operation_worker_wake.set()
        self._notify_operation_waiters()
        return request

    def operation_list(self) -> dict[str, Any]:
        """Return live work and retained completion records for the TUI/CLI."""
        entries: list[dict[str, Any]] = []
        for execution in self.operation_queue.list_executions():
            entries.append(
                {
                    "execution": {
                        "execution_id": execution.execution_id,
                        "type": execution.operation_type,
                        "target": execution.target_profile_id,
                        "status": execution.status,
                        "cancel_requested": execution.cancel_requested,
                        "error": execution.error,
                        "error_type": execution.error_type,
                    },
                    "requests": [
                        self.operation_status(request_id)
                        for request_id in execution.request_ids
                    ],
                }
            )
        return {"operations": entries}

    def operation_status(self, request_id: str) -> dict[str, Any]:
        request = self.operation_queue.get_request(request_id)
        execution = self.operation_queue.get_execution(request.execution_id)
        return {
            "request_id": request.request_id,
            "execution_id": request.execution_id,
            "type": execution.operation_type,
            "target": execution.target_profile_id,
            "status": request.status,
            "waiting": request.waiting,
            "error": request.error,
            "error_type": request.error_type,
            "execution": {
                "status": execution.status,
                "cancel_requested": execution.cancel_requested,
                "error": execution.error,
                "error_type": execution.error_type,
            },
        }

    def cancel_operation(self, request_id: str) -> dict[str, Any]:
        with self._operation_admission_lock:
            cancelled = self.operation_queue.cancel_request(request_id)
            self.coordinator.set_external_queued_transitions(
                self.operation_queue.pending_transition_count
            )
            self._operation_worker_wake.set()
        self._notify_operation_waiters()
        return {
            "cancelled": cancelled,
            "operation": self.operation_status(request_id),
        }

    def wait_for_operation(self, request_id: str, timeout: float) -> dict[str, Any]:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        deadline = time.monotonic() + timeout
        with self._operation_condition:
            while True:
                operation = self.operation_status(request_id)
                if operation["status"] in {"succeeded", "failed", "cancelled"}:
                    return operation
                if not operation["waiting"]:
                    return operation
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"operation {request_id} is still running")
                self._operation_condition.wait(timeout=remaining)

    def _stop_operation_worker(self) -> None:
        self._operation_worker_stop.set()
        self._operation_worker_wake.set()
        thread = self._operation_worker_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.coordinator._drain_timeout + self.coordinator._transition_timeout + 1)
        self._operation_worker_thread = None

    def start_recovery_watchdog(self) -> None:
        """Start the internal watchdog for the currently active backend only."""

        if RECOVERY_INTERVAL <= 0 or RECOVERY_MAX_BACKOFF <= 0:
            raise ValueError("recovery intervals must be positive")
        if self._recovery_thread is not None and self._recovery_thread.is_alive():
            return
        self._recovery_stop.clear()
        self._recovery_thread = threading.Thread(
            target=self._recovery_loop,
            name="llama-broker-backend-recovery",
            daemon=True,
        )
        self._recovery_thread.start()

    def _recovery_loop(self) -> None:
        while not self._recovery_stop.wait(RECOVERY_INTERVAL):
            try:
                active_profile = self.coordinator.active_profile
                if (
                    active_profile is not None
                    and self._recovery_profile is not None
                    and active_profile != self._recovery_profile
                ):
                    self._recovery_profile = None
                profile_id = active_profile or self._recovery_profile
                if profile_id is None or self.coordinator.quiescing:
                    self._recovery_attempts = 0
                    self._recovery_next_at = 0.0
                    continue
                profile = self.registry.get(profile_id)
                if self._backend_alias_ready(profile.alias):
                    self._recovery_attempts = 0
                    self._recovery_next_at = 0.0
                    continue
                now = time.monotonic()
                if now < self._recovery_next_at:
                    continue
                try:
                    if self.coordinator.active_profile == profile_id:
                        self.coordinator.restart_when_idle(profile_id)
                    else:
                        self.coordinator.switch_when_idle(profile_id)
                except BaseException as error:
                    self._recovery_profile = profile_id
                    self._recovery_attempts += 1
                    backoff = min(
                        RECOVERY_MAX_BACKOFF,
                        RECOVERY_INTERVAL * (2 ** min(self._recovery_attempts - 1, 5)),
                    )
                    self._recovery_next_at = time.monotonic() + backoff
                    try:
                        state = self.controller.record_recovery_failure(
                            profile_id,
                            attempts=self._recovery_attempts,
                            error=str(error),
                        )
                        self._remember_controller_state(state)
                    except BaseException as record_error:
                        _log(
                            f"[broker] backend recovery state update failed for "
                            f"{profile_id!r}: {record_error}"
                        )
                    _log(
                        f"[broker] backend recovery attempt {self._recovery_attempts} "
                        f"for {profile_id!r} failed; retrying in {backoff:g}s: {error}"
                    )
                else:
                    self._recovery_attempts = 0
                    self._recovery_next_at = 0.0
                    self._recovery_profile = None
                    _log(f"[broker] recovered active backend {profile_id!r}")
            except BaseException as error:
                _log(f"[broker] recovery watchdog check failed: {error}")

    def _stop_recovery_watchdog(self) -> None:
        self._recovery_stop.set()
        thread = self._recovery_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, RECOVERY_INTERVAL + 1.0))
        self._recovery_thread = None

    def start_manifest_watcher(self) -> None:
        """Start the background watcher that hot-reloads the profile manifest."""

        if MANIFEST_RELOAD_INTERVAL <= 0:
            return
        if self._profiles_path is None:
            return
        if self._manifest_watcher_thread is not None and self._manifest_watcher_thread.is_alive():
            return
        self._seed_manifest_baseline()
        self._manifest_watcher_stop.clear()
        thread = threading.Thread(
            target=self._manifest_watcher_loop,
            name="llama-broker-manifest-watcher",
            daemon=True,
        )
        self._manifest_watcher_thread = thread
        thread.start()

    def _seed_manifest_baseline(self) -> None:
        """Seed from the bytes actually loaded, never from a later disk read."""
        with self._operation_admission_lock:
            self._last_manifest_identity = getattr(self.registry, "manifest_identity", None)

    def _manifest_watcher_loop(self) -> None:
        while not self._manifest_watcher_stop.wait(MANIFEST_RELOAD_INTERVAL):
            try:
                self._maybe_reload_manifest()
            except Exception:
                _log("[broker] manifest watcher: unexpected error (continuing)")

    def _maybe_reload_manifest(self) -> None:
        if self._profiles_path is None:
            return
        with self._operation_admission_lock:
            if self.coordinator.shutting_down:
                return
            # Alternate application paths may swap the registry without updating
            # watcher bookkeeping. Reconcile loaded bytes, never a fresh disk read,
            # even while an existing execution still suppresses new submissions.
            self._last_manifest_identity = getattr(self.registry, "manifest_identity", None)
            pending = self._pending_manifest_reload
            if pending is not None:
                try:
                    execution = self.operation_queue.get_execution(pending.execution_id)
                except KeyError:
                    # Retained history can expire while the watcher is stopped.
                    execution = None
                if execution is not None and execution.status in {"queued", "preflight", "committed"}:
                    return
                # Cancellation before worker dispatch has no completion callback.
                # It must not permanently suppress retry of the same disk bytes.
                self._pending_manifest_reload = None
            try:
                current = _manifest_identity(self._profiles_path)
            except Exception as error:
                _log(f"[broker] manifest watcher: cannot fingerprint manifest: {error}")
                return
            if current == self._last_manifest_identity:
                return
            try:
                request = self.submit_profile_reload_operation()
            except Exception as error:
                _log(f"[broker] manifest reload submission failed; will retry: {error}")
                return
            # A changed file while this request waits is retried after its stale
            # snapshot fails, rather than flooding the queue with every revision.
            self._pending_manifest_reload = request
            _log(f"[broker] manifest changed; queued profile/catalog reload ({request.request_id[:8]}…)")

    def _stop_manifest_watcher(self) -> None:
        self._manifest_watcher_stop.set()
        thread = self._manifest_watcher_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, MANIFEST_RELOAD_INTERVAL + 1.0))
        self._manifest_watcher_thread = None

    def restart_when_idle(self) -> Any:
        """Restart the active profile or recover the last failed target."""

        if self.coordinator.active_profile is None:
            state = self._controller_state_snapshot()
            if state.get("phase") == "failed":
                target_profile = state.get("target_profile")
                if isinstance(target_profile, str) and target_profile:
                    _log(
                        f"[broker] restart requested with no active profile; "
                        f"recovering failed target {target_profile!r}"
                    )
                    return self.coordinator.switch_when_idle(target_profile)
        return self.coordinator.restart_when_idle()

    def max_output_tokens_for(self, profile: Profile) -> int:
        """Return the maximum client-requested output for a profile."""

        maximum = profile.slot_context_window
        if self.max_output_tokens is not None:
            maximum = min(self.max_output_tokens, maximum)
        return maximum

    def default_max_output_tokens_for(self, profile: Profile) -> int:
        """Return the inverse-scaled output default for omitted client budgets."""

        if self.max_output_tokens is not None:
            return self.max_output_tokens_for(profile)
        return profile.default_max_output_tokens

    def _remember_controller_state(self, state: Mapping[str, Any]) -> None:
        with self._state_lock:
            self._controller_state = dict(state)

    def _controller_state_snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._controller_state)

    def _refresh_after_lifecycle(self, *, raise_on_error: bool) -> None:
        """Reconcile cached and lease state with the durable controller state."""

        try:
            state = dict(self.controller.state())
        except BaseException as error:
            unknown = {
                "generation": self._controller_state_snapshot().get("generation", 0),
                "profile": None,
                "phase": "unknown",
                "error": f"controller state refresh failed: {error}",
            }
            self._remember_controller_state(unknown)
            self.coordinator.synchronize_after_controller_failure(None)
            if raise_on_error:
                raise BrokerStateUnknownError(
                    "controller lifecycle completed but its state cannot be proven"
                ) from error
            return

        self._remember_controller_state(state)
        active_profile = state.get("profile") if state.get("phase") == "active" else None
        if not isinstance(active_profile, str):
            active_profile = None
        self.coordinator.synchronize_after_controller_failure(active_profile)

    @contextmanager
    def _controller_lifecycle(self) -> Iterator[Any]:
        """Own one cross-process lifecycle lock around an adapter transition."""
        lifecycle = getattr(self.controller, "lifecycle", None)
        if callable(lifecycle):
            lifecycle_context: Any = lifecycle()
            with lifecycle_context as controller:
                yield controller
            return
        yield self.controller

    def _switch_profile(self, profile_id: str) -> Any:
        try:
            with self._controller_lifecycle() as controller:
                result = controller.switch(profile_id)
        except Exception as error:
            self._refresh_after_lifecycle(raise_on_error=False)
            if isinstance(error, (BackendControllerError, BrokerCoordinationError)):
                raise
            raise BrokerLifecycleError(
                f"controller switch failed: {error}"
            ) from error
        self._refresh_after_lifecycle(raise_on_error=True)
        return result

    def _switch_profile_with_commit(
        self,
        profile_id: str,
        commit_callback: Callable[[], None],
    ) -> Any:
        """Delegate commit-marker timing to the backend adapter."""
        commit_crossed = False

        def mark_committed() -> None:
            nonlocal commit_crossed
            try:
                commit_callback()
            except OperationCancelledError as error:
                raise LifecycleCancelledError(str(error)) from error
            commit_crossed = True

        try:
            with self._controller_lifecycle() as controller:
                if isinstance(controller, BackendController):
                    result = controller.switch(
                        profile_id,
                        commit_callback=mark_committed,
                    )
                else:
                    mark_committed()
                    result = controller.switch(profile_id)
        except SwitchError as error:
            self._refresh_after_lifecycle(raise_on_error=False)
            if not commit_crossed:
                raise LifecycleCancelledError(
                    f"target preflight failed before commit: {error}"
                ) from error
            raise
        except Exception as error:
            self._refresh_after_lifecycle(raise_on_error=False)
            if isinstance(error, (BackendControllerError, BrokerCoordinationError)):
                raise
            raise BrokerLifecycleError(f"controller switch failed: {error}") from error
        self._refresh_after_lifecycle(raise_on_error=True)
        return result

    def _stop_backend(self) -> Any:
        # An explicit administrative stop cancels any watchdog retry target;
        # otherwise a failed recovery could resurrect a deliberately unloaded
        # backend on the next watchdog tick.
        self._recovery_profile = None
        self._recovery_attempts = 0
        self._recovery_next_at = 0.0
        try:
            with self._controller_lifecycle() as controller:
                result = controller.stop_all()
        except Exception as error:
            self._refresh_after_lifecycle(raise_on_error=False)
            if isinstance(error, (BackendControllerError, BrokerCoordinationError)):
                raise
            raise BrokerLifecycleError(
                f"controller stop failed: {error}"
            ) from error
        self._refresh_after_lifecycle(raise_on_error=True)
        return result

    def _restart_profile(self, profile_id: str) -> Any:
        try:
            with self._controller_lifecycle() as controller:
                result = controller.restart(profile_id)
        except Exception as error:
            self._refresh_after_lifecycle(raise_on_error=False)
            if isinstance(error, (BackendControllerError, BrokerCoordinationError)):
                raise
            raise BrokerLifecycleError(
                f"controller restart failed: {error}"
            ) from error
        self._refresh_after_lifecycle(raise_on_error=True)
        return result

    def _restart_profile_with_commit(
        self,
        profile_id: str,
        commit_callback: Callable[[], None],
    ) -> Any:
        commit_crossed = False

        def mark_committed() -> None:
            nonlocal commit_crossed
            try:
                commit_callback()
            except OperationCancelledError as error:
                raise LifecycleCancelledError(str(error)) from error
            commit_crossed = True

        try:
            with self._controller_lifecycle() as controller:
                if isinstance(controller, BackendController):
                    result = controller.restart(
                        profile_id,
                        commit_callback=mark_committed,
                    )
                else:
                    mark_committed()
                    result = controller.restart(profile_id)
        except SwitchError as error:
            self._refresh_after_lifecycle(raise_on_error=False)
            if not commit_crossed:
                raise LifecycleCancelledError(
                    f"restart preflight failed before commit: {error}"
                ) from error
            raise
        except Exception as error:
            self._refresh_after_lifecycle(raise_on_error=False)
            if isinstance(error, (BackendControllerError, BrokerCoordinationError)):
                raise
            raise BrokerLifecycleError(f"controller restart failed: {error}") from error
        self._refresh_after_lifecycle(raise_on_error=True)
        return result

    def _stop_backend_with_commit(
        self,
        commit_callback: Callable[[], None],
    ) -> Any:
        def mark_committed() -> None:
            try:
                commit_callback()
            except OperationCancelledError as error:
                raise LifecycleCancelledError(str(error)) from error

        self._recovery_profile = None
        self._recovery_attempts = 0
        self._recovery_next_at = 0.0
        try:
            with self._controller_lifecycle() as controller:
                if isinstance(controller, BackendController):
                    result = controller.stop_all(commit_callback=mark_committed)
                else:
                    mark_committed()
                    result = controller.stop_all()
        except Exception as error:
            self._refresh_after_lifecycle(raise_on_error=False)
            if isinstance(error, (BackendControllerError, BrokerCoordinationError)):
                raise
            raise BrokerLifecycleError(f"controller stop failed: {error}") from error
        self._refresh_after_lifecycle(raise_on_error=True)
        return result

    @property
    def backend_url(self) -> str:
        active_profile = getattr(getattr(self, "coordinator", None), "active_profile", None)
        if isinstance(active_profile, str):
            try:
                runtime = self.registry.get(active_profile).runtime
            except ProfileError:
                return self._backend_url
            return backend_url_for_runtime(self._backend_url, runtime)
        return self._backend_url

    def backend_client(self, *, timeout: float | None = None) -> BackendClient:
        return BackendClient(
            self.backend_url,
            timeout=self.backend_timeout if timeout is None else timeout,
            ca_file=self._backend_ca_file,
            default_headers=_backend_auth_headers(self.security),
        )

    def authentication_keys(self) -> tuple[str, ...]:
        """Return the in-memory deployment keys accepted by the broker API."""

        return self.security.authentication_keys()

    def _profile_artifacts_available(self, profile: Profile) -> bool:
        """Check file presence without hashing artifacts during normal operation."""
        paths = [profile.server_binary, profile.model_path]
        if profile.mmproj_path is not None:
            paths.append(profile.mmproj_path)
        if profile.runtime == "strata":
            for field in ("strata_artifacts_manifest_path", "strata_launch_config_path"):
                path = getattr(profile, field, None)
                if not isinstance(path, Path):
                    return False
                paths.append(path)
            if not profile.strata_artifact_files:
                return False
            paths.extend(path for path, _digest in profile.strata_artifact_files)
        try:
            return all(not path.is_symlink() and path.is_file() for path in paths)
        except OSError:
            return False

    def _openai_model_record(
        self,
        raw: Mapping[str, Any],
        *,
        profile: Profile | None = None,
    ) -> dict[str, Any]:
        """Normalize one model to the OpenAI model-object contract."""

        raw_id = raw.get("id")
        if profile is not None:
            model_id = profile.alias
        elif isinstance(raw_id, str) and raw_id.strip():
            model_id = raw_id.strip()
        else:
            raise ValueError("model metadata has no valid id")

        created = raw.get("created", 0)
        if isinstance(created, bool) or not isinstance(created, int):
            created = 0
        owned_by = raw.get("owned_by")
        if not isinstance(owned_by, str) or not owned_by.strip():
            owned_by = "Llama-Broker" if profile is not None else "llama-server"

        result: dict[str, Any] = {
            "id": model_id,
            "object": "model",
            "created": created,
            "owned_by": owned_by.strip(),
            "permission": [],
        }
        if profile is not None:
            result.update(
                {
                    # Clients budget one request against one slot, not all slots.
                    "context_length": profile.slot_context_window,
                    "aggregate_context_length": profile.context_size,
                    "parallel": profile.parallel,
                    "slot_context_window": profile.slot_context_window,
                    "max_output_tokens": self.max_output_tokens_for(profile),
                    "capabilities": sorted(profile.capabilities),
                }
            )
        return result

    def model_listing(self) -> dict[str, Any]:
        """List visible broker-managed models using OpenAI-compatible objects."""

        # The running backend's model catalog is private; publish only verified
        # broker profiles, including the active profile while it is being checked.
        active_profile_id = self.coordinator.active_profile
        visible_profiles = {
            profile.alias: profile
            for profile in self.registry.profiles
            if self._profile_artifacts_available(profile) or profile.id == active_profile_id
        }
        models = [
            self._openai_model_record({}, profile=profile)
            for profile in visible_profiles.values()
        ]
        if self.system1 is not None:
            # Routing aliases are virtual models: d1 picks the real profile.
            for alias in self.system1.routing_aliases():
                if alias in visible_profiles or self.registry.try_resolve(alias):
                    continue
                models.append(
                    {
                        "id": alias,
                        "object": "model",
                        "created": 0,
                        "owned_by": "llama-broker",
                        "metadata": {"system1_router": self.system1.config.model},
                    }
                )

        return {"object": "list", "data": models}

    def decision_model(self, name: Any) -> str | None:
        """Return the model name ``/v1/decisions`` answers as, or ``None``.

        A request may name the resident decision profile by id or any alias,
        or the system 1 model name itself.
        """

        if self.system1 is None or not isinstance(name, str) or not name.strip():
            return None
        config = self.system1.config
        if name.strip().casefold() == config.model.casefold():
            return config.model
        if config.profile is not None:
            resident = self.registry.try_resolve_resident(name)
            if resident is not None and resident.id == config.profile:
                return config.model
        return None

    def system1_route(self, payload: Mapping[str, Any]) -> Any:
        """Return d1's routing decision when *payload* names a routing alias."""

        if self.system1 is None:
            return None
        requested = payload.get("model")
        router = self.system1.router
        if not router.handles(requested) or self.registry.try_resolve(requested):
            return None
        active = self.coordinator.active_profile
        decision = self.system1.route(
            payload,
            self.registry.profiles,
            active_profile=active,
            available=lambda profile: (
                self._profile_artifacts_available(profile) or profile.id == active
            ),
        )
        _controller_event_logger(
            {
                "event": "system1_route",
                "requested": str(requested),
                "active": active,
                **decision.as_dict(),
            }
        )
        return decision

    def _api_key_path(self) -> Path:
        if self.security.api_key_file is None:
            raise SecurityError("API-key management is not configured")
        return self.security.api_key_file

    def rotate_credentials(self, *, replace: bool = False) -> dict[str, Any]:
        """Rotate native backend keys and restart the active server transactionally."""

        path = self._api_key_path()
        previous = read_api_keys(path)
        new_key, _ = rotate_api_keys(path, replace=replace)
        active_profile = self.coordinator.active_profile
        try:
            if active_profile is not None:
                self.coordinator.restart_when_idle(active_profile)
        except BaseException:
            if previous:
                write_api_keys(path, previous)
            else:
                path.unlink(missing_ok=True)
            raise
        return {
            "api_key": new_key,
            "active_profile": active_profile,
            "previous_keys_retained": not replace,
        }

    def revoke_credential(self, key: str) -> dict[str, Any]:
        """Revoke one native backend key and restart the active server."""

        if not key:
            raise SecurityError("API key must not be empty")
        path = self._api_key_path()
        previous = read_api_keys(path)
        remaining = revoke_api_key(path, key)
        active_profile = self.coordinator.active_profile
        try:
            if active_profile is not None:
                self.coordinator.restart_when_idle(active_profile)
        except BaseException:
            if previous:
                write_api_keys(path, previous)
            raise
        return {
            "active_profile": active_profile,
            "remaining_keys": len(remaining),
        }

    def ollama_tags(self) -> dict[str, Any]:
        """Fetch tags from llama-server backend or fall back to profile-based."""
        # llama-server doesn't expose /api/tags, so we use profile-based
        # This maintains backward compatibility
        return {
            "models": [
                {
                    "name": profile.alias,
                    "model": profile.alias,
                    "modified_at": "",
                    "size": profile.model_path.stat().st_size,
                    "digest": profile.model_sha256,
                    "details": {"family": "Llama-Broker"},
                }
                for profile in self.registry.profiles
            ]
        }

    def health(self) -> dict[str, Any]:
        # Controller transitions can hold their own lock for minutes.  Health
        # reads the last completed state instead of joining that lock convoy.
        state = self._controller_state_snapshot()
        leases = self.coordinator.snapshot()
        if leases["transitioning"]:
            # The completed state can be inactive or belong to the old model
            # while the coordinator is still loading the requested backend.
            state["phase"] = "switching"
            state["target_profile"] = leases["transition_target"]
        phase = state.get("phase")
        profile_id = state.get("profile")
        ready = False
        if phase == "active" and isinstance(profile_id, str):
            try:
                ready = bool(
                    self._backend_alias_ready(self.registry.get(profile_id).alias)
                )
            except Exception:
                ready = False
        if phase == "active" and not ready:
            if isinstance(profile_id, str):
                self.coordinator.invalidate_when_idle(profile_id)
            state["declared_phase"] = "active"
            state["phase"] = "stale"
            state["stale_profile"] = profile_id
        crash_loops = self.crash_loops.snapshot()
        health_status = (
            "degraded"
            if self._autotune_degraded_reason
            or any(entry["tripped"] for entry in crash_loops.values())
            else (
                "ok"
                if state.get("phase") == "inactive" or (phase == "active" and ready)
                else "degraded"
            )
        )
        profile = (
            self.registry.try_resolve(str(profile_id))
            if isinstance(profile_id, str)
            else None
        )
        backend_state = {
            "host": self.backend_host,
            "port": self.backend_port,
            **state,
            "ready": ready,
        }
        if profile is not None:
            backend_state.update(
                {
                    "runtime": getattr(profile, "runtime", None),
                    "runtime_build": getattr(profile, "runtime_build", None),
                    "deployment_revision": getattr(profile, "deployment_revision", None),
                    "warnings": list(getattr(profile, "warnings", ())),
                    "degraded_features": list(getattr(profile, "degraded_features", ())),
                }
            )
        catalog_path = getattr(self.registry, "runtime_catalog_path", None)
        deployment_revision = getattr(self.registry, "deployment_revision", None)
        # One immutable queue snapshot: executions, not coalesced callers.
        executions = self.operation_queue.list_executions()
        running = next((item for item in executions if item.status in {"preflight", "committed"}), None)
        operations = {
            "queued": sum(item.status == "queued" for item in executions),
            "running": (
                {"type": running.operation_type, "target": running.target_profile_id, "status": running.status}
                if running is not None else None
            ),
        }
        return {
            "name": "Llama-Broker",
            "status": health_status,
            "catalog": {
                "path": str(catalog_path) if catalog_path else None,
                "deployment_revision": deployment_revision,
            },
            "backend": backend_state,
            "leases": leases,
            "activity": {"operations": operations},
            "crash_loops": crash_loops,
            "maintenance": {
                "autotune": self._autotune_lease.status(),
                "degraded_reason": self._autotune_degraded_reason,
            },
            "system1": (
                self.system1.status() if self.system1 is not None else {"enabled": False}
            ),
        }

    def reload_manifest_for_autotune(self) -> dict[str, Any]:
        """Reload the validated manifest once without restarting the listener."""

        if self.coordinator.quiescing or self.coordinator.active_requests > 0:
            raise BrokerCoordinationError(
                "autotune manifest reload requires a fully idle coordinator"
            )
        new_registry = ProfileRegistry.from_path(PROFILES_PATH)
        active = self.coordinator.active_profile
        if active is not None and new_registry.try_resolve(active) is None:
            raise ProfileError(
                f"autotune reload would remove the active profile {active!r}"
            )
        self.registry = new_registry
        self._autotune_degraded_reason = None
        replace_profiles = getattr(self.controller, "replace_profiles", None)
        if callable(replace_profiles):
            replace_profiles({profile.id: profile for profile in new_registry.profiles})
        install_units = getattr(self.controller, "install_units", None)
        if callable(install_units):
            install_units()
        self._remember_controller_state(dict(self.controller.state()))
        return self.health()

    def reload_profiles_for_operation(
        self,
        commit_callback: Callable[[], None],
    ) -> dict[str, Any]:
        """Apply the latest profile/catalog revision inside the ordered queue."""
        if self._profiles_path is None:
            raise LifecycleCancelledError("profile/catalog reload source path is unavailable")
        try:
            new_registry = ProfileRegistry.from_path(self._profiles_path)
            active = self.coordinator.active_profile
            if active is not None and new_registry.try_resolve(active) is None:
                raise ProfileError(
                    f"profile/catalog reload would remove the active profile {active!r}"
                )
        except Exception as error:
            raise LifecycleCancelledError(
                f"profile/catalog reload preflight failed: {error}"
            ) from error

        try:
            commit_callback()
        except OperationCancelledError as error:
            raise LifecycleCancelledError(str(error)) from error

        previous_registry = self.registry
        previous_profiles = {
            profile.id: profile for profile in previous_registry.profiles
        }
        next_profiles = {profile.id: profile for profile in new_registry.profiles}
        try:
            with self._controller_lifecycle() as controller:
                replace_profiles = getattr(controller, "replace_profiles", None)
                install_units = getattr(controller, "install_units", None)
                if callable(replace_profiles):
                    replace_profiles(next_profiles)
                if callable(install_units):
                    install_units()
        except Exception as error:
            try:
                with self._controller_lifecycle() as controller:
                    replace_profiles = getattr(controller, "replace_profiles", None)
                    install_units = getattr(controller, "install_units", None)
                    if callable(replace_profiles):
                        replace_profiles(previous_profiles)
                    if callable(install_units):
                        install_units()
            except Exception as rollback_error:
                self._autotune_degraded_reason = (
                    f"profile/catalog reload compensation failed: {rollback_error}"
                )
                raise BrokerLifecycleError(
                    f"profile/catalog reload failed and compensation failed: {rollback_error}"
                ) from error
            raise LifecycleCancelledError(
                f"profile/catalog reload failed; prior configuration restored: {error}"
            ) from error

        # Publish the applied bytes even if the queue worker already timed out.
        # Controller lifecycle locks have exited, and the coordinator invokes this
        # callback without its condition held: admission locks before that condition.
        with self._operation_admission_lock:
            self.registry = new_registry
            self._last_manifest_identity = getattr(new_registry, "manifest_identity", None)
        self._autotune_degraded_reason = None
        self._remember_controller_state(dict(self.controller.state()))
        return self.health()

    def diagnose_backend_failure(self, profile_id: str) -> BackendExit | None:
        """Ask the controller whether a failed request's backend actually exited."""

        describe_exit = getattr(self.controller, "describe_exit", None)
        if describe_exit is None:
            return None
        try:
            return describe_exit(profile_id)
        except Exception as error:
            _log(f"[broker] backend exit diagnosis failed for {profile_id!r}: {error}")
            return None

    def record_backend_crash(self, profile_id: str, cause: str) -> None:
        if self.crash_loops.record_crash(profile_id, cause):
            _log(
                f"[broker] backend {profile_id!r} is crash-looping; refusing its "
                f"requests for {self.crash_loops.cooldown:g}s instead of restarting it"
            )

    def mark_autotune_degraded(self, reason: str) -> None:
        """Fail closed after autotune compensation cannot restore the registry."""

        self._autotune_degraded_reason = str(reason)
        active = self.coordinator.active_profile
        if active is not None:
            self.coordinator.invalidate_when_idle(active)


    def shutdown(self) -> None:
        """Quiesce the control plane while preserving broker-owned backends."""

        with self._shutdown_lock:
            if self._shutdown_complete:
                if self._shutdown_error is not None:
                    raise self._shutdown_error
                return
            try:
                self.coordinator.request_shutdown()
                self._stop_operation_worker()
                self._stop_recovery_watchdog()
                self._stop_manifest_watcher()
                self.coordinator.handoff_for_process_shutdown()
            except BaseException as error:
                self._shutdown_error = error
                raise
            finally:
                self._shutdown_complete = True


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class _ClientDisconnectedError(Exception):
    """The downstream client went away while the broker was writing to it."""


class _BackendResponseError(http.client.HTTPException):
    """An upstream status or stream contract cannot follow early SSE headers."""


class _HeaderDeadlineReader:
    def __init__(self, reader: Any, connection: socket.socket, timeout: float):
        self.reader = reader
        self.connection = connection
        self.deadline = time.monotonic() + timeout
        self.pending = bytearray()

    def readline(self, limit: int = -1) -> bytes:
        limit = 65537 if limit < 0 else limit
        while True:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("request headers timed out")
            newline = self.pending.find(b"\n", 0, limit)
            if newline >= 0 or len(self.pending) >= limit:
                end = newline + 1 if newline >= 0 else limit
                line = bytes(self.pending[:end])
                del self.pending[:end]
                return line
            self.connection.settimeout(remaining)
            # read1 performs at most one underlying read. Buffered readline
            # can keep accepting trickled bytes indefinitely on one timeout.
            chunk = self.reader.read1(min(8192, limit - len(self.pending)))
            if not chunk:
                line = bytes(self.pending)
                self.pending.clear()
                return line
            self.pending.extend(chunk)

    def read1(self, limit: int = -1) -> bytes:
        if self.pending:
            end = len(self.pending) if limit < 0 else min(limit, len(self.pending))
            data = bytes(self.pending[:end])
            del self.pending[:end]
            return data
        return self.reader.read1(limit)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.reader, name)


class BrokerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Llama-Broker/2.0"

    def setup(self) -> None:
        super().setup()
        self._request_body_consumed = False
        self._proxy_response_started = False
        self._sse_started = False
        self._inference_ticket = None
        self._control_slot = False
        self._get_data_slot = False
        self.rfile = _HeaderDeadlineReader(
            self.rfile, self.connection, self.server._control_client_timeout
        )
        self.server._request_context.handler = self

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        finally:
            # Each bounded connection serves exactly one request.
            self.close_connection = True

    def finish(self) -> None:
        # rfile is closed by super().finish(); the response is already on the
        # wire because wfile is unbuffered.
        try:
            self._discard_unread_body()
        finally:
            super().finish()

    def _release_admission(self) -> None:
        # The server calls this after transport close, so finishing inference
        # cannot consume the connections reserved for administrative workers.
        if self._inference_ticket is not None:
            self.server.admission.finish_writer(self._inference_ticket)
        if self._control_slot:
            self.server._control_worker_slots.release()
        if self._get_data_slot:
            self.server._data_worker_slots.release()

    def _discard_unread_body(self) -> None:
        """Drain a request body that was rejected before it was read."""

        if self._request_body_consumed:
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (AttributeError, ValueError):
            return
        remaining = min(length, self.application.max_body_bytes)
        if remaining <= 0:
            return
        deadline = time.monotonic() + REJECTED_BODY_DRAIN_TIMEOUT
        try:
            while remaining > 0:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    return
                self.connection.settimeout(budget)
                chunk = self.rfile.read1(min(65536, remaining))
                if not chunk:
                    return
                remaining -= len(chunk)
        except (OSError, ValueError):
            return

    @property
    def application(self) -> BrokerApplication:
        return self.server.application  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        _log("[broker] " + format % args)

    def _client_is_loopback(self) -> bool:
        try:
            return ipaddress.ip_address(self.client_address[0]).is_loopback
        except ValueError:
            return False

    def _require_loopback(self) -> bool:
        if self._client_is_loopback():
            return True
        self._send_json(
            HTTPStatus.FORBIDDEN,
            {
                "error": {
                    "message": "this broker administration endpoint is loopback-only",
                    "type": "broker_admin_forbidden",
                }
            },
        )
        return False

    def _require_api_auth(self) -> bool:
        """Require the OpenAI-standard Bearer credential for every ``/v1`` route."""

        authorization = self.headers.get("Authorization", "")
        scheme, separator, credential = authorization.partition(" ")
        candidate = credential.strip() if separator and scheme.casefold() == "bearer" else ""
        configured = self.application.authentication_keys()
        if configured and candidate and any(
            secrets.compare_digest(candidate, expected) for expected in configured
        ):
            return True
        self._send_json(
            HTTPStatus.UNAUTHORIZED,
            {
                "error": {
                    "message": "invalid or missing API key",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            },
            extra_headers={"WWW-Authenticate": "Bearer"},
        )
        return False

    def _cors_target(self) -> bool:
        """The OpenAI API surface is the only cross-origin-authorized route."""
        path = urlsplit(self.path).path
        return path.startswith("/v1/") or path == "/v1"

    def do_OPTIONS(self) -> None:  # noqa: N802
        """CORS preflight for browser clients (Ouroboros page on a peer device).

        The real request is a cross-origin POST carrying ``Content-Type:
        application/json`` + ``Authorization: Bearer <key>``, which forces a
        preflight. Without this the browser blocks the call before it reaches
        the model. We answer 204 and the ``end_headers`` override below adds the
        mirrored ``Access-Control-Allow-Origin`` + allow-list.
        """
        if not self._cors_target():
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self.end_headers()
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        self.end_headers()
        return

    def end_headers(self) -> None:
        """Inject CORS for the ``/v1`` OpenAI surface.

        The Ouroboros page runs on a *peer* device's origin and talks to the
        key-gated broker cross-origin. Its chat call is a ``POST`` with
        ``Content-Type: application/json`` + ``Authorization: Bearer <key>``,
        which forces a preflight; without these headers the browser blocks the
        request before it reaches the model. Access is gated by the Bearer key
        (not the origin), so we mirror the request Origin and never use a
        wildcard. Only ``/v1`` (the OpenAI surface) is cross-origin-authorized.
        """
        # HTTP parser errors can precede initialization of path or headers.
        if hasattr(self, "path") and hasattr(self, "headers") and self._cors_target():
            # The response varies by Origin (it carries ACAO only when the request
            # has one), so mark it on *every* /v1 response — including no-Origin
            # ones — so a shared cache can't serve a no-Origin response to a browser.
            self.send_header("Vary", "Origin")
            origin = self.headers.get("Origin")
            if origin is not None:
                self.send_header("Access-Control-Allow-Origin", origin)
                if self.command == "OPTIONS":
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                    self.send_header("Access-Control-Max-Age", "600")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if not self._admit_control_reserve(path):
            return
        if path == "/admin/status" and not self._require_loopback():
            return
        if path.startswith("/v1/") and not self._require_api_auth():
            return
        if path == "/admin/operations":
            if not self._require_loopback():
                return
            self._send_json(HTTPStatus.OK, self.application.operation_list())
            return
        if path.startswith("/admin/operations/"):
            if not self._require_loopback():
                return
            request_id = unquote(path.rsplit("/", 1)[-1])
            try:
                operation = self.application.operation_status(request_id)
            except KeyError:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": {"message": "operation not found or expired", "type": "not_found"}},
                )
                return
            self._send_json(HTTPStatus.OK, {"operation": operation})
            return
        if path in {"/health", "/healthz", "/admin/status"}:
            self._send_json(HTTPStatus.OK, self.application.health())
            return
        if path == "/v1/models":
            self._send_json(HTTPStatus.OK, self.application.model_listing())
            return
        if path.startswith("/v1/models/"):
            alias = unquote(path.rsplit("/", 1)[-1])
            profile = self.application.registry.try_resolve(alias)
            if profile is None:
                self._model_not_found(alias)
                return
            if (
                profile.id != self.application.coordinator.active_profile
                and not self.application._profile_artifacts_available(profile)
            ):
                self._model_not_found(alias, message=f"model unavailable: {alias}")
                return
            # Forward backend data for this model
            client = self.application.backend_client()
            try:
                backend_info = client.model_info(
                    alias,
                    headers=self._request_headers(b""),
                )
                response = self.application._openai_model_record(
                    backend_info,
                    profile=profile,
                )
            except Exception as error:
                if self.application.security.api_key_file is not None:
                    self._send_json(
                        HTTPStatus.BAD_GATEWAY,
                        {
                            "error": {
                                "message": "backend model metadata unavailable",
                                "type": "broker_proxy_error",
                            }
                        },
                    )
                    return
                # Preserve the profile-only compatibility fallback when native
                # backend authentication is not enabled.
                _log(f"[broker] backend model info failed for {alias}: {error}")
                response = self.application._openai_model_record({}, profile=profile)
            finally:
                client.close()
            self._send_json(HTTPStatus.OK, response)
            return
        if path == "/api/tags":
            if not self._require_loopback():
                return
            self._send_json(HTTPStatus.OK, self.application.ollama_tags())
            return
        if path in {"/slots", "/props"}:
            if not self._require_loopback():
                return
            try:
                with self.application.coordinator.lease_active(probe=True):
                    self._proxy_backend_get(path)
            except BrokerCoordinationError as error:
                self._send_coordination_error(error)
            return
        self._send_json(
            HTTPStatus.NOT_FOUND,
            {"error": {"message": f"unknown broker endpoint: {path}", "type": "not_found"}},
        )

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._do_POST()
        except AdmissionError as error:
            self._send_admission_error(error)
        except TimeoutError:
            self._diagnostic("request_timeout", phase="request_io")
            try:
                self._send_json(HTTPStatus.REQUEST_TIMEOUT, {"error": {
                    "message": "client request I/O timed out", "type": "broker_request_timeout",
                }})
            except (_ClientDisconnectedError, OSError):
                self._diagnostic("client_disconnect", phase="request_timeout")
        except (_ClientDisconnectedError, OSError) as error:
            self._diagnostic("server_shutdown" if self.server._closing.is_set() else "client_disconnect",
                             phase="request", detail=type(error).__name__)

    def _do_POST(self) -> None:
        path = urlsplit(self.path).path
        if not self._admit_control_reserve(path):
            return
        if path.startswith("/admin/operations/") and path.endswith("/cancel"):
            if not self._require_loopback():
                return
            segments = path.strip("/").split("/")
            if len(segments) != 4 or segments[:2] != ["admin", "operations"]:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": {"message": "unknown operation endpoint", "type": "not_found"}},
                )
                return
            request_id = unquote(segments[2])
            try:
                self._read_payload()
                result = self.application.cancel_operation(request_id)
            except KeyError:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": {"message": "operation not found or expired", "type": "not_found"}},
                )
                return
            except ValueError as error:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": str(error), "type": "invalid_request"}},
                )
                return
            self._send_json(HTTPStatus.OK, result)
            return
        if path in {
            "/admin/switch",
            "/admin/restart",
            "/admin/stop",
            "/admin/autotune/reload",
            "/admin/autotune/degraded",
            "/admin/credentials/rotate",
            "/admin/credentials/revoke",
        }:
            if not self._require_loopback():
                return
            self._handle_admin(path)
            return
        if not path.startswith("/v1/"):
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": {"message": f"unknown broker endpoint: {path}", "type": "not_found"}},
            )
            return
        if not self._require_api_auth():
            return
        if path == "/v1/systemone":
            self._proxy_system1()
            return
        if path == "/v1/decisions":
            self._handle_decisions()
            return
        proxy_payload: Mapping[str, Any] | None = None
        payload: Mapping[str, Any] | None = None
        self._system1_decision = None
        try:
            if path == "/v1/audio/transcriptions":
                body = self._read_multipart_body()
                profile, _ = route_payload(
                    self.application.registry,
                    {},
                    active_profile=self.application.coordinator.active_profile,
                )
            else:
                payload = self._read_payload()
                decision = self.application.system1_route(payload)
                if decision is not None:
                    if decision.profile_id is None:
                        self._send_json(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            {
                                "error": {
                                    "message": (
                                        "system1 could not route this request: "
                                        f"{decision.reason}"
                                    ),
                                    "type": "system1_route_unavailable",
                                }
                            },
                        )
                        return
                    target = self.application.registry.try_resolve(decision.profile_id)
                    if target is None:
                        raise UnknownModelError(
                            f"system1 routed to unknown profile: {decision.profile_id}"
                        )
                    self._system1_decision = decision
                    payload = {**payload, "model": target.alias}
                profile, rewritten = route_payload(
                    self.application.registry,
                    payload,
                    active_profile=self.application.coordinator.active_profile,
                )
                try:
                    if path in {"/v1/chat/completions", "/v1/completions"}:
                        rewritten = ensure_single_system_message(rewritten)
                        rewritten = enforce_output_limit(
                            rewritten,
                            self.application.max_output_tokens_for(profile),
                            default=self.application.default_max_output_tokens_for(profile),
                        )
                    rewritten = apply_reasoning_policy(
                        rewritten,
                        runtime=profile.runtime,
                        max_output_tokens=self.application.max_output_tokens_for(profile),
                        request_max_output_tokens=_requested_output_tokens(rewritten),
                        batch_size=profile.batch_size or 1,
                        ubatch_size=profile.ubatch_size or 1,
                    )
                except ValueError:
                    rejected_metadata = reasoning_policy_metadata(
                        payload, payload, runtime=profile.runtime
                    )
                    if rejected_metadata is not None:
                        _controller_event_logger(
                            {
                                "decision": "rejected",
                                "event": "reasoning_policy",
                                "request_id": self._request_id(),
                                "profile": profile.id,
                                **rejected_metadata,
                            }
                        )
                    raise
                reasoning_metadata = reasoning_policy_metadata(
                    payload, rewritten, runtime=profile.runtime
                )
                if reasoning_metadata is not None:
                    _controller_event_logger(
                        {
                            "event": "reasoning_policy",
                            "request_id": self._request_id(),
                            "profile": profile.id,
                            **reasoning_metadata,
                        }
                    )
                if (
                    profile.id != self.application.coordinator.active_profile
                    and not self.application._profile_artifacts_available(profile)
                ):
                    self._model_not_found(
                        profile.alias,
                        message=f"model unavailable: {profile.alias}",
                    )
                    return
                proxy_payload = rewritten
                body = _json_bytes(rewritten)
            if path == "/v1/audio/transcriptions" and (
                profile.id != self.application.coordinator.active_profile
                and not self.application._profile_artifacts_available(profile)
            ):
                self._model_not_found(
                    profile.alias,
                    message=f"model unavailable: {profile.alias}",
                )
                return
        except UnknownModelError as error:
            requested = payload.get("model", "") if payload is not None else ""
            if self.application.registry.try_resolve_resident(requested) is not None:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": {
                            "message": (
                                f"{requested} is a resident decision model; "
                                "send it questions on POST /v1/decisions or /v1/systemone"
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                )
                return
            self._model_not_found(str(requested), message=str(error))
            return
        except BrokerCoordinationError as error:
            self._send_coordination_error(error)
            return
        except ValueError as error:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": str(error), "type": "invalid_request_error"}},
            )
            return

        crash_loop = self.application.crash_loops.check(profile.id)
        if crash_loop is not None:
            _log(
                f"[broker] rejected {self.command} {path} "
                f"(request {self._request_id()}, HTTP 503): {crash_loop.message()}"
            )
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": {"message": crash_loop.message(), "type": "backend_crash_loop"}},
                extra_headers={"Retry-After": str(crash_loop.retry_after)},
            )
            return

        self._proxy_response_started = False
        self._proxy_completed = False
        try:
            self._proxy(body, proxy_payload, profile)
            if self._proxy_completed:
                self.application.crash_loops.record_success(profile.id)
        except BrokerCoordinationError as error:
            self._send_coordination_error(error)
            return
        except BackendControllerError as error:
            self._diagnostic("backend_failure", profile=profile.id, phase="activation",
                             detail=type(error).__name__)
            _log(
                f"[broker] profile switch failed for {profile.id} "
                f"(request {self._request_id()}): {type(error).__name__}"
            )
            if not self.wfile.closed:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "error": {
                            "message": f"backend profile unavailable: {profile.id}",
                            "type": "broker_switch_error",
                        }
                    },
                )
            return
        except (OSError, TimeoutError, http.client.HTTPException) as error:
            timed_out = isinstance(error, TimeoutError)
            self._diagnostic("backend_timeout" if timed_out else "backend_failure", profile=profile.id,
                             detail=type(error).__name__)
            _log(
                f"[broker] backend proxy failure for {profile.id} "
                f"(request {self._request_id()}): {type(error).__name__}"
            )
            if not isinstance(error, _BackendResponseError):
                self.application.coordinator.invalidate_when_idle(profile.id)
            backend_exit = self.application.diagnose_backend_failure(profile.id)
            message = "backend request failed"
            if backend_exit is not None and backend_exit.crashed:
                cause = backend_exit.summary()
                message = f"backend exited during the request: {cause}"
                _log(
                    f"[broker] backend {profile.id} exited during request "
                    f"{self._request_id()}: {cause}"
                )
                self.application.record_backend_crash(profile.id, cause)
            if self._proxy_response_started and not self._sse_started:
                # The backend status line is already on the wire; appending a
                # JSON error would corrupt the response. Closing ends it.
                return
            try:
                self._send_json(
                    HTTPStatus.GATEWAY_TIMEOUT if timed_out else HTTPStatus.BAD_GATEWAY,
                    {
                        "error": {
                            "message": message,
                            "type": "broker_backend_timeout" if timed_out else "broker_proxy_error",
                        }
                    },
                )
            except OSError:
                pass
            return

    def _respond_admin_operation(self, request: RequestView, *, wait: bool) -> None:
        if not wait:
            state = self.application.health()
            state["operation"] = self.application.operation_status(request.request_id)
            self._send_json(HTTPStatus.ACCEPTED, state)
            return
        try:
            operation = self.application.wait_for_operation(
                request.request_id,
                self.application.admin_operation_timeout,
            )
        except TimeoutError:
            self._send_json(
                HTTPStatus.GATEWAY_TIMEOUT,
                {
                    "error": {
                        "message": "profile transition is still running",
                        "type": "broker_transition_timeout",
                    },
                    "operation": self.application.operation_status(request.request_id),
                },
            )
            return
        if operation["status"] == "failed":
            error_type = operation.get("error_type")
            if error_type == "BrokerDrainTimeoutError":
                response_status = HTTPStatus.GATEWAY_TIMEOUT
                response_type = "broker_drain_timeout"
            elif error_type == "BrokerTransitionTimeoutError":
                response_status = HTTPStatus.GATEWAY_TIMEOUT
                response_type = "broker_transition_timeout"
            elif error_type == "StaleOperationSnapshotError":
                response_status = HTTPStatus.CONFLICT
                response_type = "operation_snapshot_stale"
            elif error_type == "BrokerStateUnknownError":
                response_status = HTTPStatus.SERVICE_UNAVAILABLE
                response_type = "broker_state_unknown"
            else:
                response_status = HTTPStatus.SERVICE_UNAVAILABLE
                response_type = "broker_switch_error"
            self._send_json(
                response_status,
                {
                    "error": {
                        "message": operation.get("error") or "profile transition failed",
                        "type": response_type,
                    },
                    "operation": operation,
                },
            )
            return
        if operation["status"] == "cancelled":
            self._send_json(
                HTTPStatus.CONFLICT,
                {
                    "error": {"message": "transition request was cancelled", "type": "operation_cancelled"},
                    "operation": operation,
                },
            )
            return
        state = self.application.health()
        state["operation"] = operation
        self._send_json(HTTPStatus.OK, state)

    def _handle_admin(self, path: str) -> None:
        try:
            payload = self._read_payload()
            if path == "/admin/switch":
                model = payload.get("model")
                if not isinstance(model, str) or not model:
                    raise ValueError("model must be a non-empty profile ID or alias")
                wait = payload.get("wait", True)
                if not isinstance(wait, bool):
                    raise ValueError("wait must be boolean")
                request = self.application.submit_switch_operation(model)
                self._respond_admin_operation(request, wait=wait)
                return
            elif path == "/admin/restart":
                wait = payload.get("wait", True)
                if not isinstance(wait, bool):
                    raise ValueError("wait must be boolean")
                model = payload.get("model")
                if model is not None:
                    if not isinstance(model, str) or not model:
                        raise ValueError("model must be a non-empty profile ID or alias")
                    request = self.application.submit_restart_operation(model)
                else:
                    active_profile = self.application.coordinator.active_profile
                    if active_profile is None:
                        failed_target = self.application._controller_state_snapshot().get(
                            "target_profile"
                        )
                        if not isinstance(failed_target, str):
                            raise BrokerCoordinationError(
                                "there is no active or recoverable profile to restart"
                            )
                        request = self.application.submit_switch_operation(failed_target)
                    else:
                        request = self.application.submit_restart_operation(active_profile)
                self._respond_admin_operation(request, wait=wait)
                return
            elif path == "/admin/stop":
                wait = payload.get("wait", True)
                if not isinstance(wait, bool):
                    raise ValueError("wait must be boolean")
                request = self.application.submit_stop_operation()
                self._respond_admin_operation(request, wait=wait)
                return
            elif path == "/admin/autotune/reload":
                wait = payload.get("wait", True)
                if not isinstance(wait, bool):
                    raise ValueError("wait must be boolean")
                request = self.application.submit_profile_reload_operation()
                self._respond_admin_operation(request, wait=wait)
                return
            elif path == "/admin/autotune/degraded":
                reason = payload.get("reason")
                if not isinstance(reason, str) or not reason:
                    raise ValueError("reason must be a non-empty string")
                self.application.mark_autotune_degraded(reason)
                self._send_json(HTTPStatus.OK, self.application.health())
                return
            elif path == "/admin/credentials/rotate":
                replace = payload.get("replace", False)
                if not isinstance(replace, bool):
                    raise ValueError("replace must be boolean")
                self._send_json(
                    HTTPStatus.OK,
                    self.application.rotate_credentials(replace=replace),
                )
                return
            else:
                key = payload.get("key")
                if not isinstance(key, str):
                    raise ValueError("key must be a string")
                self._send_json(
                    HTTPStatus.OK,
                    self.application.revoke_credential(key),
                )
                return
            self._send_json(HTTPStatus.OK, self.application.health())
        except (ValueError, ProfileError) as error:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": str(error), "type": "invalid_request_error"}},
            )
        except QueueCapacityError as error:
            self._send_json(
                HTTPStatus.TOO_MANY_REQUESTS,
                {"error": {"message": str(error), "type": "operation_queue_full"}},
            )
        except OperationSupersededError as error:
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": {"message": str(error), "type": "operation_superseded"}},
            )
        except ReconciliationRequiredError as error:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": {"message": str(error), "type": "broker_reconciliation_required"}},
            )
        except BackendControllerError as error:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": {"message": str(error), "type": "broker_switch_error"}},
            )
        except BrokerCoordinationError as error:
            self._send_coordination_error(error)

    def _admit_control_reserve(self, path: str) -> bool:
        control_path = path in {
            "/health",
            "/healthz",
            "/admin/status",
            "/admin/operations",
            "/admin/switch",
            "/admin/restart",
            "/admin/stop",
            "/admin/autotune/reload",
            "/admin/autotune/degraded",
            "/admin/credentials/rotate",
            "/admin/credentials/revoke",
            "/v1/models",
            "/api/tags",
        } or path.startswith(("/admin/operations/", "/v1/models/"))
        if control_path:
            self._control_slot = self.server._control_worker_slots.acquire(False)
            if self._control_slot:
                self.server.release_header_slot()
                return True
        elif self.command == "POST":
            # Inference authentication, length validation and admission occur
            # before any body allocation. It never borrows a control worker.
            return True
        else:
            self._get_data_slot = self.server._data_worker_slots.acquire(False)
            if self._get_data_slot:
                self.server.release_header_slot()
                return True
        if not self._control_slot and not self._get_data_slot:
            self._diagnostic("control_full" if control_path else "worker_full")
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": {
                        "message": "broker worker capacity is exhausted",
                        "type": "broker_overloaded",
                    }
                },
            )
            return False
        return True

    def _send_coordination_error(self, error: BrokerCoordinationError) -> None:
        self._diagnostic("request_stale" if isinstance(error, BrokerRequestStaleError) else "coordination_failure",
                         detail=error.error_type)
        _log(
            f"[broker] rejected {self.command} {urlsplit(self.path).path} "
            f"(request {self._request_id()}, HTTP {error.http_status}): {error}"
        )
        self._send_json(
            error.http_status,
            {"error": {"message": str(error), "type": error.error_type}},
        )

    def _read_raw_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding") is not None:
            raise ValueError("Transfer-Encoding is not supported")
        if len(self.headers.get_all("Content-Length", [])) > 1:
            raise ValueError("multiple Content-Length headers are not supported")
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(length_header)
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length < 0:
            raise ValueError("invalid Content-Length")
        if length > self.application.max_body_bytes:
            raise ValueError(
                f"request body exceeds broker limit {self.application.max_body_bytes} bytes"
            )
        if urlsplit(self.path).path.startswith("/v1/"):
            self._inference_ticket = self.server.admission.reserve(
                length, quiescing=self.application.coordinator.quiescing
            )
            if not self._inference_ticket.worker:
                self._diagnostic("queue_wait")
            self.server.release_header_slot()
        self._request_body_consumed = True
        deadline = time.monotonic() + self.server._control_client_timeout
        if self._inference_ticket is not None:
            deadline = min(deadline, self._inference_ticket.deadline)
        chunks = bytearray()
        while len(chunks) < length:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if self._inference_ticket is not None and time.monotonic() >= self._inference_ticket.deadline:
                    raise AdmissionError("queue_expired")
                raise TimeoutError("request body timed out")
            self.connection.settimeout(remaining)
            try:
                chunk = self.rfile.read1(min(65536, length - len(chunks)))
            except TimeoutError as error:
                if self._inference_ticket is not None and time.monotonic() >= self._inference_ticket.deadline:
                    raise AdmissionError("queue_expired") from error
                raise
            if not chunk:
                self._diagnostic("server_shutdown" if self.server._closing.is_set() else "client_disconnect",
                                 phase="request_body")
                break
            chunks.extend(chunk)
        body = bytes(chunks)
        if len(body) != length:
            raise ValueError("request body is incomplete")
        return body

    def _read_payload(self) -> dict[str, Any]:
        body = self._read_raw_body()
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"request body is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("request JSON must be an object")
        return payload

    def _read_multipart_body(self) -> bytes:
        content_type = self.headers.get("Content-Type", "")
        media_type, separator, parameters = content_type.partition(";")
        if media_type.strip().casefold() != "multipart/form-data" or not separator:
            raise ValueError("Content-Type must be multipart/form-data")
        if not re.search(
            r"(?:^|;)\s*boundary\s*=\s*(?:\"[^\"]+\"|[^;\s]+)",
            parameters,
            re.IGNORECASE,
        ):
            raise ValueError("multipart/form-data boundary is required")
        return self._read_raw_body()

    def _request_headers(self, body: bytes) -> dict[str, str]:
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_REQUEST_HEADERS | {"x-request-id"}
        }
        headers["Content-Length"] = str(len(body))
        headers["Connection"] = "close"
        headers["X-Request-ID"] = self._request_id()
        return headers

    def _proxy(
        self,
        body: bytes,
        payload: Mapping[str, Any] | None,
        profile: Profile,
    ) -> None:
        ticket = self._inference_ticket
        assert ticket is not None
        streaming = (
            urlsplit(self.path).path in {"/v1/chat/completions", "/v1/completions"}
            and payload is not None and payload.get("stream") is True
        )
        pipe = BackendPipe(ticket)
        self._proxy_pipe = pipe
        self._proxy_completed = False
        self.connection.settimeout(self.server._control_client_timeout)

        def read_backend() -> None:
            connection = None
            response = None
            lease = None
            try:
                while True:
                    self.server.admission.wait_turn(
                        ticket, lambda: self.application.coordinator.inference_ready(profile.id)
                    )
                    self._require_current_profile(profile)
                    lease = self.application.coordinator.lease(
                        profile.id, transient=profile.is_transient, admission_only=True
                    )
                    try:
                        lease.__enter__()
                        break
                    except BrokerQuiescingError:
                        # An admin operation may win between readiness and lease
                        # acquisition. Nothing has been dispatched; retain FIFO.
                        lease = None
                try:
                    self._require_current_profile(profile)
                    self.server.admission.started(ticket)
                    if ticket.cancelled.is_set():
                        return
                    pipe.deadline = time.monotonic() + self.application.backend_timeout
                    connection = make_connection(
                        self.application.backend_url, timeout=self.application.backend_timeout,
                        ca_file=self.application.security.backend_tls_ca_file,
                    )
                    connection.connect()
                    pipe.attach(connection.sock)
                    if ticket.cancelled.is_set():
                        return
                    if time.monotonic() >= pipe.deadline:
                        raise TimeoutError("backend connect deadline expired")
                    connection.request(self.command, self.path, body=body,
                                       headers=self._request_headers(body))
                    # getresponse() can detach connection.sock for Connection:
                    # close. Retain the transport solely to interrupt a read.
                    response = connection.getresponse()
                    if streaming and not (
                        200 <= response.status < 300
                        and response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
                        == "text/event-stream"
                    ):
                        raise _BackendResponseError(f"backend returned incompatible SSE response (HTTP {response.status})")
                    pipe.put("headers", (
                        response.status, response.reason, response.getheaders(),
                        getattr(response, "length", None) == 0,
                    ))
                    pending = bytearray()
                    completed = False
                    while not ticket.cancelled.is_set():
                        chunk = response.read1(8192)
                        if not chunk:
                            if getattr(response, "length", 0) not in (None, 0):
                                raise http.client.IncompleteRead(b"", response.length)
                            if streaming and not completed:
                                raise _BackendResponseError("backend SSE ended before [DONE]")
                            break
                        if not streaming:
                            finished = getattr(response, "length", None) == 0
                            pipe.put("last_data" if finished else "data", chunk)
                            if finished:
                                break
                            continue
                        pending.extend(chunk)
                        while True:
                            boundary = re.search(rb"\r\n\r\n|\n\n|\r\r", pending)
                            if boundary is None:
                                break
                            if boundary.end() > 262144:
                                raise _BackendResponseError("backend SSE event exceeds 262144 bytes")
                            event = bytes(pending[:boundary.end()])
                            del pending[:boundary.end()]
                            data_lines = [
                                line.partition(b":")[2].strip()
                                for line in event.splitlines() if line.startswith(b"data:")
                            ]
                            completed = data_lines == [b"[DONE]"]
                            pipe.put("terminal" if completed else "data", event)
                            if completed:
                                break
                        if completed:
                            break
                        if len(pending) > 262144:
                            raise _BackendResponseError("backend SSE event exceeds 262144 bytes")
                    if not ticket.cancelled.is_set():
                        pipe.put("done", 200 <= response.status < 300)
                        # The lease covers delivery as well as upstream reads.
                        # The handler signals completion/disconnect in finally.
                        ticket.cancelled.wait()
                finally:
                    if response is not None:
                        response.close()
                    if connection is not None:
                        connection.close()
                    lease.__exit__(None, None, None)
            except Exception as error:
                pipe.put("error", error)
            finally:
                self.server.admission.finish_reader(ticket)

        thread = threading.Thread(target=read_backend, name="broker-backend-reader", daemon=True)
        try:
            if streaming:
                self._start_sse(profile)
            ticket.reader_started = True
            try:
                thread.start()
            except BaseException as error:
                ticket.reader_started = False
                if isinstance(error, Exception):
                    raise _BackendResponseError("backend reader could not start") from error
                raise
            heartbeat_at = time.monotonic() + SSE_HEARTBEAT_INTERVAL
            upstream_success = False
            while True:
                if self.server._closing.is_set():
                    raise AdmissionError("server_shutdown")
                if self._client_disconnected():
                    raise _ClientDisconnectedError("downstream EOF")
                if pipe.deadline is not None and time.monotonic() >= pipe.deadline:
                    raise TimeoutError("backend request deadline expired")
                if pipe.deadline is None and time.monotonic() >= ticket.deadline:
                    raise AdmissionError("queue_expired")
                try:
                    kind, value = pipe.messages.get(timeout=0.02)
                except queue.Empty:
                    if streaming and time.monotonic() >= heartbeat_at:
                        self._write_client(b": heartbeat\n\n")
                        heartbeat_at = time.monotonic() + SSE_HEARTBEAT_INTERVAL
                    continue
                if kind == "error":
                    if not isinstance(value, (AdmissionError, BrokerCoordinationError,
                                              BackendControllerError, OSError, http.client.HTTPException)):
                        raise _BackendResponseError(
                            f"backend reader failed ({type(value).__name__})"
                        ) from value
                    raise value
                if kind == "headers":
                    status, reason, headers, empty_body = value
                    upstream_success = 200 <= status < 300
                    if status >= 400:
                        self._diagnostic("backend_failure", profile=profile.id, upstream_status=status)
                    if not streaming:
                        self._send_proxy_headers(status, reason, headers, profile)
                        if empty_body:
                            self._proxy_completed = upstream_success
                            break
                elif kind in {"data", "last_data", "terminal"}:
                    self._write_client(value)
                    heartbeat_at = time.monotonic() + SSE_HEARTBEAT_INTERVAL
                    if kind != "data":
                        self._proxy_completed = upstream_success
                        break
                elif kind == "done":
                    self._proxy_completed = value
                    break
        except _ClientDisconnectedError as error:
            if self.server._closing.is_set():
                self._diagnostic("server_shutdown", profile=profile.id, phase="response")
            else:
                self._diagnostic("client_disconnect", profile=profile.id,
                                 phase="queue" if pipe.deadline is None else "response")
                _log(f"[broker] client disconnected during {profile.id} response "
                     f"(request {self._request_id()}): {error}")
        finally:
            pipe.abort()
            # Socket shutdown wakes HTTPResponse without concurrent buffered
            # reads/closes. A committed activation finishes under its existing
            # lifecycle deadline and releases any late lease without dispatch.
            if thread.ident is not None:
                thread.join(timeout=0.1)
            self._proxy_pipe = None
            self.close_connection = True

    def _client_disconnected(self) -> bool:
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            previous = self.connection.gettimeout()
            try:
                self.connection.settimeout(0.001)
                return self.connection.recv(1) == b""
            finally:
                self.connection.settimeout(previous)
        except (BlockingIOError, TimeoutError, ssl.SSLWantReadError):
            return False
        except OSError:
            return True

    def _require_current_profile(self, profile: Profile) -> None:
        try:
            current = self.application.registry.get(profile.id)
        except ProfileError:
            current = None
        if current != profile:
            raise BrokerRequestStaleError("profile changed while the request waited; resubmit it")

    def _start_sse(self, profile: Profile) -> None:
        self._send_proxy_headers(200, None, [
            ("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"),
            ("X-Accel-Buffering", "no"),
        ], profile)
        self._sse_started = True
        self._write_client(b": heartbeat\n\n")

    def _send_proxy_headers(self, status, reason, headers, profile) -> None:
        self._client_write_timeout()
        try:
            self.send_response(status, reason)
            # The llama.cpp backend already mirrors the request Origin and emits
            # its own Access-Control-Allow-Origin; we strip it here so the broker
            # (see end_headers) is the *single* CORS authority. A duplicate ACAO
            # header is exactly what makes Safari reject a cross-origin response.
            skip = HOP_BY_HOP_RESPONSE_HEADERS | {
                "x-request-id",
                "x-llama-broker-profile",
                "access-control-allow-origin",
            }
            for key, value in headers:
                if key.lower() not in skip:
                    self.send_header(key, value)
            self.send_header("X-Llama-Broker-Profile", profile.id)
            decision = getattr(self, "_system1_decision", None)
            if decision is not None:
                self.send_header("X-Llama-Broker-System1-Route", decision.reason)
            self.send_header("X-Request-ID", self._request_id())
            self.send_header("Connection", "close")
            self.end_headers()
            self._proxy_response_started = True
        except OSError as error:
            if isinstance(error, TimeoutError):
                self._client_write_timeout()
            raise _ClientDisconnectedError(str(error)) from error

    def _diagnostic(self, event: str, **metadata: Any) -> None:
        _controller_event_logger({"event": event, "request_id": self._request_id(), **metadata})

    def _send_admission_error(self, error: AdmissionError) -> None:
        self._diagnostic(error.event)
        status = HTTPStatus.GATEWAY_TIMEOUT if error.event == "queue_expired" else HTTPStatus.SERVICE_UNAVAILABLE
        try:
            self._send_json(status, {"error": {
                "message": str(error), "type": f"broker_{error.event}",
                "code": f"broker_{error.event}",
            }})
        except (_ClientDisconnectedError, OSError):
            if not self.server._closing.is_set():
                self._diagnostic("client_disconnect", phase="admission_error")

    def _client_write_timeout(self) -> None:
        timeout = self.server._control_client_timeout
        pipe = getattr(self, "_proxy_pipe", None)
        if pipe is not None:
            deadline = pipe.deadline if pipe.deadline is not None else pipe.ticket.deadline
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if pipe.deadline is None:
                    raise AdmissionError("queue_expired")
                raise TimeoutError("backend request deadline expired")
            timeout = min(timeout, remaining)
        self.connection.settimeout(timeout)

    def _write_client(self, data: bytes) -> None:
        self._client_write_timeout()
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except OSError as error:
            if isinstance(error, TimeoutError):
                self._client_write_timeout()
            raise _ClientDisconnectedError(str(error)) from error

    def _proxy_system1(self) -> None:
        """Forward a decision request to the resident system 1 backend.

        This never takes a profile lease: d1 lives outside the main backend's
        lifecycle, so decisions keep working while profiles drain or swap.
        """

        service = self.application.system1
        if service is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {
                    "error": {
                        "message": "system1 decision model is not enabled",
                        "type": "not_found",
                    }
                },
            )
            return
        try:
            body = self._read_raw_body()
            json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": str(error), "type": "invalid_request_error"}},
            )
            return
        try:
            status, content_type, data = service.post_raw(body)
        except (OSError, http.client.HTTPException) as error:
            _log(f"[broker] system1 proxy failure: {error}")
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {
                    "error": {
                        "message": "system1 backend request failed",
                        "type": "broker_proxy_error",
                    }
                },
            )
            return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Llama-Broker-System1", service.config.model)
        self.send_header("X-Request-ID", self._request_id())
        self.send_header("Connection", "close")
        self.end_headers()
        if data:
            self.wfile.write(data)
            self.wfile.flush()
        self.close_connection = True

    def _handle_decisions(self) -> None:
        """Answer an OpenAI Decisions request with the resident d1 model.

        Like ``/v1/systemone`` this never takes a profile lease or touches the
        main backend; it only translates the request and response shapes.
        """

        service = self.application.system1
        if service is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {
                    "error": {
                        "message": "no decision model is enabled on this broker",
                        "type": "not_found",
                    }
                },
            )
            return
        try:
            payload = json.loads(self._read_raw_body().decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise DecisionRequestError("request body must be a JSON object")
            if payload.get("stream"):
                raise DecisionRequestError("stream is not supported", param="stream")
            model = self.application.decision_model(payload.get("model"))
            if model is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {
                        "error": {
                            "message": (
                                f"model {payload.get('model')!r} cannot answer decisions; "
                                f"use {service.config.model!r}"
                            ),
                            "type": "invalid_request_error",
                            "param": "model",
                            "code": "model_not_found",
                        }
                    },
                )
                return
            questions = decisions.parse_questions(payload.get("questions"))
            state, images = decisions.parse_input(payload.get("input"))
        except DecisionRequestError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, error.as_error())
            return
        except (ValueError, UnicodeDecodeError) as error:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": str(error), "type": "invalid_request_error"}},
            )
            return
        try:
            result = service.decide(
                state,
                decisions.systemone_request(state, images, questions)["questions"],
                images=images,
                response_transform=lambda answers: decisions.decision_response(model, questions, answers),
            )
        except (System1Error, DecisionBackendError) as error:
            _log(f"[broker] decisions failure: {error}")
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": {"message": str(error), "type": "broker_proxy_error"}},
            )
            return
        self._send_json(
            HTTPStatus.OK, result, extra_headers={"X-Llama-Broker-System1": model}
        )

    def _model_not_found(self, alias: str, *, message: str | None = None) -> None:
        self._send_json(
            HTTPStatus.NOT_FOUND,
            {
                "error": {
                    "message": message or f"unknown model: {alias}",
                    "type": "broker_model_not_found",
                }
            },
        )

    def _request_id(self) -> str:
        current = getattr(self, "_broker_request_id", None)
        if isinstance(current, str):
            return current
        candidate = self.headers.get("X-Request-ID", "").strip()
        if candidate and len(candidate) <= 128 and "\r" not in candidate and "\n" not in candidate:
            current = candidate
        else:
            current = uuid.uuid4().hex
        self._broker_request_id = current
        return current

    def _send_json(
        self,
        status: int,
        payload: Mapping[str, Any],
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        if self._sse_started:
            error = dict(payload.get("error", {}))
            error.setdefault("code", error.get("type", "broker_proxy_error"))
            terminal = {"error": error, "request_id": self._request_id(), "status": int(status)}
            self._write_client(b"event: error\ndata: " + _json_bytes(terminal) + b"\n\n")
            self.close_connection = True
            return
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-ID", self._request_id())
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def _proxy_backend_get(self, path: str) -> None:
        client = self.application.backend_client()
        try:
            try:
                status, content_type, body = client.get_raw(
                    path,
                    headers=self._request_headers(b""),
                )
            except (OSError, http.client.HTTPException) as e:
                _log(f"[broker] backend proxy GET {path} failed: {e}")
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": {"message": "backend unavailable", "type": "broker_proxy_error"}},
                )
                return
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if body:
                self._write_client(body)
        except (OSError, _ClientDisconnectedError) as error:
            # The backend call already finished; only the client side can fail here.
            _log(
                f"[broker] client disconnected during backend GET {path} "
                f"(request {self._request_id()}): {error}"
            )
        finally:
            client.close()
            self.close_connection = True


class BrokerServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        application: BrokerApplication,
        max_workers: int,
        control_workers: int = CONTROL_WORKERS,
        control_client_timeout: float = CONTROL_CLIENT_TIMEOUT,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        for name, value in (("max_workers", max_workers), ("control_workers", control_workers),
                            ("header_workers", HEADER_WORKERS)):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (("control_client_timeout", control_client_timeout),
                            ("queue_wait_timeout", INFERENCE_QUEUE_WAIT_TIMEOUT),
                            ("heartbeat_interval", SSE_HEARTBEAT_INTERVAL)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if INFERENCE_QUEUE_MAX_SOCKETS < 0 or INFERENCE_QUEUE_MAX_BODY_BYTES <= 0:
            raise ValueError("inference queue limits must be non-negative sockets and positive bytes")
        self.application = application
        self.admission = InferenceAdmission(max_workers, INFERENCE_QUEUE_MAX_SOCKETS,
                                            INFERENCE_QUEUE_MAX_BODY_BYTES,
                                            INFERENCE_QUEUE_WAIT_TIMEOUT)
        self._data_worker_slots = self.admission.slots
        self._control_worker_slots = threading.BoundedSemaphore(control_workers)
        self._header_slots = threading.BoundedSemaphore(HEADER_WORKERS)
        self._connection_slots = threading.BoundedSemaphore(
            max_workers + INFERENCE_QUEUE_MAX_SOCKETS + control_workers + HEADER_WORKERS
        )
        self._control_client_timeout = control_client_timeout
        self._ssl_context = ssl_context
        self._request_context = threading.local()
        self._clients: set[socket.socket] = set()
        self._clients_lock = threading.Lock()
        self._closing = threading.Event()
        super().__init__(server_address, BrokerHandler)

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        # Unparsed peers have a separate short-lived bound. Inference cannot
        # occupy admin workers, and TLS handshakes never block the accept loop.
        if self._closing.is_set() or not self._connection_slots.acquire(False):
            _controller_event_logger({
                "event": "server_shutdown" if self._closing.is_set() else "connection_full",
                "request_id": uuid.uuid4().hex,
            })
            self.shutdown_request(request)
            return
        if not self._header_slots.acquire(False):
            try:
                _controller_event_logger({"event": "connection_full", "request_id": uuid.uuid4().hex})
                self.shutdown_request(request)
            finally:
                self._connection_slots.release()
            return
        with self._clients_lock:
            self._clients.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            try:
                self.shutdown_request(request)
            finally:
                with self._clients_lock:
                    self._clients.discard(request)
                self._header_slots.release()
                self._connection_slots.release()
            raise

    def release_header_slot(self) -> None:
        if getattr(self._request_context, "header_slot", False):
            self._request_context.header_slot = False
            self._header_slots.release()

    def process_request_thread(
        self, request: socket.socket, client_address: tuple[str, int]
    ) -> None:
        raw_request = request
        self._request_context.header_slot = True
        self._request_context.handler = None
        try:
            request.settimeout(self._control_client_timeout)
            if self._ssl_context is not None:
                request = self._ssl_context.wrap_socket(
                    request, server_side=True, do_handshake_on_connect=False
                )
                with self._clients_lock:
                    self._clients.discard(raw_request)
                    self._clients.add(request)
                request.do_handshake()
            if not self._closing.is_set():
                self.finish_request(request, client_address)
        except (OSError, _ClientDisconnectedError):
            # Includes bounded handshake/header/body timeouts and TLS EOF.
            pass
        except Exception:
            self.handle_error(request, client_address)
        finally:
            try:
                self.shutdown_request(request)
            finally:
                try:
                    handler = self._request_context.handler
                    if handler is not None:
                        handler._release_admission()
                finally:
                    self._request_context.handler = None
                    self.release_header_slot()
                    with self._clients_lock:
                        self._clients.discard(raw_request)
                        self._clients.discard(request)
                    self._connection_slots.release()

    def server_close(self) -> None:
        coordinator = getattr(self.application, "coordinator", None)
        if coordinator is not None:
            coordinator.request_shutdown()
        self._closing.set()
        self.admission.stop()
        with self._clients_lock:
            clients = tuple(self._clients)
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            self.application.shutdown()
        finally:
            super().server_close()


@contextmanager
def _operation_lock(path: Path = LOCK_PATH) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def acquire_process_instance_lock(path: Path = PROCESS_LOCK_PATH) -> Any:
    """Acquire the Linux-native singleton gate before any backend lifecycle work."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise BrokerAlreadyRunningError(
            f"another Llama-Broker process owns {path}"
        ) from error
    return handle


def release_process_instance_lock(handle: Any) -> None:
    try:
        fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


class LockedController:
    """Cross-process lock shared by the broker and brokerctl."""

    def __init__(self, controller: BackendController, lock_path: Path = LOCK_PATH):
        self.controller = controller
        self.lock_path = lock_path

    @property
    def state_store(self) -> StateFileStore:
        """Expose the backend's shared state adapter to the orchestrator."""
        return self.controller.state_store

    @contextmanager
    def lifecycle(self) -> Iterator[BackendController]:
        """Give the orchestration core one cross-process lifecycle lock lease."""
        with _operation_lock(self.lock_path):
            yield self.controller

    def state(self) -> dict[str, Any]:
        return self.controller.state()

    def preempt_activation(self) -> bool:
        """Signal the in-process backend activation without taking lifecycle locks."""
        return self.controller.preempt_activation()

    def replace_profiles(self, profiles: Mapping[str, Profile]) -> None:
        with self.lifecycle() as controller:
            controller.replace_profiles(profiles)

    def install_units(self) -> dict[str, Path]:
        with self.lifecycle() as controller:
            return controller.install_units()

    def switch(self, profile_id: str) -> dict[str, Any]:
        with self.lifecycle() as controller:
            return controller.switch(profile_id)

    def restart(self, profile_id: str) -> dict[str, Any]:
        with self.lifecycle() as controller:
            return controller.restart(profile_id)

    def stop_all(self) -> dict[str, Any]:
        with self.lifecycle() as controller:
            return controller.stop_all()

    def reconcile(self, *, active_probe_timeout: float | None = None) -> dict[str, Any]:
        with self.lifecycle() as controller:
            return controller.reconcile(active_probe_timeout=active_probe_timeout)

    def record_recovery_failure(
        self, profile_id: str, *, attempts: int, error: str
    ) -> dict[str, Any]:
        with self.lifecycle() as controller:
            return controller.record_recovery_failure(
                profile_id, attempts=attempts, error=error
            )

def _readiness_probe(
    alias: str,
    timeout: float,
    shutdown_requested: threading.Event | None = None,
    security: SecurityConfig | None = None,
    backend_url: str | None = None,
    preempted: threading.Event | None = None,
) -> bool:
    security = security or load_security_config()
    backend_url = backend_url or effective_backend_url(security)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if preempted is not None and preempted.is_set():
            return False
        if shutdown_requested is not None and shutdown_requested.is_set():
            raise BrokerQuiescingError("Llama-Broker startup was interrupted")
        remaining = max(0.05, deadline - time.monotonic())
        client = BackendClient(
            backend_url,
            timeout=min(1.0, remaining),
            ca_file=security.backend_tls_ca_file,
            default_headers=_backend_auth_headers(security),
        )
        try:
            payload = client.list_models()
            if isinstance(payload, Mapping) and any(
                isinstance(item, Mapping) and item.get("id") == alias
                for item in payload.get("data", [])
            ):
                return True
        except (OSError, ValueError, json.JSONDecodeError, http.client.HTTPException):
            pass
        finally:
            client.close()
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.5, remaining))
    return False


def create_production_controller(
    registry: ProfileRegistry,
    shutdown_requested: threading.Event | None = None,
    security: SecurityConfig | None = None,
    preemption_event: threading.Event | None = None,
) -> BackendController:
    preemption_event = preemption_event or threading.Event()
    if migrate_legacy_state():
        _log(f"[broker] migrated controller state to {STATE_PATH}")
    security = security or load_security_config()
    security.validate()
    backend_listen_host = effective_backend_listen_host(security)
    security.validate_network_binding(BROKER_HOST, backend_listen_host)
    backend_url = effective_backend_url(security)
    backend_parts = urlsplit(backend_url)
    backend_tls_enabled = security.backend_tls_cert_file is not None
    if (backend_parts.scheme == "https") != backend_tls_enabled:
        raise SecurityError(
            "LLM_BROKER_BACKEND_URL scheme and backend TLS certificate settings disagree"
        )
    managed_args = {
        runtime: security.backend_server_args(
            listen_host=backend_listen_host,
            listen_port=BACKEND_LISTEN_PORT,
            runtime=runtime,
        )
        for runtime in {profile.runtime for profile in registry.profiles}
    }
    readiness_profiles: dict[str, Profile] = {}

    def replace_readiness_profiles(profiles: Mapping[str, Profile]) -> None:
        readiness_profiles.clear()
        for profile in profiles.values():
            for alias in (profile.id, profile.alias, *profile.aliases):
                readiness_profiles[alias.casefold()] = profile

    def readiness_probe(alias: str, timeout: float) -> bool:
        profile = readiness_profiles.get(str(alias).strip().casefold())
        if profile is None:
            raise ProfileError(f"unknown model alias: {alias}")
        probe_url = backend_url_for_runtime(backend_url, profile.runtime)
        return _readiness_probe(
            getattr(profile, "backend_model_name", profile.alias),
            timeout,
            shutdown_requested,
            security,
            probe_url,
            preempted=preemption_event,
        )

    setattr(readiness_probe, "replace_profiles", replace_readiness_profiles)
    replace_readiness_profiles({profile.id: profile for profile in registry.profiles})

    def gpu_free_probe() -> int:
        reading = probe_gpu_memory()
        if reading is None:
            raise GpuTelemetryError("GPU memory telemetry is unavailable")
        return reading[1]

    return BackendController(
        profiles={profile.id: profile for profile in registry.profiles},
        state_path=STATE_PATH,
        systemd_dir=SYSTEMD_USER_DIR,
        service_prefix=SERVICE_PREFIX,
        command_runner=subprocess_command_runner,
        readiness_probe=readiness_probe,
        # Keep qualification admission opt-in; handoff telemetry is advisory.
        resource_probe=None,
        vram_probe=gpu_free_probe,
        event_logger=_controller_event_logger,
        service_inventory_probe=systemd_service_inventory_probe,
        service_state_probe=systemd_service_state_probe,
        exit_probe=systemd_backend_exit_probe,
        cancellation_probe=(
            shutdown_requested.is_set if shutdown_requested is not None else None
        ),
        endpoint_probe=lambda: backend_endpoint_ready(
            backend_parts.hostname or "127.0.0.1",
            backend_parts.port or 10000,
            0.25,
        ),
        backend_listen_host=backend_listen_host,
        backend_listen_port=BACKEND_LISTEN_PORT,
        managed_args=managed_args,
        # Hash artifacts only to diagnose a failed backend activation.
        artifact_verifier=lambda profile: _verify_profile_artifacts(profile, force=True),
        preemption_event=preemption_event,
        handoff_min_vram_rise_mib=HANDOFF_MIN_VRAM_RISE_MIB,
        handoff_window_samples=HANDOFF_WINDOW_SAMPLES,
        verify_command=lambda profile: (
            sys.executable,
            str(ROOT / "brokerctl.py"),
            "--profiles",
            str(PROFILES_PATH),
            "verify",
            profile.id,
        ),
    )


def _fresh_recovery_operations() -> dict[str, Any]:
    operations = TransitionQueue().to_operations_state()
    operations["reconciliation_required"] = True
    return operations


def _prepare_durable_state(controller: BackendController) -> bool:
    """Back up unsupported/corrupt state and reconstruct ownership from runtime facts."""
    state_store = controller.state_store
    try:
        state = state_store.read()
    except StateFileError:
        recovered = controller.reconcile_from_runtime()
        if recovered.get("phase") == "stale":
            raise BrokerStateUnknownError(
                "corrupt broker state could not be reconciled to one proven backend"
            )
        recovery = recovered.setdefault("recovery", {})
        if isinstance(recovery, dict):
            recovery["warning"] = (
                "durable state was corrupt; queued requests were discarded and must be resubmitted"
            )
        controller._write_state(recovered)
        return True

    operations_error: OperationStateError | None = None
    try:
        TransitionQueue.from_operations_state(state.get("operations"))
    except OperationStateError as error:
        operations_error = error

    backend_error: BaseException | None = None
    try:
        controller.state()
    except (BackendControllerError, StateFileError) as error:
        backend_error = error

    if operations_error is None and backend_error is None:
        return False

    warning = (
        f"unsupported operation state: {operations_error}"
        if operations_error is not None
        else f"invalid backend state: {backend_error}"
    )
    backend_state = {key: value for key, value in state.items() if key != "operations"}
    if backend_error is not None:
        backend_state = {
            "generation": 0,
            "profile": None,
            "phase": "stale",
            "recovery": {"action": "state_file_reconstruction", "warning": warning},
        }
    state_store.backup_and_replace(backend_state, _fresh_recovery_operations())
    recovered = controller.reconcile_from_runtime()
    if recovered.get("phase") == "stale":
        raise BrokerStateUnknownError(
            "recovered broker state is ambiguous; preserving services and refusing startup"
        )
    recovery = recovered.setdefault("recovery", {})
    if isinstance(recovery, dict):
        recovery["warning"] = (
            warning + "; queued requests were discarded and must be resubmitted"
        )
    controller._write_state(recovered)
    return True
def create_production_system1(registry: ProfileRegistry) -> System1Service | None:
    """Load ``system1.json`` and start the resident decision backend.

    System 1 is advisory: a bad config or a failed start is logged and the
    broker keeps serving profiles; routing aliases then fall back per config.
    """

    try:
        config = load_system1_config(resident_profiles=registry.resident_profiles)
    except (OSError, System1ConfigError) as error:
        _log(f"[broker] system1 disabled: {error}")
        return None
    if config is None:
        return None
    sidecar = None
    if config.managed:
        sidecar = System1Sidecar(
            config,
            command_runner=subprocess_command_runner,
            profile=(
                registry.try_resolve_resident(config.profile) if config.profile else None
            ),
            artifact_verifier=_verify_profile_artifacts,
        )
        if sidecar.unit_name.startswith(f"{SERVICE_PREFIX}-"):
            # The controller would see the unit as a stray profile backend.
            _log(
                f"[broker] system1 disabled: {sidecar.unit_name} collides with "
                f"service prefix {SERVICE_PREFIX!r}"
            )
            return None
        try:
            sidecar.ensure_running()
        except Exception as error:
            _log(f"[broker] system1 backend failed to start: {error}")
    _log(f"[broker] system1 decision model {config.model} at {config.url}")
    return System1Service(config, sidecar=sidecar)


def create_production_application(
    shutdown_requested: threading.Event | None = None,
) -> tuple[BrokerApplication, BackendController]:
    registry = ProfileRegistry.from_path(PROFILES_PATH)
    if shutdown_requested is not None and shutdown_requested.is_set():
        raise BrokerQuiescingError("Llama-Broker startup was interrupted")
    security = load_security_config()
    backend_url = effective_backend_url(security)
    preemption_event = threading.Event()
    controller = create_production_controller(
        registry,
        shutdown_requested,
        security,
        preemption_event=preemption_event,
    )
    with _operation_lock():
        _prepare_durable_state(controller)
        controller.install_units()
        if shutdown_requested is not None and shutdown_requested.is_set():
            raise BrokerQuiescingError("Llama-Broker startup was interrupted")
        reconciled = controller.reconcile(active_probe_timeout=0.5)
        if reconciled.get("phase") == "stale":
            stale_profile = reconciled.get("stale_profile")
            if not isinstance(stale_profile, str):
                raise BrokerStateUnknownError(
                    "stale backend state has no recoverable profile; preserving discovered services"
                )
            controller.switch(stale_profile)
    if shutdown_requested is not None and shutdown_requested.is_set():
        raise BrokerQuiescingError("Llama-Broker startup was interrupted")
    system1 = create_production_system1(registry)
    locked = LockedController(controller)
    parts = urlsplit(backend_url)
    application = BrokerApplication(
        registry=registry,
        controller=locked,
        backend_host=parts.hostname or "127.0.0.1",
        backend_port=parts.port or 10000,
        backend_timeout=BACKEND_TIMEOUT,
        max_body_bytes=MAX_BODY_BYTES,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        backend_url=backend_url,
        security=security,
        drain_timeout=DRAIN_TIMEOUT,
        transition_timeout=TRANSITION_TIMEOUT,
        stop_timeout=STOP_TIMEOUT,
        profiles_path=PROFILES_PATH,
        system1=system1,
    )
    application.start_recovery_watchdog()
    application.start_manifest_watcher()
    return application, controller


def stop_production_units_after_startup_failure() -> None:
    """Stop every discoverable backend when startup fails before app creation."""

    failures: dict[str, BaseException] = {}
    try:
        units = set(systemd_service_inventory_probe(SERVICE_PREFIX))
    except Exception as error:
        units = set()
        failures["service-inventory"] = error

    try:
        legacy_state = systemd_service_state_probe(LEGACY_SERVICE_UNIT)
        if not legacy_state.is_quiescent:
            units.add(LEGACY_SERVICE_UNIT)
    except Exception as error:
        failures[LEGACY_SERVICE_UNIT] = error

    for unit_name in sorted(units):
        try:
            subprocess_command_runner(
                ("systemctl", "--user", "stop", unit_name)
            )
            state = systemd_service_state_probe(unit_name)
            if not state.is_quiescent:
                raise RuntimeError(
                    f"{unit_name} remained active after emergency stop: "
                    f"ActiveState={state.active_state}, SubState={state.sub_state}, "
                    f"MainPID={state.main_pid}, ControlPID={state.control_pid}"
                )
        except Exception as error:
            failures[unit_name] = error

    backend_parts = urlsplit(BACKEND_URL)
    if backend_endpoint_ready(
        backend_parts.hostname or "127.0.0.1",
        backend_parts.port or 10000,
        0.25,
    ):
        failures["private-endpoint"] = RuntimeError(
            "the private backend endpoint is still accepting connections"
        )

    if failures:
        details = "; ".join(
            f"{unit}: {error}" for unit, error in sorted(failures.items())
        )
        raise RuntimeError(
            "could not prove all backend units stopped after broker startup failure: "
            + details
        )


def main() -> int:
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        print("usage: llama-broker [--help] [--version]")
        print("Run the Llama-Broker control plane with the configured environment.")
        return 0
    if "--version" in sys.argv[1:]:
        try:
            from importlib.metadata import version

            print(version("llama-broker"))
        except Exception:
            print("0.1.0")
        return 0

    shutdown_requested = threading.Event()
    received_signal: int | None = None

    def request_shutdown(signum: int, _frame: object) -> None:
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum
        shutdown_requested.set()

    handled_signals = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in handled_signals
    }
    for signum in handled_signals:
        signal.signal(signum, request_shutdown)

    application: BrokerApplication | None = None
    server: BrokerServer | None = None
    startup_error: BaseException | None = None
    shutdown_error: BaseException | None = None
    process_lock: Any | None = None
    try:
        try:
            process_lock = acquire_process_instance_lock()
            application, _controller = create_production_application(
                shutdown_requested=shutdown_requested
            )
            if not shutdown_requested.is_set():
                server = BrokerServer(
                    (BROKER_HOST, BROKER_PORT),
                    application=application,
                    max_workers=MAX_WORKERS,
                    control_workers=CONTROL_WORKERS,
                    control_client_timeout=CONTROL_CLIENT_TIMEOUT,
                    ssl_context=load_security_config().broker_tls_context(),
                )
                server.timeout = 0.25
                scheme = "https" if application.security.broker_tls_cert_file else "http"
                _log(f"[broker] listening on {scheme}://{BROKER_HOST}:{BROKER_PORT}")
                while not shutdown_requested.is_set():
                    server.handle_request()
        except Exception as error:
            startup_error = error
        except KeyboardInterrupt:
            received_signal = signal.SIGINT
            shutdown_requested.set()
    finally:
        if received_signal is not None:
            _log(
                f"[broker] received signal {received_signal}; "
                "quiescing and stopping all backend units"
            )
        try:
            if server is not None:
                server.server_close()
            elif application is not None:
                application.shutdown()
            elif startup_error is not None and process_lock is not None:
                if isinstance(startup_error, BrokerStateUnknownError):
                    _log(
                        "[broker] startup state is ambiguous; preserving all discovered "
                        "backend services and refusing to guess"
                    )
                else:
                    stop_production_units_after_startup_failure()
        except BaseException as error:
            shutdown_error = error
            _log(f"[broker] shutdown failed closed: {error}")
        if process_lock is not None:
            try:
                release_process_instance_lock(process_lock)
            except BaseException as error:
                if shutdown_error is None:
                    shutdown_error = error
                _log(f"[broker] singleton lock release failed: {error}")
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)

    if startup_error is not None:
        if shutdown_requested.is_set():
            _log(f"[broker] startup interrupted by shutdown: {startup_error}")
        else:
            _log(f"[broker] startup failed closed: {startup_error}")
            return 1
    if shutdown_error is not None:
        return 1
    return 130 if received_signal == signal.SIGINT else 0


if __name__ == "__main__":
    raise SystemExit(main())
