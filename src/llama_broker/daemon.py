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
import os
import re
import secrets
import signal
import socket
import ssl
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol
from urllib.parse import unquote, urlsplit

from .backend_controller import (
    BackendController,
    BackendControllerError,
    subprocess_command_runner,
    systemd_service_inventory_probe,
    systemd_service_state_probe,
)
from .broker_security import (
    SecurityConfig,
    SecurityError,
    load_security_config,
    read_api_keys,
    revoke_api_key,
    rotate_api_keys,
    write_api_keys,
)
from .tools.broker_backend import BackendClient
from .tools.broker_backend import make_connection
from .broker_profiles import Profile, ProfileError, ProfileRegistry
from .autotune import MaintenanceLease
from .profile_config import sha256_file
from .paths import (
    CONFIG_DIR as XDG_CONFIG_DIR,
    LOCK_PATH as XDG_LOCK_PATH,
    PROCESS_LOCK_PATH as XDG_PROCESS_LOCK_PATH,
    PROFILES_PATH as XDG_PROFILES_PATH,
    STATE_PATH as XDG_STATE_PATH,
    SYSTEMD_USER_DIR as XDG_SYSTEMD_USER_DIR,
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
RECOVERY_MAX_BACKOFF = float(
    _env_value("LLAMA_BROKER_RECOVERY_MAX_BACKOFF", "LLM_BROKER_RECOVERY_MAX_BACKOFF", "60")
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


def backend_serves_alias(
    host: str,
    port: int,
    alias: str,
    timeout: float = 0.5,
    *,
    scheme: str = "http",
    ca_file: str | Path | None = None,
) -> bool:
    """Return whether the private backend advertises *alias* exactly."""

    if timeout <= 0:
        raise ValueError("backend alias readiness timeout must be positive")
    host_for_url = f"[{host}]" if ":" in host and not host.startswith("[") else host
    client = BackendClient(
        f"{scheme}://{host_for_url}:{port}",
        timeout=timeout,
        ca_file=ca_file,
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


class BrokerLifecycleError(BrokerCoordinationError):
    error_type = "broker_lifecycle_failed"


class BrokerStateUnknownError(BrokerCoordinationError):
    error_type = "broker_state_unknown"


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


def _verify_profile_artifacts(profile: Any) -> None:
    """Verify every immutable artifact immediately before a backend start."""

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

    for path, expected, kind in checks:
        if not isinstance(expected, str) or not expected:
            raise BackendControllerError(
                f"{profile.id} {kind} has no configured SHA-256 digest"
            )
        try:
            actual = sha256_file(path)
        except OSError as error:
            raise BackendControllerError(
                f"{profile.id} {kind} is unavailable: {error}"
            ) from error
        if actual != expected:
            raise BackendControllerError(
                f"{profile.id} {kind} SHA mismatch: expected {expected}, "
                f"got {actual} ({path})"
            )


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
        effort = {"min": "minimal"}.get(effort.strip().casefold(), effort.strip().casefold())

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
            low_budget = max(1, budget_limit // ubatch_size)
            if effort == "minimal":
                budget = min(batch_size, budget_limit)
            elif effort == "low":
                budget = low_budget
            elif effort == "max":
                budget = budget_limit
            else:
                budget = min(budget_limit, low_budget * (2 ** (index + 1)))
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
    rewritten["model"] = profile.alias
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
        self._drain_timeout = drain_timeout
        self._transition_timeout = transition_timeout
        self._stop_timeout = stop_timeout
        self._active_requests = 0
        self._queued_transitions = 0
        self._transitioning = False
        self._transition_name: str | None = None
        self._transition_target: str | None = None
        self._pending_restore: tuple[str, str] | None = None
        self._restore_generation = object()
        self._temporary_restore: tuple[object, str, str | None] | None = None
        self._shutdown_requested = False
        self._condition = threading.Condition()

    def _quiescing_unlocked(self) -> bool:
        return (
            self._shutdown_requested
            or self._transitioning
            or self._queued_transitions > 0
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
    def quiescing(self) -> bool:
        with self._condition:
            return self._quiescing_unlocked()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "profile": self._active_profile,
                "active_requests": self._active_requests,
                "quiescing": self._quiescing_unlocked(),
                "transitioning": self._transitioning,
                "queued_transitions": self._queued_transitions,
                "transition": self._transition_name,
                "transition_target": self._transition_target,
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

    def _wait_for_transition_slot(
        self,
        operation_name: str,
        *,
        force_after_drain_timeout: bool = False,
    ) -> bool:
        deadline = time.monotonic() + self._drain_timeout
        forced = False
        with self._condition:
            try:
                if self._shutdown_requested:
                    raise BrokerQuiescingError("Llama-Broker is shutting down")
                while self._transitioning or self._active_requests > 0:
                    if self._shutdown_requested:
                        raise BrokerQuiescingError("Llama-Broker is shutting down")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if force_after_drain_timeout and not self._transitioning:
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
                self._condition.notify_all()
                return forced
            except BaseException:
                self._queued_transitions -= 1
                self._condition.notify_all()
                raise

    def _finish_empty_transition(self) -> None:
        with self._condition:
            failed_target = self._transition_target
            self._transitioning = False
            self._transition_name = None
            self._transition_target = None
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
                if outcome["error"] is not None:
                    # A failed lifecycle operation leaves backend ownership
                    # unknown; never continue leasing the prior profile.
                    self._active_profile = None
                self._transitioning = False
                self._transition_name = None
                self._transition_target = None
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
            raise error
        return outcome["result"]

    def _quiescing_error_unlocked(self) -> BrokerQuiescingError:
        reason = self._transition_name or "a queued transition"
        return BrokerQuiescingError(
            f"broker is quiescing for {reason}; retry after the transition settles"
        )

    def _acquire(
        self,
        profile_id: str,
        *,
        transient: bool = False,
    ) -> tuple[object, str, str | None] | None:
        temporary_restore: tuple[object, str, str | None] | None = None
        with self._condition:
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
            self._queue_transition_unlocked()
            if temporary_restore is not None:
                self._temporary_restore = temporary_restore

        operation_name = f"request switch to {profile_id!r}"
        try:
            self._wait_for_transition_slot(operation_name)
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

    def _acquire_active(self) -> str:
        with self._condition:
            if self._quiescing_unlocked():
                raise self._quiescing_error_unlocked()
            if self._active_profile is None:
                raise BrokerQuiescingError("private backend has no active profile")
            self._active_requests += 1
            return self._active_profile

    def _release(self) -> None:
        with self._condition:
            if self._active_requests <= 0:
                raise RuntimeError("profile lease underflow")
            self._active_requests -= 1
            if self._active_requests == 0:
                self._condition.notify_all()

    @contextmanager
    def lease(self, profile_id: str, transient: bool = False) -> Iterator[None]:
        temporary_restore = self._acquire(profile_id, transient=transient)
        try:
            yield
        finally:
            self._release()
            if temporary_restore is not None:
                self._complete_temporary_restore(temporary_restore)

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
    def lease_active(self) -> Iterator[str]:
        profile_id = self._acquire_active()
        try:
            yield profile_id
        finally:
            self._release()

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

    def _run_switch_transition(self, target_profile: str) -> Any:
        operation_name = f"admin switch to {target_profile!r}"
        self._wait_for_transition_slot(operation_name)
        return self._run_transition(
            operation_name,
            lambda: self._switch_profile(target_profile),
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
        """Drain active requests, then restart the currently active profile."""

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
        self._wait_for_transition_slot(operation_name)
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
    ) -> None:
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.registry = registry
        self.controller = controller
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.backend_timeout = backend_timeout
        self.max_body_bytes = max_body_bytes
        self.max_output_tokens = max_output_tokens
        self.security = security or SecurityConfig()
        self._artifact_verification_cache: dict[str, tuple[tuple[Any, ...], bool]] = {}
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
        state = dict(controller.state())
        self._controller_state = state

        self._backend_url = (backend_url or f"http://{backend_host}:{backend_port}").rstrip("/")
        backend_parts = urlsplit(self._backend_url)
        self._backend_ca_file = self.security.backend_tls_ca_file
        probe_timeout = min(max(backend_timeout, 0.05), 0.5)
        self._backend_ready = backend_ready or (
            lambda: backend_endpoint_ready(backend_host, backend_port, probe_timeout)
        )
        self._backend_alias_ready = backend_alias_ready or (
            lambda alias: backend_serves_alias(
                backend_host,
                backend_port,
                alias,
                probe_timeout,
                scheme=backend_parts.scheme or "http",
                ca_file=self._backend_ca_file,
            )
        )

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
        )

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

    def _switch_profile(self, profile_id: str) -> Any:
        try:
            result = self.controller.switch(profile_id)
        except Exception as error:
            self._refresh_after_lifecycle(raise_on_error=False)
            if isinstance(error, (BackendControllerError, BrokerCoordinationError)):
                raise
            raise BrokerLifecycleError(
                f"controller switch failed: {error}"
            ) from error
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
            result = self.controller.stop_all()
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
            result = self.controller.restart(profile_id)
        except Exception as error:
            self._refresh_after_lifecycle(raise_on_error=False)
            if isinstance(error, (BackendControllerError, BrokerCoordinationError)):
                raise
            raise BrokerLifecycleError(
                f"controller restart failed: {error}"
            ) from error
        self._refresh_after_lifecycle(raise_on_error=True)
        return result

    @property
    def backend_url(self) -> str:
        return self._backend_url

    def backend_client(self, *, timeout: float | None = None) -> BackendClient:
        return BackendClient(
            self._backend_url,
            timeout=self.backend_timeout if timeout is None else timeout,
            ca_file=self._backend_ca_file,
        )

    def authentication_keys(self) -> tuple[str, ...]:
        """Return the in-memory deployment keys accepted by the broker API."""

        return self.security.authentication_keys()

    def _profile_artifacts_verified(self, profile: Profile) -> bool:
        checks = [
            (profile.server_binary, profile.server_sha256),
            (profile.model_path, profile.model_sha256),
        ]
        if profile.mmproj_path is not None:
            if profile.mmproj_sha256 is None:
                return False
            checks.append((profile.mmproj_path, profile.mmproj_sha256))
        fingerprint: list[Any] = []
        for path, expected in checks:
            try:
                stat = path.stat()
            except OSError:
                return False
            fingerprint.extend((str(path), stat.st_size, stat.st_mtime_ns, expected))
        cache_key = tuple(fingerprint)
        cached = self._artifact_verification_cache.get(profile.id)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        verified = True
        for path, expected in checks:
            try:
                if sha256_file(path) != expected:
                    verified = False
                    break
            except OSError:
                verified = False
                break
        self._artifact_verification_cache[profile.id] = (cache_key, verified)
        return verified

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
                    # ``context_length`` is the aggregate ``-c`` value.  The
                    # per-slot window is exposed separately for parallel
                    # backends using ``-np``.
                    "context_length": profile.context_size,
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
            if self._profile_artifacts_verified(profile) or profile.id == active_profile_id
        }
        models = [
            self._openai_model_record({}, profile=profile)
            for profile in visible_profiles.values()
        ]

        return {"object": "list", "data": models}

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
        health_status = (
            "degraded"
            if self._autotune_degraded_reason
            else (
                "ok"
                if state.get("phase") == "inactive" or (phase == "active" and ready)
                else "degraded"
            )
        )
        return {
            "name": "Llama-Broker",
            "status": health_status,
            "backend": {
                "host": self.backend_host,
                "port": self.backend_port,
                **state,
                "ready": ready,
            },
            "leases": self.coordinator.snapshot(),
            "maintenance": {
                "autotune": self._autotune_lease.status(),
                "degraded_reason": self._autotune_degraded_reason,
            },
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
                self._stop_recovery_watchdog()
                self.coordinator.handoff_for_process_shutdown()
            except BaseException as error:
                self._shutdown_error = error
                raise
            finally:
                self._shutdown_complete = True


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class BrokerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Llama-Broker/2.0"

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

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if not self._admit_control_reserve(path):
            return
        if path == "/admin/status" and not self._require_loopback():
            return
        if path.startswith("/v1/") and not self._require_api_auth():
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
                not self.application._profile_artifacts_verified(profile)
                and profile.id != self.application.coordinator.active_profile
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
                with self.application.coordinator.lease_active():
                    self._proxy_backend_get(path)
            except BrokerCoordinationError as error:
                self._send_coordination_error(error)
            return
        self._send_json(
            HTTPStatus.NOT_FOUND,
            {"error": {"message": f"unknown broker endpoint: {path}", "type": "not_found"}},
        )

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if not self._admit_control_reserve(path):
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
        proxy_payload: Mapping[str, Any] | None = None
        payload: Mapping[str, Any] | None = None
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
                profile, rewritten = route_payload(
                    self.application.registry,
                    payload,
                    active_profile=self.application.coordinator.active_profile,
                )
                try:
                    rewritten = apply_reasoning_policy(
                        rewritten,
                        runtime=profile.runtime,
                        max_output_tokens=self.application.max_output_tokens_for(profile),
                        request_max_output_tokens=_requested_output_tokens(payload),
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
                            "profile": profile.id,
                            **reasoning_metadata,
                        }
                    )
                if (
                    not self.application._profile_artifacts_verified(profile)
                    and profile.id != self.application.coordinator.active_profile
                ):
                    self._model_not_found(
                        profile.alias,
                        message=f"model unavailable: {profile.alias}",
                    )
                    return
                if path in {"/v1/chat/completions", "/v1/completions"}:
                    rewritten = ensure_single_system_message(rewritten)
                    rewritten = enforce_output_limit(
                        rewritten,
                        self.application.max_output_tokens_for(profile),
                        default=self.application.default_max_output_tokens_for(profile),
                    )
                proxy_payload = rewritten
                body = _json_bytes(rewritten)
            if path == "/v1/audio/transcriptions" and (
                not self.application._profile_artifacts_verified(profile)
                and profile.id != self.application.coordinator.active_profile
            ):
                self._model_not_found(
                    profile.alias,
                    message=f"model unavailable: {profile.alias}",
                )
                return
        except UnknownModelError as error:
            requested = payload.get("model", "") if payload is not None else ""
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

        try:
            with self.application.coordinator.lease(
                profile.id, transient=profile.is_transient
            ):
                self._proxy(body, proxy_payload, profile)
        except BrokerCoordinationError as error:
            self._send_coordination_error(error)
            return
        except BackendControllerError as error:
            _log(f"[broker] profile switch failed for {profile.id}: {error}")
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
            _log(f"[broker] backend proxy failure for {profile.id}: {error}")
            self.application.coordinator.invalidate_when_idle(profile.id)
            try:
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {
                        "error": {
                            "message": "backend request failed",
                            "type": "broker_proxy_error",
                        }
                    },
                )
            except (BrokenPipeError, ConnectionError):
                pass
            return

    def _handle_admin(self, path: str) -> None:
        try:
            payload = self._read_payload()
            if path == "/admin/switch":
                profile = self.application.registry.resolve(payload.get("model"))
                wait = payload.get("wait", True)
                if not isinstance(wait, bool):
                    raise ValueError("wait must be boolean")
                if wait:
                    self.application.coordinator.switch_when_idle(profile.id)
                else:
                    self.application.coordinator.switch_when_idle_async(profile.id)
                    state = self.application.health()
                    state["operation"] = {
                        "status": "accepted",
                        "target": profile.id,
                        "type": "switch",
                    }
                    self._send_json(HTTPStatus.ACCEPTED, state)
                    return
            elif path == "/admin/restart":
                model = payload.get("model")
                if model is None:
                    self.application.restart_when_idle()
                else:
                    profile = self.application.registry.resolve(model)
                    if self.application.coordinator.active_profile == profile.id:
                        self.application.coordinator.restart_when_idle(profile.id)
                    else:
                        self.application.coordinator.switch_when_idle(profile.id)
            elif path == "/admin/stop":
                self.application.coordinator.stop_when_idle()
            elif path == "/admin/autotune/reload":
                self._send_json(
                    HTTPStatus.OK,
                    self.application.reload_manifest_for_autotune(),
                )
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
        except BackendControllerError as error:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": {"message": str(error), "type": "broker_switch_error"}},
            )
        except BrokerCoordinationError as error:
            self._send_coordination_error(error)

    def _admit_control_reserve(self, path: str) -> bool:
        using_reserve = bool(
            getattr(self.server, "request_uses_control_reserve", lambda: False)()
        )
        control_path = path in {
            "/health",
            "/healthz",
            "/admin/status",
            "/admin/switch",
            "/admin/restart",
            "/admin/stop",
            "/admin/autotune/reload",
            "/admin/autotune/degraded",
            "/admin/credentials/rotate",
            "/admin/credentials/revoke",
            "/v1/models",
            "/api/tags",
        }
        if using_reserve and not control_path:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": {
                        "message": "data-plane workers are saturated",
                        "type": "broker_overloaded",
                    }
                },
            )
            return False
        return True

    def _send_coordination_error(self, error: BrokerCoordinationError) -> None:
        self._send_json(
            error.http_status,
            {"error": {"message": str(error), "type": error.error_type}},
        )

    def _read_raw_body(self) -> bytes:
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
        body = self.rfile.read(length)
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
            if key.lower() not in HOP_BY_HOP_REQUEST_HEADERS
        }
        headers["Content-Length"] = str(len(body))
        headers["Connection"] = "close"
        return headers

    def _proxy(
        self,
        body: bytes,
        payload: Mapping[str, Any] | None,
        profile: Profile,
    ) -> None:
        connection = make_connection(
            self.application.backend_url,
            timeout=self.application.backend_timeout,
            ca_file=self.application.security.backend_tls_ca_file,
        )
        response: http.client.HTTPResponse | None = None
        response_started = False
        try:
            connection.request(
                self.command,
                self.path,
                body=body,
                headers=self._request_headers(body),
            )
            response = connection.getresponse()
            streaming = bool(payload and payload.get("stream")) or (
                response.getheader("Content-Type", "").startswith("text/event-stream")
            )
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                lower = key.lower()
                if lower in HOP_BY_HOP_RESPONSE_HEADERS:
                    continue
                if streaming and lower == "content-length":
                    continue
                self.send_header(key, value)
            self.send_header("X-Llama-Broker-Profile", profile.id)
            self.send_header("X-Request-ID", self._request_id())
            self.send_header("Connection", "close")
            self.end_headers()
            response_started = True

            if streaming:
                reader = getattr(response, "read1", response.read)
                while True:
                    chunk = reader(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            else:
                data = response.read()
                if data:
                    self.wfile.write(data)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as error:
            if not response_started:
                raise ConnectionError(str(error)) from error
            _log(f"[broker] client disconnected during {profile.id} response: {error}")
        finally:
            if response is not None:
                response.close()
            connection.close()
            self.close_connection = True

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
            status, content_type, body = client.get_raw(
                path,
                headers=self._request_headers(b""),
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if body:
                self.wfile.write(body)
                self.wfile.flush()
        except (OSError, http.client.HTTPException) as e:
            _log(f"[broker] backend proxy GET {path} failed: {e}")
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": {"message": "backend unavailable", "type": "broker_proxy_error"}},
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
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if control_workers <= 0:
            raise ValueError("control_workers must be positive")
        if control_client_timeout <= 0:
            raise ValueError("control_client_timeout must be positive")
        self.application = application
        self._data_worker_slots = threading.BoundedSemaphore(max_workers)
        self._control_worker_slots = threading.BoundedSemaphore(control_workers)
        self._control_client_timeout = control_client_timeout
        self._request_lanes: dict[int, str] = {}
        self._request_lanes_lock = threading.Lock()
        self._request_context = threading.local()
        super().__init__(server_address, BrokerHandler)
        if ssl_context is not None:
            self.socket = ssl_context.wrap_socket(self.socket, server_side=True)

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        if self._data_worker_slots.acquire(blocking=False):
            lane = "data"
        elif self._control_worker_slots.acquire(blocking=False):
            lane = "control"
        else:
            self._reject_over_capacity(request)
            return
        with self._request_lanes_lock:
            self._request_lanes[id(request)] = lane
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self._request_lanes_lock:
                self._request_lanes.pop(id(request), None)
            self._release_lane(lane)
            raise

    def process_request_thread(
        self, request: socket.socket, client_address: tuple[str, int]
    ) -> None:
        with self._request_lanes_lock:
            lane = self._request_lanes.pop(id(request))
        self._request_context.control_reserve = lane == "control"
        try:
            if lane == "control":
                request.settimeout(self._control_client_timeout)
            super().process_request_thread(request, client_address)
        finally:
            self._request_context.control_reserve = False
            self._release_lane(lane)

    def request_uses_control_reserve(self) -> bool:
        return bool(getattr(self._request_context, "control_reserve", False))

    def _release_lane(self, lane: str) -> None:
        if lane == "control":
            self._control_worker_slots.release()
        else:
            self._data_worker_slots.release()

    def _reject_over_capacity(self, request: socket.socket) -> None:
        body = _json_bytes(
            {
                "error": {
                    "message": "broker worker capacity is exhausted",
                    "type": "broker_overloaded",
                }
            }
        )
        response = (
            b"HTTP/1.1 503 Service Unavailable\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n"
            + body
        )
        try:
            request.settimeout(self._control_client_timeout)
            request.sendall(response)
        except OSError:
            pass
        finally:
            self.shutdown_request(request)

    def server_close(self) -> None:
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

    def state(self) -> dict[str, Any]:
        return self.controller.state()

    def switch(self, profile_id: str) -> dict[str, Any]:
        with _operation_lock(self.lock_path):
            return self.controller.switch(profile_id)

    def restart(self, profile_id: str) -> dict[str, Any]:
        with _operation_lock(self.lock_path):
            return self.controller.restart(profile_id)

    def stop_all(self) -> dict[str, Any]:
        with _operation_lock(self.lock_path):
            return self.controller.stop_all()

    def record_recovery_failure(
        self, profile_id: str, *, attempts: int, error: str
    ) -> dict[str, Any]:
        with _operation_lock(self.lock_path):
            return self.controller.record_recovery_failure(
                profile_id, attempts=attempts, error=error
            )

def _readiness_probe(
    alias: str,
    timeout: float,
    shutdown_requested: threading.Event | None = None,
    security: SecurityConfig | None = None,
    backend_url: str | None = None,
) -> bool:
    security = security or load_security_config()
    backend_url = backend_url or effective_backend_url(security)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if shutdown_requested is not None and shutdown_requested.is_set():
            raise BrokerQuiescingError("Llama-Broker startup was interrupted")
        remaining = max(0.05, deadline - time.monotonic())
        client = BackendClient(
            backend_url,
            timeout=min(1.0, remaining),
            ca_file=security.backend_tls_ca_file,
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
) -> BackendController:
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
    return BackendController(
        profiles={profile.id: profile for profile in registry.profiles},
        state_path=STATE_PATH,
        systemd_dir=SYSTEMD_USER_DIR,
        service_prefix=SERVICE_PREFIX,
        command_runner=subprocess_command_runner,
        readiness_probe=lambda alias, timeout: _readiness_probe(
            alias, timeout, shutdown_requested, security, backend_url
        ),
        # Production keeps resource probing advisory: the Phase 2 contract
        # forbids a mandatory nvidia-smi/NVML or fixed-free-VRAM gate.
        resource_probe=None,
        event_logger=_controller_event_logger,
        service_inventory_probe=systemd_service_inventory_probe,
        service_state_probe=systemd_service_state_probe,
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
        artifact_verifier=_verify_profile_artifacts,
        verify_command=lambda profile: (
            sys.executable,
            str(ROOT / "brokerctl.py"),
            "--profiles",
            str(PROFILES_PATH),
            "verify",
            profile.id,
        ),
    )


def create_production_application(
    shutdown_requested: threading.Event | None = None,
) -> tuple[BrokerApplication, BackendController]:
    registry = ProfileRegistry.from_path(PROFILES_PATH)
    if shutdown_requested is not None and shutdown_requested.is_set():
        raise BrokerQuiescingError("Llama-Broker startup was interrupted")
    security = load_security_config()
    backend_url = effective_backend_url(security)
    controller = create_production_controller(registry, shutdown_requested, security)
    with _operation_lock():
        controller.install_units()
        if shutdown_requested is not None and shutdown_requested.is_set():
            raise BrokerQuiescingError("Llama-Broker startup was interrupted")
        reconciled = controller.reconcile(active_probe_timeout=0.5)
        if reconciled.get("phase") == "stale":
            stale_profile = reconciled.get("stale_profile")
            if not isinstance(stale_profile, str):
                raise BackendControllerError(
                    "stale backend state has no recoverable profile"
                )
            controller.switch(stale_profile)
    if shutdown_requested is not None and shutdown_requested.is_set():
        raise BrokerQuiescingError("Llama-Broker startup was interrupted")
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
    )
    application.start_recovery_watchdog()
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
