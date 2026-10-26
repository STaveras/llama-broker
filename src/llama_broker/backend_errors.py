"""Bounded, OpenAI-shaped backend errors for streams with committed headers."""

from __future__ import annotations

import json
import re
from typing import Any


MAX_BACKEND_ERROR_BYTES = 65536
_MAX_MESSAGE_CHARS = 4096
_CONTEXT_CODES = frozenset({
    "context_length_exceeded", "context_window_exceeded", "exceed_context_size",
})
_LLAMA_CONTEXT_MESSAGE = re.compile(
    r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)",
    re.IGNORECASE,
)
_TOKEN_FIELDS = (
    "n_prompt_tokens", "n_ctx", "n_ctx_slot", "prompt_tokens", "context_length",
)


def _text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    # JSON can contain lone surrogate escapes; never let these break SSE encoding.
    return value[:limit].encode("utf-8", errors="replace").decode("utf-8")


def backend_error(status: int, body: bytes) -> dict[str, Any]:
    """Retain recovery fields, not arbitrary backend debug objects or HTML.

    ``status_code`` lives inside the error because streaming SDKs discard the
    enclosing SSE object when raising APIError after an HTTP-200 handshake.
    """
    fallback = {
        "message": f"backend returned HTTP {status} without a usable JSON error",
        "type": "broker_proxy_error", "code": "broker_proxy_error",
        "status_code": status,
    }
    if len(body) > MAX_BACKEND_ERROR_BYTES:
        return fallback
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return fallback
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return fallback
    message = _text(error.get("message"), _MAX_MESSAGE_CHARS)
    if message is None:
        return fallback
    result: dict[str, Any] = {
        "message": message,
        "type": _text(error.get("type"), 256) or (
            "invalid_request_error" if status < 500 else "server_error"
        ),
        "code": _text(error.get("code"), 256) or "broker_proxy_error",
        "status_code": status,
    }
    native_code = error.get("code")
    if isinstance(native_code, int) and not isinstance(native_code, bool):
        result["code"] = native_code
    if "param" in error and (error["param"] is None or isinstance(error["param"], str)):
        result["param"] = _text(error["param"], 256)
    for name in _TOKEN_FIELDS:
        value = error.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2**63 - 1:
            result[name] = value
    match = _LLAMA_CONTEXT_MESSAGE.search(message)
    if (str(result["code"]).lower() in _CONTEXT_CODES
            or result["type"].lower() in _CONTEXT_CODES or match):
        if result["code"] != "context_length_exceeded":
            result["upstream_code"] = result["code"]
        result["code"] = "context_length_exceeded"
        if match:
            result.setdefault("prompt_tokens", int(match[1]))
            result.setdefault("context_length", int(match[2]))
        if "n_prompt_tokens" in result:
            result.setdefault("prompt_tokens", result["n_prompt_tokens"])
        for name in ("n_ctx_slot", "n_ctx"):
            if name in result:
                result.setdefault("context_length", result[name])
    return result
