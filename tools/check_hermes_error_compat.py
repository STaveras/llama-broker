#!/usr/bin/env python3
"""Offline compatibility probe against an installed Hermes and OpenAI SDK.

Run with Hermes' Python environment and --hermes-root /path/to/hermes-agent.
Uses real loopback HTTP and broker handlers, a scripted backend, and a scripted
compression hook. No model inference, credentials, or production services are used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


OVERFLOW_MESSAGE = (
    "request (147362 tokens) exceeds the available context size (131072 tokens), "
    "try increasing it"
)


def fixture_backend(handler):
    payload = json.loads(handler.rfile.read(int(handler.headers["Content-Length"])))
    handler.server.seen_payloads.append(payload)
    status = payload.get("fixture_error_status")
    if status:
        body = json.dumps({"error": {
            "code": status, "type": "exceed_context_size", "message": OVERFLOW_MESSAGE,
            "n_prompt_tokens": 147362, "n_ctx": 131072,
        }}).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    else:
        chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 0,
                 "model": "main", "choices": [{"index": 0, "delta": {"content": "fixture recovered"},
                                               "finish_reason": "stop"}]}
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n")
    handler.wfile.flush()
    handler.close_connection = True


def probe(status, fixture, client, classify_api_error, recover_from_overflow, TurnRetryState, APIError):
    messages = [{"role": "user", "content": "long fixture input " * 2000}]
    caught = None
    with client.chat.completions.create(
        model="main", messages=messages, stream=True, extra_body={"fixture_error_status": status},
    ) as stream:
        try:
            list(stream)
        except APIError as error:
            caught = error
    assert caught is not None, "SDK failed to raise the broker's SSE error"
    classified = classify_api_error(
        caught, provider="llama-broker", model="main", approx_tokens=147362,
        context_length=131072, num_messages=len(messages), base_url=str(client.base_url),
    )
    assert classified.reason.value == "context_overflow", classified
    assert classified.should_compress and not classified.should_fallback, classified
    # APIError from an HTTP-200 stream is status-less. Hermes versions differ
    # in whether they also lift status_code out of the SDK's flat error body.
    assert classified.status_code in (None, status), classified
    assert caught.body["status_code"] == status, caught.body
    assert caught.body["code"] == "context_length_exceeded", caught.body
    calls = []
    compacted = [{"role": "user", "content": "short fixture summary"}]

    def compress(_messages, _system, **kwargs):
        calls.append(kwargs)
        return compacted, None

    compressor = SimpleNamespace(context_length=131072, update_model=lambda **kwargs: None)
    agent = SimpleNamespace(
        context_compressor=compressor, tools=[], model="main", provider="llama-broker",
        base_url=str(client.base_url), api_mode="chat_completions", api_key=fixture.auth_key,
        max_tokens=64, log_prefix="[compat-fixture] ", _compress_context=compress,
        _buffer_vprint=lambda *_: None, _buffer_diagnostic_status=lambda *_: None,
    )
    retry = TurnRetryState()
    verdict = recover_from_overflow(
        agent, caught, classified, retry, status_code=classified.status_code,
        error_msg=str(caught), wrapped_output_cap_budget=None, messages=messages,
        api_messages=messages, system_message=None, active_system_prompt=None,
        conversation_history=None, approx_tokens=147362, compression_attempts=0,
        max_compression_attempts=3, api_call_count=1, effective_task_id="offline-compat",
    )
    assert len(calls) == 1 and calls[0]["trigger"] == "overflow", calls
    assert calls[0]["bypass_cooldown"] is True, calls
    assert verdict.action == "break" and retry.restart_with_compressed_messages, verdict
    assert verdict.provider_overflow_recovery_pending, verdict
    with client.chat.completions.create(model="main", messages=verdict.messages, stream=True) as stream:
        text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    assert text == "fixture recovered", text
    assert fixture.broker.application.coordinator.active_requests == 0
    return {"upstream_status": status, "sdk_exception": type(caught).__name__,
            "code": caught.body["code"], "classification": classified.reason.value,
            "compression_calls": len(calls), "retry_with_compacted_messages": True,
            "scripted_retry_succeeded": True, "fallback_requested": classified.should_fallback}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    hermes_root = args.hermes_root.resolve()
    if not (hermes_root / "agent/error_classifier.py").is_file():
        parser.error("--hermes-root must contain agent/error_classifier.py")
    sys.path[:0] = [str(root / "src"), str(root), str(hermes_root)]
    # Isolate even metadata caches that Hermes' recovery may persist.
    with tempfile.TemporaryDirectory(prefix="broker-hermes-error-compat-") as directory:
        home = Path(directory)
        env = {"HOME": directory, "HERMES_HOME": str(home / ".hermes"),
               "HERMES_DISABLE_LAZY_INSTALLS": "1",
               "XDG_CONFIG_HOME": str(home / "config"), "XDG_STATE_HOME": str(home / "state"),
               "XDG_DATA_HOME": str(home / "data"), "XDG_CACHE_HOME": str(home / "cache"),
               "XDG_RUNTIME_DIR": str(home / "run"),
               "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(home / "no-bus"),
               "DBUS_SYSTEM_BUS_ADDRESS": "unix:path=" + str(home / "no-system-bus")}
        with patch.dict(os.environ, env):
            from openai import APIError, OpenAI
            from agent.error_classifier import classify_api_error
            from agent.turn_overflow import recover_from_overflow
            from agent.turn_retry_state import TurnRetryState
            from tests.test_broker_http import BackendHandler, BrokerHttpTests

            fixture = BrokerHttpTests()
            with patch.object(BackendHandler, "do_POST", fixture_backend):
                fixture.setUp()
                try:
                    with OpenAI(base_url=f"http://127.0.0.1:{fixture.broker.server_port}/v1",
                                api_key=fixture.auth_key, max_retries=0, timeout=5) as client:
                        reports = [probe(status, fixture, client, classify_api_error,
                                         recover_from_overflow, TurnRetryState, APIError)
                                   for status in (400, 500)]
                    assert len(getattr(fixture.backend, "seen_payloads")) == 4
                    assert fixture.controller.transitions == []
                    print(json.dumps({"scope": "offline fixtures; scripted compression, no inference",
                                      "hermes_root": str(hermes_root), "reports": reports}, indent=2))
                finally:
                    fixture.tearDown()


if __name__ == "__main__":
    main()
