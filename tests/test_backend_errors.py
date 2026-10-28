"""Upstream rejections must remain actionable after early SSE headers."""

import io
import json
import threading

import pytest

from tests.test_connection_reliability import BackendConnection, harness  # noqa: F401


OVERFLOW_MESSAGE = (
    "request (147362 tokens) exceeds the available context size (131072 tokens), "
    "try increasing it"
)


class ErrorResponse:
    reason = "Bad Request"

    def __init__(self, body, status=400, content_type="application/json"):
        self.status = status
        self.content_type = content_type
        self.body = io.BytesIO(body)
        self.length = len(body)
        self.bytes_read = 0

    def getheader(self, name, default=None):
        return dict(self.getheaders()).get(name, default)

    def getheaders(self):
        return [("Content-Type", self.content_type), ("Content-Length", str(self.length))]

    def read1(self, size=8192):
        data = self.body.read(size)
        self.bytes_read += len(data)
        self.length -= len(data)
        return data

    def close(self):
        self.body.close()


def error_event(wire):
    assert wire.startswith(b"HTTP/1.1 200 OK\r\n")
    assert b": heartbeat\n\n" in wire
    assert b"data: [DONE]" not in wire
    events = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith(b"data: ")]
    assert len(events) == 1
    return events[0]


def reject(monkeypatch, body, *, status=400, content_type="application/json"):
    response = ErrorResponse(body, status, content_type)
    monkeypatch.setattr(BackendConnection, "getresponse", lambda self: response)
    return response


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.parametrize("status", [400, 500])
def test_streaming_context_overflow_preserves_recovery_details(harness, monkeypatch, path, status):
    h = harness()
    original = BackendConnection.getresponse
    reject(monkeypatch, json.dumps({"error": {
        "code": status, "type": "exceed_context_size", "message": OVERFLOW_MESSAGE,
        "param": "messages", "n_prompt_tokens": 147362, "n_ctx": 131072,
    }}).encode(), status=status)
    event = error_event(h.finished(h.request({"stream": True}, path=path, request_id="oversized")))
    error = event["error"]
    assert error["message"] == OVERFLOW_MESSAGE
    assert error["code"] == "context_length_exceeded"
    assert error["type"] == "exceed_context_size"
    assert error["param"] == "messages"
    assert error["status_code"] == status
    assert error["upstream_code"] == status
    assert error["prompt_tokens"] == 147362
    assert error["context_length"] == 131072
    assert error["n_prompt_tokens"] == 147362
    assert error["n_ctx"] == 131072
    assert error["request_id"] == event["request_id"] == "oversized"
    assert error["profile"] == "fixture-text-7b"
    assert event["status"] == status
    assert len(h.seen) == 1
    assert h.app.coordinator.active_requests == 0
    assert not h.app.crash_loops.snapshot()
    # A client correction must not require a backend reload or lose admission capacity.
    monkeypatch.setattr(BackendConnection, "getresponse", original)
    assert b"data: [DONE]" in h.finished(h.request({"stream": True}))
    assert len(h.seen) == 2
    assert not h.controller.transitions


def test_streaming_validation_error_keeps_code_type_and_param(harness, monkeypatch):
    h = harness()
    reject(monkeypatch, json.dumps({"error": {
        "message": "temperature must be a number", "type": "invalid_request_error",
        "code": "invalid_parameter", "param": "temperature",
        "prompt": "DO NOT REFLECT INPUT", "headers": {"Authorization": "SECRET"},
    }}).encode())
    event = error_event(h.finished(h.request({"stream": True})))
    assert event["error"]["message"] == "temperature must be a number"
    assert event["error"]["code"] == "invalid_parameter"
    assert event["error"]["type"] == "invalid_request_error"
    assert event["error"]["param"] == "temperature"
    assert b"SECRET" not in json.dumps(event).encode()
    assert b"DO NOT REFLECT INPUT" not in json.dumps(event).encode()


def test_streaming_overflow_message_without_native_code_is_normalized(harness, monkeypatch):
    h = harness()
    reject(monkeypatch, json.dumps({"error": {"message": OVERFLOW_MESSAGE}}).encode())
    error = error_event(h.finished(h.request({"stream": True})))["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["prompt_tokens"] == 147362
    assert error["context_length"] == 131072


@pytest.mark.parametrize("body", [b"<html>SECRET internal proxy debug</html>", b"not json", b'[]', b'{"error": []}'])
def test_unstructured_errors_are_safe_and_correlated(harness, monkeypatch, body):
    h = harness()
    reject(monkeypatch, body, status=503)
    event = error_event(h.finished(h.request({"stream": True})))
    assert event["status"] == 503
    assert event["error"]["status_code"] == 503
    assert event["error"]["code"] == "broker_proxy_error"
    assert "SECRET" not in json.dumps(event)


def test_backend_error_body_read_is_bounded(harness, monkeypatch):
    h = harness()
    response = reject(monkeypatch, b'x' * 1000000, status=500)
    event = error_event(h.finished(h.request({"stream": True})))
    assert event["error"]["code"] == "broker_proxy_error"
    assert response.bytes_read <= 65537
    assert len(json.dumps(event)) < 1024


@pytest.mark.parametrize("message", ["temperature out of range", "context_size must be a positive integer"])
def test_other_validation_errors_do_not_become_context_overflow(harness, monkeypatch, message):
    h = harness()
    reject(monkeypatch, json.dumps({"error": {
        "message": message, "type": "invalid_request_error", "code": "invalid_parameter",
        "status_code": 200, "n_ctx": True, "n_prompt_tokens": -1,
    }}).encode())
    error = error_event(h.finished(h.request({"stream": True})))["error"]
    assert error["code"] == "invalid_parameter"
    assert error["status_code"] == 400
    assert "n_ctx" not in error
    assert "n_prompt_tokens" not in error


@pytest.mark.parametrize("message", ["x" * 50000, "bad surrogate: \ud800"])
def test_error_messages_are_bounded_and_encodable(harness, monkeypatch, message):
    h = harness()
    reject(monkeypatch, json.dumps({"error": {"message": message}}).encode())
    error = error_event(h.finished(h.request({"stream": True})))["error"]
    assert 0 < len(error["message"]) <= 4096
    error["message"].encode("utf-8")


def test_stalled_error_body_keeps_existing_deadline_and_heartbeats(harness, monkeypatch):
    h = harness()
    gate = threading.Event()
    response = reject(monkeypatch, b'{"error":{"message":"never arrives"}}')
    original_read = response.read1

    def blocked_read(size=8192):
        gate.wait(2)
        return original_read(size)

    monkeypatch.setattr(response, "read1", blocked_read)
    try:
        event = error_event(h.finished(h.request({"stream": True})))
        assert event["error"]["code"] == "broker_backend_timeout"
        assert event["status"] == 504
    finally:
        gate.set()


def test_nonstreaming_backend_error_passes_through_without_heartbeats(harness, monkeypatch):
    h = harness()
    body = json.dumps({"error": {"message": OVERFLOW_MESSAGE, "type": "exceed_context_size"}}).encode()
    reject(monkeypatch, body)
    wire = h.finished(h.request({"stream": False}))
    assert wire.startswith(b"HTTP/1.1 400 Bad Request\r\n")
    assert b": heartbeat" not in wire
    assert wire.endswith(body)
