"""Exercise real handlers/coordinators with disposable in-memory transports.

The separate wire tests exercise HTTP/TLS. These tests also run in sandboxes
that prohibit socket creation; the transport double is only at the socket edge.
"""

import http.client
import io
import json
import threading
import time
from dataclasses import replace
from unittest.mock import patch

import pytest

import llama_broker.daemon as daemon
from llama_broker.broker_profiles import ProfileRegistry
from llama_broker.broker_security import SecurityConfig
from tests.fixtures import materialize_fixture
from tests.test_broker_http import FakeController


MAIN = "fixture-text-7b"
VISION = "fixture-vision-12b"


def until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition did not become true"
        time.sleep(0.005)


class MemorySocket:
    def __init__(self, wire=b"", *, tls=False):
        self.input = io.BytesIO(wire)
        self.output = bytearray()
        self.lock = threading.Lock()
        self.closed = False
        self.disconnected = False
        self.tls = tls
        self.timeout = None

    def makefile(self, *_args, **_kwargs):
        return self.input

    def sendall(self, data):
        if self.disconnected:
            raise BrokenPipeError("fixture disconnected")
        with self.lock:
            self.output.extend(data)

    def recv(self, _size, flags=0):
        if self.tls and flags:
            raise ValueError("non-zero flags not allowed on SSLSocket")
        if self.disconnected:
            return b""
        raise BlockingIOError()

    def settimeout(self, value):
        self.timeout = value

    def gettimeout(self):
        return self.timeout

    def shutdown(self, _how):
        self.disconnected = True

    def close(self):
        self.closed = True

    def bind(self, _address):
        pass

    def listen(self, _backlog):
        pass

    def setsockopt(self, *_args):
        pass

    def getsockname(self):
        return ("127.0.0.1", 12345)


class BackendResponse:
    status = 200
    reason = "OK"

    def __init__(self, backend, payload):
        self.backend = backend
        self.payload = payload
        self.sent = False
        self.streaming = payload.get("stream") is True
        self.length = None if self.streaming else 11

    def getheader(self, name, default=None):
        return dict(self.getheaders()).get(name, default)

    def getheaders(self):
        if self.streaming:
            return [("Content-Type", "text/event-stream")]
        return [("Content-Type", "application/json"), ("Content-Length", "11")]

    def read1(self, _size=8192):
        if not self.sent:
            self.backend.body_gate.wait(2)
            self.sent = True
            if self.payload.get("fail_body"):
                raise ConnectionResetError("fixture reset")
            if not self.streaming:
                self.length = 0
            return b'data: {"token":"ok"}\n\ndata: [DONE]\n\n' if self.streaming else b'{"ok":true}'
        return b""

    def read(self, _size=None):
        return self.read1()

    def close(self):
        pass


class BackendConnection:
    sock = None

    def __init__(self, backend):
        self.backend = backend

    def connect(self):
        pass

    def request(self, method, path, body, headers):
        self.payload = json.loads(body) if not path.endswith("transcriptions") else {}
        self.backend.seen.append((path, self.payload, headers))

    def getresponse(self):
        self.backend.header_gate.wait(2)
        if self.payload.get("fail_headers"):
            raise ConnectionResetError("fixture reset")
        response = BackendResponse(self.backend, self.payload)
        if self.payload.get("error_status"):
            response.status = 500
        return response

    def close(self):
        pass


class Harness:
    def __init__(self, tmp_path, monkeypatch, *, workers=1, cap=3, wait=1, body_cap=4096, drain=0.5):
        monkeypatch.setattr(daemon, "INFERENCE_QUEUE_MAX_SOCKETS", cap)
        monkeypatch.setattr(daemon, "INFERENCE_QUEUE_MAX_BODY_BYTES", body_cap)
        monkeypatch.setattr(daemon, "INFERENCE_QUEUE_WAIT_TIMEOUT", wait)
        monkeypatch.setattr(daemon, "SSE_HEARTBEAT_INTERVAL", 0.03, raising=False)
        # select is only used to detect downstream EOF, not to read the backend.
        monkeypatch.setattr(daemon.select, "select", lambda reads, *_args: (
            [s for s in reads if s.disconnected], [], []
        ))
        self.profiles_path = materialize_fixture(tmp_path)
        registry = ProfileRegistry.from_path(self.profiles_path)
        registry = ProfileRegistry(tuple(
            replace(p, role="transient", restore_profile=MAIN) if p.id == VISION else p
            for p in registry.profiles
        ))
        self.controller = FakeController(MAIN)
        self.app = daemon.BrokerApplication(
            registry=registry, controller=self.controller,
            backend_host="127.0.0.1", backend_port=10000, backend_timeout=0.25,
            max_body_bytes=10000, security=SecurityConfig(api_key="fixture-key"),
            backend_alias_ready=lambda _alias: True,
            drain_timeout=drain, transition_timeout=0.5, stop_timeout=0.5,
            profiles_path=self.profiles_path,
        )
        self.header_gate = threading.Event()
        self.body_gate = threading.Event()
        self.header_gate.set()
        self.body_gate.set()
        self.seen = []
        self.events = []
        monkeypatch.setattr(daemon, "_controller_event_logger", self.events.append)
        monkeypatch.setattr(daemon, "make_connection", lambda *_args, **_kwargs: BackendConnection(self))
        with patch.object(daemon.socket, "socket", return_value=MemorySocket()):
            self.server = daemon.BrokerServer(("127.0.0.1", 0), application=self.app,
                                              max_workers=workers, control_workers=1)
        self.clients = []

    def request(self, payload=None, *, request_id="fixture", path="/v1/chat/completions", auth=True, tls=False):
        method = "GET" if payload is None else "POST"
        body = b"" if payload is None else json.dumps({"model": MAIN, **payload}).encode()
        headers = f"{method} {path} HTTP/1.1\r\nHost: fixture\r\nX-Request-ID: {request_id}\r\n"
        if auth:
            headers += "Authorization: Bearer fixture-key\r\n"
        headers += f"Content-Length: {len(body)}\r\nContent-Type: application/json\r\n\r\n"
        client = MemorySocket(headers.encode() + body, tls=tls)
        self.clients.append(client)
        self.server.process_request(client, ("127.0.0.1", 1234))
        return client

    def finished(self, client):
        until(lambda: client.closed)
        return bytes(client.output)

    def close(self):
        self.header_gate.set()
        self.body_gate.set()
        self.server.server_close()
        until(lambda: all(c.closed for c in self.clients))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    instances = []

    def create(**kwargs):
        instance = Harness(tmp_path, monkeypatch, **kwargs)
        instances.append(instance)
        return instance

    yield create
    for instance in instances:
        instance.close()


@pytest.mark.parametrize(("request_bytes", "status"), [
    pytest.param(
        b"GET /v1/models EXTRA HTTP/1.1\r\nHost: fixture\r\n\r\n",
        400, id="malformed-request-line",
    ),
    pytest.param(
        b"GET /" + b"x" * 65537 + b" HTTP/1.1\r\n\r\n",
        414, id="oversized-request-line",
    ),
    pytest.param(
        b"GET /v1/models HTTP/1.1\r\nX-Large: " + b"x" * 65537 + b"\r\n\r\n",
        431, id="oversized-header",
    ),
])
def test_cors_preserves_http_parser_errors(harness, request_bytes, status):
    h = harness()
    client = MemorySocket(request_bytes)
    h.clients.append(client)
    h.server.process_request(client, ("127.0.0.1", 1))
    assert h.finished(client).startswith(f"HTTP/1.1 {status} ".encode())


@pytest.mark.parametrize("origin", [None, "https://peer.example"])
@pytest.mark.parametrize(("upstream_vary", "expected_fields"), [
    pytest.param(
        [("Vary", "Accept-Language, Origin")], {"accept-language", "origin"},
        id="combined-fields",
    ),
    pytest.param([("Vary", "*")], {"*", "origin"}, id="wildcard"),
    pytest.param(
        [("Vary", "Accept-Encoding"), ("vArY", "Accept-Language")],
        {"accept-encoding", "accept-language", "origin"}, id="multiple-headers",
    ),
])
def test_cors_preserves_upstream_vary(harness, monkeypatch, origin, upstream_vary, expected_fields):
    h = harness()
    original = BackendResponse.getheaders
    monkeypatch.setattr(BackendResponse, "getheaders", lambda self: original(self) + upstream_vary + [
        ("Access-Control-Allow-Origin", "https://backend.example"),
    ])
    body = json.dumps({"model": MAIN}).encode()
    headers = (
        "POST /v1/completions HTTP/1.1\r\nHost: fixture\r\n"
        "Authorization: Bearer fixture-key\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
    )
    if origin is not None:
        headers += f"Origin: {origin}\r\n"
    client = MemorySocket(headers.encode() + b"\r\n" + body)
    h.clients.append(client)
    h.server.process_request(client, ("127.0.0.1", 1))
    response = http.client.HTTPResponse(MemorySocket(h.finished(client)))
    response.begin()
    assert response.status == 200
    fields = {
        field.strip().lower()
        for value in response.headers.get_all("Vary", [])
        for field in value.split(",")
    }
    assert fields == expected_fields
    assert response.headers.get_all("Access-Control-Allow-Origin", []) == (
        [] if origin is None else [origin]
    )
    assert response.read() == b'{"ok":true}'


def test_waits_through_temporary_switch_without_taking_a_lease(harness):
    h = harness(workers=2)
    gate = threading.Event()
    entered = threading.Event()
    original = h.controller.switch

    def switch(target):
        if target == VISION:
            entered.set()
            gate.wait(2)
        return original(target)

    h.controller.switch = switch
    first = h.request({"model": VISION}, request_id="vision")
    assert entered.wait(1)
    second = h.request({}, request_id="switch-wait")
    try:
        time.sleep(0.08)
        assert not second.closed, bytes(second.output)
        assert h.app.coordinator.active_requests == 0
    finally:
        gate.set()
    assert b"200 OK" in h.finished(first)
    assert b"200 OK" in h.finished(second)
    assert h.controller.transitions == [VISION, MAIN]


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_heartbeat_before_backend_headers_and_during_prefill(harness, path):
    h = harness()
    h.header_gate.clear()
    h.body_gate.clear()
    client = h.request({"stream": True}, request_id="header-wait", path=path)
    try:
        until(lambda: b": heartbeat\n\n" in client.output, 0.15)
        assert b"200 OK" in client.output
        assert h.seen
        h.header_gate.set()
        before = bytes(client.output).count(b": heartbeat")
        until(lambda: bytes(client.output).count(b": heartbeat") > before, 0.15)
    finally:
        h.body_gate.set()
        h.header_gate.set()
    wire = h.finished(client)
    assert b'data: {"token":"ok"}' in wire
    assert b"data: [DONE]" in wire


def test_saturated_tls_like_handlers_wait_fifo_and_control_reserve_stays_free(harness):
    h = harness()
    h.body_gate.clear()
    active = h.request({"stream": True}, request_id="active", tls=True)
    until(lambda: len(h.seen) == 1)
    queued = []
    for index in range(3):
        queued.append(h.request({}, request_id=f"queued-{index}", tls=True))
        time.sleep(0.02)
    try:
        assert all(not c.closed for c in queued)
        assert h.app.coordinator.active_requests == 1
        control = h.request(None, path="/health")
        assert b"200 OK" in h.finished(control)
        denied = h.request({}, auth=False)
        assert b"401 Unauthorized" in h.finished(denied)
    finally:
        h.body_gate.set()
    assert b"200 OK" in h.finished(active)
    for client in queued:
        assert b"200 OK" in h.finished(client)
    assert [headers["X-Request-ID"] for _, _, headers in h.seen] == [
        "active", "queued-0", "queued-1", "queued-2"
    ]


def test_queue_cap_and_expiry_are_distinct_correlated_events(harness):
    h = harness(cap=1, wait=0.12)
    h.body_gate.clear()
    active = h.request({"stream": True}, request_id="active")
    until(lambda: h.seen)
    waiting = h.request({}, request_id="expiry")
    time.sleep(0.02)
    rejected = h.request({}, request_id="full")
    try:
        assert b"broker_queue_full" in h.finished(rejected)
        assert b"broker_queue_expired" in h.finished(waiting)
        assert {(e["event"], e["request_id"]) for e in h.events} >= {
            ("queue_full", "full"), ("queue_expired", "expiry")
        }
        assert len(h.seen) == 1
    finally:
        h.body_gate.set()
    h.finished(active)


def test_queued_body_memory_is_reserved_before_read(harness):
    h = harness(body_cap=100)
    h.body_gate.clear()
    active = h.request({"stream": True})
    until(lambda: h.seen)
    queued = h.request({"padding": "x" * 40}, request_id="fits")
    time.sleep(0.02)
    too_large = h.request({"padding": "x" * 40}, request_id="body-cap")
    try:
        assert b"broker_queue_full" in h.finished(too_large)
        assert not queued.closed
        assert any(e["event"] == "queue_full" and e["request_id"] == "body-cap" for e in h.events)
    finally:
        h.body_gate.set()
    h.finished(active)
    assert b"200 OK" in h.finished(queued)


def test_queued_disconnect_frees_capacity_without_dispatch(harness):
    h = harness(cap=1)
    h.body_gate.clear()
    active = h.request({"stream": True})
    until(lambda: h.seen)
    gone = h.request({}, request_id="gone")
    time.sleep(0.03)
    gone.disconnected = True
    h.finished(gone)
    assert any(e["event"] == "client_disconnect" and e["request_id"] == "gone" for e in h.events)
    next_client = h.request({}, request_id="next")
    assert not next_client.closed
    h.body_gate.set()
    h.finished(active)
    assert b"200 OK" in h.finished(next_client)
    assert [headers["X-Request-ID"] for _, _, headers in h.seen] == ["fixture", "next"]


@pytest.mark.parametrize("failure", ["fail_headers", "fail_body", "error_status"])
def test_early_sse_errors_are_explicit_terminal_and_never_retried(harness, failure):
    h = harness()
    client = h.request({"stream": True, failure: True}, request_id="backend-error")
    wire = h.finished(client)
    assert b"HTTP/1.1 200 OK" in wire
    assert b"event: error\ndata: " in wire
    assert b'"request_id":"backend-error"' in wire
    assert b"data: [DONE]" not in wire
    assert any(e["event"] == "backend_failure" and e["request_id"] == "backend-error" for e in h.events)
    assert len(h.seen) == 1


def test_finite_backend_timeout_with_heartbeats(harness):
    h = harness()
    h.header_gate.clear()
    client = h.request({"stream": True}, request_id="backend-timeout")
    try:
        wire = h.finished(client)
        assert b": heartbeat" in wire
        assert b"event: error" in wire
        assert b"broker_backend_timeout" in wire
        assert any(e["event"] == "backend_timeout" and e["request_id"] == "backend-timeout" for e in h.events)
        assert len(h.seen) == 1
    finally:
        h.header_gate.set()


def test_shutdown_removes_waiters_without_leases_or_dispatch(harness):
    h = harness()
    h.body_gate.clear()
    active = h.request({"stream": True})
    until(lambda: h.seen)
    queued = h.request({"stream": True}, request_id="shutdown-wait")
    time.sleep(0.03)
    closer = threading.Thread(target=h.server.server_close)
    closer.start()
    until(lambda: h.server._closing.is_set())
    h.body_gate.set()
    closer.join(1)
    assert not closer.is_alive()
    h.finished(active)
    h.finished(queued)
    assert len(h.seen) == 1
    assert h.app.coordinator.active_requests == 0
    assert h.controller.stops == 0


def test_json_and_audio_never_receive_heartbeats(harness):
    h = harness()
    normal = h.request({"stream": False})
    wire = h.finished(normal)
    assert b"application/json" in wire
    assert b": heartbeat" not in wire
    assert wire.endswith(b'{"ok":true}')
    audio = b"--fixture\r\nbinary-audio\r\n--fixture--\r\n"
    client = MemorySocket(
        b"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: fixture\r\n"
        b"Authorization: Bearer fixture-key\r\n"
        b"Content-Type: multipart/form-data; boundary=fixture\r\n"
        + f"Content-Length: {len(audio)}\r\n\r\n".encode() + audio
    )
    h.clients.append(client)
    h.server.process_request(client, ("127.0.0.1", 1))
    audio_wire = h.finished(client)
    assert b"200 OK" in audio_wire
    assert b": heartbeat" not in audio_wire
    assert b"event: error" not in audio_wire
    assert h.seen[-1][0] == "/v1/audio/transcriptions"


def test_generated_request_id_is_echoed_and_forwarded(harness):
    h = harness()
    client = h.request({}, request_id="")
    wire = h.finished(client)
    response_id = wire.split(b"X-Request-ID: ", 1)[1].split(b"\r\n", 1)[0].decode()
    assert len(response_id) == 32
    assert h.seen[0][2]["X-Request-ID"] == response_id


def test_header_deadline_is_absolute_even_with_trickled_bytes():
    class SlowReader:
        def readline(self, _limit):
            time.sleep(0.12)
            return b"GET /health HTTP/1.1\r\n"

        def read1(self, _limit):
            time.sleep(0.01)
            return b"G"

    reader = daemon._HeaderDeadlineReader(SlowReader(), MemorySocket(), 0.03)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        reader.readline(65537)
    assert time.monotonic() - started < 0.1


def test_disconnect_during_activation_does_not_dispatch_or_leak_lease(harness):
    h = harness()
    gate = threading.Event()
    entered = threading.Event()
    original = h.controller.switch

    def switch(target):
        if target == VISION:
            entered.set()
            gate.wait(2)
        return original(target)

    h.controller.switch = switch
    client = h.request({"stream": True, "model": VISION}, request_id="activation-disconnect")
    assert entered.wait(1)
    client.disconnected = True
    try:
        h.finished(client)
        assert h.seen == []
        assert h.app.coordinator.active_requests == 0
    finally:
        gate.set()
    until(lambda: not h.server.admission.tickets)
    assert h.app.coordinator.active_requests == 0
    assert h.seen == []
    assert h.controller.transitions == [VISION, MAIN]
    assert any(e["event"] == "client_disconnect" and e["request_id"] == "activation-disconnect" for e in h.events)


def test_activation_wait_has_a_deadline_and_late_lease_is_released(harness):
    h = harness(wait=0.08)
    gate = threading.Event()
    original = h.controller.switch

    def switch(target):
        if target == VISION:
            gate.wait(2)
        return original(target)

    h.controller.switch = switch
    client = h.request({"stream": True, "model": VISION}, request_id="activation-expiry")
    try:
        wire = h.finished(client)
        assert b"broker_queue_expired" in wire
        assert h.seen == []
    finally:
        gate.set()
    until(lambda: not h.server.admission.tickets)
    assert h.app.coordinator.active_requests == 0
    assert h.seen == []


def test_disconnect_during_connect_prevents_a_late_dispatch(harness, monkeypatch):
    h = harness()
    gate = threading.Event()
    entered = threading.Event()

    class SlowConnect(BackendConnection):
        def connect(self):
            entered.set()
            gate.wait(2)

    monkeypatch.setattr(daemon, "make_connection", lambda *_args, **_kwargs: SlowConnect(h))
    client = h.request({"stream": True}, request_id="connect-disconnect")
    assert entered.wait(1)
    client.disconnected = True
    try:
        h.finished(client)
    finally:
        gate.set()
    until(lambda: not h.server.admission.tickets)
    assert h.seen == []
    assert h.app.coordinator.active_requests == 0


@pytest.mark.parametrize("chunked", [False, True])
def test_real_http_response_buffer_is_drained_without_socket_select(harness, monkeypatch, chunked):
    h = harness()
    event = b'data: {"token":"buffered"}\n\ndata: [DONE]\n\n'
    headers = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n"
    if chunked:
        encoded = f"{len(event):x}\r\n".encode() + event + b"\r\n0\r\n\r\n"
        headers += b"Transfer-Encoding: chunked\r\n"
    else:
        encoded = event
        headers += f"Content-Length: {len(event)}\r\n".encode()

    class BufferedConnection(BackendConnection):
        def getresponse(self):
            response = http.client.HTTPResponse(MemorySocket(headers + b"\r\n" + encoded))
            response.begin()
            return response

    monkeypatch.setattr(daemon, "make_connection", lambda *_args, **_kwargs: BufferedConnection(h))
    client = h.request({"stream": True})
    wire = h.finished(client)
    assert event in wire
    assert b"event: error" not in wire


def test_heartbeat_does_not_split_a_fragmented_backend_event(harness, monkeypatch):
    h = harness()
    fragment = threading.Event()
    arrived = threading.Event()

    class FragmentedResponse(BackendResponse):
        def read1(self, _size=8192):
            if not self.sent:
                self.sent = True
                arrived.set()
                return b'data: {"token":'
            fragment.wait(2)
            return b'"fragmented"}\n\ndata: [DONE]\n\n'

    class FragmentedConnection(BackendConnection):
        def getresponse(self):
            return FragmentedResponse(h, self.payload)

    monkeypatch.setattr(daemon, "make_connection", lambda *_args, **_kwargs: FragmentedConnection(h))
    client = h.request({"stream": True})
    try:
        assert arrived.wait(1)
        until(lambda: bytes(client.output).count(b": heartbeat") >= 2)
        assert b'data: {"token":' not in client.output
    finally:
        fragment.set()
    wire = h.finished(client)
    assert b'data: {"token":"fragmented"}\n\n' in wire


def test_nonstreaming_backend_error_keeps_status_and_logs_request_id(harness):
    h = harness()
    client = h.request({"error_status": True}, request_id="json-error")
    wire = h.finished(client)
    assert b"HTTP/1.1 500" in wire
    assert b": heartbeat" not in wire
    assert any(e["event"] == "backend_failure" and e["request_id"] == "json-error" for e in h.events)


def test_lease_and_worker_cover_a_blocked_downstream_writer(harness, monkeypatch):
    h = harness()
    h.body_gate.clear()
    write_gate = threading.Event()
    writing = threading.Event()
    original = MemorySocket.sendall

    def send(sock, data):
        if data == b'{"ok":true}':
            writing.set()
            write_gate.wait(2)
        original(sock, data)

    monkeypatch.setattr(MemorySocket, "sendall", send)
    active = h.request({}, request_id="writer")
    until(lambda: h.seen)
    h.body_gate.set()
    try:
        assert writing.wait(1)
        queued = h.request({}, request_id="writer-wait")
        time.sleep(0.04)
        assert len(h.seen) == 1
        assert h.app.coordinator.active_requests == 1
        assert b"200 OK" in h.finished(h.request(None, path="/health"))
    finally:
        write_gate.set()
    h.finished(active)
    assert b"200 OK" in h.finished(queued)
    until(lambda: not h.server.admission.tickets)
    assert h.app.coordinator.active_requests == 0


def test_disabling_extra_queue_slots_keeps_idle_inference_usable(harness):
    h = harness(cap=0)
    assert b"200 OK" in h.finished(h.request({}))
    h.body_gate.clear()
    active = h.request({"stream": True})
    until(lambda: len(h.seen) == 2)
    try:
        assert b"broker_queue_full" in h.finished(h.request({}, request_id="no-slots"))
    finally:
        h.body_gate.set()
    h.finished(active)


@pytest.mark.parametrize("setting,value", [
    ("INFERENCE_QUEUE_MAX_SOCKETS", -1),
    ("INFERENCE_QUEUE_MAX_BODY_BYTES", 0),
    ("INFERENCE_QUEUE_WAIT_TIMEOUT", float("inf")),
    ("SSE_HEARTBEAT_INTERVAL", float("nan")),
])
def test_invalid_bounds_fail_before_opening_a_listener(monkeypatch, setting, value):
    monkeypatch.setattr(daemon, setting, value)
    with pytest.raises(ValueError):
        daemon.BrokerServer(("127.0.0.1", 0), application=None, max_workers=1)


def test_transfer_encoding_and_duplicate_lengths_do_not_enter_admission(harness):
    h = harness()
    for framing in (b"Transfer-Encoding: chunked\r\n", b"Content-Length: 2\r\nContent-Length: 2\r\n"):
        client = MemorySocket(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: fixture\r\n"
            b"Authorization: Bearer fixture-key\r\n" + framing + b"\r\n{}"
        )
        h.clients.append(client)
        h.server.process_request(client, ("127.0.0.1", 1))
        assert b"400 Bad Request" in h.finished(client)
        assert not h.server.admission.tickets
    assert h.seen == []


def test_queue_expiry_during_body_upload_is_not_reported_as_disconnect(harness):
    h = harness(wait=0.05)
    h.body_gate.clear()
    active = h.request({"stream": True})
    until(lambda: h.seen)
    headers = (
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: fixture\r\n"
        b"Authorization: Bearer fixture-key\r\nX-Request-ID: slow-upload\r\n"
        b"Content-Length: 10\r\n\r\n"
    )

    class SlowUpload(io.BytesIO):
        def read1(self, size):
            if self.tell() < len(headers):
                return super().read1(size)
            time.sleep(0.07)
            raise TimeoutError("fixture upload timeout")

    client = MemorySocket()
    client.input = SlowUpload(headers)
    h.clients.append(client)
    h.server.process_request(client, ("127.0.0.1", 1))
    try:
        assert b"broker_queue_expired" in h.finished(client)
        assert any(e["event"] == "queue_expired" and e["request_id"] == "slow-upload" for e in h.events)
        assert not any(e["event"] == "client_disconnect" and e["request_id"] == "slow-upload" for e in h.events)
    finally:
        h.body_gate.set()
    h.finished(active)


def test_http_inference_waits_for_an_administrative_switch(harness):
    h = harness(workers=2)
    gate = threading.Event()
    entered = threading.Event()
    original = h.controller.switch

    def switch(target):
        entered.set()
        gate.wait(2)
        return original(target)

    h.controller.switch = switch
    operation = h.app.submit_switch_operation(VISION)
    assert entered.wait(1)
    client = h.request({"stream": True}, request_id="admin-switch-wait")
    try:
        until(lambda: bytes(client.output).count(b": heartbeat") >= 2)
        assert not client.closed
        assert h.app.coordinator.active_requests == 0
        assert h.seen == []
    finally:
        gate.set()
    assert h.app.wait_for_operation(operation.request_id, 1)["status"] == "succeeded"
    assert b"data: [DONE]" in h.finished(client)


def test_emergency_stop_uses_control_reserve_with_a_full_inference_queue(harness):
    h = harness(cap=1, drain=0.03)
    h.body_gate.clear()
    active = h.request({"stream": True}, request_id="active")
    until(lambda: h.seen)
    queued = h.request({}, request_id="waiting")
    until(lambda: any(e["event"] == "queue_wait" and e["request_id"] == "waiting" for e in h.events))
    try:
        control = h.request({}, path="/admin/stop", request_id="stop")
        assert b"200 OK" in h.finished(control)
        assert h.controller.stops == 1
        assert len(h.seen) == 1
    finally:
        h.body_gate.set()
    h.finished(active)
    assert b"200 OK" in h.finished(queued)


def test_waiter_does_not_dispatch_a_payload_validated_against_changed_profile(harness):
    h = harness()
    h.body_gate.clear()
    active = h.request({"stream": True})
    until(lambda: h.seen)
    queued = h.request({"stream": True}, request_id="changed-profile")
    until(lambda: any(e["event"] == "queue_wait" and e["request_id"] == "changed-profile" for e in h.events))
    manifest = json.loads(h.profiles_path.read_text())
    manifest["profiles"][0]["display_name"] = "Updated profile revision"
    h.profiles_path.write_text(json.dumps(manifest))
    try:
        operation = h.app.submit_profile_reload_operation()
        assert h.app.wait_for_operation(operation.request_id, 1)["status"] == "succeeded"
    finally:
        h.body_gate.set()
    h.finished(active)
    wire = h.finished(queued)
    assert b"broker_request_stale" in wire
    assert b"event: error" in wire
    assert len(h.seen) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_client_closing_after_complete_delivery_is_success(harness, monkeypatch, stream):
    h = harness()
    h.app.crash_loops.record_crash(MAIN, "fixture prior crash")
    original = MemorySocket.sendall

    def send(sock, data):
        original(sock, data)
        if b"data: [DONE]" in data or data == b'{"ok":true}':
            sock.disconnected = True

    monkeypatch.setattr(MemorySocket, "sendall", send)
    client = h.request({"stream": stream}, request_id="completed")
    wire = h.finished(client)
    assert b"200 OK" in wire
    assert h.app.crash_loops.snapshot() == {}
    assert not any(e["event"] == "client_disconnect" and e["request_id"] == "completed" for e in h.events)


def test_connections_are_bounded_until_the_transport_is_closed(harness, monkeypatch):
    monkeypatch.setattr(daemon, "HEADER_WORKERS", 1)
    h = harness(cap=0)
    closing = threading.Event()
    original = MemorySocket.close

    def close(sock):
        if sock.output:
            closing.wait(2)
        original(sock)

    monkeypatch.setattr(MemorySocket, "close", close)
    try:
        # Hold one inference, one control and one rejected parsing transport.
        active = h.request({}, request_id="closing-inference")
        until(lambda: b"200 OK" in active.output)
        control = h.request(None, path="/health", request_id="closing-control")
        until(lambda: b"200 OK" in control.output)
        parsing = h.request({}, auth=False, request_id="closing-rejection")
        until(lambda: b"401 Unauthorized" in parsing.output)
        rejected = h.request({}, request_id="connection-limit")
        time.sleep(0.03)
        assert len(h.seen) == 1
        assert any(e["event"] == "connection_full" for e in h.events)
        assert rejected.closed
    finally:
        closing.set()


def test_closing_transport_retains_inference_slot_and_control_reserve(harness, monkeypatch):
    h = harness(cap=0)
    closing = threading.Event()
    original = MemorySocket.close

    def close(sock):
        if b"X-Request-ID: closing-worker\r\n" in sock.output:
            closing.wait(2)
        original(sock)

    monkeypatch.setattr(MemorySocket, "close", close)
    active = h.request({}, request_id="closing-worker")
    try:
        until(lambda: b'{"ok":true}' in active.output)
        time.sleep(0.02)
        assert b"broker_queue_full" in h.finished(h.request({}, request_id="no-reuse"))
        assert b"200 OK" in h.finished(h.request(None, path="/health"))
        assert len(h.seen) == 1
    finally:
        closing.set()
    h.finished(active)


def test_response_write_reaching_backend_deadline_is_a_timeout(harness, monkeypatch):
    h = harness()
    h.app.backend_timeout = 0.04
    original = MemorySocket.sendall

    def send(sock, data):
        if data.startswith(b"data:"):
            time.sleep(sock.timeout + 0.01)
            raise TimeoutError("fixture stalled response write")
        original(sock, data)

    monkeypatch.setattr(MemorySocket, "sendall", send)
    client = h.request({"stream": True}, request_id="delivery-deadline")
    wire = h.finished(client)
    assert b"broker_backend_timeout" in wire
    assert any(e["event"] == "backend_timeout" and e["request_id"] == "delivery-deadline" for e in h.events)
    assert not any(e["event"] == "client_disconnect" and e["request_id"] == "delivery-deadline" for e in h.events)
    assert len(h.seen) == 1


def test_client_eof_during_body_upload_is_correlated_and_releases_admission(harness):
    h = harness()
    client = MemorySocket(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: fixture\r\n"
        b"Authorization: Bearer fixture-key\r\nX-Request-ID: upload-eof\r\n"
        b"Content-Length: 10\r\n\r\n{"
    )
    h.clients.append(client)
    h.server.process_request(client, ("127.0.0.1", 1))
    h.finished(client)
    assert any(e["event"] == "client_disconnect" and e["request_id"] == "upload-eof" for e in h.events)
    assert not h.server.admission.tickets
    assert not h.seen


def test_different_profile_waits_for_generation_without_triggering_drain_timeout(harness):
    h = harness(workers=2, drain=0.03)
    h.body_gate.clear()
    active = h.request({"stream": True}, request_id="main-running")
    until(lambda: h.seen)
    waiting = h.request({"model": VISION, "stream": True}, request_id="vision-wait")
    try:
        time.sleep(0.09)  # beyond the coordinator's drain deadline
        assert not waiting.closed, bytes(waiting.output)
        assert bytes(waiting.output).count(b": heartbeat") >= 2
        assert h.app.coordinator.active_requests == 1
        assert h.controller.transitions == []
    finally:
        h.body_gate.set()
    assert b"data: [DONE]" in h.finished(active)
    assert b"data: [DONE]" in h.finished(waiting)
    assert h.controller.transitions == [VISION, MAIN]


def test_backend_reader_start_failure_is_terminal_and_releases_admission(harness, monkeypatch):
    h = harness()
    original = threading.Thread.start

    def start(thread):
        if thread.name == "broker-backend-reader":
            raise RuntimeError("fixture reader cannot start")
        original(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    client = h.request({"stream": True}, request_id="reader-start")
    wire = h.finished(client)
    assert b"event: error" in wire
    assert b"broker_proxy_error" in wire
    assert any(e["event"] == "backend_failure" and e["request_id"] == "reader-start" for e in h.events)
    assert not h.server.admission.tickets
    assert not h.seen


@pytest.mark.parametrize("failure", ["truncated", "oversized", "content_type"])
def test_invalid_backend_sse_terminates_explicitly_without_done(harness, monkeypatch, failure):
    h = harness()

    class InvalidResponse(BackendResponse):
        def __init__(self, *args):
            super().__init__(*args)
            self.remaining = 262145 if failure == "oversized" else 0

        def getheaders(self):
            return [("Content-Type", "application/json" if failure == "content_type" else "text/event-stream")]

        def read1(self, size):
            length = min(size, self.remaining)
            self.remaining -= length
            return b"x" * length

    class InvalidConnection(BackendConnection):
        def getresponse(self):
            return InvalidResponse(h, self.payload)

    monkeypatch.setattr(daemon, "make_connection", lambda *_args, **_kwargs: InvalidConnection(h))
    client = h.request({"stream": True}, request_id="sse-contract")
    wire = h.finished(client)
    assert b"event: error" in wire
    assert b"broker_proxy_error" in wire
    assert b"data: [DONE]" not in wire
    assert any(e["event"] == "backend_failure" and e["request_id"] == "sse-contract" for e in h.events)
    assert len(h.seen) == 1


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_backend_timeout_must_remain_finite_and_positive(timeout):
    with pytest.raises(ValueError, match="backend_timeout must be finite and positive"):
        daemon.BrokerApplication(
            registry=None, controller=None, backend_host="127.0.0.1",
            backend_port=1, backend_timeout=timeout, max_body_bytes=1,
        )
