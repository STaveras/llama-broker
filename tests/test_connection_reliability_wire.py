"""Disposable slow HTTP/TLS peers; never models, services, or live config."""

import http.client
import json
import socket
import ssl
import subprocess
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import llama_broker.daemon as daemon
from llama_broker.broker_profiles import ProfileRegistry
from llama_broker.broker_security import SecurityConfig
from tests.fixtures import materialize_fixture
from tests.test_broker_http import FakeController
from tests.test_connection_reliability import MAIN, VISION, until


class SlowBackend(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        h = self.server.fixture
        body = self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.loads(body)
        h.seen.append(self.headers["X-Request-ID"])
        mode = payload.get("_fixture")
        try:
            if mode in {"slow", "timeout"}:
                h.headers.wait(4)
            if mode == "error":
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"fixture failure"}}')
            elif payload.get("stream") is True:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Connection", "close")
                self.end_headers()
                if mode == "slow":
                    h.prefill.wait(4)
                # Deliberately split a line: heartbeat comments must never be
                # inserted in the middle of the backend's event.
                self.wfile.write(b'data: {"token":')
                self.wfile.flush()
                if mode == "slow":
                    h.fragment.wait(4)
                self.wfile.write(b'"fixture"}\n\n')
                self.wfile.flush()
                if mode == "slow":
                    h.idle.wait(4)
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                data = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            self.wfile.flush()
        except OSError:
            pass
        finally:
            self.close_connection = True


@pytest.fixture(params=[False, True], ids=["http", "tls"])
def wire(request, tmp_path, monkeypatch):
    tls = request.param
    registry = ProfileRegistry.from_path(materialize_fixture(tmp_path))
    registry = ProfileRegistry(tuple(
        replace(p, role="transient", restore_profile=MAIN) if p.id == VISION else p
        for p in registry.profiles
    ))
    context = None
    ca_file = None
    if tls:
        cert = tmp_path / "fixture.crt"
        key = tmp_path / "fixture.key"
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(cert), "-days", "1", "-subj", "/CN=fixture",
            "-addext", "subjectAltName=IP:127.0.0.1",
        ], check=True, capture_output=True)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        ca_file = cert
    h = SimpleNamespace(seen=[], events=[], tls=tls, client_context=None)
    for name in ("headers", "prefill", "fragment", "idle"):
        setattr(h, name, threading.Event())
    backend = ThreadingHTTPServer(("127.0.0.1", 0), SlowBackend)
    backend.daemon_threads = True
    backend.fixture = h
    if tls:
        backend.socket = context.wrap_socket(backend.socket, server_side=True)
        h.client_context = ssl.create_default_context(cafile=str(ca_file))
    controller = FakeController(MAIN)
    monkeypatch.setattr(daemon, "INFERENCE_QUEUE_MAX_SOCKETS", 2)
    monkeypatch.setattr(daemon, "INFERENCE_QUEUE_MAX_BODY_BYTES", 4096)
    monkeypatch.setattr(daemon, "INFERENCE_QUEUE_WAIT_TIMEOUT", 1)
    monkeypatch.setattr(daemon, "SSE_HEARTBEAT_INTERVAL", 0.03)
    monkeypatch.setattr(daemon, "_controller_event_logger", h.events.append)
    app = daemon.BrokerApplication(
        registry=registry, controller=controller, backend_host="127.0.0.1",
        backend_port=backend.server_port, backend_timeout=1, max_body_bytes=8192,
        backend_url=f"{'https' if tls else 'http'}://127.0.0.1:{backend.server_port}",
        security=SecurityConfig(api_key="fixture-key", backend_tls_ca_file=ca_file),
        backend_alias_ready=lambda _alias: True, drain_timeout=0.3, transition_timeout=0.5,
    )
    broker = daemon.BrokerServer(("127.0.0.1", 0), application=app, max_workers=1,
                                 control_workers=1, control_client_timeout=0.3,
                                 ssl_context=context)
    h.broker = broker
    h.app = app
    h.controller = controller
    h.connections = []

    def send(payload=None, request_id="wire", path="/v1/chat/completions", auth=True):
        if tls:
            connection = http.client.HTTPSConnection("127.0.0.1", broker.server_port,
                                                     timeout=2, context=h.client_context)
        else:
            connection = http.client.HTTPConnection("127.0.0.1", broker.server_port, timeout=2)
        h.connections.append(connection)
        headers = {"X-Request-ID": request_id}
        if auth:
            headers["Authorization"] = "Bearer fixture-key"
        if payload is None:
            connection.request("GET", path, headers=headers)
        else:
            headers["Content-Type"] = "application/json"
            connection.request("POST", path, body=json.dumps({"model": MAIN, **payload}), headers=headers)
        return connection

    h.send = send
    threads = [threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
               for s in (backend, broker)]
    for thread in threads:
        thread.start()
    try:
        yield h
    finally:
        for name in ("headers", "prefill", "fragment", "idle"):
            getattr(h, name).set()
        for connection in h.connections:
            connection.close()
        broker.shutdown()
        broker.server_close()
        backend.shutdown()
        backend.server_close()
        for thread in threads:
            thread.join(1)


def heartbeat(response):
    while True:
        line = response.readline()
        assert line, "stream ended before heartbeat"
        if line == b": heartbeat\n":
            assert response.readline() == b"\n"
            return


def test_slow_header_prefill_fragment_idle_and_saturated_fifo(wire):
    h = wire
    active = h.send({"stream": True, "_fixture": "slow"}, "active").getresponse()
    assert active.status == 200
    heartbeat(active)
    until(lambda: h.seen == ["active"])
    second_connection = h.send({"stream": True}, "second")
    second = second_connection.getresponse()
    heartbeat(second)  # waiting for the only inference worker
    third_connection = h.send({}, "third")
    until(lambda: any(e["event"] == "queue_wait" and e["request_id"] == "third" for e in h.events))
    rejected = h.send({}, "full").getresponse()
    assert rejected.status == 503
    assert json.loads(rejected.read())["error"]["code"] == "broker_queue_full"
    health = h.send(None, "health", "/health").getresponse()
    assert health.status == 200
    health.read()
    denied = h.send({}, "unauthenticated", auth=False).getresponse()
    assert denied.status == 401
    denied.read()
    assert h.app.coordinator.active_requests == 1
    assert h.seen == ["active"]
    heartbeat(active)  # backend headers have not arrived
    h.headers.set()
    heartbeat(active)  # prefill has not produced bytes
    h.prefill.set()
    heartbeat(active)  # a partial SSE data line is buffered upstream
    h.fragment.set()
    while True:
        line = active.readline()
        assert line, "stream ended before token"
        if line == b'data: {"token":"fixture"}\n':
            break
    assert active.readline() == b"\n"
    heartbeat(active)  # idle after the first complete event
    h.idle.set()
    assert b"data: [DONE]" in active.read()
    assert b"data: [DONE]" in second.read()
    third = third_connection.getresponse()
    assert third.status == 200
    assert json.loads(third.read())["ok"]
    assert h.seen == ["active", "second", "third"]


def test_wire_expiry_and_disconnect_reclaim_slots(wire):
    h = wire
    h.broker.admission.timeout = 0.15
    active = h.send({"stream": True, "_fixture": "slow"}, "active").getresponse()
    heartbeat(active)
    until(lambda: h.seen == ["active"])
    gone = h.send({}, "gone")
    until(lambda: any(e["event"] == "queue_wait" and e["request_id"] == "gone" for e in h.events))
    gone.close()
    until(lambda: any(e["event"] == "client_disconnect" and e["request_id"] == "gone" for e in h.events))
    expiry = h.send({"stream": True}, "expiry").getresponse()
    assert expiry.status == 200
    result = expiry.read()
    assert b"event: error\ndata:" in result
    assert b"broker_queue_expired" in result
    assert b'"request_id":"expiry"' in result
    assert h.seen == ["active"]
    for name in ("headers", "prefill", "fragment", "idle"):
        getattr(h, name).set()
    active.read()


@pytest.mark.parametrize("mode,event,code", [
    ("error", "backend_failure", "broker_proxy_error"),
    ("timeout", "backend_timeout", "broker_backend_timeout"),
])
def test_wire_backend_error_and_timeout_are_terminal_without_retry(wire, mode, event, code):
    h = wire
    h.app.backend_timeout = 0.15
    response = h.send({"stream": True, "_fixture": mode}, "failure").getresponse()
    assert response.status == 200
    wire_bytes = response.read()
    assert b"event: error" in wire_bytes
    assert code.encode() in wire_bytes
    assert b"data: [DONE]" not in wire_bytes
    assert h.seen == ["failure"]
    assert any(e["event"] == event and e["request_id"] == "failure" for e in h.events)


def test_stalled_handshake_or_headers_does_not_block_control_acceptance(wire):
    h = wire
    stalled = socket.create_connection(("127.0.0.1", h.broker.server_port), timeout=1)
    try:
        response = h.send(None, "control", "/healthz").getresponse()
        assert response.status == 200
        response.read()
    finally:
        stalled.close()


def test_wire_waits_through_temporary_activation_without_leases(wire):
    h = wire
    entered = threading.Event()
    activation = threading.Event()
    original = h.controller.switch

    def switch(target):
        if target == VISION:
            entered.set()
            activation.wait(4)
        return original(target)

    h.controller.switch = switch
    vision = h.send({"model": VISION, "stream": True}, "vision-load").getresponse()
    try:
        heartbeat(vision)
        assert entered.wait(1)
        waiting = h.send({"stream": True}, "switch-wait").getresponse()
        heartbeat(waiting)
        heartbeat(waiting)
        assert h.app.coordinator.active_requests == 0
        assert not h.seen
    finally:
        activation.set()
    assert b"data: [DONE]" in vision.read()
    assert b"data: [DONE]" in waiting.read()
    assert h.seen == ["vision-load", "switch-wait"]
    assert h.controller.transitions == [VISION, MAIN]
