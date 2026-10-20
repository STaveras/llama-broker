"""Wire-level bounds and framing checks with an inert local HTTP upstream."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time

import pytest
from llama_broker.basalt_bridge import BridgeHandler, BridgeServer, _loopback


@pytest.fixture
def bridge():
    class Native(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            if self.path == '/truncated':
                self.send_response(200)
                self.send_header('Content-Length', '20')
                self.end_headers()
                self.wfile.write(b'short')
                return
            if self.path == '/drip':
                self.send_response(200)
                self.end_headers()
                for _ in range(30):
                    try:
                        self.wfile.write(b'x'); self.wfile.flush()
                    except OSError:
                        break
                    time.sleep(.05)
                return
            if self.path == '/hop':
                self.send_response(200)
                self.send_header('Connection', 'x-private')
                self.send_header('X-Private', 'native-only')
                self.send_header('Authorization', 'native-only')
                self.send_header('Content-Length', '2')
                self.end_headers(); self.wfile.write(b'ok')
                return
            self.send_response(200); self.send_header('Content-Length', '2')
            self.end_headers(); self.wfile.write(b'ok')
        do_POST = do_GET
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Native)
    upstream.daemon_threads = True
    class Handler(BridgeHandler):
        upstream_host = '127.0.0.1'
        upstream_port = upstream.server_port
        upstream_timeout = .3
        bridge_tokens = ('first-fixture', 'second-fixture')
    server = BridgeServer(('127.0.0.1', 0), Handler)
    server.client_timeout = .2
    server.max_connections = 2
    for item in (server, upstream):
        threading.Thread(target=item.serve_forever, daemon=True).start()
    yield server
    for item in (server, upstream):
        item.shutdown(); item.server_close()


def request(server, path='/', headers=None):
    conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
    conn.request('GET', path, headers=headers or {'Authorization': 'Bearer second-fixture'})
    return conn, conn.getresponse()


def test_upstream_truncation_is_visible_to_client(bridge):
    conn, response = request(bridge, '/truncated')
    try:
        with pytest.raises(http.client.IncompleteRead): response.read()
    finally: conn.close()


def test_response_connection_nominations_and_credentials_are_private(bridge):
    conn, response = request(bridge, '/hop')
    try:
        assert response.read() == b'ok'
        assert response.getheader('X-Private') is None
        assert response.getheader('Authorization') is None
    finally: conn.close()


def test_dripping_upstream_cannot_extend_absolute_deadline(bridge):
    started = time.monotonic()
    conn, response = request(bridge, '/drip')
    try:
        try: response.read()
        except (OSError, http.client.IncompleteRead): pass
        assert time.monotonic() - started < .9
    finally: conn.close()


def test_stalled_request_body_is_closed_without_forwarding(bridge):
    with socket.create_connection(('127.0.0.1', bridge.server_port), timeout=1) as conn:
        conn.sendall(b'POST / HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer first-fixture\r\nContent-Length: 5\r\n\r\nx')
        started = time.monotonic()
        try: data = conn.recv(4096)
        except socket.timeout: pytest.fail('unbounded client body read')
        assert b'200 OK' not in data
        assert time.monotonic() - started < .8


def test_non_ascii_and_duplicate_credentials_fail_closed(bridge):
    conn, response = request(bridge, headers={'Authorization': 'Bearer \xff'})
    assert response.status == 401
    response.read(); conn.close()
    conn = http.client.HTTPConnection('127.0.0.1', bridge.server_port, timeout=2)
    conn.putrequest('GET', '/')
    conn.putheader('Authorization', 'Bearer first-fixture')
    conn.putheader('Authorization', 'Bearer wrong-fixture')
    conn.endheaders()
    response = conn.getresponse()
    assert response.status == 401
    response.read(); conn.close()


def test_connection_slots_recover_after_stalled_clients(bridge):
    clients = [socket.create_connection(('127.0.0.1', bridge.server_port), timeout=1) for _ in range(2)]
    time.sleep(.05)
    extra = socket.create_connection(('127.0.0.1', bridge.server_port), timeout=.5)
    try:
        try: assert extra.recv(1) == b''
        except ConnectionResetError: pass
        except socket.timeout: pytest.fail('connection bound was not enforced')
    finally:
        extra.close()
        for conn in clients: conn.close()
    time.sleep(.1)
    conn, response = request(bridge)
    assert response.read() == b'ok'
    conn.close()


@pytest.mark.parametrize('path', ['/unexpected', '/prefix/'])
def test_upstream_path_must_not_be_silently_ignored(path):
    with pytest.raises(ValueError): _loopback('http://127.0.0.1:8080'+path)
