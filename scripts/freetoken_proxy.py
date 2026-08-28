#!/usr/bin/env python3
"""Run FreeToken behind the broker's TLS and API-key boundary."""

from __future__ import annotations

import argparse
import hmac
import http.client
import os
import signal
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class ProxyServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler, *, upstream_port: int, api_keys: tuple[str, ...]):
        super().__init__(address, handler)
        self.upstream_port = upstream_port
        self.api_keys = api_keys


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FreeTokenBrokerAdapter/1"

    @property
    def proxy_server(self) -> ProxyServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write("freetoken-proxy: " + (format % args) + "\n")
        sys.stderr.flush()

    def _is_loopback(self) -> bool:
        host = str(self.client_address[0])
        return host in {"127.0.0.1", "::1", "localhost"}

    def _authorized(self) -> bool:
        if self._is_loopback():
            return True
        if not self.proxy_server.api_keys:
            return False
        authorization = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            return False
        presented = authorization[len(prefix):].strip()
        return any(hmac.compare_digest(presented, key) for key in self.proxy_server.api_keys)

    def _reject_unauthorized(self) -> None:
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "0")
        self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Connection", "close")
        self.end_headers()

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > 64 * 1024 * 1024:
            raise ValueError("request body exceeds 64 MiB adapter limit")
        return self.rfile.read(length) if length else b""

    def _request_headers(self, body: bytes) -> dict[str, str]:
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP and key.lower() != "authorization"
        }
        headers["Host"] = f"127.0.0.1:{self.proxy_server.upstream_port}"
        headers["Content-Length"] = str(len(body))
        headers["Connection"] = "close"
        return headers

    def _proxy(self) -> None:
        if not self._authorized():
            self._reject_unauthorized()
            return
        try:
            body = self._read_body()
            connection = http.client.HTTPConnection(
                "127.0.0.1", self.proxy_server.upstream_port, timeout=1800
            )
            connection.request(
                self.command,
                self.path,
                body=body,
                headers=self._request_headers(body),
            )
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() in HOP_BY_HOP:
                    continue
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                while chunk := response.read(64 * 1024):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            connection.close()
        except (OSError, ValueError, http.client.HTTPException) as exc:
            self.log_message("upstream request failed: %s", exc)
            if not self.wfile.closed:
                try:
                    self.send_error(502, "FreeToken upstream unavailable")
                except OSError:
                    pass

    do_GET = _proxy
    do_HEAD = _proxy
    do_POST = _proxy
    do_PUT = _proxy
    do_DELETE = _proxy
    do_OPTIONS = _proxy


def _read_api_keys(path: str | None) -> tuple[str, ...]:
    if not path:
        return ()
    values = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return tuple(dict.fromkeys(values))


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--freetoken-external-host", required=True)
    parser.add_argument("--freetoken-external-port", required=True, type=int)
    parser.add_argument("--freetoken-upstream-port", required=True, type=int)
    parser.add_argument("--freetoken-tls-cert")
    parser.add_argument("--freetoken-tls-key")
    parser.add_argument("--freetoken-api-key-file")
    return parser.parse_known_args()


def _wait_for_upstream(port: int, child: subprocess.Popen, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"FreeToken exited during startup with status {child.returncode}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/health", headers={"Connection": "close"})
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status < 500:
                return
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(1)
    raise TimeoutError(f"FreeToken did not become ready on 127.0.0.1:{port}")


def main() -> int:
    options, freetoken_args = _parse_args()
    if options.freetoken_external_port <= 0 or options.freetoken_upstream_port <= 0:
        raise ValueError("proxy ports must be positive")
    if options.freetoken_external_port == options.freetoken_upstream_port:
        raise ValueError("proxy and FreeToken upstream ports must differ")
    if bool(options.freetoken_tls_cert) != bool(options.freetoken_tls_key):
        raise ValueError("FreeToken adapter TLS certificate and key must be provided together")
    if options.freetoken_external_host not in {"127.0.0.1", "::1", "localhost"}:
        if not options.freetoken_tls_cert:
            raise ValueError("non-loopback FreeToken adapter binding requires TLS")

    child_command = [
        sys.executable,
        "-m",
        "freetoken",
        *freetoken_args,
        "--host",
        "127.0.0.1",
        "--port",
        str(options.freetoken_upstream_port),
    ]
    child_env = dict(os.environ)
    child_env["TVM_FFI_CUDA_ARCH_LIST"] = "12.0"
    child = subprocess.Popen(child_command, env=child_env)
    server: ProxyServer | None = None
    try:
        _wait_for_upstream(options.freetoken_upstream_port, child)
        keys = _read_api_keys(options.freetoken_api_key_file)
        server = ProxyServer(
            (options.freetoken_external_host, options.freetoken_external_port),
            ProxyHandler,
            upstream_port=options.freetoken_upstream_port,
            api_keys=keys,
        )
        if options.freetoken_tls_cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(options.freetoken_tls_cert, options.freetoken_tls_key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        stopped = threading.Event()

        def stop(_signum, _frame) -> None:
            if stopped.is_set():
                return
            stopped.set()
            if server is not None:
                threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        server.serve_forever(poll_interval=0.5)
        return 0
    finally:
        if server is not None:
            server.server_close()
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"freetoken-proxy: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
