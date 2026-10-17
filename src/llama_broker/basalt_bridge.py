"""Authenticated HTTPS passthrough for Basalt's private native HTTP listener.

The child may speak HTTP on loopback; the broker-facing endpoint is always TLS.
Provider credentials are never forwarded to the child or across this boundary.
"""
from __future__ import annotations
import http.client
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import urlsplit

_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length"}

class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    max_body_bytes = 8 * 1024 * 1024
    upstream_host = "127.0.0.1"
    upstream_port = 0
    bridge_token = ""

    def _authorized(self) -> bool:
        supplied = self.headers.get("X-Basalt-Bridge-Token", "")
        return bool(self.bridge_token) and hmac.compare_digest(supplied, self.bridge_token)

    def _reject(self, status: int, message: str) -> None:
        body = message.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers(); self.wfile.write(body)

    def _proxy(self) -> None:
        if not self._authorized():
            return self._reject(401, "invalid bridge credentials")
        if self.command not in {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}:
            return self._reject(405, "method not allowed")
        length = self.headers.get("Content-Length")
        try: size = int(length or "0")
        except ValueError: return self._reject(400, "invalid content length")
        if size < 0 or size > self.max_body_bytes:
            return self._reject(413, "request body too large")
        body = self.rfile.read(size) if size else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP and k.lower() not in {"authorization", "x-api-key", "x-basalt-bridge-token"}}
        headers["Host"] = f"{self.upstream_host}:{self.upstream_port}"
        conn = http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=30)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse(); data = response.read(self.max_body_bytes + 1)
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in _HOP: self.send_header(key, value)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        finally: conn.close()

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _proxy
    def log_message(self, *_args): pass

class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

__all__ = ["BridgeHandler", "BridgeServer"]
