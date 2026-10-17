"""Authenticated TLS bridge for Basalt's private native HTTP listener."""
from __future__ import annotations

import argparse
import hmac
import http.client
import ipaddress
import os
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from pathlib import Path
from .broker_security import read_api_keys

_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length"}
_STRIP = {"authorization", "x-api-key", "x-basalt-bridge-token"}


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    max_body_bytes = 8 * 1024 * 1024
    upstream_host = "127.0.0.1"
    upstream_port = 0
    upstream_path = ""
    bridge_token = ""
    bridge_tokens = ()
    upstream_token = ""

    def _authorized(self) -> bool:
        supplied = self.headers.get("X-Basalt-Bridge-Token", "")
        authorization = self.headers.get("Authorization", "")
        bearer = authorization[7:].strip() if authorization.startswith("Bearer ") else ""
        matches = [hmac.compare_digest(supplied, token) | hmac.compare_digest(bearer, token)
                   for token in (self.bridge_tokens or ((self.bridge_token,) if self.bridge_token else ()))]
        return any(matches)

    def _reject(self, status: int, message: str) -> None:
        body = message.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _proxy(self) -> None:
        if not self._authorized():
            return self._reject(401, "invalid bridge credentials")
        if self.command not in {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}:
            return self._reject(405, "method not allowed")
        if self.headers.get_all("Transfer-Encoding", []):
            return self._reject(400, "transfer encoding is unsupported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != len(set(lengths)) or len(lengths) > 1:
            return self._reject(400, "duplicate content length")
        try:
            size = int(lengths[0]) if lengths else 0
        except ValueError:
            return self._reject(400, "invalid content length")
        if size < 0:
            return self._reject(400, "invalid content length")
        if size > self.max_body_bytes:
            return self._reject(413, "request body too large")
        body = self.rfile.read(size) if size else None
        nominated = {x.strip().lower() for value in self.headers.get_all('Connection', []) for x in value.split(',')}
        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in _HOP and k.lower() not in _STRIP and k.lower() not in nominated
        }
        headers["Host"] = f"{self.upstream_host}:{self.upstream_port}"
        if self.upstream_token:
            headers["Authorization"] = f"Bearer {self.upstream_token}"
        conn = http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=300)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in _HOP:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            while True:
                chunk = response.read1(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _proxy

    def log_message(self, *_args):
        pass


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _loopback(url: str):
    parsed = urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("--upstream-url must be an http loopback URL")
    try:
        address = ipaddress.ip_address(parsed.hostname)
        allowed = address.is_loopback
    except ValueError:
        allowed = parsed.hostname.lower() == "localhost"
    if not allowed:
        raise ValueError("--upstream-url must target loopback")
    if parsed.query or parsed.fragment:
        raise ValueError("--upstream-url must not contain query or fragment")
    return parsed.hostname, parsed.port or 80


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the authenticated Basalt TLS bridge")
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--auth-token-file", required=True)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--upstream-token-file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--max-body-bytes", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args(argv)
    try:
        host, port = _loopback(args.upstream_url)
        if args.max_body_bytes < 0 or args.port < 0 or args.port > 65535:
            raise ValueError("invalid size or port")
        tokens = read_api_keys(Path(args.auth_token_file))
        if not tokens:
            raise ValueError("auth token file is empty")
        upstream_token = ""
        if args.upstream_token_file:
            with open(args.upstream_token_file, encoding="utf-8") as token_file:
                upstream_token = token_file.read().strip()
            if not upstream_token:
                raise ValueError("upstream token file is empty")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert, args.key)
        server = BridgeServer((args.host, args.port), BridgeHandler)
        BridgeHandler.upstream_host, BridgeHandler.upstream_port = host, port
        BridgeHandler.bridge_tokens = tuple(tokens)
        BridgeHandler.bridge_token = ""
        BridgeHandler.upstream_token = upstream_token
        BridgeHandler.max_body_bytes = args.max_body_bytes
        server.socket = context.wrap_socket(server.socket, server_side=True)
    except (OSError, ValueError, ssl.SSLError) as exc:
        parser.error(str(exc))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    main()


__all__ = ["BridgeHandler", "BridgeServer", "main"]
