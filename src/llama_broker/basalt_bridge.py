"""Authenticated TLS bridge for Basalt's private native HTTP listener."""
from __future__ import annotations

import argparse
import hmac
import http.client
import ipaddress
import os
import ssl
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from pathlib import Path
from .broker_security import read_api_keys
from jsonschema import Draft202012Validator, SchemaError

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
    upstream_timeout = 900

    def setup(self):
        self.request.settimeout(self.server.client_timeout)
        self._upstream = None
        self._upstream_socket = None
        self._headers_sent = False
        self._deadline_timer = threading.Timer(self.upstream_timeout, self._expire)
        self._deadline_timer.daemon = True
        super().setup()
        self._deadline_timer.start()

    def _expire(self):
        # Shutdown interrupts blocked reads/writes, including a trickling peer.
        for sock in (self.connection, self._upstream_socket, getattr(self._upstream, 'sock', None)):
            if sock is not None:
                try: sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass

    @staticmethod
    def validate_upstream_timeout(value: int) -> int:
        if not 1 <= value <= 900:
            raise ValueError("--upstream-timeout must be between 1 and 900 seconds")
        return value

    def _authorized(self) -> bool:
        if any(len(self.headers.get_all(name, [])) > 1 for name in ('Authorization', 'X-Basalt-Bridge-Token')):
            return False
        supplied = self.headers.get("X-Basalt-Bridge-Token", "")
        authorization = self.headers.get("Authorization", "")
        bearer = authorization[7:].strip() if authorization.startswith("Bearer ") else ""
        matches = [hmac.compare_digest(supplied.encode(), token.encode()) | hmac.compare_digest(bearer.encode(), token.encode())
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
        try:
            self._forward()
        except (OSError, http.client.HTTPException):
            # Never append a second HTTP response or complete a partial stream.
            self.close_connection = True
            if not self._headers_sent:
                try: self._reject(502, 'upstream transport failed')
                except OSError: pass

    def _forward(self) -> None:
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
        if size and len(body) != size:
            return self._reject(400, 'incomplete request body')
        if body and self.headers.get_content_type() == "application/json":
            try:
                payload = json.loads(body)
                requested = payload.get("response_format", {})
                schema = requested.get("json_schema", {}).get("schema") if isinstance(requested, dict) else None
                if schema is not None:
                    Draft202012Validator.check_schema(schema)
            except (ValueError, AttributeError, TypeError, SchemaError):
                return self._reject(400, 'invalid response schema')
        nominated = {x.strip().lower() for value in self.headers.get_all('Connection', []) for x in value.split(',')}
        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in _HOP and k.lower() not in _STRIP and k.lower() not in nominated
        }
        headers["Host"] = f"{self.upstream_host}:{self.upstream_port}"
        if self.upstream_token:
            headers["Authorization"] = f"Bearer {self.upstream_token}"
        conn = http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=self.upstream_timeout)
        self._upstream = conn
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            self._upstream_socket = conn.sock
            response = conn.getresponse()
            lengths = response.headers.get_all('Content-Length', [])
            encodings = response.headers.get_all('Transfer-Encoding', [])
            if len(lengths) > 1 or (lengths and encodings) or (encodings and encodings != ['chunked']):
                return self._reject(502, 'invalid upstream framing')
            if lengths and (not lengths[0].isascii() or not lengths[0].isdecimal()):
                return self._reject(502, 'invalid upstream length')
            nominated_response = {x.strip().lower() for value in response.headers.get_all('Connection', []) for x in value.split(',')}
            has_body = response.status not in (204, 304) and response.status >= 200
            chunked = has_body and not lengths
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in _HOP | _STRIP | nominated_response:
                    self.send_header(key, value)
            if lengths and has_body:
                self.send_header('Content-Length', lengths[0])
            elif chunked:
                self.send_header('Transfer-Encoding', 'chunked')
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            self._headers_sent = True
            while has_body:
                chunk = response.read1(8192)
                if not chunk:
                    if response.length:
                        raise http.client.IncompleteRead(b'', response.length)
                    break
                if chunked:
                    self.wfile.write(f'{len(chunk):x}\r\n'.encode())
                self.wfile.write(chunk)
                if chunked: self.wfile.write(b'\r\n')
                self.wfile.flush()
            if chunked:
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
        finally:
            conn.close()
            self._upstream = None
            self._upstream_socket = None

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _proxy

    def finish(self):
        """Finish TLS shutdown within a bounded interval, including silent peers."""
        self._deadline_timer.cancel()
        try:
            if not self.wfile.closed:
                self.wfile.flush()
        except OSError:
            pass
        try:
            self.wfile.close()
        except OSError:
            pass
        try:
            self.rfile.close()
        except OSError:
            pass
        connection = self.connection
        try:
            if isinstance(connection, ssl.SSLSocket):
                connection.settimeout(min(1, self.server.client_timeout))
                connection.unwrap().close()
            else:
                connection.close()
        except (BrokenPipeError, ConnectionResetError, ssl.SSLError, OSError):
            connection.close()

    def log_message(self, *_args):
        pass


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    client_timeout = 10
    max_connections = 32
    ssl_context = None
    _slots = None

    def process_request(self, request, client_address):
        if self._slots is None:
            self._slots = threading.BoundedSemaphore(self.max_connections)
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            request.settimeout(self.client_timeout)
            if self.ssl_context is not None:
                request = self.ssl_context.wrap_socket(request, server_side=True)
            super().process_request_thread(request, client_address)
        except OSError:
            self.shutdown_request(request)
        finally:
            self._slots.release()


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
    if parsed.query or parsed.fragment or parsed.path not in ('', '/'):
        raise ValueError("--upstream-url must not contain a path, query or fragment")
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
    parser.add_argument("--upstream-timeout", type=int, default=900)
    parser.add_argument('--client-timeout', type=int, default=10)
    parser.add_argument('--max-connections', type=int, default=32)
    args = parser.parse_args(argv)
    try:
        host, port = _loopback(args.upstream_url)
        if args.max_body_bytes < 0 or args.port < 0 or args.port > 65535:
            raise ValueError("invalid size or port")
        BridgeHandler.validate_upstream_timeout(args.upstream_timeout)
        if not 1 <= args.client_timeout <= 900 or not 1 <= args.max_connections <= 256:
            raise ValueError('invalid client timeout or connection bound')
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
        class Handler(BridgeHandler): pass
        Handler.upstream_host, Handler.upstream_port = host, port
        Handler.bridge_tokens = tuple(tokens)
        Handler.bridge_token = ""
        Handler.upstream_token = upstream_token
        Handler.max_body_bytes = args.max_body_bytes
        Handler.upstream_timeout = args.upstream_timeout
        server = BridgeServer((args.host, args.port), Handler)
        server.client_timeout = args.client_timeout
        server.max_connections = args.max_connections
        server.ssl_context = context
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
