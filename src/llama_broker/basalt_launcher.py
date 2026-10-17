"""Safely supervise a native Basalt child and its authenticated TLS bridge."""
from __future__ import annotations

import argparse
import http.client
import os
import secrets
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _read_secret(path: str) -> str:
    value = Path(path).read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"secret file is empty: {path}")
    return value


def _reserve_port() -> int:
    for _ in range(8):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
        finally:
            sock.close()
    raise RuntimeError("could not reserve native port")


def _validate(args):
    if args.port < 0 or args.port > 65535 or args.bridge_port < 0 or args.bridge_port > 65535:
        raise ValueError("invalid port")
    _read_secret(args.auth_token_file)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(args.cert, args.key)
    return context


def _wait_native(port: int, token: str, child=None, cancelled=lambda: False) -> None:
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if cancelled():
            raise RuntimeError('startup cancelled')
        if child is not None and child.poll() is not None:
            raise RuntimeError('native Basalt child exited during startup')
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health", headers={"Authorization": f"Bearer {token}", "X-API-Key": token})
            response = conn.getresponse()
            response.read()
            conn.close()
            if response.status in (200, 503):
                return
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(.1)
    raise RuntimeError("native Basalt child did not become reachable")


def _stop(proc, sig=signal.SIGTERM):
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--basalt-binary', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--cert', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--auth-token-file', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--bridge-port', type=int, required=True)
    args = parser.parse_args(argv)

    child = bridge = None
    stopping = False
    def stop(signum, _frame):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        _stop(bridge, signum)
        _stop(child, signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        _validate(args)
        public_token = _read_secret(args.auth_token_file)
        native_port = args.port or _reserve_port()
        # Keep short-lived credentials in tmpfs, and remove them on every exit.
        tmp_root = '/dev/shm' if os.path.isdir('/dev/shm') else os.environ.get('XDG_RUNTIME_DIR', tempfile.gettempdir())
        with tempfile.TemporaryDirectory(dir=tmp_root) as secrets_dir:
            upstream_path = Path(secrets_dir) / 'native-token'
            upstream_token = secrets.token_urlsafe(32)
            upstream_path.write_text(upstream_token, encoding='utf-8')
            os.chmod(upstream_path, 0o600)
            child = subprocess.Popen([
                args.basalt_binary, 'serve', '--config', args.config,
                '--host', '127.0.0.1', '--port', str(native_port), '--api-key', upstream_token,
            ], start_new_session=True)
            _wait_native(native_port, upstream_token, child, lambda: stopping)
            bridge = subprocess.Popen([
                sys.executable, '-m', 'llama_broker.basalt_bridge', '--cert', args.cert,
                '--key', args.key, '--auth-token-file', args.auth_token_file,
                '--upstream-token-file', str(upstream_path), '--upstream-url', f'http://127.0.0.1:{native_port}',
                '--host', args.host, '--port', str(args.bridge_port),
            ], start_new_session=True)
            def stop(signum, _frame):
                nonlocal stopping
                if stopping:
                    return
                stopping = True
                _stop(bridge, signum)
                _stop(child, signum)
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            while True:
                if stopping:
                    break
                if child.poll() is not None or bridge.poll() is not None:
                    break
                time.sleep(.05)
            if not stopping:
                _stop(bridge)
                _stop(child)
            for proc in (bridge, child):
                if proc is not None:
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        _stop(proc, signal.SIGKILL)
                        proc.wait(timeout=5)
            if stopping:
                return 0
            code = child.returncode if child and child.returncode is not None else bridge.returncode if bridge else 1
            return 128 + (-code) if code < 0 else (code or 1)
    except (OSError, ValueError, RuntimeError, ssl.SSLError) as exc:
        _stop(bridge)
        _stop(child)
        for proc in (bridge, child):
            if proc is not None:
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _stop(proc, signal.SIGKILL)
                    proc.wait(timeout=5)
        print(f"basalt launcher: {exc}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
