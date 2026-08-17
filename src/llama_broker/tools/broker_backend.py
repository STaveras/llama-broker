#!/usr/bin/env python3
"""Backend communication utilities for llama-broker.

This module provides a thin HTTP client to llama-server endpoints for
fetching model metadata and other backend information.
"""

import http.client
import json
import ssl
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any, Mapping


def make_connection(
    base_url: str,
    *,
    timeout: float,
    ca_file: str | Path | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> http.client.HTTPConnection:
    """Create an HTTP(S) connection matching *base_url*."""

    parsed = urlsplit(base_url)
    host = parsed.hostname or "127.0.0.1"
    if parsed.scheme == "https":
        context = ssl_context or ssl.create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH,
            cafile=str(ca_file) if ca_file else None,
        )
        return http.client.HTTPSConnection(
            host,
            parsed.port or 443,
            timeout=timeout,
            context=context,
        )
    if parsed.scheme not in {"", "http"}:
        raise ValueError(f"unsupported backend URL scheme: {parsed.scheme!r}")
    return http.client.HTTPConnection(host, parsed.port or 80, timeout=timeout)


class BackendClient:
    """Thin client to llama-server endpoints."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 1800.0,
        *,
        ca_file: str | Path | None = None,
        ssl_context: ssl.SSLContext | None = None,
        default_headers: Mapping[str, str] | None = None,
    ):
        parsed = urlsplit(base_url)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._base_url = base_url
        self._ca_file = ca_file
        self._ssl_context = ssl_context
        self._default_headers = dict(default_headers or {})
        self._timeout = timeout
        self._closed = False
        self._conn = self._make_connection()

    def _make_connection(self) -> http.client.HTTPConnection:
        return make_connection(
            self._base_url,
            timeout=self._timeout,
            ca_file=self._ca_file,
            ssl_context=self._ssl_context,
        )

    def _headers(self, headers: Mapping[str, str] | None = None) -> dict[str, str]:
        merged = dict(self._default_headers)
        if headers:
            merged.update(headers)
        merged.setdefault("Connection", "close")
        return merged

    def _get_conn(self) -> http.client.HTTPConnection:
        if self._closed:
            raise RuntimeError("BackendClient is closed")
        if self._conn is None:
            self._conn = self._make_connection()
        return self._conn

    def _request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        conn = self._get_conn()
        try:
            conn.request(method, path, headers=self._headers(headers))
            response = conn.getresponse()
            body = response.read().decode("utf-8")
            if response.status < 200 or response.status >= 300:
                raise http.client.HTTPException(
                    f"Backend error: {response.status} {response.reason}"
                )
            return json.loads(body)
        except (http.client.RemoteDisconnected, ConnectionResetError, OSError):
            self._conn = None
            conn = self._get_conn()
            conn.request(method, path, headers=self._headers(headers))
            response = conn.getresponse()
            body = response.read().decode("utf-8")
            if response.status < 200 or response.status >= 300:
                raise http.client.HTTPException(
                    f"Backend error: {response.status} {response.reason}"
                )
            return json.loads(body)
        finally:
            self._conn = None

    def list_models(self, *, headers: Mapping[str, str] | None = None) -> dict[str, Any]:
        return self._request("GET", "/v1/models", headers=headers)

    def model_info(
        self,
        model_name: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._request("GET", f"/v1/models/{model_name}", headers=headers)

    def health(self, *, headers: Mapping[str, str] | None = None) -> dict[str, Any]:
        return self._request("GET", "/health", headers=headers)

    def get_raw(
        self,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, str, bytes]:
        """Raw GET returning (status, content_type, body_bytes)."""
        conn = self._get_conn()
        try:
            conn.request("GET", path, headers=self._headers(headers))
            response = conn.getresponse()
            content_type = response.getheader("Content-Type", "application/json")
            body = response.read()
            return response.status, content_type, body
        except (http.client.RemoteDisconnected, ConnectionResetError, OSError):
            self._conn = None
            conn = self._get_conn()
            conn.request("GET", path, headers=self._headers(headers))
            response = conn.getresponse()
            content_type = response.getheader("Content-Type", "application/json")
            body = response.read()
            return response.status, content_type, body
        finally:
            self._conn = None

    def close(self) -> None:
        if not self._closed:
            if self._conn:
                self._conn.close()
                self._conn = None
            self._closed = True

    def __enter__(self) -> "BackendClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
