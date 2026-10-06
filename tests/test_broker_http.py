from __future__ import annotations

import http.client
import json
import socket
import ssl
import struct
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from llama_broker.broker_profiles import ProfileRegistry
from llama_broker.backend_controller import (
    BackendControllerError,
    BackendExit,
    LifecycleCancelledError,
)
from llama_broker.crash_loop import CrashLoopGuard
import llama_broker.daemon as llama_broker
from llama_broker.broker_security import SecurityConfig, write_api_keys
from llama_broker.profile_config import sha256_file
from llama_broker import (
    BrokerApplication,
    BrokerDrainTimeoutError,
    BrokerQuiescingError,
    BrokerServer,
    BrokerTransitionTimeoutError,
    ProfileCoordinator,
    apply_reasoning_policy,
    enforce_output_limit,
    reasoning_policy_metadata,
    route_payload,
)


class FakeController:
    def __init__(self, profile: str | None = "main") -> None:
        self.profile = profile
        self.transitions: list[str] = []
        self.stops = 0
        self.generation = 1 if profile else 0
        self._lock = threading.RLock()

    def state(self) -> dict[str, object]:
        with self._lock:
            return {
                "generation": self.generation,
                "profile": self.profile,
                "phase": "active" if self.profile else "inactive",
            }

    def switch(self, profile: str) -> dict[str, object]:
        with self._lock:
            self.transitions.append(profile)
            self.profile = profile
            self.generation += 1
            return self.state()

    def restart(self, profile: str) -> dict[str, object]:
        with self._lock:
            self.transitions.append(f"restart:{profile}")
            self.profile = profile
            return self.state()

    def stop_all(self) -> dict[str, object]:
        with self._lock:
            self.stops += 1
            self.profile = None
            self.generation += 1
            return self.state()

    def record_recovery_failure(
        self, profile: str, *, attempts: int, error: str
    ) -> dict[str, object]:
        with self._lock:
            self.profile = None
            self.generation += 1
            return {
                "generation": self.generation,
                "profile": None,
                "phase": "stale",
                "stale_profile": profile,
                "recovery": {
                    "attempts": attempts,
                    "error": error,
                    "action": "broker_watchdog_restart",
                },
            }


class BackendHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.seen_bodies.append(body)  # type: ignore[attr-defined]
        self.server.seen_headers.append(dict(self.headers))  # type: ignore[attr-defined]
        if self.headers.get("Content-Type", "").startswith("multipart/form-data"):
            payload = None
        else:
            payload = json.loads(body)
            self.server.seen_payloads.append(payload)  # type: ignore[attr-defined]
        if payload and payload.get("_test_hold_stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'data: {"token":"held"}\n\n')
            self.wfile.flush()
            with self.server.hold_stream_lock:  # type: ignore[attr-defined]
                self.server.hold_stream_count += 1  # type: ignore[attr-defined]
                if self.server.hold_stream_count == 2:  # type: ignore[attr-defined]
                    self.server.hold_streams_entered.set()  # type: ignore[attr-defined]
            self.server.hold_stream_release.wait(timeout=5)  # type: ignore[attr-defined]
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True
            return
        if payload and payload.get("_test_reset_before_headers"):
            self.connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            self.connection.close()
            self.close_connection = True
            return
        if payload and payload.get("_test_reset_stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'data: {"token":"partial"}\n\n')
            self.wfile.flush()
            self.server.reset_stream_release.wait(timeout=5)  # type: ignore[attr-defined]
            # SO_LINGER 0 turns close() into a TCP RST, like a crashed backend.
            self.connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            self.connection.close()
            self.close_connection = True
            return

        response_payload = {"ok": True}
        if payload is not None:
            response_payload["model"] = payload["model"]
        response = json.dumps(response_payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


class BrokerHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        binary = root / "llama-server"
        binary.write_bytes(b"binary")
        binary.chmod(0o755)
        main_model = root / "main.gguf"
        main_model.write_bytes(b"main")
        vision_model = root / "vision.gguf"
        vision_model.write_bytes(b"vision")
        mmproj = root / "mmproj.gguf"
        mmproj.write_bytes(b"projector")
        common = {
            "server_binary": str(binary),
            "server_sha256": sha256_file(binary),
            "context_size": 65536,
            "parallel": 1,
            "batch_size": 256,
            "ubatch_size": 64,
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo4",
            "flash_attention": "on",
            "gpu_layers": "all",
            "startup_timeout": 10,
            "safety": {
                "service": {
                    "type": "exec",
                    "memory_max_mib": 20_480,
                    "memory_swap_max_mib": 1_024,
                    "tasks_max": 512,
                    "kill_signal": "SIGINT",
                    "timeout_stop_sec": 20,
                },
                "admission": {
                    "min_ram_available_mib": 8_192,
                    "max_swap_used_mib": 1_024,
                    "max_swap_growth_mib": 128,
                    "min_total_vram_mib": 12_000,
                    "min_free_vram_before_start_mib": 8_192,
                    "driver_vram_reserve_mib": 512,
                    "vision_vram_reserve_mib": 1_536,
                    "safety_vram_reserve_mib": 1_024,
                },
            },
            "environment": {
                "GGML_CUDA_PDL": "0",
            },
            "extra_args": [],
        }
        manifest = {
            "schema_version": 1,
            "profiles": [
                {
                    **common,
                    "id": "main",
                    "alias": "main-27b",
                    "aliases": ["main", "main-27b"],
                    "display_name": "main",
                    "role": "persistent",
                    "capabilities": ["text"],
                    "model_path": str(main_model),
                    "model_sha256": sha256_file(main_model),
                },
                {
                    **common,
                    "id": "vision",
                    "alias": "qwen3.5-9b-vision",
                    "aliases": ["qwen3.5-9b-vision"],
                    "display_name": "vision",
                    "role": "transient",
                    "capabilities": ["text", "vision"],
                    "restore_profile": "main",
                    "model_path": str(vision_model),
                    "model_sha256": sha256_file(vision_model),
                    "mmproj_path": str(mmproj),
                    "mmproj_sha256": sha256_file(mmproj),
                    "context_size": 32768,
                    "parallel": 2,
                    "gpu_layers": "auto",
                    "safety": {
                        "service": {
                            "type": "exec",
                            "memory_max_mib": 16_384,
                            "memory_swap_max_mib": 512,
                            "tasks_max": 384,
                            "kill_signal": "SIGINT",
                            "timeout_stop_sec": 20,
                        },
                        "admission": {
                            "min_ram_available_mib": 8_192,
                            "max_swap_used_mib": 1_024,
                            "max_swap_growth_mib": 128,
                            "min_total_vram_mib": 12_000,
                            "min_free_vram_before_start_mib": 8_192,
                            "driver_vram_reserve_mib": 512,
                            "vision_vram_reserve_mib": 0,
                            "safety_vram_reserve_mib": 1_536,
                        },
                    },
                },
            ],
        }
        self.profiles_path = root / "profiles.json"
        self.profiles_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.registry = ProfileRegistry.from_path(self.profiles_path)
        self.auth_key = "native-key"
        self.auth_key_path = root / "api-keys"
        write_api_keys(self.auth_key_path, [self.auth_key])

        self.backend = ThreadingHTTPServer(("127.0.0.1", 0), BackendHandler)
        self.backend.seen_payloads = []  # type: ignore[attr-defined]
        self.backend.seen_bodies = []  # type: ignore[attr-defined]
        self.backend.seen_headers = []  # type: ignore[attr-defined]
        self.backend.hold_stream_lock = threading.Lock()  # type: ignore[attr-defined]
        self.backend.hold_stream_count = 0  # type: ignore[attr-defined]
        self.backend.hold_streams_entered = threading.Event()  # type: ignore[attr-defined]
        self.backend.hold_stream_release = threading.Event()  # type: ignore[attr-defined]
        self.backend.reset_stream_release = threading.Event()  # type: ignore[attr-defined]
        self.backend_thread = threading.Thread(target=self.backend.serve_forever, daemon=True)
        self.backend_thread.start()

        self.controller = FakeController("main")
        app = BrokerApplication(
            registry=self.registry,
            controller=self.controller,
            backend_host="127.0.0.1",
            backend_port=self.backend.server_port,
            backend_timeout=10,
            max_body_bytes=1_000_000,
            security=SecurityConfig(api_key_file=self.auth_key_path),
            backend_alias_ready=self._active_alias_ready,
            profiles_path=self.profiles_path,
        )
        self.broker = BrokerServer(("127.0.0.1", 0), application=app, max_workers=4)
        self.broker_thread = threading.Thread(target=self.broker.serve_forever, daemon=True)
        self.broker_thread.start()

    def _active_alias_ready(self, alias: str) -> bool:
        profile_id = self.controller.profile
        return (
            isinstance(profile_id, str)
            and self.registry.get(profile_id).alias == alias
        )

    def tearDown(self) -> None:
        self.backend.hold_stream_release.set()  # type: ignore[attr-defined]
        self.backend.reset_stream_release.set()  # type: ignore[attr-defined]
        self.broker.shutdown()
        self.broker.server_close()
        self.backend.shutdown()
        self.backend.server_close()
        self.tempdir.cleanup()

    def _post(
        self,
        payload: dict[str, object],
        path: str = "/v1/chat/completions",
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, object]]:
        connection = http.client.HTTPConnection("127.0.0.1", self.broker.server_port, timeout=5)
        body = json.dumps(payload)
        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_key}",
        }
        request_headers.update(headers or {})
        connection.request(
            "POST",
            path,
            body=body,
            headers=request_headers,
        )
        response = connection.getresponse()
        decoded = json.loads(response.read())
        connection.close()
        return response.status, decoded

    def _post_raw(
        self,
        body: bytes,
        path: str = "/v1/audio/transcriptions",
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, object], dict[str, str]]:
        connection = http.client.HTTPConnection("127.0.0.1", self.broker.server_port, timeout=5)
        request_headers = {"Authorization": f"Bearer {self.auth_key}"}
        request_headers.update(headers or {})
        connection.request("POST", path, body=body, headers=request_headers)
        response = connection.getresponse()
        decoded = json.loads(response.read())
        response_headers = dict(response.getheaders())
        connection.close()
        return response.status, decoded, response_headers

    def _post_raw_wire(
        self,
        body: bytes,
        *,
        content_length: str | None,
        headers: dict[str, str] | None = None,
        shutdown_write: bool = False,
    ) -> tuple[int, dict[str, object], dict[str, str]]:
        connection = http.client.HTTPConnection("127.0.0.1", self.broker.server_port, timeout=5)
        request_headers = {
            "Content-Type": "multipart/form-data; boundary=audio-boundary",
            "Authorization": f"Bearer {self.auth_key}",
            "Connection": "close",
        }
        if content_length is not None:
            request_headers["Content-Length"] = content_length
        request_headers.update(headers or {})
        connection.putrequest("POST", "/v1/audio/transcriptions")
        for key, value in request_headers.items():
            connection.putheader(key, value)
        connection.endheaders()
        if body:
            connection.send(body)
        if shutdown_write:
            assert connection.sock is not None
            connection.sock.shutdown(socket.SHUT_WR)
        response = connection.getresponse()
        decoded = json.loads(response.read())
        response_headers = dict(response.getheaders())
        connection.close()
        return response.status, decoded, response_headers

    def test_audio_transcription_forwards_multipart_bytes_and_headers(self) -> None:
        content_type = "multipart/form-data; boundary=audio-boundary"
        body = (
            b'--audio-boundary\r\n'
            b'Content-Disposition: form-data; name="model"\r\n\r\n'
            b"unknown-form-model\r\n"
            b'--audio-boundary\r\n'
            b'Content-Disposition: form-data; name="file"; filename="voice.wav"\r\n'
            b"Content-Type: audio/wav\r\n\r\n"
            b"RIFF\x00\xff\x10\x80binary-audio\r\n"
            b"--audio-boundary--\r\n"
        )
        request_id = "audio-request-123"

        status, payload, response_headers = self._post_raw(
            body,
            headers={
                "Content-Type": content_type,
                "Authorization": f"Bearer {self.auth_key}",
                "X-Request-ID": request_id,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["ok"], True)
        self.assertEqual(self.backend.seen_bodies[-1], body)  # type: ignore[attr-defined]
        seen = self.backend.seen_headers[-1]  # type: ignore[attr-defined]
        self.assertEqual(seen["Content-Type"], content_type)
        self.assertEqual(seen["Authorization"], f"Bearer {self.auth_key}")
        self.assertEqual(seen["X-Request-ID"], request_id)
        self.assertEqual(response_headers["X-Llama-Broker-Profile"], "main")
        self.assertEqual(response_headers["X-Request-ID"], request_id)

    def test_audio_transcription_rejects_non_multipart_content_type(self) -> None:
        request_id = "audio-invalid-content-type"

        status, payload, response_headers = self._post_raw(
            b'{"model":"main"}',
            headers={
                "Content-Type": "application/json",
                "X-Request-ID": request_id,
            },
        )

        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")  # type: ignore[index]
        self.assertEqual(self.backend.seen_bodies, [])  # type: ignore[attr-defined]
        self.assertEqual(response_headers["X-Request-ID"], request_id)

    def test_audio_transcription_rejects_oversized_body_before_backend_access(self) -> None:
        self.broker.application.max_body_bytes = 4  # type: ignore[attr-defined]
        request_id = "audio-oversized"

        status, payload, response_headers = self._post_raw(
            b"12345",
            headers={
                "Content-Type": "multipart/form-data; boundary=audio-boundary",
                "X-Request-ID": request_id,
            },
        )

        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")  # type: ignore[index]
        self.assertIn("exceeds broker limit 4 bytes", payload["error"]["message"])  # type: ignore[operator,index]
        self.assertEqual(self.backend.seen_bodies, [])  # type: ignore[attr-defined]
        self.assertEqual(response_headers["X-Request-ID"], request_id)

    def test_audio_transcription_rejects_invalid_bearer_before_backend_access(self) -> None:
        request_id = "audio-invalid-bearer"

        status, payload, response_headers = self._post_raw(
            b"binary-audio-that-must-not-be-read",
            headers={
                "Content-Type": "multipart/form-data; boundary=audio-boundary",
                "Authorization": "Bearer wrong-key",
                "X-Request-ID": request_id,
            },
        )

        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["type"], "authentication_error")  # type: ignore[index]
        self.assertEqual(self.backend.seen_bodies, [])  # type: ignore[attr-defined]
        self.assertEqual(response_headers["X-Request-ID"], request_id)

    def test_audio_transcription_rejects_missing_bearer_before_backend_access(self) -> None:
        request_id = "audio-missing-bearer"

        status, payload, response_headers = self._post_raw(
            b"binary-audio-that-must-not-be-read",
            headers={
                "Content-Type": "multipart/form-data; boundary=audio-boundary",
                "Authorization": "",
                "X-Request-ID": request_id,
            },
        )

        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["type"], "authentication_error")  # type: ignore[index]
        self.assertEqual(self.backend.seen_bodies, [])  # type: ignore[attr-defined]
        self.assertEqual(response_headers["X-Request-ID"], request_id)

    def test_audio_transcription_rejects_missing_invalid_and_incomplete_lengths(self) -> None:
        cases = (
            ("missing", b"", None, False, "Content-Length is required"),
            ("invalid", b"", "not-a-number", False, "invalid Content-Length"),
            ("incomplete", b"short", "10", True, "request body is incomplete"),
        )

        for label, body, content_length, shutdown_write, expected_message in cases:
            with self.subTest(case=label):
                request_id = f"audio-{label}-length"
                status, payload, response_headers = self._post_raw_wire(
                    body,
                    content_length=content_length,
                    headers={"X-Request-ID": request_id},
                    shutdown_write=shutdown_write,
                )

                self.assertEqual(status, 400)
                self.assertEqual(
                    payload["error"]["type"], "invalid_request_error"  # type: ignore[index]
                )
                self.assertIn(expected_message, payload["error"]["message"])  # type: ignore[operator,index]
                self.assertEqual(response_headers["X-Request-ID"], request_id)

        self.assertEqual(self.backend.seen_bodies, [])  # type: ignore[attr-defined]

    def test_audio_transcription_does_not_log_uploaded_media(self) -> None:
        marker = "audio-media-must-not-appear-in-logs"
        body = (
            b'--audio-boundary\r\nContent-Disposition: form-data; name="file"\r\n\r\n'
            + marker.encode("ascii")
            + b"\r\n--audio-boundary--\r\n"
        )

        with patch.object(llama_broker, "_log") as log:
            status, _, _ = self._post_raw(
                body,
                headers={
                    "Content-Type": "multipart/form-data; boundary=audio-boundary",
                    "X-Request-ID": "audio-log-safety",
                },
            )

        self.assertEqual(status, 200)
        messages = "\n".join(
            str(call.args[0]) for call in log.call_args_list if call.args
        )
        self.assertNotIn(marker, messages)

    def test_native_backend_credentials_are_forwarded_unchanged(self) -> None:
        status, _ = self._post(
            {"model": "main", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "Authorization": "Bearer native-key",
                "X-Api-Key": "native-key",
            },
        )

        self.assertEqual(status, 200)
        seen = self.backend.seen_headers[-1]  # type: ignore[attr-defined]
        self.assertEqual(seen["Authorization"], "Bearer native-key")
        self.assertEqual(seen["X-Api-Key"], "native-key")

    def test_log_redaction_covers_non_bearer_authorization_headers(self) -> None:
        message = llama_broker._redact_log_message(
            'Authorization: Basic dXNlcjpwYXNz, "authorization":"Custom secret-value"'
        )

        self.assertNotIn("dXNlcjpwYXNz", message)
        self.assertNotIn("secret-value", message)
        self.assertIn("Authorization: [REDACTED]", message)

    def test_credential_rotation_restarts_active_profile_and_retains_overlap(self) -> None:
        key_path = Path(self.tempdir.name) / "api-keys"
        write_api_keys(key_path, ["old-native-key"])
        application = self.broker.application  # type: ignore[attr-defined]
        application.security = SecurityConfig(api_key_file=key_path)

        result = application.rotate_credentials()

        self.assertNotEqual(result["api_key"], "old-native-key")
        self.assertEqual(result["previous_keys_retained"], True)
        self.assertEqual(application.coordinator.active_profile, "main")
        self.assertEqual(self.controller.transitions[-1], "restart:main")
        self.assertEqual(
            set(key_path.read_text(encoding="utf-8").split()),
            {"old-native-key", result["api_key"]},
        )

    def test_broker_server_wraps_listener_when_tls_context_is_configured(self) -> None:
        class Context:
            def __init__(self) -> None:
                self.server_side = False

            def wrap_socket(self, sock: object, *, server_side: bool) -> object:
                self.server_side = server_side
                return sock

        class Application:
            def shutdown(self) -> None:
                return

        context = Context()
        server = BrokerServer(
            ("127.0.0.1", 0),
            application=Application(),  # type: ignore[arg-type]
            max_workers=1,
            ssl_context=context,  # type: ignore[arg-type]
        )
        try:
            self.assertTrue(context.server_side)
        finally:
            server.server_close()

    def _get(self, path: str) -> tuple[int, dict[str, object]]:
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.broker.server_port, timeout=5
        )
        connection.request(
            "GET",
            path,
            headers={
                "Connection": "close",
                "Authorization": f"Bearer {self.auth_key}",
            },
        )
        response = connection.getresponse()
        decoded = json.loads(response.read())
        connection.close()
        return response.status, decoded

    def _replace_broker(
        self,
        *,
        max_workers: int,
        control_workers: int,
        drain_timeout: float,
        transition_timeout: float,
        stop_timeout: float,
        start_operation_worker: bool = True,
        initial_profile: str | None = "main",
    ) -> None:
        self.broker.shutdown()
        self.broker.server_close()
        self.broker_thread.join(timeout=1)
        self.controller = FakeController(initial_profile)
        application = BrokerApplication(
            registry=self.registry,
            controller=self.controller,
            backend_host="127.0.0.1",
            backend_port=self.backend.server_port,
            backend_timeout=10,
            max_body_bytes=1_000_000,
            security=SecurityConfig(api_key_file=self.auth_key_path),
            backend_ready=lambda: True,
            backend_alias_ready=self._active_alias_ready,
            profiles_path=self.profiles_path,
            drain_timeout=drain_timeout,
            transition_timeout=transition_timeout,
            stop_timeout=stop_timeout,
            start_operation_worker=start_operation_worker,
        )
        self.broker = BrokerServer(
            ("127.0.0.1", 0),
            application=application,
            max_workers=max_workers,
            control_workers=control_workers,
        )
        self.broker_thread = threading.Thread(
            target=self.broker.serve_forever, daemon=True
        )
        self.broker_thread.start()

    def test_llamacpp_reasoning_policy_is_forwarded_to_backend(self) -> None:
        status, _ = self._post(
            {
                "model": "main",
                "messages": [{"role": "user", "content": "think"}],
                "reasoning": {"effort": "high"},
            }
        )

        self.assertEqual(status, 200)
        forwarded = self.backend.seen_payloads[-1]  # type: ignore[attr-defined]
        self.assertEqual(
            forwarded["chat_template_kwargs"], {"enable_thinking": True}
        )
        self.assertEqual(forwarded["reasoning_effort"], "high")
        self.assertEqual(forwarded["thinking_budget_tokens"], 3072)

    def test_experimental_effort_uses_request_max_and_profile_ubatch(self) -> None:
        status, _ = self._post(
            {
                "model": "main",
                "messages": [{"role": "user", "content": "think"}],
                "max_tokens": 8192,
                "reasoning_effort": "high",
            }
        )

        self.assertEqual(status, 200)
        forwarded = self.backend.seen_payloads[-1]  # type: ignore[attr-defined]
        self.assertEqual(forwarded["max_tokens"], 8192)
        self.assertEqual(forwarded["thinking_budget_tokens"], 384)

    def test_llamacpp_reasoning_policy_defaults_off_in_forwarded_payload(self) -> None:
        status, _ = self._post(
            {"model": "main", "messages": [{"role": "user", "content": "fast"}]}
        )

        self.assertEqual(status, 200)
        forwarded = self.backend.seen_payloads[-1]  # type: ignore[attr-defined]
        self.assertEqual(
            forwarded["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertEqual(forwarded["reasoning_effort"], "none")
        self.assertEqual(forwarded["thinking_budget_tokens"], 0)

    def test_known_alias_is_rewritten_to_exact_backend_alias(self) -> None:
        status, response = self._post(
            {"model": "main", "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(status, 200)
        self.assertEqual(response["model"], "main-27b")
        self.assertEqual(self.backend.seen_payloads[-1]["model"], "main-27b")  # type: ignore[attr-defined]
        self.assertEqual(self.backend.seen_payloads[-1]["max_tokens"], 65536)  # type: ignore[attr-defined]
        self.assertEqual(self.controller.transitions, [])

    def test_omitted_model_uses_currently_loaded_profile(self) -> None:
        status, response = self._post(
            {"messages": [{"role": "user", "content": "use active"}]}
        )

        self.assertEqual(status, 200)
        self.assertEqual(response["model"], "main-27b")
        self.assertEqual(self.backend.seen_payloads[-1]["model"], "main-27b")  # type: ignore[attr-defined]
        self.assertEqual(self.controller.transitions, [])

    def test_omitted_model_fails_when_no_profile_is_loaded(self) -> None:
        self.broker.application.coordinator.stop_when_idle()  # type: ignore[attr-defined]

        status, response = self._post({"messages": []})

        self.assertEqual(status, 503)
        self.assertEqual(response["error"]["type"], "no_active_profile")  # type: ignore[index]
        self.assertIn("no active model loaded", response["error"]["message"])  # type: ignore[operator,index]
        self.assertEqual(self.backend.seen_payloads, [])  # type: ignore[attr-defined]

    def test_output_budget_is_defaulted_and_oversized_requests_fail_closed(self) -> None:
        self.assertEqual(enforce_output_limit({"model": "x"}, 4096)["max_tokens"], 4096)
        self.assertEqual(
            enforce_output_limit({"max_completion_tokens": 2048}, 4096)[
                "max_completion_tokens"
            ],
            2048,
        )
        self.assertEqual(
            enforce_output_limit({}, 131072, default=32768)["max_tokens"],
            32768,
        )
        self.assertEqual(
            enforce_output_limit({"max_tokens": 131072}, 131072, default=32768)[
                "max_tokens"
            ],
            131072,
        )
        status, response = self._post(
            {"model": "main", "messages": [], "max_tokens": 65537}
        )
        self.assertEqual(status, 400)
        self.assertEqual(response["error"]["type"], "invalid_request_error")  # type: ignore[index]

    def test_v1_requires_valid_api_key_but_health_is_public(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.broker.server_port, timeout=5)
        connection.request("GET", "/v1/models", headers={"Connection": "close"})
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        self.assertEqual(response.status, 401)
        self.assertTrue(response.getheader("X-Request-ID"))
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        self.assertEqual(payload["error"]["type"], "authentication_error")

        connection = http.client.HTTPConnection("127.0.0.1", self.broker.server_port, timeout=5)
        connection.request(
            "GET",
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key", "Connection": "close"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()
        self.assertEqual(response.status, 401)

        connection = http.client.HTTPConnection("127.0.0.1", self.broker.server_port, timeout=5)
        connection.request(
            "GET",
            "/v1/models",
            headers={"X-Api-Key": self.auth_key, "Connection": "close"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()
        self.assertEqual(response.status, 401)

        connection = http.client.HTTPConnection("127.0.0.1", self.broker.server_port, timeout=5)
        connection.request("GET", "/health", headers={"Connection": "close"})
        response = connection.getresponse()
        response.read()
        connection.close()
        self.assertEqual(response.status, 200)

    def test_backend_control_proxy_routes_are_loopback_only(self) -> None:
        for path in ("/slots", "/props"):
            with patch.object(llama_broker.BrokerHandler, "_client_is_loopback", return_value=False):
                status, payload = self._get(path)
            self.assertEqual(status, 403)
            self.assertEqual(payload["error"]["type"], "broker_admin_forbidden")  # type: ignore[index]

    def test_models_omit_stopped_profiles_with_broken_artifacts(self) -> None:
        broken = self.registry.profiles[-1]
        broken.model_path.unlink()
        response = self.broker.application.model_listing()  # type: ignore[attr-defined]
        ids = {item["id"] for item in response["data"]}
        self.assertNotIn(broken.alias, ids)

    def test_models_listing_publishes_only_verified_profile_aliases(self) -> None:
        backend_models = {
            "data": [
                {"id": self.registry.profiles[0].alias, "owned_by": "llamacpp"},
                {"id": "private-backend-model", "owned_by": "llamacpp"},
            ]
        }
        with patch.object(
            llama_broker.BackendClient,
            "list_models",
            return_value=backend_models,
        ):
            status, payload = self._get("/v1/models")

        self.assertEqual(status, 200)
        self.assertEqual(
            {
                (item["id"], item["owned_by"])
                for item in payload["data"]  # type: ignore[union-attr]
            },
            {
                (profile.alias, "Llama-Broker")
                for profile in self.registry.profiles
            },
        )

    def test_model_detail_hides_stopped_broken_profile(self) -> None:
        broken = self.registry.profiles[-1]
        broken.model_path.unlink()
        connection = http.client.HTTPConnection("127.0.0.1", self.broker.server_port, timeout=5)
        connection.request(
            "GET",
            f"/v1/models/{broken.alias}",
            headers={"Authorization": f"Bearer {self.auth_key}"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        self.assertEqual(response.status, 404)
        self.assertEqual(payload["error"]["type"], "broker_model_not_found")

    def test_api_does_not_start_stopped_broken_profile(self) -> None:
        broken = self.registry.profiles[-1]
        broken.model_path.unlink()
        status, payload = self._post(
            {"model": broken.alias, "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["type"], "broker_model_not_found")
        self.assertEqual(self.controller.transitions, [])

    def test_unknown_alias_fails_closed_without_touching_backend(self) -> None:
        status, response = self._post({"model": "qwen3-5-9b-vision", "messages": []})
        self.assertEqual(status, 404)
        self.assertEqual(response["error"]["type"], "broker_model_not_found")  # type: ignore[index]
        self.assertEqual(self.backend.seen_payloads, [])  # type: ignore[attr-defined]
        self.assertEqual(self.controller.transitions, [])

    def test_vision_switches_then_restores_primary_after_response(self) -> None:
        status, response = self._post(
            {"model": "qwen3.5-9b-vision", "messages": [{"role": "user", "content": "see"}]}
        )
        self.assertEqual(status, 200)
        self.assertEqual(response["model"], "qwen3.5-9b-vision")
        deadline = time.monotonic() + 1
        while self.controller.transitions != ["vision", "main"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)
        self.assertEqual(self.controller.transitions, ["vision", "main"])
        self.assertEqual(self.controller.profile, "main")

    def test_models_listing_exposes_context_and_capabilities(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.broker.server_port, timeout=5)
        connection.request(
            "GET",
            "/v1/models",
            headers={"Authorization": f"Bearer {self.auth_key}"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        self.assertEqual(response.status, 200)
        by_id = {item["id"]: item for item in payload["data"]}
        for model in payload["data"]:
            self.assertEqual(model["object"], "model")
            self.assertIsInstance(model["id"], str)
            self.assertIsInstance(model["created"], int)
            self.assertIsInstance(model["owned_by"], str)
            self.assertIsInstance(model["permission"], list)
        self.assertEqual(by_id["main-27b"]["context_length"], 65536)
        self.assertEqual(by_id["main-27b"]["max_output_tokens"], 65536)
        self.assertEqual(by_id["qwen3.5-9b-vision"]["context_length"], 32768)
        self.assertEqual(by_id["qwen3.5-9b-vision"]["parallel"], 2)
        self.assertEqual(by_id["qwen3.5-9b-vision"]["slot_context_window"], 16384)
        self.assertEqual(by_id["qwen3.5-9b-vision"]["max_output_tokens"], 16384)
        self.assertEqual(by_id["qwen3.5-9b-vision"]["capabilities"], ["text", "vision"])

    def test_admin_switch_uses_coordinator_and_reports_consistent_health(self) -> None:
        status, payload = self._post(
            {"model": "qwen3.5-9b-vision"},
            "/admin/switch",
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["backend"]["profile"], "vision")  # type: ignore[index]
        self.assertEqual(payload["leases"]["profile"], "vision")  # type: ignore[index]
        self.assertEqual(self.controller.transitions, ["vision"])

    def test_http_async_operation_can_be_inspected_and_cancelled(self) -> None:
        self._replace_broker(
            max_workers=4,
            control_workers=2,
            drain_timeout=1,
            transition_timeout=1,
            stop_timeout=1,
            start_operation_worker=False,
        )

        status, submitted = self._post(
            {"model": "qwen3.5-9b-vision", "wait": False},
            "/admin/switch",
        )
        self.assertEqual(status, 202)
        raw_operation = submitted.get("operation")
        if not isinstance(raw_operation, dict):
            self.fail("async response omitted its operation")
        request_id = raw_operation.get("request_id")
        execution_id = raw_operation.get("execution_id")
        self.assertIsInstance(request_id, str)
        self.assertIsInstance(execution_id, str)
        if not isinstance(request_id, str) or not isinstance(execution_id, str):
            self.fail("async response omitted request/execution identities")

        status, observed = self._get(f"/admin/operations/{request_id}")
        self.assertEqual(status, 200)
        observed_operation = observed.get("operation")
        if not isinstance(observed_operation, dict):
            self.fail("operation status omitted its operation record")
        self.assertEqual(observed_operation.get("request_id"), request_id)
        self.assertEqual(observed_operation.get("execution_id"), execution_id)
        self.assertEqual(observed_operation.get("status"), "queued")

        status, recent = self._get("/admin/operations")
        self.assertEqual(status, 200)
        operations = recent.get("operations")
        self.assertIsInstance(operations, list)
        if not isinstance(operations, list):
            self.fail("operation listing omitted the operations array")
        self.assertTrue(
            any(
                isinstance(entry, dict)
                and isinstance(entry.get("execution"), dict)
                and entry["execution"].get("execution_id") == execution_id
                for entry in operations
            )
        )

        status, cancelled = self._post(
            {},
            f"/admin/operations/{request_id}/cancel",
        )
        self.assertEqual(status, 200)
        cancelled_operation = cancelled.get("operation")
        if not isinstance(cancelled_operation, dict):
            self.fail("cancel response omitted its operation record")
        self.assertEqual(cancelled_operation.get("status"), "cancelled")
        self.assertEqual(self.broker.application.operation_queue.queue_depth, 0)  # type: ignore[attr-defined]

    def test_changed_profile_snapshot_fails_before_controller_transition(self) -> None:
        self._replace_broker(
            max_workers=4,
            control_workers=2,
            drain_timeout=1,
            transition_timeout=1,
            stop_timeout=1,
            start_operation_worker=False,
        )
        application = self.broker.application  # type: ignore[attr-defined]
        request = application.submit_switch_operation("vision")
        application.registry = ProfileRegistry(
            tuple(
                replace(profile, display_name=f"changed-{profile.id}")
                for profile in application.registry.profiles
            )
        )
        application.start_operation_worker()

        deadline = time.monotonic() + 2
        while application.operation_status(request.request_id)["status"] not in {
            "failed",
            "succeeded",
        }:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

        operation = application.operation_status(request.request_id)
        self.assertEqual(operation["status"], "failed")
        self.assertEqual(operation["error_type"], "StaleOperationSnapshotError")
        self.assertEqual(self.controller.transitions, [])
        self.assertEqual(application.coordinator.active_profile, "main")

    def test_changed_snapshot_fails_even_when_target_is_already_active(self) -> None:
        self._replace_broker(
            max_workers=4,
            control_workers=2,
            drain_timeout=1,
            transition_timeout=1,
            stop_timeout=1,
            start_operation_worker=False,
        )
        application = self.broker.application  # type: ignore[attr-defined]
        request = application.submit_switch_operation("main")
        application.registry = ProfileRegistry(
            tuple(
                replace(profile, display_name=f"new-{profile.id}")
                for profile in application.registry.profiles
            )
        )
        application.start_operation_worker()

        deadline = time.monotonic() + 2
        while application.operation_status(request.request_id)["status"] not in {"failed", "succeeded"}:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

        self.assertEqual(application.operation_status(request.request_id)["error_type"], "StaleOperationSnapshotError")
        self.assertEqual(self.controller.transitions, [])
        self.assertEqual(application.coordinator.active_profile, "main")

    def test_async_admin_switch_accepts_before_backend_transition_finishes(self) -> None:
        transition_started = threading.Event()
        release_transition = threading.Event()
        original_switch = self.controller.switch

        def blocked_switch(profile: str) -> dict[str, object]:
            transition_started.set()
            release_transition.wait(timeout=1)
            return original_switch(profile)

        self.controller.switch = blocked_switch  # type: ignore[method-assign]
        try:
            started = time.monotonic()
            status, payload = self._post(
                {"model": "qwen3.5-9b-vision", "wait": False},
                "/admin/switch",
            )
            elapsed = time.monotonic() - started

            self.assertLess(elapsed, 0.5)
            self.assertEqual(status, 202)
            operation = payload["operation"]
            self.assertIsInstance(operation, dict)
            if not isinstance(operation, dict):
                self.fail("async response omitted its operation record")
            request_id = operation.get("request_id")
            execution_id = operation.get("execution_id")
            if not isinstance(request_id, str) or not isinstance(execution_id, str):
                self.fail("async response omitted request/execution identities")
            self.assertEqual(operation.get("target"), "vision")
            self.assertEqual(operation.get("type"), "switch")
            self.assertTrue(transition_started.wait(timeout=1))
        finally:
            release_transition.set()

        deadline = time.monotonic() + 1
        while self.controller.transitions != ["vision"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

        status, observed = self._get(f"/admin/operations/{request_id}")
        self.assertEqual(status, 200)
        final_operation = observed.get("operation")
        if not isinstance(final_operation, dict):
            self.fail("operation status omitted its operation record")
        self.assertEqual(final_operation.get("execution_id"), execution_id)
        self.assertEqual(final_operation.get("status"), "succeeded")

    def test_async_restart_and_stop_use_priority_operations(self) -> None:
        self._replace_broker(
            max_workers=4,
            control_workers=2,
            drain_timeout=1,
            transition_timeout=1,
            stop_timeout=1,
            start_operation_worker=False,
        )

        status, restart_response = self._post(
            {"wait": False},
            "/admin/restart",
        )
        self.assertEqual(status, 202)
        restart_operation = restart_response.get("operation")
        if not isinstance(restart_operation, dict):
            self.fail("restart response omitted its operation record")
        restart_request_id = restart_operation.get("request_id")
        self.assertIsInstance(restart_request_id, str)
        self.assertEqual(restart_operation.get("type"), "restart")

        status, stop_response = self._post(
            {"wait": False},
            "/admin/stop",
        )
        self.assertEqual(status, 202)
        stop_operation = stop_response.get("operation")
        if not isinstance(stop_operation, dict):
            self.fail("stop response omitted its operation record")
        stop_request_id = stop_operation.get("request_id")
        self.assertIsInstance(stop_request_id, str)
        self.assertEqual(stop_operation.get("type"), "stop")
        self.assertEqual(
            self.broker.application.operation_status(restart_request_id)["status"],  # type: ignore[attr-defined,arg-type]
            "cancelled",
        )

        status, stopped = self._post(
            {},
            f"/admin/operations/{stop_request_id}/cancel",
        )
        self.assertEqual(status, 200)
        stopped_operation = stopped.get("operation")
        if not isinstance(stopped_operation, dict):
            self.fail("stop cancellation omitted operation status")
        self.assertEqual(stopped_operation.get("status"), "cancelled")
        self.assertEqual(
            self.broker.application.operation_queue.pending_transition_count,  # type: ignore[attr-defined]
            0,
        )

    def test_async_restart_executes_and_publishes_terminal_status(self) -> None:
        status, submitted = self._post({"wait": False}, "/admin/restart")
        self.assertEqual(status, 202)
        operation = submitted.get("operation")
        if not isinstance(operation, dict):
            self.fail("async restart response omitted its operation")
        request_id = operation.get("request_id")
        if not isinstance(request_id, str):
            self.fail("async restart response omitted request_id")

        deadline = time.monotonic() + 2
        while self.controller.transitions != ["restart:main"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)
        status, observed = self._get(f"/admin/operations/{request_id}")
        self.assertEqual(status, 200)
        self.assertEqual(observed["operation"]["status"], "succeeded")  # type: ignore[index]

    def test_async_stop_executes_and_publishes_terminal_status(self) -> None:
        status, submitted = self._post({"wait": False}, "/admin/stop")
        self.assertEqual(status, 202)
        operation = submitted.get("operation")
        if not isinstance(operation, dict):
            self.fail("async stop response omitted its operation")
        request_id = operation.get("request_id")
        if not isinstance(request_id, str):
            self.fail("async stop response omitted request_id")

        deadline = time.monotonic() + 2
        while self.controller.stops != 1:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)
        status, observed = self._get(f"/admin/operations/{request_id}")
        self.assertEqual(status, 200)
        self.assertEqual(observed["operation"]["status"], "succeeded")  # type: ignore[index]

    def test_profile_catalog_reload_is_ordered_with_load_operations(self) -> None:
        self._replace_broker(
            max_workers=4,
            control_workers=2,
            drain_timeout=1,
            transition_timeout=1,
            stop_timeout=1,
            start_operation_worker=False,
        )
        manifest = json.loads(self.profiles_path.read_text(encoding="utf-8"))
        manifest["profiles"][1]["display_name"] = "Vision revision two"  # type: ignore[index]
        self.profiles_path.write_text(json.dumps(manifest), encoding="utf-8")

        status, reload_response = self._post(
            {"wait": False},
            "/admin/autotune/reload",
        )
        self.assertEqual(status, 202)
        reload_operation = reload_response.get("operation")
        if not isinstance(reload_operation, dict):
            self.fail("profile reload did not return an operation")
        self.assertEqual(reload_operation.get("type"), "reload")
        reload_execution_id = reload_operation.get("execution_id")
        if not isinstance(reload_execution_id, str):
            self.fail("reload response omitted execution_id")
        captured_catalog = self.broker.application.operation_queue.get_profile_snapshot(  # type: ignore[attr-defined]
            reload_execution_id
        )
        self.assertEqual(
            captured_catalog,
            ProfileRegistry.from_path(self.profiles_path).snapshot_revision_identity(),
        )

        status, load_response = self._post(
            {"model": "qwen3.5-9b-vision", "wait": False},
            "/admin/switch",
        )
        self.assertEqual(status, 202)
        executions = self.broker.application.operation_queue.list_executions()  # type: ignore[attr-defined]
        self.assertEqual([execution.operation_type for execution in executions[:2]], ["reload", "switch"])
        load_snapshot = self.broker.application.operation_queue.get_profile_snapshot(  # type: ignore[attr-defined]
            executions[1].execution_id
        )
        expected_snapshot = ProfileRegistry.from_path(self.profiles_path).profile_snapshot_identity(
            "vision"
        )
        self.assertEqual(load_snapshot, expected_snapshot)

    def test_profile_catalog_reload_executes_in_operation_worker(self) -> None:
        manifest = json.loads(self.profiles_path.read_text(encoding="utf-8"))
        manifest["profiles"][1]["display_name"] = "Vision revision two"  # type: ignore[index]
        self.profiles_path.write_text(json.dumps(manifest), encoding="utf-8")

        status, response = self._post(
            {"wait": False},
            "/admin/autotune/reload",
        )
        self.assertEqual(status, 202)
        operation = response.get("operation")
        if not isinstance(operation, dict):
            self.fail("reload response omitted its operation record")
        request_id = operation.get("request_id")
        if not isinstance(request_id, str):
            self.fail("reload response omitted request_id")

        deadline = time.monotonic() + 2
        while self.broker.application.operation_status(request_id)["status"] not in {  # type: ignore[attr-defined]
            "succeeded",
            "failed",
        }:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)
        final_operation = self.broker.application.operation_status(request_id)  # type: ignore[attr-defined]
        self.assertEqual(final_operation["status"], "succeeded")
        self.assertEqual(
            self.broker.application.registry.get("vision").display_name,  # type: ignore[attr-defined]
            "Vision revision two",
        )

    def test_admin_restart_recovers_last_failed_target_without_model(self) -> None:
        application = self.broker.application  # type: ignore[attr-defined]
        application._remember_controller_state(  # type: ignore[attr-defined]
            {
                "generation": 9,
                "profile": None,
                "phase": "failed",
                "target_profile": "main",
            }
        )
        application.coordinator.synchronize_after_controller_failure(None)
        self.controller.profile = None

        status, payload = self._post({}, "/admin/restart")

        self.assertEqual(status, 200)
        self.assertEqual(payload["backend"]["profile"], "main")  # type: ignore[index]
        self.assertEqual(self.controller.transitions, ["main"])
        self.assertEqual(self.controller.profile, "main")

    def test_admin_restart_reloads_active_profile_without_unloading_it(self) -> None:
        status, payload = self._post({}, "/admin/restart")

        self.assertEqual(status, 200)
        self.assertEqual(payload["backend"]["profile"], "main")  # type: ignore[index]
        self.assertEqual(payload["leases"]["profile"], "main")  # type: ignore[index]
        self.assertEqual(self.controller.transitions, ["restart:main"])
        self.assertEqual(self.controller.profile, "main")

    def test_admin_restart_with_model_switches_when_target_is_different(self) -> None:
        status, payload = self._post(
            {"model": "qwen3.5-9b-vision"},
            "/admin/restart",
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["backend"]["profile"], "vision")  # type: ignore[index]
        self.assertEqual(self.controller.transitions, ["vision"])

    def test_successful_switch_with_unreadable_state_returns_bounded_503(self) -> None:
        original_switch = self.controller.switch
        original_state = self.controller.state
        fail_next_state_read = threading.Event()

        def switch_then_hide_state(profile_id: str) -> dict[str, object]:
            result = original_switch(profile_id)
            fail_next_state_read.set()
            return result

        def flaky_state() -> dict[str, object]:
            if fail_next_state_read.is_set():
                fail_next_state_read.clear()
                raise RuntimeError("state unavailable")
            return original_state()

        self.controller.switch = switch_then_hide_state  # type: ignore[method-assign]
        self.controller.state = flaky_state  # type: ignore[method-assign]

        status, payload = self._post(
            {"model": "qwen3.5-9b-vision"},
            "/admin/switch",
        )

        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["type"], "broker_state_unknown")  # type: ignore[index]
        health = self.broker.application.health()  # type: ignore[attr-defined]
        self.assertEqual(health["status"], "degraded")
        self.assertEqual(health["backend"]["phase"], "unknown")
        self.assertIsNone(self.broker.application.coordinator.active_profile)  # type: ignore[attr-defined]

    def test_health_downgrades_stale_active_state_and_invalidates_idle_lease(self) -> None:
        readiness = iter((True, False))

        def alias_ready(alias: str) -> bool:
            self.assertEqual(alias, "main-27b")
            return next(readiness)

        application = BrokerApplication(
            registry=self.registry,
            controller=FakeController("main"),
            backend_host="127.0.0.1",
            backend_port=self.backend.server_port,
            backend_timeout=10,
            max_body_bytes=1_000_000,
            security=SecurityConfig(api_key_file=self.auth_key_path),
            backend_ready=lambda: True,
            backend_alias_ready=alias_ready,
        )
        try:
            self.assertEqual(application.coordinator.active_profile, "main")

            payload = application.health()

            self.assertEqual(payload["status"], "degraded")
            self.assertEqual(payload["backend"]["phase"], "stale")
            self.assertEqual(payload["backend"]["declared_phase"], "active")
            self.assertFalse(payload["backend"]["ready"])
            self.assertIsNone(application.coordinator.active_profile)
        finally:
            application.shutdown()

    def test_profile_artifact_verifier_rehashes_projector_before_start(self) -> None:
        profile = self.registry.get("vision")
        self.assertIsNotNone(profile.mmproj_path)
        assert profile.mmproj_path is not None
        mmproj_path = Path(profile.mmproj_path)
        mmproj_path.write_bytes(b"tampered-projector")

        with self.assertRaisesRegex(BackendControllerError, "mmproj SHA mismatch"):
            llama_broker._verify_profile_artifacts(profile)

    def test_profile_artifact_verifier_reuses_unchanged_hashes(self) -> None:
        llama_broker._ARTIFACT_VERIFICATION_CACHE.clear()
        profile = self.registry.get("main")

        with patch.object(llama_broker, "sha256_file", wraps=sha256_file) as digest:
            llama_broker._verify_profile_artifacts(profile)
            first_call_count = digest.call_count
            llama_broker._verify_profile_artifacts(profile)
            self.assertGreater(first_call_count, 0)
            self.assertEqual(digest.call_count, first_call_count)

            Path(profile.model_path).write_bytes(b"changed-model")
            with self.assertRaisesRegex(BackendControllerError, "model SHA mismatch"):
                llama_broker._verify_profile_artifacts(profile)
            self.assertGreater(digest.call_count, first_call_count)

    def test_failed_recovery_cannot_republish_dead_profile_as_active(self) -> None:
        application = self.broker.application  # type: ignore[attr-defined]
        application._backend_alias_ready = lambda _alias: False

        class OnePassStop:
            def __init__(self) -> None:
                self.calls = 0

            def wait(self, _timeout: float) -> bool:
                self.calls += 1
                return self.calls > 1

            def set(self) -> None:
                return

        application._recovery_stop = OnePassStop()  # type: ignore[assignment]
        self.controller.restart = lambda _profile: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeError("recovery restart failed")
        )

        application._recovery_loop()

        self.assertIsNone(application.coordinator.active_profile)
        self.assertEqual(application._recovery_profile, "main")
        with self.assertRaises(BrokerQuiescingError):
            with application.coordinator.lease_active():
                self.fail("dead backend retained an active request lease")
        application.coordinator.stop_when_idle()
        self.assertIsNone(application._recovery_profile)

    def _open_chat(self, payload: dict[str, object]) -> http.client.HTTPConnection:
        connection = http.client.HTTPConnection("127.0.0.1", self.broker.server_port, timeout=5)
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(payload),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.auth_key}",
            },
        )
        return connection

    def test_client_tls_eof_mid_response_is_a_disconnect_not_a_backend_failure(self) -> None:
        class TlsEofAfterHeaders:
            """Lets the status line through, then fails like a vanished TLS peer."""

            def __init__(self, inner: object) -> None:
                self.inner = inner
                self.writes = 0

            def write(self, data: bytes) -> int:
                self.writes += 1
                if self.writes > 1:
                    raise ssl.SSLEOFError(8, "EOF occurred in violation of protocol")
                return self.inner.write(data)  # type: ignore[attr-defined]

            def flush(self) -> None:
                return None

            def close(self) -> None:
                self.inner.close()  # type: ignore[attr-defined]

            @property
            def closed(self) -> bool:
                return self.inner.closed  # type: ignore[attr-defined]

        original_setup = llama_broker.BrokerHandler.setup

        def setup(handler: llama_broker.BrokerHandler) -> None:
            original_setup(handler)
            handler.wfile = TlsEofAfterHeaders(handler.wfile)  # type: ignore[assignment]

        coordinator = self.broker.application.coordinator
        with (
            patch.object(llama_broker.BrokerHandler, "setup", setup),
            patch.object(
                coordinator,
                "invalidate_when_idle",
                wraps=coordinator.invalidate_when_idle,
            ) as invalidate,
            patch.object(self.broker, "handle_error") as handle_error,
            patch.object(llama_broker, "_log") as log,
        ):
            connection = self._open_chat({"model": "main", "messages": []})
            try:
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                with self.assertRaises(http.client.IncompleteRead):
                    response.read()
            finally:
                connection.close()

        invalidate.assert_not_called()
        handle_error.assert_not_called()
        messages = [call.args[0] for call in log.call_args_list]
        self.assertTrue(
            any("client disconnected during main response" in m for m in messages),
            messages,
        )
        self.assertFalse(any("backend proxy failure" in m for m in messages), messages)
        self.assertEqual(coordinator.active_profile, "main")

    def test_backend_reset_mid_stream_does_not_append_json_error(self) -> None:
        coordinator = self.broker.application.coordinator
        with (
            patch.object(
                coordinator,
                "invalidate_when_idle",
                wraps=coordinator.invalidate_when_idle,
            ) as invalidate,
            patch.object(self.broker, "handle_error") as handle_error,
            patch.object(llama_broker, "_log") as log,
        ):
            connection = self._open_chat(
                {"model": "main", "messages": [], "stream": True, "_test_reset_stream": True}
            )
            try:
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                first = response.readline()
                self.backend.reset_stream_release.set()  # type: ignore[attr-defined]
                rest = response.read()
            finally:
                connection.close()

        self.assertIn(b"partial", first)
        self.assertNotIn(b"broker_proxy_error", rest)
        invalidate.assert_called_once_with("main")
        handle_error.assert_not_called()
        messages = [call.args[0] for call in log.call_args_list]
        self.assertTrue(any("backend proxy failure for main" in m for m in messages), messages)

    def _crash_main_backend(self) -> tuple[int, dict[str, object]]:
        connection = self._open_chat(
            {"model": "main", "messages": [], "_test_reset_before_headers": True}
        )
        try:
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_backend_exit_during_request_is_logged_with_its_cause(self) -> None:
        self.controller.describe_exit = lambda _profile: BackendExit(  # type: ignore[attr-defined]
            result="signal",
            exit_status="SIGABRT",
            detail=("CUDA error: device not ready",),
        )
        with patch.object(llama_broker, "_log") as log:
            status, body = self._crash_main_backend()

        self.assertEqual(status, 502)
        self.assertEqual(
            body["error"]["message"],
            "backend exited during the request: signal (SIGABRT): CUDA error: device not ready",
        )
        messages = [call.args[0] for call in log.call_args_list]
        self.assertTrue(
            any(
                m.startswith("[broker] backend main exited during request ")
                and m.endswith(": signal (SIGABRT): CUDA error: device not ready")
                for m in messages
            ),
            messages,
        )
        self.assertEqual(self.broker.application.crash_loops.snapshot()["main"]["crashes"], 1)

    def test_proxy_failure_with_backend_still_running_is_not_a_crash(self) -> None:
        self.controller.describe_exit = lambda _profile: None  # type: ignore[attr-defined]
        with patch.object(llama_broker, "_log"):
            status, body = self._crash_main_backend()

        self.assertEqual(status, 502)
        self.assertEqual(body["error"]["message"], "backend request failed")
        self.assertEqual(self.broker.application.crash_loops.snapshot(), {})

    def test_repeated_backend_crashes_refuse_the_profile_instead_of_restarting_it(self) -> None:
        application = self.broker.application
        application.crash_loops = CrashLoopGuard(threshold=2, window=600, cooldown=300)
        self.controller.describe_exit = lambda _profile: BackendExit(  # type: ignore[attr-defined]
            result="signal", exit_status="SIGABRT", detail=("CUDA error: device not ready",)
        )
        with patch.object(llama_broker, "_log") as log:
            self.assertEqual(self._crash_main_backend()[0], 502)
            # The first crash restarts the backend for the next request as before.
            self.assertEqual(self._crash_main_backend()[0], 502)
            self.assertEqual(self.controller.transitions, ["main"])
            seen = len(self.backend.seen_payloads)  # type: ignore[attr-defined]

            connection = self._open_chat({"model": "main", "messages": []})
            try:
                response = connection.getresponse()
                status = response.status
                retry_after = response.getheader("Retry-After")
                error = json.loads(response.read())["error"]
            finally:
                connection.close()

        self.assertEqual(status, 503)
        self.assertEqual(retry_after, "300")
        self.assertEqual(error["type"], "backend_crash_loop")
        self.assertIn("crashed 2 times", error["message"])
        self.assertIn("CUDA error: device not ready", error["message"])
        self.assertEqual(self.controller.transitions, ["main"])
        self.assertEqual(len(self.backend.seen_payloads), seen)  # type: ignore[attr-defined]
        messages = [call.args[0] for call in log.call_args_list]
        self.assertTrue(any("is crash-looping" in m for m in messages), messages)

        health = application.health()
        self.assertEqual(health["status"], "degraded")
        self.assertTrue(health["crash_loops"]["main"]["tripped"])

        application.submit_switch_operation("main")
        self.assertEqual(application.crash_loops.snapshot(), {})

    def test_completed_response_clears_crash_history(self) -> None:
        self.controller.describe_exit = lambda _profile: BackendExit(  # type: ignore[attr-defined]
            result="signal", exit_status="SIGABRT"
        )
        with patch.object(llama_broker, "_log"):
            self._crash_main_backend()
            self.assertIn("main", self.broker.application.crash_loops.snapshot())
            status, _ = self._post({"model": "main", "messages": []})

        self.assertEqual(status, 200)
        self.assertEqual(self.broker.application.crash_loops.snapshot(), {})

    def test_saturated_streams_cannot_block_health_or_emergency_stop(self) -> None:
        self._replace_broker(
            max_workers=2,
            control_workers=2,
            drain_timeout=0.1,
            transition_timeout=1,
            stop_timeout=1,
        )
        stream_errors: list[BaseException] = []

        def held_stream() -> None:
            connection = http.client.HTTPConnection(
                "127.0.0.1", self.broker.server_port, timeout=3
            )
            try:
                body = json.dumps(
                    {
                        "model": "main",
                        "messages": [],
                        "stream": True,
                        "_test_hold_stream": True,
                    }
                )
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.auth_key}",
                    },
                )
                response = connection.getresponse()
                response.read()
            except BaseException as error:
                stream_errors.append(error)
            finally:
                connection.close()

        streams = [threading.Thread(target=held_stream) for _ in range(2)]
        for stream in streams:
            stream.start()
        self.assertTrue(
            self.backend.hold_streams_entered.wait(timeout=1),  # type: ignore[attr-defined]
            "both data workers should be occupied by held streams",
        )

        switch_result: list[tuple[int, dict[str, object]]] = []
        switch = threading.Thread(
            target=lambda: switch_result.append(
                self._post({"model": "qwen3.5-9b-vision"}, "/admin/switch")
            )
        )
        switch.start()
        deadline = time.monotonic() + 1
        while not self.broker.application.coordinator.quiescing:  # type: ignore[attr-defined]
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

        started = time.monotonic()
        health_status, health = self._get("/admin/status")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(health_status, 200)
        self.assertTrue(health["leases"]["quiescing"])  # type: ignore[index]

        switch.join(timeout=1)
        self.assertFalse(switch.is_alive(), "admin switch drain must be bounded")
        self.assertEqual(switch_result[0][0], 504)
        self.assertEqual(
            switch_result[0][1]["error"]["type"],  # type: ignore[index]
            "broker_drain_timeout",
        )

        overloaded_status, overloaded = self._post(
            {"model": "main", "messages": []}
        )
        self.assertEqual(overloaded_status, 503)
        self.assertEqual(
            overloaded["error"]["type"],  # type: ignore[index]
            "broker_overloaded",
        )

        stop_result: list[tuple[int, dict[str, object]]] = []
        stop = threading.Thread(
            target=lambda: stop_result.append(self._post({}, "/admin/stop"))
        )
        stop.start()
        deadline = time.monotonic() + 1
        while not self.broker.application.coordinator.quiescing:  # type: ignore[attr-defined]
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)
        health_status, _health = self._get("/healthz")
        self.assertEqual(health_status, 200)
        stop.join(timeout=1)
        self.assertFalse(stop.is_alive(), "emergency stop must not wait forever")
        self.assertEqual(stop_result[0][0], 200)
        self.assertEqual(self.controller.stops, 1)

        self.backend.hold_stream_release.set()  # type: ignore[attr-defined]
        for stream in streams:
            stream.join(timeout=1)
        self.assertFalse(any(stream.is_alive() for stream in streams))
        self.assertEqual(stream_errors, [])

    def _post_large(self, headers: dict[str, str] | None = None) -> tuple[int, dict[str, object]]:
        # Larger than the loopback socket buffers, so an unread body is still
        # in flight when the broker answers.
        payload = {
            "model": "main",
            "messages": [{"role": "user", "content": "x" * 900_000}],
            "stream": True,
        }
        connection = http.client.HTTPConnection("127.0.0.1", self.broker.server_port, timeout=5)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload),
                headers={"Content-Type": "application/json", **(headers or {})},
            )
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_rejection_before_body_read_is_delivered_not_reset(self) -> None:
        status, payload = self._post_large()

        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["type"], "authentication_error")  # type: ignore[index]

    def test_saturated_broker_answers_large_requests_with_503_not_reset(self) -> None:
        self._replace_broker(
            max_workers=2,
            control_workers=1,
            drain_timeout=0.1,
            transition_timeout=1,
            stop_timeout=1,
        )
        auth = {"Authorization": f"Bearer {self.auth_key}"}

        def held_stream() -> None:
            connection = http.client.HTTPConnection(
                "127.0.0.1", self.broker.server_port, timeout=3
            )
            try:
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=json.dumps(
                        {"model": "main", "messages": [], "stream": True, "_test_hold_stream": True}
                    ),
                    headers={"Content-Type": "application/json", **auth},
                )
                connection.getresponse().read()
            finally:
                connection.close()

        streams = [threading.Thread(target=held_stream) for _ in range(2)]
        for stream in streams:
            stream.start()
        self.assertTrue(
            self.backend.hold_streams_entered.wait(timeout=1),  # type: ignore[attr-defined]
            "both data workers should be occupied by held streams",
        )

        # Control lane: the handler refuses inference before reading the body.
        status, payload = self._post_large(auth)
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["type"], "broker_overloaded")  # type: ignore[index]

        # Occupy the only control worker with a client that never sends its
        # request line, so the next connection is refused at accept time.
        control_slots = self.broker._control_worker_slots  # type: ignore[attr-defined]

        def wait_for_free_control_workers(expected: int) -> None:
            deadline = time.monotonic() + 3
            while control_slots._value != expected:  # type: ignore[attr-defined]
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.005)

        wait_for_free_control_workers(1)
        stalled = socket.create_connection(("127.0.0.1", self.broker.server_port))
        try:
            wait_for_free_control_workers(0)
            # Force the accept-time rejection down the saturated-drainer fallback.
            self.broker._rejection_drains = threading.BoundedSemaphore(  # type: ignore[attr-defined]
                0
            )
            status, payload = self._post_large(auth)
        finally:
            stalled.close()
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["type"], "broker_overloaded")  # type: ignore[index]
        self.assertIn("capacity is exhausted", payload["error"]["message"])  # type: ignore[index]

        self.backend.hold_stream_release.set()  # type: ignore[attr-defined]
        for stream in streams:
            stream.join(timeout=1)

    def test_health_uses_cached_state_while_controller_transition_lock_is_held(self) -> None:
        self._assert_health_during_blocked_switch("main")

    def test_health_reports_cold_load_while_controller_transition_lock_is_held(self) -> None:
        self._assert_health_during_blocked_switch(None)

    def _assert_health_during_blocked_switch(self, initial_profile: str | None) -> None:
        self._replace_broker(
            max_workers=2,
            control_workers=2,
            drain_timeout=0.1,
            transition_timeout=0.2,
            stop_timeout=1,
            initial_profile=initial_profile,
        )
        transition_started = threading.Event()
        allow_transition = threading.Event()

        def blocked_switch(profile: str) -> dict[str, object]:
            with self.controller._lock:
                transition_started.set()
                allow_transition.wait(timeout=2)
                self.controller.transitions.append(profile)
                self.controller.profile = profile
                self.controller.generation += 1
                return self.controller.state()

        self.controller.switch = blocked_switch  # type: ignore[method-assign]
        result: list[tuple[int, dict[str, object]]] = []
        request = threading.Thread(
            target=lambda: result.append(
                self._post(
                    {"model": "qwen3.5-9b-vision", "messages": []},
                    "/v1/chat/completions",
                )
            )
        )
        request.start()
        self.assertTrue(transition_started.wait(timeout=1))
        try:
            started = time.monotonic()
            status, health = self._get("/health")
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(status, 200)
            self.assertTrue(health["leases"]["transitioning"])  # type: ignore[index]
            self.assertEqual(health["backend"]["phase"], "switching")  # type: ignore[index]
            self.assertEqual(health["backend"]["target_profile"], "vision")  # type: ignore[index]
            self.assertFalse(health["backend"]["ready"])  # type: ignore[index]

            request.join(timeout=1)
            self.assertFalse(request.is_alive())
            self.assertEqual(result[0][0], 504)
            self.assertEqual(
                result[0][1]["error"]["type"],  # type: ignore[index]
                "broker_transition_timeout",
            )
            status, health = self._get("/admin/status")
            self.assertEqual(status, 200)
            self.assertTrue(health["leases"]["quiescing"])  # type: ignore[index]
            self.assertEqual(health["backend"]["phase"], "switching")  # type: ignore[index]
            self.assertEqual(health["backend"]["target_profile"], "vision")  # type: ignore[index]
            self.assertFalse(health["backend"]["ready"])  # type: ignore[index]
        finally:
            allow_transition.set()
            request.join(timeout=1)

        deadline = time.monotonic() + 1
        while self.broker.application.coordinator.quiescing:  # type: ignore[attr-defined]
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)
        status, health = self._get("/admin/status")
        self.assertEqual(status, 200)
        self.assertEqual(health["backend"]["phase"], "active")  # type: ignore[index]
        self.assertEqual(health["backend"]["profile"], "vision")  # type: ignore[index]
        self.assertTrue(health["backend"]["ready"])  # type: ignore[index]
        self.assertNotIn("target_profile", health["backend"])


class CoordinatorTests(unittest.TestCase):
    def test_external_queue_blocks_leases_until_reserved_transition_runs(self) -> None:
        transitions: list[str] = []
        coordinator = ProfileCoordinator("main", transitions.append)

        coordinator.set_external_queued_transitions(1)
        self.assertTrue(coordinator.quiescing)
        with self.assertRaises(BrokerQuiescingError):
            with coordinator.lease("main"):
                self.fail("an inference lease entered while a durable transition was queued")

        coordinator.begin_external_transition()
        self.assertEqual(coordinator.snapshot()["queued_transitions"], 1)
        coordinator.run_reserved_switch("vision")

        self.assertEqual(transitions, ["vision"])
        self.assertEqual(coordinator.active_profile, "vision")
        self.assertFalse(coordinator.quiescing)

    def test_external_queue_cancellation_before_commit_preserves_active_profile(self) -> None:
        calls: list[str] = []

        def switch_with_commit(
            target: str,
            commit_callback: Callable[[], None],
        ) -> None:
            calls.append(f"start:{target}")
            commit_callback()
            calls.append(f"commit:{target}")

        coordinator = ProfileCoordinator(
            "main",
            calls.append,
            switch_profile_with_commit=switch_with_commit,
        )
        coordinator.set_external_queued_transitions(1)
        coordinator.begin_external_transition()

        def cancel() -> None:
            raise LifecycleCancelledError("cancelled before destructive work")

        with self.assertRaises(LifecycleCancelledError):
            coordinator.run_reserved_switch("vision", commit_callback=cancel)

        self.assertEqual(coordinator.active_profile, "main")
        self.assertFalse(coordinator.quiescing)
        self.assertEqual(calls, ["start:vision"])

    def test_reserved_profile_reload_commits_inside_fifo_transition_slot(self) -> None:
        events: list[str] = []
        coordinator = ProfileCoordinator("main", lambda _profile: None)
        coordinator.set_external_queued_transitions(1)
        coordinator.begin_external_transition()

        def reload(commit: Callable[[], None]) -> str:
            events.append("preflight")
            commit()
            events.append("apply")
            return "reloaded"

        result = coordinator.run_reserved_reload(
            reload,
            commit_callback=lambda: events.append("commit"),
        )

        self.assertEqual(result, "reloaded")
        self.assertEqual(events, ["preflight", "commit", "apply"])
        self.assertFalse(coordinator.quiescing)

    def test_transient_lease_restores_profile_active_before_switch(self) -> None:
        transitions: list[str] = []
        coordinator = ProfileCoordinator("gemma", transitions.append)

        with coordinator.lease("vision", transient=True):
            self.assertEqual(coordinator.active_profile, "vision")

        self.assertEqual(transitions, ["vision", "gemma"])
        self.assertEqual(coordinator.active_profile, "gemma")

    def _assert_transient_round_trip(self, initial_profile: str) -> None:
        transitions: list[str] = []
        coordinator = ProfileCoordinator(initial_profile, transitions.append)

        with coordinator.lease("vision", transient=True):
            self.assertEqual(coordinator.active_profile, "vision")

        self.assertEqual(transitions, ["vision", initial_profile])
        self.assertEqual(coordinator.active_profile, initial_profile)

    def test_transient_lease_restores_persistent_profile(self) -> None:
        self._assert_transient_round_trip("persistent")

    def test_transient_lease_restores_kvarn_equivalent_profile(self) -> None:
        self._assert_transient_round_trip("kvarn4")

    def test_repeated_and_nested_temporary_leases_are_bounded(self) -> None:
        transitions: list[str] = []
        coordinator = ProfileCoordinator("primary", transitions.append)

        with coordinator.lease("vision", transient=True):
            with coordinator.lease("vision", transient=True):
                pass
            with self.assertRaises(BrokerQuiescingError):
                with coordinator.lease("other", transient=True):
                    pass

        for _ in range(3):
            with coordinator.lease("vision", transient=True):
                pass

        self.assertEqual(
            transitions,
            ["vision", "primary"] * 4,
        )
        self.assertEqual(coordinator.active_profile, "primary")

    def test_administrative_switch_cancels_temporary_restore(self) -> None:
        transitions: list[str] = []
        coordinator = ProfileCoordinator("primary", transitions.append)
        entered = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        def temporary_request() -> None:
            try:
                with coordinator.lease("vision", transient=True):
                    entered.set()
                    release.wait(timeout=2)
            except BaseException as error:
                errors.append(error)

        request = threading.Thread(target=temporary_request)
        request.start()
        self.assertTrue(entered.wait(timeout=1))

        admin_errors: list[BaseException] = []

        def administrative_switch() -> None:
            try:
                coordinator.switch_when_idle("admin")
            except BaseException as error:
                admin_errors.append(error)

        admin = threading.Thread(target=administrative_switch)
        admin.start()
        deadline = time.monotonic() + 1
        while not coordinator.quiescing:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

        release.set()
        request.join(timeout=2)
        admin.join(timeout=2)

        self.assertFalse(request.is_alive())
        self.assertFalse(admin.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(admin_errors, [])
        self.assertEqual(transitions, ["vision", "admin"])
        self.assertEqual(coordinator.active_profile, "admin")

    def test_administrative_switch_cancels_deferred_restore(self) -> None:
        transition_started = threading.Event()
        allow_transition = threading.Event()
        transitions: list[str] = []

        def blocked_switch(profile: str) -> None:
            transitions.append(profile)
            if profile == "vision":
                transition_started.set()
                allow_transition.wait(timeout=2)

        coordinator = ProfileCoordinator(
            "primary",
            blocked_switch,
            drain_timeout=1,
            transition_timeout=0.1,
        )
        with self.assertRaises(BrokerTransitionTimeoutError):
            with coordinator.lease("vision"):
                self.fail("timed-out switch admitted a lease")
        self.assertTrue(transition_started.is_set())
        coordinator.restore_when_idle("vision", "primary")

        admin_errors: list[BaseException] = []

        def administrative_switch() -> None:
            try:
                coordinator.switch_when_idle("admin")
            except BaseException as error:
                admin_errors.append(error)

        admin = threading.Thread(target=administrative_switch)
        admin.start()
        deadline = time.monotonic() + 1
        while coordinator.snapshot()["queued_transitions"] == 0:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

        allow_transition.set()
        admin.join(timeout=2)

        self.assertFalse(admin.is_alive())
        self.assertEqual(admin_errors, [])
        self.assertEqual(transitions, ["vision", "admin"])
        self.assertEqual(coordinator.active_profile, "admin")

    def test_request_preempts_idle_in_flight_load_then_original_reloads(self) -> None:
        loading = threading.Event()
        preempted = threading.Event()
        transitions: list[str] = []
        preempt_calls: list[bool] = []
        served: list[str] = []

        def switch(profile: str) -> None:
            transitions.append(profile)
            if profile == "slow" and transitions.count("slow") == 1:
                loading.set()
                if preempted.wait(timeout=2):
                    raise BackendControllerError("activation preempted")

        def preempt() -> bool:
            preempt_calls.append(True)
            preempted.set()
            return True

        coordinator = ProfileCoordinator(
            "primary",
            switch,
            drain_timeout=2,
            transition_timeout=5,
            preempt_activation=preempt,
        )
        errors: list[BaseException] = []

        def slow_request() -> None:
            try:
                with coordinator.lease("slow"):
                    served.append("slow")
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=slow_request)
        thread.start()
        self.assertTrue(loading.wait(timeout=2))

        with coordinator.lease("vision"):
            served.append("vision")
        thread.join(timeout=3)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(preempt_calls, [True])
        self.assertEqual(transitions, ["slow", "vision", "slow"])
        self.assertEqual(served, ["vision", "slow"])
        self.assertEqual(coordinator.active_profile, "slow")
        self.assertEqual(coordinator.active_requests, 0)

    def test_administrative_load_is_not_preempted(self) -> None:
        loading = threading.Event()
        release = threading.Event()
        transitions: list[str] = []
        preempt_calls: list[bool] = []

        def switch(profile: str) -> None:
            transitions.append(profile)
            if profile == "admin":
                loading.set()
                release.wait(timeout=2)

        coordinator = ProfileCoordinator(
            "primary",
            switch,
            drain_timeout=2,
            transition_timeout=5,
            preempt_activation=lambda: preempt_calls.append(True) or True,
        )
        coordinator.switch_when_idle_async("admin")
        self.assertTrue(loading.wait(timeout=2))

        def finish_admin_load() -> None:
            time.sleep(0.05)
            release.set()

        threading.Thread(target=finish_admin_load).start()
        with coordinator.lease("vision"):
            pass

        self.assertEqual(preempt_calls, [])
        self.assertEqual(transitions, ["admin", "vision"])

    def test_load_with_active_lease_is_not_preempted(self) -> None:
        preempt_calls: list[bool] = []
        coordinator = ProfileCoordinator(
            "primary",
            lambda profile: None,
            drain_timeout=0.05,
            preempt_activation=lambda: preempt_calls.append(True) or True,
        )
        with coordinator.lease("primary"):
            with self.assertRaises(llama_broker.BrokerDrainTimeoutError):
                with coordinator.lease("vision"):
                    self.fail("switch admitted while a slot was in use")

        self.assertEqual(preempt_calls, [])

    def test_readiness_probe_stops_waiting_once_preempted(self) -> None:
        preempted = threading.Event()
        preempted.set()
        started = time.monotonic()

        ready = llama_broker._readiness_probe(
            "vision",
            30,
            security=SecurityConfig(),
            backend_url="http://127.0.0.1:9",
            preempted=preempted,
        )

        self.assertFalse(ready)
        self.assertLess(time.monotonic() - started, 1)

    def test_failed_temporary_restore_clears_cleanup_state(self) -> None:
        transitions: list[str] = []

        def switch(profile: str) -> None:
            transitions.append(profile)
            if profile == "primary":
                raise BackendControllerError("restore failed")

        coordinator = ProfileCoordinator("primary", switch)
        logs: list[str] = []
        with patch.object(llama_broker, "_log", logs.append):
            with coordinator.lease("vision", transient=True):
                pass

        self.assertIsNone(coordinator.active_profile)
        self.assertEqual(coordinator.active_requests, 0)
        self.assertTrue(any("failed to restore temporary profile" in log for log in logs))

        with coordinator.lease("other", transient=True):
            pass

        self.assertEqual(transitions, ["vision", "primary", "other"])

    def test_timed_out_temporary_switch_does_not_restore_late_target(self) -> None:
        transition_started = threading.Event()
        allow_transition = threading.Event()
        transitions: list[str] = []

        def blocked_switch(profile: str) -> None:
            transitions.append(profile)
            if profile == "vision":
                transition_started.set()
                allow_transition.wait(timeout=2)

        coordinator = ProfileCoordinator(
            "primary",
            blocked_switch,
            drain_timeout=1,
            transition_timeout=0.1,
        )
        with self.assertRaises(BrokerTransitionTimeoutError):
            with coordinator.lease("vision", transient=True):
                self.fail("timed-out temporary switch admitted a lease")

        self.assertTrue(transition_started.is_set())
        allow_transition.set()
        deadline = time.monotonic() + 1
        while coordinator.quiescing:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

        self.assertEqual(transitions, ["vision"])
        self.assertEqual(coordinator.active_profile, "vision")

    def test_same_profile_leases_overlap(self) -> None:
        transitions: list[str] = []
        coordinator = ProfileCoordinator("main", transitions.append)
        entered = threading.Barrier(3)
        release = threading.Event()
        active = 0
        peak = 0
        lock = threading.Lock()

        def worker() -> None:
            nonlocal active, peak
            with coordinator.lease("main"):
                with lock:
                    active += 1
                    peak = max(peak, active)
                entered.wait(timeout=2)
                release.wait(timeout=2)
                with lock:
                    active -= 1

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        entered.wait(timeout=2)
        release.set()
        for thread in threads:
            thread.join(timeout=2)
        self.assertEqual(peak, 2)
        self.assertEqual(transitions, [])

    def test_cross_profile_switch_waits_for_active_lease(self) -> None:
        transitions: list[str] = []
        coordinator = ProfileCoordinator("main", transitions.append)
        acquired = threading.Event()

        def vision() -> None:
            with coordinator.lease("vision"):
                acquired.set()

        with coordinator.lease("main"):
            thread = threading.Thread(target=vision)
            thread.start()
            time.sleep(0.05)
            self.assertFalse(acquired.is_set())
            self.assertEqual(transitions, [])
        thread.join(timeout=2)
        self.assertTrue(acquired.is_set())
        self.assertEqual(transitions, ["vision"])

    def test_administrative_switch_waits_for_active_lease(self) -> None:
        transitions: list[str] = []
        coordinator = ProfileCoordinator("main", transitions.append)
        entered = threading.Event()
        release = threading.Event()

        def active_request() -> None:
            with coordinator.lease("main"):
                entered.set()
                release.wait(timeout=2)

        request = threading.Thread(target=active_request)
        request.start()
        self.assertTrue(entered.wait(timeout=1))
        admin = threading.Thread(
            target=coordinator.switch_when_idle,
            args=("vision",),
        )
        admin.start()
        time.sleep(0.05)
        self.assertEqual(transitions, [])
        release.set()
        request.join(timeout=2)
        admin.join(timeout=2)
        self.assertEqual(transitions, ["vision"])
        self.assertEqual(coordinator.active_profile, "vision")

    def test_administrative_restart_interrupts_active_lease(self) -> None:
        transitions: list[str] = []
        restart_started = threading.Event()
        release = threading.Event()

        def restart(profile: str) -> None:
            transitions.append(f"restart:{profile}")
            restart_started.set()

        coordinator = ProfileCoordinator(
            "main",
            lambda _profile: None,
            restart_profile=restart,
            drain_timeout=1,
            transition_timeout=1,
        )
        entered = threading.Event()
        request = threading.Thread(
            target=lambda: self._hold_active_lease(coordinator, entered, release)
        )
        request.start()
        self.assertTrue(entered.wait(timeout=1))

        errors: list[BaseException] = []

        def administrative_restart() -> None:
            try:
                coordinator.restart_when_idle()
            except BaseException as error:
                errors.append(error)

        admin = threading.Thread(target=administrative_restart)
        admin.start()
        try:
            self.assertTrue(
                restart_started.wait(timeout=0.2),
                "administrative restart must not drain an active request first",
            )
            self.assertEqual(transitions, ["restart:main"])
            self.assertEqual(coordinator.active_requests, 1)
        finally:
            release.set()
            admin.join(timeout=1)
            request.join(timeout=1)

        self.assertFalse(admin.is_alive())
        self.assertFalse(request.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(coordinator.active_profile, "main")
        self.assertEqual(coordinator.active_requests, 0)

    @staticmethod
    def _hold_active_lease(
        coordinator: ProfileCoordinator,
        entered: threading.Event,
        release: threading.Event,
    ) -> None:
        with coordinator.lease("main"):
            entered.set()
            release.wait(timeout=2)

    def test_administrative_stop_waits_for_active_lease(self) -> None:
        stops: list[str] = []
        coordinator = ProfileCoordinator(
            "main", lambda _profile: None, lambda: stops.append("stopped")
        )
        entered = threading.Event()
        release = threading.Event()

        def active_request() -> None:
            with coordinator.lease("main"):
                entered.set()
                release.wait(timeout=2)

        request = threading.Thread(target=active_request)
        request.start()
        self.assertTrue(entered.wait(timeout=1))
        admin = threading.Thread(target=coordinator.stop_when_idle)
        admin.start()
        time.sleep(0.05)
        self.assertEqual(stops, [])
        release.set()
        request.join(timeout=2)
        admin.join(timeout=2)
        self.assertEqual(stops, ["stopped"])
        self.assertIsNone(coordinator.active_profile)

    def test_cross_profile_drain_is_bounded_and_closes_same_profile_gate(self) -> None:
        transitions: list[str] = []
        coordinator = ProfileCoordinator(
            "main",
            transitions.append,
            drain_timeout=0.1,
            transition_timeout=1,
        )
        entered = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        def holder() -> None:
            with coordinator.lease("main"):
                entered.set()
                release.wait(timeout=2)

        def switcher() -> None:
            try:
                with coordinator.lease("vision"):
                    pass
            except BaseException as error:
                errors.append(error)

        held = threading.Thread(target=holder)
        held.start()
        self.assertTrue(entered.wait(timeout=1))
        started = time.monotonic()
        switching = threading.Thread(target=switcher)
        switching.start()
        deadline = time.monotonic() + 1
        while not coordinator.quiescing:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

        with self.assertRaises(BrokerQuiescingError):
            with coordinator.lease("main"):
                self.fail("same-profile lease entered while a writer was queued")

        switching.join(timeout=1)
        self.assertFalse(switching.is_alive())
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], BrokerDrainTimeoutError)
        self.assertEqual(transitions, [])
        release.set()
        held.join(timeout=1)

    def test_transition_wait_with_zero_requests_is_not_reported_as_drain_timeout(self) -> None:
        transition_started = threading.Event()
        allow_transition = threading.Event()
        transitions: list[str] = []
        first_errors: list[BaseException] = []

        def blocked_switch(profile: str) -> None:
            transitions.append(profile)
            transition_started.set()
            allow_transition.wait(timeout=2)

        coordinator = ProfileCoordinator(
            "main",
            blocked_switch,
            drain_timeout=0.05,
            transition_timeout=0.1,
        )

        def first_transition() -> None:
            try:
                with coordinator.lease("vision"):
                    pass
            except BaseException as error:
                first_errors.append(error)

        first = threading.Thread(target=first_transition)
        first.start()
        self.assertTrue(transition_started.wait(timeout=1))
        self.assertEqual(coordinator.active_requests, 0)

        try:
            with self.assertRaises(BrokerTransitionTimeoutError):
                coordinator.switch_when_idle("admin")
            self.assertEqual(transitions, ["vision"])
            self.assertEqual(coordinator.active_requests, 0)
        finally:
            allow_transition.set()
            first.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertTrue(
            all(isinstance(error, BrokerTransitionTimeoutError) for error in first_errors)
        )
        deadline = time.monotonic() + 1
        while coordinator.quiescing:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)
        self.assertFalse(coordinator.snapshot()["transitioning"])
        self.assertEqual(coordinator.active_requests, 0)

    def test_transition_timeout_quarantines_late_success_without_phantom_lease(self) -> None:
        transition_started = threading.Event()
        allow_transition = threading.Event()

        def blocked_switch(_profile: str) -> None:
            transition_started.set()
            allow_transition.wait(timeout=2)

        coordinator = ProfileCoordinator(
            "main",
            blocked_switch,
            drain_timeout=1,
            transition_timeout=0.1,
        )
        started = time.monotonic()
        try:
            with self.assertRaises(BrokerTransitionTimeoutError):
                with coordinator.lease("vision"):
                    self.fail("timed-out transition admitted a request lease")
            self.assertTrue(transition_started.is_set())
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(coordinator.quiescing)
            with self.assertRaises(BrokerQuiescingError):
                with coordinator.lease("main"):
                    self.fail("late transition admitted a same-profile lease")
        finally:
            allow_transition.set()

        deadline = time.monotonic() + 1
        while coordinator.quiescing:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)
        self.assertEqual(coordinator.active_profile, "vision")
        self.assertEqual(coordinator.active_requests, 0)

    def test_late_transient_success_defers_and_completes_primary_restore(self) -> None:
        transition_started = threading.Event()
        allow_transition = threading.Event()
        transitions: list[str] = []

        def blocked_switch(profile: str) -> None:
            transitions.append(profile)
            if profile == "vision":
                transition_started.set()
                allow_transition.wait(timeout=2)

        coordinator = ProfileCoordinator(
            "main",
            blocked_switch,
            drain_timeout=1,
            transition_timeout=0.1,
        )

        with self.assertRaises(BrokerTransitionTimeoutError):
            with coordinator.lease("vision"):
                self.fail("timed-out transient switch admitted a request lease")

        self.assertTrue(transition_started.is_set())
        coordinator.restore_when_idle("vision", "main")
        self.assertEqual(
            coordinator.snapshot()["pending_restore"], ("vision", "main")
        )
        allow_transition.set()

        deadline = time.monotonic() + 1
        while coordinator.active_profile != "main" or transitions != ["vision", "main"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

        self.assertFalse(coordinator.quiescing)
        self.assertEqual(coordinator.active_requests, 0)
        self.assertIsNone(coordinator.snapshot()["pending_restore"])

    def test_failed_deferred_restore_is_caught_and_logged(self) -> None:
        transition_started = threading.Event()
        allow_transition = threading.Event()
        restore_attempted = threading.Event()

        def switch(profile: str) -> None:
            if profile == "vision":
                transition_started.set()
                allow_transition.wait(timeout=2)
                return
            restore_attempted.set()
            raise BackendControllerError("restore failed")

        coordinator = ProfileCoordinator(
            "main",
            switch,
            drain_timeout=1,
            transition_timeout=0.1,
        )
        logs: list[str] = []

        with patch.object(llama_broker, "_log", logs.append):
            with self.assertRaises(BrokerTransitionTimeoutError):
                with coordinator.lease("vision"):
                    self.fail("timed-out transient switch admitted a lease")
            coordinator.restore_when_idle("vision", "main")
            allow_transition.set()
            self.assertTrue(restore_attempted.wait(timeout=1))

            deadline = time.monotonic() + 1
            while not any(
                "deferred restore" in message and "restore failed" in message
                for message in logs
            ):
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.005)

        self.assertTrue(
            any("deferred restore" in message and "restore failed" in message for message in logs)
        )
        self.assertIsNone(coordinator.active_profile)
        self.assertEqual(coordinator.active_requests, 0)

    def test_experimental_effort_scaling_uses_request_budget_and_alias(self) -> None:
        expected = {
            "minimal": 256,
            "low": 128,
            "medium": 256,
            "high": 384,
            "extra high": 512,
            "max": 8192,
        }
        for effort, budget in expected.items():
            with self.subTest(effort=effort):
                rewritten = apply_reasoning_policy(
                    {
                        "max_tokens": 8192,
                        "reasoning_effort": (
                            "min"
                            if effort == "minimal"
                            else "med"
                            if effort == "medium"
                            else effort
                        ),
                    },
                    runtime="llamacpp",
                    max_output_tokens=131072,
                    request_max_output_tokens=8192,
                    batch_size=256,
                    ubatch_size=64,
                )
                self.assertEqual(rewritten["reasoning_effort"], effort)
                self.assertEqual(rewritten["thinking_budget_tokens"], budget)

    def test_experimental_effort_scaling_ladder_is_monotonic(self) -> None:
        for limit, batch, ubatch in (
            (8192, 256, 64),
            (32768, 256, 64),
            (65536, 512, 128),
            (4096, 128, 32),
            (1024, 128, 32),
            (256, 128, 32),
        ):
            with self.subTest(limit=limit, batch=batch, ubatch=ubatch):
                budgets = []
                for effort in ("low", "medium", "high", "extra high", "max"):
                    rewritten = apply_reasoning_policy(
                        {"reasoning_effort": effort},
                        runtime="llamacpp",
                        max_output_tokens=limit,
                        batch_size=batch,
                        ubatch_size=ubatch,
                    )
                    budgets.append(rewritten["thinking_budget_tokens"])
                # low < medium < high < extra high < max, always.
                self.assertEqual(budgets, sorted(budgets))
                self.assertLess(budgets[-2], budgets[-1])
                # Every derived level stays strictly below the slot limit.
                for budget in budgets[:-1]:
                    self.assertLess(budget, limit)
                # minimal is the 1-batch anchor, capped by the limit.
                rewritten = apply_reasoning_policy(
                    {"reasoning_effort": "minimal"},
                    runtime="llamacpp",
                    max_output_tokens=limit,
                    batch_size=batch,
                    ubatch_size=ubatch,
                )
                self.assertEqual(
                    rewritten["thinking_budget_tokens"], min(batch, limit)
                )

    def test_experimental_effort_scaling_accepts_opencode_xhigh_alias(self) -> None:
        rewritten = apply_reasoning_policy(
            {
                "max_tokens": 8192,
                "reasoning_effort": "xhigh",
            },
            runtime="llamacpp",
            max_output_tokens=131072,
            request_max_output_tokens=8192,
            batch_size=256,
            ubatch_size=64,
        )
        self.assertEqual(rewritten["reasoning_effort"], "extra high")
        self.assertEqual(rewritten["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(rewritten["thinking_budget_tokens"], 512)

    def test_experimental_effort_scaling_preserves_explicit_budget(self) -> None:
        rewritten = apply_reasoning_policy(
            {
                "max_tokens": 8192,
                "reasoning_effort": "high",
                "thinking_budget_tokens": 777,
            },
            runtime="llamacpp",
            max_output_tokens=131072,
            request_max_output_tokens=8192,
            batch_size=256,
            ubatch_size=64,
        )
        self.assertEqual(rewritten["thinking_budget_tokens"], 777)

        self.assertEqual(
            apply_reasoning_policy({}, runtime="llamacpp", max_output_tokens=32768),
            {
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_effort": "none",
                "thinking_budget_tokens": 0,
            },
        )
        rewritten = apply_reasoning_policy(
            {"reasoning": {"effort": "low"}},
            runtime="llamacpp",
            max_output_tokens=32768,
            batch_size=256,
            ubatch_size=64,
        )
        self.assertEqual(
            rewritten,
            {
                "reasoning": {"effort": "low"},
                "chat_template_kwargs": {"enable_thinking": True},
                "thinking_budget_tokens": 512,
                "reasoning_effort": "low",
            },
        )
        self.assertEqual(
            apply_reasoning_policy(
                {"reasoning": {"effort": "high"}},
                runtime="llamacpp",
                max_output_tokens=32768,
                batch_size=256,
                ubatch_size=64,
            )["thinking_budget_tokens"],
            1536,
        )

    def test_zero_thinking_budget_disables_reasoning_safely(self) -> None:
        for payload in (
            {"thinking_budget_tokens": 0},
            {"reasoning_budget_tokens": 0},
            {"thinking_budget_tokens": 0, "reasoning_budget_tokens": 5},
            {"thinking_budget_tokens": 5, "reasoning_budget_tokens": 0},
        ):
            with self.subTest(payload=payload):
                rewritten = apply_reasoning_policy(
                    {"reasoning": {"effort": "high"}, **payload},
                    runtime="llamacpp",
                    max_output_tokens=32768,
                )

                self.assertEqual(rewritten["reasoning_effort"], "none")
                self.assertEqual(rewritten["thinking_budget_tokens"], 0)
                self.assertEqual(
                    rewritten["chat_template_kwargs"], {"enable_thinking": False}
                )

    def test_llamacpp_reasoning_policy_preserves_explicit_budget(self) -> None:
        payload = {
            "reasoning": {"effort": "high"},
            "thinking_budget_tokens": 2048,
        }

        rewritten = apply_reasoning_policy(
            payload, runtime="llamacpp", max_output_tokens=32768
        )

        self.assertEqual(rewritten["thinking_budget_tokens"], 2048)
        self.assertNotIn("reasoning_budget_tokens", rewritten)
        self.assertEqual(
            rewritten["chat_template_kwargs"], {"enable_thinking": True}
        )

    def test_llamacpp_reasoning_policy_preserves_explicit_budget_without_effort(self) -> None:
        rewritten = apply_reasoning_policy(
            {"reasoning_budget_tokens": 2048},
            runtime="llamacpp",
            max_output_tokens=32768,
        )

        self.assertEqual(rewritten["thinking_budget_tokens"], 2048)
        self.assertNotIn("reasoning_budget_tokens", rewritten)
        self.assertEqual(
            rewritten["chat_template_kwargs"], {"enable_thinking": True}
        )
        self.assertNotIn("reasoning_effort", rewritten)

    def test_llamacpp_reasoning_policy_maps_extended_efforts(self) -> None:
        expected = {
            "minimal": 256,
            "low": 512,
            "medium": 1024,
            "high": 1536,
            "extra high": 2048,
            "max": 32768,
        }
        for effort, budget in expected.items():
            with self.subTest(effort=effort):
                rewritten = apply_reasoning_policy(
                    {"reasoning_effort": effort},
                    runtime="llamacpp",
                    max_output_tokens=32768,
                    batch_size=256,
                    ubatch_size=64,
                )
                self.assertEqual(rewritten["reasoning_effort"], effort)
                self.assertEqual(rewritten["thinking_budget_tokens"], budget)
                self.assertEqual(
                    rewritten["chat_template_kwargs"], {"enable_thinking": True}
                )

    def test_reasoning_policy_metadata_is_safe_and_normalized(self) -> None:
        original = {
            "reasoning": {"effort": "ultra"},
            "messages": [{"role": "user", "content": "secret prompt"}],
        }
        rewritten = {
            **original,
            "reasoning_effort": "high",
            "thinking_budget_tokens": 32768,
            "chat_template_kwargs": {"enable_thinking": True},
        }

        self.assertEqual(
            reasoning_policy_metadata(original, rewritten, runtime="llamacpp"),
            {
                "runtime": "llamacpp",
                "effort": "high",
                "budget_present": False,
                "budget_tokens": 32768,
            },
        )
        self.assertIsNone(
            reasoning_policy_metadata(original, rewritten, runtime="vllm")
        )

    def test_reasoning_policy_does_not_touch_non_llamacpp_payloads(self) -> None:
        payload = {"reasoning": {"effort": "high"}}

        self.assertEqual(apply_reasoning_policy(payload, runtime="vllm"), payload)

    def test_llamacpp_reasoning_policy_honors_explicit_template_override(self) -> None:
        payload = {
            "reasoning": {"effort": "high"},
            "chat_template_kwargs": {"enable_thinking": False},
        }

        rewritten = apply_reasoning_policy(
            payload, runtime="llamacpp", max_output_tokens=32768
        )

        self.assertEqual(
            rewritten["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertNotIn("reasoning_budget_tokens", rewritten)

    def test_route_payload_uses_active_profile_when_model_is_absent(self) -> None:
        profile, payload = route_payload(
            self.registry_fixture(), {}, active_profile="main"
        )
        self.assertEqual(profile.id, "main")
        self.assertEqual(payload["model"], "main-27b")

    def test_route_payload_rejects_omitted_model_without_active_profile(self) -> None:
        with self.assertRaisesRegex(Exception, "no active model"):
            route_payload(self.registry_fixture(), {}, active_profile=None)

    def test_route_payload_rejects_unknown_explicit_model(self) -> None:
        with self.assertRaisesRegex(Exception, "unknown model"):
            route_payload(
                self.registry_fixture(),
                {"model": "unknown"},
                active_profile=None,
            )

    @staticmethod
    def registry_fixture() -> object:
        class Profile:
            id = "main"
            alias = "main-27b"

        class Registry:
            def get(self, profile_id: str) -> object:
                if profile_id != "main":
                    raise KeyError(profile_id)
                return Profile()

            def try_resolve(self, alias: object) -> object | None:
                return Profile() if alias == "main-27b" else None

        return Registry()


if __name__ == "__main__":
    unittest.main()
