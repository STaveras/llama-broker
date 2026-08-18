from __future__ import annotations

import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llama_broker.broker_profiles import ProfileRegistry
from llama_broker.backend_controller import BackendControllerError
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
    enforce_output_limit,
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
                "GGML_CUDA_DISABLE_GRAPHS": "1",
                "GGML_CUDA_GRAPH_OPT": "0",
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
                    "role": "primary",
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
        manifest_path = root / "profiles.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.registry = ProfileRegistry.from_path(manifest_path)
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
    ) -> None:
        self.broker.shutdown()
        self.broker.server_close()
        self.broker_thread.join(timeout=1)
        self.controller = FakeController("main")
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
            drain_timeout=drain_timeout,
            transition_timeout=transition_timeout,
            stop_timeout=stop_timeout,
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
        self.assertEqual(by_id["qwen3.5-9b-vision"]["max_output_tokens"], 32768)
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
            self.assertEqual(
                payload["operation"],
                {"status": "accepted", "target": "vision", "type": "switch"},
            )
            self.assertTrue(transition_started.wait(timeout=1))
        finally:
            release_transition.set()

        deadline = time.monotonic() + 1
        while self.controller.transitions != ["vision"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

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

    def test_health_uses_cached_state_while_controller_transition_lock_is_held(self) -> None:
        self._replace_broker(
            max_workers=2,
            control_workers=2,
            drain_timeout=0.1,
            transition_timeout=0.2,
            stop_timeout=1,
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
        finally:
            allow_transition.set()

        deadline = time.monotonic() + 1
        while self.broker.application.coordinator.quiescing:  # type: ignore[attr-defined]
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)


class CoordinatorTests(unittest.TestCase):
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

    def test_transient_lease_restores_primary_profile(self) -> None:
        self._assert_transient_round_trip("primary")

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
