from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from llama_broker import BrokerApplication, BrokerServer
from llama_broker.broker_profiles import ProfileError, ProfileRegistry
from llama_broker.broker_security import SecurityConfig
from llama_broker.system1 import (
    SYSTEM1_UNIT_NAME,
    System1Client,
    System1Config,
    System1ConfigError,
    System1Error,
    System1Router,
    System1RoutingConfig,
    System1Service,
    System1Sidecar,
    choice_probabilities,
    load_system1_config,
    request_state,
)
from tests import test_broker_http as _http


def _profile(profile_id: str, *capabilities: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=profile_id,
        alias=f"{profile_id}-alias",
        display_name=profile_id.title(),
        capabilities=capabilities or ("text",),
        context_size=32768,
    )


class FakeClient:
    def __init__(self, answers: object = None, error: Exception | None = None) -> None:
        self.answers = answers
        self.error = error
        self.calls: list[tuple[object, dict[str, object]]] = []

    def decide(self, state: object, questions: dict[str, object], **_: object) -> object:
        self.calls.append((state, questions))
        if self.error is not None:
            raise self.error
        return self.answers


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write(self, record: dict[str, object]) -> Path:
        path = self.root / "system1.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_missing_or_disabled_config_turns_system1_off(self) -> None:
        self.assertIsNone(load_system1_config(self.root / "absent.json"))
        self.assertIsNone(load_system1_config(self._write({"enabled": False})))
        self.assertIsNone(load_system1_config(self._write({})))

    def test_managed_backend_requires_binary_model_and_loopback_port(self) -> None:
        with self.assertRaisesRegex(System1ConfigError, "server_binary and model_path"):
            load_system1_config(self._write({"enabled": True}))
        base = {
            "enabled": True,
            "server_binary": "/opt/llama-server",
            "model_path": "/models/d1.gguf",
        }
        with self.assertRaisesRegex(System1ConfigError, "loopback"):
            load_system1_config(self._write({**base, "url": "http://0.0.0.0:10001"}))
        with self.assertRaisesRegex(System1ConfigError, "explicit port"):
            load_system1_config(self._write({**base, "url": "http://127.0.0.1"}))
        config = load_system1_config(self._write(base))
        assert config is not None
        self.assertEqual(config.listen_port, 10001)
        self.assertEqual(config.routing.aliases, ("auto",))

    def test_unmanaged_backend_only_needs_a_url(self) -> None:
        config = load_system1_config(
            self._write(
                {"enabled": True, "managed": False, "url": "http://127.0.0.1:9000/"}
            )
        )
        assert config is not None
        self.assertEqual(config.url, "http://127.0.0.1:9000")

    def test_routing_values_are_validated(self) -> None:
        with self.assertRaisesRegex(System1ConfigError, "min_confidence"):
            System1RoutingConfig.from_mapping({"min_confidence": 2})
        with self.assertRaisesRegex(System1ConfigError, "aliases"):
            System1RoutingConfig.from_mapping({"aliases": []})
        with self.assertRaisesRegex(System1ConfigError, "descriptions"):
            System1RoutingConfig.from_mapping({"descriptions": {"main": 3}})


class ParsingTests(unittest.TestCase):
    def test_choice_answer_shapes_are_normalised(self) -> None:
        expected = {"a": 0.7, "b": 0.3}
        self.assertEqual(choice_probabilities({"choice": expected}), expected)
        self.assertEqual(choice_probabilities(expected), expected)
        self.assertEqual(
            choice_probabilities({"choice": {"probabilities": expected}}), expected
        )
        self.assertEqual(
            choice_probabilities(
                [{"label": "a", "probability": 0.7}, {"label": "b", "probability": 0.3}]
            ),
            expected,
        )
        with self.assertRaises(System1Error):
            choice_probabilities({"choice": "a"})

    def test_request_state_uses_system_prompt_latest_user_turn_and_images(self) -> None:
        state, has_images = request_state(
            {
                "messages": [
                    {"role": "system", "content": "You are a coder."},
                    {"role": "user", "content": "old question"},
                    {"role": "assistant", "content": "old answer"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is in this photo?"},
                            {"type": "image_url", "image_url": {"url": "data:x"}},
                        ],
                    },
                ]
            },
            max_chars=4000,
        )
        self.assertTrue(has_images)
        self.assertIn("You are a coder.", state)
        self.assertIn("what is in this photo?", state)
        self.assertNotIn("old question", state)
        self.assertIn("includes images", state)

    def test_request_state_keeps_the_tail_of_long_requests(self) -> None:
        state, _ = request_state(
            {"prompt": "x" * 100 + "THE ASK"}, max_chars=20
        )
        self.assertEqual(len(state), 20)
        self.assertTrue(state.endswith("THE ASK"))


class RouterTests(unittest.TestCase):
    profiles = (_profile("main"), _profile("coder"), _profile("vision", "text", "vision"))

    def _decide(
        self,
        client: FakeClient,
        *,
        active: str | None = "main",
        payload: dict[str, object] | None = None,
        **config: object,
    ):
        router = System1Router(System1RoutingConfig.from_mapping(config), client)  # type: ignore[arg-type]
        return router.decide(
            payload or {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            self.profiles,
            active_profile=active,
            available=lambda _profile: True,
        )

    def test_routes_to_the_most_probable_profile(self) -> None:
        client = FakeClient({"route": {"choice": {"main": 0.1, "coder": 0.8, "vision": 0.1}}})
        decision = self._decide(client)
        self.assertEqual(decision.profile_id, "coder")
        self.assertEqual(decision.reason, "system1")
        question = client.calls[0][1]["route"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(set(question["criteria"]), {"main", "coder", "vision"})

    def test_keeps_the_active_profile_inside_the_switch_margin(self) -> None:
        client = FakeClient({"route": {"choice": {"main": 0.4, "coder": 0.5}}})
        decision = self._decide(client, min_confidence=0.3)
        self.assertEqual(decision.profile_id, "main")
        self.assertEqual(decision.reason, "kept_active_within_switch_margin")

    def test_low_confidence_and_errors_fall_back_to_the_active_profile(self) -> None:
        low = self._decide(FakeClient({"route": {"choice": {"main": 0.3, "coder": 0.35}}}))
        self.assertEqual(low.profile_id, "main")
        self.assertTrue(low.reason.startswith("fallback_active"))
        down = self._decide(FakeClient(error=System1Error("unreachable")))
        self.assertEqual(down.profile_id, "main")
        self.assertIn("unreachable", down.reason)
        refused = self._decide(FakeClient(error=System1Error("down")), fallback="error")
        self.assertIsNone(refused.profile_id)

    def test_active_fallback_is_fail_closed_when_no_candidate_can_serve_images(self) -> None:
        router = System1Router(System1RoutingConfig.from_mapping({}), FakeClient(error=System1Error("unreachable")))  # type: ignore[arg-type]
        decision = router.decide(
            {"model": "auto", "messages": [{"role": "user", "content": [{"type": "input_image", "image_url": {"url": "data:x"}}]}]},
            self.profiles[:2],
            active_profile="main",
            available=lambda _profile: True,
        )
        self.assertIsNone(decision.profile_id)

    def test_explicit_fallback_must_be_an_eligible_candidate(self) -> None:
        router = System1Router(System1RoutingConfig.from_mapping({"fallback": "vision"}), FakeClient(error=System1Error("unreachable")))  # type: ignore[arg-type]
        decision = router.decide(
            {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
            self.profiles,
            active_profile="main",
            available=lambda profile: profile.id != "vision",
        )
        self.assertIsNone(decision.profile_id)

    def test_images_restrict_candidates_to_vision_profiles(self) -> None:
        client = FakeClient()
        decision = self._decide(
            client,
            payload={
                "model": "auto",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "data:x"}}],
                    }
                ],
            },
        )
        self.assertEqual(decision.profile_id, "vision")
        self.assertEqual(decision.reason, "only_candidate")
        self.assertEqual(client.calls, [])

    def test_candidate_list_and_descriptions_come_from_config(self) -> None:
        client = FakeClient({"route": {"choice": {"main": 0.9, "coder": 0.1}}})
        self._decide(
            client,
            candidates=["main", "coder"],
            descriptions={"coder": "Best for programming."},
        )
        criteria = client.calls[0][1]["route"]["criteria"]
        self.assertEqual(set(criteria), {"main", "coder"})
        self.assertEqual(criteria["coder"], "Best for programming.")


class SidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.binary = root / "llama-server"
        self.binary.write_text("#!/bin/sh\n")
        self.binary.chmod(0o755)
        self.model = root / "d1-3B-Q4_K_M.gguf"
        self.model.write_bytes(b"gguf")
        self.config = System1Config.from_mapping(
            {
                "server_binary": str(self.binary),
                "model_path": str(self.model),
                "environment": {"CUDA_VISIBLE_DEVICES": "0"},
            }
        )
        self.commands: list[tuple[str, ...]] = []
        self.sidecar = System1Sidecar(
            self.config,
            command_runner=self.commands.append,
            systemd_dir=root / "systemd",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_unit_runs_d1_on_its_own_port_outside_the_profile_prefix(self) -> None:
        self.assertFalse(SYSTEM1_UNIT_NAME.startswith("llama-broker-profile-"))
        unit = self.sidecar.render_unit()
        self.assertIn('"--port" "10001"', unit)
        self.assertIn('"--alias" "d1-3B"', unit)
        self.assertIn(f'"{self.model}"', unit)
        self.assertIn('Environment="CUDA_VISIBLE_DEVICES=0"', unit)
        self.assertIn("Restart=on-failure", unit)

    def test_ensure_running_installs_once_then_only_starts(self) -> None:
        self.sidecar.ensure_running()
        self.assertEqual(
            self.commands,
            [
                ("systemctl", "--user", "daemon-reload"),
                ("systemctl", "--user", "restart", SYSTEM1_UNIT_NAME),
            ],
        )
        self.commands.clear()
        self.sidecar.ensure_running()
        self.assertEqual(
            self.commands, [("systemctl", "--user", "start", SYSTEM1_UNIT_NAME)]
        )

    def test_missing_model_refuses_to_start(self) -> None:
        self.model.unlink()
        with self.assertRaisesRegex(System1ConfigError, "model_path"):
            self.sidecar.ensure_running()
        self.assertEqual(self.commands, [])


class FakeD1Handler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.server.requests.append(  # type: ignore[attr-defined]
            (self.path, json.loads(body), dict(self.headers))
        )
        response = json.dumps({"answers": self.server.answers}).encode()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


class System1HttpTests(unittest.TestCase):
    setUp_base = _http.BrokerHttpTests.setUp
    tearDown_base = _http.BrokerHttpTests.tearDown
    _active_alias_ready = _http.BrokerHttpTests._active_alias_ready
    _post = _http.BrokerHttpTests._post
    _get = _http.BrokerHttpTests._get

    def setUp(self) -> None:
        self.setUp_base()
        self.d1 = ThreadingHTTPServer(("127.0.0.1", 0), FakeD1Handler)
        self.d1.requests = []  # type: ignore[attr-defined]
        self.d1.answers = {}  # type: ignore[attr-defined]
        threading.Thread(target=self.d1.serve_forever, daemon=True).start()
        config = System1Config.from_mapping(
            {"managed": False, "url": f"http://127.0.0.1:{self.d1.server_port}"}
        )
        self.broker.shutdown()
        self.broker.server_close()
        self.controller = _http.FakeController("main")
        application = BrokerApplication(
            registry=self.registry,
            controller=self.controller,
            backend_host="127.0.0.1",
            backend_port=self.backend.server_port,
            backend_timeout=10,
            max_body_bytes=1_000_000,
            security=SecurityConfig(api_key_file=self.auth_key_path),
            backend_alias_ready=self._active_alias_ready,
            system1=System1Service(config),
        )
        self.broker = BrokerServer(("127.0.0.1", 0), application=application, max_workers=4)
        threading.Thread(target=self.broker.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.d1.shutdown()
        self.d1.server_close()
        self.tearDown_base()

    def _post_with_headers(self, payload: dict[str, object]) -> tuple[int, dict[str, object], dict[str, str]]:
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
        response = connection.getresponse()
        decoded = json.loads(response.read())
        headers = dict(response.getheaders())
        connection.close()
        return response.status, decoded, headers

    def test_auto_model_is_routed_by_d1_to_a_real_profile(self) -> None:
        self.d1.answers = {"route": {"choice": {"main": 0.1, "vision": 0.9}}}  # type: ignore[attr-defined]
        status, response, headers = self._post_with_headers(
            {"model": "auto", "messages": [{"role": "user", "content": "describe a cat"}]}
        )
        self.assertEqual(status, 200)
        self.assertEqual(response["model"], "qwen3.5-9b-vision")
        self.assertEqual(headers["X-Llama-Broker-Profile"], "vision")
        self.assertEqual(headers["X-Llama-Broker-System1-Route"], "system1")
        path, request, d1_headers = self.d1.requests[0]  # type: ignore[attr-defined]
        self.assertEqual(path, "/v1/systemone")
        self.assertIn("describe a cat", request["state"])
        self.assertNotIn("Authorization", d1_headers)

    def test_auto_model_keeps_active_profile_when_d1_is_unsure(self) -> None:
        self.d1.answers = {"route": {"choice": {"main": 0.45, "vision": 0.55}}}  # type: ignore[attr-defined]
        status, response = self._post(
            {"model": "auto", "messages": [{"role": "user", "content": "hello"}]}
        )
        self.assertEqual(status, 200)
        self.assertEqual(response["model"], "main-27b")
        self.assertEqual(self.controller.transitions, [])

    def test_explicit_model_never_consults_d1(self) -> None:
        status, _ = self._post({"model": "main", "messages": []})
        self.assertEqual(status, 200)
        self.assertEqual(self.d1.requests, [])  # type: ignore[attr-defined]

    def test_systemone_endpoint_is_proxied_to_the_resident_backend(self) -> None:
        self.d1.answers = {"refund": {"noul": 0.97}}  # type: ignore[attr-defined]
        question = {
            "state": "I was charged twice",
            "questions": {
                "refund": {"type": "noul", "instructions": "Is this a refund request?"}
            },
        }
        status, response = self._post(question, path="/v1/systemone")
        self.assertEqual(status, 200)
        self.assertEqual(response, {"answers": {"refund": {"noul": 0.97}}})
        self.assertEqual(self.d1.requests[0][1], question)  # type: ignore[attr-defined]
        self.assertEqual(self.controller.transitions, [])

    def test_models_and_health_report_system1(self) -> None:
        status, models = self._get("/v1/models")
        self.assertEqual(status, 200)
        self.assertIn("auto", [model["id"] for model in models["data"]])
        status, health = self._get("/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["system1"]["enabled"])
        self.assertTrue(health["system1"]["ready"])
        self.assertEqual(health["system1"]["routing"]["aliases"], ["auto"])


class DisabledSystem1HttpTests(unittest.TestCase):
    setUp = _http.BrokerHttpTests.setUp
    tearDown = _http.BrokerHttpTests.tearDown
    _active_alias_ready = _http.BrokerHttpTests._active_alias_ready
    _post = _http.BrokerHttpTests._post
    _get = _http.BrokerHttpTests._get

    def test_disabled_system1_rejects_auto_and_systemone(self) -> None:
        status, _ = self._post({"model": "auto", "messages": []})
        self.assertEqual(status, 404)
        status, _ = self._post({"state": "x", "questions": {}}, path="/v1/systemone")
        self.assertEqual(status, 404)
        _, health = self._get("/health")
        self.assertEqual(health["system1"], {"enabled": False})


class ClientTests(unittest.TestCase):
    def test_unreachable_backend_raises_system1_error(self) -> None:
        client = System1Client("http://127.0.0.1:9", timeout=0.2)
        with self.assertRaises(System1Error):
            client.decide("x", {})
        self.assertFalse(client.healthy())


if __name__ == "__main__":
    unittest.main()


class ResidentProfileTests(unittest.TestCase):
    """d1 defined as a resident profile in profiles.json."""

    setUp_base = _http.BrokerHttpTests.setUp
    tearDown = _http.BrokerHttpTests.tearDown
    _active_alias_ready = _http.BrokerHttpTests._active_alias_ready
    _post = _http.BrokerHttpTests._post
    _get = _http.BrokerHttpTests._get

    def setUp(self) -> None:
        self.setUp_base()
        root = Path(self.tempdir.name)
        self.d1_model = root / "d1-3B-Q4_K_M.gguf"
        self.d1_model.write_bytes(b"d1")
        self.manifest = json.loads(self.profiles_path.read_text(encoding="utf-8"))
        main = self.manifest["profiles"][0]
        self.resident = {
            **main,
            "id": "d1-3b",
            "alias": "d1-3B",
            "aliases": ["d1-3B", "d1"],
            "display_name": "Liquid d1 3B",
            "role": "resident",
            "capabilities": ["decision"],
            "model_path": str(self.d1_model),
            "model_sha256": _http.sha256_file(self.d1_model),
            "context_size": 8192,
        }

    def _registry(self, *extra: dict[str, object]):
        manifest = {**self.manifest, "profiles": [*self.manifest["profiles"], *extra]}
        self.profiles_path.write_text(json.dumps(manifest), encoding="utf-8")
        return ProfileRegistry.from_path(self.profiles_path)

    def test_resident_profile_is_listed_apart_and_never_switchable(self) -> None:
        registry = self._registry(self.resident)
        self.assertEqual([p.id for p in registry.resident_profiles], ["d1-3b"])
        self.assertNotIn("d1-3b", [p.id for p in registry.profiles])
        self.assertIsNone(registry.try_resolve("d1"))
        self.assertEqual(registry.try_resolve_resident("D1").id, "d1-3b")

    def test_resident_role_and_decision_capability_go_together(self) -> None:
        for change in (
            {"capabilities": ["text"]},
            {"capabilities": ["decision", "text"]},
            {"role": "persistent"},
            {"restore_profile": "main"},
        ):
            with self.subTest(change=change), self.assertRaises(ProfileError):
                self._registry({**self.resident, **change})

    def test_resident_aliases_cannot_shadow_switchable_profiles(self) -> None:
        with self.assertRaisesRegex(ProfileError, "duplicate alias"):
            self._registry({**self.resident, "aliases": ["d1-3B", "main"]})

    def test_resident_profile_enables_system1_without_a_config_file(self) -> None:
        registry = self._registry(self.resident)
        root = Path(self.tempdir.name)
        config = load_system1_config(
            root / "absent.json", resident_profiles=registry.resident_profiles
        )
        assert config is not None
        self.assertEqual((config.profile, config.model), ("d1-3b", "d1-3B"))
        self.assertFalse(config.routing.enabled)
        routed = root / "routed.json"
        routed.write_text(json.dumps({"routing": {"enabled": True}}), encoding="utf-8")
        tuned = load_system1_config(routed, resident_profiles=registry.resident_profiles)
        assert tuned is not None
        self.assertEqual(tuned.profile, "d1-3b")
        self.assertTrue(tuned.routing.enabled)
        off = root / "off.json"
        off.write_text(json.dumps({"enabled": False}), encoding="utf-8")
        self.assertIsNone(
            load_system1_config(off, resident_profiles=registry.resident_profiles)
        )
        wrong = root / "wrong.json"
        wrong.write_text(json.dumps({"enabled": True, "profile": "main"}), encoding="utf-8")
        with self.assertRaisesRegex(System1ConfigError, "not a resident profile"):
            load_system1_config(wrong, resident_profiles=registry.resident_profiles)
        second = {**self.resident, "id": "d1-b", "alias": "d1-b", "aliases": ["d1-b"]}
        two = self._registry(self.resident, second)
        with self.assertRaisesRegex(System1ConfigError, "several resident profiles"):
            load_system1_config(root / "absent.json", resident_profiles=two.resident_profiles)

    def test_sidecar_launches_the_resident_profile_with_its_pins(self) -> None:
        registry = self._registry(self.resident)
        profile = registry.resident_profiles[0]
        config = load_system1_config(
            Path(self.tempdir.name) / "absent.json",
            resident_profiles=registry.resident_profiles,
        )
        assert config is not None
        verified: list[str] = []
        commands: list[tuple[str, ...]] = []
        sidecar = System1Sidecar(
            config,
            command_runner=commands.append,
            systemd_dir=Path(self.tempdir.name) / "systemd",
            profile=profile,
            artifact_verifier=lambda p: verified.append(p.id),
        )
        unit = sidecar.render_unit()
        self.assertIn(f'"{self.d1_model}"', unit)
        self.assertIn('"--alias" "d1-3B"', unit)
        self.assertIn('"--port" "10001"', unit)
        self.assertIn(f"MemoryMax={profile.safety.service.memory_max_mib}M", unit)
        sidecar.ensure_running()
        self.assertEqual(verified, ["d1-3b"])
        self.assertEqual(commands[-1], ("systemctl", "--user", "restart", SYSTEM1_UNIT_NAME))

    def test_http_lists_resident_and_points_completions_at_systemone(self) -> None:
        registry = self._registry(self.resident)
        config = load_system1_config(
            Path(self.tempdir.name) / "absent.json",
            resident_profiles=registry.resident_profiles,
        )
        assert config is not None
        self.broker.shutdown()
        self.broker.server_close()
        self.registry = registry
        self.controller = _http.FakeController("main")
        application = BrokerApplication(
            registry=registry,
            controller=self.controller,
            backend_host="127.0.0.1",
            backend_port=self.backend.server_port,
            backend_timeout=10,
            max_body_bytes=1_000_000,
            security=SecurityConfig(api_key_file=self.auth_key_path),
            backend_alias_ready=self._active_alias_ready,
            system1=System1Service(config),
        )
        self.broker = BrokerServer(("127.0.0.1", 0), application=application, max_workers=4)
        threading.Thread(target=self.broker.serve_forever, daemon=True).start()

        status, models = self._get("/v1/models")
        self.assertEqual(status, 200)
        resident = [m for m in models["data"] if m["id"] == "d1-3B"]
        self.assertEqual(len(resident), 1)
        self.assertEqual(resident[0]["capabilities"], ["decision"])
        self.assertEqual(resident[0]["endpoints"], ["/v1/decisions", "/v1/systemone"])
        status, body = self._post({"model": "d1", "messages": []})
        self.assertEqual(status, 400)
        self.assertIn("/v1/decisions", body["error"]["message"])
        # Any resident alias names the decision model, so the request gets
        # past model resolution to the d1 backend (absent here: 502).
        question = {"type": "predicate", "name": "q", "instructions": "?"}
        status, body = self._post(
            {"model": "d1", "input": "x", "questions": [question]}, path="/v1/decisions"
        )
        self.assertNotEqual(status, 404, body)
        self.assertEqual(self.controller.transitions, [])
