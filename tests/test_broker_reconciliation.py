from __future__ import annotations

import unittest
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from llama_broker import BrokerApplication, backend_serves_alias


class ModelsHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        payload = json.dumps(
            {"data": [{"id": self.server.served_alias}]}  # type: ignore[attr-defined]
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@dataclass(frozen=True)
class FakeProfile:
    id: str
    alias: str


class FakeRegistry:
    def __init__(self) -> None:
        self._profiles = {
            "main": FakeProfile("main", "main-27b"),
            "vision": FakeProfile("vision", "qwen3.5-9b-vision"),
        }

    def get(self, profile_id: str) -> FakeProfile:
        return self._profiles[profile_id]


class FailingLifecycleController:
    def __init__(self, failure_state: dict[str, object]) -> None:
        self.current_state: dict[str, object] = {
            "generation": 1,
            "profile": "main",
            "phase": "active",
        }
        self.failure_state = failure_state
        self.fail_state_reads = False

    def state(self) -> dict[str, object]:
        if self.fail_state_reads:
            raise RuntimeError("state unavailable")
        return dict(self.current_state)

    def switch(self, _profile_id: str) -> dict[str, object]:
        self.current_state = dict(self.failure_state)
        raise RuntimeError("switch failed")

    def stop_all(self) -> dict[str, object]:
        self.current_state = dict(self.failure_state)
        raise RuntimeError("stop failed")


def make_application(controller: FailingLifecycleController) -> BrokerApplication:
    return BrokerApplication(
        registry=FakeRegistry(),  # type: ignore[arg-type]
        controller=controller,
        backend_host="127.0.0.1",
        backend_port=10000,
        backend_timeout=0.1,
        max_body_bytes=1024,
        backend_ready=lambda: True,
        backend_alias_ready=lambda alias: alias == "main-27b",
        drain_timeout=0.1,
        transition_timeout=1,
        stop_timeout=1,
    )


class BrokerLifecycleReconciliationTests(unittest.TestCase):
    def test_rollback_failure_quarantines_cached_and_coordinator_profile(self) -> None:
        controller = FailingLifecycleController(
            {
                "generation": 1,
                "profile": None,
                "phase": "rollback_failed",
                "prior_profile": "main",
                "target_profile": "vision",
            }
        )
        application = make_application(controller)

        with self.assertRaisesRegex(RuntimeError, "switch failed"):
            application.coordinator.switch_when_idle("vision")

        health = application.health()
        self.assertEqual(health["status"], "degraded")
        self.assertEqual(health["backend"]["phase"], "rollback_failed")
        self.assertIsNone(application.coordinator.active_profile)


    def test_failed_transition_does_not_trust_previous_profile_state(self) -> None:
        controller = FailingLifecycleController(
            {
                "generation": 2,
                "profile": None,
                "phase": "failed",
                "prior_profile": "main",
                "target_profile": "vision",
            }
        )
        application = make_application(controller)

        with self.assertRaisesRegex(RuntimeError, "switch failed"):
            application.coordinator.switch_when_idle("vision")

        self.assertIsNone(application.coordinator.active_profile)
        self.assertEqual(application.health()["status"], "degraded")

    def test_partial_stop_failure_cannot_leave_old_profile_healthy(self) -> None:
        controller = FailingLifecycleController(
            {
                "generation": 1,
                "profile": None,
                "phase": "stop_failed",
                "previous_profile": "main",
                "failed_profiles": ["vision"],
            }
        )
        application = make_application(controller)

        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            application.coordinator.stop_when_idle()

        health = application.health()
        self.assertEqual(health["status"], "degraded")
        self.assertEqual(health["backend"]["phase"], "stop_failed")
        self.assertIsNone(application.coordinator.active_profile)

    def test_unreadable_state_after_failure_is_unknown_and_quarantined(self) -> None:
        controller = FailingLifecycleController(
            {"generation": 1, "profile": "main", "phase": "active"}
        )
        application = make_application(controller)
        controller.fail_state_reads = True

        with self.assertRaisesRegex(RuntimeError, "switch failed"):
            application.coordinator.switch_when_idle("vision")

        health = application.health()
        self.assertEqual(health["status"], "degraded")
        self.assertEqual(health["backend"]["phase"], "unknown")
        self.assertIsNone(application.coordinator.active_profile)


class BackendAliasProbeTests(unittest.TestCase):
    def test_tcp_ready_backend_must_advertise_the_exact_alias(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), ModelsHandler)
        server.served_alias = "qwen3.5-9b-vision"  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.assertTrue(
                backend_serves_alias(
                    "127.0.0.1", server.server_port, "qwen3.5-9b-vision", 0.5
                )
            )
            self.assertFalse(
                backend_serves_alias(
                    "127.0.0.1", server.server_port, "main-27b", 0.5
                )
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
