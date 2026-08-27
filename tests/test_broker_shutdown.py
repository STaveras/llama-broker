from __future__ import annotations

import threading
import time
import unittest
import tempfile
import signal
import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llama_broker.daemon as llama_broker
from llama_broker import (
    BrokerAlreadyRunningError,
    BrokerApplication,
    BrokerQuiescingError,
    ProfileCoordinator,
    _readiness_probe,
    acquire_process_instance_lock,
    release_process_instance_lock,
)


class BlockingController:
    def __init__(self) -> None:
        self.stop_entered = threading.Event()
        self.stop_release = threading.Event()
        self.stops = 0
        self._lock = threading.Lock()

    def state(self) -> dict[str, object]:
        return {"generation": 1, "profile": "main", "phase": "active"}

    def switch(self, profile: str) -> dict[str, object]:
        return {"generation": 2, "profile": profile, "phase": "active"}

    def stop_all(self) -> dict[str, object]:
        with self._lock:
            self.stops += 1
        self.stop_entered.set()
        if not self.stop_release.wait(timeout=2):
            raise TimeoutError("test did not release stop_all")
        return {"generation": 2, "profile": None, "phase": "inactive"}


def make_application(controller: BlockingController) -> BrokerApplication:
    return BrokerApplication(
        registry=object(),  # type: ignore[arg-type]
        controller=controller,
        backend_host="127.0.0.1",
        backend_port=10000,
        backend_timeout=0.1,
        max_body_bytes=1024,
        backend_ready=lambda: True,
        drain_timeout=0.2,
        transition_timeout=1,
        stop_timeout=1,
    )


class BrokerApplicationShutdownTests(unittest.TestCase):
    def test_shutdown_is_synchronous_idempotent_and_preserves_runtime(self) -> None:
        controller = BlockingController()
        application = make_application(controller)

        application.shutdown()
        application.shutdown()

        self.assertEqual(controller.stops, 0)

    def test_server_construction_failure_handoffs_without_stopping_backend(self) -> None:
        lock = object()
        controller = BlockingController()
        application = make_application(controller)
        with (
            patch.object(
                llama_broker, "acquire_process_instance_lock", return_value=lock
            ),
            patch.object(
                llama_broker,
                "create_production_application",
                return_value=(application, object()),
            ),
            patch.object(
                llama_broker,
                "BrokerServer",
                side_effect=OSError("listener bind failed"),
            ),
            patch.object(
                llama_broker, "load_security_config", return_value=SimpleNamespace()
            ),
            patch.object(
                llama_broker, "stop_production_units_after_startup_failure"
            ) as emergency_stop,
            patch.object(llama_broker, "release_process_instance_lock") as release,
        ):
            self.assertEqual(llama_broker.main(), 1)

        self.assertEqual(controller.stops, 0)
        emergency_stop.assert_not_called()
        release.assert_called_once_with(lock)


class CoordinatorShutdownTests(unittest.TestCase):
    def test_failed_transition_clears_active_profile_without_restore(self) -> None:
        def fail_switch(_profile: str) -> None:
            raise RuntimeError("target transition failed closed")

        coordinator = ProfileCoordinator(
            "main",
            fail_switch,
            drain_timeout=0.1,
            transition_timeout=0.5,
        )

        with self.assertRaises(RuntimeError):
            coordinator.switch_when_idle("vision")

        self.assertIsNone(coordinator.active_profile)
        self.assertIsNone(coordinator.snapshot()["profile"])

    def test_handoff_preserves_active_profile_without_stopping_runtime(self) -> None:
        stops: list[str] = []
        coordinator = ProfileCoordinator(
            "main",
            lambda _profile: None,
            lambda: stops.append("stopped"),
            drain_timeout=0.1,
            transition_timeout=0.1,
        )

        coordinator.handoff_for_process_shutdown()

        self.assertEqual(stops, [])
        self.assertEqual(coordinator.active_profile, "main")
        self.assertTrue(coordinator.snapshot()["shutting_down"])
        with self.assertRaises(BrokerQuiescingError):
            with coordinator.lease_active():
                self.fail("lease entered after control-plane handoff")

    def test_shutdown_gate_rejects_new_requests_and_transitions(self) -> None:
        stop_entered = threading.Event()
        stop_release = threading.Event()
        stops: list[str] = []
        errors: list[BaseException] = []

        def stop_backend() -> None:
            stops.append("stopped")
            stop_entered.set()
            if not stop_release.wait(timeout=2):
                raise TimeoutError("test did not release shutdown stop")

        coordinator = ProfileCoordinator(
            "main",
            lambda _profile: None,
            stop_backend,
            drain_timeout=0.2,
            transition_timeout=1,
            stop_timeout=1,
        )

        def shutdown() -> None:
            try:
                coordinator.stop_for_process_shutdown()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=shutdown)
        thread.start()
        self.assertTrue(stop_entered.wait(timeout=1))
        self.assertTrue(coordinator.snapshot()["shutting_down"])

        try:
            with self.assertRaises(BrokerQuiescingError):
                with coordinator.lease("main"):
                    self.fail("same-profile request entered after shutdown began")
            with self.assertRaises(BrokerQuiescingError):
                with coordinator.lease("vision"):
                    self.fail("cross-profile request entered after shutdown began")
            with self.assertRaises(BrokerQuiescingError):
                with coordinator.lease_active():
                    self.fail("active-profile request entered after shutdown began")
            with self.assertRaises(BrokerQuiescingError):
                coordinator.switch_when_idle("vision")
            with self.assertRaises(BrokerQuiescingError):
                coordinator.stop_when_idle()
            with self.assertRaises(BrokerQuiescingError):
                coordinator.restore_when_idle("main", "vision")
        finally:
            stop_release.set()
            thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(stops, ["stopped"])
        self.assertIsNone(coordinator.active_profile)

    def test_transition_completing_during_shutdown_cannot_grant_lease(self) -> None:
        transition_entered = threading.Event()
        transition_release = threading.Event()
        lease_granted = threading.Event()
        stops: list[str] = []
        request_errors: list[BaseException] = []
        shutdown_errors: list[BaseException] = []

        def switch_profile(_profile: str) -> None:
            transition_entered.set()
            if not transition_release.wait(timeout=2):
                raise TimeoutError("test did not release profile transition")

        coordinator = ProfileCoordinator(
            "main",
            switch_profile,
            lambda: stops.append("stopped"),
            drain_timeout=0.2,
            transition_timeout=1,
            stop_timeout=1,
        )

        def request() -> None:
            try:
                with coordinator.lease("vision"):
                    lease_granted.set()
            except BaseException as error:
                request_errors.append(error)

        def shutdown() -> None:
            try:
                coordinator.stop_for_process_shutdown()
            except BaseException as error:
                shutdown_errors.append(error)

        request_thread = threading.Thread(target=request)
        shutdown_thread = threading.Thread(target=shutdown)
        request_thread.start()
        self.assertTrue(transition_entered.wait(timeout=1))
        shutdown_thread.start()

        deadline = time.monotonic() + 1
        while not coordinator.snapshot()["shutting_down"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

        transition_release.set()
        request_thread.join(timeout=1)
        shutdown_thread.join(timeout=1)

        self.assertFalse(request_thread.is_alive())
        self.assertFalse(shutdown_thread.is_alive())
        self.assertFalse(lease_granted.is_set())
        self.assertEqual(shutdown_errors, [])
        self.assertEqual(len(request_errors), 1)
        self.assertIsInstance(request_errors[0], BrokerQuiescingError)
        self.assertEqual(coordinator.active_requests, 0)
        self.assertIsNone(coordinator.active_profile)
        self.assertEqual(stops, ["stopped"])


class BrokerServiceShutdownPolicyTests(unittest.TestCase):
    def test_source_unit_preserves_cooperative_shutdown_window(self) -> None:
        unit_path = (
            Path(__file__).resolve().parents[1]
            / "docs/examples/llama-broker.service.in"
        )
        service_values: dict[str, str] = {}
        section = ""
        for raw_line in unit_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line
            elif section == "[Service]" and line and not line.startswith("#"):
                key, separator, value = line.partition("=")
                if separator:
                    service_values[key] = value

        self.assertEqual(service_values.get("TimeoutStopSec"), "600")
        self.assertEqual(service_values.get("KillMode"), "mixed")
        self.assertEqual(service_values.get("SendSIGKILL"), "no")


class BrokerProcessLifecycleTests(unittest.TestCase):
    def test_daemon_help_does_not_start_or_acquire_live_instance(self) -> None:
        output = io.StringIO()
        with patch.object(sys, "argv", ["llama-broker", "--help"]), patch.object(
            llama_broker, "acquire_process_instance_lock"
        ) as acquire, contextlib.redirect_stdout(output):
            result = llama_broker.main()

        self.assertEqual(result, 0)
        acquire.assert_not_called()
        self.assertIn("usage: llama-broker", output.getvalue())

    def test_linux_process_lock_rejects_second_owner_before_lifecycle_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broker.instance.lock"
            first = acquire_process_instance_lock(path)
            try:
                with self.assertRaises(BrokerAlreadyRunningError):
                    acquire_process_instance_lock(path)
            finally:
                release_process_instance_lock(first)

            replacement = acquire_process_instance_lock(path)
            release_process_instance_lock(replacement)

    def test_lock_contention_never_stops_the_existing_broker_backend(self) -> None:
        with (
            patch.object(
                llama_broker,
                "acquire_process_instance_lock",
                side_effect=BrokerAlreadyRunningError("already running"),
            ),
            patch.object(llama_broker, "create_production_application") as create,
            patch.object(
                llama_broker, "stop_production_units_after_startup_failure"
            ) as emergency_stop,
        ):
            self.assertEqual(llama_broker.main(), 1)

        create.assert_not_called()
        emergency_stop.assert_not_called()

    def test_startup_receives_signal_event_and_runs_emergency_cleanup(self) -> None:
        observed_event: list[threading.Event] = []
        lock = object()

        def interrupted_factory(
            shutdown_requested: threading.Event | None = None,
        ) -> tuple[BrokerApplication, object]:
            self.assertIsNotNone(shutdown_requested)
            assert shutdown_requested is not None
            observed_event.append(shutdown_requested)
            shutdown_requested.set()
            raise BrokerQuiescingError("startup interrupted")

        with (
            patch.object(
                llama_broker, "acquire_process_instance_lock", return_value=lock
            ),
            patch.object(
                llama_broker,
                "create_production_application",
                side_effect=interrupted_factory,
            ),
            patch.object(
                llama_broker, "stop_production_units_after_startup_failure"
            ) as emergency_stop,
            patch.object(llama_broker, "release_process_instance_lock") as release,
        ):
            self.assertEqual(llama_broker.main(), 0)

        self.assertEqual(len(observed_event), 1)
        self.assertTrue(observed_event[0].is_set())
        emergency_stop.assert_called_once_with()
        release.assert_called_once_with(lock)

    def test_interrupted_startup_probe_cancels_before_backend_poll(self) -> None:
        shutdown_requested = threading.Event()
        shutdown_requested.set()

        with patch.object(llama_broker.http.client, "HTTPConnection") as connection:
            with self.assertRaises(BrokerQuiescingError):
                _readiness_probe(
                    "main-27b", 300, shutdown_requested=shutdown_requested
                )

        connection.assert_not_called()

    def test_lock_release_failure_cannot_mask_startup_or_skip_signal_restore(self) -> None:
        lock = object()
        with (
            patch.object(llama_broker.signal, "getsignal", return_value=signal.SIG_DFL),
            patch.object(llama_broker.signal, "signal") as set_signal,
            patch.object(
                llama_broker, "acquire_process_instance_lock", return_value=lock
            ),
            patch.object(
                llama_broker,
                "create_production_application",
                side_effect=RuntimeError("startup root cause"),
            ),
            patch.object(
                llama_broker, "stop_production_units_after_startup_failure"
            ) as emergency_stop,
            patch.object(
                llama_broker,
                "release_process_instance_lock",
                side_effect=OSError("unlock failed"),
            ),
        ):
            self.assertEqual(llama_broker.main(), 1)

        emergency_stop.assert_called_once_with()
        self.assertEqual(set_signal.call_count, 4)

    def test_emergency_cleanup_rejects_surviving_private_endpoint(self) -> None:
        with (
            patch.object(
                llama_broker, "systemd_service_inventory_probe", return_value=()
            ),
            patch.object(
                llama_broker,
                "systemd_service_state_probe",
                return_value=SimpleNamespace(is_quiescent=True),
            ),
            patch.object(llama_broker, "backend_endpoint_ready", return_value=True),
            patch.object(llama_broker, "subprocess_command_runner") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "private-endpoint"):
                llama_broker.stop_production_units_after_startup_failure()

        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
