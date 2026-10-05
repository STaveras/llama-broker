import json
import subprocess
import tempfile
import unittest
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llama_broker import backend_controller

from llama_broker.backend_controller import (
    ActivationPreemptedError,
    AdmissionError,
    BackendController,
    CommandError,
    ConfigurationError,
    LifecycleCancelledError,
    QuiescenceError,
    ResourceSnapshot,
    RollbackError,
    ServiceStateSnapshot,
    ServiceOwnershipError,
    SwitchError,
    systemd_service_inventory_probe,
    systemd_service_state_probe,
)
from llama_broker.broker_profiles import AdmissionSafetyBudget, SafetyBudget, ServiceSafetyBudget
from llama_broker.daemon import _prepare_durable_state
from llama_broker.transition_operations import PersistentTransitionQueue


def fake_safety() -> SafetyBudget:
    return SafetyBudget(
        service=ServiceSafetyBudget(
            service_type="exec",
            memory_max_mib=20_480,
            memory_swap_max_mib=1_024,
            tasks_max=512,
            kill_signal="SIGINT",
            timeout_stop_sec=20,
        ),
        admission=AdmissionSafetyBudget(
            min_ram_available_mib=8_192,
            max_swap_used_mib=1_024,
            max_swap_growth_mib=128,
            min_total_vram_mib=12_000,
            min_free_vram_before_start_mib=8_192,
            driver_vram_reserve_mib=512,
            vision_vram_reserve_mib=1_536,
            safety_vram_reserve_mib=1_024,
        ),
    )


@dataclass(frozen=True)
class FakeProfile:
    id: str
    alias: str
    display_name: str
    server_binary: str
    model_path: str
    args: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    startup_timeout: float = 30.0
    safety: SafetyBudget = field(default_factory=fake_safety)
    runtime_build: str | None = None
    server_sha256: str | None = None


class Harness:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.probes: list[tuple[str, float]] = []
        self.command_failures: set[tuple[str, ...]] = set()
        self.probe_results: dict[str, bool] = {}
        self.resource_snapshots: list[ResourceSnapshot] = []
        self.resource_probe_calls = 0
        self.resource_probe_error: BaseException | None = None
        self.events: list[dict[str, object]] = []
        self.service_state_calls: list[str] = []
        self.service_states: dict[str, ServiceStateSnapshot] = {}
        self.service_inventory_calls: list[str] = []
        self.service_inventory_units: set[str] = set()
        self.service_inventory_error: BaseException | None = None
        self.quiesce_on_stop: set[str] = set()

    def run(self, command: tuple[str, ...]) -> None:
        command = tuple(command)
        self.commands.append(command)
        if command in self.command_failures:
            raise RuntimeError("injected command failure")
        if command[:3] == ("systemctl", "--user", "start"):
            self.service_states[command[3]] = ServiceStateSnapshot(
                active_state="active",
                sub_state="running",
                main_pid=4242,
                control_pid=0,
            )
        if command[:3] == ("systemctl", "--user", "stop"):
            state = self.service_states.get(command[3])
            automatically_started = (
                state is not None
                and state.active_state == "active"
                and state.sub_state == "running"
                and state.main_pid == 4242
            )
            if command[3] not in self.quiesce_on_stop and not automatically_started:
                return
            self.service_states[command[3]] = ServiceStateSnapshot(
                active_state="inactive",
                sub_state="dead",
                main_pid=0,
                control_pid=0,
            )

    def probe(self, alias: str, timeout: float) -> bool:
        self.probes.append((alias, timeout))
        return self.probe_results.get(alias, True)

    def resources(self) -> ResourceSnapshot:
        self.resource_probe_calls += 1
        if self.resource_probe_error is not None:
            raise self.resource_probe_error
        if self.resource_snapshots:
            return self.resource_snapshots.pop(0)
        return ResourceSnapshot(
            ram_available_mib=24_000,
            swap_used_mib=0,
            vram_total_mib=12_227,
            vram_free_mib=11_000,
        )

    def log_event(self, event: dict[str, object]) -> None:
        self.events.append(dict(event))

    def service_state(self, unit_name: str) -> ServiceStateSnapshot:
        self.service_state_calls.append(unit_name)
        return self.service_states.get(
            unit_name,
            ServiceStateSnapshot(
                active_state="inactive",
                sub_state="dead",
                main_pid=0,
                control_pid=0,
            ),
        )

    def service_inventory(self, service_prefix: str) -> tuple[str, ...]:
        self.service_inventory_calls.append(service_prefix)
        if self.service_inventory_error is not None:
            raise self.service_inventory_error
        return tuple(sorted(self.service_inventory_units))


class BackendControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.systemd_dir = self.root / "systemd"
        self.state_path = self.root / "state.json"
        self.harness = Harness()
        self.main = FakeProfile(
            id="main-27b",
            alias="main-27b",
            display_name="Main 27B",
            server_binary="/opt/llama cpp/llama-server",
            model_path="/models/Main 27B.gguf",
            args=("-c", "65536", "-ngl", "auto"),
            environment={
                "GGML_CUDA_DISABLE_GRAPHS": "1",
                "GGML_CUDA_GRAPH_OPT": "0",
                "LITERAL_DOLLAR": "$HOME",
                "PERCENT_VALUE": "95%",
            },
            startup_timeout=90.0,
        )
        self.vision = FakeProfile(
            id="qwen-vision",
            alias="qwen3.5-9b-vision",
            display_name="Qwen vision",
            server_binary="/opt/llama-server",
            model_path="/models/qwen.gguf",
            args=("--mmproj", "/models/mmproj-F16.gguf"),
            environment={
                "GGML_CUDA_DISABLE_GRAPHS": "1",
                "GGML_CUDA_GRAPH_OPT": "0",
            },
            startup_timeout=45.0,
        )
        self.profiles = {profile.id: profile for profile in (self.main, self.vision)}
        self.controller = BackendController(
            profiles=self.profiles,
            state_path=self.state_path,
            systemd_dir=self.systemd_dir,
            service_prefix="llama-broker-profile",
            command_runner=self.harness.run,
            readiness_probe=self.harness.probe,
            resource_probe=self.harness.resources,
            event_logger=self.harness.log_event,
            service_state_probe=self.harness.service_state,
            service_inventory_probe=self.harness.service_inventory,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def read_state(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def test_replace_profiles_refreshes_readiness_alias_mapping(self) -> None:
        class AliasProbe:
            def __init__(self, aliases: set[str]) -> None:
                self.aliases = aliases

            def __call__(self, alias: str, _timeout: float) -> bool:
                return alias in self.aliases

            def replace_profiles(self, profiles) -> None:
                self.aliases = {profile.alias for profile in profiles.values()}

        old_profile = self.main
        new_profile = replace(old_profile, alias="qwen3.8-flash-next-iq2-xs")
        probe = AliasProbe({old_profile.alias})
        self.controller.readiness_probe = probe

        self.controller.replace_profiles({new_profile.id: new_profile})

        self.assertTrue(probe(new_profile.alias, 0.1))

    def test_recovery_from_corrupt_state_uses_actual_active_unit(self) -> None:
        main_unit = self.controller.unit_name("main-27b")
        self.harness.service_inventory_units.add(main_unit)
        self.harness.service_states[main_unit] = ServiceStateSnapshot(
            active_state="active",
            sub_state="running",
            main_pid=4242,
            control_pid=0,
        )
        self.harness.probe_results[self.main.alias] = True
        self.state_path.write_text("{corrupt state", encoding="utf-8")

        recovered = self.controller.reconcile_from_runtime()

        self.assertEqual(recovered["phase"], "active")
        self.assertEqual(recovered["profile"], "main-27b")
        self.assertEqual(recovered["generation"], 0)
        self.assertEqual(self.harness.commands, [])

    def test_recovery_from_ambiguous_active_units_fails_stale_without_stopping(self) -> None:
        main_unit = self.controller.unit_name("main-27b")
        vision_unit = self.controller.unit_name("qwen-vision")
        self.harness.service_inventory_units.update({main_unit, vision_unit})
        for unit in (main_unit, vision_unit):
            self.harness.service_states[unit] = ServiceStateSnapshot(
                active_state="active",
                sub_state="running",
                main_pid=4242,
                control_pid=0,
            )
        self.state_path.write_text("{corrupt state", encoding="utf-8")

        recovered = self.controller.reconcile_from_runtime()

        self.assertEqual(recovered["phase"], "stale")
        self.assertIsNone(recovered["profile"])
        self.assertEqual(self.harness.commands, [])

    def test_failed_unit_without_process_is_quiescent(self) -> None:
        state = ServiceStateSnapshot(
            active_state="failed",
            sub_state="failed",
            main_pid=0,
            control_pid=0,
        )

        self.assertTrue(state.is_quiescent)

    def test_startup_recovery_backs_up_corrupt_state_and_reconciles_runtime(self) -> None:
        main_unit = self.controller.unit_name("main-27b")
        self.harness.service_inventory_units.add(main_unit)
        self.harness.service_states[main_unit] = ServiceStateSnapshot(
            active_state="active",
            sub_state="running",
            main_pid=4242,
            control_pid=0,
        )
        self.harness.probe_results[self.main.alias] = True
        corrupt = b'{"phase":"broken"\n'
        self.state_path.write_bytes(corrupt)

        repaired = _prepare_durable_state(self.controller)

        self.assertTrue(repaired)
        backups = list(self.root.glob("state.json.corrupt.*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), corrupt)
        state = self.read_state()
        self.assertEqual((state["phase"], state["profile"]), ("active", "main-27b"))
        self.assertTrue(state["operations"]["reconciliation_required"])
        self.assertIn("resubmitted", state["recovery"]["warning"])
        self.assertEqual(self.harness.commands, [])

    def test_startup_recovery_discards_unsupported_operation_schema(self) -> None:
        self.controller._write_state(
            {"generation": 9, "profile": "main-27b", "phase": "active"}
        )
        state = self.read_state()
        state["operations"] = {"schema_version": 999, "queue": ["old"]}
        self.state_path.write_text(json.dumps(state), encoding="utf-8")
        main_unit = self.controller.unit_name("main-27b")
        self.harness.service_inventory_units.add(main_unit)
        self.harness.service_states[main_unit] = ServiceStateSnapshot(
            active_state="active",
            sub_state="running",
            main_pid=4242,
            control_pid=0,
        )
        self.harness.probe_results[self.main.alias] = True

        repaired = _prepare_durable_state(self.controller)

        self.assertTrue(repaired)
        backups = list(self.root.glob("state.json.corrupt.*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text())["operations"]["schema_version"], 999)
        state = self.read_state()
        self.assertEqual(state["profile"], "main-27b")
        self.assertEqual(state["operations"]["executions"], {})
        self.assertTrue(PersistentTransitionQueue(self.controller.state_store).needs_reconciliation)

    def test_raw_subprocess_timeout_is_wrapped_as_controller_error(self) -> None:
        def time_out(command: tuple[str, ...]) -> None:
            raise subprocess.TimeoutExpired(command, timeout=30)

        self.controller.command_runner = time_out
        with self.assertRaises(CommandError) as caught:
            self.controller._run("systemctl", "--user", "stop", "example.service")

        self.assertIsInstance(caught.exception.__cause__, subprocess.TimeoutExpired)

    def test_stop_timeout_escalates_to_cgroup_kill_and_clears_failed_state(self) -> None:
        unit = self.controller.unit_name("main-27b")
        self.harness.service_states[unit] = ServiceStateSnapshot(
            active_state="deactivating",
            sub_state="stop-sigint",
            main_pid=4242,
            control_pid=0,
        )

        def runner(command: tuple[str, ...]) -> None:
            self.harness.commands.append(tuple(command))
            if command[:4] == ("systemctl", "--user", "stop", unit):
                raise subprocess.TimeoutExpired(command, timeout=30)
            if command[:3] == ("systemctl", "--user", "kill"):
                self.harness.service_states[unit] = ServiceStateSnapshot(
                    active_state="inactive",
                    sub_state="dead",
                    main_pid=0,
                    control_pid=0,
                )

        self.controller.command_runner = runner
        self.controller._stop_unit(unit)

        self.assertEqual(
            self.harness.commands,
            [
                ("systemctl", "--user", "stop", unit),
                (
                    "systemctl",
                    "--user",
                    "kill",
                    "--kill-who=all",
                    "--signal=SIGKILL",
                    unit,
                ),
                ("systemctl", "--user", "reset-failed", unit),
            ],
        )
        self.assertTrue(
            any(
                event.get("event") == "service_stop_escalation"
                and event.get("decision") == "force_kill"
                for event in self.harness.events
            )
        )

    def test_stop_timeout_accepts_failed_state_without_process(self) -> None:
        unit = self.controller.unit_name("main-27b")
        self.harness.service_states[unit] = ServiceStateSnapshot(
            active_state="deactivating",
            sub_state="stop-sigterm",
            main_pid=4242,
            control_pid=0,
        )
        reset_attempts = 0

        def runner(command: tuple[str, ...]) -> None:
            nonlocal reset_attempts
            self.harness.commands.append(tuple(command))
            if command[:4] == ("systemctl", "--user", "stop", unit):
                raise subprocess.TimeoutExpired(command, timeout=30)
            if command[:3] == ("systemctl", "--user", "kill"):
                self.harness.service_states[unit] = ServiceStateSnapshot(
                    active_state="failed",
                    sub_state="failed",
                    main_pid=0,
                    control_pid=0,
                )
            if command == ("systemctl", "--user", "reset-failed", unit):
                reset_attempts += 1
                if reset_attempts == 2:
                    self.harness.service_states[unit] = ServiceStateSnapshot(
                        active_state="inactive",
                        sub_state="dead",
                        main_pid=0,
                        control_pid=0,
                    )

        self.controller.command_runner = runner
        with patch.object(backend_controller.time, "sleep", return_value=None):
            self.controller._stop_unit(unit)

        reset_command = ("systemctl", "--user", "reset-failed", unit)
        self.assertEqual(self.harness.commands.count(reset_command), 1)
        self.assertEqual(reset_attempts, 1)

    def test_shutdown_after_prior_stop_never_starts_target_or_rollback(self) -> None:
        cancelled = False
        original_runner = self.harness.run

        def cancel_after_stop(command: tuple[str, ...]) -> None:
            nonlocal cancelled
            original_runner(command)
            if command == (
                "systemctl",
                "--user",
                "stop",
                self.controller.unit_name("main-27b"),
            ):
                cancelled = True

        self.controller.command_runner = cancel_after_stop
        self.controller.cancellation_probe = lambda: cancelled
        self.state_path.write_text(
            json.dumps(
                {"generation": 1, "profile": "main-27b", "phase": "active"}
            ),
            encoding="utf-8",
        )

        with self.assertRaises(SwitchError) as caught:
            self.controller.switch("qwen-vision")

        self.assertIsInstance(caught.exception.cause, LifecycleCancelledError)
        self.assertFalse(
            any(command[:3] == ("systemctl", "--user", "start") for command in self.harness.commands)
        )
        state = self.read_state()
        self.assertEqual(state["phase"], "failed")
        self.assertEqual(state["prior_profile"], "main-27b")
        self.assertEqual(state["target_profile"], "qwen-vision")

    def test_shutdown_during_quiescence_inventory_blocks_start_command(self) -> None:
        cancelled = False
        original_state_probe = self.harness.service_state

        def cancel_during_probe(unit_name: str) -> ServiceStateSnapshot:
            nonlocal cancelled
            snapshot = original_state_probe(unit_name)
            cancelled = True
            return snapshot

        self.controller.service_state_probe = cancel_during_probe
        self.controller.cancellation_probe = lambda: cancelled

        with self.assertRaises(LifecycleCancelledError):
            self.controller._start("qwen-vision")

        self.assertFalse(
            any(command[:3] == ("systemctl", "--user", "start") for command in self.harness.commands)
        )

    def test_systemd_service_state_probe_reads_exact_process_state(self) -> None:
        completed = SimpleNamespace(
            stdout=(
                "ActiveState=inactive\n"
                "SubState=dead\n"
                "MainPID=0\n"
                "ControlPID=0\n"
            )
        )
        with patch.object(
            backend_controller.subprocess, "run", return_value=completed
        ) as run:
            snapshot = systemd_service_state_probe("llama-broker-profile-main.service")

        self.assertTrue(snapshot.is_quiescent)
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ("systemctl", "--user", "show", "llama-broker-profile-main.service"))
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual(run.call_args.kwargs["timeout"], 5)

    def test_systemd_service_inventory_probe_keeps_only_exact_prefix_units(self) -> None:
        completed = SimpleNamespace(
            stdout=(
                "llama-broker-profile-main.service loaded inactive dead Main\n"
                "llama-broker-profile-retired@1.service loaded active running Old\n"
                "llama-broker-profile.service loaded active running Near miss\n"
                "llama-broker-profile-main.service.evil loaded active running Near miss\n"
                "other-llama-broker-profile-main.service loaded active running Other\n"
            )
        )
        with patch.object(
            backend_controller.subprocess, "run", return_value=completed
        ) as run:
            units = systemd_service_inventory_probe("llama-broker-profile")

        self.assertEqual(
            units,
            (
                "llama-broker-profile-main.service",
                "llama-broker-profile-retired@1.service",
            ),
        )
        command = run.call_args.args[0]
        self.assertEqual(command[:3], ("systemctl", "--user", "list-units"))
        self.assertIn("--type=service", command)
        self.assertIn("--all", command)
        self.assertIn("--plain", command)
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual(run.call_args.kwargs["timeout"], 5)

    def test_controller_rejects_profile_that_could_render_unsafe_unit(self) -> None:
        unsafe_service = replace(self.main.safety.service, service_type="simple")
        unsafe_profile = replace(
            self.main,
            safety=replace(self.main.safety, service=unsafe_service),
        )

        with self.assertRaisesRegex(ConfigurationError, "service_type"):
            BackendController(
                profiles={unsafe_profile.id: unsafe_profile},
                state_path=self.state_path,
                systemd_dir=self.systemd_dir,
                service_prefix="llama-broker-profile",
                command_runner=self.harness.run,
                readiness_probe=self.harness.probe,
                resource_probe=self.harness.resources,
                event_logger=self.harness.log_event,
            )

    def test_install_units_writes_distinct_profile_units_atomically(self) -> None:
        paths = self.controller.install_units()

        expected_main = self.systemd_dir / "llama-broker-profile-main-27b.service"
        expected_vision = self.systemd_dir / "llama-broker-profile-qwen-vision.service"
        expected_verify_main = self.systemd_dir / "llama-broker-verify-profile-main-27b.service"
        expected_verify_vision = self.systemd_dir / "llama-broker-verify-profile-qwen-vision.service"
        self.assertEqual(paths, {"main-27b": expected_main, "qwen-vision": expected_vision})
        self.assertTrue(expected_verify_main.exists())
        self.assertTrue(expected_verify_vision.exists())
        self.assertFalse((self.systemd_dir / "llama-server.service").exists())
        self.assertEqual(
            self.harness.commands,
            [
                ("systemctl", "--user", "daemon-reload"),
                (
                    "systemctl",
                    "--user",
                    "reset-failed",
                    "llama-broker-profile-main-27b.service",
                    "llama-broker-profile-qwen-vision.service",
                    "llama-broker-verify-profile-main-27b.service",
                    "llama-broker-verify-profile-qwen-vision.service",
                ),
            ],
        )

        unit = expected_main.read_text(encoding="utf-8")
        self.assertIn("X-Llama-Broker-Profile=main-27b", unit)
        self.assertIn(
            'ExecStart="/opt/llama cpp/llama-server" "-m" "/models/Main 27B.gguf" '
            '"--alias" "main-27b" "-c" "65536" "-ngl" "auto"',
            unit,
        )
        self.assertNotIn('Environment="GGML_CUDA_DISABLE_GRAPHS=1"', unit)
        self.assertNotIn('Environment="GGML_CUDA_GRAPH_OPT=0"', unit)
        self.assertIn('Environment="LITERAL_DOLLAR=$HOME"', unit)
        self.assertIn('Environment="PERCENT_VALUE=95%%"', unit)
        self.assertIn("Restart=no", unit)
        self.assertIn("Type=exec", unit)
        self.assertIn("MemoryMax=20480M", unit)
        self.assertIn("MemorySwapMax=1024M", unit)
        self.assertIn("TasksMax=512", unit)
        self.assertIn("KillSignal=SIGINT", unit)
        self.assertIn("TimeoutStopSec=20s", unit)
        self.assertIn("KillMode=control-group", unit)
        self.assertIn("SendSIGKILL=no", unit)
        self.assertNotIn("Type=simple", unit)
        self.assertEqual(list(self.systemd_dir.glob(".*.tmp")), [])

    def test_cleanup_stale_units_removes_inactive_files_after_regeneration(self) -> None:
        self.controller.install_units()
        stale_profile = self.systemd_dir / "llama-broker-profile-removed.service"
        stale_verify = self.systemd_dir / "llama-broker-verify-profile-removed.service"
        stale_profile.write_text("stale", encoding="utf-8")
        stale_verify.write_text("stale", encoding="utf-8")

        removed = self.controller.cleanup_stale_units()

        self.assertEqual(
            {path.name for path in removed},
            {stale_profile.name, stale_verify.name},
        )
        self.assertFalse(stale_profile.exists())
        self.assertFalse(stale_verify.exists())
        self.assertTrue((self.systemd_dir / "llama-broker-profile-main-27b.service").exists())
        self.assertTrue(
            (self.systemd_dir / "llama-broker-verify-profile-main-27b.service").exists()
        )
        self.assertIn(
            (
                "systemctl",
                "--user",
                "reset-failed",
                stale_profile.name,
                stale_verify.name,
            ),
            self.harness.commands,
        )

    def test_cleanup_stale_units_refuses_active_profile_and_keeps_files(self) -> None:
        self.systemd_dir.mkdir(parents=True, exist_ok=True)
        stale_profile = self.systemd_dir / "llama-broker-profile-removed.service"
        stale_verify = self.systemd_dir / "llama-broker-verify-profile-removed.service"
        stale_profile.write_text("stale", encoding="utf-8")
        stale_verify.write_text("stale", encoding="utf-8")
        self.harness.service_states[stale_profile.name] = ServiceStateSnapshot(
            active_state="active",
            sub_state="running",
            main_pid=4242,
            control_pid=0,
        )

        with self.assertRaisesRegex(ServiceOwnershipError, "removed.*unload"):
            self.controller.cleanup_stale_units()

        self.assertTrue(stale_profile.exists())
        self.assertTrue(stale_verify.exists())

    def test_render_unit_applies_network_and_native_security_arguments(self) -> None:
        self.controller.backend_listen_host = "0.0.0.0"
        self.controller.backend_listen_port = 10000
        self.controller.managed_args = (
            "--api-key-file",
            "/run/user/1000/llama-broker/api-keys",
            "--ssl-cert-file",
            "/etc/llama-broker/backend.crt",
            "--ssl-key-file",
            "/etc/llama-broker/backend.key",
        )

        unit = self.controller.render_unit("main-27b")

        self.assertIn('"--host" "0.0.0.0"', unit)
        self.assertIn('"--port" "10000"', unit)
        self.assertIn(
            '"--api-key-file" "/run/user/1000/llama-broker/api-keys"',
            unit,
        )
        self.assertIn('"--ssl-cert-file" "/etc/llama-broker/backend.crt"', unit)
        self.assertIn('"--ssl-key-file" "/etc/llama-broker/backend.key"', unit)

    def test_switch_starts_target_probes_exact_alias_and_persists_state(self) -> None:
        state = self.controller.switch("main-27b")

        self.assertEqual(
            self.harness.commands,
            [("systemctl", "--user", "start", "llama-broker-profile-main-27b.service")],
        )
        self.assertEqual(self.harness.probes, [("main-27b", 90.0)])
        self.assertEqual(state, {"generation": 1, "profile": "main-27b", "phase": "active"})

        self.assertEqual(self.read_state(), state)

        state = self.controller.switch("qwen-vision")
        self.assertEqual(
            self.harness.commands[-2:],
            [
                ("systemctl", "--user", "stop", "llama-broker-profile-main-27b.service"),
                ("systemctl", "--user", "start", "llama-broker-profile-qwen-vision.service"),
            ],
        )
        self.assertEqual(self.harness.probes[-1], ("qwen3.5-9b-vision", 45.0))
        self.assertEqual(state, {"generation": 2, "profile": "qwen-vision", "phase": "active"})

    def test_backend_lifecycle_write_preserves_namespaced_operations_state(self) -> None:
        operations = {
            "schema_version": 1,
            "queue": ["execution-1"],
            "requests": {"request-1": {"execution_id": "execution-1"}},
        }
        self.state_path.write_text(
            json.dumps(
                {
                    "generation": 0,
                    "profile": None,
                    "phase": "inactive",
                    "operations": operations,
                }
            ),
            encoding="utf-8",
        )

        self.controller.switch("main-27b")

        self.assertEqual(self.read_state()["operations"], operations)

    def test_commit_callback_runs_immediately_before_prior_backend_stop(self) -> None:
        self.controller.switch("main-27b")
        committed = False
        original_runner = self.controller.command_runner

        def run(command: tuple[str, ...]) -> object:
            if command[:3] == ("systemctl", "--user", "stop"):
                self.assertTrue(committed)
            return original_runner(command)

        self.controller.command_runner = run

        def commit() -> None:
            nonlocal committed
            committed = True

        self.controller.switch("qwen-vision", commit_callback=commit)

        self.assertTrue(committed)
        self.assertEqual(self.controller.state()["profile"], "qwen-vision")

    def test_cancelled_commit_leaves_prior_backend_and_state_untouched(self) -> None:
        self.controller.switch("main-27b")
        commands_before = list(self.harness.commands)

        def cancel() -> None:
            raise LifecycleCancelledError("request cancelled before commit")

        with self.assertRaises(LifecycleCancelledError):
            self.controller.switch("qwen-vision", commit_callback=cancel)

        self.assertEqual(self.controller.state()["profile"], "main-27b")
        self.assertEqual(self.controller.state()["phase"], "active")
        self.assertEqual(self.harness.commands, commands_before)

    def test_successful_switch_skips_artifact_verification(self) -> None:
        self.controller.switch("main-27b")

        def reject(_profile: FakeProfile) -> None:
            self.fail("successful activation must not verify artifacts")

        self.controller.artifact_verifier = reject  # type: ignore[attr-defined]
        self.controller.switch("qwen-vision")

        self.assertEqual(self.controller.state()["profile"], "qwen-vision")
        self.assertEqual(self.controller.state()["phase"], "active")

    def test_failed_start_verifies_artifacts_after_cleanup(self) -> None:
        unit = self.controller.unit_name("main-27b")
        self.harness.command_failures.add(("systemctl", "--user", "start", unit))

        def reject(_profile: FakeProfile) -> None:
            self.assertEqual(self.harness.commands[-1], ("systemctl", "--user", "stop", unit))
            raise ConfigurationError("artifact digest mismatch")

        self.controller.artifact_verifier = reject  # type: ignore[attr-defined]

        with self.assertRaises(SwitchError) as caught:
            self.controller.switch("main-27b")

        self.assertIn("injected command failure", str(caught.exception))
        self.assertEqual(self.controller.state()["artifact_verification"],
                         {"status": "failed", "error": "artifact digest mismatch"})

    def test_successful_switch_persists_concrete_running_build_identity(self) -> None:
        profile = replace(
            self.main,
            runtime_build="beellama",
            server_sha256="a" * 64,
        )
        self.profiles["main-27b"] = profile
        self.controller.profiles["main-27b"] = profile

        state = self.controller.switch("main-27b")

        self.assertEqual(state["runtime_build"], "beellama")
        self.assertEqual(state["server_sha256"], "a" * 64)
        self.assertEqual(self.read_state(), state)

    def test_switch_without_resource_probe_has_no_telemetry_dependency(self) -> None:
        controller = BackendController(
            profiles=self.profiles,
            state_path=self.state_path,
            systemd_dir=self.systemd_dir,
            service_prefix="llama-broker-profile",
            command_runner=self.harness.run,
            readiness_probe=self.harness.probe,
            resource_probe=None,
            event_logger=self.harness.log_event,
            service_state_probe=self.harness.service_state,
            service_inventory_probe=self.harness.service_inventory,
        )

        state = controller.switch("main-27b")

        self.assertEqual(state["phase"], "active")
        self.assertEqual(self.harness.resource_probe_calls, 0)
        decisions = [
            event.get("decision")
            for event in self.harness.events
            if event.get("event") == "resource_admission"
        ]
        self.assertEqual(decisions, ["not_configured", "not_configured"])

    def test_exact_alias_cannot_commit_without_running_target_unit(self) -> None:
        def command_without_process(command: tuple[str, ...]) -> None:
            self.harness.commands.append(tuple(command))

        self.controller.command_runner = command_without_process

        with self.assertRaises(SwitchError) as caught:
            self.controller.switch("main-27b")

        self.assertIsInstance(caught.exception.cause, ServiceOwnershipError)
        self.assertEqual(self.read_state()["phase"], "failed")
        self.assertNotEqual(self.read_state().get("profile"), "main-27b")

    def test_unmanaged_private_endpoint_blocks_systemd_start(self) -> None:
        self.controller.endpoint_probe = lambda: True

        with self.assertRaises(SwitchError) as caught:
            self.controller.switch("main-27b")

        self.assertIsInstance(caught.exception.cause, ServiceOwnershipError)
        self.assertFalse(
            any(command[:3] == ("systemctl", "--user", "start") for command in self.harness.commands)
        )
        self.assertEqual(self.read_state()["phase"], "failed")

    def test_successful_switch_emits_structured_transition_decisions(self) -> None:
        self.controller.switch("main-27b")

        transitions = [
            event
            for event in self.harness.events
            if event.get("event") == "profile_transition"
        ]
        self.assertEqual(
            [event["decision"] for event in transitions],
            ["requested", "target_starting", "target_ready", "committed"],
        )
        self.assertTrue(all(event["target_profile"] == "main-27b" for event in transitions))
        self.assertEqual(transitions[0]["prior_profile"], None)
        self.assertEqual(transitions[-1]["generation"], 1)

    def test_pre_start_admission_rejects_low_ram_without_starting_target(self) -> None:
        self.harness.resource_snapshots.append(
            ResourceSnapshot(
                ram_available_mib=4_096,
                swap_used_mib=0,
                vram_total_mib=12_227,
                vram_free_mib=11_000,
            )
        )

        with self.assertRaises(SwitchError):
            self.controller.switch("main-27b")

        self.assertEqual(self.harness.commands, [])
        self.assertEqual(self.harness.probes, [])
        self.assertEqual(self.harness.resource_probe_calls, 1)
        decision = next(
            event
            for event in self.harness.events
            if event.get("event") == "resource_admission"
        )
        self.assertEqual(decision["stage"], "pre_start")
        self.assertEqual(decision["decision"], "deny")
        self.assertIn("ram_available_mib", " ".join(decision["violations"]))
        self.assertEqual(self.read_state()["phase"], "failed")

    def test_unknown_vram_fails_closed_when_vram_admission_is_required(self) -> None:
        self.controller.resource_probe = lambda: ResourceSnapshot(
            ram_available_mib=24_000,
            swap_used_mib=0,
            vram_total_mib=None,
            vram_free_mib=None,
        )

        with self.assertRaises(SwitchError) as raised:
            self.controller.switch("main-27b")

        self.assertIsInstance(raised.exception.cause, AdmissionError)
        self.assertEqual(self.harness.commands, [])
        decision = next(
            event for event in self.harness.events
            if event.get("event") == "resource_admission"
        )
        self.assertEqual(decision["decision"], "deny")
        violations = str(decision.get("violations", ""))
        self.assertIn("vram_total_mib unavailable", violations)
        self.assertIn("vram_free_mib unavailable", violations)

    def test_resource_probe_failure_fails_closed_and_is_logged(self) -> None:
        self.harness.resource_probe_error = OSError("qualification probe unavailable")

        with self.assertRaises(SwitchError) as raised:
            self.controller.switch("main-27b")

        self.assertIsInstance(raised.exception.cause, AdmissionError)
        self.assertEqual(self.harness.commands, [])
        decision = next(
            event
            for event in self.harness.events
            if event.get("event") == "resource_admission"
        )
        self.assertEqual(decision["stage"], "pre_start")
        self.assertEqual(decision["decision"], "probe_error")
        self.assertIn("qualification probe unavailable", decision["error"])

    def test_inconsistent_resource_snapshot_fails_closed(self) -> None:
        self.harness.resource_snapshots.append(
            ResourceSnapshot(
                ram_available_mib=24_000,
                swap_used_mib=0,
                vram_total_mib=12_000,
                vram_free_mib=12_001,
            )
        )

        with self.assertRaises(SwitchError) as raised:
            self.controller.switch("main-27b")

        self.assertIsInstance(raised.exception.cause, AdmissionError)
        self.assertEqual(self.harness.commands, [])
        decision = next(
            event
            for event in self.harness.events
            if event.get("event") == "resource_admission"
        )
        self.assertEqual(decision["decision"], "probe_error")

    def test_switch_to_healthy_current_profile_is_a_noop(self) -> None:
        initial = self.controller.switch("main-27b")
        self.harness.commands.clear()
        self.harness.probes.clear()

        state = self.controller.switch("main-27b")

        self.assertEqual(state, initial)
        self.assertEqual(self.harness.commands, [])
        self.assertEqual(self.harness.probes, [("main-27b", 90.0)])

    def test_stop_all_stops_every_profile_and_marks_inactive(self) -> None:
        self.controller.switch("main-27b")
        self.harness.commands.clear()

        state = self.controller.stop_all()

        self.assertEqual(
            self.harness.commands,
            [
                ("systemctl", "--user", "stop", "llama-broker-profile-main-27b.service"),
                ("systemctl", "--user", "stop", "llama-broker-profile-qwen-vision.service"),
            ],
        )
        self.assertEqual(state, {"generation": 2, "profile": None, "phase": "inactive"})
        self.assertEqual(self.controller.state(), state)

    def test_stop_all_rejects_inactive_state_while_private_endpoint_survives(self) -> None:
        self.controller.endpoint_probe = lambda: True

        with self.assertRaises(QuiescenceError):
            self.controller.stop_all()

        state = self.read_state()
        self.assertEqual(state["phase"], "stop_failed")
        self.assertIsNone(state["profile"])
        self.assertIn("private-endpoint", state["failed_profiles"])

    def test_post_stop_survivor_blocks_target_and_rollback_start(self) -> None:
        self.controller.switch("main-27b")
        self.harness.commands.clear()
        main_unit = "llama-broker-profile-main-27b.service"
        self.harness.service_states[main_unit] = ServiceStateSnapshot(
            active_state="deactivating",
            sub_state="stop-sigint",
            main_pid=4242,
            control_pid=0,
        )

        with self.assertRaises(SwitchError) as raised:
            self.controller.switch("qwen-vision")

        self.assertIsInstance(raised.exception.cause, QuiescenceError)
        self.assertEqual(
            self.harness.commands,
            [("systemctl", "--user", "stop", main_unit)],
        )
        state = self.read_state()
        self.assertEqual(state["phase"], "failed")
        self.assertIsNone(state["profile"])
        self.assertEqual(state["prior_profile"], "main-27b")
        self.assertEqual(state["target_profile"], "qwen-vision")

    def test_pre_start_orphan_blocks_first_start_fail_closed(self) -> None:
        vision_unit = "llama-broker-profile-qwen-vision.service"
        self.harness.service_states[vision_unit] = ServiceStateSnapshot(
            active_state="active",
            sub_state="running",
            main_pid=5151,
            control_pid=0,
        )

        with self.assertRaises(SwitchError) as raised:
            self.controller.switch("main-27b")

        self.assertIsInstance(raised.exception.cause, QuiescenceError)
        self.assertNotIn(
            ("systemctl", "--user", "start", "llama-broker-profile-main-27b.service"),
            self.harness.commands,
        )
        self.assertEqual(self.read_state()["phase"], "failed")

    def test_active_removed_profile_blocks_start_fail_closed(self) -> None:
        stale_unit = "llama-broker-profile-retired.service"
        self.harness.service_inventory_units.add(stale_unit)
        self.harness.service_states[stale_unit] = ServiceStateSnapshot(
            active_state="active",
            sub_state="running",
            main_pid=5252,
            control_pid=0,
        )

        with self.assertRaises(SwitchError) as raised:
            self.controller.switch("main-27b")

        self.assertIsInstance(raised.exception.cause, QuiescenceError)
        self.assertFalse(
            any(command[:3] == ("systemctl", "--user", "start") for command in self.harness.commands)
        )
        self.assertIn(stale_unit, self.harness.service_state_calls)
        self.assertEqual(
            self.harness.service_inventory_calls,
            ["llama-broker-profile"],
        )

    def test_inventory_error_blocks_start_fail_closed(self) -> None:
        self.harness.service_inventory_error = OSError("systemd inventory unavailable")

        with self.assertRaises(SwitchError) as raised:
            self.controller.switch("main-27b")

        self.assertIsInstance(raised.exception.cause, QuiescenceError)
        self.assertFalse(
            any(command[:3] == ("systemctl", "--user", "start") for command in self.harness.commands)
        )
        decision = next(
            event
            for event in self.harness.events
            if event.get("event") == "service_inventory"
        )
        self.assertEqual(decision["decision"], "probe_error")
        self.assertIn("systemd inventory unavailable", decision["error"])

    def test_active_legacy_standalone_unit_blocks_start(self) -> None:
        legacy_unit = "llama-server.service"
        self.harness.service_states[legacy_unit] = ServiceStateSnapshot(
            active_state="active",
            sub_state="running",
            main_pid=5353,
            control_pid=0,
        )

        with self.assertRaises(SwitchError) as raised:
            self.controller.switch("main-27b")

        self.assertIsInstance(raised.exception.cause, QuiescenceError)
        self.assertFalse(
            any(command[:3] == ("systemctl", "--user", "start") for command in self.harness.commands)
        )
        self.assertIn(legacy_unit, self.harness.service_state_calls)

    def test_stop_all_stops_and_verifies_discovered_stale_and_active_legacy_units(self) -> None:
        stale_unit = "llama-broker-profile-retired.service"
        legacy_unit = "llama-server.service"
        self.harness.service_inventory_units.add(stale_unit)
        self.harness.service_states[legacy_unit] = ServiceStateSnapshot(
            active_state="active",
            sub_state="running",
            main_pid=5454,
            control_pid=0,
        )
        self.harness.quiesce_on_stop.add(legacy_unit)

        state = self.controller.stop_all()

        self.assertEqual(
            self.harness.commands,
            [
                ("systemctl", "--user", "stop", "llama-broker-profile-main-27b.service"),
                ("systemctl", "--user", "stop", "llama-broker-profile-qwen-vision.service"),
                ("systemctl", "--user", "stop", stale_unit),
                ("systemctl", "--user", "stop", legacy_unit),
            ],
        )
        self.assertGreaterEqual(self.harness.service_state_calls.count(stale_unit), 1)
        self.assertGreaterEqual(self.harness.service_state_calls.count(legacy_unit), 2)
        self.assertEqual(state["phase"], "inactive")

    def test_stop_all_records_every_unit_that_did_not_quiesce(self) -> None:
        vision_unit = "llama-broker-profile-qwen-vision.service"
        self.harness.service_states[vision_unit] = ServiceStateSnapshot(
            active_state="failed",
            sub_state="timeout",
            main_pid=6161,
            control_pid=0,
        )

        with self.assertRaises(QuiescenceError):
            self.controller.stop_all()

        self.assertEqual(
            self.harness.commands,
            [
                ("systemctl", "--user", "stop", "llama-broker-profile-main-27b.service"),
                ("systemctl", "--user", "stop", vision_unit),
            ],
        )
        state = self.read_state()
        self.assertEqual(state["phase"], "stop_failed")
        self.assertEqual(state["failed_profiles"], ["qwen-vision"])
        self.assertIsNone(state["profile"])

    def test_failed_target_rolls_back_prior_backend_and_state(self) -> None:
        self.controller.switch("main-27b")
        self.harness.commands.clear()
        self.harness.probes.clear()
        self.harness.probe_results["qwen3.5-9b-vision"] = False

        with self.assertRaises(SwitchError) as raised:
            self.controller.switch("qwen-vision")

        self.assertNotIsInstance(raised.exception, RollbackError)
        self.assertEqual(
            self.harness.commands,
            [
                ("systemctl", "--user", "stop", "llama-broker-profile-main-27b.service"),
                ("systemctl", "--user", "start", "llama-broker-profile-qwen-vision.service"),
                ("systemctl", "--user", "stop", "llama-broker-profile-qwen-vision.service"),
            ],
        )
        self.assertEqual(
            self.harness.probes,
            [("qwen3.5-9b-vision", 45.0)],
        )
        self.assertEqual(
            self.read_state(),
            {
                "generation": 1,
                "profile": None,
                "phase": "failed",
                "prior_profile": "main-27b",
                "target_profile": "qwen-vision",
                "error": "profile 'qwen-vision' did not serve exact alias 'qwen3.5-9b-vision' within 45s",
                "restore": "not_attempted",
            },
        )

    def test_failed_target_leaves_broker_in_fail_closed_state_without_restore(self) -> None:
        self.controller.switch("main-27b")
        self.harness.commands.clear()
        self.harness.probes.clear()
        self.harness.probe_results["qwen3.5-9b-vision"] = False

        with self.assertRaises(SwitchError):
            self.controller.switch("qwen-vision")

        self.assertEqual(
            self.harness.commands,
            [
                ("systemctl", "--user", "stop", "llama-broker-profile-main-27b.service"),
                ("systemctl", "--user", "start", "llama-broker-profile-qwen-vision.service"),
                ("systemctl", "--user", "stop", "llama-broker-profile-qwen-vision.service"),
            ],
        )
        self.assertEqual(self.harness.probes, [("qwen3.5-9b-vision", 45.0)])
        state = self.read_state()
        self.assertEqual(state["generation"], 1)
        self.assertIsNone(state["profile"])
        self.assertEqual(state["phase"], "failed")
        self.assertEqual(state["prior_profile"], "main-27b")
        self.assertEqual(state["target_profile"], "qwen-vision")

    def test_post_ready_admission_failure_stops_target_and_restores_prior(self) -> None:
        self.controller.switch("main-27b")
        self.harness.commands.clear()
        self.harness.probes.clear()
        self.harness.events.clear()
        self.harness.resource_probe_calls = 0
        self.harness.resource_snapshots.extend(
            [
                ResourceSnapshot(
                    ram_available_mib=24_000,
                    swap_used_mib=0,
                    vram_total_mib=12_227,
                    vram_free_mib=11_000,
                ),
                ResourceSnapshot(
                    ram_available_mib=20_000,
                    swap_used_mib=256,
                    vram_total_mib=12_227,
                    vram_free_mib=1_000,
                ),
            ]
        )

        with self.assertRaises(SwitchError) as raised:
            self.controller.switch("qwen-vision")

        self.assertNotIsInstance(raised.exception, RollbackError)
        self.assertEqual(
            self.harness.commands,
            [
                ("systemctl", "--user", "stop", "llama-broker-profile-main-27b.service"),
                ("systemctl", "--user", "start", "llama-broker-profile-qwen-vision.service"),
                ("systemctl", "--user", "stop", "llama-broker-profile-qwen-vision.service"),
            ],
        )
        self.assertEqual(
            self.harness.resource_probe_calls,
            2,
            "only the target receives pre/post admission snapshots",
        )
        post_decision = next(
            event
            for event in self.harness.events
            if event.get("event") == "resource_admission"
            and event.get("stage") == "post_ready"
        )
        self.assertEqual(post_decision["decision"], "deny")
        self.assertEqual(post_decision["swap_growth_mib"], 256)
        self.assertIn("vram_free_mib", " ".join(post_decision["violations"]))
        transitions = [
            event["decision"]
            for event in self.harness.events
            if event.get("event") == "profile_transition"
        ]
        self.assertEqual(
            transitions,
            [
                "requested",
                "target_starting",
                "target_failed",
                "target_stopped",
                "failed",
            ],
        )
        self.assertEqual(
            self.read_state(),
            {
                "generation": 1,
                "profile": None,
                "phase": "failed",
                "prior_profile": "main-27b",
                "target_profile": "qwen-vision",
                "error": "profile 'qwen-vision' failed post-ready resource admission: swap_growth_mib 256 > 128; vram_free_mib 1000 < 3072",
                "restore": "not_attempted",
            },
        )

    def test_failed_target_never_restores_even_if_prior_would_also_fail(self) -> None:
        self.controller.switch("main-27b")
        self.harness.commands.clear()
        self.harness.probe_results.update(
            {"qwen3.5-9b-vision": False, "main-27b": False}
        )

        with self.assertRaises(SwitchError) as raised:
            self.controller.switch("qwen-vision")

        self.assertEqual(raised.exception.target_profile, "qwen-vision")
        state = self.read_state()
        self.assertEqual(state["generation"], 1)
        self.assertIsNone(state["profile"])
        self.assertEqual(state["phase"], "failed")
        self.assertEqual(state["prior_profile"], "main-27b")
        self.assertEqual(state["target_profile"], "qwen-vision")
        self.assertEqual(
            self.harness.commands,
            [
                ("systemctl", "--user", "stop", "llama-broker-profile-main-27b.service"),
                ("systemctl", "--user", "start", "llama-broker-profile-qwen-vision.service"),
                ("systemctl", "--user", "stop", "llama-broker-profile-qwen-vision.service"),
            ],
        )

    def test_failed_first_start_stops_target_and_records_failure(self) -> None:
        self.harness.probe_results["main-27b"] = False

        with self.assertRaises(SwitchError):
            self.controller.switch("main-27b")

        self.assertEqual(
            self.harness.commands,
            [
                ("systemctl", "--user", "start", "llama-broker-profile-main-27b.service"),
                ("systemctl", "--user", "stop", "llama-broker-profile-main-27b.service"),
            ],
        )
        state = self.read_state()
        self.assertEqual(state["generation"], 0)
        self.assertIsNone(state["profile"])
        self.assertEqual(state["phase"], "failed")

    def test_start_command_failure_still_stops_partially_started_target(self) -> None:
        start = (
            "systemctl",
            "--user",
            "start",
            "llama-broker-profile-main-27b.service",
        )
        stop = (
            "systemctl",
            "--user",
            "stop",
            "llama-broker-profile-main-27b.service",
        )
        self.harness.command_failures.add(start)

        with self.assertRaises(SwitchError):
            self.controller.switch("main-27b")

        self.assertEqual(self.harness.commands, [start, stop])
        self.assertEqual(self.read_state()["phase"], "failed")

    def test_reconcile_interrupted_switch_fails_closed(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {
                    "generation": 7,
                    "profile": "main-27b",
                    "phase": "switching",
                    "target_profile": "qwen-vision",
                }
            ),
            encoding="utf-8",
        )

        state = self.controller.reconcile()

        self.assertEqual(
            self.harness.commands,
            [
                ("systemctl", "--user", "stop", "llama-broker-profile-qwen-vision.service"),
            ],
        )
        self.assertEqual(self.harness.probes, [])
        self.assertEqual(self.harness.resource_probe_calls, 0)
        self.assertEqual(
            state,
            {
                "generation": 7,
                "profile": None,
                "phase": "failed",
                "prior_profile": "main-27b",
                "target_profile": "qwen-vision",
                "error": "interrupted switch target was stopped; prior profile not restored",
                "restore": "not_attempted",
            },
        )
        self.assertEqual(self.read_state(), state)

    def test_reconcile_does_not_restore_prior_when_target_cleanup_succeeds(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {
                    "generation": 7,
                    "profile": "main-27b",
                    "phase": "switching",
                    "target_profile": "qwen-vision",
                }
            ),
            encoding="utf-8",
        )
        self.harness.probe_results["main-27b"] = False

        state = self.controller.reconcile()

        self.assertEqual(
            self.harness.commands,
            [
                ("systemctl", "--user", "stop", "llama-broker-profile-qwen-vision.service"),
            ],
        )
        self.assertEqual(
            state["phase"], "failed"
        )
        self.assertIsNone(state["profile"])
        self.assertEqual(state["prior_profile"], "main-27b")
        self.assertEqual(state["target_profile"], "qwen-vision")
        self.assertEqual(self.harness.probes, [])

    def test_record_recovery_failure_persists_stale_profile_metadata(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {"generation": 3, "profile": "main-27b", "phase": "active"}
            ),
            encoding="utf-8",
        )

        state = self.controller.record_recovery_failure(
            "main-27b", attempts=2, error="backend exited"
        )

        self.assertEqual(state["phase"], "stale")
        self.assertEqual(state["stale_profile"], "main-27b")
        self.assertEqual(state["recovery"]["attempts"], 2)
        self.assertEqual(state["recovery"]["action"], "broker_watchdog_restart")
        self.assertEqual(self.read_state(), state)

        self.state_path.write_text(
            json.dumps(
                {
                    "generation": 7,
                    "profile": "main-27b",
                    "phase": "active",
                }
            ),
            encoding="utf-8",
        )
        self.systemd_dir.mkdir(parents=True)
        alias_path = self.systemd_dir / "llama-server.service"
        alias_path.symlink_to("llama-broker-profile-main-27b.service")
        self.harness.probe_results["main-27b"] = False

        state = self.controller.reconcile(active_probe_timeout=0.25)

        self.assertEqual(self.harness.commands, [])
        self.assertEqual(self.harness.probes, [("main-27b", 0.25)])
        self.assertEqual(
            state,
            {
                "generation": 8,
                "profile": None,
                "phase": "stale",
                "stale_profile": "main-27b",
                "error": "backend readiness probe failed",
                "recovery": {
                    "attempts": 0,
                    "action": "startup_reconcile",
                    "error": "backend readiness probe failed",
                    "service": {
                        "active_state": "inactive",
                        "sub_state": "dead",
                        "main_pid": 0,
                        "control_pid": 0,
                    },
                },
            },
        )
        self.assertEqual(self.read_state(), state)
        self.assertFalse(alias_path.is_symlink())

    def test_reconcile_keeps_healthy_active_backend(self) -> None:
        active = {
            "generation": 4,
            "profile": "main-27b",
            "phase": "active",
        }
        self.state_path.write_text(json.dumps(active), encoding="utf-8")
        self.harness.service_states[self.controller.unit_name("main-27b")] = (
            ServiceStateSnapshot(
                active_state="active",
                sub_state="running",
                main_pid=4242,
                control_pid=0,
            )
        )

        state = self.controller.reconcile(active_probe_timeout=0.25)

        self.assertEqual(state, active)
        self.assertEqual(self.read_state(), active)
        self.assertEqual(self.harness.commands, [])
        self.assertEqual(self.harness.probes, [("main-27b", 0.25)])

    def test_reconcile_rejects_exact_alias_from_inactive_target_unit(self) -> None:
        active = {
            "generation": 4,
            "profile": "main-27b",
            "phase": "active",
        }
        self.state_path.write_text(json.dumps(active), encoding="utf-8")

        state = self.controller.reconcile(active_probe_timeout=0.25)

        self.assertEqual(state["phase"], "stale")
        self.assertIsNone(state["profile"])
        self.assertEqual(state["stale_profile"], "main-27b")
        self.assertEqual(self.harness.probes, [("main-27b", 0.25)])


class BackendPreemptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.harness = Harness()
        self.main = FakeProfile(
            id="main-27b",
            alias="main-27b",
            display_name="Main 27B",
            server_binary="/opt/llama-server",
            model_path="/models/main.gguf",
        )
        self.vision = FakeProfile(
            id="qwen-vision",
            alias="qwen-vision",
            display_name="Qwen vision",
            server_binary="/opt/llama-server",
            model_path="/models/qwen.gguf",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_controller(self, **overrides: object) -> BackendController:
        options: dict[str, object] = {
            "profiles": {profile.id: profile for profile in (self.main, self.vision)},
            "state_path": self.root / "state.json",
            "systemd_dir": self.root / "systemd",
            "service_prefix": "llama-broker-profile",
            "command_runner": self.harness.run,
            "readiness_probe": self.harness.probe,
            "resource_probe": None,
            "event_logger": self.harness.log_event,
            "service_state_probe": self.harness.service_state,
        }
        options.update(overrides)
        return BackendController(**options)  # type: ignore[arg-type]

    def unit(self, controller: BackendController, profile_id: str) -> str:
        return controller.unit_name(profile_id)

    def test_preempted_activation_stops_target_and_records_inactive_state(self) -> None:
        controller = self.make_controller()
        controller.switch("main-27b")

        def preempted_probe(alias: str, timeout: float) -> bool:
            self.assertTrue(controller.preempt_activation())
            return False

        controller.readiness_probe = preempted_probe
        with self.assertRaises(SwitchError) as caught:
            controller.switch("qwen-vision")

        self.assertIsInstance(caught.exception.__cause__, ActivationPreemptedError)
        vision_unit = self.unit(controller, "qwen-vision")
        self.assertEqual(
            self.harness.commands[-1],
            ("systemctl", "--user", "stop", vision_unit),
        )
        state = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["phase"], "inactive")
        self.assertIsNone(state["profile"])
        self.assertEqual(state["preempted_profile"], "qwen-vision")
        self.assertTrue(self.harness.service_state(vision_unit).is_quiescent)
        self.assertFalse(controller.preempt_activation())

        controller.readiness_probe = self.harness.probe
        state = controller.switch("main-27b")

        self.assertEqual(state["profile"], "main-27b")
        self.assertEqual(
            self.harness.commands[-1],
            ("systemctl", "--user", "start", self.unit(controller, "main-27b")),
        )

    def test_preempt_without_switch_in_progress_is_ignored(self) -> None:
        controller = self.make_controller()

        self.assertFalse(controller.preempt_activation())
        state = controller.switch("main-27b")

        self.assertEqual(state["profile"], "main-27b")


if __name__ == "__main__":
    unittest.main()
