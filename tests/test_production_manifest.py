"""Fixture-driven production-manifest contract tests.

These tests assert the *portable* invariants of a valid profile manifest
(bounded services, fail-closed admission, no-fit contract, controller
rendering, TLS binding defaults) against the machine-independent fixture in
``tests/fixtures/profiles.fixture.json``.

Machine-specific production-manifest checks (real model hashes, local binary
paths, deployment profile IDs) live in ``tests/test_local_production_manifest.py``,
which is gitignored and skipped when no production manifest is present.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llama_broker.daemon as llama_broker
from llama_broker.backend_controller import (
    BackendController,
    ResourceSnapshot,
    linux_resource_snapshot,
)
from llama_broker.broker_security import SecurityConfig
from llama_broker.broker_profiles import ProfileRegistry
from llama_broker import create_production_controller

from tests.fixtures import materialize_fixture


class ProductionManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_path = materialize_fixture()
        cls.registry = ProfileRegistry.from_path(cls.fixture_path)

    def test_aliases_resolve_explicitly(self) -> None:
        expected = {
            "fixture-text-7b": "fixture-text-7b",
            "fixture-7b": "fixture-text-7b",
            "fixture-vision-12b": "fixture-vision-12b",
            "fixture-vision": "fixture-vision-12b",
        }
        self.assertEqual(
            {alias: self.registry.resolve(alias).id for alias in expected}, expected
        )

    def test_text_profile_renders_supported_runtime_flags(self) -> None:
        profile = self.registry.get("fixture-text-7b")
        args = profile.server_args(host="127.0.0.1", port=10000)

        self.assertEqual(profile.alias, "fixture-text-7b")
        self.assertEqual(profile.capabilities, {"text"})
        self.assertIsNone(profile.mmproj_path)
        self.assertEqual(profile.context_size, 32_768)
        self.assertNotIn("--fit", args)
        self.assertNotIn("--gemma", args)
        self.assertNotIn("--arch", args)

    def test_vision_profile_is_bounded_and_capable(self) -> None:
        profile = self.registry.get("fixture-vision-12b")
        self.assertIn("vision", profile.capabilities)
        self.assertIsNotNone(profile.mmproj_path)
        self.assertIsNone(profile.fit)
        self.assertEqual(profile.gpu_layers, "auto")
        self.assertEqual(profile.safety.service.memory_max_mib, 20_480)

    def test_all_profiles_have_bounded_services_and_capacity_metadata(self) -> None:
        for profile in self.registry.profiles:
            with self.subTest(profile=profile.id):
                service = profile.safety.service
                self.assertEqual(service.service_type, "exec")
                self.assertEqual(service.kill_signal, "SIGINT")
                self.assertLessEqual(service.memory_max_mib, 20_480)
                self.assertLessEqual(service.memory_swap_max_mib, 1_024)
                self.assertLessEqual(service.tasks_max, 512)
                self.assertEqual(service.timeout_stop_sec, 20)
                self.assertEqual(
                    profile.safety.admission.min_total_vram_mib, 12_000
                )
                self.assertEqual(
                    profile.safety.admission.min_free_vram_before_start_mib, 8_192
                )

        self.assertFalse(hasattr(self.registry, "default_profile"))

    def test_production_controller_has_no_mandatory_resource_probe(self) -> None:
        controller = create_production_controller(self.registry)
        self.assertIsNone(controller.resource_probe)

    def test_secure_backend_defaults_to_lan_binding_unless_explicitly_overridden(
        self,
    ) -> None:
        security = SecurityConfig(
            api_key_file=Path("/tmp/api-keys"),
            backend_tls_cert_file=Path("/tmp/llama-server.crt"),
            backend_tls_key_file=Path("/tmp/llama-server.key"),
            backend_tls_ca_file=Path("/tmp/lan-ca.crt"),
        )

        with (
            patch.object(llama_broker, "BROKER_HOST", "127.0.0.1"),
            patch.object(llama_broker, "BACKEND_LISTEN_HOST", "127.0.0.1"),
            patch.object(llama_broker, "_BACKEND_LISTEN_HOST_EXPLICIT", False),
        ):
            controller = create_production_controller(self.registry, security=security)

        self.assertEqual(controller.backend_listen_host, "0.0.0.0")

        with (
            patch.object(llama_broker, "BROKER_HOST", "127.0.0.1"),
            patch.object(llama_broker, "BACKEND_LISTEN_HOST", "127.0.0.1"),
            patch.object(llama_broker, "_BACKEND_LISTEN_HOST_EXPLICIT", True),
        ):
            controller = create_production_controller(self.registry, security=security)

        self.assertEqual(controller.backend_listen_host, "127.0.0.1")

    def test_linux_resource_snapshot_reads_meminfo_and_primary_gpu(self) -> None:
        with self.subTest("valid sample"):
            with tempfile.TemporaryDirectory() as directory:
                meminfo = Path(directory) / "meminfo"
                meminfo.write_text(
                    "MemAvailable:       24576000 kB\n"
                    "SwapTotal:           8388608 kB\n"
                    "SwapFree:            8327168 kB\n",
                    encoding="utf-8",
                )
                with patch(
                    "llama_broker.backend_controller.subprocess.run",
                    return_value=SimpleNamespace(stdout="12227, 10913\n"),
                ) as run:
                    snapshot = linux_resource_snapshot(meminfo)

                self.assertEqual(
                    snapshot,
                    ResourceSnapshot(
                        ram_available_mib=24000,
                        swap_used_mib=60,
                        vram_total_mib=12227,
                        vram_free_mib=10913,
                    ),
                )
                self.assertEqual(run.call_args.args[0][:3], ("nvidia-smi", "-i", "0"))

    def test_controller_render_has_one_model_and_alias_pair(self) -> None:
        controller = BackendController(
            profiles={profile.id: profile for profile in self.registry.profiles},
            state_path="/tmp/not-used-broker-state.json",
            systemd_dir="/tmp/not-used-broker-systemd",
            service_prefix="llama-broker-profile",
            command_runner=lambda _command: None,
            readiness_probe=lambda _alias, _timeout: True,
            resource_probe=lambda: ResourceSnapshot(
                ram_available_mib=24_000,
                swap_used_mib=0,
                vram_total_mib=12_227,
                vram_free_mib=11_000,
            ),
            event_logger=lambda _event: None,
        )
        unit = controller.render_unit("fixture-text-7b")
        exec_line = next(
            line for line in unit.splitlines() if line.startswith("ExecStart=")
        )
        self.assertEqual(exec_line.count('"-m"'), 1)
        self.assertEqual(exec_line.count('"--alias"'), 1)
        self.assertIn('"-c" "32768"', exec_line)
        self.assertNotIn('"--fit"', exec_line)
        self.assertNotIn('"--cache-ram"', exec_line)
        self.assertNotIn('"--ctx-checkpoints"', exec_line)
        self.assertIn('"-ngl" "auto"', exec_line)
        self.assertIn("Type=exec", unit)
        self.assertIn("MemoryMax=20480M", unit)
        self.assertIn("MemorySwapMax=1024M", unit)
        self.assertIn("TasksMax=512", unit)
        self.assertIn("KillSignal=SIGINT", unit)
        self.assertIn("TimeoutStopSec=20s", unit)
        self.assertIn("KillMode=control-group", unit)
        self.assertIn("SendSIGKILL=no", unit)
        self.assertIn("Restart=no", unit)

    def test_broker_control_plane_unit_is_bounded_without_forced_kill(self) -> None:
        unit = (
            Path(__file__).resolve().parents[1]
            / "docs/examples/llama-broker.service.in"
        ).read_text(encoding="utf-8")
        self.assertIn("MemoryHigh=768M", unit)
        self.assertIn("MemoryMax=1024M", unit)
        self.assertIn("MemorySwapMax=128M", unit)
        self.assertIn("TasksMax=64", unit)
        self.assertIn("TimeoutStopSec=600", unit)
        self.assertIn("KillMode=mixed", unit)
        self.assertIn("SendSIGKILL=no", unit)


if __name__ == "__main__":
    unittest.main()
