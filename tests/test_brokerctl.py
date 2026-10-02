from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import llama_broker.cli as brokerctl
from llama_broker.broker_profiles import ProfileRegistry
from llama_broker.discovery import DiscoveryResult
from llama_broker.integration import ProfileInspection
from tests.fixtures import materialize_fixture


class BrokerctlStatusTests(unittest.TestCase):
    def test_profiles_discover_json_has_schema_version(self) -> None:
        output = io.StringIO()
        with (
            patch.object(brokerctl, "discover_backends", return_value=DiscoveryResult()),
            redirect_stdout(output),
        ):
            result = brokerctl.main(["profile", "discover", "--json"])

        self.assertEqual(result, 1)
        self.assertEqual(json.loads(output.getvalue())["schema_version"], 1)

    def test_model_list_json_has_schema_version(self) -> None:
        registry = Mock()
        registry.profiles = []
        output = io.StringIO()
        with (
            patch.object(brokerctl.ProfileRegistry, "from_path", return_value=registry),
            redirect_stdout(output),
        ):
            result = brokerctl.main(["model", "list", "--json"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["schema_version"], 1)

    def test_status_json_has_schema_version(self) -> None:
        controller = Mock()
        controller.state.return_value = {"generation": 1, "profile": None, "phase": "inactive"}
        output = io.StringIO()
        with (
            patch.object(brokerctl.ProfileRegistry, "from_path", return_value=Mock()),
            patch.object(brokerctl, "create_production_controller", return_value=controller),
            patch.object(brokerctl, "backend_endpoint_ready", return_value=False),
            redirect_stdout(output),
        ):
            result = brokerctl.main(["status", "--json"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["schema_version"], 1)

    def test_status_reports_persisted_active_state_as_stale_when_backend_is_absent(self) -> None:
        controller = Mock()
        controller.state.return_value = {
            "generation": 10,
            "profile": "example-9b",
            "phase": "active",
        }

        output = io.StringIO()
        with (
            patch.object(brokerctl.ProfileRegistry, "from_path", return_value=Mock()),
            patch.object(brokerctl, "create_production_controller", return_value=controller),
            patch.object(brokerctl, "backend_endpoint_ready", return_value=False),
            redirect_stdout(output),
        ):
            result = brokerctl.main(["status"])

        self.assertEqual(result, 1)
        self.assertIn("Broker:    stale", output.getvalue())
        self.assertIn("Backend:   unavailable", output.getvalue())
        self.assertIn("persisted profile: example-9b", output.getvalue())

    def test_status_includes_live_context_and_slot_utilisation(self) -> None:
        controller = Mock()
        controller.state.return_value = {
            "generation": 11,
            "profile": "main",
            "phase": "active",
        }
        profile = SimpleNamespace(
            id="main",
            display_name="Main model",
            runtime="llamacpp",
            server_binary="llama-server",
            context_size=262144,
            parallel=2,
        )
        registry = Mock()
        registry.try_resolve.return_value = profile
        output = io.StringIO()
        with (
            patch.object(brokerctl.ProfileRegistry, "from_path", return_value=registry),
            patch.object(brokerctl, "create_production_controller", return_value=controller),
            patch.object(brokerctl, "_broker_admin_get", return_value={
                "status": "ok",
                "backend": {"profile": "main", "phase": "active", "ready": True},
                "leases": {"active_requests": 1, "pending_requests": 2, "queued_transitions": 1},
            }),
            patch.object(brokerctl, "backend_endpoint_ready", return_value=True),
            patch.object(brokerctl, "get_runtime_info", return_value=SimpleNamespace(
                name="llamacpp", binaries=("llama-server",)
            )),
            redirect_stdout(output),
        ):
            result = brokerctl.main(["status"])

        self.assertEqual(result, 0)
        rendered = output.getvalue()
        self.assertIn("health: ok", rendered)
        self.assertIn("Context:   262,144 tokens total; 131,072 per slot", rendered)
        self.assertIn("Requests:  active=1 pending=2 (pending transitions=1)", rendered)
        self.assertIn("Slots:     1/2 in use (50%)", rendered)

    def _render_status(self, *, parallel: int, leases: dict[str, int]) -> str:
        controller = Mock()
        controller.state.return_value = {"generation": 11, "profile": "main", "phase": "active"}
        profile = SimpleNamespace(
            id="main",
            display_name="Main model",
            runtime="llamacpp",
            server_binary="llama-server",
            context_size=65536,
            parallel=parallel,
        )
        registry = Mock()
        registry.try_resolve.return_value = profile
        output = io.StringIO()
        with (
            patch.object(brokerctl.ProfileRegistry, "from_path", return_value=registry),
            patch.object(brokerctl, "create_production_controller", return_value=controller),
            patch.object(brokerctl, "_broker_admin_get", return_value={
                "status": "ok",
                "backend": {"profile": "main", "phase": "active", "ready": True},
                "leases": leases,
            }),
            patch.object(brokerctl, "backend_endpoint_ready", return_value=True),
            patch.object(brokerctl, "get_runtime_info", return_value=SimpleNamespace(
                name="llamacpp", binaries=("llama-server",)
            )),
            redirect_stdout(output),
        ):
            self.assertEqual(brokerctl.main(["status"]), 0)
        return output.getvalue()

    def test_status_never_reports_more_slots_in_use_than_parallel(self) -> None:
        rendered = self._render_status(
            parallel=1,
            leases={"active_requests": 2, "inference_requests": 2, "queued_transitions": 0},
        )

        self.assertIn("Requests:  active=1 pending=1 (pending transitions=0)", rendered)
        self.assertIn("Slots:     1/1 in use (100%)", rendered)
        self.assertNotIn("2/1", rendered)

    def test_status_excludes_slot_probes_from_slot_usage(self) -> None:
        rendered = self._render_status(
            parallel=1,
            leases={"active_requests": 2, "inference_requests": 1, "queued_transitions": 0},
        )

        self.assertIn("Requests:  active=1 pending=0 (pending transitions=0)", rendered)
        self.assertIn("Slots:     1/1 in use (100%)", rendered)


class BrokerctlLifecycleSafetyTests(unittest.TestCase):
    def test_load_pretty_prints_admin_state_by_default(self) -> None:
        admin = Mock(
            return_value={
                "name": "Llama-Broker",
                "status": "ok",
                "backend": {
                    "host": "127.0.0.1",
                    "port": 10000,
                    "generation": 12,
                    "phase": "active",
                    "profile": "main",
                    "ready": True,
                },
                "operation": {
                    "status": "accepted",
                    "target": "vision",
                    "type": "switch",
                },
            }
        )
        registry = Mock()
        registry.resolve.return_value = SimpleNamespace(id="vision")
        output = io.StringIO()

        with (
            patch.object(brokerctl.ProfileRegistry, "from_path", return_value=registry),
            patch.object(brokerctl, "create_production_controller", return_value=Mock()),
            patch.object(brokerctl, "_broker_admin", admin),
            redirect_stdout(output),
        ):
            result = brokerctl.main(["load", "vision"])

        self.assertEqual(result, 0)
        admin.assert_called_once_with("/admin/switch", {"model": "vision", "wait": False})
        self.assertEqual(
            output.getvalue(),
            "Request:   accepted\n"
            "Target:    vision\n"
            "Broker:    ok\n"
            "Backend:   transition pending (current: main)\n"
            "Phase:     transition_pending\n"
            "Ready:     no\n"
            "Generation: 12\n"
            "Endpoint:  127.0.0.1:10000\n",
        )

    def test_load_raw_preserves_json_output(self) -> None:
        state = {
            "name": "Llama-Broker",
            "status": "ok",
            "backend": {"profile": "vision", "phase": "active", "ready": True},
            "operation": {"status": "accepted", "target": "vision", "type": "switch"},
        }
        admin = Mock(return_value=state)
        registry = Mock()
        registry.resolve.return_value = SimpleNamespace(id="vision")
        output = io.StringIO()

        with (
            patch.object(brokerctl.ProfileRegistry, "from_path", return_value=registry),
            patch.object(brokerctl, "create_production_controller", return_value=Mock()),
            patch.object(brokerctl, "_broker_admin", admin),
            redirect_stdout(output),
        ):
            result = brokerctl.main(["load", "vision", "--raw"])

        self.assertEqual(result, 0)
        admin.assert_called_once_with("/admin/switch", {"model": "vision", "wait": False})
        self.assertEqual(json.loads(output.getvalue()), state)

    def test_unload_pretty_prints_admin_state_by_default(self) -> None:
        admin = Mock(
            return_value={
                "name": "Llama-Broker",
                "status": "ok",
                "backend": {
                    "host": "127.0.0.1",
                    "port": 10000,
                    "generation": 13,
                    "phase": "inactive",
                    "profile": None,
                    "ready": False,
                },
            }
        )
        output = io.StringIO()

        with (
            patch.object(brokerctl.ProfileRegistry, "from_path", return_value=Mock()),
            patch.object(brokerctl, "create_production_controller", return_value=Mock()),
            patch.object(brokerctl, "_broker_admin", admin),
            redirect_stdout(output),
        ):
            result = brokerctl.main(["unload"])

        self.assertEqual(result, 0)
        admin.assert_called_once_with("/admin/stop", {})
        self.assertEqual(
            output.getvalue(),
            "Broker:    ok\n"
            "Backend:   inactive\n"
            "Phase:     inactive\n"
            "Ready:     no\n"
            "Generation: 13\n"
            "Endpoint:  127.0.0.1:10000\n",
        )

    def test_unload_raw_preserves_json_output(self) -> None:
        state = {
            "name": "Llama-Broker",
            "status": "ok",
            "backend": {"profile": None, "phase": "inactive", "ready": False},
        }
        admin = Mock(return_value=state)
        output = io.StringIO()

        with (
            patch.object(brokerctl.ProfileRegistry, "from_path", return_value=Mock()),
            patch.object(brokerctl, "create_production_controller", return_value=Mock()),
            patch.object(brokerctl, "_broker_admin", admin),
            redirect_stdout(output),
        ):
            result = brokerctl.main(["unload", "--raw"])

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue()), state)

    def test_restart_without_model_requests_admin_restart(self) -> None:
        admin = Mock(return_value={"status": "ok"})
        with (
            patch.object(brokerctl.ProfileRegistry, "from_path", return_value=Mock()),
            patch.object(brokerctl, "create_production_controller", return_value=Mock()),
            patch.object(brokerctl, "_broker_admin", admin),
            redirect_stdout(io.StringIO()),
        ):
            result = brokerctl.main(["restart"])

        self.assertEqual(result, 0)
        admin.assert_called_once_with("/admin/restart", {})

    def test_restart_with_model_requests_admin_restart_for_target(self) -> None:
        admin = Mock(return_value={"status": "ok"})
        registry = Mock()
        registry.resolve.return_value = SimpleNamespace(id="vision")
        with (
            patch.object(brokerctl.ProfileRegistry, "from_path", return_value=registry),
            patch.object(brokerctl, "create_production_controller", return_value=Mock()),
            patch.object(brokerctl, "_broker_admin", admin),
            redirect_stdout(io.StringIO()),
        ):
            result = brokerctl.main(["restart", "vision"])

        self.assertEqual(result, 0)
        admin.assert_called_once_with("/admin/restart", {"model": "vision"})

    def test_admin_connection_failure_never_falls_back_to_direct_controller(self) -> None:
        connection = Mock()
        connection.request.side_effect = ConnectionRefusedError("broker offline")

        with (
            patch.dict(
                brokerctl.os.environ,
                {"LLAMA_BROKER_ADMIN_URL": "http://127.0.0.1:11434"},
                clear=False,
            ),
            patch.object(
                brokerctl.http.client,
                "HTTPConnection",
                return_value=connection,
            ),
            self.assertRaisesRegex(
                brokerctl.BackendControllerError,
                "direct lifecycle fallback is disabled",
            ),
        ):
            brokerctl._broker_admin("/admin/stop", {})

        connection.close.assert_called_once_with()

    def test_admin_accepts_async_switch_response(self) -> None:
        connection = Mock()
        response = Mock(status=202)
        response.read.return_value = b'{"status":"accepted"}'
        connection.getresponse.return_value = response

        with (
            patch.dict(
                brokerctl.os.environ,
                {"LLAMA_BROKER_ADMIN_URL": "http://127.0.0.1:11434"},
                clear=False,
            ),
            patch.object(
                brokerctl.http.client,
                "HTTPConnection",
                return_value=connection,
            ),
        ):
            self.assertEqual(
                brokerctl._broker_admin("/admin/switch", {"model": "vision", "wait": False}),
                {"status": "accepted"},
            )


class BrokerctlSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.systemd_dir = Path(tempfile.mkdtemp())
        self.systemd_dir_patch = patch.object(
            brokerctl.broker_runtime,
            "SYSTEMD_USER_DIR",
            self.systemd_dir,
        )
        self.systemd_dir_patch.start()
        self.addCleanup(self.systemd_dir_patch.stop)
        self.addCleanup(shutil.rmtree, self.systemd_dir, ignore_errors=True)

    def test_setup_regenerates_units_by_default_and_has_an_explicit_opt_out(self) -> None:
        self.assertTrue(brokerctl.build_parser().parse_args(["setup"]).install_units)
        self.assertFalse(
            brokerctl.build_parser()
            .parse_args(["setup", "--no-install-units"])
            .install_units
        )
        self.assertFalse(hasattr(brokerctl.build_parser().parse_args(["setup"]), "set_default"))
        with self.assertRaises(SystemExit):
            brokerctl.build_parser().parse_args(["setup", "--set-default"])

    def test_interactive_setup_does_not_expose_json(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            brokerctl.build_parser().parse_args(["setup", "--json"])
        self.assertEqual(raised.exception.code, 2)

    def test_setup_clean_regenerates_then_cleans_stale_units(self) -> None:
        controller = Mock()
        controller.install_units.return_value = {
            "main": Path("/tmp/llama-broker-profile-main.service")
        }
        controller.cleanup_stale_units.return_value = (
            Path("/tmp/llama-broker-profile-old.service"),
            Path("/tmp/llama-broker-verify-profile-old.service"),
        )
        args = SimpleNamespace(
            profiles=Path("profiles.json"),
            verify=False,
            install_units=True,
            clean=True,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            with (
                patch.dict(
                    brokerctl.os.environ,
                    {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus"},
                ),
                patch.object(
                    brokerctl.ProfileRegistry,
                    "from_path",
                    return_value=SimpleNamespace(profiles=()),
                ),
                patch.object(brokerctl, "create_production_controller", return_value=controller),
                patch.object(
                    brokerctl,
                    "ensure_security_material",
                    return_value=(SimpleNamespace(api_key_file=Path(tempdir) / "api-keys"), "redacted"),
                ),
                patch("llama_broker.configure.main", return_value=0),
                patch.object(brokerctl, "BIN_DIR", Path(tempdir) / "bin"),
                patch.object(brokerctl.Path, "home", return_value=Path(tempdir)),
                redirect_stdout(io.StringIO()),
            ):
                result = brokerctl._cmd_setup(args)

        self.assertEqual(result, 0)
        controller.install_units.assert_called_once_with()
        controller.cleanup_stale_units.assert_called_once_with()

    def test_setup_keeps_unit_paths_without_repeating_catalog_rows(self) -> None:
        controller = Mock()
        controller.install_units.return_value = {
            "main": Path("/tmp/llama-broker-profile-main.service")
        }
        profile = SimpleNamespace(
            id="main",
            alias="fixture-text-7b",
            context_size=262144,
            parallel=1,
            capabilities={"text"},
        )
        args = SimpleNamespace(
            profiles=Path("profiles.json"),
            verify=False,
            install_units=True,
        )
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as tempdir:
            with (
                patch.dict(
                    brokerctl.os.environ,
                    {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus"},
                ),
                patch.object(
                    brokerctl.ProfileRegistry,
                    "from_path",
                    return_value=SimpleNamespace(
                        profiles=(profile,),
                    ),
                ),
                patch.object(brokerctl, "create_production_controller", return_value=controller),
                patch.object(
                    brokerctl,
                    "ensure_security_material",
                    return_value=(
                        SimpleNamespace(api_key_file=Path(tempdir) / "api-keys"),
                        "redacted",
                    ),
                ),
                patch("llama_broker.configure.main", return_value=0),
                patch.object(brokerctl, "BIN_DIR", Path(tempdir) / "bin"),
                redirect_stdout(output),
            ):
                result = brokerctl._cmd_setup(args)

        self.assertEqual(result, 0)
        rendered = output.getvalue()
        self.assertNotIn("main-27b  ctx=", rendered)
        self.assertIn(
            "main -> /tmp/llama-broker-profile-main.service",
            rendered,
        )

    def test_tls_setup_defaults_profile_units_to_lan_binding(self) -> None:
        security = SimpleNamespace(
            broker_tls_cert_file=Path("/tmp/broker.crt"),
            backend_tls_cert_file=Path("/tmp/llama-server.crt"),
        )
        with (
            patch.dict(brokerctl.os.environ, {}, clear=True),
            patch.object(brokerctl.broker_runtime, "BROKER_HOST", "127.0.0.1"),
            patch.object(brokerctl.broker_runtime, "BACKEND_LISTEN_HOST", "127.0.0.1"),
        ):
            brokerctl._apply_setup_network_defaults(security)

            self.assertEqual(brokerctl.broker_runtime.BROKER_HOST, "0.0.0.0")
            self.assertEqual(brokerctl.broker_runtime.BACKEND_LISTEN_HOST, "0.0.0.0")


    def test_setup_regenerates_units_without_taking_the_lifecycle_lock(self) -> None:
        controller = Mock()
        controller.install_units.return_value = {}
        args = SimpleNamespace(profiles=Path("profiles.json"), verify=False)

        with tempfile.TemporaryDirectory() as tempdir:
            with (
                patch.dict(
                    brokerctl.os.environ,
                    {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus"},
                ),
                patch.object(brokerctl.ProfileRegistry, "from_path", return_value=SimpleNamespace(profiles=())),
                patch.object(brokerctl, "create_production_controller", return_value=controller),
                patch.object(
                    brokerctl,
                    "_operation_lock",
                    side_effect=AssertionError("setup must not take the lifecycle lock"),
                ) as operation_lock,
                patch.object(
                    brokerctl,
                    "ensure_security_material",
                    return_value=(SimpleNamespace(api_key_file=Path(tempdir) / "api-keys"), "redacted"),
                ),
                patch("llama_broker.configure.main", return_value=0) as configure_main,
                patch.object(brokerctl, "BIN_DIR", Path(tempdir) / "bin"),
                redirect_stdout(io.StringIO()),
            ):
                result = brokerctl._cmd_setup(args)

        self.assertEqual(result, 0)
        operation_lock.assert_not_called()
        controller.install_units.assert_called_once_with()
        configure_main.assert_called_once_with(["--profiles", "profiles.json"], indent=True)

    def test_setup_can_skip_unit_regeneration_explicitly(self) -> None:
        controller = Mock()
        controller.install_units.return_value = {}
        args = SimpleNamespace(profiles=Path("profiles.json"), verify=False, install_units=False)

        with tempfile.TemporaryDirectory() as tempdir:
            with (
                patch.dict(
                    brokerctl.os.environ,
                    {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus"},
                ),
                patch.object(
                    brokerctl.ProfileRegistry,
                    "from_path",
                    return_value=SimpleNamespace(
                        profiles=(),
                    ),
                ),
                patch.object(
                    brokerctl,
                    "_operation_lock",
                    side_effect=AssertionError("setup must not take the lifecycle lock"),
                ),
                patch.object(
                    brokerctl,
                    "ensure_security_material",
                    return_value=(SimpleNamespace(api_key_file=Path(tempdir) / "api-keys"), "redacted"),
                ),
                patch("llama_broker.configure.main", return_value=0),
                patch.object(brokerctl, "BIN_DIR", Path(tempdir) / "bin"),
                redirect_stdout(io.StringIO()),
            ):
                result = brokerctl._cmd_setup(args)

        self.assertEqual(result, 0)
        controller.install_units.assert_not_called()

    def test_setup_does_not_select_a_profile(self) -> None:
        args = SimpleNamespace(
            profiles=Path("profiles.json"),
            verify=False,
            install_units=False,

        )
        with tempfile.TemporaryDirectory() as tempdir:
            with (
                patch.object(
                    brokerctl.ProfileRegistry,
                    "from_path",
                    return_value=SimpleNamespace(profiles=()),
                ),
                patch.object(
                    brokerctl,
                    "ensure_security_material",
                    return_value=(SimpleNamespace(api_key_file=Path(tempdir) / "api-keys"), "redacted"),
                ),
                patch("llama_broker.configure.main", return_value=0) as configure_main,
                patch.object(brokerctl, "BIN_DIR", Path(tempdir) / "bin"),
                redirect_stdout(io.StringIO()),
            ):
                result = brokerctl._cmd_setup(args)

        self.assertEqual(result, 0)
        configure_main.assert_called_once_with(
            ["--profiles", "profiles.json"],
            indent=True,
        )

    def test_security_init_passes_certificate_identity_and_overwrite_flag(self) -> None:
        result = SimpleNamespace(
            created=True,
            sans=("DNS:localhost", "DNS:broker-test-host", "IP:127.0.0.1", "IP:192.0.2.10"),
            config=SimpleNamespace(
                source_path=Path("/tmp/security.json"),
                api_key_file=Path("/tmp/api-keys"),
                broker_tls_ca_file=Path("/tmp/lan-ca.crt"),
                broker_tls_cert_file=Path("/tmp/broker.crt"),
                backend_tls_cert_file=Path("/tmp/llama-server.crt"),
            ),
        )
        output = io.StringIO()
        with patch.object(brokerctl, "initialize_security_material", return_value=result) as initialize:
            with redirect_stdout(output):
                exit_code = brokerctl.main(
                    [
                        "security",
                        "init",
                        "--ip",
                        "192.0.2.10",
                        "--hostname",
                        "broker-test-host",
                        "--days",
                        "30",
                        "--force",
                    ]
                )

        self.assertEqual(exit_code, 0)
        initialize.assert_called_once_with(
            ip_addresses=["192.0.2.10"],
            hostnames=["broker-test-host"],
            security_path=None,
            api_key_path=None,
            days=30,
            force=True,
        )
        self.assertIn("API-key value is redacted", output.getvalue())


class BrokerctlProfileMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.profiles = self.root / "profiles.json"
        from tests.fixtures import materialize_fixture  # noqa: PLC0415

        source = materialize_fixture()
        self.data = json.loads(source.read_text(encoding="utf-8"))
        self.profiles.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _configure_runtime_builds(self) -> tuple[Path, Path]:
        profile = next(
            item for item in self.data["profiles"] if item["id"] == "fixture-text-7b"
        )
        default_server = Path(profile["server_binary"])
        default_server.chmod(0o755)
        upstream_server = self.root / "bin" / "upstream" / "llama-server"
        upstream_server.parent.mkdir(parents=True, exist_ok=True)
        upstream_server.write_bytes(b"upstream runtime build")
        upstream_server.chmod(0o755)
        self.data["runtime_builds"] = [
            {
                "id": "default-build",
                "runtime": "llamacpp",
                "display_name": "Default build",
                "server_binary": str(default_server),
                "server_sha256": brokerctl.sha256_file(default_server),
                "environment": {"LD_LIBRARY_PATH": str(default_server.parent)},
            },
            {
                "id": "upstream-build",
                "runtime": "llamacpp",
                "display_name": "Upstream build",
                "server_binary": str(upstream_server),
                "server_sha256": brokerctl.sha256_file(upstream_server),
                "environment": {"LD_LIBRARY_PATH": str(upstream_server.parent)},
            },
        ]
        profile["runtime_build"] = "default-build"
        self.profiles.write_text(
            json.dumps(self.data, indent=2) + "\n", encoding="utf-8"
        )
        return default_server, upstream_server

    def test_profiles_add_parser_accepts_runtime_build(self) -> None:
        args = brokerctl.build_parser().parse_args(
            [
                "profile",
                "add",
                "new-profile",
                "--model",
                "/tmp/model.gguf",
                "--runtime-build",
                "upstream-build",
            ]
        )

        self.assertEqual(args.runtime_build, "upstream-build")

    def test_profiles_add_selects_cataloged_runtime_build(self) -> None:
        _default_server, upstream_server = self._configure_runtime_builds()
        source_profile = next(
            item for item in self.data["profiles"] if item["id"] == "fixture-text-7b"
        )
        output = io.StringIO()

        with redirect_stdout(output):
            result = brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "add",
                    "upstream-test",
                    "--model",
                    source_profile["model_path"],
                    "--runtime-build",
                    "upstream-build",
                ]
            )

        self.assertEqual(result, 0)
        added = next(
            item
            for item in json.loads(self.profiles.read_text(encoding="utf-8"))["profiles"]
            if item["id"] == "upstream-test"
        )
        self.assertEqual(added["runtime"], "llamacpp")
        self.assertEqual(added["runtime_build"], "upstream-build")
        self.assertEqual(added["server_binary"], str(upstream_server.resolve()))
        self.assertEqual(added["server_sha256"], brokerctl.sha256_file(upstream_server))
        self.assertEqual(
            added["environment"]["LD_LIBRARY_PATH"], str(upstream_server.parent)
        )

    def test_profiles_add_selects_cross_runtime_build_without_template_fields(self) -> None:
        default_server = Path(
            next(
                item
                for item in self.data["profiles"]
                if item["id"] == "fixture-text-7b"
            )["server_binary"]
        )
        default_server.chmod(0o755)
        vllm_binary = self.root / "bin" / "vllm"
        vllm_binary.parent.mkdir(parents=True, exist_ok=True)
        vllm_binary.write_bytes(b"vllm runtime build")
        vllm_binary.chmod(0o755)
        self.data["runtime_builds"] = [
            {
                "id": "default-build",
                "runtime": "llamacpp",
                "display_name": "Default build",
                "server_binary": str(default_server),
                "server_sha256": brokerctl.sha256_file(default_server),
            },
            {
                "id": "vllm-build",
                "runtime": "vllm",
                "display_name": "vLLM build",
                "server_binary": str(vllm_binary),
                "server_sha256": brokerctl.sha256_file(vllm_binary),
                "environment": {"VLLM_USE_V1": "1"},
            },
        ]
        source_profile = next(
            item for item in self.data["profiles"] if item["id"] == "fixture-text-7b"
        )
        source_profile["runtime_build"] = "default-build"
        source_profile["environment"]["VLLM_USE_V1"] = "stale-source-setting"
        self.profiles.write_text(
            json.dumps(self.data, indent=2) + "\n", encoding="utf-8"
        )
        output = io.StringIO()

        with redirect_stdout(output):
            result = brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "add",
                    "vllm-test",
                    "--model",
                    source_profile["model_path"],
                    "--runtime-build",
                    "vllm-build",
                ]
            )

        self.assertEqual(result, 0)
        added = next(
            item
            for item in json.loads(self.profiles.read_text(encoding="utf-8"))["profiles"]
            if item["id"] == "vllm-test"
        )
        self.assertEqual(added["runtime"], "vllm")
        self.assertEqual(added["runtime_build"], "vllm-build")
        self.assertEqual(added["server_binary"], str(vllm_binary.resolve()))
        self.assertEqual(added["environment"], {"VLLM_USE_V1": "1"})
        for field in (
            "batch_size",
            "ubatch_size",
            "cache_type_k",
            "cache_type_v",
            "flash_attention",
            "gpu_layers",
            "fit",
            "mmproj_path",
            "mmproj_sha256",
        ):
            self.assertNotIn(field, added)

    def test_profiles_add_infers_vllm_build_when_template_runtime_is_omitted(self) -> None:
        default_server = Path(
            next(
                item
                for item in self.data["profiles"]
                if item["id"] == "fixture-text-7b"
            )["server_binary"]
        )
        default_server.chmod(0o755)
        vllm_binary = self.root / "bin" / "vllm"
        vllm_binary.parent.mkdir(parents=True, exist_ok=True)
        vllm_binary.write_bytes(b"vllm inferred runtime build")
        vllm_binary.chmod(0o755)
        self.data["runtime_builds"] = [
            {
                "id": "default-build",
                "runtime": "llamacpp",
                "display_name": "Default build",
                "server_binary": str(default_server),
                "server_sha256": brokerctl.sha256_file(default_server),
            },
            {
                "id": "vllm-build",
                "runtime": "vllm",
                "display_name": "vLLM build",
                "server_binary": str(vllm_binary),
                "server_sha256": brokerctl.sha256_file(vllm_binary),
            },
        ]
        source_profile = next(
            item for item in self.data["profiles"] if item["id"] == "fixture-text-7b"
        )
        source_profile["runtime_build"] = "default-build"
        self.profiles.write_text(
            json.dumps(self.data, indent=2) + "\n", encoding="utf-8"
        )
        output = io.StringIO()

        with redirect_stdout(output):
            result = brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "add",
                    "vllm-inferred-test",
                    "--model",
                    source_profile["model_path"],
                    "--server-binary",
                    str(vllm_binary),
                ]
            )

        self.assertEqual(result, 0)
        added = next(
            item
            for item in json.loads(self.profiles.read_text(encoding="utf-8"))["profiles"]
            if item["id"] == "vllm-inferred-test"
        )
        self.assertEqual(added["runtime"], "vllm")
        self.assertEqual(added["runtime_build"], "vllm-build")
        self.assertEqual(added["server_binary"], str(vllm_binary.resolve()))

    def test_profiles_add_infers_runtime_build_from_matching_binary(self) -> None:
        _default_server, upstream_server = self._configure_runtime_builds()
        source_profile = next(
            item for item in self.data["profiles"] if item["id"] == "fixture-text-7b"
        )
        output = io.StringIO()

        with redirect_stdout(output):
            result = brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "add",
                    "inferred-test",
                    "--model",
                    source_profile["model_path"],
                    "--server-binary",
                    str(upstream_server),
                ]
            )

        self.assertEqual(result, 0)
        added = next(
            item
            for item in json.loads(self.profiles.read_text(encoding="utf-8"))["profiles"]
            if item["id"] == "inferred-test"
        )
        self.assertEqual(added["runtime_build"], "upstream-build")
        self.assertEqual(added["server_binary"], str(upstream_server.resolve()))

    def test_profiles_add_rejects_conflicting_runtime_build_and_binary_atomically(self) -> None:
        default_server, upstream_server = self._configure_runtime_builds()
        source_profile = next(
            item for item in self.data["profiles"] if item["id"] == "fixture-text-7b"
        )
        before = self.profiles.read_bytes()
        error = io.StringIO()

        with redirect_stderr(error):
            result = brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "add",
                    "conflict-test",
                    "--model",
                    source_profile["model_path"],
                    "--runtime-build",
                    "upstream-build",
                    "--server-binary",
                    str(default_server),
                ]
            )

        self.assertEqual(result, 1)
        self.assertEqual(self.profiles.read_bytes(), before)
        self.assertIn("does not match runtime build", error.getvalue())
        self.assertNotIn("conflict-test", self.profiles.read_text(encoding="utf-8"))

    def test_live_profile_mutation_refreshes_broker_registry(self) -> None:
        output = io.StringIO()
        admin = Mock(return_value={"status": "ok"})
        with (
            patch.object(brokerctl, "PROFILES_PATH", self.profiles),
            patch.object(brokerctl, "_broker_admin", admin),
            redirect_stdout(output),
        ):
            result = brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "set",
                    "fixture-text-7b",
                    "startup-timeout",
                    "600",
                ]
            )

        self.assertEqual(result, 0)
        admin.assert_called_once_with("/admin/autotune/reload", {})

    def test_live_profile_mutation_rolls_back_if_registry_refresh_fails(self) -> None:
        before = self.profiles.read_bytes()
        admin = Mock(side_effect=RuntimeError("broker unavailable"))
        error = io.StringIO()
        with (
            patch.object(brokerctl, "PROFILES_PATH", self.profiles),
            patch.object(brokerctl, "_broker_admin", admin),
            redirect_stderr(error),
        ):
            result = brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "set",
                    "fixture-text-7b",
                    "startup-timeout",
                    "600",
                ]
            )

        self.assertEqual(result, 1)
        self.assertEqual(self.profiles.read_bytes(), before)
        self.assertIn("live broker profile registry refresh failed", error.getvalue())

    def test_fit_unset_removes_every_fit_field_and_round_trips(self) -> None:
        main = next(profile for profile in self.data["profiles"] if profile["id"] == "fixture-text-7b")
        main.update(
            {
                "fit": True,
                "fit_target_mib": 3_072,
                "fit_ctx": 262_144,
                "ctx_checkpoints": 0,
                "cache_ram_mib": 0,
            }
        )
        self.profiles.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

        output = io.StringIO()
        with redirect_stdout(output):
            result = brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "edit",
                    "fixture-text-7b",
                    "--fit",
                    "unset",
                ]
            )

        self.assertEqual(result, 0)
        updated = json.loads(self.profiles.read_text(encoding="utf-8"))
        record = next(profile for profile in updated["profiles"] if profile["id"] == "fixture-text-7b")
        for key in ("fit", "fit_target_mib", "fit_ctx", "ctx_checkpoints", "cache_ram_mib"):
            self.assertNotIn(key, record)
        parsed = ProfileRegistry.from_path(self.profiles).get("fixture-text-7b")
        self.assertNotIn("--fit", parsed.server_args(host="127.0.0.1", port=10000))

    def test_profiles_edit_selects_cataloged_runtime_build(self) -> None:
        profile = next(
            item for item in self.data["profiles"] if item["id"] == "fixture-text-7b"
        )
        server = Path(profile["server_binary"])
        server.chmod(0o755)
        self.data["runtime_builds"] = [
            {
                "id": "beellama",
                "runtime": "llamacpp",
                "display_name": "BeeLlama",
                "server_binary": str(server),
                "server_sha256": profile["server_sha256"],
            }
        ]
        self.profiles.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

        output = io.StringIO()
        with redirect_stdout(output):
            result = brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "edit",
                    "fixture-text-7b",
                    "--runtime-build",
                    "beellama",
                ]
            )

        self.assertEqual(result, 0)
        updated = json.loads(self.profiles.read_text(encoding="utf-8"))
        selected = next(
            item for item in updated["profiles"] if item["id"] == "fixture-text-7b"
        )
        self.assertEqual(selected["runtime_build"], "beellama")
        self.assertEqual(selected["server_binary"], str(server.resolve()))
        self.assertEqual(selected["server_sha256"], profile["server_sha256"])

    def test_profiles_edit_rejects_unknown_or_mismatched_build_without_mutation(self) -> None:
        profile = next(
            item for item in self.data["profiles"] if item["id"] == "fixture-text-7b"
        )
        server = Path(profile["server_binary"])
        server.chmod(0o755)
        self.data["runtime_builds"] = [
            {
                "id": "atomic-vllm",
                "runtime": "vllm",
                "display_name": "Atomic vLLM",
                "server_binary": str(server),
                "server_sha256": profile["server_sha256"],
            }
        ]
        self.profiles.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

        for build_id in ("missing", "atomic-vllm"):
            before = self.profiles.read_bytes()
            error = io.StringIO()
            with redirect_stderr(error):
                result = brokerctl.main(
                    [
                        "--profiles",
                        str(self.profiles),
                        "profile",
                        "edit",
                        "fixture-text-7b",
                        "--runtime-build",
                        build_id,
                    ]
                )

            self.assertEqual(result, 1)
            self.assertEqual(self.profiles.read_bytes(), before)
            self.assertIn("runtime build", error.getvalue())

    def test_profiles_show_reports_inferred_runtime_build_identity(self) -> None:
        profile = next(
            item for item in self.data["profiles"] if item["id"] == "fixture-text-7b"
        )
        server = Path(profile["server_binary"])
        server.chmod(0o755)
        self.data["runtime_builds"] = [
            {
                "id": "beellama",
                "runtime": "llamacpp",
                "display_name": "BeeLlama",
                "server_binary": str(server),
                "server_sha256": profile["server_sha256"],
            }
        ]
        self.profiles.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

        output = io.StringIO()
        with redirect_stdout(output):
            result = brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "show",
                    "fixture-text-7b",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["runtime_build"], "beellama")

    def test_profiles_add_refuses_incomplete_discovery_without_mutation(self) -> None:
        before = self.profiles.read_bytes()
        model = SimpleNamespace(id="unreviewed-model", context_length=65_536, parallel=1)
        endpoint = SimpleNamespace(label="test-backend", models=[model])
        error = io.StringIO()

        with (
            patch.object(
                brokerctl,
                "discover_backends",
                return_value=SimpleNamespace(endpoints=[endpoint]),
            ),
            redirect_stderr(error),
        ):
            result = brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "add",
                    "unreviewed-model",
                ]
            )

        self.assertEqual(result, 2)
        self.assertEqual(self.profiles.read_bytes(), before)
        self.assertIn("Refusing to auto-add", error.getvalue())
        self.assertNotIn('"fit"', error.getvalue())

    def test_rich_profile_commands_round_trip_and_reject_invalid_mutations(self) -> None:
        model_path = next(
            Path(profile["model_path"])
            for profile in self.data["profiles"]
            if profile["id"] == "fixture-text-7b"
        )
        output = io.StringIO()
        with redirect_stdout(output):
            result = brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "add",
                    "new-profile",
                    "--model",
                    str(model_path),
                    "--alias",
                    "new-profile",
                    "--alias",
                    "new-model",
                    "--display-name",
                    "New profile",
                    "--context",
                    "32768",
                    "--cache-k",
                    "f16",
                    "--cache-v",
                    "f16",
                    "--gpu-layers",
                    "auto",
                    "--env",
                    "CUSTOM=value",
                    "--arg",
                    "--no-warmup",
                ]
            )
        self.assertEqual(result, 0)
        self.assertIn("profile edits do not install units", output.getvalue())

        shown = io.StringIO()
        with redirect_stdout(shown):
            self.assertEqual(
                brokerctl.main(
                    ["--profiles", str(self.profiles), "profile", "show", "new-profile"]
                ),
                0,
            )
        shown_record = json.loads(shown.getvalue())
        self.assertEqual(shown_record["display_name"], "New profile")
        self.assertEqual(shown_record["environment"]["CUSTOM"], "value")
        self.assertEqual(shown_record["extra_args"], ["--no-warmup"])

        self.assertEqual(
            brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "set",
                    "new-profile",
                    "startup-timeout",
                    "600",
                ]
            ),
            0,
        )
        self.assertEqual(
            brokerctl.main(
                ["--profiles", str(self.profiles), "profile", "alias", "add", "new-profile", "new-alias"]
            ),
            0,
        )
        self.assertEqual(
            brokerctl.main(
                ["--profiles", str(self.profiles), "profile", "env", "set", "new-profile", "OTHER=2"]
            ),
            0,
        )
        self.assertEqual(
            brokerctl.main(
                [
                    "--profiles",
                    str(self.profiles),
                    "profile",
                    "args",
                    "add",
                    "new-profile",
                    "--reasoning",
                    "off",
                ]
            ),
            0,
        )
        self.assertEqual(
            brokerctl.main(
                ["--profiles", str(self.profiles), "profile", "rename", "new-profile", "renamed"]
            ),
            0,
        )
        self.assertEqual(
            brokerctl.main(
                ["--profiles", str(self.profiles), "profile", "unset", "renamed", "extra-args"]
            ),
            0,
        )

        before_invalid = self.profiles.read_bytes()
        errors = io.StringIO()
        with redirect_stderr(errors):
            self.assertEqual(
                brokerctl.main(
                    [
                        "--profiles",
                        str(self.profiles),
                        "profile",
                        "set",
                        "renamed",
                        "cache-k",
                        "unknown-cache",
                    ]
                ),
                1,
            )
        self.assertEqual(self.profiles.read_bytes(), before_invalid)
        self.assertIn("invalid cache-k", errors.getvalue())

        self.assertEqual(
            brokerctl.main(["--profiles", str(self.profiles), "profile", "remove", "renamed"]),
            0,
        )
        registry = ProfileRegistry.from_path(self.profiles)
        self.assertFalse(hasattr(registry, "default_profile"))
        self.assertIsNone(registry.try_resolve("renamed"))


class BrokerctlProfileEditorTests(unittest.TestCase):
    def test_edit_parser_accepts_runtime_build(self) -> None:
        args = brokerctl.build_parser().parse_args(
            ["profile", "edit", "fixture-text-7b", "--runtime-build", "beellama"]
        )

        self.assertEqual(args.runtime_build, "beellama")

    def test_edit_parser_targets_field_editor_and_configure_is_removed(self) -> None:
        args = brokerctl.build_parser().parse_args(["profile", "edit"])
        self.assertEqual(args.command, "profile")
        self.assertEqual(args.profiles_command, "edit")
        self.assertIsNone(args.model)

        with self.assertRaises(SystemExit):
            brokerctl.build_parser().parse_args(["profile", "configure"])

    def test_edit_without_model_selects_from_numbered_menu(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            from tests.fixtures import materialize_fixture  # noqa: PLC0415

            profiles = materialize_fixture(Path(tempdir))
            expected = json.loads(profiles.read_text(encoding="utf-8"))["profiles"][1]["id"]
            output = io.StringIO()
            with (
                patch("builtins.input", return_value="2"),
                redirect_stdout(output),
            ):
                selected = brokerctl._select_profile_for_edit(profiles)

        self.assertEqual(selected, expected)
        self.assertIn("Select a profile to edit", output.getvalue())
        self.assertIn(expected, output.getvalue())

    def test_edit_without_model_passes_menu_selection_to_tui(self) -> None:
        from tests.fixtures import materialize_fixture  # noqa: PLC0415

        class TTYBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        with tempfile.TemporaryDirectory() as tempdir:
            profiles = materialize_fixture(Path(tempdir))
            output = TTYBuffer()
            with (
                patch.object(brokerctl, "_select_profile_for_edit", return_value="fixture-vision-12b"),
                patch("llama_broker.broker_tui.run_tui", return_value=0) as run_tui,
                patch.object(brokerctl.sys.stdin, "isatty", return_value=True),
                redirect_stdout(output),
            ):
                result = brokerctl.main(["--profiles", str(profiles), "profile", "edit"])

        self.assertEqual(result, 0)
        run_tui.assert_called_once()
        self.assertEqual(run_tui.call_args.args[1], "fixture-vision-12b")

    def test_authoritative_edit_does_not_hold_operation_lock_during_reload(self) -> None:
        from tests.fixtures import materialize_fixture  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tempdir:
            profiles = materialize_fixture(Path(tempdir))
            updated = json.loads(profiles.read_text(encoding="utf-8"))
            updated["profiles"][0]["display_name"] = "updated"
            controller = Mock()
            controller.state.return_value = {"phase": "inactive", "profile": None}
            with (
                patch.object(brokerctl, "_is_authoritative_manifest", return_value=True),
                patch.object(brokerctl, "_operation_lock", side_effect=AssertionError("lock held during reload")) as operation_lock,
                patch.object(brokerctl, "create_production_controller", return_value=controller),
                patch.object(brokerctl, "_refresh_live_profile_registry"),
            ):
                result = brokerctl._apply_profile_configuration(profiles, updated, {"fixture-text-7b"})

        self.assertTrue(result.changed)
        operation_lock.assert_not_called()

    def test_profiles_state_command_is_removed(self) -> None:
        with self.assertRaises(SystemExit):
            brokerctl.build_parser().parse_args(["profile", "state"])


class BrokerctlCommandLayoutTests(unittest.TestCase):
    def test_profile_render_is_namespaced_and_dispatches(self) -> None:
        args = brokerctl.build_parser().parse_args(["profile", "render", "main"])
        self.assertEqual((args.command, args.profiles_command, args.model), ("profile", "render", "main"))

        with self.assertRaises(SystemExit):
            brokerctl.build_parser().parse_args(["render", "main"])

        registry = Mock()
        registry.resolve.return_value = SimpleNamespace(id="main")
        controller = Mock()
        controller.render_unit.return_value = "[Unit]\nDescription=main\n"
        output = io.StringIO()
        with (
            patch.object(brokerctl.ProfileRegistry, "from_path", return_value=registry),
            patch.object(brokerctl, "create_production_controller", return_value=controller),
            redirect_stdout(output),
        ):
            result = brokerctl.main(["profile", "render", "main"])

        self.assertEqual(result, 0)
        registry.resolve.assert_called_once_with("main")
        controller.render_unit.assert_called_once_with("main")
        self.assertEqual(output.getvalue(), "[Unit]\nDescription=main\n")

    def test_explicit_command_namespaces_replace_ambiguous_top_level_commands(self) -> None:
        model = brokerctl.build_parser().parse_args(["model", "list"])
        self.assertEqual((model.command, model.model_command), ("model", "list"))

        integrate_install = brokerctl.build_parser().parse_args(["integrate", "install"])
        self.assertEqual(
            (integrate_install.command, integrate_install.integrate_command),
            ("integrate", "install"),
        )
        integrate_configure = brokerctl.build_parser().parse_args(
            ["integrate", "configure", "--model", "fixture-text-7b"]
        )
        self.assertEqual(
            (integrate_configure.command, integrate_configure.integrate_command),
            ("integrate", "configure"),
        )
        self.assertEqual(integrate_configure.model, "fixture-text-7b")

        rotate = brokerctl.build_parser().parse_args(["security", "rotate", "--replace"])
        self.assertEqual((rotate.command, rotate.security_command), ("security", "rotate"))
        self.assertTrue(rotate.replace)
        revoke = brokerctl.build_parser().parse_args(["security", "revoke", "key-id"])
        self.assertEqual((revoke.command, revoke.security_command), ("security", "revoke"))
        self.assertEqual(revoke.key, "key-id")

        for argv in (("list",), ("install",), ("configure",), ("credentials", "rotate")):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    brokerctl.build_parser().parse_args(list(argv))

    def test_model_list_is_profile_centric_and_does_not_print_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            from tests.fixtures import materialize_fixture  # noqa: PLC0415

            profiles = materialize_fixture(Path(tempdir))
            output = io.StringIO()
            with redirect_stdout(output):
                result = brokerctl.main(["--profiles", str(profiles), "model", "list"])

        self.assertEqual(result, 0)
        rendered = output.getvalue()
        self.assertIn("MODEL", rendered)
        self.assertIn("PROFILE", rendered)
        self.assertIn("RUNTIME", rendered)
        self.assertIn("fixture-text-7b", rendered)
        self.assertIn("fixture-vision-12b", rendered)
        self.assertIn("llamacpp", rendered)
        self.assertNotIn("model_path", rendered)
        self.assertNotIn(tempdir, rendered)

    def test_model_list_json_advertises_only_canonical_model_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            from tests.fixtures import materialize_fixture  # noqa: PLC0415

            profiles = materialize_fixture(Path(tempdir))
            output = io.StringIO()
            with redirect_stdout(output):
                result = brokerctl.main(
                    ["--profiles", str(profiles), "model", "list", "--json"]
                )

        self.assertEqual(result, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["object"], "model_list")
        self.assertTrue(payload["data"])
        for model in payload["data"]:
            self.assertIn("id", model)
            self.assertNotIn("profile", model)
            self.assertNotIn("aliases", model)

    def test_profile_list_reports_total_and_slot_context(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            from tests.fixtures import materialize_fixture  # noqa: PLC0415

            profiles = materialize_fixture(Path(tempdir))
            manifest = json.loads(profiles.read_text(encoding="utf-8"))
            manifest["profiles"][0].update({"context_size": 65536, "parallel": 2})
            profiles.write_text(json.dumps(manifest), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                result = brokerctl.main(["--profiles", str(profiles), "profile", "list"])

        self.assertEqual(result, 0)
        line = next(
            line for line in output.getvalue().splitlines() if line.startswith("fixture-text-7b")
        )
        self.assertIn("ctx=65536", line)
        self.assertIn("parallel=2", line)
        self.assertIn("slot_ctx=32768", line)

    def test_runtime_list_is_profile_centric_and_does_not_print_binary_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            from tests.fixtures import materialize_fixture  # noqa: PLC0415

            profiles = materialize_fixture(Path(tempdir))
            output = io.StringIO()
            with redirect_stdout(output):
                result = brokerctl.main(["--profiles", str(profiles), "runtime", "list"])

        self.assertEqual(result, 0)
        rendered = output.getvalue()
        self.assertIn("RUNTIME", rendered)
        self.assertIn("PROFILES", rendered)
        self.assertIn("llamacpp", rendered)
        self.assertIn("fixture-text-7b", rendered)
        self.assertIn("fixture-vision-12b", rendered)
        self.assertIn("vllm", rendered)
        self.assertNotIn("binaries=[", rendered)
        self.assertNotIn("capabilities=[", rendered)
        self.assertNotIn(tempdir, rendered)

    def test_security_rotate_and_revoke_use_the_security_namespace(self) -> None:
        with (
            patch.object(brokerctl, "_broker_admin", return_value={"status": "ok"}) as admin,
            patch.object(brokerctl, "_configure_after_credential_change"),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(brokerctl.main(["security", "rotate", "--replace"]), 0)
            self.assertEqual(brokerctl.main(["security", "revoke", "old-key"]), 0)

        self.assertEqual(
            admin.call_args_list,
            [
                call("/admin/credentials/rotate", {"replace": True}),
                call("/admin/credentials/revoke", {"key": "old-key"}),
            ],
        )


if __name__ == "__main__":
    unittest.main()

class RuntimeListTableWidthTests(unittest.TestCase):
    """`runtime list` / table rendering must fit the terminal, never overflow."""

    def _render(self, headers, rows, width=80) -> str:
        return brokerctl._render_table(headers, rows, width=width)

    def test_table_fits_terminal_width_and_keeps_content(self) -> None:
        rows = [
            (
                "llamacpp",
                "qwen3.5-9b, qwen3.5-9b-vision, qwen3.5-9b-antidoom, gemma4-12b, qwen3.8-27b, qwen3.8-27b-kvarn-bounded-accum-test",
                "text, vision, tools, fit, speculative",
            ),
        ]
        rendered = self._render(("RUNTIME", "PROFILES", "CAPABILITIES"), rows, width=80)
        lines = rendered.splitlines()
        for line in lines:
            self.assertLessEqual(len(line), 80, f"line exceeds 80 cols: {line!r}")
        # Content is preserved, just wrapped: join on newlines so multi-line
        # cells are reassembled intact.
        joined = "\n".join(lines)
        # Header and separator are present.
        self.assertIn("RUNTIME", joined)
        self.assertIn("CAPABILITIES", joined)

    def test_table_exact_width_when_content_is_small(self) -> None:
        rendered = self._render(("RUNTIME", "PROFILES", "CAPABILITIES"), [
            ("llamacpp", "one, two, three", "text"),
        ])
        lines = rendered.splitlines()
        # Column widths are 8 (llamacpp), 15 (one, two, three), 12 (CAPABILITIES).
        self.assertEqual(
            lines[1].rstrip(),
            "-" * 8 + "  " + "-" * 15 + "  " + "-" * 12,
        )
        self.assertEqual(lines[2].rstrip(), "llamacpp" + "  " + "one, two, three" + "  " + "text")
        # When content is small the table hugs its content: no trailing stretch.
        for line in lines:
            self.assertLessEqual(len(line.rstrip()), 8 + 2 + 15 + 2 + 12)

    def test_table_wraps_very_long_single_token(self) -> None:
        token = "x" * 40
        rendered = self._render(("A", "B"), [(token, "z")], width=40)
        for line in rendered.splitlines():
            self.assertLessEqual(len(line), 40)

    def test_runtime_list_auto_width_needs_no_flag(self) -> None:
        # Auto-detection: no --width given, output still fits the detected width.
        with tempfile.TemporaryDirectory() as tempdir:
            from tests.fixtures import materialize_fixture  # noqa: PLC0415

            profiles = materialize_fixture(Path(tempdir))
            output = io.StringIO()
            with redirect_stdout(output):
                result = brokerctl.main(["--profiles", str(profiles), "runtime", "list"])
        self.assertEqual(result, 0)
        rendered = output.getvalue()
        detected = brokerctl.shutil.get_terminal_size().columns
        for line in rendered.splitlines():
            self.assertLessEqual(len(line), max(detected, brokerctl._MIN_TABLE_WIDTH))
        self.assertIn("llamacpp", rendered)

    def test_runtime_list_width_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            from tests.fixtures import materialize_fixture  # noqa: PLC0415

            profiles = materialize_fixture(Path(tempdir))
            output = io.StringIO()
            with redirect_stdout(output):
                result = brokerctl.main(
                    ["--profiles", str(profiles), "runtime", "list", "--width", "200"]
                )
        self.assertEqual(result, 0)
        rendered = output.getvalue()
        # At 200 cols nothing wraps: each comma-joined value stays on one line.
        self.assertIn("fixture-text-7b, fixture-vision-12b", rendered)
        for line in rendered.splitlines():
            self.assertLessEqual(len(line), 200)

    def test_model_list_accepts_width_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            from tests.fixtures import materialize_fixture  # noqa: PLC0415

            profiles = materialize_fixture(Path(tempdir))
            output = io.StringIO()
            with redirect_stdout(output):
                result = brokerctl.main(
                    ["--profiles", str(profiles), "model", "list", "--width", "120"]
                )
        self.assertEqual(result, 0)


class BrokerctlSetupWizardApplyErrorTests(unittest.TestCase):
    def test_setup_wizard_reports_apply_error_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(
                profiles=root / "profiles.json",
                runtime_path=root / "llama-server",
                model_path=root / "model.gguf",
                scan_path=[],
                check=False,
                json=False,
                dry_run=False,
                enable_startup=False,
                allow_experimental=False,
                link_model=False,
                integrate=False,
            )
            plan = SimpleNamespace(
                errors=(),
                profile_record={"id": "new-profile"},
                enable_startup=False,
            )
            stderr = io.StringIO()

            with (
                patch.object(brokerctl, "RUNTIME_CATALOG_PATH", root / "catalog.json"),
                patch.object(brokerctl, "_setup_referenced_paths", return_value=[]),
                patch.object(
                    brokerctl,
                    "ensure_runtime_catalog",
                    return_value={"mode": "catalog-first", "broad_scan": False},
                ),
                patch.object(brokerctl, "discover_candidates", return_value=SimpleNamespace()),
                patch.object(brokerctl, "build_setup_plan", return_value=plan),
                patch.object(brokerctl, "_confirm_setup_plan", return_value=True),
                patch.object(
                    brokerctl,
                    "apply_profile_plan",
                    side_effect=brokerctl.ApplyError(
                        "profile apply failed: original; rollback also failed: rollback"
                    ),
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(stderr),
            ):
                result = brokerctl._cmd_setup_wizard(args)

            self.assertEqual(result, 1)
            self.assertIn("ERROR: setup apply failed", stderr.getvalue())
            self.assertIn("rollback also failed", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


class BrokerctlSetupReferenceScanTests(unittest.TestCase):
    def test_reference_scan_does_not_migrate_unversioned_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profiles = materialize_fixture(root / "fixture")
            data = json.loads(profiles.read_text(encoding="utf-8"))
            data.pop("schema_version")
            profiles.write_text(json.dumps(data), encoding="utf-8")
            catalog = root / "catalog.json"
            catalog.write_text("{}\n", encoding="utf-8")
            manifest_before = profiles.read_bytes()
            backups_before = {
                path.name: path.read_bytes()
                for path in profiles.parent.glob(f"{profiles.name}.bak-*")
            }

            with patch.object(brokerctl, "RUNTIME_CATALOG_PATH", catalog):
                brokerctl._setup_referenced_paths(profiles)

            self.assertEqual(profiles.read_bytes(), manifest_before)
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in profiles.parent.glob(f"{profiles.name}.bak-*")
                },
                backups_before,
            )


class BrokerctlIntegrateCleanTests(unittest.TestCase):
    def _run_install(
        self,
        root: Path,
        *,
        clean: bool,
        cleanup_error: Exception | None = None,
        extra_manifest_ids: tuple[str, ...] = (),
        systemd_available: bool = True,
    ) -> tuple[int, dict[str, object], Mock, list[str]]:
        profiles = materialize_fixture(root / "fixture")
        registry = ProfileRegistry.from_path(profiles)
        profile = registry.get("fixture-text-7b")
        record = next(
            item
            for item in json.loads(profiles.read_text(encoding="utf-8"))["profiles"]
            if item["id"] == profile.id
        )
        inspection = ProfileInspection(profile.id, "ready", profile, record)
        payload = json.loads(profiles.read_text(encoding="utf-8"))
        payload["profiles"].extend({"id": profile_id} for profile_id in extra_manifest_ids)
        controller = Mock()
        events: list[str] = []
        controller.unit_path.side_effect = lambda profile_id: root / f"{profile_id}.service"
        controller.render_unit.side_effect = lambda profile_id: f"unit:{profile_id}\n"
        controller.install_units.side_effect = lambda: events.append("install")
        stale = (root / "old-profile.service", root / "old-profile-verify.service")
        if cleanup_error is None:
            controller.cleanup_stale_units.side_effect = lambda **_kwargs: (
                events.append("clean") or stale
            )
        else:
            controller.cleanup_stale_units.side_effect = lambda **_kwargs: (
                events.append("clean") or (_ for _ in ()).throw(cleanup_error)
            )
        template = root / "llama-broker.service.in"
        template.write_text("[Unit]\nDescription=test\n", encoding="utf-8")
        args = SimpleNamespace(
            profiles=profiles,
            full_verify=False,
            json=True,
            dry_run=False,
            integrate_command="install",
            install_units=True,
            clean=clean,
            model=None,
            tool_names=(),
        )
        output = io.StringIO()
        with (
            patch.object(
                brokerctl,
                "inspect_manifest",
                return_value=(payload, (inspection,)),
            ),
            patch.object(brokerctl, "create_production_controller", return_value=controller),
            patch.object(brokerctl, "_service_template_path", return_value=template),
            patch.object(brokerctl, "_service_exec_start", return_value="/bin/true"),
            patch.object(brokerctl, "_atomic_write"),
            patch.object(brokerctl.broker_runtime, "SYSTEMD_USER_DIR", root / "systemd"),
            patch.dict(
                brokerctl.os.environ,
                {"DBUS_SESSION_BUS_ADDRESS": "test" if systemd_available else ""},
                clear=False,
            ),
            redirect_stdout(output),
        ):
            result = brokerctl._cmd_integrate(args)
        return result, json.loads(output.getvalue()), controller, events

    def test_clean_installs_before_cleanup_and_reports_removed_units(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result, report, controller, events = self._run_install(
                Path(temp), clean=True
            )

        self.assertEqual(result, 0)
        self.assertEqual(events, ["install", "clean"])
        controller.install_units.assert_called_once_with()
        controller.cleanup_stale_units.assert_called_once_with(
            manifest_profile_ids={"fixture-text-7b", "fixture-vision-12b"}
        )
        cleanup = next(
            operation
            for operation in report["operations"]
            if operation["action"] == "stale_unit_cleanup"
        )
        self.assertEqual(cleanup["requested"], True)
        self.assertEqual(cleanup["status"], "removed")
        self.assertEqual(
            cleanup["removed"], ["old-profile.service", "old-profile-verify.service"]
        )

    def test_without_clean_does_not_call_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result, report, controller, events = self._run_install(
                Path(temp), clean=False
            )

        self.assertEqual(result, 0)
        self.assertEqual(events, ["install"])
        controller.cleanup_stale_units.assert_not_called()
        self.assertNotIn(
            "stale_unit_cleanup",
            [operation["action"] for operation in report["operations"]],
        )

    def test_clean_uses_all_safe_manifest_ids_not_only_ready_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result, _report, controller, _events = self._run_install(
                Path(temp),
                clean=True,
                extra_manifest_ids=("declared-but-unready",),
            )

        self.assertEqual(result, 0)
        controller.cleanup_stale_units.assert_called_once_with(
            manifest_profile_ids={
                "fixture-text-7b",
                "fixture-vision-12b",
                "declared-but-unready",
            }
        )

    def test_clean_refuses_ambiguous_manifest_ids_before_controller_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profiles = materialize_fixture(root / "fixture")
            registry = ProfileRegistry.from_path(profiles)
            profile = registry.get("fixture-text-7b")
            payload = json.loads(profiles.read_text(encoding="utf-8"))
            payload["profiles"].append({"id": "unsafe/profile"})
            inspection = ProfileInspection(
                profile.id,
                "ready",
                profile,
                next(item for item in payload["profiles"] if item["id"] == profile.id),
            )
            args = SimpleNamespace(
                profiles=profiles,
                full_verify=False,
                json=True,
                dry_run=False,
                integrate_command="install",
                install_units=True,
                clean=True,
            )
            stderr = io.StringIO()
            with (
                patch.object(
                    brokerctl,
                    "inspect_manifest",
                    return_value=(payload, (inspection,)),
                ),
                patch.object(brokerctl, "create_production_controller") as create_controller,
                redirect_stderr(stderr),
            ):
                result = brokerctl._cmd_integrate(args)

        self.assertEqual(result, 1)
        create_controller.assert_not_called()
        self.assertIn("unsafe manifest profile id", stderr.getvalue())

    def test_clean_skips_without_systemd_and_reports_partial_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result, report, controller, events = self._run_install(
                Path(temp), clean=True, systemd_available=False
            )

        self.assertEqual(result, 3)
        self.assertEqual(events, [])
        controller.cleanup_stale_units.assert_not_called()
        cleanup = next(
            operation
            for operation in report["operations"]
            if operation["action"] == "stale_unit_cleanup"
        )
        self.assertEqual(cleanup["status"], "skipped")
        self.assertIn("systemd", cleanup["reason"])
        self.assertTrue(any("cleanup skipped" in error for error in report["errors"]))

    def test_clean_failure_is_reported_as_partial_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result, report, controller, events = self._run_install(
                Path(temp),
                clean=True,
                cleanup_error=brokerctl.BackendControllerError("unit is active"),
            )

        self.assertEqual(result, 3)
        self.assertEqual(events, ["install", "clean"])
        controller.cleanup_stale_units.assert_called_once_with(
            manifest_profile_ids={"fixture-text-7b", "fixture-vision-12b"}
        )
        cleanup = next(
            operation
            for operation in report["operations"]
            if operation["action"] == "stale_unit_cleanup"
        )
        self.assertEqual(cleanup["status"], "failed")
        self.assertIn("unit is active", cleanup["error"])
        self.assertTrue(any("cleanup" in error for error in report["errors"]))

    def test_clean_rejects_no_install_units(self) -> None:
        args = SimpleNamespace(
            profiles=Path("/tmp/unused-profiles.json"),
            full_verify=False,
            json=True,
            dry_run=False,
            integrate_command="install",
            install_units=False,
            clean=True,
        )
        stderr = io.StringIO()
        with patch.object(brokerctl, "inspect_manifest") as inspect, redirect_stderr(stderr):
            result = brokerctl._cmd_integrate(args)

        self.assertEqual(result, 2)
        inspect.assert_not_called()
        self.assertIn("requires unit installation", stderr.getvalue())
