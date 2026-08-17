from __future__ import annotations

import contextlib
import argparse
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import llama_broker.cli as brokerctl
from llama_broker.integration import inspect_manifest, report_for
from llama_broker.setup_wizard import build_setup_plan, discover_candidates


class IntegrationTests(unittest.TestCase):
    def test_interactive_dry_run_rejects_json(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            result = brokerctl.main(["integrate", "--dry-run", "--json"])
        self.assertEqual(result, 2)
        self.assertIn("cannot be combined", output.getvalue())

    def test_interactive_dry_run_does_not_expose_confirmation_bypass(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            brokerctl.build_parser().parse_args(["integrate", "--dry-run", "--yes"])
        self.assertEqual(raised.exception.code, 2)

    def test_profile_inspection_keeps_ready_profile_when_another_is_broken(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "llama-server"
            runtime.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runtime.chmod(0o755)
            model = root / "demo.gguf"
            model.write_bytes(b"demo")
            discovery = discover_candidates([root])
            plan = build_setup_plan(
                discovery,
                runtime_path=runtime,
                model_path=model,
            )
            manifest = {
                "schema_version": 1,
                "default_profile": plan.profile_record["id"],
                "profiles": [plan.profile_record, {"id": "broken"}],
            }
            path = root / "profiles.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            _, inspections = inspect_manifest(path)
            statuses = {item.profile_id: item.status for item in inspections}
            self.assertEqual(statuses[plan.profile_record["id"]], "ready")
            self.assertEqual(statuses["broken"], "broken")
            self.assertEqual(report_for(inspections)["counts"]["ready"], 1)

    def test_profile_inspection_resolves_local_inheritance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "llama-server"
            runtime.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runtime.chmod(0o755)
            model = root / "demo.gguf"
            model.write_bytes(b"demo")
            plan = build_setup_plan(
                discover_candidates([root]),
                runtime_path=runtime,
                model_path=model,
            )
            parent = dict(plan.profile_record)
            child = {"id": "child", "extends": parent["id"], "alias": "child"}
            path = root / "profiles.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "default_profile": "child",
                        "profiles": [parent, child],
                    }
                ),
                encoding="utf-8",
            )

            _, inspections = inspect_manifest(path)

            statuses = {item.profile_id: item.status for item in inspections}
            self.assertEqual(statuses, {parent["id"]: "ready", "child": "ready"})

    def test_integrate_reports_partial_success_without_calling_systemd(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "llama-server"
            runtime.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runtime.chmod(0o755)
            model = root / "demo.gguf"
            model.write_bytes(b"demo")
            discovery = discover_candidates([root])
            plan = build_setup_plan(discovery, runtime_path=runtime, model_path=model)
            path = root / "profiles.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "default_profile": plan.profile_record["id"],
                        "profiles": [plan.profile_record, {"id": "broken"}],
                    }
                ),
                encoding="utf-8",
            )
            systemd_dir = root / "systemd"

            class FakeController:
                def unit_path(self, profile_id: str) -> Path:
                    return systemd_dir / f"llama-broker-profile-{profile_id}.service"

                def render_unit(self, profile_id: str) -> str:
                    return f"[Unit]\nDescription={profile_id}\n"

                def install_units(self) -> None:
                    raise AssertionError("systemd must not be called without a user session")

            args = argparse.Namespace(
                profiles=path,
                dry_run=False,
                json=True,
                verify=True,
                clean=False,
                set_default=False,
                yes=False,
                install_units=True,
            )
            output = io.StringIO()
            with (
                contextlib.redirect_stdout(output),
                patch.object(brokerctl, "create_production_controller", return_value=FakeController()),
                patch.object(brokerctl.broker_runtime, "SYSTEMD_USER_DIR", systemd_dir),
                patch("llama_broker.configure.main", return_value=0),
                patch.dict(os.environ, {"DBUS_SESSION_BUS_ADDRESS": ""}, clear=False),
            ):
                result = brokerctl._cmd_integrate(args)
            payload = json.loads(output.getvalue())
            self.assertEqual(result, 3)
            self.assertEqual(payload["counts"]["skipped"], 1)
            self.assertEqual(payload["counts"]["broken"], 1)

            args.dry_run = True
            args.json = False
            args.yes = True
            dry_output = io.StringIO()
            with (
                contextlib.redirect_stdout(dry_output),
                patch.object(brokerctl, "create_production_controller", return_value=FakeController()),
                patch.object(brokerctl.broker_runtime, "SYSTEMD_USER_DIR", systemd_dir),
                patch("llama_broker.configure.main", return_value=0) as configure_main,
                patch("builtins.input", return_value="y"),
                patch.dict(os.environ, {"DBUS_SESSION_BUS_ADDRESS": ""}, clear=False),
            ):
                dry_result = brokerctl._cmd_integrate(args)
            self.assertEqual(dry_result, 3)
            self.assertIn("broker unit", dry_output.getvalue().casefold())
            self.assertIn("client/provider", dry_output.getvalue().casefold())
            configure_main.assert_called_once()


if __name__ == "__main__":
    unittest.main()
