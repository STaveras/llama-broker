from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from llama_broker.broker_profiles import ProfileError
from llama_broker.setup_wizard import (
    apply_profile_plan,
    build_setup_plan,
    discover_candidates,
    render_report,
)


class SetupWizardTests(unittest.TestCase):
    def test_discovery_separates_qualified_and_experimental_runtimes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bin_dir = root / "bin"
            models = root / "models"
            bin_dir.mkdir()
            models.mkdir()
            llama = bin_dir / "llama-server"
            llama.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            llama.chmod(0o755)
            vllm = bin_dir / "vllm"
            vllm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            vllm.chmod(0o755)
            model = models / "demo.gguf"
            model.write_bytes(b"demo")

            report = discover_candidates([bin_dir, models])

            self.assertEqual([item.runtime for item in report.runtimes], ["llamacpp", "vllm"])
            self.assertTrue(report.runtimes[0].qualified)
            self.assertFalse(report.runtimes[1].qualified)
            self.assertEqual(report.models[0].path, model.resolve())
            self.assertTrue(report.models[0].verified)

    def test_plan_requires_explicit_runtime_and_model_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "llama-server"
            runtime.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runtime.chmod(0o755)
            model = root / "demo.gguf"
            model.write_bytes(b"demo")
            report = discover_candidates([root])

            empty = build_setup_plan(report)
            self.assertEqual(empty.operations, ())

            selected = build_setup_plan(
                report,
                runtime_path=runtime,
                model_path=model,
            )
            self.assertEqual(selected.selected_model, model.resolve())
            self.assertEqual(selected.selected_runtime, runtime.resolve())
            self.assertEqual(selected.operations[0]["action"], "create_profile")

    def test_report_has_versioned_machine_readable_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = discover_candidates([Path(temp)])
            payload = json.loads(render_report(report, json_output=True))
            self.assertEqual(payload["schema_version"], 1)
            self.assertIn("runtimes", payload)
            self.assertIn("models", payload)

    def test_apply_profile_plan_rolls_back_manifest_when_model_link_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "llama-server"
            runtime.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runtime.chmod(0o755)
            model = root / "demo.gguf"
            model.write_bytes(b"demo")
            report = discover_candidates([root])
            plan = build_setup_plan(
                report,
                runtime_path=runtime,
                model_path=model,
                create_model_link=True,
            )
            profiles = root / "profiles.json"
            models_dir = root / "models"
            models_dir.mkdir()
            (models_dir / model.name).write_text("unrelated", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                apply_profile_plan(
                    plan,
                    profiles,
                    models_dir=models_dir,
                    create_model_link=True,
                )

            self.assertFalse(profiles.exists())

    def test_setup_plan_can_include_startup_and_integrate_operations(self) -> None:
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
                enable_startup=True,
                offer_integrate=True,
            )

            actions = [item["action"] for item in plan.operations]
            self.assertEqual(
                actions,
                ["create_profile", "enable_broker_startup", "offer_integrate"],
            )

    def test_setup_does_not_overwrite_unsupported_manifest_schema(self) -> None:
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
            profiles = root / "profiles.json"
            original = '{"schema_version": 99, "profiles": []}\n'
            profiles.write_text(original, encoding="utf-8")

            with self.assertRaises(ProfileError):
                apply_profile_plan(plan, profiles)

            self.assertEqual(profiles.read_text(encoding="utf-8"), original)
            self.assertEqual(list(root.glob("profiles.json.bak-*")), [])

    def test_setup_preserves_existing_inheritance_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "llama-server"
            model = root / "model.gguf"
            runtime.write_bytes(b"runtime")
            runtime.chmod(0o755)
            model.write_bytes(b"model")
            seed = build_setup_plan(
                discover_candidates([root]),
                runtime_path=runtime,
                model_path=model,
            )
            assert seed.profile_record is not None
            base = dict(seed.profile_record)
            base["id"] = "base"
            base["alias"] = "base"
            child = {
                "id": "child",
                "extends": "base",
                "alias": "child",
                "aliases": ["child"],
            }
            profiles_path = root / "profiles.json"
            profiles_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profiles": [base, child],
                    }
                ),
                encoding="utf-8",
            )

            seed.profile_record["id"] = "added"
            seed.profile_record["alias"] = "added"
            apply_profile_plan(seed, profiles_path)

            saved = json.loads(profiles_path.read_text(encoding="utf-8"))
            saved_child = next(item for item in saved["profiles"] if item["id"] == "child")
            self.assertEqual(saved_child["extends"], "base")


if __name__ == "__main__":
    unittest.main()
