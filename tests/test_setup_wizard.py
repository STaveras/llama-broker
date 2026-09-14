from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_broker.broker_profiles import ProfileError, ProfileRegistry
from llama_broker.profile_config import ApplyError
from llama_broker.setup_wizard import (
    apply_profile_plan,
    build_setup_plan,
    discover_candidates,
    render_report,
)
from tests.fixtures import materialize_fixture


class SetupWizardTests(unittest.TestCase):
    def _schema2_setup_fixture(
        self, root: Path, *, registered: bool = True
    ) -> tuple[Path, Path, Path, Path, str, str, str]:
        profiles = materialize_fixture(root / "fixture")
        data = json.loads(profiles.read_text(encoding="utf-8"))
        profile = next(
            item for item in data["profiles"] if item["id"] == "fixture-text-7b"
        )

        bundle = root / "runtime-bundle"
        binary = bundle / "bin" / "llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"catalog-runtime")
        binary.chmod(0o755)
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        runtime_id = "beellama-kvarn-deadbeef1234"
        revision = "deployment-test-revision"
        catalog = root / "catalog.json"
        catalog.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "deployment_revision": revision,
                    "temporary_aliases": {"beellama": runtime_id},
                    "runtimes": [
                        {
                            "id": runtime_id,
                            "runtime": "llamacpp",
                            "display_name": "BeeLlama adaptive KVarN",
                            "bundle_root": str(bundle),
                            "server_binary": str(binary),
                            "server_sha256": digest,
                            "environment": {
                                "CATALOG_ONLY": "retained",
                                "CONFLICT": "catalog-loses",
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        profile.pop("runtime", None)
        profile.pop("server_binary", None)
        profile.pop("server_sha256", None)
        profile["runtime_build"] = runtime_id
        profile["environment"] = {
            "GGML_CUDA_PDL": "0",
            "CONFLICT": "profile-wins",
        }
        data["schema_version"] = 2
        data["deployment_revision"] = revision
        data.pop("runtime_builds", None)
        data["profiles"] = [profile]
        profiles.write_text(json.dumps(data), encoding="utf-8")

        model = root / "new-model.gguf"
        model.write_bytes(b"new-model")
        selected_runtime = binary
        if not registered:
            selected_runtime = root / "unregistered" / "llama-server"
            selected_runtime.parent.mkdir()
            # Keep the catalog hash valid so this case isolates an unregistered path.
            selected_runtime.write_bytes(binary.read_bytes())
            selected_runtime.chmod(0o755)
        return profiles, catalog, selected_runtime, model, runtime_id, revision, digest

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
            assert selected.profile_record is not None
            self.assertNotIn("runtime_build", selected.profile_record)
            assert selected.runtime_build_record is not None
            self.assertTrue(str(selected.runtime_build_record["id"]).startswith("llamacpp-"))
            self.assertEqual(selected.runtime_build_record["runtime"], "llamacpp")
            self.assertEqual(
                selected.runtime_build_record["server_binary"], str(runtime.resolve())
            )

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

    def test_apply_revalidates_selected_artifacts_before_publication(self) -> None:
        for changed_kind in ("runtime", "model"):
            with self.subTest(changed_kind=changed_kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                runtime = root / "llama-server"
                runtime.write_bytes(b"runtime-before")
                runtime.chmod(0o755)
                model = root / "new-model.gguf"
                model.write_bytes(b"model-before")
                plan = build_setup_plan(
                    discover_candidates([runtime, model]),
                    runtime_path=runtime,
                    model_path=model,
                    create_model_link=True,
                )
                profiles = root / "profiles.json"
                models_dir = root / "models"

                if changed_kind == "runtime":
                    runtime.write_bytes(b"runtime-after")
                else:
                    model.write_bytes(b"model-after")

                with self.assertRaisesRegex(ProfileError, r"artifact|SHA-256|hash|identity"):
                    apply_profile_plan(
                        plan,
                        profiles,
                        models_dir=models_dir,
                        create_model_link=True,
                    )

                self.assertFalse(profiles.exists())
                self.assertFalse(models_dir.exists())

    def test_apply_fsyncs_manifest_directory_after_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "llama-server"
            runtime.write_bytes(b"runtime")
            runtime.chmod(0o755)
            model = root / "new-model.gguf"
            model.write_bytes(b"model")
            plan = build_setup_plan(
                discover_candidates([runtime, model]),
                runtime_path=runtime,
                model_path=model,
            )
            profiles = root / "profiles.json"
            events: list[str] = []
            real_replace = os.replace
            real_fsync = os.fsync

            def record_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
                result = real_replace(source, destination)
                if Path(destination) == profiles:
                    events.append("replace")
                return result

            def record_fsync(fd: int) -> None:
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    events.append("directory_fsync")
                return real_fsync(fd)

            with patch("llama_broker.setup_wizard.os.replace", side_effect=record_replace), patch(
                "llama_broker.setup_wizard.os.fsync", side_effect=record_fsync
            ):
                apply_profile_plan(plan, profiles)

            self.assertEqual(events, ["replace", "directory_fsync"])

    def test_apply_reports_original_and_rollback_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profiles = materialize_fixture(root / "fixture")
            runtime = root / "llama-server"
            runtime.write_bytes(b"runtime")
            runtime.chmod(0o755)
            model = root / "new-model.gguf"
            model.write_bytes(b"model")
            plan = build_setup_plan(
                discover_candidates([runtime, model]),
                runtime_path=runtime,
                model_path=model,
            )
            original_manifest = profiles.read_bytes()
            real_fsync = os.fsync

            def fail_directory_fsync(fd: int) -> None:
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    raise OSError("publish directory fsync failed")
                return real_fsync(fd)

            with patch(
                "llama_broker.setup_wizard.os.fsync", side_effect=fail_directory_fsync
            ), self.assertRaises(ApplyError) as raised:
                apply_profile_plan(plan, profiles)

            self.assertIn("setup apply failed", str(raised.exception))
            self.assertIn("rollback also failed", str(raised.exception))
            self.assertIn("publish directory fsync failed", str(raised.exception))
            self.assertEqual(profiles.read_bytes(), original_manifest)

    def test_schema1_setup_rejects_invalid_existing_profile_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profiles = materialize_fixture(root / "fixture")
            data = json.loads(profiles.read_text(encoding="utf-8"))
            data["schema_version"] = 1
            data["profiles"][0]["context_size"] = 0
            profiles.write_text(json.dumps(data), encoding="utf-8")
            manifest_before = profiles.read_bytes()

            runtime = root / "llama-server"
            runtime.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runtime.chmod(0o755)
            model = root / "new-model.gguf"
            model.write_bytes(b"new-model")
            plan = build_setup_plan(
                discover_candidates([runtime, model]),
                runtime_path=runtime,
                model_path=model,
            )
            self.assertIsNotNone(plan.profile_record)

            with self.assertRaisesRegex(ProfileError, r"context_size.*positive integer"):
                apply_profile_plan(plan, profiles)

            self.assertEqual(profiles.read_bytes(), manifest_before)

    def test_no_selection_does_not_migrate_legacy_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profiles = materialize_fixture(root / "fixture")
            data = json.loads(profiles.read_text(encoding="utf-8"))
            data.pop("schema_version")
            profiles.write_text(json.dumps(data), encoding="utf-8")
            manifest_before = profiles.read_bytes()
            plan = build_setup_plan(discover_candidates([root]))

            self.assertIsNone(plan.profile_record)
            result = apply_profile_plan(plan, profiles)

            self.assertEqual(result["skipped"], ["no explicit selection"])
            self.assertEqual(profiles.read_bytes(), manifest_before)
            self.assertEqual(list(profiles.parent.glob(f"{profiles.name}.bak-*")), [])

    def test_apply_schema2_setup_reuses_catalog_build_id_without_downgrade(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (
                profiles,
                catalog,
                runtime,
                model,
                runtime_id,
                revision,
                digest,
            ) = self._schema2_setup_fixture(root)
            plan = build_setup_plan(
                discover_candidates([runtime, model]),
                runtime_path=runtime,
                model_path=model,
            )
            assert plan.profile_record is not None
            profile_id = str(plan.profile_record["id"])
            discovery_label = f"llamacpp-{digest[:12]}"
            self.assertNotEqual(runtime_id, discovery_label)
            catalog_before = catalog.read_bytes()

            with patch.dict(
                os.environ,
                {"LLAMA_BROKER_RUNTIME_CATALOG": str(catalog)},
                clear=False,
            ):
                apply_profile_plan(plan, profiles)
                saved = json.loads(profiles.read_text(encoding="utf-8"))
                self.assertEqual(saved["schema_version"], 2)
                self.assertEqual(saved["deployment_revision"], revision)
                self.assertNotIn("runtime_builds", saved)
                added = next(item for item in saved["profiles"] if item["id"] == profile_id)
                self.assertEqual(added["runtime_build"], runtime_id)
                self.assertNotEqual(added["runtime_build"], discovery_label)

                registry = ProfileRegistry.from_path(profiles)
                loaded = registry.get(profile_id)
                self.assertEqual(loaded.runtime_build, runtime_id)
                self.assertEqual(loaded.server_binary, runtime.resolve())
                self.assertEqual(loaded.server_sha256, digest)
                self.assertEqual(
                    loaded.command(host="127.0.0.1", port=10001)[0],
                    str(runtime.resolve()),
                )

            self.assertEqual(catalog.read_bytes(), catalog_before)

    def test_schema2_apply_revalidates_catalog_artifacts_before_publication(self) -> None:
        for changed_kind in ("runtime", "model"):
            with self.subTest(changed_kind=changed_kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                profiles, catalog, runtime, model, _runtime_id, _revision, _digest = (
                    self._schema2_setup_fixture(root)
                )
                plan = build_setup_plan(
                    discover_candidates([runtime, model]),
                    runtime_path=runtime,
                    model_path=model,
                    create_model_link=True,
                )
                manifest_before = profiles.read_bytes()
                models_dir = root / "linked-models"
                if changed_kind == "runtime":
                    runtime.write_bytes(b"catalog-runtime-replaced")
                else:
                    model.write_bytes(b"new-model-replaced")

                with patch.dict(
                    os.environ,
                    {"LLAMA_BROKER_RUNTIME_CATALOG": str(catalog)},
                    clear=False,
                ):
                    with self.assertRaisesRegex(ProfileError, r"artifact|SHA-256|hash|identity"):
                        apply_profile_plan(
                            plan,
                            profiles,
                            models_dir=models_dir,
                            create_model_link=True,
                        )

                self.assertEqual(profiles.read_bytes(), manifest_before)
                self.assertFalse(models_dir.exists())

    def test_unregistered_schema2_setup_refuses_before_mutation_or_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profiles, catalog, runtime, model, _runtime_id, _revision, _digest = (
                self._schema2_setup_fixture(root, registered=False)
            )
            plan = build_setup_plan(
                discover_candidates([runtime, model]),
                runtime_path=runtime,
                model_path=model,
                create_model_link=True,
            )
            manifest_before = profiles.read_bytes()
            catalog_before = catalog.read_bytes()
            models_dir = root / "linked-models"

            # apply_profile_plan owns manifest/model-link mutation only; startup and
            # systemd unit installation are separate CLI operations.
            original_mkstemp = tempfile.mkstemp
            with patch.dict(
                os.environ,
                {"LLAMA_BROKER_RUNTIME_CATALOG": str(catalog)},
                clear=False,
            ), patch("llama_broker.setup_wizard.os.replace") as replace, patch(
                "llama_broker.setup_wizard.tempfile.mkstemp",
                side_effect=original_mkstemp,
            ) as mkstemp:
                with self.assertRaisesRegex(ProfileError, r"unknown runtime build"):
                    apply_profile_plan(
                        plan,
                        profiles,
                        models_dir=models_dir,
                        create_model_link=True,
                    )

            replace.assert_not_called()
            mkstemp.assert_not_called()
            self.assertEqual(profiles.read_bytes(), manifest_before)
            self.assertEqual(list(profiles.parent.glob(f"{profiles.name}.bak-*")), [])
            self.assertEqual(catalog.read_bytes(), catalog_before)
            self.assertFalse(models_dir.exists())

    def test_schema2_inline_runtime_catalog_refuses_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profiles, catalog, runtime, model, _runtime_id, _revision, _digest = (
                self._schema2_setup_fixture(root)
            )
            plan = build_setup_plan(
                discover_candidates([runtime, model]),
                runtime_path=runtime,
                model_path=model,
                create_model_link=True,
            )
            assert plan.runtime_build_record is not None
            data = json.loads(profiles.read_text(encoding="utf-8"))
            data["runtime_builds"] = [plan.runtime_build_record]
            profiles.write_text(json.dumps(data), encoding="utf-8")
            manifest_before = profiles.read_bytes()
            catalog_before = catalog.read_bytes()
            models_dir = root / "linked-models"

            original_mkstemp = tempfile.mkstemp
            with patch.dict(
                os.environ,
                {"LLAMA_BROKER_RUNTIME_CATALOG": str(catalog)},
                clear=False,
            ), patch("llama_broker.setup_wizard.os.replace") as replace, patch(
                "llama_broker.setup_wizard.tempfile.mkstemp",
                side_effect=original_mkstemp,
            ) as mkstemp:
                with self.assertRaisesRegex(
                    ProfileError,
                    r"schema 2.*external catalog.*runtime_builds",
                ):
                    apply_profile_plan(
                        plan,
                        profiles,
                        models_dir=models_dir,
                        create_model_link=True,
                    )

            replace.assert_not_called()
            mkstemp.assert_not_called()
            self.assertEqual(profiles.read_bytes(), manifest_before)
            self.assertEqual(catalog.read_bytes(), catalog_before)
            self.assertFalse(models_dir.exists())

    def test_legacy_migration_rolls_back_when_profile_composition_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profiles = materialize_fixture(root / "fixture")
            data = json.loads(profiles.read_text(encoding="utf-8"))
            data.pop("schema_version")
            profiles.write_text(json.dumps(data), encoding="utf-8")
            manifest_before = profiles.read_bytes()

            existing_backup = profiles.with_name(f"{profiles.name}.bak-existing")
            existing_backup.write_bytes(b"operator backup\n")
            backups_before = {
                path.name: path.read_bytes()
                for path in profiles.parent.glob(f"{profiles.name}.bak-*")
            }

            runtime = root / "llama-server"
            runtime.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runtime.chmod(0o755)
            model = root / "fixture-text-7b.gguf"
            model.write_bytes(b"new-model")
            plan = build_setup_plan(
                discover_candidates([runtime, model]),
                runtime_path=runtime,
                model_path=model,
            )
            self.assertIsNotNone(plan.profile_record)

            with self.assertRaisesRegex(ValueError, r"profile already exists"):
                apply_profile_plan(plan, profiles)

            self.assertEqual(profiles.read_bytes(), manifest_before)
            backups_after = {
                path.name: path.read_bytes()
                for path in profiles.parent.glob(f"{profiles.name}.bak-*")
            }
            self.assertEqual(backups_after, backups_before)

    def test_legacy_rollback_preserves_concurrent_manifest_and_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profiles = materialize_fixture(root / "fixture")
            data = json.loads(profiles.read_text(encoding="utf-8"))
            data.pop("schema_version")
            profiles.write_text(json.dumps(data), encoding="utf-8")

            existing_backup = profiles.with_name(f"{profiles.name}.bak-existing")
            existing_backup.write_bytes(b"operator backup")
            concurrent_backup = profiles.with_name(f"{profiles.name}.bak-concurrent")
            concurrent_manifest = b"concurrent editor manifest\n"
            concurrent_backup_bytes = b"concurrent editor backup\n"

            runtime = root / "llama-server"
            runtime.write_bytes(b"runtime")
            runtime.chmod(0o755)
            model = root / "new-model.gguf"
            model.write_bytes(b"model")
            plan = build_setup_plan(
                discover_candidates([runtime, model]),
                runtime_path=runtime,
                model_path=model,
            )

            def concurrent_failure(
                _manifest: dict[str, object], *, directory: Path | None = None
            ) -> None:
                profiles.write_bytes(concurrent_manifest)
                concurrent_backup.write_bytes(concurrent_backup_bytes)
                raise ValueError("later apply failure")

            with patch(
                "llama_broker.setup_wizard.validate_manifest",
                side_effect=concurrent_failure,
            ):
                with self.assertRaisesRegex(ValueError, r"later apply failure"):
                    apply_profile_plan(plan, profiles)

            self.assertEqual(profiles.read_bytes(), concurrent_manifest)
            self.assertEqual(existing_backup.read_bytes(), b"operator backup")
            self.assertEqual(concurrent_backup.read_bytes(), concurrent_backup_bytes)
            backup_names = {
                path.name for path in profiles.parent.glob(f"{profiles.name}.bak-*")
            }
            self.assertEqual(backup_names, {existing_backup.name, concurrent_backup.name})

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
            assert seed.runtime_build_record is not None
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
                        "runtime_builds": [seed.runtime_build_record],
                        "profiles": [base, child],
                    }
                ),
                encoding="utf-8",
            )

            seed.profile_record["id"] = "added"
            seed.profile_record["alias"] = "added"
            seed.profile_record["aliases"] = ["added"]
            apply_profile_plan(seed, profiles_path)

            saved = json.loads(profiles_path.read_text(encoding="utf-8"))
            saved_child = next(item for item in saved["profiles"] if item["id"] == "child")
            saved_added = next(item for item in saved["profiles"] if item["id"] == "added")
            self.assertEqual(saved_child["extends"], "base")
            self.assertEqual(saved_added["runtime_build"], seed.runtime_build_record["id"])
            self.assertEqual(len(saved["runtime_builds"]), 1)
            self.assertEqual(
                saved["runtime_builds"][0]["id"],
                seed.runtime_build_record["id"],
            )


if __name__ == "__main__":
    unittest.main()
