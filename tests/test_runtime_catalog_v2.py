from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llama_broker.cli as brokerctl
from llama_broker.broker_profiles import ProfileError, ProfileRegistry
from llama_broker.integration import inspect_manifest
from tests.fixtures import materialize_fixture


class ExternalRuntimeCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.profiles = materialize_fixture(self.root / "fixture")
        self.data = json.loads(self.profiles.read_text(encoding="utf-8"))
        self.profile = next(
            item for item in self.data["profiles"] if item["id"] == "fixture-text-7b"
        )
        self.bundle = self.root / "runtime-bundle"
        self.binary = self.bundle / "bin" / "llama-server"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b"external-runtime")
        self.binary.chmod(0o755)
        self.source_repo = self.root / "source-repo"
        self.source_repo.mkdir()
        self.provenance = self.source_repo / ".runtime-provenance.json"
        self.provenance.write_text("{}\n", encoding="utf-8")
        self.catalog = self.root / "catalog.json"
        self.revision = "deployment-test-revision"
        self.runtime_id = "beellama-kvarn-deadbeef1234"
        self._make_v2_manifest()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _make_v2_manifest(self) -> None:
        self.profile.pop("runtime", None)
        self.profile.pop("server_binary", None)
        self.profile.pop("server_sha256", None)
        self.profile["runtime_build"] = self.runtime_id
        self.profile["environment"] = {
            "CATALOG_DEFAULT": "preserved",
            "CONFLICT": "profile-wins",
        }
        digest = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        catalog = {
            "schema_version": 2,
            "deployment_revision": self.revision,
            "temporary_aliases": {"beellama": self.runtime_id},
            "runtimes": [
                {
                    "id": self.runtime_id,
                    "runtime": "llamacpp",
                    "display_name": "BeeLlama adaptive KVarN",
                    "bundle_root": str(self.bundle),
                    "server_binary": str(self.binary),
                    "server_sha256": digest,
                    "environment": {
                        "GGML_CUDA_PDL": "0",
                        "CATALOG_DEFAULT": "preserved",
                        "CONFLICT": "catalog-loses",
                        "CATALOG_ONLY": "retained",
                    },
                    "source_repository": str(self.source_repo),
                    "provenance_file": str(self.provenance),
                    "source_branch": "feature/adaptive-kv-stream",
                    "source_commit": "abc1234",
                    "dirty": True,
                    "selected_profile_ids": ["fixture-text-7b"],
                }
            ],
        }
        self.catalog.write_text(json.dumps(catalog), encoding="utf-8")
        self.data["schema_version"] = 2
        self.data["deployment_revision"] = self.revision
        self.data.pop("runtime_builds", None)
        self.data["profiles"] = [self.profile]
        self.profiles.write_text(json.dumps(self.data), encoding="utf-8")

    def _registry(self) -> ProfileRegistry:
        with patch.dict(
            os.environ,
            {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.catalog)},
            clear=False,
        ):
            return ProfileRegistry.from_path(self.profiles)

    def test_v2_derives_runtime_fields_and_merges_catalog_environment(self) -> None:
        profile = self._registry().get("fixture-text-7b")

        self.assertEqual(profile.runtime_build, self.runtime_id)
        self.assertEqual(profile.runtime, "llamacpp")
        self.assertEqual(profile.server_binary, self.binary.resolve())
        self.assertEqual(profile.server_sha256, hashlib.sha256(self.binary.read_bytes()).hexdigest())
        self.assertEqual(profile.environment["CATALOG_DEFAULT"], "preserved")
        self.assertEqual(profile.environment["CATALOG_ONLY"], "retained")
        self.assertEqual(profile.environment["CONFLICT"], "profile-wins")
        self.assertEqual(profile.bundle_root, self.bundle.resolve())
        self.assertEqual(profile.source_repository, self.source_repo.resolve())
        self.assertEqual(profile.provenance_file, self.provenance.resolve())
        self.assertEqual(profile.source_branch, "feature/adaptive-kv-stream")
        self.assertEqual(profile.source_commit, "abc1234")
        self.assertTrue(profile.dirty)
        self.assertEqual(profile.selected_profile_ids, ("fixture-text-7b",))

    def test_catalog_path_environment_override_is_used(self) -> None:
        with patch.dict(
            os.environ,
            {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.catalog)},
            clear=False,
        ):
            registry = ProfileRegistry.from_path(self.profiles)
        self.assertEqual(registry.runtime_catalog_path, self.catalog.resolve())

    def test_shared_deployment_revision_mismatch_fails_closed(self) -> None:
        catalog = json.loads(self.catalog.read_text(encoding="utf-8"))
        catalog["deployment_revision"] = "different-revision"
        self.catalog.write_text(json.dumps(catalog), encoding="utf-8")

        with patch.dict(
            os.environ,
            {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.catalog)},
            clear=False,
        ), self.assertRaisesRegex(ProfileError, "deployment_revision"):
            ProfileRegistry.from_path(self.profiles)

    def test_v2_requires_runtime_build(self) -> None:
        self.profile.pop("runtime_build")
        self.profiles.write_text(json.dumps(self.data), encoding="utf-8")

        with patch.dict(
            os.environ,
            {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.catalog)},
            clear=False,
        ), self.assertRaisesRegex(ProfileError, "runtime_build"):
            ProfileRegistry.from_path(self.profiles)

    def test_explicit_legacy_runtime_fields_must_match_catalog(self) -> None:
        self.profile["server_binary"] = str(self.root / "wrong-server")
        self.profile["server_sha256"] = "0" * 64
        self.profiles.write_text(json.dumps(self.data), encoding="utf-8")

        with patch.dict(
            os.environ,
            {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.catalog)},
            clear=False,
        ), self.assertRaisesRegex(ProfileError, "does not match"):
            ProfileRegistry.from_path(self.profiles)

    def test_temporary_runtime_alias_resolves_then_disappears(self) -> None:
        self.profile["runtime_build"] = "beellama"
        self.profiles.write_text(json.dumps(self.data), encoding="utf-8")
        profile = self._registry().get("fixture-text-7b")
        self.assertEqual(profile.runtime_build, self.runtime_id)

        catalog = json.loads(self.catalog.read_text(encoding="utf-8"))
        catalog.pop("temporary_aliases")
        self.catalog.write_text(json.dumps(catalog), encoding="utf-8")
        with patch.dict(
            os.environ,
            {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.catalog)},
            clear=False,
        ), self.assertRaisesRegex(ProfileError, "unknown runtime build"):
            ProfileRegistry.from_path(self.profiles)

    def test_v2_integration_uses_external_catalog_and_reports_metadata(self) -> None:
        with patch.dict(
            os.environ,
            {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.catalog)},
            clear=False,
        ):
            _payload, inspections = inspect_manifest(self.profiles, verify_artifacts=True)

        self.assertEqual(len(inspections), 1)
        inspection = inspections[0]
        self.assertEqual(inspection.status, "ready")
        rendered = inspection.as_dict()
        self.assertEqual(rendered["runtime_build"], self.runtime_id)
        self.assertEqual(rendered["deployment_revision"], self.revision)
        self.assertIn("warnings", rendered)
        self.assertIn("degraded_features", rendered)

    def test_verify_report_exposes_revision_warnings_and_runtime_provenance(self) -> None:
        profile = self._registry().get("fixture-text-7b")

        result = brokerctl._profile_verification_result(profile, verify_artifacts=False)

        self.assertEqual(result["runtime_build"], self.runtime_id)
        self.assertEqual(result["deployment_revision"], self.revision)
        self.assertEqual(result["bundle_root"], str(self.bundle.resolve()))
        self.assertEqual(result["provenance_file"], str(self.provenance.resolve()))
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["degraded_features"], [])

    def test_runtime_list_json_exposes_catalog_build_and_selected_profiles(self) -> None:
        output = io.StringIO()
        with patch.dict(
            os.environ,
            {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.catalog)},
            clear=False,
        ), redirect_stdout(output):
            exit_code = brokerctl._cmd_runtime_list(
                SimpleNamespace(profiles=self.profiles, json=True)
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        rows = [row for row in payload["data"] if row["name"] == "llamacpp"]
        self.assertTrue(rows)
        self.assertTrue(any(row.get("builds") for row in rows))
        build = next(
            build
            for row in rows
            for build in row["builds"]
            if build["id"] == self.runtime_id
        )
        self.assertIn("fixture-text-7b", build["selected_profile_ids"])

    def test_missing_optional_draft_and_moe_cache_degrade_without_parse_failure(self) -> None:
        self.data["schema_version"] = 1
        self.data.pop("deployment_revision", None)
        self.data["runtime_builds"] = []
        self.profile["runtime"] = "llamacpp"
        self.profile.pop("runtime_build", None)
        self.profile["server_binary"] = str(self.root / "fixture" / "bin" / "llama-server")
        self.profile["server_sha256"] = hashlib.sha256(
            Path(self.profile["server_binary"]).read_bytes()
        ).hexdigest()
        self.profile["model_draft"] = str(self.root / "missing-draft.gguf")
        self.profile["environment"] = {"GGML_CUDA_PDL": "0"}
        self.profile["extra_args"] = [
            "--no-warmup",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "3",
            "--moe-cache-profile",
            str(self.root / "missing-cache.csv"),
            "--moe-cache-slots",
            "4",
        ]
        self.profiles.write_text(json.dumps(self.data), encoding="utf-8")

        profile = ProfileRegistry.from_path(self.profiles).get("fixture-text-7b")

        self.assertTrue(profile.warnings)
        self.assertIn("draft", " ".join(profile.warnings).lower())
        self.assertIn("moe", " ".join(profile.warnings).lower())
        self.assertTrue(profile.degraded_features)
        args = profile.server_args(host="127.0.0.1", port=10000)
        self.assertNotIn("--moe-cache-profile", args)
        self.assertNotIn("--moe-cache-slots", args)
        self.assertNotIn("-md", args)


if __name__ == "__main__":
    unittest.main()
