from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from llama_broker.broker_profiles import ProfileError, ProfileRegistry
from llama_broker.runtime_builds import (
    apply_runtime_build,
    infer_runtime_build,
)
from tests.fixtures import materialize_fixture


class RuntimeBuildCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.profiles = materialize_fixture(self.root)
        self.server = self.root / "bin" / "llama-server"
        self.server.chmod(0o755)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_catalog_build_is_validated_and_exposed_on_profile(self) -> None:
        server_sha256 = hashlib.sha256(self.server.read_bytes()).hexdigest()
        manifest = json.loads(self.profiles.read_text(encoding="utf-8"))
        manifest["runtime_builds"] = [
            {
                "id": "beellama",
                "runtime": "llamacpp",
                "display_name": "BeeLlama",
                "server_binary": str(self.server),
                "server_sha256": server_sha256,
            }
        ]
        profile = next(
            item for item in manifest["profiles"] if item["id"] == "fixture-text-7b"
        )
        profile["runtime_build"] = "beellama"
        self.profiles.write_text(json.dumps(manifest), encoding="utf-8")

        parsed = ProfileRegistry.from_path(self.profiles).get("fixture-text-7b")

        self.assertEqual(parsed.runtime_build, "beellama")

    def test_status_mode_can_read_stale_builds_without_relaxing_default_validation(self) -> None:
        missing = self.root / "removed" / "llama-server"
        manifest = json.loads(self.profiles.read_text(encoding="utf-8"))
        manifest["runtime_builds"] = [
            {
                "id": "stale",
                "runtime": "llamacpp",
                "display_name": "Stale",
                "server_binary": str(missing),
                "server_sha256": "0" * 64,
            }
        ]
        profile = next(
            item for item in manifest["profiles"] if item["id"] == "fixture-text-7b"
        )
        profile["runtime_build"] = "stale"
        profile["server_binary"] = str(missing)
        profile["server_sha256"] = "0" * 64
        self.profiles.write_text(json.dumps(manifest), encoding="utf-8")

        parsed = ProfileRegistry.from_path(
            self.profiles, allow_missing_artifacts=True
        ).get("fixture-text-7b")

        self.assertEqual(parsed.runtime_build, "stale")
        with self.assertRaisesRegex(ProfileError, "existing executable"):
            ProfileRegistry.from_path(self.profiles)

    def test_catalog_requires_an_executable_binary(self) -> None:
        self.server.chmod(0o644)
        manifest = json.loads(self.profiles.read_text(encoding="utf-8"))
        manifest["runtime_builds"] = [
            {
                "id": "beellama",
                "runtime": "llamacpp",
                "display_name": "BeeLlama",
                "server_binary": str(self.server),
                "server_sha256": hashlib.sha256(self.server.read_bytes()).hexdigest(),
            }
        ]
        self.profiles.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ProfileError, "executable"):
            ProfileRegistry.from_path(self.profiles)

    def test_profile_build_must_match_pinned_binary_and_hash(self) -> None:
        server_sha256 = hashlib.sha256(self.server.read_bytes()).hexdigest()
        manifest = json.loads(self.profiles.read_text(encoding="utf-8"))
        manifest["runtime_builds"] = [
            {
                "id": "beellama",
                "runtime": "llamacpp",
                "display_name": "BeeLlama",
                "server_binary": str(self.server),
                "server_sha256": server_sha256,
            }
        ]
        profile = next(
            item for item in manifest["profiles"] if item["id"] == "fixture-text-7b"
        )
        profile["runtime_build"] = "beellama"
        profile["server_sha256"] = "0" * 64
        self.profiles.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ProfileError, "does not match"):
            ProfileRegistry.from_path(self.profiles)

    def test_inference_and_application_use_the_same_catalog_identity(self) -> None:
        server_sha256 = hashlib.sha256(self.server.read_bytes()).hexdigest()
        manifest = json.loads(self.profiles.read_text(encoding="utf-8"))
        manifest["runtime_builds"] = [
            {
                "id": "beellama",
                "runtime": "llamacpp",
                "display_name": "BeeLlama",
                "server_binary": str(self.server),
                "server_sha256": server_sha256,
            }
        ]
        profile = next(
            item for item in manifest["profiles"] if item["id"] == "fixture-text-7b"
        )
        profile.pop("runtime_build", None)

        inferred = infer_runtime_build(manifest, profile)
        updated = apply_runtime_build(manifest, profile, "beellama")

        self.assertIsNotNone(inferred)
        assert inferred is not None
        self.assertEqual(inferred.id, "beellama")
        self.assertNotIn("runtime_build", profile)
        self.assertEqual(updated["runtime_build"], "beellama")
        self.assertEqual(updated["server_binary"], str(self.server.resolve()))
        self.assertEqual(updated["server_sha256"], server_sha256)


if __name__ == "__main__":
    unittest.main()
