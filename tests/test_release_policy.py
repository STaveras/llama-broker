from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import io
import unittest
import zipfile

from tools.check_release import ROOT, validate_distributions, validate_publication, validate_tag_target, validate_version


class ReleasePolicyTests(unittest.TestCase):
    def test_stable_tag_must_match_and_cannot_release_development(self):
        validate_version("0.7.1", "v0.7.1")
        for version, tag in [("0.7.1", "v0.1.0"), ("0.7.2.dev1", "v0.7.2.dev1"), ("0.7.1", "main")]:
            with self.subTest(version=version, tag=tag), self.assertRaises(ValueError):
                validate_version(version, tag)

    def test_development_snapshot_policy(self):
        for version in ("0.7.1", "0.7.2.dev0", "1.0.0.dev23"):
            validate_version(version)
        for version in ("0.7", "v0.7.1", "01.2.3", "0.7.2rc1", "0.7.2.dev01"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                validate_version(version)

    def test_tag_at_different_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.check_output(["git", "-C", tmp, *args], stderr=subprocess.DEVNULL)
            git("init", "--quiet")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--allow-empty", "--quiet", "-m", "one")
            git("tag", "v0.7.1")
            validate_tag_target(root, "v0.7.1")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--allow-empty", "--quiet", "-m", "two")
            with self.assertRaises(ValueError):
                validate_tag_target(root, "v0.7.1")

    def test_both_built_metadata_records_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            def packages(wheel_version, sdist_version):
                with zipfile.ZipFile(dist / "llama_broker-0.7.1-py3-none-any.whl", "w") as wheel:
                    wheel.writestr("llama_broker-0.7.1.dist-info/METADATA", f"Name: llama-broker\nVersion: {wheel_version}\n")
                raw = f"Name: llama-broker\nVersion: {sdist_version}\n".encode()
                with tarfile.open(dist / "llama_broker-0.7.1.tar.gz", "w:gz") as sdist:
                    info = tarfile.TarInfo("llama_broker-0.7.1/PKG-INFO")
                    info.size = len(raw)
                    sdist.addfile(info, io.BytesIO(raw))
            packages("0.7.1", "0.7.1")
            validate_distributions(dist, "0.7.1")
            for wheel_version, sdist_version in [("0.1.0", "0.7.1"), ("0.7.1", "0.1.0")]:
                packages(wheel_version, sdist_version)
                with self.assertRaises(ValueError):
                    validate_distributions(dist, "0.7.1")

    def test_retrospective_tags_cannot_publish(self):
        tags = json.loads((ROOT / "docs/retrospective-tags.json").read_text())
        self.assertEqual(len(tags), 25)
        self.assertIn("v0.7.1", tags)
        for tag in [*tags, None]:
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                validate_publication(tag, tags)
        validate_publication("v0.7.2", tags)

    def test_publication_requires_explicit_dispatch_and_environment(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("\n  push:", workflow)
        self.assertIn("default: false", workflow)
        self.assertIn("inputs.publish == true", workflow)
        self.assertIn("vars.PYPI_PUBLISH_ENABLED == 'true'", workflow)
        self.assertIn("environment: pypi", workflow)
        self.assertEqual(workflow.count("id-token: write"), 1)
        self.assertNotIn("contents: write", workflow)
        self.assertIn('--require-head-tag', workflow)
        self.assertIn('--dist dist', workflow)


if __name__ == "__main__":
    unittest.main()
