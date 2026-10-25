from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_broker.broker_profiles import ProfileError, ProfileRegistry


class NativeBasaltProfileRegistry:
    """Real, pinned Basalt catalog fixture with a deliberately small model header."""

    def __init__(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.model = self.root / "models" / "fixture.basalt"
        self.model.parent.mkdir()
        self.model.write_bytes(b"BASALT\x01\x00")
        self.config = self.root / "config" / "fixture.json"
        self.config.parent.mkdir()
        self.config.write_text(json.dumps({"server": {"model": str(self.model)}}), encoding="utf-8")
        self.server = self.root / "bin" / "basalt"
        self.server.parent.mkdir()
        self.server.write_bytes(b"basalt-server")
        self.server.chmod(0o755)
        self.catalog = self.root / "catalog.json"
        self.manifest = self.root / "profiles.json"
        self._write_manifests()

    def _write_manifests(self) -> None:
        server_sha = hashlib.sha256(self.server.read_bytes()).hexdigest()
        model_sha = hashlib.sha256(self.model.read_bytes()).hexdigest()
        config_sha = hashlib.sha256(self.config.read_bytes()).hexdigest()
        build_id = "basalt-fixture-pinned"
        profile = {
            "id": "basalt-fixture",
            "alias": "basalt-fixture",
            "aliases": ["basalt-fixture"],
            "display_name": "Basalt fixture",
            "runtime": "basalt",
            "runtime_build": build_id,
            "role": "persistent",
            "capabilities": ["text", "tools"],
            "server_binary": str(self.server),
            "server_sha256": server_sha,
            "model_path": str(self.model),
            "model_sha256": model_sha,
            "basalt_config_path": str(self.config),
            "basalt_config_sha256": config_sha,
            "context_size": 4096,
            "parallel": 1,
            "startup_timeout": 30,
            "environment": {},
            "extra_args": [],
            "safety": {
                "service": {"type": "exec", "memory_max_mib": 1024, "memory_swap_max_mib": 0,
                             "tasks_max": 64, "kill_signal": "SIGINT", "timeout_stop_sec": 5},
                "admission": {"min_ram_available_mib": 8192, "max_swap_used_mib": 0,
                              "max_swap_growth_mib": 0, "min_total_vram_mib": 12000,
                              "min_free_vram_before_start_mib": 8192, "driver_vram_reserve_mib": 512,
                              "vision_vram_reserve_mib": 1, "safety_vram_reserve_mib": 1},
            },
        }
        catalog = {
            "schema_version": 2,
            "deployment_revision": "basalt-fixture-revision",
            "temporary_aliases": {},
            "runtimes": [{
                "id": build_id, "runtime": "basalt", "display_name": "Basalt fixture build",
                "bundle_root": str(self.root), "server_binary": str(self.server),
                "server_sha256": server_sha, "source_repository": str(self.root / "source"),
                "source_commit": "fixture-pinned-sha", "dirty": False,
                "selected_profile_ids": ["basalt-fixture"],
            }],
        }
        self.catalog.write_text(json.dumps(catalog), encoding="utf-8")
        self.manifest.write_text(json.dumps({
            "schema_version": 2, "deployment_revision": "basalt-fixture-revision", "profiles": [profile]
        }), encoding="utf-8")

    def close(self) -> None:
        self.tempdir.cleanup()


class BasaltProfileRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = NativeBasaltProfileRegistry()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_native_basalt_validation_does_not_read_entire_model(self) -> None:
        model = self.fixture.model
        original_read_bytes = Path.read_bytes
        read_bytes_paths: list[Path] = []

        def read_bytes(path: Path) -> bytes:
            read_bytes_paths.append(path)
            if path == model:
                raise AssertionError("Basalt model must not be loaded with Path.read_bytes")
            return original_read_bytes(path)

        original_open = Path.open
        model_reads: list[int | None] = []

        class Reader:
            def __init__(self, handle):
                self.handle = handle
            def __enter__(self):
                self.handle.__enter__()
                return self
            def __exit__(self, *args):
                return self.handle.__exit__(*args)
            def read(self, size=-1):
                model_reads.append(size)
                return self.handle.read(size)
            def __getattr__(self, name):
                return getattr(self.handle, name)

        def open_file(path: Path, *args, **kwargs):
            handle = original_open(path, *args, **kwargs)
            return Reader(handle) if path == model else handle

        with patch.dict(os.environ, {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.fixture.catalog)}), \
                patch.object(Path, "read_bytes", read_bytes), patch.object(Path, "open", open_file):
            registry = ProfileRegistry.from_path(self.fixture.manifest)

        self.assertEqual(registry.get("basalt-fixture").basalt_config_path, self.fixture.config)
        self.assertIn(self.fixture.config, read_bytes_paths)
        self.assertNotIn(model, read_bytes_paths)
        self.assertIn(6, model_reads)
        self.assertNotIn(64 * 1024**3, model_reads)

    def test_basalt_tools_allow_missing_parser_but_vllm_tools_require_one(self) -> None:
        with patch.dict(os.environ, {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.fixture.catalog)}):
            basalt = ProfileRegistry.from_path(self.fixture.manifest).get("basalt-fixture")
        self.assertIsNone(basalt.tool_call_parser)

        data = json.loads(self.fixture.manifest.read_text(encoding="utf-8"))
        profile = data["profiles"][0]
        profile["runtime"] = "vllm"
        profile["model_path"] = str(self.fixture.root / "model-directory")
        Path(profile["model_path"]).mkdir()
        profile.pop("basalt_config_path")
        profile.pop("basalt_config_sha256")
        catalog = json.loads(self.fixture.catalog.read_text(encoding="utf-8"))
        catalog["runtimes"][0]["runtime"] = "vllm"
        self.fixture.catalog.write_text(json.dumps(catalog), encoding="utf-8")
        self.fixture.manifest.write_text(json.dumps(data), encoding="utf-8")
        with patch.dict(os.environ, {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.fixture.catalog)}), \
                self.assertRaisesRegex(ProfileError, "requires tool_call_parser"):
            ProfileRegistry.from_path(self.fixture.manifest)

    def test_catalog_listing_tolerates_absent_artifacts_but_activation_rejects_them(self):
        self.fixture.config.unlink()
        self.fixture.model.unlink()
        with patch.dict(os.environ, {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.fixture.catalog)}):
            registry = ProfileRegistry.from_path(self.fixture.manifest, allow_missing_artifacts=True)
            self.assertEqual(registry.get('basalt-fixture').alias, 'basalt-fixture')
            with self.assertRaises(ProfileError):
                ProfileRegistry.from_path(self.fixture.manifest)

    def test_native_alias_comes_from_hashed_operator_config(self):
        self.fixture.config.write_text(json.dumps({'server': {'model': str(self.fixture.model), 'alias': 'native-fixture'}}))
        self.fixture._write_manifests()
        with patch.dict(os.environ, {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.fixture.catalog)}):
            profile = ProfileRegistry.from_path(self.fixture.manifest).get('basalt-fixture')
            self.assertEqual(profile.backend_model_name, 'native-fixture')

    def test_config_model_binding_and_hash_fail_closed(self):
        self.fixture.config.write_text(json.dumps({'server': {'model': '/example/different.basalt'}}))
        with patch.dict(os.environ, {"LLAMA_BROKER_RUNTIME_CATALOG": str(self.fixture.catalog)}):
            with self.assertRaisesRegex(ProfileError, 'hash mismatch'):
                ProfileRegistry.from_path(self.fixture.manifest)
            self.fixture._write_manifests()
            with self.assertRaisesRegex(ProfileError, 'pinned to profile model'):
                ProfileRegistry.from_path(self.fixture.manifest)
