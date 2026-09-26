from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.model_store_migrate import (
    MigrationError,
    apply_migration,
    build_migration_plan,
    cleanup_migration,
    stage_migration,
    verify_migration,
)


class ModelStoreMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.source_root = self.root / "source"
        self.models_dir = self.root / "canonical-models"
        self.runtimes_dir = self.root / "canonical-runtimes"
        self.cache_dir = self.root / "cache"
        self.source_root.mkdir()
        self.server = self.source_root / "bin" / "llama-server"
        self.server.parent.mkdir()
        self.server.write_text("#!/bin/sh\necho version: fixture\n", encoding="utf-8")
        self.server.chmod(0o755)
        self.model_real = self.source_root / "Qwen3.5-9B-Q6_K.gguf"
        self.model_real.write_bytes(b"fixture-model")
        self.model_link = self.source_root / "legacy-model.gguf"
        self.model_link.symlink_to(self.model_real)
        self.deferred = self.source_root / "checkpoint.safetensors"
        self.deferred.write_bytes(b"deferred")
        self.manifest = self.root / "profiles.json"
        model_sha = hashlib.sha256(self.model_real.read_bytes()).hexdigest()
        server_sha = hashlib.sha256(self.server.read_bytes()).hexdigest()
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "runtime_builds": [
                        {
                            "id": "fixture-runtime",
                            "runtime": "llamacpp",
                            "display_name": "Fixture runtime",
                            "server_binary": str(self.server),
                            "server_sha256": server_sha,
                        }
                    ],
                    "profiles": [
                        {
                            "id": "fixture-profile",
                            "alias": "fixture-alias",
                            "aliases": ["fixture-alias", "fixture-old"],
                            "display_name": "Fixture profile",
                            "role": "persistent",
                            "capabilities": ["text"],
                            "runtime": "llamacpp",
                            "runtime_build": "fixture-runtime",
                            "server_binary": str(self.server),
                            "server_sha256": server_sha,
                            "model_path": str(self.model_link),
                            "model_sha256": model_sha,
                            "context_size": 32768,
                            "parallel": 1,
                            "batch_size": 256,
                            "ubatch_size": 64,
                            "cache_type_k": "q8_0",
                            "cache_type_v": "q8_0",
                            "flash_attention": "on",
                            "gpu_layers": "all",
                            "startup_timeout": 30,
                            "safety": {
                                "service": {
                                    "type": "exec",
                                    "memory_max_mib": 20480,
                                    "memory_swap_max_mib": 1024,
                                    "tasks_max": 512,
                                    "kill_signal": "SIGINT",
                                    "timeout_stop_sec": 20,
                                },
                                "admission": {
                                    "min_ram_available_mib": 8192,
                                    "max_swap_used_mib": 1024,
                                    "max_swap_growth_mib": 128,
                                    "min_total_vram_mib": 12000,
                                    "min_free_vram_before_start_mib": 8192,
                                    "driver_vram_reserve_mib": 512,
                                    "vision_vram_reserve_mib": 1536,
                                    "safety_vram_reserve_mib": 1024,
                                },
                            },
                            "environment": {"GGML_CUDA_PDL": "0"},
                            "extra_args": ["--no-warmup", "--reasoning", "off"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.plan_path = self.root / "migration-plan.json"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _plan(self) -> dict[str, object]:
        return build_migration_plan(
            self.manifest,
            catalog_path=self.runtimes_dir / "catalog.json",
            models_dir=self.models_dir,
            runtimes_dir=self.runtimes_dir,
            cache_dir=self.cache_dir,
            output_path=self.plan_path,
            inventory_roots=[self.source_root],
            validate_loaders=False,
        )

    def test_plan_records_exact_mapping_and_deferred_inventory_without_copying(self) -> None:
        plan = self._plan()

        self.assertEqual(plan["schema_version"], 1)
        self.assertTrue(plan["plan_sha256"])
        self.assertFalse(self.models_dir.exists())
        self.assertFalse(self.runtimes_dir.exists())
        mappings = plan["model_mappings"]
        self.assertEqual(len(mappings), 1)
        self.assertEqual(mappings[0]["source"], str(self.model_link))
        self.assertEqual(mappings[0]["resolved_source"], str(self.model_real))
        self.assertEqual(mappings[0]["target"], str(self.models_dir / "qwen-3.5" / "9b" / "q6-k.gguf"))
        self.assertTrue(any(item["path"] == str(self.deferred) for item in plan["deferred_inventory"]))

    def test_stage_is_resumable_and_recopies_a_mismatched_target(self) -> None:
        self._plan()
        stage_migration(self.plan_path)
        target = self.models_dir / "qwen-3.5" / "9b" / "q6-k.gguf"
        self.assertTrue(target.is_file())
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_bytes(), self.model_real.read_bytes())

        target.write_bytes(b"corrupt")
        stage_migration(self.plan_path)
        self.assertEqual(target.read_bytes(), self.model_real.read_bytes())

    def test_verify_rejects_source_drift_before_apply(self) -> None:
        self._plan()
        stage_migration(self.plan_path)
        self.model_real.write_bytes(b"changed")

        with self.assertRaisesRegex(MigrationError, "drift|hash"):
            verify_migration(self.plan_path, validate_loaders=False)
        self.assertFalse((self.manifest.with_suffix(".json.bak")).exists())

    def test_apply_requires_explicit_permission_and_publishes_paired_v2_state(self) -> None:
        self._plan()
        stage_migration(self.plan_path)
        verify_migration(self.plan_path, validate_loaders=False)

        with self.assertRaisesRegex(MigrationError, "apply"):
            apply_migration(self.plan_path, apply=False, validate_loaders=False)

        result = apply_migration(self.plan_path, apply=True, validate_loaders=False)
        self.assertTrue(result["applied"])
        saved_manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        saved_catalog = json.loads((self.runtimes_dir / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_manifest["schema_version"], 2)
        self.assertEqual(saved_catalog["schema_version"], 2)
        self.assertEqual(saved_manifest["deployment_revision"], saved_catalog["deployment_revision"])
        profile = saved_manifest["profiles"][0]
        self.assertEqual(profile["id"], "fixture-profile")
        self.assertEqual(profile["alias"], "fixture-alias")
        self.assertEqual(profile["runtime_build"], "fixture-runtime-" + result["runtime_id_suffix"])
        self.assertNotIn("server_binary", profile)
        self.assertNotIn("runtime", profile)
        self.assertTrue(list(self.root.glob("**/profiles.json.bak-*")))

    def test_cleanup_requires_separate_confirmation_and_refuses_unplanned_paths(self) -> None:
        plan = self._plan()
        stage_migration(self.plan_path)
        verify_migration(self.plan_path, validate_loaders=False)
        apply_migration(self.plan_path, apply=True, validate_loaders=False)

        with self.assertRaisesRegex(MigrationError, "confirm"):
            cleanup_migration(self.plan_path, cleanup=True, confirm=False)

        unplanned = self.source_root / "do-not-delete.txt"
        unplanned.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(MigrationError, "unplanned"):
            cleanup_migration(self.plan_path, cleanup=True, confirm=True, extra_paths=[unplanned])
        self.assertTrue(unplanned.exists())

        result = cleanup_migration(self.plan_path, cleanup=True, confirm=True)
        self.assertTrue(result["cleaned"])
        self.assertTrue(unplanned.exists())
        self.assertFalse(self.model_link.exists())

    def test_plan_hash_mismatch_rejects_stage(self) -> None:
        self._plan()
        payload = json.loads(self.plan_path.read_text(encoding="utf-8"))
        payload["model_mappings"][0]["target"] = str(self.models_dir / "tampered.gguf")
        self.plan_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(MigrationError, "plan hash"):
            stage_migration(self.plan_path)


if __name__ == "__main__":
    unittest.main()
