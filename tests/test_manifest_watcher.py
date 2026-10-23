from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import llama_broker.daemon as llama_broker
from llama_broker.daemon import BrokerApplication
from llama_broker.broker_profiles import ProfileRegistry


class TrivialController:
    def state(self) -> dict[str, object]:
        return {"phase": "inactive"}


def make_profile_record(profile_id: str, alias: str, model_path: Path) -> dict[str, object]:
    return {
        "id": profile_id,
        "alias": alias,
        "aliases": [alias],
        "display_name": profile_id,
        "role": "persistent",
        "capabilities": ["text"],
        "server_binary": str(model_path.parent / "llama-server"),
        "server_sha256": "1" * 64,
        "model_path": str(model_path),
        "model_sha256": "2" * 64,
        "context_size": 65536,
        "parallel": 1,
        "batch_size": 256,
        "ubatch_size": 64,
        "cache_type_k": "turbo4",
        "cache_type_v": "turbo4",
        "flash_attention": "on",
        "gpu_layers": "all",
        "startup_timeout": 180,
        "safety": {
            "service": {
                "type": "exec",
                "memory_max_mib": 20_480,
                "memory_swap_max_mib": 1_024,
                "tasks_max": 512,
                "kill_signal": "SIGINT",
                "timeout_stop_sec": 20,
            },
            "admission": {
                "min_ram_available_mib": 8_192,
                "max_swap_used_mib": 1_024,
                "max_swap_growth_mib": 128,
                "min_total_vram_mib": 12_000,
                "min_free_vram_before_start_mib": 8_192,
                "driver_vram_reserve_mib": 512,
                "vision_vram_reserve_mib": 1_536,
                "safety_vram_reserve_mib": 1_024,
            },
        },
        "environment": {"GGML_CUDA_PDL": "0"},
        "extra_args": ["--no-warmup"],
    }


class ManifestWatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.server = root / "llama-server"
        self.server.write_bytes(b"server")
        self.server.chmod(0o755)
        self.model = root / "main.gguf"
        self.model.write_bytes(b"main")
        self.profiles_path = root / "profiles.json"
        self.write_profiles([make_profile_record("main", "main", self.model)])
        application = BrokerApplication(
            registry=object(),  # type: ignore[arg-type]
            controller=TrivialController(),
            backend_host="127.0.0.1",
            backend_port=10000,
            backend_timeout=0.1,
            max_body_bytes=1024,
            backend_ready=lambda: True,
            start_operation_worker=False,
            profiles_path=self.profiles_path,
        )
        self.application = application

    def tearDown(self) -> None:
        self.application._stop_manifest_watcher()
        self.tempdir.cleanup()

    def write_profiles(self, records: list[dict[str, object]]) -> None:
        envelope = {"schema_version": 1, "profiles": records}
        self.profiles_path.write_text(json.dumps(envelope), encoding="utf-8")

    def assert_identity_loaded(self) -> None:
        """The on-disk manifest must be a real, loadable revision identity."""
        registry = ProfileRegistry.from_path(self.profiles_path)
        identity = registry.snapshot_revision_identity()
        self.assertIsInstance(identity, dict)
        self.assertIn("fingerprint", identity)

    def test_watcher_submits_reload_on_identity_change(self) -> None:
        self.assert_identity_loaded()
        with patch.object(
            self.application, "submit_profile_reload_operation"
        ) as submit:
            self.application._maybe_reload_manifest()
            self.assertEqual(submit.call_count, 0)
            self.write_profiles(
                [
                    make_profile_record("main", "main", self.model),
                    make_profile_record("extra", "extra", self.model),
                ]
            )
            self.application._maybe_reload_manifest()
            self.assertEqual(submit.call_count, 1)
        submit.assert_called_once()

    def test_watcher_noop_when_identity_stable(self) -> None:
        self.assert_identity_loaded()
        with patch.object(self.application, "submit_profile_reload_operation") as submit:
            self.application._maybe_reload_manifest()
            self.application._maybe_reload_manifest()
            self.assertEqual(submit.call_count, 0)
        submit.assert_not_called()

    def test_watcher_skips_malformed_manifest(self) -> None:
        self.profiles_path.write_text("{ not valid json", encoding="utf-8")
        with patch.object(self.application, "submit_profile_reload_operation") as submit:
            self.application._maybe_reload_manifest()
        self.assertEqual(submit.call_count, 0)
        submit.assert_not_called()

    def test_disabled_watcher_starts_no_thread(self) -> None:
        self.assertIsNotNone(self.application._profiles_path)
        with patch.object(llama_broker, "MANIFEST_RELOAD_INTERVAL", 0.0):
            self.application.start_manifest_watcher()
        self.assertIsNone(self.application._manifest_watcher_thread)


if __name__ == "__main__":
    unittest.main()
