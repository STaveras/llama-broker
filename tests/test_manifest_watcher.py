from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import llama_broker.daemon as llama_broker
from llama_broker.daemon import BrokerApplication, _manifest_identity
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
        self.application = BrokerApplication(
            registry=ProfileRegistry.from_path(self.profiles_path),
            controller=TrivialController(),
            backend_host="127.0.0.1",
            backend_port=10000,
            backend_timeout=0.1,
            max_body_bytes=1024,
            backend_ready=lambda: True,
            start_operation_worker=False,
            profiles_path=self.profiles_path,
        )

    def tearDown(self) -> None:
        self.application._stop_manifest_watcher()
        self.application._stop_operation_worker()
        self.tempdir.cleanup()

    def write_profiles(self, records: list[dict[str, object]]) -> None:
        envelope = {"schema_version": 1, "profiles": records}
        self.profiles_path.write_text(json.dumps(envelope), encoding="utf-8")

    def seed(self) -> None:
        # Production seeds the baseline from the manifest loaded at startup; the
        # tests reproduce that invariant by seeding from the current file.
        self.application._seed_manifest_baseline()

    def test_watcher_submits_reload_on_edit_and_advances_baseline(self) -> None:
        self.seed()
        baseline = self.application._last_manifest_identity
        self.application._maybe_reload_manifest()
        self.assertEqual(self.application.operation_queue.list_executions(), ())
        self.write_profiles([
            make_profile_record("main", "main", self.model),
            make_profile_record("extra", "extra", self.model),
        ])
        self.application._maybe_reload_manifest()
        self.application._maybe_reload_manifest()
        self.assertEqual(len(self.application.operation_queue.list_executions()), 1)
        self.assertEqual(self.application._last_manifest_identity, baseline)
        self.complete_reload()
        self.assertEqual(self.application._last_manifest_identity, _manifest_identity(self.profiles_path))
        self.assertIsNotNone(self.application.registry.try_resolve("extra"))

    def complete_reload(self) -> None:
        request = self.application.operation_queue.list_executions()[0].request_ids[0]
        self.application.start_operation_worker()
        self.assertEqual(self.application.wait_for_operation(request, 5)["status"], "succeeded")
        self.application._maybe_reload_manifest()

    def test_watcher_noop_when_manifest_stable(self) -> None:
        self.seed()
        with patch.object(self.application, "submit_profile_reload_operation") as submit:
            self.application._maybe_reload_manifest()
            self.application._maybe_reload_manifest()
            self.assertEqual(submit.call_count, 0)
        submit.assert_not_called()

    def test_malformed_manifest_is_not_applied_and_is_retried(self) -> None:
        # A malformed file changes the hash, so the watcher tries to submit it, but
        # the submit itself re-parses and fails — so the bad file is never applied
        # and the baseline is not advanced (the change is retried on the next tick).
        self.seed()
        baseline = self.application._last_manifest_identity
        self.assertIsNotNone(baseline)
        self.profiles_path.write_text("{ not valid json", encoding="utf-8")
        self.application._maybe_reload_manifest()
        self.assertEqual(self.application._last_manifest_identity, baseline)
        # Recovery: acceptance alone does not advance the applied baseline.
        self.write_profiles(
            [
                make_profile_record("main", "main", self.model),
                make_profile_record("extra", "extra", self.model),
            ]
        )
        self.application._maybe_reload_manifest()
        self.assertEqual(self.application._last_manifest_identity, baseline)
        self.complete_reload()
        self.assertEqual(
            self.application._last_manifest_identity, _manifest_identity(self.profiles_path)
        )

    def test_failed_submit_does_not_advance_baseline(self) -> None:
        self.seed()
        self.write_profiles(
            [
                make_profile_record("main", "main", self.model),
                make_profile_record("extra", "extra", self.model),
            ]
        )
        baseline_after_seed = self.application._last_manifest_identity
        with patch.object(
            self.application, "submit_profile_reload_operation", side_effect=RuntimeError("nope")
        ):
            self.application._maybe_reload_manifest()
            # A failed submit must keep the baseline at the last loaded state so the
            # change is retried on the next tick (not dropped).
            self.assertEqual(self.application._last_manifest_identity, baseline_after_seed)
        # Recovery must pass through a real successful registry swap.
        self.application._maybe_reload_manifest()
        self.assertEqual(self.application._last_manifest_identity, baseline_after_seed)
        self.complete_reload()
        self.assertEqual(self.application._last_manifest_identity, _manifest_identity(self.profiles_path))

    def test_seeded_baseline_matches_loaded_manifest(self) -> None:
        # The baseline seeded at startup equals the hash of the manifest that was
        # loaded, so an edit that landed while the daemon was coming up is still
        # detected on the first tick (not swallowed as the baseline).
        self.seed()
        self.assertEqual(
            self.application._last_manifest_identity, _manifest_identity(self.profiles_path)
        )
        with patch.object(self.application, "submit_profile_reload_operation") as submit:
            self.write_profiles(
                [
                    make_profile_record("main", "main", self.model),
                    make_profile_record("extra", "extra", self.model),
                ]
            )
            self.application._maybe_reload_manifest()
            self.assertEqual(submit.call_count, 1)
        submit.assert_called_once()

    def test_disabled_watcher_starts_no_thread(self) -> None:
        self.assertIsNotNone(self.application._profiles_path)
        with patch.object(llama_broker, "MANIFEST_RELOAD_INTERVAL", 0.0):
            self.application.start_manifest_watcher()
        self.assertIsNone(self.application._manifest_watcher_thread)

    def test_enabled_watcher_starts_daemon_thread(self) -> None:
        # Deterministic wiring check: the watcher is started as a daemon thread with
        # the correct target, no real thread (no sleep, no flakiness).
        self.seed()
        self.application._seed_manifest_baseline()
        with patch.object(llama_broker, "MANIFEST_RELOAD_INTERVAL", 0.5):
            with patch.object(llama_broker.threading, "Thread") as fake_thread:
                self.application.start_manifest_watcher()
        fake_thread.assert_called_once()
        kwargs = fake_thread.call_args.kwargs
        self.assertEqual(kwargs["name"], "llama-broker-manifest-watcher")
        self.assertTrue(kwargs["daemon"])
        self.assertEqual(kwargs["target"].__name__, "_manifest_watcher_loop")
        self.assertIs(kwargs["target"].__self__, self.application)
        fake_thread.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
