from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_broker.state_store import StateFileStore, atomic_write_text


class StateFileStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tempdir.name) / "state.json"
        self.store = StateFileStore(self.state_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_operations_update_preserves_existing_backend_fields(self) -> None:
        backend = {
            "generation": 12,
            "profile": "alpha",
            "phase": "active",
            "runtime_build": "build-a",
        }
        self.store.replace_backend_state(backend)

        self.store.write_operations({"schema_version": 1, "queue": ["exec-a"]})

        self.assertEqual(
            self.store.read(),
            {
                **backend,
                "operations": {"schema_version": 1, "queue": ["exec-a"]},
            },
        )

    def test_backend_update_preserves_existing_operations_section(self) -> None:
        operations = {"schema_version": 1, "queue": ["exec-a"], "history": []}
        self.store.write_operations(operations)

        backend = {"generation": 13, "profile": None, "phase": "inactive"}
        self.store.replace_backend_state(backend)

        self.assertEqual(
            self.store.read(),
            {**backend, "operations": operations},
        )

    def test_missing_file_reads_as_inactive_backend_state(self) -> None:
        self.assertEqual(
            self.store.read(),
            {"generation": 0, "profile": None, "phase": "inactive"},
        )

    def test_backup_and_replace_preserves_corrupt_bytes_before_recovery_state(self) -> None:
        corrupt = b'{"generation": broken\n'
        self.state_path.write_bytes(corrupt)
        operations = {
            "schema_version": 1,
            "queue": [],
            "restart_queue": [],
            "stop_queue": [],
            "running_execution_id": None,
            "restart_coalesced_id": None,
            "reconciliation_required": True,
            "executions": {},
            "requests": {},
            "history": [],
        }

        backup = self.store.backup_and_replace(
            {"generation": 0, "profile": None, "phase": "stale"},
            operations,
        )

        self.assertIsNotNone(backup)
        self.assertEqual(backup.read_bytes(), corrupt)  # type: ignore[union-attr]
        self.assertEqual(self.store.read()["operations"], operations)
        self.assertEqual(self.store.read()["phase"], "stale")

    def test_atomic_write_failure_leaves_last_valid_state_unchanged(self) -> None:
        original = {"generation": 4, "profile": "alpha", "phase": "active"}
        self.store.replace_backend_state(original)

        with patch("llama_broker.state_store.atomic_write_text", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.write_operations({"schema_version": 1, "queue": []})

        self.assertEqual(self.store.read(), original)

    def test_atomic_text_writer_replaces_file_contents(self) -> None:
        atomic_write_text(self.state_path, "{}\n")
        self.assertEqual(json.loads(self.state_path.read_text(encoding="utf-8")), {})


if __name__ == "__main__":
    unittest.main()
