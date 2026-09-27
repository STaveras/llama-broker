from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_broker.state_store import StateFileStore
from llama_broker.transition_operations import (
    PersistentTransitionQueue,
    QueueCapacityError,
    ReconciliationRequiredError,
)


class PersistentTransitionQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "state.json"
        self.store = StateFileStore(self.path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_submit_is_durable_before_request_returns(self) -> None:
        self.store.replace_backend_state(
            {"generation": 4, "profile": "alpha", "phase": "active"}
        )
        queue = PersistentTransitionQueue(self.store)

        request = queue.submit_load("beta")

        state = self.store.read()
        self.assertEqual(state["profile"], "alpha")
        operations = state["operations"]
        self.assertEqual(operations["schema_version"], 1)
        self.assertEqual(operations["queue"], [request.execution_id])
        self.assertEqual(
            operations["requests"][request.request_id]["execution_id"],
            request.execution_id,
        )

    def test_queued_operations_survive_reconstructing_the_manager(self) -> None:
        first = PersistentTransitionQueue(self.store)
        submitted = first.submit_load("alpha")
        first.submit_load("beta")

        restarted = PersistentTransitionQueue(self.store)

        self.assertEqual(restarted.queue_depth, 2)
        self.assertEqual(
            restarted.get_request(submitted.request_id).execution_id,
            submitted.execution_id,
        )
        self.assertEqual(restarted.next_execution().target_profile_id, "alpha")

    def test_persistence_failure_rejects_and_rolls_back_submission(self) -> None:
        queue = PersistentTransitionQueue(self.store)
        with patch.object(self.store, "write_operations", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                queue.submit_load("alpha")

        self.assertEqual(queue.queue_depth, 0)
        self.assertNotIn("operations", self.store.read())

    def test_completion_history_keeps_latest_32_executions_and_requests(self) -> None:
        queue = PersistentTransitionQueue(self.store)
        request_ids: list[str] = []
        for index in range(34):
            request = queue.submit_load(f"profile-{index}")
            request_ids.append(request.request_id)
            execution = queue.next_execution()
            queue.finish(execution.execution_id)

        with self.assertRaises(KeyError):
            queue.get_request(request_ids[0])
        self.assertEqual(queue.get_request(request_ids[-1]).status, "succeeded")
        operations = self.store.read()["operations"]
        self.assertEqual(len(operations["history"]), 32)
        self.assertEqual(len(operations["requests"]), 32)

    def test_interrupted_execution_requires_reconciliation_before_new_work(self) -> None:
        queue = PersistentTransitionQueue(self.store)
        interrupted = queue.submit_load("alpha")
        running = queue.next_execution()
        pending = queue.submit_load("beta")

        restarted = PersistentTransitionQueue(self.store)

        self.assertTrue(restarted.needs_reconciliation)
        self.assertEqual(restarted.get_request(interrupted.request_id).status, "failed")
        with self.assertRaises(ReconciliationRequiredError):
            restarted.submit_load("gamma")
        self.assertIsNone(restarted.next_execution())

        restarted.acknowledge_reconciliation()
        self.assertFalse(restarted.needs_reconciliation)
        next_execution = restarted.next_execution()
        if next_execution is None:
            self.fail("safe queued work should resume after reconciliation")
        self.assertEqual(next_execution.execution_id, pending.execution_id)
        self.assertEqual(running.execution_id, interrupted.execution_id)

    def test_priority_restart_lane_survives_persistent_reconstruction(self) -> None:
        queue = PersistentTransitionQueue(self.store)
        normal = queue.submit_load("alpha")
        restart = queue.submit_restart("alpha")

        restored = PersistentTransitionQueue(self.store)
        execution = restored.next_execution()

        self.assertIsNotNone(execution)
        if execution is None:
            self.fail("restart should be selected from the priority lane")
        self.assertEqual(execution.operation_type, "restart")
        self.assertEqual(execution.execution_id, restart.execution_id)
        restored.finish(execution.execution_id)
        execution = restored.next_execution()
        self.assertIsNotNone(execution)
        if execution is None:
            self.fail("normal FIFO work should resume after restart")
        self.assertEqual(execution.execution_id, normal.execution_id)
        restored.finish(execution.execution_id)

    def test_stop_persistence_cancels_waiting_restart(self) -> None:
        queue = PersistentTransitionQueue(self.store)
        normal = queue.submit_load("alpha")
        restart = queue.submit_restart("alpha")
        stop = queue.submit_stop()

        restored = PersistentTransitionQueue(self.store)

        self.assertEqual(restored.get_request(restart.request_id).status, "cancelled")
        execution = restored.next_execution()
        self.assertIsNotNone(execution)
        if execution is None:
            self.fail("stop should win priority after restart cancellation")
        self.assertEqual(execution.execution_id, stop.execution_id)
        self.assertEqual(execution.operation_type, "stop")
        restored.finish(execution.execution_id)
        execution = restored.next_execution()
        self.assertIsNotNone(execution)
        if execution is None:
            self.fail("normal work should remain queued after stop")
        self.assertEqual(execution.execution_id, normal.execution_id)
        restored.finish(execution.execution_id)

    def test_profile_snapshot_is_durable_without_environment_values(self) -> None:
        queue = PersistentTransitionQueue(self.store)
        snapshot = {
            "schema_version": 1,
            "profile_id": "alpha",
            "fingerprint": "a" * 64,
            "catalog_revision": "revision-1",
            "artifact_sha256": {"server": "b" * 64, "model": "c" * 64},
            "safety": {"memory_max_mib": 20480},
        }
        request = queue.submit_load("alpha", profile_snapshot=snapshot)

        restarted = PersistentTransitionQueue(self.store)

        self.assertEqual(
            restarted.get_profile_snapshot(request.execution_id),
            snapshot,
        )
        self.assertNotIn("TOKEN", json.dumps(self.store.read()))

    def test_full_queue_rejects_distinct_request_but_accepts_coalesced_request(self) -> None:
        queue = PersistentTransitionQueue(self.store)
        queue.submit_load("alpha")
        queue.submit_load("beta")
        queue.submit_load("gamma")

        with self.assertRaises(QueueCapacityError):
            queue.submit_load("delta")
        coalesced = queue.submit_load("alpha")

        self.assertEqual(queue.queue_depth, 3)
        self.assertEqual(coalesced.execution_id, queue.get_execution(coalesced.execution_id).execution_id)


if __name__ == "__main__":
    unittest.main()
