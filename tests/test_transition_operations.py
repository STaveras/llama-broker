from __future__ import annotations

import unittest
import uuid

from llama_broker.transition_operations import (
    OperationCancelledError,
    QueueCapacityError,
    TransitionQueue,
)


class TransitionQueueTests(unittest.TestCase):
    def test_duplicate_target_does_not_coalesce_across_snapshot_revisions(self) -> None:
        queue = TransitionQueue()
        first = queue.submit_load("alpha", profile_snapshot={"fingerprint": "one"})
        second = queue.submit_load("alpha", profile_snapshot={"fingerprint": "two"})

        self.assertNotEqual(first.execution_id, second.execution_id)
        self.assertEqual(queue.queue_depth, 2)
        self.assertEqual(
            queue.get_profile_snapshot(first.execution_id),
            {"fingerprint": "one"},
        )

    def test_recent_listing_includes_pending_and_retained_terminal_executions(self) -> None:
        queue = TransitionQueue()
        completed = queue.submit_load("alpha")
        execution = queue.next_execution()
        if execution is None:
            self.fail("queued operation was not returned")
        queue.finish(execution.execution_id)
        pending = queue.submit_load("beta")

        listed = queue.list_executions()
        by_id = {item.execution_id: item for item in listed}

        self.assertEqual(by_id[completed.execution_id].status, "succeeded")
        self.assertEqual(by_id[pending.execution_id].status, "queued")
        self.assertEqual(by_id[pending.execution_id].request_ids, (pending.request_id,))

    def test_profile_reload_uses_normal_fifo_lane(self) -> None:
        queue = TransitionQueue()
        first = queue.submit_load("alpha")
        reload_request = queue.submit_reload()
        duplicate_reload = queue.submit_reload()
        last = queue.submit_load("beta")

        self.assertEqual(reload_request.execution_id, duplicate_reload.execution_id)
        self.assertEqual(queue.queue_depth, 3)
        for expected_execution, expected_type in (
            (first.execution_id, "switch"),
            (reload_request.execution_id, "reload"),
            (last.execution_id, "switch"),
        ):
            execution = queue.next_execution()
            if execution is None:
                self.fail("FIFO operation missing")
            self.assertEqual(execution.execution_id, expected_execution)
            self.assertEqual(execution.operation_type, expected_type)
            queue.finish(execution.execution_id)

    def test_reload_revisions_do_not_coalesce(self) -> None:
        queue = TransitionQueue()
        first = queue.submit_reload(profile_snapshot={"fingerprint": "one"})
        second = queue.submit_reload(profile_snapshot={"fingerprint": "two"})

        self.assertNotEqual(first.execution_id, second.execution_id)
        self.assertEqual(queue.queue_depth, 2)
        self.assertEqual(
            queue.get_profile_snapshot(first.execution_id),
            {"fingerprint": "one"},
        )

    def test_submission_uses_unique_uuid_request_and_execution_ids(self) -> None:
        queue = TransitionQueue()

        first = queue.submit_load("alpha")
        second = queue.submit_load("beta")

        self.assertNotEqual(first.request_id, second.request_id)
        self.assertNotEqual(first.execution_id, second.execution_id)
        uuid.UUID(first.request_id)
        uuid.UUID(first.execution_id)
        self.assertEqual(queue.get_request(first.request_id).status, "queued")

    def test_same_queued_target_coalesces_without_adding_an_execution(self) -> None:
        queue = TransitionQueue()

        first = queue.submit_load("alpha")
        second = queue.submit_load("alpha")

        self.assertNotEqual(first.request_id, second.request_id)
        self.assertEqual(first.execution_id, second.execution_id)
        self.assertEqual(queue.queue_depth, 1)
        self.assertEqual(
            set(queue.get_execution(first.execution_id).request_ids),
            {first.request_id, second.request_id},
        )

    def test_distinct_targets_are_dequeued_in_fifo_order(self) -> None:
        queue = TransitionQueue()
        requests = [queue.submit_load(target) for target in ("alpha", "beta", "gamma")]

        for request, target in zip(requests, ("alpha", "beta", "gamma"), strict=True):
            execution = queue.next_execution()
            self.assertIsNotNone(execution)
            self.assertEqual(execution.target_profile_id, target)
            self.assertEqual(execution.execution_id, request.execution_id)
            queue.finish(execution.execution_id)

        self.assertIsNone(queue.next_execution())
        self.assertEqual(queue.queue_depth, 0)

    def test_capacity_allows_three_waiting_executions_in_addition_to_running(self) -> None:
        queue = TransitionQueue()
        active = queue.submit_load("active")
        running = queue.next_execution()
        self.assertEqual(running.execution_id, active.execution_id)

        queue.submit_load("beta")
        queue.submit_load("gamma")
        queue.submit_load("delta")
        self.assertEqual(queue.queue_depth, 3)
        with self.assertRaises(QueueCapacityError):
            queue.submit_load("epsilon")
        self.assertEqual(queue.queue_depth, 3)

        queue.finish(running.execution_id)

    def test_coalesced_request_is_admitted_even_when_queue_is_full(self) -> None:
        queue = TransitionQueue()
        first = queue.submit_load("alpha")
        queue.submit_load("beta")
        queue.submit_load("gamma")

        coalesced = queue.submit_load("alpha")

        self.assertEqual(coalesced.execution_id, first.execution_id)
        self.assertNotEqual(coalesced.request_id, first.request_id)
        self.assertEqual(queue.queue_depth, 3)

    def test_cancel_one_request_keeps_execution_for_other_request(self) -> None:
        queue = TransitionQueue()
        first = queue.submit_load("alpha")
        second = queue.submit_load("alpha")

        self.assertTrue(queue.cancel_request(first.request_id))

        self.assertEqual(queue.get_request(first.request_id).status, "cancelled")
        self.assertEqual(queue.get_request(second.request_id).status, "queued")
        self.assertEqual(queue.queue_depth, 1)
        execution = queue.next_execution()
        self.assertEqual(execution.execution_id, second.execution_id)
        queue.finish(execution.execution_id)

    def test_cancelling_last_queued_request_removes_execution(self) -> None:
        queue = TransitionQueue()
        first = queue.submit_load("alpha")
        second = queue.submit_load("alpha")

        self.assertTrue(queue.cancel_request(first.request_id))
        self.assertTrue(queue.cancel_request(second.request_id))

        self.assertEqual(queue.queue_depth, 0)
        self.assertEqual(queue.get_execution(first.execution_id).status, "cancelled")
        self.assertIsNone(queue.next_execution())

    def test_restart_is_prioritized_and_duplicate_restarts_coalesce(self) -> None:
        queue = TransitionQueue()
        first_load = queue.submit_load("alpha")
        restart = queue.submit_restart("alpha")
        duplicate_restart = queue.submit_restart("alpha")
        last_load = queue.submit_load("beta")

        self.assertEqual(restart.execution_id, duplicate_restart.execution_id)
        self.assertNotEqual(restart.request_id, duplicate_restart.request_id)
        self.assertEqual(queue.queue_depth, 2)
        execution = queue.next_execution()
        if execution is None:
            self.fail("priority restart was not scheduled")
        self.assertEqual(execution.operation_type, "restart")
        self.assertEqual(execution.target_profile_id, "alpha")
        queue.mark_committed(execution.execution_id)
        queue.finish(execution.execution_id)

        execution = queue.next_execution()
        self.assertEqual(execution.execution_id, first_load.execution_id)
        queue.finish(execution.execution_id)
        execution = queue.next_execution()
        self.assertEqual(execution.execution_id, last_load.execution_id)
        queue.finish(execution.execution_id)

    def test_stop_cancels_waiting_restart_and_precedes_normal_loads(self) -> None:
        queue = TransitionQueue()
        first_load = queue.submit_load("alpha")
        restart = queue.submit_restart("alpha")
        stop = queue.submit_stop()
        last_load = queue.submit_load("beta")

        self.assertEqual(queue.get_request(restart.request_id).status, "cancelled")
        execution = queue.next_execution()
        if execution is None:
            self.fail("unload/stop was not scheduled")
        self.assertEqual(execution.execution_id, stop.execution_id)
        self.assertEqual(execution.operation_type, "stop")
        queue.mark_committed(execution.execution_id)
        queue.finish(execution.execution_id)

        execution = queue.next_execution()
        self.assertEqual(execution.execution_id, first_load.execution_id)
        queue.finish(execution.execution_id)
        execution = queue.next_execution()
        self.assertEqual(execution.execution_id, last_load.execution_id)
        queue.finish(execution.execution_id)

    def test_stop_waits_for_committed_restart_without_interrupting_it(self) -> None:
        queue = TransitionQueue()
        restart = queue.submit_restart("alpha")
        execution = queue.next_execution()
        if execution is None:
            self.fail("restart was not scheduled")
        queue.mark_committed(execution.execution_id)
        stop = queue.submit_stop()

        self.assertEqual(queue.get_request(restart.request_id).status, "committed")
        queue.finish(execution.execution_id)
        stop_execution = queue.next_execution()
        self.assertEqual(stop_execution.execution_id, stop.execution_id)
        self.assertEqual(stop_execution.operation_type, "stop")

    def test_all_precommit_callers_cancelling_aborts_preflight(self) -> None:
        queue = TransitionQueue()
        first = queue.submit_load("alpha")
        second = queue.submit_load("alpha")
        execution = queue.next_execution()

        self.assertTrue(queue.cancel_request(first.request_id))
        self.assertTrue(queue.cancel_request(second.request_id))
        self.assertTrue(queue.get_execution(execution.execution_id).cancel_requested)
        with self.assertRaises(OperationCancelledError):
            queue.mark_committed(execution.execution_id)

    def test_cancel_after_commit_stops_waiting_but_does_not_cancel_execution(self) -> None:
        queue = TransitionQueue()
        request = queue.submit_load("alpha")
        execution = queue.next_execution()
        queue.mark_committed(execution.execution_id)

        self.assertFalse(queue.cancel_request(request.request_id))
        self.assertFalse(queue.get_request(request.request_id).waiting)
        self.assertEqual(queue.get_execution(execution.execution_id).status, "committed")


if __name__ == "__main__":
    unittest.main()
