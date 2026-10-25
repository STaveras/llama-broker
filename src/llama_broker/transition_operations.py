"""Pure orchestration primitives for queued profile-transition operations.

Persistence, backend process control, and HTTP/CLI adapters are deliberately
separate. This module models normal load executions and per-caller requests;
one execution may serve multiple coalesced requests.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .state_store import StateFileStore


COMPLETED_EXECUTION_HISTORY_LIMIT = 32
MAX_QUEUED_TRANSITIONS = 3
OPERATION_STATE_SCHEMA_VERSION = 1


def _validated_profile_snapshot(
    profile_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if profile_snapshot is None:
        return None
    if not isinstance(profile_snapshot, Mapping):
        raise ValueError("profile_snapshot must be an object")
    try:
        encoded = json.dumps(dict(profile_snapshot), sort_keys=True, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"profile_snapshot must contain JSON-safe facts: {error}") from error


class QueueCapacityError(RuntimeError):
    """Raised when a distinct execution cannot be admitted to the bounded queue."""


class OperationSupersededError(RuntimeError):
    """Raised when a new restart is rejected behind a pending unload/stop."""


class OperationCancelledError(RuntimeError):
    """Raised when all callers cancel a pre-commit execution."""


class OperationStateError(ValueError):
    """Raised when persisted transition-operation state fails validation."""


class ReconciliationRequiredError(RuntimeError):
    """Raised until interrupted backend state has been reconciled."""


@dataclass(frozen=True)
class RequestView:
    """Immutable caller-specific view of an operation request."""

    request_id: str
    execution_id: str
    status: str
    waiting: bool = True
    error: str | None = None
    operation_type: str = "switch"
    error_type: str | None = None


@dataclass(frozen=True)
class ExecutionView:
    """Immutable shared view of a transition execution."""

    execution_id: str
    target_profile_id: str | None
    request_ids: tuple[str, ...]
    status: str
    cancel_requested: bool = False
    error: str | None = None
    operation_type: str = "switch"
    error_type: str | None = None


@dataclass
class _RequestRecord:
    request_id: str
    execution_id: str
    operation_type: str = "switch"
    status: str = "queued"
    waiting: bool = True
    error: str | None = None
    error_type: str | None = None


@dataclass
class _ExecutionRecord:
    execution_id: str
    operation_type: str
    target_profile_id: str | None
    profile_snapshot: dict[str, Any] | None
    request_ids: list[str]
    status: str = "queued"
    cancel_requested: bool = False
    error: str | None = None
    error_type: str | None = None


class TransitionQueue:
    """Thread-safe operation queue with a fixed three-entry normal waiting lane.

    The active execution and priority restart/stop lanes do not consume normal
    queue capacity. Equivalent queued loads and reloads coalesce.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._queue: deque[str] = deque()
        self._restart_queue: deque[str] = deque()
        self._stop_queue: deque[str] = deque()
        self._running_execution_id: str | None = None
        self._restart_coalesced_id: str | None = None
        self._requests: dict[str, _RequestRecord] = {}
        self._executions: dict[str, _ExecutionRecord] = {}
        self._history: list[str] = []
        self.needs_reconciliation = False

    @property
    def queue_depth(self) -> int:
        """Number of distinct executions waiting to start."""
        with self._lock:
            return len(self._queue)

    @property
    def pending_transition_count(self) -> int:
        """Number of waiting executions across priority and normal lanes."""
        with self._lock:
            return len(self._stop_queue) + len(self._restart_queue) + len(self._queue)

    def submit_load(
        self,
        target_profile_id: str,
        *,
        profile_snapshot: Mapping[str, Any] | None = None,
    ) -> RequestView:
        """Admit a load request, coalescing only equivalent target snapshots."""
        if not isinstance(target_profile_id, str) or not target_profile_id:
            raise ValueError("target_profile_id must be a non-empty string")

        snapshot = _validated_profile_snapshot(profile_snapshot)
        with self._lock:
            if self.needs_reconciliation:
                raise ReconciliationRequiredError(
                    "backend reconciliation is required before accepting transitions"
                )
            execution = self._find_queued_target(target_profile_id, snapshot)
            if execution is None:
                if len(self._queue) >= MAX_QUEUED_TRANSITIONS:
                    raise QueueCapacityError(
                        f"transition queue is full ({MAX_QUEUED_TRANSITIONS} queued executions)"
                    )
                execution_id = uuid.uuid4().hex
                execution = _ExecutionRecord(
                    execution_id=execution_id,
                    operation_type="switch",
                    target_profile_id=target_profile_id,
                    profile_snapshot=snapshot,
                    request_ids=[],
                )
                self._executions[execution_id] = execution
                self._queue.append(execution_id)

            request_id = uuid.uuid4().hex
            request = _RequestRecord(
                request_id=request_id,
                execution_id=execution.execution_id,
                operation_type=execution.operation_type,
            )
            self._requests[request_id] = request
            execution.request_ids.append(request_id)
            return self._request_view(request)

    def submit_reload(
        self,
        *,
        profile_snapshot: Mapping[str, Any] | None = None,
    ) -> RequestView:
        """Queue a profile/catalog reload bound to its captured revision."""
        snapshot = _validated_profile_snapshot(profile_snapshot)
        with self._lock:
            if self.needs_reconciliation:
                raise ReconciliationRequiredError(
                    "backend reconciliation is required before accepting transitions"
                )
            existing = next(
                (
                    self._executions[execution_id]
                    for execution_id in self._queue
                    if (
                        self._executions[execution_id].operation_type == "reload"
                        and self._executions[execution_id].profile_snapshot == snapshot
                    )
                ),
                None,
            )
            if existing is not None:
                return self._new_request_for_execution(existing)
            if len(self._queue) >= MAX_QUEUED_TRANSITIONS:
                raise QueueCapacityError(
                    f"transition queue is full ({MAX_QUEUED_TRANSITIONS} queued executions)"
                )
            execution = _ExecutionRecord(
                execution_id=uuid.uuid4().hex,
                operation_type="reload",
                target_profile_id=None,
                profile_snapshot=snapshot,
                request_ids=[],
            )
            self._executions[execution.execution_id] = execution
            self._queue.append(execution.execution_id)
            return self._new_request_for_execution(execution)

    def submit_restart(
        self,
        target_profile_id: str,
        *,
        profile_snapshot: Mapping[str, Any] | None = None,
    ) -> RequestView:
        """Queue a forced restart ahead of loads, bound to its snapshot."""
        if not isinstance(target_profile_id, str) or not target_profile_id:
            raise ValueError("target_profile_id must be a non-empty string")
        snapshot = _validated_profile_snapshot(profile_snapshot)
        with self._lock:
            if self._stop_queue or self._running_is_stop():
                raise OperationSupersededError("an unload/stop is already pending")
            if self._restart_coalesced_id is not None:
                existing = self._executions.get(self._restart_coalesced_id)
                if (
                    existing is not None
                    and existing.operation_type == "restart"
                    and existing.target_profile_id == target_profile_id
                    and existing.profile_snapshot == snapshot
                    and existing.status != "cancelled"
                ):
                    return self._new_request_for_execution(existing)
            execution = _ExecutionRecord(
                execution_id=uuid.uuid4().hex,
                operation_type="restart",
                target_profile_id=target_profile_id,
                profile_snapshot=snapshot,
                request_ids=[],
            )
            self._executions[execution.execution_id] = execution
            self._restart_queue.append(execution.execution_id)
            self._restart_coalesced_id = execution.execution_id
            return self._new_request_for_execution(execution)

    def submit_stop(self) -> RequestView:
        """Queue the highest-priority unload/stop operation."""
        with self._lock:
            existing_stop = self._stop_queue[0] if self._stop_queue else None
            if existing_stop is None and self._running_is_stop():
                existing_stop = self._running_execution_id
            if existing_stop is not None:
                return self._new_request_for_execution(self._executions[existing_stop])

            for execution_id in tuple(self._restart_queue):
                self._cancel_queued_execution(execution_id)
            if self._running_is_restart() and self._executions[
                self._running_execution_id or ""
            ].status == "preflight":
                running = self._executions[self._running_execution_id or ""]
                running.cancel_requested = True
                if self._restart_coalesced_id == running.execution_id:
                    self._restart_coalesced_id = None
                for request_id in running.request_ids:
                    request = self._requests[request_id]
                    request.status = "cancelled"
                    request.waiting = False

            execution = _ExecutionRecord(
                execution_id=uuid.uuid4().hex,
                operation_type="stop",
                target_profile_id=None,
                profile_snapshot=None,
                request_ids=[],
            )
            self._executions[execution.execution_id] = execution
            self._stop_queue.append(execution.execution_id)
            return self._new_request_for_execution(execution)

    def _new_request_for_execution(self, execution: _ExecutionRecord) -> RequestView:
        request_id = uuid.uuid4().hex
        terminal = execution.status in {"succeeded", "failed", "cancelled"}
        request = _RequestRecord(
            request_id=request_id,
            execution_id=execution.execution_id,
            operation_type=execution.operation_type,
            status=execution.status,
            waiting=not terminal,
            error=execution.error,
        )
        self._requests[request_id] = request
        execution.request_ids.append(request_id)
        return self._request_view(request)

    def _running_is_stop(self) -> bool:
        if self._running_execution_id is None:
            return False
        return self._executions[self._running_execution_id].operation_type == "stop"

    def _running_is_restart(self) -> bool:
        if self._running_execution_id is None:
            return False
        return self._executions[self._running_execution_id].operation_type == "restart"

    def _cancel_queued_execution(self, execution_id: str) -> None:
        execution = self._executions[execution_id]
        lane = self._restart_queue if execution.operation_type == "restart" else self._queue
        lane.remove(execution_id)
        execution.status = "cancelled"
        execution.cancel_requested = True
        for request_id in execution.request_ids:
            request = self._requests[request_id]
            request.status = "cancelled"
            request.waiting = False
        if self._restart_coalesced_id == execution_id:
            self._restart_coalesced_id = None
        self._archive_execution(execution_id)

    def next_execution(self) -> ExecutionView | None:
        """Move the oldest queued execution into cancellable preflight."""
        with self._lock:
            if self.needs_reconciliation or self._running_execution_id is not None:
                return None
            if self._stop_queue:
                lane = self._stop_queue
            elif self._restart_queue:
                lane = self._restart_queue
            elif self._queue:
                lane = self._queue
            else:
                return None
            execution_id = lane.popleft()
            execution = self._executions[execution_id]
            execution.status = "preflight"
            self._running_execution_id = execution_id
            for request_id in execution.request_ids:
                request = self._requests[request_id]
                if request.status == "queued":
                    request.status = "preflight"
            return self._execution_view(execution)

    def mark_committed(self, execution_id: str) -> ExecutionView:
        """Cross the irreversible boundary immediately before backend teardown."""
        with self._lock:
            execution = self._require_running(execution_id)
            if execution.status != "preflight":
                raise ValueError("only a preflight execution can be committed")
            if execution.cancel_requested or not self._has_live_callers(execution):
                raise OperationCancelledError(
                    f"execution {execution_id} was cancelled before commit"
                )
            execution.status = "committed"
            for request_id in execution.request_ids:
                request = self._requests[request_id]
                if request.status == "preflight":
                    request.status = "committed"
            return self._execution_view(execution)

    def finish(
        self,
        execution_id: str,
        *,
        error: str | None = None,
        error_type: str | None = None,
    ) -> ExecutionView:
        """Publish a terminal result and release the single running slot."""
        with self._lock:
            execution = self._require_running(execution_id)
            if execution.status not in {"preflight", "committed"}:
                raise ValueError("execution is not running")
            if execution.cancel_requested and execution.status == "preflight":
                execution.status = "cancelled"
            else:
                execution.status = "failed" if error is not None else "succeeded"
            execution.error = error
            execution.error_type = error_type
            for request_id in execution.request_ids:
                request = self._requests[request_id]
                if request.status == "cancelled":
                    continue
                request.status = execution.status
                request.error = error
                request.error_type = error_type
            self._running_execution_id = None
            self._archive_execution(execution.execution_id)
            self._clear_restart_coalescing_if_idle()
            return self._execution_view(execution)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel one caller before commit, or stop waiting after commit.

        Returns true when the request itself was cancelled before commit. A
        committed execution is never interrupted; cancelling its request only
        clears that caller's waiting flag.
        """
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.status in {"cancelled", "succeeded", "failed"}:
                return False
            execution = self._executions[request.execution_id]
            if execution.status == "committed":
                request.waiting = False
                return False

            request.status = "cancelled"
            request.waiting = False
            if not self._has_live_callers(execution):
                execution.cancel_requested = True
                if execution.status == "queued":
                    lane = self._lane_for(execution.operation_type)
                    lane.remove(execution.execution_id)
                    execution.status = "cancelled"
                    self._archive_execution(execution.execution_id)
                    self._clear_restart_coalescing_if_idle()
            return True

    def get_request(self, request_id: str) -> RequestView:
        """Return a copy of one caller-specific request record."""
        with self._lock:
            try:
                return self._request_view(self._requests[request_id])
            except KeyError as error:
                raise KeyError(f"unknown request_id: {request_id}") from error

    def get_execution(self, execution_id: str) -> ExecutionView:
        """Return a copy of one shared execution record."""
        with self._lock:
            try:
                return self._execution_view(self._executions[execution_id])
            except KeyError as error:
                raise KeyError(f"unknown execution_id: {execution_id}") from error

    def get_profile_snapshot(self, execution_id: str) -> dict[str, Any] | None:
        """Return the secret-safe immutable profile facts captured at submission."""
        with self._lock:
            try:
                snapshot = self._executions[execution_id].profile_snapshot
            except KeyError as error:
                raise KeyError(f"unknown execution_id: {execution_id}") from error
            return _validated_profile_snapshot(snapshot)

    def acknowledge_reconciliation(self) -> None:
        """Allow queued work after the caller proves backend state is reconciled."""
        with self._lock:
            self.needs_reconciliation = False

    def list_executions(self) -> tuple[ExecutionView, ...]:
        """Return queued/running work followed by retained history, newest first."""
        with self._lock:
            execution_ids: list[str] = []
            if self._running_execution_id is not None:
                execution_ids.append(self._running_execution_id)
            execution_ids.extend(self._stop_queue)
            execution_ids.extend(self._restart_queue)
            execution_ids.extend(self._queue)
            execution_ids.extend(reversed(self._history))
            return tuple(
                self._execution_view(self._executions[execution_id])
                for execution_id in dict.fromkeys(execution_ids)
                if execution_id in self._executions
            )

    def to_operations_state(self) -> dict[str, Any]:
        """Return a JSON-ready snapshot of queued, active, and retained history."""
        with self._lock:
            return {
                "schema_version": OPERATION_STATE_SCHEMA_VERSION,
                "queue": list(self._queue),
                "restart_queue": list(self._restart_queue),
                "stop_queue": list(self._stop_queue),
                "running_execution_id": self._running_execution_id,
                "restart_coalesced_id": self._restart_coalesced_id,
                "reconciliation_required": self.needs_reconciliation,
                "executions": {
                    execution_id: {
                        "execution_id": execution.execution_id,
                        "operation_type": execution.operation_type,
                        "target_profile_id": execution.target_profile_id,
                        "profile_snapshot": execution.profile_snapshot,
                        "request_ids": list(execution.request_ids),
                        "status": execution.status,
                        "cancel_requested": execution.cancel_requested,
                        "error": execution.error,
                        "error_type": execution.error_type,
                    }
                    for execution_id, execution in self._executions.items()
                },
                "requests": {
                    request_id: {
                        "request_id": request.request_id,
                        "execution_id": request.execution_id,
                        "operation_type": request.operation_type,
                        "status": request.status,
                        "waiting": request.waiting,
                        "error": request.error,
                        "error_type": request.error_type,
                    }
                    for request_id, request in self._requests.items()
                },
                "history": list(self._history),
            }

    @classmethod
    def from_operations_state(
        cls,
        state: Mapping[str, Any] | None,
    ) -> "TransitionQueue":
        """Restore a validated queue snapshot; ``None`` means a fresh queue."""
        queue = cls()
        if state is None:
            return queue
        if not isinstance(state, Mapping):
            raise OperationStateError("operations state must be an object")
        if state.get("schema_version") != OPERATION_STATE_SCHEMA_VERSION:
            raise OperationStateError(
                f"unsupported operations schema_version: {state.get('schema_version')!r}"
            )

        raw_queue = state.get("queue")
        raw_restart_queue = state.get("restart_queue", [])
        raw_stop_queue = state.get("stop_queue", [])
        raw_executions = state.get("executions")
        raw_requests = state.get("requests")
        raw_history = state.get("history")
        running_id = state.get("running_execution_id")
        restart_coalesced_id = state.get("restart_coalesced_id")
        reconciliation_required = state.get("reconciliation_required", False)
        if not isinstance(reconciliation_required, bool):
            raise OperationStateError("reconciliation_required must be a boolean")
        if (
            not isinstance(raw_queue, list)
            or not all(isinstance(value, str) for value in raw_queue)
            or len(raw_queue) > MAX_QUEUED_TRANSITIONS
            or len(set(raw_queue)) != len(raw_queue)
        ):
            raise OperationStateError("operations queue is invalid")
        if any(
            not isinstance(lane, list)
            or not all(isinstance(value, str) for value in lane)
            or len(set(lane)) != len(lane)
            for lane in (raw_restart_queue, raw_stop_queue)
        ):
            raise OperationStateError("priority operation queue is invalid")
        queued_ids = list(raw_queue) + list(raw_restart_queue) + list(raw_stop_queue)
        if len(set(queued_ids)) != len(queued_ids):
            raise OperationStateError("execution appears in multiple queue lanes")
        if not isinstance(raw_executions, dict) or not isinstance(raw_requests, dict):
            raise OperationStateError("operations executions/requests must be objects")
        if not isinstance(raw_history, list) or not all(
            isinstance(value, str) for value in raw_history
        ):
            raise OperationStateError("operations history is invalid")
        if len(raw_history) > COMPLETED_EXECUTION_HISTORY_LIMIT:
            raise OperationStateError("operations history exceeds its retention limit")
        if running_id is not None and not isinstance(running_id, str):
            raise OperationStateError("running_execution_id must be a string or null")
        if restart_coalesced_id is not None and not isinstance(restart_coalesced_id, str):
            raise OperationStateError("restart_coalesced_id must be a string or null")

        valid_statuses = {"queued", "preflight", "committed", "succeeded", "failed", "cancelled"}
        valid_operation_types = {"switch", "restart", "stop", "reload"}
        for execution_id, raw in raw_executions.items():
            if not isinstance(execution_id, str) or not isinstance(raw, dict):
                raise OperationStateError("execution record is invalid")
            request_ids = raw.get("request_ids")
            operation_type = raw.get("operation_type", "switch")
            target_profile_id = raw.get("target_profile_id")
            try:
                profile_snapshot = _validated_profile_snapshot(raw.get("profile_snapshot"))
            except ValueError as error:
                raise OperationStateError(
                    f"execution snapshot {execution_id!r} is invalid: {error}"
                ) from error
            target_is_valid = (
                target_profile_id is None
                if operation_type in {"stop", "reload"}
                else isinstance(target_profile_id, str) and bool(target_profile_id)
            )
            if (
                raw.get("execution_id") != execution_id
                or operation_type not in valid_operation_types
                or not target_is_valid
                or (operation_type == "stop" and profile_snapshot is not None)
                or not isinstance(request_ids, list)
                or not all(isinstance(value, str) for value in request_ids)
                or len(set(request_ids)) != len(request_ids)
                or raw.get("status") not in valid_statuses
                or not isinstance(raw.get("cancel_requested"), bool)
            ):
                raise OperationStateError(f"execution record {execution_id!r} is malformed")
            queue._executions[execution_id] = _ExecutionRecord(
                execution_id=execution_id,
                operation_type=operation_type,
                target_profile_id=target_profile_id,
                profile_snapshot=profile_snapshot,
                request_ids=list(request_ids),
                status=raw["status"],
                cancel_requested=raw["cancel_requested"],
                error=raw.get("error") if isinstance(raw.get("error"), str) else None,
                error_type=raw.get("error_type") if isinstance(raw.get("error_type"), str) else None,
            )

        for request_id, raw in raw_requests.items():
            if not isinstance(request_id, str) or not isinstance(raw, dict):
                raise OperationStateError("request record is invalid")
            execution_id = raw.get("execution_id")
            operation_type = raw.get(
                "operation_type",
                queue._executions[execution_id].operation_type
                if isinstance(execution_id, str) and execution_id in queue._executions
                else None,
            )
            if (
                raw.get("request_id") != request_id
                or not isinstance(execution_id, str)
                or execution_id not in queue._executions
                or operation_type != queue._executions[execution_id].operation_type
                or raw.get("status") not in valid_statuses
                or not isinstance(raw.get("waiting"), bool)
                or request_id not in queue._executions[execution_id].request_ids
            ):
                raise OperationStateError(f"request record {request_id!r} is malformed")
            queue._requests[request_id] = _RequestRecord(
                request_id=request_id,
                execution_id=execution_id,
                operation_type=operation_type,
                status=raw["status"],
                waiting=raw["waiting"],
                error=raw.get("error") if isinstance(raw.get("error"), str) else None,
                error_type=raw.get("error_type") if isinstance(raw.get("error_type"), str) else None,
            )

        if any(execution_id not in queue._executions for execution_id in raw_queue):
            raise OperationStateError("queue references an unknown execution")
        for execution_id in raw_queue:
            execution = queue._executions[execution_id]
            if execution.status != "queued" or execution.operation_type not in {"switch", "reload"}:
                raise OperationStateError("normal queue contains an invalid execution")
        for execution_id in raw_restart_queue:
            execution = queue._executions.get(execution_id)
            if execution is None or execution.status != "queued" or execution.operation_type != "restart":
                raise OperationStateError("restart queue contains an invalid execution")
        for execution_id in raw_stop_queue:
            execution = queue._executions.get(execution_id)
            if execution is None or execution.status != "queued" or execution.operation_type != "stop":
                raise OperationStateError("stop queue contains an invalid execution")
        if running_id is not None:
            if running_id not in queue._executions or queue._executions[running_id].status not in {
                "preflight",
                "committed",
            }:
                raise OperationStateError("running execution record is invalid")
        if any(value not in queue._executions for value in raw_history):
            raise OperationStateError("history references an unknown execution")
        queue._queue = deque(raw_queue)
        queue._restart_queue = deque(raw_restart_queue)
        queue._stop_queue = deque(raw_stop_queue)
        queue._running_execution_id = running_id
        queue._restart_coalesced_id = restart_coalesced_id
        if restart_coalesced_id is not None:
            restart_execution = queue._executions.get(restart_coalesced_id)
            if restart_execution is None or restart_execution.operation_type != "restart":
                raise OperationStateError("restart coalescing reference is invalid")
        queue._history = list(raw_history)
        queue.needs_reconciliation = reconciliation_required
        return queue

    def fail_interrupted_execution(self) -> bool:
        """Fail an in-flight execution found during broker restart."""
        with self._lock:
            if self._running_execution_id is None:
                return False
            execution = self._executions[self._running_execution_id]
            execution.status = "failed"
            execution.error = "interrupted by broker restart; backend reconciliation required"
            for request_id in execution.request_ids:
                request = self._requests[request_id]
                if request.status != "cancelled":
                    request.status = "failed"
                    request.error = execution.error
            self._running_execution_id = None
            self.needs_reconciliation = True
            self._archive_execution(execution.execution_id)
            return True

    def _archive_execution(self, execution_id: str) -> None:
        if execution_id not in self._history:
            self._history.append(execution_id)
        while len(self._history) > COMPLETED_EXECUTION_HISTORY_LIMIT:
            evicted = self._history.pop(0)
            execution = self._executions.pop(evicted, None)
            if evicted == self._restart_coalesced_id:
                self._restart_coalesced_id = None
            if execution is not None:
                for request_id in execution.request_ids:
                    self._requests.pop(request_id, None)

    def _lane_for(self, operation_type: str) -> deque[str]:
        if operation_type in {"switch", "reload"}:
            return self._queue
        if operation_type == "restart":
            return self._restart_queue
        if operation_type == "stop":
            return self._stop_queue
        raise ValueError(f"unknown operation type: {operation_type}")

    def _clear_restart_coalescing_if_idle(self) -> None:
        if (
            self._running_execution_id is None
            and not self._stop_queue
            and not self._restart_queue
            and not self._queue
        ):
            self._restart_coalesced_id = None

    def _find_queued_target(
        self,
        target_profile_id: str,
        profile_snapshot: dict[str, Any] | None,
    ) -> _ExecutionRecord | None:
        for execution_id in self._queue:
            execution = self._executions[execution_id]
            if (
                execution.target_profile_id == target_profile_id
                and execution.profile_snapshot == profile_snapshot
            ):
                return execution
        return None

    def _has_live_callers(self, execution: _ExecutionRecord) -> bool:
        return any(
            self._requests[request_id].status != "cancelled"
            for request_id in execution.request_ids
        )

    def _require_running(self, execution_id: str) -> _ExecutionRecord:
        if self._running_execution_id != execution_id:
            raise ValueError(f"execution {execution_id} is not the running execution")
        return self._executions[execution_id]

    @staticmethod
    def _request_view(request: _RequestRecord) -> RequestView:
        return RequestView(
            request_id=request.request_id,
            execution_id=request.execution_id,
            status=request.status,
            waiting=request.waiting,
            error=request.error,
            operation_type=request.operation_type,
            error_type=request.error_type,
        )

    @staticmethod
    def _execution_view(execution: _ExecutionRecord) -> ExecutionView:
        return ExecutionView(
            execution_id=execution.execution_id,
            target_profile_id=execution.target_profile_id,
            request_ids=tuple(execution.request_ids),
            status=execution.status,
            cancel_requested=execution.cancel_requested,
            error=execution.error,
            operation_type=execution.operation_type,
            error_type=execution.error_type,
        )




class PersistentTransitionQueue:
    """Durable adapter around :class:`TransitionQueue`.

    Every state-changing call is persisted before its result is returned. A
    failed write restores the prior in-memory snapshot and rejects the call.
    """

    def __init__(self, state_store: StateFileStore) -> None:
        self.state_store = state_store
        self._lock = threading.RLock()
        state = state_store.read()
        operations = state.get("operations")
        self._queue = TransitionQueue.from_operations_state(operations)
        if self._queue.fail_interrupted_execution():
            state_store.write_operations(self._queue.to_operations_state())

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return self._queue.queue_depth

    @property
    def pending_transition_count(self) -> int:
        with self._lock:
            return self._queue.pending_transition_count

    @property
    def needs_reconciliation(self) -> bool:
        with self._lock:
            return self._queue.needs_reconciliation

    def submit_load(
        self,
        target_profile_id: str,
        *,
        profile_snapshot: Mapping[str, Any] | None = None,
    ) -> RequestView:
        return self._mutate(
            lambda queue: queue.submit_load(
                target_profile_id,
                profile_snapshot=profile_snapshot,
            )
        )

    def submit_restart(
        self,
        target_profile_id: str,
        *,
        profile_snapshot: Mapping[str, Any] | None = None,
    ) -> RequestView:
        return self._mutate(
            lambda queue: queue.submit_restart(
                target_profile_id,
                profile_snapshot=profile_snapshot,
            )
        )

    def submit_reload(
        self,
        *,
        profile_snapshot: Mapping[str, Any] | None = None,
    ) -> RequestView:
        return self._mutate(
            lambda queue: queue.submit_reload(profile_snapshot=profile_snapshot)
        )

    def submit_stop(self) -> RequestView:
        return self._mutate(lambda queue: queue.submit_stop())

    def next_execution(self) -> ExecutionView | None:
        return self._mutate(lambda queue: queue.next_execution())

    def mark_committed(self, execution_id: str) -> ExecutionView:
        return self._mutate(lambda queue: queue.mark_committed(execution_id))

    def finish(
        self,
        execution_id: str,
        *,
        error: str | None = None,
        error_type: str | None = None,
    ) -> ExecutionView:
        return self._mutate(
            lambda queue: queue.finish(
                execution_id,
                error=error,
                error_type=error_type,
            )
        )

    def cancel_request(self, request_id: str) -> bool:
        return self._mutate(lambda queue: queue.cancel_request(request_id))

    def acknowledge_reconciliation(self) -> None:
        self._mutate(lambda queue: queue.acknowledge_reconciliation())

    def get_request(self, request_id: str) -> RequestView:
        with self._lock:
            return self._queue.get_request(request_id)

    def get_execution(self, execution_id: str) -> ExecutionView:
        with self._lock:
            return self._queue.get_execution(execution_id)

    def get_profile_snapshot(self, execution_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._queue.get_profile_snapshot(execution_id)

    def list_executions(self) -> tuple[ExecutionView, ...]:
        with self._lock:
            return self._queue.list_executions()

    def _mutate(self, action: Callable[[TransitionQueue], Any]) -> Any:
        with self._lock, self.state_store.locked():
            previous = self._queue.to_operations_state()
            result = action(self._queue)
            current = self._queue.to_operations_state()
            if current == previous:
                return result
            try:
                self.state_store.write_operations(current)
            except BaseException:
                self._queue = TransitionQueue.from_operations_state(previous)
                raise
            return result


__all__ = [
    "ExecutionView",
    "OperationCancelledError",
    "OperationStateError",
    "OperationSupersededError",
    "PersistentTransitionQueue",
    "QueueCapacityError",
    "ReconciliationRequiredError",
    "RequestView",
    "TransitionQueue",
]
