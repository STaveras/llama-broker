"""Atomic shared storage for backend and transition-operation state.

The existing backend state remains at the JSON top level. The ``operations``
key is a separate namespace updated through read/modify/write under a shared
file lock so backend lifecycle writes cannot erase queued-operation records.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


class StateFileError(RuntimeError):
    """Raised when durable broker state is unreadable or has the wrong shape."""


_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.RLock] = {}


def _thread_lock_for(path: Path) -> threading.RLock:
    key = path.resolve()
    with _LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


def atomic_write_text(path: Path | str, text: str, *, mode: int = 0o644) -> None:
    """Durably replace a text file via a same-directory temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class StateFileStore:
    """Serialize backend and operation-state updates to one state file."""

    def __init__(
        self,
        state_path: Path | str,
        *,
        lock_path: Path | str | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.lock_path = (
            Path(lock_path)
            if lock_path is not None
            else self.state_path.with_name(f".{self.state_path.name}.lock")
        )
        self._thread_lock = _thread_lock_for(self.lock_path)
        self._local = threading.local()

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Acquire the shared process/thread state lock, reentrantly per thread."""
        with self._thread_lock:
            depth = getattr(self._local, "depth", 0)
            if depth:
                self._local.depth = depth + 1
                try:
                    yield
                finally:
                    self._local.depth -= 1
                return

            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                self._local.depth = 1
                try:
                    yield
                finally:
                    self._local.depth = 0
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def read(self) -> dict[str, Any]:
        """Read a complete backend/operations snapshot."""
        with self.locked():
            return self._read_unlocked()

    def replace_backend_state(self, backend_state: Mapping[str, Any]) -> dict[str, Any]:
        """Replace backend fields while preserving the operations namespace."""
        if not isinstance(backend_state, Mapping):
            raise TypeError("backend_state must be a mapping")
        with self.locked():
            current = self._read_unlocked()
            updated = dict(backend_state)
            if "operations" in current:
                updated["operations"] = current["operations"]
            self._write_unlocked(updated)
            return updated

    def write_operations(self, operations: Mapping[str, Any]) -> dict[str, Any]:
        """Replace only the namespaced operation state."""
        if not isinstance(operations, Mapping):
            raise TypeError("operations must be a mapping")
        with self.locked():
            current = self._read_unlocked()
            current["operations"] = dict(operations)
            self._write_unlocked(current)
            return current

    def backup_and_replace(
        self,
        backend_state: Mapping[str, Any],
        operations: Mapping[str, Any],
    ) -> Path | None:
        """Preserve the current file under a sibling backup, then replace it."""
        if not isinstance(backend_state, Mapping) or not isinstance(operations, Mapping):
            raise TypeError("backend_state and operations must be mappings")
        with self.locked():
            backup_path: Path | None = None
            if self.state_path.exists():
                suffix = time.time_ns()
                backup_path = self.state_path.with_name(
                    f"{self.state_path.name}.corrupt.{suffix}.bak"
                )
                while backup_path.exists():
                    suffix += 1
                    backup_path = self.state_path.with_name(
                        f"{self.state_path.name}.corrupt.{suffix}.bak"
                    )
                try:
                    os.link(self.state_path, backup_path)
                except OSError:
                    shutil.copy2(self.state_path, backup_path)
                    backup_fd = os.open(backup_path, os.O_RDONLY)
                    try:
                        os.fsync(backup_fd)
                    finally:
                        os.close(backup_fd)
                directory_fd = os.open(self.state_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)

            repaired = dict(backend_state)
            repaired["operations"] = dict(operations)
            self._write_unlocked(repaired)
            return backup_path

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            raw = self.state_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"generation": 0, "profile": None, "phase": "inactive"}
        except OSError as error:
            raise StateFileError(f"cannot read controller state: {error}") from error

        try:
            state = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as error:
            raise StateFileError(f"invalid controller state: {error}") from error
        if not isinstance(state, dict):
            raise StateFileError("controller state must be a JSON object")
        return state

    def _write_unlocked(self, state: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(state), indent=2, sort_keys=True) + "\n"
        atomic_write_text(self.state_path, payload)


__all__ = ["StateFileError", "StateFileStore", "atomic_write_text"]
