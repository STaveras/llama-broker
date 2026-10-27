"""Applied-state regressions using real reload workers and queue operations."""

import json
from pathlib import Path
import time
import threading
from unittest.mock import patch

import pytest

from llama_broker.broker_profiles import ProfileRegistry
from llama_broker.daemon import BrokerApplication, _manifest_identity
from tests.test_manifest_watcher import make_profile_record


class ReloadController:
    def __init__(self):
        self.profiles = {}
        self.fail_install = False
        self.install_hook = None

    def state(self):
        return {"phase": "inactive"}

    def replace_profiles(self, profiles):
        self.profiles = dict(profiles)

    def install_units(self):
        if self.install_hook:
            hook, self.install_hook = self.install_hook, None
            hook()
        if self.fail_install:
            self.fail_install = False
            raise RuntimeError("fixture install failure")


class WatcherFixture:
    def __init__(self, root):
        self.path = root / "profiles.json"
        self.model = root / "main.gguf"
        self.model.write_bytes(b"fixture model")
        binary = root / "llama-server"
        binary.write_bytes(b"fixture server")
        binary.chmod(0o755)
        self.write("main")
        self.loaded_identity = _manifest_identity(self.path)
        self.controller = ReloadController()
        self.app = BrokerApplication(
            registry=ProfileRegistry.from_path(self.path), controller=self.controller,
            backend_host="127.0.0.1", backend_port=10000, backend_timeout=0.1,
            max_body_bytes=1024, backend_ready=lambda: True,
            start_operation_worker=False, profiles_path=self.path,
        )

    def write(self, *names):
        self.path.write_text(json.dumps({"schema_version": 1, "profiles": [
            make_profile_record(name, name, self.model) for name in names
        ]}))
        return _manifest_identity(self.path)

    def executions(self):
        return self.app.operation_queue.list_executions()

    def terminal(self, execution):
        deadline = time.monotonic() + 3
        while True:
            with self.app._operation_admission_lock:
                current = self.app.operation_queue.get_execution(execution.execution_id)
            if current.status in {"succeeded", "failed", "cancelled"}:
                return current
            assert time.monotonic() < deadline, "reload did not reach terminal state"
            time.sleep(0.005)

    def queue_change(self, *names):
        identity = self.write(*names)
        before = {e.execution_id for e in self.executions()}
        self.app._maybe_reload_manifest()
        return identity, next(e for e in self.executions() if e.execution_id not in before)


@pytest.fixture
def watcher(tmp_path):
    fixture = WatcherFixture(tmp_path)
    try:
        yield fixture
    finally:
        fixture.app._stop_manifest_watcher()
        fixture.app._stop_operation_worker()


def test_startup_baseline_is_loaded_bytes_not_later_disk(watcher):
    changed = watcher.write("main", "extra")
    watcher.app._seed_manifest_baseline()
    assert watcher.app._last_manifest_identity == watcher.loaded_identity
    assert watcher.app._last_manifest_identity != changed
    watcher.app._maybe_reload_manifest()
    assert len(watcher.executions()) == 1
    watcher.app.start_operation_worker()
    assert watcher.terminal(watcher.executions()[0]).status == "succeeded"
    assert watcher.app.registry.try_resolve("extra") is not None


def test_registry_identity_comes_from_same_read_that_was_parsed(watcher):
    original = Path.read_bytes
    read_identity = _manifest_identity(watcher.path)

    replaced = False

    def replace_after_read(path):
        nonlocal replaced
        data = original(path)
        if path == watcher.path and not replaced:
            replaced = True
            watcher.write("main", "extra")
        return data

    with patch.object(Path, "read_bytes", replace_after_read):
        registry = ProfileRegistry.from_path(watcher.path)
    assert registry.try_resolve("extra") is None
    assert registry.manifest_identity == read_identity
    assert registry.manifest_identity != _manifest_identity(watcher.path)


def test_success_advances_baseline_without_any_waiter(watcher):
    identity, execution = watcher.queue_change("main", "extra")
    assert watcher.app._last_manifest_identity == watcher.loaded_identity
    watcher.app.start_operation_worker()
    assert watcher.terminal(execution).status == "succeeded"
    assert watcher.app._last_manifest_identity == identity
    watcher.app._maybe_reload_manifest()
    assert len(watcher.executions()) == 1


def test_timeout_late_application_then_exact_restore_reloads_original_bytes(watcher):
    original_bytes = watcher.path.read_bytes()
    identity_b, first = watcher.queue_change("main", "extra")
    entered = threading.Event()
    release = threading.Event()
    coordinator = watcher.app.coordinator
    coordinator._transition_timeout = 0.05

    def block_install():
        entered.set()
        assert release.wait(3), "test did not release install"

    watcher.controller.install_hook = block_install
    watcher.app.start_operation_worker()
    try:
        assert entered.wait(3), "reload did not enter install"
        terminal = watcher.terminal(first)
        assert terminal.status == "failed"
        assert terminal.error_type == "BrokerTransitionTimeoutError"
        assert watcher.app._pending_manifest_reload is None
        assert watcher.app.registry.try_resolve("extra") is None
        assert watcher.app._last_manifest_identity == watcher.loaded_identity
    finally:
        release.set()
        with coordinator._condition:
            assert coordinator._condition.wait_for(
                lambda: not coordinator._transitioning, timeout=3,
            ), "late transition did not finish"

    assert watcher.app.registry.manifest_identity == identity_b
    assert watcher.app.registry.try_resolve("extra") is not None
    assert "extra" in watcher.controller.profiles
    assert watcher.terminal(first).status == "failed"
    watcher.path.write_bytes(original_bytes)
    assert _manifest_identity(watcher.path) == watcher.loaded_identity
    watcher.app._maybe_reload_manifest()
    retries = [e for e in watcher.executions() if e.execution_id != first.execution_id]
    assert len(retries) == 1, "restored A must reload after B applied past its timeout"
    assert watcher.terminal(retries[0]).status == "succeeded"
    assert watcher.app.registry.try_resolve("extra") is None
    assert "extra" not in watcher.controller.profiles
    assert watcher.app.registry.manifest_identity == watcher.loaded_identity
    assert watcher.app._last_manifest_identity == watcher.loaded_identity
    watcher.app._maybe_reload_manifest()
    assert len(watcher.executions()) == 2


def test_autotune_reload_then_exact_restore_reloads_original_bytes(watcher):
    original_bytes = watcher.path.read_bytes()
    identity_b = watcher.write("main", "extra")
    with patch("llama_broker.daemon.PROFILES_PATH", watcher.path):
        watcher.app.reload_manifest_for_autotune()
    assert watcher.app.registry.manifest_identity == identity_b
    assert watcher.app.registry.try_resolve("extra") is not None
    assert "extra" in watcher.controller.profiles
    assert not watcher.executions()

    watcher.path.write_bytes(original_bytes)
    assert _manifest_identity(watcher.path) == watcher.loaded_identity
    watcher.app._maybe_reload_manifest()
    executions = watcher.executions()
    assert len(executions) == 1, "restored A must reload after alternate-path B application"
    assert watcher.app._last_manifest_identity == identity_b
    watcher.app.start_operation_worker()
    assert watcher.terminal(executions[0]).status == "succeeded"
    assert watcher.app.registry.try_resolve("extra") is None
    assert "extra" not in watcher.controller.profiles
    assert watcher.app.registry.manifest_identity == watcher.loaded_identity
    assert watcher.app._last_manifest_identity == watcher.loaded_identity
    watcher.app._maybe_reload_manifest()
    assert len(watcher.executions()) == 1


def test_failed_apply_rolls_back_and_next_tick_retries_without_waiter(watcher):
    watcher.controller.fail_install = True
    identity, execution = watcher.queue_change("main", "extra")
    watcher.app.start_operation_worker()
    assert watcher.terminal(execution).status == "failed"
    assert watcher.app.registry.try_resolve("extra") is None
    assert "extra" not in watcher.controller.profiles
    assert watcher.app._last_manifest_identity == watcher.loaded_identity
    watcher.app._maybe_reload_manifest()
    retry = next(e for e in watcher.executions() if e.execution_id != execution.execution_id)
    assert watcher.terminal(retry).status == "succeeded"
    assert watcher.app.registry.try_resolve("extra") is not None
    assert watcher.app._last_manifest_identity == identity


def test_cancelled_queued_reload_is_retried(watcher):
    identity, execution = watcher.queue_change("main", "extra")
    watcher.app.cancel_operation(execution.request_ids[0])
    assert watcher.terminal(execution).status == "cancelled"
    watcher.app._maybe_reload_manifest()
    retry = next(e for e in watcher.executions() if e.execution_id != execution.execution_id)
    watcher.app.start_operation_worker()
    assert watcher.terminal(retry).status == "succeeded"
    assert watcher.app._last_manifest_identity == identity


def test_cancelling_watcher_caller_does_not_cancel_shared_manual_reload(watcher):
    identity, execution = watcher.queue_change("main", "extra")
    manual = watcher.app.submit_profile_reload_operation()
    assert manual.execution_id == execution.execution_id
    watcher.app.cancel_operation(execution.request_ids[0])
    watcher.app._maybe_reload_manifest()
    assert len(watcher.executions()) == 1
    assert len(watcher.executions()[0].request_ids) == 2
    watcher.app.start_operation_worker()
    assert watcher.terminal(execution).status == "succeeded"
    assert watcher.app._last_manifest_identity == identity


def test_old_successful_waiter_cannot_mark_new_pending_revision_applied(watcher):
    identity_b, first = watcher.queue_change("main", "extra")
    watcher.app.start_operation_worker()
    assert watcher.terminal(first).status == "succeeded"
    watcher.app._stop_operation_worker()
    identity_c, second = watcher.queue_change("main", "extra", "third")
    assert second.status == "queued"
    watcher.app.wait_for_operation(first.request_ids[0], timeout=0.1)
    assert watcher.app._last_manifest_identity == identity_b
    assert watcher.app._last_manifest_identity != identity_c
    assert watcher.app.registry.try_resolve("third") is None
    watcher.app.start_operation_worker()
    assert watcher.terminal(second).status == "succeeded"
    assert watcher.app._last_manifest_identity == identity_c


def test_edit_while_pending_does_not_flood_queue_and_is_eventually_applied(watcher):
    _, first = watcher.queue_change("main", "extra")
    newest = watcher.write("main", "extra", "third")
    for _ in range(5):
        watcher.app._maybe_reload_manifest()
    assert len(watcher.executions()) == 1
    watcher.app.start_operation_worker()
    assert watcher.terminal(first).status == "failed"
    watcher.app._maybe_reload_manifest()
    second = next(e for e in watcher.executions() if e.execution_id != first.execution_id)
    assert watcher.terminal(second).status == "succeeded"
    assert watcher.app.registry.try_resolve("third") is not None
    assert watcher.app._last_manifest_identity == newest


def test_edit_after_commit_is_not_claimed_as_the_applied_revision(watcher):
    identity_b, first = watcher.queue_change("main", "extra")
    watcher.controller.install_hook = lambda: watcher.write("main", "extra", "third")
    watcher.app.start_operation_worker()
    assert watcher.terminal(first).status == "succeeded"
    assert watcher.app._last_manifest_identity == identity_b
    assert watcher.app.registry.try_resolve("third") is None
    assert watcher.app._last_manifest_identity != _manifest_identity(watcher.path)
    watcher.app._maybe_reload_manifest()
    second = next(e for e in watcher.executions() if e.execution_id != first.execution_id)
    assert watcher.terminal(second).status == "succeeded"
    assert watcher.app.registry.try_resolve("third") is not None


def test_evicted_cancelled_request_does_not_wedge_watcher(watcher):
    _, first = watcher.queue_change("main", "extra")
    watcher.app.cancel_operation(first.request_ids[0])
    for _ in range(40):
        request = watcher.app.submit_profile_reload_operation()
        watcher.app.cancel_operation(request.request_id)
    with pytest.raises(KeyError):
        watcher.app.operation_queue.get_execution(first.execution_id)
    watcher.app._maybe_reload_manifest()
    queued = [e for e in watcher.executions() if e.status == "queued"]
    assert len(queued) == 1
    watcher.app.start_operation_worker()
    assert watcher.terminal(queued[0]).status == "succeeded"


def test_stable_tick_never_reparses_or_revalidates_artifacts(watcher):
    watcher.app._seed_manifest_baseline()
    with patch.object(ProfileRegistry, "from_path", side_effect=AssertionError("must not reparse")):
        for _ in range(5):
            watcher.app._maybe_reload_manifest()
    assert not watcher.executions()


def test_manual_successful_reload_updates_applied_identity_without_waiter(watcher):
    identity = watcher.write("main", "extra")
    request = watcher.app.submit_profile_reload_operation()
    watcher.app.start_operation_worker()
    execution = watcher.app.operation_queue.get_execution(request.execution_id)
    assert watcher.terminal(execution).status == "succeeded"
    assert watcher.app._last_manifest_identity == identity
    watcher.app._maybe_reload_manifest()
    assert len(watcher.executions()) == 1
