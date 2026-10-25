"""Watcher acceptance tests using real queues/workers and disposable artifacts."""
import hashlib
import json
import unittest
from unittest.mock import patch

from tests import test_manifest_watcher as watcher_fixtures

TrivialController = watcher_fixtures.TrivialController
make_profile_record = watcher_fixtures.make_profile_record
from llama_broker.broker_profiles import ProfileRegistry
from llama_broker.daemon import BrokerApplication, _manifest_identity


class ReloadController(TrivialController):
    fail_install = False
    installs = 0

    def replace_profiles(self, profiles):
        self.profiles = profiles

    def install_units(self):
        self.installs += 1
        if self.fail_install:
            raise RuntimeError("fixture installation failure")


class ManifestReleaseRegressions(unittest.TestCase):
    def setUp(self):
        self.fixture = watcher_fixtures.ManifestWatcherTests()
        self.fixture.setUp()
        f = self.fixture
        self.controller = ReloadController()
        self.app = BrokerApplication(
            registry=f.application.registry, controller=self.controller,
            backend_host="127.0.0.1", backend_port=10000, backend_timeout=.1,
            max_body_bytes=1024, backend_ready=lambda: True,
            start_operation_worker=False, profiles_path=f.profiles_path,
        )
        self.app._seed_manifest_baseline()

    def tearDown(self):
        self.app._stop_operation_worker()
        self.fixture.tearDown()

    def edit(self, extra="extra"):
        f = self.fixture
        f.write_profiles([make_profile_record("main", "main", f.model),
                          make_profile_record(extra, extra, f.model)])

    def only_request(self):
        executions = self.app.operation_queue.list_executions()
        self.assertEqual(len(executions), 1)
        return executions[0].request_ids[0]

    def finish(self, request):
        self.app.start_operation_worker()
        return self.app.wait_for_operation(request, 5)

    def test_edit_before_watcher_start_is_not_swallowed(self):
        self.edit()
        self.app._seed_manifest_baseline()
        self.app._maybe_reload_manifest()
        request = self.only_request()
        self.assertEqual(self.finish(request)["status"], "succeeded")
        self.assertIsNotNone(self.app.registry.try_resolve("extra"))

    def test_baseline_does_not_advance_on_queue_acceptance(self):
        baseline = self.app._last_manifest_identity
        self.edit()
        self.app._maybe_reload_manifest()
        self.only_request()
        self.assertEqual(self.app._last_manifest_identity, baseline)
        self.assertIsNone(self.app.registry.try_resolve("extra"))

    def test_async_failure_is_retried_without_editing_file(self):
        self.controller.fail_install = True
        self.edit()
        self.app._maybe_reload_manifest()
        first = self.only_request()
        self.assertEqual(self.finish(first)["status"], "failed")
        self.assertIsNone(self.app.registry.try_resolve("extra"))
        self.controller.fail_install = False
        self.app._maybe_reload_manifest()
        executions = self.app.operation_queue.list_executions()
        requests = [r for e in executions for r in e.request_ids if r != first]
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.app.wait_for_operation(requests[0], 5)["status"], "succeeded")
        self.assertIsNotNone(self.app.registry.try_resolve("extra"))
        self.app._maybe_reload_manifest()
        self.assertEqual(self.app._last_manifest_identity, _manifest_identity(self.fixture.profiles_path))

    def test_pending_reload_coalesces_further_edits_until_completion(self):
        self.edit()
        self.app._maybe_reload_manifest()
        first = self.only_request()
        self.edit("third")
        for _ in range(3):
            self.app._maybe_reload_manifest()
        self.assertEqual(self.only_request(), first)
        # Preserve the existing immutable-snapshot guard on queued revisions.
        self.assertEqual(self.finish(first)["status"], "failed")
        self.app._maybe_reload_manifest()
        requests = [r for e in self.app.operation_queue.list_executions() for r in e.request_ids if r != first]
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.app.wait_for_operation(requests[0], 5)["status"], "succeeded")
        self.assertIsNotNone(self.app.registry.try_resolve("third"))

    def test_cancelled_reload_is_retried(self):
        self.edit()
        self.app._maybe_reload_manifest()
        first = self.only_request()
        self.app.cancel_operation(first)
        self.app._maybe_reload_manifest()
        requests = [r for e in self.app.operation_queue.list_executions() for r in e.request_ids]
        self.assertGreater(len(requests), 1)

    def test_loaded_identity_matches_parsed_bytes_not_subsequent_disk(self):
        from llama_broker import profile_config
        f = self.fixture
        original = f.profiles_path.read_bytes()
        resolve = profile_config.resolve_manifest_inheritance

        def edit_after_read(payload):
            self.edit()
            return resolve(payload)

        with patch.object(profile_config, "resolve_manifest_inheritance", side_effect=edit_after_read):
            registry = ProfileRegistry.from_path(f.profiles_path)
        self.assertIsNone(registry.try_resolve("extra"))
        self.assertEqual(registry.manifest_identity, hashlib.sha256(original).hexdigest())
        self.assertNotEqual(registry.manifest_identity, _manifest_identity(f.profiles_path))

    def test_edit_after_parse_during_reload_is_detected_again(self):
        self.edit()
        self.app._maybe_reload_manifest()
        first = self.only_request()
        install = self.controller.install_units

        def edit_during_install():
            self.edit("third")
            install()

        with patch.object(self.controller, "install_units", side_effect=edit_during_install):
            self.assertEqual(self.finish(first)["status"], "succeeded")
        self.assertIsNotNone(self.app.registry.try_resolve("extra"))
        self.assertIsNone(self.app.registry.try_resolve("third"))
        self.app._maybe_reload_manifest()
        requests = [r for e in self.app.operation_queue.list_executions() for r in e.request_ids if r != first]
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.app.wait_for_operation(requests[0], 5)["status"], "succeeded")
        self.assertIsNotNone(self.app.registry.try_resolve("third"))
