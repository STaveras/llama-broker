"""Backend telemetry exercised through real broker and injected HTTP backends."""
import concurrent.futures
import http.client
import json
import threading
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler

from llama_broker.system1 import System1Service
from llama_broker.transition_operations import TransitionQueue
from tests import test_system1 as fixtures


class ControlledD1(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.server.entered.set()
        self.server.release.wait(5)
        if self.server.disconnect:
            self.close_connection = True
            return
        data = self.server.raw
        if data is None:
            data = json.dumps({'answers': self.server.answers}).encode()
        self.send_response(self.server.code)
        self.send_header('Content-Type', self.server.content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


class TelemetryHttpTests(unittest.TestCase):
    setUp_base = fixtures.System1HttpTests.setUp_base
    tearDown_base = fixtures.System1HttpTests.tearDown_base
    _active_alias_ready = fixtures.System1HttpTests._active_alias_ready
    _post = fixtures.System1HttpTests._post
    _get = fixtures.System1HttpTests._get
    tearDown = fixtures.System1HttpTests.tearDown

    def setUp(self):
        fixtures.System1HttpTests.setUp(self)
        self.d1.RequestHandlerClass = ControlledD1
        self.d1.entered = threading.Event()
        self.d1.release = threading.Event()
        self.d1.release.set()
        self.d1.code = 200
        self.d1.content_type = 'application/json'
        self.d1.raw = None
        self.d1.disconnect = False
        self.app = self.broker.application
        self.service = self.app.system1

    def metrics(self):
        return self._get('/health')[1]['system1']['metrics']

    def direct(self, path='/v1/systemone'):
        if path.endswith('decisions'):
            payload = {'model': 'd1-3B', 'input': 'private prompt', 'questions': [
                {'type': 'predicate', 'name': 'q', 'instructions': '?'}]}
        else:
            payload = {'state': 'private prompt', 'questions': {'q': {'type': 'noul'}}}
        conn = http.client.HTTPConnection('127.0.0.1', self.broker.server_port, timeout=8)
        try:
            conn.request('POST', path, body=json.dumps(payload), headers={
                'Authorization': f'Bearer {self.auth_key}', 'Content-Type': 'application/json'})
            response = conn.getresponse()
            data = response.read()
            return response.status, data
        finally:
            conn.close()

    def route(self):
        return self._post({'model': 'auto', 'messages': [{'role': 'user', 'content': 'private prompt'}]})

    def test_initial_snapshot_and_health_probes_do_not_count(self):
        for _ in range(3):
            metrics = self.metrics()
        self.assertEqual(metrics, {
            'scope': 'since_broker_start',
            'direct_requests': {'total': 0, 'active': 0, 'errors': 0},
            'routing': {'total': 0, 'active': 0, 'errors': 0, 'fallbacks': 0,
                        'targets': {}, 'latency_ms': {'last': None, 'mean': None}}})

    def test_validated_direct_endpoints_count_once_each_and_invalid_input_is_excluded(self):
        self.d1.answers = {'q': {'noul': .9}}
        for path in ('/v1/systemone', '/v1/decisions'):
            self.assertEqual(self.direct(path)[0], 200)
        self.assertEqual(self._post({'model': 'd1-3B', 'questions': []}, path='/v1/decisions')[0], 400)
        conn = http.client.HTTPConnection('127.0.0.1', self.broker.server_port)
        conn.request('POST', '/v1/systemone', body=b'{', headers={'Authorization': f'Bearer {self.auth_key}'})
        self.assertEqual(conn.getresponse().status, 400)
        conn.close()
        self.assertEqual(self.metrics()['direct_requests'], {'total': 2, 'active': 0, 'errors': 0})
        self.assertEqual(self.metrics()['routing']['total'], 0)

    def test_direct_http_network_decode_and_translation_errors_release_active(self):
        self.d1.code = 503
        self.assertEqual(self.direct()[0], 503)
        self.d1.code = 200
        self.d1.raw = b'not json'
        # The raw endpoint remains byte-transparent, but records unusable JSON.
        self.assertEqual(self.direct()[0], 200)
        self.assertEqual(self.direct('/v1/decisions')[0], 502)
        self.d1.raw = None
        self.d1.answers = {'q': {}}
        self.assertEqual(self.direct('/v1/decisions')[0], 502)
        self.d1.disconnect = True
        self.assertEqual(self.direct()[0], 502)
        self.assertEqual(self.direct('/v1/decisions')[0], 502)
        self.assertEqual(self.metrics()['direct_requests'], {'total': 6, 'active': 0, 'errors': 6})

    def test_raw_deeply_nested_json_is_forwarded_unchanged_when_telemetry_decode_recurses(self):
        self.d1.raw = b'{"answers":{},"extra":' + b'[' * 10000 + b'0' + b']' * 10000 + b'}'
        self.d1.content_type = 'application/json; charset=utf-8'
        conn = http.client.HTTPConnection('127.0.0.1', self.broker.server_port, timeout=8)
        try:
            conn.request('POST', '/v1/systemone', body=b'{"state":"private prompt"}', headers={
                'Authorization': f'Bearer {self.auth_key}', 'Content-Type': 'application/json'})
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader('Content-Type'), self.d1.content_type)
            self.assertEqual(response.read(), self.d1.raw)
        finally:
            conn.close()
        self.assertEqual(self.metrics()['direct_requests'], {'total': 1, 'active': 0, 'errors': 1})

    def test_success_low_confidence_backend_error_unknown_labels_and_no_target(self):
        self.d1.answers = {'route': {'choice': {'main': .1, 'vision': .9}}}
        self.assertEqual(self.route()[0], 200)
        self.d1.answers = {'route': {'choice': {'main': .2, 'vision': .3}}}
        self.assertEqual(self.route()[0], 200)
        self.d1.code = 503
        self.assertEqual(self.route()[0], 200)
        self.d1.code = 200
        self.d1.answers = {'route': {'choice': {'unknown': 1}}}
        self.assertEqual(self.route()[0], 200)
        self.service.router.config = replace(self.service.router.config, fallback='error')
        self.assertNotEqual(self.route()[0], 200)
        metrics = self.metrics()
        self.assertEqual(metrics['direct_requests']['total'], 0)
        routing = metrics['routing']
        self.assertEqual((routing['total'], routing['active'], routing['errors'], routing['fallbacks']), (5, 0, 3, 3))
        self.assertEqual(sum(routing['targets'].values()), 4)
        self.assertGreaterEqual(routing['latency_ms']['last'], 0)
        self.assertGreater(routing['latency_ms']['mean'], 0)
        self.assertNotIn('private prompt', json.dumps(metrics))

    def test_single_candidate_and_no_candidate_are_route_attempts_without_backend_rpc(self):
        self.service.router.config = replace(self.service.router.config, candidates=('main',))
        self.assertEqual(self.route()[0], 200)
        self.assertFalse(self.d1.entered.is_set())
        self.service.router.config = replace(self.service.router.config, candidates=('missing',))
        self.assertNotEqual(self.route()[0], 200)
        routing = self.metrics()['routing']
        self.assertEqual((routing['total'], routing['errors'], routing['fallbacks']), (2, 1, 0))
        self.assertEqual(routing['targets'], {'main': 1})

    def test_active_snapshots_are_coherent_under_concurrent_blocked_direct_requests(self):
        self.d1.release.clear()
        self.d1.answers = {'q': {'noul': .9}}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                pending = [pool.submit(self.direct) for _ in range(3)]
                self.assertTrue(self.d1.entered.wait(2))
                for _ in range(20):
                    counts = self.metrics()['direct_requests']
                    self.assertEqual(counts['total'], counts['active'])
                    self.assertEqual(counts['errors'], 0)
                self.d1.release.set()
                self.assertEqual([p.result()[0] for p in pending], [200] * 3)
        finally:
            self.d1.release.set()
        self.assertEqual(self.metrics()['direct_requests'], {'total': 3, 'active': 0, 'errors': 0})

    def test_routing_active_while_blocked_and_finally_release_on_network_error(self):
        self.d1.release.clear()
        self.d1.disconnect = True
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pending = pool.submit(self.route)
                self.assertTrue(self.d1.entered.wait(2))
                routing = self.metrics()['routing']
                self.assertEqual((routing['total'], routing['active']), (1, 1))
                self.assertEqual(routing['latency_ms'], {'last': None, 'mean': None})
                self.d1.release.set()
                self.assertEqual(pending.result()[0], 200)
        finally:
            self.d1.release.set()
        routing = self.metrics()['routing']
        self.assertEqual((routing['active'], routing['errors'], routing['fallbacks']), (0, 1, 1))

    def test_target_cardinality_is_bounded_across_dynamic_profile_updates(self):
        for i in range(150):
            self.service.route({}, [fixtures._profile(f'dynamic-{i}')], active_profile=None, available=lambda _: True)
        routing = self.metrics()['routing']
        self.assertLessEqual(len(routing['targets']), 129)
        self.assertGreater(routing['targets']['[other]'], 0)
        self.assertEqual(sum(routing['targets'].values()), 150)
        routing['targets'].clear()
        self.assertEqual(sum(self.metrics()['routing']['targets'].values()), 150)

    def test_health_queue_snapshot_does_not_take_controller_lifecycle_lock(self):
        queue = TransitionQueue()
        self.app.operation_queue = queue
        queue.submit_load('main')
        with self.controller._lock:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                health = pool.submit(self._get, '/health').result(timeout=2)[1]
                self.assertEqual(health['activity']['operations'], {'queued': 1, 'running': None})

    def test_compatibility_active_during_backend_rpc_and_release_after_decode_failure(self):
        self.d1.release.clear()
        self.d1.raw = b'invalid'
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pending = pool.submit(self.direct, '/v1/decisions')
                self.assertTrue(self.d1.entered.wait(2))
                self.assertEqual(self.metrics()['direct_requests'], {'total': 1, 'active': 1, 'errors': 0})
                self.d1.release.set()
                self.assertEqual(pending.result()[0], 502)
        finally:
            self.d1.release.set()
        self.assertEqual(self.metrics()['direct_requests'], {'total': 1, 'active': 0, 'errors': 1})

    def test_client_disconnect_cannot_leak_active_forward_count(self):
        self.d1.release.clear()
        conn = http.client.HTTPConnection('127.0.0.1', self.broker.server_port)
        conn.request('POST', '/v1/systemone', body=b'{"state":"private prompt"}', headers={
            'Authorization': f'Bearer {self.auth_key}'})
        try:
            self.assertTrue(self.d1.entered.wait(2))
            self.assertEqual(self.metrics()['direct_requests']['active'], 1)
            conn.close()
            self.d1.release.set()
            for _ in range(100):
                counts = self.metrics()['direct_requests']
                if counts['active'] == 0:
                    break
            self.assertEqual(counts, {'total': 1, 'active': 0, 'errors': 0})
        finally:
            conn.close()
            self.d1.release.set()

    def test_concurrent_route_completions_have_coherent_totals_targets_and_latency(self):
        profiles = [fixtures._profile('main')]
        def select(_):
            return self.service.route({}, profiles, active_profile=None, available=lambda _: True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(select, range(500)))
        routing = self.metrics()['routing']
        self.assertEqual((routing['total'], routing['active'], routing['errors'], routing['fallbacks']), (500, 0, 0, 0))
        self.assertEqual(routing['targets'], {'main': 500})
        self.assertGreaterEqual(routing['latency_ms']['last'], 0)
        self.assertGreater(routing['latency_ms']['mean'], 0)

    def test_switch_margin_is_not_fallback_but_explicit_confidence_fallback_is(self):
        self.service.router.config = replace(self.service.router.config, min_confidence=.3)
        self.d1.answers = {'route': {'choice': {'main': .45, 'vision': .55}}}
        self.assertEqual(self.route()[0], 200)
        self.service.router.config = replace(self.service.router.config, min_confidence=.8, fallback='vision')
        self.assertEqual(self.route()[0], 200)
        routing = self.metrics()['routing']
        self.assertEqual((routing['total'], routing['errors'], routing['fallbacks']), (2, 0, 1))
        self.assertEqual(routing['targets'], {'main': 1, 'vision': 1})

    def test_queue_snapshot_coalesces_callers_excludes_terminal_and_does_not_advance(self):
        queue = TransitionQueue()
        self.app.operation_queue = queue
        finished = queue.submit_load('done')
        queue.next_execution()
        queue.finish(finished.execution_id)
        first = queue.submit_load('main')
        second = queue.submit_load('main')
        queue.submit_reload()
        self.assertEqual(first.execution_id, second.execution_id)
        self.assertEqual(self._get('/health')[1]['activity']['operations'], {'queued': 2, 'running': None})
        running = queue.next_execution()
        for status in ('preflight', 'committed'):
            if status == 'committed':
                queue.mark_committed(running.execution_id)
            before = queue.list_executions()
            for _ in range(2):
                self.assertEqual(self._get('/health')[1]['activity']['operations'], {
                    'queued': 1, 'running': {'type': 'switch', 'target': 'main', 'status': status}})
            self.assertEqual(queue.list_executions(), before)
        queue.finish(running.execution_id)
        queue.next_execution()
        self.assertEqual(self._get('/health')[1]['activity']['operations']['running'], {
            'type': 'reload', 'target': None, 'status': 'preflight'})
