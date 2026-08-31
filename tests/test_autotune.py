from __future__ import annotations

import json
import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence, cast

from llama_broker.autotune import (
    ADAPTERS,
    AutotuneEngine,
    AutotuneTransaction,
    BenchmarkHarness,
    BenchmarkResult,
    Candidate,
    CandidateRejected,
    KnobSpec,
    LeaseBusy,
    MaintenanceLease,
    ResponseMeasurement,
    ReloadCompensationError,
    SystemdCandidateLifecycle,
    TuningSchema,
    apply_winner_to_manifest,
    discover_schema,
    generate_candidates,
    parse_runtime_help,
    patch_autotune_extra_args,
)
from llama_broker.cli import _autotune_report, _autotune_unit_directives


class FakeSession:
    def __init__(self) -> None:
        self.requests: list[tuple[str, bool, bool]] = []
        self.stopped = False

    def request(self, mode: str, *, streaming: bool, concurrent: bool) -> ResponseMeasurement:
        self.requests.append((mode, streaming, concurrent))
        return ResponseMeasurement(
            generated_tokens=100,
            end_to_end_seconds=1.0,
            decode_seconds=0.5,
            ttft_seconds=0.1 if streaming else None,
        )

    def stop(self) -> None:
        self.stopped = True


class FakeLifecycle:
    def __init__(self) -> None:
        self.candidates: list[Candidate] = []
        self.sessions: list[FakeSession] = []

    def start(self, candidate: Candidate) -> FakeSession:
        self.candidates.append(candidate)
        session = FakeSession()
        self.sessions.append(session)
        return session


class AutotuneEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.profile = SimpleNamespace(
            id="demo",
            runtime="llamacpp",
            context_size=4096,
            parallel=1,
            batch_size=256,
            ubatch_size=64,
            cache_type_k="f16",
            cache_type_v="f16",
            flash_attention="auto",
            gpu_layers="99",
            extra_args=("--jinja", "--custom-manual", "keep"),
            capabilities=frozenset({"text", "tools"}),
        )

    def test_autotune_unit_directives_force_graph_disable(self) -> None:
        profile = SimpleNamespace(
            runtime="llamacpp",
            environment={
                "CUSTOM": "keep",
                "GGML_CUDA_DISABLE_GRAPHS": "0",
                "GGML_CUDA_GRAPH_OPT": "1",
                "LLAMA_GRAPH_REUSE_DISABLE": "0",
            },
            safety=SimpleNamespace(
                service=SimpleNamespace(
                    memory_max_mib=20_480,
                    memory_swap_max_mib=1_024,
                    tasks_max=512,
                )
            ),
        )

        directives = _autotune_unit_directives(profile)

        self.assertIn("Environment=GGML_CUDA_DISABLE_GRAPHS=1", directives)
        self.assertIn("Environment=GGML_CUDA_GRAPH_OPT=0", directives)
        self.assertIn("Environment=LLAMA_GRAPH_REUSE_DISABLE=1", directives)
        self.assertIn("Environment=CUSTOM=keep", directives)
        self.assertNotIn("Environment=GGML_CUDA_DISABLE_GRAPHS=0", directives)
        self.assertNotIn("Environment=GGML_CUDA_GRAPH_OPT=1", directives)
        self.assertNotIn("Environment=LLAMA_GRAPH_REUSE_DISABLE=0", directives)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_adapter_schema_has_stable_order_and_fit_is_opt_in(self) -> None:
        schema = discover_schema(self.profile)
        self.assertEqual(schema.runtime, "llamacpp")
        self.assertEqual(
            schema.names,
            ("batch_size", "ubatch_size"),
        )
        self.assertNotIn("context_size", schema.names)
        self.assertNotIn("cache_type_k", schema.names)
        self.assertNotIn("cache_type_v", schema.names)
        self.assertNotIn("parallel", schema.names)
        self.assertNotIn("flash_attention", schema.names)
        self.assertFalse(schema.fit_enabled)
        self.assertEqual(schema.confidence, "high")

        fit_schema = discover_schema(self.profile, include_fit=True)
        self.assertTrue(fit_schema.fit_enabled)
        self.assertNotIn("fit", fit_schema.names)

    def test_help_fallback_only_admits_documented_flags_and_is_low_confidence(self) -> None:
        schema = parse_runtime_help(
            "llamacpp",
            "Usage: -b N --batch-size N\n--unknown-value PATH\n--flash-attn auto|on|off\n",
            profile=self.profile,
        )
        self.assertEqual(schema.source, "runtime-help-fallback")
        self.assertEqual(schema.confidence, "low")
        self.assertEqual(schema.names, ("batch_size",))
        self.assertNotIn("flash_attention", schema.names)

    def test_report_includes_before_after_changes(self) -> None:
        baseline_candidate = Candidate(
            0, {"batch_size": 256}, self.profile.extra_args, "baseline"
        )
        winner_candidate = Candidate(
            1, {"batch_size": 512}, self.profile.extra_args, "one-knob"
        )
        baseline = BenchmarkResult(baseline_candidate, (), True, (), 10.0, 2.0, 2.0, True)
        winner = BenchmarkResult(winner_candidate, (), True, (), 11.0, 1.9, 1.9, False)
        search = SimpleNamespace(
            baseline=baseline,
            winner=winner,
            best_latency=winner,
            evaluated=(baseline, winner),
            stopped_reason="candidate_limit",
        )

        report = cast(dict[str, Any], _autotune_report(cast(Any, self.profile), search))

        self.assertEqual(
            report["winner_changes"],
            {"batch_size": {"before": 256, "after": 512}},
        )
        self.assertEqual(report["experiments"][0]["changes"], {})
        self.assertEqual(
            report["experiments"][1]["changes"],
            {"batch_size": {"before": 256, "after": 512}},
        )

    def test_candidate_order_cap_and_interaction_are_deterministic(self) -> None:
        schema = TuningSchema(
            "llamacpp",
            (
                KnobSpec("batch_size", (128, 256, 512), "adapter", coupled_group="batch"),
                KnobSpec("ubatch_size", (32, 64, 128), "adapter", coupled_group="batch"),
                KnobSpec("parallel", (1, 2), "adapter"),
            ),
            "high",
            "adapter-defined",
        )
        candidates = generate_candidates(self.profile, schema, max_candidates=8)
        self.assertEqual(candidates[0].strategy, "baseline")
        self.assertEqual(candidates[1].changes, {"batch_size": 128})
        self.assertEqual(candidates[2].changes, {"batch_size": 512})
        self.assertLessEqual(len(candidates), 8)
        self.assertEqual(candidates[-1].index, len(candidates) - 1)

    def test_extra_args_patch_preserves_unrelated_manual_arguments(self) -> None:
        result = patch_autotune_extra_args(
            ("--custom-manual", "keep", "--batch-size", "256"),
            ("--batch-size", "512"),
        )
        self.assertEqual(result, ("--custom-manual", "keep", "--batch-size", "512"))
        self.assertEqual(
            patch_autotune_extra_args(("--custom-manual", "keep"), ("--batch-size", "512")),
            ("--custom-manual", "keep", "--batch-size", "512"),
        )
        with self.assertRaises(CandidateRejected):
            patch_autotune_extra_args(("--custom-manual", "keep"), ("--custom-manual", "changed"))

    def test_manifest_application_rejects_forbidden_identity_changes(self) -> None:
        candidate = Candidate(1, {"model_path": "/tmp/evil"}, (), "one-knob")
        with self.assertRaises(CandidateRejected):
            ADAPTERS["llamacpp"].apply({"extra_args": []}, candidate)

    def test_benchmark_uses_a_fresh_session_and_collects_modes(self) -> None:
        lifecycle = FakeLifecycle()
        fixture = Path(__file__)
        harness = BenchmarkHarness(lifecycle, image_fixture=fixture)
        candidate = Candidate(0, {}, (), "baseline")
        result = harness.run(
            candidate,
            capabilities=("text", "tools", "vision"),
            parallelism=2,
            baseline=True,
        )
        self.assertTrue(result.valid)
        self.assertEqual(len(lifecycle.candidates), 1)
        self.assertTrue(lifecycle.sessions[0].stopped)
        self.assertEqual({mode for mode, _, _ in lifecycle.sessions[0].requests}, {"text", "tools-json", "vision"})
        self.assertEqual(result.modes[0].scored_requests, 9)

    def test_mode_result_records_variance_and_confidence_interval(self) -> None:
        mode = BenchmarkHarness._mode_result(
            "text",
            (
                ResponseMeasurement(100, 1.0, decode_seconds=0.5),
                ResponseMeasurement(100, 1.2, decode_seconds=0.5),
                ResponseMeasurement(100, 0.8, decode_seconds=0.5),
            ),
            (),
            1,
        )

        self.assertAlmostEqual(mode.latency_stddev_seconds or 0.0, 0.2)
        self.assertIsNotNone(mode.latency_ci95_seconds)
        self.assertIsNotNone(mode.throughput_ci95_tokens_per_second)
        self.assertEqual(mode.throughput_stddev_tokens_per_second, 0.0)

    def test_engine_requires_valid_baseline_and_selects_throughput_then_latency(self) -> None:
        schema = TuningSchema("llamacpp", (), "high", "adapter-defined")
        baseline = Candidate(0, {}, (), "baseline")
        candidate = Candidate(1, {"batch_size": 512}, (), "one-knob")
        results = {
            0: BenchmarkResult(baseline, (), True, (), 10.0, 2.0, 2.0, True),
            1: BenchmarkResult(candidate, (), True, (), 10.0, 1.0, 1.0, False),
        }
        seen: list[tuple[int, bool]] = []

        def benchmark(item: Candidate, is_baseline: bool) -> BenchmarkResult:
            seen.append((item.index, is_baseline))
            return results[item.index]

        result = AutotuneEngine(
            schema=schema,
            candidates=(baseline, candidate),
            benchmark=benchmark,
        ).run()
        self.assertEqual(seen, [(0, True), (1, False)])
        self.assertEqual(result.winner.candidate.index, 1)
        self.assertEqual(result.best_latency.candidate.index, 1)

    def test_engine_combines_sequential_knob_winners(self) -> None:
        schema = TuningSchema(
            "llamacpp",
            (
                KnobSpec("knob_a", (0, 1, 2), "test", "high"),
                KnobSpec("knob_b", (0, 1, 2), "test", "high"),
            ),
            "high",
            "test",
        )
        candidates = (
            Candidate(0, {"knob_a": 0, "knob_b": 0}, (), "baseline"),
            Candidate(1, {"knob_a": 1}, (), "one-knob", {"knob": "knob_a"}),
            Candidate(2, {"knob_a": 2}, (), "one-knob", {"knob": "knob_a"}),
            Candidate(3, {"knob_b": 1}, (), "one-knob", {"knob": "knob_b"}),
            Candidate(4, {"knob_b": 2}, (), "one-knob", {"knob": "knob_b"}),
        )
        scores = {
            (0, 0): 10.0,
            (1, 0): 12.0,
            (2, 0): 11.0,
            (1, 1): 13.0,
            (1, 2): 12.5,
        }
        seen: list[dict[str, int]] = []

        def benchmark(item: Candidate, is_baseline: bool) -> BenchmarkResult:
            changes = {key: int(value) for key, value in item.changes.items()}
            seen.append(changes)
            score = scores[(changes["knob_a"], changes["knob_b"])]
            return BenchmarkResult(item, (), True, (), score, 1.0, 1.0, is_baseline)

        result = AutotuneEngine(
            schema=schema,
            candidates=candidates,
            benchmark=benchmark,
        ).run()

        self.assertEqual(result.winner.candidate.changes, {"knob_a": 1, "knob_b": 1})
        self.assertEqual(seen, [
            {"knob_a": 0, "knob_b": 0},
            {"knob_a": 1, "knob_b": 0},
            {"knob_a": 2, "knob_b": 0},
            {"knob_a": 1, "knob_b": 1},
            {"knob_a": 1, "knob_b": 2},
        ])

    def test_engine_enforces_latency_guardrail_for_knob_winners(self) -> None:
        schema = TuningSchema(
            "llamacpp",
            (KnobSpec("knob", (0, 1, 2), "test", "high"),),
            "high",
            "test",
        )
        baseline = Candidate(0, {"knob": 0}, (), "baseline")
        too_slow = Candidate(1, {"knob": 1}, (), "one-knob", {"knob": "knob"})
        acceptable = Candidate(2, {"knob": 2}, (), "one-knob", {"knob": "knob"})

        def benchmark(item: Candidate, is_baseline: bool) -> BenchmarkResult:
            if item.changes["knob"] == 1:
                return BenchmarkResult(item, (), True, (), 20.0, 1.1, 1.1, is_baseline)
            if item.changes["knob"] == 2:
                return BenchmarkResult(item, (), True, (), 11.0, 0.9, 0.9, is_baseline)
            return BenchmarkResult(item, (), True, (), 10.0, 1.0, 1.0, is_baseline)

        result = AutotuneEngine(
            schema=schema,
            candidates=(baseline, too_slow, acceptable),
            benchmark=benchmark,
        ).run()

        self.assertEqual(result.winner.candidate.changes, {"knob": 2})

    def test_engine_reports_candidate_progress(self) -> None:
        schema = TuningSchema("llamacpp", (), "high", "adapter-defined")
        baseline = Candidate(0, {}, (), "baseline")
        candidate = Candidate(1, {"batch_size": 512}, (), "one-knob")
        result = BenchmarkResult(baseline, (), True, (), 10.0, 2.0, 2.0, True)
        progress: list[tuple[str, int, int, int, bool | None]] = []

        search = AutotuneEngine(
            schema=schema,
            candidates=(baseline, candidate),
            benchmark=lambda item, is_baseline: result if is_baseline else replace(
                result, candidate=item, baseline=False
            ),
            progress=lambda phase, item, completed, total, measured: progress.append(
                (phase, item.index, completed, total, measured is not None)
            ),
        ).run()

        self.assertTrue(search.baseline.valid)
        self.assertEqual(
            progress,
            [("started", 0, 0, 2, False), ("completed", 0, 1, 2, True),
             ("started", 1, 1, 2, False), ("completed", 1, 2, 2, True)],
        )

    def test_maintenance_lease_expires_and_can_be_reclaimed(self) -> None:
        now = [100.0]
        alive = {41: True, 42: True}
        lease_path = Path(self._tmpdir.name) / "maintenance.json"
        first = MaintenanceLease(
            lease_path, ttl_seconds=10, pid=41, clock=lambda: now[0], pid_alive=lambda pid: alive.get(pid, False)
        )
        second = MaintenanceLease(
            lease_path, ttl_seconds=10, pid=42, clock=lambda: now[0], pid_alive=lambda pid: alive.get(pid, False)
        )
        first.acquire()
        with self.assertRaises(LeaseBusy):
            second.acquire()
        now[0] = 111.0
        self.assertTrue(second.status()["expired"])
        second.acquire()
        self.assertTrue(second.status()["held"])
        first.release()  # Cannot remove the replacement owner's lease.
        second.ensure_valid()
        second.release()

    def test_systemd_lifecycle_cleans_unique_unit_after_session_stop(self) -> None:
        commands: list[tuple[str, ...]] = []
        unit_dir = Path(self._tmpdir.name) / "units"
        lifecycle = SystemdCandidateLifecycle(
            lambda _candidate: ("/bin/true", "--safe"),
            lambda _unit, _mode, _streaming, _concurrent: ResponseMeasurement(1, 1.0),
            unit_dir=unit_dir,
            runner=lambda command: commands.append(tuple(command)),
            readiness_probe=lambda _candidate: True,
        )
        session = lifecycle.start(Candidate(3, {}, (), "one-knob"))
        self.assertEqual(len(list(unit_dir.glob("*.service"))), 1)
        session.stop()
        self.assertEqual(list(unit_dir.glob("*.service")), [])
        self.assertTrue(any(command[-2:] == ("start", getattr(session, "unit_name")) for command in commands))
        self.assertTrue(any(len(command) >= 2 and command[-2] == "stop" for command in commands))
        self.assertTrue(any(len(command) >= 2 and command[-2] == "reset-failed" for command in commands))

    def test_systemd_lifecycle_removes_unit_when_start_fails(self) -> None:
        commands: list[tuple[str, ...]] = []

        def runner(command: Sequence[str]) -> None:
            commands.append(tuple(command))
            if len(command) >= 2 and command[-2] == "start":
                raise RuntimeError("start failed")

        # The runner raises on the exact start command while cleanup remains
        # best-effort; the unit file must not survive the interruption.
        lifecycle = SystemdCandidateLifecycle(
            lambda _candidate: ("/bin/true",),
            lambda _unit, _mode, _streaming, _concurrent: ResponseMeasurement(1, 1.0),
            unit_dir=Path(self._tmpdir.name) / "failed-units",
            runner=runner,
        )
        with self.assertRaises(RuntimeError):
            lifecycle.start(Candidate(4, {}, (), "one-knob"))
        self.assertEqual(list((Path(self._tmpdir.name) / "failed-units").glob("*.service")), [])

    def test_transaction_reloads_once_and_restores_active_profile(self) -> None:
        path = Path(self._tmpdir.name) / "profiles.json"
        manifest = {"schema_version": 1, "profiles": [{"id": "demo", "extra_args": []}]}
        path.write_text(json.dumps(manifest), encoding="utf-8")
        reloads: list[int] = []
        restored: list[str | None] = []
        transaction = AutotuneTransaction(
            path,
            reload_registry=lambda: reloads.append(1),
            restore_active=lambda profile: restored.append(profile),
        )
        result = transaction.commit(
            manifest,
            profile_id="demo",
            winner=Candidate(1, {"batch_size": 512}, (), "one-knob"),
            adapter=ADAPTERS["llamacpp"],
            previous_active="demo",
        )
        self.assertEqual(result["reloads"], 1)
        self.assertEqual(len(reloads), 1)
        self.assertEqual(restored, ["demo"])
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["profiles"][0]["batch_size"], 512)

    def test_transaction_compensates_manifest_and_registry_on_reload_failure(self) -> None:
        path = Path(self._tmpdir.name) / "profiles.json"
        manifest = {"schema_version": 1, "profiles": [{"id": "demo", "extra_args": []}]}
        original = json.dumps(manifest).encode()
        path.write_bytes(original)
        calls = [0]
        restored: list[str | None] = []

        def reload_registry() -> None:
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError("new registry invalid")

        transaction = AutotuneTransaction(
            path,
            reload_registry=reload_registry,
            restore_active=lambda profile: restored.append(profile),
        )
        with self.assertRaises(Exception) as raised:
            transaction.commit(
                manifest,
                profile_id="demo",
                winner=Candidate(1, {"batch_size": 512}, (), "one-knob"),
                adapter=ADAPTERS["llamacpp"],
                previous_active="demo",
            )
        self.assertNotIsInstance(raised.exception, ReloadCompensationError)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(calls[0], 2)
        self.assertEqual(restored, ["demo"])

    def test_transaction_marks_degraded_when_compensation_also_fails(self) -> None:
        path = Path(self._tmpdir.name) / "profiles.json"
        manifest = {"schema_version": 1, "profiles": [{"id": "demo", "extra_args": []}]}
        path.write_text(json.dumps(manifest), encoding="utf-8")
        reasons: list[str] = []
        transaction = AutotuneTransaction(
            path,
            reload_registry=lambda: (_ for _ in ()).throw(RuntimeError("registry unavailable")),
            restore_active=lambda _profile: None,
            mark_degraded=reasons.append,
        )
        with self.assertRaises(ReloadCompensationError):
            transaction.commit(
                manifest,
                profile_id="demo",
                winner=Candidate(1, {"batch_size": 512}, (), "one-knob"),
                adapter=ADAPTERS["llamacpp"],
                previous_active="demo",
            )
        self.assertEqual(len(reasons), 1)
        self.assertIn("compensation failed", reasons[0])


if __name__ == "__main__":
    unittest.main()
