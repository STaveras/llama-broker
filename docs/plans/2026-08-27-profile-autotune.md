# Profile Autotune Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement `brokerctl profile autotune [PROFILE]` as a bounded, performance-centric maintenance workflow that safely benchmarks runtime candidates, persists winners, and recovers the broker on failure.

**Architecture:** Add an orchestration layer above the existing profile registry, runtime adapters, profile-config transaction helpers, and backend systemd controller. The orchestrator creates disposable candidate units, runs a deterministic local workload, applies only autotune-owned fields, performs one daemon reload after the batch, and uses compensating rollback when persistence or reload recovery fails. Keep all candidate/lifecycle behavior injectable so CI remains offline.

**Tech Stack:** Python 3.11+, existing `argparse`, `pathlib`, `subprocess`, `urllib`, `json`, `statistics`, `hashlib`, systemd user units, existing unittest suite.

---

## Invariants and decisions

- Canonical command is singular: `brokerctl profile autotune [PROFILE]`; remove `profiles`.
- No profile argument processes eligible profiles in manifest order.
- `--force` retunes completed profiles and resumes matching incomplete checkpoints; without it, incomplete attempts are discarded and restarted.
- `--dry-run` performs the real benchmark but does not persist winners.
- `--yes` is required only when stopping the active profile.
- Candidate units are generated temporary systemd user units, loopback-only, sequential, and fresh per candidate.
- No direct subprocess fallback when user systemd is unavailable.
- Only `llamacpp` and `vllm` are supported in v1.
- Candidate generation is adapter-owned adaptive coordinate descent with bounded per-knob sweeps.
- Controller-owned, identity, integrity, security, safety, and lifecycle fields are immutable to autotune.
- `max_output_tokens` is a validated per-profile serving policy and is not an autotune candidate knob; explicit profile/request limits remain supported.
- Primary score is aggregate generated completion tokens/sec; each knob is selected sequentially against the accumulated incumbent, subject to the baseline p95-latency guardrail.
- Per-request latency/throughput variance and 95% confidence intervals are recorded for each mode.
- Prompt preprocessing/prefill is currently included only indirectly in end-to-end timing; TTFT/prefill throughput is not an independent winner-scoring term until first-token timing is measured separately.
- No minimum improvement threshold; the best valid candidate or baseline is persisted.
- Semantic accuracy is not measured; reports must say so.
- Reports retain aggregate metrics and sanitized errors only; no raw logs or request/response bodies.
- Profile winners persist independently; daemon reload happens once after the batch.
- Reload failure triggers compensating rollback; unrecoverable failure remains fail-closed/degraded.
- Any manual/external profile edit invalidates completion through a canonical profile revision digest.
- There is no hardware fingerprint and results are reusable across machines.
- `context_size`, cache types, parallel slots/streams, and flash attention remain
  fixed constraints by default; future opt-in autotune flags such as
  `--context_size`, `--cache_types`, `--parallel`, and `--flash_attention` are
  deferred to later follow-ups.

---

## Task 1: Define autotune metadata and profile revision semantics

**Objective:** Add validated profile metadata for status, ownership, reports, attempts, and revision matching without duplicating configuration authority.

**Files:**
- Modify: `src/llama_broker/broker_profiles.py`
- Modify: `src/llama_broker/profile_config.py`
- Test: `tests/test_broker_profiles.py`
- Test: `tests/test_profile_config.py`

**Steps:**

1. Write failing tests for the allowed `autotune` metadata object and statuses `never`, `complete`, `failed`, `interrupted`, and `conflicted`.
2. Add validation for `max_output_tokens` as a per-profile policy.
3. Add canonical profile revision serialization/digesting that excludes only volatile autotune/report fields.
4. Ensure normal profile fields and `extra_args` remain the sole configuration authority.
5. Test that invalid metadata, invalid output budgets, and protected-field changes fail closed.
6. Run:

```bash
python3 -m unittest tests.test_broker_profiles tests.test_profile_config -v
```

Expected: all targeted tests pass.

---

## Task 2: Rename the CLI namespace and add autotune arguments

**Objective:** Make `profile` the only profile-management namespace and parse the bounded autotune command surface.

**Files:**
- Modify: `src/llama_broker/cli.py`
- Modify: `tests/test_brokerctl.py`
- Modify: `README.md`
- Modify: `SETUP_GUIDE.md`
- Modify: `docs/proposals/unified-cli.md`

**Steps:**

1. Add failing parser tests for:

```text
brokerctl profile autotune [PROFILE]
--force --dry-run --yes --max-experiments --time-budget --json
```

2. Remove the plural `profiles` parser and dispatch path.
3. Reject unknown autotune options and validate positive budget values.
4. Keep parsing separate from execution so tests do not stop services.
5. Document singular `profile` examples and the maintenance requirement.
6. Run:

```bash
python3 -m unittest tests.test_brokerctl -v
python3 -m llama_broker.cli profile autotune --help
```

Expected: parser tests pass and help shows only the agreed first-version surface.

---

## Task 3: Add adapter-owned tuning schemas

**Objective:** Give `llamacpp` and `vllm` explicit bounded tuning policies and a safe schema-less fallback.

**Files:**
- Modify: `src/llama_broker/runtimes/registry.py`
- Modify: `src/llama_broker/runtimes/llamacpp.py`
- Modify: `src/llama_broker/runtimes/vllm.py`
- Create: `src/llama_broker/autotune_schema.py`
- Test: `tests/test_runtimes.py`
- Test: `tests/test_autotune_schema.py`

**Steps:**

1. Define immutable schema types for tunable fields/flags, value generators, priority, incompatibilities, and restart requirements.
2. Add the controller-owned denylist for host, port, alias, model/binary paths, credentials/TLS, supervision, PID, and logging ownership.
3. Define native performance families for each registered runtime.
4. Keep llama.cpp fit controls in a separate opt-in family disabled by default.
5. Implement `--help` parsing only for explicit finite boolean/enum/bounded numeric domains.
6. Implement safe `extra_args` ownership/patching and reject ambiguous duplicates.
7. Test that arbitrary paths, strings, unbounded values, and protected arguments never become candidates.
8. Run:

```bash
python3 -m unittest tests.test_runtimes tests.test_autotune_schema -v
```

Expected: schemas generate deterministic candidates and fallback discovery rejects unsafe domains.

---

## Task 4: Implement deterministic workload probes and scoring

**Objective:** Build the standard-library benchmark protocol used identically for baseline and candidates.

**Files:**
- Create: `src/llama_broker/autotune_benchmark.py`
- Create: `tests/fixtures/autotune_vision_fixture.bin` or another approved small local fixture
- Test: `tests/test_autotune_benchmark.py`

**Steps:**

1. Write failing tests for deterministic probe payloads, fixed sampling settings, warmups, serial requests, concurrency waves, and repeat caps.
2. Implement capability probe selection for text, tools/JSON, and vision.
3. Implement both non-streaming and streaming measurements.
4. Record aggregate completion tokens/sec, prompt tokens/sec, request rate, p50/p95 latency, TTFT, startup time, counts, and sanitized error classes.
5. Treat startup/readiness failures, crashes, timeouts, malformed responses, truncation, and context rejection as invalid candidates.
6. Ensure reports contain aggregate data only—no bodies, previews, hashes, or raw logs.
7. Implement lexicographic winner selection: maximum throughput, exact-throughput p95-latency tie-break, separate best-latency result.
8. Run:

```bash
python3 -m unittest tests.test_autotune_benchmark -v
```

Expected: fake HTTP responses produce deterministic scores and rejection reasons.

---

## Task 5: Implement staged candidate planning and checkpoints

**Objective:** Generate baseline-first staged coordinate-descent plans with bounded interaction passes and resumable checkpoints.

**Files:**
- Create: `src/llama_broker/autotune_search.py`
- Test: `tests/test_autotune_search.py`

**Steps:**

1. Write failing tests for baseline inclusion, adapter priority ordering, one-knob refinement, coupled batch/micro-batch interaction, stable fallback ordering, and 64-candidate/30-minute limits.
2. Implement deterministic candidate identity from profile revision, owned settings, and candidate-plan version.
3. Add checkpoint records for completed candidates and current stage.
4. Make `--force` resume only matching immutable identity, baseline, and plan; stale checkpoints start fresh.
5. Ensure the best valid baseline is retained when no candidate improves it.
6. Run:

```bash
python3 -m unittest tests.test_autotune_search -v
```

Expected: candidate plans are deterministic, bounded, and resumable.

---

## Task 6: Add temporary systemd candidate lifecycle

**Objective:** Reuse the existing unit-rendering/lifecycle path for isolated candidate services and safe cleanup.

**Files:**
- Modify: `src/llama_broker/backend_controller.py`
- Create: `src/llama_broker/autotune_units.py`
- Test: `tests/test_autotune_units.py`
- Test: `tests/test_backend_controller.py`

**Steps:**

1. Write fake-systemd tests for generated names, loopback ports, unique aliases, bounded safety, start/readiness/stop, and exact-prefix cleanup.
2. Generate units named `llama-broker-autotune-<run>-<candidate>.service` without installing persistent profile units.
3. Use `systemctl --user` and `daemon-reload` through injectable command helpers.
4. Verify exact candidate alias and process quiescence after stop.
5. Discover and clean only orphaned autotune-prefixed units before a run.
6. Fail before mutation if the user systemd session is unavailable.
7. Run:

```bash
python3 -m unittest tests.test_autotune_units tests.test_backend_controller -v
```

Expected: fake lifecycle tests cover normal completion, failed startup, timeout, and cleanup.

---

## Task 7: Add broker maintenance lock and lease state

**Objective:** Coordinate CLI maintenance with the daemon while keeping read-only health/status available.

**Files:**
- Create: `src/llama_broker/autotune_lease.py`
- Modify: `src/llama_broker/daemon.py`
- Modify: `tests/test_broker_http.py`
- Create: `tests/test_autotune_lease.py`

**Steps:**

1. Write failing tests for exclusive mutation locking, lease ownership, progress updates, expiry, and read-only status visibility.
2. Add a local-only lease protocol that cannot remotely initiate autotune.
3. Expose maintenance phase, current profile/candidate, counters, intentional backend downtime, and recovery state in `/health` and `/admin/status`.
4. Make inference return the existing unavailable response during maintenance.
5. On lease expiry, clean autotune units, discard unapplied changes, reload pre-run state if necessary, and restore the previously healthy active profile.
6. Run:

```bash
python3 -m unittest tests.test_autotune_lease tests.test_broker_http -v
```

Expected: lease loss is recoverable and conflicting mutators fail fast.

---

## Task 8: Implement daemon profile hot-reload

**Objective:** Reload a validated manifest into an idle daemon without restarting its listener or corrupting coordinator state.

**Files:**
- Modify: `src/llama_broker/daemon.py`
- Modify: `src/llama_broker/broker_profiles.py`
- Modify: `tests/test_broker_http.py`
- Modify: `tests/test_backend_controller.py`

**Steps:**

1. Write failing tests for reload while busy, reload while idle, invalid manifest, registry swap, and preserved coordinator state.
2. Require a fully idle/quiescent coordinator before swapping the registry.
3. Validate the current manifest completely before changing in-memory state.
4. Add a loopback-only maintenance reload operation used by the local CLI, not a remote autotune trigger.
5. Ensure data-plane listener/control-plane process remains running.
6. Run:

```bash
python3 -m unittest tests.test_broker_http tests.test_backend_controller -v
```

Expected: reload is atomic from request users’ perspective and invalid manifests leave the old registry active.

---

## Task 9: Implement report persistence and three-way autotune application

**Objective:** Apply only autotune-owned winners, preserve unrelated edits, and maintain latest/last-success reports safely.

**Files:**
- Create: `src/llama_broker/autotune_reports.py`
- Modify: `src/llama_broker/profile_config.py`
- Test: `tests/test_autotune_reports.py`
- Modify: `tests/test_profile_config.py`

**Steps:**

1. Write failing tests for aggregate-only reports, latest/last-success retention, atomic finalization, and sanitized errors.
2. Add per-profile atomic persistence for successful winners.
3. Merge unrelated external edits while allowing autotune to overwrite only declared owned fields and autotune metadata.
4. Refuse protected-field conflicts and mark the profile conflicted.
5. Implement compensating restoration of all pre-run snapshots when daemon reload fails.
6. Ensure failed/interrupted runs leave normal profile settings unchanged.
7. Delete older reports only after the replacement report is safely finalized.
8. Run:

```bash
python3 -m unittest tests.test_autotune_reports tests.test_profile_config -v
```

Expected: no partial failed-profile writes and no protected-field clobbering.

---

## Task 10: Implement the autotune orchestrator and CLI execution

**Objective:** Connect selection, maintenance, lifecycle, benchmark, persistence, reload, restoration, output, and exit status.

**Files:**
- Create: `src/llama_broker/autotune.py`
- Modify: `src/llama_broker/cli.py`
- Test: `tests/test_autotune.py`
- Modify: `tests/test_brokerctl.py`

**Steps:**

1. Write failing orchestration tests for inactive profiles, active-profile maintenance, all-profile manifest order, skips, failures, partial success, dry-run, force/resume, and no-op exit codes.
2. Acquire the maintenance lock and publish the lease.
3. Clean orphan candidates.
4. Determine eligible profiles and require `--yes` only if the active profile must stop.
5. Gracefully drain/stop the active profile when needed.
6. For each profile, run a fresh disposable baseline and staged candidates sequentially.
7. Persist successful winners independently unless `--dry-run` is active.
8. Reload the daemon once after the batch.
9. Restore the pre-maintenance active profile through the reloaded registry.
10. Handle interruption, lease expiry, reload failure, rollback, and degraded outcomes.
11. Emit human output or clean JSON and return the agreed exit status.
12. Run:

```bash
python3 -m unittest tests.test_autotune tests.test_brokerctl -v
```

Expected: the complete fake lifecycle state machine passes without real models or systemd.

---

## Task 11: Documentation and release-boundary updates

**Objective:** Document the command without implying semantic accuracy guarantees or hardware fingerprinting.

**Files:**
- Modify: `README.md`
- Modify: `SETUP_GUIDE.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/proposals/profile-autotune-design.md`

**Steps:**

1. Document command syntax, maintenance behavior, budgets, reports, skip/force semantics, and recovery states.
2. State explicitly that semantic accuracy is not measured.
3. State that hardware fingerprints are not used and results may be reused across machines.
4. Document `/health` and `/admin/status` maintenance state.
5. Document systemd as a prerequisite and no subprocess fallback.
6. Run Markdown/link checks available in the repository.

---

## Final verification

Run the complete offline suite and static checks:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -q
python3 tests/run_offline.py
python3 -m compileall -q src tests
python3 -m llama_broker.cli profile autotune --help
git diff --check
```

Expected:

- all tests pass
- no real model, GPU, systemd unit, or network service is required by CI
- CLI help exposes the agreed singular command and options
- no whitespace errors are reported
- real hardware/runtime qualification is run separately before claiming measured performance gains

## Commit boundaries

Use focused commits after each verified task, for example:

```bash
git add tests/test_autotune_schema.py src/llama_broker/autotune_schema.py
git commit -m "feat: add autotune schema model"
```

Do not combine implementation with release publication, tags, or pushes. The existing release workflow remains a separate operation.
