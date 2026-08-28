# Proposal: `brokerctl profile autotune`

**Status:** Agreed design; implementation not started.

## Goal

Add an on-demand maintenance command that searches runtime-specific serving settings for higher performance on a local broker installation, while preserving operational reliability and safe broker lifecycle behavior.

The command is performance-centric. It does **not** maintain or run a semantic-accuracy benchmark library and must not claim that factual, reasoning, or general model quality was preserved. It qualifies only operational reliability and measured performance.

## Command surface

The canonical namespace is singular:

```bash
brokerctl profile autotune [PROFILE]
```

`brokerctl profiles` is removed rather than retained as an alias.

First-version options:

```text
--force
--dry-run
--yes
--max-experiments N
--time-budget SECONDS
--json
```

Behavior:

- With no `PROFILE`, process eligible profiles in manifest order.
- With `PROFILE`, process only that profile.
- Without `--force`, completed profiles are skipped when their profile revision still matches; incomplete attempts are discarded and a fresh run starts.
- With `--force`, completed profiles are retuned and incomplete attempts are resumed when their checkpoint matches the immutable profile identity, baseline configuration, and candidate plan. A stale checkpoint starts fresh.
- `--dry-run` runs the real experiments and reports the result but does not persist profile changes.
- `--yes` is required only when the active profile must be stopped.
- Default output is human-readable progress and a final summary. `--json` emits only the stable machine-readable result schema.

## Optimization objective

Candidate gating is strict and performance selection is lexicographic:

1. Candidate startup and readiness must succeed.
2. Required probes must complete without crashes, timeouts, malformed responses, truncation, or context rejection.
3. A candidate must not introduce a new operational error class or worsen a clean baseline.
4. Sweep each knob’s finite values against the current incumbent, retaining a value only when it improves aggregate generated completion tokens per second and stays within the baseline p95-latency guardrail.
5. Evaluate each later knob on top of the values retained by earlier knobs; the final incumbent is the balanced composite.
6. Report the best-latency candidate separately when it differs from the persisted winner.

There is no minimum improvement threshold. The best valid candidate is persisted even when the gain is negligible. If the baseline remains best, persist no setting changes and record `winner_source: baseline`.

Semantic accuracy is explicitly outside the v1 score:

```json
{
  "qualification": "performance_and_operational_reliability",
  "semantic_accuracy": "not_measured"
}
```

## Candidate search

Search is adapter-owned adaptive coordinate descent:

- always measure the current configuration as the baseline
- sweep every finite value for one knob at a time in adapter-defined priority order
- evaluate each knob on top of the incumbent selected by earlier knobs
- retain a knob value only when throughput improves and the baseline p95-latency guardrail holds
- use stable lexical ordering for fallback-discovered flags
- enforce both default limits: 64 candidates and 30 minutes per profile
- allow `--max-experiments` and `--time-budget` overrides, but never unlimited defaults
- run candidates sequentially, never concurrently
- start a fresh backend process for every candidate

The current registered runtimes are the v1 scope: `llamacpp` and `vllm`. Unknown runtimes are ineligible.

### Tuning ownership

Autotune may explore any runtime configuration field declared by the adapter, including runtime-specific `extra_args`, except for permanently protected fields:

- model and binary identity, paths, and hashes
- endpoint host, port, alias, and broker routing
- credentials, TLS, and security settings
- environment secrets
- safety/resource budgets
- lifecycle, supervision, PID, and log-ownership settings

`extra_args` are patched only for flags owned by autotune. Unrelated manual arguments remain intact. Ambiguous duplicate or malformed argv patterns are rejected.

Fit-related llama.cpp controls remain an explicit opt-in schema family and are disabled in ordinary autotune runs.

### Schema discovery

When an adapter has a versioned tuning schema, it supplies:

- tunable field/flag names and types
- bounded values or candidate generators
- knob priority
- incompatible combinations
- restart requirements
- controller-owned denylist enforcement

When no schema exists, fallback discovery may:

- parse the runtime executable's `--help`
- inspect existing `extra_args`
- add only finite, explicit boolean/enum domains or bounded numeric ranges exposed by help
- perturb existing numeric arguments

It must never invent arbitrary strings, paths, model names, or unbounded numeric values. Fallback runs are marked lower-confidence.

## Workload protocol

Each candidate uses deterministic local probes and the same protocol as the baseline:

- two warmups per capability mode
- three scored serial requests for latency and TTFT
- three concurrency waves up to configured profile parallelism
- at most 32 scored requests per capability mode
- adaptive repeats up to seven measurement repetitions when variance is high
- report per-request latency/throughput standard deviation and 95% confidence intervals

Exercise every declared capability:

- text chat/completion
- tools/structured JSON
- vision using a bundled local image fixture

Measure both transport modes:

- non-streaming: aggregate throughput and end-to-end latency
- streaming: TTFT and streaming decode throughput

**Current implementation note:** prompt preprocessing/prefill is not isolated as a
separate measurement yet. The recorded TTFT value currently uses full request
elapsed time, and prompt preprocessing affects candidates only indirectly through
aggregate end-to-end throughput and p95 latency. TTFT/prefill throughput is not
an independent winner-scoring term. A future measurement pass should capture
first-token timing and prefill tokens/sec separately.

Probe requests use temperature 0, a fixed seed where supported, fixed prompts, and fixed output limits. Probe settings do not alter persisted user-facing sampling defaults.

`max_output_tokens` is a validated per-profile serving policy, not a backend-performance knob, and is never generated or changed by ordinary autotune. It remains available for explicit profile/request limits. Context and output limits may be explored only through explicit serving-policy changes; truncation or context-window rejection invalidates the candidate.

## Candidate execution and maintenance

Candidate processes are temporary generated systemd user units:

```text
llama-broker-autotune-<run>-<candidate>.service
```

They use loopback-only ephemeral ports and unique aliases. The benchmark talks directly to the candidate endpoint; normal broker routing is untouched.

Requirements:

- use the existing generated-unit/systemd lifecycle path
- call `daemon-reload` after creating/removing temporary units
- verify exact-alias readiness
- stop and verify quiescence after every candidate
- clean only the exact autotune unit prefix
- never use Docker or a direct subprocess fallback
- fail closed before changing anything if the user systemd session is unavailable
- acquire an exclusive broker-owned maintenance lock for the full run
- conflicting mutating commands fail fast; read-only health/status remains available

If the active profile is selected, perform a normal graceful drain and stop of that profile only. The control plane remains running. No separate announcement is required, but `--yes` is mandatory.

If the active profile was healthy before maintenance, interruption or lease expiry cleans candidates, discards unapplied changes, and restores that pre-maintenance profile. If recovery fails, the broker remains fail-closed/degraded.

An explicitly tuned inactive profile is never activated automatically.

## Daemon maintenance lease and reload

Autotune is initiated locally by the CLI, not through an HTTP trigger. The CLI owns a loopback-only maintenance-state lease with the daemon:

- lease expiry is safe if the CLI dies
- `/health` and `/admin/status` expose maintenance phase, profile/candidate progress, intentional backend downtime, and recovery state
- inference remains unavailable while maintenance is active
- the lease cannot be used as a remote autotune trigger

After the maintenance batch:

1. persist successful profile winners independently and atomically
2. reload the validated manifest once while the coordinator is fully idle
3. preserve coordinator/listener state; do not restart the broker daemon
4. restore the pre-maintenance active profile through the reloaded registry

If reload fails after winners were committed, restore all pre-run manifest snapshots, reload the old registry, and restore the prior active profile. If any compensating step fails, remain fail-closed/degraded and preserve evidence.

For a normal partial run, independent successful profiles may be committed. If the active profile's own tuning failed, do not claim recovery; leave the broker fail-closed/degraded according to the lifecycle state machine.

## Persistence and report model

Winning settings remain authoritative in normal profile fields and `extra_args`. The `autotune` metadata object stores provenance/status only; report copies are evidence, not configuration authority.

Profile statuses:

- `never`
- `complete`
- `failed`
- `interrupted`
- `conflicted`

`running` belongs to the maintenance lease/latest-attempt state so a rerun does not erase the last successful result.

A canonical profile revision digest excludes only volatile autotune/report fields. Any manual or external manifest edit changes the digest and makes the profile eligible for retuning. There is intentionally no hardware fingerprint: results are reusable across machines.

Reports are stored under:

```text
~/.config/llama-broker/autotune/
```

Retain only:

- `latest.json`: newest attempt of any status
- `last-success.json`: most recent successful tuning result

Older attempts are deleted only after the new report is safely finalized. Reports contain aggregate metrics, settings snapshots, lifecycle counts/timings, exit status, and sanitized bounded error classifications. They do not contain request/response bodies, previews, hashes, or raw logs.

Failed or interrupted profiles remain unchanged. Ineligible profiles are skipped during all-profile runs and recorded in the final summary/report; an explicitly named ineligible profile fails clearly.

## Exit statuses

- `0`: every eligible selected profile completed successfully, including baseline-retained outcomes, or at least one profile was already complete and all other profiles were cleanly skipped
- nonzero: a selected profile failed, recovery was incomplete, an explicitly named profile was ineligible, the maintenance prerequisite failed, or an all-profile run had no successful eligible profile

## Testing boundary

CI remains offline and hardware-independent. v1 requires:

- deterministic candidate-generation and scoring tests
- a minimal fake lifecycle harness covering temporary units, stop/start, cleanup, interruption, lease expiry, reload, rollback, restoration, locking, and exit codes
- separate real-GPU/runtime qualification outside standard CI

No semantic accuracy claim is made by these tests.
