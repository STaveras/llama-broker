# Retrospective changelog

Current assigned version: **0.6.12**.

These versions were assigned retrospectively to existing first-parent main checkpoints. Dates below are original committer calendar dates, not verified publication dates. Entries describe the changes at each checkpoint; reading through this file gives the cumulative history. Subsequent entries may supersede earlier behavior. Side branches inherit the latest assigned checkpoint on their first-parent lineage; this file does not claim to describe additional unmerged branch work. Existing older changelog version/date/publication claims are superseded by this evidence-based chronology.

Historical checkpoints were not executed individually during this preparation. Current-main verification results and limitations are recorded in the separate preparation report. A second preparation pass intentionally sanitizes code/configuration and aligns package versions. Historical publication workflows are disabled; the newest main/development snapshots include the replacement manual release workflow. No retrospective tag qualifies a package for publication.

## [0.6.12] - 2026-08-27 — Keep parallelism and flash attention fixed

Original checkpoint: `52bdd25d0dccffd8050b40e08b0147caf448f26c`.

- Remove llama.cpp parallel/flash_attention and vLLM tensor_parallel_size from autotune knob allowlists; adjust schema and help-fallback tests.

## [0.6.11] - 2026-08-27 — Remove deferred README note

Original checkpoint: `480db28c41e191f6a28d359764035541b5ad42cd`.

- Remove the deferred opt-in autotune constraint-family TODO from README. This checkpoint changes documentation only; the context/cache constraints introduced in 0.6.10 remain.

## [0.6.10] - 2026-08-27 — Keep context and cache policy fixed

Original checkpoint: `5e6cac5a800386d8cbf7549d3db45804d1ff5bdb`.

- Remove context_size and cache-type fields from default llama.cpp/vLLM autotune knob allowlists and update schema tests.
- Add a README follow-up note about possible explicit opt-in constraint families; no such opt-in controls are implemented here.

## [0.6.9] - 2026-08-27 — Reset failed candidates before stopping

Original checkpoint: `9856be8bc468aeca204aac60d577a122b2cef9f5`.

- Move reset-failed ahead of stop in candidate cleanup so systemd can clear the loaded unit's failed result before it is garbage-collected; retain best-effort cleanup handling.

## [0.6.8] - 2026-08-27 — Protect the active backend while tuning

Original checkpoint: `13e885a903bdd7f3cc3cb773d95ca0d6d3e13e14`.

- Require the active-backend stop/confirmation path whenever any eligible profile is tuned, including when the active profile is outside the selected set.
- Regenerate controller units when the autotune manifest is reloaded.

## [0.6.7] - 2026-08-27 — Autotune progress and bounded commands

Original checkpoint: `52c8c7ec2f82bd12f9ae3ea7620da54f2af0d0b8`.

- Report candidate start/completion progress and measured results through an engine callback and CLI progress output.
- Bound candidate lifecycle subprocesses and workload request timeouts, heartbeat maintenance during requests, and clear failed systemd candidate state during cleanup.

## [0.6.6] - 2026-08-27 — Apply candidate settings to the live workload

Original checkpoint: `1c9c759dd11c08db36afb354a49398b6644bc0e2`.

- Apply candidate field changes to the temporary profile used by the live HTTP workload, so its settings match the candidate being benchmarked.
- Remove the now-unused benchmark-JSON conversion helper after the switch to live workloads; keep the engine's benchmark callback live and on demand.
- Correct --force help to describe retuning completed profiles without promising matching-work resumption.

## [0.6.5] - 2026-08-27 — Concurrent benchmark waves and cleanup

Original checkpoint: `d71f4ed05082528afcf6e8a6f1b96c7cc5d113a3`.

- Execute bounded concurrency waves with a thread pool instead of serial requests and close HTTP connections in a finally block.
- Allow profile unset autotune to clear completion metadata; show winner throughput, p95 latency, and report location in human-readable output.

## [0.6.4] - 2026-08-27 — Live on-demand autotune

Original checkpoint: `699922468afe536e2a0d8ad5723f816256210b1f`.

- Replace the CLI replay workflow with live isolated candidate runs using a local HTTP workload for text, tools/JSON, and vision-capable profiles.
- Run benchmarks when the engine requests each candidate; collect JSON/stream response measurements and reject malformed, empty, context-rejected, or failed responses.
- Persist per-profile reports, completion metadata and revision fingerprints; skip unchanged completed profiles unless --force is supplied.
- Add active-profile stop confirmation, --yes, experiment/time limits, failure reporting, and real-benchmark --dry-run behavior that avoids persisting winners.

## [0.6.3] - 2026-08-27 — Profile edits preserve the running backend

Original checkpoint: `d79a64aca5421fd3b8fe517139d4de9cccb1862a`.

- Publish validated profile edits and regenerate units without invoking the backend restart callback, including edits to the active profile.
- Retain manifest/unit rollback on installation failure and update the profile-apply regression expectations.

## [0.6.2] - 2026-08-27 — Autotune all profiles by default

Original checkpoint: `0b245aa67808e8ff0e5c8eb25b0218b7cf04b3d8`.

- Make the autotune profile selector optional: omission plans/evaluates all profiles and reports unsupported runtimes as skipped.
- Support per-profile replay result maps and publish multiple winners within one reload/compensation boundary.
- Remove the temporary plural profiles alias and update CLI tests to use profile. Evaluation still uses offline benchmark replay at this checkpoint.

## [0.6.1] - 2026-08-27 — Singular profile command

Original checkpoint: `9a4c3c47d7d48b1d6bd35202f7aa2cebb3f897ee`.

- Make profile the canonical CLI namespace, retaining profiles as an alias at this checkpoint; dispatch autotune and other profile operations through either spelling.

## [0.6.0] - 2026-08-27 — Bounded autotune foundation

Original checkpoint: `5a3cbfdadb5b499df9384b474192fa5909229a6c`.

- Introduce llama.cpp/vLLM tuning schemas, allowlisted field/extra-argument changes, deterministic baseline-first candidate generation, and bounded benchmark/search primitives (64 candidates, up to 30 minutes per profile).
- Add benchmark measurement, maintenance leases, isolated systemd candidate lifecycle helpers, and transactional winner publication with registry reload, active-profile restoration, and compensation/degraded handling.
- Expose profiles autotune with candidate-preview dry runs and offline benchmark-JSON replay; the CLI at this checkpoint requires replay data for evaluation and explicit --commit for publication. Live CLI benchmarking arrives in later checkpoints.
- Expose autotune maintenance/degraded status and loopback administrative reload/degraded endpoints; add engine, lease, lifecycle, and compensation tests.

## [0.5.1] - 2026-08-27 — Backend lifecycle integrity

Original checkpoint: `4c785079e728718e89abcd1b211383d4d3cd84d8`.

- Verify pinned server, model, and optional projector artifacts immediately before backend start through the production controller's verifier hook.
- Retain the watchdog recovery target across failed attempts, switch to it when no profile is active, and cancel recovery state on an explicit administrative stop.
- Add regression cases for integrity failures, recovery behavior, and preserving runtime services during control-plane shutdown.

## [0.5.0] - 2026-08-27 — Runtime build profiles and lifecycle handoff

Original checkpoint: `1a7f5b0de8fbca441dad4a4f5481d8f57c0db38f`.

- Add an inline runtime-build catalog with executable/path/hash validation, build inference, explicit profile selection, setup integration, and concrete running-build identity in state/status.
- Make the profile editor runtime-family/build aware, snapshot its catalog, merge concurrent manifest drafts, and keep active build edits from restarting the backend.
- Preserve runtime services during normal daemon shutdown/handoff; add recovery-watchdog behavior and treat failed units without surviving processes as quiescent.
- Extend reasoning efforts, normalize budgets to thinking_budget_tokens, make an explicit zero budget disable thinking, and emit normalized reasoning metadata. Reject the obsolete primary profile role.

## [0.4.0] - 2026-08-18 — llama.cpp reasoning policy

Original checkpoint: `240237bdbdb145c36e5eb0205ad175170d2f6ee9`.

- Apply a llama.cpp-specific reasoning policy before forwarding requests: default thinking off, map none/low/medium/high efforts, and derive reasoning budgets from the profile output limit.
- Respect explicit thinking-template controls and explicit budgets; leave other runtimes unchanged. Add policy and HTTP forwarding regression cases.

## [0.3.0] - 2026-08-18 — State-driven routing and multipart audio

Original checkpoint: `f657dd12ed281e77a7f2835633c40fd4367d5858`.

- Route omitted model requests to the loaded profile and fail when none is active; remove manifest default_profile selection and require explicit client-model selection for configuration.
- Publish only visible verified broker profiles in the model catalog, rather than exposing the private backend catalog.
- Forward bounded authenticated multipart audio transcription bodies and boundary-bearing content types unchanged to the active profile; validate lengths and avoid logging uploaded media.
- Bound explicit output requests by the per-slot context window; derive omitted output budgets as context_size / parallel cubed (integer division, minimum one), unless an operator ceiling is set.

## [0.2.0] - 2026-08-17 — Command organization

Original checkpoint: `e3b72a41be3d6918687229b46cd9e0bafd87b177`.

- Organize model list, integrate install/configure, and security rotate/revoke into explicit command namespaces; remove the ambiguous top-level list/install/configure and credentials commands.
- Render profile-backed model/runtime tables and schema-versioned JSON without artifact paths; normalize model metadata returned by the HTTP API.
- Separate unit installation from provider configuration and add optional full artifact verification to integration inspection, with tests for the command and integration boundaries.

## [0.1.0] - 2026-08-17 — Initial source snapshot

Original checkpoint: `81661c4765c0183ce38ddfab411ae4a92c56624d`.

- Introduce the src-layout Python package, brokerctl and daemon entry points, setup discovery, profile editing, and client integration adapters.
- Serve an authenticated OpenAI-compatible API with streaming, profile aliases, request leases, bounded drain-on-switch transitions, and one active backend.
- Provide llama.cpp and experimental vLLM adapters, pinned artifact validation, bounded systemd profile units, TLS provisioning, and credential rotation/revocation.
- Include vision/transient-profile handling, an offline regression runner, packaging, and CI/release workflow definitions. This snapshot does not establish that a package or hosted release was published.

