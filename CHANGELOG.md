# Retrospective changelog

Current assigned version: **0.4.0**.

These versions were assigned retrospectively to existing first-parent main checkpoints. Dates below are original committer calendar dates, not verified publication dates. Entries describe the changes at each checkpoint; reading through this file gives the cumulative history. Subsequent entries may supersede earlier behavior. Side branches inherit the latest assigned checkpoint on their first-parent lineage; this file does not claim to describe additional unmerged branch work. Existing older changelog version/date/publication claims are superseded by this evidence-based chronology.

Historical checkpoints were not executed individually during this preparation. Current-main verification results and limitations are recorded in the separate preparation report. A second preparation pass intentionally sanitizes code/configuration and aligns package versions. Historical publication workflows are disabled; the newest main/development snapshots include the replacement manual release workflow. No retrospective tag qualifies a package for publication.

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

