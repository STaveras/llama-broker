# Retrospective changelog

Current assigned version: **0.1.0**.

These versions were assigned retrospectively to existing first-parent main checkpoints. Dates below are original committer calendar dates, not verified publication dates. Entries describe the changes at each checkpoint; reading through this file gives the cumulative history. Subsequent entries may supersede earlier behavior. Side branches inherit the latest assigned checkpoint on their first-parent lineage; this file does not claim to describe additional unmerged branch work. Existing older changelog version/date/publication claims are superseded by this evidence-based chronology.

Historical checkpoints were not executed individually during this preparation. Current-main verification results and limitations are recorded in the separate preparation report. A second preparation pass intentionally sanitizes code/configuration and aligns package versions. Historical publication workflows are disabled; the newest main/development snapshots include the replacement manual release workflow. No retrospective tag qualifies a package for publication.

## [0.1.0] - 2026-08-17 — Initial source snapshot

Original checkpoint: `81661c4765c0183ce38ddfab411ae4a92c56624d`.

- Introduce the src-layout Python package, brokerctl and daemon entry points, setup discovery, profile editing, and client integration adapters.
- Serve an authenticated OpenAI-compatible API with streaming, profile aliases, request leases, bounded drain-on-switch transitions, and one active backend.
- Provide llama.cpp and experimental vLLM adapters, pinned artifact validation, bounded systemd profile units, TLS provisioning, and credential rotation/revocation.
- Include vision/transient-profile handling, an offline regression runner, packaging, and CI/release workflow definitions. This snapshot does not establish that a package or hosted release was published.

