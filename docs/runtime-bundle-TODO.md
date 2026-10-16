# Runtime bundle control plane — remaining work

Feature branch: feat/runtime-bundle-control-plane. Production not deployed.

## Release blockers
- [ ] Correct CUDA bundle manifest: record actual recipe, CUDA selection, resolved default ref, and toolchain provenance rather than hardcoded CPU values.
- [ ] Exercise a real CUDA build in the minimal Bubblewrap filesystem; explicitly pass selected compiler/toolkit paths and verify required mounts.
- [ ] Remove arbitrary executable fallback; require the intended llama-server artifact.
- [ ] Independently review manifest path validation before reading any declared file; enforce schema types, traversal rejection, and symlink containment.
- [ ] Make trust-index publication locked and atomic; document its local trust model (not cryptographic authenticity against same-user modification).
- [ ] Separate build directories/IDs by recipe and configuration, not just commit, preventing CPU/CUDA collisions.

## Requested workflow
- [ ] Implement pinned, hash-verified missing-dependency fetch/build outside offline compilation; do not execute arbitrary bootstrap scripts.
- [ ] Complete dependency closure and runtime loader policy: compatible host libraries vs packaged libraries, transitive dependencies and RPATH.
- [ ] Include source URL in human-readable existing-runtime listings, not only JSON or the separate bundle list.
- [ ] Clearly distinguish staged candidate bundles from registered/active runtimes; activation remains explicit.

## Acceptance
- [x] Real CPU fixture build with host-file and network isolation tests.
- [x] Exact commit selection and UI regression tests.
- [x] Full suite at ae90b1b: 703 passed, 119 subtests passed.
- [ ] Full CUDA fixture/build and executable dependency verification.
- [ ] Broker-managed guarded smoke test and independent rollback before production promotion.

Paused to explore Basalt at user's request. Do not treat passing unit tests as full CUDA acceptance.
