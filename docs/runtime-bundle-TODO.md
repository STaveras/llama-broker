# Runtime bundle control plane — review status

Work is on the isolated `fix/runtime-bundle-review` branch. Production is not
deployed. The CPU-only initial scope has been superseded by CPU/CUDA support.
See [the contract](plans/runtime-bundle-control-plane.md).

## Reviewed fixes

- [x] Actual recipe, CUDA architecture, resolved source ref and full commit,
  configure flags and compiler/toolkit/binutils provenance are recorded.
- [x] CPU/CUDA configuration identity includes full canonical build inputs;
  bounded IDs and per-builder archived source/build directories prevent mixing.
- [x] Intended `llama-server` is mandatory; executable fallback is removed.
- [x] Strict schema 2 validates nested paths, hashes, file/link types, dependency
  records and provenance before artifact consumption. Unsafe schema 1 requires
  rebuilding; no automatic migration changes old trust anchors.
- [x] Locked publication, atomic/fsynced index replacement, rollback and journal
  recovery cover competing builders and interrupted publication.
- [x] Transitive ELF closure, packaged non-glibc libraries, SONAME links and
  origin RPATH are verified. Only explicitly pinned host glibc/loader remain
  host supplied. `ldconfig` inventory is no longer called portable closure.
- [x] Existing registered-runtime human listing includes source URL and wraps
  with the terminal width. Artifact-first model listing is preserved.
- [x] Candidates remain staged until a separate explicit activation workflow.
- [x] CUDA selects compiler/host compiler/toolkit paths explicitly and mounts
  the versioned toolkit read only; real sandbox compile/link preflight blocks
  unavailable or unsupported environments with a capability error.

## Acceptance evidence

- [x] Real CPU C and C++ fixture compile, verify and minimal-filesystem execution.
- [x] Actual host secret isolation and unreachable-from-sandbox host listener.
- [x] Real inert CUDA 13.4 fixture compilation for 120a using the selected system
  toolchain in Bubblewrap on the review host. CPU/CUDA same-ref builds coexist.
- [x] Multiprocess different/identical builder tests, index write and
  post-replacement rollback tests, and killed-publisher recovery.
- [x] Captured RED/GREEN regression logs; full integrated suite: 798 passed,
  153 subtests passed. ELF architecture and nonregular catalog-file rejection
  are covered, with spawned multiprocess tests and no fork warnings.

## Remaining work outside these fixes

- [ ] Pinned hash-verified dependency acquisition before offline compilation.
  Missing or unsupported ELF dependencies currently fail closed. No bootstrap
  scripts or implicit network fetches are introduced.
- [ ] Full production llama.cpp GPU build and GPU execution/driver acceptance.
  The inert fixture verifies compilation and ELF packaging only. Dynamic GPU
  driver loading and arbitrary `dlopen` plugins are outside the static ELF
  closure claim. Other host/toolkit combinations remain unvalidated until their
  real capability preflight passes.
- [ ] Broker-managed guarded model smoke test and independent rollback before
  production promotion. No model, service, live configuration or release was
  touched by this review fix.
