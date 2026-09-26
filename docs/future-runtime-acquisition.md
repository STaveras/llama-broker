# Future goal: broker-managed source, runtime, and model acquisition

This is future work, not part of the current runtime-binary reconciliation.

## Goal

Allow the broker to manage the complete local inference lifecycle while keeping
production serving fail-closed and reviewable:

1. Clone or update source repositories under the canonical runtime root.
2. Check out an exact source commit and apply declared build configuration.
3. Build in an isolated WSL build workspace without mutating a live bundle.
4. Stage a standalone runtime bundle containing the executable and every shared
   library it needs.
5. Record source repository, commit, dirty-content state, build options,
   executable/library hashes, and provenance in the runtime catalog.
6. Atomically publish the bundle and bind profiles to its content-addressed
   runtime ID.
7. Acquire model files from pinned network sources into a temporary download
   area, verify revision and SHA-256, then atomically install them under the
   canonical model store.
8. Never switch a live profile or delete an older bundle implicitly. Promotion,
   rollback, cleanup, and model selection remain explicit broker operations.

## Development-environment exception

For this WSL development environment, the authoritative BeeLlama source
checkout may live in `~/workspace` and the shared runtime-root path may point to
it by symlink. Built bundles remain standalone and must not depend on the Git
checkout at runtime.

## Required invariants

- `profiles.json` remains the model-to-runtime execution authority.
- `catalog.json` remains the concrete bundle/hash/provenance authority.
- A source checkout is never treated as a serving bundle.
- A bundle's executable and shared libraries are self-contained under its
  bundle directory.
- Dirty builds are visible in provenance and cannot be promoted as clean
  release builds without an explicit qualification decision.
- All acquisition and build operations are transactional, resumable, and
  auditable.
