# Runtime Bundle Control Plane Contract

The reviewed scope is now CPU **and CUDA**, superseding the initial CPU-only
plan. These commands stage candidate bundles; they do not register, select,
activate or promote a runtime, change profiles, or control services.

## Source, configuration and provenance

`bundle plan` resolves its displayed source ref to a full immutable commit.
Fetched defaults use the actual `refs/remotes/origin/<branch>` ref; interactive
selection pins the chosen full commit. `bundle build` archives that commit into
a private temporary source directory. It never checks out, cleans or modifies
the registered source repository. Uncommitted files are excluded.

The bounded bundle ID contains a readable project prefix, commit prefix and
128 bits of a SHA-256 digest of canonical JSON containing the **full** project
name, source URL, full commit, recipe, ordered configure flags, host/CUDA
architecture and selected toolchain paths, versions and executable hashes.
CPU native tuning is disabled. C, C++, CMake, linker, assembler and make are
recorded, plus nvcc for CUDA. Each builder gets separate source/build scratch
directories, including concurrent builders of the same configuration. Existing
bundle IDs are immutable and cannot be overwritten. `manifest.json` is the
stable manifest filename; branch/ref metadata describes that build's actual
selection, even when multiple refs resolve to the same configuration identity.

Supported recipes are `cmake-cpu` and `cmake-cuda --cuda-arch 120a`. CUDA requires
a versioned system toolkit, CUDA >= 12.8, and an actual inert CMake/CUDA compile
and link preflight in the build sandbox. Selected C/C++/CUDA compiler paths,
CUDA host compiler and toolkit root are explicit CMake arguments. Failure
returns a blocked plan with `toolchain_issue` and an unvalidated capability;
build refuses it. A successful version string alone is insufficient.
Configured recipe cache values are checked before and after compilation;
configuration overrides fail closed. Exactly one regular executable named
`llama-server` is required. Compiler probes or other executables are never a
fallback.

## Sandbox and dependencies

Compilation uses Bubblewrap with all namespaces unshared, no network, dropped
capabilities, a cleared environment, read-only source, writable private build
output, private `/tmp`, `/proc`, and a minimal device filesystem. Distribution
tools, headers, libraries and CMake data directories are mounted read only.
Only compiler alternatives and the loader cache are exposed from `/etc`.
Homes, `/mnt`, general `/usr/local`, host GPU devices and other host paths are
absent. The selected versioned system CUDA toolkit is an explicit read-only
mount. Missing mount/toolchain support blocks the CUDA preflight.

Bundle creation and verification inspect ELF metadata using `readelf`, never
`ldd` or execution of an untrusted artifact on the host. The intended server,
built shared libraries and their transitive `DT_NEEDED` dependencies form a
verified flat `lib/` closure. ELF64 little-endian x86-64/AArch64 machine and executable/shared-object types
are checked against the host before publication. Non-glibc system/toolkit dependencies are copied,
with SONAME symlink chains preserved once. Ambiguous names, escaped/dangling
links, unresolved dependencies and unsupported loader paths reject publication.
The executable uses origin-relative inherited RPATH so packaged system
libraries without their own RUNPATH can resolve transitive packaged libraries.
Only `$ORIGIN` and `$ORIGIN/../lib` are allowed; each lookup must actually reach
`lib/`. Verification recomputes the graph and hashes, including host dependencies.

The explicit host ABI contract is **same-host glibc and loader**, each recorded
with its exact path and SHA-256. Their transitive ELF dependencies are also
inspected. This is not a claim of portable closure based on the host `ldconfig`
cache. Changed glibc/loader content invalidates verification and requires a
rebuild/review. Arbitrary `dlopen` plugins, GPU driver compatibility, inference,
and GPU execution are outside this ELF closure claim. Promotion still needs a
separate operator-controlled runtime/driver validation.

## Manifest and publication

Bundle manifests use strict **schema 2**. Schema 1 lacked safe nested validation
and truthful configuration identity: it is rejected with **rebuild required**;
there is no automatic migration or trust-index rewrite. Rebuild into a fresh
control root and retain the legacy catalog for inspection; do not copy its
unsafe manifests or anchors into the new catalog. The catalog trust-index
format remains schema 1 (`bundles` maps IDs to manifest SHA-256 values).

All nested types, hashes, paths, link chains, recipe/provenance fields and
dependency records are validated before artifact consumption. Absolute paths,
empty/dot/traversal components, symlink parent directories, escaped links and
nonregular file types are rejected as `RuntimeControlError`. Declared library
links must end at a declared regular library. Host dependency records may name
only the fixed ABI policy paths; verification derives its own trusted paths
rather than reading arbitrary manifest-provided paths. Hash mismatches and
changed dependency facts produce an invalid verification result.

Builders and readers share a catalog filesystem lock. Publication fsyncs the
staged bundle, records a durable pending transaction, renames the bundle, and
atomically replaces/fsyncs the trust index. Readers cannot observe an incomplete
pair. A failed publication restores the previous index and removes the new
bundle. Recovery under the next catalog lock removes an interrupted unanchored
publication, or preserves a fully anchored committed one. If storage failure
also prevents rollback, the error explicitly reports that publication committed
and rollback failed; the anchored complete bundle and recovery record are
preserved. An abrupt process kill can leave private build scratch directories;
these are never selected as bundles.

The index detects accidental changes and corruption. It does **not** provide
cryptographic authenticity against another process running as the same user,
which can replace both index and artifacts. Same-user concurrent malicious
filesystem mutation is outside this local trust model.

## Evidence and acceptance limits

Regression tests exercise real isolated CPU C/C++ builds, transitive shared
libraries, minimal-filesystem execution, a reachable host network fixture,
unsafe manifest/filesystem entries, multiprocess publication and injected
write/crash failures. A real inert CUDA fixture compiles using the selected
system toolkit and validates separate CPU/CUDA identities and manifests.
Tests use disposable state and do not operate on models or production services.

Successful inert CUDA compilation is evidence of sandbox/toolchain support on
the test host, not acceptance of a full production llama.cpp GPU build or GPU
execution. Hosts without the capability receive a visible blocked preflight;
that environment remains unvalidated. See `docs/runtime-bundle-TODO.md` for
promotion and dependency-acquisition work outside this fix.
