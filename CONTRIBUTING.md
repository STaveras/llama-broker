# Contributing

Thanks for your interest in llama-broker. This document is short on purpose —
the project is young and wants to stay approachable.

## Project shape

- `src/llama_broker/daemon.py` — broker daemon (request leasing, profile switching, TLS).
- `src/llama_broker/cli.py` — administrative CLI.
- `src/llama_broker/broker_profiles.py` — profile manifest parsing and validation.
- `src/llama_broker/backend_controller.py` — systemd unit install/management.
- `src/llama_broker/runtimes/` — runtime adapters.
- `tests/` — offline pytest and unittest suite (no model or systemd required).

## Development workflow

1. Create a feature branch off `development`.
2. Make focused changes. Keep the offline test suite green:
   ```bash
   uv run --extra dev pytest -q --ignore=tests/test_local_production_manifest.py
   uv run python tools/check_public_tree.py
   uv run python tools/check_release.py --development
   ```
3. Never depend on this machine's `profiles.json` in tests — use
   `tests/fixtures` (see `tests/fixtures/__init__.py`) or construct profiles
   in a tempdir.
4. Open a PR. Describe what changed and why; include test evidence.

Use disposable HOME/XDG directories and unreachable session/system D-Bus
addresses when running the suite on a machine with a broker installation.
HTTP/TLS tests use loopback fixtures. Positive runtime-build tests also need
Linux namespace permission, Bubblewrap, CMake and a compiler. They explicitly
skip with an unvalidated-acceptance reason when that capability is absent;
passing portable tests alone does not qualify a runtime build or GPU execution.
Use a failing regression before changing production code. See `CONTEXT.md` for
the lifecycle contract and `docs/RELEASING.md` for release gates.

## Ground rules

- **Fail closed.** Profiles are validated against the same invariants the
  broker enforces at runtime. If a change relaxes a safety invariant, it will
  be rejected — loosen validation only with a written safety argument.
- **No secrets, no personal data.** Never commit API keys, machine-specific
  absolute paths (`$HOME`, `/mnt/...`), or personal model names. Use
  placeholders and `$HOME`/repo-relative forms.
- **Offline tests only.** Tests must never start a model or talk to systemd.
  Real integration checks belong in `test-scripts.sh`-style scripts, run by
  the operator on their own machine.
- **One verb, one job** (aspirationally): prefer small, composable CLI
  commands over many flags.

## Reporting security issues

Open a private issue, or email the maintainers (see the git log). Do not
publish exploit details before a fix is released. Relevant trust boundaries:
API-key file permissions, TLS material handling, profile-manifest integrity,
and the fail-closed admission gates.
