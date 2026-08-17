# Contributing

Thanks for your interest in llama-broker. This document is short on purpose —
the project is young and wants to stay approachable.

## Project shape

- `llama_broker.py` — broker daemon (request leasing, profile switching, TLS).
- `brokerctl.py` — administrative CLI.
- `broker_profiles.py` — profile manifest parsing and fail-closed validation.
- `backend_controller.py` — systemd unit install/management.
- `runtimes/` — runtime adapters (llama.cpp, vLLM).
- `tests/` — offline unittest suite (no model, no systemd required).

## Development workflow

1. Create a feature branch off `dev/0.0.1` (or the current dev branch).
2. Make focused changes. Keep the offline test suite green:
   ```bash
   python3 -m py_compile brokerctl.py llama_broker.py backend_controller.py
   PYTHONPATH=. python3 -m unittest discover -s tests -v
   ```
3. Never depend on this machine's `profiles.json` in tests — use
   `tests/fixtures` (see `tests/fixtures/__init__.py`) or construct profiles
   in a tempdir.
4. Open a PR. Describe what changed and why; include test evidence.

## Ground rules

- **Fail closed.** Profiles are validated against the same invariants the
  broker enforces at runtime. If a change relaxes a safety invariant, it will
  be rejected — loosen validation only with a written safety argument.
- **No secrets, no personal data.** Never commit API keys, machine-specific
  absolute paths (`/home/...`, `/mnt/...`), or personal model names. Use
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
