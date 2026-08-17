# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (0.1.0 roadmap, not yet cut)

- Packaging: `pyproject.toml` + `src/` layout, console entry points.
- Deep rename: environment-variable prefixes, config directory name, and
  systemd unit names (branding transition).
- CLI consolidation toward a single-verb-per-job surface.

## [0.0.1-beta] - 2026-08-07

First public release. A local model broker / control plane for llama.cpp and
vLLM backends with an OpenAI-compatible API, immutable systemd profile units,
and fail-closed resource admission.

### Added

- `brokerctl` CLI: `list`, `status`, `install`, `render`, `load`/`start`,
  `restart`, `unload`, `verify`, `setup`, `runtime`, `security`, `configure`,
  `credentials`, `profiles`.
- Broker daemon (`llama_broker.py`) with request leasing, drain-on-switch, and
  TLS support.
- Immutable per-profile systemd units with bounded resource budgets
  (memory, swap, tasks) and cooperative shutdown.
- Profile manifest with fail-closed validation: pinned SHA-256 hashes for
  server/model/projector, CUDA-graph safety environment, VRAM/swap admission
  gates.
- vLLM runtime abstraction (`runtimes/`): adapter registry for llama.cpp and
  vLLM.
- `brokerctl setup`: one-shot idempotent install (API-key material, profile
  units, tool configuration, `brokerctl` symlink).
- Interactive profile editor (`brokerctl profiles edit|add|discover`).
- Security: per-deployment API keys, TLS material provisioning, credential
  rotation/revocation.
- Vision-capable profiles with mmproj pinning and VRAM reservation.

### Changed

- Profiles now resolve from the user config directory
  (`~/.config/llama-broker/profiles.json`, or `$XDG_CONFIG_HOME`); a checked-in
  `profiles.example.json` serves as the template. The repository copy is no
  longer read by default.
- The broker systemd unit is rendered by `brokerctl setup` from
  `docs/examples/llama-broker.service.in` instead of being tracked with
  machine-specific paths.
- Generated artifacts (`generated/`, `profiles.json`, local docs) are
  gitignored and no longer shipped.

### Fixed

- `status` output: distinct `Model`/`Backend` labels; Backend shows the runtime
  name and primary binary path, with a safe fallback for profiles without an
  explicit `server_binary`.

### Security

- No secrets ship in the repository: API keys are generated at runtime into
  the user config directory and never printed.
- Public release history is a fresh orphan `master` branch; the private
  pre-release history is not published.
