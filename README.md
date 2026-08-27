# Llama-Broker

Llama-Broker is a local model broker / control plane for self-hosted LLM
serving. The current release runs one serving backend at a time because of
local hardware limits, exposes an OpenAI-compatible API on `:11434/v1`, and
keeps every model profile behind fail-closed validation and bounded resource
isolation.

The target architecture supports multiple broker-managed runtime instances when
memory allows, including multiple instances of the same model. Until the
residency and idle-eviction manager is implemented, clients talk to the broker
and the broker talks to one immutable per-profile systemd unit. Switching models
is a transactional, drain-on-switch operation - active requests finish before
the old backend stops. A failed transition leaves the broker fail-closed;
clients request the next profile after it is repaired.

```text
Hermes and other local clients
             |
             v
Llama-Broker :11434 (TLS-capable, OpenAI-compatible)
  - validates an explicit model alias
  - leases the active profile for the complete response or SSE stream
  - drains active leases before changing profiles
             |
             v
one llama-broker-profile-<id>.service on :10000 (native TLS)
             |
             v
llama-server (llama.cpp) or vllm serve, per profile
```

The native `llama-server` web UI is available directly at `:10000` and belongs
to the currently active profile; a broker profile switch changes the model
shown there.

## Highlights

- **Current one-backend policy, transactional switches.** Requests for a
  different profile wait for active leases to drain; the new backend must
  advertise its exact configured alias before the switch commits.
  Successful transient profiles restore their prior profile after their
  leases drain. If the active backend crashes, the broker records the stale
  profile and retries recovery through the same coordinator with bounded
  exponential backoff; it never relies on systemd to restart profile units.
  Multiple resident instances are a planned
  memory-managed capability, not a current release feature.
- **Immutable, bounded profile units.** Every profile renders to its own
  systemd unit with explicit memory, swap, task, kill-signal, and timeout
  bounds from a validated manifest budget.
- **Fail-closed profiles.** Each entry pins the runtime, model, optional
  projector (mmproj), SHA-256 hashes, context, cache types, and safety
  environment. CUDA graph capture/optimization/PDL must be explicitly
  disabled; unknown or empty model requests are rejected, never substituted.
- **llama.cpp + vLLM runtimes.** A small runtime registry
  (`runtimes/`) owns each backend's complete command shape, so vLLM profiles
  emit `vllm serve MODEL` and llama.cpp profiles emit llama-server flags —
  with cross-runtime field validation.
- **Vision-capable profiles.** A transient vision profile can drain the text
  profile, serve a vision request, and restore the prior profile when leases
  finish.
- **TLS + API keys.** Optional LAN binding with deployment-owned TLS material
  and a private API-key file. Keys are generated at runtime, never printed,
  never committed.

## Quick start (0.1.0)

```bash
# Installed workflow (recommended)
uv tool install llama-broker
brokerctl setup

# Apply reviewed service/provider integration
brokerctl integrate

# Clone workflow remains supported
python3 brokerctl.py setup

# Load a verified model profile explicitly
python3 brokerctl.py load <profile-id>
python3 brokerctl.py status
```

Profiles resolve from `~/.config/llama-broker/profiles.json`
(`$XDG_CONFIG_HOME`-aware, `LLAMA_BROKER_PROFILES` overrides). `brokerctl setup` discovers existing runtimes and model files, shows qualified
and experimental runtimes separately, and requires explicit profile selection.
It never downloads runtimes, copies models, or loads a backend implicitly. Use
`brokerctl setup --check` for a non-mutating human-readable catalog report and
`brokerctl integrate --dry-run` to review integration changes. When setup creates
a profile from a verified runtime executable, it also registers that executable
in the manifest's `runtime_builds` catalog using a stable SHA-256-derived ID, so
later profile edits can select the discovered build without manual catalog work.

The current release keeps one backing runtime active at a time because of the
local hardware budget. The planned residency manager will allow multiple
broker-owned instances, including duplicate instances of one model, and will
evict only the oldest idle instance when memory pressure requires it. That
future behavior is intentionally deferred from 0.1.0.

For TLS on a LAN:

```bash
python3 brokerctl.py security init --ip <lan-ip> --hostname <lan-hostname>
```

## CLI reference

| Command | Purpose |
| --- | --- |
| `brokerctl` (bare) | (planned) launches the interactive TUI |
| `brokerctl model list` | list client-facing model references from canonical profiles (`--json` for a schema-versioned report) |
| `brokerctl status` | show durable/live backend state (`--json` for schema-versioned JSON) |
| `brokerctl render <model>` | render a profile unit without installing |
| `brokerctl load <model>` | queue a profile load (`start` is an alias) |
| `brokerctl restart` | reload the active model or start a new one |
| `brokerctl unload` | stop the current backing backend |
| `brokerctl verify` | verify pinned runtime/model/projector SHA-256 |
| `brokerctl setup` | first-time discovery wizard; `--check`/`--dry-run` are non-mutating |
| `brokerctl runtime list` | list registered runtimes and the profiles selecting each (`--json` for a schema-versioned report) |
| `brokerctl security init` | provision TLS + API-key material |
| `brokerctl security rotate\|revoke` | rotate or revoke native API keys |
| `brokerctl integrate` | install broker units and configure supported provider integrations; partial success is reported |
| `brokerctl integrate install` | install broker-owned units only |
| `brokerctl integrate configure [TOOLS...]` | configure coding tools only |
| `brokerctl integrate --dry-run` | show the combined integration plan and ask before applying it |
| `brokerctl update` | deferred from 0.1.0; runtime acquisition is never automatic |
| `brokerctl profiles list\|discover\|edit\|add` | profile management; profiles remain the canonical model-to-runtime binding and client naming source |
| `brokerctl profiles calibrate ... --autotemp` | (planned) hardware calibration with optional model-keyed behavioral evaluation |

## API and security

The stable additive OpenAI-compatible subset is under `/v1`:

| Route | Contract |
| --- | --- |
| `GET /v1/models` | Authenticated list of currently healthy/serving models; stopped broken or unverifiable profiles are hidden. |
| `GET /v1/models/{model}` | Authenticated model metadata for a verified profile, or the currently healthy active backend. |
| `POST /v1/chat/completions` | Authenticated chat request; an explicit `model` id/alias is routed to the matching profile, while an omitted model uses the currently loaded profile. `stream: true` preserves SSE streaming. |
| `POST /v1/completions` | Authenticated completion request with the same explicit/currently-loaded model routing and streaming behavior. |
| `POST /v1/audio/transcriptions` | Authenticated, bounded `multipart/form-data` upload; the exact body and content type are forwarded to the currently loaded profile without parsing form fields. |

## TODO

- [x] **Forward multipart audio transcription requests.** The broker accepts
  authenticated, bounded `multipart/form-data` uploads at
  `POST /v1/audio/transcriptions`, validates `Content-Length`, preserves the
  exact body and boundary-bearing content type through the authenticated proxy,
  uses the currently loaded profile without parsing form fields, keeps request
  IDs and fail-closed error handling intact, and does not log uploaded media.
  JSON `input_audio` content through `POST /v1/chat/completions` remains the
  other broker-supported audio path.

Requests must provide the OpenAI-standard `Authorization: Bearer $API_KEY` header.
The credential order is raw `LLAMA_BROKER_API_KEY` first, then
`LLAMA_BROKER_API_KEY_FILE`, then locally generated credentials. Missing/invalid
credentials fail closed. `/health` and `/healthz` remain minimal
unauthenticated probes; administrative and backend-control endpoints remain
loopback-only. Every response carries an additive `X-Request-ID`; errors use
an OpenAI-compatible `{ "error": { "message", "type", "code" } }` object
without stack traces or secrets. Normal request logs contain operational
metadata only; prompt/completion bodies are not logged unless explicitly
requested at an appropriate debug/verbose level.

The default binding is loopback. LAN exposure requires explicit opt-in, TLS,
and API-key authentication. Plaintext non-loopback binding is rejected.

Never persist raw secrets in config, profiles, logs, or generated documentation.
OS-level dependency installation is out of scope. The broker may detect missing
dependencies and provide guidance, but must not invoke `sudo`, `apt`, `dnf`,
driver installation, compiler installation, or similar package managers
automatically. Model artifacts should not be copied, moved, overwritten, or
deleted automatically.

## Architecture

- `src/llama_broker/` — installed Python package and reusable implementation.
- `llama_broker.py` — checkout daemon entry point for direct debugging: request leasing, drain-on-switch,
  TLS, credential forwarding.
- `brokerctl.py` — checkout administrative entry point; installed users use the `brokerctl` console command.
- `broker_profiles.py` — profile manifest parsing and fail-closed validation.
- `backend_controller.py` — systemd unit install/management and the switch
  coordinator.
- `broker_security.py` — TLS material and API-key lifecycle.
- `runtimes/` — runtime adapters (llamacpp, vllm).
- `configure/` — per-tool provider configuration (Hermes, Codex, Claude Code,
  OpenCode, OpenClaw).

The administrative interface is `brokerctl`; `manage-services.sh` is a
compatibility shim. Profile switching must go through the broker's drain
coordinator (via `brokerctl`), not by calling `systemctl` on profile units
directly.

## Security model

- **No secrets in the repo.** API keys are generated at runtime into
  `~/.config/llama-broker/api-keys` (mode-restricted) and never printed.
- **TLS with deployment-owned material.** `security init` creates a local CA
  plus broker/backend certificates covering `127.0.0.1`, the LAN IP, and the
  LAN hostname. Private keys are mode 600. A non-loopback binding without both
  TLS and the API-key file is rejected at startup.
- **Fail-closed admission.** Profiles pin hashes (server/model/projector),
  enforce CUDA-graph safety environment, bound memory/swap/tasks per unit, and
  cap explicit output requests at the per-slot context window. Requests without
  a client budget use the inverse-parallel default
  (`slot_context_window // parallel`); `LLAMA_BROKER_MAX_OUTPUT_TOKENS` may set
  an explicit operator ceiling.
- **No GPU-telemetry dependency for normal switching.** Resource bounds come
  from systemd cgroups; `nvidia-smi`-based admission is only used in explicit
  qualification experiments.
- **Lifecycle failure handling.** Successful transient requests retain the
  current restore behavior. Failed transitions stop/quarantine the failed
  target, clear the active alias, and do not automatically restore or restart
  the previous profile; structured `[broker-event]` JSON records the outcome.

## Development

Offline-only regression suite — never starts a model, never touches systemd,
and does not require GPUs, CUDA, model files, runtime binaries, drivers,
compilers, Hermes, or OpenCode:

```bash
python3 tests/run_offline.py
python3 -m pip wheel . --no-deps --wheel-dir /tmp/llama-broker-dist
```

Tests must not depend on this machine's `profiles.json`; use
`tests/fixtures` (see `tests/fixtures/__init__.py`).

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and ground
rules (fail-closed, no secrets, offline tests).

## 0.1.0 release boundary

The qualified first-release runtime is llama.cpp. vLLM and retained adapters
are experimental and require explicit selection. Runtime acquisition, compiler
or driver installation, multi-instance residency/eviction, cloud control,
scoped credentials, and calibration/autotune are deferred. Standard CI is
offline and hardware-independent; GPU/runtime qualification is a separate
self-hosted activity.

## License

[MIT](LICENSE)
