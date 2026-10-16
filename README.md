<!-- retrospective-release:start -->
## Retrospective version: 0.6.16

**Profile editor row and dirty-state fixes** — source checkpoint c34c5e8fba6f (2026-08-27).

Assigned retrospectively during release preparation; this is not evidence of a historical publication. [Cumulative release notes](CHANGELOG.md) describe this checkpoint and all earlier assigned versions.

The README below is historical text with portable examples. During a second preparation pass, private deployment snapshots were removed, machine-specific examples and migration defaults were sanitized, package versions were aligned, and release workflows were made explicitly manual. These are preparation changes, not evidence of historical package publication.
<!-- retrospective-release:end -->

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
  environment. `GGML_CUDA_PDL=0` remains an explicit llama.cpp safety
  invariant; CUDA graph capture/optimization controls are scoped to autotune
  candidate units and are omitted from production units. Unknown or empty model
  requests are rejected, never substituted.
- **llama.cpp, vLLM, and Strata runtimes.** A small runtime registry
  (`runtimes/`) owns each backend's command and transport contract. vLLM emits
  `vllm serve MODEL`, llama.cpp emits llama-server flags, and Strata uses a
  pinned bridge with a loopback-only HTTP backend. Runtime-specific profile
  fields are validated fail-closed.
- **Strata IQ2_XS packages.** Strata profiles pin both GGUF shards, the native
  pack inventory, launch config, engine, and bridge. They are text-only,
  single-slot, MCP-disabled, and refused unless their loopback/API-key and RAM
  admission checks pass.
- **External chat templates.** Advanced profile editing accepts a template
  file path for llama.cpp and vLLM profiles. llama.cpp launches with `--jinja`
  and `--chat-template-file`; vLLM passes the path as `--chat-template`.
- **Vision-capable profiles.** A transient vision profile can drain the text
  profile, serve a vision request, and restore the prior profile when leases
  finish.
- **TLS + API keys.** Optional LAN binding with deployment-owned TLS material
  and a private API-key file. Keys are generated at runtime, never printed,
  never committed.

## Connection reliability

Authenticated inference requests wait in FIFO order when inference workers are
occupied or a model transition is temporarily blocking admission. Waiting
requests hold no model lease, so they cannot prevent draining or restoration of
a temporary profile. Administration retains its own worker reserve. A newer
inference request cannot bypass an accepted waiter, including while the head
request's body is being validated or its profile is activating.
Requests for a different profile also wait for existing generations to finish
before starting activation, using the inference queue deadline.
Later HTTP requests wait for an earlier activation to settle; the coordinator's
optional idle-load preemption for direct lease callers does not override HTTP
FIFO admission.

| Setting | Default | Bound |
| --- | --- | --- |
| `LLAMA_BROKER_MAX_WORKERS` | `2` | Inference worker reservations, including response writers, closing transports and committed activations finishing after cancellation |
| `LLAMA_BROKER_CONTROL_WORKERS` | `2` | Separate administrative workers |
| `LLAMA_BROKER_INFERENCE_QUEUE_MAX_SOCKETS` | `16` | Additional waiting inference connections; `0` disables extra waiting slots |
| `LLAMA_BROKER_INFERENCE_QUEUE_MAX_BODY_BYTES` | `67108864` | Aggregate declared body bytes for queued requests, reserved before body reads |
| `LLAMA_BROKER_INFERENCE_QUEUE_WAIT_TIMEOUT` | `360` seconds | Time from authenticated body admission through validation, queue waiting and profile activation |
| `LLAMA_BROKER_MAX_BODY_BYTES` | `33554432` | Maximum body bytes per request, including active requests |
| `LLAMA_BROKER_CONTROL_CLIENT_TIMEOUT` | `2` seconds | TLS handshake, total header-read and body-read deadlines; downstream writes also have this finite timeout |
| `LLAMA_BROKER_SSE_HEARTBEAT_INTERVAL` | `15` seconds | Interval between SSE comment heartbeats during quiet periods |
| `LLAMA_BROKER_BACKEND_TIMEOUT` | `1800` seconds | Finite total inference proxy deadline after acquiring a lease, including connect, headers, prefill and response delivery |

Each connection serves one request. At most 16 additional connections can be
parsing headers, handshaking or delivering/draining a rejection. Once that
pre-admission capacity is exhausted, further connections close without an HTTP
response; no credentials or request ID have been parsed yet. Otherwise queue
capacity errors return HTTP 503 with `broker_queue_full`, and queue expiry
returns HTTP 504 with `broker_queue_expired`. Raw queued bodies and active bodies
have the bounds above; JSON objects and rewritten payloads add bounded Python
allocation overhead. Slow uploads and downstream readers can hit their I/O
deadline independently of the queue deadline. These settings cannot extend a
client's own deadline.
Retained client transports, including connections still closing, have a hard
limit of inference workers + queue slots + control workers + 16 (36 by default).
Clients must keep the request transport open until the response finishes;
send-side EOF after the complete body is treated as cancellation. A committed
activation can retain its bounded worker reservation until the lifecycle worker
settles, even after the caller's wait deadline has expired.
If a catalog reload changes or removes a request's validated profile while it
waits, admission fails closed with `broker_request_stale` (409); the client must
resubmit against the new profile facts.

Only `POST /v1/chat/completions` and `POST /v1/completions` with JSON
`stream: true` receive broker SSE heartbeats. After authentication, body and
model validation, and bounded admission, the broker commits HTTP 200 with
`Content-Type: text/event-stream` immediately. It writes `: heartbeat` comments
while waiting in the queue, activating a profile, waiting for upstream headers,
prefilling, or receiving an idle/fragmented stream. Comments are inserted between
complete SSE events. JSON, audio and other endpoints never receive comments.
One handler writes downstream; a separate reader owns all upstream HTTP reads
and passes data through a bounded buffer. SSE events are limited to 262144 bytes;
the supported completion streams must finish with upstream `data: [DONE]`.

**HTTP 200 acknowledges an accepted stream, not successful generation.** An
upstream non-success status, incompatible response type, truncated stream,
backend failure/timeout, queue expiry, or activation failure after these headers
produces a terminal SSE error, then closes the connection:

```text
event: error
data: {"error":{"message":"backend request failed","type":"broker_proxy_error","code":"broker_proxy_error"},"request_id":"example-request","status":502}

```

Clients must inspect these error events; the broker does not synthesize `[DONE]`
after an error. Delivering a terminal error requires a writable client transport;
disconnected or stalled writers may only observe the connection closing.
Early validation and admission rejection still use ordinary
HTTP errors with JSON bodies. Non-streaming upstream responses retain their
status and content type; failures after their headers close the response.
Dispatched generation is never retried. Disconnects cancel waiting work and
interrupt upstream I/O; an already committed profile activation finishes under
its lifecycle deadline, releases any late lease, and does not dispatch generation
for the cancelled caller. Lifecycle deadlines bound waiting; controller work that
outlives its deadline remains quarantined by the coordinator.
Shutdown rejects new admission, wakes waiters and
interrupts client transports while preserving the existing backend handoff
policy.

Structured `[broker-event]` records distinguish `queue_wait`, `queue_full`,
`queue_expired`, `backend_failure`, `backend_timeout`, `client_disconnect`, and
`server_shutdown`, correlated with the response's `X-Request-ID`. IDs generated
for clients without one are also forwarded upstream. Request bodies and backend
error bodies are not included in these diagnostics. Unparsed capacity rejection
uses `connection_full` with a broker-generated ID.
Client I/O expiry and changed profile facts use `request_timeout` and
`request_stale` respectively.

The disposable HTTP/TLS reliability fixtures need loopback socket access and
`openssl` for temporary test certificates; they never start a model or operate
on services or live configuration:

```bash
uv run --extra dev pytest -q tests/test_connection_reliability.py tests/test_connection_reliability_wire.py
uv run --extra dev pytest -q
```

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
External template files are supplied by the operator and must be readable by
the selected serving runtime.

The current release keeps one backing runtime active at a time because of the
local hardware budget. The planned residency manager will allow multiple
broker-owned instances, including duplicate instances of one model, and will
evict only the oldest idle instance when memory pressure requires it. That
future behavior is intentionally deferred from 0.1.0.

For TLS on a LAN:

```bash
python3 brokerctl.py security init --ip <lan-ip> --hostname <lan-hostname>
```

## Local model bundle listing

`brokerctl model list` keeps the configured profile table and adds a file tree
for local models under the XDG models directory (`LLAMA_BROKER_MODELS_DIR`
overrides). Use `--scan-path DIR` to scan another directory, including a Hugging
Face cache's `hub` or `snapshots` directory; repeat it to scan several roots.
Listing only reads directory entries, small config/index files and file sizes.
It never hashes weights, updates the catalog, imports profiles or contacts a
backend. The existing profile manifest is still required.

HF safetensors are grouped at the directory containing a model `config.json`
(with a nonempty string `model_type` or list of architecture names) or a
`*.safetensors.index.json`. An index defines the weight membership, including
relative paths into weight subdirectories. Without an index, each unsharded
weight or conventional `-00001-of-00002.safetensors` set is a separate bundle.
Known sidecars in that root (config, generation/processor/quantization config,
tokenizer, vocab/merges, chat templates, README and license files) appear as
children. Arbitrary files and metadata subdirectories are not inferred as members.
Nested model roots, different indexes, unindexed weight variants and independent
GGUFs remain separate. Conventional split GGUFs retain their own shard grouping.

HF cache snapshots retain their semantic filenames rather than blob hashes.
Different snapshots are separate bundles, even if they share blobs. Overlapping
scan roots and directory symlink aliases are deduplicated; symlink cycles, hidden
directories, cache `blobs`/`refs`, and partial download names are excluded.
Files have deterministic root-relative paths. Each bundle's byte total counts
each physical `(device, inode)` once, including sidecars: symlinks and hard links
do not inflate it, but byte-identical separate files do. Shared bytes can be
counted again in another logical bundle, so adding bundle totals does not give
unique disk usage for the whole store.

`complete` means the enumerated weight files are readable and nonempty.
`incomplete` means a shard is missing/empty/unfinished, or an index is unreadable,
malformed, empty, unsafe, or claims another model's files. Partial artifacts and
broken symlinks are omitted from children; missing members and index issues are
reported explicitly. Neither status asserts integrity, tokenizer validity,
runtime compatibility or serving readiness. In particular, a renamed but still
growing final weight file cannot be detected without download coordination.

The normal tree includes every member and its size; `--verbose` (`-v`) adds full
bundle roots and file references. `--width` wraps human output. `--json` always
includes full paths and all members, independent of verbosity. Its schema-1
`object: "model_list"` and canonical `data` profile records remain unchanged;
the additive `bundles` array contains `name`, `path`, `bundle_root`, `format`,
`status`, `files` (`path`, `relative_path`, `role`, `size`), `size`, `size_policy`,
`missing_files` and `issues`. `size_policy` is `unique-inode-per-bundle`.
`path` is a representative present weight file (or the index/expected file for
an incomplete bundle); `bundle_root` is display metadata. Bundles do not create
loadable IDs. Continue selecting `data[].id` for `load`, and use explicit file
paths for existing setup/import operations. Listing does not change import,
load or cleanup/delete targets; no operation treats a bundle root as permission
to act on that directory.

## CLI reference

| Command | Purpose |
| --- | --- |
| `brokerctl` (bare) | (planned) launches the interactive TUI |
| `brokerctl model list` | list canonical profile references and local model bundles with their files (`--json` for machine output, `-v` for full bundle paths, repeatable `--scan-path DIR` to override the models directory) |
| `brokerctl status` | show durable/live backend state (`--json` for schema-versioned JSON) |
| `brokerctl profile render <model>` | render a profile unit without installing |
| `brokerctl load <model>` | queue a profile load (`start` is an alias) |
| `brokerctl restart` | reload the active model or start a new one (`--json` for the full machine-readable response) |
| `brokerctl unload` | stop the current backing backend |
| `brokerctl verify` | verify pinned runtime/model/projector SHA-256 |
| `brokerctl clean [models\|runtimes\|backups\|caches]` | list unused models, orphaned runtime bundles and old manifest backups, then ask before cleaning them (scripts and `--json` only get the list); `--apply` cleans without asking. Cleaning deletes; with `--archive-dir DIR` (or `LLAMA_BROKER_ARCHIVE_DIR`) models and backups are copied there and SHA-256 verified first. Package caches are included only when `caches` is named. Paths in `~/.config/llama-broker/clean-keep` (or `--keep`) are never touched |
| `brokerctl setup` | first-time discovery wizard; `--check`/`--dry-run` are non-mutating |
| `brokerctl runtime list` | list registered runtimes and the profiles selecting each (`--json` for a schema-versioned report) |
| `brokerctl security init` | provision TLS + API-key material |
| `brokerctl security rotate\|revoke` | rotate or revoke native API keys |
| `brokerctl integrate` | install broker units and configure supported provider integrations; partial success is reported |
| `brokerctl integrate install` | install broker-owned units only |
| `brokerctl integrate configure [TOOLS...]` | configure coding tools only |
| `brokerctl integrate --dry-run` | show the combined integration plan and ask before applying it |
| `brokerctl update` | deferred from 0.1.0; runtime acquisition is never automatic |
| `brokerctl profiles list\|discover\|edit\|add` | profile management; `profile edit` opens a numbered selection menu when no model ID/alias is supplied; profiles remain the canonical model-to-runtime binding and client naming source |
| `brokerctl profiles calibrate ... --autotemp` | (planned) hardware calibration with optional model-keyed behavioral evaluation |

## API and security

The stable additive OpenAI-compatible subset is under `/v1`:

| Route | Contract |
| --- | --- |
| `GET /v1/models` | Authenticated list of currently healthy/serving broker profiles; stopped, broken, or unverifiable profiles are hidden. Each model reports aggregate `context_length` (the profile's `-c` value), `parallel` (the profile's `-np` value), and per-slot `slot_context_window`. |
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
- `runtimes/` — runtime adapters (llamacpp, vllm, strata).
- `configure/` — per-tool provider configuration (Hermes, Codex, Claude Code,
  OpenCode, OpenClaw, Pi Coding Agent).

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
- **GPU telemetry without vendor CLIs.** The shared capability probe reads AMD
  VRAM from DRM sysfs and NVIDIA/WSL VRAM through the CUDA driver API via ctypes;
  it never invokes `nvidia-smi` or treats host RAM as VRAM. Unsupported or
  ambiguous device selection remains unknown. Set `LLAMA_BROKER_GPU_DEVICE` to
  a visible ordinal when more than one device is exposed; with
  `CUDA_VISIBLE_DEVICES`, ordinals refer to the restricted visible list.
- **Bounded early handoff.** When one selected GPU reports a sustained, monotonic
  free-memory rise across the configured sample window after an asynchronous
  stop, the next backend may start before the old process exits. The endpoint
  must be absent and unrelated broker units quiescent; the old unit is reaped
  before readiness/commit. This advisory heuristic does not prove target fit;
  normal resource admission still applies. Set `LLAMA_BROKER_HANDOFF_MIN_VRAM_RISE_MIB`
  and `LLAMA_BROKER_HANDOFF_WINDOW_SAMPLES` to tune it.
- **Lifecycle failure handling.** Successful transient requests retain the
  current restore behavior. Failed transitions stop/quarantine the failed
  target, clear the active alias, and do not automatically restore or restart
  the previous profile; structured `[broker-event]` JSON records the outcome.

## Canonical model and runtime store

Broker-served artifacts use the XDG roots below:

```text
~/.local/share/models/
~/.local/share/runtimes/catalog.json
~/.local/share/runtimes/<runtime-bundle-id>/
~/.cache/llama-broker/<family>/<model>/
```

The canonical overrides are `LLAMA_BROKER_MODELS_DIR`,
`LLAMA_BROKER_RUNTIMES_DIR`, `LLAMA_BROKER_CACHE_DIR`, and
`LLAMA_BROKER_RUNTIME_CATALOG`. Existing scripts retain `MODELS_DIR` as a
compatibility fallback. The superseded `$HOME/local/share/models` path is not used.

Schema-v1 manifests with inline `runtime_builds` remain readable. Schema-v2
manifests use `runtime_build` IDs and resolve executable, hash, bundle root,
launch environment, repository, and provenance metadata from the external
catalog. The manifest and catalog share a `deployment_revision`; mismatches
fail closed. Profile IDs, aliases, restore targets, and serving settings are
not rewritten by normal loading.

The reusable migration tool is non-destructive by default:

```bash
python3 tools/model_store_migrate.py plan --profiles ~/.config/llama-broker/profiles.json --output migration-plan.json
python3 tools/model_store_migrate.py stage --plan migration-plan.json
python3 tools/model_store_migrate.py verify --plan migration-plan.json
python3 tools/model_store_migrate.py apply --plan migration-plan.json --apply
python3 tools/model_store_migrate.py cleanup --plan migration-plan.json --cleanup --confirm
```

`apply` and `cleanup` are separate gates. Staging verifies hashes and loader
resolution without changing the manifest. Cleanup removes only exact paths
recorded in the verified plan, after paired backups, broker-coordinated live
smoke tests, and provenance checks succeed. Missing optional drafts or MoE
cache files degrade the affected profile with warnings; missing required
models, projectors, or runtimes fail closed.


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
