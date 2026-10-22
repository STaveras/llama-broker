# Llama-Broker

Llama-Broker gives local AI tools one OpenAI-compatible address for models on a
Linux machine. This checkout is **0.8.0.dev0**. It manages saved model
profiles, validates them before use, and switches one generation backend at a
time after active requests drain.

<!-- retrospective-release:start -->
## Retrospective version: 0.6.16

**Profile editor row and dirty-state fixes** — source checkpoint c34c5e8fba6f (2026-08-27).

Assigned retrospectively during release preparation; this is not evidence of a historical publication. [Cumulative release notes](CHANGELOG.md) describe this checkpoint and all earlier assigned versions.

The README below is historical text with portable examples. During a second preparation pass, private deployment snapshots were removed, machine-specific examples and migration defaults were sanitized, package versions were aligned, and release workflows were made explicitly manual. These are preparation changes, not evidence of historical package publication.
<!-- retrospective-release:end -->

The current development snapshot is not a stable or published release. See
[docs/RELEASING.md](docs/RELEASING.md) for release qualification.

## Install from a development checkout

You need Linux with systemd user services, Python 3.11+, `uv`, an installed
runtime, and model files. This checkout workflow uses the package's actual
console entry point:

```bash
uv venv
source .venv/bin/activate
uv pip install --editable .
brokerctl setup --check
```

The broker does not download or install runtimes or models. `setup --check` is
read-only. To create profiles and install the broker's user services, run:

```bash
brokerctl setup
brokerctl integrate install
brokerctl profile list
```

Profiles are saved in `~/.config/llama-broker/profiles.json` by default;
`LLAMA_BROKER_PROFILES` overrides that path. Use
`brokerctl integrate --dry-run` to review integration changes before applying
them.

## Start and switch the broker

```bash
brokerctl load <profile-id>
brokerctl operation wait <request-id>
brokerctl status
```

`load` is asynchronous: its acknowledgement means the operation was accepted,
not that generation is ready. Wait for the terminal result before sending
inference or cleaning up. A profile switch drains active responses, including
streams. Do not start or stop profile units directly with `systemctl`.

## Model and runtime files

Setup can discover files outside these defaults. `brokerctl model list` is a
read-only artifact listing; it does not create profile IDs or select a model.
Use `brokerctl profile list` for canonical client model names and load targets.

| Files | Default location | Override |
| --- | --- | --- |
| Model files | `~/.local/share/models/` | `LLAMA_BROKER_MODELS_DIR` |
| Runtime bundles | `~/.local/share/runtimes/<id>/` | `LLAMA_BROKER_RUNTIMES_DIR` |
| Runtime catalog | `~/.local/share/runtimes/catalog.json` | `LLAMA_BROKER_RUNTIME_CATALOG` |

`model list --json` reports schema-2 artifact bundles, including incomplete or
missing files. Artifact status does not assert model integrity, tokenizer
validity, runtime compatibility, or serving readiness.

## Connect a client

The local API is `http://127.0.0.1:11434/v1`. Requests use an API key as a
bearer token. The default key file is
`~/.config/llama-broker/api-keys`. In an OpenAI-compatible client, set the base
URL, use the broker key, and select a profile ID or alias as the model.

| Route | Purpose |
| --- | --- |
| `GET /v1/models` | List healthy callable generation profiles. Resident decision profiles are intentionally hidden. |
| `GET /v1/models/{model}` | Show verified profile metadata. |
| `POST /v1/chat/completions` | Chat generation; `stream: true` supports SSE. |
| `POST /v1/completions` | Text completion; supports SSE. |
| `POST /v1/audio/transcriptions` | Bounded authenticated multipart transcription upload. |
| `POST /v1/decisions` | OpenAI-compatible Decisions API backed by the separate d1 system 1 backend. |
| `POST /v1/systemone` | Ask d1 directly without switching the generation backend. |

`GET /health` and `GET /healthz` are minimal unauthenticated probes. Every
response has an `X-Request-ID`; errors use an OpenAI-compatible `error` object.
Streaming failures are terminal SSE errors, not a synthetic `[DONE]`.

### System 1 and automatic routing

System 1 is a separate optional d1 backend, normally configured as a resident
profile. It stays beside the active generation profile and is not loaded,
unloaded, or advertised by `GET /v1/models`. It remains visible to
`brokerctl profile list` and can be named for `/v1/decisions` or
`/v1/systemone`. See [docs/system1.md](docs/system1.md).

Automatic routing is not enabled merely by configuring d1. Set
`system1.json`'s `routing.enabled` to `true` to allow the `auto` alias to ask
d1 to choose among generation profiles. Without that explicit opt-in, use a
profile ID or alias, or omit the model to use the active generation profile.

## Manage models and services

| Command | Purpose |
| --- | --- |
| `brokerctl status` | Show broker and active-backend state. |
| `brokerctl profile list` | Show saved profiles, including resident profiles. |
| `brokerctl model list` | List on-disk artifacts, not loadable profile IDs. |
| `brokerctl runtime list` | List registered runtimes and selected profiles. |
| `brokerctl load <profile-id>` | Queue a start or profile switch. |
| `brokerctl operation wait <request-id>` | Wait for an asynchronous lifecycle result. |
| `brokerctl restart` / `brokerctl unload` | Restart or stop the active generation backend. |
| `brokerctl profile edit [profile-id]` | Edit a saved profile. |
| `brokerctl verify` | Check pinned runtime, model, and projector hashes. |
| `brokerctl security rotate` / `revoke` | Rotate or revoke API keys. |

For a missing or failing profile, run `brokerctl setup --check`, then
`brokerctl verify` and inspect `journalctl --user -u llama-broker`.

## Security and runtimes

The default API binding is loopback. LAN access requires TLS and an API key;
plaintext non-loopback binding is rejected. Provision local material with:

```bash
brokerctl security init --ip <lan-ip> --hostname <lan-name>
```

Never put real API keys, private paths, or deployment-specific hashes in
README examples. The broker keeps request bodies out of normal operational logs,
uses bounded request/queue/backend deadlines, and validates profile hashes and
resource limits fail-closed. The detailed reliability and context contracts
are documented in [CONTEXT.md](CONTEXT.md) and [SETUP_GUIDE.md](SETUP_GUIDE.md).

llama.cpp is the qualified first-release runtime. vLLM and Strata are
experimental and require explicit selection. Basalt is also experimental; its
[acceptance contract](docs/basalt-acceptance.md) describes operator checks and
does not claim deployment, hardware, or runtime qualification.

## Development

```bash
uv run --extra dev pytest -q --ignore=tests/test_local_production_manifest.py
uv run python tools/check_public_tree.py
uv run python tools/check_release.py --development
```

The suite is offline and must not depend on a machine's private
`profiles.json`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development
rules, [SETUP_GUIDE.md](SETUP_GUIDE.md) for manual service details, and
[docs/RELEASING.md](docs/RELEASING.md) for release gates. The project is
licensed under [MIT](LICENSE).