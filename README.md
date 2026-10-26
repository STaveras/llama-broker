# Llama-Broker

Llama-Broker gives local AI tools one OpenAI-compatible address for models on a
Linux machine. It manages saved model profiles, validates them before use, and
switches one generation backend at a time after active requests drain.

**Latest release: [v0.8.0](https://github.com/STaveras/llama-broker/releases/tag/v0.8.0)**
· [Changelog](CHANGELOG.md) · [Setup guide](SETUP_GUIDE.md)
· [Contributing](CONTRIBUTING.md) · [MIT license](LICENSE)

## What it does

- **One client endpoint:** OpenAI-compatible chat and text completions, including
  streaming, with canonical profile IDs and aliases.
- **Controlled model switching:** ordered asynchronous operations, bounded
  inference admission, and draining of active responses before switching.
- **Live profile updates:** validated manifest reloads without restarting the
  generation backend; failed swaps preserve the previous registry.
- **Artifact and runtime management:** inspect model files, register runtimes,
  and verify pinned hashes without confusing artifacts with loadable profiles.
- **Optional System 1 decisions:** a separate resident decision backend and
  explicitly opt-in automatic routing.
- **Authenticated access:** bearer-token API keys, loopback by default, and TLS
  required for non-loopback binding.

## Install v0.8.0

You need Linux with systemd user services, Python 3.11+, [`uv`](https://docs.astral.sh/uv/),
an installed runtime, and model files. Start from the tagged source release:

```bash
git clone --branch v0.8.0 --depth 1 https://github.com/STaveras/llama-broker.git
cd llama-broker
uv venv
source .venv/bin/activate
uv pip install .
brokerctl --version
brokerctl setup --check
```

Prebuilt wheel and source archives are also available on the
[release page](https://github.com/STaveras/llama-broker/releases/tag/v0.8.0).
This release is distributed through GitHub; these instructions do not assume
that a package has been published to PyPI.

The broker does not download or install runtimes or models. `setup --check` is
read-only. To create profiles and install the broker's user services, run:

```bash
brokerctl setup
brokerctl integrate install
brokerctl profile list
```

Profiles are saved in `~/.config/llama-broker/profiles.json` by default;
`LLAMA_BROKER_PROFILES` overrides that path. Once setup has saved at least one
profile, use `brokerctl integrate --dry-run` to review integration changes
before applying them. With no profiles yet, integration reports that setup is
required; `setup --check` alone does not create them.

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

### Live profile reload

By default the broker watches `~/.config/llama-broker/profiles.json` and, when
its content changes, re-validates and hot-swaps the profile catalog — without
restarting the generation backend, so in-flight requests keep serving on their
existing profile snapshot. This makes "I added or edited a profile" a live
change; you no longer need `systemctl --user restart llama-broker` for it.
The reload reuses the same non-disruptive path as the manual
`profile add/set/remove` mutations, so it inherits its guards: a reload that
would remove the active profile is refused, and a failed swap rolls back to
the previous registry.

The watcher polls every 2 seconds using a content fingerprint rather than a
file timestamp, so atomic rewrites are handled. Malformed or mid-write files
are logged and retried without aborting the daemon. Failed or cancelled reload
operations are also retried; accepting a queued reload does not mark the new
manifest as applied.

To turn the watcher off (manual reload only), set
`LLAMA_BROKER_MANIFEST_RELOAD_INTERVAL=0` in the broker service environment.
Catalog reloads do not reconfigure a running backend process. Apply backend
changes through a broker-managed restart or profile switch so request draining
and lifecycle safeguards remain in effect.

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

For changes beyond the stable release, use the `development` branch and an
editable installation:

```bash
git clone --branch development https://github.com/STaveras/llama-broker.git llama-broker-dev
cd llama-broker-dev
uv venv
source .venv/bin/activate
uv pip install --editable '.[dev]'
uv run --extra dev pytest -q --ignore=tests/test_local_production_manifest.py
uv run python tools/check_public_tree.py
uv run python tools/check_release.py --development
```

The suite is offline and must not depend on a machine's private
`profiles.json`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development
rules, [SETUP_GUIDE.md](SETUP_GUIDE.md) for manual service details, and
[docs/RELEASING.md](docs/RELEASING.md) for release gates. The project is
licensed under [MIT](LICENSE).