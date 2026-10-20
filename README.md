<!-- retrospective-release:start -->
## Retrospective version: 0.7.1

**Current workflow documentation** — source checkpoint 7e72f29b40b5 (2026-09-26).

Assigned retrospectively during release preparation; this is not evidence of a historical publication. [Cumulative release notes](CHANGELOG.md) describe this checkpoint and all earlier assigned versions.

The README below is historical text with portable examples. During a second preparation pass, private deployment snapshots were removed, machine-specific examples and migration defaults were sanitized, package versions were aligned, and release workflows were made explicitly manual. These are preparation changes, not evidence of historical package publication.
<!-- retrospective-release:end -->

# Llama-Broker

Llama-Broker gives local AI tools one OpenAI-compatible address for models on a
Linux machine. It manages starting the model server and switching between saved
model profiles. A profile records which model to run and its server settings.

Only one model server can be active at a time. Its configured slots determine
how many requests it can handle together, regardless of how many client apps
connect. A request can name a profile ID or alias; when that profile differs
from the active one, the broker waits for current responses to finish, including
streams, before switching. If a request omits the model, the active profile is
used.

## Install

You need Linux with systemd user services, Python 3.11+, `uv`, and an installed
`llama-server` with a model file.

```bash
uv tool install llama-broker
```

The broker finds existing runtimes and model files. It does not download or
install them.

## Start the broker

```bash
brokerctl setup
brokerctl integrate install
brokerctl profile list
brokerctl load <profile-id>
brokerctl status
```

Setup finds runtimes and model files already on the machine, then asks which
ones to add as profiles. It does not download models or runtimes.
`brokerctl integrate install` installs the broker's systemd user services.
`load` starts the chosen profile, and `status` confirms what is active. Use
`brokerctl setup --check` to scan without making changes. Run
`brokerctl integrate --dry-run` to preview integration changes. Profiles are
saved in `~/.config/llama-broker/profiles.json` by default; set
`LLAMA_BROKER_PROFILES` to use another file.

## Model and runtime files

These are the default locations for broker-managed files. Setup can also find
models and runtimes already installed elsewhere on the machine.

| Files | Default location | Override |
| --- | --- | --- |
| Model files | `~/.local/share/models/` | `LLAMA_BROKER_MODELS_DIR` |
| Runtime bundles | `~/.local/share/runtimes/<id>/` | `LLAMA_BROKER_RUNTIMES_DIR` |
| Runtime catalog | `~/.local/share/runtimes/catalog.json` | `LLAMA_BROKER_RUNTIME_CATALOG` |

The defaults follow `XDG_DATA_HOME` when it is set. The runtime catalog is in
the runtime directory unless `LLAMA_BROKER_RUNTIME_CATALOG` points elsewhere.

To configure Hermes to use a profile:

```bash
brokerctl integrate configure hermes --model <profile-id>
```

For other supported clients, see `brokerctl integrate configure --help`.

## Connect a client

The local API address is `http://127.0.0.1:11434/v1`. Requests need an API key
as a bearer token. The default key file is
`~/.config/llama-broker/api-keys`; `brokerctl integrate configure` can save the
address and key in supported clients. In an OpenAI-compatible client, set its
base URL to that address, use the broker API key, and select a profile ID or
alias as the model. Multiple clients can share the same address and key.

| Route | Purpose |
| --- | --- |
| `GET /v1/models` | List healthy model profiles clients can use. |
| `GET /v1/models/{model}` | Show details for one profile. |
| `POST /v1/chat/completions` | Send a chat request; supports streaming. |
| `POST /v1/completions` | Send a text completion request. |
| `POST /v1/audio/transcriptions` | Transcribe an audio file. |
| `POST /v1/systemone` | Ask the optional system 1 decision model (d1) a question. |

With the optional [system 1 decision model](docs/system1.md) enabled, clients
can also request the `auto` model and let Liquid AI's d1 pick the profile.

For basic monitoring, `GET /health` and `GET /healthz` are available without an
API key.

## Manage models and services

| Command | Purpose |
| --- | --- |
| `brokerctl status` | Show the active profile and broker health. |
| `brokerctl profile list` | Show saved profiles and their IDs. |
| `brokerctl model list` | List model names that clients can request. |
| `brokerctl runtime list` | List registered runtimes. |
| `brokerctl load <profile-id>` | Start or switch to a profile. |
| `brokerctl restart` / `brokerctl unload` | Restart or stop the active model server. |
| `brokerctl profile edit [profile-id]` | Edit a saved profile. |
| `brokerctl profile render <profile-id>` | Preview a service definition. |
| `brokerctl verify` | Check model and runtime files against their saved hashes. |

Use `brokerctl` to change profiles so active requests finish before a switch.
Do not start or stop profile services directly with `systemctl`.
Inspect the broker service with `systemctl --user status llama-broker`; view its
logs with `journalctl --user -u llama-broker`.

If a profile is missing or will not start, run `brokerctl setup --check` to
rescan installed files and `brokerctl verify` to check the files recorded in
the profile. Use the service logs to see startup errors.

### API keys

The broker stores configuration under `~/.config/llama-broker/` by default.
`security.json` stores certificate and key file paths; API keys are kept in the
private `api-keys` file. Rotate keys with:

```bash
brokerctl security rotate
```

Normal rotation keeps existing keys valid so clients can be updated. Use
`--replace` to remove the old keys immediately, or revoke one key with
`brokerctl security revoke <key>`.

## Security and supported runtimes

The API listens on this machine only by default. LAN access requires TLS and an
API key; plain HTTP on a LAN address is rejected. Create certificates and key
files with `brokerctl security init --ip <lan-ip> --hostname <lan-name>`, then
set `LLAMA_BROKER_HOST` to the LAN address.

llama.cpp is the supported runtime. vLLM is experimental.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and test instructions.
See [SETUP_GUIDE.md](SETUP_GUIDE.md) for manual setup and service details.
The project is licensed under [MIT](LICENSE).
