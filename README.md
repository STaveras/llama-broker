# Llama-Broker

Run your local models behind one OpenAI-compatible API. Llama-Broker manages
model profiles and switches between them without changing your client settings.
It waits for active responses to finish before loading another model.

[Download v0.8.0](https://github.com/STaveras/llama-broker/releases/tag/v0.8.0)
· [Setup guide](SETUP_GUIDE.md) · [Changelog](CHANGELOG.md)

## Features

- Chat and text completions with streaming support.
- Saved profiles for different models and runtime settings.
- Model switching from the command line or through client requests.
- Profile updates without restarting the broker.
- API-key authentication and TLS for access from other devices.
- Optional [System 1 routing](docs/system1.md) to help choose a model for a request.

## Requirements

- Linux with systemd user services.
- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
- An installed model runtime and compatible model files.

Start with llama.cpp. Other runtime integrations are experimental.
Llama-Broker uses your existing runtimes and models; it does not download them.

## Install

```bash
git clone --branch v0.8.0 --depth 1 https://github.com/STaveras/llama-broker.git
cd llama-broker
uv venv
source .venv/bin/activate
uv pip install .
```

Wheel and source archives are also available on the
[release page](https://github.com/STaveras/llama-broker/releases/tag/v0.8.0).

## Set up your models

Run setup to discover your runtimes and model files and save model profiles.
Then install the background services and list your profiles:

```bash
brokerctl setup
brokerctl integrate install
brokerctl profile list
```

See the [setup guide](SETUP_GUIDE.md) for configuration options.

## Load a model

Choose a profile ID from the list:

```bash
brokerctl load <profile-id>
```

Loading runs in the background. The command returns a request ID; use it to
wait until loading finishes:

```bash
brokerctl operation wait <request-id>
brokerctl status
```

To switch models, load another profile the same way. Use `brokerctl unload`
when you want to stop the active model.

## Connect your client

In your OpenAI-compatible application, set:

- **Base URL:** `http://127.0.0.1:11434/v1`
- **API key:** your broker key from `~/.config/llama-broker/api-keys`
- **Model:** a profile ID or alias from `brokerctl profile list`

The default address accepts connections from the same machine. To connect from
another device, configure HTTPS and use your broker's network address instead.
See the [setup guide](SETUP_GUIDE.md) for network configuration.

## Help and documentation

- [Setup guide](SETUP_GUIDE.md): installation, configuration, and troubleshooting.
- [System 1](docs/system1.md): optional decision models and automatic routing.
- [Report a bug](https://github.com/STaveras/llama-broker/issues).
- [Contributing](CONTRIBUTING.md).

## License

[MIT](LICENSE)
