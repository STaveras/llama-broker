# Llama-Broker

Llama-Broker puts one OpenAI-compatible address in front of your local models.
Your tools keep the same URL. The broker handles starting a model, switching to
another, and letting a response finish before anything gets unloaded.

If you already have the model files and a runtime command you're happy with,
that's the part this project is for. Save the setup as a profile and use it
from your clients without starting and stopping processes by hand.

[Download v0.8.0](https://github.com/STaveras/llama-broker/releases/tag/v0.8.0)
· [Setup guide](SETUP_GUIDE.md) · [Changelog](CHANGELOG.md)

## Why a broker?

This started with two setups for the same model: one with more room for text
and coding, another with vision enabled and a smaller context. Switching meant
unloading one and loading the other. Keeping the heavier setup loaded all the
time wasn't a good trade.

The useful part was being able to make that switch without changing every tool
that talked to the model. Profiles are how the broker remembers each setup:
model files, runtime settings, and a name your client can ask for.

## What it handles

- Chat and text completions, including streaming responses.
- Loading and switching between saved model profiles.
- Picking up profile edits without restarting the broker.
- API keys for client access and HTTPS for connections from other devices.

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
- [Status output](docs/status.md): model availability, current activity, and System 1 statistics.
- [System 1](docs/system1.md): optional decision models and automatic routing.
- [Report a bug](https://github.com/STaveras/llama-broker/issues).
- [Contributing](CONTRIBUTING.md).

## License

[MIT](LICENSE)
