# System 1 decision model (d1)

A "system 1" model makes fast judgments instead of generating text: it reads a
state plus named, typed questions and returns a probability per answer in one
forward pass. Llama-Broker supports Liquid AI's
[d1-3B](https://huggingface.co/LiquidAI/d1-3B-GGUF), which `llama-server`
serves on `POST /v1/systemone`.

The usual way to turn it on is a **resident profile** in `profiles.json`: a
profile with `"role": "resident"` and `"capabilities": ["decision"]`. It has the
same fields, pinned hashes and `brokerctl verify`/`profile edit` tooling as
any other llama.cpp profile, but the broker never switches to it, leases it or
unloads it: it stays loaded beside whichever profile is active. Chat or
completion requests that name it get a 400 pointing at `/v1/systemone`, and
client integrations leave it out of their model lists.

```json
{
  "id": "d1-3b",
  "alias": "d1-3B",
  "aliases": ["d1-3B", "d1"],
  "display_name": "Liquid d1 3B (Q4_K_M)",
  "role": "resident",
  "capabilities": ["decision"],
  "model_path": "~/.local/share/models/LiquidAI/d1-3B-GGUF/d1-3B-Q4_K_M.gguf",
  "context_size": 8192,
  "parallel": 1,
  "gpu_layers": "all",
  "...": "server_binary, hashes, cache/batch fields and safety as in other profiles"
}
```

One resident decision profile turns system 1 on by itself: `/v1/systemone`
works, but automatic routing stays off until `system1.json` sets
`"routing": {"enabled": true}` (write `routing.descriptions` first; with bare
display names d1 tends to pick the smallest model, which would swap the active
one out). With a resident profile, a `system1.json` that omits `enabled` keeps
system 1 on. `~/.config/llama-broker/system1.json` (override the path with
`LLAMA_BROKER_SYSTEM1_CONFIG`) is then optional: use it to tune `routing`,
change `url`, name the resident with `"profile"` when there are several, or
switch system 1 off with `"enabled": false`. Without a resident profile,
system 1 stays off unless `system1.json` has `"enabled": true` and the inline
`server_binary`/`model_path` fields from
[`examples/system1.example.json`](examples/system1.example.json).

Resident is a stop-gap: a later redesign of `persistent`/`transient` into
multi-model residency is expected to fold it into `persistent`.

## How it runs

d1 runs as a small, always-resident side backend in its own systemd user unit,
`llama-broker-system1.service`, on its own loopback port (`10001` by default).
It is outside the profile lifecycle: profile switches, drains and restarts
never touch it, and the controller does not count it as a profile backend. The
Q4_K_M file is about 1.7 GB, so expect it to hold roughly 2 GB of VRAM for as
long as it runs; profile admission sees that through normal free-VRAM checks.
Set the resident profile's `gpu_layers` to `0` to keep it on the CPU instead.

On startup the broker writes the unit and starts it (a running unit is left
alone). On broker shutdown it keeps running, like profile backends. A missing
model file or failed start is logged and the broker keeps serving profiles.
Set `"managed": false` and point `url` at a server you run yourself to skip
the unit entirely.

The d1 model needs a `llama-server` build that includes `/v1/systemone`
support. Download it with, for example:

```sh
hf download LiquidAI/d1-3B-GGUF --include '*Q4_K_M*' \
  --local-dir ~/.local/share/models/LiquidAI/d1-3B-GGUF
```

then add the resident profile above (or copy
[`examples/system1.example.json`](examples/system1.example.json) to
`~/.config/llama-broker/system1.json` and fix its two paths). Restart the
broker; `GET /health` shows a `system1` section with `ready: true` once d1 is
serving.

## Asking d1 directly

`POST /v1/systemone` on the broker port forwards the body unchanged to d1 and
returns its answer. It uses the normal broker API key and never waits on, or
switches, the main profile.

```sh
curl -s http://127.0.0.1:11434/v1/systemone \
  -H "Authorization: Bearer $(head -n1 ~/.config/llama-broker/api-keys)" \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "I was charged twice this month, please refund one of them.",
    "questions": {
      "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
      "team": {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "Payments", "technical": "Bugs", "fraud": "Abuse"}}
    }
  }'
```

## Automatic routing

When `routing.enabled` is true, a request whose `model` is one of
`routing.aliases` (`auto` by default) is routed by d1. The broker asks one
`choice` question whose labels are the candidate profile IDs, then forwards
the request to the winner exactly as if the client had named it. The response
carries `X-Llama-Broker-Profile` and `X-Llama-Broker-System1-Route` (the
reason) headers, and each decision is logged as a `system1_route` event.

- **Candidates** are every profile whose files verify (or the active one),
  narrowed to `routing.candidates` when set. Requests with images only consider
  profiles with the `vision` capability. One candidate skips d1.
- **What d1 sees** is the system prompt (trimmed), the latest user turn, a note
  about tools or images, capped at `max_state_chars`. Each profile is described
  by its display name, capabilities and context size, or by
  `routing.descriptions[profile-id]`, which is usually worth writing.
- **Swaps are expensive**, so a different profile must beat the active one by
  `switch_margin` before d1 can cause a switch.
- **Fallback**: if d1 is down, answers below `min_confidence`, or names no
  candidate, the request goes to the active profile (`"fallback": "active"`), a
  fixed profile ID, or fails with 503 (`"fallback": "error"`).

A profile alias always wins over a routing alias with the same name.
