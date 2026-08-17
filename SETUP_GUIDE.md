# Llama-Broker Setup and Operations Guide

This guide describes the active local deployment as of 2026-07-30. It is
specific to the reviewed files in `<repo>` and the
Phase 2 KVarN runtime used by the primary Example 9B v3 profile.
The historical example-first sections remain available for generic rollback support;
the active manifest and current operational guidance take precedence.

## Canonical paths and identities

| Purpose | Path or identity |
| --- | --- |
| Broker source | `<repo>` |
| Broker branch | `integration/llama-broker-256k-safety` |
| Execution manifest | `<repo>/profiles.json` |
| Durable switch state | `<repo>/state.json` |
| Broker service | `$HOME/.config/systemd/user/llama-broker.service` |
| Profile units | `$HOME/.config/systemd/user/llama-broker-profile-*.service` |
| Broker API | `https://<host>:11434/v1` |
| Native backend/UI | `https://<host>:10000/` |
| Native API keys | `$HOME/.config/llama-broker/api-keys` |
| Security config | `$HOME/.config/llama-broker/security.json` |
| Active primary runtime | `<llama.cpp-build>/<fork-dir>/build-ui-scroll/bin/llama-server` |
| Runtime source | `<llama.cpp-build>/<fork-dir>` |
| Runtime branch/commit | `<fork-branch>` / `<fork-commit>` |
| Runtime SHA-256 | `(hash computed at profile add)` |
| Stable legacy rollback link | `<llama.cpp-build>/hermes-runtime/bin/llama-server` (still points to Phase 1) |

The stable legacy link remains preserved for rollback and is not the active
primary profile executable. Do not substitute it for the Phase 2 binary merely
because it can open a model; use the broker manifest for model-specific paths.

## Service architecture

`llama-broker.service` is the only enabled control-plane entry point. The
provided unit listens with TLS on 11434 and owns one native TLS
`llama-server` on 10000; both ports conflict with any legacy standalone
server using the same port.

`brokerctl.py integrate install` renders one stable, disabled unit for every allowlisted
profile. The broker's transactional controller starts exactly one profile unit
on native port 10000. A profile unit has `Restart=no`; a CUDA or model-load
failure therefore remains visible. The broker daemon itself uses
`Restart=on-failure`, but restarting it does not authorize a different model.
Its HTTP worker pool is bounded at two, and completion requests are defaulted
or limited to 16,384 output tokens.

The broker persists this state atomically:

```json
{
  "generation": 1,
  "phase": "active",
  "profile": "example-9b-v3"
}
```

During a change, `phase` becomes `switching` and includes the target. On broker
startup, `reconcile()` detects an interrupted switch, stops the target, and
attempts to restore the prior profile before serving traffic.

The broker does not create a second inference API-key authority. OpenAI-compatible
`/v1/*` routes, including `/v1/models`, require the standard
`Authorization: Bearer $API_KEY` header. Broker administration (`/admin/*`)
and the compatibility `/api/tags` route are loopback-only. Health endpoints
remain minimal and unauthenticated.

LAN mode requires both broker and backend certificate/key pairs plus the native
API-key file. Startup rejects a non-loopback plaintext binding. The backend
certificate must cover `127.0.0.1` for the broker's local HTTPS readiness and
the LAN name/IP used to open the built-in UI.

## Reviewed profile manifest

Only `profiles.json` authorizes execution. Expected artifact identities are:

| Artifact | SHA-256 |
| --- | --- |
| Primary Phase 2 `llama-server` | (hash computed at profile add) |
| Example 27B Q1_0 | (hash computed at profile add) |
| Example 9B stock Q6_K | `(hash computed at profile add)` |
| Example 9B v2 Q6_K | (hash computed at profile add) |
| Example 9B v3 Q6_K | (hash computed at profile add) |
| Example F16 projector | (hash computed at profile add) |

The corresponding pinned files are:

```text
<models-dir>/example-27b/Example-27B-Q1_0.gguf
$HOME/.local/share/models/example-9b-stock/Example-9B-Stock-Q6_K.gguf
$HOME/.local/share/models/example-9b-v2/Example-9B-v2-Q6_K.gguf
$HOME/.local/share/models/mmproj-F16.gguf
```

The effective profile settings are:

| ID | Alias | Total context / slots | Placement | Model and special behavior |
| --- | --- | --- | --- | --- |
| `example-9b-v3` | `example-9b-v3` | 262,144 / 1 | `-ngl all` | Example 9B v3 Q6_K; KVarN4; active primary |
| `example-27b` | `example-27b` | 262,144 / 1 | `-ngl auto` | Example 27B Q1_0; secondary rollback |
| `example-9b-stock` | `example-9b-stock` | 262,144 / 2 | `-ngl all` | stock Example 9B text; 131,072 tokens per slot |
| `example-9b` | `example-9b` | 262,144 / 2 | `-ngl all` | Example 9B v2 text; 131,072 tokens per slot |
| `example-9b-vision` | `example-9b-vision` | 32,768 / 1 | `-ngl auto` | stock Qwen plus F16 projector; transient; restores v3 |
| `example-9b-mtp` | `example-9b-mtp` | 32,768 / 1 | `-ngl all` | embedded MTP profile |

Legacy profiles use their pinned settings. The active v3 profile uses:

```text
-b 64 -ub 64 -ctk kvarn4 -ctv kvarn4 --flash-attn on
--no-context-shift --no-warmup
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_GRAPH_OPT=0
GGML_CUDA_PDL=0 LLAMA_GRAPH_REUSE_DISABLE=1
LLAMA_KVARN_MULTI_STREAM=1 Restart=no
```

`GGML_CUDA_DISABLE_GRAPHS=1` disables CUDA graph capture; the separate graph
optimizer, PDL, and llama graph-reuse controls are also disabled for the active
KVarN qualification profile. This is a conservative stability baseline; graph A/B testing is not a
production gate yet.

### Full-context semantics

Example 27B uses `-c 262144 -np 1`. `-c` is the total physical server context and
`-np` is the slot count, so one request can use the fixed 262,144-token slot
subject to chat-template and output headroom. A two-slot configuration would
halve the per-slot window and is not the canonical profile.

The stock and Example 9B text profiles use two slots in a 262,144-token server
context, yielding 131,072 tokens per slot. Vision and MTP retain their smaller
single-slot windows. `/v1/models` reports these profile-specific values.

Qwen 3.5 MRoPE KV shifting is unsupported. `--no-context-shift` is therefore a
required invariant; the configured contexts are fixed windows rather than
rolling histories.

### Native device placement

Production profiles omit every fit-related flag. Omission means no `--fit`,
`--fit-target`, `--fit-ctx`, `--cache-ram`, or `--ctx-checkpoints` is emitted.
Example 27B and vision use `-ngl auto`; text profiles use `-ngl all`. This
restores native llama-server placement after fit planning caused
performance regressions in some workloads. Inspect the backend log to verify
actual placement and acceleration for each qualified workload.

### Declarative resource budgets and lifecycle gates

Each profile declares two safety layers in `profiles.json`:

- `safety.service` renders `Type=exec`, `MemoryMax`, `MemorySwapMax`,
  `TasksMax`, `KillSignal=SIGINT`, and `TimeoutStopSec` into the generated user
  unit. `Restart=no` remains mandatory.
- `safety.admission` records capacity assumptions for explicit qualification
  experiments; production switching does not enforce these telemetry values.

Production has no `nvidia-smi` dependency and no fixed free-VRAM gate.
llama-server owns device placement, while systemd enforces the declared RAM,
swap, and task ceilings. The controller keeps optional injectable resource
admission for controlled tests, but the production controller configures none
and logs `resource_admission=not_configured`.

Lifecycle safety remains active: the old unit must be proven inactive with no
processes, the private endpoint must be absent, the new unit must own the exact
configured alias, and cancellation, drain, stop, rollback, and restore paths are
bounded and state-reconciled.

### Interactive profile editing

`profiles.json` remains the single execution authority, but it no longer needs
to be edited by hand for routine changes. Use the field-oriented editor:

```bash
cd <repo>
python3 brokerctl.py profiles edit
python3 brokerctl.py profiles edit example-9b-v3
```

The editor selects individual fields instead of walking through a fixed prompt
sequence. Enter opens a text box, dropdown, or checklist; Tab reveals runtime
and model paths, hashes, fit controls, safety budgets, environment, and extra
arguments. Save runs complete `ProfileRegistry` validation, regenerates units
from the manifest, and restarts the control plane only when the active profile
changed. Explicit `profiles edit MODEL --field value` flags remain available
for non-interactive callers.

Discovery is authoritative only as a source of candidates. Import/adoption
requires a local GGUF and a broker-owned `llama-server` path; the profile editor
hashes both and never creates a passthrough to an unrelated OpenAI endpoint.
Manual GGUF imports follow the same path. `profiles add` remains the compatibility
entry point for candidate inspection/import; there is no separate profile
configuration command.

Apply is transactional: it validates and atomically publishes the manifest,
regenerates profile units, and restarts `llama-broker.service` only if the
active profile changed. The broker control plane performs the drain and
reconciliation. A failed install or restart restores the previous manifest and
units, then attempts to restore the previous broker state; a failure is
reported instead of leaving an unvalidated profile live.

## Install or update the deployment

### Security material and direct native UI

Create the local CA, server certificates, API key, and path-only security
configuration before enabling the LAN listener:

```bash
cd <repo>
python3 brokerctl.py security init \
  --ip <lan-ip> \
  --hostname <hostname>
```

`security init` is safe to rerun when the generated material is complete; use
`--force` only for an intentional certificate replacement. It never prints the
API-key value. The generated `lan-ca.crt` must be imported into the Windows
browser trust store and installed into WSL/Linux trust for local HTTPS tools.

The resulting path-only security configuration is:

```json
{
  "api_key_file": "$HOME/.config/llama-broker/api-keys",
  "broker_tls_cert_file": "$HOME/.config/llama-broker/broker.crt",
  "broker_tls_key_file": "$HOME/.config/llama-broker/broker.key",
  "broker_tls_ca_file": "$HOME/.config/llama-broker/lan-ca.crt",
  "backend_tls_cert_file": "$HOME/.config/llama-broker/llama-server.crt",
  "backend_tls_key_file": "$HOME/.config/llama-broker/llama-server.key",
  "backend_tls_ca_file": "$HOME/.config/llama-broker/lan-ca.crt"
}
```

`brokerctl setup` is the first-time discovery wizard. It scans for existing
runtimes and model artifacts, reports qualified candidates, and creates profiles
only after explicit selection. It never downloads runtimes, copies models, or
loads a backend implicitly. Use `--check` or `--dry-run` for non-mutating
reports. After reviewing the resulting profile manifest, run
`brokerctl integrate install` to install broker-owned units and
`brokerctl integrate configure` to refresh coding-tool providers; use
`--set-default` only with the latter for an explicit model/provider selection.
Certificate private keys and the API-key file must be mode 600. Open the active
llama-server UI at `https://<lan-host>:10000/`; its API calls require the native
key.

Use the drain-aware credential lifecycle commands for rotation:

```bash
python3 brokerctl.py security rotate
python3 brokerctl.py security rotate --replace
python3 brokerctl.py security revoke '<old-key>'
```

Normal rotation retains the old key until clients have been updated and then
restarts the active profile once. `--replace` removes the overlap. The CLI
refreshes broker-owned provider credentials without changing a tool's active
endpoint or selected model; use `brokerctl integrate configure --set-default`
only for an explicit selection.

Run the offline checks and content verification first:

```bash
cd <repo>

./test-scripts.sh
python3 brokerctl.py verify
python3 brokerctl.py model list
python3 brokerctl.py render example-9b-v3 | less
```

Then install the profile units and start the broker:

```bash
systemctl --user disable --now llama-server.service
python3 brokerctl.py setup --verify
systemctl --user enable --now llama-broker.service
python3 brokerctl.py switch example-9b-v3
```

Use `python3 brokerctl.py setup --set-default` only when the coding tools
should explicitly select the manifest default through Llama-Broker.

If a profile is already running while its unit is regenerated, the existing
process retains its old command. Apply a reviewed manifest update by draining
and restarting it:

```bash
python3 brokerctl.py stop
python3 brokerctl.py switch example-9b-v3
```

Never enable the individual profile units. The controller owns them.

## Administrative commands

```bash
cd <repo>

python3 brokerctl.py model list
python3 brokerctl.py status
python3 brokerctl.py verify
python3 brokerctl.py verify example-9b-vision
python3 brokerctl.py render example-9b-v3
python3 brokerctl.py switch example-9b
python3 brokerctl.py switch example-9b
python3 brokerctl.py switch example-9b-v3
python3 brokerctl.py stop
```

When the broker is online, `switch` and `stop` use its loopback admin endpoint
and wait for all active leases. When it is offline, the same CLI takes the
cross-process lock and operates the controller directly.

Do not call the `/admin/*` endpoints from application code and do not use
`systemctl restart` on a profile as a switching mechanism.

## Request leases and switching

The coordinator permits concurrent requests only while they target the same
profile. A cross-profile request or administrative switch waits until all
active requests, including complete SSE streams, have released their leases.
The controller then:

1. records the target and `switching` phase;
2. stops the prior profile unit;
3. starts the target profile unit;
4. probes private `/v1/models` for the target's exact alias;
5. records the new active generation only after readiness succeeds.

On failure it stops the target and attempts to restore the prior profile. A
failed target with a successful restoration is reported as an error rather
than being disguised as a successful switch. A failed restoration records
`rollback_failed` and requires operator attention.

Unknown explicit model aliases fail closed. Only an omitted `model` field uses
the manifest default.

## Hermes configuration

Hermes should use provider `llama-broker` and public port 11434 for every local
profile. The canonical context declarations are:

```yaml
model:
  default: example-9b-v3
  provider: llama-broker
  base_url: https://127.0.0.1:11434/v1
  max_tokens: 16384

auxiliary:
  vision:
    provider: llama-broker
    model: example-9b-vision
    base_url: https://127.0.0.1:11434/v1
    timeout: 300

context:
  engine: local-safe-compressor
  max_threshold_tokens: 192000
  safe_models:
    - example-9b-v3

providers:
  llama-broker:
    base_url: https://127.0.0.1:11434/v1
    api_key: <managed-native-api-key>
    default_model: example-9b-v3
    transport: chat_completions
    discover_models: true
    models:
      example-9b-v3:
        context_length: 262144
      example-9b-stock:
        context_length: 131072
      example-9b:
        context_length: 131072
      example-9b-vision:
        context_length: 32768
      example-9b-mtp:
        context_length: 32768
```

Keep the explicit model aliases from the quick-start guide. Do not configure a
normal Hermes route to private port 10000: its active model changes during
broker operation.

## Vision switching and restoration

`example-9b-vision` is a transient profile using:

```text
$HOME/.local/share/models/example-9b-stock/Example-9B-Stock-Q6_K.gguf
$HOME/.local/share/models/mmproj-F16.gguf
--mmproj-offload
```

When Hermes sends a vision request, Llama-Broker drains the current text
profile, starts vision, and holds the lease through the full response or SSE
stream. After every vision lease finishes, it restores
`example-9b-v3`.

Validate the route and restoration with a real Hermes image request, then run:

```bash
python3 brokerctl.py status
  curl --fail --silent --cacert $HOME/.config/llama-broker/lan-ca.crt \
    https://127.0.0.1:11434/health | python3 -m json.tool
journalctl --user -u llama-broker.service -n 120 --no-pager
```

If restoration fails, the broker logs a `CRITICAL` message. Stop new traffic,
inspect both units, and use the drain-aware recovery command:

```bash
python3 brokerctl.py switch example-9b-v3
```

## Controlled prompt safety ladder

The ladder is an operator qualification test for a newly changed runtime,
driver, cache format, or long-context profile. It is not a routine startup
check. Qualification runs should be scheduled in a maintenance window with
appropriate work preserved.

Before the test:

1. stop or pause Hermes gateways, cron jobs, and every other broker client;
2. verify the pinned files and installed unit;
3. switch to `example-9b-v3` and confirm its backend PID is stable;
4. record current GPU state and choose an evidence file on persistent storage;
5. ensure the legacy shared server remains disabled.

The Linux-first ladder is portable and requires no Windows tooling:

```bash
cd <repo>
set -o pipefail

python3 brokerctl.py verify example-9b-v3
python3 brokerctl.py switch example-9b-v3

python3 tools/safety_ladder.py \
  --model example-9b-v3 \
  --timeout 1800 \
  8192 16384 32768 49152 65536 98304 131072 196608 245760 262000 \
  | tee /path/to/persistent-evidence/example-v3-safety-ladder.jsonl
```

Targets are sequential and unique. The final target leaves reserved room for
the template and 128 requested output tokens. After every request the tool:

- checks the server-reported prompt length and opening marker;
- verifies `X-Llama-Broker-Profile`;
- verifies that the backend PID did not change;
- records prompt/decode rates and `nvidia-smi` memory, utilization,
  temperature, and power;
- scans the portable Linux kernel journal and the exact profile journal for
  CUDA failures, Xid reports, a fallen-off GPU, device assertions, and related
  backend errors, including a llama-server OOM kill.

On WSL, Windows LiveKernelReport detection can be added explicitly:

```bash
python3 tools/safety_ladder.py \
  --model example-9b-v3 \
  --timeout 1800 \
  --windows-wer \
  8192 16384 32768 49152 65536 98304 131072 196608 245760 262000
```

The default path never calls PowerShell, which keeps Linux deployment
portable. `--windows-wer` is an additional WSL gate, not a replacement for the
Linux journal checks.

Stop at the first validation error or GPU fault. Do not skip ahead, retry the
same dangerous prompt repeatedly, or leave unattended clients active. After an
abort, contain the backend before investigating:

```bash
python3 brokerctl.py stop
systemctl --user disable --now llama-broker.service
rm -f $HOME/.config/systemd/user/llama-server.service.d/10-llama-broker-quarantine.conf
```

Long-context success establishes the tested serial path only. Test sustained
decode, repeated requests, Hermes, vision restoration, and controlled
concurrency as separate gates.

## Health, logs, and runtime inspection

```bash
python3 brokerctl.py status
 curl --fail --silent --cacert $HOME/.config/llama-broker/lan-ca.crt \
   https://127.0.0.1:11434/health | python3 -m json.tool
 curl --fail --silent --cacert $HOME/.config/llama-broker/lan-ca.crt \
   https://127.0.0.1:11434/v1/models | python3 -m json.tool

systemctl --user status llama-broker.service --no-pager -l
journalctl --user -u llama-broker.service -n 200 --no-pager
journalctl --user -u llama-broker-profile-example-9b-v3.service -n 200 --no-pager
nvidia-smi
```

Confirm the active safety environment from the running process:

```bash
pid=$(systemctl --user show \
  llama-broker-profile-example-9b-v3.service -p MainPID --value)
tr '\0' '\n' < "/proc/$pid/environ" | \
  grep -E '^(GGML_CUDA_DISABLE_GRAPHS|GGML_CUDA_GRAPH_OPT|GGML_CUDA_PDL|LLAMA_GRAPH_REUSE_DISABLE)='
readlink -f "/proc/$pid/exe"
```

Expected values are `GGML_CUDA_DISABLE_GRAPHS=1`, `GGML_CUDA_GRAPH_OPT=0`,
`GGML_CUDA_PDL=0`, and `LLAMA_GRAPH_REUSE_DISABLE=1`. The resolved executable
must be the Phase 2 primary runtime shown in the manifest.

## Conservative rollback

### Transactional profile rollback

A normal failed switch automatically attempts to restore the prior profile.
To return manually to the primary profile after a successful alternate-model
request:

```bash
python3 brokerctl.py switch example-9b-v3
```

### Emergency containment

Drain and stop the backend before disabling the broker daemon:

```bash
cd <repo>
python3 brokerctl.py stop
systemctl --user disable --now llama-broker.service
```

Stopping the broker daemon alone can leave its independently managed profile
server running.

### Known-safe direct-mode rollback

The validated 32K direct-mode artifacts from the July 15 watchdog remediation
are retained at:

```text
<llama.cpp-build>/inventory/20260715-143408/
  diagnosis-gpu-lock-20260715-183102/post-mitigation-validation/
```

The matching files are:

```text
llama-server.service.final  SHA-256 (hash computed at profile add)
hermes-config.final.yaml     SHA-256 (hash computed at profile add)
```

Archive the current broker/Hermes configuration first. Then perform the
rollback as one coordinated change:

```bash
SAFE=<llama.cpp-build>/inventory/20260715-143408/diagnosis-gpu-lock-20260715-183102/post-mitigation-validation
SNAPSHOT=<llama.cpp-build>/inventory/rollback-before-direct-$(date +%Y%m%d-%H%M%S).tar.gz

umask 077
tar -C $HOME -czf "$SNAPSHOT" \
  .hermes/config.yaml \
  .config/systemd/user/llama-broker.service \
  .config/systemd/user/llama-server.service \
  workspace/llama-broker/profiles.json

cd <repo>
python3 brokerctl.py stop
systemctl --user disable --now llama-broker.service

{
  echo "(hash computed at profile add)  $SAFE/llama-server.service.final"
  echo "(hash computed at profile add)  $SAFE/hermes-config.final.yaml"
} | sha256sum --check
install -m 0644 "$SAFE/llama-server.service.final" \
  $HOME/.config/systemd/user/llama-server.service
install -m 0600 "$SAFE/hermes-config.final.yaml" \
  $HOME/.hermes/config.yaml

systemctl --user daemon-reload
systemctl --user enable --now llama-server.service
systemctl --user restart hermes-gateway.service
curl --fail --silent http://127.0.0.1:10000/health
```

Do not restore the July 17 legacy direct unit with SHA-256
`(hash computed at profile add)`.
It contains the unsafe two-slot 262K command that prompted containment. Do not
mix one deployment's unit with the other deployment's Hermes configuration.

## Preservation and provenance

The broker's pre-overhaul state is preserved at:

```text
branch: preservation/original-20260717
commit: 2ea6c7acb2b4f017f05af943964d8ed43be822b5
bundle: <llama.cpp-build>/inventory/20260717-113345-broker-kv-overhaul/broker-original-20260717.bundle
bundle SHA-256: (hash computed at profile add)
```

The active primary profile uses the Phase 2 KVarN4 cache. Example 27B and legacy
profiles retain their pinned Turbo4/Prism-compatible formats. KVarN source refs,
patches, qualification results, and remaining blockers are documented in
`<llama.cpp-build>/inventory/20260730-implementation-preflight/` and the
authoritative goal document.

## Inventory-only model discovery

`sync-models.py` may inventory GGUF files under `~/.local/share/models` and
excludes `mmproj-*` and `mtp-*` auxiliaries from standalone results. Its output
does not alter `profiles.json` and cannot be switched to until an operator adds
a reviewed, hashed profile.

`download-models.sh` does not contact Hugging Face unless `--download` is
provided. Downloads still require an explicit review and profile entry before
execution.
