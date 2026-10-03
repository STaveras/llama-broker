# Reasoning capability detection

Status: implemented, uncommitted in the `development` working tree.

The broker propagates a profile's reasoning capability from its launch
config, so tools that gate on `capabilities` (e.g. the Pi Coding Agent
adapter) light up without a manual catalog edit.

## Rule

| Launch config | Declared capabilities | Effective capabilities |
|---|---|---|
| `--reasoning on` / `auto` | anything | `+ reasoning` (auto-added) |
| `--reasoning off` or absent | no `reasoning` | unchanged |
| `--reasoning off` or absent | `reasoning` declared | `ProfileError` |

Mirrors the vision precedent: `mmproj` implies the `vision` capability; a
declared capability the launch config cannot serve is a validation error.

## Allowed `--reasoning` modes

`off`, `on`, `auto` (`ALLOWED_REASONING_MODES`). Anything else is an
unsupported-value `ProfileError`.

## Files

| File | Change |
|---|---|
| `src/llama_broker/broker_profiles.py` | `reasoning` added to `ALLOWED_CAPABILITIES`; `ALLOWED_REASONING_MODES` gains `on`/`auto`; `_parse_extra_args` returns `reasoning_mode`; effective capability augmented after extra-args parse; declared-but-disabled is rejected |
| `src/llama_broker/broker_tui.py` | editor capability options include `reasoning` |
| `tests/test_broker_profiles.py` | +4: auto-add on/auto, no-add on off, declared-requires-enabled, unsupported mode |

## End-to-end effect

Daemon model listing exposes `Profile.capabilities` -> `ToolProfile
.from_broker` -> Pi adapter `_model_entry` adds `reasoning: true` and
`thinkingLevelMap`. The Pi reasoning branch is no longer dead code.

## Out of scope

- Model introspection (GGUF has no reliable reasoning key); the flag is
  the operator assertion, same trust model as `mmproj`.
- `setup_wizard.py` default stays `--reasoning off`.
