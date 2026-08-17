# Proposal: Unified `llm-broker` CLI

## Problem

Broker management is fragmented across multiple entry points:
- `brokerctl.py` — profile management (list, switch, stop, verify)
- `python3 -m configure` — tool configuration
- `manage-services.sh` — shim for systemd
- No backend discovery (must manually know ports)

Users need a single `llm-broker` command with subcommands for all operations.

## Goal

Consolidate into `brokerctl.py` as the main entry point, installable as `llm-broker` via symlink.

## Commands

| Command | Description |
|---------|-------------|
| `llm-broker setup` | Idempotent full setup (units, configs, verify) |
| `llm-broker configure [tools]` | Configure AI coding tools |
| `llm-broker profiles list` | List all profiles + status |
| `llm-broker profiles discover` | Port sweep with dirty tracking |
| `llm-broker profiles edit MODEL` | Edit profile parameters |
| `llm-broker profiles add MODEL` | Add new profile from discovery |
| `llm-broker status` | Show broker state |
| `llm-broker start MODEL` | Switch to profile |
| `llm-broker stop` | Stop backend |
| `llm-broker verify [MODEL]` | Verify SHA256 hashes |

## Design

### Entry Points
- `brokerctl.py` — main script
- `~/bin/llm-broker → brokerctl.py` — global symlink
- `python3 -m brokerctl` — package mode

### Backend Discovery
- Reuse pattern from `simple-chat-client/src/backend_discovery.py`
- Fixed candidate list: `[10000, 11434, 1234, 5000]`
- Optional broad sweep: `--sweep 8000-12000`
- Dirty port tracking: `~/.config/llm-broker/port_state.json`
- Dirty = not checked in >5 minutes

### Profile Editing
- Interactive mode: step through parameters with prompts
- Flag mode: `--context N --parallel N` for scripting
- Detects TTY for auto-selection

### Configuration
- Delegates to existing `configure/` package
- `brokerctl configure` calls `configure.main()`

## Files to Create/Modify

| File | Action |
|------|--------|
| `brokerctl.py` | Expand with new subcommands |
| `discovery.py` | New: port sweep + dirty tracking |
| `brokerctl/__main__.py` | New: package mode entry point |
| `llm-broker` | New: wrapper script for symlink |

## Implementation Steps

1. Create `discovery.py` — port sweep + dirty tracking
2. Create `brokerctl/__main__.py` — package mode
3. Expand `brokerctl.py` — add profiles/discover/edit/setup
4. Create `llm-broker` wrapper script
5. Install symlink
6. Test all subcommands
