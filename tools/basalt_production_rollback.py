#!/usr/bin/env python3
"""Bounded Basalt rollback hook; never restarts its owning broker unit."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path


def run(argv: list[str], timeout: float) -> int:
    try:
        return subprocess.run(argv, timeout=timeout, check=False).returncode
    except subprocess.TimeoutExpired:
        return 124
    except OSError:
        return 127


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--recovery-unit", default="llama-broker-recovery.service")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    state = args.state
    armed = state / "ARMED"
    success = state / "SUCCESS"
    if success.exists() or not armed.exists():
        return 0
    # Startup gating: verify the independent recovery unit exists before any restore mutation.
    check = ["systemctl", "--user", "cat", args.recovery_unit]
    if args.dry_run:
        return 0 if run(check, 5) == 0 else 2
    if run(check, 5) != 0:
        (state / "rollback-result.json").write_text(json.dumps({"result": "blocked", "error": "recovery unit unavailable"}) + "\n")
        return 2
    state.mkdir(parents=True, exist_ok=True)
    (state / "RECOVERY_REQUEST").write_text("rollback\n")
    rc = run(["systemctl", "--user", "start", args.recovery_unit], 10)
    (state / "rollback-result.json").write_text(json.dumps({"result": "recovery-dispatched" if rc == 0 else "recovery-dispatch-failed", "returncode": rc}) + "\n")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
