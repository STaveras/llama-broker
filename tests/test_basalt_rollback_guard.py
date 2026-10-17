import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "basalt_production_rollback.py"


def test_rollback_hook_dispatches_independent_unit_not_owner():
    text = SCRIPT.read_text()
    assert "llama-broker.service" not in text
    assert '"start", args.recovery_unit' in text
    assert "--dry-run" in text


def test_guard_has_no_recursive_restart():
    text = (ROOT / "tools" / "basalt-production-guard.conf.example").read_text()
    assert "ExecStopPost=" in text
    assert "restart llama-broker.service" not in text
    assert "llama-broker-recovery.service" in text


def test_dry_run_is_bounded_and_does_not_mutate_state(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "ARMED").write_text("armed\n")
    fake = tmp_path / "systemctl"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o700)
    result = subprocess.run(
        ["python3", str(SCRIPT), "--state", str(state), "--dry-run"],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin"}, timeout=3,
    )
    assert result.returncode == 0
    assert not (state / "RECOVERY_REQUEST").exists()
