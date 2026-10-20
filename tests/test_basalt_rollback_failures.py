import importlib.util
from pathlib import Path
import subprocess
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/basalt_production_rollback.py'
spec = importlib.util.spec_from_file_location('rollback_hook', SCRIPT)
rollback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rollback)


def test_service_dispatch_timeout_returns_bounded_failure():
    with patch.object(rollback.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['systemctl'], 1)):
        assert rollback.run(['systemctl'], 1) == 124


def test_missing_systemctl_is_a_reportable_failure():
    with patch.object(rollback.subprocess, 'run', side_effect=FileNotFoundError):
        assert rollback.run(['systemctl'], 1) == 127
