"""Missing isolation must fail closed even when positive build tests skip."""
import pytest
from llama_broker import runtime_control as rc


def test_missing_bubblewrap_never_falls_back_to_host_build(tmp_path, monkeypatch):
    monkeypatch.setattr(rc.shutil, "which", lambda _name: None)
    with pytest.raises(rc.RuntimeControlError, match="bubblewrap"):
        rc._sandbox(tmp_path, tmp_path, {})
