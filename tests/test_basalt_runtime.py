from types import SimpleNamespace
import sys
import pytest
from llama_broker.runtimes import get_runtime, runtime_names
from llama_broker.daemon import backend_url_for_runtime


def test_basalt_is_explicit_runtime_and_builds_loopback_server_command():
    profile = SimpleNamespace(server_binary="/bundle/basalt", basalt_config_path="/bundle/model.json", model_path=None)
    assert "basalt" in runtime_names()
    assert get_runtime("basalt").build_command(
        profile, host="203.0.113.8", port=10001,
        managed_args=("--host", "127.0.0.1", "--port", "7400", "--api-key-file", "/run/key", "--ssl-cert-file", "/run/cert", "--ssl-key-file", "/run/key.pem"),
    ) == (
        sys.executable, "-m", "llama_broker.basalt_launcher",
        "--basalt-binary", "/bundle/basalt", "--config", "/bundle/model.json",
        "--cert", "/run/cert", "--key", "/run/key.pem", "--auth-token-file", "/run/key",
        "--host", "203.0.113.8", "--port", "0", "--bridge-port", "10001"
    )


def test_basalt_backend_never_downgrades_tls():
    assert backend_url_for_runtime("https://127.0.0.1:10001", "basalt") == "https://127.0.0.1:10001"
    with pytest.raises(Exception, match="HTTPS"):
        backend_url_for_runtime("http://127.0.0.1:10001", "basalt")
