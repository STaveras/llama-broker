from types import SimpleNamespace
import pytest
from llama_broker.runtimes import get_runtime, runtime_names
from llama_broker.daemon import backend_url_for_runtime


def test_basalt_is_explicit_runtime_and_builds_loopback_server_command():
    profile = SimpleNamespace(server_binary="/bundle/basalt", basalt_config_path="/bundle/model.json", model_path=None)
    assert "basalt" in runtime_names()
    assert get_runtime("basalt").build_command(profile, host="0.0.0.0", port=10001) == (
        "/bundle/basalt", "serve", "--config", "/bundle/model.json", "--host", "127.0.0.1", "--port", "10001"
    )


def test_basalt_backend_downgrades_tls_only_on_loopback():
    assert backend_url_for_runtime("https://127.0.0.1:10001", "basalt") == "http://127.0.0.1:10001"
    with pytest.raises(Exception, match="loopback"):
        backend_url_for_runtime("https://203.0.113.5:10001", "basalt")
