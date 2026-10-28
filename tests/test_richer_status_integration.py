"""Exercise real broker HTTP status and the CLI renderer without production services."""
import http.client
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

from llama_broker import cli
from tests.test_system1 import System1HttpTests as _Harness


def test_real_broker_telemetry_reaches_cli_after_direct_and_routed_requests():
    fixture = _Harness(methodName="test_models_and_health_report_system1")
    fixture.setUp()
    try:
        fixture.d1.answers = {"route": {"choice": {"main": 0.1, "vision": 0.9}}}
        status, _, _ = fixture._post_with_headers({
            "model": "auto", "messages": [{"role": "user", "content": "fixture request"}],
        })
        assert status == 200
        connection = http.client.HTTPConnection("127.0.0.1", fixture.broker.server_port, timeout=5)
        try:
            connection.request("POST", "/v1/systemone", json.dumps({"questions": {}}),
                               {"Content-Type": "application/json", "Authorization": f"Bearer {fixture.auth_key}"})
            response = connection.getresponse()
            response.read()
            assert response.status == 200
        finally:
            connection.close()

        def status_output(*flags):
            output = io.StringIO()
            with (
                patch.object(cli.ProfileRegistry, "from_path", return_value=fixture.registry),
                patch.object(cli, "create_production_controller", return_value=fixture.controller),
                patch.object(cli, "_broker_admin_url", return_value=f"http://127.0.0.1:{fixture.broker.server_port}"),
                patch.object(cli, "effective_backend_url", return_value=f"http://127.0.0.1:{fixture.backend.server_port}"),
                redirect_stdout(output),
            ):
                assert cli.main(["status", *flags]) == 0
            return output.getvalue()

        text = status_output()
        assert "System 1:  available" in text
        assert "S1 usage:  1 direct; 1 routes (since broker start)" in text
        assert "S1 targets: vision=1" in text
        assert "S1 timing:" in text
        detail = json.loads(status_output("--json"))
        m = detail["live"]["system1"]["metrics"]
        assert m["direct_requests"] == {"total": 1, "active": 0, "errors": 0}
        assert m["routing"]["targets"] == {"vision": 1}
        assert detail["live"]["activity"]["operations"] == {"queued": 0, "running": None}
        # Reading status must not itself increment model usage.
        assert json.loads(status_output("--json"))["live"]["system1"]["metrics"] == m
    finally:
        fixture.tearDown()
