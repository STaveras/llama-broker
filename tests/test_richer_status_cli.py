"""Human status contracts exercised through the real CLI entry point."""
import io
import json
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import pytest

from llama_broker import cli


def render(live, *flags):
    controller = Mock()
    controller.state.return_value = {"phase": "inactive", "profile": None}
    output = io.StringIO()
    with (
        patch.object(cli.ProfileRegistry, "from_path", return_value=Mock()),
        patch.object(cli, "create_production_controller", return_value=controller),
        patch.object(cli, "backend_endpoint_ready", return_value=False),
        patch.object(cli, "_broker_admin_get", return_value=live),
        redirect_stdout(output),
    ):
        code = cli.main(["status", *flags])
    return code, output.getvalue()


def service(**extra):
    return {"enabled": True, "ready": True, "model": "resident-d1",
            "routing": {"enabled": True}, **extra}


def metrics():
    return {"scope": "since_broker_start",
            "direct_requests": {"total": 12, "active": 1, "errors": 1},
            "routing": {"total": 40, "active": 0, "errors": 2, "fallbacks": 3,
                        "targets": {"main": 28, "small": 12},
                        "latency_ms": {"last": 8.5, "mean": 11.25}}}


@pytest.mark.parametrize("s1", [None, {}, {"enabled": False}, {"enabled": False, "ready": True}])
def test_absent_or_disabled_system1_preserves_quiet_output(s1):
    live = {"status": "ok"}
    if s1 is not None:
        live["system1"] = s1
    assert render(live) == (0, "Broker:    inactive (port 11434; health: ok)\nBackend:   (none)\n")


def test_available_system1_visible_without_primary_backend():
    code, output = render({"status": "ok", "system1": service()})
    assert code == 0
    assert "System 1:  available — resident-d1 (routing enabled)" in output
    assert "S1 usage:" not in output  # Old daemon has no telemetry; do not invent zeros.


@pytest.mark.parametrize("ready,word", [(False, "unavailable"), (None, "unknown")])
def test_configured_unhealthy_or_unknown_system1_is_not_hidden(ready, word):
    _, output = render({"system1": service(ready=ready)})
    assert f"System 1:  {word} — resident-d1" in output


def test_routing_disabled_is_distinct_from_model_unavailable():
    _, output = render({"system1": service(routing={"enabled": False})})
    assert "available — resident-d1 (routing disabled)" in output


def test_routing_missing_is_unknown_not_disabled():
    _, output = render({"system1": service(routing={})})
    assert "routing unknown" in output


def test_metrics_distinguish_direct_routing_errors_fallbacks_and_latency():
    _, output = render({"system1": service(metrics=metrics())})
    assert "S1 usage:  12 direct; 40 routes (since broker start)" in output
    assert "S1 active: 1 direct; 0 routes" in output
    assert "S1 errors: 1 direct; 2 routes; 3 fallbacks" in output
    assert "S1 timing: last 8.5 ms; mean 11.2 ms (routing)" in output
    assert "S1 targets: main=28, small=12" in output


def test_zero_metrics_keep_idle_capability_compact():
    m = metrics()
    m["direct_requests"] = {"total": 0, "active": 0, "errors": 0}
    m["routing"] = {"total": 0, "active": 0, "errors": 0, "fallbacks": 0,
                    "targets": {}, "latency_ms": {"last": None, "mean": None}}
    _, output = render({"system1": service(metrics=m)})
    assert "S1 usage:  0 direct; 0 routes (since broker start)" in output
    assert all(label not in output for label in ("S1 active:", "S1 errors:", "S1 timing:", "S1 targets:"))


def test_target_summary_is_ranked_and_bounded():
    m = metrics()
    m["routing"]["targets"] = {"a": 1, "b": 10, "c": 5, "d": 3}
    _, output = render({"system1": service(metrics=m)})
    assert "S1 targets: b=10, c=5, d=3 (+1 more)" in output


def test_missing_metric_counts_are_unknown_not_zero():
    _, output = render({"system1": service(metrics={"direct_requests": {}, "routing": {}})})
    assert "S1 usage:  unknown direct; unknown routes (scope unknown)" in output


def test_invalid_latency_is_not_rendered_as_real_measurement():
    m = metrics()
    m["routing"]["latency_ms"] = {"last": float("nan"), "mean": -1}
    _, output = render({"system1": service(metrics=m)})
    assert "S1 timing:" not in output


def test_inference_activity_is_shown_even_without_resolved_primary_profile():
    _, output = render({"leases": {"inference_requests": 2, "active_requests": 3}})
    assert "Activity:  2 in-flight requests" in output


def test_health_probe_count_is_not_presented_as_inference_activity():
    _, output = render({"leases": {"inference_requests": 0, "active_requests": 1}})
    assert "Activity:" not in output


def test_transition_explains_operation_and_target_with_waiting_queue():
    _, output = render({
        "leases": {"transitioning": True, "transition": "switch", "transition_target": "large"},
        "activity": {"operations": {"queued": 2, "running": {"type": "switch", "target": "large", "status": "committed"}}},
    })
    assert "Changing:  switching to large" in output
    assert "Queue:     2 waiting operations" in output
    assert "Operation:" not in output


def test_preflight_is_not_mislabeled_as_an_applied_model_switch():
    _, output = render({"activity": {"operations": {
        "queued": 1, "running": {"type": "reload", "target": None, "status": "preflight"},
    }}})
    assert "Operation: reloading profiles (preflight)" in output
    assert "Queue:     1 waiting operation\n" in output


def test_old_daemon_pending_transitions_have_honest_fallback_label():
    _, output = render({"leases": {"queued_transitions": 2}})
    assert "Queue:     2 pending transitions" in output


def test_no_live_snapshot_does_not_invent_idle_activity_or_system1():
    _, output = render({})
    assert all(label not in output for label in ("Activity:", "Changing:", "Queue:", "System 1:"))


def test_json_remains_machine_readable_and_preserves_additive_telemetry():
    live = {"system1": service(metrics=metrics()), "activity": {"operations": {"queued": 2, "running": None}}}
    code, output = render(live, "--json")
    detail = json.loads(output)
    assert code == 0
    assert detail["schema_version"] == 1
    assert detail["live"] == live
    assert "System 1:" not in output


def test_verbose_includes_same_live_telemetry():
    live = {"system1": service(metrics=metrics())}
    _, output = render(live, "--verbose")
    assert "S1 usage:  12 direct" in output
    assert json.loads(output[output.index("\n{") + 1:])["live"] == live


def test_status_labels_cannot_inject_terminal_control_or_extra_lines():
    _, output = render({"system1": service(model="d1\n\x1b[31mforged")})
    assert "\x1b" not in output
    assert "d1??[31mforged" in output
