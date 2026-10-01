"""Strata's local OpenAI-compatible server command adapter."""

from __future__ import annotations

from typing import Any, Sequence


def _without_endpoint_args(args: Sequence[str]) -> tuple[str, ...]:
    """Drop controller endpoint values before Strata's loopback endpoint is set."""
    result: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"--host", "--port"} and index + 1 < len(args):
            index += 2
            continue
        result.append(token)
        index += 1
    return tuple(result)


class StrataRuntime:
    """Launch the broker-packaged Strata bridge, always on loopback."""

    name: str = "strata"
    binaries: tuple[str, ...] = ("strata-server",)
    capabilities: tuple[str, ...] = ("text",)
    editable_fields: frozenset[str] = frozenset()

    def build_args(
        self,
        profile: Any,
        *,
        host: str,
        port: int,
        managed_args: Sequence[str] = (),
    ) -> tuple[str, ...]:
        del host  # Strata's HTTP server is private to the local broker.
        args = [
            "--config",
            str(profile.strata_launch_config_path),
            "--config-sha256",
            str(profile.strata_launch_config_sha256),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--min-available-ram-mib",
            str(profile.safety.admission.min_ram_available_mib),
            "--max-swap-used-mib",
            str(profile.safety.admission.max_swap_used_mib),
            "--memory-max-mib",
            str(profile.safety.service.memory_max_mib),
            "--memory-swap-max-mib",
            str(profile.safety.service.memory_swap_max_mib),
            "--min-total-vram-mib",
            str(profile.safety.admission.min_total_vram_mib),
            "--min-free-vram-mib",
            str(profile.safety.admission.min_free_vram_before_start_mib),
        ]
        args.extend(_without_endpoint_args(managed_args))
        return tuple(args)

    def build_command(
        self,
        profile: Any,
        *,
        host: str,
        port: int,
        managed_args: Sequence[str] = (),
    ) -> tuple[str, ...]:
        return (str(profile.server_binary), *self.build_args(
            profile, host=host, port=port, managed_args=managed_args
        ))
