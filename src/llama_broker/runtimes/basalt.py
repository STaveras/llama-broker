"""Basalt native HTTP server runtime adapter.

The broker owns the listener address and transport. Basalt is launched in its
source tree through the immutable bundle's basalt launcher; no model conversion
or download is performed here.
"""
from __future__ import annotations
import sys
from typing import Any, Sequence


def _without_endpoint_args(args: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    i = 0
    while i < len(args):
        if args[i] in {"--host", "--port"} and i + 1 < len(args):
            i += 2
        else:
            result.append(args[i]); i += 1
    return tuple(result)


class BasaltRuntime:
    name = "basalt"
    binaries = ("basalt",)
    capabilities = ("text", "tools", "streaming")
    editable_fields = frozenset({"basalt_config_path", "basalt_config_sha256"})

    def sync_capability_edit(self, draft, extra_args, capabilities):
        unsupported = set(capabilities) - set(self.capabilities)
        if unsupported:
            raise ValueError(f"basalt does not support capabilities: {', '.join(sorted(unsupported))}")
        return list(extra_args)

    def build_args(self, profile: Any, *, host: str, port: int, managed_args: Sequence[str] = ()) -> tuple[str, ...]:
        del host
        config = getattr(profile, "basalt_config_path", None)
        if config is None:
            raise ValueError("Basalt profile requires a pinned basalt_config_path")
        args = ["serve", "--config", str(config)]
        args += ["--host", "127.0.0.1", "--port", str(port)]
        args.extend(_without_endpoint_args(managed_args))
        return tuple(args)

    def build_command(self, profile: Any, *, host: str, port: int, managed_args: Sequence[str] = ()) -> tuple[str, ...]:
        # The wrapper owns the private native listener and the public TLS bridge.
        if not hasattr(profile, "basalt_config_sha256"):
            return (str(profile.server_binary), *self.build_args(profile, host=host, port=port, managed_args=managed_args))
        return (sys.executable, "-m", "llama_broker.basalt_launcher", "--basalt-binary", str(profile.server_binary), "--config", str(profile.basalt_config_path), "--bridge-port", str(port), *tuple(managed_args))


__all__ = ["BasaltRuntime"]
