"""Basalt native HTTP server runtime adapter.

The broker owns the listener address and transport. Basalt is launched in its
source tree through the immutable bundle's basalt launcher; no model conversion
or download is performed here.
"""
from __future__ import annotations
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
    editable_fields = frozenset()

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
        return (str(profile.server_binary), *self.build_args(profile, host=host, port=port, managed_args=managed_args))


__all__ = ["BasaltRuntime"]
