"""FreeToken broker runtime command adapter."""

from __future__ import annotations

from typing import Any, Sequence


def _without_endpoint_args(args: Sequence[str]) -> tuple[str, ...]:
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


class FreeTokenRuntime:
    """Build the broker-owned FreeToken TLS-wrapper command for an FTW directory."""

    name = "freetoken"
    binaries: tuple[str, ...] = ("python", "python3")
    capabilities: tuple[str, ...] = ("text", "vision", "tools", "tensor-parallel", "moe-offload", "nvfp4")
    editable_fields = frozenset()

    def build_args(
        self,
        profile: Any,
        *,
        host: str,
        port: int,
        managed_args: Sequence[str] = (),
    ) -> tuple[str, ...]:
        args: list[str] = [
            "--model-path",
            str(profile.model_path),
            "--served-model-name",
            profile.alias,
            "--max-seq-len-override",
            str(profile.context_size),
            "--tensor-parallel-size",
            str(profile.parallel),
        ]
        if profile.max_output_tokens is not None:
            args.extend(["--max-output-tokens", str(profile.max_output_tokens)])

        args.extend(profile.extra_args)
        args.extend(_without_endpoint_args(managed_args))
        args.extend([
            "--freetoken-external-host",
            host,
            "--freetoken-external-port",
            str(port),
            "--freetoken-upstream-port",
            str(port + 1),
        ])
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


FreeTokenAdapter = FreeTokenRuntime
