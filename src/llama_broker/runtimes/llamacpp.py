"""llama.cpp/llama-server command adapter."""

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


class LlamaCppRuntime:
    """Preserve the broker's existing llama-server argv contract."""

    name = "llamacpp"
    binaries = ("llama-server",)
    capabilities = ("text", "vision", "tools", "fit", "speculative")
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
            "-m",
            str(profile.model_path),
            "--alias",
            profile.alias,
            "-c",
            str(profile.context_size),
            "-np",
            str(profile.parallel),
            "-b",
            str(profile.batch_size),
            "-ub",
            str(profile.ubatch_size),
            "-ctk",
            profile.cache_type_k,
            "-ctv",
            profile.cache_type_v,
            "--flash-attn",
            profile.flash_attention,
            "--no-context-shift",
        ]
        if profile.fit is True:
            assert profile.fit_target_mib is not None
            assert profile.fit_ctx is not None
            args.extend(
                [
                    "--fit",
                    "on",
                    "--fit-target",
                    str(profile.fit_target_mib),
                    "--fit-ctx",
                    str(profile.fit_ctx),
                ]
            )
            if profile.cache_ram_mib is not None:
                args.extend(["--cache-ram", str(profile.cache_ram_mib)])
            if profile.ctx_checkpoints is not None:
                args.extend(["--ctx-checkpoints", str(profile.ctx_checkpoints)])
        elif profile.fit is False:
            args.extend(["--no-fit", "-ngl", profile.gpu_layers])
        else:
            args.extend(["-ngl", profile.gpu_layers])

        if profile.mmproj_path is not None:
            args.extend(["--mmproj", str(profile.mmproj_path), "--mmproj-offload"])
        if profile.model_draft is not None:
            args.extend(["-md", str(profile.model_draft)])

        args.extend(profile.extra_args)
        if "tools" in profile.capabilities:
            args.extend(["--tools", "all"])

        args.extend(["--host", host, "--port", str(port)])
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


LlamaCppAdapter = LlamaCppRuntime
