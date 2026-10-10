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
        effective_draft = getattr(profile, "effective_model_draft", profile.model_draft)
        if effective_draft is not None:
            args.extend(["-md", str(effective_draft)])

        effective_extra_args = list(
            getattr(profile, "effective_extra_args", profile.extra_args)
        )
        if profile.chat_template:
            # llama.cpp needs Jinja enabled before the external template is
            # selected. Treat --jinja as broker-managed in this case so a
            # profile that already declares it does not add the flag twice.
            args.append("--jinja")
            args.extend(["--chat-template-file", str(profile.chat_template)])
            effective_extra_args = [
                arg for arg in effective_extra_args if arg != "--jinja"
            ]
        args.extend(effective_extra_args)
        # Function-call support does not imply built-in server tools are compiled in.
        if "tools" in profile.capabilities and getattr(profile, "builtin_tools_available", True):
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
