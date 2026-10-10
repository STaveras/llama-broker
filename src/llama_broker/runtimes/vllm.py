"""vLLM OpenAI-compatible server command adapter."""

from __future__ import annotations

import json
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


class VllmRuntime:
    """Build vLLM's ``vllm serve MODEL`` argv, never llama.cpp flags."""

    name = "vllm"
    binaries = ("vllm",)
    capabilities = ("text", "vision", "tools", "tensor-parallel", "kv-cache-dtype")
    editable_fields = frozenset()

    def sync_capability_edit(self, draft, extra_args, capabilities):
        if "reasoning" in capabilities:
            raise ValueError("vllm does not support the reasoning capability")
        return list(extra_args)

    def build_args(
        self,
        profile: Any,
        *,
        host: str,
        port: int,
        managed_args: Sequence[str] = (),
    ) -> tuple[str, ...]:
        args: list[str] = [
            "serve",
            str(profile.model_path),
            "--served-model-name",
            profile.alias,
            "--max-model-len",
            str(profile.context_size),
            "--tensor-parallel-size",
            str(profile.tensor_parallel_size),
        ]
        if profile.gpu_memory_utilization is not None:
            args.extend(["--gpu-memory-utilization", str(profile.gpu_memory_utilization)])
        if profile.kv_cache_dtype is not None:
            args.extend(["--kv-cache-dtype", profile.kv_cache_dtype])

        if "tools" in profile.capabilities:
            args.append("--enable-auto-tool-choice")
            if profile.tool_call_parser:
                args.extend(["--tool-call-parser", profile.tool_call_parser])
        if profile.tool_parser_plugin:
            args.extend(["--tool-parser-plugin", profile.tool_parser_plugin])
        if profile.chat_template:
            args.extend(["--chat-template", profile.chat_template])

        if profile.limit_mm_per_prompt is not None:
            args.extend(
                [
                    "--limit-mm-per-prompt",
                    json.dumps(dict(profile.limit_mm_per_prompt), separators=(",", ":")),
                ]
            )
        if profile.allowed_local_media_path:
            args.extend(["--allowed-local-media-path", profile.allowed_local_media_path])

        args.extend(profile.extra_args)
        args.extend(_without_endpoint_args(managed_args))
        args.extend(["--host", host, "--port", str(port)])
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


VllmAdapter = VllmRuntime
