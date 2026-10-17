"""Small static registry of supported model-serving runtimes.

Adapters deliberately depend only on the profile object's public attributes;
the profile parser owns validation and this registry owns command shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .llamacpp import LlamaCppRuntime
from .strata import StrataRuntime
from .vllm import VllmRuntime
from .basalt import BasaltRuntime


class RuntimeError(ValueError):
    """A profile selected an unknown or unusable runtime."""


class RuntimeAdapterProtocol(Protocol):
    name: str
    binaries: tuple[str, ...]
    capabilities: tuple[str, ...]
    editable_fields: frozenset[str]

    def sync_capability_edit(self, draft: Mapping[str, Any], extra_args: list[str], capabilities: list[str]) -> list[str]:
        raise NotImplementedError

    def build_args(
        self,
        profile: Any,
        *,
        host: str,
        port: int,
        managed_args: Sequence[str] = (),
    ) -> tuple[str, ...]:
        raise NotImplementedError

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


@dataclass(frozen=True)
class RuntimeInfo:
    """Registry metadata plus the adapter that builds process argv."""

    name: str
    binaries: tuple[str, ...]
    capabilities: tuple[str, ...]
    editable_fields: frozenset[str]
    adapter: RuntimeAdapterProtocol

    def sync_capability_edit(self, draft: Mapping[str, Any], extra_args: list[str], capabilities: list[str]) -> list[str]:
        return self.adapter.sync_capability_edit(draft, extra_args, capabilities)


_BASE_FIELDS = frozenset(
    {
        "runtime",
        "id",
        "alias",
        "aliases",
        "display_name",
        "role",
        "capabilities",
        "native_vision",
        "restore_profile",
        "runtime_build",
        "server_binary",
        "server_sha256",
        "model_path",
        "model_sha256",
        "startup_timeout",
        "safety",
        "environment",
        "extra_args",
        "architecture",
        "jinja_support",
        "max_output_tokens",
        "autotune",
    }
)

_LLAMACPP_FIELDS = _BASE_FIELDS | frozenset(
    {
        "mmproj_path",
        "mmproj_sha256",
        "context_size",
        "parallel",
        "batch_size",
        "ubatch_size",
        "cache_type_k",
        "cache_type_v",
        "flash_attention",
        "gpu_layers",
        "fit",
        "fit_target_mib",
        "fit_ctx",
        "ctx_checkpoints",
        "cache_ram_mib",
        "model_draft",
        "model_draft_sha256",
        "chat_template",
        "moe_cache_slots_limit",
    }
)

_VLLM_FIELDS = _BASE_FIELDS | frozenset(
    {
        "context_size",
        "parallel",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "kv_cache_dtype",
        "tool_call_parser",
        "tool_parser_plugin",
        "chat_template",
        "limit_mm_per_prompt",
        "allowed_local_media_path",
    }
)


_STRATA_FIELDS = frozenset(
    {
        "runtime",
        "id",
        "alias",
        "aliases",
        "display_name",
        "role",
        "capabilities",
        "native_vision",
        "runtime_build",
        "server_binary",
        "server_sha256",
        "model_path",
        "model_sha256",
        "startup_timeout",
        "safety",
        "environment",
        "max_output_tokens",
        "context_size",
        "parallel",
        "strata_artifacts_manifest_path",
        "strata_artifacts_manifest_sha256",
        "strata_launch_config_path",
        "strata_launch_config_sha256",
    }
)


_LLAMACPP = LlamaCppRuntime()
_VLLM = VllmRuntime()
_STRATA = StrataRuntime()
_BASALT = BasaltRuntime()

_BASALT_FIELDS = _BASE_FIELDS | frozenset({"basalt_config_path", "context_size", "parallel"})

RUNTIME_REGISTRY: Mapping[str, RuntimeInfo] = {
    "llamacpp": RuntimeInfo(
        name="llamacpp",
        binaries=("llama-server",),
        capabilities=_LLAMACPP.capabilities,
        editable_fields=_LLAMACPP_FIELDS,
        adapter=_LLAMACPP,
    ),
    "vllm": RuntimeInfo(
        name="vllm",
        binaries=("vllm",),
        capabilities=("text", "vision", "tools", "tensor-parallel", "kv-cache-dtype"),
        editable_fields=_VLLM_FIELDS,
        adapter=_VLLM,
    ),
    "strata": RuntimeInfo(
        name="strata",
        binaries=("strata-server",),
        capabilities=_STRATA.capabilities,
        editable_fields=_STRATA_FIELDS,
        adapter=_STRATA,
    ),
    "basalt": RuntimeInfo(
        name="basalt",
        binaries=("basalt",),
        capabilities=_BASALT.capabilities,
        editable_fields=_BASALT_FIELDS,
        adapter=_BASALT,
    ),
}


def runtime_names() -> tuple[str, ...]:
    return tuple(RUNTIME_REGISTRY)


def list_runtimes() -> tuple[RuntimeInfo, ...]:
    return tuple(RUNTIME_REGISTRY.values())


def get_runtime(name: str) -> RuntimeAdapterProtocol:
    key = str(name).strip().lower()
    try:
        return RUNTIME_REGISTRY[key].adapter
    except KeyError as error:
        raise RuntimeError(
            f"unknown runtime {name!r}; registered runtimes: {', '.join(runtime_names())}"
        ) from error


def get_runtime_info(name: str) -> RuntimeInfo:
    key = str(name).strip().lower()
    try:
        return RUNTIME_REGISTRY[key]
    except KeyError as error:
        raise RuntimeError(
            f"unknown runtime {name!r}; registered runtimes: {', '.join(runtime_names())}"
        ) from error
