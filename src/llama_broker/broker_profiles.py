#!/usr/bin/env python3
"""Validated, allowlisted runtime profiles for the local Hermes broker.

Automatic GGUF discovery remains useful for inventory, but it is deliberately
not execution authority.  Only profiles in ``profiles.json`` can start a
backend process.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .runtimes import get_runtime, get_runtime_info
from .runtimes.registry import RuntimeError as RuntimeRegistryError
from .runtime_builds import (
    RuntimeBuild,
    RuntimeBuildError,
    RuntimeCatalog,
    external_catalog_for_manifest,
    parse_runtime_builds,
    resolve_profile_runtime,
)


PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_ROLES = frozenset({"persistent", "transient"})
ALLOWED_CAPABILITIES = frozenset({"text", "vision", "tools", "reasoning"})
ALLOWED_CACHE_TYPES = frozenset(
    {"f16", "f32", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "turbo4", "kvarn4"}
)
# Profiles may request the full trained window, but the fail-closed single-slot,
# fit, and memory controls below still apply.
MAX_CONTEXT_SIZE = 262_144
MAX_BATCH_SIZE = 512
MAX_UBATCH_SIZE = 128
QUALIFIED_LLAMPAMPERE_BUILD = "llamampere-v054-tls-sm120-fb389de8bf76"
QUALIFIED_LLAMPAMPERE_SHA256 = "fb389de8bf76833ef01e4779daacc382609229d34c0e1d653e253831cf990168"
QUALIFIED_LLAMPAMPERE_COMMIT = "e7093a5526aa82e3e271120358dc021ab222060a"
LLAMPAMPERE_EXTRA_VALUES = {
    "--cache-type-s": "q8_0",
    "--spec-draft-type-k": "q8_0",
    "--spec-draft-type-v": "q8_0",
    "--spec-draft-window": "8192",
    "--cache-ram": "0",
    "--fit": "off",
    "--sjkvarn-body-type": "sjkvarn4",
}
LLAMPAMPERE_FLAG_ONLY = frozenset({"--no-cache-disk", "--no-webui", "--offline", "--no-context-shift"})
MAX_FIT_TARGET_MIB = 8_192
MAX_TASKS_MAX = 512
MAX_MEMORY_MAX_MIB = 20_480
MAX_STRATA_MEMORY_MAX_MIB = 57_344
MIN_STRATA_MEMORY_MAX_MIB = 8_192
MIN_STRATA_RAM_AVAILABLE_MIB = 6_144
MIN_STRATA_KV_RESIDENT_CELLS = 20_480
# KV formats that support --kv-resident streaming (k8v4 cannot stream)
STRATA_KV_FORMATS = frozenset({"fp16", "int8", "q4_0"})
STRATA_RUNTIME_ENVIRONMENT_FIELDS = frozenset(
    {"LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS"}
)
MAX_MEMORY_SWAP_MAX_MIB = 1_024
MAX_TIMEOUT_STOP_SEC = 20
MAX_SPECULATIVE_DRAFT_TOKENS = 64
MAX_MOE_CACHE_SLOTS = 8
MAX_QUALIFIED_MOE_CACHE_SLOTS = 112
MAX_CACHE_RAM_SHARE_DIVISOR = 2
MIN_ADMISSION_RAM_AVAILABLE_MIB = 8_192
MAX_ADMISSION_SWAP_USED_MIB = 1_024
MAX_ADMISSION_SWAP_GROWTH_MIB = 128
MIN_ADMISSION_TOTAL_VRAM_MIB = 12_000
MIN_ADMISSION_FREE_VRAM_MIB = 8_192
FLAG_ONLY_EXTRA_ARGS = frozenset({"--no-warmup", "--jinja"})
ALLOWED_REASONING_MODES = frozenset({"off", "on", "auto"})
ALLOWED_SPECULATIVE_TYPES = frozenset({"mtp", "draft-mtp", "nextn"})
SPECULATIVE_COUNT_FLAGS = frozenset(
    {"--draft-max", "--spec-draft-n-max", "--draft-min"}
)
MOE_CACHE_VALUE_FLAGS = frozenset(
    {"--moe-cache-profile", "--moe-cache-slots"}
)
LLAMACPP_AUTOTUNE_ENVIRONMENT = MappingProxyType(
    {
        "GGML_CUDA_DISABLE_GRAPHS": "1",
        "GGML_CUDA_GRAPH_OPT": "0",
        "LLAMA_GRAPH_REUSE_DISABLE": "1",
    }
)
LLAMACPP_AUTOTUNE_ONLY_ENVIRONMENT = frozenset(LLAMACPP_AUTOTUNE_ENVIRONMENT)


def launch_environment(
    environment: Mapping[str, str],
    *,
    runtime: str,
    autotune: bool = False,
) -> dict[str, str]:
    """Return the environment for a normal or autotune runtime launch.

    CUDA graph controls are deliberately candidate-only.  Strip legacy values
    from normal llama.cpp launches, then force the conservative values for
    autotune candidates so profile data cannot leak the tuning policy into
    production units.
    """

    effective = dict(environment)
    if runtime != "llamacpp":
        return effective

    for name in LLAMACPP_AUTOTUNE_ONLY_ENVIRONMENT:
        effective.pop(name, None)
    if autotune:
        effective.update(LLAMACPP_AUTOTUNE_ENVIRONMENT)
    return effective


VLLM_KV_CACHE_DTYPES = frozenset(
    {
        "auto",
        "bfloat16",
        "float16",
        "fp8",
        "fp8_ds_mla",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_inc",
        "fp8_per_token_head",
        "int4_per_token_head",
        "int8_per_token_head",
        "nvfp4",
        "turboquant_3bit_nc",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_k8v4",
    }
)
VLLM_FLAG_ONLY_EXTRA_ARGS = frozenset(
    {
        "--trust-remote-code",
        "--enforce-eager",
        "--disable-log-requests",
        "--enable-prefix-caching",
        "--disable-frontend-multiprocessing",
    }
)
VLLM_VALUE_EXTRA_ARGS = frozenset(
    {
        "--dtype",
        "--quantization",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--swap-space",
        "--cpu-offload-gb",
        "--served-model-name",
        "--task",
        "--reasoning-parser",
        "--reasoning-parser-plugin",
    }
)
VLLM_LLAMA_ONLY_FIELDS = frozenset(
    {
        "mmproj_path",
        "mmproj_sha256",
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
    }
)
LLAMACPP_VLLM_ONLY_FIELDS = frozenset(
    {
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "kv_cache_dtype",
        "tool_call_parser",
        "tool_parser_plugin",
        "limit_mm_per_prompt",
        "allowed_local_media_path",
    }
)


class ProfileError(ValueError):
    """A profile manifest violates a broker safety invariant."""


def _required(record: Mapping[str, Any], key: str) -> Any:
    if key not in record:
        raise ProfileError(f"missing required profile field: {key}")
    return record[key]


def _positive_int(record: Mapping[str, Any], key: str) -> int:
    value = _required(record, key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProfileError(f"{key} must be a positive integer")
    return value


def _nonnegative_int(record: Mapping[str, Any], key: str) -> int:
    value = _required(record, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProfileError(f"{key} must be a non-negative integer")
    return value


def _optional_nonnegative_int(
    record: Mapping[str, Any], key: str, *, default: int = 0
) -> int:
    value = record.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProfileError(f"{key} must be a non-negative integer")
    return value


def _parse_vllm_extra_args(
    args: tuple[str, ...], profile_id: str
) -> tuple[tuple[str, ...], frozenset[str]]:
    seen_flags: set[str] = set()
    index = 0
    while index < len(args):
        flag = args[index]
        if flag in seen_flags:
            raise ProfileError(f"duplicate extra_args flag for {profile_id!r}: {flag}")
        if flag in VLLM_FLAG_ONLY_EXTRA_ARGS:
            seen_flags.add(flag)
            index += 1
            continue
        if flag not in VLLM_VALUE_EXTRA_ARGS:
            raise ProfileError(
                f"extra_args for vllm profile {profile_id!r} contain unsupported "
                f"token: {flag!r}"
            )
        if index + 1 >= len(args) or args[index + 1].startswith("--"):
            raise ProfileError(
                f"extra_args flag {flag} for {profile_id!r} requires a value"
            )
        seen_flags.add(flag)
        index += 2
    return args, frozenset(), None


def _parse_extra_args(
    raw: object,
    profile_id: str,
    *,
    runtime: str,
    moe_cache_slots_limit: int = MAX_MOE_CACHE_SLOTS,
    qualified_ampere: bool = False,
) -> tuple[tuple[str, ...], frozenset[str]]:
    if not isinstance(raw, list) or not all(
        isinstance(item, str) and item for item in raw
    ):
        raise ProfileError(f"extra_args must be a string list for {profile_id!r}")

    args = tuple(raw)
    if runtime == "strata":
        if args:
            raise ProfileError(f"Strata profiles do not accept extra_args for {profile_id!r}")
        return (), frozenset(), None
    if runtime == "vllm":
        return _parse_vllm_extra_args(args, profile_id)

    speculative_types: set[str] = set()
    reasoning_mode: str | None = None
    seen_flags: set[str] = set()
    i = 0
    while i < len(args):
        flag = args[i]
        if qualified_ampere and flag not in (
            set(LLAMPAMPERE_EXTRA_VALUES) | set(LLAMPAMPERE_FLAG_ONLY)
            | {"--no-warmup", "--reasoning", "--spec-type", "--spec-draft-n-max"}
        ):
            raise ProfileError(f"unsupported qualified llamAmpere flag: {flag!r}")
        if flag in seen_flags:
            raise ProfileError(f"duplicate extra_args flag for {profile_id!r}: {flag}")
        if flag in FLAG_ONLY_EXTRA_ARGS or (qualified_ampere and flag in LLAMPAMPERE_FLAG_ONLY):
            seen_flags.add(flag)
            i += 1
            continue
        if flag not in {
            "--reasoning",
            "--spec-type",
            *SPECULATIVE_COUNT_FLAGS,
            *MOE_CACHE_VALUE_FLAGS,
            *(LLAMPAMPERE_EXTRA_VALUES if qualified_ampere else ()),
        }:
            raise ProfileError(
                f"extra_args for {profile_id!r} contain unsupported token: {flag!r}"
            )
        if i + 1 >= len(args) or args[i + 1].startswith("--"):
            raise ProfileError(
                f"extra_args flag {flag} for {profile_id!r} requires a value"
            )

        value = args[i + 1]
        seen_flags.add(flag)
        if qualified_ampere and flag in {"--reasoning", "--spec-type", "--spec-draft-n-max"}:
            expected = {"--reasoning": "on", "--spec-type": "draft-mtp", "--spec-draft-n-max": "3"}[flag]
            if value != expected:
                raise ProfileError(f"unqualified llamAmpere value for {flag}: {value!r}")
        if qualified_ampere and flag in LLAMPAMPERE_EXTRA_VALUES:
            if value != LLAMPAMPERE_EXTRA_VALUES[flag]:
                raise ProfileError(f"unqualified llamAmpere value for {flag}: {value!r}")
        elif flag == "--moe-cache-profile":
            path = Path(value).expanduser()
            if not path.is_absolute() or not path.is_file():
                raise ProfileError(
                    f"extra_args flag {flag} for {profile_id!r} requires an "
                    f"absolute existing file: {value!r}"
                )
        elif flag == "--moe-cache-slots":
            if not re.fullmatch(r"[0-9]+", value):
                raise ProfileError(
                    f"extra_args flag {flag} for {profile_id!r} must be a "
                    "non-negative integer"
                )
            slots = int(value, 10)
            if slots > moe_cache_slots_limit:
                raise ProfileError(
                    f"extra_args flag {flag} for {profile_id!r} exceeds "
                    f"profile cache-slot limit {moe_cache_slots_limit}"
                )
        elif flag == "--reasoning":
            if value not in ALLOWED_REASONING_MODES:
                raise ProfileError(
                    f"extra_args flag {flag} for {profile_id!r} has unsupported "
                    f"value {value!r}"
                )
            reasoning_mode = value
        elif flag == "--spec-type":
            requested = tuple(part.strip() for part in value.split(","))
            if (
                not requested
                or any(not part for part in requested)
                or not set(requested) <= ALLOWED_SPECULATIVE_TYPES
            ):
                raise ProfileError(
                    f"extra_args flag {flag} for {profile_id!r} has unsupported "
                    f"value {value!r}"
                )
            speculative_types.update(requested)
        else:
            if not re.fullmatch(r"[0-9]+", value):
                raise ProfileError(
                    f"extra_args flag {flag} for {profile_id!r} must be an integer"
                )
            count = int(value, 10)
            if count > MAX_SPECULATIVE_DRAFT_TOKENS:
                raise ProfileError(
                    f"extra_args flag {flag} for {profile_id!r} exceeds "
                    f"fail-closed limit {MAX_SPECULATIVE_DRAFT_TOKENS}"
                )
        i += 2

    return args, frozenset(speculative_types), reasoning_mode


def _absolute_existing_path(
    record: Mapping[str, Any],
    key: str,
    *,
    allow_directory: bool = False,
    allow_missing: bool = False,
) -> Path:
    raw = str(_required(record, key)).strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ProfileError(f"{key} must be an absolute path: {raw!r}")
    if not allow_missing and not path.is_file() and not (allow_directory and path.is_dir()):
        raise ProfileError(f"{key} does not exist or is not a file: {path}")
    return path


def _sha256(record: Mapping[str, Any], key: str) -> str:
    value = str(_required(record, key)).strip().lower()
    if not SHA256_RE.fullmatch(value):
        raise ProfileError(f"{key} must be a lowercase 64-character SHA-256")
    return value


@dataclass(frozen=True)
class ServiceSafetyBudget:
    service_type: str
    memory_max_mib: int
    memory_swap_max_mib: int
    tasks_max: int
    kill_signal: str
    timeout_stop_sec: int


@dataclass(frozen=True)
class AdmissionSafetyBudget:
    min_ram_available_mib: int
    max_swap_used_mib: int
    max_swap_growth_mib: int
    min_total_vram_mib: int
    min_free_vram_before_start_mib: int
    driver_vram_reserve_mib: int
    vision_vram_reserve_mib: int
    safety_vram_reserve_mib: int

    @property
    def post_ready_min_free_vram_mib(self) -> int:
        return (
            self.driver_vram_reserve_mib
            + self.vision_vram_reserve_mib
            + self.safety_vram_reserve_mib
        )


@dataclass(frozen=True)
class SafetyBudget:
    service: ServiceSafetyBudget
    admission: AdmissionSafetyBudget


def _parse_safety(record: Mapping[str, Any], profile_id: str, *, runtime: str) -> SafetyBudget:
    raw_safety = _required(record, "safety")
    if not isinstance(raw_safety, dict):
        raise ProfileError(f"safety must be an object for {profile_id!r}")
    raw_service = _required(raw_safety, "service")
    raw_admission = _required(raw_safety, "admission")
    if not isinstance(raw_service, dict):
        raise ProfileError(f"safety.service must be an object for {profile_id!r}")
    if not isinstance(raw_admission, dict):
        raise ProfileError(f"safety.admission must be an object for {profile_id!r}")

    service_type = str(_required(raw_service, "type")).strip()
    kill_signal = str(_required(raw_service, "kill_signal")).strip()
    if service_type != "exec":
        raise ProfileError(f"safety.service.type must be 'exec' for {profile_id!r}")
    if kill_signal != "SIGINT":
        raise ProfileError(f"safety.service.kill_signal must be 'SIGINT' for {profile_id!r}")

    service = ServiceSafetyBudget(
        service_type=service_type,
        memory_max_mib=_positive_int(raw_service, "memory_max_mib"),
        memory_swap_max_mib=_nonnegative_int(raw_service, "memory_swap_max_mib"),
        tasks_max=_positive_int(raw_service, "tasks_max"),
        kill_signal=kill_signal,
        timeout_stop_sec=_positive_int(raw_service, "timeout_stop_sec"),
    )
    if service.memory_swap_max_mib > service.memory_max_mib:
        raise ProfileError(
            f"safety.service.memory_swap_max_mib cannot exceed memory_max_mib "
            f"for {profile_id!r}"
        )
    memory_max_limit = (
        MAX_STRATA_MEMORY_MAX_MIB if runtime == "strata" else MAX_MEMORY_MAX_MIB
    )
    if runtime == "strata" and service.memory_max_mib < MIN_STRATA_MEMORY_MAX_MIB:
        raise ProfileError(
            f"safety.service.memory_max_mib for Strata profile {profile_id!r} "
            f"must be at least {MIN_STRATA_MEMORY_MAX_MIB} MiB"
        )
    if service.memory_max_mib > memory_max_limit:
        raise ProfileError(
            f"safety.service.memory_max_mib for {profile_id!r} exceeds "
            f"fail-closed limit {memory_max_limit}"
        )
    if service.memory_swap_max_mib > MAX_MEMORY_SWAP_MAX_MIB:
        raise ProfileError(
            f"safety.service.memory_swap_max_mib for {profile_id!r} exceeds "
            f"fail-closed limit {MAX_MEMORY_SWAP_MAX_MIB}"
        )
    if service.tasks_max > MAX_TASKS_MAX:
        raise ProfileError(
            f"safety.service.tasks_max for {profile_id!r} exceeds fail-closed "
            f"limit {MAX_TASKS_MAX}"
        )
    if service.timeout_stop_sec > MAX_TIMEOUT_STOP_SEC:
        raise ProfileError(
            f"safety.service.timeout_stop_sec for {profile_id!r} exceeds "
            f"fail-closed limit {MAX_TIMEOUT_STOP_SEC}"
        )

    admission = AdmissionSafetyBudget(
        min_ram_available_mib=_positive_int(
            raw_admission, "min_ram_available_mib"
        ),
        max_swap_used_mib=_nonnegative_int(raw_admission, "max_swap_used_mib"),
        max_swap_growth_mib=_nonnegative_int(
            raw_admission, "max_swap_growth_mib"
        ),
        min_total_vram_mib=_positive_int(raw_admission, "min_total_vram_mib"),
        min_free_vram_before_start_mib=_positive_int(
            raw_admission, "min_free_vram_before_start_mib"
        ),
        driver_vram_reserve_mib=_positive_int(
            raw_admission, "driver_vram_reserve_mib"
        ),
        vision_vram_reserve_mib=_nonnegative_int(
            raw_admission, "vision_vram_reserve_mib"
        ),
        safety_vram_reserve_mib=_positive_int(
            raw_admission, "safety_vram_reserve_mib"
        ),
    )
    if runtime == "strata" and admission.min_ram_available_mib < MIN_STRATA_RAM_AVAILABLE_MIB:
        raise ProfileError(
            f"safety.admission.min_ram_available_mib for Strata profile {profile_id!r} "
            f"must be at least {MIN_STRATA_RAM_AVAILABLE_MIB} MiB"
        )
    if runtime == "strata" and admission.min_ram_available_mib > service.memory_max_mib:
        raise ProfileError(
            f"Strata min_ram_available_mib cannot exceed service memory_max_mib for {profile_id!r}"
        )
    admission_lower_bounds = {
        "min_ram_available_mib": (
            MIN_STRATA_RAM_AVAILABLE_MIB if runtime == "strata" else MIN_ADMISSION_RAM_AVAILABLE_MIB
        ),
        "min_total_vram_mib": MIN_ADMISSION_TOTAL_VRAM_MIB,
        "min_free_vram_before_start_mib": MIN_ADMISSION_FREE_VRAM_MIB,
    }
    for field_name, lower_bound in admission_lower_bounds.items():
        if getattr(admission, field_name) < lower_bound:
            raise ProfileError(
                f"safety.admission.{field_name} for {profile_id!r} is below "
                f"fail-closed floor {lower_bound}"
            )
    admission_upper_bounds = {
        "max_swap_used_mib": MAX_ADMISSION_SWAP_USED_MIB,
        "max_swap_growth_mib": MAX_ADMISSION_SWAP_GROWTH_MIB,
    }
    for field_name, upper_bound in admission_upper_bounds.items():
        if getattr(admission, field_name) > upper_bound:
            raise ProfileError(
                f"safety.admission.{field_name} for {profile_id!r} exceeds "
                f"fail-closed limit {upper_bound}"
            )
    if admission.min_free_vram_before_start_mib > admission.min_total_vram_mib:
        raise ProfileError(
            f"safety.admission.min_free_vram_before_start_mib for {profile_id!r} "
            "cannot exceed min_total_vram_mib"
        )

    return SafetyBudget(service=service, admission=admission)


@dataclass(frozen=True)
class Profile:
    id: str
    runtime: str
    alias: str
    aliases: tuple[str, ...]
    display_name: str
    role: str
    capabilities: frozenset[str]
    native_vision: bool
    restore_profile: str | None
    runtime_build: str | None
    server_binary: Path
    server_sha256: str
    model_path: Path
    model_sha256: str
    mmproj_path: Path | None
    mmproj_sha256: str | None
    context_size: int
    parallel: int
    batch_size: int | None
    ubatch_size: int | None
    cache_type_k: str | None
    cache_type_v: str | None
    flash_attention: str | None
    gpu_layers: str | None
    tensor_parallel_size: int
    gpu_memory_utilization: float | None
    kv_cache_dtype: str | None
    tool_call_parser: str | None
    tool_parser_plugin: str | None
    chat_template: str | None
    limit_mm_per_prompt: Mapping[str, int] | None
    allowed_local_media_path: str | None
    fit: bool | None
    fit_target_mib: int | None
    fit_ctx: int | None
    ctx_checkpoints: int | None
    cache_ram_mib: int | None
    startup_timeout: int
    safety: SafetyBudget
    environment: Mapping[str, str]
    extra_args: tuple[str, ...]
    model_draft: Path | None
    moe_cache_slots_limit: int
    max_output_tokens: int | None = None
    autotune: Mapping[str, Any] | None = None
    bundle_root: Path | None = None
    source_repository: Path | None = None
    provenance_file: Path | None = None
    source_branch: str | None = None
    source_commit: str | None = None
    dirty: bool | None = None
    selected_profile_ids: tuple[str, ...] = ()
    builtin_tools_available: bool = True
    warnings: tuple[str, ...] = ()
    degraded_features: tuple[str, ...] = ()
    declared_extra_args: tuple[str, ...] = ()
    effective_extra_args: tuple[str, ...] = ()
    effective_model_draft: Path | None = None
    model_draft_sha256: str | None = None
    deployment_revision: str | None = None
    strata_artifacts_manifest_path: Path | None = None
    strata_artifacts_manifest_sha256: str | None = None
    strata_launch_config_path: Path | None = None
    strata_launch_config_sha256: str | None = None
    strata_artifact_files: tuple[tuple[Path, str], ...] = ()
    strata_engine_binary: Path | None = None
    strata_engine_sha256: str | None = None
    strata_model_name: str | None = None
    basalt_config_path: Path | None = None
    basalt_config_sha256: str | None = None

    @property
    def backend_model_name(self) -> str:
        """Return the backend's model identifier behind the public profile alias."""
        return self.strata_model_name or self.alias

    @property
    def is_transient(self) -> bool:
        return self.role == "transient"

    @property
    def slot_context_window(self) -> int:
        """Return the context window available to one parallel slot."""

        return self.context_size // self.parallel

    @property
    def default_max_output_tokens(self) -> int:
        """Return the swarm-friendly inverse-scaled output default."""

        if self.max_output_tokens is not None:
            return self.max_output_tokens
        return max(1, self.slot_context_window // (self.parallel**2))

    @property
    def args(self) -> tuple[str, ...]:
        """Controller-compatible arguments excluding controller-owned model/alias."""
        args = self.server_args(host="127.0.0.1", port=10000)
        return args[4:] if self.runtime == "llamacpp" else args

    def server_args(self, *, host: str, port: int) -> tuple[str, ...]:
        return get_runtime(self.runtime).build_args(self, host=host, port=port)

    def command(
        self,
        *,
        host: str,
        port: int,
        managed_args: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return the complete executable command for the selected runtime."""

        return get_runtime(self.runtime).build_command(
            self, host=host, port=port, managed_args=managed_args
        )


class ProfileRegistry:
    def __init__(
        self,
        profiles: tuple[Profile, ...],
        *,
        runtime_catalog: RuntimeCatalog | None = None,
    ):
        by_id = {profile.id: profile for profile in profiles}

        aliases: dict[str, Profile] = {}
        for profile in profiles:
            for alias in (profile.id, profile.alias, *profile.aliases):
                key = alias.casefold()
                owner = aliases.get(key)
                if owner is not None and owner.id != profile.id:
                    raise ProfileError(
                        f"duplicate alias {alias!r}: {owner.id!r} and {profile.id!r}"
                    )
                aliases[key] = profile

        for profile in profiles:
            if profile.restore_profile and profile.restore_profile not in by_id:
                raise ProfileError(
                    f"restore_profile for {profile.id!r} does not exist: "
                    f"{profile.restore_profile!r}"
                )

        self._profiles = profiles
        self._by_id = MappingProxyType(by_id)
        self._by_alias = MappingProxyType(aliases)
        self._runtime_catalog = runtime_catalog

    @classmethod
    def from_path(
        cls, path: Path | str, *, allow_missing_artifacts: bool = False
    ) -> "ProfileRegistry":
        manifest_path = Path(path)
        try:
            from .profile_config import load_manifest

            payload = load_manifest(manifest_path)
        except ProfileError:
            raise
        except OSError as exc:
            raise ProfileError(f"cannot read profile manifest {manifest_path}: {exc}") from exc
        schema = payload.get("schema_version")
        if schema not in {1, 2}:
            raise ProfileError(f"unsupported profile manifest schema_version: {schema!r}")
        raw_profiles = payload.get("profiles")
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise ProfileError("profile manifest must contain a non-empty profiles list")

        try:
            runtime_catalog = external_catalog_for_manifest(
                payload,
                allow_missing_artifacts=allow_missing_artifacts,
            )
            runtime_builds = (
                runtime_catalog.builds
                if runtime_catalog is not None
                else parse_runtime_builds(
                    payload.get("runtime_builds", []),
                    allow_missing_artifacts=allow_missing_artifacts,
                )
            )
        except RuntimeBuildError as error:
            raise ProfileError(str(error)) from error

        profiles: list[Profile] = []
        for item in raw_profiles:
            try:
                if isinstance(item, dict):
                    resolved_item, resolved_build = resolve_profile_runtime(
                        payload,
                        item,
                        allow_missing_artifacts=allow_missing_artifacts,
                    )
                else:
                    resolved_item, resolved_build = item, None
                profiles.append(
                    _parse_profile(
                        resolved_item,
                        allow_missing_artifacts=allow_missing_artifacts,
                        runtime_builds=runtime_builds,
                        runtime_catalog=runtime_catalog,
                        resolved_runtime_build=resolved_build,
                    )
                )
            except RuntimeBuildError as error:
                profile_id = (
                    item.get("id", "<unknown>")
                    if isinstance(item, dict)
                    else "<unknown>"
                )
                raise ProfileError(
                    f"{error} for profile {profile_id!r}"
                ) from error
        parsed_profiles = tuple(profiles)
        ids = [profile.id for profile in parsed_profiles]
        if len(ids) != len(set(ids)):
            raise ProfileError("duplicate profile id")
        return cls(parsed_profiles, runtime_catalog=runtime_catalog)

    @property
    def profiles(self) -> tuple[Profile, ...]:
        return self._profiles

    @property
    def runtime_catalog(self) -> RuntimeCatalog | None:
        return self._runtime_catalog

    @property
    def runtime_catalog_path(self) -> Path | None:
        return self._runtime_catalog.path if self._runtime_catalog is not None else None

    @property
    def deployment_revision(self) -> str | None:
        return (
            self._runtime_catalog.deployment_revision
            if self._runtime_catalog is not None
            else None
        )

    @property
    def runtime_builds(self) -> tuple[RuntimeBuild, ...]:
        if self._runtime_catalog is not None:
            return self._runtime_catalog.builds
        return tuple(
            RuntimeBuild(
                id=profile.runtime_build or "",
                runtime=profile.runtime,
                display_name=profile.display_name,
                server_binary=profile.server_binary,
                server_sha256=profile.server_sha256,
            )
            for profile in self._profiles
            if profile.runtime_build
        )

    def get(self, profile_id: str) -> Profile:
        try:
            return self._by_id[profile_id]
        except KeyError as exc:
            raise ProfileError(f"unknown profile: {profile_id}") from exc

    def profile_snapshot_identity(self, profile_id: str) -> dict[str, Any]:
        """Return a secret-safe fingerprint and pinned facts for one profile.

        The full resolved profile is hashed in memory; the durable operation
        record keeps only the digest, artifact pins, safety budget, and catalog
        revision. A worker must compare this identity before executing rather
        than silently resolving changed profile data.
        """
        profile = self.get(profile_id)
        canonical_profile = _profile_snapshot_value(profile)
        encoded = json.dumps(
            canonical_profile,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return {
            "schema_version": 1,
            "profile_id": profile.id,
            "fingerprint": hashlib.sha256(encoded).hexdigest(),
            "catalog_revision": self.deployment_revision,
            "runtime_build": profile.runtime_build,
            "alias": profile.alias,
            "role": profile.role,
            "capabilities": sorted(profile.capabilities),
            "artifact_sha256": {
                "server": profile.server_sha256,
                "model": profile.model_sha256,
                "mmproj": profile.mmproj_sha256,
                "draft": profile.model_draft_sha256,
                "strata_manifest": profile.strata_artifacts_manifest_sha256,
                "strata_launch_config": profile.strata_launch_config_sha256,
                "strata_engine": profile.strata_engine_sha256,
            },
            "safety": _profile_snapshot_value(profile.safety),
        }

    def snapshot_revision_identity(self) -> dict[str, Any]:
        """Fingerprint the complete resolved profile/catalog revision without secrets."""
        profiles = [
            self.profile_snapshot_identity(profile.id)
            for profile in sorted(self._profiles, key=lambda item: item.id)
        ]
        canonical = {
            "deployment_revision": self.deployment_revision,
            "profiles": profiles,
        }
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return {
            "schema_version": 1,
            "deployment_revision": self.deployment_revision,
            "fingerprint": hashlib.sha256(encoded).hexdigest(),
            "profile_count": len(profiles),
        }

    def try_resolve(self, alias: object) -> Profile | None:
        value = str(alias or "").strip()
        if not value:
            return None
        return self._by_alias.get(value.casefold())

    def resolve(self, alias: object) -> Profile:
        profile = self.try_resolve(alias)
        if profile is None:
            raise ProfileError(f"unknown model alias: {alias}")
        return profile


def _profile_snapshot_value(value: Any) -> Any:
    """Convert resolved profile facts to deterministic secret-safe JSON data."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _profile_snapshot_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _profile_snapshot_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        converted = [_profile_snapshot_value(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return [_profile_snapshot_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported profile snapshot value: {type(value).__name__}")


def _optional_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strata_relative_artifact(
    root: Path,
    raw_path: object,
    *,
    profile_id: str,
    label: str,
    directory: bool = False,
    allow_missing: bool = False,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path or chr(92) in raw_path:
        raise ProfileError(f"{label} must be a safe relative path for {profile_id!r}")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ProfileError(f"{label} path escapes its artifact root for {profile_id!r}")
    candidate = root.joinpath(relative)
    root_resolved = root.resolve()
    for parent in (candidate, *candidate.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ProfileError(f"{label} must not use symlinks for {profile_id!r}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ProfileError(f"{label} path escapes its artifact root for {profile_id!r}")
    if not allow_missing and not (
        resolved.is_dir() if directory else resolved.is_file()
    ):
        raise ProfileError(f"{label} does not exist for {profile_id!r}: {resolved}")
    return resolved


def _parse_strata_runtime_sources(
    root: Path,
    entries: object,
    *,
    profile_id: str,
    allow_missing: bool,
) -> tuple[tuple[Path, str], ...]:
    if not root.is_dir() or root.is_symlink():
        raise ProfileError(f"Strata working directory is missing or symlinked for {profile_id!r}")
    expected_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or "__pycache__" in path.parts or path.suffix == ".pyc":
            raise ProfileError(f"Strata working directory contains a symlink or bytecode file for {profile_id!r}")
        if path.is_file() and path.suffix == ".py":
            expected_paths.add(path.relative_to(root).as_posix())
    if not isinstance(entries, list) or not entries:
        raise ProfileError(f"Strata launch config must pin its Python source inventory for {profile_id!r}")
    observed: dict[str, tuple[Path, str]] = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ProfileError(f"invalid Strata runtime source entry for {profile_id!r}")
        raw_path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(raw_path, str) or Path(raw_path).suffix != ".py":
            raise ProfileError(f"Strata runtime source path must name a Python file for {profile_id!r}")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ProfileError(f"invalid Strata runtime source hash for {profile_id!r}")
        if raw_path in observed:
            raise ProfileError(f"duplicate Strata runtime source path for {profile_id!r}: {raw_path}")
        path = _strata_relative_artifact(
            root, raw_path, profile_id=profile_id, label="runtime Python source",
            allow_missing=allow_missing,
        )
        if path.exists() and _optional_file_sha256(path) != digest:
            raise ProfileError(f"Strata runtime source SHA mismatch for {profile_id!r}: {path}")
        observed[raw_path] = (path, digest)
    if set(observed) != expected_paths:
        raise ProfileError(f"Strata runtime source inventory is incomplete or contains extras for {profile_id!r}")
    return tuple(observed[key] for key in sorted(observed))


def _verify_strata_source_checkout(
    source_repository: Path,
    expected_commit: str,
    profile_id: str,
) -> set[str]:
    git = shutil.which("git", path=os.defpath)
    if git is None:
        raise ProfileError(f"cannot verify Strata source provenance for {profile_id!r}: git is unavailable")
    env = {
        "PATH": os.defpath,
        "HOME": str(Path.home()),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
    }

    def git_read(*arguments: str) -> str:
        try:
            result = subprocess.run(
                [
                    git,
                    "-C",
                    str(source_repository),
                    "-c",
                    "core.fsmonitor=false",
                    "-c",
                    "core.untrackedCache=false",
                    *arguments,
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                stdin=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ProfileError(f"cannot verify Strata source provenance for {profile_id!r}: {error}") from error
        if result.returncode != 0:
            raise ProfileError(f"Strata source provenance check failed for {profile_id!r}")
        return result.stdout

    if Path(git_read("rev-parse", "--show-toplevel").strip()).resolve() != source_repository.resolve():
        raise ProfileError(f"Strata source_repository is not a Git worktree root for {profile_id!r}")
    if git_read("rev-parse", "--verify", "HEAD").strip() != expected_commit:
        raise ProfileError(f"Strata source checkout HEAD does not match the pinned commit for {profile_id!r}")
    if git_read("status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ProfileError(f"Strata source checkout is dirty for {profile_id!r}")
    tracked_paths = git_read("ls-files", "-z").split("\0")
    return {path for path in tracked_paths if path.endswith(".py")}


def _strata_hash_entry(item: object, profile_id: str, label: str) -> tuple[str, int]:
    if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
        raise ProfileError(f"invalid {label} entry for {profile_id!r}")
    digest = item.get("sha256")
    size = item.get("size")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ProfileError(f"invalid {label} sha256 for {profile_id!r}")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ProfileError(f"invalid {label} size for {profile_id!r}")
    return digest, size


def _parse_strata_mtp_artifacts(
    raw: object,
    *,
    model_root: Path,
    profile_id: str,
    allow_missing_artifacts: bool,
) -> tuple[Path, tuple[tuple[Path, str], ...]]:
    required = {"repository", "revision", "path", "runtime_path", "files"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ProfileError(f"Strata profile {profile_id!r} requires a pinned MTP runtime")
    repository = raw.get("repository")
    revision = raw.get("revision")
    if repository != "Qwen/Qwen3.8-Flash-Next" or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ProfileError(f"Strata MTP source repository/revision is not pinned for {profile_id!r}")
    mtp_root = _strata_relative_artifact(
        model_root, raw.get("path"), profile_id=profile_id,
        label="MTP root", directory=True, allow_missing=allow_missing_artifacts,
    )
    runtime_root = _strata_relative_artifact(
        model_root, raw.get("runtime_path"), profile_id=profile_id,
        label="MTP runtime directory", directory=True, allow_missing=allow_missing_artifacts,
    )
    if not runtime_root.is_relative_to(mtp_root):
        raise ProfileError(f"Strata MTP runtime directory must be inside its pinned root for {profile_id!r}")
    raw_files = raw.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ProfileError(f"Strata MTP file inventory is empty for {profile_id!r}")
    artifacts: list[tuple[Path, str]] = []
    file_records: dict[str, tuple[Path, str, int]] = {}
    for item in raw_files:
        digest, size = _strata_hash_entry(item, profile_id, "MTP artifact")
        relative = item.get("path")
        path = _strata_relative_artifact(
            model_root, relative, profile_id=profile_id,
            label="MTP artifact", allow_missing=allow_missing_artifacts,
        )
        if not isinstance(relative, str) or not relative.startswith(str(raw["path"]) + "/"):
            raise ProfileError(f"Strata MTP artifact is outside its root for {profile_id!r}")
        if relative in file_records:
            raise ProfileError(f"duplicate Strata MTP artifact for {profile_id!r}: {relative}")
        if path.exists() and path.stat().st_size != size:
            raise ProfileError(f"Strata MTP artifact size mismatch for {profile_id!r}: {path}")
        file_records[relative] = (path, digest, size)
        artifacts.append((path, digest))
    required_files = {
        f"{raw['path']}/mtp-inventory.json",
        f"{raw['path']}/mtp-manifest.json",
        f"{raw['path']}/mtp-q2_0.gguf",
        f"{raw['runtime_path']}/experts.bin",
        f"{raw['runtime_path']}/dense.bin",
        f"{raw['runtime_path']}/dense.txt",
    }
    if not required_files <= file_records.keys():
        missing = sorted(required_files - file_records.keys())
        raise ProfileError(f"Strata MTP artifact set is incomplete for {profile_id!r}: {', '.join(missing)}")
    if not allow_missing_artifacts:
        inventory_path = file_records[f"{raw['path']}/mtp-inventory.json"][0]
        manifest_path = file_records[f"{raw['path']}/mtp-manifest.json"][0]
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            tensor_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProfileError(f"cannot read pinned Strata MTP metadata for {profile_id!r}: {error}") from error
        expected_repo = f"https://huggingface.co/{repository}/resolve/{revision}/"
        if not isinstance(inventory, dict) or inventory.get("repo") != expected_repo:
            raise ProfileError(f"Strata MTP inventory is not pinned to the declared revision for {profile_id!r}")
        inventory_tensors = inventory.get("tensors")
        if not isinstance(inventory_tensors, list) or not isinstance(tensor_manifest, list) or not tensor_manifest:
            raise ProfileError(f"Strata MTP tensor inventory/manifest is empty for {profile_id!r}")
        inventory_names = {item.get("name") for item in inventory_tensors if isinstance(item, dict)}
        manifest_names: set[str] = set()
        for item in tensor_manifest:
            if not isinstance(item, dict):
                raise ProfileError(f"Strata MTP tensor manifest entry is invalid for {profile_id!r}")
            name, relative = item.get("name"), item.get("file")
            digest, size = item.get("sha256"), item.get("bytes")
            if not isinstance(name, str) or not isinstance(relative, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ProfileError(f"Strata MTP tensor manifest entry is unpinned for {profile_id!r}")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or name in manifest_names:
                raise ProfileError(f"Strata MTP tensor manifest entry is duplicated/invalid for {profile_id!r}")
            manifest_names.add(name)
            source_path = f"{raw['path']}/{relative}"
            source = file_records.get(source_path)
            if source is None or source[1] != digest or source[2] != size:
                raise ProfileError(f"Strata MTP raw tensor pin mismatch for {profile_id!r}: {name}")
        if manifest_names != inventory_names:
            raise ProfileError(f"Strata MTP inventory and tensor manifest disagree for {profile_id!r}")
    return runtime_root, tuple(artifacts)


def _parse_strata_artifacts(
    record: Mapping[str, Any],
    *,
    profile_id: str,
    model_path: Path,
    model_sha256: str,
    context_size: int,
    runtime_build: RuntimeBuild | None,
    allow_missing_artifacts: bool,
) -> dict[str, Any]:
    if runtime_build is None or runtime_build.bundle_root is None:
        raise ProfileError(f"Strata profile {profile_id!r} requires a runtime bundle")
    bundle_root = runtime_build.bundle_root.resolve()
    source_repository = runtime_build.source_repository
    source_commit = runtime_build.source_commit
    if source_repository is None or (not allow_missing_artifacts and not source_repository.is_dir()):
        raise ProfileError(f"Strata runtime source repository is missing for {profile_id!r}")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ProfileError(f"Strata runtime source commit must be pinned for {profile_id!r}")
    if runtime_build.dirty is not False:
        raise ProfileError(f"Strata runtime bundle must be built from a clean source tree for {profile_id!r}")
    source_python_paths = None
    if source_repository.is_dir():
        source_python_paths = _verify_strata_source_checkout(source_repository, source_commit, profile_id)
    manifest_path = _absolute_existing_path(
        record, "strata_artifacts_manifest_path", allow_missing=allow_missing_artifacts
    )
    manifest_sha256 = _sha256(record, "strata_artifacts_manifest_sha256")
    launch_path = _absolute_existing_path(
        record, "strata_launch_config_path", allow_missing=allow_missing_artifacts
    )
    launch_sha256 = _sha256(record, "strata_launch_config_sha256")
    if manifest_path.is_symlink() or launch_path.is_symlink():
        raise ProfileError(f"Strata metadata files must not be symlinks for {profile_id!r}")
    model_root = manifest_path.parent.resolve()
    if manifest_path.parent.is_symlink() or launch_path.parent.is_symlink():
        raise ProfileError(f"Strata metadata directory must not be a symlink for {profile_id!r}")
    if launch_path.parent.resolve() != model_root:
        raise ProfileError(f"Strata launch config must sit beside its artifact manifest for {profile_id!r}")
    if manifest_path.is_symlink() or launch_path.is_symlink() or not manifest_path.is_file() or not launch_path.is_file():
        raise ProfileError(f"Strata metadata files must be regular files for {profile_id!r}")
    try:
        manifest_bytes = manifest_path.read_bytes()
        launch_bytes = launch_path.read_bytes()
    except OSError as error:
        raise ProfileError(f"cannot read Strata metadata for {profile_id!r}: {error}") from error
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ProfileError(f"Strata artifact manifest hash mismatch for {profile_id!r}")
    if hashlib.sha256(launch_bytes).hexdigest() != launch_sha256:
        raise ProfileError(f"Strata launch config hash mismatch for {profile_id!r}")
    try:
        manifest = json.loads(manifest_bytes)
        launch = json.loads(launch_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProfileError(f"cannot decode Strata metadata for {profile_id!r}: {error}") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "runtime_family", "model", "pack"
    }:
        raise ProfileError(f"invalid Strata artifact manifest for {profile_id!r}")
    if manifest.get("schema_version") != 1 or manifest.get("runtime_family") != "strata":
        raise ProfileError(f"unsupported Strata artifact manifest for {profile_id!r}")
    model = manifest.get("model")
    pack = manifest.get("pack")
    if not isinstance(model, dict) or set(model) != {
        "name", "quant", "license", "repository", "revision", "shards"
    }:
        raise ProfileError(f"invalid Strata model metadata for {profile_id!r}")
    if model.get("quant") != "IQ2_XS":
        raise ProfileError(f"Strata runtime currently requires IQ2_XS for {profile_id!r}")
    revision = model.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ProfileError(f"Strata model revision must be a pinned commit for {profile_id!r}")
    shards = model.get("shards")
    if not isinstance(shards, list) or len(shards) != 2:
        raise ProfileError(f"Strata model must declare exactly two shards for {profile_id!r}")
    shard_paths: list[Path] = []
    artifact_files: list[tuple[Path, str]] = []
    for index, shard in enumerate(shards):
        digest, size = _strata_hash_entry(shard, profile_id, f"model shard {index + 1}")
        path = _strata_relative_artifact(
            model_root, shard.get("path"), profile_id=profile_id,
            label=f"model shard {index + 1}", allow_missing=allow_missing_artifacts,
        )
        if path.exists() and path.stat().st_size != size:
            raise ProfileError(f"Strata model shard size mismatch for {profile_id!r}: {path}")
        shard_paths.append(path)
        if index > 0:
            artifact_files.append((path, digest))
    if model_path.is_symlink():
        raise ProfileError(f"Strata model shard must not be a symlink for {profile_id!r}")
    if shard_paths[0] != model_path.resolve() or shards[0]["sha256"] != model_sha256:
        raise ProfileError(f"Strata model_path must be the pinned first shard for {profile_id!r}")
    if not isinstance(pack, dict) or set(pack) != {
        "format", "path", "source_repository", "source_commit", "mtp", "files"
    }:
        raise ProfileError(f"invalid Strata pack metadata for {profile_id!r}")
    if pack.get("format") != "strata-native-iq":
        raise ProfileError(f"Strata profile {profile_id!r} requires a native IQ pack")
    source_commit = pack.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ProfileError(f"Strata pack source commit must be pinned for {profile_id!r}")
    if source_commit != runtime_build.source_commit:
        raise ProfileError(f"Strata pack/runtime source commit mismatch for {profile_id!r}")
    pack_path = _strata_relative_artifact(
        model_root, pack.get("path"), profile_id=profile_id,
        label="pack directory", directory=True, allow_missing=allow_missing_artifacts,
    )
    pack_files = pack.get("files")
    if not isinstance(pack_files, list) or not pack_files:
        raise ProfileError(f"Strata pack file list is empty for {profile_id!r}")
    required_pack_files = {
        f"{pack.get('path')}/dense.bin",
        f"{pack.get('path')}/index.txt",
        f"{pack.get('path')}/native_experts.txt",
        f"{pack.get('path')}/tokenizer/vocab.json",
        f"{pack.get('path')}/tokenizer/merges.txt",
        f"{pack.get('path')}/tokenizer/chat_template.jinja",
    }
    observed_pack_files: set[str] = set()
    for item in pack_files:
        digest, size = _strata_hash_entry(item, profile_id, "pack file")
        relative_text = item.get("path")
        path = _strata_relative_artifact(
            model_root, relative_text, profile_id=profile_id,
            label="pack file", allow_missing=allow_missing_artifacts,
        )
        if not isinstance(relative_text, str) or not relative_text.startswith(str(pack.get("path")) + "/"):
            raise ProfileError(f"Strata pack file is outside the pack directory for {profile_id!r}")
        if relative_text in observed_pack_files:
            raise ProfileError(f"duplicate Strata pack file for {profile_id!r}: {relative_text}")
        observed_pack_files.add(relative_text)
        if path.exists() and path.stat().st_size != size:
            raise ProfileError(f"Strata pack file size mismatch for {profile_id!r}: {path}")
        artifact_files.append((path, digest))
    if not required_pack_files <= observed_pack_files:
        missing = sorted(required_pack_files - observed_pack_files)
        raise ProfileError(f"Strata pack is missing required files for {profile_id!r}: {', '.join(missing)}")
    mtp_runtime_path, mtp_files = _parse_strata_mtp_artifacts(
        pack.get("mtp"),
        model_root=model_root,
        profile_id=profile_id,
        allow_missing_artifacts=allow_missing_artifacts,
    )
    artifact_files.extend(mtp_files)
    if not isinstance(launch, dict) or set(launch) != {
        "schema_version", "model_name", "engine_binary", "engine_sha256",
        "expert_profile_sha256", "engine_args", "working_directory", "tokenizer_path",
        "runtime_files",
    }:
        forbidden = sorted(set(launch) & {"host", "api_key", "mcp", "mcp_servers", "mcp_config"}) if isinstance(launch, dict) else []
        detail = f" (forbidden fields: {', '.join(forbidden)})" if forbidden else ""
        raise ProfileError(f"invalid or unsafe Strata launch config for {profile_id!r}{detail}")
    if launch.get("schema_version") != 1:
        raise ProfileError(f"unsupported Strata launch config schema for {profile_id!r}")
    model_name = launch.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ProfileError(f"Strata launch config requires a non-empty model_name for {profile_id!r}")
    engine_binary = Path(str(launch.get("engine_binary", ""))).expanduser()
    if not engine_binary.is_absolute() or engine_binary.is_symlink():
        raise ProfileError(f"Strata engine_binary must be an absolute non-symlink path for {profile_id!r}")
    engine_binary = engine_binary.resolve()
    if not engine_binary.is_relative_to(bundle_root):
        raise ProfileError(f"Strata engine_binary must be inside its bundle for {profile_id!r}")
    engine_sha256 = launch.get("engine_sha256")
    if not isinstance(engine_sha256, str) or not SHA256_RE.fullmatch(engine_sha256):
        raise ProfileError(f"invalid Strata engine sha256 for {profile_id!r}")
    if not allow_missing_artifacts and (not engine_binary.is_file() or not os.access(engine_binary, os.X_OK)):
        raise ProfileError(f"Strata engine is missing or not executable for {profile_id!r}: {engine_binary}")
    if engine_binary.is_file():
        artifact_files.append((engine_binary, engine_sha256))
    expert_profile_sha256 = launch.get("expert_profile_sha256")
    if not isinstance(expert_profile_sha256, str) or not SHA256_RE.fullmatch(expert_profile_sha256):
        raise ProfileError(f"invalid Strata expert profile sha256 for {profile_id!r}")
    expert_profile_path: Path | None = None
    working_directory = Path(str(launch.get("working_directory", ""))).expanduser()
    if (
        not working_directory.is_absolute()
        or working_directory.is_symlink()
        or not working_directory.resolve().is_relative_to(bundle_root)
    ):
        raise ProfileError(f"Strata working_directory must be inside its bundle for {profile_id!r}")
    if not allow_missing_artifacts and not working_directory.is_dir():
        raise ProfileError(f"Strata working_directory is missing for {profile_id!r}")
    runtime_source_files = _parse_strata_runtime_sources(
        working_directory,
        launch.get("runtime_files"),
        profile_id=profile_id,
        allow_missing=allow_missing_artifacts,
    )
    if source_python_paths is not None:
        bundled_source_paths = {
            path.relative_to(working_directory).as_posix()
            for path, _expected_sha in runtime_source_files
        }
        if bundled_source_paths != source_python_paths:
            raise ProfileError(f"Strata bundle Python inventory differs from its pinned checkout for {profile_id!r}")
    if source_repository is not None and source_repository.is_dir():
        for bundled_path, expected_source_sha in runtime_source_files:
            relative = bundled_path.relative_to(working_directory)
            source_path = source_repository / relative
            if source_path.is_symlink() or not source_path.is_file():
                raise ProfileError(f"Strata runtime source is absent from its pinned checkout for {profile_id!r}: {relative}")
            if _optional_file_sha256(source_path) != expected_source_sha:
                raise ProfileError(f"Strata runtime source differs from its pinned checkout for {profile_id!r}: {relative}")
    artifact_files.extend(runtime_source_files)
    tokenizer_path = Path(str(launch.get("tokenizer_path", ""))).expanduser()
    if (
        not tokenizer_path.is_absolute()
        or tokenizer_path.is_symlink()
        or tokenizer_path.resolve() != (pack_path / "tokenizer").resolve()
    ):
        raise ProfileError(f"Strata tokenizer_path must select the pinned pack tokenizer for {profile_id!r}")
    if not allow_missing_artifacts and not tokenizer_path.is_dir():
        raise ProfileError(f"Strata tokenizer_path is missing for {profile_id!r}")
    raw_args = launch.get("engine_args")
    if not isinstance(raw_args, list) or not all(isinstance(value, str) and value for value in raw_args):
        raise ProfileError(f"Strata engine_args must be a non-empty string list for {profile_id!r}")
    allowed_values = {
        "--pack": str(pack_path),
        "--native": str(shard_paths[0]),
        "--ple-gguf": str(shard_paths[1]),
        "--expert-cache": "auto",
        "--prefill": "auto",
        "--max-context": str(context_size),
        "--spec": "4",
        "--spec-min-p": "0.5",
        "--mtp": str(mtp_runtime_path),
    }
    seen: set[str] = set()
    boolean_flags = {"--mmap-experts"}
    # optional: RAM (MiB) for parking the active conversation while another prompt runs
    optional_value_flags = {"--conversation-cache-mib", "--resident-budget-gib", "--kv"}
    uses_mmap_experts = False
    index = 0
    while index < len(raw_args):
        flag = raw_args[index]
        if flag in boolean_flags:
            if flag in seen:
                raise ProfileError(f"unsupported or duplicate Strata engine argument for {profile_id!r}: {flag}")
            seen.add(flag)
            uses_mmap_experts = True
            index += 1
            continue
        if flag not in {*allowed_values, "--kv-resident", "--expert-profile", *optional_value_flags} or flag in seen:
            raise ProfileError(f"unsupported or duplicate Strata engine argument for {profile_id!r}: {flag}")
        if index + 1 >= len(raw_args) or raw_args[index + 1].startswith("--"):
            raise ProfileError(f"Strata engine argument requires a value for {profile_id!r}: {flag}")
        value = raw_args[index + 1]
        if flag == "--kv-resident":
            if not value.isdecimal() or int(value) < MIN_STRATA_KV_RESIDENT_CELLS:
                raise ProfileError(
                    f"Strata KV streaming requires at least "
                    f"{MIN_STRATA_KV_RESIDENT_CELLS} resident cells for {profile_id!r}"
                )
        elif flag == "--conversation-cache-mib":
            if not value.isdecimal() or not 0 < int(value) <= 16384:
                raise ProfileError(f"Strata conversation cache must be 1-16384 MiB for {profile_id!r}")
        elif flag == "--resident-budget-gib":
            if not value.isdecimal() or not 0 < int(value) <= 64:
                raise ProfileError(f"Strata resident budget must be 1-64 GiB for {profile_id!r}")
        elif flag == "--kv":
            if value not in STRATA_KV_FORMATS:
                raise ProfileError(f"Strata KV format must be fp16, int8 or q4_0 for {profile_id!r}")
        elif flag in allowed_values and value != allowed_values[flag]:
            raise ProfileError(f"Strata engine argument does not match pinned profile data for {profile_id!r}: {flag}")
        if flag == "--expert-profile":
            raw_expert_profile = Path(value).expanduser()
            if not raw_expert_profile.is_absolute() or raw_expert_profile.is_symlink():
                raise ProfileError(f"Strata expert profile must be an absolute regular path for {profile_id!r}")
            expert_profile = raw_expert_profile.resolve()
            if not expert_profile.is_relative_to(bundle_root):
                raise ProfileError(f"Strata expert profile must be inside its bundle for {profile_id!r}")
            if not allow_missing_artifacts and not expert_profile.is_file():
                raise ProfileError(f"Strata expert profile is missing for {profile_id!r}")
            expert_profile_path = expert_profile
        seen.add(flag)
        index += 2
    required_flags = {*allowed_values, "--kv-resident", "--expert-profile"}
    if (
        not required_flags.issubset(seen)
        or seen - required_flags - boolean_flags - optional_value_flags
        or expert_profile_path is None
    ):
        raise ProfileError(f"Strata engine_args are missing required controls for {profile_id!r}")
    if uses_mmap_experts and f"{pack.get('path')}/experts.bin" not in observed_pack_files:
        raise ProfileError(f"Strata mmap-experts mode requires the pack's experts.bin for {profile_id!r}")
    if expert_profile_path.is_file():
        artifact_files.append((expert_profile_path, expert_profile_sha256))
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "launch_path": launch_path,
        "launch_sha256": launch_sha256,
        "model_name": model_name,
        "artifact_files": tuple(artifact_files),
        "engine_binary": engine_binary,
        "engine_sha256": engine_sha256,
    }


def _degrade_optional_moe_args(
    raw: object,
    profile_id: str,
    *,
    slots_limit: int,
) -> tuple[object, tuple[str, ...], tuple[str, ...]]:
    """Remove only an invalid optional MoE cache pair from effective argv."""
    if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
        return raw, (), ()
    args = list(raw)
    cache_flags = {"--moe-cache-profile", "--moe-cache-slots"}
    positions: dict[str, list[int]] = {flag: [] for flag in cache_flags}
    for index, value in enumerate(args):
        if value in cache_flags:
            positions[value].append(index)

    if not positions["--moe-cache-profile"]:
        return args, (), ()

    valid = all(len(positions[flag]) == 1 for flag in cache_flags)
    profile_index = positions["--moe-cache-profile"][0] if len(positions["--moe-cache-profile"]) == 1 else -1
    slots_index = positions["--moe-cache-slots"][0] if len(positions["--moe-cache-slots"]) == 1 else -1
    if valid:
        if profile_index + 1 >= len(args):
            valid = False
        else:
            cache_path = Path(args[profile_index + 1]).expanduser()
            valid = cache_path.is_absolute() and cache_path.is_file()
        if slots_index + 1 >= len(args):
            valid = False
        else:
            raw_slots = args[slots_index + 1]
            valid = valid and bool(re.fullmatch(r"[0-9]+", raw_slots)) and int(raw_slots) <= slots_limit

    if valid:
        return args, (), ()

    cleaned: list[str] = []
    index = 0
    while index < len(args):
        if args[index] in cache_flags:
            index += 2 if index + 1 < len(args) else 1
            continue
        cleaned.append(args[index])
        index += 1
    return (
        cleaned,
        (f"{profile_id}: optional MoE cache optimization is unavailable",),
        ("moe-cache",),
    )


def _parse_profile(
    raw: object,
    *,
    allow_missing_artifacts: bool = False,
    runtime_builds: tuple[RuntimeBuild, ...] = (),
    runtime_catalog: RuntimeCatalog | None = None,
    resolved_runtime_build: RuntimeBuild | None = None,
) -> Profile:
    if not isinstance(raw, dict):
        raise ProfileError("every profile must be a JSON object")
    record: Mapping[str, Any] = raw
    resolved_build = resolved_runtime_build
    if resolved_build is None and isinstance(record.get("runtime_build"), str):
        requested_build = record["runtime_build"].strip().casefold()
        resolved_build = next(
            (
                build
                for build in runtime_builds
                if build.id.casefold() == requested_build
                or build.display_name.casefold() == requested_build
            ),
            None,
        )

    runtime = str(record.get("runtime", "llamacpp")).strip().lower()
    try:
        runtime_info = get_runtime_info(runtime)
    except RuntimeRegistryError as error:
        raise ProfileError(str(error)) from error
    unsupported_fields = sorted(
        key
        for key in record
        if key not in runtime_info.editable_fields
        and key
        not in {
            "schema_version",
        }
    )
    if unsupported_fields:
        raise ProfileError(
            f"runtime {runtime!r} does not support profile fields for "
            f"{record.get('id', '<unknown>')!r}: {', '.join(unsupported_fields)}"
        )

    profile_id = str(_required(record, "id")).strip()
    if not PROFILE_ID_RE.fullmatch(profile_id):
        raise ProfileError(f"invalid profile id: {profile_id!r}")
    alias = str(_required(record, "alias")).strip()
    if not alias:
        raise ProfileError(f"alias is empty for profile {profile_id!r}")

    raw_aliases = record.get("aliases", [])
    if not isinstance(raw_aliases, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_aliases
    ):
        raise ProfileError(f"aliases must be a list of non-empty strings for {profile_id!r}")
    aliases = tuple(dict.fromkeys([alias, *(item.strip() for item in raw_aliases)]))

    role = str(record.get("role", "persistent")).strip()
    if role not in ALLOWED_ROLES:
        raise ProfileError(f"invalid role for {profile_id!r}: {role!r}")
    raw_capabilities = record.get("capabilities", ["text"])
    if not isinstance(raw_capabilities, list):
        raise ProfileError(f"capabilities must be a list for {profile_id!r}")
    capabilities = frozenset(str(item).strip() for item in raw_capabilities)
    if not capabilities or not capabilities <= ALLOWED_CAPABILITIES:
        raise ProfileError(f"invalid capabilities for {profile_id!r}: {sorted(capabilities)!r}")
    if runtime == "strata" and capabilities != frozenset({"text"}):
        raise ProfileError(f"Strata profiles currently support text capability only for {profile_id!r}")

    warnings: list[str] = []
    degraded_features: list[str] = []
    server_binary = _absolute_existing_path(
        record, "server_binary", allow_missing=allow_missing_artifacts
    )
    server_sha256 = _sha256(record, "server_sha256")
    runtime_build: str | None = None
    build: RuntimeBuild | None = None
    if "runtime_build" in record:
        raw_runtime_build = record["runtime_build"]
        if not isinstance(raw_runtime_build, str) or not raw_runtime_build.strip():
            raise ProfileError(f"runtime_build must be a non-empty string for {profile_id!r}")
        build = resolved_build
        if build is None:
            runtime_build_name = raw_runtime_build.strip().casefold()
            build = next(
                (
                    item
                    for item in runtime_builds
                    if item.id.casefold() == runtime_build_name
                    or item.display_name.casefold() == runtime_build_name
                ),
                None,
            )
        if build is None:
            raise ProfileError(
                f"unknown runtime build for profile {profile_id!r}: {raw_runtime_build.strip()!r}"
            )
        runtime_build = build.id
        if build.runtime != runtime:
            raise ProfileError(
                f"runtime build {runtime_build!r} does not match runtime {runtime!r} "
                f"for profile {profile_id!r}"
            )
        if build.server_binary != server_binary.resolve() or build.server_sha256 != server_sha256:
            raise ProfileError(
                f"runtime build {runtime_build!r} does not match the pinned "
                f"server binary/hash for profile {profile_id!r}"
            )
    model_path = _absolute_existing_path(
        record,
        "model_path",
        allow_directory=runtime == "vllm",
        allow_missing=allow_missing_artifacts,
    )
    model_sha256 = _sha256(record, "model_sha256")
    basalt_config_path: Path | None = None
    basalt_config_sha256: str | None = None
    if runtime == "basalt":
        basalt_config_path = _absolute_existing_path(record, "basalt_config_path", allow_missing=allow_missing_artifacts)
        basalt_config_sha256 = _sha256(record, "basalt_config_sha256")
        if model_path.suffix.lower() != ".basalt" or model_path.is_symlink():
            raise ProfileError(f"Basalt model_path must be a regular .basalt file for {profile_id!r}")
        if not allow_missing_artifacts:
            if model_path.stat().st_size < 8 or model_path.read_bytes()[:6] != b"BASALT":
                raise ProfileError(f"Basalt model_path is not a native Basalt artifact for {profile_id!r}")
            if hashlib.sha256(basalt_config_path.read_bytes()).hexdigest() != basalt_config_sha256:
                raise ProfileError(f"Basalt config hash mismatch for {profile_id!r}")
        try:
            config = json.loads(basalt_config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProfileError(f"cannot read Basalt config for {profile_id!r}: {error}") from error
        if not isinstance(config, dict):
            raise ProfileError(f"Basalt config must be an object for {profile_id!r}")
        server = config.get("server")
        if not isinstance(server, dict) or server.get("model") != str(model_path):
            raise ProfileError(f"Basalt config server.model must be pinned to profile model for {profile_id!r}")
    mmproj_path: Path | None = None
    mmproj_sha256: str | None = None
    if record.get("mmproj_path"):
        mmproj_path = _absolute_existing_path(
            record, "mmproj_path", allow_missing=allow_missing_artifacts
        )
        mmproj_sha256 = _sha256(record, "mmproj_sha256")
    native_vision = record.get("native_vision", False)
    if not isinstance(native_vision, bool):
        raise ProfileError(f"native_vision must be a boolean for {profile_id!r}")
    if runtime == "strata" and native_vision:
        raise ProfileError(f"Strata native vision is not enabled for {profile_id!r}")
    if native_vision and "vision" not in capabilities:
        raise ProfileError(f"native_vision requires the vision capability for {profile_id!r}")
    if (
        runtime == "llamacpp"
        and "vision" in capabilities
        and mmproj_path is None
        and not native_vision
    ):
        raise ProfileError(f"vision profile {profile_id!r} requires an existing mmproj_path")

    model_draft: Path | None = None
    effective_model_draft: Path | None = None
    model_draft_sha256: str | None = None
    if record.get("model_draft"):
        raw_draft = str(record["model_draft"]).strip()
        candidate = Path(raw_draft).expanduser()
        if not candidate.is_absolute():
            model_draft = candidate
            warnings.append(f"{profile_id}: optional model draft path is not absolute")
            degraded_features.append("speculative-draft")
        else:
            model_draft = candidate.resolve()
            raw_draft_hash = record.get("model_draft_sha256")
            if raw_draft_hash is not None:
                if isinstance(raw_draft_hash, str) and SHA256_RE.fullmatch(raw_draft_hash.strip().lower()):
                    model_draft_sha256 = raw_draft_hash.strip().lower()
                else:
                    warnings.append(f"{profile_id}: optional model draft hash is invalid")
                    degraded_features.append("speculative-draft")
            if not model_draft.is_file():
                warnings.append(f"{profile_id}: optional model draft is unavailable")
                degraded_features.append("speculative-draft")
            elif model_draft_sha256 is not None:
                try:
                    actual_draft_hash = _optional_file_sha256(model_draft)
                except OSError:
                    actual_draft_hash = ""
                if actual_draft_hash != model_draft_sha256:
                    warnings.append(f"{profile_id}: optional model draft hash mismatch")
                    degraded_features.append("speculative-draft")
                else:
                    effective_model_draft = model_draft
            else:
                effective_model_draft = model_draft

    context_size = _positive_int(record, "context_size")
    if context_size > MAX_CONTEXT_SIZE:
        raise ProfileError(
            f"context_size for {profile_id!r} exceeds fail-closed limit {MAX_CONTEXT_SIZE}"
        )
    parallel = _positive_int(record, "parallel")
    if runtime == "strata" and parallel != 1:
        raise ProfileError(f"Strata currently serves one request at a time; parallel must be 1 for {profile_id!r}")
    max_output_tokens = (
        _positive_int(record, "max_output_tokens")
        if "max_output_tokens" in record
        else None
    )
    autotune = record.get("autotune")
    if runtime == "strata" and autotune is not None:
        raise ProfileError(f"Strata autotune is not supported for {profile_id!r}")
    if autotune is not None:
        if not isinstance(autotune, dict):
            raise ProfileError(f"autotune metadata must be an object for {profile_id!r}")
        allowed_autotune_fields = {
            "status",
            "winner_candidate",
            "strategy",
            "confidence",
            "report",
            "last_success_report",
            "profile_revision",
            "experiment_count",
            "updated_at",
            "accuracy",
        }
        unknown_autotune_fields = sorted(set(autotune) - allowed_autotune_fields)
        if unknown_autotune_fields:
            raise ProfileError(
                f"unknown autotune metadata fields for {profile_id!r}: "
                + ", ".join(unknown_autotune_fields)
            )
        status = autotune.get("status", "never")
        if status not in {"never", "complete", "failed", "interrupted", "conflicted"}:
            raise ProfileError(f"invalid autotune status for {profile_id!r}: {status!r}")
    chat_template = None
    raw_chat_template = record.get("chat_template")
    if raw_chat_template is not None:
        raw_template_path = str(raw_chat_template).strip()
        if not raw_template_path:
            raise ProfileError(f"chat_template must not be empty for {profile_id!r}")
        chat_template = str(Path(raw_template_path).expanduser())

    qualified_ampere = (
        build is not None
        and build.id == QUALIFIED_LLAMPAMPERE_BUILD
        and build.server_sha256 == QUALIFIED_LLAMPAMPERE_SHA256
        and build.source_commit == QUALIFIED_LLAMPAMPERE_COMMIT
    )
    if qualified_ampere and (context_size != 98304 or parallel != 1):
        raise ProfileError("qualified llamAmpere requires context_size=98304 and parallel=1")
    if runtime == "llamacpp":
        batch_size = _positive_int(record, "batch_size")
        ubatch_size = _positive_int(record, "ubatch_size")
        if qualified_ampere:
            if batch_size > 4096 or ubatch_size > 512:
                raise ProfileError(f"batch/ubatch exceeds qualified llamAmpere limits for {profile_id!r}")
        elif batch_size > MAX_BATCH_SIZE or ubatch_size > MAX_UBATCH_SIZE:
            raise ProfileError(f"batch/ubatch exceeds safe limit for {profile_id!r}")
        if ubatch_size > batch_size:
            raise ProfileError(f"ubatch_size cannot exceed batch_size for {profile_id!r}")

        cache_type_k = str(_required(record, "cache_type_k")).strip().lower()
        cache_type_v = str(_required(record, "cache_type_v")).strip().lower()
        for cache_field, cache_type in (
            ("cache_type_k", cache_type_k),
            ("cache_type_v", cache_type_v),
        ):
            if cache_type not in ALLOWED_CACHE_TYPES and not (qualified_ampere and cache_type == "sjkvarn4"):
                raise ProfileError(
                    f"invalid {cache_field} for {profile_id!r}: {cache_type!r}; "
                    f"supported cache types: {', '.join(sorted(ALLOWED_CACHE_TYPES))}"
                )
        if qualified_ampere and (cache_type_k, cache_type_v) != ("sjkvarn4", "sjkvarn4"):
            raise ProfileError("qualified llamAmpere requires scalar SJ-KVaRN4 K/V")
        flash_attention = str(record.get("flash_attention", "auto")).strip().lower()
        if qualified_ampere and flash_attention != "on":
            raise ProfileError("qualified llamAmpere requires flash_attention=on")
        if flash_attention not in {"on", "off", "auto"}:
            raise ProfileError(f"invalid flash_attention for {profile_id!r}")
        if (cache_type_k.startswith("turbo") or cache_type_v.startswith("turbo")) and (
            flash_attention != "on"
        ):
            raise ProfileError(f"Turbo KV requires flash_attention=on for {profile_id!r}")

        fit_raw = record.get("fit")
        if "fit" in record and not isinstance(fit_raw, bool):
            raise ProfileError(f"fit must be a boolean for {profile_id!r}")
        fit = fit_raw
        gpu_layers = str(record.get("gpu_layers", "auto")).strip().lower()
        if fit and gpu_layers != "auto":
            raise ProfileError(
                f"fit profile {profile_id!r} must leave gpu_layers=auto so fit may spill to CPU"
            )
        if not fit and gpu_layers not in {"all", "auto"} and not gpu_layers.isdigit():
            raise ProfileError(f"invalid gpu_layers for {profile_id!r}: {gpu_layers!r}")

        if fit is True:
            fit_target_mib = _positive_int(record, "fit_target_mib")
            if fit_target_mib > MAX_FIT_TARGET_MIB:
                raise ProfileError(
                    f"fit_target_mib for {profile_id!r} exceeds fail-closed limit "
                    f"{MAX_FIT_TARGET_MIB}"
                )
            fit_ctx = _positive_int(record, "fit_ctx")
            if fit_ctx > context_size:
                raise ProfileError(f"fit_ctx cannot exceed context_size for {profile_id!r}")
            ctx_checkpoints = (
                _optional_nonnegative_int(record, "ctx_checkpoints")
                if "ctx_checkpoints" in record
                else None
            )
            cache_ram_mib = (
                _optional_nonnegative_int(record, "cache_ram_mib")
                if "cache_ram_mib" in record
                else None
            )
        else:
            fit_target_mib = None
            fit_ctx = None
            ctx_checkpoints = None
            cache_ram_mib = None
            unexpected_fit_fields = sorted(
                key
                for key in ("fit_target_mib", "fit_ctx", "ctx_checkpoints", "cache_ram_mib")
                if key in record
            )
            if unexpected_fit_fields:
                raise ProfileError(
                    f"fit-only fields for {profile_id!r} require fit=true: "
                    + ", ".join(unexpected_fit_fields)
                )

        tensor_parallel_size = parallel
        gpu_memory_utilization = None
        kv_cache_dtype = None
        tool_call_parser = None
        tool_parser_plugin = None
        limit_mm_per_prompt = None
        allowed_local_media_path = None
    else:
        batch_size = None
        ubatch_size = None
        cache_type_k = None
        cache_type_v = None
        flash_attention = None
        gpu_layers = None
        fit = None
        fit_target_mib = None
        fit_ctx = None
        ctx_checkpoints = None
        cache_ram_mib = None
        tensor_parallel_size = _positive_int(record, "tensor_parallel_size") if "tensor_parallel_size" in record else parallel
        if tensor_parallel_size > 16:
            raise ProfileError(
                f"tensor_parallel_size for {profile_id!r} exceeds safe limit 16"
            )

        raw_gpu_memory = record.get("gpu_memory_utilization")
        if raw_gpu_memory is not None:
            if isinstance(raw_gpu_memory, bool) or not isinstance(raw_gpu_memory, (int, float)):
                raise ProfileError(
                    f"gpu_memory_utilization must be a number for {profile_id!r}"
                )
            gpu_memory_utilization = float(raw_gpu_memory)
            if not 0 < gpu_memory_utilization <= 1:
                raise ProfileError(
                    f"gpu_memory_utilization for {profile_id!r} must be between 0 and 1"
                )
        else:
            gpu_memory_utilization = None

        raw_kv_dtype = record.get("kv_cache_dtype")
        kv_cache_dtype = (
            str(raw_kv_dtype).strip().lower() if raw_kv_dtype is not None else None
        )
        if kv_cache_dtype is not None and kv_cache_dtype not in VLLM_KV_CACHE_DTYPES:
            raise ProfileError(
                f"unsupported vllm kv_cache_dtype for {profile_id!r}: {kv_cache_dtype!r}"
            )

        def optional_text(field_name: str) -> str | None:
            value = record.get(field_name)
            if value is None:
                return None
            text = str(value).strip()
            if not text:
                raise ProfileError(f"{field_name} must not be empty for {profile_id!r}")
            return text

        tool_call_parser = optional_text("tool_call_parser")
        tool_parser_plugin = optional_text("tool_parser_plugin")
        if "tools" in capabilities and not tool_call_parser:
            raise ProfileError(
                f"vllm tools profile {profile_id!r} requires tool_call_parser"
            )

        raw_mm_limit = record.get("limit_mm_per_prompt")
        if raw_mm_limit is not None:
            if not isinstance(raw_mm_limit, dict) or not all(
                isinstance(key, str)
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for key, value in raw_mm_limit.items()
            ):
                raise ProfileError(
                    f"limit_mm_per_prompt must map media names to positive integers "
                    f"for {profile_id!r}"
                )
            limit_mm_per_prompt = MappingProxyType(dict(raw_mm_limit))
        else:
            limit_mm_per_prompt = None
        allowed_local_media_path = optional_text("allowed_local_media_path")

    raw_environment = record.get("environment", {})
    if not isinstance(raw_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_environment.items()
    ):
        raise ProfileError(f"environment must be a string mapping for {profile_id!r}")
    environment = MappingProxyType(dict(raw_environment))
    if runtime == "llamacpp":
        if environment.get("GGML_CUDA_PDL") != "0":
            raise ProfileError(f"{profile_id!r} must disable CUDA PDL")
        if (
            parallel > 1
            and cache_type_k is not None
            and cache_type_v is not None
            and (cache_type_k.startswith("kvarn") or cache_type_v.startswith("kvarn"))
            and environment.get("LLAMA_KVARN_MULTI_STREAM") != "1"
        ):
            raise ProfileError(
                f"{profile_id!r} uses KVarN with parallel={parallel}; "
                "set LLAMA_KVARN_MULTI_STREAM=1 explicitly"
            )
    elif runtime == "vllm" and any(
        key.startswith(("GGML_", "LLAMA_")) for key in environment
    ):
        raise ProfileError(
            f"vllm profile {profile_id!r} contains llama.cpp-only environment settings"
        )
    elif runtime == "strata":
        build_environment = dict(build.environment) if build is not None else {}
        if build is None or dict(environment) != build_environment:
            raise ProfileError(f"Strata environment must match its pinned runtime bundle for {profile_id!r}")
        unsafe_environment = sorted(
            key for key in build_environment
            if key not in STRATA_RUNTIME_ENVIRONMENT_FIELDS
        )
        if unsafe_environment:
            raise ProfileError(
                f"Strata runtime bundle environment contains unsupported names for {profile_id!r}: "
                + ", ".join(unsafe_environment)
            )

    if runtime == "llamacpp":
        if "moe_cache_slots_limit" in record:
            moe_cache_slots_limit = _positive_int(record, "moe_cache_slots_limit")
        else:
            moe_cache_slots_limit = MAX_MOE_CACHE_SLOTS
        if moe_cache_slots_limit > MAX_QUALIFIED_MOE_CACHE_SLOTS:
            raise ProfileError(
                f"moe_cache_slots_limit for {profile_id!r} exceeds fail-closed limit "
                f"{MAX_QUALIFIED_MOE_CACHE_SLOTS}"
            )
    else:
        moe_cache_slots_limit = MAX_MOE_CACHE_SLOTS

    raw_extra_args = record.get("extra_args", [])
    declared_extra_args = (
        tuple(raw_extra_args)
        if isinstance(raw_extra_args, list)
        and all(isinstance(item, str) and item for item in raw_extra_args)
        else ()
    )
    effective_raw_extra_args: object = raw_extra_args
    if runtime == "llamacpp" and not qualified_ampere:
        effective_raw_extra_args, cache_warnings, cache_degraded = _degrade_optional_moe_args(
            raw_extra_args,
            profile_id,
            slots_limit=moe_cache_slots_limit,
        )
        warnings.extend(cache_warnings)
        degraded_features.extend(cache_degraded)
    extra_args, speculative_types, reasoning_mode = _parse_extra_args(
        effective_raw_extra_args,
        profile_id,
        runtime=runtime,
        moe_cache_slots_limit=moe_cache_slots_limit,
        qualified_ampere=qualified_ampere,
    )
    if qualified_ampere:
        required_flags = set(LLAMPAMPERE_EXTRA_VALUES) | set(LLAMPAMPERE_FLAG_ONLY)
        if not required_flags <= set(extra_args):
            raise ProfileError("qualified llamAmpere requires all validated memory/offline/scalar flags")
        if environment.get("LLAMA_MTP_DRAFT_COMPUTE_LEAN") != "1":
            raise ProfileError("qualified llamAmpere requires LLAMA_MTP_DRAFT_COMPUTE_LEAN=1")
    if "reasoning" in capabilities:
        if reasoning_mode in (None, "off"):
            raise ProfileError(
                f"reasoning capability for {profile_id!r} requires "
                f"--reasoning enabled in extra_args"
            )
    elif reasoning_mode is not None and reasoning_mode != "off":
        # The launch config asserts the model produces reasoning output;
        # the broker propagates that into the effective capabilities.
        capabilities = capabilities | {"reasoning"}
    safety = _parse_safety(record, profile_id, runtime=runtime)
    strata_metadata: dict[str, Any] = {}
    if runtime == "strata":
        if role != "persistent":
            raise ProfileError(f"Strata profiles must be persistent for {profile_id!r}")
        strata_metadata = _parse_strata_artifacts(
            record,
            profile_id=profile_id,
            model_path=model_path,
            model_sha256=model_sha256,
            context_size=context_size,
            runtime_build=build,
            allow_missing_artifacts=allow_missing_artifacts,
        )

    if (
        "draft-mtp" in speculative_types
        and model_draft is not None
        and model_draft.resolve() == model_path.resolve()
    ):
        raise ProfileError(
            f"model_draft for draft-mtp profile {profile_id!r} resolves to "
            "model_path; omit model_draft to reuse the target model's embedded "
            "MTP heads"
        )

    max_cache_ram_mib = safety.service.memory_max_mib // MAX_CACHE_RAM_SHARE_DIVISOR
    if cache_ram_mib is not None and cache_ram_mib > max_cache_ram_mib:
        raise ProfileError(
            f"cache_ram_mib for {profile_id!r} cannot exceed half of "
            f"safety.service.memory_max_mib ({max_cache_ram_mib} MiB)"
        )
    if (
        role == "transient"
        and "vision" in capabilities
        and safety.admission.vision_vram_reserve_mib != 0
    ):
        raise ProfileError(
            f"transient vision profile {profile_id!r} must consume rather than "
            "retain vision_vram_reserve_mib"
        )
    if (
        safety.admission.post_ready_min_free_vram_mib
        > safety.admission.min_free_vram_before_start_mib
    ):
        raise ProfileError(
            f"min_free_vram_before_start_mib for {profile_id!r} must cover "
            "the post-ready driver/vision/safety reserves"
        )
    if (
        fit is True
        and fit_target_mib is not None
        and safety.admission.post_ready_min_free_vram_mib > fit_target_mib
    ):
        raise ProfileError(
            f"fit_target_mib for {profile_id!r} must cover the declared "
            "post-ready driver/vision/safety VRAM reserves"
        )

    return Profile(
        id=profile_id,
        runtime=runtime,
        alias=alias,
        aliases=aliases,
        display_name=str(record.get("display_name", profile_id)).strip() or profile_id,
        role=role,
        capabilities=capabilities,
        native_vision=native_vision,
        restore_profile=(str(record.get("restore_profile", "")).strip() or None),
        runtime_build=runtime_build,
        server_binary=server_binary,
        server_sha256=server_sha256,
        model_path=model_path,
        model_sha256=model_sha256,
        mmproj_path=mmproj_path,
        mmproj_sha256=mmproj_sha256,
        context_size=context_size,
        parallel=parallel,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        flash_attention=flash_attention,
        gpu_layers=gpu_layers,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
        tool_call_parser=tool_call_parser,
        tool_parser_plugin=tool_parser_plugin,
        chat_template=chat_template,
        limit_mm_per_prompt=limit_mm_per_prompt,
        allowed_local_media_path=allowed_local_media_path,
        fit=fit,
        fit_target_mib=fit_target_mib,
        fit_ctx=fit_ctx,
        ctx_checkpoints=ctx_checkpoints,
        cache_ram_mib=cache_ram_mib,
        startup_timeout=_positive_int(record, "startup_timeout"),
        safety=safety,
        environment=environment,
        extra_args=extra_args,
        model_draft=model_draft,
        moe_cache_slots_limit=moe_cache_slots_limit,
        max_output_tokens=max_output_tokens,
        autotune=autotune,
        bundle_root=build.bundle_root if build is not None else None,
        source_repository=build.source_repository if build is not None else None,
        provenance_file=build.provenance_file if build is not None else None,
        source_branch=build.source_branch if build is not None else None,
        source_commit=build.source_commit if build is not None else None,
        dirty=build.dirty if build is not None else None,
        selected_profile_ids=(build.selected_profile_ids if build is not None else ()),
        builtin_tools_available=(build is None or build.build_options.get("LLAMA_SUBPROCESS") is not False),
        warnings=tuple(dict.fromkeys(warnings)),
        degraded_features=tuple(dict.fromkeys(degraded_features)),
        declared_extra_args=declared_extra_args,
        effective_extra_args=extra_args,
        effective_model_draft=effective_model_draft,
        model_draft_sha256=model_draft_sha256,
        deployment_revision=(
            runtime_catalog.deployment_revision if runtime_catalog is not None else None
        ),
        strata_artifacts_manifest_path=strata_metadata.get("manifest_path"),
        strata_artifacts_manifest_sha256=strata_metadata.get("manifest_sha256"),
        strata_launch_config_path=strata_metadata.get("launch_path"),
        strata_launch_config_sha256=strata_metadata.get("launch_sha256"),
        strata_artifact_files=strata_metadata.get("artifact_files", ()),
        strata_engine_binary=strata_metadata.get("engine_binary"),
        strata_engine_sha256=strata_metadata.get("engine_sha256"),
        strata_model_name=strata_metadata.get("model_name"),
        basalt_config_path=basalt_config_path,
        basalt_config_sha256=basalt_config_sha256,
    )
