#!/usr/bin/env python3
"""Validated, allowlisted runtime profiles for the local Hermes broker.

Automatic GGUF discovery remains useful for inventory, but it is deliberately
not execution authority.  Only profiles in ``profiles.json`` can start a
backend process.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .runtimes import get_runtime, get_runtime_info
from .runtimes.registry import RuntimeError as RuntimeRegistryError
from .runtime_builds import RuntimeBuild, RuntimeBuildError, parse_runtime_builds


PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_ROLES = frozenset({"persistent", "transient"})
ALLOWED_CAPABILITIES = frozenset({"text", "vision", "tools"})
ALLOWED_CACHE_TYPES = frozenset(
    {"f16", "f32", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "turbo4", "kvarn4"}
)
# Profiles may request the full trained window, but the fail-closed single-slot,
# fit, and memory controls below still apply.
MAX_CONTEXT_SIZE = 262_144
MAX_PARALLEL = 2
MAX_BATCH_SIZE = 512
MAX_UBATCH_SIZE = 128
MAX_FIT_TARGET_MIB = 8_192
MAX_TASKS_MAX = 512
MAX_MEMORY_MAX_MIB = 22_528
MAX_MEMORY_SWAP_MAX_MIB = 1_024
MAX_TIMEOUT_STOP_SEC = 20
MAX_SPECULATIVE_DRAFT_TOKENS = 64
MAX_CACHE_RAM_SHARE_DIVISOR = 2
MIN_ADMISSION_RAM_AVAILABLE_MIB = 8_192
MAX_ADMISSION_SWAP_USED_MIB = 1_024
MAX_ADMISSION_SWAP_GROWTH_MIB = 128
MIN_ADMISSION_TOTAL_VRAM_MIB = 12_000
MIN_ADMISSION_FREE_VRAM_MIB = 8_192
FLAG_ONLY_EXTRA_ARGS = frozenset({"--no-warmup", "--jinja"})
ALLOWED_REASONING_MODES = frozenset({"off"})
ALLOWED_SPECULATIVE_TYPES = frozenset({"mtp", "draft-mtp", "nextn"})
SPECULATIVE_COUNT_FLAGS = frozenset(
    {"--draft-max", "--spec-draft-n-max", "--draft-min"}
)
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
FREETOKEN_FLAG_ONLY_EXTRA_ARGS = frozenset(
    {
        "--disable-pynccl",
        "--moe-cache-auto",
        "--disable-moe-prefill-overlap",
        "--enable-special-token-ckpt",
        "--moe-prefill-hit-d2d",
        "--enable-cache-report",
    }
)
FREETOKEN_VALUE_EXTRA_ARGS = frozenset(
    {
        "--dtype",
        "--max-running-requests",
        "--max-seq-len-override",
        "--max-output-tokens",
        "--memory-ratio",
        "--cuda-graph-max-bs",
        "--num-tokenizer",
        "--max-prefill-length",
        "--num-pages",
        "--num-tokens",
        "--page-size",
        "--attention-backend",
        "--cache-type",
        "--served-model-name",
        "--tool-call-parser",
        "--reasoning-parser",
        "--sampling-defaults",
        "--moe-backend",
        "--nvfp4-backend",
        "--expert-load",
        "--moe-cache-size",
        "--moe-cache-rate",
        "--kv-reserve-tokens",
        "--moe-cache-policy",
        "--moe-cpu-threads",
        "--moe-cpu-layers",
        "--moe-hybrid-max-fetch",
        "--cors-origins",
    }
)
LLAMACPP_VLLM_ONLY_FIELDS = frozenset(
    {
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
    return args, frozenset()


def _parse_freetoken_extra_args(
    args: tuple[str, ...], profile_id: str
) -> tuple[tuple[str, ...], frozenset[str]]:
    seen_flags: set[str] = set()
    index = 0
    while index < len(args):
        flag = args[index]
        if flag in seen_flags:
            raise ProfileError(f"duplicate extra_args flag for {profile_id!r}: {flag}")
        if flag in FREETOKEN_FLAG_ONLY_EXTRA_ARGS:
            seen_flags.add(flag)
            index += 1
            continue
        if flag not in FREETOKEN_VALUE_EXTRA_ARGS:
            raise ProfileError(
                f"extra_args for freetoken profile {profile_id!r} contain unsupported "
                f"token: {flag!r}"
            )
        if index + 1 >= len(args) or args[index + 1].startswith("--"):
            raise ProfileError(
                f"extra_args flag {flag} for {profile_id!r} requires a value"
            )
        seen_flags.add(flag)
        index += 2
    return args, frozenset()


def _parse_extra_args(
    raw: object, profile_id: str, *, runtime: str
) -> tuple[tuple[str, ...], frozenset[str]]:
    if not isinstance(raw, list) or not all(
        isinstance(item, str) and item for item in raw
    ):
        raise ProfileError(f"extra_args must be a string list for {profile_id!r}")

    args = tuple(raw)
    if runtime == "vllm":
        return _parse_vllm_extra_args(args, profile_id)
    if runtime == "freetoken":
        return _parse_freetoken_extra_args(args, profile_id)

    speculative_types: set[str] = set()
    seen_flags: set[str] = set()
    i = 0
    while i < len(args):
        flag = args[i]
        if flag in seen_flags:
            raise ProfileError(f"duplicate extra_args flag for {profile_id!r}: {flag}")
        if flag in FLAG_ONLY_EXTRA_ARGS:
            seen_flags.add(flag)
            i += 1
            continue
        if flag not in {
            "--reasoning",
            "--spec-type",
            *SPECULATIVE_COUNT_FLAGS,
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
        if flag == "--reasoning":
            if value not in ALLOWED_REASONING_MODES:
                raise ProfileError(
                    f"extra_args flag {flag} for {profile_id!r} has unsupported "
                    f"value {value!r}"
                )
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

    return args, frozenset(speculative_types)


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


def _parse_safety(record: Mapping[str, Any], profile_id: str) -> SafetyBudget:
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
    if service.memory_max_mib > MAX_MEMORY_MAX_MIB:
        raise ProfileError(
            f"safety.service.memory_max_mib for {profile_id!r} exceeds "
            f"fail-closed limit {MAX_MEMORY_MAX_MIB}"
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
    admission_lower_bounds = {
        "min_ram_available_mib": MIN_ADMISSION_RAM_AVAILABLE_MIB,
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
    max_output_tokens: int | None = None
    autotune: Mapping[str, Any] | None = None

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
    def __init__(self, profiles: tuple[Profile, ...]):
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
        if payload.get("schema_version") != 1:
            raise ProfileError("unsupported profile manifest schema_version")
        raw_profiles = payload.get("profiles")
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise ProfileError("profile manifest must contain a non-empty profiles list")

        try:
            runtime_builds = parse_runtime_builds(
                payload.get("runtime_builds", []),
                allow_missing_artifacts=allow_missing_artifacts,
            )
        except RuntimeBuildError as error:
            raise ProfileError(str(error)) from error
        profiles = tuple(
            _parse_profile(
                item,
                allow_missing_artifacts=allow_missing_artifacts,
                runtime_builds=runtime_builds,
            ) for item in raw_profiles
        )
        ids = [profile.id for profile in profiles]
        if len(ids) != len(set(ids)):
            raise ProfileError("duplicate profile id")
        return cls(profiles)

    @property
    def profiles(self) -> tuple[Profile, ...]:
        return self._profiles

    def get(self, profile_id: str) -> Profile:
        try:
            return self._by_id[profile_id]
        except KeyError as exc:
            raise ProfileError(f"unknown profile: {profile_id}") from exc

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


def _parse_profile(
    raw: object,
    *,
    allow_missing_artifacts: bool = False,
    runtime_builds: tuple[RuntimeBuild, ...] = (),
) -> Profile:
    if not isinstance(raw, dict):
        raise ProfileError("every profile must be a JSON object")
    record: Mapping[str, Any] = raw

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

    server_binary = _absolute_existing_path(
        record, "server_binary", allow_missing=allow_missing_artifacts
    )
    server_sha256 = _sha256(record, "server_sha256")
    runtime_build: str | None = None
    if "runtime_build" in record:
        raw_runtime_build = record["runtime_build"]
        if not isinstance(raw_runtime_build, str) or not raw_runtime_build.strip():
            raise ProfileError(f"runtime_build must be a non-empty string for {profile_id!r}")
        runtime_build = raw_runtime_build.strip()
        build = next((item for item in runtime_builds if item.id == runtime_build), None)
        if build is None:
            raise ProfileError(
                f"unknown runtime build for profile {profile_id!r}: {runtime_build!r}"
            )
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
        allow_directory=runtime in {"vllm", "freetoken"},
        allow_missing=allow_missing_artifacts,
    )
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
    if record.get("model_draft"):
        model_draft = _absolute_existing_path(
            record, "model_draft", allow_missing=allow_missing_artifacts
        )

    context_size = _positive_int(record, "context_size")
    if context_size > MAX_CONTEXT_SIZE:
        raise ProfileError(
            f"context_size for {profile_id!r} exceeds fail-closed limit {MAX_CONTEXT_SIZE}"
        )
    parallel = _positive_int(record, "parallel")
    if runtime == "llamacpp" and parallel > MAX_PARALLEL:
        raise ProfileError(f"parallel slots for {profile_id!r} exceed safe limit {MAX_PARALLEL}")
    max_output_tokens = (
        _positive_int(record, "max_output_tokens")
        if "max_output_tokens" in record
        else None
    )
    autotune = record.get("autotune")
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
    if runtime == "llamacpp":
        batch_size = _positive_int(record, "batch_size")
        ubatch_size = _positive_int(record, "ubatch_size")
        if batch_size > MAX_BATCH_SIZE or ubatch_size > MAX_UBATCH_SIZE:
            raise ProfileError(f"batch/ubatch exceeds safe limit for {profile_id!r}")
        if ubatch_size > batch_size:
            raise ProfileError(f"ubatch_size cannot exceed batch_size for {profile_id!r}")

        cache_type_k = str(_required(record, "cache_type_k")).strip().lower()
        cache_type_v = str(_required(record, "cache_type_v")).strip().lower()
        for cache_field, cache_type in (
            ("cache_type_k", cache_type_k),
            ("cache_type_v", cache_type_v),
        ):
            if cache_type not in ALLOWED_CACHE_TYPES:
                raise ProfileError(
                    f"invalid {cache_field} for {profile_id!r}: {cache_type!r}; "
                    f"supported cache types: {', '.join(sorted(ALLOWED_CACHE_TYPES))}"
                )
        flash_attention = str(record.get("flash_attention", "auto")).strip().lower()
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
        chat_template = None
        limit_mm_per_prompt = None
        allowed_local_media_path = None
    elif runtime == "vllm":
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
        chat_template = optional_text("chat_template")
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
        tensor_parallel_size = parallel
        gpu_memory_utilization = None
        kv_cache_dtype = None
        tool_call_parser = None
        tool_parser_plugin = None
        chat_template = None
        limit_mm_per_prompt = None
        allowed_local_media_path = None

    raw_environment = record.get("environment", {})
    if not isinstance(raw_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_environment.items()
    ):
        raise ProfileError(f"environment must be a string mapping for {profile_id!r}")
    environment = MappingProxyType(dict(raw_environment))
    if runtime == "llamacpp":
        if environment.get("GGML_CUDA_DISABLE_GRAPHS") != "1":
            raise ProfileError(f"{profile_id!r} must disable CUDA graph capture")
        if environment.get("GGML_CUDA_GRAPH_OPT") != "0":
            raise ProfileError(f"{profile_id!r} must disable CUDA graph optimization")
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

    extra_args, speculative_types = _parse_extra_args(
        record.get("extra_args", []), profile_id, runtime=runtime
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

    safety = _parse_safety(record, profile_id)
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
        model_sha256=_sha256(record, "model_sha256"),
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
        max_output_tokens=max_output_tokens,
        autotune=autotune,
    )
