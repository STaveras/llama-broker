#!/usr/bin/env python3
"""Broker-only loopback bridge for Strata's HTTP server."""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler
import importlib
import ipaddress
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Sequence


try:
    from llama_broker.gpu_memory import GpuTelemetryError, probe_gpu_memory
except ModuleNotFoundError:  # packaged bridge: shared module is bundled beside the launcher
    from gpu_memory import GpuTelemetryError, probe_gpu_memory


class StrataBridgeError(ValueError):
    """The Strata wrapper configuration violates the broker contract."""


STRATA_MIN_AVAILABLE_RAM_MIB = 6_144
STRATA_MAX_SWAP_USED_MIB = 1_024
STRATA_MIN_CGROUP_MEMORY_MIB = 8_192
STRATA_MAX_CGROUP_MEMORY_MIB = 57_344
STRATA_MAX_CGROUP_SWAP_MIB = 1_024
STRATA_IQ2_XS_ARENA_GIB = 35.5
STRATA_RESIDENT_RAM_GIB = 48.0
STRATA_LOW_RAM_HOST_HEADROOM_GIB = 6.0
STRATA_LOW_RAM_VRAM_RESERVE_GIB = 5.0
STRATA_TOTAL_VRAM_MIN_MIB = 12_000
STRATA_FREE_VRAM_MIN_MIB = 8_192
STRATA_MIN_KV_RESIDENT_CELLS = 20_480
# KV formats that support --kv-resident streaming (k8v4 cannot stream)
STRATA_KV_FORMATS = frozenset({"fp16", "int8", "q4_0"})

def _is_loopback(host: str) -> bool:
    value = str(host).strip().lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def read_api_keys_file(path: Path | str) -> tuple[str, ...]:
    """Read every currently accepted broker key from its private file."""
    key_path = Path(path).expanduser()
    if not key_path.is_absolute() or key_path.is_symlink() or not key_path.is_file():
        raise StrataBridgeError("broker API-key file must be an absolute regular file")
    if os.name != "nt" and stat.S_IMODE(key_path.stat().st_mode) & 0o077:
        raise StrataBridgeError("broker API-key file permissions must be 0600 or stricter")
    keys = tuple(
        line.strip()
        for line in key_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not keys:
        raise StrataBridgeError("broker API-key file is empty")
    return keys


def read_api_key_file(path: Path | str) -> str:
    """Read the current broker key without exposing it in config or argv."""
    return read_api_keys_file(path)[-1]


def enforce_resource_floor(
    *,
    available_ram_mib: int,
    min_available_ram_mib: int,
    swap_used_mib: int,
    max_swap_used_mib: int,
) -> str | None:
    """Hard-gate free RAM, but treat host-wide swap as advisory only.

    The service's effective cgroup MemoryMax/MemorySwapMax are checked
    separately and remain the hard per-process resource limits.
    """
    if min_available_ram_mib < STRATA_MIN_AVAILABLE_RAM_MIB:
        raise StrataBridgeError("Strata minimum available-RAM gate cannot be weakened")
    if max_swap_used_mib > STRATA_MAX_SWAP_USED_MIB:
        raise StrataBridgeError("Strata host-swap warning threshold cannot be weakened")
    if available_ram_mib < min_available_ram_mib:
        raise StrataBridgeError(
            f"available RAM is {available_ram_mib} MiB; Strata profile requires "
            f"at least {min_available_ram_mib} MiB"
        )
    if swap_used_mib > max_swap_used_mib:
        return (
            f"host swap usage is {swap_used_mib} MiB, above the advisory "
            f"threshold of {max_swap_used_mib} MiB; the systemd service cgroup "
            "still enforces this process's MemoryMax and MemorySwapMax"
        )
    return None


def _system_resource_mib() -> tuple[int, int, int]:
    """Return (MemTotal, MemAvailable, swap used) in MiB from Linux procfs."""
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"} and parts:
                values[key] = int(parts[0])
        total_kib = values["MemTotal"]
        available_kib = values["MemAvailable"]
        swap_used_kib = values["SwapTotal"] - values["SwapFree"]
    except (OSError, KeyError, ValueError) as error:
        raise StrataBridgeError(f"cannot read Linux memory admission data: {error}") from error
    return total_kib // 1024, available_kib // 1024, max(0, swap_used_kib // 1024)


def assess_strata_memory_mode(
    *,
    system_ram_mib: int,
    gpu_vram_total_mib: int | None,
    mmap_experts: bool,
    has_experts_bin: bool,
    resident_budget_gib: int | None = None,
) -> str | None:
    """Return fit warnings for mmap or bounded resident-expert modes."""
    if mmap_experts:
        if not has_experts_bin:
            raise StrataBridgeError("Strata mmap-experts mode requires the pack's experts.bin")
        if gpu_vram_total_mib is None:
            raise StrataBridgeError("GPU VRAM telemetry is unavailable; cannot assess Strata mmap mode")
        ram_gib = system_ram_mib / 1024
        vram_gib = gpu_vram_total_mib / 1024
        estimated_cache_gib = (
            ram_gib - STRATA_LOW_RAM_HOST_HEADROOM_GIB
            + max(0.0, vram_gib - STRATA_LOW_RAM_VRAM_RESERVE_GIB)
        )
        if estimated_cache_gib < STRATA_IQ2_XS_ARENA_GIB:
            return (
                f"Strata low-RAM estimate is {estimated_cache_gib:.1f} GiB for a "
                f"{STRATA_IQ2_XS_ARENA_GIB:.1f} GiB expert arena; mmap mode may page heavily from the SSD"
            )
        return None
    if resident_budget_gib is not None:
        if resident_budget_gib <= 0 or resident_budget_gib > 64:
            raise StrataBridgeError("Strata resident budget must be 1-64 GiB")
        if system_ram_mib < (resident_budget_gib + int(STRATA_LOW_RAM_HOST_HEADROOM_GIB)) * 1024:
            raise StrataBridgeError(
                f"bounded resident Strata budget of {resident_budget_gib} GiB requires "
                f"at least {resident_budget_gib + int(STRATA_LOW_RAM_HOST_HEADROOM_GIB)} GiB system RAM"
            )
        if system_ram_mib < int(STRATA_RESIDENT_RAM_GIB * 1024):
            return (
                f"bounded resident budget of {resident_budget_gib} GiB on "
                f"{system_ram_mib / 1024:.1f} GiB system RAM may still page cold experts"
            )
        return None
    if system_ram_mib < int(STRATA_RESIDENT_RAM_GIB * 1024):
        raise StrataBridgeError(
            f"resident Strata IQ2_XS mode requires about {STRATA_RESIDENT_RAM_GIB:.0f} GiB system RAM; "
            "select an explicitly packaged mmap-experts mode instead"
        )
    return None


def enforce_gpu_floor(
    *,
    gpu_total_mib: int | None,
    gpu_free_mib: int | None,
    min_total_vram_mib: int,
    min_free_vram_mib: int,
) -> tuple[int, int]:
    if min_total_vram_mib < STRATA_TOTAL_VRAM_MIN_MIB:
        raise StrataBridgeError("Strata minimum total-VRAM gate cannot be weakened")
    if min_free_vram_mib < STRATA_FREE_VRAM_MIN_MIB:
        raise StrataBridgeError("Strata minimum free-VRAM gate cannot be weakened")
    if gpu_total_mib is None or gpu_free_mib is None:
        raise StrataBridgeError(
            "GPU VRAM telemetry is unavailable; refusing Strata admission"
        )
    if gpu_total_mib < min_total_vram_mib:
        raise StrataBridgeError(f"total VRAM is {gpu_total_mib} MiB; Strata profile requires {min_total_vram_mib} MiB")
    if gpu_free_mib < min_free_vram_mib:
        raise StrataBridgeError(f"free VRAM is {gpu_free_mib} MiB; Strata profile requires {min_free_vram_mib} MiB")
    return gpu_total_mib, gpu_free_mib


def enforce_cgroup_budget(
    *,
    total_system_ram_mib: int,
    declared_memory_max_mib: int,
    declared_memory_swap_max_mib: int,
    effective_memory_max_mib: int,
    effective_memory_swap_max_mib: int,
) -> None:
    if not STRATA_MIN_CGROUP_MEMORY_MIB <= declared_memory_max_mib <= min(
        STRATA_MAX_CGROUP_MEMORY_MIB, total_system_ram_mib
    ):
        raise StrataBridgeError("Strata cgroup memory budget is outside host and broker policy")
    if declared_memory_swap_max_mib < 0 or declared_memory_swap_max_mib > STRATA_MAX_CGROUP_SWAP_MIB:
        raise StrataBridgeError("Strata cgroup swap budget exceeds the broker policy")
    if effective_memory_max_mib < STRATA_MIN_CGROUP_MEMORY_MIB or effective_memory_max_mib > declared_memory_max_mib:
        raise StrataBridgeError("effective cgroup memory limit does not match the Strata profile budget")
    if effective_memory_swap_max_mib < 0 or effective_memory_swap_max_mib > declared_memory_swap_max_mib:
        raise StrataBridgeError("effective cgroup swap limit exceeds the Strata profile budget")


def _effective_cgroup_limit_mib(
    filename: str,
    *,
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_root_path: Path = Path("/sys/fs/cgroup"),
) -> int:
    cgroup_root = cgroup_root_path.resolve()
    try:
        row = next(
            line for line in proc_cgroup_path.read_text(encoding="ascii").splitlines()
            if line.startswith("0::")
        )
        relative = Path(row.split("::", 1)[1].lstrip("/"))
    except (OSError, StopIteration, ValueError) as error:
        raise StrataBridgeError(f"cannot locate unified cgroup for Strata admission: {error}") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise StrataBridgeError("process cgroup path is unsafe")
    current = (cgroup_root / relative).resolve()
    if current == cgroup_root or not current.is_relative_to(cgroup_root):
        raise StrataBridgeError("Strata bridge must run inside a bounded systemd service cgroup")

    def read_limit(path: Path, *, required: bool) -> int | None:
        try:
            raw = path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            if required:
                raise StrataBridgeError(f"required cgroup controller is missing: {path}")
            return None
        except OSError as error:
            raise StrataBridgeError(f"cannot read {filename} cgroup limit: {error}") from error
        if raw == "max":
            if required:
                raise StrataBridgeError(f"Strata service cgroup has no {filename} limit")
            return None
        try:
            return int(raw) // (1024 * 1024)
        except ValueError as error:
            raise StrataBridgeError(f"invalid {filename} cgroup limit") from error

    current_limit = read_limit(current / filename, required=True)
    assert current_limit is not None
    limits = [current_limit]
    parent = current.parent
    while parent != cgroup_root:
        parent_limit = read_limit(parent / filename, required=False)
        if parent_limit is not None:
            limits.append(parent_limit)
        parent = parent.parent
    return min(limits)


def _current_cgroup_limits_mib() -> tuple[int, int]:
    return (
        _effective_cgroup_limit_mib("memory.max"),
        _effective_cgroup_limit_mib("memory.swap.max"),
    )


def install_private_loopback_serve(
    server_module: ModuleType | object,
    api_key_file: Path | str,
) -> None:
    """Require a broker key and force Strata's HTTP listener to loopback."""
    keys = read_api_keys_file(api_key_file)
    original_serve = getattr(server_module, "serve", None)
    original_make_handler = getattr(server_module, "make_handler", None)
    if not callable(original_serve) or not callable(original_make_handler):
        raise StrataBridgeError("Strata server module is missing its serve/handler functions")

    def private_handler(service):
        handler_class = original_make_handler(service)
        original_parse_request = getattr(handler_class, "parse_request", None)
        original_expect_100 = getattr(handler_class, "handle_expect_100", None)
        if original_parse_request is not BaseHTTPRequestHandler.parse_request:
            raise StrataBridgeError("Strata request parser differs from the audited standard-library parser")
        if not callable(original_expect_100):
            raise StrataBridgeError("Strata request handler has no Expect/Continue hook")
        assert callable(original_expect_100)

        def authorized(handler) -> bool:
            try:
                accepted_keys = read_api_keys_file(api_key_file)
            except StrataBridgeError:
                accepted_keys = ()
            authorization = handler.headers.get("Authorization", "")
            supplied = (
                authorization[7:].strip()
                if authorization.lower().startswith("bearer ")
                else handler.headers.get("x-api-key", "")
            )
            if supplied in accepted_keys:
                return True
            handler._json(
                401,
                {"error": {"type": "authentication_error", "message": "missing or wrong API key"}},
            )
            return False

        def private_parse_request(handler) -> bool:
            if not original_parse_request(handler):
                return False
            return authorized(handler)

        def private_expect_100(handler) -> bool:
            if not authorized(handler):
                return False
            return bool(original_expect_100(handler))

        setattr(handler_class, "_authorized", authorized)
        setattr(handler_class, "parse_request", private_parse_request)
        setattr(handler_class, "handle_expect_100", private_expect_100)
        return handler_class

    def private_serve(service, *, host: str = "127.0.0.1", port: int = 10000):
        if not _is_loopback(host):
            raise StrataBridgeError("Strata backend may only bind to loopback")
        service.api_key = keys[-1]
        return original_serve(service, host="127.0.0.1", port=port)

    setattr(server_module, "make_handler", private_handler)
    setattr(server_module, "serve", private_serve)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_runtime_source_inventory(root: Path, entries: object) -> None:
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise StrataBridgeError("Strata source root must be an absolute directory")
    expected: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or "__pycache__" in path.parts or path.suffix == ".pyc":
            raise StrataBridgeError("Strata source tree contains a symlink or bytecode file")
        if path.is_file() and path.suffix == ".py":
            expected.add(path.relative_to(root).as_posix())
    if not isinstance(entries, list) or not entries:
        raise StrataBridgeError("Strata launch config has no pinned Python source inventory")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise StrataBridgeError("Strata runtime source manifest entry is malformed")
        relative = entry.get("path")
        expected_sha = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or chr(92) in relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).suffix != ".py"
        ):
            raise StrataBridgeError("Strata runtime source path is unsafe")
        if relative in seen:
            raise StrataBridgeError("Strata runtime source inventory contains duplicates")
        seen.add(relative)
        source = (root / relative).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file() or source.is_symlink():
            raise StrataBridgeError("Strata runtime source file is missing or unsafe")
        if not isinstance(expected_sha, str) or _hash_file(source) != expected_sha:
            raise StrataBridgeError(f"Strata runtime source SHA-256 mismatch: {relative}")
    if seen != expected:
        raise StrataBridgeError("Strata runtime source inventory is incomplete or has extras")


def _load_launch_config(path: Path, expected_sha256: str) -> dict:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise StrataBridgeError("Strata launch config must be an absolute regular file")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise StrataBridgeError(f"cannot read Strata launch config: {error}") from error
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise StrataBridgeError("Strata launch config SHA-256 mismatch")
    try:
        config = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise StrataBridgeError(f"cannot decode Strata launch config: {error}") from error
    if not isinstance(config, dict) or set(config) != {
        "schema_version", "model_name", "engine_binary", "engine_sha256",
        "expert_profile_sha256", "engine_args", "working_directory", "tokenizer_path",
        "runtime_files",
    }:
        raise StrataBridgeError("Strata launch config has unknown or missing fields")
    if config.get("schema_version") != 1 or not isinstance(config.get("model_name"), str):
        raise StrataBridgeError("unsupported Strata launch config schema")
    engine = Path(str(config["engine_binary"])).expanduser()
    if not engine.is_absolute() or not engine.is_file() or engine.is_symlink() or not os.access(engine, os.X_OK):
        raise StrataBridgeError("Strata engine binary must be an absolute executable regular file")
    expected_engine_sha = config.get("engine_sha256")
    if not isinstance(expected_engine_sha, str) or _hash_file(engine) != expected_engine_sha:
        raise StrataBridgeError("Strata engine binary SHA-256 mismatch")
    args = config.get("engine_args")
    value_flags = {
        "--pack", "--native", "--ple-gguf", "--expert-profile",
        "--expert-cache", "--prefill", "--max-context", "--kv-resident", "--spec",
        "--spec-min-p", "--mtp",
    }
    boolean_flags = {"--mmap-experts"}
    # Optional: parks the active conversation in RAM while another prompt runs
    optional_value_flags = {"--conversation-cache-mib", "--resident-budget-gib", "--kv"}
    if not isinstance(args, list) or not all(isinstance(arg, str) and arg for arg in args):
        raise StrataBridgeError("Strata engine_args must be a string list")
    seen: set[str] = set()
    expert_profile: Path | None = None
    pack_path: Path | None = None
    mtp_path: Path | None = None
    index = 0
    while index < len(args):
        flag = args[index]
        if flag in boolean_flags:
            if flag in seen:
                raise StrataBridgeError("Strata engine_args contain a duplicate flag")
            seen.add(flag)
            index += 1
            continue
        if flag not in value_flags | optional_value_flags or flag in seen or index + 1 >= len(args):
            raise StrataBridgeError("Strata engine_args contain an unsupported or duplicate flag")
        value = args[index + 1]
        if value.startswith("--"):
            raise StrataBridgeError("Strata engine_args flag is missing its value")
        seen.add(flag)
        if flag == "--expert-profile":
            expert_profile = Path(value).expanduser()
        elif flag == "--pack":
            pack_path = Path(value).expanduser()
        elif flag == "--mtp":
            mtp_path = Path(value).expanduser()
        elif flag == "--spec" and value != "4":
            raise StrataBridgeError("Strata profile must pin the supported four-token MTP window")
        elif flag == "--spec-min-p" and value != "0.5":
            raise StrataBridgeError("Strata profile must pin the supported MTP acceptance threshold")
        elif flag == "--conversation-cache-mib" and not value.isdecimal():
            raise StrataBridgeError("Strata conversation cache size must be a whole number of MiB")
        elif flag == "--kv" and value not in STRATA_KV_FORMATS:
            raise StrataBridgeError("Strata KV format must be fp16, int8 or q4_0 with KV streaming")
        elif flag == "--kv-resident" and (
            not value.isdecimal() or int(value) < STRATA_MIN_KV_RESIDENT_CELLS
        ):
            raise StrataBridgeError(
                f"Strata KV streaming requires at least {STRATA_MIN_KV_RESIDENT_CELLS} resident cells"
            )
        index += 2
    required_flags = value_flags
    if not required_flags.issubset(seen) or seen - required_flags - boolean_flags - optional_value_flags:
        raise StrataBridgeError("Strata engine_args are missing required controls")
    if expert_profile is None:
        raise StrataBridgeError("Strata engine_args are missing the expert profile")
    if not expert_profile.is_absolute() or expert_profile.is_symlink() or not expert_profile.is_file():
        raise StrataBridgeError("Strata expert profile must be an absolute regular file")
    expected_expert_sha = config.get("expert_profile_sha256")
    if not isinstance(expected_expert_sha, str) or _hash_file(expert_profile) != expected_expert_sha:
        raise StrataBridgeError("Strata expert profile SHA-256 mismatch")
    if "--mmap-experts" in seen:
        if pack_path is None or not pack_path.is_absolute() or pack_path.is_symlink() or not pack_path.is_dir():
            raise StrataBridgeError("Strata mmap-experts mode requires an absolute regular pack directory")
        expert_file = pack_path / "experts.bin"
        if expert_file.is_symlink() or not expert_file.is_file():
            raise StrataBridgeError("Strata mmap-experts mode requires the pack's experts.bin")
    if mtp_path is None or not mtp_path.is_absolute() or mtp_path.is_symlink() or not mtp_path.is_dir():
        raise StrataBridgeError("Strata native-pack serving requires an absolute MTP runtime directory")
    for name in ("experts.bin", "dense.bin", "dense.txt"):
        file_path = mtp_path / name
        if file_path.is_symlink() or not file_path.is_file():
            raise StrataBridgeError(f"Strata MTP runtime is missing {name}")
    working_directory = Path(str(config["working_directory"])).expanduser()
    tokenizer_path = Path(str(config["tokenizer_path"])).expanduser()
    if not working_directory.is_absolute() or not working_directory.is_dir():
        raise StrataBridgeError("Strata working directory is missing")
    if not tokenizer_path.is_absolute() or not tokenizer_path.is_dir():
        raise StrataBridgeError("Strata tokenizer directory is missing")
    _verify_runtime_source_inventory(working_directory, config.get("runtime_files"))
    return config


def _runtime_log_path(
    model_name: str,
    *,
    state_home: Path | str | None = None,
) -> Path:
    """Create a private, bounded-size startup log for Strata's stderr."""
    name = str(model_name).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
        raise StrataBridgeError("Strata model_name is unsafe for a runtime log filename")
    if state_home is None:
        configured = os.environ.get("XDG_STATE_HOME")
        base = Path(configured).expanduser() if configured else Path.home() / ".local" / "state"
    else:
        base = Path(state_home).expanduser()
    if not base.is_absolute() or base.is_symlink():
        raise StrataBridgeError("Strata runtime log directory must be an absolute non-symlink path")
    directory = base / "llama-broker" / "strata"
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise StrataBridgeError("Strata runtime log directory must be a regular directory")
        os.chmod(directory, 0o700)
        path = directory / f"{name}.log"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise StrataBridgeError("Strata runtime log must be a regular file")
            if info.st_size > 64 * 1024 * 1024:
                os.ftruncate(descriptor, 0)
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
    except StrataBridgeError:
        raise
    except OSError as error:
        raise StrataBridgeError(f"cannot create private Strata runtime log: {error}") from error
    return path


def _write_server_config(config: dict, log_path: Path) -> Path:
    if not log_path.is_absolute() or log_path.is_symlink() or not log_path.is_file():
        raise StrataBridgeError("Strata server log path must be an absolute private regular file")
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    directory = Path(runtime_dir) if runtime_dir else Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="strata-broker-", suffix=".json", dir=directory)
    path = Path(name)
    try:
        payload = {
            "exe": config["engine_binary"],
            "args": config["engine_args"],
            "cwd": config["working_directory"],
            "tokenizer": config["tokenizer_path"],
            "model_name": config["model_name"],
            "log": str(log_path),
        }
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o600)
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--min-available-ram-mib", type=int, required=True)
    parser.add_argument("--max-swap-used-mib", type=int, required=True)
    parser.add_argument("--memory-max-mib", type=int, required=True)
    parser.add_argument("--memory-swap-max-mib", type=int, required=True)
    parser.add_argument("--min-total-vram-mib", type=int, required=True)
    parser.add_argument("--min-free-vram-mib", type=int, required=True)
    parser.add_argument("--broker-api-key-file", required=True)
    args = parser.parse_args(argv)
    if not _is_loopback(args.host):
        parser.error("Strata backend may only bind to loopback")
    if args.port <= 0 or args.port > 65535:
        parser.error("port must be between 1 and 65535")
    if (
        args.min_available_ram_mib <= 0
        or args.max_swap_used_mib < 0
        or args.memory_max_mib <= 0
        or args.memory_swap_max_mib < 0
        or args.min_total_vram_mib <= 0
        or args.min_free_vram_mib <= 0
    ):
        parser.error("memory and VRAM admission limits must be positive/non-negative MiB values")
    config_path = Path(args.config).expanduser()
    config = _load_launch_config(config_path, args.config_sha256)
    engine_args = config["engine_args"]
    mmap_experts = "--mmap-experts" in engine_args
    resident_budget_gib = None
    if "--resident-budget-gib" in engine_args:
        resident_budget_gib = int(engine_args[engine_args.index("--resident-budget-gib") + 1])
    pack_arg = engine_args[engine_args.index("--pack") + 1]
    has_experts_bin = (Path(pack_arg) / "experts.bin").is_file()
    try:
        total_ram_mib, available_ram_mib, swap_used_mib = _system_resource_mib()
        gpu_total_mib = gpu_free_mib = None
        try:
            gpu_reading = probe_gpu_memory()
        except GpuTelemetryError as error:
            raise StrataBridgeError(f"GPU telemetry failed: {error}") from error
        if gpu_reading is not None:
            gpu_total_mib, gpu_free_mib = gpu_reading
        swap_warning = enforce_resource_floor(
            available_ram_mib=available_ram_mib,
            min_available_ram_mib=args.min_available_ram_mib,
            swap_used_mib=swap_used_mib,
            max_swap_used_mib=args.max_swap_used_mib,
        )
        if swap_warning:
            print(f"[strata] WARNING: {swap_warning}", file=sys.stderr, flush=True)
        memory_warning = assess_strata_memory_mode(
            system_ram_mib=total_ram_mib,
            gpu_vram_total_mib=gpu_total_mib,
            mmap_experts=mmap_experts,
            has_experts_bin=has_experts_bin,
            resident_budget_gib=resident_budget_gib,
        )
        if memory_warning:
            print(f"[strata] WARNING: {memory_warning}", file=sys.stderr, flush=True)
        gpu_total_mib, gpu_free_mib = enforce_gpu_floor(
            gpu_total_mib=gpu_total_mib,
            gpu_free_mib=gpu_free_mib,
            min_total_vram_mib=args.min_total_vram_mib,
            min_free_vram_mib=args.min_free_vram_mib,
        )
        effective_memory_max_mib, effective_memory_swap_max_mib = _current_cgroup_limits_mib()
        enforce_cgroup_budget(
            total_system_ram_mib=total_ram_mib,
            declared_memory_max_mib=args.memory_max_mib,
            declared_memory_swap_max_mib=args.memory_swap_max_mib,
            effective_memory_max_mib=effective_memory_max_mib,
            effective_memory_swap_max_mib=effective_memory_swap_max_mib,
        )
    except StrataBridgeError as error:
        parser.error(str(error))
    try:
        log_path = _runtime_log_path(str(config["model_name"]))
    except StrataBridgeError as error:
        parser.error(str(error))
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    server_root = Path(config["working_directory"]).resolve()
    sys.path.insert(0, str(server_root))
    server_module = importlib.import_module("serve.server")
    install_private_loopback_serve(server_module, args.broker_api_key_file)
    server_config = _write_server_config(config, log_path)
    original_argv = sys.argv
    try:
        sys.argv = [
            "strata-server",
            "--engine", "strata",
            "--config", str(server_config),
            "--host", "127.0.0.1",
            "--port", str(args.port),
        ]
        return int(server_module.main())
    finally:
        sys.argv = original_argv
        server_config.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
