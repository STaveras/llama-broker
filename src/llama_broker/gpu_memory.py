"""Capability-detected GPU memory telemetry without vendor command-line tools.

The probe returns one selected device's ``(total_mib, free_mib)`` pair.  An
unsupported platform returns ``None``; malformed or failing supported providers
raise :class:`GpuTelemetryError` so callers can distinguish unknown capability
from an actual telemetry failure.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Any, Callable


class GpuTelemetryError(RuntimeError):
    """A supported GPU provider was found but could not provide valid data."""


CudaLoader = Callable[[], Any]


def _cuda_visible_count(driver_count: int) -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return driver_count
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens or tokens == ["-1"] or tokens == ["NoDevFiles"]:
        return 0
    return len(tokens)


def _selected_index(*, count: int, provider: str) -> int:
    if count <= 0:
        raise GpuTelemetryError(f"{provider} reports no visible devices")
    raw = os.environ.get("LLAMA_BROKER_GPU_DEVICE", "").strip()
    if raw:
        try:
            index = int(raw, 10)
        except ValueError as error:
            raise GpuTelemetryError("LLAMA_BROKER_GPU_DEVICE must be a non-negative integer") from error
        if index < 0 or index >= count:
            raise GpuTelemetryError(f"selected GPU device {index} is outside {count} visible devices")
        return index
    if count != 1:
        raise GpuTelemetryError(f"ambiguous {provider} device selection: {count} devices are visible")
    return 0


def _mib(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GpuTelemetryError(f"invalid {field} GPU memory value")
    return value // (1024 * 1024)


def _amd_cards(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    cards: list[Path] = []
    for card in sorted(root.glob("card[0-9]*")):
        device = card / "device"
        if (device / "mem_info_vram_total").is_file() or (device / "mem_info_vram_used").is_file():
            cards.append(device)
    return cards


def probe_amd_memory(*, sysfs_drm_root: Path | str = "/sys/class/drm") -> tuple[int, int] | None:
    cards = _amd_cards(Path(sysfs_drm_root))
    if not cards:
        return None
    index = _selected_index(count=len(cards), provider="AMD")
    device = cards[index]
    try:
        total = int((device / "mem_info_vram_total").read_text(encoding="ascii").strip(), 10)
        used = int((device / "mem_info_vram_used").read_text(encoding="ascii").strip(), 10)
    except (OSError, ValueError) as error:
        raise GpuTelemetryError(f"cannot read AMD VRAM telemetry from {device}: {error}") from error
    total_mib = _mib(total, field="total")
    used_mib = _mib(used, field="used")
    if total_mib <= 0 or used_mib > total_mib:
        raise GpuTelemetryError("AMD VRAM telemetry has an invalid total/used pair")
    if used > total:
        raise GpuTelemetryError("AMD VRAM usage exceeds capacity")
    return total_mib, (total - used) // (1024 * 1024)


def _cuda_error(loader: Any, result: int, operation: str) -> GpuTelemetryError:
    text = ""
    get_name = getattr(loader, "cuGetErrorName", None)
    if callable(get_name):
        name = ctypes.c_char_p()
        try:
            if get_name(result, ctypes.byref(name)) == 0 and name.value:
                text = f" ({name.value.decode('ascii', 'replace')})"
        except Exception:
            pass
    return GpuTelemetryError(f"CUDA {operation} failed with code {result}{text}")


def probe_nvidia_memory(*, loader: Any | None = None, cuda_loader: CudaLoader | None = None) -> tuple[int, int] | None:
    if loader is None:
        try:
            library = cuda_loader() if cuda_loader is not None else ctypes.CDLL("libcuda.so.1")
        except (OSError, AttributeError) as error:
            if cuda_loader is not None:
                raise GpuTelemetryError(f"cannot load CUDA driver API: {error}") from error
            return None
    else:
        library = loader

    # Bind ABI explicitly. Tests and injected compatibility loaders may expose
    # plain Python callables, so only assign attributes present on ctypes funcs.
    signatures = {
        "cuInit": (ctypes.c_int, [ctypes.c_uint]),
        "cuDeviceGetCount": (ctypes.c_int, [ctypes.POINTER(ctypes.c_int)]),
        "cuDeviceGet": (ctypes.c_int, [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
        "cuDevicePrimaryCtxRetain": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]),
        "cuCtxPushCurrent": (ctypes.c_int, [ctypes.c_void_p]),
        "cuMemGetInfo": (ctypes.c_int, [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]),
        "cuCtxPopCurrent": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)]),
        "cuDevicePrimaryCtxRelease": (ctypes.c_int, [ctypes.c_int]),
    }
    aliases = {
        "cuCtxPushCurrent": ("cuCtxPushCurrent_v2", "cuCtxPushCurrent"),
        "cuMemGetInfo": ("cuMemGetInfo_v2", "cuMemGetInfo"),
        "cuCtxPopCurrent": ("cuCtxPopCurrent_v2", "cuCtxPopCurrent"),
    }
    resolved: dict[str, Any] = {}
    for name, (restype, argtypes) in signatures.items():
        function = next((getattr(library, candidate, None) for candidate in aliases.get(name, (name,)) if callable(getattr(library, candidate, None))), None)
        if function is None:
            raise GpuTelemetryError(f"CUDA driver API is missing {name}")
        if hasattr(function, "restype"):
            function.restype = restype
            function.argtypes = argtypes
        resolved[name] = function

    def call(name: str, *args: Any) -> None:
        result = int(resolved[name](*args))
        if result != 0:
            raise _cuda_error(library, result, name)

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    # The driver enumerates the CUDA-visible namespace; do not manufacture
    # a device count from environment tokens (invalid lists may expose fewer).
    index = _selected_index(count=count.value, provider="NVIDIA")
    device = ctypes.c_int()
    call("cuDeviceGet", ctypes.byref(device), index)
    context = ctypes.c_void_p()
    retained = False
    pushed = False
    try:
        call("cuDevicePrimaryCtxRetain", ctypes.byref(context), device)
        retained = True
        call("cuCtxPushCurrent", context)
        pushed = True
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        call("cuMemGetInfo", ctypes.byref(free), ctypes.byref(total))
        total_mib = _mib(total.value, field="total")
        free_mib = _mib(free.value, field="free")
        if total_mib <= 0 or free_mib > total_mib:
            raise GpuTelemetryError("CUDA VRAM telemetry has an invalid total/free pair")
        return total_mib, free_mib
    finally:
        cleanup_error: GpuTelemetryError | None = None
        if pushed:
            try:
                call("cuCtxPopCurrent", ctypes.byref(ctypes.c_void_p()))
            except GpuTelemetryError as error:
                cleanup_error = error
        if retained:
            try:
                call("cuDevicePrimaryCtxRelease", device)
            except GpuTelemetryError as error:
                cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            raise cleanup_error


def probe_gpu_memory(*, sysfs_drm_root: Path | str = "/sys/class/drm", cuda_loader: CudaLoader | None = None) -> tuple[int, int] | None:
    """Probe exactly one unambiguously selected GPU using native APIs."""
    amd = probe_amd_memory(sysfs_drm_root=sysfs_drm_root)
    if amd is not None:
        return amd
    return probe_nvidia_memory(cuda_loader=cuda_loader)


__all__ = ["GpuTelemetryError", "probe_amd_memory", "probe_gpu_memory", "probe_nvidia_memory"]
