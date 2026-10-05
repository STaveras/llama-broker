from __future__ import annotations

import ctypes
from pathlib import Path
from unittest.mock import patch

import pytest

from llama_broker.gpu_memory import (
    GpuTelemetryError,
    probe_gpu_memory,
    probe_nvidia_memory,
)


def _write_amd_card(root: Path, name: str, total: int, used: int) -> None:
    device = root / name / "device"
    device.mkdir(parents=True)
    (device / "mem_info_vram_total").write_text(str(total), encoding="ascii")
    (device / "mem_info_vram_used").write_text(str(used), encoding="ascii")


def test_amd_sysfs_probe_reports_bytes_as_mib_and_selects_single_card(tmp_path: Path) -> None:
    drm = tmp_path / "class" / "drm"
    _write_amd_card(drm, "card0", 12_227 * 1024 * 1024, 327 * 1024 * 1024)

    assert probe_gpu_memory(sysfs_drm_root=drm, cuda_loader=lambda: None) == (12_227, 11_900)


def test_amd_probe_rejects_ambiguous_cards_without_selection(tmp_path: Path) -> None:
    drm = tmp_path / "class" / "drm"
    _write_amd_card(drm, "card0", 12_000 * 1024 * 1024, 100 * 1024 * 1024)
    _write_amd_card(drm, "card1", 24_000 * 1024 * 1024, 100 * 1024 * 1024)

    with pytest.raises(GpuTelemetryError, match="ambiguous"):
        probe_gpu_memory(sysfs_drm_root=drm, cuda_loader=lambda: None)


def test_nvidia_context_is_balanced_on_memory_error() -> None:
    calls: list[str] = []
    context = ctypes.c_void_p(123)

    class FakeCuda:
        def cuInit(self, flags):
            calls.append("init")
            return 0

        def cuDeviceGetCount(self, count):
            count._obj.value = 1
            return 0

        def cuDeviceGet(self, device, ordinal):
            device._obj.value = 0
            return 0

        def cuDevicePrimaryCtxRetain(self, ctx, device):
            calls.append("retain")
            ctx._obj.value = context.value
            return 0

        def cuCtxPushCurrent(self, ctx):
            calls.append("push")
            return 0

        def cuMemGetInfo(self, free, total):
            calls.append("mem")
            return 2

        def cuCtxPopCurrent(self, ctx):
            calls.append("pop")
            return 0

        def cuDevicePrimaryCtxRelease(self, device):
            calls.append("release")
            return 0

    with pytest.raises(GpuTelemetryError, match="cuMemGetInfo"):
        probe_nvidia_memory(loader=FakeCuda())
    assert calls == ["init", "retain", "push", "mem", "pop", "release"]


def test_nvidia_multiple_visible_devices_requires_explicit_selection() -> None:
    class FakeCuda:
        def cuInit(self, flags): return 0
        def cuDeviceGetCount(self, count):
            count._obj.value = 2
            return 0
        def cuDeviceGet(self, device, ordinal): return 0
        def cuDevicePrimaryCtxRetain(self, ctx, device): return 0
        def cuCtxPushCurrent(self, ctx): return 0
        def cuMemGetInfo(self, free, total): return 0
        def cuCtxPopCurrent(self, ctx): return 0
        def cuDevicePrimaryCtxRelease(self, device): return 0
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
        with pytest.raises(GpuTelemetryError, match="ambiguous"):
            probe_nvidia_memory(loader=FakeCuda())


def test_nvidia_single_visible_device_uses_visible_ordinal_not_host_gpu_zero() -> None:
    calls: list[int] = []

    class FakeCuda:
        def cuInit(self, flags): return 0
        def cuDeviceGetCount(self, count):
            count._obj.value = 1
            return 0
        def cuDeviceGet(self, device, ordinal):
            calls.append(ordinal)
            device._obj.value = ordinal
            return 0
        def cuDevicePrimaryCtxRetain(self, ctx, device): ctx._obj.value = 1; return 0
        def cuCtxPushCurrent(self, ctx): return 0
        def cuMemGetInfo(self, free, total):
            free._obj.value = 8 * 1024 * 1024
            total._obj.value = 12 * 1024 * 1024
            return 0
        def cuCtxPopCurrent(self, ctx): return 0
        def cuDevicePrimaryCtxRelease(self, device): return 0

    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "3"}, clear=False):
        assert probe_nvidia_memory(loader=FakeCuda()) == (12, 8)
    assert calls == [0]
