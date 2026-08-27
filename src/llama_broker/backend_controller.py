"""Transactional lifecycle control for immutable llama-server profile units.

The controller deliberately knows nothing about HTTP routing or profile discovery.
Each profile owns a distinct systemd unit; switching starts and stops those units
without rewriting a shared ``llama-server.service`` file.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


class Profile(Protocol):
    """Minimum profile shape consumed by :class:`BackendController`.

    Runtime adapters own the complete backend command. ``args`` remains in the
    protocol for small legacy test doubles and downstream integrations.
    """

    id: str
    runtime: str
    runtime_build: str | None
    alias: str
    display_name: str
    server_binary: str | os.PathLike[str]
    model_path: str | os.PathLike[str]
    args: tuple[str, ...]
    environment: Mapping[str, str]
    startup_timeout: float
    safety: Any

    def command(
        self,
        *,
        host: str,
        port: int,
        managed_args: tuple[str, ...] = (),
    ) -> tuple[str, ...]: ...


CommandRunner = Callable[[tuple[str, ...]], Any]
ReadinessProbe = Callable[[str, float], bool]
EndpointProbe = Callable[[], bool]
VerifyCommand = Callable[[Profile], Sequence[str]]
ArtifactVerifier = Callable[[Profile], None]


@dataclass(frozen=True)
class ResourceSnapshot:
    """Linux host and primary-GPU capacity sampled at one transition boundary."""

    ram_available_mib: int
    swap_used_mib: int
    vram_total_mib: int
    vram_free_mib: int

    def as_dict(self) -> dict[str, int]:
        return {
            "ram_available_mib": self.ram_available_mib,
            "swap_used_mib": self.swap_used_mib,
            "vram_total_mib": self.vram_total_mib,
            "vram_free_mib": self.vram_free_mib,
        }


ResourceProbe = Callable[[], ResourceSnapshot]
EventLogger = Callable[[dict[str, object]], None]
CancellationProbe = Callable[[], bool]


def linux_resource_snapshot(
    proc_meminfo: str | os.PathLike[str] = "/proc/meminfo",
    *,
    nvidia_smi: str = "nvidia-smi",
    timeout: float = 5.0,
) -> ResourceSnapshot:
    """Sample host RAM/swap and primary-GPU capacity using Linux-native probes.

    The probe intentionally fails closed when either `/proc/meminfo` or the
    primary NVIDIA device cannot be read; callers convert that failure into a
    denied profile transition instead of starting a model without a safety
    snapshot. Available RAM is rounded down and swap usage is rounded up so
    admission never overstates usable capacity.
    """

    if timeout <= 0:
        raise ValueError("resource probe timeout must be positive")

    meminfo = Path(proc_meminfo).read_text(encoding="utf-8")
    values_kib: dict[str, int] = {}
    for line in meminfo.splitlines():
        key, separator, remainder = line.partition(":")
        if not separator:
            continue
        fields = remainder.split()
        if len(fields) < 2 or fields[1] != "kB":
            continue
        try:
            value = int(fields[0])
        except ValueError as error:
            raise ValueError(f"invalid /proc/meminfo value for {key}") from error
        if value < 0:
            raise ValueError(f"negative /proc/meminfo value for {key}")
        values_kib[key] = value

    required = ("MemAvailable", "SwapTotal", "SwapFree")
    missing = [key for key in required if key not in values_kib]
    if missing:
        raise ValueError("/proc/meminfo is missing " + ", ".join(missing))
    if values_kib["SwapFree"] > values_kib["SwapTotal"]:
        raise ValueError("/proc/meminfo reports SwapFree above SwapTotal")

    result = subprocess.run(
        (
            nvidia_smi,
            "-i",
            "0",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    line = next((candidate.strip() for candidate in result.stdout.splitlines() if candidate.strip()), "")
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 2:
        raise ValueError("nvidia-smi returned no primary-GPU memory pair")
    try:
        vram_total_mib = int(fields[0])
        vram_free_mib = int(fields[1])
    except ValueError as error:
        raise ValueError("nvidia-smi returned non-numeric primary-GPU memory") from error
    if vram_total_mib <= 0 or vram_free_mib < 0 or vram_free_mib > vram_total_mib:
        raise ValueError("nvidia-smi returned an invalid primary-GPU memory pair")

    def floor_mib(kib: int) -> int:
        return kib // 1024

    def ceil_mib(kib: int) -> int:
        return (kib + 1023) // 1024

    return ResourceSnapshot(
        ram_available_mib=floor_mib(values_kib["MemAvailable"]),
        swap_used_mib=ceil_mib(values_kib["SwapTotal"] - values_kib["SwapFree"]),
        vram_total_mib=vram_total_mib,
        vram_free_mib=vram_free_mib,
    )


@dataclass(frozen=True)
class ServiceStateSnapshot:
    """Process-bearing systemd state used to prove a backend is gone."""

    active_state: str
    sub_state: str
    main_pid: int
    control_pid: int

    @property
    def is_quiescent(self) -> bool:
        return (
            self.active_state in {"inactive", "failed"}
            and self.sub_state in {"dead", "failed"}
            and self.main_pid == 0
            and self.control_pid == 0
        )

    def as_dict(self) -> dict[str, str | int]:
        return {
            "active_state": self.active_state,
            "sub_state": self.sub_state,
            "main_pid": self.main_pid,
            "control_pid": self.control_pid,
        }


ServiceStateProbe = Callable[[str], ServiceStateSnapshot]
ServiceInventoryProbe = Callable[[str], Sequence[str]]


def systemd_service_state_probe(unit_name: str) -> ServiceStateSnapshot:
    """Read exact user-unit process state without any Windows-side IPC."""

    result = subprocess.run(
        (
            "systemctl",
            "--user",
            "show",
            unit_name,
            "--no-pager",
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            "--property=ControlPID",
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    missing = [
        key
        for key in ("ActiveState", "SubState", "MainPID", "ControlPID")
        if key not in values
    ]
    if missing:
        raise RuntimeError(
            f"systemctl show omitted properties for {unit_name!r}: {', '.join(missing)}"
        )
    try:
        main_pid = int(values["MainPID"])
        control_pid = int(values["ControlPID"])
    except ValueError as error:
        raise RuntimeError(
            f"systemctl show returned invalid PIDs for {unit_name!r}"
        ) from error
    if main_pid < 0 or control_pid < 0:
        raise RuntimeError(f"systemctl show returned negative PIDs for {unit_name!r}")
    return ServiceStateSnapshot(
        active_state=values["ActiveState"],
        sub_state=values["SubState"],
        main_pid=main_pid,
        control_pid=control_pid,
    )


class BackendControllerError(RuntimeError):
    """Base class for controller failures."""


class ConfigurationError(BackendControllerError):
    """A profile, unit name, or persisted state is invalid."""


class CommandError(BackendControllerError):
    """An injected lifecycle command returned a non-zero status."""


class ReadinessError(BackendControllerError):
    """A backend did not become ready with its exact configured alias."""


class AdmissionError(BackendControllerError):
    """A fail-closed resource admission decision rejected an activation."""


class QuiescenceError(BackendControllerError):
    """A backend unit could not be proven process-free."""


class LifecycleCancelledError(BackendControllerError):
    """A start or admission step was cancelled before model activation."""


class ServiceOwnershipError(BackendControllerError):
    """An exact-alias endpoint is not owned by the expected systemd unit."""


class SwitchError(BackendControllerError):
    """A target switch failed and the controller recorded fail-closed state."""

    def __init__(self, target_profile: str, cause: BaseException) -> None:
        self.target_profile = target_profile
        self.cause = cause
        super().__init__(f"failed to switch to profile {target_profile!r}: {cause}")


class RollbackError(SwitchError):
    """Legacy compatibility error; new transitions do not attempt rollback."""

    def __init__(
        self,
        target_profile: str,
        prior_profile: str,
        target_error: BaseException,
        rollback_error: BaseException,
    ) -> None:
        self.prior_profile = prior_profile
        self.target_error = target_error
        self.rollback_error = rollback_error
        super().__init__(
            target_profile,
            RuntimeError(
                f"target error: {target_error}; rollback of {prior_profile!r} failed: "
                f"{rollback_error}"
            ),
        )


_UNIT_COMPONENT = re.compile(r"^[A-Za-z0-9_.@-]+$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LEGACY_SERVICE_UNIT = "llama-server.service"


def _profile_id_from_unit_name(unit_name: str, service_prefix: str) -> str | None:
    prefix = f"{service_prefix}-"
    suffix = ".service"
    if not unit_name.startswith(prefix) or not unit_name.endswith(suffix):
        return None
    profile_id = unit_name[len(prefix) : -len(suffix)]
    if not profile_id or not _UNIT_COMPONENT.fullmatch(profile_id):
        return None
    return profile_id


def systemd_service_inventory_probe(service_prefix: str) -> tuple[str, ...]:
    """List loaded user services belonging to one exact broker prefix."""

    if not _UNIT_COMPONENT.fullmatch(service_prefix) or service_prefix.endswith(
        ".service"
    ):
        raise ValueError(f"invalid service prefix: {service_prefix!r}")
    result = subprocess.run(
        (
            "systemctl",
            "--user",
            "list-units",
            "--type=service",
            "--all",
            "--no-legend",
            "--plain",
            "--no-pager",
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    units: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=1)
        if not fields:
            continue
        unit_name = fields[0]
        if _profile_id_from_unit_name(unit_name, service_prefix) is not None:
            units.add(unit_name)
    return tuple(sorted(units))


def _reject_control(value: str, field: str, *, allow_empty: bool = False) -> None:
    if not value and not allow_empty:
        raise ConfigurationError(f"{field} must not be empty")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise ConfigurationError(f"{field} contains a forbidden control character")


def _unit_quote(value: str, *, escape_dollar: bool = False) -> str:
    """Quote one systemd command/environment word without shell evaluation."""

    _reject_control(value, "unit value", allow_empty=True)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if escape_dollar:
        escaped = escaped.replace("$", "$$")
    return f'"{escaped}"'


def _atomic_write(path: Path, text: str) -> None:
    """Durably replace *path* using a temporary file in the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)

        # Persist the directory entry where the platform supports directory fsync.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _replace_option(args: list[str], option: str, value: str) -> list[str]:
    """Replace a two-token option, appending it when a test profile omits it."""

    result: list[str] = []
    replaced = False
    index = 0
    while index < len(args):
        token = args[index]
        if token == option:
            result.extend((option, value))
            replaced = True
            index += 2 if index + 1 < len(args) else 1
            continue
        result.append(token)
        index += 1
    if not replaced:
        result.extend((option, value))
    return result


class BackendController:
    """Install profile units and switch among them transactionally.

    ``command_runner`` receives an argv tuple and should raise on failure.
    Returning a ``CompletedProcess``-like value with a non-zero ``returncode`` is
    also treated as failure. ``readiness_probe`` receives the *exact* configured
    alias and timeout for the profile and must return truthy only after that alias
    is served by the backend.
    """

    def __init__(
        self,
        *,
        profiles: Mapping[str, Profile],
        state_path: str | Path,
        systemd_dir: str | Path,
        service_prefix: str,
        command_runner: CommandRunner,
        readiness_probe: ReadinessProbe,
        resource_probe: ResourceProbe | None,
        event_logger: EventLogger,
        service_state_probe: ServiceStateProbe | None = None,
        service_inventory_probe: ServiceInventoryProbe | None = None,
        cancellation_probe: CancellationProbe | None = None,
        endpoint_probe: EndpointProbe | None = None,
        backend_listen_host: str | None = None,
        backend_listen_port: int | None = None,
        managed_args: Sequence[str] | Mapping[str, Sequence[str]] = (),
        verify_command: VerifyCommand | None = None,
        artifact_verifier: ArtifactVerifier | None = None,
    ) -> None:
        if not profiles:
            raise ConfigurationError("at least one profile is required")
        if not _UNIT_COMPONENT.fullmatch(service_prefix):
            raise ConfigurationError(f"invalid service prefix: {service_prefix!r}")
        if service_prefix.endswith(".service"):
            raise ConfigurationError("service prefix must not include .service")

        self.profiles = dict(profiles)
        self.state_path = Path(state_path)
        self.systemd_dir = Path(systemd_dir)
        self.service_prefix = service_prefix
        self.command_runner = command_runner
        self.readiness_probe = readiness_probe
        self.resource_probe = resource_probe
        self.event_logger = event_logger
        self.service_state_probe = service_state_probe
        self.service_inventory_probe = service_inventory_probe
        self.cancellation_probe = cancellation_probe
        self.endpoint_probe = endpoint_probe
        if backend_listen_port is not None and backend_listen_port <= 0:
            raise ConfigurationError("backend_listen_port must be positive")
        self.backend_listen_host = backend_listen_host
        self.backend_listen_port = backend_listen_port
        self.verify_command = verify_command
        self.artifact_verifier = artifact_verifier
        if isinstance(managed_args, Mapping):
            self.managed_args = {
                runtime: tuple(args) for runtime, args in managed_args.items()
            }
        else:
            self.managed_args = tuple(managed_args)
        self._lock = threading.RLock()
        self._validate_profiles()

    def _managed_args_for(self, profile: Profile) -> tuple[str, ...]:
        if isinstance(self.managed_args, Mapping):
            runtime = str(getattr(profile, "runtime", "llamacpp"))
            return tuple(self.managed_args.get(runtime, ()))
        return tuple(self.managed_args)

    def _command_for(self, profile: Profile) -> tuple[str, ...]:
        host = self.backend_listen_host or "127.0.0.1"
        port = self.backend_listen_port or 10000
        builder = getattr(profile, "command", None)
        if callable(builder):
            return tuple(
                builder(
                    host=host,
                    port=port,
                    managed_args=self._managed_args_for(profile),
                )
            )

        # Compatibility path for legacy profile-shaped test doubles. Production
        # profiles always expose command(), so no runtime-specific argv is
        # assembled here.
        profile_args = list(profile.args)
        if self.backend_listen_host is not None:
            profile_args = _replace_option(profile_args, "--host", host)
        if self.backend_listen_port is not None:
            profile_args = _replace_option(profile_args, "--port", str(port))
        return (
            str(profile.server_binary),
            "-m",
            str(profile.model_path),
            "--alias",
            profile.alias,
            *profile_args,
            *self._managed_args_for(profile),
        )

    def _validate_profiles(self) -> None:
        aliases: set[str] = set()
        for key, profile in self.profiles.items():
            if key != profile.id:
                raise ConfigurationError(
                    f"profile mapping key {key!r} does not match id {profile.id!r}"
                )
            if not _UNIT_COMPONENT.fullmatch(profile.id):
                raise ConfigurationError(f"invalid profile id: {profile.id!r}")
            for field_name in ("alias", "display_name"):
                value = getattr(profile, field_name)
                if not isinstance(value, str):
                    raise ConfigurationError(f"{profile.id}.{field_name} must be a string")
                _reject_control(value, f"{profile.id}.{field_name}")
            for field_name in ("server_binary", "model_path"):
                value = getattr(profile, field_name)
                if not isinstance(value, (str, os.PathLike)):
                    raise ConfigurationError(
                        f"{profile.id}.{field_name} must be a string or path-like value"
                    )
                _reject_control(str(value), f"{profile.id}.{field_name}")
            if profile.alias in aliases:
                raise ConfigurationError(f"duplicate profile alias: {profile.alias!r}")
            aliases.add(profile.alias)
            if profile.startup_timeout <= 0:
                raise ConfigurationError(
                    f"{profile.id}.startup_timeout must be greater than zero"
                )
            builder = getattr(profile, "command", None)
            if callable(builder):
                command = self._command_for(profile)
                for index, argument in enumerate(command):
                    if not isinstance(argument, str):
                        raise ConfigurationError(
                            f"{profile.id}.command[{index}] must be a string"
                        )
                    _reject_control(
                        argument, f"{profile.id}.command[{index}]", allow_empty=True
                    )
            else:
                for index, argument in enumerate(profile.args):
                    if not isinstance(argument, str):
                        raise ConfigurationError(f"{profile.id}.args[{index}] must be a string")
                    _reject_control(
                        argument, f"{profile.id}.args[{index}]", allow_empty=True
                    )
                    if argument in {"-m", "--model", "--alias"}:
                        raise ConfigurationError(
                            f"{profile.id}.args must not override controller-owned "
                            f"argument {argument!r}"
                        )
            for name, value in profile.environment.items():
                if not _ENVIRONMENT_NAME.fullmatch(name):
                    raise ConfigurationError(
                        f"invalid environment name in {profile.id}: {name!r}"
                    )
                if not isinstance(value, str):
                    raise ConfigurationError(
                        f"environment value {profile.id}.{name} must be a string"
                    )
                _reject_control(
                    value,
                    f"{profile.id}.environment[{name!r}]",
                    allow_empty=True,
                )
            try:
                service = profile.safety.service
                admission = profile.safety.admission
            except AttributeError as error:
                raise ConfigurationError(
                    f"{profile.id}.safety is missing required service/admission budgets"
                ) from error
            if service.service_type != "exec":
                raise ConfigurationError(
                    f"{profile.id}.safety.service.service_type must be 'exec'"
                )
            if service.kill_signal != "SIGINT":
                raise ConfigurationError(
                    f"{profile.id}.safety.service.kill_signal must be 'SIGINT'"
                )
            for field_name in (
                "memory_max_mib",
                "tasks_max",
                "timeout_stop_sec",
            ):
                value = getattr(service, field_name, None)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ConfigurationError(
                        f"{profile.id}.safety.service.{field_name} must be positive"
                    )
            if (
                isinstance(service.memory_swap_max_mib, bool)
                or not isinstance(service.memory_swap_max_mib, int)
                or service.memory_swap_max_mib < 0
                or service.memory_swap_max_mib > service.memory_max_mib
            ):
                raise ConfigurationError(
                    f"{profile.id}.safety.service.memory_swap_max_mib is invalid"
                )
            for field_name in (
                "min_ram_available_mib",
                "max_swap_used_mib",
                "max_swap_growth_mib",
                "min_total_vram_mib",
                "min_free_vram_before_start_mib",
                "driver_vram_reserve_mib",
                "vision_vram_reserve_mib",
                "safety_vram_reserve_mib",
            ):
                value = getattr(admission, field_name, None)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ConfigurationError(
                        f"{profile.id}.safety.admission.{field_name} is invalid"
                    )
            if admission.min_ram_available_mib <= 0 or admission.min_total_vram_mib <= 0:
                raise ConfigurationError(
                    f"{profile.id}.safety admission RAM/VRAM floors must be positive"
                )
            if (
                admission.post_ready_min_free_vram_mib
                > admission.min_free_vram_before_start_mib
            ):
                raise ConfigurationError(
                    f"{profile.id}.safety post-ready reserve exceeds pre-start floor"
                )

    def unit_name(self, profile_id: str) -> str:
        self._profile(profile_id)
        return f"{self.service_prefix}-{profile_id}.service"

    def unit_path(self, profile_id: str) -> Path:
        return self.systemd_dir / self.unit_name(profile_id)

    def _verify_service_prefix(self) -> str:
        if self.service_prefix.endswith("-profile"):
            return f"{self.service_prefix[:-len('-profile')]}-verify-profile"
        return f"{self.service_prefix}-verify-profile"

    def verify_unit_name(self, profile_id: str) -> str:
        self._profile(profile_id)
        return f"{self._verify_service_prefix()}-{profile_id}.service"

    def verify_unit_path(self, profile_id: str) -> Path:
        return self.systemd_dir / self.verify_unit_name(profile_id)

    def owned_unit_names(self) -> tuple[str, ...]:
        profile_units = [self.unit_name(profile_id) for profile_id in sorted(self.profiles)]
        verify_units = [
            self.verify_unit_name(profile_id) for profile_id in sorted(self.profiles)
        ]
        return tuple(profile_units + verify_units)

    def _update_alias(self, profile_id: str | None) -> None:
        """Maintain a ``llama-server.service`` symlink to the active profile unit.

        When *profile_id* is ``None``, the symlink is removed.
        """
        alias_path = self.systemd_dir / "llama-server.service"
        try:
            if profile_id is None:
                if alias_path.is_symlink() or alias_path.exists():
                    alias_path.unlink()
            else:
                target = self.unit_name(profile_id)
                if alias_path.is_symlink():
                    current = alias_path.readlink().name if alias_path.exists() else None
                    if current == target:
                        return
                    alias_path.unlink()
                elif alias_path.exists():
                    alias_path.unlink()
                alias_path.symlink_to(target)
        except OSError:
            pass  # Best-effort; don't fail the operation over a symlink

    def render_unit(self, profile_id: str) -> str:
        profile = self._profile(profile_id)
        service = profile.safety.service
        command: Sequence[str] = self._command_for(profile)
        lines = [
            "[Unit]",
            f"Description=llama-broker profile: {profile.display_name}",
            "After=network-online.target",
            "Wants=network-online.target",
            f"X-Llama-Broker-Profile={profile.id}",
            "",
            "[Service]",
            f"Type={service.service_type}",
            f"MemoryMax={service.memory_max_mib}M",
            f"MemorySwapMax={service.memory_swap_max_mib}M",
            f"TasksMax={service.tasks_max}",
            f"KillSignal={service.kill_signal}",
            f"TimeoutStopSec={service.timeout_stop_sec}s",
            "KillMode=control-group",
            "SendSIGKILL=no",
            "ExecStart="
            + " ".join(_unit_quote(word, escape_dollar=True) for word in command),
        ]
        lines.extend(
            f"Environment={_unit_quote(f'{name}={value}')}"
            for name, value in sorted(profile.environment.items())
        )
        lines.extend(
            [
                "Restart=no",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        )
        return "\n".join(lines)

    def render_verify_unit(self, profile_id: str) -> str:
        profile = self._profile(profile_id)
        if self.verify_command is None:
            command: tuple[str, ...] = ("/usr/bin/true",)
        else:
            command = tuple(self.verify_command(profile))
        if not command or any(not isinstance(word, str) or not word for word in command):
            raise ConfigurationError(
                f"verification command for {profile.id!r} must contain non-empty strings"
            )
        lines = [
            "[Unit]",
            f"Description=Verify llama-broker profile: {profile.display_name}",
            "After=network-online.target",
            "Wants=network-online.target",
            f"X-Llama-Broker-Profile={profile.id}",
            "",
            "[Service]",
            "Type=oneshot",
            "ExecStart="
            + " ".join(_unit_quote(word, escape_dollar=True) for word in command),
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
        return "\n".join(lines)

    def install_units(self) -> dict[str, Path]:
        """Atomically install one stable unit file per configured profile."""

        with self._lock:
            installed: dict[str, Path] = {}
            for profile_id in sorted(self.profiles):
                path = self.unit_path(profile_id)
                _atomic_write(path, self.render_unit(profile_id))
                _atomic_write(
                    self.verify_unit_path(profile_id), self.render_verify_unit(profile_id)
                )
                installed[profile_id] = path
            self._run("systemctl", "--user", "daemon-reload")
            # daemon-reload updates definitions but preserves a prior failed
            # result; clear only those recorded states so one stale crash does
            # not block the next lifecycle transition.
            try:
                self._run(
                    "systemctl",
                    "--user",
                    "reset-failed",
                    *self.owned_unit_names(),
                )
            except CommandError:
                # A freshly installed inactive unit is valid, but systemd exits
                # non-zero when reset-failed is given a unit that has never
                # entered a failed state.  Clear the manager's failed set as a
                # whole instead of treating that harmless state as startup
                # failure; genuine daemon-reload errors still propagate above.
                self._run("systemctl", "--user", "reset-failed")
            state = self._read_state()
            if state.get("phase") == "active" and state.get("profile"):
                self._update_alias(state["profile"])
            return installed

    def cleanup_stale_units(self) -> tuple[Path, ...]:
        """Remove inactive broker-owned unit files absent from the manifest."""

        with self._lock:
            if self.service_state_probe is None:
                raise ServiceOwnershipError(
                    "cannot clean stale broker units without a systemd state probe"
                )

            manifest_ids = set(self.profiles)
            prefixes = (self.service_prefix, self._verify_service_prefix())
            candidates: dict[str, Path] = {}
            for prefix in prefixes:
                if not self.systemd_dir.is_dir():
                    continue
                for path in self.systemd_dir.glob(f"{prefix}-*.service"):
                    profile_id = _profile_id_from_unit_name(path.name, prefix)
                    if profile_id is not None and profile_id not in manifest_ids:
                        candidates[path.name] = path

            stale = tuple(candidates[name] for name in sorted(candidates))
            if not stale:
                return ()

            active_profiles: set[str] = set()
            for path in stale:
                profile_id = _profile_id_from_unit_name(path.name, self.service_prefix)
                if profile_id is None:
                    profile_id = _profile_id_from_unit_name(
                        path.name, self._verify_service_prefix()
                    )
                try:
                    snapshot = self.service_state_probe(path.name)
                except Exception as error:
                    raise ServiceOwnershipError(
                        f"could not determine whether stale unit {path.name!r} is active: {error}"
                    ) from error
                if not isinstance(snapshot, ServiceStateSnapshot):
                    raise ServiceOwnershipError(
                        f"systemd state probe returned an invalid snapshot for {path.name!r}"
                    )
                if not snapshot.is_quiescent:
                    active_profiles.add(profile_id or path.name)

            if active_profiles:
                profiles = ", ".join(sorted(active_profiles))
                raise ServiceOwnershipError(
                    "refusing to remove active broker unit(s) for profile(s): "
                    f"{profiles}; unload the profile first"
                )

            for path in stale:
                path.unlink()

            self._run("systemctl", "--user", "daemon-reload")
            removed_names = tuple(path.name for path in stale)
            try:
                self._run("systemctl", "--user", "reset-failed", *removed_names)
            except CommandError:
                self._run("systemctl", "--user", "reset-failed")
            return stale

    def switch(
        self,
        profile_id: str,
        *,
        force_restart: bool = False,
    ) -> dict[str, Any]:
        """Switch to *profile_id*, restoring the prior profile on any failure."""

        with self._lock:
            self._require_not_cancelled("switch")
            target = self._profile(profile_id)
            prior_state = self._read_state()
            if prior_state["phase"] == "switching":
                prior_state = self.reconcile()
            prior_id = prior_state.get("profile")
            if prior_id is not None:
                self._profile(prior_id)
            generation = prior_state["generation"]
            self._emit(
                "profile_transition",
                decision="requested",
                prior_profile=prior_id,
                target_profile=profile_id,
                generation=generation,
            )

            restart_current = force_restart
            if prior_id == profile_id and prior_state["phase"] == "active":
                if not restart_current:
                    try:
                        self._probe_owned(profile_id, target)
                    except (ReadinessError, ServiceOwnershipError):
                        # Persisted state can outlive the process.  Fall through to a
                        # controlled restart when the exact alias is no longer ready.
                        restart_current = True
                    else:
                        self._emit(
                            "profile_transition",
                            decision="already_active",
                            prior_profile=prior_id,
                            target_profile=profile_id,
                            generation=generation,
                        )
                        return prior_state

            switching_state = {
                "generation": generation,
                "profile": prior_id,
                "phase": "switching",
                "target_profile": profile_id,
            }
            self._write_state(switching_state)

            target_start_attempted = False
            try:
                if prior_id is not None and (
                    prior_id != profile_id
                    or prior_state["phase"] != "active"
                    or restart_current
                ):
                    self._stop(prior_id)
                pre_start_snapshot = self._admit_pre_start(target)
                self._emit(
                    "profile_transition",
                    decision="target_starting",
                    prior_profile=prior_id,
                    target_profile=profile_id,
                    generation=generation,
                )
                target_start_attempted = True
                self._start(profile_id)
                self._probe_owned(profile_id, target)
                self._admit_post_ready(target, pre_start_snapshot)
                self._emit(
                    "profile_transition",
                    decision="target_ready",
                    prior_profile=prior_id,
                    target_profile=profile_id,
                    generation=generation,
                )
            except Exception as target_error:
                self._emit(
                    "profile_transition",
                    decision="target_failed",
                    prior_profile=prior_id,
                    target_profile=profile_id,
                    generation=generation,
                    error_type=type(target_error).__name__,
                    error=str(target_error),
                )
                self._handle_failed_switch(
                    profile_id=profile_id,
                    prior_id=prior_id,
                    generation=generation,
                    target_error=target_error,
                    cleanup_target=target_start_attempted,
                )
                raise AssertionError("failed switch handler must raise")

            active_state = {
                "generation": generation + 1,
                "profile": profile_id,
                "phase": "active",
            }
            runtime_build = getattr(target, "runtime_build", None)
            server_sha256 = getattr(target, "server_sha256", None)
            if isinstance(runtime_build, str) and runtime_build and isinstance(server_sha256, str) and server_sha256:
                active_state["runtime_build"] = runtime_build
                active_state["server_sha256"] = server_sha256
            self._write_state(active_state)
            self._update_alias(profile_id)
            self._emit(
                "profile_transition",
                decision="committed",
                prior_profile=prior_id,
                target_profile=profile_id,
                generation=active_state["generation"],
            )
            return active_state

    def restart(self, profile_id: str) -> dict[str, Any]:
        """Restart a profile even when its alias is currently healthy."""

        return self.switch(profile_id, force_restart=True)

    def state(self) -> dict[str, Any]:
        """Return a validated snapshot of the durable controller state."""

        with self._lock:
            return dict(self._read_state())

    def record_recovery_failure(
        self, profile_id: str, *, attempts: int, error: str
    ) -> dict[str, Any]:
        """Persist a bounded recovery attempt without starting a backend."""

        with self._lock:
            self._profile(profile_id)
            state = self._read_state()
            failure_state = {
                "generation": state["generation"] + 1,
                "profile": None,
                "phase": "stale",
                "stale_profile": profile_id,
                "recovery": {
                    "attempts": max(1, int(attempts)),
                    "error": str(error),
                    "action": "broker_watchdog_restart",
                },
            }
            self._write_state(failure_state)
            self._update_alias(None)
            self._emit(
                "profile_recovery",
                decision="failed",
                profile=profile_id,
                attempts=failure_state["recovery"]["attempts"],
                error=str(error),
            )
            return failure_state

    def stop_all(self) -> dict[str, Any]:
        """Cooperatively stop every broker unit without ever starting a model."""

        with self._lock:
            state = self._read_state()
            generation = state["generation"]
            failures: dict[str, BaseException] = {}

            try:
                discovered_units = self._inventory_unit_names()
            except QuiescenceError as error:
                discovered_units = set()
                failures["service-inventory"] = error

            profile_units = {
                self.unit_name(profile_id) for profile_id in self.profiles
            }
            for unit_name in sorted(profile_units | discovered_units):
                try:
                    self._stop_unit(unit_name)
                except Exception as error:
                    failures[self._unit_label(unit_name)] = error

            # The historical standalone unit is outside the strict broker prefix.
            # Avoid issuing ``stop`` for a missing, already-quiescent unit, while
            # still stopping and verifying it whenever it can carry a process.
            if self.service_state_probe is not None:
                try:
                    self._require_unit_quiescent(
                        _LEGACY_SERVICE_UNIT, stage="pre_stop_all"
                    )
                except QuiescenceError:
                    try:
                        self._stop_unit(_LEGACY_SERVICE_UNIT)
                    except Exception as error:
                        failures[_LEGACY_SERVICE_UNIT] = error

            try:
                self._require_endpoint_absent(stage="post_stop_all")
            except ServiceOwnershipError as error:
                failures["private-endpoint"] = error

            if failures:
                failed_state = {
                    "generation": generation,
                    "profile": None,
                    "phase": "stop_failed",
                    "previous_profile": state.get("profile"),
                    "failed_profiles": sorted(failures),
                    "errors": {
                        profile_id: str(error)
                        for profile_id, error in failures.items()
                    },
                }
                self._write_state(failed_state)
                self._update_alias(None)
                failed_ids = ", ".join(sorted(failures))
                raise QuiescenceError(
                    f"could not prove all broker backends stopped: {failed_ids}"
                ) from next(iter(failures.values()))

            inactive_state = {
                "generation": generation + 1,
                "profile": None,
                "phase": "inactive",
            }
            self._write_state(inactive_state)
            self._update_alias(None)
            return inactive_state

    def reconcile(
        self, *, active_probe_timeout: float | None = None
    ) -> dict[str, Any]:
        """Reconcile durable state with the backend that is actually serving.

        Interrupted switches are failed closed after the target is quiesced;
        the prior profile is never restarted automatically.  When
        ``active_probe_timeout`` is supplied, an ``active`` record must also
        pass a bounded exact-alias readiness probe.  A missing backend is
        recorded as stale without trying to start or stop any service.
        """

        with self._lock:
            state = self._read_state()
            if state["phase"] == "active" and active_probe_timeout is not None:
                if active_probe_timeout <= 0:
                    raise ValueError("active_probe_timeout must be positive")
                profile_id = state.get("profile")
                if not isinstance(profile_id, str):
                    raise ConfigurationError("active state is missing profile")
                profile = self._profile(profile_id)
                try:
                    ready = bool(
                        self.readiness_probe(
                            profile.alias, float(active_probe_timeout)
                        )
                    )
                except Exception:
                    ready = False
                if ready:
                    try:
                        self._require_running(profile_id, stage="reconcile_active")
                    except ServiceOwnershipError:
                        ready = False
                if ready:
                    self._update_alias(profile_id)
                    return state

                recovery: dict[str, Any] = {
                    "attempts": 0,
                    "action": "startup_reconcile",
                    "error": "backend readiness probe failed",
                }
                if self.service_state_probe is not None:
                    try:
                        recovery["service"] = self.service_state_probe(
                            self.unit_name(profile_id)
                        ).as_dict()
                    except Exception as service_error:
                        recovery["service_error"] = str(service_error)
                stale_state = {
                    "generation": state["generation"] + 1,
                    "profile": None,
                    "phase": "stale",
                    "stale_profile": profile_id,
                    "error": "backend readiness probe failed",
                    "recovery": recovery,
                }
                self._write_state(stale_state)
                self._update_alias(None)
                return stale_state

            if state["phase"] != "switching":
                return state

            generation = state["generation"]
            prior_id = state.get("profile")
            target_id = state.get("target_profile")
            if not isinstance(target_id, str):
                raise ConfigurationError("switching state is missing target_profile")
            self._profile(target_id)
            if prior_id is not None:
                self._profile(prior_id)

            try:
                self._stop(target_id)
            except Exception as cleanup_error:
                failed_state = {
                    "generation": generation,
                    "profile": None,
                    "phase": "failed",
                    "prior_profile": prior_id,
                    "target_profile": target_id,
                    "error": "interrupted switch target could not be quiesced",
                    "cleanup_error": str(cleanup_error),
                    "restore": "not_attempted",
                }
                self._write_state(failed_state)
                self._update_alias(None)
                raise SwitchError(target_id, cleanup_error) from cleanup_error

            failed_state = {
                "generation": generation,
                "profile": None,
                "phase": "failed",
                "prior_profile": prior_id,
                "target_profile": target_id,
                "error": "interrupted switch target was stopped; prior profile not restored",
                "restore": "not_attempted",
            }
            self._write_state(failed_state)
            self._update_alias(None)
            self._emit(
                "profile_transition",
                decision="interrupted_failed",
                prior_profile=prior_id,
                target_profile=target_id,
                generation=generation,
                restore="not_attempted",
            )
            return failed_state

    def _handle_failed_switch(
        self,
        *,
        profile_id: str,
        prior_id: str | None,
        generation: int,
        target_error: BaseException,
        cleanup_target: bool,
    ) -> None:
        cleanup_error: BaseException | None = None
        if cleanup_target:
            try:
                self._stop(profile_id)
                self._emit(
                    "profile_transition",
                    decision="target_stopped",
                    prior_profile=prior_id,
                    target_profile=profile_id,
                    generation=generation,
                )
            except Exception as error:
                cleanup_error = error
                self._emit(
                    "profile_transition",
                    decision="target_stop_failed",
                    prior_profile=prior_id,
                    target_profile=profile_id,
                    generation=generation,
                    error_type=type(error).__name__,
                    error=str(error),
                )

        failed_state = {
            "generation": generation,
            "profile": None,
            "phase": "failed",
            "prior_profile": prior_id,
            "target_profile": profile_id,
            "error": str(target_error),
            "restore": "not_attempted",
        }
        if cleanup_error is not None:
            failed_state["cleanup_error"] = str(cleanup_error)
        self._write_state(failed_state)
        self._update_alias(None)
        self._emit(
            "profile_transition",
            decision="failed",
            prior_profile=prior_id,
            target_profile=profile_id,
            generation=generation,
            error_type=type(target_error).__name__,
            error=str(target_error),
            restore="not_attempted",
        )
        failure = cleanup_error or target_error
        raise SwitchError(profile_id, failure) from failure

    def _profile(self, profile_id: str) -> Profile:
        try:
            return self.profiles[profile_id]
        except KeyError as error:
            raise ConfigurationError(f"unknown profile: {profile_id!r}") from error

    def replace_profiles(self, profiles: Mapping[str, Profile]) -> None:
        """Replace the validated profile registry without restarting systemd."""

        if not profiles:
            raise ConfigurationError("at least one profile is required")
        with self._lock:
            old_profiles = self.profiles
            self.profiles = dict(profiles)
            try:
                self._validate_profiles()
            except BaseException:
                self.profiles = old_profiles
                raise

    def _emit(self, event: str, **fields: object) -> None:
        record: dict[str, object] = {"event": event, **fields}
        try:
            self.event_logger(record)
        except Exception:
            # Logging must never bypass transaction cleanup or fail-closed state.
            pass

    def _require_not_cancelled(self, stage: str) -> None:
        if self.cancellation_probe is None:
            return
        try:
            cancelled = bool(self.cancellation_probe())
        except Exception as error:
            raise LifecycleCancelledError(
                f"lifecycle cancellation probe failed at {stage}: {error}"
            ) from error
        if cancelled:
            raise LifecycleCancelledError(
                f"backend activation cancelled at {stage}"
            )

    def _admit_pre_start(self, profile: Profile) -> ResourceSnapshot | None:
        self._require_not_cancelled(f"pre_start_admission:{profile.id}")
        if self.resource_probe is None:
            self._emit(
                "resource_admission",
                profile=profile.id,
                stage="pre_start",
                decision="not_configured",
            )
            return None
        snapshot = self._sample_resources(profile, stage="pre_start")
        admission = profile.safety.admission
        violations: list[str] = []
        if snapshot.ram_available_mib < admission.min_ram_available_mib:
            violations.append(
                "ram_available_mib "
                f"{snapshot.ram_available_mib} < {admission.min_ram_available_mib}"
            )
        if snapshot.swap_used_mib > admission.max_swap_used_mib:
            violations.append(
                "swap_used_mib "
                f"{snapshot.swap_used_mib} > {admission.max_swap_used_mib}"
            )
        if snapshot.vram_total_mib < admission.min_total_vram_mib:
            violations.append(
                "vram_total_mib "
                f"{snapshot.vram_total_mib} < {admission.min_total_vram_mib}"
            )
        if snapshot.vram_free_mib < admission.min_free_vram_before_start_mib:
            violations.append(
                "vram_free_mib "
                f"{snapshot.vram_free_mib} < {admission.min_free_vram_before_start_mib}"
            )

        self._emit(
            "resource_admission",
            profile=profile.id,
            stage="pre_start",
            decision="deny" if violations else "allow",
            snapshot=snapshot.as_dict(),
            violations=violations,
        )
        if violations:
            raise AdmissionError(
                f"profile {profile.id!r} failed pre-start resource admission: "
                + "; ".join(violations)
            )
        return snapshot

    def _admit_post_ready(
        self, profile: Profile, pre_start: ResourceSnapshot | None
    ) -> ResourceSnapshot | None:
        self._require_not_cancelled(f"post_ready_admission:{profile.id}")
        if self.resource_probe is None or pre_start is None:
            self._emit(
                "resource_admission",
                profile=profile.id,
                stage="post_ready",
                decision="not_configured",
            )
            return None
        snapshot = self._sample_resources(profile, stage="post_ready")
        admission = profile.safety.admission
        swap_growth_mib = max(0, snapshot.swap_used_mib - pre_start.swap_used_mib)
        violations: list[str] = []
        if snapshot.ram_available_mib < admission.min_ram_available_mib:
            violations.append(
                "ram_available_mib "
                f"{snapshot.ram_available_mib} < {admission.min_ram_available_mib}"
            )
        if snapshot.swap_used_mib > admission.max_swap_used_mib:
            violations.append(
                "swap_used_mib "
                f"{snapshot.swap_used_mib} > {admission.max_swap_used_mib}"
            )
        if swap_growth_mib > admission.max_swap_growth_mib:
            violations.append(
                "swap_growth_mib "
                f"{swap_growth_mib} > {admission.max_swap_growth_mib}"
            )
        if snapshot.vram_total_mib < admission.min_total_vram_mib:
            violations.append(
                "vram_total_mib "
                f"{snapshot.vram_total_mib} < {admission.min_total_vram_mib}"
            )
        if snapshot.vram_free_mib < admission.post_ready_min_free_vram_mib:
            violations.append(
                "vram_free_mib "
                f"{snapshot.vram_free_mib} < {admission.post_ready_min_free_vram_mib}"
            )

        self._emit(
            "resource_admission",
            profile=profile.id,
            stage="post_ready",
            decision="deny" if violations else "allow",
            snapshot=snapshot.as_dict(),
            swap_growth_mib=swap_growth_mib,
            violations=violations,
        )
        if violations:
            raise AdmissionError(
                f"profile {profile.id!r} failed post-ready resource admission: "
                + "; ".join(violations)
            )
        return snapshot

    def _sample_resources(self, profile: Profile, *, stage: str) -> ResourceSnapshot:
        try:
            resource_probe = self.resource_probe
            if resource_probe is None:
                raise RuntimeError("resource admission is not configured")
            snapshot = resource_probe()
            values = snapshot.as_dict()
            invalid = [
                name
                for name, value in values.items()
                if isinstance(value, bool) or not isinstance(value, int) or value < 0
            ]
            if snapshot.vram_total_mib <= 0:
                invalid.append("vram_total_mib")
            if snapshot.vram_free_mib > snapshot.vram_total_mib:
                invalid.append("vram_free_mib")
            if invalid:
                raise ValueError(
                    "invalid resource snapshot fields: " + ", ".join(sorted(set(invalid)))
                )
        except Exception as error:
            self._emit(
                "resource_admission",
                profile=profile.id,
                stage=stage,
                decision="probe_error",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise AdmissionError(
                f"profile {profile.id!r} resource probe failed at {stage}: {error}"
            ) from error
        return snapshot

    def _start(self, profile_id: str) -> None:
        profile = self._profile(profile_id)
        self._require_not_cancelled(f"start:{profile_id}")
        self._require_all_quiescent(stage=f"pre_start:{profile_id}")
        self._require_endpoint_absent(stage=f"pre_start:{profile_id}")
        self._require_not_cancelled(f"pre_command_start:{profile_id}")
        if self.artifact_verifier is not None:
            self.artifact_verifier(profile)
        self._run("systemctl", "--user", "start", self.unit_name(profile_id))

    def _stop(self, profile_id: str) -> None:
        self._stop_unit(self.unit_name(profile_id))

    def _stop_unit(self, unit_name: str) -> None:
        try:
            self._run("systemctl", "--user", "stop", unit_name)
        except BaseException as stop_error:
            # A llama-server can block SIGINT/SIGTERM while tearing down a
            # CUDA/KV-cache graph.  The historical SendSIGKILL=no unit policy
            # otherwise leaves a process in the cgroup forever, and the broker
            # can only report a quarantined transition.  Once this method is
            # reached, the coordinator has already drained all request leases;
            # use systemd's explicit last-resort cgroup kill and clear its
            # failed result before proving quiescence again.
            self._emit(
                "service_stop_escalation",
                unit=unit_name,
                decision="force_kill",
                error_type=type(stop_error).__name__,
                error=str(stop_error),
            )
            try:
                self._run(
                    "systemctl",
                    "--user",
                    "kill",
                    "--kill-who=all",
                    "--signal=SIGKILL",
                    unit_name,
                )
            except BaseException as kill_error:
                # A unit may have disappeared between stop and kill.  The
                # subsequent state probe decides whether that is harmless.
                self._emit(
                    "service_stop_escalation",
                    unit=unit_name,
                    decision="force_kill_failed",
                    error_type=type(kill_error).__name__,
                    error=str(kill_error),
                )
            # ``systemctl kill`` returns before PID 1 has necessarily published
            # the final failed/dead state.  Resetting too early is a no-op, and
            # the broker would incorrectly quarantine an already-dead worker.
            # Retry reset/probe for a short bounded window; this is deliberately
            # independent of the long model stop timeout and never waits on a
            # process that systemd still reports as owning a PID forever.
            last_quiescence_error: BaseException | None = None
            for attempt in range(8):
                if attempt:
                    time.sleep(0.25)
                try:
                    self._run("systemctl", "--user", "reset-failed", unit_name)
                except BaseException as reset_error:
                    self._emit(
                        "service_stop_escalation",
                        unit=unit_name,
                        decision="reset_failed_failed",
                        attempt=attempt + 1,
                        error_type=type(reset_error).__name__,
                        error=str(reset_error),
                    )
                try:
                    self._require_unit_quiescent(
                        unit_name, stage=f"post_force_kill_{attempt + 1}"
                    )
                except BaseException as quiescence_error:
                    last_quiescence_error = quiescence_error
                    continue
                return
            if last_quiescence_error is not None:
                self._emit(
                    "service_stop_escalation",
                    unit=unit_name,
                    decision="force_kill_quiescence_timeout",
                    attempts=8,
                    error_type=type(last_quiescence_error).__name__,
                    error=str(last_quiescence_error),
                )
            raise stop_error
        self._require_unit_quiescent(unit_name, stage="post_stop")

    def _inventory_unit_names(self) -> set[str]:
        if self.service_inventory_probe is None:
            return set()
        try:
            reported_units = self.service_inventory_probe(self.service_prefix)
            if isinstance(reported_units, (str, bytes)):
                raise TypeError("service inventory probe returned a string")
            units: set[str] = set()
            for unit_name in reported_units:
                if not isinstance(unit_name, str):
                    raise TypeError(
                        "service inventory probe returned a non-string unit name"
                    )
                if _profile_id_from_unit_name(unit_name, self.service_prefix) is None:
                    raise ValueError(
                        f"service inventory probe returned an invalid unit: {unit_name!r}"
                    )
                units.add(unit_name)
        except Exception as error:
            self._emit(
                "service_inventory",
                service_prefix=self.service_prefix,
                decision="probe_error",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise QuiescenceError(
                f"could not inventory loaded broker units: {error}"
            ) from error

        self._emit(
            "service_inventory",
            service_prefix=self.service_prefix,
            decision="allow",
            units=sorted(units),
        )
        return units

    def _all_backend_unit_names(self) -> set[str]:
        units = {self.unit_name(profile_id) for profile_id in self.profiles}
        units.add(_LEGACY_SERVICE_UNIT)
        units.update(self._inventory_unit_names())
        return units

    def _unit_label(self, unit_name: str) -> str:
        profile_id = _profile_id_from_unit_name(unit_name, self.service_prefix)
        return profile_id if profile_id is not None else unit_name

    def _require_all_quiescent(self, *, stage: str) -> None:
        for unit_name in sorted(self._all_backend_unit_names()):
            self._require_unit_quiescent(unit_name, stage=stage)

    def _require_unit_quiescent(self, unit_name: str, *, stage: str) -> None:
        if self.service_state_probe is None:
            return
        profile_id = _profile_id_from_unit_name(unit_name, self.service_prefix)
        try:
            snapshot = self.service_state_probe(unit_name)
            if not isinstance(snapshot, ServiceStateSnapshot):
                raise TypeError("service state probe returned an invalid snapshot type")
            values = snapshot.as_dict()
            if (
                not snapshot.active_state
                or not snapshot.sub_state
                or isinstance(snapshot.main_pid, bool)
                or isinstance(snapshot.control_pid, bool)
                or snapshot.main_pid < 0
                or snapshot.control_pid < 0
            ):
                raise ValueError("service state probe returned invalid fields")
        except Exception as error:
            self._emit(
                "service_quiescence",
                profile=profile_id,
                unit=unit_name,
                stage=stage,
                decision="probe_error",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise QuiescenceError(
                f"could not verify backend unit {unit_name!r} at {stage}: {error}"
            ) from error

        self._emit(
            "service_quiescence",
            profile=profile_id,
            unit=unit_name,
            stage=stage,
            decision="allow" if snapshot.is_quiescent else "deny",
            snapshot=values,
        )
        if not snapshot.is_quiescent:
            raise QuiescenceError(
                f"backend unit {unit_name!r} is not quiescent at {stage}: "
                f"ActiveState={snapshot.active_state}, SubState={snapshot.sub_state}, "
                f"MainPID={snapshot.main_pid}, ControlPID={snapshot.control_pid}"
            )

    def _require_endpoint_absent(self, *, stage: str) -> None:
        if self.endpoint_probe is None:
            return
        try:
            endpoint_present = bool(self.endpoint_probe())
        except Exception as error:
            raise ServiceOwnershipError(
                f"could not prove the private backend endpoint absent at {stage}: {error}"
            ) from error
        if endpoint_present:
            raise ServiceOwnershipError(
                f"an unmanaged process already owns the private backend endpoint at {stage}"
            )

    def _require_running(self, profile_id: str, *, stage: str) -> None:
        if self.service_state_probe is None:
            return
        unit_name = self.unit_name(profile_id)
        try:
            snapshot = self.service_state_probe(unit_name)
            if not isinstance(snapshot, ServiceStateSnapshot):
                raise TypeError("service state probe returned an invalid snapshot type")
            if (
                not snapshot.active_state
                or not snapshot.sub_state
                or isinstance(snapshot.main_pid, bool)
                or isinstance(snapshot.control_pid, bool)
                or snapshot.main_pid < 0
                or snapshot.control_pid < 0
            ):
                raise ValueError("service state probe returned invalid fields")
        except Exception as error:
            raise ServiceOwnershipError(
                f"could not verify backend {profile_id!r} ownership at {stage}: {error}"
            ) from error

        if not (
            snapshot.active_state == "active"
            and snapshot.sub_state == "running"
            and snapshot.main_pid > 0
            and snapshot.control_pid == 0
        ):
            raise ServiceOwnershipError(
                f"backend {profile_id!r} alias is not owned by {unit_name} at {stage}: "
                f"ActiveState={snapshot.active_state}, SubState={snapshot.sub_state}, "
                f"MainPID={snapshot.main_pid}, ControlPID={snapshot.control_pid}"
            )

    def _probe(self, profile: Profile) -> None:
        self._require_not_cancelled(f"readiness:{profile.id}")
        if not self.readiness_probe(profile.alias, float(profile.startup_timeout)):
            raise ReadinessError(
                f"profile {profile.id!r} did not serve exact alias {profile.alias!r} "
                f"within {profile.startup_timeout:g}s"
            )

    def _probe_owned(self, profile_id: str, profile: Profile) -> None:
        self._probe(profile)
        self._require_running(profile_id, stage="post_readiness")

    def _run(self, *command: str) -> Any:
        argv = tuple(command)
        try:
            result = self.command_runner(argv)
        except BackendControllerError:
            raise
        except Exception as error:
            raise CommandError(f"command failed: {argv!r}: {error}") from error
        returncode = getattr(result, "returncode", 0)
        if returncode:
            raise CommandError(f"command returned {returncode}: {argv!r}")
        if isinstance(result, int) and result != 0:
            raise CommandError(f"command returned {result}: {argv!r}")
        return result

    def _read_state(self) -> dict[str, Any]:
        try:
            raw = self.state_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"generation": 0, "profile": None, "phase": "inactive"}
        try:
            state = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as error:
            raise ConfigurationError(f"invalid controller state: {error}") from error
        if not isinstance(state, dict):
            raise ConfigurationError("controller state must be a JSON object")
        generation = state.get("generation")
        profile_id = state.get("profile")
        phase = state.get("phase")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ConfigurationError("controller state generation must be a non-negative integer")
        if profile_id is not None and not isinstance(profile_id, str):
            raise ConfigurationError("controller state profile must be a string or null")
        if not isinstance(phase, str) or not phase:
            raise ConfigurationError("controller state phase must be a non-empty string")
        return state

    def _write_state(self, state: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(state), indent=2, sort_keys=True) + "\n"
        _atomic_write(self.state_path, payload)


def subprocess_command_runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    """Opt-in real command runner suitable for production composition."""

    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


__all__ = [
    "BackendController",
    "BackendControllerError",
    "AdmissionError",
    "CommandError",
    "ConfigurationError",
    "LifecycleCancelledError",
    "Profile",
    "QuiescenceError",
    "ReadinessError",
    "ResourceSnapshot",
    "linux_resource_snapshot",
    "RollbackError",
    "ServiceInventoryProbe",
    "ServiceStateSnapshot",
    "ServiceOwnershipError",
    "SwitchError",
    "subprocess_command_runner",
    "systemd_service_inventory_probe",
    "systemd_service_state_probe",
]
