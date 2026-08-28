#!/usr/bin/env python3
"""Security material and TLS configuration for the broker deployment.

The broker does not validate inference API keys itself.  This module only
manages the native ``llama-server`` key file and the TLS contexts needed by the
broker listener and its HTTPS connection to that server.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import socket
import secrets
import ssl
import stat
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paths import API_KEY_PATH, SECURITY_PATH


DEFAULT_SECURITY_PATH = SECURITY_PATH
DEFAULT_API_KEY_PATH = API_KEY_PATH
DEFAULT_CERTIFICATE_DAYS = 825


class SecurityError(ValueError):
    """A deployment security setting is missing or unsafe."""


@dataclass(frozen=True)
class SecurityInitResult:
    """Material produced by the explicit local TLS bootstrap command."""

    config: "SecurityConfig"
    sans: tuple[str, ...]
    created: bool


def _absolute_path(value: object, field_name: str) -> Path | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise SecurityError(f"{field_name} must be an absolute path or null")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SecurityError(f"{field_name} must be an absolute path: {value!r}")
    return path


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _security_config_path(path: str | Path | None = None) -> Path:
    raw_path = path or _env_value(
        "LLAMA_BROKER_SECURITY_FILE", "LLM_BROKER_SECURITY_FILE"
    )
    config_path = Path(raw_path).expanduser() if raw_path else DEFAULT_SECURITY_PATH
    if not config_path.is_absolute():
        raise SecurityError(f"security config must be absolute: {config_path}")
    return config_path


def _is_loopback_host(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_pair(
    cert_file: Path | None,
    key_file: Path | None,
    label: str,
) -> None:
    if (cert_file is None) != (key_file is None):
        raise SecurityError(f"{label} certificate and key must be configured together")


def _validate_file(path: Path, label: str, *, private: bool = False) -> None:
    if not path.is_file():
        raise SecurityError(f"{label} does not exist: {path}")
    if private:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise SecurityError(
                f"{label} is accessible by group/other users; chmod 600 {path}"
            )


@dataclass(frozen=True)
class SecurityConfig:
    """Deployment-owned paths for native backend auth and TLS."""

    # This value may only come from the process environment or an explicit
    # in-memory constructor.  It is deliberately excluded from as_mapping().
    api_key: str | None = None
    api_key_file: Path | None = None
    broker_tls_cert_file: Path | None = None
    broker_tls_key_file: Path | None = None
    broker_tls_ca_file: Path | None = None
    backend_tls_cert_file: Path | None = None
    backend_tls_key_file: Path | None = None
    backend_tls_ca_file: Path | None = None
    source_path: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        source_path: Path | None = None,
    ) -> "SecurityConfig":
        return cls(
            api_key_file=_absolute_path(mapping.get("api_key_file"), "api_key_file"),
            broker_tls_cert_file=_absolute_path(
                mapping.get("broker_tls_cert_file"), "broker_tls_cert_file"
            ),
            broker_tls_key_file=_absolute_path(
                mapping.get("broker_tls_key_file"), "broker_tls_key_file"
            ),
            broker_tls_ca_file=_absolute_path(
                mapping.get("broker_tls_ca_file"), "broker_tls_ca_file"
            ),
            backend_tls_cert_file=_absolute_path(
                mapping.get("backend_tls_cert_file"), "backend_tls_cert_file"
            ),
            backend_tls_key_file=_absolute_path(
                mapping.get("backend_tls_key_file"), "backend_tls_key_file"
            ),
            backend_tls_ca_file=_absolute_path(
                mapping.get("backend_tls_ca_file"), "backend_tls_ca_file"
            ),
            source_path=source_path,
        )

    def with_api_key_file(self, path: Path) -> "SecurityConfig":
        return replace(self, api_key_file=path)

    def authentication_keys(self) -> tuple[str, ...]:
        """Return deployment API keys without exposing them in diagnostics."""

        if self.api_key is not None and self.api_key.strip():
            return (self.api_key,)
        if self.api_key_file is not None:
            return tuple(dict.fromkeys(read_api_keys(self.api_key_file)))
        return ()

    def as_mapping(self) -> dict[str, str]:
        fields = (
            "api_key_file",
            "broker_tls_cert_file",
            "broker_tls_key_file",
            "broker_tls_ca_file",
            "backend_tls_cert_file",
            "backend_tls_key_file",
            "backend_tls_ca_file",
        )
        return {
            field: str(value)
            for field in fields
            if (value := getattr(self, field)) is not None
        }

    def validate(self, *, require_files: bool = False) -> None:
        _validate_pair(
            self.broker_tls_cert_file,
            self.broker_tls_key_file,
            "broker TLS",
        )
        _validate_pair(
            self.backend_tls_cert_file,
            self.backend_tls_key_file,
            "backend TLS",
        )
        if self.api_key is not None and not self.api_key.strip():
            raise SecurityError("API key must not be empty")
        if not require_files:
            return

        if self.api_key_file is not None:
            _validate_file(self.api_key_file, "API-key file", private=True)
            if not read_api_keys(self.api_key_file):
                raise SecurityError(f"API-key file is empty: {self.api_key_file}")
        for path, label, private in (
            (self.broker_tls_cert_file, "broker TLS certificate", False),
            (self.broker_tls_key_file, "broker TLS private key", True),
            (self.broker_tls_ca_file, "broker TLS CA", False),
            (self.backend_tls_cert_file, "backend TLS certificate", False),
            (self.backend_tls_key_file, "backend TLS private key", True),
            (self.backend_tls_ca_file, "backend TLS CA", False),
        ):
            if path is not None:
                _validate_file(path, label, private=private)

    def validate_network_binding(self, broker_host: str, backend_host: str) -> None:
        """Require native auth and TLS before exposing either service on a LAN."""

        if not _is_loopback_host(broker_host):
            if self.broker_tls_cert_file is None or self.broker_tls_key_file is None:
                raise SecurityError(
                    "non-loopback broker binding requires broker TLS certificate and key"
                )
            if not self.authentication_keys():
                raise SecurityError("non-loopback broker binding requires API key")
        if not _is_loopback_host(backend_host):
            if self.api_key_file is None:
                raise SecurityError("non-loopback backend binding requires API-key file")
            if self.backend_tls_cert_file is None or self.backend_tls_key_file is None:
                raise SecurityError(
                    "non-loopback backend binding requires backend TLS certificate and key"
                )

    def backend_server_args(
        self,
        *,
        listen_host: str,
        listen_port: int,
        runtime: str = "llamacpp",
    ) -> tuple[str, ...]:
        """Return deployment-owned network/auth arguments for one runtime."""

        args = ["--host", listen_host, "--port", str(listen_port)]
        if runtime == "llamacpp":
            if self.api_key_file is not None:
                args.extend(["--api-key-file", str(self.api_key_file)])
            if self.backend_tls_cert_file is not None:
                args.extend(
                    [
                        "--ssl-cert-file",
                        str(self.backend_tls_cert_file),
                        "--ssl-key-file",
                        str(self.backend_tls_key_file),
                    ]
                )
            return tuple(args)
        if runtime == "vllm":
            if self.api_key_file is not None:
                keys = read_api_keys(self.api_key_file)
                if not keys:
                    raise SecurityError(f"API-key file is empty: {self.api_key_file}")
                # vLLM accepts the key value, not llama.cpp's key-file option.
                args.extend(["--api-key", keys[-1]])
            if self.backend_tls_cert_file is not None:
                args.extend(
                    [
                        "--ssl-certfile",
                        str(self.backend_tls_cert_file),
                        "--ssl-keyfile",
                        str(self.backend_tls_key_file),
                    ]
                )
            if self.backend_tls_ca_file is not None:
                args.extend(["--ssl-ca-certs", str(self.backend_tls_ca_file)])
            return tuple(args)
        if runtime == "freetoken":
            if self.api_key_file is not None:
                args.extend(["--freetoken-api-key-file", str(self.api_key_file)])
            if self.backend_tls_cert_file is not None:
                args.extend(
                    [
                        "--freetoken-tls-cert",
                        str(self.backend_tls_cert_file),
                        "--freetoken-tls-key",
                        str(self.backend_tls_key_file),
                    ]
                )
            return tuple(args)
        raise SecurityError(f"unsupported backend runtime for security args: {runtime}")

    def broker_tls_context(self) -> ssl.SSLContext | None:
        if self.broker_tls_cert_file is None and self.broker_tls_key_file is None:
            return None
        self.validate(require_files=True)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(
            certfile=str(self.broker_tls_cert_file),
            keyfile=str(self.broker_tls_key_file),
        )
        return context

    def backend_client_context(self) -> ssl.SSLContext | None:
        if self.backend_tls_cert_file is None:
            return None
        self.validate(require_files=True)
        context = ssl.create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH,
            cafile=str(self.backend_tls_ca_file) if self.backend_tls_ca_file else None,
        )
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        return context

    def broker_client_context(self) -> ssl.SSLContext | None:
        if self.broker_tls_cert_file is None:
            return None
        self.validate(require_files=True)
        context = ssl.create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH,
            cafile=str(self.broker_tls_ca_file) if self.broker_tls_ca_file else None,
        )
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        return context


def load_security_config(path: str | Path | None = None) -> SecurityConfig:
    """Load JSON security settings, with environment variables as overrides."""

    config_path = _security_config_path(path)

    data: dict[str, Any] = {}
    if config_path.is_file():
        try:
            decoded = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SecurityError(f"could not read security config {config_path}: {error}") from error
        if not isinstance(decoded, dict):
            raise SecurityError(f"security config must contain a JSON object: {config_path}")
        data.update(decoded)

    env_names = {
        "api_key_file": ("LLAMA_BROKER_API_KEY_FILE", "LLM_BROKER_API_KEY_FILE"),
        "broker_tls_cert_file": (
            "LLAMA_BROKER_TLS_CERT_FILE",
            "LLM_BROKER_TLS_CERT_FILE",
        ),
        "broker_tls_key_file": (
            "LLAMA_BROKER_TLS_KEY_FILE",
            "LLM_BROKER_TLS_KEY_FILE",
        ),
        "broker_tls_ca_file": (
            "LLAMA_BROKER_TLS_CA_FILE",
            "LLM_BROKER_TLS_CA_FILE",
        ),
        "backend_tls_cert_file": (
            "LLAMA_BROKER_BACKEND_TLS_CERT_FILE",
            "LLM_BROKER_BACKEND_TLS_CERT_FILE",
        ),
        "backend_tls_key_file": (
            "LLAMA_BROKER_BACKEND_TLS_KEY_FILE",
            "LLM_BROKER_BACKEND_TLS_KEY_FILE",
        ),
        "backend_tls_ca_file": (
            "LLAMA_BROKER_BACKEND_TLS_CA_FILE",
            "LLM_BROKER_BACKEND_TLS_CA_FILE",
        ),
    }
    for field_name, names in env_names.items():
        if (value := _env_value(*names)) is not None:
            data[field_name] = value
    config = SecurityConfig.from_mapping(data, source_path=config_path)
    if "LLAMA_BROKER_API_KEY" in os.environ:
        raw_key = os.environ["LLAMA_BROKER_API_KEY"]
        if not raw_key.strip():
            raise SecurityError("LLAMA_BROKER_API_KEY (API key) must not be empty")
        config = replace(config, api_key=raw_key)
    return config


def write_security_config(config: SecurityConfig) -> Path:
    """Atomically write the path-only security configuration."""

    path = config.source_path or DEFAULT_SECURITY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(config.as_mapping(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return path


def read_api_keys(path: Path) -> list[str]:
    """Read native llama-server API keys without logging their values."""

    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _write_api_keys(path: Path, keys: list[str]) -> None:
    if not keys or any(not isinstance(key, str) or not key.strip() for key in keys):
        raise SecurityError("API-key file must contain at least one non-empty key")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("\n".join(keys) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def write_api_keys(path: Path, keys: list[str]) -> None:
    """Publish an explicit native key set while keeping it private on disk."""

    _write_api_keys(path, keys)


def generate_api_key() -> str:
    return "sk-llama-broker-" + secrets.token_urlsafe(32)


def ensure_api_key_file(path: Path) -> tuple[Path, str]:
    """Create key material if necessary and return the newest key."""

    keys = read_api_keys(path)
    if not keys:
        keys = [generate_api_key()]
        _write_api_keys(path, keys)
    else:
        os.chmod(path, 0o600)
    return path, keys[-1]


def rotate_api_keys(path: Path, *, replace: bool = False) -> tuple[str, list[str]]:
    """Rotate native keys; retain the prior key unless ``replace`` is set."""

    current = read_api_keys(path)
    new_key = generate_api_key()
    updated = [new_key] if replace else [*current, new_key]
    _write_api_keys(path, updated)
    return new_key, current


def revoke_api_key(path: Path, key: str) -> list[str]:
    current = read_api_keys(path)
    if key not in current:
        raise SecurityError("API key is not present")
    updated = [candidate for candidate in current if candidate != key]
    if not updated:
        raise SecurityError("cannot revoke the last API key")
    _write_api_keys(path, updated)
    return updated


def ensure_security_material() -> tuple[SecurityConfig, str]:
    """Ensure setup has a native key and persist its path-only configuration."""

    config = load_security_config()
    key_path = config.api_key_file or DEFAULT_API_KEY_PATH
    _, key = ensure_api_key_file(key_path)
    if config.api_key_file is None:
        config = config.with_api_key_file(key_path)
        write_security_config(config)
    return config, key


_HOSTNAME_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")


def _normalise_init_ip(value: str) -> str:
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError as error:
        raise SecurityError(f"invalid certificate IP address: {value!r}") from error


def _normalise_init_hostname(value: str) -> str:
    raw = value.strip().rstrip(".")
    if not raw:
        raise SecurityError("certificate hostname cannot be empty")
    try:
        hostname = raw.encode("idna").decode("ascii")
    except UnicodeError as error:
        raise SecurityError(f"invalid certificate hostname: {value!r}") from error
    if len(hostname) > 253 or not _HOSTNAME_PATTERN.fullmatch(hostname):
        raise SecurityError(f"invalid certificate hostname: {value!r}")
    return hostname


def _certificate_sans(
    ip_addresses: Sequence[str], hostnames: Sequence[str],
) -> tuple[str, ...]:
    if not ip_addresses:
        raise SecurityError("security init requires at least one --ip/--lan-ip value")

    ips: list[str] = ["127.0.0.1"]
    for value in ip_addresses:
        normalised = _normalise_init_ip(value)
        if normalised not in ips:
            ips.append(normalised)

    names: list[str] = ["localhost"]
    supplied_names = list(hostnames)
    if not supplied_names:
        try:
            supplied_names = [socket.gethostname()]
        except OSError:
            supplied_names = []
    for value in supplied_names:
        normalised = _normalise_init_hostname(value)
        if normalised not in names:
            names.append(normalised)

    return tuple([*(f"DNS:{name}" for name in names), *(f"IP:{ip}" for ip in ips)])


def _run_openssl(binary: str, arguments: Sequence[str], cwd: Path) -> None:
    try:
        completed = subprocess.run(
            [binary, *arguments],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as error:
        raise SecurityError("openssl is required for security init") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise SecurityError(f"openssl failed while creating TLS material{suffix}") from error
    if completed.returncode != 0:
        raise SecurityError("openssl failed while creating TLS material")


def initialize_security_material(
    *,
    ip_addresses: Sequence[str],
    hostnames: Sequence[str] = (),
    security_path: str | Path | None = None,
    api_key_path: str | Path | None = None,
    days: int = DEFAULT_CERTIFICATE_DAYS,
    force: bool = False,
) -> SecurityInitResult:
    """Create a local CA and native server certificates without overwriting by default."""

    if days < 1:
        raise SecurityError("certificate lifetime must be at least one day")

    config_path = _security_config_path(security_path)
    existing = load_security_config(config_path)
    key_path = Path(api_key_path).expanduser() if api_key_path else existing.api_key_file
    key_path = key_path or DEFAULT_API_KEY_PATH
    if not key_path.is_absolute():
        raise SecurityError(f"API-key path must be absolute: {key_path}")

    sans = _certificate_sans(ip_addresses, hostnames)
    security_dir = config_path.parent
    target_paths = {
        "ca_key": security_dir / "lan-ca.key",
        "ca_cert": security_dir / "lan-ca.crt",
        "broker_key": security_dir / "broker.key",
        "broker_cert": security_dir / "broker.crt",
        "backend_key": security_dir / "llama-server.key",
        "backend_cert": security_dir / "llama-server.crt",
    }
    expected_tls = {
        "broker_tls_cert_file": target_paths["broker_cert"],
        "broker_tls_key_file": target_paths["broker_key"],
        "broker_tls_ca_file": target_paths["ca_cert"],
        "backend_tls_cert_file": target_paths["backend_cert"],
        "backend_tls_key_file": target_paths["backend_key"],
        "backend_tls_ca_file": target_paths["ca_cert"],
    }
    existing_tls = {
        field: getattr(existing, field)
        for field in expected_tls
        if getattr(existing, field) is not None
    }
    if existing_tls and existing_tls != expected_tls and not force:
        raise SecurityError(
            f"{config_path} already defines different TLS material; use --force to replace it"
        )

    present_targets = [path for path in target_paths.values() if path.exists()]
    if present_targets and not force:
        if len(present_targets) == len(target_paths) and existing_tls == expected_tls:
            ensure_api_key_file(key_path)
            config = SecurityConfig(
                api_key_file=key_path,
                **expected_tls,
                source_path=config_path,
            )
            write_security_config(config)
            config.validate(require_files=True)
            return SecurityInitResult(config=config, sans=sans, created=False)
        raise SecurityError(
            f"TLS material already exists under {security_dir}; use --force to replace it"
        )

    openssl = shutil.which("openssl")
    if not openssl:
        raise SecurityError("openssl is required for security init")

    security_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(security_dir, 0o700)
    with tempfile.TemporaryDirectory(prefix=".security-init-", dir=security_dir) as temporary_name:
        temporary_dir = Path(temporary_name)
        extension_file = temporary_dir / "server.ext"
        extension_file.write_text(
            "[v3_server]\n"
            f"subjectAltName = {','.join(sans)}\n"
            "extendedKeyUsage = serverAuth\n",
            encoding="utf-8",
        )
        os.chmod(extension_file, 0o600)
        serial_file = temporary_dir / "lan-ca.srl"
        serial_file.write_text("01\n", encoding="ascii")
        os.chmod(serial_file, 0o600)

        ca_key = temporary_dir / target_paths["ca_key"].name
        ca_cert = temporary_dir / target_paths["ca_cert"].name
        _run_openssl(
            openssl,
            [
                "req",
                "-x509",
                "-newkey",
                "rsa:4096",
                "-nodes",
                "-sha256",
                "-days",
                str(days),
                "-keyout",
                str(ca_key),
                "-out",
                str(ca_cert),
                "-subj",
                "/CN=Llama-Broker local CA",
                "-addext",
                "basicConstraints=critical,CA:TRUE,pathlen:1",
                "-addext",
                "keyUsage=critical,keyCertSign,cRLSign",
            ],
            temporary_dir,
        )

        primary_name = next(
            (
                value.removeprefix("DNS:")
                for value in sans
                if value.startswith("DNS:") and value != "DNS:localhost"
            ),
            next(value.removeprefix("IP:") for value in sans if value.startswith("IP:")),
        )
        server_targets = {
            "broker": (target_paths["broker_key"], target_paths["broker_cert"]),
            "llama-server": (target_paths["backend_key"], target_paths["backend_cert"]),
        }
        for name, (key_target, cert_target) in server_targets.items():
            server_key = temporary_dir / key_target.name
            server_cert = temporary_dir / cert_target.name
            csr = temporary_dir / f"{name}.csr"
            _run_openssl(
                openssl,
                [
                    "req",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    str(server_key),
                    "-out",
                    str(csr),
                    "-subj",
                    f"/CN={primary_name}",
                ],
                temporary_dir,
            )
            _run_openssl(
                openssl,
                [
                    "x509",
                    "-req",
                    "-in",
                    str(csr),
                    "-CA",
                    str(ca_cert),
                    "-CAkey",
                    str(ca_key),
                    "-CAserial",
                    str(serial_file),
                    "-out",
                    str(server_cert),
                    "-days",
                    str(days),
                    "-sha256",
                    "-extfile",
                    str(extension_file),
                    "-extensions",
                    "v3_server",
                ],
                temporary_dir,
            )

        for key, target in target_paths.items():
            generated = temporary_dir / target.name
            if not generated.is_file():
                raise SecurityError(f"openssl did not create expected file: {generated.name}")
            os.replace(generated, target)
            os.chmod(target, 0o600 if key.endswith("_key") else 0o644)

    ensure_api_key_file(key_path)
    config = SecurityConfig(
        api_key_file=key_path,
        **expected_tls,
        source_path=config_path,
    )
    write_security_config(config)
    config.validate(require_files=True)
    return SecurityInitResult(config=config, sans=sans, created=True)
