"""Deterministic XDG locations used by Llama Broker.

Importing this module only resolves paths; it never creates directories or
moves existing artifacts.  Individual commands decide when a directory may be
created.
"""

from __future__ import annotations

import os
from pathlib import Path


def _home_path(env_name: str, fallback: Path) -> Path:
    value = os.environ.get(env_name)
    path = Path(value).expanduser() if value else fallback
    if not path.is_absolute():
        raise ValueError(f"{env_name} must be an absolute path: {path}")
    return path


def _override(env_name: str, fallback: Path) -> Path:
    value = os.environ.get(env_name)
    path = Path(value).expanduser() if value else fallback
    if not path.is_absolute():
        raise ValueError(f"{env_name} must be an absolute path: {path}")
    return path


XDG_CONFIG_HOME = _home_path("XDG_CONFIG_HOME", Path.home() / ".config")
XDG_DATA_HOME = _home_path("XDG_DATA_HOME", Path.home() / ".local" / "share")
XDG_CACHE_HOME = _home_path("XDG_CACHE_HOME", Path.home() / ".cache")
XDG_RUNTIME_DIR = _home_path(
    "XDG_RUNTIME_DIR", Path("/run/user") / str(os.getuid())
)

CONFIG_DIR = _override("LLAMA_BROKER_CONFIG_DIR", XDG_CONFIG_HOME / "llama-broker")
DATA_DIR = _override("LLAMA_BROKER_DATA_DIR", XDG_DATA_HOME / "llama-broker")
MODELS_DIR = _override("LLAMA_BROKER_MODELS_DIR", XDG_DATA_HOME / "models")
RUNTIMES_DIR = _override("LLAMA_BROKER_RUNTIMES_DIR", XDG_DATA_HOME / "runtimes")
CACHE_DIR = _override("LLAMA_BROKER_CACHE_DIR", XDG_CACHE_HOME / "llama-broker")
RUNTIME_STATE_DIR = _override(
    "LLAMA_BROKER_RUNTIME_DIR",
    XDG_RUNTIME_DIR / "llama-broker",
)
SYSTEMD_USER_DIR = _override(
    "SYSTEMD_USER_DIR",
    Path.home() / ".config" / "systemd" / "user",
)

PROFILES_PATH = _override(
    "LLAMA_BROKER_PROFILES",
    CONFIG_DIR / "profiles.json",
)
SECURITY_PATH = _override(
    "LLAMA_BROKER_SECURITY_FILE",
    CONFIG_DIR / "security.json",
)
API_KEY_PATH = _override(
    "LLAMA_BROKER_API_KEY_FILE",
    CONFIG_DIR / "api-keys",
)
STATE_PATH = _override(
    "LLAMA_BROKER_STATE_FILE",
    DATA_DIR / "state.json",
)
LOCK_PATH = _override(
    "LLAMA_BROKER_LOCK_FILE",
    RUNTIME_STATE_DIR / "model-switch.lock",
)
PROCESS_LOCK_PATH = _override(
    "LLAMA_BROKER_PROCESS_LOCK_FILE",
    RUNTIME_STATE_DIR / "instance.lock",
)
