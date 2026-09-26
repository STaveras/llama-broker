#!/usr/bin/env python3
"""Shared field mapping registry for llama-broker multi-tool configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..broker_security import load_security_config, read_api_keys
from ..paths import PROFILES_PATH


def _default_broker_url() -> str:
    explicit = os.environ.get("LLAMA_BROKER_CLIENT_URL")
    if explicit:
        return explicit.rstrip("/")
    security = load_security_config()
    scheme = "https" if security.broker_tls_cert_file else "http"
    return f"{scheme}://127.0.0.1:11434/v1"


BROKER_URL = _default_broker_url()
DEFAULT_PROFILES_PATHS = [
    PROFILES_PATH,
    Path(__file__).resolve().parents[1] / "data/profiles.example.json",
    Path(__file__).resolve().parents[3] / "profiles.example.json",
]


@dataclass
class ConfigureResult:
    """Result of a tool configuration attempt."""
    written: bool = False
    path: str = ""
    skipped: bool = False
    reason: str = ""
    details: list[str] = field(default_factory=list)


@dataclass
class ToolProfile:
    """Normalized profile data for tool configuration."""
    alias: str
    display_name: str
    context_size: int
    parallel: int
    capabilities: list[str]
    role: str
    profile_id: str = ""
    aliases: tuple[str, ...] = ()

    @classmethod
    def from_broker(cls, profile: dict[str, Any]) -> "ToolProfile":
        raw_aliases = profile.get("aliases", [])
        aliases = tuple(
            str(alias)
            for alias in raw_aliases
            if isinstance(alias, str) and alias.strip()
        ) if isinstance(raw_aliases, list) else ()
        return cls(
            alias=profile["alias"],
            display_name=profile.get("display_name", profile["alias"]),
            context_size=profile["context_size"],
            parallel=profile["parallel"],
            capabilities=profile.get("capabilities", ["text"]),
            role=profile.get("role", "persistent"),
            profile_id=str(profile.get("id", "")),
            aliases=aliases,
        )


class ToolRegistry:
    """Central registry providing normalized profile data to tool providers."""

    def __init__(
        self,
        broker_url: str = BROKER_URL,
        profiles_path: str | None = None,
        api_key: str | None = None,
    ):
        self.broker_url = broker_url.rstrip("/")
        self.profiles_path = self._resolve_profiles_path(profiles_path)
        if api_key is None:
            security = load_security_config()
            if security.api_key is not None:
                api_key = security.api_key
            elif security.api_key_file is not None:
                keys = read_api_keys(security.api_key_file)
                if not keys:
                    raise ValueError(f"API-key file is empty: {security.api_key_file}")
                api_key = keys[-1]
        self.api_key = api_key
        self._raw_profiles = self._load_profiles()
        self.profiles = [
            ToolProfile.from_broker(p) for p in self._raw_profiles
        ]

    def _resolve_profiles_path(self, explicit: str | None) -> Path:
        if explicit:
            p = Path(explicit)
            if p.is_file():
                return p
            raise FileNotFoundError(f"profiles.json not found: {explicit}")
        for candidate in DEFAULT_PROFILES_PATHS:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "Cannot find profiles.json. Use --profiles to specify."
        )

    def _load_profiles(self) -> list[dict[str, Any]]:
        data = json.loads(self.profiles_path.read_text(encoding="utf-8"))
        return data.get("profiles", [])

    def get_model_names(self) -> list[str]:
        """Return only canonical public model IDs for generated configs."""
        return [p.alias for p in self.profiles]

    def canonical_model_name(self, reference: str) -> str | None:
        """Resolve an internal profile reference to its public model ID.

        Profile IDs and compatibility aliases remain valid inputs for
        resolution, but generated client model catalogs must contain only the
        canonical ``alias`` value.
        """
        value = str(reference or "").strip()
        if not value:
            return None
        folded = value.casefold()
        for profile in self.profiles:
            references = (profile.profile_id, profile.alias, *profile.aliases)
            if any(
                candidate and str(candidate).strip().casefold() == folded
                for candidate in references
            ):
                return profile.alias
        return None

    def get_model_by_alias(self, alias: str) -> ToolProfile | None:
        for p in self.profiles:
            if p.alias == alias or alias in (p.alias,):
                return p
        return None
