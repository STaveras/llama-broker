#!/usr/bin/env python3
"""OpenClaw configuration provider for llama-broker."""

from __future__ import annotations

from pathlib import Path

from .base import ToolProvider
from .registry import ConfigureResult, ToolRegistry

OPENCLAW_DIR = Path.home() / ".config" / "openclaw"
OPENCLAW_ENV = OPENCLAW_DIR / "openclaw.env"


class OpenClawProvider(ToolProvider):

    @property
    def name(self) -> str:
        return "OpenClaw"

    @property
    def config_path(self) -> Path:
        return OPENCLAW_ENV

    def configure(
        self,
        registry: ToolRegistry,
        dry_run: bool = False,
        set_default: bool = False,
    ) -> ConfigureResult:
        env_path = OPENCLAW_ENV
        env_data = self._read_env(env_path)

        changes = []

        if set_default:
            # These environment variables are OpenClaw's active route, so do
            # not rewrite them unless the operator explicitly opts in.
            new_url = registry.broker_url
            old_url = env_data.get("LLAMA_BROKER_URL", "")
            if old_url != new_url:
                env_data["LLAMA_BROKER_URL"] = new_url
                changes.append(f"LLAMA_BROKER_URL: {old_url or '(unset)'} -> {new_url}")

            if registry.api_key and env_data.get("LLAMA_BROKER_API_KEY") != registry.api_key:
                env_data["LLAMA_BROKER_API_KEY"] = registry.api_key
                changes.append("LLAMA_BROKER_API_KEY: applied (value redacted)")

            default_model = registry.default_model
            if default_model and env_data.get("LLAMA_BROKER_DEFAULT_MODEL") != default_model:
                old_model = env_data.get("LLAMA_BROKER_DEFAULT_MODEL", "")
                env_data["LLAMA_BROKER_DEFAULT_MODEL"] = default_model
                changes.append(
                    f"LLAMA_BROKER_DEFAULT_MODEL: {old_model or '(unset)'} -> {default_model}"
                )

        if not changes:
            return ConfigureResult()

        result = ConfigureResult(
            written=True,
            path=str(env_path),
            details=changes,
        )

        if not dry_run:
            self._write_env(env_path, env_data)

        return result
