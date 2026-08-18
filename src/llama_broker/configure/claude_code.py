#!/usr/bin/env python3
"""Claude Code configuration provider for llama-broker."""

from __future__ import annotations

from pathlib import Path

from .base import ToolProvider
from .registry import ConfigureResult, ToolRegistry

CLAUDE_ENV = Path.home() / ".claude" / ".env"
CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"


class ClaudeCodeProvider(ToolProvider):

    @property
    def name(self) -> str:
        return "Claude Code"

    @property
    def config_path(self) -> Path:
        return CLAUDE_SETTINGS

    def configure(
        self,
        registry: ToolRegistry,
        dry_run: bool = False,
        model: str | None = None,
    ) -> ConfigureResult:
        changes = []

        # Claude Code uses ANTHROPIC_BASE_URL env var for custom endpoints
        env_path = CLAUDE_ENV
        env_data = self._read_env(env_path)

        if model:
            # Claude Code exposes one active endpoint through this env file;
            # preserve it unless the operator explicitly selects the broker.
            new_url = registry.broker_url
            old_url = env_data.get("ANTHROPIC_BASE_URL", "")

            if old_url != new_url:
                env_data["ANTHROPIC_BASE_URL"] = new_url
                changes.append(f"ANTHROPIC_BASE_URL: {old_url or '(unset)'} -> {new_url}")

            if registry.api_key and env_data.get("ANTHROPIC_API_KEY") != registry.api_key:
                env_data["ANTHROPIC_API_KEY"] = registry.api_key
                changes.append("ANTHROPIC_API_KEY: applied (value redacted)")

            if env_data.get("ANTHROPIC_MODEL") != model:
                env_data["ANTHROPIC_MODEL"] = model
                changes.append(f"ANTHROPIC_MODEL: selected {model}")

        if not changes:
            return ConfigureResult()

        if not dry_run:
            self._write_env(env_path, env_data)

        return ConfigureResult(
            written=True,
            path=str(env_path),
            details=changes + [
                "NOTE: Claude Code uses ANTHROPIC_BASE_URL for provider routing.",
                "Model selection remains invocation-controlled unless --model is used.",
            ],
        )
