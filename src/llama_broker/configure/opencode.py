#!/usr/bin/env python3
"""OpenCode configuration provider for llama-broker."""

from __future__ import annotations

from pathlib import Path

from .base import ToolProvider
from .registry import ConfigureResult, ToolRegistry

OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"


class OpenCodeProvider(ToolProvider):

    @property
    def name(self) -> str:
        return "OpenCode"

    @property
    def config_path(self) -> Path:
        return OPENCODE_CONFIG

    def configure(
        self,
        registry: ToolRegistry,
        dry_run: bool = False,
        model: str | None = None,
    ) -> ConfigureResult:
        config_path = self.config_path
        if not config_path.is_file():
            return ConfigureResult(
                skipped=True,
                reason=f"OpenCode config not found: {config_path}",
            )

        data = self._read_json(config_path)
        changes = []

        # Ensure provider structure
        if "provider" not in data:
            data["provider"] = {}
        if "llamacpp" not in data["provider"]:
            data["provider"]["llamacpp"] = {
                "name": "llama.cpp",
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": registry.broker_url},
                "models": {},
            }
            changes.append("Added llamacpp provider")

        provider = data["provider"]["llamacpp"]

        if registry.api_key:
            options = provider.setdefault("options", {})
            if options.get("apiKey") != registry.api_key:
                options["apiKey"] = registry.api_key
                changes.append("apiKey: applied to llamacpp provider")

        # Update baseURL
        old_url = provider.get("options", {}).get("baseURL", "")
        if old_url != registry.broker_url:
            if "options" not in provider:
                provider["options"] = {}
            provider["options"]["baseURL"] = registry.broker_url
            changes.append(f"baseURL: {old_url} -> {registry.broker_url}")

        # Update models
        if "models" not in provider:
            provider["models"] = {}

        for profile in registry.profiles:
            alias = profile.alias
            slot_ctx = profile.context_size // profile.parallel
            if alias not in provider["models"]:
                provider["models"][alias] = {
                    "name": profile.display_name,
                    "options": {
                        "num_ctx": slot_ctx,
                        "temperature": 0.6,
                    },
                }
                changes.append(f"Added model: {alias}")
            else:
                model_entry = provider["models"][alias]
                old_ctx = model_entry.get("options", {}).get("num_ctx")
                if old_ctx != slot_ctx:
                    if "options" not in model_entry:
                        model_entry["options"] = {}
                    model_entry["options"]["num_ctx"] = slot_ctx
                    changes.append(f"{alias}: num_ctx {old_ctx} -> {slot_ctx}")

        if model:
            selected = f"llamacpp/{model}"
            if data.get("model") != selected:
                data["model"] = selected
                changes.append(f"model: selected {selected}")

        if not changes:
            return ConfigureResult()

        result = ConfigureResult(
            written=True,
            path=str(config_path),
            details=changes,
        )

        if not dry_run:
            self._write_json(config_path, data)

        return result
