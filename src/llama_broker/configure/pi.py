#!/usr/bin/env python3
"""Pi coding agent configuration provider for llama-broker."""

from __future__ import annotations

from pathlib import Path

from .base import ToolProvider
from .registry import ConfigureResult, ToolRegistry

PI_AGENT_DIR = Path.home() / ".pi" / "agent"
PI_MODELS_CONFIG = PI_AGENT_DIR / "models.json"
PI_SETTINGS_CONFIG = PI_AGENT_DIR / "settings.json"

# The broker serves the OpenAI chat-completions API to pi.
PROVIDER_ID = "llamacpp"

# Thinking levels pi offers for custom models; the broker maps them onto
# its effort ladder.
_THINKING_LEVEL_MAP = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


class PiProvider(ToolProvider):

    @property
    def name(self) -> str:
        return "Pi Coding Agent"

    @property
    def config_path(self) -> Path:
        return PI_MODELS_CONFIG

    def configure(
        self,
        registry: ToolRegistry,
        dry_run: bool = False,
        model: str | None = None,
    ) -> ConfigureResult:
        # A fresh pi install has no models.json; that is normal.  Only skip
        # when pi has never been used at all.
        if not PI_AGENT_DIR.parent.is_dir():
            return ConfigureResult(
                skipped=True,
                reason=f"Pi agent directory not found: {PI_AGENT_DIR.parent}",
            )

        models_changes = self._configure_models(registry, dry_run)
        settings_changes = (
            self._configure_default_model(model, dry_run) if model else []
        )

        if not models_changes and not settings_changes:
            return ConfigureResult()

        if models_changes:
            path = str(PI_MODELS_CONFIG)
        else:
            path = str(PI_SETTINGS_CONFIG)
        result = ConfigureResult(
            written=True,
            path=path,
            details=models_changes + settings_changes,
        )
        return result

    def _configure_models(
        self,
        registry: ToolRegistry,
        dry_run: bool,
    ) -> list[str]:
        """Sync the broker provider entry in models.json.  Returns changes."""
        data = self._read_json(PI_MODELS_CONFIG)
        if "providers" not in data or not isinstance(data["providers"], dict):
            data["providers"] = {}
        providers = data["providers"]
        changes: list[str] = []
        if PROVIDER_ID not in providers:
            providers[PROVIDER_ID] = {
                "baseUrl": registry.broker_url,
                "api": "openai-completions",
                "models": [],
            }
            changes.append(f"Added {PROVIDER_ID} provider")
        provider = providers[PROVIDER_ID]

        old_url = provider.get("baseUrl", "")
        if old_url != registry.broker_url:
            provider["baseUrl"] = registry.broker_url
            changes.append(
                f"baseUrl: {old_url or '(unset)'} -> {registry.broker_url}"
            )

        if provider.get("api") != "openai-completions":
            provider["api"] = "openai-completions"
            changes.append("api: set openai-completions")

        if registry.api_key and provider.get("apiKey") != registry.api_key:
            provider["apiKey"] = registry.api_key
            changes.append("apiKey: applied")

        if not isinstance(provider.get("models"), list):
            provider["models"] = []
        models: list[dict] = provider["models"]
        available_aliases = {profile.alias for profile in registry.profiles}
        for entry in list(models):
            if isinstance(entry, dict) and entry.get("id") not in available_aliases:
                models.remove(entry)
                changes.append(f"Removed unavailable model: {entry.get('id')}")
        for profile in registry.profiles:
            slot_ctx = max(1, profile.context_size // profile.parallel)
            expected = self._model_entry(profile, slot_ctx)
            existing = next(
                (
                    m
                    for m in models
                    if isinstance(m, dict) and m.get("id") == profile.alias
                ),
                None,
            )
            if existing is None:
                models.append(expected)
                changes.append(f"Added model: {profile.alias}")
            elif existing != expected:
                models[models.index(existing)] = expected
                changes.append(f"Updated model: {profile.alias}")

        if changes and not dry_run:
            self._write_json(PI_MODELS_CONFIG, data)
        return changes

    def _configure_default_model(self, model: str, dry_run: bool) -> list[str]:
        """Point the pi default model at the broker.  Returns changes."""
        selected = f"{PROVIDER_ID}/{model}"
        settings = self._read_json(PI_SETTINGS_CONFIG)
        changes: list[str] = []
        if settings.get("defaultProvider") != PROVIDER_ID:
            changes.append(
                f"defaultProvider: "
                f"{settings.get('defaultProvider', '(unset)')} -> {PROVIDER_ID}"
            )
        if settings.get("defaultModel") != selected:
            changes.append(
                f"defaultModel: "
                f"{settings.get('defaultModel', '(unset)')} -> {selected}"
            )
        if changes and not dry_run:
            settings["defaultProvider"] = PROVIDER_ID
            settings["defaultModel"] = selected
            self._write_json(PI_SETTINGS_CONFIG, settings)
        return changes

    @staticmethod
    def _model_entry(profile, slot_ctx: int) -> dict:
        entry: dict = {
            "id": profile.alias,
            "name": profile.display_name,
            "contextWindow": slot_ctx,
            "maxOutputTokens": slot_ctx,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }
        if "reasoning" in profile.capabilities:
            entry["reasoning"] = True
            entry["thinkingLevelMap"] = dict(_THINKING_LEVEL_MAP)
        return entry
