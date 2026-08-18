#!/usr/bin/env python3
"""Hermes Agent configuration provider for llama-broker."""

from __future__ import annotations

import re
from pathlib import Path

from ..broker_security import load_security_config

from .base import ToolProvider
from .registry import ConfigureResult, ToolRegistry

HERMES_CONFIG = Path.home() / ".hermes" / "config.yaml"


class HermesProvider(ToolProvider):

    @property
    def name(self) -> str:
        return "Hermes Agent"

    @property
    def config_path(self) -> Path:
        return HERMES_CONFIG

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
                reason=f"Hermes config not found: {config_path}",
            )

        content = self._read_yaml(config_path)
        changes = []

        # Update only the named broker provider.  The top-level ``model`` block
        # may point at an external provider and must never be rewritten here.
        content, field_status = self._set_provider_field(
            content,
            "base_url",
            registry.broker_url,
        )
        if field_status == "added":
            changes.append("base_url: added to llama-broker provider")
        elif field_status == "updated":
            changes.append("base_url: updated for llama-broker provider")

        if model:
            default_model = model
            for field, value in (
                ("default", default_model),
                ("provider", "llama-broker"),
                ("base_url", registry.broker_url),
            ):
                content, model_status = self._set_model_field(content, field, value)
                if model_status in {"added", "updated"}:
                    changes.append(f"model.{field}: selected llama-broker")
            content, provider_status = self._set_provider_field(
                content,
                "default_model",
                default_model,
            )
            if provider_status in {"added", "updated"}:
                changes.append("llama-broker.default_model: selected explicit broker model")

        if registry.api_key:
            content, field_status = self._set_provider_field(
                content, "api_key", registry.api_key
            )
            if field_status == "added":
                changes.append("api_key: applied to llama-broker provider")
            elif field_status == "updated":
                changes.append("api_key: rotated for llama-broker provider")

        if registry.broker_url.lower().startswith("https://"):
            security = load_security_config()
            if security.broker_tls_ca_file is not None:
                content, field_status = self._set_provider_field(
                    content,
                    "ssl_ca_cert",
                    str(security.broker_tls_ca_file),
                )
                if field_status in {"added", "updated"}:
                    changes.append("ssl_ca_cert: applied to llama-broker provider")

        # Replace models block under llama-broker provider
        # Find the models: section under the llama-broker provider
        models_pattern = re.compile(
            r"(  llama-broker:.*?models:\s*\n)(.*?)(\n(?:  \w|\Z))",
            re.DOTALL,
        )
        match = models_pattern.search(content)
        if match:
            prefix = match.group(1)
            old_models = match.group(2)
            suffix = match.group(3)

            new_models = self._format_models(registry)
            new_block = prefix + new_models + suffix

            if old_models.strip() != new_models.strip():
                content = content[:match.start()] + new_block + content[match.end():]
                changes.append(f"models: updated {len(registry.get_model_names())} entries")
        else:
            changes.append("NOTE: could not find models section to update")

        if not changes:
            return ConfigureResult()

        result = ConfigureResult(
            written=True,
            path=str(config_path),
            details=changes,
        )

        if not dry_run:
            self._write_yaml(config_path, content)

        return result

    def _set_provider_field(
        self,
        content: str,
        field: str,
        value: str,
    ) -> tuple[str, str]:
        """Set a scalar field in Hermes' llama-broker provider block."""

        lines = content.splitlines(keepends=True)
        provider_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.rstrip("\r\n") == "  llama-broker:"
            ),
            None,
        )
        if provider_index is None:
            return content, "missing"

        end = len(lines)
        for index in range(provider_index + 1, len(lines)):
            if lines[index].strip() and not lines[index].startswith((" ", "\t")):
                end = index
                break

        field_prefix = f"    {field}:"
        field_index = next(
            (
                index
                for index in range(provider_index + 1, end)
                if lines[index].startswith(field_prefix)
            ),
            None,
        )
        new_line = f"    {field}: {value}\n"
        if field_index is None:
            lines.insert(provider_index + 1, new_line)
            return "".join(lines), "added"
        if lines[field_index] == new_line:
            return content, "unchanged"
        lines[field_index] = new_line
        return "".join(lines), "updated"

    def _set_model_field(
        self,
        content: str,
        field: str,
        value: str,
    ) -> tuple[str, str]:
        """Set a scalar field in Hermes' top-level model block."""

        lines = content.splitlines(keepends=True)
        model_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.rstrip("\r\n") == "model:"
            ),
            None,
        )
        if model_index is None:
            return content, "missing"

        end = len(lines)
        for index in range(model_index + 1, len(lines)):
            if lines[index].strip() and not lines[index].startswith((" ", "\t")):
                end = index
                break

        field_prefix = f"  {field}:"
        field_index = next(
            (
                index
                for index in range(model_index + 1, end)
                if lines[index].startswith(field_prefix)
            ),
            None,
        )
        new_line = f"  {field}: {value}\n"
        if field_index is None:
            lines.insert(model_index + 1, new_line)
            return "".join(lines), "added"
        if lines[field_index] == new_line:
            return content, "unchanged"
        lines[field_index] = new_line
        return "".join(lines), "updated"

    def _format_models(self, registry: ToolRegistry) -> str:
        lines = []
        for profile in registry.profiles:
            slot_ctx = profile.context_size // profile.parallel
            lines.append(f"      {profile.alias}:")
            lines.append(f"        context_length: {slot_ctx}")
        return "\n".join(lines) + "\n"
