#!/usr/bin/env python3
"""Codex CLI configuration provider for llama-broker."""

from __future__ import annotations

from pathlib import Path

from .base import ToolProvider
from .registry import ConfigureResult, ToolRegistry

CODEX_CONFIG = Path.home() / ".codex" / "config.toml"


class CodexProvider(ToolProvider):

    @property
    def name(self) -> str:
        return "Codex CLI"

    @property
    def config_path(self) -> Path:
        return CODEX_CONFIG

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
                reason=f"Codex config not found: {config_path}",
            )

        content = self._read_yaml(config_path)
        lines = content.splitlines()
        changes = []

        # Codex supports --oss with local providers (lmstudio, ollama).  Keep
        # the selected model unless the operator explicitly opts in.
        new_lines = []
        found_model = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("model ="):
                found_model = True
                if model:
                    old_model = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                    if old_model != model:
                        new_lines.append(f'model = "{model}"')
                        changes.append(
                            f"model: {old_model} -> {model}"
                        )
                        continue
                new_lines.append(line)
            else:
                new_lines.append(line)

        if model and not found_model:
            new_lines.insert(0, f'model = "{model}"')
            changes.append(f"Added model = {model}")
            found_model = True

        # Add a comment about --oss usage
        has_oss_comment = any(
            "oss" in line.lower() and line.strip().startswith("#")
            for line in new_lines
        )
        if not has_oss_comment and found_model:
            # Add comment after model line
            for i, line in enumerate(new_lines):
                if line.strip().startswith("model ="):
                    new_lines.insert(i + 1, "")
                    new_lines.insert(i + 2, "# Codex local model usage: codex --oss --local-provider ollama")
                    new_lines.insert(i + 3, "# Or set model to any name recognized by the broker")
                    changes.append(
                        "Added local-provider usage guidance; "
                        + (
                            "explicit broker model selected"
                            if model
                            else "selected model preserved"
                        )
                    )
                    break

        if not changes:
            return ConfigureResult()

        new_content = "\n".join(new_lines) + "\n"
        result = ConfigureResult(
            written=True,
            path=str(config_path),
            details=changes + [
                "NOTE: Codex uses --oss flag for local models.",
                "Run: codex --oss --local-provider ollama",
            ],
        )

        if not dry_run:
            self._write_yaml(config_path, new_content)

        return result
