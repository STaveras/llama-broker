#!/usr/bin/env python3
"""Base class for tool configuration providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path

from .registry import ConfigureResult, ToolRegistry


class ToolProvider(ABC):
    """Base class for tool-specific configuration providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable tool name."""

    @property
    @abstractmethod
    def config_path(self) -> Path:
        """Path to the tool's configuration file."""

    @abstractmethod
    def configure(
        self,
        registry: ToolRegistry,
        dry_run: bool = False,
        set_default: bool = False,
    ) -> ConfigureResult:
        """Configure the tool to use llama-broker.

        Args:
            registry: The tool registry with broker profile data
            dry_run: If True, return what would be written without modifying files
            set_default: If True, explicitly select the broker default model/provider

        Returns:
            ConfigureResult with details about what was done
        """

    def _ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _read_json(self, path: Path) -> dict:
        """Read and parse a JSON config file."""
        import json
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, data: dict, indent: int = 2) -> None:
        """Write a dict as formatted JSON."""
        import json
        self._ensure_parent(path)
        path.write_text(
            json.dumps(data, indent=indent, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.chmod(path, 0o600)

    def _read_yaml(self, path: Path) -> str:
        """Read a YAML file as raw text."""
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_yaml(self, path: Path, content: str) -> None:
        """Write raw YAML content."""
        self._ensure_parent(path)
        path.write_text(content, encoding="utf-8")

    def _read_env(self, path: Path) -> dict[str, str]:
        """Read a .env file into a dict."""
        result = {}
        if not path.is_file():
            return result
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
        return result

    def _write_env(self, path: Path, data: dict[str, str]) -> None:
        """Write a .env file from a dict."""
        self._ensure_parent(path)
        lines = []
        for key, value in data.items():
            lines.append(f"{key}={value}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.chmod(path, 0o600)
