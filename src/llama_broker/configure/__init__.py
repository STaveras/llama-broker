#!/usr/bin/env python3
"""llama-broker multi-tool configuration system.

Generates and writes provider configurations for supported AI coding tools
from the broker's profiles.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .registry import BROKER_URL, ToolRegistry

# Lazy-import providers to avoid import errors when only some are needed
ALL_TOOLS = ["hermes", "opencode", "claude-code", "codex", "openclaw"]
DEFAULT_TOOLS = ["hermes", "opencode", "claude-code", "openclaw"]


def _select_tools(requested: list[str]) -> list[str]:
    """Expand ``all`` without implicitly rewriting the active Codex config."""

    selected = list(DEFAULT_TOOLS) if "all" in requested else []
    for name in requested:
        if name != "all" and name not in selected:
            selected.append(name)
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llama-broker configure",
        description="Configure AI coding tools to use llama-broker",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=ALL_TOOLS + ["all"],
        default=list(DEFAULT_TOOLS),
        help="Tools to configure (default: all except Codex; name codex to opt in)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without writing files",
    )
    parser.add_argument(
        "--model",
        metavar="ALIAS",
        help="explicitly select this broker profile alias for the configured tool",
    )
    parser.add_argument(
        "--broker-url",
        default=BROKER_URL,
        help=f"Broker base URL (default: {BROKER_URL})",
    )
    parser.add_argument(
        "--profiles",
        default=None,
        help="Path to profiles.json (default: auto-detect)",
    )
    return parser


def _get_provider(name: str):
    if name == "hermes":
        from .hermes import HermesProvider
        return HermesProvider()
    elif name == "opencode":
        from .opencode import OpenCodeProvider
        return OpenCodeProvider()
    elif name == "claude-code":
        from .claude_code import ClaudeCodeProvider
        return ClaudeCodeProvider()
    elif name == "codex":
        from .codex import CodexProvider
        return CodexProvider()
    elif name == "openclaw":
        from .openclaw import OpenClawProvider
        return OpenClawProvider()
    else:
        raise ValueError(f"Unknown tool: {name}")


def main(argv: list[str] | None = None, *, indent: bool = False) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    prefix = "  " if indent else ""

    tools = _select_tools(args.tools)

    registry = ToolRegistry(
        broker_url=args.broker_url,
        profiles_path=args.profiles,
    )
    if args.model is not None and registry.get_model_by_alias(args.model) is None:
        parser.error(f"unknown broker profile alias: {args.model}")

    for tool_name in tools:
        try:
            provider = _get_provider(tool_name)
        except Exception as e:
            print(f"{prefix}[{tool_name}] ERROR: {e}")
            continue

        try:
            result = provider.configure(
                registry,
                dry_run=args.dry_run,
                model=args.model,
            )
            if result.written:
                mode = "WOULD WRITE" if args.dry_run else "WROTE"
                print(f"{prefix}[{tool_name}] {mode}: {result.path}")
                if result.details:
                    for line in result.details:
                        print(f"{prefix}  {line}")
            elif result.skipped:
                print(f"{prefix}[{tool_name}] SKIPPED: {result.reason}")
            else:
                print(f"{prefix}[{tool_name}] OK (no changes needed)")
        except Exception as e:
            print(f"{prefix}[{tool_name}] ERROR: {e}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
