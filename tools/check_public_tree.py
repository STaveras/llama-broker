#!/usr/bin/env python3
"""Fail if a checkout still contains private release artifacts or host paths."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

PRIVATE_PATH_PATTERNS = (
    re.compile(r"(^|/)profiles\.json(?:\.bak-.*)?$"),
    re.compile(r"(^|/)AGENTS\.md$"),
    re.compile(r"(^|/)DESIGN\.md$"),
    re.compile(r"(^|/)state\.json$"),
    re.compile(r"(^|/)(?:security\.json|api-keys)$"),
)
PUBLIC_EXCLUDED_PATH_PATTERNS = (
    # These maintainer-only prompts/plans are excluded by MANIFEST.in and are
    # intentionally not part of the public source distribution.
    re.compile(r"(^|/)docs/plans(?:/|$)"),
    re.compile(r"(^|/)docs/goals(?:/|$)"),
    re.compile(r"(^|/)docs/broker-autotune-workstream\.md$"),
)
HOST_PATTERNS = (
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    re.compile(r"/mnt/[A-Za-z]/Users/", re.IGNORECASE),
    re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE),
)
SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(?:api[-_]?key|authorization|bearer|password|token|secret)"
    r"\s*(?:=|:)\s*[\"'](?!\[REDACTED\])[^\"']{8,}[\"']"
)


def tracked_files(root: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"], text=False
    )
    return [root / item for item in output.decode().split("\0") if item]


def check(root: Path) -> list[str]:
    failures: list[str] = []
    for path in tracked_files(root):
        rel = path.relative_to(root).as_posix()
        if any(pattern.search(rel) for pattern in PUBLIC_EXCLUDED_PATH_PATTERNS):
            continue
        if any(pattern.search(rel) for pattern in PRIVATE_PATH_PATTERNS):
            failures.append(f"private path: {rel}")
            continue
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for pattern in HOST_PATTERNS:
            if pattern.search(text):
                failures.append(f"host-specific path in {rel}")
                break
        if SECRET_VALUE_PATTERN.search(text):
            failures.append(f"possible raw secret in {rel}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    failures = check(args.root.resolve())
    if failures:
        print("public-tree check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("public-tree check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
