#!/usr/bin/env python3
"""Local heuristic check; emits locations only, never matched values."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

PRIVATE_PATH_PATTERNS = (
    re.compile(r'^docs/basalt-(?:production-stage/|(?:packaged-tls-parent|cache-budget-tuning|candidate-(?:catalog|profile|stagingprofiles-manifest)|packaged-tls-qualification)\.json$|r[56]-(?:deployment-report|final-qualification)\.md$)'),
    re.compile(r"^candidate/(?:.*\.json|.*\.service)$"),
    re.compile(r"(?:^|/)(?:profiles\.json(?:\.bak-.*)?|catalog\.json|security\.json|state\.json|api-keys|AGENTS\.md|DESIGN\.md|\.env(?:\..*)?)$"),
    re.compile(r"\.(?:pem|key|p12|pfx)$"),
)
PATTERNS = {
    "personal absolute path": re.compile(rb"(?:/home/(?!user(?:/|\b)|example(?:/|\b)|<)[A-Za-z0-9_.-]+/|/mnt/[a-z]/Users/(?!(?:user|example)(?:/|\b))[^/\s]+/|[A-Za-z]:[\\/]+Users[\\/]+(?!(?:user|example)(?:[\\/]|\b))[^\\/\s]+)"),
    "private network example": re.compile(rb"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"),
    "provider token": re.compile(rb"\b(?:AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{24,})\b"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
}
CREDENTIAL = re.compile(rb'''(?i)(?:api[_-]?key|access[_-]?token|secret|password)["']?\s*[:=]\s*["']([^"'\r\n]{8,})["']''')


def inspect_blob(rel: str, data: bytes, fixture_allowlist: dict[str, list[str]]) -> list[str]:
    failures = []
    if not rel.endswith((".example", ".example.json")) and any(p.search(rel) for p in PRIVATE_PATH_PATTERNS):
        failures.append(f"private configuration: {rel}")
    if b"\0" in data:
        failures.append(f"binary blob requires manual review: {rel}")
        return failures
    for category, pattern in PATTERNS.items():
        for match in pattern.finditer(data):
            failures.append(f"{category}: {rel}:{data[:match.start()].count(bytes([10])) + 1}")
    for match in CREDENTIAL.finditer(data):
        digest = hashlib.sha256(match.group(1)).hexdigest()
        if digest not in fixture_allowlist.get(rel, []):
            failures.append(f"unreviewed credential literal: {rel}:{data[:match.start()].count(bytes([10])) + 1}")
    return failures


def check(root: Path) -> list[str]:
    allowlist = json.loads((root / "tests/fixtures/synthetic-credential-sha256.json").read_text())
    names = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
    failures = []
    for rel in filter(None, names):
        path = root / rel
        if path.is_symlink():
            failures.append(f"symlink requires manual review: {rel}")
        elif not path.is_file():
            failures.append(f"missing tracked file: {rel}")
        else:
            failures.extend(inspect_blob(rel, path.read_bytes(), allowlist))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    failures = check(parser.parse_args().root.resolve())
    for failure in failures:
        print(failure)
    print("Public-tree heuristic check " + ("failed" if failures else "passed; external audit not implied"))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
