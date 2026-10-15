"""CLI-only presentation and interactive selection for runtime control."""
from __future__ import annotations

import subprocess
import sys
from typing import Callable, Iterable, Mapping, Sequence, TextIO

from .runtime_control import RuntimeControlError, load_projects, project_commits


def _git(repo, args: Sequence[str]) -> list[str]:
    proc = subprocess.run(["git", *args], cwd=str(repo), text=True,
                          capture_output=True, check=True, shell=False)
    return [line for line in proc.stdout.splitlines() if line]


def project_branches(root, name: str) -> list[str]:
    """Read local/remote branch names without modifying the repository."""
    item = next((x for x in load_projects(root) if x.get("name") == name), None)
    if not item:
        raise RuntimeControlError(f"unknown project: {name}")
    branches = _git(item["path"], ["branch", "--format=%(refname:short)", "--no-color", "--"])
    return sorted(dict.fromkeys(branches))


def default_branch(root, name: str) -> str | None:
    item = next((x for x in load_projects(root) if x.get("name") == name), None)
    if not item:
        raise RuntimeControlError(f"unknown project: {name}")
    value = item.get("default_branch") or item.get("resolved_default_ref")
    return str(value) if value else None


def select_source(root, name: str, *, input_stream: TextIO = sys.stdin,
                  output_stream: TextIO = sys.stdout) -> dict:
    if not input_stream.isatty():
        raise RuntimeControlError("--select requires an interactive TTY; pass --branch or --commit")
    branches = project_branches(root, name)
    if not branches:
        raise RuntimeControlError("project has no branches")
    print("Branches:", file=output_stream)
    for i, branch in enumerate(branches, 1):
        print(f"  {i}. {branch}", file=output_stream)
    try:
        choice = input_stream.readline()
    except (EOFError, KeyboardInterrupt):
        raise RuntimeControlError("selection cancelled")
    if not choice.strip():
        raise RuntimeControlError("selection cancelled")
    try:
        branch = branches[int(choice.strip()) - 1]
    except (ValueError, IndexError):
        raise RuntimeControlError("invalid branch selection")
    commits = project_commits(root, name, branch)
    if not commits:
        raise RuntimeControlError(f"no commits found on branch {branch!r}")
    print(f"Commits for {branch}:", file=output_stream)
    for i, commit in enumerate(commits, 1):
        print(f"  {i}. {commit['hash']} {commit['subject']}", file=output_stream)
    try:
        choice = input_stream.readline()
    except (EOFError, KeyboardInterrupt):
        raise RuntimeControlError("selection cancelled")
    if not choice.strip():
        raise RuntimeControlError("selection cancelled")
    try:
        selected = commits[int(choice.strip()) - 1]
    except (ValueError, IndexError):
        raise RuntimeControlError("invalid commit selection")
    return {"branch": branch, "commit": selected["hash"], "subject": selected["subject"]}


def bundle_table_rows(manifests: Iterable[Mapping[str, object]]) -> list[tuple[str, ...]]:
    rows = []
    for item in manifests:
        rows.append((str(item.get("runtime_family", "unknown")), str(item.get("source_url", "unknown")),
                     str(item.get("resolved_commit", "unknown")), str(item.get("subject", "unknown")),
                     str(item.get("status", "published"))))
    return rows
