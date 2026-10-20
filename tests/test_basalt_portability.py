"""Portable acceptance checks; operator artifacts must stay outside Git."""
from pathlib import Path
import subprocess

from tools.check_public_tree import inspect_blob

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DOCS = (
    "docs/basalt-packaged-tls-parent.json",
    "docs/basalt-production-stage/merged-profiles.json",
    "docs/basalt-candidate-catalog.json",
    "docs/basalt-r5-deployment-report.md",
    "docs/basalt-r6-final-qualification.md",
)


def test_public_checker_rejects_operator_snapshots_even_without_secrets():
    for name in PRIVATE_DOCS:
        assert inspect_blob(name, b"{}", {}), name


def test_operator_artifacts_are_untracked_and_explicitly_ignored():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT).decode().splitlines()
    assert not any(name.startswith("docs/basalt-production-stage/") for name in tracked)
    for name in PRIVATE_DOCS:
        assert name not in tracked
        assert subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", name], cwd=ROOT
        ).returncode == 0, name
    assert subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "tests/test_basalt_portability.py"], cwd=ROOT
    ).returncode == 1


def test_acceptance_document_has_no_deployment_attestations():
    text = (ROOT / "docs/basalt-acceptance.md").read_text()
    assert "[x]" not in text
    assert "operator" in text.lower()
    assert "qualification" in text.lower()
