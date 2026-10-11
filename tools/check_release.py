#!/usr/bin/env python3
"""Validate source/tag and built distribution metadata before any publication."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:\.dev(?:0|[1-9]\d*))?")


def validate_version(version: str, tag: str | None = None) -> None:
    if not isinstance(version, str) or not VERSION.fullmatch(version):
        raise ValueError("version must be X.Y.Z or X.Y.Z.devN")
    if tag is not None and (".dev" in version or tag != f"v{version}"):
        raise ValueError("release tag must exactly match a stable package version")


def validate_tag_target(root: Path, tag: str) -> None:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
    target = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"]
    ).strip()
    if target != head:
        raise ValueError("release tag must point at the checked-out commit")


def validate_publication(tag: str | None, reserved: list[str]) -> None:
    if tag is None or tag in reserved:
        raise ValueError("retrospective tags are notes-only; publication requires a new release tag")


def validate_distributions(dist: Path, version: str) -> None:
    wheels, sdists = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("expected exactly one wheel and one sdist")
    if not wheels[0].name.startswith(f"llama_broker-{version}-") or sdists[0].name not in (f"llama_broker-{version}.tar.gz", f"llama-broker-{version}.tar.gz"):
        raise ValueError("distribution filenames must match the project name and version")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("wheel must contain exactly one metadata record")
        wheel_metadata = archive.read(names[0])
    with tarfile.open(sdists[0], "r:gz") as archive:
        names = [n for n in archive.getnames() if n.count("/") == 1 and n.endswith("/PKG-INFO")]
        if len(names) != 1:
            raise ValueError("sdist must contain exactly one root metadata record")
        sdist_metadata = archive.extractfile(names[0]).read()
    for raw in (wheel_metadata, sdist_metadata):
        metadata = BytesParser().parsebytes(raw)
        if metadata["Name"] != "llama-broker" or metadata["Version"] != version:
            raise ValueError("distribution name/version differs from pyproject.toml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tag")
    mode.add_argument("--development", action="store_true")
    parser.add_argument("--require-head-tag", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    try:
        validate_version(version, args.tag)
        if args.require_head_tag or args.publish:
            if not args.tag:
                raise ValueError("publication requires an explicit release tag")
            validate_tag_target(ROOT, args.tag)
        if args.publish:
            reserved = json.loads((ROOT / "docs/retrospective-tags.json").read_text())
            validate_publication(args.tag, reserved)
        if args.dist:
            validate_distributions(args.dist, version)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"release validation failed: {error}\n")
    print("Release metadata validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
