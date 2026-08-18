"""Fixture materializer for the generic profile manifest.

The committed manifest (tests/fixtures/profiles.fixture.json) is machine-independent:
paths are ``__FIXTURE_ROOT__`` tokens and hashes are placeholders.  Tests call
:func:`materialize_fixture` to produce a real, loadable profiles.json inside a
temp directory, with empty files standing in for the model/server binaries and
correct SHA-256 hashes computed from them.  This keeps the release tree free of
any machine-specific absolute paths while exercising the real fail-closed
validation in :mod:`broker_profiles`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent
FIXTURE_MANIFEST = FIXTURES_DIR / "profiles.fixture.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_fixture(dest: Path | None = None) -> Path:
    """Write a loadable profiles.json and return its path.

    Creates ``<dest>/profiles.json`` (or a fresh temp dir when *dest* is None)
    with empty ``bin/llama-server`` and ``models/*.gguf`` placeholder files, and
    returns the path to the materialized manifest.
    """
    if dest is None:
        dest = Path(tempfile.mkdtemp(prefix="llama-broker-fixture-"))
    dest = dest.resolve()

    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))

    server = dest / "bin" / "llama-server"
    server.parent.mkdir(parents=True, exist_ok=True)
    server.touch()

    mmproj = dest / "models" / "mmproj-F16.gguf"
    mmproj.parent.mkdir(parents=True, exist_ok=True)
    mmproj.touch()

    model_paths: dict[str, Path] = {}
    for profile in manifest["profiles"]:
        model_name = Path(profile["model_path"]).name
        model = dest / "models" / model_name
        model.parent.mkdir(parents=True, exist_ok=True)
        model.touch()
        model_paths[profile["id"]] = model
        profile["server_binary"] = str(server)
        profile["server_sha256"] = _sha256(server)
        profile["model_path"] = str(model)
        profile["model_sha256"] = _sha256(model)
        if "mmproj_path" in profile:
            profile["mmproj_path"] = str(mmproj)
            profile["mmproj_sha256"] = _sha256(mmproj)

    out = dest / "profiles.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def fixture_bin(dest: Path | None = None) -> Path:
    """Return the materialized ``bin/llama-server`` path (creating it if needed)."""
    materialized = materialize_fixture(dest)
    return materialized.parent / "bin" / "llama-server"


if __name__ == "__main__":
    from broker_profiles import ProfileRegistry  # noqa: PLC0415

    path = materialize_fixture()
    registry = ProfileRegistry.from_path(path)
    print(f"materialized: {path}")
    print(f"profiles: {[p.id for p in registry.profiles]}")
