#!/usr/bin/env python3
"""Discover local GGUF models and generate launch artifacts for llama-broker."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path.home() / ".local/share/models")).expanduser()
GENERATED_DIR = Path(os.environ.get("LLAMA_BROKER_GENERATED_DIR", ROOT / "generated"))
AUXILIARY_PREFIXES = ("mmproj-", "mtp-")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "model"


GGUF_TYPES = {
    0: ("B", 1),   # UINT8
    1: ("b", 1),   # INT8
    2: ("H", 2),   # UINT16
    3: ("h", 2),   # INT16
    4: ("I", 4),   # UINT32
    5: ("i", 4),   # INT32
    6: ("f", 4),   # FLOAT32
    7: ("?", 1),   # BOOL
    10: ("Q", 8),  # UINT64
    11: ("q", 8),  # INT64
    12: ("d", 8),  # FLOAT64
}


def _read_string(handle) -> str:
    length_data = handle.read(8)
    if len(length_data) != 8:
        raise ValueError("truncated GGUF string length")
    length = struct.unpack("<Q", length_data)[0]
    if length > 1 << 20:
        raise ValueError("unreasonable GGUF string length")
    value = handle.read(length)
    if len(value) != length:
        raise ValueError("truncated GGUF string")
    return value.decode("utf-8", errors="replace")


def _skip_value(handle, value_type: int) -> object | None:
    if value_type == 8:  # STRING
        return _read_string(handle)
    if value_type == 9:  # ARRAY
        element_type_data = handle.read(4)
        count_data = handle.read(8)
        if len(element_type_data) != 4 or len(count_data) != 8:
            raise ValueError("truncated GGUF array")
        element_type = struct.unpack("<I", element_type_data)[0]
        count = struct.unpack("<Q", count_data)[0]
        if count > 1 << 24:
            raise ValueError("unreasonable GGUF array length")
        for _ in range(count):
            _skip_value(handle, element_type)
        return None
    spec = GGUF_TYPES.get(value_type)
    if spec is None:
        raise ValueError(f"unsupported GGUF metadata type {value_type}")
    fmt, size = spec
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF metadata value")
    return struct.unpack("<" + fmt, data)[0]


def read_metadata(path: Path) -> dict[str, object]:
    """Read only the GGUF metadata header; never loads tensor data."""
    with path.open("rb") as handle:
        header = handle.read(24)
        if len(header) != 24 or header[:4] != b"GGUF":
            return {}
        _, version, _, metadata_count = struct.unpack("<4sIQQ", header)
        if version not in {1, 2, 3} or metadata_count > 1 << 20:
            return {}
        metadata: dict[str, object] = {}
        for _ in range(metadata_count):
            key = _read_string(handle)
            value_type_data = handle.read(4)
            if len(value_type_data) != 4:
                return {}
            value_type = struct.unpack("<I", value_type_data)[0]
            value = _skip_value(handle, value_type)
            if key in {"general.name", "general.basename", "general.architecture"}:
                metadata[key] = value
        return metadata


def model_display_name(path: Path) -> tuple[str, dict[str, object]]:
    try:
        metadata = read_metadata(path)
    except (OSError, ValueError, struct.error):
        metadata = {}
    name = str(metadata.get("general.name") or metadata.get("general.basename") or path.stem).strip()
    return name, metadata


def simple_alias(name: str, path: Path, metadata: dict[str, object]) -> str:
    """Prefer embedded names, falling back from generic exporter names."""
    source = name
    normalized = slugify(source)
    if normalized in {"merged", "model", "unknown", "untitled"}:
        source = path.stem
    elif normalized.endswith("-hf"):
        source = source[:-2].rstrip()
    alias = slugify(source)
    # Quantization and file-format suffixes do not belong in a user-facing alias.
    alias = re.sub(
        r"-(?:ud-)?(?:q\d+(?:-[a-z0-9]+)*|iq\d+(?:-[a-z0-9]+)*|f(?:16|32)|bf16)$",
        "",
        alias,
    )
    return alias or slugify(path.stem)


def discover() -> list[dict[str, object]]:
    if not MODELS_DIR.is_dir():
        return []

    candidates = []
    for path in sorted(MODELS_DIR.rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file() or path.suffix.lower() != ".gguf":
            continue
        if path.name.lower().startswith(AUXILIARY_PREFIXES):
            continue
        candidates.append(path)

    used_aliases: set[str] = set()
    used_slugs: set[str] = set()
    records: list[dict[str, object]] = []
    for path in candidates:
        name, metadata = model_display_name(path)
        slug_base = slugify(path.stem)
        slug = slug_base
        number = 2
        while slug in used_slugs:
            slug = f"{slug_base}-{number}"
            number += 1
        used_slugs.add(slug)

        alias_base = simple_alias(name, path, metadata)
        alias = alias_base
        number = 2
        while alias in used_aliases:
            # Keep aliases short and human-readable while remaining deterministic.
            alias = f"{alias_base}-{number}"
            number += 1
        used_aliases.add(alias)
        records.append(
            {
                "slug": slug,
                "alias": alias,
                "name": name,
                "architecture": metadata.get("general.architecture", ""),
                # Keep the standardized models-root path even when it is a symlink.
                "path": str(path.absolute()),
                "size": path.stat().st_size,
            }
        )
    return records


def write_artifacts(records: list[dict[str, object]]) -> None:
    scripts_dir = GENERATED_DIR / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    run_script = scripts_dir / "run-model.sh"
    run_script.write_text(
        "#!/usr/bin/env bash\n"
        'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
        'exec "$SCRIPT_DIR/../../manage-services.sh" switch "${1:?usage: run-model.sh MODEL}" "${@:2}"\n',
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    for path in scripts_dir.glob("start-*.sh"):
        path.unlink()
    # Older versions generated one service and one launcher per model.
    # Remove those stale artifacts; the manager now renders one live unit.
    old_services_dir = GENERATED_DIR / "services"
    if old_services_dir.is_dir():
        for path in old_services_dir.glob("*.service"):
            path.unlink()
        try:
            old_services_dir.rmdir()
        except OSError:
            pass

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    (GENERATED_DIR / "models.json").write_text(
        json.dumps({"models_dir": str(MODELS_DIR), "models": records}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    records = discover()
    write_artifacts(records)
    if not args.quiet:
        print(f"Discovered {len(records)} model(s) under {MODELS_DIR}")
        for record in records:
            print(f"  {record['slug']}: {record['path']}")
        print(f"Registry: {GENERATED_DIR / 'models.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
