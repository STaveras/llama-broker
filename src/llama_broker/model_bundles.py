"""Read-only local model listing. Bundle roots are display data, not action targets.

Weights are never read or hashed. Snapshot symlinks retain their semantic names;
sizes count each (device, inode) once within a bundle, including its metadata.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


_SPLIT = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.(?P<format>gguf|safetensors)$"
)
_INDEX_SUFFIX = ".safetensors.index.json"
_PARTIAL_SUFFIXES = (".incomplete", ".partial", ".part", ".tmp", ".download", ".lock")
_METADATA_NAMES = {
    "config.json", "generation_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
    "preprocessor_config.json", "processor_config.json", "quantize_config.json",
    "quant_config.json", "quantization_config.json", "chat_template.json",
}


def _excluded(name: str) -> bool:
    return name.startswith(".") or name.casefold().endswith(_PARTIAL_SUFFIXES)


def _file_stat(path: Path) -> os.stat_result | None:
    try:
        # HF cache entries point outside snapshots, into blobs. Reject partial
        # blob targets but retain the snapshot path in all displayed records.
        if _excluded(path.name) or any(
            part.casefold().endswith(_PARTIAL_SUFFIXES) for part in path.resolve().parts
        ):
            return None
        result = path.stat()
        if stat.S_ISREG(result.st_mode) and os.access(path, os.R_OK):
            return result
    except (OSError, RuntimeError, ValueError):
        pass
    return None


def _json_object(path: Path) -> dict[str, Any] | None:
    if _file_stat(path) is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _has_model_config(path: Path) -> bool:
    config = _json_object(path / "config.json") or {}
    model_type = config.get("model_type")
    architectures = config.get("architectures")
    return (isinstance(model_type, str) and bool(model_type.strip())) or (
        isinstance(architectures, list) and bool(architectures)
        and all(isinstance(name, str) and bool(name.strip()) for name in architectures)
    )


def _directories(paths: Iterable[Path]) -> dict[Path, tuple[Path, ...]]:
    roots: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser().absolute()
        try:
            roots.add((path.parent if path.is_file() else path).resolve())
        except (OSError, RuntimeError):
            continue
    directories: dict[Path, tuple[Path, ...]] = {}
    seen: set[tuple[int, int]] = set()
    for root in sorted(roots):
        for directory, children, filenames in os.walk(root, followlinks=True):
            try:
                current = Path(directory).resolve()
                info = current.stat()
                identity = (info.st_dev, info.st_ino)
                if (
                    identity in seen or current in root.parents
                    or any(part.casefold().endswith(_PARTIAL_SUFFIXES) for part in current.parts)
                ):
                    children[:] = []
                    continue
                seen.add(identity)
                children[:] = sorted(
                    name for name in children
                    if not _excluded(name) and name not in {"blobs", "refs"}
                )
                directories[current] = tuple(
                    current / name for name in sorted(filenames) if not _excluded(name)
                )
            except (OSError, RuntimeError):
                children[:] = []
    return directories


def _metadata(path: Path) -> bool:
    name = path.name.casefold()
    return (
        name in _METADATA_NAMES
        or name.startswith(("tokenizer", "chat_template", "license", "readme"))
        and path.suffix.casefold() in {".json", ".model", ".tiktoken", ".jinja", ".txt", ".md", ""}
    )


def _split_names(path: Path) -> list[str]:
    match = _SPLIT.fullmatch(path.name)
    if match and 0 < int(match["index"]) <= int(match["count"]):
        return [
            f"{match['stem']}-{index:05d}-of-{int(match['count']):05d}.{match['format']}"
            for index in range(1, int(match["count"]) + 1)
        ]
    return [path.name]


def _family(path: Path) -> str:
    match = _SPLIT.fullmatch(path.name)
    return match["stem"] if match else path.stem


def _bundle(
    root: Path,
    *,
    format: str,
    name: str,
    weights: Iterable[str],
    metadata: Iterable[Path] = (),
    index: Path | None = None,
    issues: Iterable[str] = (),
) -> dict[str, Any]:
    roles = {path: "metadata" for path in metadata}
    expected = sorted(set(weights))
    roles.update({root / name: "weight" for name in expected})
    if index is not None:
        roles[index] = "index"
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    physical: set[tuple[int, int]] = set()
    total = 0
    for path, role in sorted(roles.items()):
        info = _file_stat(path)
        if info is None or (role == "weight" and info.st_size == 0):
            if role in {"weight", "index"}:
                missing.append(path.relative_to(root).as_posix())
            continue
        files.append({
            "path": str(path), "relative_path": path.relative_to(root).as_posix(),
            "role": role, "size": info.st_size,
        })
        identity = (info.st_dev, info.st_ino)
        if identity not in physical:
            physical.add(identity)
            total += info.st_size
    issues = sorted(set(issues))
    present_weights = [item["path"] for item in files if item["role"] == "weight"]
    return {
        "name": name,
        "path": present_weights[0] if present_weights else str(index or root / expected[0]),
        "bundle_root": str(root),
        "format": format,
        "status": "incomplete" if missing or issues or not present_weights else "complete",
        "files": files,
        "size": total,
        "size_policy": "unique-inode-per-bundle",
        "missing_files": sorted(missing),
        "issues": issues,
    }


def discover_model_bundles(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Find HF bundles with config/index evidence and independent GGUF sets.

    Completeness checks file presence only, never loadability or integrity.
    Distinct directories, indexes, weight variants and GGUF split stems remain
    distinct, even when they share physical blobs or identical file contents.
    """
    directories = _directories(paths)
    config_roots = {root for root in directories if _has_model_config(root)}
    model_roots = config_roots | {
        root for root, files in directories.items()
        if any(path.name.endswith(_INDEX_SUFFIX) for path in files)
    }
    bundles: list[dict[str, Any]] = []
    for root, files in sorted(directories.items()):
        # GGUF has its own split convention; HF config never absorbs GGUFs.
        seen_gguf: set[str] = set()
        for path in files:
            if (
                path.suffix.casefold() != ".gguf" or path.name in seen_gguf
                or _file_stat(path) is None
            ):
                continue
            names = _split_names(path)
            seen_gguf.update(names)
            bundles.append(_bundle(root, format="gguf", name=_family(path), weights=names))
        if root not in model_roots:
            continue
        metadata = [path for path in files if _metadata(path)]
        indexes = [path for path in files if path.name.endswith(_INDEX_SUFFIX)]
        indexed_families: set[str] = set()
        claimed: set[str] = set()
        for index in indexes:
            family = index.name.removesuffix(_INDEX_SUFFIX)
            indexed_families.add(family)
            payload = _json_object(index)
            weight_map = payload.get("weight_map") if payload is not None else None
            issues: list[str] = []
            names: set[str] = set()
            if not isinstance(weight_map, dict) or not weight_map:
                issues.append(f"{index.name}: invalid or empty weight_map")
            else:
                for value in weight_map.values():
                    relative = PurePosixPath(value) if isinstance(value, str) else None
                    if (
                        relative is None or relative.is_absolute() or "\\" in value
                        or "\x00" in value
                        or any(part == ".." or _excluded(part) for part in relative.parts)
                        or relative.suffix != ".safetensors"
                    ):
                        issues.append(f"{index.name}: invalid weight path {value!r}")
                        continue
                    member = root / relative
                    # An index cannot claim a separate model's files, including
                    # a directory alias pointing at that model.
                    try:
                        parent = member.parent.resolve()
                    except (OSError, RuntimeError, ValueError):
                        issues.append(f"{index.name}: cannot resolve weight path: {value}")
                        continue
                    if any(
                        boundary != root and boundary not in root.parents
                        and (boundary == parent or boundary in parent.parents)
                        for boundary in model_roots
                    ):
                        issues.append(f"{index.name}: weight crosses model boundary: {value}")
                        continue
                    names.add(relative.as_posix())
            claimed.update(names)
            bundles.append(_bundle(
                root, format="hf-safetensors",
                name=root.name if family == "model" else f"{root.name}/{family}",
                weights=names, metadata=metadata, index=index, issues=issues,
            ))
        if root not in config_roots:
            continue
        seen_weights = set(claimed)
        for path in files:
            if (
                path.suffix != ".safetensors" or path.name in seen_weights
                or _family(path) in indexed_families or _file_stat(path) is None
            ):
                continue
            names = _split_names(path)
            seen_weights.update(names)
            family = _family(path)
            bundles.append(_bundle(
                root, format="hf-safetensors",
                name=root.name if family == "model" else f"{root.name}/{family}",
                weights=names, metadata=metadata,
            ))
    return sorted(bundles, key=lambda bundle: (bundle["bundle_root"], bundle["path"], bundle["name"]))
