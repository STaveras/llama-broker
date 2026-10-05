"""Disk hygiene for broker data: find what nothing references and remove it.

A file counts as referenced when a profile pins it, when it sits inside a
directory a profile pins, or when its absolute path appears anywhere in the
live profile manifest, runtime catalog, provenance index, or a Strata
manifest/launch config.  Everything else under the broker's model and runtime
roots is a candidate.

Candidates are deleted.  When an archive directory is given, model files and
manifest backups are archived instead: copied to a directory on another
filesystem, verified by SHA-256, and only then removed.  Orphaned runtime
bundles and package caches are always deleted, and caches only when asked for
by name.

Nothing here runs without an explicit apply; :func:`build_plan` only reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .broker_profiles import ProfileError, ProfileRegistry


SCHEMA_VERSION = 1
CATEGORIES = ("models", "runtimes", "backups", "caches")
DEFAULT_CATEGORIES = ("models", "runtimes", "backups")
ARCHIVE_ACTION = "archive"
DELETE_ACTION = "delete"
# Only these are worth a copy; runtimes rebuild and caches refill.
ARCHIVABLE_CATEGORIES = frozenset({"models", "backups"})
# Manifests whose backups accumulate next to the live file.
BACKUP_BASENAMES = ("profiles.json", "state.json", "catalog.json")
BUNDLE_STEM_MARKER = "runtime-bundle"
WSL_COMPACT_THRESHOLD_BYTES = 5 * 1024**3
_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w.~-])(/[^\s\"'=,;]+)")
_GGUF_SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")
_HASH_CHUNK = 8 * 1024 * 1024


class HygieneError(ValueError):
    """The cleanup plan cannot be built or applied safely."""


@dataclass(frozen=True)
class Candidate:
    category: str
    path: Path
    root: Path
    size_bytes: int
    action: str
    reason: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "category": self.category,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "action": self.action,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Skipped:
    category: str
    path: Path
    reason: str

    def as_mapping(self) -> dict[str, object]:
        return {"category": self.category, "path": str(self.path), "reason": self.reason}


@dataclass(frozen=True)
class CleanRoots:
    """Where each category looks; injectable so tests never touch real data."""

    profiles_path: Path
    catalog_path: Path
    provenance_path: Path
    models_dir: Path
    runtimes_dir: Path
    config_dir: Path
    cache_home: Path
    home: Path


@dataclass(frozen=True)
class CleanOptions:
    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    min_size_bytes: int = 256 * 1024 * 1024
    min_age_seconds: float = 7 * 86400.0
    keep_backups: int = 10
    keep_paths: tuple[Path, ...] = ()
    archive: bool = False

    def action_for(self, category: str) -> str:
        if self.archive and category in ARCHIVABLE_CATEGORIES:
            return ARCHIVE_ACTION
        return DELETE_ACTION


@dataclass(frozen=True)
class CleanPlan:
    candidates: tuple[Candidate, ...]
    skipped: tuple[Skipped, ...]
    references: frozenset[Path] = field(repr=False, default=frozenset())

    def total_bytes(self, action: str) -> int:
        return sum(c.size_bytes for c in self.candidates if c.action == action)


@dataclass
class ApplyResult:
    archived: list[dict[str, object]] = field(default_factory=list)
    deleted: list[dict[str, object]] = field(default_factory=list)
    failed: list[dict[str, object]] = field(default_factory=list)
    skipped: list[dict[str, object]] = field(default_factory=list)
    run_dir: Path | None = None

    def as_mapping(self) -> dict[str, object]:
        return {
            "archive_run_dir": str(self.run_dir) if self.run_dir else None,
            "archived": self.archived,
            "deleted": self.deleted,
            "failed": self.failed,
            "skipped": self.skipped,
        }


# ── References ──────────────────────────────────────────────────────────────

def _normal(path: Path | str) -> Path:
    return Path(os.path.normpath(str(path)))


def _with_resolved(paths: Iterable[Path]) -> set[Path]:
    result: set[Path] = set()
    for path in paths:
        if not Path(path).is_absolute():
            continue
        result.add(_normal(path))
        result.add(Path(os.path.realpath(path)))
    return result


def _paths_in_document(path: Path | None) -> set[Path]:
    """Every absolute path mentioned anywhere in a JSON document."""
    if path is None or not path.is_file():
        return set()
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HygieneError(f"cannot read {path} to find referenced files: {error}") from error
    found: set[Path] = set()
    stack: list[Any] = [decoded]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str):
            found.update(Path(match) for match in _ABSOLUTE_PATH_RE.findall(value))
    return found


def _split_gguf_siblings(path: Path) -> set[Path]:
    """llama.cpp loads every shard of a split GGUF from the first one's path."""
    match = _GGUF_SPLIT_RE.match(path.name)
    if match is None:
        return set()
    count = int(match["count"])
    return {
        path.with_name(f"{match['stem']}-{index:05d}-of-{count:05d}.gguf")
        for index in range(1, count + 1)
    }


def collect_references(roots: CleanRoots, keep_paths: Sequence[Path] = ()) -> frozenset[Path]:
    """Return every path a live profile or runtime document refers to."""
    try:
        registry = ProfileRegistry.from_path(roots.profiles_path, allow_missing_artifacts=True)
    except ProfileError as error:
        raise HygieneError(
            f"cannot tell which files are in use because the profiles do not load: {error}"
        ) from error
    found: set[Path] = set(keep_paths)
    documents: list[Path | None] = [roots.profiles_path, roots.catalog_path, roots.provenance_path]
    for profile in registry.profiles:
        for value in (
            profile.server_binary,
            profile.model_path,
            profile.mmproj_path,
            profile.model_draft,
            profile.effective_model_draft,
            profile.bundle_root,
            profile.source_repository,
            profile.provenance_file,
            profile.strata_artifacts_manifest_path,
            profile.strata_launch_config_path,
            profile.strata_engine_binary,
        ):
            if value is not None:
                found.add(Path(value))
        found.update(Path(path) for path, _digest in profile.strata_artifact_files)
        documents.extend((profile.strata_artifacts_manifest_path, profile.strata_launch_config_path))
    for document in documents:
        found |= _paths_in_document(Path(document) if document is not None else None)
    for path in list(found):
        found |= _split_gguf_siblings(path)
    return frozenset(_with_resolved(found))


def _protected(path: Path, references: frozenset[Path], ancestors: frozenset[Path]) -> bool:
    """True when the path is referenced, inside a reference, or contains one."""
    if path in ancestors:
        return True
    return any(candidate in references for candidate in (path, *path.parents))


def _ancestors_of(references: frozenset[Path]) -> frozenset[Path]:
    return frozenset(parent for reference in references for parent in reference.parents)


# ── Tree measurements ───────────────────────────────────────────────────────

def _walk_entries(path: Path) -> Iterator[tuple[Path, os.stat_result]]:
    yield path, path.lstat()
    if path.is_dir() and not path.is_symlink():
        for directory, dirnames, filenames in os.walk(path, followlinks=False):
            for name in (*dirnames, *filenames):
                child = Path(directory) / name
                try:
                    yield child, child.lstat()
                except FileNotFoundError:
                    continue


def tree_stats(path: Path, *, exclusive_only: bool = False) -> tuple[int, float]:
    """Return (bytes, newest mtime) counting each inode once.

    With ``exclusive_only`` only files with a single hard link count, which is
    what deleting the tree actually frees (uv's cache, for one, hard-links
    into virtualenvs).
    """
    seen: set[tuple[int, int]] = set()
    total = 0
    newest = 0.0
    for _entry, info in _walk_entries(path):
        newest = max(newest, info.st_mtime)
        if not stat.S_ISREG(info.st_mode):
            continue
        inode = (info.st_dev, info.st_ino)
        if inode in seen:
            continue
        seen.add(inode)
        if exclusive_only and info.st_nlink > 1:
            continue
        total += info.st_size
    return total, newest


def _age_days(newest: float, now: float) -> float:
    return max(0.0, (now - newest) / 86400.0)


# ── Candidate discovery ─────────────────────────────────────────────────────

def _unreferenced_entries(root: Path, references: frozenset[Path]) -> list[Path]:
    """Largest subtrees under ``root`` that contain nothing referenced."""
    root = Path(os.path.realpath(root))
    if not root.is_dir():
        return []
    # A reference to the root itself or above it (e.g. a catalog that records
    # the model store location) says nothing about the files inside it.
    scoped = frozenset(ref for ref in references if root in ref.parents)
    ancestors = _ancestors_of(scoped)
    found: list[Path] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(directory.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_symlink() or child in scoped:
                continue
            if child in ancestors and child.is_dir():
                stack.append(child)
                continue
            found.append(child)
    return found


def _running_executables() -> tuple[Path, ...]:
    executables: list[Path] = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            executables.append(Path(os.readlink(proc / "exe")))
        except OSError:
            continue
    return tuple(executables)


def _bundle_stem(name: str) -> str | None:
    index = name.find(BUNDLE_STEM_MARKER)
    if index < 0:
        return None
    return name[: index + len(BUNDLE_STEM_MARKER)]


def _runtime_bundle_paths(roots: CleanRoots, references: frozenset[Path]) -> list[Path]:
    """Runtime directories nothing references, plus orphaned sibling bundles."""
    runtimes_dir = Path(os.path.realpath(roots.runtimes_dir))
    paths = [path for path in _unreferenced_entries(runtimes_dir, references) if path.is_dir()]
    paths = [path for path in paths if path.parent == runtimes_dir]
    # Bundles kept outside the runtime store (Strata's are built under
    # ~/builds) leave superseded siblings behind with the same name stem.
    registered = {
        ref for ref in references
        if ref.is_dir() and _bundle_stem(ref.name) and runtimes_dir not in ref.parents
    }
    ancestors = _ancestors_of(references)
    for bundle in sorted(registered):
        stem = _bundle_stem(bundle.name)
        assert stem is not None
        try:
            siblings = sorted(bundle.parent.iterdir())
        except OSError:
            continue
        for sibling in siblings:
            if (
                sibling.is_dir()
                and not sibling.is_symlink()
                and sibling.name.startswith(stem)
                and (sibling / "bin").is_dir()
                and sibling not in paths
                and not _protected(sibling, references, ancestors)
            ):
                paths.append(sibling)
    return paths


def _backup_paths(roots: CleanRoots) -> list[tuple[str, list[Path]]]:
    groups: list[tuple[str, list[Path]]] = []
    for directory in (roots.config_dir, roots.runtimes_dir):
        for base in BACKUP_BASENAMES:
            matches = [
                path for path in directory.glob(f"{base}.*")
                if path.is_file()
                and not path.is_symlink()
                and not path.name.endswith((".tmp", ".lock"))
            ]
            if matches:
                matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
                groups.append((str(directory / base), matches))
    return groups


def _cache_paths(roots: CleanRoots) -> list[tuple[str, Path]]:
    entries = [
        ("uv package cache", roots.cache_home / "uv"),
        ("pip cache", roots.cache_home / "pip"),
        ("npm cache", roots.home / ".npm" / "_cacache"),
        ("npx cache", roots.home / ".npm" / "_npx"),
        ("ccache", roots.cache_home / "ccache"),
        ("ccache", roots.home / ".ccache"),
    ]
    # Model snapshots in the Hugging Face cache can back vLLM profiles; only
    # downloaded datasets are treated as disposable.
    hub = roots.cache_home / "huggingface" / "hub"
    if hub.is_dir():
        entries.extend(("Hugging Face dataset", path) for path in sorted(hub.glob("datasets--*")))
    return [(label, path) for label, path in entries if path.is_dir() and not path.is_symlink()]


def build_plan(
    roots: CleanRoots,
    options: CleanOptions,
    *,
    now: float | None = None,
    running_executables: Callable[[], Sequence[Path]] = _running_executables,
) -> CleanPlan:
    """Find cleanup candidates without changing anything."""
    for category in options.categories:
        if category not in CATEGORIES:
            raise HygieneError(f"unknown clean category {category!r}")
    now = time.time() if now is None else now
    references = collect_references(roots, options.keep_paths)
    candidates: list[Candidate] = []
    skipped: list[Skipped] = []

    def too_recent(category: str, path: Path, newest: float) -> bool:
        if now - newest >= options.min_age_seconds:
            return False
        skipped.append(Skipped(
            category, path, f"modified {_age_days(newest, now):.1f} days ago"
        ))
        return True

    if "models" in options.categories:
        models_root = Path(os.path.realpath(roots.models_dir))
        for path in _unreferenced_entries(models_root, references):
            size, newest = tree_stats(path)
            if size < options.min_size_bytes or too_recent("models", path, newest):
                continue
            candidates.append(Candidate(
                "models", path, models_root, size, options.action_for("models"),
                "no profile, catalog or launch config references it",
            ))

    if "runtimes" in options.categories:
        executables = tuple(_normal(path) for path in running_executables())
        for path in _runtime_bundle_paths(roots, references):
            size, newest = tree_stats(path)
            if too_recent("runtimes", path, newest):
                continue
            if any(path in executable.parents for executable in executables):
                skipped.append(Skipped("runtimes", path, "a running process executes from it"))
                continue
            candidates.append(Candidate(
                "runtimes", path, path.parent, size, options.action_for("runtimes"),
                "orphaned runtime bundle: not in the catalog or any profile",
            ))

    if "backups" in options.categories:
        for label, paths in _backup_paths(roots):
            for path in paths[options.keep_backups:]:
                if path in references:
                    continue
                size, newest = tree_stats(path)
                if too_recent("backups", path, newest):
                    continue
                candidates.append(Candidate(
                    "backups", path, path.parent, size, options.action_for("backups"),
                    f"older than the newest {options.keep_backups} backups of {Path(label).name}",
                ))

    if "caches" in options.categories:
        for label, path in _cache_paths(roots):
            size, _newest = tree_stats(path, exclusive_only=True)
            if size == 0:
                continue
            candidates.append(Candidate(
                "caches", path, roots.home, size, options.action_for("caches"),
                f"{label}; regenerates on demand",
            ))

    return CleanPlan(tuple(candidates), tuple(skipped), references)


def still_unreferenced(roots: CleanRoots, options: CleanOptions) -> Callable[[Candidate], bool]:
    """Re-read the live manifests right before removing each candidate."""

    def check(candidate: Candidate) -> bool:
        if candidate.category == "caches":
            return True
        references = collect_references(roots, options.keep_paths)
        return not _protected(candidate.path, references, _ancestors_of(references))

    return check


# ── Archiving ───────────────────────────────────────────────────────────────

class _HashingReader:
    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self._handle.read(size)
        self.digest.update(chunk)
        return chunk


def _sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(_HASH_CHUNK):
        digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(info: os.stat_result) -> tuple[int, int]:
    return info.st_size, info.st_mtime_ns


def _archive_file(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("xb") as writer:
        while chunk := reader.read(_HASH_CHUNK):
            digest.update(chunk)
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    shutil.copystat(source, destination)
    if _fingerprint(source.stat()) != _fingerprint(before):
        raise HygieneError(f"{source} changed while it was being archived")
    with destination.open("rb") as reader:
        copied = _sha256_stream(reader)
    if copied != digest.hexdigest():
        raise HygieneError(f"archived copy of {source} does not match the original")
    return {"kind": "file", "sha256": copied}


def _archive_tree(source: Path, destination: Path) -> dict[str, object]:
    """Write ``source`` to an uncompressed tar, then prove the tar matches it.

    A tar keeps symlinks, hard links and modes intact on filesystems (NTFS
    under WSL) that cannot represent them.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected: dict[str, tuple[str, str, tuple[int, int]]] = {}
    with destination.open("xb") as handle, tarfile.open(fileobj=handle, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path, info in _walk_entries(source):
            name = (Path(source.name) / path.relative_to(source)).as_posix()
            member = archive.gettarinfo(str(path), arcname=name)
            if member is None:
                raise HygieneError(f"{path} is a socket or device and cannot be archived")
            if member.isreg():
                with path.open("rb") as reader:
                    hashed = _HashingReader(reader)
                    archive.addfile(member, hashed)
                expected[name] = ("file", hashed.digest.hexdigest(), _fingerprint(info))
            else:
                archive.addfile(member)
                if member.issym():
                    expected[name] = ("symlink", member.linkname, (0, 0))
                elif member.islnk():
                    expected[name] = ("hardlink", member.linkname, (0, 0))
                else:
                    expected[name] = ("dir", "", (0, 0))
        handle.flush()
        os.fsync(handle.fileno())
    for path, info in _walk_entries(source):
        name = (Path(source.name) / path.relative_to(source)).as_posix()
        kind, _value, fingerprint = expected.get(name, ("missing", "", (0, 0)))
        if kind == "missing" or (kind == "file" and _fingerprint(info) != fingerprint):
            raise HygieneError(f"{source} changed while it was being archived")
    seen: set[str] = set()
    with tarfile.open(destination, mode="r:") as archive:
        for member in archive:
            kind, value, _fingerprint_ = expected.get(member.name, ("missing", "", (0, 0)))
            if member.isreg():
                extracted = archive.extractfile(member)
                assert extracted is not None
                ok = kind == "file" and _sha256_stream(extracted) == value
            elif member.issym():
                ok = kind == "symlink" and member.linkname == value
            elif member.islnk():
                ok = kind == "hardlink" and member.linkname == value
            else:
                ok = kind == "dir"
            if not ok:
                raise HygieneError(f"archive of {source} does not match the original at {member.name}")
            seen.add(member.name)
    if seen != set(expected):
        raise HygieneError(f"archive of {source} is missing entries")
    return {"kind": "tar", "members": len(seen)}


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _write_manifest(run_dir: Path, entries: list[dict[str, object]]) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "restore": "copy a file back to original_path, or `tar -xf ARCHIVE -C PARENT` for a .tar",
        "entries": entries,
    }
    temporary = run_dir / "MANIFEST.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, run_dir / "MANIFEST.json")


def apply_plan(
    plan: CleanPlan,
    *,
    archive_dir: Path | None,
    still_unreferenced: Callable[[Candidate], bool],
    now: float | None = None,
    require_other_filesystem: bool = True,
    log: Callable[[str], None] = lambda _message: None,
) -> ApplyResult:
    """Archive or delete each candidate; a failure leaves that original intact."""
    result = ApplyResult()
    to_archive = [c for c in plan.candidates if c.action == ARCHIVE_ACTION]
    run_dir: Path | None = None
    if to_archive:
        if archive_dir is None:
            raise HygieneError("the plan archives files but no archive directory was given")
        archive_dir = Path(archive_dir).expanduser()
        if not archive_dir.is_absolute() or not archive_dir.is_dir():
            raise HygieneError(f"archive directory must be an existing absolute directory: {archive_dir}")
        needed = sum(c.size_bytes for c in to_archive)
        free = shutil.disk_usage(archive_dir).free
        if free < needed + 256 * 1024 * 1024:
            raise HygieneError(
                f"archive directory has {free / 1024**3:.1f} GiB free; "
                f"{needed / 1024**3:.1f} GiB is needed"
            )
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(now))
        run_dir = archive_dir / f"brokerctl-clean-{stamp}"
        run_dir.mkdir(exist_ok=False)
        result.run_dir = run_dir
    manifest: list[dict[str, object]] = []

    for candidate in plan.candidates:
        record: dict[str, object] = candidate.as_mapping()
        try:
            if not candidate.path.exists():
                result.skipped.append({**record, "reason": "already gone"})
                continue
            if not still_unreferenced(candidate):
                result.skipped.append({**record, "reason": "a profile now references it"})
                continue
            if candidate.action == DELETE_ACTION:
                log(f"deleting {candidate.path}")
                _remove(candidate.path)
                result.deleted.append(record)
                continue
            assert run_dir is not None
            if require_other_filesystem and candidate.path.stat().st_dev == run_dir.stat().st_dev:
                raise HygieneError("archive directory is on the same filesystem, so archiving frees nothing")
            relative = Path(candidate.category) / candidate.path.relative_to(candidate.root)
            if candidate.path.is_dir():
                destination = run_dir / relative.with_name(relative.name + ".tar")
                log(f"archiving {candidate.path} -> {destination}")
                details = _archive_tree(candidate.path, destination)
            else:
                destination = run_dir / relative
                log(f"archiving {candidate.path} -> {destination}")
                details = _archive_file(candidate.path, destination)
            # The copy is verified; recheck references once more, then remove.
            if not still_unreferenced(candidate):
                result.skipped.append({**record, "reason": "a profile now references it; archive kept, original kept"})
                continue
            _remove(candidate.path)
            entry = {**record, **details, "original_path": str(candidate.path), "archive_path": str(destination)}
            manifest.append(entry)
            _write_manifest(run_dir, manifest)
            result.archived.append(entry)
        except (OSError, HygieneError, tarfile.TarError) as error:
            result.failed.append({**record, "error": str(error)})
    return result


# ── WSL virtual disk ────────────────────────────────────────────────────────

def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="ascii", errors="ignore").lower()
    except OSError:
        return False


def parse_lxss_base_path(reg_output: str, distribution: str) -> str | None:
    """Find a distribution's BasePath in `reg.exe query ...\\Lxss /s` output."""
    blocks = re.split(r"\r?\n(?=HKEY_)", reg_output)
    for block in blocks:
        values = dict(re.findall(r"^\s+(\w+)\s+REG_\w+\s+(.*?)\s*$", block, flags=re.MULTILINE))
        if values.get("DistributionName", "").lower() == distribution.lower() and "BasePath" in values:
            return values["BasePath"].removeprefix("\\\\?\\")
    return None


def windows_to_wsl_path(path: str) -> Path | None:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if match is None:
        return None
    return Path("/mnt") / match[1].lower() / match[2].replace("\\", "/")


def wsl_disk_report(*, timeout: float = 10.0) -> dict[str, object] | None:
    """Compare this distribution's ext4.vhdx with what its filesystem uses."""
    if not is_wsl():
        return None
    distribution = os.environ.get("WSL_DISTRO_NAME", "")
    report: dict[str, object] = {"distribution": distribution or None}
    try:
        completed = subprocess.run(
            ["reg.exe", "query", r"HKCU\Software\Microsoft\Windows\CurrentVersion\Lxss", "/s"],
            capture_output=True, text=True, timeout=timeout, check=False, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {**report, "error": f"cannot query WSL registration: {error}"}
    base = parse_lxss_base_path(completed.stdout, distribution) if distribution else None
    if base is None:
        return {**report, "error": "cannot find this distribution's virtual disk"}
    windows_vhdx = base.rstrip("\\") + "\\ext4.vhdx"
    vhdx = windows_to_wsl_path(windows_vhdx)
    try:
        if vhdx is None:
            raise OSError(f"unsupported Windows path {windows_vhdx}")
        info = vhdx.stat()
        filesystem = os.statvfs("/")
    except OSError as error:
        return {**report, "vhdx_path": windows_vhdx, "error": str(error)}
    allocated = info.st_blocks * 512 if info.st_blocks else info.st_size
    vhdx_bytes = min(info.st_size, allocated)
    used = (filesystem.f_blocks - filesystem.f_bfree) * filesystem.f_frsize
    return {
        **report,
        "vhdx_path": windows_vhdx,
        "vhdx_bytes": vhdx_bytes,
        "filesystem_used_bytes": used,
        "reclaimable_bytes": max(0, vhdx_bytes - used),
    }


def wsl_compaction_steps(vhdx_path: str) -> tuple[str, ...]:
    return (
        "sudo fstrim -v /",
        "In an administrator PowerShell (this stops WSL and every model):",
        f'  wsl --shutdown; "select vdisk file=`"{vhdx_path}`"`nattach vdisk readonly`n'
        'compact vdisk`ndetach vdisk" | Set-Content "$env:TEMP\\compact-wsl.txt"; '
        'diskpart /s "$env:TEMP\\compact-wsl.txt"',
    )
