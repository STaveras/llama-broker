"""Package the reviewed Strata bridge as a pinned runtime executable."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


class PackageError(ValueError):
    """The bridge package inputs do not meet the bundle contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_launcher(
    bundle_root: Path | str,
    python_executable: Path | str,
    *,
    source_path: Path | str | None = None,
) -> dict[str, str]:
    root = Path(bundle_root).expanduser()
    interpreter = Path(python_executable).expanduser()
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise PackageError("runtime bundle root must be an absolute non-symlink directory")
    if not interpreter.is_absolute() or interpreter.is_symlink() or not interpreter.is_file():
        raise PackageError("bundle Python must be an absolute regular file")
    interpreter = interpreter.resolve()
    root = root.resolve()
    if not interpreter.is_relative_to(root):
        raise PackageError("bundle Python must be inside runtime bundle")
    if not os.access(interpreter, os.X_OK):
        raise PackageError("bundle Python is not executable")
    if any(char.isspace() for char in str(interpreter)):
        raise PackageError("bundle Python path cannot contain whitespace in a shebang")

    source = (
        Path(source_path).expanduser()
        if source_path is not None
        else Path(__file__).resolve().parents[1]
        / "src"
        / "llama_broker"
        / "runtimes"
        / "strata_bridge.py"
    )
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise PackageError("bridge source must be an absolute regular file")
    body = source.read_text(encoding="utf-8")
    first_line, separator, remainder = body.partition("\n")
    if not separator or not first_line.startswith("#!"):
        raise PackageError("bridge source must begin with a shebang")
    destination = root / "bin" / "strata-server"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
    except FileExistsError as error:
        raise PackageError(f"bridge launcher already exists: {destination}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"#!{interpreter}\n")
            handle.write(remainder)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(destination, 0o755)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {
        "path": str(destination),
        "sha256": _sha256(destination),
        "source_sha256": _sha256(source),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path, help="executable Python inside the bundle")
    parser.add_argument("--source", type=Path, help="bridge source (defaults to the broker's audited bridge)")
    args = parser.parse_args(argv)
    try:
        result = package_launcher(args.bundle_root, args.python, source_path=args.source)
    except (OSError, PackageError) as error:
        print(f"package_strata_bridge: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
