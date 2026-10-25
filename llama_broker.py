#!/usr/bin/env python3
"""Checkout entry point for direct Llama Broker daemon debugging.

When imported from an uninstalled checkout, this launcher also exposes the
package submodules so the source tree remains usable without an editable
installation. Installed users import the real ``src/llama_broker`` package.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version as _installed_version
import sys
import tomllib
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
_PACKAGE_DIR = _SRC / "llama_broker"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if __name__ == "__main__":
    from llama_broker.daemon import main

    raise SystemExit(main())

# The checkout file name necessarily shadows the src-layout package when the
# repository root is first on sys.path. Mark this module as a package and
# forward its public symbols to the implementation package so imports such as
# ``import llama_broker`` continue to work in an uninstalled checkout.
__path__ = [str(_PACKAGE_DIR)]

try:
    __version__ = _installed_version("llama-broker")
except PackageNotFoundError:
    with (_SRC.parent / "pyproject.toml").open("rb") as _handle:
        __version__ = str(tomllib.load(_handle)["project"]["version"])

_daemon = importlib.import_module(f"{__name__}.daemon")
for _name, _value in vars(_daemon).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
