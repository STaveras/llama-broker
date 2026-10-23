"""Llama Broker runtime package.

The daemon's public and legacy module-level symbols are re-exported here so
existing imports such as ``from llama_broker import BrokerApplication`` remain
valid after the source-layout migration.
"""

from importlib.metadata import PackageNotFoundError, version as _installed_version
from pathlib import Path
import tomllib


def _package_version() -> str:
    try:
        return _installed_version("llama-broker")
    except PackageNotFoundError:
        # Source checkouts have no installed distribution metadata. Read the
        # single canonical value from pyproject.toml instead of duplicating it.
        try:
            project = Path(__file__).resolve().parents[2] / "pyproject.toml"
            with project.open("rb") as handle:
                return str(tomllib.load(handle)["project"]["version"])
        except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
            return "unknown"


__version__ = _package_version()

from . import daemon as _daemon
from .daemon import *  # noqa: F401,F403

# Preserve access to legacy private helpers for integrations that imported the
# old flat ``llama_broker.py`` module directly. New code should import the
# implementation from ``llama_broker.daemon`` instead.
for _name, _value in vars(_daemon).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
