"""Llama Broker runtime package.

The daemon's public and legacy module-level symbols are re-exported here so
existing imports such as ``from llama_broker import BrokerApplication`` remain
valid after the source-layout migration.
"""

from . import daemon as _daemon
from .daemon import *  # noqa: F401,F403

# Preserve access to legacy private helpers for integrations that imported the
# old flat ``llama_broker.py`` module directly. New code should import the
# implementation from ``llama_broker.daemon`` instead.
for _name, _value in vars(_daemon).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
