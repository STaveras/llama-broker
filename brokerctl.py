#!/usr/bin/env python3
"""Checkout entry point for the Llama Broker control interface."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llama_broker.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())