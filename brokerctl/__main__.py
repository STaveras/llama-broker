#!/usr/bin/env python3
"""Package mode entry point for brokerctl.

Usage:
    python3 -m brokerctl [command] [args...]
    python3 -m brokerctl model list
    python3 -m brokerctl profiles discover
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import brokerctl.py
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Import main from brokerctl.py (one directory up)
from brokerctl import main

raise SystemExit(main())
