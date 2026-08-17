#!/usr/bin/env python3
"""Run the hardware-independent release suite without host-only manifests."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        if path.name == "test_local_production_manifest.py":
            continue
        module = importlib.import_module(f"tests.{path.stem}")
        suite.addTests(loader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
