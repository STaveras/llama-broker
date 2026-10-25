from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.package_strata_bridge import PackageError, package_launcher


class StrataBridgePackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.python = self.bundle / "venv" / "bin" / "python"
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b"interpreter")
        self.python.chmod(0o755)
        self.source = self.root / "strata_bridge.py"
        self.source.write_text("#!/usr/bin/env python3\nprint('bridge')\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_packages_executable_bridge_with_bundle_interpreter_and_hash(self) -> None:
        result = package_launcher(self.bundle, self.python, source_path=self.source)
        target = self.bundle / "bin" / "strata-server"
        source_body = self.source.read_text(encoding="utf-8").split("\n", 1)[1]

        self.assertEqual(target.read_text(encoding="utf-8"), f"#!{self.python}\n{source_body}")
        self.assertTrue(os.access(target, os.X_OK))
        self.assertEqual(result["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual(result["source_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_packaged_entrypoint_runs_with_pinned_bundle_python(self) -> None:
        python_shim = self.bundle / "venv" / "bin" / "python"
        python_shim.write_text(
            "#!/bin/sh\nexec " + shlex.quote(sys.executable) + " \"$@\"\n",
            encoding="utf-8",
        )
        python_shim.chmod(0o755)
        source = Path(__file__).resolve().parents[1] / "src" / "llama_broker" / "runtimes" / "strata_bridge.py"

        result = package_launcher(self.bundle, python_shim, source_path=source)
        process = subprocess.run(
            [result["path"], "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--config-sha256", process.stdout)
        self.assertIn("--min-available-ram-mib", process.stdout)

    def test_refuses_interpreter_outside_bundle_or_launcher_overwrite(self) -> None:
        external = self.root / "python"
        external.write_bytes(b"outside")
        external.chmod(0o755)
        with self.assertRaisesRegex(PackageError, "inside runtime bundle"):
            package_launcher(self.bundle, external, source_path=self.source)
        package_launcher(self.bundle, self.python, source_path=self.source)
        with self.assertRaisesRegex(PackageError, "already exists"):
            package_launcher(self.bundle, self.python, source_path=self.source)


if __name__ == "__main__":
    unittest.main()
