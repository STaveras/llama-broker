from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from tools import safety_ladder


class SafetyLadderPortabilityTests(unittest.TestCase):
    def test_script_entrypoint_resolves_repo_modules(self) -> None:
        script = Path(safety_ladder.__file__).resolve()
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ladder_reaches_full_window_with_template_reserve(self) -> None:
        self.assertEqual(safety_ladder.MAX_TARGET_TOKENS, 262_000)

    def test_gpu_state_is_unknown_without_a_portable_provider(self) -> None:
        with patch.object(
            safety_ladder.subprocess,
            "run",
            side_effect=AssertionError("GPU telemetry must not invoke a vendor utility"),
        ):
            self.assertIsNone(safety_ladder.gpu_state())

    def test_default_fault_collection_has_no_windows_dependency(self) -> None:
        with (
            patch.object(safety_ladder, "linux_gpu_faults", return_value=[]),
            patch.object(
                safety_ladder,
                "windows_live_kernel_reports",
                side_effect=AssertionError("Windows probe must be opt-in"),
            ),
        ):
            self.assertEqual(
                safety_ladder.collect_faults("2026-07-17T12:00:00-04:00"), []
            )
            safety_ladder.linux_gpu_faults.assert_called_once_with(  # type: ignore[attr-defined]
                "2026-07-17T12:00:00-04:00", "example-model-7b"
            )

    def test_windows_probe_can_be_added_explicitly(self) -> None:
        report = {"source": "windows-wer", "message": "WATCHDOG.dmp"}
        with (
            patch.object(safety_ladder, "linux_gpu_faults", return_value=[]),
            patch.object(
                safety_ladder, "windows_live_kernel_reports", return_value=[report]
            ),
        ):
            self.assertEqual(
                safety_ladder.collect_faults(
                    "2026-07-17T12:00:00-04:00", include_windows_wer=True
                ),
                [report],
            )


if __name__ == "__main__":
    unittest.main()
