from __future__ import annotations

import unittest

from llama_broker.crash_loop import CrashLoopGuard


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class CrashLoopGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.guard = CrashLoopGuard(threshold=3, window=600, cooldown=300, clock=self.clock)

    def _crash(self, cause: str = "signal (SIGABRT)") -> bool:
        tripped = self.guard.record_crash("main", cause)
        self.clock.now += 80
        return tripped

    def test_trips_on_threshold_and_refuses_for_the_cooldown(self) -> None:
        self.assertFalse(self._crash())
        self.assertFalse(self._crash())
        self.assertIsNone(self.guard.check("main"))
        self.assertTrue(self.guard.record_crash("main", "signal (SIGABRT): CUDA error"))

        rejection = self.guard.check("main")
        assert rejection is not None
        self.assertEqual(rejection.crashes, 3)
        self.assertEqual(rejection.retry_after, 300)
        self.assertEqual(rejection.last_cause, "signal (SIGABRT): CUDA error")
        self.assertIn("crashed 3 times", rejection.message())
        self.assertIsNone(self.guard.check("other"))

        self.clock.now += 299.5
        self.assertEqual(self.guard.check("main").retry_after, 1)  # type: ignore[union-attr]

    def test_crashes_outside_the_window_do_not_count(self) -> None:
        self._crash()
        self._crash()
        self.clock.now += 600
        self.assertFalse(self._crash())
        self.assertIsNone(self.guard.check("main"))

    def test_one_crash_on_probation_trips_again(self) -> None:
        for _ in range(3):
            self._crash()
        self.clock.now += 300
        self.assertIsNone(self.guard.check("main"))
        self.assertTrue(self.guard.snapshot()["main"]["probation"])
        self.clock.now += 1000

        self.assertTrue(self._crash())
        self.assertIsNotNone(self.guard.check("main"))

    def test_success_and_clear_forget_history(self) -> None:
        self._crash()
        self._crash()
        self.guard.record_success("main")
        self.assertFalse(self._crash())

        for _ in range(2):
            self._crash()
        self.assertIsNotNone(self.guard.check("main"))
        self.guard.clear("main")
        self.assertIsNone(self.guard.check("main"))

        self.guard.record_crash("a", "x")
        self.guard.record_crash("b", "y")
        self.guard.clear()
        self.assertEqual(self.guard.snapshot(), {})

    def test_zero_threshold_disables_the_guard(self) -> None:
        guard = CrashLoopGuard(threshold=0, window=600, cooldown=300, clock=self.clock)
        for _ in range(10):
            self.assertFalse(guard.record_crash("main", "x"))
        self.assertIsNone(guard.check("main"))
        self.assertEqual(guard.snapshot(), {})

    def test_snapshot_reports_tripped_state(self) -> None:
        for _ in range(3):
            self._crash("cause")
        self.assertEqual(
            self.guard.snapshot(),
            {
                "main": {
                    "crashes": 3,
                    "last_cause": "cause",
                    "tripped": True,
                    "retry_after": 220,
                    "probation": False,
                }
            },
        )
        self.assertTrue(self.guard.any_tripped())

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            CrashLoopGuard(threshold=-1, window=600, cooldown=300)
        with self.assertRaises(ValueError):
            CrashLoopGuard(threshold=3, window=0, cooldown=300)
        with self.assertRaises(ValueError):
            CrashLoopGuard(threshold=3, window=600, cooldown=0)


if __name__ == "__main__":
    unittest.main()
