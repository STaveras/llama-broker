"""Stop request-driven restarts from replaying a request into a known crash.

A backend that dies while serving a request is restarted by the next request
for its profile. When the crash is deterministic (a prompt that exhausts VRAM,
say), a retrying client replays it forever, paying a full model load and
prefill each time. The guard counts crashes per profile and, past a threshold,
refuses that profile's requests for a cooldown instead of restarting it.

After the cooldown one request is let through on probation; if it crashes too
the guard trips again at once. A completed response, or an explicit lifecycle
operation, clears the profile's history.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class CrashLoopRejection:
    profile_id: str
    crashes: int
    retry_after: int
    last_cause: str

    def message(self) -> str:
        return (
            f"backend {self.profile_id} crashed {self.crashes} times while serving "
            f"requests (last: {self.last_cause}); refusing its requests for "
            f"{self.retry_after}s instead of restarting it. "
            "`brokerctl restart` retries now."
        )


@dataclass
class _ProfileCrashes:
    times: deque[float] = field(default_factory=deque)
    last_cause: str = ""
    tripped_until: float | None = None
    probation: bool = False


class CrashLoopGuard:
    def __init__(
        self,
        *,
        threshold: int,
        window: float,
        cooldown: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if threshold < 0:
            raise ValueError("crash loop threshold must not be negative")
        if window <= 0 or cooldown <= 0:
            raise ValueError("crash loop window and cooldown must be positive")
        self.threshold = threshold
        self.window = window
        self.cooldown = cooldown
        self._clock = clock
        self._profiles: dict[str, _ProfileCrashes] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.threshold > 0

    def record_crash(self, profile_id: str, cause: str) -> bool:
        """Count a crash during a request; return True if this trips the guard."""

        if not self.enabled:
            return False
        now = self._clock()
        with self._lock:
            entry = self._profiles.setdefault(profile_id, _ProfileCrashes())
            entry.times.append(now)
            entry.last_cause = cause
            while entry.times and now - entry.times[0] > self.window:
                entry.times.popleft()
            if entry.probation or len(entry.times) >= self.threshold:
                entry.tripped_until = now + self.cooldown
                entry.probation = False
                return True
            return False

    def record_success(self, profile_id: str) -> None:
        with self._lock:
            self._profiles.pop(profile_id, None)

    def clear(self, profile_id: str | None = None) -> None:
        with self._lock:
            if profile_id is None:
                self._profiles.clear()
            else:
                self._profiles.pop(profile_id, None)

    def check(self, profile_id: str) -> CrashLoopRejection | None:
        """Return a rejection while the profile is tripped, else None."""

        now = self._clock()
        with self._lock:
            entry = self._profiles.get(profile_id)
            if entry is None or entry.tripped_until is None:
                return None
            if now >= entry.tripped_until:
                entry.tripped_until = None
                entry.probation = True
                return None
            return CrashLoopRejection(
                profile_id=profile_id,
                crashes=len(entry.times),
                retry_after=max(1, math.ceil(entry.tripped_until - now)),
                last_cause=entry.last_cause,
            )

    def snapshot(self) -> dict[str, dict[str, Any]]:
        now = self._clock()
        with self._lock:
            return {
                profile_id: {
                    "crashes": len(entry.times),
                    "last_cause": entry.last_cause,
                    "tripped": entry.tripped_until is not None and now < entry.tripped_until,
                    "retry_after": (
                        max(1, math.ceil(entry.tripped_until - now))
                        if entry.tripped_until is not None and now < entry.tripped_until
                        else 0
                    ),
                    "probation": entry.probation
                    or (entry.tripped_until is not None and now >= entry.tripped_until),
                }
                for profile_id, entry in self._profiles.items()
            }

    def any_tripped(self) -> bool:
        return any(entry["tripped"] for entry in self.snapshot().values())
