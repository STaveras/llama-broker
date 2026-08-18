"""Per-profile validation and reporting for ``brokerctl integrate``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .broker_profiles import Profile, ProfileError, _parse_profile
from .profile_config import load_manifest, sha256_file


@dataclass(frozen=True)
class ProfileInspection:
    profile_id: str
    status: str
    profile: Profile | None = None
    record: dict[str, Any] | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "status": self.status,
            "reason": self.reason,
        }


def _verify(profile: Profile) -> str | None:
    checks = [
        (profile.server_binary, profile.server_sha256, "runtime"),
        (profile.model_path, profile.model_sha256, "model"),
    ]
    if profile.mmproj_path is not None and profile.mmproj_sha256 is not None:
        checks.append((profile.mmproj_path, profile.mmproj_sha256, "projector"))
    for path, expected, label in checks:
        try:
            actual = sha256_file(path)
        except OSError as error:
            return f"{label} cannot be read: {path} ({error})"
        if actual != expected:
            return f"{label} hash mismatch: expected {expected}, got {actual} ({path})"
    return None


def inspect_manifest(
    path: Path, *, verify_artifacts: bool = True
) -> tuple[dict[str, Any], tuple[ProfileInspection, ...]]:
    """Parse profiles independently, optionally verifying their artifacts."""

    payload = load_manifest(path)
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raise ProfileError("profile manifest must contain a profiles list")
    inspections: list[ProfileInspection] = []
    for index, raw in enumerate(raw_profiles):
        if not isinstance(raw, dict):
            inspections.append(ProfileInspection(f"<index:{index}>", "broken", reason="profile is not an object"))
            continue
        profile_id = str(raw.get("id", f"<index:{index}>")).strip()
        try:
            profile = _parse_profile(raw, allow_missing_artifacts=not verify_artifacts)
        except (ProfileError, ValueError, TypeError, OSError) as error:
            inspections.append(ProfileInspection(profile_id, "broken", record=raw, reason=str(error)))
            continue
        failure = _verify(profile) if verify_artifacts else None
        if failure is not None:
            inspections.append(ProfileInspection(profile.id, "unverified", profile, raw, failure))
        else:
            inspections.append(ProfileInspection(profile.id, "ready", profile, raw))
    return payload, tuple(inspections)


def report_for(inspections: tuple[ProfileInspection, ...]) -> dict[str, object]:
    counts = {name: 0 for name in ("ready", "installed", "updated", "unchanged", "broken", "unverified", "skipped")}
    for item in inspections:
        counts[item.status] = counts.get(item.status, 0) + 1
    return {
        "schema_version": 1,
        "profiles": [item.as_dict() for item in inspections],
        "counts": counts,
    }


def filtered_manifest(payload: dict[str, Any], inspections: tuple[ProfileInspection, ...]) -> dict[str, Any] | None:
    """Return a temporary provider manifest containing ready records only."""

    records = [item.record for item in inspections if item.status == "ready" and item.record is not None]
    if not records:
        return None
    return {"schema_version": 1, "profiles": records}
