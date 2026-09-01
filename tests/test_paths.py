from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def resolve_paths(env: dict[str, str | None]) -> dict[str, str]:
    code = (
        "import sys; sys.path.insert(0, %r); " % str(SRC)
        + "import json; from llama_broker import paths; "
        + "print(json.dumps({name: str(getattr(paths, name)) for name in "
        + "('CONFIG_DIR', 'PROFILES_PATH', 'STATE_PATH', 'DATA_DIR', 'MODELS_DIR', "
        + "'RUNTIMES_DIR', 'CACHE_DIR', 'RUNTIME_STATE_DIR', 'SYSTEMD_USER_DIR')}))"
    )
    full_env = dict(os.environ)
    for name in (
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
        "LLAMA_BROKER_PROFILES",
        "LLAMA_BROKER_STATE_FILE",
        "LLAMA_BROKER_LOCK_FILE",
        "LLAMA_BROKER_PROCESS_LOCK_FILE",
        "SYSTEMD_USER_DIR",
    ):
        full_env.pop(name, None)
    for name, value in env.items():
        if value is None:
            full_env.pop(name, None)
        else:
            full_env[name] = value
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=full_env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def migrate_state(env: dict[str, str]) -> dict[str, object]:
    code = (
        "import sys; sys.path.insert(0, %r); " % str(SRC)
        + "import json; from llama_broker import paths; "
        + "paths.LEGACY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True); "
        + "paths.LEGACY_STATE_PATH.write_text('legacy-state', encoding='utf-8'); "
        + "migrated = paths.migrate_legacy_state(); "
        + "print(json.dumps({'migrated': migrated, "
        + "'new_exists': paths.STATE_PATH.exists(), "
        + "'legacy_exists': paths.LEGACY_STATE_PATH.exists(), "
        + "'new_contents': paths.STATE_PATH.read_text(encoding='utf-8')}))"
    )
    full_env = dict(os.environ)
    for name in (
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
        "LLAMA_BROKER_PROFILES",
        "LLAMA_BROKER_STATE_FILE",
        "LLAMA_BROKER_LOCK_FILE",
        "LLAMA_BROKER_PROCESS_LOCK_FILE",
        "SYSTEMD_USER_DIR",
    ):
        full_env.pop(name, None)
    full_env.update(env)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=full_env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


class XDGPathTests(unittest.TestCase):
    def test_defaults_use_linux_xdg_locations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = resolve_paths(
                {
                    "XDG_CONFIG_HOME": str(root / "config"),
                    "XDG_DATA_HOME": str(root / "data"),
                    "XDG_CACHE_HOME": str(root / "cache"),
                    "XDG_RUNTIME_DIR": str(root / "run"),
                }
            )
        self.assertEqual(values["CONFIG_DIR"], str(root / "config" / "llama-broker"))
        self.assertEqual(values["PROFILES_PATH"], str(root / "config" / "llama-broker" / "profiles.json"))
        self.assertEqual(values["STATE_PATH"], str(root / "config" / "llama-broker" / "state.json"))
        self.assertEqual(values["DATA_DIR"], str(root / "data" / "llama-broker"))
        self.assertEqual(values["MODELS_DIR"], str(root / "data" / "models"))
        self.assertEqual(values["RUNTIMES_DIR"], str(root / "data" / "runtimes"))
        self.assertEqual(values["CACHE_DIR"], str(root / "cache" / "llama-broker"))
        self.assertEqual(values["RUNTIME_STATE_DIR"], str(root / "run" / "llama-broker"))
        self.assertEqual(values["SYSTEMD_USER_DIR"], str(Path.home() / ".config/systemd/user"))

    def test_legacy_state_is_migrated_to_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = migrate_state(
                {
                    "XDG_CONFIG_HOME": str(root / "config"),
                    "XDG_DATA_HOME": str(root / "data"),
                    "XDG_CACHE_HOME": str(root / "cache"),
                    "XDG_RUNTIME_DIR": str(root / "run"),
                }
            )
        self.assertEqual(
            values,
            {
                "migrated": True,
                "new_exists": True,
                "legacy_exists": False,
                "new_contents": "legacy-state",
            },
        )

    def test_explicit_profile_and_systemd_paths_override_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = resolve_paths(
                {
                    "XDG_CONFIG_HOME": str(root / "config"),
                    "LLAMA_BROKER_PROFILES": str(root / "profiles.json"),
                    "SYSTEMD_USER_DIR": str(root / "systemd"),
                }
            )
        self.assertEqual(values["PROFILES_PATH"], str(root / "profiles.json"))
        self.assertEqual(values["SYSTEMD_USER_DIR"], str(root / "systemd"))


if __name__ == "__main__":
    unittest.main()
