"""Config-path resolution tests.

Covers the XDG default for ``PROFILES_PATH``, the ``LLAMA_BROKER_PROFILES``
override, and setup seeding of a missing manifest. These are environment-level
contracts for where the broker looks for profiles, independent of any
particular machine's layout.

``PROFILES_PATH`` is computed at import time, so the resolution tests run the
module in a subprocess with a controlled environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_in_subprocess(env: dict[str, str]) -> str:
    """Return the PROFILES_PATH/CONFIG_DIR a fresh interpreter would compute."""
    code = (
        "import sys; sys.path.insert(0, %r); " % str(REPO_ROOT / "src")
        + "import json, llama_broker; "
        "print(json.dumps({'profiles': str(llama_broker.PROFILES_PATH), "
        "'config': str(llama_broker.CONFIG_DIR)}))"
    )
    full_env = dict(os.environ)
    full_env.pop("LLAMA_BROKER_PROFILES", None)
    full_env.pop("XDG_CONFIG_HOME", None)
    full_env.update({k: v for k, v in env.items() if v is not None})
    src_path = str(REPO_ROOT / "src")
    existing_pythonpath = full_env.get("PYTHONPATH")
    full_env["PYTHONPATH"] = (
        src_path
        if not existing_pythonpath
        else os.pathsep.join((src_path, existing_pythonpath))
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=full_env,
        check=True,
    )
    return json.loads(result.stdout.strip())


class ConfigPathResolutionTests(unittest.TestCase):
    def test_profiles_path_defaults_to_xdg_config_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            resolved = resolve_in_subprocess({"XDG_CONFIG_HOME": str(xdg)})
            self.assertEqual(
                resolved["profiles"],
                str(xdg / "llama-broker" / "profiles.json"),
            )
            self.assertEqual(resolved["config"], str(xdg / "llama-broker"))

    def test_profiles_path_uses_home_dot_config_without_xdg(self) -> None:
        resolved = resolve_in_subprocess({"XDG_CONFIG_HOME": None})
        self.assertEqual(
            resolved["profiles"],
            str(Path.home() / ".config" / "llama-broker" / "profiles.json"),
        )
        self.assertEqual(
            resolved["config"], str(Path.home() / ".config" / "llama-broker")
        )

    def test_llama_broker_profiles_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom-profiles.json"
            resolved = resolve_in_subprocess(
                {
                    "XDG_CONFIG_HOME": str(Path(tmp) / "xdg"),
                    "LLAMA_BROKER_PROFILES": str(custom),
                }
            )
            self.assertEqual(resolved["profiles"], str(custom))
            # CONFIG_DIR still follows XDG (keys dir, TLS material).
            self.assertEqual(
                resolved["config"], str(Path(tmp) / "xdg" / "llama-broker")
            )

    def test_canonical_environment_names_override_legacy_runtime_names(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r); " % str(REPO_ROOT / "src")
            + "import json; from llama_broker import daemon; "
            + "print(json.dumps({'url': daemon.BACKEND_URL, 'host': daemon.BROKER_HOST, "
            + "'port': daemon.BROKER_PORT}))"
        )
        env = dict(os.environ)
        for name in (
            "LLAMA_BROKER_BACKEND_URL",
            "LLM_BROKER_BACKEND_URL",
            "LLAMA_BROKER_HOST",
            "LLM_BROKER_HOST",
            "LLAMA_BROKER_PORT",
            "LLM_BROKER_PORT",
        ):
            env.pop(name, None)
        env.update(
            {
                "LLM_BROKER_BACKEND_URL": "http://legacy:10000",
                "LLAMA_BROKER_BACKEND_URL": "http://canonical:10000",
                "LLM_BROKER_HOST": "legacy-host",
                "LLAMA_BROKER_HOST": "canonical-host",
                "LLM_BROKER_PORT": "10001",
                "LLAMA_BROKER_PORT": "10002",
            }
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        values = json.loads(result.stdout)
        self.assertEqual(
            values,
            {
                "url": "http://canonical:10000",
                "host": "canonical-host",
                "port": 10002,
            },
        )

    def test_profiles_example_is_valid_json_with_generic_ids(self) -> None:
        example = REPO_ROOT / "profiles.example.json"
        manifest = json.loads(example.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertNotIn("default_profile", manifest)
        self.assertTrue(manifest["profiles"])
        for profile in manifest["profiles"]:
            self.assertNotIn("/home/", profile.get("server_binary", ""))
            self.assertNotIn("/home/", profile.get("model_path", ""))

    def test_config_dir_constant_follows_xdg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"
            xdg.mkdir()
            resolved = resolve_in_subprocess({"XDG_CONFIG_HOME": str(xdg)})
            self.assertEqual(resolved["config"], str(xdg / "llama-broker"))


if __name__ == "__main__":
    unittest.main()
