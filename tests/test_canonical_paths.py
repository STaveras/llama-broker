from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def _paths_in_subprocess(env: dict[str, str]) -> dict[str, str]:
    code = (
        "import sys; sys.path.insert(0, %r); " % str(SRC)
        + "import json; from llama_broker import paths; "
        + "print(json.dumps({name: str(getattr(paths, name)) for name in "
        + "('MODELS_DIR', 'RUNTIMES_DIR', 'CACHE_DIR', 'RUNTIME_CATALOG_PATH')}))"
    )
    full_env = dict(os.environ)
    for name in (
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
        "LLAMA_BROKER_MODELS_DIR",
        "LLAMA_BROKER_RUNTIMES_DIR",
        "LLAMA_BROKER_CACHE_DIR",
        "LLAMA_BROKER_RUNTIME_CATALOG",
        "MODELS_DIR",
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


class CanonicalPathTests(unittest.TestCase):
    def test_xdg_defaults_include_external_runtime_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values = _paths_in_subprocess(
                {
                    "XDG_DATA_HOME": str(root / "data"),
                    "XDG_CACHE_HOME": str(root / "cache"),
                    "XDG_CONFIG_HOME": str(root / "config"),
                    "XDG_RUNTIME_DIR": str(root / "run"),
                }
            )
        self.assertEqual(values["MODELS_DIR"], str(root / "data" / "models"))
        self.assertEqual(values["RUNTIMES_DIR"], str(root / "data" / "runtimes"))
        self.assertEqual(values["CACHE_DIR"], str(root / "cache" / "llama-broker"))
        self.assertEqual(
            values["RUNTIME_CATALOG_PATH"],
            str(root / "data" / "runtimes" / "catalog.json"),
        )

    def test_canonical_overrides_win_over_legacy_models_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values = _paths_in_subprocess(
                {
                    "LLAMA_BROKER_MODELS_DIR": str(root / "canonical-models"),
                    "LLAMA_BROKER_RUNTIMES_DIR": str(root / "canonical-runtimes"),
                    "LLAMA_BROKER_CACHE_DIR": str(root / "canonical-cache"),
                    "LLAMA_BROKER_RUNTIME_CATALOG": str(root / "catalog.json"),
                    "MODELS_DIR": str(root / "legacy-models"),
                }
            )
        self.assertEqual(values["MODELS_DIR"], str(root / "canonical-models"))
        self.assertEqual(values["RUNTIMES_DIR"], str(root / "canonical-runtimes"))
        self.assertEqual(values["CACHE_DIR"], str(root / "canonical-cache"))
        self.assertEqual(values["RUNTIME_CATALOG_PATH"], str(root / "catalog.json"))

    def test_sync_models_accepts_legacy_models_dir_only_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = REPO_ROOT / "sync-models.py"
            spec = importlib.util.spec_from_file_location("sync_models_test", script)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            previous = dict(os.environ)
            try:
                os.environ.pop("LLAMA_BROKER_MODELS_DIR", None)
                os.environ["MODELS_DIR"] = str(root / "legacy")
                spec.loader.exec_module(module)
                self.assertEqual(module.MODELS_DIR, (root / "legacy").resolve())
            finally:
                os.environ.clear()
                os.environ.update(previous)

    def test_download_script_prefers_canonical_models_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = dict(os.environ)
            env.update(
                {
                    "LLAMA_BROKER_MODELS_DIR": str(root / "canonical"),
                    "MODELS_DIR": str(root / "legacy"),
                }
            )
            result = subprocess.run(
                [str(REPO_ROOT / "download-models.sh")],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
        self.assertIn(str(root / "canonical"), result.stdout)
        self.assertNotIn(str(root / "legacy"), result.stdout)

    def test_superseded_non_dot_local_path_is_not_in_store_scripts(self) -> None:
        for path in (REPO_ROOT / "sync-models.py", REPO_ROOT / "download-models.sh"):
            self.assertNotIn("/home/stan/local/share/models", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
