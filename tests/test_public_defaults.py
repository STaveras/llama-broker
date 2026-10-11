from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools import model_store_migrate as migration
from tools.check_public_tree import inspect_blob


class PortableDefaultsTests(unittest.TestCase):
    def test_migration_infers_generic_names_and_accepts_explicit_overrides(self):
        self.assertEqual(migration.MODEL_TARGET_OVERRIDES, {})
        self.assertEqual(migration.RUNTIME_REPOSITORY_OVERRIDES, {})
        self.assertEqual(migration.CANONICAL_RUNTIME_REPOSITORIES, {})
        source = "/example/models/Qwen3.5-9B-Q6_K.gguf"
        self.assertEqual(migration.canonical_model_target(source), "qwen-3.5/9b/q6-k.gguf")
        self.assertEqual(migration.canonical_model_target(source, overrides={source: "custom/model.gguf"}), "custom/model.gguf")
        with self.assertRaises(migration.MigrationError):
            migration.canonical_model_target("/example/models/ambiguous.gguf")

    def test_example_environment_resolves_to_xdg_paths(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            env = {k:v for k,v in os.environ.items() if not k.startswith("LLAMA_BROKER_")}
            env.update(XDG_CONFIG_HOME=tmp + "/config", XDG_DATA_HOME=tmp + "/data")
            command = 'source docs/examples/local.env.example; PYTHONPATH=src python3 -c \'import json; from llama_broker import paths; print(json.dumps([str(paths.PROFILES_PATH), str(paths.MODELS_DIR), str(paths.RUNTIME_CATALOG_PATH)]))\''
            output = subprocess.check_output(["bash", "-c", command], cwd=root, env=env)
            self.assertEqual(json.loads(output), [tmp+"/config/llama-broker/profiles.json", tmp+"/data/models", tmp+"/data/runtimes/catalog.json"])

    def test_public_tree_scans_ignored_tracked_snapshots_and_exact_fixtures(self):
        self.assertTrue(inspect_blob("candidate/profiles.production.snapshot.json", b"{}", {}))
        fixture = b'api_key="fixture-only-example"'
        self.assertTrue(inspect_blob("tests/test_synthetic.py", fixture, {}))
        import hashlib
        allow = {"tests/test_synthetic.py": [hashlib.sha256(b"fixture-only-example").hexdigest()]}
        self.assertFalse(inspect_blob("tests/test_synthetic.py", fixture, allow))
        self.assertTrue(inspect_blob("src/config.py", fixture, allow))
