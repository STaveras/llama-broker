from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_sync_models():
    spec = importlib.util.spec_from_file_location("sync_models_under_test", REPO_ROOT / "sync-models.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyncModelsNamingTests(unittest.TestCase):
    def test_variant_aware_canonical_name_preserves_antidoom_variant(self) -> None:
        module = load_sync_models()
        name, metadata = module.model_display_name(
            Path("/tmp/Qwen3.5-9B-Antidoom-v3-Q6_K.gguf")
        )
        target = module.canonical_model_target(
            Path("/tmp/Qwen3.5-9B-Antidoom-v3-Q6_K.gguf"), name, metadata
        )
        self.assertEqual(target, "qwen-3.5/9b/q6-antidoom-v3.gguf")

    def test_canonical_target_collision_fails_closed(self) -> None:
        module = load_sync_models()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "Qwen3.5-9B-Q6_K.gguf"
            second = root / "Qwen3.5-9B-Q6_K-copy.gguf"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            with patch.object(module, "MODELS_DIR", root):
                with self.assertRaisesRegex(ValueError, "collision"):
                    module.discover()

    def test_deferred_hf_assets_are_not_serving_candidates(self) -> None:
        module = load_sync_models()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "Qwen3.5-9B-Q6_K.gguf"
            deferred = root / "Qwen3.5-9B.safetensors"
            model.write_bytes(b"model")
            deferred.write_bytes(b"weights")
            with patch.object(module, "MODELS_DIR", root):
                records = module.discover()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["path"], str(model))


if __name__ == "__main__":
    unittest.main()
