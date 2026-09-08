from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llama_broker.setup_wizard import ensure_runtime_catalog


class SetupCatalogLifecycleTests(unittest.TestCase):
    def test_first_run_inventories_once_and_creates_external_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory_root = root / "inventory"
            (inventory_root / "nested").mkdir(parents=True)
            model = inventory_root / "nested" / "model.gguf"
            model.write_bytes(b"model")
            runtime = inventory_root / "nested" / "llama-server"
            runtime.write_text("#!/bin/sh\n", encoding="utf-8")
            runtime.chmod(0o755)
            deferred = inventory_root / "checkpoint.safetensors"
            deferred.write_bytes(b"deferred")
            catalog_path = root / "runtimes" / "catalog.json"

            report = ensure_runtime_catalog(
                catalog_path,
                inventory_roots=[inventory_root],
            )

            self.assertEqual(report["mode"], "initial-inventory")
            self.assertTrue(report["broad_scan"])
            self.assertTrue(catalog_path.is_file())
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(catalog["schema_version"], 2)
            self.assertTrue(catalog["deployment_revision"])
            paths = {item["path"] for item in catalog["inventory"]}
            self.assertIn(str(model.resolve()), paths)
            self.assertIn(str(runtime.resolve()), paths)
            self.assertIn(str(deferred.resolve()), paths)
            self.assertEqual(catalog["runtimes"], [])

    def test_later_setup_uses_catalog_and_only_inspects_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory_root = root / "inventory"
            referenced = inventory_root / "referenced.gguf"
            unreferenced = inventory_root / "unreferenced.gguf"
            inventory_root.mkdir()
            referenced.write_bytes(b"referenced")
            unreferenced.write_bytes(b"unreferenced")
            catalog_path = root / "runtimes" / "catalog.json"
            ensure_runtime_catalog(catalog_path, inventory_roots=[inventory_root])

            report = ensure_runtime_catalog(
                catalog_path,
                inventory_roots=[inventory_root],
                referenced_paths=[referenced],
            )

            self.assertEqual(report["mode"], "catalog-first")
            self.assertFalse(report["broad_scan"])
            self.assertEqual(report["inspected_paths"], [str(referenced.resolve())])
            self.assertEqual(report["inventory_added"], [])


if __name__ == "__main__":
    unittest.main()
