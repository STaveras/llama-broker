from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llama_broker.broker_tui import EDITOR_FIELDS, TUIModel, run_guided_wizard


class BrokerTUIModelTests(unittest.TestCase):
    def test_capabilities_editor_exposes_tools(self) -> None:
        field = next(item for item in EDITOR_FIELDS if item.key == "capabilities")
        self.assertIn("tools", field.options)

    def test_selection_wraps_and_simple_update_is_shared(self) -> None:
        manifest = {
            "profiles": [
                {"id": "one", "alias": "one", "context_size": 32768, "parallel": 1},
                {"id": "two", "alias": "two", "context_size": 65536, "parallel": 1},
            ]
        }
        model = TUIModel(manifest)
        model.move(-1)
        self.assertEqual(model.current["id"], "two")
        model.update_current({"context_size": 131072, "parallel": 2})
        self.assertEqual(model.current["context_size"], 131072)
        self.assertEqual(model.current["parallel"], 2)

    def test_field_editor_navigates_individual_fields_and_rehashes_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            binary = Path(tempdir) / "llama-server"
            binary.write_bytes(b"new runtime")
            manifest = {
                "profiles": [
                    {
                        "id": "one",
                        "alias": "one",
                        "aliases": ["one"],
                        "display_name": "One",
                        "context_size": 32768,
                        "parallel": 1,
                        "server_binary": str(binary),
                        "server_sha256": "old",
                    }
                ]
            }
            model = TUIModel(manifest)
            self.assertEqual(model.current_field.key, "display_name")
            model.move_field(1)
            self.assertEqual(model.current_field.key, "alias")
            model.set_field("alias", "one-new")
            self.assertEqual(model.current["aliases"], ["one-new"])
            model.set_field("server_binary", str(binary))
            self.assertNotEqual(model.current["server_sha256"], "old")
            self.assertTrue(model.dirty)

    def test_advanced_fields_are_hidden_until_toggled(self) -> None:
        model = TUIModel({"profiles": [{"id": "one", "alias": "one", "fit": True}]})
        self.assertNotIn("server_binary", {field.key for field in model.visible_fields})
        model.toggle_advanced()
        self.assertIn("server_binary", {field.key for field in model.visible_fields})


class GuidedWizardTests(unittest.TestCase):
    def test_existing_profile_preview_uses_shared_callback_without_write(self) -> None:
        from tests.fixtures import materialize_fixture  # noqa: PLC0415

        source = materialize_fixture()
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "profiles.json"
            manifest_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            # display, alias, context, parallel, batch, ubatch, cache K, cache V,
            # gpu layers, projector, advanced?, apply?
            answers = iter(
                [
                    "",
                    "",
                    "",
                    "",
                    "131072",
                    "2",
                    "64",
                    "64",
                    "q8_0",
                    "q8_0",
                    "auto",
                    "",
                    "n",
                    "y",
                ]
            )
            callback = Mock()
            with patch("builtins.input", side_effect=lambda prompt: next(answers)):
                result = run_guided_wizard(
                    manifest_path,
                    "fixture-vision-12b",
                    apply_callback=callback,
                )
            self.assertEqual(result, 0)
            callback.assert_called_once()
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))
            updated, changed = callback.call_args.args
            self.assertIn("fixture-vision-12b", changed)
            self.assertEqual(
                next(item for item in updated["profiles"] if item["id"] == "fixture-vision-12b")["context_size"],
                131072,
            )


if __name__ == "__main__":
    unittest.main()
