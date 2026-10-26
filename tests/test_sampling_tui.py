"""Numeric sampling fields in the existing profile editor; no live configuration."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from llama_broker import broker_tui
from llama_broker.broker_tui import TUIModel
from llama_broker.sampling import SAMPLING_FIELDS, qwen27b_sampling_defaults


class SamplingTUITests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"schema_version": 1, "profiles": [
            {"id": "main", "alias": "main", "runtime": "llamacpp"},
            {"id": "other", "alias": "other", "runtime": "llamacpp"},
        ]}
        self.model = TUIModel(self.manifest)

    def test_numeric_fields_are_visible_without_advanced_mode(self):
        keys = {field.key for field in self.model.visible_fields}
        self.assertIn("sampling_preset", keys)
        for mode in ("instruct", "thinking"):
            for name in SAMPLING_FIELDS:
                self.assertIn(f"sampling.{mode}.{name}", keys)

    def test_preset_matches_supplied_values_and_only_changes_selected_draft(self):
        self.model.set_field("sampling_preset", "Qwen 27B")
        self.assertEqual(self.model.current["sampling_defaults"], qwen27b_sampling_defaults())
        self.assertEqual(self.model.field_value("sampling_preset"), "Qwen 27B")
        self.assertTrue(self.model.dirty)
        self.assertEqual(self.manifest, self.model.base_manifest)
        self.model.move(1)
        self.assertNotIn("sampling_defaults", self.model.current)

    def test_float_and_zero_edits_preserve_other_mode_and_siblings(self):
        self.model.set_field("sampling_preset", "Qwen 27B")
        self.model.set_field("sampling.instruct.temperature", "0")
        self.assertEqual(self.model.current["sampling_defaults"]["instruct"]["temperature"], 0)
        self.assertEqual(self.model.field_value("sampling.thinking.temperature"), 1.0)
        self.assertEqual(self.model.field_value("sampling.instruct.top_p"), 0.8)
        self.assertEqual(self.model.field_value("sampling_preset"), "Custom")

    def test_clear_one_value_leaves_other_fields_intact(self):
        self.model.set_field("sampling_preset", "Qwen 27B")
        self.model.set_field("sampling.instruct.temperature", "")
        self.assertEqual(self.model.field_value("sampling.instruct.temperature"), "")
        self.assertEqual(self.model.field_value("sampling.thinking.temperature"), 1.0)
        self.assertEqual(self.model.field_value("sampling.instruct.top_p"), 0.8)

    def test_clear_last_field_removes_empty_maps(self):
        self.model.set_field("sampling.instruct.temperature", "0.5")
        self.model.set_field("sampling.instruct.temperature", "")
        self.assertNotIn("sampling_defaults", self.model.current)
        self.assertFalse(self.model.dirty)

    def test_clear_all_returns_to_runtime_defaults(self):
        self.model.set_field("sampling_preset", "Qwen 27B")
        self.model.set_field("sampling_preset", "Runtime defaults")
        self.assertNotIn("sampling_defaults", self.model.current)
        self.assertFalse(self.model.dirty)

    def test_invalid_edits_do_not_change_draft_or_dirty_state(self):
        for key, value in [("temperature", "nan"), ("temperature", "inf"),
                           ("temperature", "-1"), ("temperature", True),
                           ("top_p", "1.01"), ("top_k", "2.5"),
                           ("min_p", "-1"), ("presence_penalty", "3"),
                           ("repetition_penalty", "0"), ("unknown", "3")]:
            with self.subTest(key=key, value=value):
                old = copy.deepcopy(self.model.manifest)
                with self.assertRaises(ValueError):
                    self.model.set_field(f"sampling.instruct.{key}", value)
                self.assertEqual(self.model.manifest, old)
                self.assertFalse(self.model.dirty)

    def test_noop_edit_does_not_mark_dirty(self):
        self.model.set_field("sampling.instruct.temperature", "")
        self.assertFalse(self.model.dirty)
        self.assertNotIn("sampling_defaults", self.model.current)

    def test_sampling_fields_are_hidden_for_other_runtimes(self):
        for runtime in ("vllm", "strata", "basalt"):
            self.model.current["runtime"] = runtime
            self.assertFalse(any(f.key.startswith("sampling") for f in self.model.visible_fields))
            with self.assertRaises(ValueError):
                self.model.set_field("sampling_preset", "Qwen 27B")

    def test_runtime_change_clears_inapplicable_sampling_defaults_in_draft(self):
        self.model.set_field("sampling_preset", "Qwen 27B")
        self.model.set_field("runtime", "vllm")
        self.assertNotIn("sampling_defaults", self.model.current)
        self.assertTrue(self.model.dirty)
        self.assertNotIn("sampling_defaults", self.manifest["profiles"][0])

    def test_prompt_allows_blank_to_clear_sampling_field(self):
        key = "sampling.instruct.temperature"
        self.model.set_field(key, "0.7")
        self.model.field_index = next(i for i, f in enumerate(self.model.visible_fields) if f.key == key)
        with patch.object(broker_tui, "_curses_prompt", return_value="") as prompt:
            broker_tui._edit_current_field(object(), self.model)
        self.assertTrue(prompt.call_args.kwargs["allow_empty"])
        self.assertNotIn("sampling_defaults", self.model.current)

    def test_cancelling_a_zero_value_prompt_preserves_zero(self):
        from unittest.mock import Mock
        window = Mock()
        window.getmaxyx.return_value = (24, 80)
        with patch.object(broker_tui, "_read_curses_key", return_value=27):
            result = broker_tui._curses_prompt(window, "Temperature", 0, allow_empty=True)
        self.assertEqual(result, "0")

    def test_cancelled_preset_selection_does_not_change_draft(self):
        self.model.field_index = next(i for i, f in enumerate(self.model.visible_fields) if f.key == "sampling_preset")
        with patch.object(broker_tui, "_curses_select", return_value=("Qwen 27B", True)):
            broker_tui._edit_current_field(object(), self.model)
        self.assertFalse(self.model.dirty)
        self.assertEqual(self.model.manifest, self.manifest)


if __name__ == "__main__":
    unittest.main()
