"""Offline request defaults: client settings win, profiles are isolated."""
from __future__ import annotations

import copy
import unittest

from llama_broker import daemon


INSTRUCT = dict(temperature=0.7, top_p=0.80, top_k=20, min_p=0.0,
                presence_penalty=1.5, repetition_penalty=1.0)
THINKING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                presence_penalty=0.0, repetition_penalty=1.0)
DEFAULTS = {"instruct": INSTRUCT, "thinking": THINKING}


class SamplingDefaultsTests(unittest.TestCase):
    def apply(self, payload, defaults=None, runtime="llamacpp"):
        return daemon.apply_sampling_defaults(
            payload, DEFAULTS if defaults is None else defaults, runtime=runtime
        )

    def resolved(self, payload):
        return daemon.apply_reasoning_policy(payload, runtime="llamacpp", max_output_tokens=4096)

    def test_missing_reasoning_uses_instruct(self):
        result = self.apply(self.resolved({"messages": []}))
        for key, value in INSTRUCT.items():
            self.assertEqual(result["repeat_penalty" if key == "repetition_penalty" else key], value)
        self.assertNotIn("repetition_penalty", result)

    def test_resolved_reasoning_uses_thinking(self):
        for request in [
            {"reasoning_effort": "high"}, {"reasoning": {"effort": "low"}},
            {"chat_template_kwargs": {"enable_thinking": True}},
            {"thinking_budget_tokens": 100}, {"reasoning_budget_tokens": 100},
        ]:
            with self.subTest(request=request):
                result = self.apply(self.resolved(request))
                self.assertEqual(result["temperature"], 1.0)
                self.assertEqual(result["presence_penalty"], 0.0)

    def test_effective_off_overrides_requested_reasoning(self):
        for request in [
            {"reasoning_effort": "none"},
            {"reasoning_effort": "high", "thinking_budget_tokens": 0},
            {"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}},
        ]:
            with self.subTest(request=request):
                self.assertEqual(self.apply(self.resolved(request))["temperature"], 0.7)

    def test_partial_explicit_settings_override_individually_including_zero(self):
        result = self.apply({"temperature": 0, "top_p": 0.5, "presence_penalty": 0,
                             "repeat_penalty": 1.2})
        self.assertEqual(result["temperature"], 0)
        self.assertEqual(result["top_p"], 0.5)
        self.assertEqual(result["presence_penalty"], 0)
        self.assertEqual(result["repeat_penalty"], 1.2)
        self.assertEqual(result["top_k"], 20)

    def test_repetition_alias_is_translated_without_conflicting_default(self):
        result = self.apply({"repetition_penalty": 1.15})
        self.assertEqual(result["repeat_penalty"], 1.15)
        self.assertNotIn("repetition_penalty", result)

    def test_native_repetition_field_wins_if_client_supplies_both(self):
        result = self.apply({"repeat_penalty": 1.2, "repetition_penalty": 1.15})
        self.assertEqual(result["repeat_penalty"], 1.2)
        self.assertNotIn("repetition_penalty", result)

    def test_no_config_leaves_request_unchanged(self):
        request = {"temperature": 0.25, "messages": []}
        self.assertEqual(self.apply(request, {}), request)

    def test_unconfigured_mode_does_not_borrow_other_mode(self):
        self.assertNotIn("temperature", self.apply({}, {"thinking": THINKING}))

    def test_no_request_or_defaults_mutation(self):
        request = {"messages": [], "chat_template_kwargs": {"enable_thinking": True}}
        original = copy.deepcopy(request)
        defaults = copy.deepcopy(DEFAULTS)
        self.apply(request, defaults)
        self.assertEqual(request, original)
        self.assertEqual(defaults, DEFAULTS)

    def test_null_does_not_silently_become_a_profile_default(self):
        self.assertIsNone(self.apply({"temperature": None})["temperature"])

    def test_unsupported_runtime_does_not_receive_sampling_extensions(self):
        self.assertEqual(self.apply({}, runtime="strata"), {})


class SamplingValidationTests(unittest.TestCase):
    def validate(self, value):
        from llama_broker.sampling import validate_sampling_defaults
        return validate_sampling_defaults(value)

    def test_presets_validate_and_are_immutable(self):
        result = self.validate(DEFAULTS)
        self.assertEqual(dict(result["instruct"]), INSTRUCT)
        with self.assertRaises(TypeError):
            result["thinking"]["temperature"] = 2.0

    def test_invalid_configs_are_rejected(self):
        for value in [None, [], {"other": {}}, {"thinking": None},
                      {"instruct": {"unknown": 1}},
                      {"instruct": {"temperature": True}},
                      {"instruct": {"temperature": float("nan")}},
                      {"instruct": {"temperature": float("inf")}},
                      {"instruct": {"top_k": 2.5}},
                      {"instruct": {"temperature": -0.1}},
                      {"instruct": {"top_p": 1.1}},
                      {"instruct": {"min_p": -0.1}},
                      {"instruct": {"presence_penalty": 3}},
                      {"instruct": {"repetition_penalty": 0}},
                      {"instruct": {"temperature": "0.7"}}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(value)

    def test_empty_and_partial_configs_are_valid(self):
        self.assertEqual(dict(self.validate({})), {})
        self.assertEqual(dict(self.validate({"instruct": {"temperature": 0}})["instruct"]), {"temperature": 0})


if __name__ == "__main__":
    unittest.main()
