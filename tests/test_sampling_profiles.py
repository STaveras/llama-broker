"""Manifest persistence and validation for per-profile request defaults."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llama_broker.broker_profiles import ProfileError, ProfileRegistry
from tests.test_manifest_watcher import make_profile_record
from tests.test_sampling_defaults import DEFAULTS


class SamplingProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "profiles.json"
        self.record = make_profile_record("main", "main", self.root / "model.gguf")

    def registry(self, records=None):
        self.path.write_text(json.dumps({"schema_version": 1, "profiles": records or [self.record]}))
        return ProfileRegistry.from_path(self.path, allow_missing_artifacts=True)

    def test_defaults_round_trip_as_immutable_profile_data(self):
        self.record["sampling_defaults"] = DEFAULTS
        profile = self.registry().get("main")
        self.assertEqual(dict(profile.sampling_defaults["thinking"]), DEFAULTS["thinking"])
        with self.assertRaises(TypeError):
            profile.sampling_defaults["thinking"]["temperature"] = 0.2

    def test_omitting_defaults_is_backwards_compatible(self):
        self.assertFalse(self.registry().get("main").sampling_defaults)

    def test_invalid_defaults_fail_manifest_validation(self):
        for invalid in [None, [], {"instruct": {"top_p": 1.1}}, {"thinking": {"top_k": True}}]:
            self.record["sampling_defaults"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ProfileError):
                self.registry()

    def test_sampler_edits_change_snapshot_identity_but_not_launch_command(self):
        old = self.registry()
        self.record["sampling_defaults"] = DEFAULTS
        new = self.registry()
        self.assertNotEqual(old.profile_snapshot_identity("main"), new.profile_snapshot_identity("main"))
        self.assertEqual(old.get("main").command(host="127.0.0.1", port=10000),
                         new.get("main").command(host="127.0.0.1", port=10000))

    def test_unsupported_runtimes_reject_the_profile_setting(self):
        self.record["runtime"] = "vllm"
        self.record["sampling_defaults"] = DEFAULTS
        with self.assertRaisesRegex(ProfileError, "sampling_defaults"):
            self.registry()

    def test_inherited_profiles_can_override_one_mode_without_changing_parent(self):
        self.record["sampling_defaults"] = DEFAULTS
        child = {"id": "other", "alias": "other", "aliases": ["other"], "extends": "main",
                 "sampling_defaults": {"instruct": {"temperature": 0.4}}}
        registry = self.registry([self.record, child])
        self.assertEqual(registry.get("other").sampling_defaults["instruct"]["temperature"], 0.4)
        self.assertEqual(registry.get("other").sampling_defaults["thinking"]["temperature"], 1.0)
        self.assertEqual(registry.get("main").sampling_defaults["instruct"]["temperature"], 0.7)

    def test_cleared_inherited_defaults_stay_cleared_after_save_and_reload(self):
        from llama_broker.broker_tui import TUIModel
        from llama_broker.profile_config import write_manifest_atomically, load_manifest_source
        self.record["sampling_defaults"] = DEFAULTS
        # Validation uses real disposable artifacts; no runtime is executed.
        server = self.root / "llama-server"
        server.write_bytes(b"offline binary")
        server.chmod(0o755)
        (self.root / "model.gguf").write_bytes(b"offline model")
        child = {"id": "other", "alias": "other", "aliases": ["other"], "extends": "main"}
        self.registry([self.record, child])
        model = TUIModel(load_manifest_source(self.path))
        model.move(1)
        model.set_field("sampling.instruct.temperature", "")
        write_manifest_atomically(self.path, model.manifest)
        parsed = ProfileRegistry.from_path(self.path).get("other")
        self.assertNotIn("temperature", parsed.sampling_defaults["instruct"])
        self.assertEqual(parsed.sampling_defaults["thinking"]["temperature"], 1.0)
        model.set_field("sampling_preset", "Runtime defaults")
        write_manifest_atomically(self.path, model.manifest)
        self.assertFalse(ProfileRegistry.from_path(self.path).get("other").sampling_defaults)
        self.assertEqual(load_manifest_source(self.path)["profiles"][1]["extends"], "main")

    def test_sampling_inheritance_flag_requires_a_boolean(self):
        self.record["sampling_defaults_inherit"] = "false"
        with self.assertRaises(ProfileError):
            self.registry()

    def test_one_profile_does_not_inherit_another_profiles_defaults(self):
        self.record["sampling_defaults"] = DEFAULTS
        other = make_profile_record("other", "other", self.root / "other.gguf")
        registry = self.registry([self.record, other])
        self.assertFalse(registry.get("other").sampling_defaults)


if __name__ == "__main__":
    unittest.main()
