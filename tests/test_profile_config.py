from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from llama_broker.broker_profiles import ProfileRegistry
from llama_broker.profile_config import (
    apply_manifest_transaction,
    apply_simple_fields,
    draft_from_gguf,
    find_profile_record,
    load_manifest,
    semantic_diff,
)
from llama_broker.manifest_resolution import resolve_manifest_inheritance


class ProfileConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.fixtures import materialize_fixture  # noqa: PLC0415

        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.manifest_path = self.root / "profiles.json"
        self.data = json.loads(materialize_fixture().read_text(encoding="utf-8"))
        self.manifest_path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_simple_projection_updates_projector_hash_and_vision_capability(self) -> None:
        projector = self.root / "mmproj.gguf"
        projector.write_bytes(b"projector")
        original = next(item for item in self.data["profiles"] if item["id"] == "fixture-vision-12b")
        updated = apply_simple_fields(
            original,
            {
                "context_size": 65536,
                "parallel": 2,
                "mmproj_path": str(projector),
            },
        )
        self.assertEqual(updated["context_size"], 65536)
        self.assertEqual(updated["parallel"], 2)
        self.assertIn("vision", updated["capabilities"])
        self.assertEqual(len(updated["mmproj_sha256"]), 64)

    def test_gguf_draft_is_pinned_and_keeps_broker_owned_runtime(self) -> None:
        model = self.root / "Imported-Q4.gguf"
        model.write_bytes(b"GGUF-not-a-full-model")
        draft = draft_from_gguf(self.data, model, profile_id="imported-model")
        self.assertEqual(draft["id"], "imported-model")
        self.assertEqual(Path(draft["model_path"]), model.resolve())
        self.assertEqual(len(draft["model_sha256"]), 64)
        self.assertEqual(draft["parallel"], 1)
        self.assertNotIn("fit", draft)
        candidate = dict(self.data)
        candidate["profiles"] = [*candidate["profiles"], draft]
        candidate_path = self.root / "candidate.json"
        candidate_path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
        ProfileRegistry.from_path(candidate_path)

    def test_find_profile_accepts_alias_and_diff_is_stable(self) -> None:
        found = find_profile_record(self.data, "fixture-vision")
        self.assertIsNotNone(found)
        changed = dict(self.data)
        changed["schema_version"] = 1
        self.assertEqual(semantic_diff(self.data, changed), [])

    def test_inheritance_merges_maps_and_replaces_lists(self) -> None:
        payload = {
            "schema_version": 1,
            "profiles": [
                {"id": "base", "nested": {"one": 1}, "items": [1, 2]},
                {
                    "id": "child",
                    "extends": "base",
                    "nested": {"two": 2},
                    "items": [3],
                },
            ],
        }
        resolved = resolve_manifest_inheritance(payload)
        child = resolved["profiles"][1]
        self.assertEqual(child["nested"], {"one": 1, "two": 2})
        self.assertEqual(child["items"], [3])
        self.assertNotIn("extends", child)

    def test_inheritance_rejects_unknown_parent_cycles_and_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown profile parent"):
            resolve_manifest_inheritance({"profiles": [{"id": "child", "extends": "missing"}]})
        with self.assertRaisesRegex(ValueError, "cycle"):
            resolve_manifest_inheritance(
                {"profiles": [{"id": "a", "extends": "b"}, {"id": "b", "extends": "a"}]}
            )
        with self.assertRaisesRegex(ValueError, "forbidden"):
            resolve_manifest_inheritance({"profiles": [{"id": "a", "extends": "https://example.test/a"}]})

    def test_legacy_manifest_migrates_with_adjacent_backup(self) -> None:
        legacy = json.loads(json.dumps(self.data))
        legacy.pop("schema_version")
        self.manifest_path.write_text(json.dumps(legacy), encoding="utf-8")
        migrated = load_manifest(self.manifest_path)
        self.assertEqual(migrated["schema_version"], 1)
        persisted = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["schema_version"], 1)
        backups = list(self.root.glob("profiles.json.bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text(encoding="utf-8")), legacy)

    def test_ambiguous_or_unsupported_manifest_is_untouched(self) -> None:
        original = json.dumps({"profiles": [{"id": "ambiguous"}]})
        self.manifest_path.write_text(original, encoding="utf-8")
        with self.assertRaisesRegex(Exception, "ambiguous"):
            load_manifest(self.manifest_path)
        self.assertEqual(self.manifest_path.read_text(encoding="utf-8"), original)
        self.assertFalse(list(self.root.glob("profiles.json.bak-*")))

        unsupported = json.dumps({"schema_version": 2, "profiles": []})
        self.manifest_path.write_text(unsupported, encoding="utf-8")
        with self.assertRaisesRegex(Exception, "unsupported"):
            load_manifest(self.manifest_path)
        self.assertEqual(self.manifest_path.read_text(encoding="utf-8"), unsupported)

    def test_active_apply_updates_without_restarting(self) -> None:
        updated = json.loads(json.dumps(self.data))
        record = next(item for item in updated["profiles"] if item["id"] == "fixture-vision-12b")
        record["batch_size"] = 32
        record["ubatch_size"] = 32
        install_calls: list[str] = []
        restart = Mock()

        def install() -> None:
            install_calls.append("install")

        result = apply_manifest_transaction(
            self.manifest_path,
            updated,
            active_profile="fixture-vision-12b",
            install_units=install,
            restart_broker=restart,
        )

        self.assertTrue(result.changed)
        self.assertFalse(result.restarted)
        self.assertEqual(load_manifest(self.manifest_path)["profiles"][1]["batch_size"], 32)
        self.assertEqual(install_calls, ["install"])
        restart.assert_not_called()

    def test_inactive_apply_does_not_restart(self) -> None:
        updated = json.loads(json.dumps(self.data))
        record = next(item for item in updated["profiles"] if item["id"] == "fixture-text-7b")
        record["batch_size"] = 128
        restart = Mock()
        apply_manifest_transaction(
            self.manifest_path,
            updated,
            active_profile="fixture-vision-12b",
            install_units=Mock(),
            restart_broker=restart,
        )
        restart.assert_not_called()
        self.assertEqual(load_manifest(self.manifest_path)["profiles"][0]["batch_size"], 128)

    def test_active_runtime_build_change_does_not_restart(self) -> None:
        updated = json.loads(json.dumps(self.data))
        record = next(item for item in updated["profiles"] if item["id"] == "fixture-vision-12b")
        record["runtime"] = "llamacpp"
        record["runtime_build"] = "new-build"
        Path(record["server_binary"]).chmod(0o755)
        updated["runtime_builds"] = [
            {
                "id": "new-build",
                "runtime": record["runtime"],
                "display_name": "New build",
                "server_binary": record["server_binary"],
                "server_sha256": record["server_sha256"],
            }
        ]
        restart = Mock()
        install = Mock()

        result = apply_manifest_transaction(
            self.manifest_path,
            updated,
            active_profile="fixture-vision-12b",
            install_units=install,
            restart_broker=restart,
        )

        self.assertTrue(result.changed)
        self.assertFalse(result.restarted)
        restart.assert_not_called()
        install.assert_called_once_with()
        self.assertEqual(
            next(item for item in load_manifest(self.manifest_path)["profiles"] if item["id"] == "fixture-vision-12b")["runtime_build"],
            "new-build",
        )


if __name__ == "__main__":
    unittest.main()
