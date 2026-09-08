from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llama_broker.broker_tui import (
    EDITOR_FIELDS,
    TUIModel,
    _curses_prompt,
    _curses_select,
    _read_curses_key,
    run_guided_wizard,
)
from llama_broker.profile_config import merge_manifest_drafts


class BrokerTUIModelTests(unittest.TestCase):
    def test_runtime_family_reveals_backend_build_and_applies_catalog_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            binary = Path(tempdir) / "llama-server"
            binary.write_bytes(b"beellama")
            binary.chmod(0o755)
            server_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
            manifest = {
                "runtime_builds": [
                    {
                        "id": "beellama",
                        "runtime": "llamacpp",
                        "display_name": "BeeLlama",
                        "server_binary": str(binary),
                        "server_sha256": server_sha256,
                    },
                    {
                        "id": "atomic-vllm",
                        "runtime": "vllm",
                        "display_name": "Atomic vLLM",
                        "server_binary": str(binary),
                        "server_sha256": server_sha256,
                    },
                ],
                "profiles": [
                    {
                        "id": "one",
                        "alias": "one",
                        "runtime": "llamacpp",
                        "server_binary": str(binary),
                        "server_sha256": server_sha256,
                    }
                ],
            }

            model = TUIModel(manifest)

            family = next(item for item in model.visible_fields if item.key == "runtime_family")
            build = next(item for item in model.visible_fields if item.key == "runtime_build")
            self.assertEqual(family.options, ("llamacpp — 1 build", "vllm — 1 build", "(custom/manual)"))
            self.assertEqual(model.field_value("runtime_family"), "llamacpp — 1 build")
            self.assertTrue(any(option.startswith("BeeLlama — beellama\n") for option in build.options))
            model.set_field("runtime_build", build.options[0])
            self.assertEqual(model.current["runtime_build"], "beellama")
            self.assertEqual(model.current["server_binary"], str(binary.resolve()))
            self.assertEqual(model.current["server_sha256"], server_sha256)

    def test_runtime_family_and_backend_build_are_dependent(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            binary = Path(tempdir) / "llama-server"
            binary.write_bytes(b"shared runtime")
            binary.chmod(0o755)
            server_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
            manifest = {
                "runtime_builds": [
                    {
                        "id": "beellama",
                        "runtime": "llamacpp",
                        "display_name": "BeeLlama",
                        "server_binary": str(binary),
                        "server_sha256": server_sha256,
                    },
                    {
                        "id": "atomic",
                        "runtime": "llamacpp",
                        "display_name": "Atomic",
                        "server_binary": str(binary),
                        "server_sha256": server_sha256,
                    },
                    {
                        "id": "atomic-vllm",
                        "runtime": "vllm",
                        "display_name": "Atomic vLLM",
                        "server_binary": str(binary),
                        "server_sha256": server_sha256,
                    },
                ],
                "profiles": [
                    {
                        "id": "one",
                        "alias": "one",
                        "runtime": "llamacpp",
                        "runtime_build": "beellama",
                        "server_binary": str(binary),
                        "server_sha256": server_sha256,
                    }
                ],
            }

            model = TUIModel(manifest)
            keys = {field.key for field in model.visible_fields}
            family = next(field for field in model.visible_fields if field.key == "runtime_family")

            self.assertIn("runtime_family", keys)
            self.assertIn("runtime_build", keys)
            self.assertNotIn("runtime", keys)
            self.assertEqual(
                family.options,
                ("llamacpp — 2 builds", "vllm — 1 build", "(custom/manual)"),
            )
            self.assertEqual(model.field_value("runtime_family"), "llamacpp — 2 builds")

            model.set_field("runtime_build", "Atomic — atomic")
            assert model.current is not None
            self.assertEqual(model.current["runtime"], "llamacpp")
            self.assertEqual(model.current["runtime_build"], "atomic")

            model.set_field("runtime_family", "vllm")
            self.assertEqual(model.current["runtime"], "vllm")
            self.assertEqual(model.current["runtime_build"], "atomic-vllm")

    def test_entering_and_canceling_a_field_keeps_model_clean(self) -> None:
        manifest = {
            "profiles": [
                {"id": "one", "alias": "one", "context_size": 32768, "capabilities": ["text"]}
            ]
        }
        model = TUIModel(manifest)

        model.set_field("context_size", 32768)
        self.assertFalse(model.dirty)
        model.set_field("context_size", 65536)
        self.assertTrue(model.dirty)
        model.set_field("context_size", 32768)
        self.assertFalse(model.dirty)

    def test_multiline_and_long_field_values_are_safe_for_root_list(self) -> None:
        from llama_broker.broker_tui import _format_field_value, _truncate_field_value

        value = _format_field_value(
            "BeeLlama — beellama\n"
            "  binary: /example/user/.local/share/runtimes/beellama.cpp/runtime/bin/llama-server\n"
            "  sha256: deadbeef"
        )

        self.assertNotIn("\n", value)
        self.assertLessEqual(len(_truncate_field_value(value, 32)), 32)

    def test_curses_prompt_edits_existing_text_and_refreshes(self) -> None:
        class FakeWindow:
            def __init__(self, keys: list[int]) -> None:
                self.keys = iter(keys)
                self.refresh_count = 0

            def getmaxyx(self) -> tuple[int, int]:
                return (24, 80)

            def move(self, _row: int, _column: int) -> None:
                pass

            def clrtoeol(self) -> None:
                pass

            def addnstr(self, _row: int, _column: int, _text: str, _count: int) -> None:
                pass

            def refresh(self) -> None:
                self.refresh_count += 1

            def getch(self) -> int:
                return next(self.keys)

        window = FakeWindow([21, *map(ord, "New display name"), 10])

        value = _curses_prompt(window, "Display name", "Old display name")

        self.assertEqual(value, "New display name")
        self.assertGreater(window.refresh_count, 0)

        model = TUIModel({"profiles": [{"id": "one", "display_name": "Old display name"}]})
        model.set_field("display_name", value)
        assert model.current is not None
        self.assertEqual(model.current["display_name"], "New display name")

    def test_curses_select_clips_multiline_choices_to_window(self) -> None:
        import curses

        class BoundsWindow:
            def __init__(self, height: int, width: int, keys: list[int]) -> None:
                self.height = height
                self.width = width
                self.keys = iter(keys)
                self.rows: list[int] = []

            def getmaxyx(self) -> tuple[int, int]:
                return self.height, self.width

            def erase(self) -> None:
                pass

            def addnstr(self, row: int, column: int, _text: str, _count: int, *_args: int) -> None:
                if not (0 <= row < self.height and 0 <= column < self.width):
                    raise AssertionError(
                        f"draw outside {self.height}x{self.width}: row={row}, column={column}"
                    )
                self.rows.append(row)

            def getch(self) -> int:
                return next(self.keys)

        choices = tuple(
            f"build-{index}\n  runtime: llamacpp\n  binary: /runtime/{index}\n  sha256: {'0' * 64}"
            for index in range(6)
        )
        window = BoundsWindow(height=12, width=80, keys=[10])

        value, cancelled = _curses_select(window, "Backend build", choices[0], choices)

        self.assertEqual(value, choices[0])
        self.assertFalse(cancelled)
        self.assertTrue(window.rows)
        self.assertLess(max(window.rows), window.height)

        navigation_window = BoundsWindow(
            height=12,
            width=80,
            keys=[curses.KEY_DOWN] * (len(choices) - 1) + [10],
        )
        value, cancelled = _curses_select(
            navigation_window, "Backend build", choices[0], choices
        )

        self.assertEqual(value, choices[-1])
        self.assertFalse(cancelled)
        self.assertLess(max(navigation_window.rows), navigation_window.height)

    def test_alt_arrow_sequences_do_not_become_escape(self) -> None:
        import curses

        class FakeWindow:
            def __init__(self, keys: list[int]) -> None:
                self.keys = iter(keys)
                self.timeouts: list[int] = []

            def getch(self) -> int:
                try:
                    return next(self.keys)
                except StopIteration:
                    return -1

            def timeout(self, milliseconds: int) -> None:
                self.timeouts.append(milliseconds)

        sequences = (
            (bytes([27]) + b"[1;3A", curses.KEY_UP),
            (bytes([27]) + b"[1;3B", curses.KEY_DOWN),
            (bytes([27]) + b"[1;3C", curses.KEY_RIGHT),
            (bytes([27]) + b"[1;3D", curses.KEY_LEFT),
            (bytes([27]) + b"OA", curses.KEY_UP),
        )
        for raw_sequence, expected in sequences:
            with self.subTest(sequence=raw_sequence):
                window = FakeWindow(list(raw_sequence))
                self.assertEqual(_read_curses_key(window), expected)
                self.assertEqual(window.timeouts, [50, -1])

    def test_standalone_escape_remains_escape(self) -> None:
        class FakeWindow:
            def __init__(self) -> None:
                self.timeouts: list[int] = []

            def getch(self) -> int:
                return 27 if not self.timeouts else -1

            def timeout(self, milliseconds: int) -> None:
                self.timeouts.append(milliseconds)

        window = FakeWindow()
        self.assertEqual(_read_curses_key(window), 27)
        self.assertEqual(window.timeouts, [50, -1])

    def test_curses_prompt_cancels_cleanly_on_keyboard_interrupt(self) -> None:
        class InterruptingWindow:
            def getmaxyx(self) -> tuple[int, int]:
                return (24, 80)

            def move(self, _row: int, _column: int) -> None:
                pass

            def clrtoeol(self) -> None:
                pass

            def addnstr(self, _row: int, _column: int, _text: str, _count: int) -> None:
                pass

            def refresh(self) -> None:
                pass

            def getch(self) -> int:
                raise KeyboardInterrupt

        self.assertEqual(
            _curses_prompt(InterruptingWindow(), "Display name", "Old display name"),
            "Old display name",
        )

    def test_runtime_change_keeps_runtime_build_selection_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            binary = Path(tempdir) / "llama-server"
            binary.write_bytes(b"shared runtime")
            binary.chmod(0o755)
            server_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
            manifest = {
                "runtime_builds": [
                    {
                        "id": "beellama",
                        "runtime": "llamacpp",
                        "display_name": "BeeLlama",
                        "server_binary": str(binary),
                        "server_sha256": server_sha256,
                    },
                    {
                        "id": "atomic-vllm",
                        "runtime": "vllm",
                        "display_name": "Atomic vLLM",
                        "server_binary": str(binary),
                        "server_sha256": server_sha256,
                    },
                ],
                "profiles": [
                    {
                        "id": "one",
                        "alias": "one",
                        "runtime": "llamacpp",
                        "runtime_build": "beellama",
                        "server_binary": str(binary),
                        "server_sha256": server_sha256,
                    }
                ],
            }

            model = TUIModel(manifest)
            model.set_field("runtime_family", "vllm")

            field = next(item for item in model.visible_fields if item.key == "runtime_family")
            self.assertEqual(field.options, ("llamacpp — 1 build", "vllm — 1 build", "(custom/manual)"))
            self.assertEqual(model.field_value("runtime_family"), "vllm — 1 build")
            assert model.current is not None
            self.assertEqual(model.current["runtime_build"], "atomic-vllm")

    def test_custom_manual_clears_build_and_reveals_manual_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            binary = Path(tempdir) / "llama-server"
            binary.write_bytes(b"runtime")
            binary.chmod(0o755)
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            manifest = {
                "runtime_builds": [
                    {
                        "id": "beellama",
                        "runtime": "llamacpp",
                        "display_name": "BeeLlama",
                        "server_binary": str(binary),
                        "server_sha256": digest,
                    }
                ],
                "profiles": [
                    {
                        "id": "one",
                        "alias": "one",
                        "runtime": "llamacpp",
                        "runtime_build": "beellama",
                        "server_binary": str(binary),
                        "server_sha256": digest,
                    }
                ],
            }
            model = TUIModel(manifest)
            model.set_field("runtime_family", "(custom/manual)")
            assert model.current is not None
            self.assertNotIn("runtime_build", model.current)
            keys = {field.key for field in model.visible_fields}
            self.assertIn("runtime", keys)
            self.assertNotIn("runtime_build", keys)
            self.assertTrue(model.dirty)

    def test_family_filter_keeps_invalid_builds_and_sorts_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            valid_binary = Path(tempdir) / "valid-server"
            valid_binary.write_bytes(b"valid")
            valid_binary.chmod(0o755)
            digest = hashlib.sha256(valid_binary.read_bytes()).hexdigest()
            manifest = {
                "runtime_builds": [
                    {
                        "id": "z-valid",
                        "runtime": "llamacpp",
                        "display_name": "Zed",
                        "server_binary": str(valid_binary),
                        "server_sha256": digest,
                    },
                    {
                        "id": "a-invalid",
                        "runtime": "llamacpp",
                        "display_name": "Alpha",
                        "server_binary": str(Path(tempdir) / "missing-server"),
                        "server_sha256": "0" * 64,
                    },
                    {
                        "id": "vllm-only-invalid",
                        "runtime": "vllm",
                        "display_name": "VLLM",
                        "server_binary": str(Path(tempdir) / "missing-vllm"),
                        "server_sha256": "1" * 64,
                    },
                ],
                "profiles": [{"id": "one", "alias": "one", "runtime": "llamacpp"}],
            }
            model = TUIModel(manifest)
            model.set_field("runtime_family", "llamacpp")
            family = next(field for field in model.visible_fields if field.key == "runtime_family")
            self.assertEqual(family.options, ("llamacpp — 2 builds", "(custom/manual)"))
            build = next(field for field in model.visible_fields if field.key == "runtime_build")
            self.assertTrue(build.options[0].startswith("Alpha — a-invalid"))
            self.assertTrue(build.options[1].startswith("Zed — z-valid"))

    def test_catalog_is_snapshotted_for_editor_session(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            binary = Path(tempdir) / "llama-server"
            binary.write_bytes(b"runtime")
            binary.chmod(0o755)
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            manifest = {
                "runtime_builds": [
                    {
                        "id": "one-build",
                        "runtime": "llamacpp",
                        "display_name": "One",
                        "server_binary": str(binary),
                        "server_sha256": digest,
                    }
                ],
                "profiles": [{"id": "one", "alias": "one", "runtime": "llamacpp"}],
            }
            model = TUIModel(manifest)
            model.manifest["runtime_builds"].append(
                {
                    "id": "two-build",
                    "runtime": "vllm",
                    "display_name": "Two",
                    "server_binary": str(binary),
                    "server_sha256": digest,
                }
            )
            family = next(field for field in model.visible_fields if field.key == "runtime_family")
            self.assertNotIn("vllm", family.options)

    def test_no_valid_catalog_builds_keeps_custom_manual_available(self) -> None:
        model = TUIModel({"runtime_builds": [], "profiles": [{"id": "one", "alias": "one"}]})
        family = next(field for field in model.visible_fields if field.key == "runtime_family")
        self.assertEqual(
            family.options,
            ("(no valid catalog builds available)", "(custom/manual)"),
        )

    def test_manifest_merge_preserves_external_fields_and_reports_conflicts(self) -> None:
        base = {
            "profiles": [
                {"id": "one", "alias": "one", "context_size": 32768, "parallel": 1}
            ]
        }
        disk = {
            "profiles": [
                {
                    "id": "one",
                    "alias": "one-on-disk",
                    "context_size": 65536,
                    "parallel": 1,
                }
            ],
            "external_setting": True,
        }
        draft = {
            "profiles": [
                {"id": "one", "alias": "one", "context_size": 131072, "parallel": 1}
            ]
        }

        merged, conflicts = merge_manifest_drafts(base, disk, draft)

        self.assertEqual(merged["external_setting"], True)
        profile = merged["profiles"][0]
        self.assertEqual(profile["alias"], "one-on-disk")
        self.assertEqual(profile["context_size"], 131072)
        self.assertEqual(conflicts, ["profiles[one].context_size"])

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
