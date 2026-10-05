from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llama_broker.cli as brokerctl
import llama_broker.hygiene as hygiene
from llama_broker.hygiene import (
    CleanOptions,
    CleanRoots,
    HygieneError,
    apply_plan,
    build_plan,
    parse_lxss_base_path,
    windows_to_wsl_path,
)

MIB = 1024 * 1024
LATER = time.time() + 30 * 86400  # every fixture file looks a month old


def _profile(**paths: object) -> SimpleNamespace:
    fields = {
        "server_binary": None, "model_path": None, "mmproj_path": None,
        "model_draft": None, "effective_model_draft": None, "bundle_root": None,
        "source_repository": None, "provenance_file": None,
        "strata_artifacts_manifest_path": None, "strata_launch_config_path": None,
        "strata_engine_binary": None, "strata_artifact_files": (),
    }
    fields.update(paths)
    return SimpleNamespace(**fields)


def _write(path: Path, size: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


class HygieneTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(os.path.realpath(self.tempdir.name))
        self.roots = CleanRoots(
            profiles_path=self.base / "config" / "profiles.json",
            catalog_path=self.base / "runtimes" / "catalog.json",
            provenance_path=self.base / "config" / "provenance.json",
            models_dir=self.base / "models",
            runtimes_dir=self.base / "runtimes",
            config_dir=self.base / "config",
            cache_home=self.base / "cache",
            home=self.base / "home",
        )
        for directory in (self.roots.models_dir, self.roots.runtimes_dir, self.roots.config_dir):
            directory.mkdir(parents=True)
        self.roots.profiles_path.write_text("{}", encoding="utf-8")
        self.profiles: list[SimpleNamespace] = []
        registry = patch.object(
            hygiene.ProfileRegistry, "from_path",
            side_effect=lambda *_args, **_kwargs: SimpleNamespace(profiles=list(self.profiles)),
        )
        registry.start()
        self.addCleanup(registry.stop)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def plan(self, **options: object):
        defaults: dict[str, object] = {"min_size_bytes": MIB}
        defaults.update(options)
        return build_plan(
            self.roots, CleanOptions(**defaults), now=LATER, running_executables=lambda: (),
        )

    def candidate_paths(self, plan, category: str) -> set[Path]:
        return {c.path for c in plan.candidates if c.category == category}


class CandidateDiscoveryTests(HygieneTestCase):
    def test_unused_model_subtrees_are_candidates_and_pinned_files_are_kept(self) -> None:
        models = self.roots.models_dir
        used = _write(models / "qwen" / "iq2" / "model.gguf", 2 * MIB)
        _write(models / "qwen" / "iq2" / "README.md", 10)
        _write(models / "qwen" / "iq2" / "old-pack" / "experts.bin", 2 * MIB)
        _write(models / "gemma" / "unused.gguf", 2 * MIB)
        self.profiles.append(_profile(model_path=used))

        plan = self.plan(categories=("models",))

        self.assertEqual(
            self.candidate_paths(plan, "models"),
            {models / "qwen" / "iq2" / "old-pack", models / "gemma"},
        )
        self.assertTrue(all(c.action == "delete" for c in plan.candidates))

    def test_an_archive_dir_switches_only_models_and_backups_to_archiving(self) -> None:
        _write(self.roots.models_dir / "old.gguf", 2 * MIB)
        _write(self.roots.runtimes_dir / "llama-old" / "bin" / "llama-server")
        backup = _write(self.roots.config_dir / "profiles.json.bak-1")
        os.utime(backup, (1_000_000, 1_000_000))

        plan = self.plan(keep_backups=0, archive=True)

        self.assertEqual(
            {c.category: c.action for c in plan.candidates},
            {"models": "archive", "runtimes": "delete", "backups": "archive"},
        )

    def test_split_gguf_shards_are_protected_by_the_first_shard(self) -> None:
        models = self.roots.models_dir
        first = _write(models / "m" / "q-00001-of-00003.gguf", 2 * MIB)
        _write(models / "m" / "q-00002-of-00003.gguf", 2 * MIB)
        _write(models / "m" / "q-00003-of-00003.gguf", 2 * MIB)
        self.profiles.append(_profile(model_path=first))

        self.assertEqual(self.plan(categories=("models",)).candidates, ())

    def test_any_path_mentioned_in_the_catalog_is_protected(self) -> None:
        models = self.roots.models_dir
        pack = models / "m" / "pack"
        _write(pack / "experts.bin", 2 * MIB)
        self.roots.catalog_path.write_text(
            json.dumps({"runtimes": [{"build_options": {"note": f"--pack={pack}"}}]}),
            encoding="utf-8",
        )

        self.assertEqual(self.plan(categories=("models",)).candidates, ())

    def test_a_reference_to_the_model_root_does_not_hide_its_contents(self) -> None:
        _write(self.roots.models_dir / "stale.gguf", 2 * MIB)
        self.roots.catalog_path.write_text(
            json.dumps({"inventory": [{"root": str(self.roots.models_dir)}]}), encoding="utf-8"
        )

        plan = self.plan(categories=("models",))

        self.assertEqual(self.candidate_paths(plan, "models"), {self.roots.models_dir / "stale.gguf"})

    def test_keep_paths_and_recent_files_are_left_alone(self) -> None:
        kept = _write(self.roots.models_dir / "kept" / "weights.safetensors", 2 * MIB)
        fresh = _write(self.roots.models_dir / "fresh.gguf", 2 * MIB)

        plan = build_plan(
            self.roots,
            CleanOptions(categories=("models",), min_size_bytes=MIB, keep_paths=(kept.parent,)),
            running_executables=lambda: (),
        )

        self.assertEqual(plan.candidates, ())
        self.assertEqual([s.path for s in plan.skipped], [fresh])

    def test_orphaned_runtime_bundles_are_deleted_and_registered_ones_kept(self) -> None:
        runtimes = self.roots.runtimes_dir
        registered = runtimes / "llama-abc"
        _write(registered / "bin" / "llama-server")
        _write(runtimes / "llama-old" / "bin" / "llama-server")
        builds = self.base / "builds"
        current = builds / "strata-runtime-bundle-v6"
        _write(current / "bin" / "strata-server")
        _write(builds / "strata-runtime-bundle-v5" / "bin" / "strata-server")
        _write(builds / "strata-runtime-bundle-v4" / "src" / "only-source.txt")
        _write(builds / "other-runtime-bundle" / "bin" / "x")
        _write(builds / "strata-cuda-build" / "bin" / "strata")
        self.profiles.append(_profile(server_binary=registered / "bin" / "llama-server", bundle_root=current))

        plan = self.plan(categories=("runtimes",))

        self.assertEqual(
            self.candidate_paths(plan, "runtimes"),
            {runtimes / "llama-old", builds / "strata-runtime-bundle-v5"},
        )
        self.assertTrue(all(c.action == "delete" for c in plan.candidates))

    def test_a_bundle_a_running_process_executes_from_is_kept(self) -> None:
        orphan = self.roots.runtimes_dir / "llama-old"
        binary = _write(orphan / "bin" / "llama-server")

        plan = build_plan(
            self.roots, CleanOptions(categories=("runtimes",)), now=LATER,
            running_executables=lambda: (binary,),
        )

        self.assertEqual(plan.candidates, ())
        self.assertIn("running process", plan.skipped[0].reason)

    def test_only_backups_beyond_the_newest_are_candidates(self) -> None:
        config = self.roots.config_dir
        for index in range(4):
            backup = _write(config / f"profiles.json.bak-{index}")
            os.utime(backup, (1_000_000 + index, 1_000_000 + index))
        _write(config / "profiles.json.tmp")

        plan = self.plan(categories=("backups",), keep_backups=2)

        self.assertEqual(
            self.candidate_paths(plan, "backups"),
            {config / "profiles.json.bak-0", config / "profiles.json.bak-1"},
        )

    def test_caches_count_only_what_deleting_frees(self) -> None:
        uv = self.roots.cache_home / "uv"
        _write(uv / "own.whl", 3 * MIB)
        shared = _write(uv / "shared.whl", 5 * MIB)
        os.link(shared, self.base / "venv-copy.whl")
        _write(self.roots.cache_home / "huggingface" / "hub" / "models--x" / "w.bin", MIB)
        _write(self.roots.cache_home / "huggingface" / "hub" / "datasets--y" / "d.parquet", MIB)

        self.assertEqual(self.plan().candidates, ())
        plan = self.plan(categories=("caches",))

        sizes = {c.path: c.size_bytes for c in plan.candidates}
        self.assertEqual(sizes[uv], 3 * MIB)
        self.assertIn(self.roots.cache_home / "huggingface" / "hub" / "datasets--y", sizes)
        self.assertNotIn(self.roots.cache_home / "huggingface" / "hub" / "models--x", sizes)

    def test_unloadable_profiles_refuse_to_plan(self) -> None:
        with patch.object(hygiene.ProfileRegistry, "from_path", side_effect=hygiene.ProfileError("bad")):
            with self.assertRaisesRegex(HygieneError, "do not load"):
                self.plan()


class ApplyTests(HygieneTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.archive = self.base / "archive"
        self.archive.mkdir()

    def apply(self, plan, **kwargs: object):
        options = {
            "archive_dir": self.archive,
            "still_unreferenced": lambda _candidate: True,
            "require_other_filesystem": False,
        }
        options.update(kwargs)
        return apply_plan(plan, **options)

    def test_model_files_and_trees_are_archived_verified_and_removed(self) -> None:
        models = self.roots.models_dir
        single = _write(models / "a" / "old.gguf", 2 * MIB)
        pack = models / "b" / "pack"
        _write(pack / "experts.bin", 2 * MIB)
        (pack / "latest").symlink_to("experts.bin")

        result = self.apply(self.plan(categories=("models",), archive=True))

        self.assertEqual(result.failed, [])
        self.assertFalse(single.exists())
        self.assertFalse((models / "b").exists())
        manifest = json.loads((result.run_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        by_original = {entry["original_path"]: entry for entry in manifest["entries"]}
        self.assertEqual(by_original[str(models / "a")]["kind"], "tar")
        with tarfile.open(by_original[str(models / "b")]["archive_path"]) as archive:
            link = archive.getmember("b/pack/latest")
            self.assertTrue(link.issym())
            self.assertEqual(link.linkname, "experts.bin")

    def test_without_an_archive_dir_models_are_deleted(self) -> None:
        model = _write(self.roots.models_dir / "old.gguf", 2 * MIB)

        result = self.apply(self.plan(categories=("models",)), archive_dir=None)

        self.assertFalse(model.exists())
        self.assertEqual([entry["path"] for entry in result.deleted], [str(model)])
        self.assertIsNone(result.run_dir)

    def test_runtime_bundles_are_deleted_without_an_archive_dir(self) -> None:
        orphan = self.roots.runtimes_dir / "llama-old"
        _write(orphan / "bin" / "llama-server")

        result = self.apply(self.plan(categories=("runtimes",)), archive_dir=None)

        self.assertFalse(orphan.exists())
        self.assertEqual(len(result.deleted), 1)
        self.assertIsNone(result.run_dir)

    def test_missing_archive_dir_fails_before_anything_is_deleted(self) -> None:
        orphan = self.roots.runtimes_dir / "llama-old"
        _write(orphan / "bin" / "llama-server")
        _write(self.roots.models_dir / "old.gguf", 2 * MIB)

        with self.assertRaisesRegex(HygieneError, "no archive directory"):
            self.apply(self.plan(categories=("models", "runtimes"), archive=True), archive_dir=None)

        self.assertTrue(orphan.exists())

    def test_a_failed_verification_keeps_the_original(self) -> None:
        model = _write(self.roots.models_dir / "old.gguf", 2 * MIB)

        with patch.object(hygiene, "_sha256_stream", return_value="0" * 64):
            result = self.apply(self.plan(categories=("models",), archive=True))

        self.assertTrue(model.exists())
        self.assertEqual(len(result.failed), 1)
        self.assertIn("does not match", result.failed[0]["error"])

    def test_a_path_a_profile_starts_using_is_kept(self) -> None:
        orphan = self.roots.runtimes_dir / "llama-old"
        _write(orphan / "bin" / "llama-server")

        result = self.apply(self.plan(categories=("runtimes",)), still_unreferenced=lambda _c: False)

        self.assertTrue(orphan.exists())
        self.assertIn("now references", result.skipped[0]["reason"])

    def test_archiving_onto_the_same_filesystem_is_refused(self) -> None:
        model = _write(self.roots.models_dir / "old.gguf", 2 * MIB)

        result = self.apply(self.plan(categories=("models",), archive=True), require_other_filesystem=True)

        self.assertTrue(model.exists())
        self.assertIn("same filesystem", result.failed[0]["error"])


class WslDiskTests(unittest.TestCase):
    def test_base_path_is_found_for_the_named_distribution(self) -> None:
        output = (
            "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss\n"
            "    DefaultDistribution    REG_SZ    {a}\n\n"
            "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss\\{a}\n"
            "    BasePath    REG_SZ    \\\\?\\C:\\Users\\example\\wsl\\Debian\n"
            "    DistributionName    REG_SZ    Debian\n\n"
            "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss\\{b}\n"
            "    BasePath    REG_SZ    C:\\Users\\example\\AppData\\Local\\Packages\\Ubuntu\\LocalState\n"
            "    DistributionName    REG_SZ    Ubuntu\n"
        )
        self.assertEqual(
            parse_lxss_base_path(output, "ubuntu"),
            "C:\\Users\\example\\AppData\\Local\\Packages\\Ubuntu\\LocalState",
        )
        self.assertEqual(parse_lxss_base_path(output, "Debian"), "C:\\Users\\example\\wsl\\Debian")
        self.assertIsNone(parse_lxss_base_path(output, "Arch"))

    def test_windows_paths_map_to_drvfs_mounts(self) -> None:
        self.assertEqual(windows_to_wsl_path("C:\\Users\\example\\ext4.vhdx"), Path("/mnt/c/Users/example/ext4.vhdx"))
        self.assertIsNone(windows_to_wsl_path("\\\\server\\share"))


class BrokerctlCleanTests(HygieneTestCase):
    def run_cli(self, *argv: str, terminal: bool = False, answer: object = None) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        prompt = patch("builtins.input", side_effect=[answer] if answer is not None else AssertionError("prompted"))
        with (
            prompt,
            patch.object(brokerctl, "_interactive_terminal", return_value=terminal),
            patch.object(brokerctl, "_clean_roots", return_value=self.roots),
            patch.object(brokerctl, "CONFIG_DIR", self.roots.config_dir),
            patch.object(brokerctl, "wsl_disk_report", return_value=None),
            patch.object(brokerctl, "build_plan", side_effect=lambda roots, options: build_plan(
                roots, options, now=LATER, running_executables=lambda: ())),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = brokerctl.main(["clean", *argv])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_json_dry_run_lists_candidates_and_changes_nothing(self) -> None:
        orphan = self.roots.runtimes_dir / "llama-old"
        _write(orphan / "bin" / "llama-server")

        code, stdout, _stderr = self.run_cli("--json")

        report = json.loads(stdout)
        self.assertEqual(code, 0)
        self.assertEqual(report["schema_version"], 1)
        self.assertTrue(report["dry_run"])
        self.assertEqual([c["path"] for c in report["candidates"]], [str(orphan)])
        self.assertTrue(orphan.exists())

    def test_dry_run_deletes_by_default_and_archives_with_an_archive_dir(self) -> None:
        _write(self.roots.models_dir / "old.gguf", 300 * MIB)

        _code, deleting, _stderr = self.run_cli("models", "--json")
        _code, archiving, _stderr = self.run_cli("models", "--json", "--archive-dir", str(self.base))
        with patch.dict(os.environ, {"LLAMA_BROKER_ARCHIVE_DIR": str(self.base)}):
            _code, from_env, _stderr = self.run_cli("models", "--json")

        self.assertEqual(json.loads(deleting)["candidates"][0]["action"], "delete")
        self.assertEqual(json.loads(archiving)["candidates"][0]["action"], "archive")
        self.assertEqual(json.loads(from_env)["candidates"][0]["action"], "archive")
        self.assertTrue((self.roots.models_dir / "old.gguf").exists())

    def test_a_dry_run_at_a_terminal_cleans_after_yes(self) -> None:
        orphan = self.roots.runtimes_dir / "llama-old"
        _write(orphan / "bin" / "llama-server")

        code, stdout, _stderr = self.run_cli(terminal=True, answer="y")

        self.assertEqual(code, 0)
        self.assertFalse(orphan.exists())
        self.assertIn("Deleted 1 item(s)", stdout)

    def test_a_dry_run_at_a_terminal_changes_nothing_unless_yes(self) -> None:
        orphan = self.roots.runtimes_dir / "llama-old"
        _write(orphan / "bin" / "llama-server")

        for answer in ("", "n", "yes please", EOFError()):
            code, stdout, _stderr = self.run_cli(terminal=True, answer=answer)
            self.assertEqual(code, 0)
            self.assertTrue(orphan.exists())
            self.assertIn("Nothing changed.", stdout)

    def test_scripts_json_and_apply_never_prompt(self) -> None:
        orphan = self.roots.runtimes_dir / "llama-old"
        _write(orphan / "bin" / "llama-server")

        _code, stdout, _stderr = self.run_cli()
        self.assertIn("Run the same command with --apply", stdout)
        self.run_cli("--json", terminal=True)
        self.assertTrue(orphan.exists())

        self.run_cli("--apply", terminal=True)
        self.assertFalse(orphan.exists())

    def test_keep_file_protects_listed_paths(self) -> None:
        model = _write(self.roots.models_dir / "keep-me" / "w.safetensors", 300 * MIB)
        (self.roots.config_dir / "clean-keep").write_text(f"# mine\n{model.parent}\n", encoding="utf-8")

        code, stdout, _stderr = self.run_cli("models", "--json")

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["candidates"], [])

    def test_unknown_category_is_an_error(self) -> None:
        code, _stdout, stderr = self.run_cli("models", "everything")

        self.assertEqual(code, 1)
        self.assertIn("unknown clean category everything", stderr)


if __name__ == "__main__":
    unittest.main()
