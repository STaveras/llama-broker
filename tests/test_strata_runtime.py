from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import llama_broker.cli as brokerctl
from llama_broker.broker_profiles import ProfileError, ProfileRegistry
from llama_broker.broker_security import SecurityConfig
from llama_broker.broker_tui import TUIModel
from llama_broker.cli import _set_scalar_field
from llama_broker.daemon import (
    BackendControllerError,
    BrokerApplication,
    _verify_profile_artifacts,
    backend_url_for_runtime,
    route_payload,
)
from llama_broker.profile_config import write_manifest_atomically


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StrataProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.model_root = self.root / "models" / "iq2-xs"
        self.model_root.mkdir(parents=True)
        self.bundle = self.root / "runtime-bundle"
        self.source_repo = self.root / "strata-source"
        self.source_repo.mkdir()
        self.strata_root = self.bundle / "strata"
        self.binary = self.bundle / "bin" / "strata-server"
        self.engine = self.strata_root / "bin" / "strata"
        self.expert_profile = self.strata_root / "data" / "expert-profile.bin"
        for path, data in (
            (self.binary, b"strata bridge"),
            (self.engine, b"strata engine"),
            (self.expert_profile, b"expert profile"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.binary.chmod(0o755)
        self.engine.chmod(0o755)
        serve_dir = self.strata_root / "serve"
        serve_dir.mkdir(parents=True, exist_ok=True)
        (serve_dir / "__init__.py").write_text("", encoding="utf-8")
        (serve_dir / "server.py").write_text("# pinned server module\n", encoding="utf-8")
        source_serve_dir = self.source_repo / "serve"
        source_serve_dir.mkdir(parents=True, exist_ok=True)
        (source_serve_dir / "__init__.py").write_text("", encoding="utf-8")
        (source_serve_dir / "server.py").write_text("# pinned server module\n", encoding="utf-8")
        subprocess.run(["git", "init", "--quiet", str(self.source_repo)], check=True)
        subprocess.run(["git", "-C", str(self.source_repo), "add", "serve"], check=True)
        subprocess.run(
            ["git", "-C", str(self.source_repo), "-c", "user.name=Fixture",
             "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "fixture"],
            check=True,
        )
        self.source_commit = subprocess.run(
            ["git", "-C", str(self.source_repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        self.shard1 = self.model_root / "model-00001-of-00002.gguf"
        self.shard2 = self.model_root / "model-00002-of-00002.gguf"
        self.shard1.write_bytes(b"shard one")
        self.shard2.write_bytes(b"shard two")
        self.pack = self.model_root / "strata-pack"
        self.pack.mkdir()
        self.pack_files = {}
        for relative, data in (
            ("dense.bin", b"dense"),
            ("index.txt", b"index"),
            ("native_experts.txt", b"experts"),
            ("experts.bin", b"expert arena source"),
            ("tokenizer/vocab.json", b"{}"),
            ("tokenizer/merges.txt", b"merges"),
            ("tokenizer/chat_template.jinja", b"{{ messages }}"),
        ):
            path = self.pack / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.pack_files[relative] = path
        self.manifest_path = self.model_root / "strata-artifacts.json"
        self.launch_path = self.model_root / "strata-launch.json"
        self.mtp_root = self.model_root / "mtp"
        self.mtp_revision = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
        self.mtp_files = {}
        raw_tensor = b"pinned mtp tensor"
        tensor_sha = hashlib.sha256(raw_tensor).hexdigest()
        mtp_payloads = {
            "mtp-inventory.json": json.dumps({
                "repo": f"https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/{self.mtp_revision}/",
                "total_bytes": len(raw_tensor),
                "tensors": [{"name": "mtp.test.weight", "bytes": len(raw_tensor)}],
            }).encode(),
            "mtp-manifest.json": json.dumps([{
                "name": "mtp.test.weight", "file": "tensors/mtp.test.weight.bin",
                "sha256": tensor_sha, "bytes": len(raw_tensor),
            }]).encode(),
            "mtp-q2_0.gguf": b"packed mtp gguf",
            "tensors/mtp.test.weight.bin": raw_tensor,
            "rt/experts.bin": b"mtp runtime experts",
            "rt/dense.bin": b"mtp runtime dense",
            "rt/dense.txt": b"mtp runtime index",
        }
        for relative, data in mtp_payloads.items():
            path = self.mtp_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.mtp_files[relative] = path
        self.profile_id = "strata-iq2-xs"
        self.alias = "qwen3.8-flash-next-iq2-xs"
        self._write_manifest()
        self._write_launch_config()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_manifest(self) -> None:
        self.manifest = {
            "schema_version": 1,
            "runtime_family": "strata",
            "model": {
                "name": "Qwen3.8-Flash-Next",
                "quant": "IQ2_XS",
                "license": "apache-2.0",
                "repository": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                "revision": "a" * 40,
                "shards": [
                    {"path": self.shard1.name, "size": self.shard1.stat().st_size,
                     "sha256": sha256(self.shard1)},
                    {"path": self.shard2.name, "size": self.shard2.stat().st_size,
                     "sha256": sha256(self.shard2)},
                ],
            },
            "pack": {
                "format": "strata-native-iq",
                "path": self.pack.name,
                "source_repository": "Niko1221/Strata",
                "source_commit": self.source_commit,
                "mtp": {
                    "repository": "Qwen/Qwen3.8-Flash-Next",
                    "revision": self.mtp_revision,
                    "path": self.mtp_root.name,
                    "runtime_path": f"{self.mtp_root.name}/rt",
                    "files": [
                        {"path": f"{self.mtp_root.name}/{relative}", "size": p.stat().st_size,
                         "sha256": sha256(p)}
                        for relative, p in sorted(self.mtp_files.items())
                    ],
                },
                "files": [
                    {"path": f"{self.pack.name}/{relative}", "size": p.stat().st_size,
                     "sha256": sha256(p)}
                    for relative, p in self.pack_files.items()
                ],
            },
        }
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def _write_launch_config(self) -> None:
        self.launch_config = {
            "schema_version": 1,
            "model_name": self.alias,
            "engine_binary": str(self.engine),
            "engine_sha256": sha256(self.engine),
            "expert_profile_sha256": sha256(self.expert_profile),
            "engine_args": [
                "--pack", str(self.pack),
                "--native", str(self.shard1),
                "--ple-gguf", str(self.shard2),
                "--expert-profile", str(self.expert_profile),
                "--expert-cache", "auto",
                "--prefill", "auto",
                "--max-context", "8192",
                "--kv-resident", "20480",
                "--spec", "4",
                "--spec-min-p", "0.5",
                "--mtp", str(self.mtp_root / "rt"),
                "--mmap-experts",
            ],
            "working_directory": str(self.strata_root),
            "tokenizer_path": str(self.pack / "tokenizer"),
            "runtime_files": [
                {"path": path.relative_to(self.strata_root).as_posix(), "sha256": sha256(path)}
                for path in sorted(self.strata_root.rglob("*.py"))
            ],
        }
        self.launch_path.write_text(json.dumps(self.launch_config), encoding="utf-8")

    def _profile(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "id": self.profile_id,
            "alias": self.alias,
            "aliases": [self.alias],
            "display_name": "Qwen3.8 Flash Next IQ2_XS (Strata)",
            "role": "persistent",
            "runtime": "strata",
            "runtime_build": "strata-test",
            "capabilities": ["text"],
            "server_binary": str(self.binary),
            "server_sha256": sha256(self.binary),
            "model_path": str(self.shard1),
            "model_sha256": sha256(self.shard1),
            "context_size": 8192,
            "parallel": 1,
            "strata_artifacts_manifest_path": str(self.manifest_path),
            "strata_artifacts_manifest_sha256": sha256(self.manifest_path),
            "strata_launch_config_path": str(self.launch_path),
            "strata_launch_config_sha256": sha256(self.launch_path),
            "startup_timeout": 240,
            "safety": {
                "service": {
                    "type": "exec", "memory_max_mib": 20480,
                    "memory_swap_max_mib": 1024, "tasks_max": 512,
                    "kill_signal": "SIGINT", "timeout_stop_sec": 20,
                },
                "admission": {
                    "min_ram_available_mib": 6144, "max_swap_used_mib": 1024,
                    "max_swap_growth_mib": 128, "min_total_vram_mib": 12000,
                    "min_free_vram_before_start_mib": 8192,
                    "driver_vram_reserve_mib": 512, "vision_vram_reserve_mib": 0,
                    "safety_vram_reserve_mib": 1024,
                },
            },
        }
        row.update(overrides)
        return row

    def _write_profiles(
        self,
        record: dict[str, object],
        *,
        runtime_overrides: dict[str, object] | None = None,
    ) -> Path:
        path = self.root / "profiles.json"
        runtime = {
            "id": "strata-test",
            "runtime": "strata",
            "display_name": "Strata test bundle",
            "bundle_root": str(self.bundle),
            "server_binary": str(self.binary),
            "server_sha256": sha256(self.binary),
            "source_repository": str(self.source_repo),
            "source_branch": "main",
            "source_commit": self.source_commit,
            "dirty": False,
            "selected_profile_ids": [self.profile_id],
            "environment": {},
            "build_options": {},
        }
        runtime.update(runtime_overrides or {})
        path.write_text(json.dumps({"schema_version": 1, "profiles": [record],
                                    "runtime_builds": [runtime]}), encoding="utf-8")
        return path

    def test_strata_profile_accepts_bounded_resident_budget_mode(self) -> None:
        resident_launch = dict(self.launch_config)
        args = list(resident_launch["engine_args"])
        mmap_index = args.index("--mmap-experts")
        args[mmap_index:mmap_index + 1] = ["--resident-budget-gib", "9"]
        resident_launch["engine_args"] = args
        self.launch_path.write_text(json.dumps(resident_launch), encoding="utf-8")

        profile = ProfileRegistry.from_path(self._write_profiles(self._profile())).get(self.profile_id)

        self.assertEqual(profile.strata_launch_config_path, self.launch_path)

    def test_strata_profile_parses_pinned_bundle_and_model_artifacts(self) -> None:
        profile = ProfileRegistry.from_path(self._write_profiles(self._profile())).get(self.profile_id)

        self.assertEqual(profile.runtime, "strata")
        self.assertEqual(profile.parallel, 1)
        self.assertEqual(profile.model_path, self.shard1)
        self.assertEqual(profile.strata_artifacts_manifest_path, self.manifest_path)
        self.assertEqual(profile.strata_launch_config_path, self.launch_path)
        self.assertEqual(profile.bundle_root, self.bundle)

    def test_editor_rename_is_usable_without_restarting_loaded_strata_backend(self) -> None:
        path = self._write_profiles(self._profile())
        registry = ProfileRegistry.from_path(path)
        renamed = "qwen3.8-flash-next"
        original_launch = self.launch_path.read_bytes()
        controller = SimpleNamespace(
            state=lambda: {"profile": self.profile_id, "phase": "active", "generation": 1},
            replace_profiles=Mock(),
            install_units=Mock(),
            switch=Mock(side_effect=AssertionError("rename started a model")),
            restart=Mock(side_effect=AssertionError("rename restarted a model")),
            stop_all=Mock(side_effect=AssertionError("rename stopped a model")),
        )

        def make_app(current_registry: ProfileRegistry) -> BrokerApplication:
            return BrokerApplication(
                registry=current_registry, controller=controller,
                backend_host="127.0.0.1", backend_port=10000,
                backend_timeout=2.0, max_body_bytes=1024,
                backend_ready=lambda: True, security=SecurityConfig(),
                start_operation_worker=False, profiles_path=path,
            )

        def edit_in_tui(profiles_path: Path, selector: str, *, apply_callback=None) -> int:
            model = TUIModel(json.loads(profiles_path.read_text(encoding="utf-8")))
            model.set_field("alias", renamed)
            if apply_callback is None:
                write_manifest_atomically(profiles_path, model.manifest)
            else:
                apply_callback(model.manifest, {selector})
            return 0

        with patch("llama_broker.daemon.backend_serves_alias",
                   side_effect=lambda _host, _port, alias, *_args, **_kwargs: alias == self.alias):
            app = make_app(registry)
            try:
                with (
                    patch.object(brokerctl.sys.stdin, "isatty", return_value=True),
                    patch.object(brokerctl.sys.stdout, "isatty", return_value=True),
                    patch("llama_broker.broker_tui.run_tui", side_effect=edit_in_tui),
                    patch.object(brokerctl, "_is_authoritative_manifest", return_value=True),
                    patch.object(brokerctl, "create_production_controller", return_value=controller),
                    patch.object(brokerctl, "_refresh_live_profile_registry",
                                 side_effect=lambda _path: app.reload_profiles_for_operation(lambda: None)),
                ):
                    self.assertEqual(brokerctl.main(
                        ["--profiles", str(path), "profile", "edit", self.profile_id]), 0)

                profile, payload = route_payload(
                    app.registry, {"model": renamed}, active_profile=app.coordinator.active_profile)
                self.assertEqual(profile.alias, renamed)
                self.assertEqual(payload["model"], self.alias)
                self.assertEqual(app.health()["backend"]["phase"], "active")
                with app.coordinator.lease(profile.id, transient=False):
                    self.assertEqual(app.coordinator.active_profile, self.profile_id)
                self.assertEqual(self.launch_path.read_bytes(), original_launch)
                controller.switch.assert_not_called()
                controller.restart.assert_not_called()
                controller.stop_all.assert_not_called()
            finally:
                app.shutdown()

            # A new broker process must also recognize the already loaded backend
            # when its public alias differs from the pinned Strata server name.
            app = make_app(ProfileRegistry.from_path(path))
            try:
                self.assertEqual(app.coordinator.active_profile, self.profile_id)
                self.assertEqual(app.health()["backend"]["phase"], "active")
            finally:
                app.shutdown()

    def test_strata_profile_accepts_262144_context_when_launch_uses_streaming(self) -> None:
        args = list(self.launch_config["engine_args"])
        args[args.index("--max-context") + 1] = "262144"
        self.launch_config["engine_args"] = args
        self.launch_path.write_text(json.dumps(self.launch_config), encoding="utf-8")
        profile = ProfileRegistry.from_path(
            self._write_profiles(self._profile(context_size=262144))
        ).get(self.profile_id)
        self.assertEqual(profile.context_size, 262144)

    def test_strata_profile_rejects_engine_context_that_disagrees_with_profile(self) -> None:
        args = list(self.launch_config["engine_args"])
        args[args.index("--max-context") + 1] = "262144"
        self.launch_config["engine_args"] = args
        self.launch_path.write_text(json.dumps(self.launch_config), encoding="utf-8")
        with self.assertRaisesRegex(ProfileError, "does not match pinned profile data.*--max-context"):
            ProfileRegistry.from_path(self._write_profiles(self._profile()))

    def test_strata_profile_rejects_kv_resident_below_floor(self) -> None:
        args = list(self.launch_config["engine_args"])
        args[args.index("--kv-resident") + 1] = "20479"
        self.launch_config["engine_args"] = args
        self.launch_path.write_text(json.dumps(self.launch_config), encoding="utf-8")

        with self.assertRaisesRegex(ProfileError, "resident cells"):
            ProfileRegistry.from_path(self._write_profiles(self._profile()))

    def test_tui_runtime_switch_drops_fields_unsupported_by_target_runtime(self) -> None:
        record = self._profile(
            batch_size=256,
            cache_type_k="f16",
            gpu_layers="all",
            environment={"GGML_CUDA_PDL": "0"},
            extra_args=["--no-warmup"],
        )
        manifest = json.loads(self._write_profiles(record).read_text(encoding="utf-8"))
        tui = TUIModel(manifest)

        updated = tui._apply_build(record, tui._catalog[0])

        for key in ("batch_size", "cache_type_k", "gpu_layers", "extra_args"):
            self.assertNotIn(key, updated)
        self.assertEqual(updated["environment"], {})
        self.assertIn("strata_artifacts_manifest_path", updated)

    def test_strata_artifact_and_config_paths_are_visible_in_advanced_tui(self) -> None:
        manifest_path = self._write_profiles(self._profile())
        tui = TUIModel(json.loads(manifest_path.read_text(encoding="utf-8")))
        tui.toggle_advanced()

        keys = {field.key for field in tui.visible_fields}
        self.assertIn("strata_artifacts_manifest_path", keys)
        self.assertIn("strata_launch_config_path", keys)
        self.assertNotIn("environment", keys)

    def test_strata_tui_path_edit_refreshes_pinned_config_hash(self) -> None:
        manifest_path = self._write_profiles(self._profile())
        replacement = self.model_root / "alternate-launch.json"
        replacement.write_bytes(self.launch_path.read_bytes())
        tui = TUIModel(json.loads(manifest_path.read_text(encoding="utf-8")))
        tui.advanced = True

        updated = tui.set_field("strata_launch_config_path", str(replacement))

        self.assertEqual(updated["strata_launch_config_path"], str(replacement.resolve()))
        self.assertEqual(updated["strata_launch_config_sha256"], sha256(replacement))

    def test_strata_cli_path_edit_refreshes_pinned_config_hash(self) -> None:
        replacement = self.model_root / "cli-launch.json"
        replacement.write_bytes(self.launch_path.read_bytes())
        record = self._profile()

        _set_scalar_field(record, "strata-launch-config", str(replacement))

        self.assertEqual(record["strata_launch_config_path"], str(replacement.resolve()))
        self.assertEqual(record["strata_launch_config_sha256"], sha256(replacement))

    def test_strata_cli_profile_edit_pins_manifest_paths(self) -> None:
        profiles_path = self._write_profiles(self._profile())
        replacement = self.model_root / "alternate-launch-cli.json"
        replacement.write_bytes(self.launch_path.read_bytes())

        exit_code = brokerctl.main([
            "--profiles", str(profiles_path), "profile", "edit", self.profile_id,
            "--strata-launch-config-path", str(replacement),
        ])

        self.assertEqual(exit_code, 0)
        updated = json.loads(profiles_path.read_text(encoding="utf-8"))["profiles"][0]
        self.assertEqual(updated["strata_launch_config_path"], str(replacement.resolve()))
        self.assertEqual(updated["strata_launch_config_sha256"], sha256(replacement))

    def test_cli_add_constructs_strata_profile_from_pinned_model_metadata(self) -> None:
        source = self._profile()
        source["runtime"] = "llamacpp"
        source.pop("runtime_build", None)
        data = json.loads(self._write_profiles(source).read_text(encoding="utf-8"))
        args: Any = SimpleNamespace(
            profile_id="strata-added",
            runtime_build="strata-test",
            server_binary=None,
            model=str(self.shard1),
            alias=[self.alias],
            display_name="Qwen Strata",
            role=None,
            context=8192,
            parallel=1,
            batch=None,
            ubatch=None,
            cache_k=None,
            cache_v=None,
            flash=None,
            gpu_layers=None,
            mmproj=None,
            restore=None,
            env=None,
            extra_args=None,
            strata_artifacts_manifest=str(self.manifest_path),
            strata_launch_config=str(self.launch_path),
        )

        added = brokerctl._build_added_profile(args, data)

        self.assertEqual(added["runtime"], "strata")
        self.assertEqual(added["model_path"], str(self.shard1))
        self.assertEqual(added["strata_artifacts_manifest_path"], str(self.manifest_path))
        self.assertEqual(added["strata_artifacts_manifest_sha256"], sha256(self.manifest_path))
        self.assertEqual(added["strata_launch_config_sha256"], sha256(self.launch_path))
        self.assertEqual(added["parallel"], 1)
        self.assertEqual(added["safety"]["service"]["memory_max_mib"], 20480)
        self.assertNotIn("extra_args", added)

    def test_cli_profile_add_registers_a_strata_build_and_validates_artifacts(self) -> None:
        profiles_path = self._write_profiles(self._profile())
        new_alias = "qwen3.8-flash-next-iq2-xs-added"
        launch = dict(self.launch_config)
        launch["model_name"] = new_alias
        launch_path = self.model_root / "strata-launch-added.json"
        launch_path.write_text(json.dumps(launch), encoding="utf-8")

        exit_code = brokerctl.main([
            "--profiles", str(profiles_path), "profile", "add", "strata-added",
            "--model", str(self.shard1), "--alias", new_alias,
            "--runtime-build", "strata-test",
            "--strata-artifacts-manifest", str(self.manifest_path),
            "--strata-launch-config", str(launch_path),
        ])

        self.assertEqual(exit_code, 0)
        registry = ProfileRegistry.from_path(profiles_path)
        added = registry.get("strata-added")
        self.assertEqual(added.alias, new_alias)
        self.assertEqual(added.runtime, "strata")
        self.assertEqual(added.model_path, self.shard1)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                brokerctl.main([
                    "--profiles", str(profiles_path), "profile", "render", "strata-added"
                ]),
                0,
            )
        rendered = output.getvalue()
        self.assertIn("--config-sha256", rendered)
        self.assertIn("--min-available-ram-mib", rendered)
        self.assertIn("--memory-max-mib", rendered)
        self.assertIn("127.0.0.1", rendered)
        self.assertIn("MemoryMax=20480M", rendered)
        self.assertIn("MemorySwapMax=1024M", rendered)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                brokerctl.main(["--profiles", str(profiles_path), "verify", "strata-added", "--json"]),
                0,
            )
        verified = json.loads(output.getvalue())
        self.assertEqual(verified["profiles"][0]["status"], "ok")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                brokerctl.main(["--profiles", str(profiles_path), "runtime", "list", "--json"]),
                0,
            )
        self.assertIn('"strata"', output.getvalue())

    def test_strata_artifact_verification_covers_every_pack_file(self) -> None:
        profile = ProfileRegistry.from_path(self._write_profiles(self._profile())).get(self.profile_id)
        _verify_profile_artifacts(profile, force=True)

        self.pack_files["dense.bin"].write_bytes(b"tampered")
        with self.assertRaisesRegex(BackendControllerError, "Strata artifact SHA mismatch"):
            _verify_profile_artifacts(profile, force=True)

    def test_strata_artifact_verification_covers_mtp_files(self) -> None:
        profile = ProfileRegistry.from_path(self._write_profiles(self._profile())).get(self.profile_id)
        _verify_profile_artifacts(profile, force=True)

        mtp_experts = self.mtp_files["rt/experts.bin"]
        mtp_experts.write_bytes(b"tampered mtp experts")
        with self.assertRaisesRegex(BackendControllerError, "Strata artifact SHA mismatch"):
            _verify_profile_artifacts(profile, force=True)

    def test_strata_mtp_inventory_revision_must_match_pinned_manifest(self) -> None:
        self.mtp_revision = "b" * 40
        self._write_manifest()
        with self.assertRaisesRegex(ProfileError, "not pinned to the declared revision"):
            ProfileRegistry.from_path(self._write_profiles(self._profile()))

    def test_broker_model_visibility_does_not_hash_strata_pack(self) -> None:
        registry = ProfileRegistry.from_path(self._write_profiles(self._profile()))
        controller: Any = SimpleNamespace(
            state=lambda: {"profile": self.profile_id, "phase": "active", "generation": 1}
        )
        app = BrokerApplication(
            registry=registry,
            controller=controller,
            backend_host="127.0.0.1",
            backend_port=10000,
            backend_timeout=2.0,
            max_body_bytes=1024,
            backend_alias_ready=lambda _alias: True,
            backend_ready=lambda: True,
            start_operation_worker=False,
        )
        try:
            profile = registry.get(self.profile_id)
            self.assertTrue(app._profile_artifacts_available(profile))
            self.pack_files["dense.bin"].write_bytes(b"tampered")
            self.assertTrue(app._profile_artifacts_available(profile))
        finally:
            app.shutdown()

    def test_start_verifier_rejects_symlink_substitution(self) -> None:
        profile = ProfileRegistry.from_path(self._write_profiles(self._profile())).get(self.profile_id)
        real_model = self.model_root / "model-copy.gguf"
        self.shard1.replace(real_model)
        self.shard1.symlink_to(real_model)

        with self.assertRaisesRegex(BackendControllerError, "not a regular file"):
            _verify_profile_artifacts(profile, force=True)

    def test_strata_artifact_verification_covers_runtime_server_sources(self) -> None:
        profile = ProfileRegistry.from_path(self._write_profiles(self._profile())).get(self.profile_id)
        server_source = self.strata_root / "serve" / "server.py"
        server_source.write_text("# tampered server source\n", encoding="utf-8")

        with self.assertRaisesRegex(BackendControllerError, "Strata artifact SHA mismatch"):
            _verify_profile_artifacts(profile, force=True)

    def test_strata_requires_bounded_cgroup_and_ram_headroom_budgets(self) -> None:
        for service_memory, available_ram in ((4096, 6144), (57345, 6144), (20480, 4095), (8192, 9000)):
            with self.subTest(service_memory=service_memory, available_ram=available_ram):
                record = self._profile()
                record["safety"] = json.loads(json.dumps(record["safety"]))
                record["safety"]["service"]["memory_max_mib"] = service_memory
                record["safety"]["admission"]["min_ram_available_mib"] = available_ram
                with self.assertRaises(ProfileError):
                    ProfileRegistry.from_path(self._write_profiles(record))

    def test_strata_requires_clean_pinned_runtime_source(self) -> None:
        cases: list[dict[str, object]] = [
            {"dirty": True}, {"source_commit": "not-a-commit"}, {"source_repository": None}
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ProfileError):
                    ProfileRegistry.from_path(
                        self._write_profiles(self._profile(), runtime_overrides=overrides)
                    )
        source_file = self.source_repo / "serve" / "server.py"
        original = source_file.read_text(encoding="utf-8")
        source_file.write_text("# locally modified source\n", encoding="utf-8")
        try:
            with self.assertRaisesRegex(ProfileError, "dirty"):
                ProfileRegistry.from_path(self._write_profiles(self._profile()))
        finally:
            source_file.write_text(original, encoding="utf-8")

    def test_strata_rejects_incomplete_python_inventory_vs_pinned_checkout(self) -> None:
        extra_source = self.source_repo / "tools" / "strata_tokenizer.py"
        extra_source.parent.mkdir(parents=True)
        extra_source.write_text("# committed source file\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.source_repo), "add", "tools"], check=True)
        subprocess.run(
            ["git", "-C", str(self.source_repo), "-c", "user.name=Fixture",
             "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "add source file"],
            check=True,
        )
        self.source_commit = subprocess.run(
            ["git", "-C", str(self.source_repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        self._write_manifest()
        self._write_launch_config()

        with self.assertRaisesRegex(ProfileError, "inventory differs"):
            ProfileRegistry.from_path(self._write_profiles(self._profile()))

    def test_strata_rejects_symlinked_engine_and_working_directory_paths(self) -> None:
        engine_link = self.strata_root / "bin" / "strata-link"
        engine_link.symlink_to(self.engine)
        linked_engine_config = dict(self.launch_config)
        linked_engine_config["engine_binary"] = str(engine_link)
        linked_engine_config["engine_sha256"] = sha256(self.engine)
        self.launch_path.write_text(json.dumps(linked_engine_config), encoding="utf-8")
        with self.assertRaisesRegex(ProfileError, "non-symlink"):
            ProfileRegistry.from_path(self._write_profiles(self._profile()))
        engine_link.unlink()

        working_link = self.bundle / "strata-link"
        working_link.symlink_to(self.strata_root, target_is_directory=True)
        linked_working_config = dict(self.launch_config)
        linked_working_config["working_directory"] = str(working_link)
        self.launch_path.write_text(json.dumps(linked_working_config), encoding="utf-8")
        with self.assertRaisesRegex(ProfileError, "working_directory"):
            ProfileRegistry.from_path(self._write_profiles(self._profile()))

    def test_strata_rejects_non_loopback_or_tool_capabilities(self) -> None:
        for override in (
            {"capabilities": ["text", "tools"]},
            {"capabilities": ["text", "vision"]},
            {"parallel": 2},
        ):
            with self.subTest(override=override):
                with self.assertRaises(ProfileError):
                    ProfileRegistry.from_path(self._write_profiles(self._profile(**override)))

    def test_strata_rejects_unsafe_launch_configuration(self) -> None:
        self.launch_config["api_key"] = "must-not-be-stored"
        self.launch_path.write_text(json.dumps(self.launch_config), encoding="utf-8")
        record = self._profile(strata_launch_config_sha256=sha256(self.launch_path))

        with self.assertRaisesRegex(ProfileError, "api_key"):
            ProfileRegistry.from_path(self._write_profiles(record))

    def test_strata_rejects_path_traversal_in_artifact_manifest(self) -> None:
        self.manifest["pack"]["files"][0]["path"] = "../outside.bin"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        record = self._profile(strata_artifacts_manifest_sha256=sha256(self.manifest_path))

        with self.assertRaisesRegex(ProfileError, "path"):
            ProfileRegistry.from_path(self._write_profiles(record))

    def test_strata_rejects_arbitrary_extra_args_and_environment(self) -> None:
        for override in (
            {"extra_args": ["--host", "0.0.0.0"]},
            {"environment": {"STRATA_API_KEY": "secret"}},
        ):
            with self.subTest(override=override):
                with self.assertRaises(ProfileError):
                    ProfileRegistry.from_path(self._write_profiles(self._profile(**override)))

    def test_backend_transport_uses_http_only_for_loopback_strata(self) -> None:
        self.assertEqual(
            backend_url_for_runtime("https://127.0.0.1:10000", "strata"),
            "http://127.0.0.1:10000",
        )
        self.assertEqual(
            backend_url_for_runtime("https://127.0.0.1:10000", "llamacpp"),
            "https://127.0.0.1:10000",
        )
        with self.assertRaisesRegex(ValueError, "loopback"):
            backend_url_for_runtime("https://192.0.2.10:10000", "strata")

    def test_broker_routes_loopback_strata_backend_over_http(self) -> None:
        registry = ProfileRegistry.from_path(self._write_profiles(self._profile()))
        controller: Any = SimpleNamespace(
            state=lambda: {"profile": self.profile_id, "phase": "active", "generation": 1}
        )
        with patch("llama_broker.daemon.backend_serves_alias", return_value=True) as probe:
            app = BrokerApplication(
                registry=registry,
                controller=controller,
                backend_host="127.0.0.1",
                backend_port=10000,
                backend_timeout=2.0,
                max_body_bytes=1024,
                backend_url="https://127.0.0.1:10000",
                security=SecurityConfig(),
                start_operation_worker=False,
            )
            try:
                self.assertEqual(app.backend_url, "http://127.0.0.1:10000")
                self.assertEqual(probe.call_args.kwargs["scheme"], "http")
            finally:
                app.shutdown()


if __name__ == "__main__":
    unittest.main()
