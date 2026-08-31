from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llama_broker.broker_profiles import (
    ProfileError,
    ProfileRegistry,
    launch_environment,
)


class ProfileRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.server = self.root / "llama-server"
        self.server.write_bytes(b"server")
        self.server.chmod(0o755)
        self.main_model = self.root / "main.gguf"
        self.main_model.write_bytes(b"main")
        self.vision_model = self.root / "vision.gguf"
        self.vision_model.write_bytes(b"vision")
        self.mmproj = self.root / "mmproj.gguf"
        self.mmproj.write_bytes(b"projector")
        self.moe_profile = self.root / "routing.csv"
        self.moe_profile.write_text("0,0,1\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _profile(
        self,
        profile_id: str,
        alias: str,
        model_path: Path,
        **overrides: object,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "id": profile_id,
            "alias": alias,
            "aliases": [alias],
            "display_name": profile_id,
            "role": "persistent",
            "capabilities": ["text"],
            "server_binary": str(self.server),
            "server_sha256": "1" * 64,
            "model_path": str(model_path),
            "model_sha256": "2" * 64,
            "context_size": 65536,
            "parallel": 1,
            "batch_size": 256,
            "ubatch_size": 64,
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo4",
            "flash_attention": "on",
            "gpu_layers": "all",
            "startup_timeout": 180,
            "safety": {
                "service": {
                    "type": "exec",
                    "memory_max_mib": 20_480,
                    "memory_swap_max_mib": 1_024,
                    "tasks_max": 512,
                    "kill_signal": "SIGINT",
                    "timeout_stop_sec": 20,
                },
                "admission": {
                    "min_ram_available_mib": 8_192,
                    "max_swap_used_mib": 1_024,
                    "max_swap_growth_mib": 128,
                    "min_total_vram_mib": 12_000,
                    "min_free_vram_before_start_mib": 8_192,
                    "driver_vram_reserve_mib": 512,
                    "vision_vram_reserve_mib": 1_536,
                    "safety_vram_reserve_mib": 1_024,
                },
            },
            "environment": {
                "GGML_CUDA_PDL": "0",
            },
            "extra_args": ["--no-warmup"],
        }
        record.update(overrides)
        if (
            "safety" not in overrides
            and record.get("role") == "transient"
            and "vision" in record.get("capabilities", [])  # type: ignore[operator]
        ):
            safety = dict(record["safety"])  # type: ignore[arg-type]
            admission = dict(safety["admission"])  # type: ignore[arg-type]
            admission["vision_vram_reserve_mib"] = 0
            admission["safety_vram_reserve_mib"] = 1_536
            safety["admission"] = admission
            record["safety"] = safety
        return record

    def test_profile_exposes_declarative_service_and_admission_budgets(self) -> None:
        profile = ProfileRegistry.from_path(
            self._write([self._profile("main", "main-27b", self.main_model)])
        ).get("main")

        self.assertEqual(profile.safety.service.service_type, "exec")
        self.assertEqual(profile.safety.service.memory_max_mib, 20_480)
        self.assertEqual(profile.safety.service.memory_swap_max_mib, 1_024)
        self.assertEqual(profile.safety.service.tasks_max, 512)
        self.assertEqual(profile.safety.service.kill_signal, "SIGINT")
        self.assertEqual(profile.safety.service.timeout_stop_sec, 20)
        self.assertEqual(profile.safety.admission.min_ram_available_mib, 8_192)
        self.assertEqual(profile.safety.admission.max_swap_growth_mib, 128)
        self.assertEqual(profile.safety.admission.min_total_vram_mib, 12_000)
        self.assertEqual(profile.safety.admission.post_ready_min_free_vram_mib, 3_072)

    def test_output_budget_scales_inverse_cubically_with_parallel_slots(self) -> None:
        profile = ProfileRegistry.from_path(
            self._write(
                [
                    self._profile(
                        "main",
                        "main-27b",
                        self.main_model,
                        context_size=262_144,
                        parallel=2,
                    )
                ]
            )
        ).get("main")

        self.assertEqual(profile.slot_context_window, 131_072)
        self.assertEqual(profile.default_max_output_tokens, 32_768)

    def test_service_swap_budget_cannot_exceed_memory_budget(self) -> None:
        profile = self._profile("main", "main-27b", self.main_model)
        safety = dict(profile["safety"])  # type: ignore[arg-type]
        service = dict(safety["service"])  # type: ignore[arg-type]
        service["memory_swap_max_mib"] = 20_481
        safety["service"] = service
        profile["safety"] = safety

        with self.assertRaisesRegex(ProfileError, "memory_swap_max_mib"):
            ProfileRegistry.from_path(self._write([profile]))

    def test_service_tasks_budget_has_a_fail_closed_upper_bound(self) -> None:
        profile = self._profile("main", "main-27b", self.main_model)
        safety = dict(profile["safety"])  # type: ignore[arg-type]
        service = dict(safety["service"])  # type: ignore[arg-type]
        service["tasks_max"] = 513
        safety["service"] = service
        profile["safety"] = safety

        with self.assertRaisesRegex(ProfileError, "tasks_max"):
            ProfileRegistry.from_path(self._write([profile]))

    def test_service_memory_budget_has_a_fail_closed_upper_bound(self) -> None:
        profile = self._profile("main", "main-27b", self.main_model)
        safety = dict(profile["safety"])  # type: ignore[arg-type]
        service = dict(safety["service"])  # type: ignore[arg-type]
        service["memory_max_mib"] = 20_481
        safety["service"] = service
        profile["safety"] = safety

        with self.assertRaisesRegex(ProfileError, "memory_max_mib"):
            ProfileRegistry.from_path(self._write([profile]))

    def test_service_swap_budget_has_a_fail_closed_upper_bound(self) -> None:
        profile = self._profile("main", "main-27b", self.main_model)
        safety = dict(profile["safety"])  # type: ignore[arg-type]
        service = dict(safety["service"])  # type: ignore[arg-type]
        service["memory_swap_max_mib"] = 1_025
        safety["service"] = service
        profile["safety"] = safety

        with self.assertRaisesRegex(ProfileError, "memory_swap_max_mib"):
            ProfileRegistry.from_path(self._write([profile]))

    def test_service_stop_timeout_has_a_fail_closed_upper_bound(self) -> None:
        profile = self._profile("main", "main-27b", self.main_model)
        safety = dict(profile["safety"])  # type: ignore[arg-type]
        service = dict(safety["service"])  # type: ignore[arg-type]
        service["timeout_stop_sec"] = 21
        safety["service"] = service
        profile["safety"] = safety

        with self.assertRaisesRegex(ProfileError, "timeout_stop_sec"):
            ProfileRegistry.from_path(self._write([profile]))

    def test_fit_is_optional_and_strict_when_present(self) -> None:
        for unsafe_value in (None, "false", 0, 1, [], {}):
            with self.subTest(value=unsafe_value):
                profile = self._profile("main", "main-27b", self.main_model)
                profile["fit"] = unsafe_value
                with self.assertRaisesRegex(ProfileError, "fit must be a boolean"):
                    ProfileRegistry.from_path(self._write([profile]))

        parsed = ProfileRegistry.from_path(
            self._write([self._profile("main", "main-27b", self.main_model)])
        ).get("main")
        args = parsed.server_args(host="127.0.0.1", port=10000)
        self.assertIsNone(parsed.fit)
        self.assertNotIn("--fit", args)
        self.assertNotIn("--cache-ram", args)
        self.assertNotIn("--ctx-checkpoints", args)

        explicit_off = self._profile("off", "off", self.main_model, fit=False)
        off_args = ProfileRegistry.from_path(self._write([explicit_off])).get(
            "off"
        ).server_args(host="127.0.0.1", port=10000)
        self.assertNotIn("--fit", off_args)
        self.assertEqual(off_args[off_args.index("-ngl") + 1], "all")

    def test_tools_capability_emits_valid_all_tools_flag_per_profile(self) -> None:
        tool_profile = self._profile(
            "tools",
            "tools",
            self.main_model,
            capabilities=["text", "tools"],
        )
        text_profile = self._profile("text", "text", self.vision_model)
        registry = ProfileRegistry.from_path(
            self._write([tool_profile, text_profile])
        )

        tool_args = registry.get("tools").server_args(host="127.0.0.1", port=10000)
        text_args = registry.get("text").server_args(host="127.0.0.1", port=10000)

        self.assertEqual(tool_args[tool_args.index("--tools") + 1], "all")
        self.assertNotIn("--tools", text_args)

    def test_prompt_cache_cannot_consume_more_than_half_service_memory(self) -> None:
        profile = self._profile(
            "main",
            "main-27b",
            self.main_model,
            fit=True,
            gpu_layers="auto",
            fit_target_mib=3_072,
            fit_ctx=65_536,
        )
        profile["cache_ram_mib"] = 10_241
        with self.assertRaisesRegex(ProfileError, "cache_ram_mib.*half"):
            ProfileRegistry.from_path(self._write([profile]))

        profile["cache_ram_mib"] = 10_240
        parsed = ProfileRegistry.from_path(self._write([profile])).get("main")
        self.assertEqual(parsed.cache_ram_mib, 10_240)

    def test_speculative_draft_counts_are_integer_bounded_and_have_values(self) -> None:
        for flag in ("--draft-max", "--spec-draft-n-max", "--draft-min"):
            for unsafe_value in ("65", "not-an-integer"):
                with self.subTest(flag=flag, value=unsafe_value):
                    profile = self._profile(
                        "main",
                        "main-27b",
                        self.main_model,
                        extra_args=[flag, unsafe_value],
                    )
                    with self.assertRaisesRegex(ProfileError, flag):
                        ProfileRegistry.from_path(self._write([profile]))

            profile = self._profile(
                "main",
                "main-27b",
                self.main_model,
                extra_args=[flag],
            )
            with self.subTest(flag=flag, value="missing"):
                with self.assertRaisesRegex(ProfileError, flag):
                    ProfileRegistry.from_path(self._write([profile]))

            profile = self._profile(
                "main",
                "main-27b",
                self.main_model,
                extra_args=[flag, "64"],
            )
            parsed = ProfileRegistry.from_path(self._write([profile])).get("main")
            self.assertIn(flag, parsed.extra_args)

    def test_draft_mtp_rejects_explicit_draft_resolving_to_target(self) -> None:
        target_alias = self.root / "same-target.gguf"
        target_alias.symlink_to(self.main_model)
        profile = self._profile(
            "main",
            "main-27b",
            self.main_model,
            model_draft=str(target_alias),
            extra_args=["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
        )

        with self.assertRaisesRegex(ProfileError, "model_draft.*model_path"):
            ProfileRegistry.from_path(self._write([profile]))

        del profile["model_draft"]
        parsed = ProfileRegistry.from_path(self._write([profile])).get("main")
        self.assertIsNone(parsed.model_draft)
        self.assertNotIn("-md", parsed.server_args(host="127.0.0.1", port=10000))

    def test_unsupported_gemma_runtime_flags_are_rejected(self) -> None:
        for extra_args in (["--gemma"], ["--arch", "gemma4"]):
            with self.subTest(extra_args=extra_args):
                profile = self._profile(
                    "main",
                    "main-27b",
                    self.main_model,
                    extra_args=extra_args,
                )
                with self.assertRaisesRegex(ProfileError, "unsupported token"):
                    ProfileRegistry.from_path(self._write([profile]))

    def test_moe_cache_flags_are_allowlisted_and_rendered(self) -> None:
        profile = self._profile(
            "moe",
            "moe",
            self.main_model,
            extra_args=[
                "--no-warmup",
                "--moe-cache-profile",
                str(self.moe_profile),
                "--moe-cache-slots",
                "1",
            ],
        )

        parsed = ProfileRegistry.from_path(self._write([profile])).get("moe")
        args = parsed.server_args(host="127.0.0.1", port=10000)

        self.assertEqual(
            args[args.index("--moe-cache-profile") + 1], str(self.moe_profile)
        )
        self.assertEqual(args[args.index("--moe-cache-slots") + 1], "1")

    def test_moe_cache_slots_accept_explicit_profile_limit(self) -> None:
        profile = self._profile(
            "moe-qualified",
            "moe-qualified",
            self.main_model,
            moe_cache_slots_limit=88,
            extra_args=["--moe-cache-slots", "88"],
        )

        parsed = ProfileRegistry.from_path(self._write([profile])).get("moe-qualified")

        self.assertEqual(parsed.moe_cache_slots_limit, 88)
        self.assertEqual(parsed.extra_args[-1], "88")

    def test_moe_cache_slots_limit_has_hard_ceiling(self) -> None:
        profile = self._profile(
            "moe-qualified",
            "moe-qualified",
            self.main_model,
            moe_cache_slots_limit=113,
            extra_args=["--moe-cache-slots", "88"],
        )

        with self.assertRaisesRegex(ProfileError, "moe_cache_slots_limit"):
            ProfileRegistry.from_path(self._write([profile]))

    def test_moe_cache_profile_must_be_an_absolute_existing_file(self) -> None:
        for value in ("routing.csv", str(self.root / "missing.csv")):
            with self.subTest(value=value):
                profile = self._profile(
                    "moe",
                    "moe",
                    self.main_model,
                    extra_args=["--moe-cache-profile", value],
                )
                with self.assertRaisesRegex(ProfileError, "absolute existing file"):
                    ProfileRegistry.from_path(self._write([profile]))

    def test_moe_cache_slots_are_nonnegative_and_bounded(self) -> None:
        for value in ("-1", "one", "9"):
            with self.subTest(value=value):
                profile = self._profile(
                    "moe",
                    "moe",
                    self.main_model,
                    extra_args=["--moe-cache-slots", value],
                )
                with self.assertRaisesRegex(ProfileError, "moe-cache-slots"):
                    ProfileRegistry.from_path(self._write([profile]))

        profile = self._profile(
            "moe",
            "moe",
            self.main_model,
            extra_args=["--moe-cache-slots", "8"],
        )
        parsed = ProfileRegistry.from_path(self._write([profile])).get("moe")
        self.assertEqual(parsed.extra_args[-1], "8")

    def test_native_vision_profile_does_not_require_projector(self) -> None:
        profile = self._profile(
            "gemma4",
            "gemma4-12b-qat",
            self.main_model,
            capabilities=["text", "vision"],
            native_vision=True,
        )
        parsed = ProfileRegistry.from_path(self._write([profile])).get("gemma4")
        args = parsed.server_args(host="127.0.0.1", port=10000)

        self.assertTrue(parsed.native_vision)
        self.assertIsNone(parsed.mmproj_path)
        self.assertNotIn("--mmproj", args)

    def test_admission_thresholds_cannot_be_weakened_below_host_safety_policy(self) -> None:
        unsafe_values = {
            "min_ram_available_mib": 8_191,
            "max_swap_used_mib": 1_025,
            "max_swap_growth_mib": 129,
            "min_total_vram_mib": 11_999,
            "min_free_vram_before_start_mib": 8_191,
        }
        for field_name, unsafe_value in unsafe_values.items():
            with self.subTest(field=field_name):
                profile = self._profile("main", "main-27b", self.main_model)
                safety = dict(profile["safety"])  # type: ignore[arg-type]
                admission = dict(safety["admission"])  # type: ignore[arg-type]
                admission[field_name] = unsafe_value
                safety["admission"] = admission
                profile["safety"] = safety

                with self.assertRaisesRegex(ProfileError, field_name):
                    ProfileRegistry.from_path(self._write([profile]))

    def test_primary_role_is_rejected(self) -> None:
        profile = self._profile(
            "main", "main-27b", self.main_model, role="primary"
        )

        with self.assertRaisesRegex(ProfileError, "invalid role"):
            ProfileRegistry.from_path(self._write([profile]))

    def test_post_ready_reserve_cannot_exceed_native_fit_reserve(self) -> None:
        profile = self._profile(
            "main",
            "main-27b",
            self.main_model,
            fit=True,
            gpu_layers="auto",
            fit_target_mib=2_048,
            fit_ctx=65_536,
        )

        with self.assertRaisesRegex(ProfileError, "fit_target_mib"):
            ProfileRegistry.from_path(self._write([profile]))

    def test_transient_vision_budget_consumes_instead_of_reserving_vision_vram(self) -> None:
        persistent_safety = self._profile(
            "main", "main-27b", self.main_model
        )["safety"]
        vision = self._profile(
            "vision",
            "qwen3.5-9b-vision",
            self.vision_model,
            role="transient",
            capabilities=["text", "vision"],
            restore_profile="main",
            mmproj_path=str(self.mmproj),
            mmproj_sha256="3" * 64,
            gpu_layers="auto",
            fit=True,
            fit_target_mib=3_072,
            fit_ctx=65_536,
            safety=persistent_safety,
        )

        with self.assertRaisesRegex(ProfileError, "vision_vram_reserve_mib"):
            ProfileRegistry.from_path(
                self._write(
                    [self._profile("main", "main-27b", self.main_model), vision]
                )
            )

    def test_pre_start_vram_threshold_must_cover_post_ready_reserves(self) -> None:
        profile = self._profile("main", "main-27b", self.main_model)
        safety = dict(profile["safety"])  # type: ignore[arg-type]
        admission = dict(safety["admission"])  # type: ignore[arg-type]
        admission["safety_vram_reserve_mib"] = 7_000
        safety["admission"] = admission
        profile["safety"] = safety

        with self.assertRaisesRegex(ProfileError, "min_free_vram_before_start_mib"):
            ProfileRegistry.from_path(self._write([profile]))

    def test_pre_start_vram_threshold_cannot_exceed_total_vram_floor(self) -> None:
        profile = self._profile("main", "main-27b", self.main_model)
        safety = dict(profile["safety"])  # type: ignore[arg-type]
        admission = dict(safety["admission"])  # type: ignore[arg-type]
        admission["min_free_vram_before_start_mib"] = 12_001
        safety["admission"] = admission
        profile["safety"] = safety

        with self.assertRaisesRegex(ProfileError, "min_free_vram_before_start_mib"):
            ProfileRegistry.from_path(self._write([profile]))

    def _write(self, profiles: list[dict[str, object]]) -> Path:
        path = self.root / "profiles.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profiles": profiles,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_exact_aliases_resolve_and_unknown_alias_fails_closed(self) -> None:
        main = self._profile(
            "main",
            "main-27b",
            self.main_model,
            aliases=["main-27b", "main"],
        )
        vision = self._profile(
            "vision",
            "qwen3.5-9b-vision",
            self.vision_model,
            aliases=["qwen3.5-9b-vision"],
            role="transient",
            capabilities=["text", "vision"],
            restore_profile="main",
            mmproj_path=str(self.mmproj),
            mmproj_sha256="3" * 64,
            gpu_layers="auto",
            context_size=32768,
        )
        registry = ProfileRegistry.from_path(self._write([main, vision]))

        self.assertEqual(registry.resolve("main").id, "main")
        self.assertEqual(registry.resolve("qwen3.5-9b-vision").id, "vision")
        self.assertIsNone(registry.try_resolve("qwen3-5-9b-vision"))
        with self.assertRaises(ProfileError):
            registry.resolve("does-not-exist")

    def test_main_launch_args_leave_fit_unset_for_native_gpu_placement(self) -> None:
        main = self._profile("main", "main-27b", self.main_model)
        profile = ProfileRegistry.from_path(self._write([main])).get("main")
        args = profile.server_args(host="127.0.0.1", port=10000)

        expected_pairs = {
            "-c": "65536",
            "-np": "1",
            "-b": "256",
            "-ub": "64",
            "-ctk": "turbo4",
            "-ctv": "turbo4",
            "--flash-attn": "on",
            "-ngl": "all",
        }
        for flag, value in expected_pairs.items():
            index = args.index(flag)
            self.assertEqual(args[index + 1], value)
        self.assertIn("--no-context-shift", args)
        self.assertNotIn("--fit", args)
        self.assertNotIn("--cache-ram", args)
        self.assertNotIn("--ctx-checkpoints", args)
        self.assertNotIn("--mmproj", args)
        self.assertNotIn("-np 2", " ".join(args))
        self.assertEqual(profile.environment["GGML_CUDA_PDL"], "0")
        self.assertNotIn("GGML_CUDA_DISABLE_GRAPHS", profile.environment)
        self.assertNotIn("GGML_CUDA_GRAPH_OPT", profile.environment)

    def test_llamacpp_graph_controls_are_not_profile_requirements(self) -> None:
        profile = self._profile("main", "main-27b", self.main_model)
        profile["environment"] = {"GGML_CUDA_PDL": "0"}

        parsed = ProfileRegistry.from_path(self._write([profile])).get("main")

        self.assertEqual(dict(parsed.environment), {"GGML_CUDA_PDL": "0"})

    def test_launch_environment_scopes_graph_controls_to_autotune(self) -> None:
        environment = {
            "CUSTOM": "keep",
            "GGML_CUDA_DISABLE_GRAPHS": "legacy",
            "GGML_CUDA_GRAPH_OPT": "legacy",
            "GGML_CUDA_PDL": "0",
            "LLAMA_GRAPH_REUSE_DISABLE": "legacy",
        }

        normal = launch_environment(environment, runtime="llamacpp")
        autotune = launch_environment(environment, runtime="llamacpp", autotune=True)

        self.assertEqual(
            normal,
            {"CUSTOM": "keep", "GGML_CUDA_PDL": "0"},
        )
        self.assertEqual(
            autotune,
            {
                "CUSTOM": "keep",
                "GGML_CUDA_DISABLE_GRAPHS": "1",
                "GGML_CUDA_GRAPH_OPT": "0",
                "GGML_CUDA_PDL": "0",
                "LLAMA_GRAPH_REUSE_DISABLE": "1",
            },
        )

    def test_vision_launch_prioritizes_mmproj_with_native_auto_placement(self) -> None:
        main = self._profile("main", "main-27b", self.main_model)
        vision = self._profile(
            "vision",
            "qwen3.5-9b-vision",
            self.vision_model,
            role="transient",
            capabilities=["text", "vision"],
            restore_profile="main",
            mmproj_path=str(self.mmproj),
            mmproj_sha256="3" * 64,
            context_size=32768,
            gpu_layers="auto",
        )
        profile = ProfileRegistry.from_path(self._write([main, vision])).get("vision")
        args = profile.server_args(host="127.0.0.1", port=10000)

        self.assertEqual(args[args.index("--mmproj") + 1], str(self.mmproj))
        self.assertIn("--mmproj-offload", args)
        self.assertEqual(args[args.index("-ngl") + 1], "auto")
        self.assertNotIn("--fit", args)
        self.assertNotIn("--cache-ram", args)
        self.assertNotIn("--ctx-checkpoints", args)

    def test_explicit_fit_profile_emits_only_declared_fit_controls(self) -> None:
        record = self._profile(
            "main",
            "main-27b",
            self.main_model,
            fit=True,
            gpu_layers="auto",
            fit_target_mib=3_072,
            fit_ctx=65_536,
            cache_ram_mib=512,
            ctx_checkpoints=2,
        )
        args = ProfileRegistry.from_path(self._write([record])).get("main").server_args(
            host="127.0.0.1", port=10000
        )

        self.assertEqual(args[args.index("--fit") + 1], "on")
        self.assertEqual(args[args.index("--fit-target") + 1], "3072")
        self.assertEqual(args[args.index("--fit-ctx") + 1], "65536")
        self.assertEqual(args[args.index("--cache-ram") + 1], "512")
        self.assertEqual(args[args.index("--ctx-checkpoints") + 1], "2")
        self.assertNotIn("-ngl", args)

    def test_explicit_fit_omits_undeclared_cache_controls(self) -> None:
        record = self._profile(
            "main",
            "main-27b",
            self.main_model,
            fit=True,
            gpu_layers="auto",
            fit_target_mib=3_072,
            fit_ctx=65_536,
        )
        args = ProfileRegistry.from_path(self._write([record])).get("main").server_args(
            host="127.0.0.1", port=10000
        )

        self.assertNotIn("--cache-ram", args)
        self.assertNotIn("--ctx-checkpoints", args)

    def test_duplicate_alias_and_invalid_restore_target_are_rejected(self) -> None:
        main = self._profile("main", "same", self.main_model)
        duplicate = self._profile("other", "same", self.vision_model)
        with self.assertRaisesRegex(ProfileError, "duplicate alias"):
            ProfileRegistry.from_path(self._write([main, duplicate]))

        broken = self._profile(
            "vision",
            "vision",
            self.vision_model,
            role="transient",
            restore_profile="missing",
        )
        with self.assertRaisesRegex(ProfileError, "restore_profile"):
            ProfileRegistry.from_path(self._write([main, broken]))

    def test_vision_capability_requires_existing_projector(self) -> None:
        vision = self._profile(
            "vision",
            "vision",
            self.vision_model,
            capabilities=["text", "vision"],
            mmproj_path=str(self.root / "missing.gguf"),
            mmproj_sha256="3" * 64,
        )
        with self.assertRaisesRegex(ProfileError, "mmproj"):
            ProfileRegistry.from_path(self._write([vision]))

    def test_fit_target_is_bounded_before_reaching_native_argument_parser(self) -> None:
        profile = self._profile(
            "main",
            "main-27b",
            self.main_model,
            gpu_layers="auto",
            fit=True,
            fit_target_mib=8193,
            fit_ctx=65_536,
        )
        with self.assertRaisesRegex(ProfileError, "fit_target_mib"):
            ProfileRegistry.from_path(self._write([profile]))

        profile["fit_target_mib"] = 8192
        self.assertEqual(
            ProfileRegistry.from_path(self._write([profile])).get("main").fit_target_mib,
            8192,
        )

    def test_kvarn_parallel_requires_explicit_multi_stream_opt_in(self) -> None:
        profile = self._profile(
            "kvarn",
            "kvarn",
            self.main_model,
            parallel=2,
            cache_type_k="kvarn4",
            cache_type_v="kvarn4",
        )
        with self.assertRaisesRegex(ProfileError, "LLAMA_KVARN_MULTI_STREAM=1"):
            ProfileRegistry.from_path(self._write([profile]))

        environment = dict(profile["environment"])  # type: ignore[arg-type]
        environment["LLAMA_KVARN_MULTI_STREAM"] = "1"
        profile["environment"] = environment
        parsed = ProfileRegistry.from_path(self._write([profile])).get("kvarn")
        self.assertEqual(parsed.parallel, 2)
        self.assertEqual(parsed.environment["LLAMA_KVARN_MULTI_STREAM"], "1")

    def test_extra_args_cannot_override_broker_owned_safety_controls(self) -> None:
        profile = self._profile(
            "main",
            "main-27b",
            self.main_model,
            extra_args=["--no-warmup", "--fit-target=18446744073709551615"],
        )
        with self.assertRaisesRegex(ProfileError, "extra_args"):
            ProfileRegistry.from_path(self._write([profile]))


if __name__ == "__main__":
    unittest.main()
