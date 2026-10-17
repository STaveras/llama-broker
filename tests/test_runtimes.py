from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import llama_broker.cli as brokerctl
from llama_broker.backend_controller import BackendController
from llama_broker.broker_profiles import ProfileError, ProfileRegistry
from llama_broker.runtimes import list_runtimes, runtime_names


class RuntimeDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.binary = self.root / "vllm"
        self.binary.write_bytes(b"vllm")
        self.binary.chmod(0o755)
        self.model = self.root / "model"
        self.model.write_bytes(b"model")
        self.media = self.root / "media"
        self.media.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _safety(self) -> dict[str, object]:
        return {
            "service": {
                "type": "exec",
                "memory_max_mib": 16_384,
                "memory_swap_max_mib": 512,
                "tasks_max": 256,
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
        }

    def _vllm_profile(self, **overrides: object) -> dict[str, object]:
        record: dict[str, object] = {
            "id": "vllm-main",
            "alias": "vllm-main",
            "aliases": ["vllm-main"],
            "display_name": "vLLM main",
            "role": "persistent",
            "runtime": "vllm",
            "capabilities": ["text"],
            "server_binary": str(self.binary),
            "server_sha256": "1" * 64,
            "model_path": str(self.model),
            "model_sha256": "2" * 64,
            "context_size": 8192,
            "parallel": 2,
            "gpu_memory_utilization": 0.85,
            "kv_cache_dtype": "fp8",
            "startup_timeout": 180,
            "safety": self._safety(),
            "environment": {},
            "extra_args": [],
        }
        record.update(overrides)
        return record

    def _write(self, profiles: list[dict[str, object]]) -> Path:
        path = self.root / "profiles.json"
        path.write_text(
            json.dumps(
                {"schema_version": 1, "profiles": profiles},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_registry_lists_only_supported_runtimes(self) -> None:
        self.assertEqual(runtime_names(), ("llamacpp", "vllm", "strata", "basalt"))
        runtimes = {runtime.name: runtime for runtime in list_runtimes()}
        self.assertEqual(set(runtimes), {"llamacpp", "vllm", "strata", "basalt"})
        self.assertIn("llama-server", runtimes["llamacpp"].binaries)
        self.assertIn("vllm", runtimes["vllm"].binaries)
        self.assertEqual(runtimes["strata"].capabilities, ("text",))
        self.assertIn("strata-server", runtimes["strata"].binaries)

    def test_strata_command_is_loopback_only_and_filters_managed_endpoint_args(self) -> None:
        profile = SimpleNamespace(
            server_binary=self.binary,
            strata_launch_config_path=self.root / "strata-launch.json",
            strata_launch_config_sha256="c" * 64,
            safety=SimpleNamespace(
                service=SimpleNamespace(
                    memory_max_mib=20480,
                    memory_swap_max_mib=1024,
                ),
                admission=SimpleNamespace(
                    min_ram_available_mib=6144,
                    max_swap_used_mib=1024,
                    min_total_vram_mib=12000,
                    min_free_vram_before_start_mib=8192,
                ),
            ),
        )
        adapter = next(runtime.adapter for runtime in list_runtimes() if runtime.name == "strata")

        command = adapter.build_command(
            profile,
            host="0.0.0.0",
            port=10000,
            managed_args=(
                "--host", "0.0.0.0", "--port", "10000",
                "--broker-api-key-file", "/tmp/api-keys",
            ),
        )

        self.assertEqual(
            command,
            (
                str(self.binary),
                "--config", str(self.root / "strata-launch.json"),
                "--config-sha256", "c" * 64,
                "--host", "127.0.0.1", "--port", "10000",
                "--min-available-ram-mib", "6144",
                "--max-swap-used-mib", "1024",
                "--memory-max-mib", "20480",
                "--memory-swap-max-mib", "1024",
                "--min-total-vram-mib", "12000",
                "--min-free-vram-mib", "8192",
                "--broker-api-key-file", "/tmp/api-keys",
            ),
        )

    def test_runtime_field_defaults_to_llamacpp_and_preserves_command_shape(self) -> None:
        llama = self._vllm_profile(
            id="legacy",
            alias="legacy",
            runtime=None,
            capabilities=["text"],
            server_binary=str(self.binary),
            batch_size=256,
            ubatch_size=64,
            cache_type_k="f16",
            cache_type_v="f16",
            flash_attention="off",
            gpu_layers="all",
            environment={
                "GGML_CUDA_PDL": "0",
            },
        )
        llama.pop("runtime")
        llama.pop("gpu_memory_utilization")
        llama.pop("kv_cache_dtype")
        parsed = ProfileRegistry.from_path(self._write([llama])).get("legacy")

        self.assertEqual(parsed.runtime, "llamacpp")
        self.assertEqual(
            parsed.command(host="127.0.0.1", port=10000)[:5],
            (str(self.binary), "-m", str(self.model), "--alias", "legacy"),
        )

    def test_vllm_command_uses_serve_shape_and_common_mappings(self) -> None:
        profile = self._vllm_profile(
            capabilities=["text", "vision", "tools"],
            tool_call_parser="hermes",
            limit_mm_per_prompt={"image": 2},
            allowed_local_media_path=str(self.media),
        )
        parsed = ProfileRegistry.from_path(self._write([profile])).get("vllm-main")
        command = parsed.command(host="0.0.0.0", port=10000)

        self.assertEqual(command[:3], (str(self.binary), "serve", str(self.model)))
        self.assertIn(("--max-model-len", "8192"), tuple(zip(command, command[1:])))
        self.assertIn(("--tensor-parallel-size", "2"), tuple(zip(command, command[1:])))
        self.assertIn(("--gpu-memory-utilization", "0.85"), tuple(zip(command, command[1:])))
        self.assertIn(("--kv-cache-dtype", "fp8"), tuple(zip(command, command[1:])))
        self.assertIn("--enable-auto-tool-choice", command)
        self.assertIn(("--tool-call-parser", "hermes"), tuple(zip(command, command[1:])))
        self.assertIn("--limit-mm-per-prompt", command)
        self.assertIn("--allowed-local-media-path", command)
        self.assertIn(("--host", "0.0.0.0"), tuple(zip(command, command[1:])))
        self.assertIn(("--port", "10000"), tuple(zip(command, command[1:])))
        for llama_flag in ("-m", "--alias", "-c", "-np", "-ctk", "--mmproj"):
            self.assertNotIn(llama_flag, command)

    def test_vllm_rejects_llamacpp_only_fields(self) -> None:
        for field, value in (
            ("batch_size", 256),
            ("gpu_layers", "all"),
            ("cache_type_k", "f16"),
            ("fit", False),
            ("mmproj_path", str(self.media / "mmproj.gguf")),
        ):
            with self.subTest(field=field):
                profile = self._vllm_profile(**{field: value})
                with self.assertRaisesRegex(ProfileError, field):
                    ProfileRegistry.from_path(self._write([profile]))

    def test_controller_renders_runtime_command_without_model_alias_assumptions(self) -> None:
        profile = ProfileRegistry.from_path(self._write([self._vllm_profile()])).get("vllm-main")
        controller = BackendController(
            profiles={profile.id: profile},
            state_path=self.root / "state.json",
            systemd_dir=self.root / "systemd",
            service_prefix="llama-broker-profile",
            command_runner=lambda _command: None,
            readiness_probe=lambda _alias, _timeout: True,
            resource_probe=None,
            event_logger=lambda _event: None,
        )

        unit = controller.render_unit(profile.id)
        exec_line = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
        self.assertIn(f'"{self.binary}" "serve"', exec_line)
        self.assertIn(f'"{self.model}"', exec_line)
        self.assertNotIn('"-m"', exec_line)
        self.assertNotIn('"--alias"', exec_line)

    def test_llamacpp_unit_does_not_append_runtime_name_to_execstart(self) -> None:
        record = self._vllm_profile(
            id="llama-main",
            alias="llama-main",
            display_name="llama.cpp main",
            runtime="llamacpp",
            batch_size=256,
            ubatch_size=64,
            cache_type_k="f16",
            cache_type_v="f16",
            flash_attention="off",
            gpu_layers="all",
            environment={
                "GGML_CUDA_PDL": "0",
            },
        )
        record.pop("gpu_memory_utilization")
        record.pop("kv_cache_dtype")
        profile = ProfileRegistry.from_path(self._write([record])).get("llama-main")
        controller = BackendController(
            profiles={profile.id: profile},
            state_path=self.root / "state.json",
            systemd_dir=self.root / "systemd",
            service_prefix="llama-broker-profile",
            command_runner=lambda _command: None,
            readiness_probe=lambda _alias, _timeout: True,
            resource_probe=None,
            event_logger=lambda _event: None,
            backend_listen_host="0.0.0.0",
            backend_listen_port=10000,
            managed_args={
                "llamacpp": ("--host", "0.0.0.0", "--port", "10000"),
            },
        )

        command = controller._command_for(profile)
        self.assertNotIn("llamacpp", command)
        self.assertEqual(command[-4:], ("--host", "0.0.0.0", "--port", "10000"))

        exec_line = next(
            line
            for line in controller.render_unit(profile.id).splitlines()
            if line.startswith("ExecStart=")
        )
        self.assertNotIn('"llamacpp"', exec_line)
        self.assertTrue(exec_line.endswith('"--host" "0.0.0.0" "--port" "10000"'))

    def test_brokerctl_runtime_list_reports_metadata_and_profiles(self) -> None:
        path = self._write([self._vllm_profile()])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                brokerctl.main(["--profiles", str(path), "runtime", "list"]),
                0,
            )

        rendered = output.getvalue()
        self.assertIn("RUNTIME", rendered)
        self.assertIn("BUNDLE", rendered)
        self.assertIn("PROFILES", rendered)
        self.assertIn("llamacpp", rendered)
        self.assertIn("vllm", rendered)
        self.assertNotIn("llama-server", rendered)
        self.assertIn("vllm-main", rendered)

    def test_profile_edit_rejects_unsupported_field_without_mutation(self) -> None:
        path = self._write([self._vllm_profile()])
        before = path.read_bytes()
        with self.assertRaisesRegex(ProfileError, "batch_size"):
            brokerctl._interactive_edit(
                "vllm-main",
                path,
                {"batch_size": "256"},
                flag_mode=True,
            )
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
