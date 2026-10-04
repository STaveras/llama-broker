from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace

from llama_broker.runtimes.strata_bridge import (
    StrataBridgeError,
    _effective_cgroup_limit_mib,
    _load_launch_config,
    assess_strata_memory_mode,
    enforce_cgroup_budget,
    enforce_gpu_floor,
    enforce_resource_floor,
    install_private_loopback_serve,
    read_api_key_file,
    read_api_keys_file,
    _runtime_log_path,
    _write_server_config,
)


class StrataBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.key_file = self.root / "api-keys"
        self.key_file.write_text("# broker key\nsecret-key\n", encoding="utf-8")
        self.key_file.chmod(0o600)
        self.engine = self.root / "bundle" / "strata" / "bin" / "strata"
        self.engine.parent.mkdir(parents=True)
        self.engine.write_bytes(b"engine")
        self.engine.chmod(0o755)
        self.expert_profile = self.root / "bundle" / "strata" / "data" / "expert-profile.bin"
        self.expert_profile.parent.mkdir(parents=True)
        self.expert_profile.write_bytes(b"expert profile")
        self.working_directory = self.root / "bundle" / "strata"
        self.pack_path = self.root / "model" / "strata-pack"
        self.tokenizer_path = self.pack_path / "tokenizer"
        self.tokenizer_path.mkdir(parents=True)
        (self.pack_path / "experts.bin").write_bytes(b"mmap expert data")
        self.mtp_path = self.root / "model" / "mtp" / "rt"
        self.mtp_path.mkdir(parents=True)
        for name in ("experts.bin", "dense.bin", "dense.txt"):
            (self.mtp_path / name).write_bytes(b"mtp " + name.encode())
        self.serve_dir = self.working_directory / "serve"
        self.serve_dir.mkdir(parents=True, exist_ok=True)
        (self.serve_dir / "__init__.py").write_text("", encoding="utf-8")
        (self.serve_dir / "server.py").write_text("# pinned server module\n", encoding="utf-8")
        self.config_path = self.root / "launch.json"
        self.launch_config = {
            "schema_version": 1,
            "model_name": "qwen-strata-test",
            "engine_binary": str(self.engine),
            "engine_sha256": hashlib.sha256(self.engine.read_bytes()).hexdigest(),
            "expert_profile_sha256": hashlib.sha256(self.expert_profile.read_bytes()).hexdigest(),
            "engine_args": [
                "--pack", str(self.pack_path),
                "--native", str(self.root / "model" / "model-00001.gguf"),
                "--ple-gguf", str(self.root / "model" / "model-00002.gguf"),
                "--expert-profile", str(self.expert_profile),
                "--expert-cache", "auto", "--prefill", "auto",
                "--max-context", "8192", "--kv-resident", "20480", "--spec", "4",
                "--spec-min-p", "0.5", "--mtp", str(self.mtp_path),
                "--mmap-experts",
            ],
            "working_directory": str(self.working_directory),
            "tokenizer_path": str(self.tokenizer_path),
            "runtime_files": [
                {"path": path.relative_to(self.working_directory).as_posix(),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(self.working_directory.rglob("*.py"))
            ],
        }
        self.config_path.write_text(json.dumps(self.launch_config), encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_launch_config_checks_engine_and_expert_hashes_and_rejects_unknown_fields(self) -> None:
        self.assertEqual(
            _load_launch_config(
                self.config_path,
                hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            )["model_name"],
            "qwen-strata-test",
        )

        unsafe = dict(self.launch_config)
        unsafe["api_key"] = "must-not-be-stored"
        self.config_path.write_text(json.dumps(unsafe), encoding="utf-8")
        with self.assertRaisesRegex(StrataBridgeError, "unknown or missing"):
            _load_launch_config(
                self.config_path,
                hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            )

        self.config_path.write_text(json.dumps(self.launch_config), encoding="utf-8")
        (self.serve_dir / "server.py").write_text("# tampered server module\n", encoding="utf-8")
        with self.assertRaisesRegex(StrataBridgeError, "runtime source SHA"):
            _load_launch_config(
                self.config_path,
                hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            )

        (self.serve_dir / "server.py").write_text("# pinned server module\n", encoding="utf-8")
        self.engine.write_bytes(b"tampered engine")
        with self.assertRaisesRegex(StrataBridgeError, "engine binary SHA"):
            _load_launch_config(
                self.config_path,
                hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            )

    def test_kv_resident_cell_floor_is_enforced(self) -> None:
        args = list(self.launch_config["engine_args"])
        args[args.index("--kv-resident") + 1] = "20479"
        self.launch_config["engine_args"] = args
        self.config_path.write_text(json.dumps(self.launch_config), encoding="utf-8")

        with self.assertRaisesRegex(StrataBridgeError, "resident cells"):
            _load_launch_config(
                self.config_path,
                hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            )

    def test_resident_budget_mode_is_accepted(self) -> None:
        args = list(self.launch_config["engine_args"])
        mmap_index = args.index("--mmap-experts")
        args[mmap_index:mmap_index + 1] = ["--resident-budget-gib", "9"]
        self.launch_config["engine_args"] = args
        self.config_path.write_text(json.dumps(self.launch_config), encoding="utf-8")

        loaded = _load_launch_config(
            self.config_path,
            hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
        )

        self.assertIn("--resident-budget-gib", loaded["engine_args"])
        warning = assess_strata_memory_mode(
            system_ram_mib=24 * 1024,
            gpu_vram_total_mib=12 * 1024,
            mmap_experts=False,
            has_experts_bin=True,
            resident_budget_gib=9,
        )
        self.assertIn("bounded resident", warning or "")

    def test_reads_last_key_from_private_file(self) -> None:
        self.key_file.write_text("older-key\ncurrent-key\n", encoding="utf-8")
        self.key_file.chmod(0o600)

        self.assertEqual(read_api_key_file(self.key_file), "current-key")

    def test_reads_all_keys_for_overlapping_broker_rotation(self) -> None:
        self.key_file.write_text("older-key\ncurrent-key\n", encoding="utf-8")
        self.key_file.chmod(0o600)

        self.assertEqual(read_api_keys_file(self.key_file), ("older-key", "current-key"))

    def test_bridge_auth_accepts_all_rotating_keys_then_tracks_revocation(self) -> None:
        self.key_file.write_text("older-key\ncurrent-key\n", encoding="utf-8")
        self.key_file.chmod(0o600)

        class FakeHandler(BaseHTTPRequestHandler):
            def __init__(self, token: str) -> None:
                self.raw_requestline = b"GET /web/app.js HTTP/1.1\r\n"
                self.rfile = io.BytesIO(
                    f"Host: localhost\r\nAuthorization: Bearer {token}\r\n\r\n".encode()
                )
                self.wfile = io.BytesIO()
                self.auth_responses = []
                self.close_connection = False

            def _json(self, status, payload):
                self.auth_responses.append((status, payload))

        service = SimpleNamespace(api_key="")
        module = SimpleNamespace(
            serve=lambda *_args, **_kwargs: None,
            make_handler=lambda _service: FakeHandler,
        )
        install_private_loopback_serve(module, self.key_file)
        handler_type = module.make_handler(service)

        self.assertTrue(handler_type("older-key").parse_request())
        self.assertTrue(handler_type("current-key").parse_request())
        self.key_file.write_text("current-key\nnew-key\n", encoding="utf-8")
        self.key_file.chmod(0o600)
        self.assertFalse(handler_type("older-key").parse_request())
        self.assertTrue(handler_type("new-key").parse_request())

    def test_rejects_insecure_or_empty_key_files(self) -> None:
        self.key_file.chmod(0o644)
        with self.assertRaisesRegex(StrataBridgeError, "permissions"):
            read_api_key_file(self.key_file)
        self.key_file.chmod(0o600)
        self.key_file.write_text("# only a comment\n", encoding="utf-8")
        with self.assertRaisesRegex(StrataBridgeError, "empty"):
            read_api_key_file(self.key_file)

    def test_resource_floor_keeps_ram_gate_and_makes_host_swap_advisory(self) -> None:
        enforce_resource_floor(
            available_ram_mib=8192,
            min_available_ram_mib=6144,
            swap_used_mib=0,
            max_swap_used_mib=1024,
        )
        with self.assertRaisesRegex(StrataBridgeError, "cannot be weakened"):
            enforce_resource_floor(
                available_ram_mib=64_000,
                min_available_ram_mib=40_960,
                swap_used_mib=0,
                max_swap_used_mib=2_048,
            )
        with self.assertRaisesRegex(StrataBridgeError, "available RAM"):
            enforce_resource_floor(
                available_ram_mib=5_000,
                min_available_ram_mib=6_144,
                swap_used_mib=0,
                max_swap_used_mib=1_024,
            )

        warning = enforce_resource_floor(
            available_ram_mib=8_192,
            min_available_ram_mib=6_144,
            swap_used_mib=6_144,
            max_swap_used_mib=1_024,
        )

        self.assertIn("host swap usage is 6144 MiB", warning or "")
        self.assertIn("cgroup", warning or "")

    def test_cgroup_gate_rejects_unbounded_or_host_exceeding_service(self) -> None:
        enforce_cgroup_budget(
            total_system_ram_mib=24576,
            declared_memory_max_mib=20480,
            declared_memory_swap_max_mib=1024,
            effective_memory_max_mib=20480,
            effective_memory_swap_max_mib=1024,
        )
        enforce_cgroup_budget(
            total_system_ram_mib=24576,
            declared_memory_max_mib=20480,
            declared_memory_swap_max_mib=1024,
            effective_memory_max_mib=16384,
            effective_memory_swap_max_mib=0,
        )
        with self.assertRaisesRegex(StrataBridgeError, "effective cgroup memory"):
            enforce_cgroup_budget(
                total_system_ram_mib=24576,
                declared_memory_max_mib=20480,
                declared_memory_swap_max_mib=1024,
                effective_memory_max_mib=4096,
                effective_memory_swap_max_mib=0,
            )
        with self.assertRaisesRegex(StrataBridgeError, "host and broker policy"):
            enforce_cgroup_budget(
                total_system_ram_mib=24576,
                declared_memory_max_mib=49152,
                declared_memory_swap_max_mib=1024,
                effective_memory_max_mib=24576,
                effective_memory_swap_max_mib=0,
            )

    def test_low_ram_mode_accounts_for_total_system_ram_and_vram(self) -> None:
        from llama_broker.runtimes.strata_bridge import assess_strata_memory_mode

        warning = assess_strata_memory_mode(
            system_ram_mib=24 * 1024,
            gpu_vram_total_mib=12 * 1024,
            mmap_experts=True,
            has_experts_bin=True,
        )
        self.assertIn("low-RAM estimate", warning or "")
        self.assertIsNone(
            assess_strata_memory_mode(
                system_ram_mib=48 * 1024,
                gpu_vram_total_mib=12 * 1024,
                mmap_experts=False,
                has_experts_bin=False,
            )
        )
        with self.assertRaisesRegex(StrataBridgeError, "experts.bin"):
            assess_strata_memory_mode(
                system_ram_mib=24 * 1024,
                gpu_vram_total_mib=12 * 1024,
                mmap_experts=True,
                has_experts_bin=False,
            )

    def test_cgroup_reader_uses_process_group_and_ancestors(self) -> None:
        cgroup_root = self.root / "cgroup"
        service_group = cgroup_root / "user.slice" / "strata.service"
        service_group.mkdir(parents=True)
        (service_group / "memory.max").write_text(str(49152 * 1024 * 1024), encoding="ascii")
        (service_group / "memory.swap.max").write_text(str(1024 * 1024 * 1024), encoding="ascii")
        parent = service_group.parent
        (parent / "memory.max").write_text(str(57344 * 1024 * 1024), encoding="ascii")
        (parent / "memory.swap.max").write_text("max\n", encoding="ascii")
        proc_cgroup = self.root / "proc-self-cgroup"
        proc_cgroup.write_text("0::/user.slice/strata.service\n", encoding="ascii")

        self.assertEqual(
            _effective_cgroup_limit_mib(
                "memory.max", proc_cgroup_path=proc_cgroup, cgroup_root_path=cgroup_root
            ),
            49152,
        )
        self.assertEqual(
            _effective_cgroup_limit_mib(
                "memory.swap.max", proc_cgroup_path=proc_cgroup, cgroup_root_path=cgroup_root
            ),
            1024,
        )

    def test_gpu_admission_checks_total_and_post_stop_free_vram(self) -> None:
        enforce_gpu_floor(
            gpu_total_mib=12_227,
            gpu_free_mib=11_900,
            min_total_vram_mib=12_000,
            min_free_vram_mib=8192,
        )
        with self.assertRaisesRegex(StrataBridgeError, "free VRAM"):
            enforce_gpu_floor(
                gpu_total_mib=12_227,
                gpu_free_mib=678,
                min_total_vram_mib=12_000,
                min_free_vram_mib=8192,
            )

    def test_runtime_engine_log_is_private_and_included_in_server_config(self) -> None:
        from unittest.mock import patch

        log_path = _runtime_log_path(
            "qwen3.8-flash-next-iq2-xs", state_home=self.root / "state"
        )
        self.assertTrue(log_path.is_file())
        self.assertEqual(log_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(log_path.parent.stat().st_mode & 0o777, 0o700)
        runtime_dir = self.root / "runtime"
        with patch.dict("os.environ", {"XDG_RUNTIME_DIR": str(runtime_dir)}):
            config_path = _write_server_config(self.launch_config, log_path)
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["log"], str(log_path))
        finally:
            config_path.unlink(missing_ok=True)

    def test_patched_server_requires_loopback_and_keeps_key_out_of_argv(self) -> None:
        calls = []
        service = SimpleNamespace(api_key="")

        def original_serve(svc, *, host, port):
            calls.append((svc, host, port))
            return "server"

        class FakeHandler(BaseHTTPRequestHandler):
            def _json(self, _status, _payload):
                return None

        module = SimpleNamespace(
            serve=original_serve,
            make_handler=lambda _service: FakeHandler,
        )
        install_private_loopback_serve(module, self.key_file)

        with self.assertRaisesRegex(StrataBridgeError, "loopback"):
            module.serve(service, host="0.0.0.0", port=10000)

        self.assertEqual(module.serve(service, host="127.0.0.1", port=10000), "server")
        self.assertEqual(service.api_key, "secret-key")
        self.assertEqual(calls[0][1:], ("127.0.0.1", 10000))


if __name__ == "__main__":
    unittest.main()
