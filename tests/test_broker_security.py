from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_broker import broker_security
from llama_broker.daemon import _redact_log_message
from llama_broker.broker_security import (
    SecurityError,
    SecurityConfig,
    ensure_api_key_file,
    initialize_security_material,
    load_security_config,
    read_api_keys,
    revoke_api_key,
    rotate_api_keys,
    write_security_config,
)


class BrokerSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_strata_backend_args_pin_loopback_and_pass_only_api_key_file_path(self) -> None:
        config = SecurityConfig(
            api_key_file=self.root / "api-keys",
            backend_tls_cert_file=self.root / "backend.crt",
            backend_tls_key_file=self.root / "backend.key",
        )

        args = config.backend_server_args(
            listen_host="0.0.0.0", listen_port=10000, runtime="strata"
        )

        self.assertEqual(
            args,
            (
                "--host", "127.0.0.1",
                "--port", "10000",
                "--broker-api-key-file", str(self.root / "api-keys"),
            ),
        )
        self.assertNotIn("--ssl-cert-file", args)
        self.assertNotIn("--api-key", args)

    def test_strata_backend_requires_a_broker_api_key_file(self) -> None:
        with self.assertRaisesRegex(SecurityError, "requires API-key file"):
            SecurityConfig().backend_server_args(
                listen_host="127.0.0.1", listen_port=10000, runtime="strata"
            )

    def test_native_server_args_include_key_and_tls_without_broker_auth_authority(self) -> None:
        config = SecurityConfig(
            api_key_file=self.root / "api-keys",
            backend_tls_cert_file=self.root / "backend.crt",
            backend_tls_key_file=self.root / "backend.key",
        )

        self.assertEqual(
            config.backend_server_args(listen_host="0.0.0.0", listen_port=10000),
            (
                "--host",
                "0.0.0.0",
                "--port",
                "10000",
                "--api-key-file",
                str(self.root / "api-keys"),
                "--ssl-cert-file",
                str(self.root / "backend.crt"),
                "--ssl-key-file",
                str(self.root / "backend.key"),
            ),
        )

    def test_non_loopback_binding_requires_native_auth_and_tls(self) -> None:
        with self.assertRaisesRegex(ValueError, "broker TLS"):
            SecurityConfig().validate_network_binding("0.0.0.0", "127.0.0.1")
        with self.assertRaisesRegex(ValueError, "API-key"):
            SecurityConfig(
                broker_tls_cert_file=self.root / "broker.crt",
                broker_tls_key_file=self.root / "broker.key",
            ).validate_network_binding("127.0.0.1", "0.0.0.0")

    def test_raw_broker_key_satisfies_broker_lan_auth_requirement(self) -> None:
        config = SecurityConfig(
            api_key="[REDACTED]",
            broker_tls_cert_file=self.root / "broker.crt",
            broker_tls_key_file=self.root / "broker.key",
        )
        config.validate_network_binding("0.0.0.0", "127.0.0.1")

    def test_key_file_is_private_and_rotation_supports_overlap_and_revoke(self) -> None:
        path = self.root / "api-keys"
        _, first = ensure_api_key_file(path)
        self.assertEqual(read_api_keys(path), [first])
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

        second, previous = rotate_api_keys(path)
        self.assertEqual(previous, [first])
        self.assertEqual(read_api_keys(path), [first, second])

        remaining = revoke_api_key(path, first)
        self.assertEqual(remaining, [second])
        self.assertEqual(read_api_keys(path), [second])

    def test_security_config_file_and_environment_overrides(self) -> None:
        config_path = self.root / "security.json"
        api_key_path = self.root / "api-keys"
        config = SecurityConfig(
            api_key_file=api_key_path,
            source_path=config_path,
        )
        write_security_config(config)

        with patch.dict(
            os.environ,
            {"LLAMA_BROKER_SECURITY_FILE": str(config_path)},
            clear=False,
        ):
            loaded = load_security_config()

        self.assertEqual(loaded.api_key_file, api_key_path)
        self.assertEqual(loaded.source_path, config_path)

    def test_raw_api_key_is_loaded_from_environment_but_never_serialized(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLAMA_BROKER_SECURITY_FILE": str(self.root / "security.json"),
                "LLAMA_BROKER_API_KEY": "[REDACTED]",
            },
            clear=False,
        ):
            loaded = load_security_config()

        self.assertEqual(loaded.api_key, "[REDACTED]")
        self.assertNotIn("[REDACTED]", json.dumps(loaded.as_mapping()))
        self.assertEqual(loaded.authentication_keys(), ("[REDACTED]",))

    def test_raw_api_key_takes_precedence_over_key_file(self) -> None:
        key_path = self.root / "api-keys"
        key_path.write_text("file-key\n", encoding="utf-8")

        with patch.dict(
            os.environ,
            {
                "LLAMA_BROKER_API_KEY": "[REDACTED]",
                "LLAMA_BROKER_API_KEY_FILE": str(key_path),
            },
            clear=False,
        ):
            loaded = load_security_config()

        self.assertEqual(loaded.authentication_keys(), ("[REDACTED]",))

    def test_empty_raw_api_key_fails_closed(self) -> None:
        with patch.dict(
            os.environ,
            {"LLAMA_BROKER_API_KEY": "   "},
            clear=False,
        ):
            with self.assertRaisesRegex(SecurityError, "API key"):
                load_security_config(self.root / "security.json")

    def test_log_redaction_removes_authorization_and_key_values(self) -> None:
        message = _redact_log_message(
            'Authorization: Bearer raw-secret api_key=another-secret "token":"third-secret"'
        )
        self.assertNotIn("raw-secret", message)
        self.assertNotIn("another-secret", message)
        self.assertNotIn("third-secret", message)
        self.assertEqual(message.count("[REDACTED]"), 3)

    def test_security_init_generates_san_certificates_and_path_only_config(self) -> None:
        calls: list[tuple[str, ...]] = []
        extensions: list[str] = []

        def fake_openssl(binary: str, arguments: tuple[str, ...], cwd: Path) -> None:
            del binary
            calls.append(arguments)
            if "-extfile" in arguments:
                extensions.append(Path(arguments[arguments.index("-extfile") + 1]).read_text())
            for flag in ("-keyout", "-out"):
                if flag in arguments:
                    output = Path(arguments[arguments.index(flag) + 1])
                    output.write_text("generated\n", encoding="utf-8")

        config_path = self.root / "security.json"
        api_key_path = self.root / "api-keys"
        with (
            patch.object(broker_security.shutil, "which", return_value="/usr/bin/openssl"),
            patch.object(broker_security, "_run_openssl", side_effect=fake_openssl),
        ):
            result = initialize_security_material(
                ip_addresses=["192.0.2.10"],
                hostnames=["broker-test-host"],
                security_path=config_path,
                api_key_path=api_key_path,
            )

        self.assertTrue(result.created)
        self.assertEqual(
            result.sans,
            (
                "DNS:localhost",
                "DNS:broker-test-host",
                "IP:127.0.0.1",
                "IP:192.0.2.10",
            ),
        )
        self.assertEqual(len(calls), 5)
        self.assertIn(
            "subjectAltName = DNS:localhost,DNS:broker-test-host,IP:127.0.0.1,IP:192.0.2.10",
            extensions[0],
        )
        self.assertIn("extendedKeyUsage = serverAuth", extensions[0])
        self.assertEqual(stat.S_IMODE(config_path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(api_key_path.stat().st_mode), 0o600)
        for path in (
            self.root / "lan-ca.key",
            self.root / "broker.key",
            self.root / "llama-server.key",
        ):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        for path in (
            self.root / "lan-ca.crt",
            self.root / "broker.crt",
            self.root / "llama-server.crt",
        ):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config_data["api_key_file"], str(api_key_path))
        self.assertNotIn(read_api_keys(api_key_path)[-1], config_path.read_text(encoding="utf-8"))

    def test_security_init_is_idempotent_and_does_not_overwrite_material(self) -> None:
        config_path = self.root / "security.json"
        with (
            patch.object(broker_security.shutil, "which", return_value="/usr/bin/openssl"),
            patch.object(
                broker_security,
                "_run_openssl",
                side_effect=lambda _binary, arguments, _cwd: [
                    Path(arguments[index + 1]).write_text("generated\n", encoding="utf-8")
                    for index, flag in enumerate(arguments[:-1])
                    if flag in {"-keyout", "-out"}
                ],
            ),
        ):
            first = initialize_security_material(
                ip_addresses=["192.0.2.10"],
                hostnames=["broker-test-host"],
                security_path=config_path,
            )

        original_key = (self.root / "broker.key").read_bytes()
        with patch.object(broker_security, "_run_openssl") as run_openssl:
            second = initialize_security_material(
                ip_addresses=["192.0.2.10"],
                hostnames=["broker-test-host"],
                security_path=config_path,
            )

        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual((self.root / "broker.key").read_bytes(), original_key)
        run_openssl.assert_not_called()

    def test_security_init_refuses_partial_material_without_force(self) -> None:
        (self.root / "broker.key").write_text("existing\n", encoding="utf-8")

        with self.assertRaisesRegex(SecurityError, "already exists"):
            initialize_security_material(
                ip_addresses=["192.0.2.10"],
                security_path=self.root / "security.json",
            )


if __name__ == "__main__":
    unittest.main()
