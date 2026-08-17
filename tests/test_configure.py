from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llama_broker.cli as brokerctl
from llama_broker import configure
from llama_broker.broker_security import SecurityConfig
from llama_broker.configure import DEFAULT_TOOLS, _build_parser, _select_tools
from llama_broker.configure.claude_code import ClaudeCodeProvider
from llama_broker.configure.codex import CodexProvider
from llama_broker.configure.hermes import HermesProvider
from llama_broker.configure.openclaw import OpenClawProvider
from llama_broker.configure.opencode import OpenCodeProvider
from llama_broker.configure.registry import ToolRegistry


class ConfigureSafetyTests(unittest.TestCase):
    def test_configuration_output_reports_tools_without_catalog_rows(self) -> None:
        registry = SimpleNamespace(broker_url="http://127.0.0.1:11434/v1", profiles=[
            SimpleNamespace(alias="main-27b", context_size=262144, parallel=1)
        ])
        provider = SimpleNamespace(
            configure=lambda *_args, **_kwargs: SimpleNamespace(
                written=False,
                skipped=False,
                reason="",
                details=[],
            )
        )
        output = io.StringIO()

        with (
            patch.object(configure, "ToolRegistry", return_value=registry),
            patch.object(configure, "_get_provider", return_value=provider),
            redirect_stdout(output),
        ):
            result = configure.main(["--tools", "hermes"])
            self.assertEqual(result, 0)
            self.assertEqual(output.getvalue(), "[hermes] OK (no changes needed)\n\n")

            output.seek(0)
            output.truncate(0)
            result = configure.main(["--tools", "hermes"], indent=True)

        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), "  [hermes] OK (no changes needed)\n\n")

    def test_default_configuration_excludes_codex(self) -> None:
        args = _build_parser().parse_args([])

        self.assertEqual(args.tools, DEFAULT_TOOLS)
        self.assertNotIn("codex", _select_tools(args.tools))

    def test_all_excludes_codex_unless_it_is_named_explicitly(self) -> None:
        self.assertNotIn("codex", _select_tools(["all"]))
        self.assertIn("codex", _select_tools(["all", "codex"]))
        self.assertEqual(_select_tools(["codex"]), ["codex"])

    def test_default_selection_requires_an_explicit_flag(self) -> None:
        self.assertFalse(_build_parser().parse_args([]).set_default)
        self.assertTrue(_build_parser().parse_args(["--set-default"]).set_default)
        self.assertTrue(
            _build_parser().parse_args(["--set-default-model"]).set_default
        )

    def test_brokerctl_passes_explicit_tool_names_via_tools_option(self) -> None:
        args = argparse.Namespace(
            tool_names=["codex"],
            dry_run=True,
            set_default=False,
            integrate_command="configure",
            profiles=Path("/tmp/profiles.json"),
        )
        with patch("llama_broker.configure.main", return_value=0) as configure_main:
            result = brokerctl._cmd_integrate_configure(args)

        self.assertEqual(result, 0)
        configure_main.assert_called_once_with(
            [
                "--tools",
                "codex",
                "--dry-run",
                "--profiles",
                "/tmp/profiles.json",
            ]
        )

    def test_supported_tools_receive_the_native_api_key_during_setup(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = ToolRegistry(
            broker_url="https://127.0.0.1:11434/v1",
            profiles_path=str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "profiles.fixture.json"),
            api_key="[REDACTED]",
        )

        with tempfile.TemporaryDirectory() as tempdir:
            temp = Path(tempdir)
            opencode_path = temp / "opencode.json"
            opencode_path.write_text(
                '{"model": "user-selected-opencode"}\n',
                encoding="utf-8",
            )
            hermes_path = temp / "config.yaml"
            hermes_path.write_text(
                "providers:\n"
                "  llama-broker:\n"
                "    base_url: http://127.0.0.1:11434/v1\n"
                "    default_model: user-selected-hermes\n"
                "    models:\n"
                "      main:\n"
                "        context_length: 1024\n",
                encoding="utf-8",
            )
            claude_path = temp / "claude.env"
            claude_path.write_text(
                "ANTHROPIC_MODEL=user-selected-claude\n",
                encoding="utf-8",
            )
            openclaw_path = temp / "openclaw.env"
            openclaw_path.write_text(
                "LLAMA_BROKER_DEFAULT_MODEL=user-selected-openclaw\n",
                encoding="utf-8",
            )

            with (
                patch("llama_broker.configure.opencode.OPENCODE_CONFIG", opencode_path),
                patch("llama_broker.configure.hermes.HERMES_CONFIG", hermes_path),
                patch(
                    "llama_broker.configure.hermes.load_security_config",
                    return_value=SecurityConfig(
                        broker_tls_ca_file=Path("/tmp/lan-ca.crt")
                    ),
                ),
                patch("llama_broker.configure.claude_code.CLAUDE_ENV", claude_path),
                patch("llama_broker.configure.openclaw.OPENCLAW_ENV", openclaw_path),
            ):
                self.assertTrue(OpenCodeProvider().configure(registry).written)
                self.assertTrue(HermesProvider().configure(registry).written)
                self.assertTrue(
                    ClaudeCodeProvider().configure(registry, set_default=True).written
                )
                self.assertTrue(
                    OpenClawProvider().configure(registry, set_default=True).written
                )

            opencode = json.loads(opencode_path.read_text(encoding="utf-8"))
            self.assertEqual(
                opencode["provider"]["llamacpp"]["options"]["apiKey"],
                "[REDACTED]",
            )
            self.assertEqual(opencode["model"], "user-selected-opencode")
            self.assertIn("api_key: [REDACTED]", hermes_path.read_text(encoding="utf-8"))
            self.assertIn("ANTHROPIC_API_KEY=[REDACTED]", claude_path.read_text(encoding="utf-8"))
            self.assertIn("LLAMA_BROKER_API_KEY=[REDACTED]", openclaw_path.read_text(encoding="utf-8"))

            self.assertIn(
                "default_model: user-selected-hermes",
                hermes_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "ANTHROPIC_MODEL=fixture-text-7b",
                claude_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "LLAMA_BROKER_DEFAULT_MODEL=fixture-text-7b",
                openclaw_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "ssl_ca_cert: /tmp/lan-ca.crt",
                hermes_path.read_text(encoding="utf-8"),
            )

    def test_codex_setup_preserves_the_selected_model(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = ToolRegistry(
            broker_url="https://127.0.0.1:11434/v1",
            profiles_path=str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "profiles.fixture.json"),
            api_key="[REDACTED]",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.toml"
            config_path.write_text('model = "user-selected-codex"\n', encoding="utf-8")
            with patch("llama_broker.configure.codex.CODEX_CONFIG", config_path):
                result = CodexProvider().configure(registry)

            self.assertTrue(result.written)
            content = config_path.read_text(encoding="utf-8")
            self.assertIn('model = "user-selected-codex"', content)
            self.assertNotIn('model = "main-27b"', content)

    def test_hermes_setup_does_not_rewrite_the_active_model_route(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = ToolRegistry(
            broker_url="https://127.0.0.1:11434/v1",
            profiles_path=str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "profiles.fixture.json"),
            api_key="[REDACTED]",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            config_path.write_text(
                "model:\n"
                "  default: gpt-5.6-luna\n"
                "  provider: openai-codex\n"
                "  base_url: https://chatgpt.com/backend-api/codex\n"
                "providers:\n"
                "  llama-broker:\n"
                "    base_url: https://old-broker.example/v1\n"
                "    models:\n"
                "      main:\n"
                "        context_length: 1024\n",
                encoding="utf-8",
            )
            with (
                patch("llama_broker.configure.hermes.HERMES_CONFIG", config_path),
                patch(
                    "llama_broker.configure.hermes.load_security_config",
                    return_value=SecurityConfig(
                        broker_tls_ca_file=Path("/tmp/lan-ca.crt")
                    ),
                ),
            ):
                HermesProvider().configure(registry)

            content = config_path.read_text(encoding="utf-8")
            self.assertIn(
                "base_url: https://chatgpt.com/backend-api/codex",
                content,
            )
            self.assertIn(
                "base_url: https://127.0.0.1:11434/v1",
                content,
            )

    def test_hermes_set_default_explicitly_selects_broker_model_and_provider(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = ToolRegistry(
            broker_url="https://127.0.0.1:11434/v1",
            profiles_path=str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "profiles.fixture.json"),
            api_key="[REDACTED]",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            config_path.write_text(
                "model:\n"
                "  default: gpt-5.6-luna\n"
                "  provider: openai-codex\n"
                "  base_url: https://chatgpt.com/backend-api/codex\n"
                "providers:\n"
                "  llama-broker:\n"
                "    base_url: https://old-broker.example/v1\n"
                "    models:\n"
                "      main:\n"
                "        context_length: 1024\n",
                encoding="utf-8",
            )
            with (
                patch("llama_broker.configure.hermes.HERMES_CONFIG", config_path),
                patch(
                    "llama_broker.configure.hermes.load_security_config",
                    return_value=SecurityConfig(
                        broker_tls_ca_file=Path("/tmp/lan-ca.crt")
                    ),
                ),
            ):
                HermesProvider().configure(registry, set_default=True)

            content = config_path.read_text(encoding="utf-8")
            self.assertIn("  default: fixture-text-7b", content)
            self.assertIn("  provider: llama-broker", content)
            self.assertIn("  base_url: https://127.0.0.1:11434/v1", content)
            self.assertIn("    default_model: fixture-text-7b", content)

    def test_active_route_tools_preserve_endpoints_without_opt_in(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = ToolRegistry(
            broker_url="https://127.0.0.1:11434/v1",
            profiles_path=str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "profiles.fixture.json"),
            api_key="[REDACTED]",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            temp = Path(tempdir)
            claude_path = temp / "claude.env"
            claude_original = (
                "ANTHROPIC_MODEL=user-selected-claude\n"
                "ANTHROPIC_BASE_URL=https://api.anthropic.com/v1\n"
            )
            claude_path.write_text(claude_original, encoding="utf-8")
            openclaw_path = temp / "openclaw.env"
            openclaw_original = (
                "LLAMA_BROKER_DEFAULT_MODEL=user-selected-openclaw\n"
                "LLAMA_BROKER_URL=https://external.example/v1\n"
                "LLAMA_BROKER_API_KEY=external-key\n"
            )
            openclaw_path.write_text(openclaw_original, encoding="utf-8")

            with (
                patch("llama_broker.configure.claude_code.CLAUDE_ENV", claude_path),
                patch("llama_broker.configure.openclaw.OPENCLAW_ENV", openclaw_path),
            ):
                self.assertFalse(ClaudeCodeProvider().configure(registry).written)
                self.assertFalse(OpenClawProvider().configure(registry).written)

            self.assertEqual(claude_path.read_text(encoding="utf-8"), claude_original)
            self.assertEqual(openclaw_path.read_text(encoding="utf-8"), openclaw_original)

    def test_opencode_set_default_selects_broker_model(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = ToolRegistry(
            broker_url="https://127.0.0.1:11434/v1",
            profiles_path=str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "profiles.fixture.json"),
            api_key="[REDACTED]",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "opencode.json"
            config_path.write_text(
                '{"model": "opencode/big-pickle", "provider": {"llamacpp": {}}}\n',
                encoding="utf-8",
            )
            with patch("llama_broker.configure.opencode.OPENCODE_CONFIG", config_path):
                OpenCodeProvider().configure(registry, set_default=True)

            data = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(data["model"], "llamacpp/fixture-text-7b")

    def test_codex_set_default_selects_broker_model(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = ToolRegistry(
            broker_url="https://127.0.0.1:11434/v1",
            profiles_path=str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "profiles.fixture.json"),
            api_key="[REDACTED]",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.toml"
            config_path.write_text('model = "gpt-5.6-luna"\n', encoding="utf-8")
            with patch("llama_broker.configure.codex.CODEX_CONFIG", config_path):
                CodexProvider().configure(registry, set_default=True)

            content = config_path.read_text(encoding="utf-8")
            self.assertIn('model = "fixture-text-7b"', content)


if __name__ == "__main__":
    unittest.main()
