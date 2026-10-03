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
from llama_broker.configure.pi import PiProvider
from llama_broker.configure.registry import ToolProfile, ToolRegistry


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

    def test_model_selection_requires_an_explicit_alias(self) -> None:
        self.assertIsNone(_build_parser().parse_args([]).model)
        self.assertEqual(
            _build_parser().parse_args(["--model", "fixture-text-7b"]).model,
            "fixture-text-7b",
        )
        with self.assertRaises(SystemExit):
            _build_parser().parse_args(["--set-default"])

    def test_brokerctl_passes_explicit_tool_names_via_tools_option(self) -> None:
        args = argparse.Namespace(
            tool_names=["codex"],
            dry_run=True,
            model=None,
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
                    ClaudeCodeProvider().configure(registry, model="fixture-text-7b").written
                )
                self.assertTrue(
                    OpenClawProvider().configure(registry, model="fixture-text-7b").written
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
                HermesProvider().configure(registry, model="fixture-text-7b")

            content = config_path.read_text(encoding="utf-8")
            self.assertIn("  default: fixture-text-7b", content)
            self.assertIn("  provider: llama-broker", content)
            self.assertIn("  base_url: https://127.0.0.1:11434/v1", content)
            self.assertIn("    default_model: fixture-text-7b", content)

    def test_hermes_integration_normalizes_internal_profile_id_to_public_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp = Path(tempdir)
            profiles_path = temp / "profiles.json"
            profiles_path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "id": "gemma4-12b-qat",
                                "alias": "gemma4-12b",
                                "aliases": ["gemma4-12b", "gemma4-12b-qat"],
                                "context_size": 131072,
                                "parallel": 2,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            registry = ToolRegistry(
                broker_url="https://127.0.0.1:11434/v1",
                profiles_path=str(profiles_path),
                api_key="[REDACTED]",
            )
            config_path = temp / "config.yaml"
            config_path.write_text(
                "providers:\n"
                "  llama-broker:\n"
                "    default_model: gemma4-12b-qat\n"
                "    models:\n"
                "      gemma4-12b:\n"
                "        context_length: 65536\n",
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
                result = HermesProvider().configure(registry)

            content = config_path.read_text(encoding="utf-8")

        self.assertTrue(result.written)
        self.assertIn("default_model: gemma4-12b\n", content)
        self.assertNotIn("default_model: gemma4-12b-qat\n", content)
        self.assertIn("      gemma4-12b:\n", content)
        self.assertNotIn("      gemma4-12b-qat:\n", content)

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
                OpenCodeProvider().configure(registry, model="fixture-text-7b")

            data = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(data["model"], "llamacpp/fixture-text-7b")

    def test_opencode_removes_model_aliases_not_in_broker_registry(self) -> None:
        registry = ToolRegistry(
            broker_url="https://127.0.0.1:11434/v1",
            profiles_path=str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "profiles.fixture.json"),
            api_key="[REDACTED]",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "opencode.json"
            config_path.write_text(
                json.dumps({
                    "provider": {
                        "llamacpp": {
                            "name": "llama.cpp",
                            "npm": "@ai-sdk/openai-compatible",
                            "options": {
                                "baseURL": registry.broker_url,
                                "apiKey": registry.api_key,
                            },
                            "models": {
                                "obsolete-alias": {"name": "Removed model", "options": {}},
                                "fixture-text-7b": {"name": "Current model", "options": {}},
                            },
                        }
                    }
                }),
                encoding="utf-8",
            )
            with patch("llama_broker.configure.opencode.OPENCODE_CONFIG", config_path):
                result = OpenCodeProvider().configure(registry)

            models = json.loads(config_path.read_text(encoding="utf-8"))["provider"]["llamacpp"]["models"]
            self.assertNotIn("obsolete-alias", models)
            self.assertIn("fixture-text-7b", models)
            self.assertIn("Removed unavailable model: obsolete-alias", result.details)

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
                CodexProvider().configure(registry, model="fixture-text-7b")

            content = config_path.read_text(encoding="utf-8")
            self.assertIn('model = "fixture-text-7b"', content)



    def test_pi_setup_writes_broker_provider_into_models_json(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = ToolRegistry(
            broker_url="https://127.0.0.1:11434/v1",
            profiles_path=str(root / "tests" / "fixtures" / "profiles.fixture.json"),
            api_key="[REDACTED]",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            agent_dir = Path(tempdir) / "agent"
            agent_dir.mkdir(parents=True)
            with (
                patch("llama_broker.configure.pi.PI_AGENT_DIR", agent_dir),
                patch("llama_broker.configure.pi.PI_MODELS_CONFIG", agent_dir / "models.json"),
                patch("llama_broker.configure.pi.PI_SETTINGS_CONFIG", agent_dir / "settings.json"),
            ):
                result = PiProvider().configure(registry)

            self.assertTrue(result.written)
            data = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))
            provider = data["providers"]["llamacpp"]
            self.assertEqual(provider["baseUrl"], "https://127.0.0.1:11434/v1")
            self.assertEqual(provider["api"], "openai-completions")
            self.assertEqual(provider["apiKey"], "[REDACTED]")
            self.assertEqual(
                [m["id"] for m in provider["models"]],
                ["fixture-text-7b", "fixture-vision-12b"],
            )
            model_entry = provider["models"][0]
            self.assertEqual(model_entry["contextWindow"], 32768)
            self.assertEqual(model_entry["maxOutputTokens"], 32768)
            self.assertNotIn("reasoning", model_entry)
            # A fresh install must not be pointed at the broker by default.
            self.assertFalse((agent_dir / "settings.json").exists())

    def test_pi_setup_is_idempotent_and_keeps_other_providers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = ToolRegistry(
            broker_url="https://127.0.0.1:11434/v1",
            profiles_path=str(root / "tests" / "fixtures" / "profiles.fixture.json"),
            api_key="[REDACTED]",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            agent_dir = Path(tempdir) / "agent"
            agent_dir.mkdir(parents=True)
            models_path = agent_dir / "models.json"
            models_path.write_text(
                json.dumps(
                    {
                        "providers": {
                            "anthropic": {
                                "baseUrl": "https://example.invalid/v1",
                                "models": [{"id": "claude-sonnet-5"}],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch("llama_broker.configure.pi.PI_AGENT_DIR", agent_dir),
                patch("llama_broker.configure.pi.PI_MODELS_CONFIG", models_path),
                patch("llama_broker.configure.pi.PI_SETTINGS_CONFIG", agent_dir / "settings.json"),
            ):
                first = PiProvider().configure(registry)
                second = PiProvider().configure(registry)

            self.assertTrue(first.written)
            self.assertFalse(second.written)
            data = json.loads(models_path.read_text(encoding="utf-8"))
            self.assertIn("anthropic", data["providers"])
            self.assertIn("llamacpp", data["providers"])

    def test_pi_setup_removes_unavailable_models_and_updates_stale_url(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = ToolRegistry(
            broker_url="https://127.0.0.1:11434/v1",
            profiles_path=str(root / "tests" / "fixtures" / "profiles.fixture.json"),
            api_key="[REDACTED]",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            agent_dir = Path(tempdir) / "agent"
            agent_dir.mkdir(parents=True)
            models_path = agent_dir / "models.json"
            models_path.write_text(
                json.dumps(
                    {
                        "providers": {
                            "llamacpp": {
                                "baseUrl": "http://127.0.0.1:9999/v1",
                                "api": "openai-completions",
                                "models": [{"id": "obsolete-alias"}],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch("llama_broker.configure.pi.PI_AGENT_DIR", agent_dir),
                patch("llama_broker.configure.pi.PI_MODELS_CONFIG", models_path),
                patch("llama_broker.configure.pi.PI_SETTINGS_CONFIG", agent_dir / "settings.json"),
            ):
                result = PiProvider().configure(registry)

            data = json.loads(models_path.read_text(encoding="utf-8"))
            provider = data["providers"]["llamacpp"]
            self.assertEqual(provider["baseUrl"], "https://127.0.0.1:11434/v1")
            self.assertEqual(
                [m["id"] for m in provider["models"]],
                ["fixture-text-7b", "fixture-vision-12b"],
            )
            self.assertIn("Removed unavailable model: obsolete-alias", result.details)
            self.assertIn("baseUrl: http://127.0.0.1:9999/v1 -> https://127.0.0.1:11434/v1", result.details)

    def test_pi_set_default_explicitly_selects_broker_model(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = ToolRegistry(
            broker_url="https://127.0.0.1:11434/v1",
            profiles_path=str(root / "tests" / "fixtures" / "profiles.fixture.json"),
            api_key="[REDACTED]",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            agent_dir = Path(tempdir) / "agent"
            agent_dir.mkdir(parents=True)
            settings_path = agent_dir / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "defaultProvider": "anthropic",
                        "defaultModel": "anthropic/claude-sonnet-5",
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch("llama_broker.configure.pi.PI_AGENT_DIR", agent_dir),
                patch("llama_broker.configure.pi.PI_MODELS_CONFIG", agent_dir / "models.json"),
                patch("llama_broker.configure.pi.PI_SETTINGS_CONFIG", settings_path),
            ):
                result = PiProvider().configure(registry, model="fixture-text-7b")

            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertEqual(settings["defaultProvider"], "llamacpp")
            self.assertEqual(settings["defaultModel"], "llamacpp/fixture-text-7b")
            self.assertEqual(result.path, str(agent_dir / "models.json"))

    def test_pi_skips_when_pi_is_not_installed(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            agent_dir = Path(tempdir) / "not-installed" / "agent"
            with (
                patch("llama_broker.configure.pi.PI_AGENT_DIR", agent_dir),
                patch("llama_broker.configure.pi.PI_MODELS_CONFIG", agent_dir / "models.json"),
                patch("llama_broker.configure.pi.PI_SETTINGS_CONFIG", agent_dir / "settings.json"),
            ):
                result = PiProvider().configure(
                    ToolRegistry(
                        broker_url="http://127.0.0.1:11434/v1",
                        profiles_path=str(
                            Path(__file__).resolve().parents[1]
                            / "tests" / "fixtures" / "profiles.fixture.json"
                        ),
                        api_key="[REDACTED]",
                    )
                )
            self.assertTrue(result.skipped)
            self.assertFalse((agent_dir / "models.json").exists())

    def test_pi_model_entry_declares_reasoning_when_capability_is_present(self) -> None:
        profile = ToolProfile.from_broker(
            {
                "alias": "reasoning-27b",
                "display_name": "Reasoning 27B",
                "context_size": 65536,
                "parallel": 2,
                "capabilities": ["text", "reasoning"],
            }
        )
        entry = PiProvider._model_entry(profile, 32768)
        self.assertTrue(entry["reasoning"])
        self.assertNotIn("thinkingLevelMap", entry)
        self.assertEqual(entry["contextWindow"], 32768)
if __name__ == "__main__":
    unittest.main()
