"""Tests for the API-key configuration CLI."""

import getpass

import pytest

from autoprepml import cli_config
from autoprepml.config_manager import AutoPrepMLConfig


def _patch_set_api_key(monkeypatch, calls):
    def set_api_key(provider, api_key):
        calls.append((provider, api_key))

    monkeypatch.setattr(AutoPrepMLConfig, "set_api_key", staticmethod(set_api_key))


def test_interactive_skip_ollama_and_save(monkeypatch):
    """The wizard handles skip, local, and credential-backed providers."""
    monkeypatch.setattr("builtins.input", lambda _prompt: "5")
    assert cli_config.configure_interactive() == 0

    monkeypatch.setattr("builtins.input", lambda _prompt: "4")
    assert cli_config.configure_interactive() == 0

    calls = []
    _patch_set_api_key(monkeypatch, calls)
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")
    monkeypatch.setattr(getpass, "getpass", lambda _prompt: "openai-secret")
    assert cli_config.configure_interactive() == 0
    assert calls == [("openai", "openai-secret")]


@pytest.mark.parametrize("input_error", [ValueError, KeyboardInterrupt])
def test_interactive_cancel(monkeypatch, input_error):
    """Invalid input and cancellation do not expose a traceback."""

    def fail_input(_prompt):
        if input_error is ValueError:
            return "not-a-number"
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", fail_input)
    assert cli_config.configure_interactive() == 2


def test_main_without_arguments_delegates_to_wizard(monkeypatch):
    """Programmatic callers can invoke the no-argument interactive mode."""
    monkeypatch.setattr(cli_config, "configure_interactive", lambda: 7)

    assert cli_config.main([]) == 7


def test_main_help(capsys):
    """The configuration CLI exposes standard argparse help."""
    with pytest.raises(SystemExit) as exc_info:
        cli_config.main(["--help"])

    assert exc_info.value.code == 0
    assert "Manage API keys" in capsys.readouterr().out


def test_list_and_info_commands(monkeypatch, capsys):
    """List and info commands delegate and return success."""
    listed = []
    monkeypatch.setattr(
        AutoPrepMLConfig,
        "list_api_keys",
        staticmethod(lambda: listed.append(True)),
    )
    assert cli_config.main(["--list"]) == 0
    assert listed == [True]

    assert cli_config.main(["--info"]) == 0
    assert "AutoPrepML" in capsys.readouterr().out


def test_set_command_handles_provider_and_empty_key(monkeypatch, capsys):
    """Set supports local providers and reports empty secrets."""
    assert cli_config.main(["--set", "unknown"]) == 2
    assert "Unknown provider" in capsys.readouterr().out

    assert cli_config.main(["--set", "ollama"]) == 0
    assert "no API key needed" in capsys.readouterr().out

    calls = []
    _patch_set_api_key(monkeypatch, calls)
    monkeypatch.setattr(getpass, "getpass", lambda _prompt: "")
    assert cli_config.main(["--set", "OPENAI"]) == 0
    assert calls == []
    assert "No API key entered" in capsys.readouterr().out

    monkeypatch.setattr(getpass, "getpass", lambda _prompt: "secret")
    assert cli_config.main(["--set", "OPENAI"]) == 0
    assert calls == [("openai", "secret")]


def test_remove_command(monkeypatch, capsys):
    """Remove validates providers and delegates valid requests."""
    assert cli_config.main(["--remove", "unknown"]) == 2
    assert "Unknown provider" in capsys.readouterr().out

    removed = []
    monkeypatch.setattr(
        AutoPrepMLConfig,
        "remove_api_key",
        staticmethod(lambda provider: removed.append(provider)),
    )
    assert cli_config.main(["--remove", "ANTHROPIC"]) == 0
    assert removed == ["anthropic"]


def test_check_command_reports_all_states(monkeypatch, capsys):
    """Check distinguishes configured, missing, and local providers."""
    assert cli_config.main(["--check", "unknown"]) == 2
    capsys.readouterr()

    monkeypatch.setattr(
        AutoPrepMLConfig,
        "get_api_key",
        staticmethod(lambda provider: "long-secret-value" if provider == "openai" else None),
    )
    assert cli_config.main(["--check", "openai"]) == 0
    assert "configured" in capsys.readouterr().out

    assert cli_config.main(["--check", "ollama"]) == 0
    assert "doesn't require" in capsys.readouterr().out

    assert cli_config.main(["--check", "google"]) == 0
    output = capsys.readouterr().out
    assert "not configured" in output
    assert "Configure it with" in output
