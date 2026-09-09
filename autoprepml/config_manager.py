"""Configuration management for AutoPrepML API keys and settings."""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional


class AutoPrepMLConfig:
    """Manage AutoPrepML configuration and API keys"""

    CONFIG_DIR = Path.home() / ".autoprepml"
    CONFIG_FILE = CONFIG_DIR / "config.json"

    PROVIDERS = {
        "openai": {
            "name": "OpenAI",
            "env_var": "OPENAI_API_KEY",
            "instructions": "Get your API key from https://platform.openai.com/api-keys",
        },
        "anthropic": {
            "name": "Anthropic (Claude)",
            "env_var": "ANTHROPIC_API_KEY",
            "instructions": "Get your API key from https://console.anthropic.com/settings/keys",
        },
        "google": {
            "name": "Google (Gemini)",
            "env_var": "GOOGLE_API_KEY",
            "instructions": "Get your API key from https://makersuite.google.com/app/apikey",
        },
        "ollama": {
            "name": "Ollama (Local)",
            "env_var": None,
            "instructions": "Install Ollama from https://ollama.ai/ - No API key needed!",
        },
    }

    @classmethod
    def ensure_config_dir(cls):
        """Create config directory if it doesn't exist"""
        cls.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            cls.CONFIG_DIR.chmod(0o700)

    @classmethod
    def _validate_provider(cls, provider: str) -> Dict:
        """Return provider metadata or raise a consistent validation error."""
        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. Valid providers: {', '.join(cls.PROVIDERS)}"
            )
        return cls.PROVIDERS[provider]

    @staticmethod
    def _saved_api_keys(config: Dict) -> Dict[str, str]:
        """Return saved keys while rejecting malformed configuration data."""
        api_keys = config.get("api_keys", {})
        if not isinstance(api_keys, dict):
            raise ValueError("Configuration field 'api_keys' must be an object")
        return api_keys

    @classmethod
    def load_config(cls) -> Dict:
        """Load configuration from file"""
        if cls.CONFIG_FILE.exists():
            try:
                with cls.CONFIG_FILE.open("r", encoding="utf-8") as config_file:
                    config = json.load(config_file)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in configuration file: {cls.CONFIG_FILE}") from exc
            if not isinstance(config, dict):
                raise ValueError(
                    f"Configuration file must contain a JSON object: {cls.CONFIG_FILE}"
                )
            return config
        return {}

    @classmethod
    def save_config(cls, config: Dict):
        """Save configuration to file"""
        if not isinstance(config, dict):
            raise TypeError("Configuration must be a dictionary")

        cls.ensure_config_dir()
        temporary_path = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=".config-",
                suffix=".tmp",
                dir=cls.CONFIG_DIR,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as config_file:
                json.dump(config, config_file, indent=2, sort_keys=True)
                config_file.write("\n")
                config_file.flush()
                os.fsync(config_file.fileno())
            if os.name != "nt":
                temporary_path.chmod(0o600)
            os.replace(temporary_path, cls.CONFIG_FILE)
            if os.name != "nt":
                cls.CONFIG_FILE.chmod(0o600)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def set_api_key(cls, provider: str, api_key: str):
        """Set API key for a provider"""
        provider_info = cls._validate_provider(provider)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("API key must be a non-empty string")

        config = cls.load_config()
        api_keys = cls._saved_api_keys(config)
        config["api_keys"] = api_keys

        api_keys[provider] = api_key
        cls.save_config(config)

        print(f"✅ API key for {provider_info['name']} saved successfully!")

    @classmethod
    def get_api_key(cls, provider: str) -> Optional[str]:
        """Get API key for a provider (from config or environment)"""
        provider_info = cls._validate_provider(provider)

        # First check environment variable
        if (env_var := provider_info.get("env_var")) and (env_key := os.getenv(env_var)):
            return env_key

        # Then check config file
        config = cls.load_config()
        return cls._saved_api_keys(config).get(provider)

    @classmethod
    def remove_api_key(cls, provider: str):
        """Remove API key for a provider"""
        provider_info = cls._validate_provider(provider)
        config = cls.load_config()
        api_keys = cls._saved_api_keys(config)
        if provider in api_keys:
            del api_keys[provider]
            cls.save_config(config)
            print(f"✅ API key for {provider_info['name']} removed!")
        else:
            print(f"ℹ️  No API key found for {provider_info['name']}")

    @classmethod
    def list_api_keys(cls):
        """List all configured API keys (masked)"""
        print("\n🔑 AutoPrepML API Key Configuration")
        print("=" * 60)

        config = cls.load_config()
        saved_keys = cls._saved_api_keys(config)

        for provider, info in cls.PROVIDERS.items():
            provider_name = info["name"]
            env_var = info["env_var"]

            # Check environment variable
            env_key = os.getenv(env_var) if env_var else None
            # Check config file
            config_key = saved_keys.get(provider)

            if env_key:
                masked = f"{env_key[:8]}...{env_key[-4:]}" if len(env_key) > 12 else "***"
                print(f"✅ {provider_name:20} (from env): {masked}")
            elif config_key:
                masked = f"{config_key[:8]}...{config_key[-4:]}" if len(config_key) > 12 else "***"
                print(f"✅ {provider_name:20} (saved):    {masked}")
            elif provider == "ollama":
                print(f"ℹ️  {provider_name:20} (local):    No API key needed")
            else:
                print(f"❌ {provider_name:20} Not configured")

        print("\n💡 Tip: Use 'autoprepml-config --set <provider>' to configure API keys")
        print("=" * 60 + "\n")
