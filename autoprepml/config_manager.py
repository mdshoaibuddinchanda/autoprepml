"""Configuration management for AutoPrepML API keys and settings"""
import os
import json
from pathlib import Path
from typing import Optional, Dict


class AutoPrepMLConfig:
    """Manage AutoPrepML configuration and API keys"""
    
    CONFIG_DIR = Path.home() / ".autoprepml"
    CONFIG_FILE = CONFIG_DIR / "config.json"
    
    PROVIDERS = {
        'openai': {
            'name': 'OpenAI',
            'env_var': 'OPENAI_API_KEY',
            'instructions': 'Get your API key from https://platform.openai.com/api-keys'
        },
        'anthropic': {
            'name': 'Anthropic (Claude)',
            'env_var': 'ANTHROPIC_API_KEY',
            'instructions': 'Get your API key from https://console.anthropic.com/settings/keys'
        },
        'google': {
            'name': 'Google (Gemini)',
            'env_var': 'GOOGLE_API_KEY',
            'instructions': 'Get your API key from https://makersuite.google.com/app/apikey'
        },
        'ollama': {
            'name': 'Ollama (Local)',
            'env_var': None,
            'instructions': 'Install Ollama from https://ollama.ai/ - No API key needed!'
        }
    }
    
    @classmethod
    def ensure_config_dir(cls):
        """Create config directory if it doesn't exist"""
        cls.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
    @classmethod
    def load_config(cls) -> Dict:
        """Load configuration from file"""
        if cls.CONFIG_FILE.exists():
            with open(cls.CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {}
        
    @classmethod
    def save_config(cls, config: Dict):
        """Save configuration to file"""
        cls.ensure_config_dir()
        with open(cls.CONFIG_FILE, 'w') as f:
            json.dump(config, indent=2, fp=f)
            
    @classmethod
    def set_api_key(cls, provider: str, api_key: str):
        """Set API key for a provider"""
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Valid providers: {', '.join(cls.PROVIDERS.keys())}")
            
        config = cls.load_config()
        if 'api_keys' not in config:
            config['api_keys'] = {}
            
        config['api_keys'][provider] = api_key
        cls.save_config(config)
        
        print(f"✅ API key for {cls.PROVIDERS[provider]['name']} saved successfully!")
        
    @classmethod
    def get_api_key(cls, provider: str) -> Optional[str]:
        """Get API key for a provider (from config or environment)"""
        # First check environment variable
        if (env_var := cls.PROVIDERS.get(provider, {}).get('env_var')) and (env_key := os.getenv(env_var)):
            return env_key

        # Then check config file
        config = cls.load_config()
        return config.get('api_keys', {}).get(provider)
        
    @classmethod
    def remove_api_key(cls, provider: str):
        """Remove API key for a provider"""
        config = cls.load_config()
        if 'api_keys' in config and provider in config['api_keys']:
            del config['api_keys'][provider]
            cls.save_config(config)
            print(f"✅ API key for {cls.PROVIDERS[provider]['name']} removed!")
        else:
            print(f"ℹ️  No API key found for {cls.PROVIDERS[provider]['name']}")
            
    @classmethod
    def list_api_keys(cls):
        """List all configured API keys (masked)"""
        print("\n🔑 AutoPrepML API Key Configuration")
        print("=" * 60)
        
        config = cls.load_config()
        saved_keys = config.get('api_keys', {})
        
        for provider, info in cls.PROVIDERS.items():
            provider_name = info['name']
            env_var = info['env_var']
            
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
            elif provider == 'ollama':
                print(f"ℹ️  {provider_name:20} (local):    No API key needed")
            else:
                print(f"❌ {provider_name:20} Not configured")

        print("\n💡 Tip: Use 'autoprepml-config --set <provider>' to configure API keys")
        print("=" * 60 + "\n")
