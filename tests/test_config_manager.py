"""Tests for configuration manager"""
import pytest
import os
import json
from pathlib import Path
from autoprepml.config_manager import AutoPrepMLConfig


class TestAutoPrepMLConfig:
    """Test configuration management"""
    
    def setup_method(self):
        """Setup test environment"""
        # Use a test config directory
        self.test_config_dir = Path.home() / ".autoprepml_test"
        self.test_config_file = self.test_config_dir / "config.json"
        
        # Override class attributes for testing
        AutoPrepMLConfig.CONFIG_DIR = self.test_config_dir
        AutoPrepMLConfig.CONFIG_FILE = self.test_config_file
        
        # Clean up any existing test config
        if self.test_config_file.exists():
            self.test_config_file.unlink()
        if self.test_config_dir.exists():
            self.test_config_dir.rmdir()
            
    def teardown_method(self):
        """Cleanup test environment"""
        if self.test_config_file.exists():
            self.test_config_file.unlink()
        if self.test_config_dir.exists():
            self.test_config_dir.rmdir()
    
    def _capture_list_output(self, capsys):
        """Helper to capture the output of listing API keys."""
        AutoPrepMLConfig.list_api_keys()
        return capsys.readouterr()
    
    def test_ensure_config_dir(self):
        """Test config directory creation"""
        assert not self.test_config_dir.exists()
        AutoPrepMLConfig.ensure_config_dir()
        assert self.test_config_dir.exists()
        
    def test_set_and_get_api_key(self):
        """Test setting and getting API keys"""
        test_key = "sk-test-key-12345"
        self._extracted_from_test_config_file_priority_over_missing_env_4(test_key)
        
    def test_multiple_api_keys(self):
        """Test managing multiple API keys"""
        AutoPrepMLConfig.set_api_key('openai', 'sk-openai-test')
        AutoPrepMLConfig.set_api_key('anthropic', 'sk-ant-test')
        
        assert AutoPrepMLConfig.get_api_key('openai') == 'sk-openai-test'
        assert AutoPrepMLConfig.get_api_key('anthropic') == 'sk-ant-test'
        
    def test_remove_api_key(self, capsys):
        """Test removing API key"""
        AutoPrepMLConfig.set_api_key('openai', 'test-key')
        assert AutoPrepMLConfig.get_api_key('openai') is not None
        
        AutoPrepMLConfig.remove_api_key('openai')
        assert AutoPrepMLConfig.get_api_key('openai') is None
        
        captured = capsys.readouterr()
        assert "removed" in captured.out.lower()
        
    def test_remove_nonexistent_key(self, capsys):
        """Test removing non-existent API key"""
        AutoPrepMLConfig.remove_api_key('openai')
        captured = capsys.readouterr()
        assert "no api key found" in captured.out.lower()
        
    def test_environment_variable_priority(self):
        """Test that environment variables are checked"""
        # Set environment variable
        os.environ['OPENAI_API_KEY'] = 'env-test-key'
        
        try:
            # Should get from environment
            key = AutoPrepMLConfig.get_api_key('openai')
            assert key == 'env-test-key'
        finally:
            # Clean up
            del os.environ['OPENAI_API_KEY']
            
    def test_config_file_priority_over_missing_env(self):
        """Test config file is used when env var doesn't exist"""
        # Make sure env var doesn't exist
# sourcery skip: no-conditionals-in-tests
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']

        self._extracted_from_test_config_file_priority_over_missing_env_4(
            'config-file-key'
        )

    # TODO Rename this here and in `test_set_and_get_api_key` and `test_config_file_priority_over_missing_env`
    def _extracted_from_test_config_file_priority_over_missing_env_4(self, arg0):
        AutoPrepMLConfig.set_api_key('openai', arg0)
        retrieved_key = AutoPrepMLConfig.get_api_key('openai')
        assert retrieved_key == arg0
        
    def test_invalid_provider(self):
        """Test setting API key for invalid provider"""
        with pytest.raises(ValueError, match="Unknown provider"):
            AutoPrepMLConfig.set_api_key('invalid_provider', 'test-key')
            
    def test_providers_list(self):
        """Test that all expected providers are supported"""
        expected_providers = ['openai', 'anthropic', 'google', 'ollama']
# sourcery skip: no-loop-in-tests
        for provider in expected_providers:
            assert provider in AutoPrepMLConfig.PROVIDERS
            
    def test_provider_metadata(self):
        """Test provider metadata structure"""
# sourcery skip: no-loop-in-tests
        for provider, info in AutoPrepMLConfig.PROVIDERS.items():
            assert 'name' in info
    def test_list_api_keys_empty(self, capsys):
        """Test listing when no keys configured"""
        captured = self._capture_list_output(capsys)
        
        assert "API Key Configuration" in captured.out
        assert "Not configured" in captured.out
        AutoPrepMLConfig.list_api_keys()
        captured = capsys.readouterr()
    def test_list_api_keys_with_keys(self, capsys):
        """Test listing configured keys"""
        AutoPrepMLConfig.set_api_key('openai', 'sk-proj-test12345678')
        AutoPrepMLConfig.list_api_keys()
        captured = capsys.readouterr()
        
        assert "OpenAI" in captured.out
        assert "sk-proj-..." in captured.out  # Should be masked
        assert "test12345678" not in captured.out  # Full key should not appear
        
    def test_config_persistence(self):
        """Test that config persists across instances"""
        AutoPrepMLConfig.set_api_key('openai', 'persistent-key')
        
        # Load config in a fresh instance
        config = AutoPrepMLConfig.load_config()
        assert config['api_keys']['openai'] == 'persistent-key'
        
    def test_empty_config_load(self):
        """Test loading non-existent config returns empty dict"""
        config = AutoPrepMLConfig.load_config()
        assert isinstance(config, dict)
        assert len(config) == 0
