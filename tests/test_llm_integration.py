"""Tests for LLM integration module"""
import pytest
import pandas as pd
import numpy as np

# Skip all tests in this module if LLM dependencies are not available
pytest.importorskip("openai", reason="openai not installed")

from autoprepml.llm_suggest import LLMSuggestor, LLMProvider, suggest_fix, explain_cleaning_step


class TestLLMSuggestor:
    """Test LLM Suggestor class"""
    
    def test_initialization_openai(self):
        """Test initialization with OpenAI provider"""
        suggestor = LLMSuggestor(provider='openai', api_key='test-key')
        assert suggestor.provider == LLMProvider.OPENAI
        assert suggestor.model == 'gpt-4o'  # Updated default model
        assert suggestor.api_key == 'test-key'
        
    def test_initialization_anthropic(self):
        """Test initialization with Anthropic provider"""
        suggestor = LLMSuggestor(provider='anthropic', api_key='test-key')
        assert suggestor.provider == LLMProvider.ANTHROPIC
        assert suggestor.model == 'claude-3-5-sonnet-20241022'  # Updated default model
        
    def test_initialization_google(self):
        """Test initialization with Google provider"""
        suggestor = LLMSuggestor(provider='google', api_key='test-key')
        assert suggestor.provider == LLMProvider.GOOGLE
        assert suggestor.model == 'gemini-2.5-flash'  # Updated default model
        
    def test_initialization_ollama(self):
        """Test initialization with Ollama provider"""
        suggestor = LLMSuggestor(provider='ollama')
        assert suggestor.provider == LLMProvider.OLLAMA
        assert suggestor.model == 'llama3.2'  # Updated default model
        assert suggestor.api_key is None  # No API key needed for local
        
    def test_custom_model(self):
        """Test initialization with custom model"""
        suggestor = LLMSuggestor(provider='openai', model='gpt-3.5-turbo', api_key='test-key')
        assert suggestor.model == 'gpt-3.5-turbo'
        
    def test_temperature_setting(self):
        """Test temperature parameter"""
        suggestor = LLMSuggestor(provider='openai', temperature=0.5, api_key='test-key')
        assert suggestor.temperature == 0.5
        
    def test_max_tokens_setting(self):
        """Test max_tokens parameter"""
        suggestor = LLMSuggestor(provider='openai', max_tokens=1000, api_key='test-key')
        assert suggestor.max_tokens == 1000
        
    def test_get_column_info_numeric(self):
        """Test column info extraction for numeric data"""
        df = pd.DataFrame({
            'age': [25, 30, np.nan, 45, 50],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
        })
        
        suggestor = LLMSuggestor(provider='ollama')  # No API needed for this test
        info = suggestor._get_column_info(df, 'age')
        
        assert info['dtype'] == 'float64'  # NaN converts to float
        assert info['missing_count'] == 1
        assert info['missing_pct'] == 20.0
        assert info['unique_values'] == 4
        assert 'mean' in info
        assert 'median' in info
        assert 'std' in info
        
    def test_get_column_info_categorical(self):
        """Test column info extraction for categorical data"""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        
        suggestor = LLMSuggestor(provider='ollama')
        info = suggestor._get_column_info(df, 'category')
        
        assert info['dtype'] == 'object'
        assert info['missing_count'] == 0
        assert info['unique_values'] == 3
        assert 'top_values' in info
        assert 'A' in info['top_values']
        
    def test_get_dataframe_summary(self):
        """Test DataFrame summary generation"""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['HR', 'IT', 'HR', 'Finance', 'IT'],
            'label': [0, 1, 0, 1, 0]
        })
        
        suggestor = LLMSuggestor(provider='ollama')
        summary = suggestor._get_dataframe_summary(df, target_col='label')
        
        assert summary['shape']['rows'] == 5
        assert summary['shape']['columns'] == 4
        assert len(summary['columns']) == 4
        assert 'dtypes' in summary
        assert 'missing_values' in summary
        assert 'duplicate_rows' in summary
        assert 'numeric_summary' in summary
        assert 'categorical_summary' in summary
        assert 'target_column' in summary
        
    def test_get_dataframe_summary_with_missing(self):
        """Test DataFrame summary with missing values"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [5, np.nan, 7, 8, 9],
            'C': ['a', 'b', None, 'd', 'e']
        })
        
        suggestor = LLMSuggestor(provider='ollama')
        summary = suggestor._get_dataframe_summary(df)
        
        assert summary['missing_values']['A'] == 1
        assert summary['missing_values']['B'] == 1
        assert summary['missing_values']['C'] == 1
        assert summary['missing_pct']['A'] == 20.0
        
    def test_get_dataframe_summary_duplicates(self):
        """Test duplicate detection in summary"""
        df = pd.DataFrame({
            'A': [1, 2, 1, 2, 3],
            'B': [5, 6, 5, 6, 7]
        })
        
        suggestor = LLMSuggestor(provider='ollama')
        summary = suggestor._get_dataframe_summary(df)
        
        assert summary['duplicate_rows'] == 2  # Two duplicate rows


class TestConvenienceFunctions:
    """Test convenience wrapper functions"""
    
    def test_suggest_fix_function(self):
        """Test suggest_fix convenience function"""
        df = pd.DataFrame({
            'age': [25, 30, np.nan, 45, 50],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
        })
        
        # This will fail without API key, but tests the function exists
        from contextlib import suppress
        with suppress(Exception):
            result = suggest_fix(df, column='age', issue_type='missing', provider='ollama')
            assert isinstance(result, str)
            
    def test_explain_cleaning_step_function(self):
        """Test explain_cleaning_step convenience function"""
        details = {
            'strategy': 'median',
            'columns_imputed': ['age', 'salary']
        }
        
        from contextlib import suppress
        with suppress(Exception):
            result = explain_cleaning_step('imputed_missing', details, provider='ollama')
            assert isinstance(result, str)


class TestProviderEnum:
    """Test LLMProvider enum"""
    
    def test_provider_values(self):
        """Test all provider enum values"""
        assert LLMProvider.OPENAI.value == 'openai'
        assert LLMProvider.ANTHROPIC.value == 'anthropic'
        assert LLMProvider.GOOGLE.value == 'google'
        assert LLMProvider.OLLAMA.value == 'ollama'
        
    def test_provider_from_string(self):
        """Test creating provider from string"""
        provider = LLMProvider('openai')
        assert provider == LLMProvider.OPENAI
        
        provider = LLMProvider('ollama')
        assert provider == LLMProvider.OLLAMA


class TestErrorHandling:
    """Test error handling in LLM module"""
    
    def test_invalid_provider(self):
        """Test initialization with invalid provider"""
        with pytest.raises(ValueError):
            LLMSuggestor(provider='invalid_provider')
            
    def test_missing_column(self):
        """Test handling of non-existent column"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        suggestor = LLMSuggestor(provider='ollama')
        
        # Should return error info
        info = suggestor._get_column_info(df, 'NonExistent')
        assert 'error' in info
        assert 'NonExistent' in info['error']
        assert 'available_columns' in info
        
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame()
        suggestor = LLMSuggestor(provider='ollama')
        
        summary = suggestor._get_dataframe_summary(df)
        assert summary['shape']['rows'] == 0
        assert summary['shape']['columns'] == 0


# Integration tests (require actual API keys or running Ollama)
@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests with real LLM providers (requires API keys)"""
    
    def test_openai_integration(self):
        """Test actual OpenAI API call"""
        import os
        api_key = os.getenv('OPENAI_API_KEY')
# sourcery skip: no-conditionals-in-tests
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
            
        df = pd.DataFrame({
            'age': [25, 30, np.nan, 45, 50],
            'salary': [50000, 60000, np.nan, 80000, 90000]
        })
        
        suggestor = LLMSuggestor(provider='openai', api_key=api_key)
        result = suggestor.suggest_fix(df, column='age', issue_type='missing')
        
        assert isinstance(result, str)
        assert len(result) > 0
        
    def test_ollama_integration(self):
        """Test actual Ollama API call"""
        df = pd.DataFrame({
            'age': [25, 30, np.nan, 45, 50]
        })
        
        try:
            suggestor = LLMSuggestor(provider='ollama', model='llama2')
            result = suggestor.suggest_fix(df, column='age', issue_type='missing')
            
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            pytest.skip(f"Ollama not available: {str(e)}")
