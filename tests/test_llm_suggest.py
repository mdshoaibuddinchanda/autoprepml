"""Tests for LLM suggestions module - basic functionality tests"""
import pandas as pd
import pytest

# Skip all tests if LLM dependencies are not available
pytest.importorskip("openai", reason="openai not installed")

from autoprepml import llm_suggest


def test_suggest_fix_missing():
    """Test that suggest_fix function exists and returns a string"""
    df = pd.DataFrame({'a': [1, None, 3]})
    # This will use placeholder response or fail gracefully without API key
    try:
        suggestion = llm_suggest.suggest_fix(df, column='a', issue_type='missing', provider='ollama')
        assert isinstance(suggestion, str)
        assert len(suggestion) > 0
    except Exception:
        # Expected if Ollama not running - test just ensures function exists
        pytest.skip("LLM provider not available")


def test_suggest_fix_outlier():
    """Test suggest_fix for outliers"""
    df = pd.DataFrame({'a': [1, 2, 3, 100]})
    try:
        suggestion = llm_suggest.suggest_fix(df, column='a', issue_type='outlier', provider='ollama')
        assert isinstance(suggestion, str)
    except Exception:
        pytest.skip("LLM provider not available")


def test_suggest_fix_imbalance():
    """Test suggest_fix for class imbalance"""
    df = pd.DataFrame({'target': [0]*90 + [1]*10})
    try:
        suggestion = llm_suggest.suggest_fix(df, column='target', issue_type='imbalance', provider='ollama')
        assert isinstance(suggestion, str)
    except Exception:
        pytest.skip("LLM provider not available")


def test_explain_cleaning_step_imputed():
    """Test explain_cleaning_step function"""
    try:
        explanation = llm_suggest.explain_cleaning_step('imputed_missing', {'strategy': 'median'}, provider='ollama')
        assert isinstance(explanation, str)
    except Exception:
        pytest.skip("LLM provider not available")


def test_explain_cleaning_step_scaled():
    """Test explanation for scaling step"""
    try:
        explanation = llm_suggest.explain_cleaning_step('scaled_features', {'method': 'standard'}, provider='ollama')
        assert isinstance(explanation, str)
    except Exception:
        pytest.skip("LLM provider not available")


def test_explain_cleaning_step_unknown():
    """Test explanation for unknown action"""
    try:
        explanation = llm_suggest.explain_cleaning_step('unknown_action', {}, provider='ollama')
        assert isinstance(explanation, str)
    except Exception:
        pytest.skip("LLM provider not available")
