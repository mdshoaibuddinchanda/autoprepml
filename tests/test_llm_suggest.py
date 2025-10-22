"""Tests for LLM suggestions module"""
import pandas as pd
from autoprepml import llm_suggest


def test_suggest_fix_missing():
    df = pd.DataFrame({'a': [1, None, 3]})
    suggestion = llm_suggest.suggest_fix(df, column='a', issue_type='missing')
    assert isinstance(suggestion, str)
    assert len(suggestion) > 0
    assert 'missing' in suggestion.lower() or 'imputation' in suggestion.lower()


def test_suggest_fix_outlier():
    df = pd.DataFrame({'a': [1, 2, 3, 100]})
    suggestion = llm_suggest.suggest_fix(df, column='a', issue_type='outlier')
    assert isinstance(suggestion, str)
    assert 'outlier' in suggestion.lower()


def test_suggest_fix_imbalance():
    df = pd.DataFrame({'target': [0]*90 + [1]*10})
    suggestion = llm_suggest.suggest_fix(df, column='target', issue_type='imbalance')
    assert isinstance(suggestion, str)
    assert 'imbalance' in suggestion.lower() or 'class' in suggestion.lower()


def test_explain_cleaning_step_imputed():
    explanation = llm_suggest.explain_cleaning_step('imputed_missing', {'strategy': 'median'})
    assert isinstance(explanation, str)
    assert 'median' in explanation.lower()


def test_explain_cleaning_step_scaled():
    explanation = llm_suggest.explain_cleaning_step('scaled_features', {'method': 'standard'})
    assert isinstance(explanation, str)
    assert 'standard' in explanation.lower()


def test_explain_cleaning_step_unknown():
    explanation = llm_suggest.explain_cleaning_step('unknown_action', {})
    assert isinstance(explanation, str)
    assert 'unknown_action' in explanation.lower()
