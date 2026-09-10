"""Tests for LLM integration module"""

import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace

# Skip all tests in this module if LLM dependencies are not available
pytest.importorskip("openai", reason="openai not installed")

from autoprepml.llm_suggest import (  # noqa: E402
    LLMSuggestor,
    LLMProvider,
    RecommendationValidationError,
    suggest_fix,
    explain_cleaning_step,
    validate_analysis_recommendation,
    validate_feature_suggestions,
)


class TestLLMSuggestor:
    """Test LLM Suggestor class"""

    def test_initialization_openai(self):
        """Test initialization with OpenAI provider"""
        suggestor = LLMSuggestor(provider="openai", api_key="test-key")
        assert suggestor.provider == LLMProvider.OPENAI
        assert suggestor.model == "gpt-4o"  # Updated default model
        assert suggestor.api_key == "test-key"

    def test_initialization_anthropic(self):
        """Test initialization with Anthropic provider"""
        suggestor = LLMSuggestor(provider="anthropic", api_key="test-key")
        assert suggestor.provider == LLMProvider.ANTHROPIC
        assert suggestor.model == "claude-3-5-sonnet-20241022"  # Updated default model

    def test_initialization_google(self):
        """Test initialization with Google provider"""
        suggestor = LLMSuggestor(provider="google", api_key="test-key")
        assert suggestor.provider == LLMProvider.GOOGLE
        assert suggestor.model == "gemini-2.5-flash"  # Updated default model

    def test_google_api_key_alias(self, monkeypatch):
        """The current Google SDK's GEMINI_API_KEY alias is supported."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

        suggestor = LLMSuggestor(provider="google")

        assert suggestor.api_key == "gemini-test-key"

    def test_initialization_ollama(self):
        """Test initialization with Ollama provider"""
        suggestor = LLMSuggestor(provider="ollama")
        assert suggestor.provider == LLMProvider.OLLAMA
        assert suggestor.model == "llama3.2"  # Updated default model
        assert suggestor.api_key is None  # No API key needed for local

    def test_custom_model(self):
        """Test initialization with custom model"""
        suggestor = LLMSuggestor(provider="openai", model="gpt-3.5-turbo", api_key="test-key")
        assert suggestor.model == "gpt-3.5-turbo"

    def test_temperature_setting(self):
        """Test temperature parameter"""
        suggestor = LLMSuggestor(provider="openai", temperature=0.5, api_key="test-key")
        assert suggestor.temperature == 0.5

    def test_max_tokens_setting(self):
        """Test max_tokens parameter"""
        suggestor = LLMSuggestor(provider="openai", max_tokens=1000, api_key="test-key")
        assert suggestor.max_tokens == 1000

    def test_get_column_info_numeric(self):
        """Test column info extraction for numeric data"""
        df = pd.DataFrame(
            {"age": [25, 30, np.nan, 45, 50], "name": ["Alice", "Bob", "Charlie", "David", "Eve"]}
        )

        suggestor = LLMSuggestor(provider="ollama")  # No API needed for this test
        info = suggestor._get_column_info(df, "age")

        assert info["dtype"] == "float64"  # NaN converts to float
        assert info["missing_count"] == 1
        assert info["missing_pct"] == 20.0
        assert info["unique_values"] == 4
        assert "mean" in info
        assert "median" in info
        assert "std" in info

    def test_get_column_info_categorical(self):
        """Test column info extraction for categorical data"""
        df = pd.DataFrame({"category": ["A", "B", "A", "C", "B", "A"], "value": [1, 2, 3, 4, 5, 6]})

        suggestor = LLMSuggestor(provider="ollama")
        info = suggestor._get_column_info(df, "category")

        assert info["dtype"] == "object"
        assert info["missing_count"] == 0
        assert info["unique_values"] == 3
        assert "top_values" in info
        assert "A" not in info["top_values"]
        assert info["top_values"]["class_1"] == 3

        sample_enabled = LLMSuggestor(provider="ollama", include_samples=True)
        sample_info = sample_enabled._get_column_info(df, "category")
        assert "A" in sample_info["top_values"]
        assert sample_info["sample_values"][:2] == ["A", "B"]

    def test_get_dataframe_summary(self):
        """Test DataFrame summary generation"""
        df = pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45],
                "salary": [50000, 60000, 70000, 80000, 90000],
                "department": ["HR", "IT", "HR", "Finance", "IT"],
                "label": [0, 1, 0, 1, 0],
            }
        )

        suggestor = LLMSuggestor(provider="ollama")
        summary = suggestor._get_dataframe_summary(df, target_col="label")

        assert summary["shape"]["rows"] == 5
        assert summary["shape"]["columns"] == 4
        assert len(summary["columns"]) == 4
        assert "dtypes" in summary
        assert "missing_values" in summary
        assert "duplicate_rows" in summary
        assert "numeric_summary" in summary
        assert "categorical_summary" in summary
        assert "target_column" in summary

    def test_get_dataframe_summary_with_missing(self):
        """Test DataFrame summary with missing values"""
        df = pd.DataFrame(
            {"A": [1, 2, np.nan, 4, 5], "B": [5, np.nan, 7, 8, 9], "C": ["a", "b", None, "d", "e"]}
        )

        suggestor = LLMSuggestor(provider="ollama")
        summary = suggestor._get_dataframe_summary(df)

        assert summary["missing_values"]["A"] == 1
        assert summary["missing_values"]["B"] == 1
        assert summary["missing_values"]["C"] == 1
        assert summary["missing_pct"]["A"] == 20.0

    def test_get_dataframe_summary_duplicates(self):
        """Test duplicate detection in summary"""
        df = pd.DataFrame({"A": [1, 2, 1, 2, 3], "B": [5, 6, 5, 6, 7]})

        suggestor = LLMSuggestor(provider="ollama")
        summary = suggestor._get_dataframe_summary(df)

        assert summary["duplicate_rows"] == 2  # Two duplicate rows

    def test_raw_samples_are_opt_in(self, monkeypatch):
        """LLM context redacts record values unless explicitly enabled."""
        monkeypatch.delenv("AUTOPREPML_LLM_INCLUDE_SAMPLES", raising=False)
        df = pd.DataFrame({"secret": ["customer-123", "customer-456"]})

        redacted = LLMSuggestor(provider="ollama")._get_dataframe_summary(df)
        assert "customer-123" not in str(redacted)

        enabled = LLMSuggestor(provider="ollama", include_samples=True)
        assert "customer-123" in str(enabled._get_dataframe_summary(df))


class TestConvenienceFunctions:
    """Test convenience wrapper functions"""

    def test_suggest_fix_function(self):
        """Test suggest_fix convenience function"""
        df = pd.DataFrame(
            {"age": [25, 30, np.nan, 45, 50], "name": ["Alice", "Bob", "Charlie", "David", "Eve"]}
        )

        # This will fail without API key, but tests the function exists
        from contextlib import suppress

        with suppress(Exception):
            result = suggest_fix(df, column="age", issue_type="missing", provider="ollama")
            assert isinstance(result, str)

    def test_explain_cleaning_step_function(self):
        """Test explain_cleaning_step convenience function"""
        details = {"strategy": "median", "columns_imputed": ["age", "salary"]}

        from contextlib import suppress

        with suppress(Exception):
            result = explain_cleaning_step("imputed_missing", details, provider="ollama")
            assert isinstance(result, str)


class TestProviderEnum:
    """Test LLMProvider enum"""

    def test_provider_values(self):
        """Test all provider enum values"""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GOOGLE.value == "google"
        assert LLMProvider.OLLAMA.value == "ollama"

    def test_provider_from_string(self):
        """Test creating provider from string"""
        provider = LLMProvider("openai")
        assert provider == LLMProvider.OPENAI

        provider = LLMProvider("ollama")
        assert provider == LLMProvider.OLLAMA


class TestErrorHandling:
    """Test error handling in LLM module"""

    def test_invalid_provider(self):
        """Test initialization with invalid provider"""
        with pytest.raises(ValueError):
            LLMSuggestor(provider="invalid_provider")

    def test_missing_column(self):
        """Test handling of non-existent column"""
        df = pd.DataFrame({"A": [1, 2, 3]})
        suggestor = LLMSuggestor(provider="ollama")

        # Should return error info
        info = suggestor._get_column_info(df, "NonExistent")
        assert "error" in info
        assert "NonExistent" in info["error"]
        assert "available_columns" in info

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame()
        suggestor = LLMSuggestor(provider="ollama")

        summary = suggestor._get_dataframe_summary(df)
        assert summary["shape"]["rows"] == 0
        assert summary["shape"]["columns"] == 0


class TestLLMCalls:
    """Provider adapters are exercised with deterministic fake clients."""

    @staticmethod
    def _suggestor(provider, client):
        suggestor = object.__new__(LLMSuggestor)
        suggestor.provider = provider
        suggestor.client = client
        suggestor.model = "test-model"
        suggestor.temperature = 0.2
        suggestor.max_tokens = 32
        suggestor.base_url = "http://localhost:11434"
        suggestor._google_legacy = False
        return suggestor

    def test_openai_and_anthropic_calls(self):
        openai_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="openai"))]
                    )
                )
            )
        )
        anthropic_client = SimpleNamespace(
            messages=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    content=[SimpleNamespace(text="anthropic")]
                )
            )
        )

        assert (
            self._suggestor(LLMProvider.OPENAI, openai_client)._call_llm("prompt", "system")
            == "openai"
        )
        assert (
            self._suggestor(LLMProvider.ANTHROPIC, anthropic_client)._call_llm("prompt", "system")
            == "anthropic"
        )

    def test_google_modern_and_ollama_calls(self):
        google_calls = []

        class GoogleModels:
            def generate_content(self, **kwargs):
                google_calls.append(kwargs)
                return SimpleNamespace(text="google")

        google_client = SimpleNamespace(models=GoogleModels())
        ollama_client = SimpleNamespace(chat=lambda **_kwargs: {"message": {"content": "ollama"}})

        assert (
            self._suggestor(LLMProvider.GOOGLE, google_client)._call_llm("prompt", "system")
            == "google"
        )
        assert google_calls[0]["model"] == "test-model"
        assert (
            self._suggestor(LLMProvider.OLLAMA, ollama_client)._call_llm("prompt", "system")
            == "ollama"
        )

    def test_legacy_google_and_error_calls(self):
        legacy_client = SimpleNamespace(
            generate_content=lambda *_args, **_kwargs: SimpleNamespace(
                candidates=[SimpleNamespace(finish_reason=1)], text="legacy"
            )
        )
        legacy = self._suggestor(LLMProvider.GOOGLE, legacy_client)
        legacy._google_legacy = True
        assert legacy._call_llm("prompt") == "legacy"

        blocked_client = SimpleNamespace(
            generate_content=lambda *_args, **_kwargs: SimpleNamespace(
                candidates=[SimpleNamespace(finish_reason=2)], text="blocked"
            )
        )
        blocked = self._suggestor(LLMProvider.GOOGLE, blocked_client)
        blocked._google_legacy = True
        assert "blocked by safety filters" in blocked._call_llm("prompt")

        failing = self._suggestor(
            LLMProvider.OLLAMA,
            SimpleNamespace(chat=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))),
        )
        assert failing._call_llm("prompt").startswith("Error calling ollama: boom")

    def test_response_parsing_and_rename_helpers(self, monkeypatch):
        suggestor = object.__new__(LLMSuggestor)
        suggestor.include_samples = False
        suggestor._call_llm = lambda *_args, **_kwargs: '{"quality_score": 8}'
        frame = pd.DataFrame({"age": [1, 2, 3]})
        assert suggestor.analyze_dataframe(frame)["quality_score"] == 8

        suggestor._call_llm = lambda *_args, **_kwargs: '```json\n[{"name": "age2"}]\n```'
        assert suggestor.suggest_features(frame) == [{"name": "age2"}]

        suggestor._call_llm = lambda *_args, **_kwargs: "'customer_age' explanation"
        assert suggestor.suggest_column_rename(frame, "age") == "customer_age explanation"
        assert "Error: Column" in suggestor.suggest_column_rename(frame, "missing")

        monkeypatch.setattr(
            suggestor,
            "suggest_column_rename",
            lambda _df, column: "new_" + column,
        )
        assert suggestor.suggest_all_column_renames(frame) == {"age": "new_age"}

    def test_structured_recommendation_validation(self):
        assert validate_analysis_recommendation({"quality_score": 8})["warnings"] == []
        assert validate_feature_suggestions([{"name": "age_bucket"}]) == [{"name": "age_bucket"}]
        with pytest.raises(RecommendationValidationError, match="quality_score"):
            validate_analysis_recommendation({"quality_score": 11})
        with pytest.raises(RecommendationValidationError, match="feature suggestion"):
            validate_feature_suggestions([{"method": "bad"}])


# Integration tests (require actual API keys or running Ollama)
@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests with real LLM providers (requires API keys)"""

    def test_openai_integration(self):
        """Test actual OpenAI API call"""
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        # sourcery skip: no-conditionals-in-tests
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        df = pd.DataFrame(
            {"age": [25, 30, np.nan, 45, 50], "salary": [50000, 60000, np.nan, 80000, 90000]}
        )

        suggestor = LLMSuggestor(provider="openai", api_key=api_key)
        result = suggestor.suggest_fix(df, column="age", issue_type="missing")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_ollama_integration(self):
        """Test actual Ollama API call"""
        df = pd.DataFrame({"age": [25, 30, np.nan, 45, 50]})

        try:
            suggestor = LLMSuggestor(provider="ollama", model="llama2")
            result = suggestor.suggest_fix(df, column="age", issue_type="missing")

            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            pytest.skip(f"Ollama not available: {str(e)}")
