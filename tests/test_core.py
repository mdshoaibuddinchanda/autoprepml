"""Tests for core AutoPrepML class"""

import pandas as pd
import pytest
from autoprepml.core import AutoPrepML, _data_fingerprint


def test_autoprepml_init():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    prep = AutoPrepML(df)
    assert prep.original_df.shape == (3, 2)
    assert len(prep.log) > 0  # initialization log


def test_autoprepml_rejects_invalid_input():
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        AutoPrepML([1, 2, 3])

    with pytest.raises(ValueError, match="DataFrame cannot be empty"):
        AutoPrepML(pd.DataFrame())


def test_detect():
    df = pd.DataFrame({"a": [1, 2, None, 4], "b": ["x", None, "y", "z"]})
    prep = AutoPrepML(df)
    results = prep.detect()
    assert "missing_values" in results
    assert "outliers" in results


def test_detect_uses_configured_detection_options():
    df = pd.DataFrame({"value": [1.0, 2.0, 100.0, 4.0]})
    prep = AutoPrepML(
        df,
        config={"detection": {"outlier_method": "zscore", "zscore_threshold": 1.0}},
    )

    results = prep.detect()

    assert results["outliers"]["method"] == "zscore"
    assert results["outliers"]["outlier_count"] > 0


def test_detect_reuses_cache_and_returns_isolated_results(monkeypatch):
    """Repeated detection avoids recomputation without sharing mutable results."""
    calls = []

    def fake_detect_all(*_args, **_kwargs):
        calls.append(True)
        return {"missing_values": {}, "outliers": {"outlier_count": 0}}

    monkeypatch.setattr("autoprepml.core.detection.detect_all", fake_detect_all)
    prep = AutoPrepML(pd.DataFrame({"value": [1, 2, 3]}))

    first = prep.detect()
    first["outliers"]["outlier_count"] = 99
    second = prep.detect()

    assert len(calls) == 1
    assert second["outliers"]["outlier_count"] == 0
    assert any(entry["action"] == "detection_cache_hit" for entry in prep.log)


def test_detect_cache_invalidates_when_data_or_options_change(monkeypatch):
    """Data and detector configuration changes never reuse stale results."""
    calls = []

    def fake_detect_all(*_args, **_kwargs):
        calls.append(True)
        return {"missing_values": {}, "outliers": {"outlier_count": len(calls)}}

    monkeypatch.setattr("autoprepml.core.detection.detect_all", fake_detect_all)
    prep = AutoPrepML(pd.DataFrame({"value": [1, 2, 3]}))

    prep.detect()
    prep.df.loc[0, "value"] = 10
    prep.detect()
    prep.config["detection"]["zscore_threshold"] = 2.0
    prep.detect()

    assert len(calls) == 3


def test_detect_can_bypass_cache(monkeypatch):
    """Callers can force a fresh detection pass when needed."""
    calls = []

    def fake_detect_all(*_args, **_kwargs):
        calls.append(True)
        return {"missing_values": {}, "outliers": {"outlier_count": 0}}

    monkeypatch.setattr("autoprepml.core.detection.detect_all", fake_detect_all)
    prep = AutoPrepML(pd.DataFrame({"value": [1, 2, 3]}))

    prep.detect()
    prep.detect(use_cache=False)

    assert len(calls) == 2


def test_data_fingerprint_bypasses_unhashable_frames(monkeypatch):
    """Unsupported extension values disable caching without breaking detection."""
    monkeypatch.setattr(
        "autoprepml.core.pd.util.hash_pandas_object",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("unhashable")),
    )

    assert _data_fingerprint(pd.DataFrame({"value": [1, 2]})) is None


def test_summary():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    prep = AutoPrepML(df)
    summary = prep.summary()
    assert summary["shape"] == (3, 2)
    assert "a" in summary["numeric_columns"]
    assert "b" in summary["categorical_columns"]


def test_clean():
    df = pd.DataFrame({"a": [1, 2, None, 4], "b": ["x", "y", "z", "x"]})
    prep = AutoPrepML(df)
    clean_df, report = prep.clean()
    assert clean_df["a"].isnull().sum() == 0
    assert "detection_results" in report


def test_clean_uses_onehot_encoding_by_default_for_nominal_features():
    df = pd.DataFrame({"feature": [1.0, 2.0, 3.0], "category": ["a", "b", "a"]})
    cleaned, _ = AutoPrepML(df).clean()

    assert any(column.startswith("category_") for column in cleaned.columns)


def test_clean_auto_false_does_not_transform():
    df = pd.DataFrame({"value": [1.0, None, 3.0]})
    prep = AutoPrepML(df)

    cleaned, report = prep.clean(auto=False)

    pd.testing.assert_frame_equal(cleaned, df)
    assert report["cleaned_shape"] == df.shape


def test_clean_honors_explicit_balance_method():
    df = pd.DataFrame(
        {
            "feature": range(10),
            "target": [0] * 8 + [1] * 2,
        }
    )
    prep = AutoPrepML(df)

    cleaned, _ = prep.clean(
        task="classification",
        target_col="target",
        balance_method="undersample",
    )

    assert cleaned["target"].value_counts().to_dict() == {0: 2, 1: 2}


def test_clean_classification():
    df = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5, 6],
            "feat2": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 0, 0, 0, 0, 1],  # imbalanced
        }
    )
    prep = AutoPrepML(df)
    clean_df, report = prep.clean(task="classification", target_col="target")
    # Check that balancing was applied
    assert len(clean_df) >= len(df)


def test_clean_drops_missing_targets_instead_of_imputing_labels():
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, None, 4.0],
            "target": [0, 1, None, 1],
        }
    )
    cleaned, report = AutoPrepML(df).clean(task="classification", target_col="target")

    assert len(cleaned) == 3
    assert cleaned["target"].isna().sum() == 0
    assert any(log["action"] == "dropped_missing_targets" for log in report["logs"])


def test_clean_rejects_all_missing_targets():
    df = pd.DataFrame({"feature": [1.0, 2.0], "target": [None, None]})

    with pytest.raises(ValueError, match="No rows remain"):
        AutoPrepML(df).clean(task="classification", target_col="target")


@pytest.mark.parametrize("method", ["knn", "iterative"])
def test_clean_supports_advanced_imputation(method):
    """Advanced imputation methods are wired through the high-level API."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, None, 4.0, 5.0, 6.0],
            "b": [5.0, None, 7.0, 8.0, 9.0, 10.0],
        }
    )

    cleaned, _ = AutoPrepML(df).clean(use_advanced=True, imputation_method=method)

    assert cleaned.isna().sum().sum() == 0


def test_clean_honors_configured_outlier_encoding_and_scaling():
    """Configured cleaning steps are applied without mutating the source frame."""
    df = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 100.0],
            "category": ["a", "b", "a", "b"],
        }
    )
    prep = AutoPrepML(
        df,
        config={
            "cleaning": {
                "remove_outliers": True,
                "encode_method": "onehot",
                "scale_method": "minmax",
            }
        },
    )

    cleaned, _ = prep.clean()

    assert len(cleaned) < len(df)
    assert any(column.startswith("category_") for column in cleaned.columns)
    assert df.equals(prep.original_df)


def test_disabled_and_enabled_llm_helpers():
    """LLM convenience methods have stable behavior in both modes."""
    prep = AutoPrepML(pd.DataFrame({"value": [1, 2, 3]}))
    message = "LLM support not enabled. Initialize with enable_llm=True"
    assert prep.get_llm_suggestions() == message
    assert prep.analyze_with_llm() == message
    assert prep.get_feature_suggestions() == [message]
    assert prep.explain_step("scaled", {}) == message

    class FakeSuggestor:
        def suggest_fix(self, *_args, **_kwargs):
            return "fix"

        def analyze_dataframe(self, *_args, **_kwargs):
            return "analysis"

        def suggest_features(self, *_args, **_kwargs):
            return ["feature"]

        def explain_cleaning_step(self, *_args, **_kwargs):
            return "explanation"

    prep.llm_enabled = True
    prep.llm_suggestor = FakeSuggestor()
    assert prep.get_llm_suggestions(column="value") == "fix"
    assert prep.get_llm_suggestions() == "analysis"
    assert prep.analyze_with_llm(task="regression") == "analysis"
    assert prep.get_feature_suggestions() == ["feature"]
    assert prep.explain_step("scaled", {}) == "explanation"


def test_report():
    df = pd.DataFrame({"a": [1, 2, 3]})
    prep = AutoPrepML(df)
    prep.detect()
    report = prep.report(include_plots=False)
    assert "timestamp" in report
    assert "original_shape" in report
    assert "logs" in report


def test_save_report_creates_nested_parent_and_accepts_path(tmp_path):
    prep = AutoPrepML(pd.DataFrame({"value": [1, 2, 3]}))
    output_path = tmp_path / "nested" / "report.JSON"

    prep.save_report(output_path)

    assert output_path.exists()
