"""Tests for core AutoPrepML class"""
import pandas as pd
import pytest
from autoprepml.core import AutoPrepML


def test_autoprepml_init():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    prep = AutoPrepML(df)
    assert prep.original_df.shape == (3, 2)
    assert len(prep.log) > 0  # initialization log


def test_autoprepml_rejects_invalid_input():
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        AutoPrepML([1, 2, 3])

    with pytest.raises(ValueError, match="DataFrame cannot be empty"):
        AutoPrepML(pd.DataFrame())


def test_detect():
    df = pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': ['x', None, 'y', 'z']
    })
    prep = AutoPrepML(df)
    results = prep.detect()
    assert 'missing_values' in results
    assert 'outliers' in results


def test_detect_uses_configured_detection_options():
    df = pd.DataFrame({'value': [1.0, 2.0, 100.0, 4.0]})
    prep = AutoPrepML(
        df,
        config={'detection': {'outlier_method': 'zscore', 'zscore_threshold': 1.0}},
    )

    results = prep.detect()

    assert results['outliers']['method'] == 'zscore'
    assert results['outliers']['outlier_count'] > 0


def test_summary():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    prep = AutoPrepML(df)
    summary = prep.summary()
    assert summary['shape'] == (3, 2)
    assert 'a' in summary['numeric_columns']
    assert 'b' in summary['categorical_columns']


def test_clean():
    df = pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': ['x', 'y', 'z', 'x']
    })
    prep = AutoPrepML(df)
    clean_df, report = prep.clean()
    assert clean_df['a'].isnull().sum() == 0
    assert 'detection_results' in report


def test_clean_auto_false_does_not_transform():
    df = pd.DataFrame({'value': [1.0, None, 3.0]})
    prep = AutoPrepML(df)

    cleaned, report = prep.clean(auto=False)

    pd.testing.assert_frame_equal(cleaned, df)
    assert report['cleaned_shape'] == df.shape


def test_clean_honors_explicit_balance_method():
    df = pd.DataFrame({
        'feature': range(10),
        'target': [0] * 8 + [1] * 2,
    })
    prep = AutoPrepML(df)

    cleaned, _ = prep.clean(
        task='classification',
        target_col='target',
        balance_method='undersample',
    )

    assert cleaned['target'].value_counts().to_dict() == {0: 2, 1: 2}


def test_clean_classification():
    df = pd.DataFrame({
        'feat1': [1, 2, 3, 4, 5, 6],
        'feat2': ['a', 'b', 'a', 'b', 'a', 'b'],
        'target': [0, 0, 0, 0, 0, 1]  # imbalanced
    })
    prep = AutoPrepML(df)
    clean_df, report = prep.clean(task='classification', target_col='target')
    # Check that balancing was applied
    assert len(clean_df) >= len(df)


def test_report():
    df = pd.DataFrame({'a': [1, 2, 3]})
    prep = AutoPrepML(df)
    prep.detect()
    report = prep.report(include_plots=False)
    assert 'timestamp' in report
    assert 'original_shape' in report
    assert 'logs' in report


def test_save_report_creates_nested_parent_and_accepts_path(tmp_path):
    prep = AutoPrepML(pd.DataFrame({'value': [1, 2, 3]}))
    output_path = tmp_path / 'nested' / 'report.JSON'

    prep.save_report(output_path)

    assert output_path.exists()
