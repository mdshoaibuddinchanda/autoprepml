"""Tests for core AutoPrepML class"""
import pandas as pd
import numpy as np
from autoprepml.core import AutoPrepML


def test_autoprepml_init():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    prep = AutoPrepML(df)
    assert prep.original_df.shape == (3, 2)
    assert len(prep.log) > 0  # initialization log


def test_detect():
    df = pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': ['x', None, 'y', 'z']
    })
    prep = AutoPrepML(df)
    results = prep.detect()
    assert 'missing_values' in results
    assert 'outliers' in results


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

