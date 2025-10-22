"""Tests for visualization module"""
import pandas as pd
import numpy as np
from autoprepml import visualization


def test_plot_missing():
    df = pd.DataFrame({
        'a': [1, None, 3],
        'b': ['x', 'y', 'z']
    })
    img_str = visualization.plot_missing(df)
    assert isinstance(img_str, str)
    assert len(img_str) > 0  # base64 string


def test_plot_missing_no_missing():
    df = pd.DataFrame({'a': [1, 2, 3]})
    img_str = visualization.plot_missing(df)
    assert isinstance(img_str, str)


def test_plot_outliers():
    df = pd.DataFrame({'x': [1, 2, 3, 100]})
    img_str = visualization.plot_outliers(df, outlier_indices=[3])
    assert isinstance(img_str, str)
    assert len(img_str) > 0


def test_plot_distributions():
    df = pd.DataFrame({'a': np.random.randn(100), 'b': np.random.randn(100)})
    img_str = visualization.plot_distributions(df)
    assert isinstance(img_str, str)
    assert len(img_str) > 0


def test_plot_correlation():
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [2, 4, 6, 8]})
    img_str = visualization.plot_correlation(df)
    assert isinstance(img_str, str)
    assert len(img_str) > 0


def test_plot_correlation_insufficient_columns():
    df = pd.DataFrame({'a': [1, 2, 3]})
    img_str = visualization.plot_correlation(df)
    assert isinstance(img_str, str)


def test_generate_all_plots():
    df = pd.DataFrame({
        'a': [1, 2, None, 100],
        'b': [10, 20, 30, 40]
    })
    plots = visualization.generate_all_plots(df, outlier_indices=[3])
    assert 'missing_plot' in plots
    assert 'outlier_plot' in plots
    assert 'distribution_plot' in plots
    assert 'correlation_plot' in plots
    assert all(isinstance(v, str) for v in plots.values())
