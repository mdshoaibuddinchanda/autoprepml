"""Tests for detection module"""
import pandas as pd
import numpy as np
from autoprepml import detection


def test_detect_missing():
    df = pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': ['x', None, 'y', 'z'],
        'c': [1, 2, 3, 4]
    })
    result = detection.detect_missing(df)
    assert 'a' in result
    assert 'b' in result
    assert 'c' not in result
    assert result['a']['count'] == 1
    assert result['a']['percent'] == 25.0


def test_detect_missing_no_missing():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    result = detection.detect_missing(df)
    assert len(result) == 0


def test_detect_outliers_iforest():
    df = pd.DataFrame({'x': [1, 2, 3, 1000, 2, 3]})
    result = detection.detect_outliers(df, method='iforest', contamination=0.2)
    assert result['outlier_count'] >= 1
    assert 'outlier_indices' in result


def test_detect_outliers_zscore():
    df = pd.DataFrame({'x': [1, 2, 3, 100, 2, 3]})
    result = detection.detect_outliers(df, method='zscore', threshold=2.0)
    assert 'outlier_count' in result
    assert result['method'] == 'zscore'


def test_detect_outliers_no_numeric():
    df = pd.DataFrame({'a': ['x', 'y', 'z']})
    result = detection.detect_outliers(df)
    assert result['outlier_count'] == 0


def test_detect_imbalance():
    df = pd.DataFrame({'target': [0]*90 + [1]*10})
    result = detection.detect_imbalance(df, 'target', threshold=0.3)
    assert result['is_imbalanced'] == True
    assert result['minority_class'] == 1  # Integer from pandas index
    assert result['minority_proportion'] == 0.1


def test_detect_imbalance_balanced():
    df = pd.DataFrame({'target': [0]*50 + [1]*50})
    result = detection.detect_imbalance(df, 'target', threshold=0.3)
    assert result['is_imbalanced'] == False


def test_detect_all():
    df = pd.DataFrame({
        'a': [1, None, 3, 1000],
        'b': ['x', 'y', None, 'z'],
        'target': [0, 0, 1, 1]
    })
    result = detection.detect_all(df, target_col='target')
    assert 'missing_values' in result
    assert 'outliers' in result
    assert 'class_imbalance' in result
