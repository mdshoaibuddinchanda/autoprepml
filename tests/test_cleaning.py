"""Tests for cleaning module"""
import pandas as pd
import numpy as np
import pytest
from autoprepml import cleaning


def test_impute_missing_auto():
    df = pd.DataFrame({
        'num': [1, 2, None, 4],
        'cat': ['a', None, 'b', 'a']
    })
    result = cleaning.impute_missing(df, strategy='auto')
    assert result['num'].isnull().sum() == 0
    assert result['cat'].isnull().sum() == 0
    assert result['num'].iloc[2] == 2.0  # median of [1,2,4]


def test_impute_missing_median():
    df = pd.DataFrame({'x': [1, 2, None, 4]})
    result = cleaning.impute_missing(df, strategy='median')
    assert result['x'].iloc[2] == 2.0


def test_impute_missing_drop():
    df = pd.DataFrame({'x': [1, None, 3]})
    result = cleaning.impute_missing(df, strategy='drop')
    assert len(result) == 2


def test_scale_features_standard():
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
    result = cleaning.scale_features(df, method='standard')
    assert np.isclose(result['a'].mean(), 0, atol=1e-10)
    assert np.isclose(result['a'].std(ddof=0), 1, atol=1e-10)  # Use ddof=0 for population std


def test_scale_features_minmax():
    df = pd.DataFrame({'a': [1, 2, 3, 4]})
    result = cleaning.scale_features(df, method='minmax')
    assert result['a'].min() == 0.0
    assert result['a'].max() == 1.0


def test_scale_features_exclude():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
    result = cleaning.scale_features(df, method='standard', exclude_cols=['b'])
    assert not np.isclose(result['b'].mean(), 0, atol=0.1)  # b should not be scaled


def test_encode_categorical_label():
    df = pd.DataFrame({'cat': ['a', 'b', 'a', 'c']})
    result = cleaning.encode_categorical(df, method='label')
    assert result['cat'].dtype in [np.int32, np.int64]
    assert len(result['cat'].unique()) == 3


def test_encode_categorical_onehot():
    df = pd.DataFrame({'cat': ['a', 'b', 'a']})
    result = cleaning.encode_categorical(df, method='onehot')
    assert 'cat_b' in result.columns or 'cat_a' in result.columns


def test_balance_classes_oversample():
    df = pd.DataFrame({'x': range(15), 'target': [0]*10 + [1]*5})
    result = cleaning.balance_classes(df, 'target', method='oversample')
    assert result['target'].value_counts()[0] == result['target'].value_counts()[1]
    assert len(result) == 20  # 10 + 10


def test_balance_classes_undersample():
    df = pd.DataFrame({'x': range(15), 'target': [0]*10 + [1]*5})
    result = cleaning.balance_classes(df, 'target', method='undersample')
    assert result['target'].value_counts()[0] == result['target'].value_counts()[1]
    assert len(result) == 10  # 5 + 5


def test_remove_outliers():
    df = pd.DataFrame({'a': range(10)})
    result = cleaning.remove_outliers(df, outlier_indices=[0, 5, 9])
    assert len(result) == 7
    # After reset_index, old indices are gone
    assert result['a'].tolist() == [1, 2, 3, 4, 6, 7, 8]
