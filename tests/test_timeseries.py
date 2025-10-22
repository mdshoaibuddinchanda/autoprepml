"""Tests for time series preprocessing module"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from autoprepml.timeseries import TimeSeriesPrepML


@pytest.fixture
def sample_timeseries_df():
    """Create sample time series DataFrame"""
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    return pd.DataFrame({
        'date': dates,
        'value': [10, 12, np.nan, 15, 18, 20, 150, 25, 28, 30]  # Note: 150 is outlier
    })


@pytest.fixture
def timeseries_with_gaps():
    """Create time series with missing dates"""
    dates = ['2024-01-01', '2024-01-02', '2024-01-05', '2024-01-08']  # Gaps on 3,4,6,7
    return pd.DataFrame({
        'date': pd.to_datetime(dates),
        'value': [10, 20, 30, 40]
    })


def test_timeseriesprepml_init(sample_timeseries_df):
    """Test TimeSeriesPrepML initialization"""
    prep = TimeSeriesPrepML(sample_timeseries_df, timestamp_column='date', value_column='value')
    
    assert prep.timestamp_column == 'date'
    assert prep.value_column == 'value'
    assert pd.api.types.is_datetime64_any_dtype(prep.df['date'])


def test_timeseriesprepml_init_invalid_column():
    """Test initialization with invalid column"""
    df = pd.DataFrame({'date': ['2024-01-01'], 'value': [10]})
    with pytest.raises(ValueError, match="not found"):
        TimeSeriesPrepML(df, timestamp_column='nonexistent')


def test_timeseriesprepml_init_auto_convert():
    """Test automatic datetime conversion"""
    df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'value': [10, 20, 30]
    })
    
    prep = TimeSeriesPrepML(df, timestamp_column='date', value_column='value')
    assert pd.api.types.is_datetime64_any_dtype(prep.df['date'])


def test_detect_issues(sample_timeseries_df):
    """Test time series issue detection"""
    prep = TimeSeriesPrepML(sample_timeseries_df, timestamp_column='date', value_column='value')
    issues = prep.detect_issues()
    
    assert 'duplicate_timestamps' in issues
    assert 'detected_gaps' in issues
    assert 'is_chronological' in issues
    assert 'missing_values' in issues
    assert issues['total_records'] == 10
    assert issues['missing_values'] == 1


def test_sort_by_time():
    """Test sorting by time"""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-03', '2024-01-01', '2024-01-02']),
        'value': [30, 10, 20]
    })
    
    prep = TimeSeriesPrepML(df, timestamp_column='date', value_column='value')
    result = prep.sort_by_time()
    
    assert result['date'].is_monotonic_increasing
    assert result['value'].tolist() == [10, 20, 30]


def test_remove_duplicate_timestamps():
    """Test duplicate timestamp removal"""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02']),
        'value': [10, 15, 20]
    })
    
    prep = TimeSeriesPrepML(df, timestamp_column='date', value_column='value')
    result = prep.remove_duplicate_timestamps(keep='first')
    
    assert len(result) == 2
    assert result['value'].iloc[0] == 10


def test_remove_duplicate_timestamps_aggregate():
    """Test aggregating duplicate timestamps"""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02']),
        'value': [10, 20, 30]
    })
    
    prep = TimeSeriesPrepML(df, timestamp_column='date', value_column='value')
    result = prep.remove_duplicate_timestamps(aggregate='mean')
    
    assert len(result) == 2
    assert result['value'].iloc[0] == 15  # Average of 10 and 20


def test_fill_missing_timestamps(timeseries_with_gaps):
    """Test filling missing timestamps"""
    prep = TimeSeriesPrepML(timeseries_with_gaps, timestamp_column='date', value_column='value')
    result = prep.fill_missing_timestamps(freq='D')
    
    # Should have 8 days (Jan 1-8)
    assert len(result) == 8
    assert result['value'].isnull().sum() > 0  # New rows have NaN


def test_interpolate_missing(sample_timeseries_df):
    """Test linear interpolation"""
    prep = TimeSeriesPrepML(sample_timeseries_df, timestamp_column='date', value_column='value')
    result = prep.interpolate_missing(method='linear')
    
    # NaN should be filled
    assert result['value'].isnull().sum() == 0


def test_interpolate_missing_ffill():
    """Test forward fill interpolation"""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5, freq='D'),
        'value': [10, np.nan, np.nan, 20, 30]
    })
    
    prep = TimeSeriesPrepML(df, timestamp_column='date', value_column='value')
    result = prep.interpolate_missing(method='ffill')
    
    # NaN values should be forward-filled with 10
    assert result['value'].iloc[1] == 10
    assert result['value'].iloc[2] == 10


def test_detect_outliers_zscore(sample_timeseries_df):
    """Test outlier detection with z-score"""
    prep = TimeSeriesPrepML(sample_timeseries_df, timestamp_column='date', value_column='value')
    result = prep.detect_outliers(method='zscore', threshold=2.0)
    
    assert 'value_is_outlier' in result.columns
    # 150 should be detected as outlier
    outlier_values = result[result['value_is_outlier']]['value'].values
    assert 150 in outlier_values


def test_detect_outliers_iqr(sample_timeseries_df):
    """Test outlier detection with IQR"""
    prep = TimeSeriesPrepML(sample_timeseries_df, timestamp_column='date', value_column='value')
    result = prep.detect_outliers(method='iqr', threshold=1.5)
    
    assert 'value_is_outlier' in result.columns


def test_add_time_features():
    """Test time feature extraction"""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-06-15 14:30:00', '2024-12-25 09:00:00']),
        'value': [10, 20]
    })
    
    prep = TimeSeriesPrepML(df, timestamp_column='date', value_column='value')
    result = prep.add_time_features()
    
    assert 'date_year' in result.columns
    assert 'date_month' in result.columns
    assert 'date_day' in result.columns
    assert 'date_hour' in result.columns
    assert 'date_dayofweek' in result.columns
    assert 'date_quarter' in result.columns
    assert 'date_is_weekend' in result.columns
    
    # Verify values
    assert result['date_year'].iloc[0] == 2024
    assert result['date_month'].iloc[0] == 6
    assert result['date_quarter'].iloc[1] == 4


def test_add_lag_features(sample_timeseries_df):
    """Test lag feature creation"""
    prep = TimeSeriesPrepML(sample_timeseries_df, timestamp_column='date', value_column='value')
    result = prep.add_lag_features(lags=[1, 2])
    
    assert 'value_lag_1' in result.columns
    assert 'value_lag_2' in result.columns
    
    # Verify lag values
    assert pd.isna(result['value_lag_1'].iloc[0])  # First row has no lag
    assert result['value_lag_1'].iloc[1] == result['value'].iloc[0]


def test_add_rolling_features(sample_timeseries_df):
    """Test rolling window statistics"""
    prep = TimeSeriesPrepML(sample_timeseries_df, timestamp_column='date', value_column='value')
    result = prep.add_rolling_features(windows=[3], functions=['mean', 'std'])
    
    assert 'value_rolling_mean_3' in result.columns
    assert 'value_rolling_std_3' in result.columns
    
    # First 2 rows should be NaN (window size 3)
    assert pd.isna(result['value_rolling_mean_3'].iloc[0])
    assert pd.isna(result['value_rolling_mean_3'].iloc[1])


def test_resample_daily_to_weekly():
    """Test resampling to different frequency"""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=14, freq='D'),
        'value': range(14)
    })
    
    prep = TimeSeriesPrepML(df, timestamp_column='date', value_column='value')
    result = prep.resample(freq='W', agg_func='sum')
    
    # 14 days = 2 weeks (approximately)
    assert len(result) <= 3


def test_resample_mean():
    """Test resampling with mean aggregation"""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=6, freq='D'),
        'value': [10, 20, 30, 40, 50, 60]
    })
    
    prep = TimeSeriesPrepML(df, timestamp_column='date', value_column='value')
    result = prep.resample(freq='2D', agg_func='mean')
    
    # Should have 3 rows (6 days / 2)
    assert len(result) == 3


def test_report(sample_timeseries_df):
    """Test report generation"""
    prep = TimeSeriesPrepML(sample_timeseries_df, timestamp_column='date', value_column='value')
    prep.sort_by_time()
    prep.interpolate_missing()
    
    report = prep.report()
    
    assert 'original_shape' in report
    assert 'current_shape' in report
    assert 'timestamp_column' in report
    assert 'value_column' in report
    assert 'logs' in report
    assert 'issues' in report
    assert len(report['logs']) > 0


def test_chained_operations(timeseries_with_gaps):
    """Test chaining multiple operations"""
    prep = TimeSeriesPrepML(timeseries_with_gaps, timestamp_column='date', value_column='value')
    
    prep.sort_by_time()
    prep.fill_missing_timestamps(freq='D')
    prep.interpolate_missing(method='linear')
    prep.add_time_features()
    result = prep.add_lag_features(lags=[1])
    
    # All operations should be applied
    assert len(result) == 8  # 8 days total
    assert result['value'].isnull().sum() == 0  # No missing values
    assert 'date_year' in result.columns  # Time features added
    assert 'value_lag_1' in result.columns  # Lag features added


def test_multiple_value_columns():
    """Test with multiple value columns"""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5, freq='D'),
        'sales': [100, 120, 130, 110, 140],
        'profit': [20, 25, 30, 22, 35]
    })
    
    # Process sales column
    prep = TimeSeriesPrepML(df, timestamp_column='date', value_column='sales')
    result = prep.add_lag_features(lags=[1])
    
    assert 'sales_lag_1' in result.columns
    assert 'profit' in result.columns  # Other column preserved
