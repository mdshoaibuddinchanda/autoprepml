"""Time series preprocessing module for AutoPrepML"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class TimeSeriesPrepML:
    """Time series preprocessing class.
    
    Example:
        >>> prep = TimeSeriesPrepML(df, timestamp_column='date', value_column='sales')
        >>> clean_df = prep.clean()
        >>> prep.save_report('timeseries_report.html')
    """
    
    def __init__(self, df: pd.DataFrame, timestamp_column: str, value_column: Optional[str] = None):
        """Initialize TimeSeriesPrepML.
        
        Args:
            df: Input DataFrame
            timestamp_column: Name of column containing timestamps
            value_column: Name of column with values (optional)
        """
        self.df = df.copy()
        self.timestamp_column = timestamp_column
        self.value_column = value_column
        self.original_df = df.copy()
        self.log = []

        if timestamp_column not in df.columns:
            raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

        # Convert timestamp column to datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[timestamp_column]):
            try:
                self.df[timestamp_column] = pd.to_datetime(self.df[timestamp_column])
            except Exception as e:
                raise ValueError(
                    f"Could not convert '{timestamp_column}' to datetime: {e}"
                ) from e
    
    def detect_issues(self) -> Dict[str, Any]:
        """Detect time series data quality issues.
        
        Returns:
            Dictionary with detected issues
        """
        ts_col = self.df[self.timestamp_column]
        
        # Check for duplicates
        duplicate_timestamps = ts_col.duplicated().sum()
        
        # Check for missing timestamps (gaps)
        sorted_ts = ts_col.sort_values()
        if len(sorted_ts) > 1:
            time_diffs = sorted_ts.diff().dropna()
            median_diff = time_diffs.median()
            gaps = (time_diffs > median_diff * 2).sum()
        else:
            gaps = 0
        
        # Check for non-chronological order
        is_sorted = ts_col.is_monotonic_increasing
        
        issues = {
            'duplicate_timestamps': int(duplicate_timestamps),
            'detected_gaps': int(gaps),
            'is_chronological': bool(is_sorted),
            'total_records': len(self.df),
            'date_range': {
                'start': str(ts_col.min()),
                'end': str(ts_col.max())
            }
        }
        
        if self.value_column:
            val_col = self.df[self.value_column]
            issues['missing_values'] = int(val_col.isnull().sum())
            issues['zero_values'] = int((val_col == 0).sum())
            issues['negative_values'] = int((val_col < 0).sum())
        
        self.log.append({'action': 'detect_issues', 'result': issues})
        return issues
    
    def sort_by_time(self) -> pd.DataFrame:
        """Sort DataFrame by timestamp column.
        
        Returns:
            Sorted DataFrame
        """
        self.df = self.df.sort_values(by=self.timestamp_column).reset_index(drop=True)
        self.log.append({'action': 'sort_by_time'})
        return self.df
    
    def remove_duplicate_timestamps(self, keep: str = 'first', aggregate: Optional[str] = None) -> pd.DataFrame:
        """Remove or aggregate duplicate timestamps.
        
        Args:
            keep: 'first', 'last', or False
            aggregate: Aggregation method ('mean', 'sum', 'max', 'min') instead of dropping
            
        Returns:
            DataFrame with duplicates handled
        """
        original_len = len(self.df)
        
        if aggregate and self.value_column:
            # Aggregate duplicates
            self.df = self.df.groupby(self.timestamp_column, as_index=False).agg({
                col: aggregate if col == self.value_column else 'first'
                for col in self.df.columns if col != self.timestamp_column
            })
        else:
            # Drop duplicates
            self.df = self.df.drop_duplicates(subset=[self.timestamp_column], keep=keep)
        
        removed = original_len - len(self.df)
        self.log.append({'action': 'remove_duplicate_timestamps', 'removed': removed, 'method': aggregate or keep})
        return self.df
    
    def fill_missing_timestamps(self, freq: str = 'D') -> pd.DataFrame:
        """Fill missing timestamps with specified frequency.
        
        Args:
            freq: Pandas frequency string ('D'=daily, 'H'=hourly, 'W'=weekly, etc.)
            
        Returns:
            DataFrame with complete time range
        """
        self.df = self.df.sort_values(by=self.timestamp_column)
        self.df = self.df.set_index(self.timestamp_column)
        
        # Create complete date range
        full_range = pd.date_range(
            start=self.df.index.min(),
            end=self.df.index.max(),
            freq=freq
        )
        
        # Reindex to fill gaps
        original_len = len(self.df)
        self.df = self.df.reindex(full_range)
        self.df.index.name = self.timestamp_column
        self.df = self.df.reset_index()
        
        added = len(self.df) - original_len
        self.log.append({'action': 'fill_missing_timestamps', 'added': added, 'freq': freq})
        return self.df
    
    def interpolate_missing(self, method: str = 'linear') -> pd.DataFrame:
        """Interpolate missing values in time series.
        
        Args:
            method: Interpolation method ('linear', 'time', 'ffill', 'bfill')
            
        Returns:
            DataFrame with interpolated values
        """
        if not self.value_column:
            raise ValueError("value_column must be specified for interpolation")

        missing_before = self.df[self.value_column].isnull().sum()

        if method in {'linear', 'time'}:
            self.df[self.value_column] = self.df[self.value_column].interpolate(method=method)
        elif method == 'ffill':
            self.df[self.value_column] = self.df[self.value_column].ffill()
        elif method == 'bfill':
            self.df[self.value_column] = self.df[self.value_column].bfill()
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        missing_after = self.df[self.value_column].isnull().sum()
        filled = missing_before - missing_after

        self.log.append({'action': 'interpolate_missing', 'filled': filled, 'method': method})
        return self.df
    
    def detect_outliers(self, method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """Detect outliers in time series values.
        
        Args:
            method: 'zscore' or 'iqr'
            threshold: Z-score threshold or IQR multiplier
            
        Returns:
            DataFrame with outlier flag column
        """
        if not self.value_column:
            raise ValueError("value_column must be specified for outlier detection")

        values = self.df[self.value_column].dropna()

        if method == 'iqr':
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outliers = (values < lower) | (values > upper)
        elif method == 'zscore':
            z_scores = np.abs((values - values.mean()) / values.std())
            outliers = z_scores > threshold
        else:
            raise ValueError(f"Unknown method: {method}")

        self.df[f'{self.value_column}_is_outlier'] = False
        self.df.loc[values.index, f'{self.value_column}_is_outlier'] = outliers.values

        outlier_count = outliers.sum()
        self.log.append({'action': 'detect_outliers', 'count': outlier_count, 'method': method})
        return self.df
    
    def add_time_features(self) -> pd.DataFrame:
        """Extract time-based features from timestamp.
        
        Returns:
            DataFrame with added time features
        """
        ts_col = self.df[self.timestamp_column]
        
        self.df[f'{self.timestamp_column}_year'] = ts_col.dt.year
        self.df[f'{self.timestamp_column}_month'] = ts_col.dt.month
        self.df[f'{self.timestamp_column}_day'] = ts_col.dt.day
        self.df[f'{self.timestamp_column}_dayofweek'] = ts_col.dt.dayofweek
        self.df[f'{self.timestamp_column}_hour'] = ts_col.dt.hour
        self.df[f'{self.timestamp_column}_quarter'] = ts_col.dt.quarter
        self.df[f'{self.timestamp_column}_is_weekend'] = ts_col.dt.dayofweek.isin([5, 6])
        
        self.log.append({'action': 'add_time_features', 'features': 7})
        return self.df
    
    def add_lag_features(self, lags: list = None) -> pd.DataFrame:
        """Add lag features for time series forecasting.
        
        Args:
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        if lags is None:
            lags = [1, 7, 30]
        if not self.value_column:
            raise ValueError("value_column must be specified for lag features")

        for lag in lags:
            self.df[f'{self.value_column}_lag_{lag}'] = self.df[self.value_column].shift(lag)

        self.log.append({'action': 'add_lag_features', 'lags': lags})
        return self.df
    
    def add_rolling_features(self, windows: list = None, functions: list = None) -> pd.DataFrame:
        """Add rolling window statistics.
        
        Args:
            windows: List of window sizes
            functions: List of functions ('mean', 'std', 'min', 'max')
            
        Returns:
            DataFrame with rolling features
        """
        if windows is None:
            windows = [7, 30]
        if functions is None:
            functions = ['mean']
        if not self.value_column:
            raise ValueError("value_column must be specified for rolling features")

        for window in windows:
            for func in functions:
                col_name = f'{self.value_column}_rolling_{func}_{window}'
                if func == 'mean':
                    self.df[col_name] = self.df[self.value_column].rolling(window=window).mean()
                elif func == 'std':
                    self.df[col_name] = self.df[self.value_column].rolling(window=window).std()
                elif func == 'min':
                    self.df[col_name] = self.df[self.value_column].rolling(window=window).min()
                elif func == 'max':
                    self.df[col_name] = self.df[self.value_column].rolling(window=window).max()

        self.log.append({'action': 'add_rolling_features', 'windows': windows, 'functions': functions})
        return self.df
    
    def resample(self, freq: str, agg_func: str = 'sum') -> pd.DataFrame:
        """Resample time series to different frequency.
        
        Args:
            freq: Target frequency ('D', 'W', 'M', 'Q', 'Y')
            agg_func: Aggregation function ('sum', 'mean', 'min', 'max', 'count')
            
        Returns:
            Resampled DataFrame
        """
        self.df = self.df.set_index(self.timestamp_column)
        
        if agg_func == 'sum':
            self.df = self.df.resample(freq).sum()
        elif agg_func == 'mean':
            self.df = self.df.resample(freq).mean()
        elif agg_func == 'min':
            self.df = self.df.resample(freq).min()
        elif agg_func == 'max':
            self.df = self.df.resample(freq).max()
        elif agg_func == 'count':
            self.df = self.df.resample(freq).count()
        
        self.df = self.df.reset_index()
        self.log.append({'action': 'resample', 'freq': freq, 'agg_func': agg_func})
        return self.df
    
    def report(self) -> Dict[str, Any]:
        """Generate preprocessing report.
        
        Returns:
            Report dictionary
        """
        return {
            'original_shape': self.original_df.shape,
            'current_shape': self.df.shape,
            'timestamp_column': self.timestamp_column,
            'value_column': self.value_column,
            'logs': self.log,
            'issues': self.detect_issues()
        }
    
    def save_report(self, output_path: str) -> None:
        """Save preprocessing report to file.
        
        Args:
            output_path: Path to save report (supports .json, .html)
        """
        from .reports import generate_json_report, generate_html_report
        
        report = self.report()
        
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generate_json_report(report))
        elif output_path.endswith('.html'):
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generate_html_report(report))
        else:
            raise ValueError("Output path must end with .json or .html")
