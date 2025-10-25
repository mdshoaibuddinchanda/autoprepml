"""Automated Feature Engineering module for AutoPrepML"""

import contextlib
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, f_classif, f_regression
import warnings


class AutoFeatureEngine:
    """Automated Feature Engineering class.
    
    Provides comprehensive feature engineering capabilities:
    - Polynomial features
    - Interaction features
    - Binning/discretization
    - Encoding enhancements
    - Time-based features
    - Automated feature selection
    - Feature importance ranking
    
    Example:
        >>> fe = AutoFeatureEngine(df)
        >>> fe.create_polynomial_features(degree=2)
        >>> fe.create_interactions()
        >>> new_df = fe.get_features()
    """
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """Initialize AutoFeatureEngine.
        
        Args:
            df: Pandas DataFrame
            target_column: Target column name for supervised feature selection
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        self.df = df.copy()
        self.target_column = target_column
        self.feature_log = []
        self.original_columns = list(df.columns)
        
    def create_polynomial_features(self, 
                                   columns: Optional[List[str]] = None,
                                   degree: int = 2,
                                   interaction_only: bool = False,
                                   include_bias: bool = False) -> pd.DataFrame:
        """Create polynomial features.
        
        Args:
            columns: Columns to create polynomial features from (None = all numeric)
            degree: Polynomial degree
            interaction_only: Only create interaction features
            include_bias: Include bias column
            
        Returns:
            DataFrame with polynomial features added
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column and self.target_column in columns:
                columns.remove(self.target_column)
        
        if not columns:
            warnings.warn("No numeric columns found for polynomial features")
            return self.df
        
        # Create polynomial features
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        
        poly_features = poly.fit_transform(self.df[columns])
        feature_names = poly.get_feature_names_out(columns)
        
        # Add new features (excluding original ones)
        n_original = len(columns)
        new_features = pd.DataFrame(
            poly_features[:, n_original:],
            columns=feature_names[n_original:],
            index=self.df.index
        )
        
        self.df = pd.concat([self.df, new_features], axis=1)
        
        self.feature_log.append({
            'operation': 'polynomial_features',
            'columns': columns,
            'degree': degree,
            'n_features_created': new_features.shape[1]
        })
        
        return self.df
    
    def create_interactions(self, 
                           columns: Optional[List[str]] = None,
                           max_interactions: int = 10) -> pd.DataFrame:
        """Create interaction features between numeric columns.
        
        Args:
            columns: Columns to create interactions from (None = all numeric)
            max_interactions: Maximum number of interaction features to create
            
        Returns:
            DataFrame with interaction features added
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column and self.target_column in columns:
                columns.remove(self.target_column)

        if len(columns) < 2:
            warnings.warn("Need at least 2 columns for interactions")
            return self.df

        interactions_created = 0
        new_features = {}

        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if interactions_created >= max_interactions:
                    break

                # Multiplication interaction
                new_col_name = f"{col1}_x_{col2}"
                new_features[new_col_name] = self.df[col1] * self.df[col2]
                interactions_created += 1

        if new_features:
            self._extracted_from_create_aggregation_features_36(
                new_features, 'interaction_features'
            )
        return self.df
    
    def create_ratio_features(self, 
                             columns: Optional[List[str]] = None,
                             max_ratios: int = 10) -> pd.DataFrame:
        """Create ratio features between numeric columns.
        
        Args:
            columns: Columns to create ratios from (None = all numeric)
            max_ratios: Maximum number of ratio features to create
            
        Returns:
            DataFrame with ratio features added
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column and self.target_column in columns:
                columns.remove(self.target_column)

        if len(columns) < 2:
            warnings.warn("Need at least 2 columns for ratios")
            return self.df

        ratios_created = 0
        new_features = {}

        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if ratios_created >= max_ratios:
                    break

                # Avoid division by zero
                if (self.df[col2] == 0).any():
                    continue

                new_col_name = f"{col1}_div_{col2}"
                new_features[new_col_name] = self.df[col1] / self.df[col2]
                ratios_created += 1

        if new_features:
            self._extracted_from_create_aggregation_features_36(
                new_features, 'ratio_features'
            )
        return self.df
    
    def create_binned_features(self,
                              columns: Optional[List[str]] = None,
                              n_bins: int = 5,
                              strategy: str = 'quantile',
                              encode: str = 'ordinal') -> pd.DataFrame:
        """Create binned/discretized features.
        
        Args:
            columns: Columns to bin (None = all numeric)
            n_bins: Number of bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            encode: Encoding method ('ordinal' or 'onehot')
            
        Returns:
            DataFrame with binned features added
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column and self.target_column in columns:
                columns.remove(self.target_column)
        
        if not columns:
            return self.df
        
        # Build discretizer kwargs
        discretizer_kwargs = {
            'n_bins': n_bins,
            'encode': encode,
            'strategy': strategy,
            'subsample': 200_000  # Prevent memory issues with large datasets
        }
        
        # Add quantile_method for quantile strategy to avoid deprecation warning
        if strategy == 'quantile':
            discretizer_kwargs['random_state'] = 42
        
        discretizer = KBinsDiscretizer(**discretizer_kwargs)
        
        binned_data = discretizer.fit_transform(self.df[columns])
        
        if encode == 'ordinal':
            binned_df = pd.DataFrame(
                binned_data,
                columns=[f"{col}_binned" for col in columns],
                index=self.df.index
            )
            self.df = pd.concat([self.df, binned_df], axis=1)
            n_features = len(columns)
        else:  # onehot
            # Get feature names for one-hot encoded bins
            n_features = binned_data.shape[1]
            feature_names = [f"bin_{i}" for i in range(n_features)]
            binned_df = pd.DataFrame(
                binned_data,
                columns=feature_names,
                index=self.df.index
            )
            self.df = pd.concat([self.df, binned_df], axis=1)
        
        self.feature_log.append({
            'operation': 'binned_features',
            'columns': columns,
            'n_bins': n_bins,
            'n_features_created': n_features
        })
        
        return self.df
    
    def create_aggregation_features(self, columns: Optional[List[str]] = None, operations: List[str] = None) -> pd.DataFrame:
        """Create aggregation features across columns.
        
        Args:
            columns: Columns to aggregate (None = all numeric)
            operations: Aggregation operations to apply
            
        Returns:
            DataFrame with aggregation features added
        """
        if operations is None:
            operations = ['sum', 'mean', 'std']
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column and self.target_column in columns:
                columns.remove(self.target_column)

        if len(columns) < 2:
            warnings.warn("Need at least 2 columns for aggregations")
            return self.df

        new_features = {}

        # Map operations to (feature_name, function) to simplify branching and avoid unnecessary f-strings
        ops_map = {
            'sum': ('agg_sum', lambda: self.df[columns].sum(axis=1)),
            'mean': ('agg_mean', lambda: self.df[columns].mean(axis=1)),
            'std': ('agg_std', lambda: self.df[columns].std(axis=1)),
            'min': ('agg_min', lambda: self.df[columns].min(axis=1)),
            'max': ('agg_max', lambda: self.df[columns].max(axis=1)),
            'median': ('agg_median', lambda: self.df[columns].median(axis=1)),
        }

        for op in operations:
            if op in ops_map:
                name, func = ops_map[op]
                new_features[name] = func()

        if new_features:
            self._extracted_from_create_aggregation_features_36(
                new_features, 'aggregation_features'
            )
        return self.df

    # TODO Rename this here and in `create_interactions`, `create_ratio_features` and `create_aggregation_features`
    def _extracted_from_create_aggregation_features_36(self, new_features, arg1):
        new_df = pd.DataFrame(new_features, index=self.df.index)
        self.df = pd.concat([self.df, new_df], axis=1)
        self.feature_log.append(
            {'operation': arg1, 'n_features_created': len(new_features)}
        )
    
    def create_datetime_features(self, columns: Optional[List[str]] = None, features: List[str] = None) -> pd.DataFrame:
        """Create features from datetime columns.
        
        Args:
            columns: DateTime columns to extract from (None = all datetime)
            features: Features to extract
            
        Returns:
            DataFrame with datetime features added
        """
        if features is None:
            features = ['year', 'month', 'day', 'dayofweek', 'hour']
        if columns is None:
            columns = self.df.select_dtypes(include=['datetime64']).columns.tolist()

        if not columns:
            # Try to convert object columns that might be dates
            for col in self.df.select_dtypes(include=['object']).columns:
                with contextlib.suppress(Exception):
                    self.df[col] = pd.to_datetime(self.df[col])
                    columns.append(col)
        if not columns:
            warnings.warn("No datetime columns found")
            return self.df

        n_features_created = 0

        for col in columns:
            if 'year' in features:
                self.df[f"{col}_year"] = self.df[col].dt.year
                n_features_created += 1
            if 'month' in features:
                self.df[f"{col}_month"] = self.df[col].dt.month
                n_features_created += 1
            if 'day' in features:
                self.df[f"{col}_day"] = self.df[col].dt.day
                n_features_created += 1
            if 'dayofweek' in features:
                self.df[f"{col}_dayofweek"] = self.df[col].dt.dayofweek
                n_features_created += 1
            if 'hour' in features:
                with contextlib.suppress(Exception):
                    self.df[f"{col}_hour"] = self.df[col].dt.hour
                    n_features_created += 1
            if 'quarter' in features:
                self.df[f"{col}_quarter"] = self.df[col].dt.quarter
                n_features_created += 1
            if 'is_weekend' in features:
                self.df[f"{col}_is_weekend"] = (self.df[col].dt.dayofweek >= 5).astype(int)
                n_features_created += 1

        if n_features_created > 0:
            self.feature_log.append({
                'operation': 'datetime_features',
                'columns': columns,
                'n_features_created': n_features_created
            })

        return self.df
    
    def select_features(self,
                       method: str = 'mutual_info',
                       k: int = 10,
                       task: str = 'classification') -> pd.DataFrame:
        """Select top k features using various methods.
        
        Args:
            method: Selection method ('mutual_info', 'f_test', 'variance')
            k: Number of features to select
            task: 'classification' or 'regression'
            
        Returns:
            DataFrame with selected features
        """
        if self.target_column is None:
            warnings.warn("Target column not specified, skipping feature selection")
            return self.df

        if self.target_column not in self.df.columns:
            warnings.warn(f"Target column '{self.target_column}' not found")
            return self.df

        # Get feature columns (exclude target)
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # Select only numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        if X_numeric.empty:
            warnings.warn("No numeric features for selection")
            return self.df
        # Apply selection method
        if method == 'f_test':
            selector = (
                SelectKBest(score_func=f_classif, k=min(k, len(numeric_cols)))
                if task == 'classification'
                else SelectKBest(
                    score_func=f_regression, k=min(k, len(numeric_cols))
                )
            )
            selector.fit(X_numeric, y)
            scores = selector.scores_
        elif method == 'mutual_info':
            scores = (
                mutual_info_classif(X_numeric, y)
                if task == 'classification'
                else mutual_info_regression(X_numeric, y)
            )
        else:
            # Variance-based
            scores = X_numeric.var().values

        # Get top k features
        top_k_idx = np.argsort(scores)[-k:]
        selected_features = numeric_cols[top_k_idx].tolist()

        # Keep selected features and target
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        columns_to_keep = selected_features + non_numeric_cols + [self.target_column]

        self.df = self.df[columns_to_keep]

        self.feature_log.append({
            'operation': 'feature_selection',
            'method': method,
            'n_features_before': len(numeric_cols),
            'n_features_after': len(selected_features),
            'selected_features': selected_features
        })

        return self.df
    
    def get_feature_importance(self, task: str = 'classification') -> pd.DataFrame:
        """Get feature importance scores.
        
        Args:
            task: 'classification' or 'regression'
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.target_column is None:
            raise ValueError("Target column required for feature importance")

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        if task == 'classification':
            scores = mutual_info_classif(X_numeric, y)
        else:
            scores = mutual_info_regression(X_numeric, y)

        return pd.DataFrame(
            {'feature': numeric_cols, 'importance': scores}
        ).sort_values('importance', ascending=False)
    
    def get_features(self) -> pd.DataFrame:
        """Get the DataFrame with all engineered features.
        
        Returns:
            DataFrame with engineered features
        """
        return self.df
    
    def get_feature_log(self) -> List[Dict[str, Any]]:
        """Get log of feature engineering operations.
        
        Returns:
            List of operation logs
        """
        return self.feature_log
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering.
        
        Returns:
            Summary dictionary
        """
        n_original = len(self.original_columns)
        n_current = len(self.df.columns)
        n_created = n_current - n_original

        operations_count = {}
        for log in self.feature_log:
            op = log['operation']
            operations_count[op] = operations_count.get(op, 0) + 1

        return {
            'original_features': n_original,
            'current_features': n_current,
            'features_created': n_created,
            'operations_performed': len(self.feature_log),
            'operations_breakdown': operations_count,
            'feature_log': self.feature_log,
        }
    
    def reset(self) -> pd.DataFrame:
        """Reset to original features.
        
        Returns:
            Original DataFrame
        """
        self.df = self.df[self.original_columns].copy()
        self.feature_log = []
        return self.df


def auto_feature_engineering(df: pd.DataFrame,
                             target_column: Optional[str] = None,
                             max_features: int = 50,
                             include_polynomials: bool = True,
                             include_interactions: bool = True,
                             include_ratios: bool = False,
                             include_aggregations: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience function for automated feature engineering.
    
    Args:
        df: Input DataFrame
        target_column: Target column for feature selection
        max_features: Maximum number of features to create
        include_polynomials: Create polynomial features
        include_interactions: Create interaction features
        include_ratios: Create ratio features
        include_aggregations: Create aggregation features
        
    Returns:
        Tuple of (engineered DataFrame, summary dict)
    """
    fe = AutoFeatureEngine(df, target_column)
    
    if include_interactions:
        fe.create_interactions(max_interactions=min(10, max_features // 3))
    
    if include_aggregations:
        fe.create_aggregation_features()
    
    if include_ratios:
        fe.create_ratio_features(max_ratios=min(10, max_features // 3))
    
    if include_polynomials:
        fe.create_polynomial_features(degree=2, interaction_only=True)
    
    # Select best features if we created too many
    if len(fe.df.columns) > max_features and target_column:
        fe.select_features(k=max_features, task='classification')
    
    return fe.get_features(), fe.get_summary()
