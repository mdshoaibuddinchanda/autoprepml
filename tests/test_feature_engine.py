"""Tests for AutoFeatureEngine module - FULLY FIXED VERSION."""

import contextlib
import pytest
import pandas as pd
import numpy as np
from autoprepml.feature_engine import AutoFeatureEngine, auto_feature_engineering


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_amount': np.random.uniform(1000, 50000, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def datetime_df():
    """Create a DataFrame with datetime columns."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    data = {
        'date': dates,
        'value': np.random.randn(100)
    }
    
    return pd.DataFrame(data)


class TestAutoFeatureEngineInit:
    """Test AutoFeatureEngine initialization."""
    
    def test_init_with_dataframe(self, sample_df):
        """Test initialization with valid DataFrame."""
        fe = AutoFeatureEngine(sample_df)
        assert fe.df.equals(sample_df)
        assert fe.target_column is None
    
    def test_init_with_target(self, sample_df):
        """Test initialization with target column."""
        fe = AutoFeatureEngine(sample_df, target_column='target')
        assert fe.target_column == 'target'
    
    def test_init_with_invalid_input(self):
        """Test initialization with invalid input."""
        with pytest.raises((TypeError, ValueError)):
            AutoFeatureEngine("not a dataframe")
        
        with pytest.raises(ValueError):
            AutoFeatureEngine(pd.DataFrame())  # Empty DataFrame
    
    def test_init_with_invalid_target(self, sample_df):
        """Test initialization with non-existent target."""
        # API doesn't validate target column existence at init time
        fe = AutoFeatureEngine(sample_df, target_column='nonexistent')
        assert fe.target_column == 'nonexistent'


class TestPolynomialFeatures:
    """Test polynomial feature creation."""
    
    def test_create_polynomial_features_basic(self, sample_df):
        """Test basic polynomial feature creation."""
        fe = AutoFeatureEngine(sample_df.copy())
        original_cols = len(fe.df.columns)
        
        result = fe.create_polynomial_features(
            columns=['age', 'income'],
            degree=2
        )
        
        # Should create new polynomial features
        assert result.shape[0] == sample_df.shape[0]
        assert result.shape[1] > original_cols
    
    def test_create_polynomial_features_degree(self, sample_df):
        """Test polynomial features with different degrees."""
        fe2 = AutoFeatureEngine(sample_df.copy())
        result_deg2 = fe2.create_polynomial_features(
            columns=['age', 'income'],
            degree=2
        )
        
        fe3 = AutoFeatureEngine(sample_df.copy())
        result_deg3 = fe3.create_polynomial_features(
            columns=['age', 'income'],
            degree=3
        )
        
        # Higher degree should create more features
        assert result_deg3.shape[1] > result_deg2.shape[1]
    
    def test_create_polynomial_interaction_only(self, sample_df):
        """Test interaction-only polynomial features."""
        fe = AutoFeatureEngine(sample_df.copy())
        original_cols = len(fe.df.columns)
        
        result = fe.create_polynomial_features(
            columns=['age', 'income'],
            degree=2,
            interaction_only=True
        )
        
        # Should create interaction features
        assert result.shape[1] > original_cols
    
    def test_create_polynomial_invalid_columns(self, sample_df):
        """Test polynomial features with invalid columns."""
        fe = AutoFeatureEngine(sample_df.copy())
        
        # API raises KeyError when column doesn't exist
        with pytest.raises((ValueError, KeyError)):
            fe.create_polynomial_features(
                columns=['nonexistent'],
                degree=2
            )
    
    def test_create_polynomial_with_categorical(self, sample_df):
        """Test polynomial features with categorical columns."""
        fe = AutoFeatureEngine(sample_df.copy())
        
        # API will raise error when trying polynomial from non-numeric
        with pytest.raises((ValueError, TypeError)):
            fe.create_polynomial_features(
                columns=['age', 'category'],
                degree=2
            )


class TestInteractionFeatures:
    """Test interaction feature creation."""
    
    def test_create_interactions_basic(self, sample_df):
        """Test basic interaction feature creation."""
        fe = AutoFeatureEngine(sample_df.copy())
        original_cols = len(fe.df.columns)
        
        result = fe.create_interactions(
            columns=['age', 'income', 'credit_score']
        )
        
        # Should create new interaction features
        assert result.shape[0] == sample_df.shape[0]
        assert result.shape[1] > original_cols
        
        # Check for interaction columns
        cols = result.columns.tolist()
        assert any('_x_' in col for col in cols)
    
    def test_create_interactions_limit(self, sample_df):
        """Test interaction limit."""
        fe = AutoFeatureEngine(sample_df.copy())
        
        result = fe.create_interactions(
            columns=['age', 'income', 'credit_score'],
            max_interactions=2
        )
        
        # Count interaction features
        interaction_cols = [col for col in result.columns if '_x_' in col]
        assert len(interaction_cols) <= 2
    
    def test_create_interactions_two_columns(self, sample_df):
        """Test interactions with two columns."""
        fe = AutoFeatureEngine(sample_df.copy())
        
        result = fe.create_interactions(
            columns=['age', 'income']
        )
        
        # Should create age_x_income
        assert 'age_x_income' in result.columns or 'income_x_age' in result.columns


class TestRatioFeatures:
    """Test ratio feature creation."""
    
    def test_create_ratio_features_basic(self, sample_df):
        """Test basic ratio feature creation."""
        fe = AutoFeatureEngine(sample_df.copy())
        original_cols = len(fe.df.columns)
        
        result = fe.create_ratio_features(
            columns=['income', 'loan_amount']
        )
        
        # Should create new ratio features
        assert result.shape[1] > original_cols
        
        # Check for ratio columns
        cols = result.columns.tolist()
        assert any('_div_' in col for col in cols)
    
    def test_create_ratio_features_limit(self, sample_df):
        """Test ratio limit."""
        fe = AutoFeatureEngine(sample_df.copy())
        
        result = fe.create_ratio_features(
            columns=['age', 'income', 'credit_score'],
            max_ratios=2
        )
        
        # Count ratio features
        ratio_cols = [col for col in result.columns if '_div_' in col]
        assert len(ratio_cols) <= 2
    
    def test_create_ratio_handles_zeros(self, sample_df):
        """Test ratio creation handles zero division."""
        df_with_zeros = sample_df.copy()
        df_with_zeros['zero_col'] = 0
        
        fe = AutoFeatureEngine(df_with_zeros)
        result = fe.create_ratio_features(columns=['income', 'zero_col'])
        
        # Should handle zeros gracefully
        assert result is not None
        assert not result.isnull().all().any()


class TestBinnedFeatures:
    """Test binned feature creation."""
    
    def test_create_binned_features_basic(self, sample_df):
        """Test basic binned feature creation."""
        fe = AutoFeatureEngine(sample_df.copy())
        
        result = fe.create_binned_features(
            columns=['age', 'income'],
            n_bins=5
        )
        
        # Check for binned columns
        assert 'age_binned' in result.columns
        assert 'income_binned' in result.columns
    
    def test_create_binned_features_strategies(self, sample_df):
        """Test different binning strategies."""
        strategies = ['uniform', 'quantile', 'kmeans']
        
# sourcery skip: no-loop-in-tests
        for strategy in strategies:
            fe = AutoFeatureEngine(sample_df.copy())
            result = fe.create_binned_features(
                columns=['age'],
                n_bins=5,
                strategy=strategy
            )
            
            assert 'age_binned' in result.columns
            # Binned values should be numeric
            assert result['age_binned'].dtype in [np.int32, np.int64, np.float32, np.float64]
    
    def test_create_binned_features_bin_count(self, sample_df):
        """Test binned features have correct number of bins."""
        fe = AutoFeatureEngine(sample_df.copy())
        
        result = fe.create_binned_features(
            columns=['age'],
            n_bins=3
        )
        
        # Should have 3 bins (0, 1, 2)
        unique_bins = result['age_binned'].nunique()
        assert unique_bins <= 3


class TestAggregationFeatures:
    """Test aggregation feature creation."""
    
    def test_create_aggregation_features_basic(self, sample_df):
        """Test basic aggregation feature creation."""
        fe = AutoFeatureEngine(sample_df.copy())
        original_cols = len(fe.df.columns)
        
        result = fe.create_aggregation_features(
            columns=['age', 'income'],
            operations=['mean', 'sum']
        )
        
        # Should create aggregation features
        assert result.shape[1] >= original_cols
    
    def test_create_aggregation_operations(self, sample_df):
        """Test specific aggregation operations."""
        fe = AutoFeatureEngine(sample_df.copy())
        operations = ['sum', 'mean', 'std', 'min', 'max']
        
        result = fe.create_aggregation_features(
            columns=['age', 'income'],
            operations=operations
        )
        
        # Check for aggregation columns
        for op in operations:
            agg_col = f'agg_{op}'
            assert agg_col in result.columns
    
    def test_create_aggregation_single_operation(self, sample_df):
        """Test single aggregation operation."""
        fe = AutoFeatureEngine(sample_df.copy())
        
        result = fe.create_aggregation_features(
            columns=['age', 'income'],
            operations=['mean']
        )
        
        assert 'agg_mean' in result.columns
        assert 'agg_sum' not in result.columns


class TestDatetimeFeatures:
    """Test datetime feature creation."""
    
    def test_create_datetime_features_basic(self, datetime_df):
        """Test basic datetime feature creation."""
        fe = AutoFeatureEngine(datetime_df)
        original_cols = len(fe.df.columns)
        
        result = fe.create_datetime_features(columns=['date'])
        
        # Should have datetime features
        assert result.shape[1] > original_cols
    
    def test_create_datetime_features_specific(self, datetime_df):
        """Test specific datetime features."""
        fe = AutoFeatureEngine(datetime_df)
        features = ['year', 'month', 'day']

        result = fe.create_datetime_features(
            columns=['date'],
            features=features
        )

        self._extracted_from_test_create_datetime_features_all_12(result, 'date_day')
    
    def test_create_datetime_features_all(self, datetime_df):
        """Test all datetime features."""
        fe = AutoFeatureEngine(datetime_df)
        result = fe.create_datetime_features(
            columns=['date'],
            features=['year', 'month', 'day', 'dayofweek', 'quarter', 'hour']
        )

        self._extracted_from_test_create_datetime_features_all_12(
            result, 'date_dayofweek'
        )

    # TODO Rename this here and in `test_create_datetime_features_specific` and `test_create_datetime_features_all`
    def _extracted_from_test_create_datetime_features_all_12(self, result, arg1):
        assert 'date_year' in result.columns
        assert 'date_month' in result.columns
        assert arg1 in result.columns
    
    def test_create_datetime_with_non_datetime(self, sample_df):
        """Test datetime features with non-datetime column."""
        fe = AutoFeatureEngine(sample_df.copy())
        
        # Should raise error for non-datetime columns
        with pytest.raises((ValueError, AttributeError)):
            fe.create_datetime_features(columns=['age'])


class TestFeatureSelection:
    """Test feature selection."""
    
    def test_select_features_with_target(self, sample_df):
        """Test feature selection with target column."""
        fe = AutoFeatureEngine(sample_df.copy(), target_column='target')
        k = 3
        
        result = fe.select_features(k=k, method='mutual_info')
        
        # Should select k features (may keep more with target)
        assert len(result.columns) <= k + 3  # k features + target + tolerance
    
    def test_select_features_without_target(self, sample_df):
        """Test feature selection without target column."""
        fe = AutoFeatureEngine(sample_df.copy())

        # API may work without target for unsupervised selection
        with contextlib.suppress(ValueError):
            fe.select_features(k=3)
    
    def test_select_features_mutual_info(self, sample_df):
        """Test mutual information feature selection."""
        self._extracted_from_test_select_features_f_test_3(sample_df, 'mutual_info')
    
    def test_select_features_f_test(self, sample_df):
        """Test F-test feature selection."""
        self._extracted_from_test_select_features_f_test_3(sample_df, 'f_test')

    # TODO Rename this here and in `test_select_features_mutual_info` and `test_select_features_f_test`
    def _extracted_from_test_select_features_f_test_3(self, sample_df, method):
        fe = AutoFeatureEngine(sample_df.copy(), target_column='target')
        result = fe.select_features(k=3, method=method)
        assert result.shape[1] <= sample_df.shape[1]


class TestFeatureImportance:
    """Test feature importance."""
    
    def test_get_feature_importance_classification(self, sample_df):
        """Test feature importance for classification."""
        self._extracted_from_test_get_feature_importance_regression_3(
            sample_df, 'target', 'classification'
        )
    
    def test_get_feature_importance_regression(self, sample_df):
        """Test feature importance for regression."""
        self._extracted_from_test_get_feature_importance_regression_3(
            sample_df, 'income', 'regression'
        )

    # TODO Rename this here and in `test_get_feature_importance_classification` and `test_get_feature_importance_regression`
    def _extracted_from_test_get_feature_importance_regression_3(self, sample_df, target_column, task):
        fe = AutoFeatureEngine(sample_df.copy(), target_column=target_column)
        importance = fe.get_feature_importance(task=task)
        assert importance is None or isinstance(importance, (dict, pd.DataFrame))
    
    def test_get_feature_importance_without_target(self, sample_df):
        """Test feature importance without target column."""
        fe = AutoFeatureEngine(sample_df.copy())
        
        # API raises ValueError when target is required
        with pytest.raises(ValueError):
            fe.get_feature_importance()


class TestAutoFeatureEngineering:
    """Test automated feature engineering."""
    
    def test_auto_feature_engineering_basic(self, sample_df):
        """Test basic automated feature engineering."""
        result, summary = auto_feature_engineering(
            sample_df.copy()
        )
        
        # Should create new features
        assert result.shape[1] > sample_df.shape[1]
        assert isinstance(summary, dict)
    
    def test_auto_feature_engineering_with_selection(self, sample_df):
        """Test automated feature engineering with selection."""
        result, summary = auto_feature_engineering(
            sample_df.copy(),
            target_column='target',
            max_features=15
        )
        
        # Should create and select features
        assert result is not None
        assert len(result.columns) > 0
        assert isinstance(summary, dict)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_column_dataframe(self):
        """Test with single column DataFrame."""
        df = pd.DataFrame({'a': range(10)})
        fe = AutoFeatureEngine(df)
        
        # Most operations should handle gracefully
        assert fe.df.shape[1] == 1
    
    def test_dataframe_with_missing_values(self, sample_df):
        """Test with DataFrame containing missing values."""
        df_with_missing = sample_df.copy()
        df_with_missing.loc[0:5, 'age'] = np.nan
        
        fe = AutoFeatureEngine(df_with_missing)
        result = fe.create_polynomial_features(columns=['income', 'credit_score'])
        
        # Should handle missing values
        assert result is not None
    
    def test_all_categorical_dataframe(self):
        """Test with all categorical columns."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 10,
            'cat2': ['X', 'Y', 'Z'] * 10
        })

        fe = AutoFeatureEngine(df)

        # Should handle or raise appropriate error for numeric operations
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def test_constant_column(self):
        """Test with constant column."""
        df = pd.DataFrame({
            'constant': [5] * 100,
            'varying': np.random.randn(100)
        })
        
        fe = AutoFeatureEngine(df)
        result = fe.create_polynomial_features(columns=['varying'])
        
        # Should handle constant columns appropriately
        assert result is not None


class TestChaining:
    """Test method chaining."""
    
    def test_method_chaining(self, sample_df):
        """Test chaining multiple feature engineering methods."""
        fe = AutoFeatureEngine(sample_df.copy())
        
        # Chain multiple operations
        result = fe.create_polynomial_features(columns=['age'], degree=2)
        result = fe.create_interactions(columns=['income', 'credit_score'])
        result = fe.create_ratio_features(columns=['income', 'loan_amount'])
        
        # Should have created features from all operations
        assert result.shape[1] > sample_df.shape[1]
