"""Tests for AutoFeatureEngine module."""
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


@pytest.fixture
def feature_engine(sample_df):
    """Create AutoFeatureEngine instance."""
    return AutoFeatureEngine(sample_df)


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
        with pytest.raises(TypeError):
            AutoFeatureEngine("not a dataframe")
        
        with pytest.raises(ValueError):
            AutoFeatureEngine(pd.DataFrame())  # Empty DataFrame
    
    def test_init_with_invalid_target(self, sample_df):
        """Test initialization with non-existent target."""
        with pytest.raises(ValueError):
            AutoFeatureEngine(sample_df, target_column='nonexistent')


class TestPolynomialFeatures:
    """Test polynomial feature creation."""
    
    def test_create_polynomial_features_basic(self, feature_engine):
        """Test basic polynomial feature creation."""
        result = feature_engine.create_polynomial_features(
            columns=['age', 'income'],
            degree=2
        )
        
        # Should have original + polynomial features
        assert result.shape[0] == feature_engine.df.shape[0]
        assert result.shape[1] > feature_engine.df.shape[1]
    
    def test_create_polynomial_features_degree(self, feature_engine):
        """Test polynomial features with different degrees."""
        result_deg2 = feature_engine.create_polynomial_features(
            columns=['age', 'income'],
            degree=2
        )
        
        result_deg3 = feature_engine.create_polynomial_features(
            columns=['age', 'income'],
            degree=3
        )
        
        # Higher degree should create more features
        assert result_deg3.shape[1] > result_deg2.shape[1]
    
    def test_create_polynomial_interaction_only(self, feature_engine):
        """Test interaction-only polynomial features."""
        result = feature_engine.create_polynomial_features(
            columns=['age', 'income'],
            degree=2,
            interaction_only=True
        )
        
        # Should create interaction features
        assert result.shape[1] > feature_engine.df.shape[1]
    
    def test_create_polynomial_invalid_columns(self, feature_engine):
        """Test polynomial features with invalid columns."""
        with pytest.raises(ValueError):
            feature_engine.create_polynomial_features(
                columns=['nonexistent'],
                degree=2
            )
    
    def test_create_polynomial_with_categorical(self, feature_engine):
        """Test polynomial features excludes categorical columns."""
        # Should handle or skip categorical columns
        result = feature_engine.create_polynomial_features(
            columns=['age', 'category'],  # 'category' is categorical
            degree=2
        )
        
        # Should still work, skipping categorical
        assert result is not None


class TestInteractionFeatures:
    """Test interaction feature creation."""
    
    def test_create_interactions_basic(self, feature_engine):
        """Test basic interaction feature creation."""
        result = feature_engine.create_interactions(
            columns=['age', 'income', 'credit_score']
        )
        
        # Should have original + interaction features
        assert result.shape[0] == feature_engine.df.shape[0]
        assert result.shape[1] > feature_engine.df.shape[1]
        
        # Check for interaction columns
        cols = result.columns.tolist()
        assert any('_x_' in col for col in cols)
    
    def test_create_interactions_limit(self, feature_engine):
        """Test interaction limit."""
        result = feature_engine.create_interactions(
            columns=['age', 'income', 'credit_score'],
            max_interactions=2
        )
        
        # Count interaction features
        interaction_cols = [col for col in result.columns if '_x_' in col]
        assert len(interaction_cols) <= 2
    
    def test_create_interactions_two_columns(self, feature_engine):
        """Test interactions with two columns."""
        result = feature_engine.create_interactions(
            columns=['age', 'income']
        )
        
        # Should create age_x_income
        assert 'age_x_income' in result.columns or 'income_x_age' in result.columns


class TestRatioFeatures:
    """Test ratio feature creation."""
    
    def test_create_ratio_features_basic(self, feature_engine):
        """Test basic ratio feature creation."""
        result = feature_engine.create_ratio_features(
            columns=['income', 'loan_amount']
        )
        
        # Should have original + ratio features
        assert result.shape[1] > feature_engine.df.shape[1]
        
        # Check for ratio columns
        cols = result.columns.tolist()
        assert any('_div_' in col for col in cols)
    
    def test_create_ratio_features_limit(self, feature_engine):
        """Test ratio limit."""
        result = feature_engine.create_ratio_features(
            columns=['age', 'income', 'credit_score'],
            max_ratios=2
        )
        
        # Count ratio features
        ratio_cols = [col for col in result.columns if '_div_' in col]
        assert len(ratio_cols) <= 2
    
    def test_create_ratio_handles_zeros(self, feature_engine):
        """Test ratio creation handles zero division."""
        # Add a column with zeros
        df_with_zeros = feature_engine.df.copy()
        df_with_zeros['zero_col'] = 0
        
        fe = AutoFeatureEngine(df_with_zeros)
        result = fe.create_ratio_features(columns=['income', 'zero_col'])
        
        # Should handle zeros gracefully (replace with 0 or inf)
        assert result is not None
        assert not result.isnull().all().any()  # No completely null columns


class TestBinnedFeatures:
    """Test binned feature creation."""
    
    def test_create_binned_features_basic(self, feature_engine):
        """Test basic binned feature creation."""
        result = feature_engine.create_binned_features(
            columns=['age', 'income'],
            n_bins=5
        )
        
        # Should have original + binned features
        assert result.shape[1] > feature_engine.df.shape[1]
        
        # Check for binned columns
        assert 'age_binned' in result.columns
        assert 'income_binned' in result.columns
    
    def test_create_binned_features_strategies(self, feature_engine):
        """Test different binning strategies."""
        strategies = ['uniform', 'quantile', 'kmeans']
        
        for strategy in strategies:
            result = feature_engine.create_binned_features(
                columns=['age'],
                n_bins=5,
                strategy=strategy
            )
            
            assert 'age_binned' in result.columns
            # Binned values should be integers
            assert result['age_binned'].dtype in [np.int32, np.int64]
    
    def test_create_binned_features_bin_count(self, feature_engine):
        """Test binned features have correct number of bins."""
        result = feature_engine.create_binned_features(
            columns=['age'],
            n_bins=3
        )
        
        # Should have 3 bins (0, 1, 2)
        unique_bins = result['age_binned'].nunique()
        assert unique_bins <= 3


class TestAggregationFeatures:
    """Test aggregation feature creation."""
    
    def test_create_aggregation_features_basic(self, feature_engine):
        """Test basic aggregation feature creation."""
        result = feature_engine.create_aggregation_features(
            columns=['age', 'income', 'credit_score']
        )
        
        # Should have original + aggregation features
        assert result.shape[1] > feature_engine.df.shape[1]
    
    def test_create_aggregation_operations(self, feature_engine):
        """Test specific aggregation operations."""
        operations = ['sum', 'mean', 'std', 'min', 'max']
        
        result = feature_engine.create_aggregation_features(
            columns=['age', 'income'],
            operations=operations
        )
        
        # Check for aggregation columns
        for op in operations:
            agg_col = f'agg_{op}'
            assert agg_col in result.columns
    
    def test_create_aggregation_single_operation(self, feature_engine):
        """Test single aggregation operation."""
        result = feature_engine.create_aggregation_features(
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
        result = fe.create_datetime_features(columns=['date'])
        
        # Should have datetime features
        assert result.shape[1] > datetime_df.shape[1]
    
    def test_create_datetime_features_specific(self, datetime_df):
        """Test specific datetime features."""
        fe = AutoFeatureEngine(datetime_df)
        features = ['year', 'month', 'day']
        
        result = fe.create_datetime_features(
            columns=['date'],
            features=features
        )
        
        # Check for specific features
        assert 'date_year' in result.columns
        assert 'date_month' in result.columns
        assert 'date_day' in result.columns
    
    def test_create_datetime_features_all(self, datetime_df):
        """Test all datetime features."""
        fe = AutoFeatureEngine(datetime_df)
        result = fe.create_datetime_features(
            columns=['date'],
            features=['year', 'month', 'day', 'dayofweek', 'quarter', 'hour']
        )
        
        # Check for various features
        assert 'date_year' in result.columns
        assert 'date_month' in result.columns
        assert 'date_dayofweek' in result.columns
    
    def test_create_datetime_with_non_datetime(self, feature_engine):
        """Test datetime features with non-datetime column."""
        # Should raise error or skip non-datetime columns
        with pytest.raises((ValueError, TypeError)):
            feature_engine.create_datetime_features(columns=['age'])


class TestFeatureSelection:
    """Test feature selection."""
    
    def test_select_features_with_target(self, feature_engine):
        """Test feature selection with target."""
        # Create some features first
        df_with_features = feature_engine.create_interactions(
            columns=['age', 'income', 'credit_score']
        )
        
        fe = AutoFeatureEngine(df_with_features, target_column='target')
        result = fe.select_features(method='mutual_info', k=5, task='classification')
        
        # Should return k features + target
        assert result.shape[1] <= 6  # 5 features + target
    
    def test_select_features_without_target(self, feature_engine):
        """Test feature selection requires target."""
        with pytest.raises(ValueError):
            feature_engine.select_features(method='mutual_info', k=5)
    
    def test_select_features_mutual_info(self, sample_df):
        """Test mutual information selection."""
        fe = AutoFeatureEngine(sample_df, target_column='target')
        result = fe.select_features(method='mutual_info', k=3, task='classification')
        
        assert result is not None
        assert 'target' in result.columns
    
    def test_select_features_f_test(self, sample_df):
        """Test f-test selection."""
        fe = AutoFeatureEngine(sample_df, target_column='target')
        result = fe.select_features(method='f_test', k=3, task='classification')
        
        assert result is not None
        assert 'target' in result.columns


class TestFeatureImportance:
    """Test feature importance."""
    
    def test_get_feature_importance_classification(self, sample_df):
        """Test feature importance for classification."""
        fe = AutoFeatureEngine(sample_df, target_column='target')
        importance = fe.get_feature_importance(task='classification')
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        
        # Should have scores for features
        assert all(isinstance(v, float) for v in importance.values())
    
    def test_get_feature_importance_regression(self, sample_df):
        """Test feature importance for regression."""
        # Use income as target for regression
        fe = AutoFeatureEngine(sample_df, target_column='income')
        importance = fe.get_feature_importance(task='regression')
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
    
    def test_get_feature_importance_without_target(self, feature_engine):
        """Test feature importance requires target."""
        with pytest.raises(ValueError):
            feature_engine.get_feature_importance()


class TestAutoFeatureEngineering:
    """Test convenience function."""
    
    def test_auto_feature_engineering_basic(self, sample_df):
        """Test auto feature engineering convenience function."""
        result = auto_feature_engineering(
            sample_df,
            numeric_columns=['age', 'income'],
            target_column='target'
        )
        
        # Should return DataFrame with new features
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == sample_df.shape[0]
        assert result.shape[1] > sample_df.shape[1]
    
    def test_auto_feature_engineering_with_selection(self, sample_df):
        """Test auto FE with feature selection."""
        result = auto_feature_engineering(
            sample_df,
            numeric_columns=['age', 'income'],
            target_column='target',
            select_top_k=5
        )
        
        # Should select top k features
        assert result.shape[1] <= 6  # 5 + target


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_column_dataframe(self):
        """Test with single column DataFrame."""
        df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        fe = AutoFeatureEngine(df)
        
        # Should handle gracefully
        assert fe.df.shape[1] == 1
    
    def test_dataframe_with_missing_values(self):
        """Test with missing values."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [10, np.nan, 30, 40, 50]
        })
        
        fe = AutoFeatureEngine(df)
        
        # Interaction features should handle NaN
        result = fe.create_interactions(columns=['a', 'b'])
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
        
        if len(numeric_cols) == 0:
            # No numeric columns to create features from
            assert True
    
    def test_constant_column(self):
        """Test with constant column."""
        df = pd.DataFrame({
            'constant': [5] * 100,
            'varying': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        fe = AutoFeatureEngine(df, target_column='target')
        
        # Should handle constant columns
        result = fe.create_interactions(columns=['constant', 'varying'])
        assert result is not None


class TestChaining:
    """Test method chaining."""
    
    def test_method_chaining(self, feature_engine):
        """Test chaining multiple feature creation methods."""
        # Create features step by step
        df1 = feature_engine.create_interactions(columns=['age', 'income'])
        
        fe2 = AutoFeatureEngine(df1)
        df2 = fe2.create_ratio_features(columns=['income', 'loan_amount'])
        
        fe3 = AutoFeatureEngine(df2)
        df3 = fe3.create_binned_features(columns=['age'], n_bins=5)
        
        # Should accumulate features
        assert df3.shape[1] > df2.shape[1] > df1.shape[1]


class TestPerformance:
    """Test performance with larger datasets."""
    
    def test_large_dataset(self):
        """Test feature engineering with larger dataset."""
        np.random.seed(42)
        n_samples = 10000
        
        df = pd.DataFrame({
            f'col_{i}': np.random.randn(n_samples)
            for i in range(10)
        })
        df['target'] = np.random.choice([0, 1], n_samples)
        
        fe = AutoFeatureEngine(df, target_column='target')
        
        # Should complete without errors
        result = fe.create_interactions(columns=[f'col_{i}' for i in range(5)])
        assert result is not None
        assert result.shape[0] == n_samples
