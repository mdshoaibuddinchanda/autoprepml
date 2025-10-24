"""Tests for advanced imputation and SMOTE functionality"""
import pytest
import pandas as pd
import numpy as np
from autoprepml.cleaning import impute_knn, impute_iterative, balance_classes_smote


class TestAdvancedImputation:
    """Test advanced imputation methods"""
    
    def test_knn_imputation_basic(self):
        """Test basic KNN imputation"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [5, np.nan, 7, 8, 9],
            'C': [10, 11, 12, 13, 14]
        })

        result = impute_knn(df, n_neighbors=2)

        self._extracted_from_test_iterative_imputation_basic_12(result)
        
    def test_knn_imputation_no_missing(self):
        """Test KNN imputation with no missing values"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 6, 7, 8, 9]
        })
        
        result = impute_knn(df, n_neighbors=2)
        
        # Should return unchanged DataFrame
        pd.testing.assert_frame_equal(result, df)
        
    def test_knn_imputation_exclude_cols(self):
        """Test KNN imputation with excluded columns"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [5, np.nan, 7, 8, 9],
            'C': [10, 11, 12, 13, 14]
        })
        
        result = impute_knn(df, n_neighbors=2, exclude_cols=['B'])
        
        # Column A should be imputed
        assert not result['A'].isnull().any()
        # Column B should still have missing value
        assert result['B'].isnull().sum() == 1
        
    def test_knn_imputation_non_numeric(self):
        """Test KNN imputation ignores non-numeric columns"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = impute_knn(df, n_neighbors=2)
        
        # Numeric column should be imputed
        assert not result['A'].isnull().any()
        # Non-numeric column should be unchanged
        assert (result['B'] == df['B']).all()
        
    def test_iterative_imputation_basic(self):
        """Test basic iterative imputation"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [5, np.nan, 7, 8, 9],
            'C': [10, 11, 12, 13, 14]
        })

        result = impute_iterative(df, max_iter=10, random_state=42)

        self._extracted_from_test_iterative_imputation_basic_12(result)

    # TODO Rename this here and in `test_knn_imputation_basic` and `test_iterative_imputation_basic`
    def _extracted_from_test_iterative_imputation_basic_12(self, result):
        assert result.isnull().sum().sum() == 0
        assert result.loc[0, 'A'] == 1
        assert result.loc[0, 'B'] == 5
        
    def test_iterative_imputation_reproducible(self):
        """Test iterative imputation is reproducible"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [5, np.nan, 7, 8, 9]
        })
        
        result1 = impute_iterative(df, max_iter=10, random_state=42)
        result2 = impute_iterative(df, max_iter=10, random_state=42)
        
        # Results should be identical with same random_state
        pd.testing.assert_frame_equal(result1, result2)
        
    def test_iterative_imputation_no_missing(self):
        """Test iterative imputation with no missing values"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 6, 7, 8, 9]
        })
        
        result = impute_iterative(df, max_iter=10, random_state=42)
        
        # Should return unchanged DataFrame
        pd.testing.assert_frame_equal(result, df)
        
    def test_iterative_imputation_exclude_cols(self):
        """Test iterative imputation with excluded columns"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [5, np.nan, 7, 8, 9],
            'C': [10, 11, 12, 13, 14]
        })
        
        result = impute_iterative(df, max_iter=10, random_state=42, exclude_cols=['B'])
        
        # Column A should be imputed
        assert not result['A'].isnull().any()
        # Column B should still have missing value
        assert result['B'].isnull().sum() == 1


class TestSMOTE:
    """Test SMOTE class balancing"""
    
    def test_smote_basic(self):
        """Test basic SMOTE balancing"""
        # Create imbalanced dataset: 90 class 0, 10 class 1
        df = pd.DataFrame({
            'A': list(range(100)),
            'B': list(range(100, 200)),
            'label': [0] * 90 + [1] * 10
        })
        
        try:
            result = balance_classes_smote(df, target_col='label', random_state=42)
            
            # Should have balanced classes
            value_counts = result['label'].value_counts()
            assert value_counts[0] == value_counts[1]
            # Should have more samples than original
            assert len(result) > len(df)
            # Should preserve column names
            assert set(result.columns) == set(df.columns)
            
        except ImportError:
            pytest.skip("imbalanced-learn not installed")
            
    def test_smote_sampling_strategy_minority(self):
        """Test SMOTE with minority sampling strategy"""
        df = pd.DataFrame({
            'A': list(range(100)),
            'B': list(range(100, 200)),
            'label': [0] * 90 + [1] * 10
        })
        
        try:
            result = balance_classes_smote(
                df, target_col='label', 
                sampling_strategy='minority',
                random_state=42
            )
            
            # Should have balanced classes
            value_counts = result['label'].value_counts()
            assert value_counts[0] == value_counts[1]
            
        except ImportError:
            pytest.skip("imbalanced-learn not installed")
            
    def test_smote_missing_target_column(self):
        """Test SMOTE with missing target column"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 6, 7, 8, 9]
        })
        
        try:
            with pytest.raises(ValueError, match="Target column 'label' not found"):
                balance_classes_smote(df, target_col='label')
        except ImportError:
            pytest.skip("imbalanced-learn not installed")
            
    def test_smote_non_numeric_features(self):
        """Test SMOTE with non-numeric features raises error"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5] * 20,
            'B': ['a', 'b', 'c', 'd', 'e'] * 20,
            'label': [0] * 90 + [1] * 10
        })
        
        try:
            with pytest.raises(ValueError, match="SMOTE requires all features to be numeric"):
                balance_classes_smote(df, target_col='label')
        except ImportError:
            pytest.skip("imbalanced-learn not installed")
            
    def test_smote_k_neighbors(self):
        """Test SMOTE with custom k_neighbors"""
        df = pd.DataFrame({
            'A': list(range(100)),
            'B': list(range(100, 200)),
            'label': [0] * 90 + [1] * 10
        })
        
        try:
            result = balance_classes_smote(
                df, target_col='label',
                k_neighbors=3,
                random_state=42
            )
            
            # Should have balanced classes
            value_counts = result['label'].value_counts()
            assert value_counts[0] == value_counts[1]
            
        except ImportError:
            pytest.skip("imbalanced-learn not installed")
            
    def test_smote_too_few_samples(self):
        """Test SMOTE with too few samples for k_neighbors"""
        # Only 3 minority samples, but k_neighbors=5 (default)
        df = pd.DataFrame({
            'A': [1, 2, 3, 10, 11, 12, 13, 14, 15, 16],
            'B': [5, 6, 7, 50, 51, 52, 53, 54, 55, 56],
            'label': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        })
        
        try:
            with pytest.raises(ValueError, match="SMOTE balancing failed"):
                balance_classes_smote(df, target_col='label', k_neighbors=5)
        except ImportError:
            pytest.skip("imbalanced-learn not installed")
            
    def test_smote_multiclass(self):
        """Test SMOTE with multiclass target"""
        df = pd.DataFrame({
            'A': list(range(100)),
            'B': list(range(100, 200)),
            'label': [0] * 70 + [1] * 20 + [2] * 10
        })
        
        try:
            result = balance_classes_smote(df, target_col='label', random_state=42)
            
            # All classes should have same count
            value_counts = result['label'].value_counts()
            assert value_counts[0] == value_counts[1] == value_counts[2]
            
        except ImportError:
            pytest.skip("imbalanced-learn not installed")
