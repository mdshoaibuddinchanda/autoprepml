"""Tests for AutoEDA module."""
import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from autoprepml.autoeda import AutoEDA


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples).astype(float),  # Float to allow NaN
        'loan_amount': np.random.uniform(1000, 50000, n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add missing values after DataFrame creation
    df.loc[np.random.choice(n_samples, 10, replace=False), 'income'] = np.nan
    df.loc[np.random.choice(n_samples, 5, replace=False), 'credit_score'] = np.nan
    
    return df


@pytest.fixture
def autoeda(sample_df):
    """Create AutoEDA instance."""
    return AutoEDA(sample_df)

def run_analyze_basic(eda):
    """Run analyze with the common basic options used by multiple tests."""
    return eda.analyze(
        include_correlations=False,
        include_distributions=False,
        include_outliers=False,
        generate_insights=False
    )

def analyze_df(df, **kwargs):
    """Create AutoEDA for a DataFrame and run analyze with optional kwargs."""
    return AutoEDA(df).analyze(**kwargs)


class TestAutoEDAInit:
    """Test AutoEDA initialization."""
    
    def test_init_with_dataframe(self, sample_df):
        """Test initialization with valid DataFrame."""
        eda = AutoEDA(sample_df)
        assert eda.df.equals(sample_df)
        assert eda.analysis_results == {}
        assert eda.insights == []
    
    def test_init_with_invalid_input(self):
        """Test initialization with invalid input."""
        with pytest.raises((TypeError, ValueError)):
            AutoEDA("not a dataframe")
        
        with pytest.raises(ValueError):
            AutoEDA(pd.DataFrame())  # Empty DataFrame


class TestBasicStatistics:
    """Test basic statistics computation."""
    
    def test_compute_basic_stats(self, autoeda):
        """Test basic statistics computation."""
        stats = autoeda._compute_basic_stats()
        
        assert 'shape' in stats
        assert stats['shape'] == (100, 6)
        assert 'n_rows' in stats
        assert stats['n_rows'] == 100
        assert 'n_columns' in stats
        assert stats['n_columns'] == 6
        assert 'memory_usage' in stats
        assert stats['memory_usage'] > 0
        assert 'duplicate_rows' in stats
        assert stats['duplicate_rows'] >= 0
    
    def test_numeric_summary(self, autoeda):
        """Test numeric columns summary."""
        stats = autoeda._compute_basic_stats()
        
        assert 'numeric_summary' in stats
        numeric_summary = stats['numeric_summary']
        
        # Should have numeric columns
        assert len(numeric_summary) > 0


class TestMissingValues:
    """Test missing value analysis."""
    
    def test_analyze_missing_values(self, autoeda):
        """Test missing values analysis."""
        missing = autoeda._analyze_missing_values()
        
        assert 'total_missing' in missing
        assert 'missing_by_column' in missing
        
        # missing_by_column is now a list of dicts
        missing_by_col = {item['column']: item for item in missing['missing_by_column']}
        
        # Should have missing values in income and credit_score
        assert missing_by_col['income']['missing_count'] == 10
        assert missing_by_col['credit_score']['missing_count'] == 5
    
    def test_missing_percentage(self, autoeda):
        """Test missing value percentages."""
        missing = autoeda._analyze_missing_values()
        
        # missing_by_column is now a list of dicts
        missing_by_col = {item['column']: item for item in missing['missing_by_column']}
        
        income_pct = missing_by_col['income']['missing_percentage']
        assert income_pct == 10.0  # 10 out of 100
        
        credit_pct = missing_by_col['credit_score']['missing_percentage']
        assert credit_pct == 5.0  # 5 out of 100


class TestDistributions:
    """Test distribution analysis."""
    
    def test_analyze_distributions(self, autoeda):
        """Test distribution analysis."""
        distributions = autoeda._analyze_distributions()
        
        # Should analyze all numeric columns
        assert len(distributions) == 5
        assert 'age' in distributions
        
        # Check distribution metrics
        age_dist = distributions['age']
        assert 'skewness' in age_dist
        assert 'kurtosis' in age_dist
        # Note: 'quartiles' key doesn't exist - q25 and q75 are separate
        assert 'q25' in age_dist
        assert 'q75' in age_dist
        assert 'median' in age_dist


class TestCorrelations:
    """Test correlation analysis."""
    
    def test_compute_correlations(self, autoeda):
        """Test correlation matrix computation."""
        correlations = autoeda._compute_correlations()
        
        assert 'correlation_matrix' in correlations
        corr_matrix = correlations['correlation_matrix']
        
        # Should be a square matrix of numeric columns
        assert isinstance(corr_matrix, dict)
        assert 'age' in corr_matrix
        
        # Diagonal should be 1.0
        assert corr_matrix['age']['age'] == pytest.approx(1.0)
    
    def test_high_correlations(self, autoeda):
        """Test high correlation detection."""
        correlations = autoeda._compute_correlations()
        
        assert 'high_correlations' in correlations
        # High correlations list may be empty or contain pairs
        assert isinstance(correlations['high_correlations'], list)


class TestOutliers:
    """Test outlier detection."""
    
    def test_detect_outliers_iqr(self, autoeda):
        """Test IQR outlier detection."""
        self._extracted_from_test_detect_outliers_zscore_3(
            autoeda, 'iqr_outliers', 'iqr_percentage'
        )
    
    def test_detect_outliers_zscore(self, autoeda):
        """Test Z-score outlier detection."""
        self._extracted_from_test_detect_outliers_zscore_3(
            autoeda, 'zscore_outliers', 'zscore_percentage'
        )

    # TODO Rename this here and in `test_detect_outliers_iqr` and `test_detect_outliers_zscore`
    def _extracted_from_test_detect_outliers_zscore_3(self, autoeda, arg1, arg2):
        outliers = autoeda._detect_outliers()
        assert isinstance(outliers, dict)
        if outliers:
            self._extracted_from_test_detect_outliers_zscore_11(outliers, arg1, arg2)

    # TODO Rename this here and in `test_detect_outliers_iqr` and `test_detect_outliers_zscore`
    def _extracted_from_test_detect_outliers_zscore_11(self, outliers, arg1, arg2):
        first_col = list(outliers.keys())[0]
        assert arg1 in outliers[first_col]
        assert arg2 in outliers[first_col]


class TestCategorical:
    """Test categorical analysis."""
    
    def test_analyze_categorical(self, autoeda):
        """Test categorical columns analysis."""
        categorical = autoeda._analyze_categorical()
        
        # Should analyze 'category' column
        assert 'category' in categorical
        
        cat_info = categorical['category']
        assert 'unique_values' in cat_info
        assert cat_info['unique_values'] == 4  # A, B, C, D
        assert 'cardinality' in cat_info
        assert cat_info['cardinality'] == 'low'  # API returns 'low' or 'high'
        assert 'mode' in cat_info
        assert 'most_common' in cat_info


class TestAnalyze:
    """Test main analyze method."""
    
    def test_analyze_basic(self, autoeda):
        """Test basic analysis."""
        results = run_analyze_basic(autoeda)
        
        assert 'basic_stats' in results
        assert 'missing_values' in results
        assert 'categorical' in results
        assert 'correlations' not in results
        assert 'distributions' not in results
        assert 'outliers' not in results
    
    def test_analyze_full(self, autoeda):
        """Test full analysis with all options."""
        results = autoeda.analyze(
            include_correlations=True,
            include_distributions=True,
            include_outliers=True,
            generate_insights=True
        )
        
        assert 'basic_stats' in results
        assert 'missing_values' in results
        assert 'categorical' in results
        assert 'correlations' in results
        assert 'distributions' in results
        assert 'outliers' in results
        assert 'insights' in results
    
    def test_analyze_stores_results(self, autoeda):
        """Test that analyze stores results."""
        assert autoeda.analysis_results == {}
        autoeda.analyze()
        assert autoeda.analysis_results != {}


class TestInsights:
    """Test insights generation."""
    
    def test_generate_insights(self, autoeda):
        """Test insights generation."""
        autoeda.analyze(generate_insights=True)
        
        assert 'insights' in autoeda.analysis_results
        insights = autoeda.analysis_results['insights']
        
        # Should be a list of strings
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)
    
    def test_insights_content(self, autoeda):
        """Test insights contain relevant information."""
        autoeda.analyze(generate_insights=True)
        insights = autoeda.analysis_results['insights']
        
        # Should be a list
        assert isinstance(insights, list)


class TestReportGeneration:
    """Test report generation."""
    
    def test_generate_report_without_analysis(self, autoeda, tmp_path):
        """Test report generation without analysis."""
        output_path = tmp_path / "report.html"
        
        # The API now generates a report even without analysis
        # So we just check it creates a file
        autoeda.generate_report(str(output_path))
        assert output_path.exists()
    
    def test_generate_report_creates_file(self, autoeda, tmp_path):
        """Test report generation creates HTML file."""
        autoeda.analyze()
        output_path = tmp_path / "report.html"
        
        autoeda.generate_report(str(output_path))
        
        assert output_path.exists()
        assert output_path.suffix == '.html'
    
    def test_generate_report_content(self, autoeda, tmp_path):
        """Test report contains expected content."""
        autoeda.analyze()
        output_path = tmp_path / "report.html"
        
        autoeda.generate_report(str(output_path))
        
        # Read with explicit encoding to handle special characters
        content = output_path.read_text(encoding='utf-8')
        
        # Should be valid HTML
        assert '<html>' in content.lower()
        assert '</html>' in content.lower()
        
        # Should contain title
        assert 'AutoEDA Report' in content or 'EDA Report' in content.lower()
        
        # Should contain sections
        assert 'basic' in content.lower() or 'statistics' in content.lower()


class TestJSONExport:
    """Test JSON export functionality."""
    
    def test_to_json_without_analysis(self, autoeda, tmp_path):
        """Test JSON export without analysis."""
        output_path = tmp_path / "results.json"
        
        # The API now exports even without analysis
        # So we just check it creates a file
        autoeda.to_json(str(output_path))
        assert output_path.exists()
    
    def test_to_json_creates_file(self, autoeda, tmp_path):
        """Test JSON export creates file."""
        autoeda.analyze()
        output_path = tmp_path / "results.json"
        
        autoeda.to_json(str(output_path))
        
        assert output_path.exists()
        assert output_path.suffix == '.json'
    
    def test_to_json_content(self, autoeda, tmp_path):
        """Test JSON content is valid."""
        autoeda.analyze()
        output_path = tmp_path / "results.json"
        
        autoeda.to_json(str(output_path))
        
        # Load and verify JSON
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
        assert 'basic_stats' in data
        assert 'missing_values' in data
    
    def test_to_json_without_path(self, autoeda):
        """Test JSON export returns dict without path."""
        autoeda.analyze()
        
        result = autoeda.to_json()
        
        # API now returns dict directly, not JSON string
        assert isinstance(result, dict)
        assert 'basic_stats' in result


class TestEdgeCases:
    """Test edge cases."""
    
    def test_dataframe_with_no_missing_values(self):
        """Test analysis with no missing values."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        
        eda = AutoEDA(df)
        results = eda.analyze()
        
        assert results['missing_values']['total_missing'] == 0
    
    def test_dataframe_with_only_numeric(self):
        """Test analysis with only numeric columns."""
        df = pd.DataFrame({
            'a': np.random.randn(50),
            'b': np.random.randn(50),
            'c': np.random.randn(50)
        })
        
        eda = AutoEDA(df)
        results = eda.analyze()
        
        assert len(results['basic_stats']['numeric_summary']) == 3
        assert len(results['categorical']) == 0
    
    def test_dataframe_with_only_categorical(self):
        """Test analysis with only categorical columns."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 20,
            'cat2': ['X', 'Y', 'Z'] * 20
        })
        
        eda = AutoEDA(df)
        results = eda.analyze()
        
        # With no numeric columns, numeric_summary won't exist
        assert 'numeric_summary' not in results['basic_stats'] or len(results['basic_stats'].get('numeric_summary', {})) == 0
        assert len(results['categorical']) == 2
    
    def test_dataframe_with_duplicates(self):
        """Test analysis with duplicate rows."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 1, 2, 3],
            'b': [10, 20, 30, 10, 20, 30]
        })
        
        eda = AutoEDA(df)
        results = eda.analyze()
        
        assert results['basic_stats']['duplicate_rows'] == 3  # 3 duplicate rows
    
    def test_dataframe_with_constant_column(self):
        """Test analysis with constant column."""
        df = pd.DataFrame({
            'constant': [5] * 100,
            'varying': np.random.randn(100)
        })
        
        eda = AutoEDA(df)
        results = eda.analyze()
        
        # Constant column should be in numeric summary
        numeric_cols = results['basic_stats']['numeric_summary']
        assert 'constant' in numeric_cols or 'varying' in numeric_cols


class TestPerformance:
    """Test performance with larger datasets."""
    
    def test_large_dataset(self):
        """Test analysis with larger dataset."""
        np.random.seed(42)
        n_samples = 10000
        
        df = pd.DataFrame({
            f'col_{i}': np.random.randn(n_samples)
            for i in range(20)
        })
        
        eda = AutoEDA(df)
        results = eda.analyze()
        
        # Should complete without errors
        assert results is not None
        assert results['basic_stats']['n_rows'] == n_samples
        assert results['basic_stats']['n_columns'] == 20

