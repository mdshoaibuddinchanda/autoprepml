"""Tests for Dashboard module."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from autoprepml.dashboard import (
    InteractiveDashboard,
    create_plotly_dashboard,
    create_correlation_heatmap,
    create_missing_data_plot,
    generate_streamlit_app
)


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
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    
    # Add missing values
    data['income'][np.random.choice(n_samples, 10, replace=False)] = np.nan
    data['credit_score'][np.random.choice(n_samples, 5, replace=False)] = np.nan
    
    return pd.DataFrame(data)


@pytest.fixture
def dashboard(sample_df):
    """Create InteractiveDashboard instance."""
    return InteractiveDashboard(sample_df)


class TestInteractiveDashboardInit:
    """Test InteractiveDashboard initialization."""
    
    def test_init_with_dataframe(self, sample_df):
        """Test initialization with valid DataFrame."""
        dashboard = InteractiveDashboard(sample_df)
        assert dashboard.df.equals(sample_df)
    
    def test_init_with_invalid_input(self):
        """Test initialization with invalid input."""
        with pytest.raises(TypeError):
            InteractiveDashboard("not a dataframe")
        
        with pytest.raises(ValueError):
            InteractiveDashboard(pd.DataFrame())  # Empty DataFrame


class TestCreateDashboard:
    """Test dashboard creation."""
    
    def test_create_dashboard_basic(self, dashboard, tmp_path):
        """Test basic dashboard creation."""
        output_path = tmp_path / "dashboard.html"
        
        dashboard.create_dashboard(
            title="Test Dashboard",
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        assert output_path.suffix == '.html'
    
    def test_create_dashboard_content(self, dashboard, tmp_path):
        """Test dashboard contains expected content."""
        output_path = tmp_path / "dashboard.html"
        
        dashboard.create_dashboard(output_path=str(output_path))
        
        content = output_path.read_text()
        
        # Should be valid HTML with Plotly
        assert '<html>' in content.lower()
        assert 'plotly' in content.lower()
    
    def test_create_dashboard_no_output(self, dashboard):
        """Test dashboard creation without output path."""
        # Should return HTML string
        result = dashboard.create_dashboard()
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert 'plotly' in result.lower()


class TestPlotlyDashboard:
    """Test Plotly dashboard convenience function."""
    
    def test_create_plotly_dashboard(self, sample_df, tmp_path):
        """Test create_plotly_dashboard function."""
        output_path = tmp_path / "plotly_dashboard.html"
        
        create_plotly_dashboard(
            sample_df,
            title="Test Plotly Dashboard",
            output_path=str(output_path)
        )
        
        assert output_path.exists()
    
    def test_create_plotly_dashboard_custom_title(self, sample_df, tmp_path):
        """Test dashboard with custom title."""
        output_path = tmp_path / "custom_dashboard.html"
        title = "My Custom Dashboard"
        
        create_plotly_dashboard(
            sample_df,
            title=title,
            output_path=str(output_path)
        )
        
        content = output_path.read_text()
        assert title in content


class TestCorrelationHeatmap:
    """Test correlation heatmap creation."""
    
    def test_create_correlation_heatmap_basic(self, dashboard, tmp_path):
        """Test basic correlation heatmap."""
        output_path = tmp_path / "correlation.html"
        
        dashboard.create_correlation_heatmap(output_path=str(output_path))
        
        assert output_path.exists()
        assert output_path.suffix == '.html'
    
    def test_create_correlation_heatmap_content(self, dashboard, tmp_path):
        """Test heatmap contains correlation data."""
        output_path = tmp_path / "correlation.html"
        
        dashboard.create_correlation_heatmap(output_path=str(output_path))
        
        content = output_path.read_text()
        
        # Should contain Plotly and heatmap elements
        assert 'plotly' in content.lower()
        assert '<html>' in content.lower()
    
    def test_create_correlation_heatmap_no_numeric(self):
        """Test heatmap with no numeric columns."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 10,
            'cat2': ['X', 'Y', 'Z'] * 10
        })
        
        dashboard = InteractiveDashboard(df)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(ValueError):
            dashboard.create_correlation_heatmap()
    
    def test_create_correlation_heatmap_function(self, sample_df, tmp_path):
        """Test correlation heatmap convenience function."""
        output_path = tmp_path / "corr_func.html"
        
        create_correlation_heatmap(sample_df, output_path=str(output_path))
        
        assert output_path.exists()


class TestMissingDataPlot:
    """Test missing data visualization."""
    
    def test_create_missing_data_plot_basic(self, dashboard, tmp_path):
        """Test basic missing data plot."""
        output_path = tmp_path / "missing.html"
        
        dashboard.create_missing_data_plot(output_path=str(output_path))
        
        assert output_path.exists()
        assert output_path.suffix == '.html'
    
    def test_create_missing_data_plot_content(self, dashboard, tmp_path):
        """Test missing data plot contains data."""
        output_path = tmp_path / "missing.html"
        
        dashboard.create_missing_data_plot(output_path=str(output_path))
        
        content = output_path.read_text()
        
        # Should contain Plotly visualization
        assert 'plotly' in content.lower()
    
    def test_create_missing_data_plot_no_missing(self):
        """Test missing data plot with no missing values."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        
        dashboard = InteractiveDashboard(df)
        result = dashboard.create_missing_data_plot()
        
        # Should still create visualization showing 0% missing
        assert result is not None
    
    def test_create_missing_data_plot_function(self, sample_df, tmp_path):
        """Test missing data plot convenience function."""
        output_path = tmp_path / "missing_func.html"
        
        create_missing_data_plot(sample_df, output_path=str(output_path))
        
        assert output_path.exists()


class TestStreamlitAppGeneration:
    """Test Streamlit app generation."""
    
    def test_generate_streamlit_app_basic(self, dashboard, tmp_path):
        """Test basic Streamlit app generation."""
        output_path = tmp_path / "app.py"
        
        dashboard.generate_streamlit_app(output_path=str(output_path))
        
        assert output_path.exists()
        assert output_path.suffix == '.py'
    
    def test_generate_streamlit_app_content(self, dashboard, tmp_path):
        """Test Streamlit app contains expected code."""
        output_path = tmp_path / "app.py"
        
        dashboard.generate_streamlit_app(output_path=str(output_path))
        
        content = output_path.read_text()
        
        # Should contain Streamlit imports and code
        assert 'import streamlit' in content
        assert 'st.' in content  # Streamlit function calls
        assert 'def ' in content  # Function definitions
    
    def test_generate_streamlit_app_runnable(self, dashboard, tmp_path):
        """Test generated Streamlit app is syntactically valid."""
        output_path = tmp_path / "app.py"
        
        dashboard.generate_streamlit_app(output_path=str(output_path))
        
        # Try to compile the generated code
        content = output_path.read_text()
        try:
            compile(content, str(output_path), 'exec')
            assert True
        except SyntaxError:
            pytest.fail("Generated Streamlit app has syntax errors")
    
    def test_generate_streamlit_app_function(self, sample_df, tmp_path):
        """Test Streamlit app generation convenience function."""
        output_path = tmp_path / "app_func.py"
        
        generate_streamlit_app(sample_df, output_path=str(output_path))
        
        assert output_path.exists()
        
        content = output_path.read_text()
        assert 'import streamlit' in content


class TestDashboardWithDifferentDataTypes:
    """Test dashboard with different data types."""
    
    def test_dashboard_numeric_only(self, tmp_path):
        """Test dashboard with only numeric columns."""
        df = pd.DataFrame({
            'col1': np.random.randn(50),
            'col2': np.random.randn(50),
            'col3': np.random.randn(50)
        })
        
        dashboard = InteractiveDashboard(df)
        output_path = tmp_path / "numeric_dashboard.html"
        
        dashboard.create_dashboard(output_path=str(output_path))
        
        assert output_path.exists()
    
    def test_dashboard_categorical_only(self, tmp_path):
        """Test dashboard with only categorical columns."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 20,
            'cat2': ['X', 'Y', 'Z'] * 20,
            'cat3': ['P', 'Q', 'R'] * 20
        })
        
        dashboard = InteractiveDashboard(df)
        output_path = tmp_path / "categorical_dashboard.html"
        
        dashboard.create_dashboard(output_path=str(output_path))
        
        assert output_path.exists()
    
    def test_dashboard_mixed_types(self, tmp_path):
        """Test dashboard with mixed data types."""
        df = pd.DataFrame({
            'numeric': np.random.randn(50),
            'categorical': ['A', 'B', 'C'] * 16 + ['A', 'B'],
            'integer': np.random.randint(0, 100, 50),
            'boolean': np.random.choice([True, False], 50)
        })
        
        dashboard = InteractiveDashboard(df)
        output_path = tmp_path / "mixed_dashboard.html"
        
        dashboard.create_dashboard(output_path=str(output_path))
        
        assert output_path.exists()


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_column_dataframe(self, tmp_path):
        """Test dashboard with single column."""
        df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        dashboard = InteractiveDashboard(df)
        
        output_path = tmp_path / "single_col.html"
        dashboard.create_dashboard(output_path=str(output_path))
        
        assert output_path.exists()
    
    def test_single_row_dataframe(self, tmp_path):
        """Test dashboard with single row."""
        df = pd.DataFrame({
            'a': [1],
            'b': [2],
            'c': [3]
        })
        
        dashboard = InteractiveDashboard(df)
        output_path = tmp_path / "single_row.html"
        
        # Should handle gracefully
        dashboard.create_dashboard(output_path=str(output_path))
        assert output_path.exists()
    
    def test_dataframe_with_all_missing(self, tmp_path):
        """Test dashboard with column of all missing values."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [np.nan] * 5,
            'c': [10, 20, 30, 40, 50]
        })
        
        dashboard = InteractiveDashboard(df)
        output_path = tmp_path / "all_missing.html"
        
        dashboard.create_dashboard(output_path=str(output_path))
        assert output_path.exists()
    
    def test_dataframe_with_unicode_columns(self, tmp_path):
        """Test dashboard with unicode column names."""
        df = pd.DataFrame({
            'año': [1, 2, 3, 4, 5],
            '名前': [10, 20, 30, 40, 50],
            'Größe': [100, 200, 300, 400, 500]
        })
        
        dashboard = InteractiveDashboard(df)
        output_path = tmp_path / "unicode.html"
        
        dashboard.create_dashboard(output_path=str(output_path))
        assert output_path.exists()
    
    def test_dataframe_with_special_chars_in_columns(self, tmp_path):
        """Test dashboard with special characters in column names."""
        df = pd.DataFrame({
            'col with spaces': [1, 2, 3, 4, 5],
            'col-with-dashes': [10, 20, 30, 40, 50],
            'col_with_underscores': [100, 200, 300, 400, 500]
        })
        
        dashboard = InteractiveDashboard(df)
        output_path = tmp_path / "special_chars.html"
        
        dashboard.create_dashboard(output_path=str(output_path))
        assert output_path.exists()


class TestOutputFormats:
    """Test different output formats and paths."""
    
    def test_dashboard_with_relative_path(self, dashboard):
        """Test dashboard creation with relative path."""
        output_path = "test_dashboard.html"
        
        try:
            dashboard.create_dashboard(output_path=output_path)
            assert Path(output_path).exists()
        finally:
            # Cleanup
            if Path(output_path).exists():
                Path(output_path).unlink()
    
    def test_dashboard_with_nested_path(self, dashboard, tmp_path):
        """Test dashboard creation with nested directory."""
        output_path = tmp_path / "subdir" / "nested" / "dashboard.html"
        
        dashboard.create_dashboard(output_path=str(output_path))
        
        # Should create nested directories
        assert output_path.exists()
    
    def test_dashboard_overwrite_existing(self, dashboard, tmp_path):
        """Test overwriting existing dashboard file."""
        output_path = tmp_path / "dashboard.html"
        
        # Create first dashboard
        dashboard.create_dashboard(output_path=str(output_path))
        assert output_path.exists()
        
        first_size = output_path.stat().st_size
        
        # Overwrite with new dashboard
        dashboard.create_dashboard(output_path=str(output_path))
        assert output_path.exists()
        
        # File should still exist (may have different size)
        assert output_path.stat().st_size > 0


class TestIntegration:
    """Test integration between different dashboard components."""
    
    def test_create_all_visualizations(self, dashboard, tmp_path):
        """Test creating all visualization types."""
        # Create main dashboard
        dashboard.create_dashboard(output_path=str(tmp_path / "main.html"))
        
        # Create correlation heatmap
        dashboard.create_correlation_heatmap(output_path=str(tmp_path / "corr.html"))
        
        # Create missing data plot
        dashboard.create_missing_data_plot(output_path=str(tmp_path / "missing.html"))
        
        # Generate Streamlit app
        dashboard.generate_streamlit_app(output_path=str(tmp_path / "app.py"))
        
        # All should exist
        assert (tmp_path / "main.html").exists()
        assert (tmp_path / "corr.html").exists()
        assert (tmp_path / "missing.html").exists()
        assert (tmp_path / "app.py").exists()


class TestPerformance:
    """Test performance with larger datasets."""
    
    def test_large_dataset(self, tmp_path):
        """Test dashboard with larger dataset."""
        np.random.seed(42)
        n_samples = 10000
        
        df = pd.DataFrame({
            f'col_{i}': np.random.randn(n_samples)
            for i in range(20)
        })
        
        dashboard = InteractiveDashboard(df)
        output_path = tmp_path / "large_dashboard.html"
        
        # Should complete without errors
        dashboard.create_dashboard(output_path=str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_many_categories(self, tmp_path):
        """Test dashboard with many categorical values."""
        n_samples = 1000
        
        df = pd.DataFrame({
            'category': [f'cat_{i}' for i in range(n_samples)],
            'value': np.random.randn(n_samples)
        })
        
        dashboard = InteractiveDashboard(df)
        output_path = tmp_path / "many_cats.html"
        
        # Should handle high cardinality
        dashboard.create_dashboard(output_path=str(output_path))
        
        assert output_path.exists()
