"""Automated Exploratory Data Analysis (AutoEDA) module for AutoPrepML"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json


class AutoEDA:
    """Automated Exploratory Data Analysis class.
    
    Provides comprehensive EDA capabilities:
    - Statistical summaries
    - Distribution analysis
    - Correlation analysis
    - Outlier detection and visualization
    - Missing value analysis
    - Automated insights generation
    - HTML report generation
    
    Example:
        >>> eda = AutoEDA(df)
        >>> eda.analyze()
        >>> insights = eda.get_insights()
        >>> eda.generate_report('eda_report.html')
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize AutoEDA.
        
        Args:
            df: Pandas DataFrame to analyze
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        self.df = df.copy()
        self.analysis_results = {}
        self.insights = []
        
    def analyze(self, 
                include_correlations: bool = True,
                include_distributions: bool = True,
                include_outliers: bool = True,
                generate_insights: bool = True) -> Dict[str, Any]:
        """Run comprehensive EDA analysis.
        
        Args:
            include_correlations: Include correlation analysis
            include_distributions: Include distribution analysis
            include_outliers: Include outlier detection
            generate_insights: Generate automated insights
            
        Returns:
            Dictionary with all analysis results
        """
        # Basic statistics
        self.analysis_results['basic_stats'] = self._compute_basic_stats()
        
        # Data types and info
        self.analysis_results['data_info'] = self._get_data_info()
        
        # Missing values
        self.analysis_results['missing_values'] = self._analyze_missing_values()
        
        # Numeric analysis
        if include_distributions:
            self.analysis_results['distributions'] = self._analyze_distributions()
        
        # Correlations
        if include_correlations:
            self.analysis_results['correlations'] = self._compute_correlations()
        
        # Outlier detection
        if include_outliers:
            self.analysis_results['outliers'] = self._detect_outliers()
        
        # Categorical analysis
        self.analysis_results['categorical'] = self._analyze_categorical()
        
        # Generate insights
        if generate_insights:
            self.insights = self._generate_insights()
            self.analysis_results['insights'] = self.insights
        
        return self.analysis_results
    
    def _compute_basic_stats(self) -> Dict[str, Any]:
        """Compute basic statistical measures."""
        stats = {
            'shape': self.df.shape,
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'duplicate_rows': self.df.duplicated().sum(),
        }
        
        # Numeric statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats['numeric_summary'] = self.df[numeric_cols].describe().to_dict()
        
        return stats
    
    def _get_data_info(self) -> Dict[str, Any]:
        """Get data type information."""
        dtypes_count = self.df.dtypes.value_counts().to_dict()
        
        info = {
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'dtype_counts': {str(k): int(v) for k, v in dtypes_count.items()},
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': self.df.select_dtypes(include=['datetime64']).columns.tolist(),
        }
        
        return info
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values."""
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df)) * 100
        
        missing_data = []
        for col in self.df.columns:
            if missing_counts[col] > 0:
                missing_data.append({
                    'column': col,
                    'missing_count': int(missing_counts[col]),
                    'missing_percentage': float(missing_pct[col]),
                    'dtype': str(self.df[col].dtype)
                })
        
        return {
            'total_missing': int(self.df.isnull().sum().sum()),
            'columns_with_missing': len(missing_data),
            'missing_by_column': missing_data,
            'completely_empty_columns': [col for col in self.df.columns if self.df[col].isnull().all()]
        }
    
    def _analyze_distributions(self) -> Dict[str, Any]:
        """Analyze distributions of numeric columns."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        distributions = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                distributions[col] = {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'unique_values': int(data.nunique()),
                    'zeros_count': int((data == 0).sum()),
                }
        
        return distributions
    
    def _compute_correlations(self) -> Dict[str, Any]:
        """Compute correlation matrix for numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {'message': 'Less than 2 numeric columns for correlation'}
        
        corr_matrix = numeric_df.corr()
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Strong correlation threshold
                    high_corr.append({
                        'column1': corr_matrix.columns[i],
                        'column2': corr_matrix.columns[j],
                        'correlation': float(corr_val),
                        'strength': 'strong' if abs(corr_val) > 0.9 else 'moderate'
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr,
            'n_numeric_columns': len(numeric_df.columns)
        }
    
    def _detect_outliers(self) -> Dict[str, Any]:
        """Detect outliers using IQR and Z-score methods."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) < 4:  # Need at least 4 values
                continue
            
            # IQR method
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            
            # Z-score method
            z_scores = np.abs((data - data.mean()) / data.std())
            zscore_outliers = (z_scores > 3).sum()
            
            if iqr_outliers > 0 or zscore_outliers > 0:
                outliers[col] = {
                    'iqr_outliers': int(iqr_outliers),
                    'iqr_percentage': float((iqr_outliers / len(data)) * 100),
                    'zscore_outliers': int(zscore_outliers),
                    'zscore_percentage': float((zscore_outliers / len(data)) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                }
        
        return outliers
    
    def _analyze_categorical(self) -> Dict[str, Any]:
        """Analyze categorical columns."""
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        categorical = {}
        
        for col in cat_cols:
            value_counts = self.df[col].value_counts()
            categorical[col] = {
                'unique_values': int(self.df[col].nunique()),
                'most_common': value_counts.head(10).to_dict(),
                'cardinality': 'high' if self.df[col].nunique() > 50 else 'low',
                'mode': str(self.df[col].mode()[0]) if len(self.df[col].mode()) > 0 else None,
                'missing': int(self.df[col].isnull().sum())
            }
        
        return categorical
    
    def _generate_insights(self) -> List[str]:
        """Generate automated insights from analysis."""
        insights = []
        
        # Data size insights
        n_rows, n_cols = self.df.shape
        insights.append(f"📊 Dataset contains {n_rows:,} rows and {n_cols} columns")
        
        # Memory usage
        mem_mb = self.df.memory_usage(deep=True).sum() / 1024**2
        insights.append(f"💾 Memory usage: {mem_mb:.2f} MB")
        
        # Duplicates
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            insights.append(f"⚠️  Found {dup_count:,} duplicate rows ({(dup_count/n_rows)*100:.1f}%)")
        
        # Missing values
        missing_info = self.analysis_results.get('missing_values', {})
        if missing_info.get('total_missing', 0) > 0:
            insights.append(f"⚠️  Missing values in {missing_info['columns_with_missing']} columns")
        
        # High correlations
        corr_info = self.analysis_results.get('correlations', {})
        high_corr = corr_info.get('high_correlations', [])
        if high_corr:
            insights.append(f"🔗 Found {len(high_corr)} pairs of highly correlated features")
        
        # Outliers
        outlier_info = self.analysis_results.get('outliers', {})
        if outlier_info:
            insights.append(f"📈 Detected outliers in {len(outlier_info)} numeric columns")
        
        # Categorical cardinality
        cat_info = self.analysis_results.get('categorical', {})
        high_card_cols = [col for col, info in cat_info.items() if info['cardinality'] == 'high']
        if high_card_cols:
            insights.append(f"🏷️  {len(high_card_cols)} categorical columns with high cardinality")
        
        # Data type distribution
        data_info = self.analysis_results.get('data_info', {})
        n_numeric = len(data_info.get('numeric_columns', []))
        n_cat = len(data_info.get('categorical_columns', []))
        insights.append(f"📋 Data types: {n_numeric} numeric, {n_cat} categorical")
        
        return insights
    
    def get_insights(self) -> List[str]:
        """Get generated insights.
        
        Returns:
            List of insight strings
        """
        if not self.insights:
            self.insights = self._generate_insights()
        return self.insights
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a compact summary of the analysis.
        
        Returns:
            Dictionary with key findings
        """
        if not self.analysis_results:
            self.analyze()
        
        summary = {
            'dataset_shape': self.analysis_results['basic_stats']['shape'],
            'n_numeric_columns': len(self.analysis_results['data_info']['numeric_columns']),
            'n_categorical_columns': len(self.analysis_results['data_info']['categorical_columns']),
            'total_missing_values': self.analysis_results['missing_values']['total_missing'],
            'duplicate_rows': self.analysis_results['basic_stats']['duplicate_rows'],
            'columns_with_outliers': len(self.analysis_results.get('outliers', {})),
            'high_correlations': len(self.analysis_results.get('correlations', {}).get('high_correlations', [])),
            'insights': self.get_insights()
        }
        
        return summary
    
    def generate_report(self, output_path: str, title: str = "AutoEDA Report") -> None:
        """Generate HTML report with visualizations.
        
        Args:
            output_path: Path to save HTML report
            title: Report title
        """
        if not self.analysis_results:
            self.analyze()
        
        html = self._create_html_report(title)
        
        Path(output_path).write_text(html, encoding='utf-8')
        print(f"✅ EDA report saved to: {output_path}")
    
    def _create_html_report(self, title: str) -> str:
        """Create HTML report content."""
        insights_html = "\n".join([f"<li>{insight}</li>" for insight in self.get_insights()])
        
        # Missing values table
        missing_data = self.analysis_results['missing_values']['missing_by_column']
        missing_html = ""
        if missing_data:
            missing_rows = "\n".join([
                f"<tr><td>{item['column']}</td><td>{item['missing_count']}</td>"
                f"<td>{item['missing_percentage']:.2f}%</td></tr>"
                for item in missing_data
            ])
            missing_html = f"""
            <h3>Missing Values</h3>
            <table>
                <tr><th>Column</th><th>Missing Count</th><th>Percentage</th></tr>
                {missing_rows}
            </table>
            """
        
        # Outliers table
        outliers = self.analysis_results.get('outliers', {})
        outliers_html = ""
        if outliers:
            outlier_rows = "\n".join([
                f"<tr><td>{col}</td><td>{info['iqr_outliers']}</td>"
                f"<td>{info['iqr_percentage']:.2f}%</td></tr>"
                for col, info in outliers.items()
            ])
            outliers_html = f"""
            <h3>Outliers (IQR Method)</h3>
            <table>
                <tr><th>Column</th><th>Outlier Count</th><th>Percentage</th></tr>
                {outlier_rows}
            </table>
            """
        
        # Correlations
        corr_info = self.analysis_results.get('correlations', {})
        high_corr = corr_info.get('high_correlations', [])
        corr_html = ""
        if high_corr:
            corr_rows = "\n".join([
                f"<tr><td>{item['column1']}</td><td>{item['column2']}</td>"
                f"<td>{item['correlation']:.3f}</td><td>{item['strength']}</td></tr>"
                for item in high_corr
            ])
            corr_html = f"""
            <h3>High Correlations</h3>
            <table>
                <tr><th>Column 1</th><th>Column 2</th><th>Correlation</th><th>Strength</th></tr>
                {corr_rows}
            </table>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                h1 {{ margin: 0; }}
                .section {{ background: white; padding: 25px; margin: 20px 0; 
                           border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #667eea; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .insights {{ list-style: none; padding: 0; }}
                .insights li {{ padding: 10px; margin: 5px 0; background: #e3f2fd; 
                               border-left: 4px solid #2196F3; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                              gap: 15px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; 
                             text-align: center; border: 1px solid #dee2e6; }}
                .stat-value {{ font-size: 32px; font-weight: bold; color: #667eea; }}
                .stat-label {{ color: #6c757d; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Automated Exploratory Data Analysis</p>
            </div>
            
            <div class="section">
                <h2>Key Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{self.df.shape[0]:,}</div>
                        <div class="stat-label">Rows</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{self.df.shape[1]}</div>
                        <div class="stat-label">Columns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{self.analysis_results['missing_values']['total_missing']:,}</div>
                        <div class="stat-label">Missing Values</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{self.analysis_results['basic_stats']['duplicate_rows']:,}</div>
                        <div class="stat-label">Duplicates</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Automated Insights</h2>
                <ul class="insights">
                    {insights_html}
                </ul>
            </div>
            
            <div class="section">
                {missing_html}
            </div>
            
            <div class="section">
                {outliers_html}
            </div>
            
            <div class="section">
                {corr_html}
            </div>
            
            <div class="section">
                <h3>Data Types</h3>
                <p>Numeric columns: {len(self.analysis_results['data_info']['numeric_columns'])}</p>
                <p>Categorical columns: {len(self.analysis_results['data_info']['categorical_columns'])}</p>
                <p>DateTime columns: {len(self.analysis_results['data_info']['datetime_columns'])}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def to_json(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Export analysis results to JSON.
        
        Args:
            output_path: Optional path to save JSON file
            
        Returns:
            Dictionary with analysis results
        """
        if not self.analysis_results:
            self.analyze()
        
        # Convert numpy types to Python types for JSON serialization
        results = json.loads(json.dumps(self.analysis_results, default=str))
        
        if output_path:
            Path(output_path).write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(f"✅ Analysis results saved to: {output_path}")
        
        return results
