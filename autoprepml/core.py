"""Core high-level interface for AutoPrepML"""
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import logging
from datetime import datetime

from . import detection
from . import cleaning
from . import visualization
from .config import load_config, DEFAULT_CONFIG


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('autoprepml')


class AutoPrepML:
    """Main AutoPrepML class for automated data preprocessing.
    
    Example:
        >>> df = pd.read_csv('data.csv')
        >>> prep = AutoPrepML(df)
        >>> clean_df, report = prep.clean(task='classification', target_col='label')
        >>> prep.save_report('report.html')
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None, 
                 config_path: Optional[str] = None):
        """Initialize AutoPrepML with a DataFrame.
        
        Args:
            df: Input DataFrame to preprocess
            config: Optional configuration dictionary
            config_path: Optional path to YAML/JSON config file
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.cleaned_df = None
        self.log = []
        self.detection_results = {}
        self.plots = {}
        
        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = DEFAULT_CONFIG.copy()
        
        self._log_action('initialized', {'shape': df.shape, 'columns': list(df.columns)})
    
    def _log_action(self, action: str, details: Any) -> None:
        """Internal logging helper."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        self.log.append(entry)
        logger.info(f"{action}: {details}")
    
    def detect(self, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Run all detection functions.
        
        Args:
            target_col: Optional target column for imbalance detection
            
        Returns:
            Dictionary containing all detection results
        """
        self.detection_results = detection.detect_all(self.df, target_col)
        self._log_action('detection_complete', self.detection_results)
        return self.detection_results
    
    def clean(self, task: Optional[str] = None, target_col: Optional[str] = None,
              auto: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean the dataset automatically.
        
        Args:
            task: 'classification', 'regression', or None
            target_col: Name of target column (for imbalance handling)
            auto: If True, apply all cleaning steps automatically
            
        Returns:
            Tuple of (cleaned_df, report_dict)
        """
        df_clean = self.df.copy()
        
        # Run detection first
        if not self.detection_results:
            self.detect(target_col)
        
        # Step 1: Handle missing values
        if self.detection_results.get('missing_values'):
            strategy = self.config['cleaning']['missing_strategy']
            df_clean = cleaning.impute_missing(df_clean, strategy=strategy)
            self._log_action('imputed_missing', {'strategy': strategy})
        
        # Step 2: Handle outliers (optional)
        outliers = self.detection_results.get('outliers', {})
        if outliers.get('outlier_count', 0) > 0 and self.config['cleaning']['remove_outliers']:
            df_clean = cleaning.remove_outliers(df_clean, outliers['outlier_indices'])
            self._log_action('removed_outliers', {'count': outliers['outlier_count']})
        
        # Step 3: Encode categorical variables
        df_clean = cleaning.encode_categorical(df_clean, 
                                               method=self.config['cleaning']['encode_method'],
                                               exclude_cols=[target_col] if target_col else None)
        self._log_action('encoded_categorical', {'method': self.config['cleaning']['encode_method']})
        
        # Step 4: Scale features
        df_clean = cleaning.scale_features(df_clean, 
                                           method=self.config['cleaning']['scale_method'],
                                           exclude_cols=[target_col] if target_col else None)
        self._log_action('scaled_features', {'method': self.config['cleaning']['scale_method']})
        
        # Step 5: Balance classes (if classification task and imbalanced)
        if task == 'classification' and target_col:
            imbalance = self.detection_results.get('class_imbalance', {})
            if imbalance.get('is_imbalanced'):
                df_clean = cleaning.balance_classes(df_clean, target_col,
                                                    method=self.config['cleaning']['balance_method'])
                self._log_action('balanced_classes', {'method': self.config['cleaning']['balance_method']})
        
        self.cleaned_df = df_clean
        
        # Generate report
        report = self.report()
        
        return df_clean, report
    
    def summary(self) -> Dict[str, Any]:
        """Get a quick summary of the dataset.
        
        Returns:
            Dictionary with basic dataset statistics
        """
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'missing_values': detection.detect_missing(self.df),
            'numeric_columns': self.df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
    
    def report(self, include_plots: bool = True) -> Dict[str, Any]:
        """Generate comprehensive preprocessing report.
        
        Args:
            include_plots: Whether to generate visualization plots
            
        Returns:
            Complete report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'original_shape': self.original_df.shape,
            'cleaned_shape': self.cleaned_df.shape if self.cleaned_df is not None else None,
            'detection_results': self.detection_results,
            'logs': self.log,
            'config': self.config
        }
        
        # Generate plots if requested
        if include_plots and self.config['reporting']['include_plots']:
            outlier_indices = self.detection_results.get('outliers', {}).get('outlier_indices', [])
            self.plots = visualization.generate_all_plots(self.original_df, outlier_indices)
            report['plots'] = self.plots
        
        return report
    
    def save_report(self, output_path: str) -> None:
        """Save report to file.
        
        Args:
            output_path: Path to save report (supports .json, .html)
        """
        from .reports import generate_json_report, generate_html_report
        
        report = self.report(include_plots=True)
        
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generate_json_report(report))
        elif output_path.endswith('.html'):
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generate_html_report(report))
        else:
            raise ValueError("Output path must end with .json or .html")
        
        self._log_action('saved_report', {'path': output_path})
