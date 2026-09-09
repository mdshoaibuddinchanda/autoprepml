"""Core high-level interface for AutoPrepML"""

import copy
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from . import detection
from . import cleaning
from . import visualization
from .config import load_config

# Optional LLM support
try:
    from .llm_suggest import LLMSuggestor

    HAS_LLM_SUPPORT = True
except ImportError:
    HAS_LLM_SUPPORT = False


# Libraries must not configure the host application's logging globally.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _data_fingerprint(df: pd.DataFrame) -> Optional[str]:
    """Return a process-local fingerprint for safe detection-cache reuse."""
    try:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
        digest.update(repr(tuple(df.columns)).encode("utf-8"))
        digest.update(repr(tuple(str(dtype) for dtype in df.dtypes)).encode("utf-8"))
        return digest.hexdigest()
    except (TypeError, ValueError):
        # Extension objects that cannot be hashed remain fully supported; they
        # simply bypass the optional cache.
        return None


class AutoPrepML:
    """Main AutoPrepML class for automated data preprocessing.

    Example:
        >>> df = pd.read_csv('data.csv')
        >>> prep = AutoPrepML(df)
        >>> clean_df, report = prep.clean(task='classification', target_col='label')
        >>> prep.save_report('report.html')

        >>> # With LLM suggestions
        >>> prep = AutoPrepML(df, enable_llm=True, llm_provider='openai')
        >>> suggestions = prep.get_llm_suggestions(column='age', issue_type='missing')
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        enable_llm: bool = False,
        llm_provider: str = "openai",
        llm_api_key: Optional[str] = None,
    ):
        """Initialize AutoPrepML with a DataFrame.

        Args:
            df: Input DataFrame to preprocess
            config: Optional configuration dictionary
            config_path: Optional path to YAML/JSON config file
            enable_llm: Enable LLM-powered suggestions
            llm_provider: LLM provider ('openai', 'anthropic', 'google', 'ollama')
            llm_api_key: API key for LLM provider (optional, can use config)
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")

        self.original_df = df.copy()
        self.df = df.copy()
        self.cleaned_df = None
        self.log = []
        self.detection_results = {}
        self._detection_target_col = None
        self._detection_cache_key = None
        self._detection_cache = None
        self.plots = {}

        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = copy.deepcopy(config)
        else:
            self.config = load_config()

        # Initialize LLM if enabled
        self.llm_enabled = enable_llm
        self.llm_suggestor = None
        if enable_llm:
            if HAS_LLM_SUPPORT:
                try:
                    self.llm_suggestor = LLMSuggestor(provider=llm_provider, api_key=llm_api_key)
                    self._log_action("llm_initialized", {"provider": llm_provider})
                except Exception as e:
                    logger.warning(f"Failed to initialize LLM: {e}")
                    self.llm_enabled = False
            else:
                logger.warning(
                    "LLM support not available. Install with: pip install autoprepml[llm]"
                )
                self.llm_enabled = False

        self._log_action("initialized", {"shape": df.shape, "columns": list(df.columns)})

    def _log_action(self, action: str, details: Any) -> None:
        """Internal logging helper."""
        entry = {"timestamp": datetime.now().isoformat(), "action": action, "details": details}
        self.log.append(entry)
        logger.info(f"{action}: {details}")

    def detect(
        self,
        target_col: Optional[str] = None,
        *,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Run all detection functions.

        Args:
            target_col: Optional target column for imbalance detection
            use_cache: Reuse results when the DataFrame, target, and detection
                configuration are unchanged.

        Returns:
            Dictionary containing all detection results
        """
        detection_config = self.config.get("detection", {})
        cache_key = None
        if use_cache:
            fingerprint = _data_fingerprint(self.df)
            if fingerprint is not None:
                cache_key = (fingerprint, target_col, repr(detection_config))
                if cache_key == self._detection_cache_key and self._detection_cache is not None:
                    self.detection_results = copy.deepcopy(self._detection_cache)
                    self._detection_target_col = target_col
                    self._log_action("detection_cache_hit", {"target_col": target_col})
                    return copy.deepcopy(self.detection_results)

        self.detection_results = detection.detect_all(
            self.df,
            target_col,
            outlier_method=detection_config.get("outlier_method", "iforest"),
            contamination=detection_config.get("contamination", 0.05),
            zscore_threshold=detection_config.get("zscore_threshold", 3.0),
            imbalance_threshold=detection_config.get("imbalance_threshold", 0.3),
        )
        self._detection_target_col = target_col
        if cache_key is not None:
            self._detection_cache_key = cache_key
            self._detection_cache = copy.deepcopy(self.detection_results)
        self._log_action(
            "detection_complete",
            {
                "sections": list(self.detection_results),
                "missing_columns": len(self.detection_results.get("missing_values", {})),
                "outlier_count": self.detection_results.get("outliers", {}).get("outlier_count", 0),
            },
        )
        return self.detection_results

    def clean(
        self,
        task: Optional[str] = None,
        target_col: Optional[str] = None,
        auto: bool = True,
        use_advanced: bool = False,
        imputation_method: str = "simple",
        balance_method: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean the dataset automatically.

        Args:
            task: 'classification', 'regression', or None
            target_col: Name of target column (for imbalance handling)
            auto: If True, apply all cleaning steps automatically
            use_advanced: Use advanced imputation methods (KNN, Iterative)
            imputation_method: 'simple', 'knn', or 'iterative'
            balance_method: 'oversample', 'undersample', or 'smote'

        Returns:
            Tuple of (cleaned_df, report_dict)
        """
        if task not in (None, "classification", "regression"):
            raise ValueError("task must be 'classification', 'regression', or None")
        if target_col is not None and target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        df_clean = self.df.copy()

        # Run detection first
        if not self.detection_results or self._detection_target_col != target_col:
            self.detect(target_col)

        if not auto:
            self.cleaned_df = df_clean
            return df_clean, self.report()

        # Step 1: Handle missing values with advanced options
        if self.detection_results.get("missing_values"):
            if use_advanced and imputation_method == "knn":
                df_clean = cleaning.impute_knn(df_clean)
                self._log_action("imputed_missing", {"strategy": "knn"})
            elif use_advanced and imputation_method == "iterative":
                df_clean = cleaning.impute_iterative(df_clean)
                self._log_action("imputed_missing", {"strategy": "iterative"})
            else:
                cleaning_config = self.config.get("cleaning", {})
                strategy = cleaning_config.get("missing_strategy", "auto")
                df_clean = cleaning.impute_missing(
                    df_clean,
                    strategy=strategy,
                    numeric_strategy=cleaning_config.get("numeric_strategy", "median"),
                    categorical_strategy=cleaning_config.get("categorical_strategy", "mode"),
                )
                self._log_action("imputed_missing", {"strategy": strategy})

        # Step 2: Handle outliers (optional)
        outliers = self.detection_results.get("outliers", {})
        if outliers.get("outlier_count", 0) > 0 and self.config.get("cleaning", {}).get(
            "remove_outliers", False
        ):
            df_clean = cleaning.remove_outliers(df_clean, outliers["outlier_indices"])
            self._log_action("removed_outliers", {"count": outliers["outlier_count"]})

        # Step 3: Encode categorical variables
        cleaning_config = self.config.get("cleaning", {})
        df_clean = cleaning.encode_categorical(
            df_clean,
            method=cleaning_config.get("encode_method", "label"),
            exclude_cols=[target_col] if target_col else None,
        )
        self._log_action(
            "encoded_categorical", {"method": cleaning_config.get("encode_method", "label")}
        )

        # Step 4: Scale features
        df_clean = cleaning.scale_features(
            df_clean,
            method=cleaning_config.get("scale_method", "standard"),
            exclude_cols=[target_col] if target_col else None,
        )
        self._log_action(
            "scaled_features", {"method": cleaning_config.get("scale_method", "standard")}
        )

        # Step 5: Balance classes (if classification task and imbalanced)
        if task == "classification" and target_col:
            imbalance = self.detection_results.get("class_imbalance", {})
            if imbalance.get("is_imbalanced"):
                selected_balance_method = balance_method or cleaning_config.get(
                    "balance_method", "oversample"
                )
                if selected_balance_method == "smote":
                    df_clean = cleaning.balance_classes_smote(df_clean, target_col)
                    self._log_action("balanced_classes", {"method": "smote"})
                else:
                    df_clean = cleaning.balance_classes(
                        df_clean, target_col, method=selected_balance_method
                    )
                    self._log_action("balanced_classes", {"method": selected_balance_method})

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
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": detection.detect_missing(self.df),
            "numeric_columns": self.df.select_dtypes(include=["number"]).columns.tolist(),
            "categorical_columns": self.df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
        }

    def report(self, include_plots: bool = True) -> Dict[str, Any]:
        """Generate comprehensive preprocessing report.

        Args:
            include_plots: Whether to generate visualization plots

        Returns:
            Complete report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "original_shape": self.original_df.shape,
            "cleaned_shape": self.cleaned_df.shape if self.cleaned_df is not None else None,
            "detection_results": self.detection_results,
            "logs": self.log,
            "config": self.config,
        }

        # Generate plots if requested
        if include_plots and self.config.get("reporting", {}).get("include_plots", True):
            outlier_indices = self.detection_results.get("outliers", {}).get("outlier_indices", [])
            self.plots = visualization.generate_all_plots(self.original_df, outlier_indices)
            report["plots"] = self.plots

        return report

    def save_report(self, output_path: str) -> None:
        """Save report to file.

        Args:
            output_path: Path to save report (supports .json, .html)
        """
        from .reports import generate_json_report, generate_html_report

        report = self.report(include_plots=True)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.suffix.lower() == ".json":
            with output_file.open("w", encoding="utf-8") as f:
                f.write(generate_json_report(report))
        elif output_file.suffix.lower() == ".html":
            with output_file.open("w", encoding="utf-8") as f:
                f.write(generate_html_report(report))
        else:
            raise ValueError("Output path must end with .json or .html")

        self._log_action("saved_report", {"path": output_path})

    def get_llm_suggestions(self, column: Optional[str] = None, issue_type: str = "missing") -> str:
        """Get LLM-powered suggestions for data quality issues.

        Args:
            column: Column name to get suggestions for (optional)
            issue_type: Type of issue ('missing', 'outliers', 'duplicates')

        Returns:
            AI-generated suggestions as string
        """
        if not self.llm_enabled or not self.llm_suggestor:
            return "LLM support not enabled. Initialize with enable_llm=True"

        if column:
            return self.llm_suggestor.suggest_fix(self.df, column, issue_type)
        else:
            return self.llm_suggestor.analyze_dataframe(self.df)

    def analyze_with_llm(self, task: Optional[str] = None, target_col: Optional[str] = None) -> str:
        """Get comprehensive LLM analysis of the dataset.

        Args:
            task: ML task type ('classification', 'regression')
            target_col: Target column name

        Returns:
            Comprehensive AI analysis and recommendations
        """
        if not self.llm_enabled or not self.llm_suggestor:
            return "LLM support not enabled. Initialize with enable_llm=True"

        return self.llm_suggestor.analyze_dataframe(self.df, task=task, target_col=target_col)

    def get_feature_suggestions(
        self, task: str = "classification", target_col: Optional[str] = None
    ) -> list:
        """Get LLM-powered feature engineering suggestions.

        Args:
            task: ML task type ('classification', 'regression')
            target_col: Target column name

        Returns:
            List of feature engineering suggestions
        """
        if not self.llm_enabled or not self.llm_suggestor:
            return ["LLM support not enabled. Initialize with enable_llm=True"]

        return self.llm_suggestor.suggest_features(self.df, task=task, target_col=target_col)

    def explain_step(self, step_name: str, step_details: Dict[str, Any]) -> str:
        """Get natural language explanation of a cleaning step.

        Args:
            step_name: Name of the cleaning step
            step_details: Details/parameters of the step

        Returns:
            Human-readable explanation
        """
        if not self.llm_enabled or not self.llm_suggestor:
            return "LLM support not enabled. Initialize with enable_llm=True"

        return self.llm_suggestor.explain_cleaning_step(step_name, step_details)
