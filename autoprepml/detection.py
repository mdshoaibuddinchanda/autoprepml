"""Detection functions for AutoPrepML - identify data quality issues"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_missing(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect missing values in DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with column names as keys and missing stats as values
    """
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df) * 100).round(2)

    return {
        col: {
            "count": int(missing_counts[col]),
            "percent": float(missing_percent[col]),
            "dtype": str(df[col].dtype),
        }
        for col in df.columns
        if missing_counts[col] > 0
    }


def detect_outliers(
    df: pd.DataFrame,
    method: str = "iforest",
    contamination: float = 0.05,
    threshold: float = 3.0,
    exclude_cols: list = None,
) -> Dict[str, Any]:
    """Detect outliers in numeric columns.

    Args:
        df: Input DataFrame
        method: 'iforest' (Isolation Forest) or 'zscore' (Z-score method)
        contamination: Expected proportion of outliers (for iforest)
        threshold: Z-score threshold (for zscore method)
        exclude_cols: Columns excluded from outlier detection, such as a target
            column that is not an input feature.

    Returns:
        Dictionary with outlier statistics and indices
    """
    exclude_cols = set(exclude_cols or [])
    numeric_cols = [
        column
        for column in df.select_dtypes(include=[np.number]).columns.tolist()
        if column not in exclude_cols and not pd.api.types.is_bool_dtype(df[column])
    ]

    if len(numeric_cols) == 0:
        return {"error": "No numeric columns found", "outlier_count": 0}

    non_finite = ~np.isfinite(df[numeric_cols].to_numpy(dtype=float))
    non_finite_count = int(non_finite.sum())
    numeric_df = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(numeric_df) == 0:
        return {"error": "No valid numeric data after dropping NaN", "outlier_count": 0}

    result = {
        "method": method,
        "numeric_columns": numeric_cols,
        "non_finite_count": non_finite_count,
    }

    if method == "iforest":
        iso = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso.fit_predict(numeric_df)
        outlier_mask = predictions == -1
        result["outlier_count"] = int(outlier_mask.sum())
        result["outlier_indices"] = numeric_df.index[outlier_mask].tolist()

    elif method == "zscore":
        z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std()
        outlier_mask = (np.abs(z_scores) > threshold).any(axis=1)
        result["outlier_count"] = int(outlier_mask.sum())
        result["outlier_indices"] = numeric_df.index[outlier_mask].tolist()

    else:
        result["error"] = f"Unknown method: {method}"
        result["outlier_count"] = 0

    return result


def detect_imbalance(df: pd.DataFrame, target_col: str, threshold: float = 0.3) -> Dict[str, Any]:
    """Detect class imbalance in target column.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        threshold: Minimum proportion for minority class (default 0.3 = 30%)

    Returns:
        Dictionary with class distribution and imbalance flag
    """
    if target_col not in df.columns:
        return {"error": f"Column {target_col} not found"}

    target = df[target_col].dropna()
    missing_count = int(df[target_col].isna().sum())
    if target.empty:
        return {"error": f"Column {target_col} contains no non-missing labels"}
    value_counts = target.value_counts()
    total = len(target)
    proportions = (value_counts / total).round(4)

    min_proportion = proportions.min()
    is_imbalanced = min_proportion < threshold

    return {
        "target_column": target_col,
        "missing_labels": missing_count,
        "class_distribution": value_counts.to_dict(),
        "class_proportions": proportions.to_dict(),
        "is_imbalanced": is_imbalanced,
        "minority_class": proportions.idxmin(),
        "minority_proportion": min_proportion,
        "imbalance_ratio": proportions.max() / min_proportion,
    }


def detect_all(
    df: pd.DataFrame,
    target_col: str = None,
    *,
    outlier_method: str = "iforest",
    contamination: float = 0.05,
    zscore_threshold: float = 3.0,
    imbalance_threshold: float = 0.3,
    exclude_cols: list = None,
) -> Dict[str, Any]:
    """Run all detection functions and return comprehensive report.

    Args:
        df: Input DataFrame
        target_col: Optional target column for imbalance detection

    Returns:
        Dictionary containing all detection results
    """
    results = {
        "missing_values": detect_missing(df),
        "outliers": detect_outliers(
            df,
            method=outlier_method,
            contamination=contamination,
            threshold=zscore_threshold,
            exclude_cols=exclude_cols,
        ),
    }

    if target_col:
        results["class_imbalance"] = detect_imbalance(
            df,
            target_col,
            threshold=imbalance_threshold,
        )

    return results
