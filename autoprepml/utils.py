"""Small, dependency-light utility functions for AutoPrepML."""

from typing import Any, Dict

import pandas as pd


def summarize_missing(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize missing values without exposing row-level data.

    Args:
        df: DataFrame to inspect.

    Returns:
        Mapping of columns containing missing values to their count and
        percentage of rows. Empty DataFrames return an empty mapping.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        return {}

    miss = df.isna().sum()
    pct = (miss / len(df) * 100).round(2)
    return {
        col: {"count": int(miss.loc[col]), "percent": float(pct.loc[col])}
        for col in df.columns
        if miss.loc[col] > 0
    }
