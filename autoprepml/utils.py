"""Utility functions for AutoPrepML"""
from typing import Any, Dict


def summarize_missing(df) -> Dict[str, Any]:
    miss = df.isnull().sum()
    pct = (miss / len(df) * 100).round(2)
    return {col: {'count': int(miss.loc[col]), 'percent': float(pct.loc[col])} for col in df.columns if miss.loc[col] > 0}
