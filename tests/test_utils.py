"""Tests for public utility helpers."""

import pandas as pd
import pytest

from autoprepml.utils import summarize_missing


def test_summarize_missing_reports_count_and_percentage():
    """Only columns with missing values are included in the summary."""
    frame = pd.DataFrame({"complete": [1, 2], "partial": [None, 2], "all": [None, None]})

    assert summarize_missing(frame) == {
        "partial": {"count": 1, "percent": 50.0},
        "all": {"count": 2, "percent": 100.0},
    }


def test_summarize_missing_handles_empty_and_complete_frames():
    """Empty and complete data have no missing-value entries."""
    assert summarize_missing(pd.DataFrame(columns=["a", "b"])) == {}
    assert summarize_missing(pd.DataFrame({"a": [1, 2]})) == {}


def test_summarize_missing_validates_input():
    """The helper fails clearly for non-DataFrame input."""
    with pytest.raises(TypeError, match="pandas DataFrame"):
        summarize_missing({"a": [1, 2]})
