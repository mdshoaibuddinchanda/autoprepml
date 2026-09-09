"""Tests for sklearn model pipeline integration."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from autoprepml.pipeline import make_model_pipeline, make_preprocessing_pipeline


def _training_frame():
    return pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45],
            "city": ["a", "a", "b", "b", "c", "c"],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )


def test_model_pipeline_fits_and_handles_unknown_categories():
    frame = _training_frame()
    pipeline = make_model_pipeline(
        frame,
        LogisticRegression(max_iter=200),
        target_col="target",
    )
    pipeline.fit(frame.drop(columns="target"), frame["target"])
    predictions = pipeline.predict(pd.DataFrame({"age": [28, 42], "city": ["new-city", "c"]}))

    assert len(predictions) == 2
    assert "model" in pipeline.named_steps


def test_preprocessing_pipeline_validates_frame():
    with pytest.raises(TypeError, match="DataFrame"):
        make_preprocessing_pipeline([[1]])
    with pytest.raises(ValueError, match="not found"):
        make_preprocessing_pipeline(_training_frame(), target_col="missing")
    with pytest.raises(ValueError, match="at least one"):
        make_preprocessing_pipeline(pd.DataFrame({"target": [1, 2]}), target_col="target")
