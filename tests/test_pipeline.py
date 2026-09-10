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


@pytest.mark.parametrize("scale_method", ["minmax", "robust", "maxabs"])
def test_preprocessing_pipeline_supports_explicit_scalers(scale_method):
    frame = _training_frame()
    frame["active"] = [True, False, True, False, True, False]
    pipeline = make_preprocessing_pipeline(frame, target_col="target", scale_method=scale_method)
    transformed = pipeline.fit_transform(frame.drop(columns="target"))

    assert transformed.shape[0] == len(frame)
    assert (
        pipeline.named_steps["preprocessor"]
        .transformers_[0][1]
        .named_steps["scaler"]
        .__class__.__name__.lower()
        .startswith(scale_method)
    )


def test_preprocessing_pipeline_handles_bool_only_features():
    frame = pd.DataFrame({"active": [True, False, True, False], "target": [0, 1, 0, 1]})
    pipeline = make_preprocessing_pipeline(frame, target_col="target")

    transformed = pipeline.fit_transform(frame.drop(columns="target"))

    assert transformed.shape == (4, 2)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"scale_numeric": "yes"}, "scale_numeric"),
        ({"scale_method": None}, "scale_method"),
        ({"scale_method": "unknown"}, "scale_method"),
        ({"encode_method": None}, "encode_method"),
        ({"encode_method": "unknown"}, "encode_method"),
    ],
)
def test_preprocessing_pipeline_validates_scaler_and_encoder_options(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        make_preprocessing_pipeline(_training_frame(), **kwargs)


def test_preprocessing_pipeline_can_disable_scaling_and_model_requires_estimator():
    frame = _training_frame()
    pipeline = make_preprocessing_pipeline(frame, target_col="target", scale_numeric=False)
    transformed = pipeline.fit_transform(frame.drop(columns="target"))
    assert transformed.shape[0] == len(frame)
    with pytest.raises(TypeError, match="estimator"):
        make_model_pipeline(frame, estimator=None)
    with pytest.raises(TypeError, match="estimator"):
        make_model_pipeline(frame, estimator=object())
