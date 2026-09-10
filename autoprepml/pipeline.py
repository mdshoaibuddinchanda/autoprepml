"""Scikit-learn pipeline integration for leakage-safe model workflows."""

import inspect
from typing import Any, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    FunctionTransformer,
    OrdinalEncoder,
)


def _as_object(frame: Any) -> Any:
    """Convert boolean extension blocks to an imputer-compatible object block."""
    return frame.astype(object)


def make_preprocessing_pipeline(
    frame: pd.DataFrame,
    target_col: Optional[str] = None,
    scale_numeric: bool = True,
    scale_method: str = "standard",
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    encode_method: str = "onehot",
) -> Pipeline:
    """Build a fitted-state-free sklearn preprocessing pipeline.

    The returned pipeline learns imputation, scaling, and category mappings
    only during ``fit``. It can therefore be composed with an estimator and
    serialized with standard sklearn tooling without leaking validation data.

    Args:
        frame: Representative training frame used to identify column roles.
        target_col: Optional target column to exclude from preprocessing.
        scale_numeric: Add a numeric scaler after imputation.
        scale_method: ``standard``, ``minmax``, ``robust``, or ``maxabs``.
        numeric_strategy: Strategy passed to ``SimpleImputer``.
        categorical_strategy: Strategy passed to ``SimpleImputer``.
        encode_method: ``onehot`` for nominal features or ``label`` for an
            explicit ordinal encoding with deterministic unknown handling.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if target_col is not None and target_col not in frame.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    if not isinstance(scale_numeric, bool):
        raise TypeError("scale_numeric must be a boolean")
    if not isinstance(scale_method, str):
        raise TypeError("scale_method must be a string")
    scale_method = scale_method.lower().strip()
    if scale_method not in {"standard", "minmax", "robust", "maxabs"}:
        raise ValueError("scale_method must be standard, minmax, robust, or maxabs")
    if not isinstance(encode_method, str):
        raise TypeError("encode_method must be a string")
    encode_method = encode_method.lower().strip()
    if encode_method not in {"onehot", "label"}:
        raise ValueError("encode_method must be onehot or label")

    feature_frame = frame.drop(columns=[target_col]) if target_col else frame
    numeric_columns = [
        column
        for column in feature_frame.select_dtypes(include=["number"]).columns
        if not pd.api.types.is_bool_dtype(feature_frame[column])
    ]
    boolean_columns = feature_frame.select_dtypes(include=["bool"]).columns.tolist()
    categorical_columns = [
        column
        for column in feature_frame.columns
        if column not in numeric_columns and column not in boolean_columns
    ]

    transformers = []
    if numeric_columns:
        numeric_steps = [("imputer", SimpleImputer(strategy=numeric_strategy))]
        if scale_numeric:
            scaler_types = {
                "standard": StandardScaler,
                "minmax": MinMaxScaler,
                "robust": RobustScaler,
                "maxabs": MaxAbsScaler,
            }
            numeric_steps.append(("scaler", scaler_types[scale_method]()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns or boolean_columns:
        categorical_columns = categorical_columns + boolean_columns
        if encode_method == "label":
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        else:
            encoder_options = {"handle_unknown": "ignore"}
            if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
                encoder_options["sparse_output"] = True
            else:
                encoder_options["sparse"] = True
            encoder = OneHotEncoder(**encoder_options)
        categorical_pipeline = Pipeline(
            [
                # scikit-learn's SimpleImputer rejects a bool-only block;
                # object conversion keeps booleans categorical and preserves
                # the missing-value policy for mixed or bool-only frames.
                (
                    "as_object",
                    FunctionTransformer(
                        _as_object,
                        validate=False,
                        feature_names_out="one-to-one",
                    ),
                ),
                ("imputer", SimpleImputer(strategy=categorical_strategy)),
                ("encoder", encoder),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("frame must contain at least one feature column")

    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=transformers,
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            )
        ]
    )


def make_model_pipeline(
    frame: pd.DataFrame,
    estimator: Any,
    target_col: Optional[str] = None,
    **preprocessing_kwargs: Any,
) -> Pipeline:
    """Compose AutoPrepML column-role inference with any sklearn estimator.

    Fit the resulting object on training rows only, then call ``predict`` or
    ``predict_proba`` on future rows. ``estimator`` is intentionally supplied
    by the caller so model selection remains independent from preprocessing.
    """
    if estimator is None or not hasattr(estimator, "fit"):
        raise TypeError("estimator must implement fit")
    pipeline = make_preprocessing_pipeline(
        frame,
        target_col=target_col,
        **preprocessing_kwargs,
    )
    return Pipeline(pipeline.steps + [("model", estimator)])
