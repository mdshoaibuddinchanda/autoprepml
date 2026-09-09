"""Scikit-learn pipeline integration for leakage-safe model workflows."""

import inspect
from typing import Any, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessing_pipeline(
    frame: pd.DataFrame,
    target_col: Optional[str] = None,
    scale_numeric: bool = True,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> Pipeline:
    """Build a fitted-state-free sklearn preprocessing pipeline.

    The returned pipeline learns imputation, scaling, and category mappings
    only during ``fit``. It can therefore be composed with an estimator and
    serialized with standard sklearn tooling without leaking validation data.

    Args:
        frame: Representative training frame used to identify column roles.
        target_col: Optional target column to exclude from preprocessing.
        scale_numeric: Add ``StandardScaler`` after numeric imputation.
        numeric_strategy: Strategy passed to ``SimpleImputer``.
        categorical_strategy: Strategy passed to ``SimpleImputer``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if target_col is not None and target_col not in frame.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    if not isinstance(scale_numeric, bool):
        raise TypeError("scale_numeric must be a boolean")

    feature_frame = frame.drop(columns=[target_col]) if target_col else frame
    numeric_columns = feature_frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in feature_frame.columns if column not in numeric_columns
    ]

    transformers = []
    if numeric_columns:
        numeric_steps = [("imputer", SimpleImputer(strategy=numeric_strategy))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        encoder_options = {"handle_unknown": "ignore"}
        if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
            encoder_options["sparse_output"] = True
        else:
            encoder_options["sparse"] = True
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy=categorical_strategy)),
                ("encoder", OneHotEncoder(**encoder_options)),
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
