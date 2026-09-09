"""Leakage-safe normalization utilities for tabular and image data.

Normalization is a fitted operation.  Statistics must be learned from the
training partition and reused unchanged for validation, test, and production
records.  The classes in this module make that lifecycle explicit while
preserving pandas schemas and image channel semantics.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted


_TABULAR_METHODS = {"standard", "minmax", "robust", "maxabs"}
_IMAGE_METHODS = {"none", "zero_one", "minus_one_one", "standard"}


def _validate_method(method: str, supported: set[str], name: str) -> str:
    if not isinstance(method, str):
        raise TypeError(f"{name} must be a string")
    method = method.lower().strip()
    if method not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"Unknown {name} '{method}'. Choose one of: {choices}")
    return method


class TabularNormalizer(BaseEstimator, TransformerMixin):
    """Fit and apply a numeric scaler without changing other columns.

    The normalizer intentionally does not impute missing values.  Imputation
    belongs before scaling and should be fitted on the same training rows.
    Raise-on-missing behavior prevents a silent train/test mismatch.

    Args:
        method: ``standard`` (z-score), ``minmax`` (range), ``robust``
            (median/IQR), or ``maxabs``.
        columns: Numeric columns to scale.  If omitted, all non-boolean
            numeric columns present during ``fit`` are selected.
        feature_range: Output range for ``minmax``.
        quantile_range: Quantiles used by ``robust``.
        clip: Clip values outside the fitted range for ``minmax`` transforms.

    Example:
        >>> normalizer = TabularNormalizer(method="robust")
        >>> train_scaled = normalizer.fit_transform(train_frame)
        >>> test_scaled = normalizer.transform(test_frame)
    """

    def __init__(
        self,
        method: str = "standard",
        columns: Optional[Sequence[str]] = None,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        clip: bool = False,
    ) -> None:
        self.method = method
        self.columns = columns
        self.feature_range = feature_range
        self.quantile_range = quantile_range
        self.clip = clip

    def _make_scaler(self):
        method = _validate_method(self.method, _TABULAR_METHODS, "method")
        if not isinstance(self.clip, (bool, np.bool_)):
            raise TypeError("clip must be a boolean")
        if method == "standard":
            return StandardScaler()
        if method == "minmax":
            if (
                not isinstance(self.feature_range, (tuple, list))
                or len(self.feature_range) != 2
                or not np.isfinite(self.feature_range).all()
                or self.feature_range[0] >= self.feature_range[1]
            ):
                raise ValueError("feature_range must be two finite values in ascending order")
            return MinMaxScaler(feature_range=tuple(self.feature_range), clip=bool(self.clip))
        if method == "robust":
            if (
                not isinstance(self.quantile_range, (tuple, list))
                or len(self.quantile_range) != 2
                or not np.isfinite(self.quantile_range).all()
                or not 0 <= self.quantile_range[0] < self.quantile_range[1] <= 100
            ):
                raise ValueError("quantile_range must satisfy 0 <= low < high <= 100")
            return RobustScaler(quantile_range=tuple(self.quantile_range))
        return MaxAbsScaler()

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

    def _select_columns(self, frame: pd.DataFrame) -> list[str]:
        if self.columns is None:
            columns = [
                column
                for column in frame.columns
                if pd.api.types.is_numeric_dtype(frame[column])
                and not pd.api.types.is_bool_dtype(frame[column])
            ]
        else:
            columns = list(self.columns)
            if len(columns) != len(set(columns)):
                raise ValueError("columns must not contain duplicates")
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
            non_numeric = [
                column
                for column in columns
                if not pd.api.types.is_numeric_dtype(frame[column])
                or pd.api.types.is_bool_dtype(frame[column])
            ]
            if non_numeric:
                raise TypeError(f"Columns must be numeric and not boolean: {non_numeric}")
        if not columns:
            raise ValueError("No numeric columns available for normalization")
        return columns

    def _validate_values(self, frame: pd.DataFrame, columns: Iterable[str]) -> None:
        columns = list(columns)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Input is missing fitted columns: {missing}")
        if frame[columns].isna().any().any():
            missing_columns = frame[columns].columns[frame[columns].isna().any()].tolist()
            raise ValueError(
                "Normalize after fitting an imputer; missing values found in "
                f"columns: {missing_columns}"
            )
        values = frame[columns].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("Normalization input contains infinite values")

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TabularNormalizer":
        """Learn scaling statistics from one training frame."""
        self._validate_frame(X)
        self.columns_ = self._select_columns(X)
        self._validate_values(X, self.columns_)
        self.scaler_ = self._make_scaler()
        self.scaler_.fit(X[self.columns_].to_numpy(dtype=np.float64))
        self.n_features_in_ = len(self.columns_)
        self.feature_names_in_ = np.asarray(self.columns_, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted statistics and preserve index, columns, and extras."""
        check_is_fitted(self, ["columns_", "scaler_"])
        self._validate_frame(X)
        self._validate_values(X, self.columns_)
        result = X.copy()
        result.loc[:, self.columns_] = self.scaler_.transform(
            X[self.columns_].to_numpy(dtype=np.float64)
        )
        return result

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Restore fitted columns to their original units."""
        check_is_fitted(self, ["columns_", "scaler_"])
        self._validate_frame(X)
        missing = [column for column in self.columns_ if column not in X.columns]
        if missing:
            raise ValueError(f"Input is missing fitted columns: {missing}")
        result = X.copy()
        result.loc[:, self.columns_] = self.scaler_.inverse_transform(
            X[self.columns_].to_numpy(dtype=np.float64)
        )
        return result

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """Return fitted column names for sklearn-compatible composition."""
        check_is_fitted(self, ["columns_"])
        if input_features is not None:
            input_features = list(input_features)
            missing = [column for column in self.columns_ if column not in input_features]
            if missing:
                raise ValueError(f"Input features are missing fitted columns: {missing}")
        return np.asarray(self.columns_, dtype=object)


def _channel_values(
    values: Optional[Union[Sequence[float], float]],
    array: np.ndarray,
    channel_axis: Optional[int],
    name: str,
) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required for standard image normalization")
    result = np.asarray(values, dtype=np.float32)
    if result.ndim > 1 or result.size == 0:
        raise ValueError(f"{name} must be a scalar or one-dimensional sequence")
    if channel_axis is None:
        if result.size != 1:
            raise ValueError(f"{name} must be a scalar for grayscale images")
        return result.reshape(1)
    axis = channel_axis if channel_axis >= 0 else array.ndim + channel_axis
    if axis < 0 or axis >= array.ndim:
        raise ValueError("channel_axis is out of bounds")
    channels = array.shape[axis]
    if result.size not in (1, channels):
        raise ValueError(f"{name} has {result.size} values but the image has {channels} channels")
    return result


def _broadcast_channel_values(values: np.ndarray, array: np.ndarray, channel_axis: Optional[int]):
    if channel_axis is None or values.size == 1:
        return values
    axis = channel_axis if channel_axis >= 0 else array.ndim + channel_axis
    shape = [1] * array.ndim
    shape[axis] = values.size
    return values.reshape(shape)


def _to_unit_interval(images: np.ndarray) -> np.ndarray:
    if not np.issubdtype(images.dtype, np.number):
        raise TypeError("images must contain numeric pixel values")
    values = images.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("images contain non-finite pixel values")
    if values.size == 0:
        return values.copy()
    minimum, maximum = float(values.min()), float(values.max())
    if minimum < 0 or maximum > 255:
        raise ValueError("pixel values must be within the range [0, 255]")
    return values if maximum <= 1.0 else values / 255.0


def normalize_image_array(
    images: np.ndarray,
    mode: str = "zero_one",
    mean: Optional[Union[Sequence[float], float]] = None,
    std: Optional[Union[Sequence[float], float]] = None,
    channel_axis: Optional[int] = -1,
) -> np.ndarray:
    """Normalize image pixels using explicit, reproducible conventions.

    ``standard`` expects mean and standard deviation measured on the training
    set after conversion to the ``[0, 1]`` scale.  It never computes those
    statistics from the batch being transformed.
    """
    mode = _validate_method(mode, _IMAGE_METHODS, "image normalization mode")
    values = np.asarray(images)
    if values.ndim < 2:
        raise ValueError("images must have at least height and width dimensions")
    if mode == "none":
        return np.array(values, copy=True)
    unit = _to_unit_interval(values)
    if mode == "zero_one":
        return unit
    if mode == "minus_one_one":
        return unit * 2.0 - 1.0
    mean_values = _channel_values(mean, unit, channel_axis, "mean")
    std_values = _channel_values(std, unit, channel_axis, "std")
    if not np.isfinite(std_values).all() or np.any(std_values <= 0):
        raise ValueError("std must contain finite values greater than zero")
    mean_values = _broadcast_channel_values(mean_values, unit, channel_axis)
    std_values = _broadcast_channel_values(std_values, unit, channel_axis)
    return (unit - mean_values) / std_values


def denormalize_image_array(
    images: np.ndarray,
    mode: str = "zero_one",
    mean: Optional[Union[Sequence[float], float]] = None,
    std: Optional[Union[Sequence[float], float]] = None,
    channel_axis: Optional[int] = -1,
) -> np.ndarray:
    """Convert normalized image values back to uint8 pixels for persistence."""
    mode = _validate_method(mode, _IMAGE_METHODS, "image normalization mode")
    values = np.asarray(images, dtype=np.float32)
    if mode == "none":
        return np.array(values, copy=True)
    if mode == "zero_one":
        unit = values
    elif mode == "minus_one_one":
        unit = (values + 1.0) / 2.0
    else:
        mean_values = _channel_values(mean, values, channel_axis, "mean")
        std_values = _channel_values(std, values, channel_axis, "std")
        if np.any(std_values <= 0) or not np.isfinite(std_values).all():
            raise ValueError("std must contain finite values greater than zero")
        unit = values * _broadcast_channel_values(std_values, values, channel_axis)
        unit = unit + _broadcast_channel_values(mean_values, values, channel_axis)
    return np.rint(np.clip(unit, 0.0, 1.0) * 255).astype(np.uint8)


def fit_image_statistics(
    images: np.ndarray,
    channel_axis: Optional[int] = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and standard deviation on training images.

    The returned statistics are measured after conversion to ``[0, 1]`` and
    can be passed to :func:`normalize_image_array` with ``mode='standard'``.
    A constant channel is rejected because it cannot provide a meaningful
    standardized feature.
    """
    values = _to_unit_interval(np.asarray(images))
    if channel_axis is None:
        reduce_axes = tuple(range(values.ndim))
    else:
        axis = channel_axis if channel_axis >= 0 else values.ndim + channel_axis
        if axis < 0 or axis >= values.ndim:
            raise ValueError("channel_axis is out of bounds")
        reduce_axes = tuple(index for index in range(values.ndim) if index != axis)
    mean = values.mean(axis=reduce_axes)
    std = values.std(axis=reduce_axes)
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("training images contain a channel with zero or invalid variance")
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def fit_tabular_normalizer(
    frame: pd.DataFrame,
    method: str = "standard",
    columns: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> TabularNormalizer:
    """Fit and return a :class:`TabularNormalizer` for a training frame."""
    return TabularNormalizer(method=method, columns=columns, **kwargs).fit(frame)


__all__ = [
    "TabularNormalizer",
    "fit_tabular_normalizer",
    "normalize_image_array",
    "denormalize_image_array",
    "fit_image_statistics",
]
