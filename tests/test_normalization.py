"""Tests for fitted, modality-aware normalization utilities."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from autoprepml import (
    TabularNormalizer,
    denormalize_image_array,
    fit_image_statistics,
    fit_tabular_normalizer,
    normalize_image_array,
)


def test_tabular_normalizer_fits_train_statistics_and_preserves_schema():
    train = pd.DataFrame(
        {"amount": [10.0, 20.0, 30.0], "flag": [True, False, True], "label": [0, 1, 0]}
    )
    test = pd.DataFrame({"amount": [40.0], "flag": [False], "label": [1]}, index=[9])

    normalizer = TabularNormalizer(columns=["amount"])
    train_scaled = normalizer.fit_transform(train)
    test_scaled = normalizer.transform(test)

    assert list(train_scaled.columns) == list(train.columns)
    assert test_scaled.index.tolist() == [9]
    assert bool(test_scaled.loc[9, "flag"]) is False
    assert test_scaled.loc[9, "label"] == 1
    assert np.isclose(train_scaled["amount"].mean(), 0.0)
    assert np.isclose(test_scaled.loc[9, "amount"], 2.449489743, atol=1e-6)


@pytest.mark.parametrize("method", ["standard", "minmax", "robust", "maxabs"])
def test_tabular_normalizer_supports_scaling_methods(method):
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    result = TabularNormalizer(method=method).fit_transform(frame)
    restored = TabularNormalizer(method=method).fit(frame).inverse_transform(result)
    np.testing.assert_allclose(restored["x"], frame["x"])


def test_tabular_normalizer_requires_imputation_before_fit():
    with pytest.raises(ValueError, match="imputer"):
        TabularNormalizer().fit(pd.DataFrame({"x": [1.0, np.nan]}))


def test_tabular_normalizer_rejects_unfitted_transform():
    with pytest.raises(NotFittedError):
        TabularNormalizer().transform(pd.DataFrame({"x": [1.0]}))


def test_image_zero_one_and_minus_one_one_modes():
    pixels = np.array([[[0, 128, 255]]], dtype=np.uint8)
    zero_one = normalize_image_array(pixels, mode="zero_one", channel_axis=-1)
    minus_one = normalize_image_array(pixels, mode="minus_one_one", channel_axis=-1)
    np.testing.assert_allclose(zero_one, [[[0.0, 128 / 255, 1.0]]])
    np.testing.assert_allclose(minus_one, [[[-1.0, 2 * 128 / 255 - 1, 1.0]]], atol=1e-6)


def test_image_standard_mode_uses_explicit_channel_statistics():
    pixels = np.array([[[0, 128, 255]]], dtype=np.uint8)
    standardized = normalize_image_array(
        pixels,
        mode="standard",
        mean=[0.0, 0.5, 1.0],
        std=[1.0, 0.25, 0.5],
        channel_axis=-1,
    )
    np.testing.assert_allclose(standardized, [[[0.0, (128 / 255 - 0.5) / 0.25, 0.0]]], atol=1e-6)


def test_image_statistics_are_computed_per_channel():
    pixels = np.array([[[0, 128, 255]], [[32, 64, 96]]], dtype=np.uint8)
    mean, std = fit_image_statistics(pixels, channel_axis=-1)
    np.testing.assert_allclose(mean, [16 / 255, 96 / 255, 175.5 / 255], atol=1e-6)
    assert np.all(std > 0)


def test_image_statistics_reject_constant_channel():
    with pytest.raises(ValueError, match="zero"):
        fit_image_statistics(np.zeros((2, 2, 3), dtype=np.uint8), channel_axis=-1)


def test_image_standard_mode_round_trips_for_persistence():
    pixels = np.array([[[0, 128, 255]]], dtype=np.uint8)
    standardized = normalize_image_array(
        pixels, mode="standard", mean=[0.1, 0.2, 0.3], std=[0.2, 0.3, 0.4], channel_axis=-1
    )
    restored = denormalize_image_array(
        standardized,
        mode="standard",
        mean=[0.1, 0.2, 0.3],
        std=[0.2, 0.3, 0.4],
        channel_axis=-1,
    )
    np.testing.assert_allclose(restored, pixels, atol=1)


def test_image_normalization_rejects_invalid_statistics():
    with pytest.raises(ValueError, match="std"):
        normalize_image_array(
            np.zeros((2, 2, 3), dtype=np.uint8),
            mode="standard",
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 0.0, 1.0],
        )


@pytest.mark.parametrize("method", [None, "unknown", " STANDARD "])
def test_tabular_normalizer_validates_method(method):
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    if method == " STANDARD ":
        assert fit_tabular_normalizer(frame, method=method).columns_ == ["x"]
    elif method is None:
        with pytest.raises(TypeError, match="method"):
            TabularNormalizer(method=method).fit(frame)
    else:
        with pytest.raises(ValueError, match="Unknown method"):
            TabularNormalizer(method=method).fit(frame)


def test_tabular_normalizer_selects_numeric_columns_and_rejects_invalid_selection():
    frame = pd.DataFrame({"number": [1.0, 2.0], "flag": [True, False], "text": ["a", "b"]})
    fitted = TabularNormalizer().fit(frame)
    assert fitted.columns_ == ["number"]
    with pytest.raises(ValueError, match="duplicates"):
        TabularNormalizer(columns=["number", "number"]).fit(frame)
    with pytest.raises(ValueError, match="not found"):
        TabularNormalizer(columns=["missing"]).fit(frame)
    with pytest.raises(TypeError, match="numeric"):
        TabularNormalizer(columns=["text"]).fit(frame)
    with pytest.raises(ValueError, match="No numeric"):
        TabularNormalizer().fit(frame[["flag", "text"]])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"feature_range": (1.0, 1.0)}, "feature_range"),
        ({"feature_range": (0.0, np.inf)}, "feature_range"),
        ({"feature_range": (0.0,)}, "feature_range"),
        ({"quantile_range": (75.0, 25.0)}, "quantile_range"),
        ({"quantile_range": (-1.0, 50.0)}, "quantile_range"),
        ({"clip": "yes"}, "clip"),
    ],
)
def test_tabular_normalizer_validates_scaler_options(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        TabularNormalizer(method="minmax" if "feature_range" in kwargs else "robust", **kwargs).fit(
            pd.DataFrame({"x": [1.0, 2.0]})
        )


def test_tabular_normalizer_validates_transform_inputs_and_feature_names():
    fitted = TabularNormalizer(columns=["x"]).fit(pd.DataFrame({"x": [1.0, 2.0]}))
    with pytest.raises(TypeError, match="DataFrame"):
        fitted.transform(np.array([[1.0]]))
    with pytest.raises(ValueError, match="missing fitted"):
        fitted.transform(pd.DataFrame({"y": [1.0]}))
    with pytest.raises(ValueError, match="infinite"):
        fitted.transform(pd.DataFrame({"x": [np.inf]}))
    with pytest.raises(ValueError, match="missing fitted"):
        fitted.inverse_transform(pd.DataFrame({"y": [1.0]}))
    assert fitted.get_feature_names_out().tolist() == ["x"]
    assert fitted.get_feature_names_out(["x", "other"]).tolist() == ["x"]
    with pytest.raises(ValueError, match="missing fitted"):
        fitted.get_feature_names_out(["other"])


@pytest.mark.parametrize(
    "images, kwargs, error",
    [
        (np.array([1]), {}, "at least"),
        (np.array([["x"]]), {}, "numeric"),
        (np.array([[np.nan]]), {}, "non-finite"),
        (np.array([[-1.0]]), {}, "range"),
        (
            np.zeros((2, 2, 3)),
            {"mode": "standard", "mean": [0.0, 0.0], "std": [1.0, 1.0]},
            "values",
        ),
        (
            np.zeros((2, 2, 3)),
            {"mode": "standard", "mean": [0, 0, 0], "std": [1, 1, 1], "channel_axis": 4},
            "bounds",
        ),
    ],
)
def test_image_normalization_validates_shape_values_and_channels(images, kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        normalize_image_array(images, **kwargs)


def test_image_none_mode_copies_and_grayscale_standard_uses_scalars():
    pixels = np.array([[0, 255]], dtype=np.uint8)
    copied = normalize_image_array(pixels, mode="none", channel_axis=None)
    copied[0, 0] = 99
    assert pixels[0, 0] == 0
    standardized = normalize_image_array(
        pixels, mode="standard", mean=0.5, std=0.25, channel_axis=None
    )
    restored = denormalize_image_array(
        standardized, mode="standard", mean=0.5, std=0.25, channel_axis=None
    )
    np.testing.assert_array_equal(restored, pixels)


def test_image_denormalization_modes_and_statistics_validation():
    values = np.array([[-1.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        denormalize_image_array(values, mode="minus_one_one"), np.array([[0, 255]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        denormalize_image_array(np.array([[0.0, 2.0]]), mode="zero_one"),
        np.array([[0, 255]], dtype=np.uint8),
    )
    copied = denormalize_image_array(values, mode="none")
    assert copied.dtype == np.float32
    with pytest.raises(ValueError, match="mean is required"):
        normalize_image_array(np.zeros((2, 2)), mode="standard", std=1.0, channel_axis=None)
    with pytest.raises(ValueError, match="std"):
        denormalize_image_array(
            np.zeros((2, 2)), mode="standard", mean=0.0, std=0.0, channel_axis=None
        )


def test_fit_image_statistics_grayscale_and_invalid_axis():
    pixels = np.array([[0, 255], [64, 128]], dtype=np.uint8)
    mean, std = fit_image_statistics(pixels, channel_axis=None)
    assert mean.shape == ()
    assert float(std) > 0
    with pytest.raises(ValueError, match="bounds"):
        fit_image_statistics(pixels, channel_axis=2)
