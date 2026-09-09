"""Tests for fitted, modality-aware normalization utilities."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from autoprepml import (
    TabularNormalizer,
    denormalize_image_array,
    fit_image_statistics,
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
