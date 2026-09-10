"""Boundary and fallback tests for deterministic dataset fingerprints."""

import numpy as np
import pandas as pd
import pytest

from autoprepml.fingerprints import (
    DatasetFingerprint,
    _canonical_dtype,
    _json_default,
    fingerprint_dataframe,
    schema_fingerprint,
)


def test_fingerprint_validates_frame_mode_and_sample_size():
    frame = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(TypeError, match="DataFrame"):
        schema_fingerprint([1, 2])
    with pytest.raises(TypeError, match="DataFrame"):
        fingerprint_dataframe([1, 2])
    with pytest.raises(ValueError, match="mode"):
        fingerprint_dataframe(frame, mode="partial")
    for sample_rows in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="sample_rows"):
            fingerprint_dataframe(frame, sample_rows=sample_rows)
    with pytest.raises(ValueError, match="mode"):
        fingerprint_dataframe(frame, mode="partial", sample_rows=1)


def test_fingerprint_handles_unhashable_object_values_and_sampling_boundaries():
    frame = pd.DataFrame({"payload": [[1, 2], [3], [4, 5], [6]]})
    full = fingerprint_dataframe(frame, mode="full")
    sampled = fingerprint_dataframe(frame, mode="sampled", sample_rows=3)
    sampled_one = fingerprint_dataframe(frame, mode="sampled", sample_rows=1)
    assert full.content and sampled.content and sampled_one.content
    assert full.content != sampled.content
    assert full.feature_count == 1
    assert DatasetFingerprint(**full.to_dict()).to_dict() == full.to_dict()


def test_fingerprint_json_conversion_and_dtype_canonicalization():
    assert _json_default(np.int64(3)) == 3
    assert _json_default(np.float32(1.5)) == pytest.approx(1.5)
    assert _json_default(np.bool_(True)) is True
    assert _json_default(pd.Timestamp("2024-01-01")) == "2024-01-01T00:00:00"
    assert _json_default(pd.Timedelta(days=1)) == "P1DT0H0M0S"
    assert _json_default(object()).startswith("<object")
    assert _canonical_dtype("object") == _canonical_dtype("string") == "string"


def test_schema_fingerprint_includes_target_and_order():
    left = pd.DataFrame({"a": [1], "b": ["x"]})
    right = pd.DataFrame({"b": ["x"], "a": [1]})
    assert schema_fingerprint(left, target="a") != schema_fingerprint(left, target="b")
    assert schema_fingerprint(left) != schema_fingerprint(right)
