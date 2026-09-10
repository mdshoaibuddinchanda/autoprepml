"""Tests for the canonical fitted DataPlan workflow."""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from autoprepml import (
    ColumnContract,
    DataContract,
    DataPlan,
    fingerprint_dataframe,
)
from autoprepml.exceptions import ArtifactError, ContractError, NotFittedError


@pytest.fixture
def training_frame():
    return pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0],
            "country": ["GB", "US", "GB", "US"],
            "label": [0, 1, 0, 1],
        }
    )


def test_plan_state_and_not_fitted_guard(training_frame):
    plan = DataPlan.infer(training_frame, target="label", task="classification")

    assert plan.state == "UNFITTED"
    with pytest.raises(NotFittedError):
        plan.transform(training_frame)

    plan.fit(training_frame)
    assert plan.state == "FITTED"


def test_transform_is_ordered_unknown_category_safe_and_does_not_change_state(training_frame):
    plan = DataPlan.infer(training_frame, target="label", task="classification").fit(training_frame)
    before = plan.state_fingerprint
    future = pd.DataFrame({"country": ["FR"], "age": [100.0]})

    transformed = plan.transform(future)

    assert transformed.index.tolist() == [0]
    assert transformed.columns.tolist() == ["age", "country_GB", "country_US"]
    assert np.isfinite(transformed.to_numpy()).all()
    assert plan.state_fingerprint == before
    assert plan.validate(future).status == "WARN"


def test_transform_ignores_target_and_extra_columns(training_frame):
    plan = DataPlan.infer(training_frame, target="label").fit(training_frame)
    with_target = training_frame.assign(extra=1)
    without_target = with_target.drop(columns="label")

    pd.testing.assert_frame_equal(plan.transform(with_target), plan.transform(without_target))


def test_report_matches_inspection_before_fit_and_manifest_after_fit(training_frame):
    plan = DataPlan.infer(training_frame, target="label")

    assert plan.report() == plan.inspect()
    plan.fit(training_frame)

    assert plan.report() == plan.manifest()


def test_contract_accepts_pandas_string_dtype_as_text_compatibility():
    contract = DataContract(
        columns=(ColumnContract("country", "object", allowed_categories=("GB", "US")),)
    )
    frame = pd.DataFrame({"country": pd.Series(["GB", "FR"], dtype="string")})

    report = contract.validate(frame, mode="compatible")

    assert report.status == "WARN"
    assert {issue.code for issue in report.issues} == {"unknown_category"}


def test_contract_reports_compatible_and_strict_findings():
    contract = DataContract(
        columns=(
            ColumnContract("age", "int64", nullable=False),
            ColumnContract("country", "object", allowed_categories=("GB", "US")),
        )
    )
    frame = pd.DataFrame({"country": ["FR"], "age": [1], "extra": [True]})

    compatible = contract.validate(frame, mode="compatible")
    strict = contract.validate(frame, mode="strict")

    assert compatible.status == "WARN"
    assert strict.status == "FAIL"
    assert {issue.code for issue in strict.issues} == {"unexpected_column", "unknown_category"}


def test_contract_rejects_duplicate_columns():
    frame = pd.DataFrame([[1, 2]], columns=["value", "value"])
    with pytest.raises(ContractError, match="unique"):
        DataContract.infer(frame)


def test_fingerprints_are_deterministic_and_mode_aware(training_frame):
    first = fingerprint_dataframe(training_frame, target="label")
    second = fingerprint_dataframe(training_frame.copy(), target="label")
    sampled = fingerprint_dataframe(training_frame, mode="sampled", target="label")
    schema_only = fingerprint_dataframe(training_frame, mode="schema", target="label")

    assert first == second
    assert first.content is not None
    assert sampled.content is not None
    assert schema_only.content is None
    assert first.schema == sampled.schema == schema_only.schema


def test_fit_resample_is_explicit_and_does_not_change_transform(training_frame):
    plan = DataPlan.infer(training_frame, target="label").fit(training_frame)
    before = plan.state_fingerprint
    features = training_frame.drop(columns="label")

    sampled_features, sampled_target = plan.fit_resample(
        features, training_frame["label"], "oversample"
    )

    assert len(sampled_features) == len(sampled_target)
    assert len(sampled_features) >= len(features)
    assert plan.state_fingerprint == before
    assert len(plan.transform(features)) == len(features)


def test_plan_save_load_round_trip_and_manifest(training_frame, tmp_path):
    plan = DataPlan.infer(training_frame, target="label", task="classification").fit(training_frame)
    path = plan.save(tmp_path / "dataset.apml")
    loaded = DataPlan.load(path)

    pd.testing.assert_frame_equal(plan.transform(training_frame), loaded.transform(training_frame))
    manifest = loaded.manifest()
    assert manifest["artifact_format_version"] == "1.0"
    assert manifest["target"] == "label"
    assert manifest["transformations"]
    assert "state_sha256" in json.loads(zipfile.ZipFile(path).read("manifest.json"))


def test_corrupt_plan_is_rejected(training_frame, tmp_path):
    path = (
        DataPlan.infer(training_frame, target="label")
        .fit(training_frame)
        .save(tmp_path / "dataset.apml")
    )
    corrupt_path = tmp_path / "corrupt.apml"
    with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(corrupt_path, "w") as target:
        target.writestr("manifest.json", source.read("manifest.json"))
        target.writestr("state.pkl", b"corrupt")

    with pytest.raises(ArtifactError, match="checksum"):
        DataPlan.load(corrupt_path)


def test_label_encoding_handles_unknown_categories(training_frame):
    plan = DataPlan.infer(
        training_frame,
        target="label",
        config={"cleaning": {"encode_method": "label"}},
    ).fit(training_frame)

    transformed = plan.transform(pd.DataFrame({"age": [25.0], "country": ["FR"]}))

    assert transformed.shape == (1, 2)
    assert transformed["country"].iloc[0] == -1
