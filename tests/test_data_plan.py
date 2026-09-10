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
    PrepProtocol,
    fingerprint_dataframe,
)
from autoprepml.exceptions import ArtifactError, ContractError, NotFittedError
from autoprepml.exceptions import ConfigurationError


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


def test_fit_missing_target_raises_contract_error(training_frame):
    plan = DataPlan.infer(training_frame, target="label")

    with pytest.raises(ContractError, match="label"):
        plan.fit(training_frame.drop(columns="label"))


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

    assert isinstance(plan, PrepProtocol)
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


@pytest.mark.parametrize("mode", ["schema", "sampled", "full"])
def test_fingerprint_mode_is_preserved_through_fit_and_save_load(training_frame, mode, tmp_path):
    plan = DataPlan.infer(
        training_frame,
        target="label",
        fingerprint_mode=mode,
        fingerprint_sample_rows=2,
    ).fit(training_frame)

    assert plan.fingerprint_mode == mode
    assert plan.manifest()["dataset_fingerprint"]["mode"] == mode
    loaded = DataPlan.load(plan.save(tmp_path / f"{mode}.apml"))
    assert loaded.fingerprint_mode == mode
    assert loaded.fingerprint_sample_rows == 2


def test_incompatible_numeric_dtype_is_a_blocking_contract_error(training_frame):
    plan = DataPlan.infer(training_frame, target="label").fit(training_frame)
    future = pd.DataFrame({"age": ["not-a-number"], "country": ["GB"]})

    report = plan.validate(future)

    assert report.status == "FAIL"
    assert {issue.code for issue in report.issues} == {"incompatible_dtype"}


def test_numeric_text_dtype_variation_remains_compatible(training_frame):
    plan = DataPlan.infer(training_frame, target="label").fit(training_frame)
    future = pd.DataFrame({"age": ["100"], "country": ["GB"]})

    report = plan.validate(future)

    assert report.status == "WARN"
    assert {issue.code for issue in report.issues} == {"dtype_mismatch"}


def test_sparse_output_policy_avoids_dense_conversion(training_frame):
    plan = DataPlan.infer(
        training_frame,
        target="label",
        output_format="sparse",
    ).fit(training_frame)

    transformed = plan.transform(training_frame.drop(columns="label"))

    assert isinstance(transformed, pd.DataFrame)
    assert all(isinstance(dtype, pd.SparseDtype) for dtype in transformed.dtypes)


def test_dense_output_limit_is_enforced_for_sparse_pipeline(training_frame):
    sparse_training = pd.DataFrame(
        {
            "age": list(range(20)),
            "country": [f"country-{index}" for index in range(20)],
            "label": [0, 1] * 10,
        }
    )
    plan = DataPlan.infer(
        sparse_training,
        target="label",
        max_dense_elements=1,
    ).fit(sparse_training)

    with pytest.raises(MemoryError, match="max_dense_elements"):
        plan.transform(sparse_training.drop(columns="label"))


def test_numpy_output_policy_returns_array(training_frame):
    plan = DataPlan.infer(
        training_frame,
        target="label",
        output_format="numpy",
    ).fit(training_frame)

    transformed = plan.transform(training_frame.drop(columns="label"))

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(training_frame)


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


def test_data_plan_validates_constructor_and_inference_options(training_frame):
    contract = DataContract.infer(training_frame, target="label")
    for kwargs, error in [
        ({"fingerprint_mode": "bad"}, "fingerprint_mode"),
        ({"fingerprint_sample_rows": 0}, "fingerprint_sample_rows"),
        ({"output_format": "arrow"}, "output_format"),
        ({"max_dense_elements": 0}, "max_dense_elements"),
    ]:
        with pytest.raises(ValueError, match=error):
            DataPlan(contract, {}, **kwargs)
    with pytest.raises(TypeError, match="DataFrame"):
        DataPlan.infer([1, 2])
    with pytest.raises(TypeError, match="random_state"):
        DataPlan.infer(training_frame, random_state=True)
    with pytest.raises(ContractError, match="string"):
        DataPlan.infer(training_frame.rename(columns={"age": 1}), target="label")
    with pytest.raises(ValueError, match="mode"):
        DataPlan.infer(training_frame, fingerprint_mode="bad")


def test_data_plan_lineage_and_pipeline_configuration_branches(training_frame, monkeypatch):
    no_scale = DataPlan(
        DataContract.infer(training_frame, target="label"), {"cleaning": {"scale_method": None}}
    )
    assert no_scale._pipeline_kwargs()["scale_numeric"] is False
    assert [item["name"] for item in no_scale._build_lineage(training_frame)] == [
        "median_imputation",
        "onehot_encoding",
    ]
    with pytest.raises(ContractError, match="feature"):
        DataPlan.infer(pd.DataFrame({"label": [0, 1]}), target="label").fit(
            pd.DataFrame({"label": [0, 1]})
        )

    import autoprepml.data_plan as module

    monkeypatch.setattr(
        module,
        "make_preprocessing_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(TypeError("bad pipeline")),
    )
    with pytest.raises(ConfigurationError, match="Could not fit"):
        DataPlan.infer(training_frame, target="label").fit(training_frame)


def test_data_plan_fit_transform_and_resampling_guards(training_frame):
    plan = DataPlan.infer(training_frame, target="label").fit(training_frame)
    with pytest.raises(TypeError, match="DataFrame"):
        plan.transform([1])
    with pytest.raises(TypeError, match="DataFrame"):
        plan.fit_transform([1])
    with pytest.raises(ValueError, match="same number"):
        plan.fit_resample(training_frame.drop(columns="label"), [1])
    assert plan.fit_resample(training_frame.drop(columns="label"), [0, 1, 0, 1], "disabled")[
        1
    ].tolist() == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="none"):
        plan.fit_resample(training_frame.drop(columns="label"), [0, 1, 0, 1], "invalid")
    with pytest.raises(TypeError, match="DataFrame"):
        plan.fit_resample([1], [0])
    with pytest.raises(ContractError, match="dtype"):
        plan.validate(pd.DataFrame({"age": ["bad"], "country": ["GB"]})).raise_for_error()


@pytest.mark.parametrize("payload", [b"not an archive", b""])
def test_data_plan_load_rejects_invalid_archives(tmp_path, payload):
    path = tmp_path / "bad.apml"
    path.write_bytes(payload)
    with pytest.raises(ArtifactError, match="Invalid"):
        DataPlan.load(path)


def test_data_plan_load_rejects_unsupported_manifest_version(training_frame, tmp_path):
    path = (
        DataPlan.infer(training_frame, target="label")
        .fit(training_frame)
        .save(tmp_path / "plan.apml")
    )
    rewritten = tmp_path / "unsupported.apml"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(rewritten, "w") as target:
        manifest = json.loads(source.read("manifest.json"))
        manifest["artifact_format_version"] = "2.0"
        target.writestr("manifest.json", json.dumps(manifest))
        target.writestr("state.pkl", source.read("state.pkl"))
    with pytest.raises(ArtifactError, match="Unsupported"):
        DataPlan.load(rewritten)
