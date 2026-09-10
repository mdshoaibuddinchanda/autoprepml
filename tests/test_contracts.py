"""Contract validation edge cases and export behavior."""

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from autoprepml.contracts import (
    ColumnContract,
    DataContract,
    ValidationIssue,
    ValidationReport,
    _numeric_conversion_is_safe,
)
from autoprepml.exceptions import ContractError, IntegrationError, ValidationError


def test_contract_inference_and_serialization_are_stable():
    frame = pd.DataFrame(
        {
            "category": pd.Series(["b", "a", None], dtype="category"),
            "flag": [True, False, True],
            "target": [0, 1, 0],
        }
    )
    contract = DataContract.infer(frame, target="target", task="classification")
    assert contract.feature_columns == ("category", "flag")
    payload = contract.to_dict()
    assert payload["target"] == "target"
    assert payload["schema_fingerprint"]
    assert payload["columns"][0]["allowed_categories"] == ["a", "b"]
    with pytest.raises(ContractError, match="Target"):
        DataContract.infer(frame, target="missing")
    with pytest.raises(ValueError, match="task"):
        DataContract.infer(frame, task="multilabel")
    with pytest.raises(ValueError, match="max_categories"):
        DataContract.infer(frame, max_categories=0)


def test_contract_reports_missing_optional_order_nulls_ranges_and_uniqueness():
    contract = DataContract(
        columns=(
            ColumnContract("optional", "int64", required=False),
            ColumnContract("bounded", "float64", minimum=0, maximum=1, nullable=False),
            ColumnContract("key", "object", unique=True),
            ColumnContract("target", "int64", semantic_role="target"),
        ),
        target="target",
        require_column_order=True,
    )
    frame = pd.DataFrame({"key": ["x", "x"], "bounded": [-1.0, 2.0]})
    report = contract.validate(frame, mode="compatible")
    codes = {issue.code for issue in report.issues}
    assert report.status == "FAIL"
    assert {
        "missing_optional_column",
        "below_minimum",
        "above_maximum",
        "unique_constraint",
    }.issubset(codes)
    strict = contract.validate(frame, mode="strict", include_target=True)
    assert any(issue.code == "missing_required_column" for issue in strict.issues)
    assert report.to_dict()["compatible"] is False
    with pytest.raises(ContractError):
        report.raise_for_error()


def test_contract_validation_input_and_dtype_branches():
    contract = DataContract(
        columns=(ColumnContract("number", "float64"), ColumnContract("text", "object")),
        require_column_order=True,
    )
    with pytest.raises(ValidationError, match="DataFrame"):
        contract.validate([1])
    with pytest.raises(ValidationError, match="mode"):
        contract.validate(pd.DataFrame({"number": [1.0], "text": ["x"]}), mode="bad")
    report = contract.validate(pd.DataFrame({"text": ["x"], "number": ["not-a-number"]}))
    assert report.status == "FAIL"
    assert {issue.code for issue in report.issues} >= {
        "incompatible_dtype",
        "column_order_mismatch",
    }
    assert _numeric_conversion_is_safe(pd.Series(["1", "2"]), "float64")
    assert not _numeric_conversion_is_safe(pd.Series(["bad"]), "float64")
    assert _numeric_conversion_is_safe(pd.Series(["bad"]), "object")


def test_validation_report_status_and_safe_values():
    empty = ValidationReport("compatible")
    assert empty.status == "PASS" and empty.compatible
    warning = ValidationReport("compatible", (ValidationIssue("w", "WARN", None, "warning"),))
    assert warning.status == "WARN" and warning.compatible
    assert ColumnContract(
        "x", "object", allowed_categories=(pd.Timestamp("2024-01-01"),)
    ).to_dict()["allowed_categories"] == ["2024-01-01T00:00:00"]


def test_contract_pandera_export_is_actionable_when_dependency_missing(monkeypatch):
    contract = DataContract((ColumnContract("x", "float64", minimum=0),))
    monkeypatch.setitem(sys.modules, "pandera", None)
    with pytest.raises(IntegrationError, match="Pandera"):
        contract.to_pandera()


def test_contract_pandera_export_uses_checks(monkeypatch):
    calls = []

    class Check:
        @staticmethod
        def ge(value):
            calls.append(("ge", value))
            return ("ge", value)

        @staticmethod
        def le(value):
            calls.append(("le", value))
            return ("le", value)

        @staticmethod
        def isin(value):
            calls.append(("isin", value))
            return ("isin", value)

    class Column:
        def __init__(self, *args, **kwargs):
            calls.append(("column", args, kwargs))

    class DataFrameSchema:
        def __init__(self, columns, **kwargs):
            self.columns, self.kwargs = columns, kwargs

    fake = SimpleNamespace(Check=Check, Column=Column, DataFrameSchema=DataFrameSchema)
    monkeypatch.setitem(sys.modules, "pandera", fake)
    contract = DataContract(
        (
            ColumnContract(
                "x", "float64", minimum=0, maximum=1, allowed_categories=(0, 1), unique=True
            ),
        ),
        require_column_order=True,
    )
    schema = contract.to_pandera()
    assert schema.kwargs["strict"] is True
    assert {item[0] for item in calls} >= {"ge", "le", "isin", "column"}
