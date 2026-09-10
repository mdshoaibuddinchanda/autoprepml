"""Tests for structured data-readiness reports."""

import pandas as pd
import pytest

from autoprepml import DataPlan, DataReadinessReport, assess_data_readiness


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "country": ["GB", "US", "GB", "US"],
            "label": [0, 1, 0, 1],
        }
    )


def test_fitted_readiness_report_is_pass_with_compatible_future(frame):
    plan = DataPlan.infer(frame, target="label").fit(frame)

    report = assess_data_readiness(plan, frame.drop(columns="label"))

    assert isinstance(report, DataReadinessReport)
    assert report.status == "PASS"
    assert report.ready
    assert {check.name for check in report.checks} == {
        "schema_integrity",
        "leakage_safety",
        "reproducibility",
        "dataset_identity",
        "lineage",
    }


def test_readiness_report_surfaces_schema_warning(frame):
    plan = DataPlan.infer(frame, target="label").fit(frame)
    future = pd.DataFrame({"age": [20], "country": ["FR"]})

    report = plan.readiness_report(future)

    assert report.status == "WARN"
    assert not report.ready
    assert report.to_dict()["checks"][0]["status"] == "WARN"


def test_unfitted_readiness_report_is_not_ready(frame):
    plan = DataPlan.infer(frame, target="label")

    report = assess_data_readiness(plan)

    assert report.status == "FAIL"
    assert not report.ready


def test_readiness_rejects_non_plan():
    with pytest.raises(TypeError, match="DataPlan-compatible"):
        assess_data_readiness(object())
