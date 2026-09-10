"""Tests for dependency-free experiment tracking."""

import json

import pandas as pd
import pytest

from autoprepml import DataPlan, ExperimentRunProtocol, ExperimentTrackerProtocol
from autoprepml.experiments import LocalExperimentTracker


def test_local_experiment_run_persists_metrics_and_artifacts(tmp_path):
    tracker = LocalExperimentTracker(tmp_path / "runs")
    artifact = tmp_path / "result.csv"
    pd.DataFrame({"value": [1, 2]}).to_csv(artifact, index=False)
    train = pd.DataFrame({"value": [1, 2, 3, 4], "label": [0, 1, 0, 1]})
    plan = DataPlan.infer(train, target="label").fit(train)

    with tracker.start_run("smoke", tags={"source": "test"}) as run:
        assert isinstance(run, ExperimentRunProtocol)
        run.log_params({"chunksize": 2})
        run.log_metrics({"accuracy": 0.75})
        copied = run.log_artifact(artifact)
        plan_artifact = run.log_plan(plan)

    manifest = tracker.list_runs()[0]
    assert manifest["status"] == "finished"
    assert manifest["metrics"]["accuracy"] == 0.75
    assert copied.exists()
    assert plan_artifact.exists()
    assert "data-plan.json" in manifest["artifacts"]
    assert json.loads((copied.parent.parent / "run.json").read_text())["name"] == "smoke"


def test_local_experiment_run_marks_failures(tmp_path):
    tracker = LocalExperimentTracker(tmp_path / "runs")
    with pytest.raises(RuntimeError):
        with tracker.start_run("failed") as run:
            run.log_metrics({"partial": 1})
            raise RuntimeError("boom")

    assert tracker.list_runs()[0]["status"] == "failed"


def test_experiment_tracker_validates_inputs(tmp_path):
    tracker = LocalExperimentTracker(tmp_path / "runs")
    assert isinstance(tracker, ExperimentTrackerProtocol)
    train = pd.DataFrame({"value": [1, 2, 3, 4], "label": [0, 1, 0, 1]})
    plan = DataPlan.infer(train, target="label").fit(train)
    with pytest.raises(ValueError):
        tracker.start_run("")
    with tracker.start_run("validation") as run:
        with pytest.raises(TypeError):
            run.log_metrics({"bad": "value"})
        with pytest.raises(FileNotFoundError):
            run.log_artifact(tmp_path / "missing.txt")
        with pytest.raises(TypeError):
            run.log_plan(object())
        with pytest.raises(ValueError):
            run.log_plan(plan, artifact_name="nested/plan.json")
