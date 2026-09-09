"""Tests for dependency-free experiment tracking."""

import json

import pandas as pd
import pytest

from autoprepml.experiments import LocalExperimentTracker


def test_local_experiment_run_persists_metrics_and_artifacts(tmp_path):
    tracker = LocalExperimentTracker(tmp_path / "runs")
    artifact = tmp_path / "result.csv"
    pd.DataFrame({"value": [1, 2]}).to_csv(artifact, index=False)

    with tracker.start_run("smoke", tags={"source": "test"}) as run:
        run.log_params({"chunksize": 2})
        run.log_metrics({"accuracy": 0.75})
        copied = run.log_artifact(artifact)

    manifest = tracker.list_runs()[0]
    assert manifest["status"] == "finished"
    assert manifest["metrics"]["accuracy"] == 0.75
    assert copied.exists()
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
    with pytest.raises(ValueError):
        tracker.start_run("")
    with tracker.start_run("validation") as run:
        with pytest.raises(TypeError):
            run.log_metrics({"bad": "value"})
        with pytest.raises(FileNotFoundError):
            run.log_artifact(tmp_path / "missing.txt")
