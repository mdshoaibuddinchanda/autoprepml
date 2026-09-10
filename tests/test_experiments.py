"""Tests for dependency-free experiment tracking."""

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pytest

from autoprepml import DataPlan, ExperimentRunProtocol, ExperimentTrackerProtocol
from autoprepml.experiments import ExperimentRun, LocalExperimentTracker, MLflowExperimentTracker


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


def test_local_run_manual_cancellation_and_serialization(tmp_path):
    tracker = LocalExperimentTracker(tmp_path / "runs")
    run = tracker.start_run(tags={1: 2})
    assert run.tags == {"1": "2"}
    run.log_params({"timestamp": object()}, ignored=True)
    run.end("cancelled")
    assert tracker.list_runs()[0]["status"] == "cancelled"
    with pytest.raises(ValueError, match="finished"):
        run.end("unknown")
    assert isinstance(run, ExperimentRun)


def test_mlflow_adapter_is_optional_but_has_compatible_protocol(monkeypatch, tmp_path):
    calls = []

    class FakeContext:
        def __enter__(self):
            calls.append("enter")
            return object()

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append(("exit", exc_type is not None))
            return False

    @contextmanager
    def start_run(**kwargs):
        calls.append(("start_run", kwargs))
        yield object()

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: calls.append(("uri", uri)),
        set_experiment=lambda name: calls.append(("experiment", name)),
        start_run=start_run,
        log_params=lambda params: calls.append(("params", params)),
        log_metrics=lambda metrics: calls.append(("metrics", metrics)),
        log_artifact=lambda path, **kwargs: calls.append(("artifact", path, kwargs)),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    tracker = MLflowExperimentTracker("demo", tracking_uri="memory://mlruns")
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("artifact", encoding="utf-8")
    plan = DataPlan.infer(pd.DataFrame({"x": [1, 2], "label": [0, 1]}), target="label").fit(
        pd.DataFrame({"x": [1, 2], "label": [0, 1]})
    )
    with tracker.start_run("trial", tags={"kind": "test"}) as run:
        run.log_params({"batch": 2}, ignored=True)
        run.log_metrics({"score": 0.5}, ignored=True)
        run.log_artifact(artifact, artifact_name="ignored.txt")
        run.log_plan(plan)
    assert any(item[0] == "start_run" for item in calls if isinstance(item, tuple))
    assert any(item[0] == "uri" for item in calls if isinstance(item, tuple))
    assert any(item[0] == "artifact" for item in calls if isinstance(item, tuple))

    with tracker.start_run("failure") as run:
        run.log_params({})
        with pytest.raises(TypeError, match="manifest"):
            run.log_plan(object())


def test_mlflow_import_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(ImportError, match="MLflow"):
        MLflowExperimentTracker()
