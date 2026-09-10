"""Tests for the reproducible OpenML smoke-test helper."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import smoke_openml_adult
from scripts import benchmark_v15


def test_load_adult_sample_is_deterministic(monkeypatch):
    """The loader joins the target and applies deterministic sampling."""
    frame = pd.DataFrame({"feature": range(5)})
    target = pd.Series(["a", "b", "a", "b", "a"], name="class")
    monkeypatch.setattr(
        smoke_openml_adult,
        "fetch_openml",
        lambda *_args, **_kwargs: SimpleNamespace(data=frame, target=target),
    )

    sampled = smoke_openml_adult.load_adult_sample(rows=3, seed=7)

    assert sampled.shape == (3, 2)
    assert sampled.columns.tolist() == ["feature", "target"]
    pd.testing.assert_frame_equal(
        sampled,
        smoke_openml_adult.load_adult_sample(rows=3, seed=7),
    )


def test_load_adult_sample_validates_rows(monkeypatch):
    """Invalid sample sizes fail before an unnecessary data operation."""
    monkeypatch.setattr(
        smoke_openml_adult,
        "fetch_openml",
        lambda *_args, **_kwargs: SimpleNamespace(
            data=pd.DataFrame({"feature": [1]}), target=pd.Series([1])
        ),
    )

    with pytest.raises(ValueError, match="rows must be positive"):
        smoke_openml_adult.load_adult_sample(rows=0)


def test_run_smoke_returns_aggregate_results(monkeypatch):
    """Smoke output contains shapes and issue counts, never raw records."""
    frame = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    monkeypatch.setattr(smoke_openml_adult, "load_adult_sample", lambda **_kwargs: frame)

    class FakePrep:
        def __init__(self, data, **_kwargs):
            assert data is frame

        def detect(self, **_kwargs):
            return {
                "missing_values": {"feature": {"count": 1}},
                "outliers": {"outlier_count": 2},
                "class_imbalance": {"is_imbalanced": True},
            }

        def clean(self, **_kwargs):
            return frame.copy(), {}

    monkeypatch.setattr(smoke_openml_adult, "AutoPrepML", FakePrep)

    assert smoke_openml_adult.run_smoke(rows=2) == {
        "dataset": "openml:adult-v2",
        "seed": 42,
        "input_shape": [2, 2],
        "output_shape": [2, 2],
        "missing_columns": 1,
        "outlier_count": 2,
        "class_imbalanced": True,
    }


def test_main_writes_json_output(monkeypatch, tmp_path, capsys):
    """The helper supports both stdout and nested JSON output files."""
    result = {"dataset": "openml:adult-v2", "output_shape": [10, 2]}
    monkeypatch.setattr(smoke_openml_adult, "run_smoke", lambda **_kwargs: result)

    assert smoke_openml_adult.main([]) == 0
    assert "openml:adult-v2" in capsys.readouterr().out

    output_path = tmp_path / "nested" / "smoke.json"
    assert (
        smoke_openml_adult.main(["--rows", "25", "--seed", "9", "--output", str(output_path)]) == 0
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == result


def test_benchmark_is_deterministic_and_uses_aggregate_results():
    first = benchmark_v15.run_benchmark(rows=20, chunksize=7, n_jobs=1, seed=9)
    second = benchmark_v15.run_benchmark(rows=20, chunksize=7, n_jobs=1, seed=9)

    assert first["rows"] == 20
    assert first["transformed_shape"] == [20, 3]
    assert first["chunk_shape"] == [20, 4]
    assert first["seed"] == second["seed"]
    assert first["transformed_shape"] == second["transformed_shape"]


def test_benchmark_validates_resource_options():
    with pytest.raises(ValueError, match="chunksize"):
        benchmark_v15.run_benchmark(rows=2, chunksize=0)
    with pytest.raises(ValueError, match="n_jobs"):
        benchmark_v15.run_benchmark(rows=2, n_jobs=0)
