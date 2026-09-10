"""Tests for the command-line interface."""

from pathlib import Path

import pandas as pd
import pytest

from autoprepml import cli


def _write_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "department": ["HR", "IT", "HR", "Finance"],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(path, index=False)


def test_help_exits_successfully(capsys):
    """The parser exposes help through the regular CLI contract."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
    assert "Automated Data Preprocessing Pipeline" in capsys.readouterr().out


def test_output_is_required_unless_detect_only(tmp_path):
    """Detection-only mode can inspect a file without creating an output file."""
    input_path = tmp_path / "input.csv"
    _write_csv(input_path)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--input", str(input_path)])
    assert exc_info.value.code == 2


def test_detect_only_without_output(tmp_path, capsys):
    """Detection-only mode returns a success status."""
    input_path = tmp_path / "input.csv"
    _write_csv(input_path)

    assert cli.main(["--input", str(input_path), "--detect-only"]) == 0
    assert "Detection complete" in capsys.readouterr().out


def test_missing_input_returns_usage_error(tmp_path, capsys):
    """Missing input files are reported without a traceback."""
    result = cli.main(
        ["--input", str(tmp_path / "missing.csv"), "--output", str(tmp_path / "out.csv")]
    )

    assert result == 2
    assert "Input file not found" in capsys.readouterr().err


def test_input_and_report_extensions_are_validated(tmp_path, capsys):
    """The CLI rejects unsupported data and report formats early."""
    input_path = tmp_path / "input.txt"
    input_path.write_text("not,csv", encoding="utf-8")
    output_path = tmp_path / "out.csv"

    assert cli.main(["--input", str(input_path), "--output", str(output_path)]) == 2
    assert "must be a CSV" in capsys.readouterr().err

    input_path = tmp_path / "input.csv"
    _write_csv(input_path)
    assert (
        cli.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--report",
                str(tmp_path / "report.txt"),
            ]
        )
        == 2
    )
    assert "Report file must be" in capsys.readouterr().err


def test_csv_load_failure_returns_error(tmp_path, monkeypatch, capsys):
    """Read failures return a non-zero status and useful message."""
    input_path = tmp_path / "input.csv"
    input_path.write_text("a,b\n1,2\n", encoding="utf-8")

    def fail_read_csv(_path):
        raise ValueError("invalid CSV")

    monkeypatch.setattr(cli.pd, "read_csv", fail_read_csv)

    result = cli.main(["--input", str(input_path), "--output", str(tmp_path / "out.csv")])

    assert result == 1
    assert "Error loading CSV: invalid CSV" in capsys.readouterr().err


def test_processing_failure_returns_error(tmp_path, monkeypatch, capsys):
    """Unexpected pipeline errors are converted to a stable CLI status."""
    input_path = tmp_path / "input.csv"
    _write_csv(input_path)

    class FailingPrep:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("pipeline failed")

    monkeypatch.setattr(cli, "AutoPrepML", FailingPrep)

    result = cli.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "out.csv"),
            "--verbose",
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "Error during preprocessing: pipeline failed" in captured.err
    assert "RuntimeError" in captured.err


def test_successful_cleaning_writes_output_and_report(tmp_path, capsys):
    """A normal invocation produces both requested artifacts."""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "nested" / "cleaned.csv"
    report_path = tmp_path / "nested" / "report.json"
    _write_csv(input_path)

    result = cli.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--no-plots",
        ]
    )

    assert result == 0
    assert output_path.exists()
    assert report_path.exists()
    assert "completed successfully" in capsys.readouterr().out


def test_modern_version_command(capsys):
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == cli.__version__


def test_modern_inspect_fit_validate_and_transform_commands(tmp_path, capsys):
    input_path = tmp_path / "input.csv"
    features_path = tmp_path / "features.csv"
    transformed_path = tmp_path / "transformed.csv"
    plan_path = tmp_path / "plan.apml"
    _write_csv(input_path)
    pd.read_csv(input_path).drop(columns="label").to_csv(features_path, index=False)

    assert cli.main(["inspect", str(input_path), "--target", "label"]) == 0
    inspection = capsys.readouterr().out
    assert '"feature_columns"' in inspection

    assert (
        cli.main(
            [
                "fit",
                str(input_path),
                "--target",
                "label",
                "--output",
                str(plan_path),
            ]
        )
        == 0
    )
    assert plan_path.exists()
    capsys.readouterr()

    assert cli.main(["validate", str(features_path), "--plan", str(plan_path)]) == 0
    assert '"status": "PASS"' in capsys.readouterr().out

    assert (
        cli.main(
            [
                "transform",
                str(features_path),
                "--plan",
                str(plan_path),
                "--output",
                str(transformed_path),
            ]
        )
        == 0
    )
    assert transformed_path.exists()
    assert not pd.read_csv(transformed_path).empty


def test_modern_commands_report_input_errors(tmp_path, capsys):
    assert cli.main(["inspect", str(tmp_path / "missing.csv")]) == 1
    assert "does not exist" in capsys.readouterr().err
