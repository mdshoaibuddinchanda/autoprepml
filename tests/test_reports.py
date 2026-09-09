"""Tests for reporting module"""

from autoprepml.reports import (
    generate_json_report,
    generate_html_report,
    generate_universal_html_report,
)


def test_generate_json_report():
    report = {
        "timestamp": "2025-10-23",
        "original_shape": (4, 2),
        "cleaned_shape": (4, 2),
        "detection_results": {"missing_values": {}},
        "logs": [{"action": "test", "details": {}}],
    }
    json_str = generate_json_report(report)
    assert "timestamp" in json_str
    assert "original_shape" in json_str


def test_generate_html_report():
    report = {
        "timestamp": "2025-10-23",
        "original_shape": (4, 2),
        "cleaned_shape": (4, 2),
        "detection_results": {"missing_values": {}, "outliers": {"outlier_count": 0}},
        "logs": [{"action": "test"}],
    }
    html_str = generate_html_report(report)
    assert "<html>" in html_str or "<!DOCTYPE html>" in html_str
    assert "AutoPrepML Report" in html_str


def test_generate_html_with_plots():
    report = {
        "timestamp": "2025-10-23",
        "original_shape": (10, 3),
        "cleaned_shape": (10, 3),
        "detection_results": {
            "missing_values": {"col1": {"count": 2, "percent": 20.0}},
            "outliers": {"outlier_count": 1, "method": "iforest"},
        },
        "logs": [],
        "plots": {"missing_plot": "base64string", "outlier_plot": "base64string"},
    }
    html_str = generate_html_report(report)
    assert "data:image/png;base64" in html_str
    assert "base64string" in html_str


def test_universal_report_does_not_mutate_input():
    report = {"issues": {"missing": 2}}

    generate_universal_html_report(report)

    assert "timestamp" not in report


def test_json_report_does_not_replace_original_plots():
    report = {"plots": {"missing_plot": "secret-base64"}}

    generate_json_report(report)

    assert report["plots"]["missing_plot"] == "secret-base64"


def test_html_report_escapes_untrusted_values():
    """Column names and log values cannot inject markup into HTML reports."""
    report = {
        "timestamp": "2025-10-23",
        "original_shape": (1, 1),
        "cleaned_shape": None,
        "detection_results": {
            "missing_values": {
                "<script>alert(1)</script>": {
                    "count": 1,
                    "percent": 100.0,
                    "dtype": "object",
                }
            },
            "outliers": {"outlier_count": 0},
        },
        "logs": [{"action": "<img src=x onerror=alert(1)>"}],
    }

    html = generate_html_report(report)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;img" in html or r"\u003cimg" in html
