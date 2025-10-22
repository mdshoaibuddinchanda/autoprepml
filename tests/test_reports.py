"""Tests for reporting module"""
from autoprepml.reports import generate_json_report, generate_html_report


def test_generate_json_report():
    report = {
        'timestamp': '2025-10-23',
        'original_shape': (4, 2),
        'cleaned_shape': (4, 2),
        'detection_results': {'missing_values': {}},
        'logs': [{'action': 'test', 'details': {}}]
    }
    json_str = generate_json_report(report)
    assert 'timestamp' in json_str
    assert 'original_shape' in json_str


def test_generate_html_report():
    report = {
        'timestamp': '2025-10-23',
        'original_shape': (4, 2),
        'cleaned_shape': (4, 2),
        'detection_results': {
            'missing_values': {},
            'outliers': {'outlier_count': 0}
        },
        'logs': [{'action': 'test'}]
    }
    html_str = generate_html_report(report)
    assert '<html>' in html_str or '<!DOCTYPE html>' in html_str
    assert 'AutoPrepML Report' in html_str


def test_generate_html_with_plots():
    report = {
        'timestamp': '2025-10-23',
        'original_shape': (10, 3),
        'cleaned_shape': (10, 3),
        'detection_results': {
            'missing_values': {'col1': {'count': 2, 'percent': 20.0}},
            'outliers': {'outlier_count': 1, 'method': 'iforest'}
        },
        'logs': [],
        'plots': {
            'missing_plot': 'base64string',
            'outlier_plot': 'base64string'
        }
    }
    html_str = generate_html_report(report)
    assert 'data:image/png;base64' in html_str
    assert 'base64string' in html_str

