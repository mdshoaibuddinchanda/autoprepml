"""Reporting utilities for AutoPrepML"""
import json
from jinja2 import Template
from typing import Dict, Any
from datetime import datetime

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AutoPrepML Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        .info-box {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .plot-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
        }
        pre {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 12px;
        }
        .stat {
            display: inline-block;
            margin: 10px 20px 10px 0;
        }
        .stat-label {
            font-weight: bold;
            color: #7f8c8d;
        }
        .stat-value {
            color: #2c3e50;
            font-size: 18px;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge-success { background: #2ecc71; color: white; }
        .badge-warning { background: #f39c12; color: white; }
        .badge-danger { background: #e74c3c; color: white; }
    </style>
</head>
<body>
    <h1>🔬 AutoPrepML Report</h1>
    
    <div class="info-box">
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        <div class="stat">
            <span class="stat-label">Original Shape:</span>
            <span class="stat-value">{{ original_shape }}</span>
        </div>
        {% if cleaned_shape %}
        <div class="stat">
            <span class="stat-label">Cleaned Shape:</span>
            <span class="stat-value">{{ cleaned_shape }}</span>
        </div>
        {% endif %}
    </div>
    
    <h2>📊 Detection Results</h2>
    <div class="info-box">
        <h3>Missing Values</h3>
        {% if detection_results.missing_values %}
            <ul>
            {% for col, stats in detection_results.missing_values.items() %}
                <li><strong>{{ col }}:</strong> {{ stats.count }} missing ({{ stats.percent }}%)
                    <span class="badge badge-warning">{{ stats.dtype }}</span>
                </li>
            {% endfor %}
            </ul>
        {% else %}
            <p class="badge badge-success">✓ No missing values detected</p>
        {% endif %}
        
        <h3>Outliers</h3>
        {% if detection_results.outliers.outlier_count > 0 %}
            <p><span class="badge badge-danger">{{ detection_results.outliers.outlier_count }} outliers detected</span></p>
            <p>Method: {{ detection_results.outliers.method }}</p>
        {% else %}
            <p class="badge badge-success">✓ No significant outliers detected</p>
        {% endif %}
        
        {% if detection_results.class_imbalance %}
        <h3>Class Imbalance</h3>
        {% if detection_results.class_imbalance.is_imbalanced %}
            <p><span class="badge badge-warning">⚠ Dataset is imbalanced</span></p>
            <p><strong>Minority class:</strong> {{ detection_results.class_imbalance.minority_class }} 
               ({{ (detection_results.class_imbalance.minority_proportion * 100) | round(2) }}%)</p>
            <p><strong>Imbalance ratio:</strong> {{ detection_results.class_imbalance.imbalance_ratio | round(2) }}:1</p>
        {% else %}
            <p class="badge badge-success">✓ Classes are balanced</p>
        {% endif %}
        {% endif %}
    </div>
    
    {% if plots %}
    <h2>📈 Visualizations</h2>
    
    <div class="plot-container">
        <h3>Missing Values</h3>
        <img src="data:image/png;base64,{{ plots.missing_plot }}" alt="Missing Values Plot">
    </div>
    
    <div class="plot-container">
        <h3>Outlier Detection</h3>
        <img src="data:image/png;base64,{{ plots.outlier_plot }}" alt="Outlier Plot">
    </div>
    
    <div class="plot-container">
        <h3>Feature Distributions</h3>
        <img src="data:image/png;base64,{{ plots.distribution_plot }}" alt="Distribution Plot">
    </div>
    
    <div class="plot-container">
        <h3>Correlation Heatmap</h3>
        <img src="data:image/png;base64,{{ plots.correlation_plot }}" alt="Correlation Plot">
    </div>
    {% endif %}
    
    <h2>📝 Processing Log</h2>
    <div class="info-box">
        <pre>{{ logs | tojson(indent=2) }}</pre>
    </div>
</body>
</html>
"""


def generate_json_report(report: Dict[str, Any]) -> str:
    """Generate JSON report from report dictionary.
    
    Args:
        report: Report data dictionary
        
    Returns:
        JSON string
    """
    # Remove base64 plots from JSON (too large)
    report_copy = report.copy()
    if 'plots' in report_copy:
        report_copy['plots'] = {k: '<base64_image_data>' for k in report_copy['plots'].keys()}
    
    return json.dumps(report_copy, indent=2, default=str)


def generate_html_report(report: Dict[str, Any]) -> str:
    """Generate HTML report from report dictionary.
    
    Args:
        report: Report data dictionary
        
    Returns:
        HTML string
    """
    # Return universal template for non-AutoPrepML reports early
    if 'detection_results' not in report:
        return generate_universal_html_report(report)
    # Use detailed template for AutoPrepML
    tpl = Template(HTML_TEMPLATE)
    return tpl.render(**report)


def generate_universal_html_report(report: Dict[str, Any]) -> str:
    """Generate universal HTML report for Text/TimeSeries/Graph preprocessing.
    
    Args:
        report: Report data dictionary
        
    Returns:
        HTML string
    """
    UNIVERSAL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AutoPrepML Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        .info-box {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        pre {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 12px;
            max-height: 400px;
        }
        .stat {
            display: inline-block;
            margin: 10px 20px 10px 0;
        }
        .stat-label {
            font-weight: bold;
            color: #7f8c8d;
        }
        .stat-value {
            color: #2c3e50;
            font-size: 18px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
    </style>
</head>
<body>
    <h1>🔬 AutoPrepML Report</h1>
    
    <div class="info-box">
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        {% if original_shape %}
        <div class="stat">
            <span class="stat-label">Original Shape:</span>
            <span class="stat-value">{{ original_shape }}</span>
        </div>
        {% endif %}
        {% if current_shape %}
        <div class="stat">
            <span class="stat-label">Current Shape:</span>
            <span class="stat-value">{{ current_shape }}</span>
        </div>
        {% endif %}
        {% if original_nodes_shape %}
        <div class="stat">
            <span class="stat-label">Original Nodes:</span>
            <span class="stat-value">{{ original_nodes_shape }}</span>
        </div>
        {% endif %}
        {% if current_nodes_shape %}
        <div class="stat">
            <span class="stat-label">Current Nodes:</span>
            <span class="stat-value">{{ current_nodes_shape }}</span>
        </div>
        {% endif %}
        {% if original_edges_shape %}
        <div class="stat">
            <span class="stat-label">Original Edges:</span>
            <span class="stat-value">{{ original_edges_shape }}</span>
        </div>
        {% endif %}
        {% if current_edges_shape %}
        <div class="stat">
            <span class="stat-label">Current Edges:</span>
            <span class="stat-value">{{ current_edges_shape }}</span>
        </div>
        {% endif %}
    </div>
    
    {% if issues %}
    <h2>📊 Detected Issues</h2>
    <div class="info-box">
        <table>
            <tr>
                <th>Issue Type</th>
                <th>Count/Value</th>
            </tr>
            {% for key, value in issues.items() %}
            <tr>
                <td>{{ key | replace('_', ' ') | title }}</td>
                <td>{{ value }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
    
    {% if graph_stats %}
    <h2>📈 Graph Statistics</h2>
    <div class="info-box">
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            {% for key, value in graph_stats.items() %}
            <tr>
                <td>{{ key | replace('_', ' ') | title }}</td>
                <td>{{ value }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
    
    {% if logs %}
    <h2>📝 Processing Log</h2>
    <div class="info-box">
        <pre>{{ logs | tojson(indent=2) }}</pre>
    </div>
    {% endif %}
    
    <div class="info-box">
        <p style="text-align: center; color: #7f8c8d;">
            Generated by AutoPrepML - AI-Assisted Data Preprocessing Pipeline
        </p>
    </div>
</body>
</html>
    """
    
    report['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tpl = Template(UNIVERSAL_TEMPLATE)
    return tpl.render(**report)

