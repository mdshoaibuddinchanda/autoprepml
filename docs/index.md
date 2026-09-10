# AutoPrepML Documentation

AutoPrepML is a Python framework for ML data readiness: inspection,
leakage-safe preprocessing, validation, provenance, and reproducible
transformation across tabular, text, time-series, graph, and image data.

## What is AutoPrepML?

AutoPrepML detects, cleans, and reports common data issues with minimal code. It is designed for data scientists and ML engineers who need a transparent path from raw data to model ready data.

## Key Features

- **Fitted preprocessing**: Learn transformations from training rows and reuse them safely.
- **Issue detection**: Identify missing values, outliers, and class imbalance.
- **Reports**: Export HTML or JSON reports with statistics and visualizations.
- **Configuration**: Use YAML or JSON files to make runs reproducible.
- **Command line interface**: Run repeatable batch jobs from a shell or CI system.
- **Large-data execution**: Process bounded chunks in parallel and stream results to storage.
- **Integrations**: Use local or fsspec storage adapters, experiment tracking, and scikit-learn model pipelines.
- **Normalization standards**: Apply train-only, modality-aware normalization for tabular, image, text, time-series, and graph data.
- **Quality gates**: The [root README testing section](https://github.com/mdshoaibuddinchanda/autoprepml/blob/main/README.md#testing) is the canonical source for the current baseline. CI runs the coverage suite on Python 3.9 through 3.14 and enforces lint, test, security, package, and documentation checks.

## Quick Start

### Installation

```bash
pip install autoprepml
```

### Python API

```python
import pandas as pd
from autoprepml import AutoPrepML

# Load your data
df = pd.read_csv('data.csv')

# Initialize and clean
prep = AutoPrepML(df)
clean_df, report = prep.clean(task='classification', target_col='label')

# Save report
prep.save_report('report.html')
```

### Command Line

```bash
autoprepml --input data.csv --output cleaned.csv --report report.html
```

## Features

### Detection

- **Missing values**: Identifies columns with missing data and calculates percentages
- **Outliers**: Uses Isolation Forest or Z-score methods
- **Class imbalance**: Detects imbalanced target variables for classification

### Cleaning

- **Imputation**: Median for numeric, mode for categorical
- **Scaling**: Standard, min-max, robust, or max-absolute scaling
- **Encoding**: One-hot encoding by default, with explicit label encoding for ordinal values
- **Balancing**: Oversampling or undersampling for imbalanced classes

### Reporting

- **JSON format**: Machine-readable logs and statistics
- **HTML format**: Structured reports with plots
- **Plots included**: Missing values, outliers, distributions, correlations

## Documentation Contents

- [Usage Guide](usage.md): Step by step usage patterns
- [API Reference](api_reference.md): Public classes and functions
- [Tutorials](tutorials.md): End to end examples
- [Normalization standards](normalization.md): Production rules for schemas, scaling, leakage prevention, and modality-specific processing
- [Getting started](getting_started.md): The canonical `DataPlan` workflow
- [DataPlan architecture](concepts/data_plan.md): Fit, transform, validate, and save semantics
- [Data contracts](concepts/contracts.md): Structured compatibility checks
- [Leakage prevention](concepts/leakage.md): Train-only fitting and resampling rules
- [Reproducibility](concepts/reproducibility.md): Fingerprints and repeatability metadata
- [Transformation lineage](concepts/lineage.md): Machine-readable provenance manifests
- [Migration to v1.5](migration_v1.5.md): Adopt the fitted DataPlan workflow
- [CLI reference](cli.md): Inspect, fit, validate, transform, and version commands
- [Limitations](limitations.md): Explicit product boundaries and artifact trust guidance

## Links

- [GitHub Repository](https://github.com/mdshoaibuddinchanda/autoprepml)
- [Issue Tracker](https://github.com/mdshoaibuddinchanda/autoprepml/issues)
- [PyPI Package](https://pypi.org/project/autoprepml/)
