# AutoPrepML Documentation

AutoPrepML is a Python library for repeatable data quality assessment and preprocessing in machine learning workflows.

## What is AutoPrepML?

AutoPrepML detects, cleans, and reports common data issues with minimal code. It is designed for data scientists and ML engineers who need a transparent path from raw data to model ready data.

## Key Features

- **One call preprocessing**: Transform a dataset with a single library call.
- **Issue detection**: Identify missing values, outliers, and class imbalance.
- **Reports**: Export HTML or JSON reports with statistics and visualizations.
- **Configuration**: Use YAML or JSON files to make runs reproducible.
- **Command line interface**: Run repeatable batch jobs from a shell or CI system.
- **Quality gates**: The repository currently passes 323 tests with 88 percent local line coverage; CI enforces lint, test, security, package, and documentation checks.

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
- **Scaling**: StandardScaler or MinMaxScaler
- **Encoding**: Label encoding or one-hot encoding
- **Balancing**: Oversampling or undersampling for imbalanced classes

### Reporting

- **JSON format**: Machine-readable logs and statistics
- **HTML format**: Structured reports with plots
- **Plots included**: Missing values, outliers, distributions, correlations

## Documentation Contents

- [Usage Guide](usage.md): Step by step usage patterns
- [API Reference](api_reference.md): Public classes and functions
- [Tutorials](tutorials.md): End to end examples

## Links

- [GitHub Repository](https://github.com/mdshoaibuddinchanda/autoprepml)
- [Issue Tracker](https://github.com/mdshoaibuddinchanda/autoprepml/issues)
- [PyPI Package](https://pypi.org/project/autoprepml/)
