# AutoPrepML Documentation

Welcome to AutoPrepML - an automated data preprocessing pipeline for machine learning.

## What is AutoPrepML?

AutoPrepML automatically detects, cleans, and reports common data issues with minimal code. It's designed for data scientists, ML engineers, and students who want to quickly move from raw CSV to model-ready data.

## Key Features

- ✨ **One-line data cleaning** - Transform messy datasets with a single function call
- 🔍 **Automatic issue detection** - Missing values, outliers, class imbalance
- 📊 **Visual reports** - HTML reports with embedded plots and statistics
- ⚙️ **Configurable** - YAML/JSON configuration for reproducible pipelines
- 🚀 **CLI support** - Command-line interface for batch processing
- 🧪 **Well-tested** - Comprehensive unit tests with 90%+ coverage

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

- **Missing values** - Identifies columns with missing data and calculates percentages
- **Outliers** - Uses Isolation Forest or Z-score methods
- **Class imbalance** - Detects imbalanced target variables for classification

### Cleaning

- **Imputation** - Median for numeric, mode for categorical
- **Scaling** - StandardScaler or MinMaxScaler
- **Encoding** - Label encoding or one-hot encoding
- **Balancing** - Oversampling or undersampling for imbalanced classes

### Reporting

- **JSON format** - Machine-readable logs and statistics
- **HTML format** - Beautiful visual reports with plots
- **Plots included** - Missing values, outliers, distributions, correlations

## Documentation Contents

- [Usage Guide](usage.md) - Step-by-step tutorials
- [API Reference](api_reference.md) - Complete function documentation
- [Tutorials](tutorials.md) - Real-world examples

## Links

- [GitHub Repository](https://github.com/yourusername/autoprepml)
- [Issue Tracker](https://github.com/yourusername/autoprepml/issues)
- [PyPI Package](https://pypi.org/project/autoprepml/)
