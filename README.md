<div align="center">
  <img src="assets/logo.png" alt="AutoPrepML Logo" width="180"/>
  
  # AutoPrepML
  
  **Multi-Modal Data Preprocessing Pipeline**
  
  [![PyPI version](https://img.shields.io/badge/pypi-v1.3.0-blue.svg)](https://pypi.org/project/autoprepml/)
  [![CI](https://github.com/mdshoaibuddinchanda/autoprepml/workflows/CI/badge.svg)](https://github.com/mdshoaibuddinchanda/autoprepml/actions)
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Tests](https://img.shields.io/badge/tests-323%20passing-brightgreen.svg)](tests/)
  [![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen.svg)](https://codecov.io/gh/mdshoaibuddinchanda/autoprepml)
  
  <p align="center">
    <a href="#quick-start-guide">Quick Start</a> |
    <a href="#installation">Installation</a> |
    <a href="#examples-directory">Examples</a> |
    <a href="#documentation">Documentation</a> |
    <a href="#contributing">Contributing</a>
  </p>
</div>

<br>

> **A practical preprocessing library for tabular, text, time series, graph, and image data.**

A comprehensive Python library that automatically detects, cleans, and transforms data across multiple modalities. Built for real-world ML pipelines with one-line automation and detailed reporting.

The processing flow is straightforward:

1. Load raw data from a supported source.
2. Detect quality issues and record the findings.
3. Apply configured cleaning and feature transformations.
4. Export the processed data and a reproducible report.

## Features

### Core Features
- **Multi-Modal Support**: Works with 5 different data types out of the box
- **Automatic Issue Detection**: Missing values, outliers, duplicates, anomalies
- **Visual Reports**: HTML reports with embedded plots and statistics
- **Highly Configurable**: YAML/JSON configuration for reproducibility
- **CLI + Python API**: Use from command line or Python scripts
- **Production readiness baseline**: 323 tests passing, 88% local line coverage, and blocking CI/CD gates

### Advanced Features (v1.3.0)
- **AutoEDA**: Automated exploratory data analysis with insights generation
- **AutoFeatureEngine**: Intelligent feature engineering with 8 creation methods
- **Interactive Dashboards**: Plotly visualizations and Streamlit app generation
- **Enhanced LLM Assistant**: Column renaming, documentation, quality analysis

### Previous Releases
- **LLM Integration**: AI-powered suggestions with GPT-4, Claude, Gemini, Ollama (v1.2.0)
- **Image Preprocessing**: Automatic image cleaning, resizing, normalization (v1.2.0)
- **Advanced Imputation**: KNN and Iterative (MICE) imputation methods (v1.1.0)
- **SMOTE Balancing**: Synthetic minority oversampling for imbalanced data (v1.1.0)

## Quick Navigation

| Section | Description |
|---------|-------------|
| [Supported Data Types](#supported-data-types) | Overview of tabular, text, time series, graph, and image data |
| [Installation](#installation) | Install from source or PyPI |
| [Quick Start](#quick-start-guide) | A short tutorial for each data type |
| [Version 1.3.0 Features](#v130-new-features) | AutoEDA, feature engineering, dashboards |
| [Advanced Features](docs/ADVANCED_FEATURES.md) | KNN and iterative imputation, and SMOTE |
| [LLM Integration](docs/LLM_CONFIGURATION.md) | Model assisted suggestions from multiple providers |
| [Dynamic LLM Configuration](docs/DYNAMIC_LLM_CONFIGURATION.md) | Configure supported models at runtime |
| [CLI Configuration](docs/QUICK_START_CLI.md) | Manage provider credentials with autoprepml-config |
| [CLI Reference](#command-line-usage) | Command line options and examples |
| [Examples](#examples-directory) | Working demo scripts with outputs |
| [Full API](#complete-feature-reference) | Function and class reference |
| [Configuration](#configuration) | YAML and JSON configuration for reproducibility |
| [Testing](#testing) | Run tests and inspect coverage |
| [Development](#development-setup) | Contribution and development guidance |

## Supported Data Types

| Data Type | Module | Use Cases | Status |
|-----------|--------|-----------|--------|
| **Tabular** | `AutoPrepML` | Classification, regression, and general machine learning | Ready |
| **Text and NLP** | `TextPrepML` | Sentiment analysis, topic modeling, and classification | Ready |
| **Time series** | `TimeSeriesPrepML` | Forecasting, trend analysis, and anomaly detection | Ready |
| **Graph** | `GraphPrepML` | Social networks, recommendation systems, and link prediction | Ready |
| **Image** | `ImagePrepML` | Computer vision, image classification, and object detection | Ready |

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Option 1: Install from PyPI

```bash
# Basic installation
pip install autoprepml

# With LLM support (AI-powered suggestions)
pip install autoprepml[llm]

# With all optional dependencies
pip install autoprepml[all]
```

### Option 2: Install from Source (Latest Development Version)

```bash
git clone https://github.com/mdshoaibuddinchanda/autoprepml.git
cd autoprepml
pip install -e .

# Or with LLM support
pip install -e ".[llm]"
```

### Option 3: With Development Tools

```bash
pip install -e ".[dev]"  # Includes pytest, coverage, linting tools
pip install -e ".[all]"  # Everything (dev + llm + docs)
```

### Configure LLM Support (Optional)

After installing with LLM support, configure your API keys:

```bash
# Interactive configuration wizard
autoprepml-config

# Or set a specific provider
autoprepml-config --set openai
autoprepml-config --set anthropic
autoprepml-config --set google

# Use Ollama for a local model; no API key is needed
# Just install Ollama from https://ollama.ai
```

See [LLM Configuration Guide](docs/LLM_CONFIGURATION.md) for detailed instructions.

### Verify Installation

```bash
python -c "from autoprepml import AutoPrepML; print('Installation successful')"
autoprepml --help
```

## v1.3.0 New Features

### AutoEDA: Automated Exploratory Data Analysis

Comprehensive automated EDA with insights generation:

```python
from autoprepml import AutoEDA

# Initialize with your DataFrame
eda = AutoEDA(df)

# Run full analysis
results = eda.analyze(
    include_correlations=True,
    include_distributions=True,
    include_outliers=True,
    generate_insights=True
)

# Generate interactive HTML report
eda.generate_report('eda_report.html')

# Export results to JSON
eda.to_json('eda_results.json')

# Access specific analysis results
print(results['insights'])
print(results['correlations']['high_correlations'])
print(results['outliers']['iqr_outliers'])
```

**Features:**
- Statistical summaries (mean, std, quartiles, skewness, kurtosis)
- Missing value analysis with percentages
- Correlation matrix with high correlation detection (>0.7)
- Distribution analysis (skewness, kurtosis, quartiles)
- Outlier detection (IQR and Z-score methods)
- Categorical analysis (cardinality, mode, value counts)
- Automated insights generation in natural language
- Interactive HTML reports with visualizations
- JSON export for programmatic access

### AutoFeatureEngine: Intelligent Feature Engineering

Create powerful features automatically with 8 different methods:

```python
from autoprepml import AutoFeatureEngine, auto_feature_engineering

# Initialize with your DataFrame
fe = AutoFeatureEngine(df, target_column='target')

# 1. Polynomial features (degree 2 or 3)
df_poly = fe.create_polynomial_features(columns=['age', 'income'], degree=2)

# 2. Interaction features (multiplication)
df_interact = fe.create_interactions(columns=['age', 'income', 'score'])

# 3. Ratio features (division-based)
df_ratio = fe.create_ratio_features(columns=['income', 'loan_amount'])

# 4. Binned features (discretization)
df_binned = fe.create_binned_features(columns=['age'], n_bins=5, strategy='quantile')

# 5. Aggregation features (sum, mean, std, min, max)
df_agg = fe.create_aggregation_features(columns=['col1', 'col2', 'col3'])

# 6. Datetime features (year, month, day, hour, quarter)
df_date = fe.create_datetime_features(columns=['date'], features=['year', 'month', 'day'])

# 7. Feature selection (keep best k features)
df_selected = fe.select_features(method='mutual_info', k=10, task='classification')

# 8. Feature importance ranking
importance = fe.get_feature_importance(task='classification')
print(importance)

# Quick auto feature engineering
df_enhanced = auto_feature_engineering(
    df,
    numeric_columns=['age', 'income', 'score'],
    target_column='target',
    select_top_k=15
)
```

**Methods:**
- `create_polynomial_features()`: Polynomial & interaction terms
- `create_interactions()`: Pairwise multiplications
- `create_ratio_features()`: Division-based features
- `create_binned_features()`: Discretization (uniform, quantile, kmeans)
- `create_aggregation_features()`: Row-wise aggregations
- `create_datetime_features()`: Extract temporal components
- `select_features()`: Mutual info or F-test selection
- `get_feature_importance()`: Rank features by importance

### Interactive Dashboards: Visualization & Streamlit

Create interactive dashboards with Plotly and generate full Streamlit apps:

```python
from autoprepml import InteractiveDashboard, create_plotly_dashboard, generate_streamlit_app

# Initialize dashboard
dashboard = InteractiveDashboard(df)

# Create comprehensive Plotly dashboard
dashboard.create_dashboard(
    title="My Data Dashboard",
    output_path="dashboard.html"
)

# Create correlation heatmap
dashboard.create_correlation_heatmap(output_path="correlation.html")

# Create missing data visualization
dashboard.create_missing_data_plot(output_path="missing_data.html")

# Generate full Streamlit app
dashboard.generate_streamlit_app(output_path="app.py")

# Run the generated Streamlit app
# streamlit run app.py

# Or use convenience functions
create_plotly_dashboard(df, title="Quick Dashboard", output_path="quick_dash.html")
generate_streamlit_app(df, output_path="my_app.py")
```

**Features:**
- Multi-subplot Plotly dashboards (histograms, box plots, scatter, bar charts)
- Interactive correlation heatmaps
- Missing data visualizations
- Full Streamlit app generation with:
  - File upload functionality
  - Overview tab (shape, dtypes, memory)
  - EDA tab (distributions, correlations, missing values)
  - Preprocessing tab (missing value handling, encoding)
  - Feature engineering tab (interactions, polynomial, binning)

### Enhanced LLM Assistant: Intelligent Data Cleaning

Advanced AI-powered assistance for data preprocessing:

```python
from autoprepml import LLMSuggestor, suggest_column_rename, generate_data_documentation

# Initialize LLM suggestor
suggestor = LLMSuggestor(provider='openai')  # or 'anthropic', 'google', 'ollama'

# 1. Suggest better column names
new_names = suggestor.suggest_all_column_renames(df)
df_renamed = df.rename(columns=new_names)

# 2. Get specific column rename suggestion
new_name = suggest_column_rename(df, column='col1')
print(f"Suggested name: {new_name}")

# 3. Explain data quality issues in natural language
explanation = suggestor.explain_data_quality_issues(df)
print(explanation)

# 4. Generate comprehensive data documentation
documentation = generate_data_documentation(df)
with open('data_docs.md', 'w') as f:
    f.write(documentation)

# 5. Get preprocessing pipeline recommendations
pipeline = suggestor.suggest_preprocessing_pipeline(df, task='classification')
print(pipeline)

# 6. Get specific fix suggestions
fix = suggestor.suggest_fix(df, column='age', issue_type='missing')
print(fix)
```

**New LLM Capabilities:**
- `suggest_column_rename()`: AI-powered intelligent column naming
- `suggest_all_column_renames()`: Batch rename all columns
- `explain_data_quality_issues()`: Natural language quality explanations
- `generate_data_documentation()`: Auto-generate Markdown documentation
- `suggest_preprocessing_pipeline()`: Complete pipeline recommendations
- Works with OpenAI (GPT-4), Anthropic (Claude), Google (Gemini), and Ollama (local)

### New Dependencies

v1.3.0 adds optional dependencies for visualization:

```bash
# Install with visualization support
pip install autoprepml[viz]

# Or install manually
pip install plotly streamlit
```

## Quick Start Guide

### Step 1: Import the Library

```python
import pandas as pd
from autoprepml import AutoPrepML, TextPrepML, TimeSeriesPrepML, GraphPrepML
```

### Step 2: Choose Your Data Type

#### **Tabular Data** (CSV, Excel, JSON)

```python
# Load your data
df = pd.read_csv('data.csv')

# Initialize and clean
prep = AutoPrepML(df)
clean_df, target = prep.clean(task='classification', target_col='label')

# Generate report
prep.save_report('report.html')
```

#### **With AI-Powered Suggestions** (v1.2.0+)

```python
# Enable LLM support for AI suggestions
prep = AutoPrepML(df, enable_llm=True, llm_provider='openai')

# Get AI analysis of your dataset
analysis = prep.analyze_with_llm(task='classification', target_col='label')
print(analysis)

# Get suggestions for missing values
suggestions = prep.get_llm_suggestions(column='age', issue_type='missing')
print(suggestions)

# Get feature engineering ideas
features = prep.get_feature_suggestions(task='classification', target_col='label')
for feature in features:
    print(f"  {feature}")

# Clean with advanced methods
clean_df, report = prep.clean(
    task='classification',
    target_col='label',
    use_advanced=True,
    imputation_method='knn',  # or 'iterative'
    balance_method='smote'     # Advanced class balancing
)
```

#### **Text/NLP Data** (Reviews, Documents, Tweets)

```python
# Load text data
df = pd.read_csv('reviews.csv')

# Initialize with text column
prep = TextPrepML(df, text_column='review_text')

# Clean text
prep.clean_text(lowercase=True, remove_urls=True, remove_html=True)
prep.remove_stopwords()
prep.extract_features()

# Get cleaned data
cleaned_df = prep.df
```

#### **Time Series Data** (Sales, Sensor Data, Logs)

```python
# Load time series
df = pd.read_csv('sales.csv')

# Initialize with timestamp and value columns
prep = TimeSeriesPrepML(df, timestamp_column='date', value_column='sales')

# Fill gaps and add features
prep.fill_missing_timestamps(freq='D')
prep.interpolate_missing(method='linear')
prep.add_time_features()
prep.add_lag_features(lags=[1, 7, 30])

# Get enhanced data
enhanced_df = prep.df
```

#### **Graph Data** (Social Networks, Relationships)

```python
# Load nodes and edges
nodes_df = pd.read_csv('nodes.csv')
edges_df = pd.read_csv('edges.csv')

# Initialize graph
prep = GraphPrepML(nodes_df=nodes_df, edges_df=edges_df,
                   node_id_col='id', source_col='source', target_col='target')

# Validate and clean
prep.validate_node_ids()
prep.validate_edges(remove_self_loops=True, remove_dangling=True)
prep.add_node_features()

# Get cleaned graph
clean_nodes = prep.nodes_df
clean_edges = prep.edges_df
```

#### **Image Data** (Computer Vision, ML Models)

```python
from autoprepml import ImagePrepML

# Initialize with image directory
prep = ImagePrepML(
    image_dir='./images',
    target_size=(224, 224),
    color_mode='rgb',
    normalize=True
)

# Detect issues
issues = prep.detect()

# Clean and preprocess
processed_images = prep.clean(
    remove_corrupted=True,
    resize=True,
    convert_mode=True
)

# Split dataset
train, val, test = prep.split_dataset(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Save processed images
prep.save_processed('./output', format='png')

# Generate report
prep.save_report('image_report.html')
```
```

## Command Line Usage

### Quick Reference

| Option          | Short | Description                         | Example             |
| --------------- | ----- | ----------------------------------- | ------------------- |
| `--input`       | `-i`  | Input CSV file                      | `-i data.csv`       |
| `--output`      | `-o`  | Output CSV file                     | `-o cleaned.csv`    |
| `--task`        | `-t`  | ML task (classification/regression) | `-t classification` |
| `--target`      |       | Target column name                  | `--target label`    |
| `--report`      | `-r`  | HTML report path                    | `-r report.html`    |
| `--config`      | `-c`  | Config file (YAML/JSON)             | `-c config.yaml`    |
| `--detect-only` |       | Only detect issues, no cleaning     | `--detect-only`     |
| `--verbose`     | `-v`  | Verbose output                      | `-v`                |


### Common Workflows

```bash
# 1. Quick data inspection
autoprepml -i data.csv --detect-only -v

# 2. Clean and generate report
autoprepml -i raw.csv -o clean.csv -r report.html -t classification --target label

# 3. Use custom configuration
autoprepml -i data.csv -o cleaned.csv -c config.yaml

# 4. Classification task with balancing
autoprepml -i train.csv -o train_clean.csv -t classification --target Survived

# 5. Regression task with outlier removal
autoprepml -i housing.csv -o housing_clean.csv -t regression --target price -v
```

## Complete Feature Reference

### 1 Tabular Data (AutoPrepML)

**Detection Capabilities:**
- Missing values (count, percentage by column)
- Outliers (Isolation Forest, Z-score methods)
- Class imbalance (for classification tasks)
- Data type validation

**Cleaning Operations:**
- Imputation (mean, median, mode, auto)
- Scaling (StandardScaler, MinMaxScaler)
- Encoding (Label, One-Hot)
- Class balancing (Oversampling, Undersampling)
- Outlier removal

**Example:**
```python
from autoprepml import AutoPrepML

df = pd.read_csv('titanic.csv')
prep = AutoPrepML(df)

# Detect issues
issues = prep.detect(target_col='Survived')
print(f"Missing values: {issues['missing_values']}")
print(f"Outliers: {issues['outliers']['outlier_count']}")

# Auto-clean
clean_df, target = prep.clean(task='classification', target_col='Survived', auto=True)

# Generate report
prep.save_report('titanic_report.html')
```

### 2 Text/NLP Data (TextPrepML)

**Detection Capabilities:**
- Missing/empty text
- Very short/long texts
- URLs, emails, HTML tags
- Average text length
- Duplicates

**Cleaning Operations:**
- Text cleaning (lowercase, remove URLs/HTML/emails)
- Special character & number removal
- Stopword removal (English + custom)
- Tokenization (word/sentence)
- Feature extraction (length, word count, etc.)
- Language detection (heuristic)
- Duplicate removal
- Length filtering

**Example:**
```python
from autoprepml import TextPrepML

df = pd.read_csv('reviews.csv')
prep = TextPrepML(df, text_column='review_text')

# Detect issues
issues = prep.detect_issues()
print(f"Contains URLs: {issues['contains_urls']}")
print(f"Contains HTML: {issues['contains_html']}")

# Clean text
prep.clean_text(lowercase=True, remove_urls=True, remove_html=True)
prep.remove_stopwords()
prep.filter_by_length(min_length=10, max_length=500)

# Extract features
prep.extract_features()
prep.tokenize(method='word')

# Get vocabulary
vocab = prep.get_vocabulary(top_n=50)

# Save
cleaned_df = prep.df
cleaned_df.to_csv('reviews_cleaned.csv', index=False)
```

### 3 Time Series Data (TimeSeriesPrepML)

**Detection Capabilities:**
- Duplicate timestamps
- Missing dates/gaps
- Chronological order validation
- Missing values in series
- Negative/zero values

**Cleaning Operations:**
- Sort by timestamp
- Remove/aggregate duplicate timestamps
- Fill missing timestamps (any frequency)
- Interpolation (linear, forward-fill, back-fill)
- Outlier detection (Z-score, IQR)
- Time feature extraction (year, month, day, hour, day of week, quarter, weekend)
- Lag features (1-day, 7-day, 30-day, custom)
- Rolling window statistics (mean, std, min, max)
- Resampling to different frequencies

**Example:**
```python
from autoprepml import TimeSeriesPrepML

df = pd.read_csv('sales.csv')
prep = TimeSeriesPrepML(df, timestamp_column='date', value_column='sales')

# Detect issues
issues = prep.detect_issues()
print(f"Detected gaps: {issues['detected_gaps']}")
print(f"Duplicate timestamps: {issues['duplicate_timestamps']}")

# Clean and enhance
prep.sort_by_time()
prep.remove_duplicate_timestamps(aggregate='mean')
prep.fill_missing_timestamps(freq='D')  # Daily frequency
prep.interpolate_missing(method='linear')

# Feature engineering for ML
prep.add_time_features()
prep.add_lag_features(lags=[1, 7, 30])
prep.add_rolling_features(windows=[7, 30], functions=['mean', 'std'])

# Optional: Detect outliers
prep.detect_outliers(method='zscore', threshold=3.0)

# Save enhanced data
enhanced_df = prep.df
enhanced_df.to_csv('sales_enhanced.csv', index=False)
```

### 4 Graph Data (GraphPrepML)

**Detection Capabilities:**
- Duplicate node IDs
- Missing node IDs
- Duplicate edges
- Self-loops
- Dangling edges (edges to non-existent nodes)
- Isolated nodes

**Cleaning Operations:**
- Node ID validation
- Edge validation (remove self-loops, dangling edges)
- Duplicate removal (nodes and edges)
- Node feature extraction (in/out/total degree)
- Edge feature extraction
- Connected component identification (BFS algorithm)
- Isolated node filtering
- Graph statistics (density, average degree)
- Format conversion (edge list, adjacency dict)

**Example:**
```python
from autoprepml import GraphPrepML

nodes = pd.read_csv('users.csv')
edges = pd.read_csv('friendships.csv')

prep = GraphPrepML(nodes_df=nodes, edges_df=edges,
                   node_id_col='user_id',
                   source_col='from_user',
                   target_col='to_user')

# Detect issues
issues = prep.detect_issues()
print(f"Duplicate nodes: {issues['nodes']['duplicate_node_ids']}")
print(f"Dangling edges: {issues['edges']['dangling_edges']}")

# Clean graph
prep.validate_node_ids()
prep.validate_edges(remove_self_loops=True, remove_dangling=True)
prep.remove_duplicate_edges()

# Feature extraction
prep.add_node_features()  # Adds degree centrality
prep.identify_components()  # Finds connected components

# Get statistics
stats = prep.get_graph_stats()
print(f"Graph density: {stats['density']:.4f}")
print(f"Average degree: {stats['avg_degree']:.2f}")

# Save cleaned data
prep.nodes_df.to_csv('users_cleaned.csv', index=False)
prep.edges_df.to_csv('friendships_cleaned.csv', index=False)
```

## Configuration

AutoPrepML supports YAML/JSON configuration files for reproducible workflows.

### Create Configuration File

**config.yaml:**
```yaml
cleaning:
  missing_strategy: auto  # auto, mean, median, mode, drop
  outlier_method: iforest  # iforest, zscore
  outlier_contamination: 0.1
  scale_method: standard  # standard, minmax
  encode_method: label  # label, onehot
  balance_method: oversample  # oversample, undersample
  remove_outliers: false

detection:
  outlier_method: iforest
  outlier_contamination: 0.1
  imbalance_threshold: 0.3

reporting:
  include_plots: true
  plot_dpi: 100

logging:
  level: INFO
```

### Use Configuration

```python
from autoprepml import AutoPrepML

# Load with config file
prep = AutoPrepML(df, config_path='config.yaml')
clean_df, target = prep.clean(task='classification', target_col='label')

# Or pass config dict directly
config = {
    'cleaning': {
        'missing_strategy': 'median',
        'scale_method': 'minmax'
    }
}
prep = AutoPrepML(df, config=config)
```

## Examples Directory

The `examples/` directory contains working demo scripts for all data types.

### Available Demos

| Demo Script | Input Data | Generated Output | Features Shown |
|-------------|------------|------------------|----------------|
| **demo_script.py** | Iris dataset (150 rows) | `iris_cleaned.csv`<br>`iris_report.html` | Tabular preprocessing, scaling, encoding, HTML reports |
| **demo_text.py** | Customer reviews (100 texts) | `reviews_cleaned.csv` | Text cleaning, stopword removal, tokenization, feature extraction |
| **demo_timeseries.py** | Sales data with gaps (365 days) | `sales_cleaned.csv` | Gap filling, interpolation, lag features, rolling statistics |
| **demo_graph.py** | Social network (50 nodes, 100 edges) | `social_network_nodes_cleaned.csv`<br>`social_network_edges_cleaned.csv` | Graph validation, component detection, degree centrality |
| **demo_all.py** | All 4 data types | Console output | Multi-modal preprocessing in one script |

### Run Demos

```bash
# Navigate to project directory
cd autoprepml

# Run individual demos
python examples/demo_script.py        # Tabular data (Iris)
python examples/demo_text.py          # Text/NLP (reviews)
python examples/demo_timeseries.py    # Time series (sales)
python examples/demo_graph.py         # Graph data (social network)
python examples/demo_all.py           # All data types

# Check generated files
ls *.csv *.html
```

### Expected Output Files
After running demos, you'll find these files in your directory:
- `iris_cleaned.csv`, `iris_report.html`
- `reviews_cleaned.csv`
- `sales_cleaned.csv`
- `social_network_nodes_cleaned.csv`, `social_network_edges_cleaned.csv`

## Testing

The repository currently passes 323 tests locally. The latest local run reports 88.04 percent line coverage. Continuous integration enforces a 75 percent minimum and also runs linting, security checks, packaging checks, and the strict documentation build.

### Run the test suite

```bash
pytest tests/ -v
```

### Generate a coverage report

```bash
pytest tests/ --cov=autoprepml --cov-report=term-missing --cov-report=html
```

Open `htmlcov/index.html` to inspect uncovered lines. Optional provider tests remain skipped when their external services are not configured.

## Project Structure

The repository is organised around a small public package and a set of focused modules:

* `autoprepml/` contains the library implementation, including modality specific processors, detection, cleaning, reporting, configuration, the command line interface, and optional LLM integrations.
* `tests/` contains unit and integration coverage for the public API.
* `examples/` contains runnable demonstrations for each supported data type.
* `docs/` contains the user guide, API reference, tutorials, feature guides, and release notes.
* `scripts/` contains test, documentation, release, and OpenML smoke test helpers.
* `pyproject.toml` defines package metadata, dependencies, optional extras, and tool configuration.

## Development Setup

### For Contributors

```bash
# 1. Fork and clone the repository
git clone https://github.com/mdshoaibuddinchanda/autoprepml.git
cd autoprepml

# 2. Create a virtual environment (recommended)
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# 3. Install in development mode with dev dependencies
pip install -e ".[dev]"

# 4. Run tests to verify setup
pytest tests/ -v

# 5. Make your changes and run tests again
pytest tests/ -v
```

### Development Commands

```bash
# Run tests with coverage
pytest tests/ --cov=autoprepml --cov-report=html

# Run tests for specific module
pytest tests/test_text.py -v

# Run linting (if configured)
black autoprepml/ tests/
ruff check autoprepml/

# Build the documentation site
mkdocs build --strict

# Create distribution packages
python -m build
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Usage Guide](docs/usage.md)**: Step by step guidance for each data type
- **[API Reference](docs/api_reference.md)**: Public classes and functions
- **[Tutorials](docs/tutorials.md)**: End to end examples and practices
- **[Advanced Features](docs/ADVANCED_FEATURES.md)**: Imputation and class balancing
- **[LLM Configuration](docs/LLM_CONFIGURATION.md)**: Provider setup and credential handling

### Build Documentation Locally

```bash
pip install -e ".[docs]"
mkdocs build --strict
```

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick start:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run tests: `pytest tests/ -v`
5. Commit: `git commit -m "Add amazing feature"`
6. Push and open a Pull Request

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import Error | `pip install -e .` |
| CLI not recognized | Reinstall: `pip uninstall autoprepml && pip install -e .` |
| Tests failing | Install dev dependencies: `pip install -e ".[dev]"` |
| Matplotlib backend issues | Set backend: `import matplotlib; matplotlib.use('Agg')` |
| Memory issues | Process in chunks: `pd.read_csv('file.csv', chunksize=10000)` |

For more help, see [GitHub Issues](https://github.com/mdshoaibuddinchanda/autoprepml/issues) or [Discussions](https://github.com/mdshoaibuddinchanda/autoprepml/discussions).

## Performance

### Performance guidance

Processing time and memory use depend on the modality, schema, optional transformations, and report settings. Treat any benchmark as workload specific and measure representative data before setting service limits.

### Optimization Tips

```python
# 1. Use auto mode for faster processing
prep.clean(task='classification', target_col='label', auto=True)

# 2. Disable reporting for speed
prep = AutoPrepML(df, config={'reporting': {'include_plots': False}})

# 3. Process in chunks for large data
for chunk in pd.read_csv('big.csv', chunksize=10000):
    prep = AutoPrepML(chunk)
    # Process
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), and [matplotlib](https://matplotlib.org/)
- Inspired by the need for faster data preprocessing in ML workflows
- Thanks to all [contributors](https://github.com/mdshoaibuddinchanda/autoprepml/graphs/contributors)

## Contact

For support, use the [issue tracker](https://github.com/mdshoaibuddinchanda/autoprepml/issues) or [GitHub Discussions](https://github.com/mdshoaibuddinchanda/autoprepml/discussions). The project is maintained by [MD Shoaibuddin Chanda](https://github.com/mdshoaibuddinchanda).

## Roadmap

### Delivered

- [x] Preprocessing for tabular, text, time series, graph, and image data.
- [x] Detection, cleaning, feature engineering, visualisation, and HTML or JSON reporting.
- [x] YAML and JSON configuration and command line workflows.
- [x] Advanced imputation, SMOTE balancing, AutoEDA, dashboards, and optional LLM assistance.
- [x] Blocking CI checks for tests, linting, security, packaging, and documentation.

### Planned

- [ ] Publish reproducible performance benchmarks for representative workloads.
- [ ] Add chunked and parallel processing for large datasets.
- [ ] Expand storage adapters and streaming integrations.
- [ ] Add experiment tracking and model pipeline integrations.
- [ ] Continue raising coverage and strengthening contract, integration, and smoke tests.

## Use Cases

### By Industry

| Industry | Use Cases |
|----------|-----------|
| **E-Commerce** | Customer review sentiment (Text), Sales forecasting (Time Series), Product recommendations (Graph) |
| **Finance** | Fraud detection (Tabular), Stock prediction (Time Series), Transaction networks (Graph) |
| **Healthcare** | Patient data (Tabular), Medical reports (Text), Disease tracking (Time Series), Provider networks (Graph) |
| **Social Media** | User behavior (Tabular), Content moderation (Text), Trend detection (Time Series), Social networks (Graph) |

### By Task

- **Machine Learning**: Feature engineering, data quality assessment, automated preprocessing
- **Data Science**: EDA, data cleaning for visualization, statistical analysis
- **Research**: Dataset preparation, reproducible workflows, benchmark creation

## Additional resources

Read the [documentation](docs/), browse the [examples](examples/), review the [changelog](CHANGELOG.md), or see the [contribution guide](CONTRIBUTING.md).
