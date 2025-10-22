# AutoPrepML – Multi-Modal Data Preprocessing Pipeline

[![CI](https://github.com/mdshoaibuddinchanda/autoprepml/workflows/CI/badge.svg)](https://github.com/mdshoaibuddinchanda/autoprepml/actions)
[![codecov](https://codecov.io/gh/mdshoaibuddinchanda/autoprepml/branch/main/graph/badge.svg)](https://codecov.io/gh/mdshoaibuddinchanda/autoprepml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-103%20passed-brightgreen.svg)](tests/)

> **Automate data preprocessing for ANY data type — Tabular, Text, Time Series, and Graphs.**

A comprehensive Python library that automatically detects, cleans, and transforms data across multiple modalities. Built for real-world ML pipelines with one-line automation and detailed reporting.

## 🎯 Features

- ✨ **Multi-Modal Support** - Works with 4 different data types out of the box
- 🔍 **Automatic Issue Detection** - Missing values, outliers, duplicates, anomalies
- 📊 **Visual Reports** - HTML reports with embedded plots and statistics
- ⚙️ **Highly Configurable** - YAML/JSON configuration for reproducibility
- 🚀 **CLI + Python API** - Use from command line or Python scripts
- 🧪 **Production Ready** - 103 tests passing, 95%+ code coverage

## 📊 Supported Data Types

| Data Type | Module | Use Cases | Status |
|-----------|--------|-----------|--------|
| **Tabular** | `AutoPrepML` | Classification, Regression, General ML | ✅ Ready |
| **Text/NLP** | `TextPrepML` | Sentiment Analysis, Topic Modeling, Classification | ✅ Ready |
| **Time Series** | `TimeSeriesPrepML` | Forecasting, Trend Analysis, Anomaly Detection | ✅ Ready |
| **Graph** | `GraphPrepML` | Social Networks, Recommendation Systems, Link Prediction | ✅ Ready |

## 📦 Installation

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/mdshoaibuddinchanda/autoprepml.git

# Navigate to the project directory
cd autoprepml

# Install in development mode
pip install -e .
```

### Option 2: Install with Dependencies

```bash
# Clone and install
git clone https://github.com/mdshoaibuddinchanda/autoprepml.git
cd autoprepml
pip install -e ".[dev]"  # Includes testing and development tools
```

### Option 3: Install from PyPI (Coming Soon)

```bash
pip install autoprepml
```

### Verify Installation

```bash
# Check if autoprepml is installed
python -c "from autoprepml import AutoPrepML, TextPrepML, TimeSeriesPrepML, GraphPrepML; print('✓ Installation successful!')"

# Check CLI is available
autoprepml --help
```

## 🚀 Quick Start Guide

### Step 1: Import the Library

```python
import pandas as pd
from autoprepml import AutoPrepML, TextPrepML, TimeSeriesPrepML, GraphPrepML
```

### Step 2: Choose Your Data Type

#### 📊 **Tabular Data** (CSV, Excel, JSON)

```python
# Load your data
df = pd.read_csv('data.csv')

# Initialize and clean
prep = AutoPrepML(df)
clean_df, target = prep.clean(task='classification', target_col='label')

# Generate report
prep.save_report('report.html')
```

#### 📝 **Text/NLP Data** (Reviews, Documents, Tweets)

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

#### ⏰ **Time Series Data** (Sales, Sensor Data, Logs)

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

#### 🕸️ **Graph Data** (Social Networks, Relationships)

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
```

## 💻 Command Line Usage

AutoPrepML provides a powerful CLI for quick data preprocessing without writing code.

### Basic Commands

```bash
# Tabular data preprocessing
autoprepml --input data.csv --output cleaned.csv

# With task specification
autoprepml --input train.csv --output clean_train.csv \
           --task classification --target label

# Generate HTML report
autoprepml --input data.csv --output cleaned.csv \
           --report report.html

# Detection only (no cleaning)
autoprepml --input data.csv --detect-only

# Verbose output
autoprepml --input data.csv --output cleaned.csv --verbose

# With custom configuration
autoprepml --input data.csv --output cleaned.csv \
           --config config.yaml
```

### CLI Options Reference

| Option | Description | Example |
|--------|-------------|---------|
| `--input`, `-i` | Input CSV file path | `--input data.csv` |
| `--output`, `-o` | Output CSV file path | `--output cleaned.csv` |
| `--task`, `-t` | ML task type | `--task classification` |
| `--target` | Target column name | `--target label` |
| `--report`, `-r` | HTML report path | `--report report.html` |
| `--config`, `-c` | Config file (YAML/JSON) | `--config config.yaml` |
| `--detect-only` | Only detect issues | `--detect-only` |
| `--verbose`, `-v` | Verbose output | `--verbose` |

### Example Workflows

#### 1. Quick Data Inspection
```bash
# See what issues exist in your data
autoprepml --input messy_data.csv --detect-only --verbose
```

#### 2. Clean and Report
```bash
# Clean data and generate visual report
autoprepml --input raw_data.csv \
           --output clean_data.csv \
           --report data_report.html \
           --task regression \
           --target price
```

#### 3. Use Custom Configuration
```bash
# Create config.yaml first (see Configuration section)
autoprepml --input data.csv \
           --output cleaned.csv \
           --config my_config.yaml
```

## � Complete Feature Reference

### 1️⃣ Tabular Data (AutoPrepML)

**Detection Capabilities:**
- ✅ Missing values (count, percentage by column)
- ✅ Outliers (Isolation Forest, Z-score methods)
- ✅ Class imbalance (for classification tasks)
- ✅ Data type validation

**Cleaning Operations:**
- ✅ Imputation (mean, median, mode, auto)
- ✅ Scaling (StandardScaler, MinMaxScaler)
- ✅ Encoding (Label, One-Hot)
- ✅ Class balancing (Oversampling, Undersampling)
- ✅ Outlier removal

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

### 2️⃣ Text/NLP Data (TextPrepML)

**Detection Capabilities:**
- ✅ Missing/empty text
- ✅ Very short/long texts
- ✅ URLs, emails, HTML tags
- ✅ Average text length
- ✅ Duplicates

**Cleaning Operations:**
- ✅ Text cleaning (lowercase, remove URLs/HTML/emails)
- ✅ Special character & number removal
- ✅ Stopword removal (English + custom)
- ✅ Tokenization (word/sentence)
- ✅ Feature extraction (length, word count, etc.)
- ✅ Language detection (heuristic)
- ✅ Duplicate removal
- ✅ Length filtering

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

### 3️⃣ Time Series Data (TimeSeriesPrepML)

**Detection Capabilities:**
- ✅ Duplicate timestamps
- ✅ Missing dates/gaps
- ✅ Chronological order validation
- ✅ Missing values in series
- ✅ Negative/zero values

**Cleaning Operations:**
- ✅ Sort by timestamp
- ✅ Remove/aggregate duplicate timestamps
- ✅ Fill missing timestamps (any frequency)
- ✅ Interpolation (linear, forward-fill, back-fill)
- ✅ Outlier detection (Z-score, IQR)
- ✅ Time feature extraction (year, month, day, hour, day of week, quarter, weekend)
- ✅ Lag features (1-day, 7-day, 30-day, custom)
- ✅ Rolling window statistics (mean, std, min, max)
- ✅ Resampling to different frequencies

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

### 4️⃣ Graph Data (GraphPrepML)

**Detection Capabilities:**
- ✅ Duplicate node IDs
- ✅ Missing node IDs
- ✅ Duplicate edges
- ✅ Self-loops
- ✅ Dangling edges (edges to non-existent nodes)
- ✅ Isolated nodes

**Cleaning Operations:**
- ✅ Node ID validation
- ✅ Edge validation (remove self-loops, dangling edges)
- ✅ Duplicate removal (nodes and edges)
- ✅ Node feature extraction (in/out/total degree)
- ✅ Edge feature extraction
- ✅ Connected component identification (BFS algorithm)
- ✅ Isolated node filtering
- ✅ Graph statistics (density, average degree)
- ✅ Format conversion (edge list, adjacency dict)

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

## ⚙️ Configuration

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

## 📖 Running the Examples

The `examples/` directory contains working demo scripts for all data types.

### Run Individual Demos

```bash
# Navigate to project directory
cd autoprepml

# Run tabular data demo
python examples/demo_script.py

# Run text/NLP demo
python examples/demo_text.py

# Run time series demo
python examples/demo_timeseries.py

# Run graph data demo
python examples/demo_graph.py

# Run comprehensive multi-modal demo
python examples/demo_all.py
```

### What Each Demo Does

| Demo | Input | Output | Features Demonstrated |
|------|-------|--------|----------------------|
| `demo_script.py` | Iris dataset | `iris_cleaned.csv`<br>`iris_report.html` | Tabular preprocessing, HTML reports |
| `demo_text.py` | Customer reviews | `reviews_cleaned.csv` | Text cleaning, stopword removal, tokenization |
| `demo_timeseries.py` | Sales data with gaps | `sales_cleaned.csv` | Gap filling, interpolation, feature engineering |
| `demo_graph.py` | Social network | `social_network_nodes_cleaned.csv`<br>`social_network_edges_cleaned.csv` | Graph validation, component detection |
| `demo_all.py` | All 4 data types | Console output | Multi-modal preprocessing in one script |

### Expected Demo Output

After running demos, you'll find generated files in the project directory:
- `*_cleaned.csv` - Cleaned data files
- `*_report.html` - Visual HTML reports (for tabular data)

```bash
# Check generated files
ls -la *.csv *.html
```

## 🧪 Testing

AutoPrepML has comprehensive test coverage with 103 tests.

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=autoprepml --cov-report=html

# Run specific test file
pytest tests/test_text.py -v

# Run tests for specific module
pytest tests/test_timeseries.py -v
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `core.py` | 6 tests | 95% |
| `detection.py` | 8 tests | 98% |
| `cleaning.py` | 11 tests | 96% |
| `visualization.py` | 7 tests | 92% |
| `reports.py` | 3 tests | 90% |
| `text.py` | 18 tests | 95% |
| `timeseries.py` | 18 tests | 95% |
| `graph.py` | 26 tests | 97% |
| **Total** | **103 tests** | **95%** |

### Quick Test Command

```bash
# Just see if everything passes
pytest tests/ -q

# Output: 103 passed, 7 warnings in 5.01s
```

## 🏗️ Project Structure

```
autoprepml/
├── autoprepml/              # Core library
│   ├── __init__.py         # Package initialization
│   ├── core.py             # AutoPrepML class (tabular data)
│   ├── text.py             # TextPrepML class (text/NLP)
│   ├── timeseries.py       # TimeSeriesPrepML class (time series)
│   ├── graph.py            # GraphPrepML class (graph data)
│   ├── detection.py        # Issue detection functions
│   ├── cleaning.py         # Data cleaning transformations
│   ├── visualization.py    # Plot generation
│   ├── reports.py          # JSON/HTML report generators
│   ├── config.py           # Configuration management
│   ├── llm_suggest.py      # AI suggestions (placeholder)
│   ├── cli.py              # Command-line interface
│   └── utils.py            # Helper utilities
├── tests/                   # Test suite (103 tests)
│   ├── test_core.py        # Tabular data tests (6)
│   ├── test_text.py        # Text preprocessing tests (18)
│   ├── test_timeseries.py  # Time series tests (18)
│   ├── test_graph.py       # Graph data tests (26)
│   ├── test_detection.py   # Detection tests (8)
│   ├── test_cleaning.py    # Cleaning tests (11)
│   ├── test_visualization.py # Visualization tests (7)
│   ├── test_reports.py     # Reporting tests (3)
│   └── test_llm_suggest.py # LLM tests (6)
├── examples/                # Demo scripts
│   ├── demo_script.py      # Tabular data demo
│   ├── demo_text.py        # Text/NLP demo
│   ├── demo_timeseries.py  # Time series demo
│   ├── demo_graph.py       # Graph data demo
│   ├── demo_all.py         # Multi-modal demo
│   └── demo_notebook.ipynb # Jupyter notebook demo
├── docs/                    # Documentation
│   ├── index.md            # Documentation home
│   ├── usage.md            # Usage guide
│   ├── api_reference.md    # API documentation
│   └── tutorials.md        # Detailed tutorials
├── scripts/                 # Utility scripts
│   ├── run_tests.sh        # Test runner
│   ├── build_docs.sh       # Documentation builder
│   └── release.sh          # Release automation
├── setup.py                # Package setup
├── pyproject.toml          # Modern Python packaging
├── requirements.txt        # Dependencies
├── README.md               # This file
├── LICENSE                 # MIT License
├── .gitignore              # Git ignore rules
└── autoprepml.yaml         # Sample configuration
```

## 🛠️ Development Setup

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

# Build documentation
cd docs
mkdocs serve  # View at http://localhost:8000

# Create distribution packages
python -m build
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Usage Guide](docs/usage.md)** - Step-by-step tutorials for each data type
- **[API Reference](docs/api_reference.md)** - Complete function and class documentation
- **[Tutorials](docs/tutorials.md)** - Real-world examples and best practices
- **[MULTI_MODAL_SUMMARY.md](MULTI_MODAL_SUMMARY.md)** - Multi-modal feature overview

### Build Documentation Locally

```bash
# Install MkDocs (if not already installed)
pip install mkdocs mkdocs-material

# Navigate to docs directory
cd docs

# Serve documentation locally
mkdocs serve

# Build static site
mkdocs build
```

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

1. **Report Bugs** - Open an issue with detailed reproduction steps
2. **Suggest Features** - Share your ideas for new features
3. **Submit Pull Requests** - Fix bugs or add new features
4. **Improve Documentation** - Help make docs clearer and more comprehensive
5. **Write Tests** - Increase test coverage
6. **Share Use Cases** - Tell us how you're using AutoPrepML

### Contribution Workflow

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/autoprepml.git
cd autoprepml

# 3. Create a feature branch
git checkout -b feature/amazing-feature

# 4. Make your changes
# - Edit code
# - Add tests
# - Update documentation

# 5. Run tests
pytest tests/ -v

# 6. Commit your changes
git add .
git commit -m "Add amazing feature"

# 7. Push to your fork
git push origin feature/amazing-feature

# 8. Open a Pull Request on GitHub
```

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions and classes
- Write tests for new features
- Keep functions focused and modular

### Testing Guidelines

```bash
# Run all tests
pytest tests/ -v

# Test specific module
pytest tests/test_text.py -v

# Check coverage
pytest tests/ --cov=autoprepml

# Coverage should be >90%
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Error: Module not found**
```bash
# Solution: Install in development mode
pip install -e .
```

**2. CLI command not recognized**
```bash
# Solution: Reinstall package
pip uninstall autoprepml
pip install -e .
```

**3. Tests failing**
```bash
# Solution: Install dev dependencies
pip install -e ".[dev]"
pytest tests/ -v
```

**4. Matplotlib backend issues**
```python
# Solution: Set backend explicitly
import matplotlib
matplotlib.use('Agg')
```

**5. Memory issues with large datasets**
```python
# Solution: Process in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    prep = AutoPrepML(chunk)
    # Process chunk
```

### Getting Help

- **GitHub Issues**: [Report bugs or ask questions](https://github.com/mdshoaibuddinchanda/autoprepml/issues)
- **Discussions**: [Community forum](https://github.com/mdshoaibuddinchanda/autoprepml/discussions)
- **Documentation**: Check [docs/](docs/) for detailed guides

## 📊 Performance

### Benchmarks

| Dataset Size | Data Type | Processing Time | Memory Usage |
|--------------|-----------|----------------|--------------|
| 1K rows | Tabular | <0.5s | <50MB |
| 10K rows | Tabular | <2s | <100MB |
| 100K rows | Tabular | <10s | <500MB |
| 1K texts | Text/NLP | <1s | <100MB |
| 10K texts | Text/NLP | <5s | <300MB |
| 1K timestamps | Time Series | <1s | <80MB |
| 10K nodes/edges | Graph | <2s | <150MB |

*Benchmarks run on: Intel Core i5, 16GB RAM, Python 3.10*

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

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), and [matplotlib](https://matplotlib.org/)
- Inspired by the need for faster data preprocessing in ML workflows
- Thanks to all [contributors](https://github.com/yourusername/autoprepml/graphs/contributors)

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/autoprepml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/autoprepml/discussions)
- **Email**: your.email@example.com

## 🗺️ Roadmap

### ✅ Phase 1 - Core MVP (COMPLETED)
- [x] Tabular data preprocessing
- [x] Basic detection and cleaning
- [x] JSON/HTML reports with visualizations
- [x] CLI support
- [x] Comprehensive unit tests (41 tests)

### ✅ Phase 2 - Multi-Modal Support (COMPLETED)
- [x] Text/NLP preprocessing module
- [x] Time series preprocessing module
- [x] Graph data preprocessing module
- [x] 62 additional tests (103 total)
- [x] Demo scripts for all data types
- [x] Multi-modal documentation

### 🚧 Phase 3 - Enhanced Features (IN PROGRESS)
- [x] YAML/JSON configuration system
- [ ] Advanced imputation (KNN, iterative)
- [ ] SMOTE for class balancing
- [ ] LLM integration for smart suggestions
- [ ] Image data preprocessing
- [ ] Audio/video metadata extraction

### 📋 Phase 4 - Production & Scale
- [ ] PyPI package release
- [ ] Distributed processing (Dask, Spark)
- [ ] Cloud storage integration (S3, GCS, Azure)
- [ ] Real-time streaming support
- [ ] MLOps integration (MLflow, Weights & Biases)
- [ ] Docker containers

### 🌟 Phase 5 - Community & Ecosystem
- [ ] Documentation website
- [ ] Video tutorials
- [ ] Community plugins
- [ ] Integration with popular ML frameworks
- [ ] Kaggle kernels and examples

## 💡 Use Cases

### By Industry

**E-Commerce**
- Customer review sentiment analysis (Text)
- Sales forecasting (Time Series)
- Product recommendation networks (Graph)

**Finance**
- Fraud detection (Tabular)
- Stock price prediction (Time Series)
- Transaction network analysis (Graph)

**Healthcare**
- Patient data preprocessing (Tabular)
- Medical report analysis (Text)
- Disease outbreak tracking (Time Series)
- Healthcare provider networks (Graph)

**Social Media**
- User behavior analysis (Tabular)
- Content moderation (Text)
- Trend detection (Time Series)
- Social network analysis (Graph)

### By Task

**Machine Learning**
- Feature engineering for models
- Data quality assessment
- Automated preprocessing pipelines

**Data Science**
- Exploratory data analysis
- Data cleaning for visualization
- Statistical analysis preparation

**Research**
- Dataset preparation for experiments
- Reproducible preprocessing workflows
- Benchmark dataset creation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Built with**: [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/)
- **Inspired by**: The need for faster, more automated data preprocessing in ML workflows
- **Thanks to**: All contributors and users of AutoPrepML

### Dependencies

**Core:**
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- jinja2 >= 3.0.0
- pyyaml >= 5.4.0

**Development:**
- pytest >= 7.0.0
- pytest-cov >= 3.0.0

## 📧 Contact & Support

- **Author**: MD Shoaibuddin Chanda
- **GitHub**: [@mdshoaibuddinchanda](https://github.com/mdshoaibuddinchanda)
- **Issues**: [GitHub Issues](https://github.com/mdshoaibuddinchanda/autoprepml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mdshoaibuddinchanda/autoprepml/discussions)

## 📈 Project Status

**Current Version**: 0.1.0  
**Status**: Active Development  
**Test Coverage**: 95%+  
**Tests Passing**: 103/103  
**Python Support**: 3.8, 3.9, 3.10, 3.11+

## 📚 Citation

If you use AutoPrepML in your research or project, please cite:

```bibtex
@software{autoprepml2025,
  author = {Chanda, MD Shoaibuddin},
  title = {AutoPrepML: Multi-Modal Data Preprocessing Pipeline},
  year = {2025},
  url = {https://github.com/mdshoaibuddinchanda/autoprepml}
}
```

---

<div align="center">

**Made with ❤️ by [MD Shoaibuddin Chanda](https://github.com/mdshoaibuddinchanda)**

**AutoPrepML** - Automate your data preprocessing workflow

[⭐ Star on GitHub](https://github.com/mdshoaibuddinchanda/autoprepml) • 
[📖 Read the Docs](docs/) • 
[🐛 Report Bug](https://github.com/mdshoaibuddinchanda/autoprepml/issues) • 
[💡 Request Feature](https://github.com/mdshoaibuddinchanda/autoprepml/issues)

</div>
