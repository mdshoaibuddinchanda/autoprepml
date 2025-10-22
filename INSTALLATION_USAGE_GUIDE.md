# AutoPrepML - Complete Installation & Usage Guide

## 📦 Installation Guide

### Prerequisites

Before installing AutoPrepML, ensure you have:
- Python 3.10 or higher
- pip (Python package installer)
- Git (for cloning from repository)

### Check Your Python Version

```bash
python --version
# Should show Python 3.10.x or higher
```

If you don't have Python installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python3`
- **Linux**: `sudo apt-get install python3`

---

## 🚀 Step-by-Step Installation

### Method 1: Install from GitHub (Recommended)

#### Step 1: Clone the Repository

```bash
# Open terminal/command prompt and run:
git clone https://github.com/mdshoaibuddinchanda/autoprepml.git
```

#### Step 2: Navigate to Project Directory

```bash
cd autoprepml
```

#### Step 3: Install the Package

```bash
# Install in development/editable mode
pip install -e .
```

This command installs AutoPrepML and all its dependencies.

#### Step 4: Verify Installation

```bash
# Test Python import
python -c "from autoprepml import AutoPrepML, TextPrepML, TimeSeriesPrepML, GraphPrepML; print('✓ Installation successful!')"

# Test CLI
autoprepml --help
```

If you see the help message, installation is successful!

---

### Method 2: Install with Development Tools

If you want to run tests and contribute:

```bash
# Clone repository
git clone https://github.com/mdshoaibuddinchanda/autoprepml.git
cd autoprepml

# Install with development dependencies
pip install -e ".[dev]"

# Verify with tests
pytest tests/ -v
```

---

### Method 3: Using Virtual Environment (Recommended for Isolation)

#### Windows:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install AutoPrepML
cd autoprepml
pip install -e .
```

#### macOS/Linux:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install AutoPrepML
cd autoprepml
pip install -e .
```

---

## 📖 Complete Usage Guide

### 1. Tabular Data Preprocessing

#### Command Line Usage

```bash
# Basic preprocessing
autoprepml --input data.csv --output cleaned.csv

# With classification task
autoprepml --input train.csv --output clean_train.csv \
           --task classification --target label

# With regression task
autoprepml --input housing.csv --output clean_housing.csv \
           --task regression --target price

# Generate HTML report
autoprepml --input data.csv --output cleaned.csv \
           --report report.html

# Detection only (no cleaning)
autoprepml --input data.csv --detect-only

# Verbose output for debugging
autoprepml --input data.csv --output cleaned.csv --verbose
```

#### Python API Usage

```python
import pandas as pd
from autoprepml import AutoPrepML

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize AutoPrepML
prep = AutoPrepML(df)

# Option 1: Automatic cleaning
clean_df, target = prep.clean(task='classification', target_col='label', auto=True)

# Option 2: Detect issues first
issues = prep.detect(target_col='label')
print(f"Missing values: {issues['missing_values']}")
print(f"Outliers: {issues['outliers']}")
print(f"Class imbalance: {issues['class_imbalance']}")

# Then clean
clean_df, target = prep.clean(task='classification', target_col='label')

# Option 3: Generate summary
summary = prep.summary()
print(summary)

# Generate report
report = prep.report(include_plots=True)
prep.save_report('my_report.html')

# Save cleaned data
clean_df.to_csv('cleaned_data.csv', index=False)
```

---

### 2. Text/NLP Data Preprocessing

#### Python API Usage

```python
from autoprepml import TextPrepML
import pandas as pd

# Load text data
df = pd.read_csv('reviews.csv')

# Initialize with text column name
prep = TextPrepML(df, text_column='review_text')

# Step 1: Detect issues
issues = prep.detect_issues()
print(f"Contains URLs: {issues['contains_urls']}")
print(f"Contains HTML: {issues['contains_html']}")
print(f"Very short texts: {issues['very_short']}")

# Step 2: Clean text
prep.clean_text(
    lowercase=True,           # Convert to lowercase
    remove_urls=True,         # Remove http:// links
    remove_emails=True,       # Remove email addresses
    remove_html=True,         # Remove HTML tags
    remove_special_chars=False,  # Keep punctuation
    remove_numbers=False,     # Keep numbers
    remove_extra_spaces=True  # Remove extra whitespace
)

# Step 3: Remove stopwords
prep.remove_stopwords()  # Uses default English stopwords

# Or with custom stopwords
custom_stopwords = ['the', 'a', 'an', 'and', 'or', 'but']
prep.remove_stopwords(stopwords=custom_stopwords)

# Step 4: Filter by length
prep.filter_by_length(min_length=10, max_length=500)

# Step 5: Remove duplicates
prep.remove_duplicates(keep='first')

# Step 6: Extract features
prep.extract_features()
# Adds: text_length, word_count, upper_count, digit_count, etc.

# Step 7: Tokenize
prep.tokenize(method='word')  # or method='sentence'

# Step 8: Get vocabulary
vocab = prep.get_vocabulary(top_n=100)
print("Top words:", list(vocab.keys())[:10])

# Get cleaned data
cleaned_df = prep.df
cleaned_df.to_csv('reviews_cleaned.csv', index=False)
```

#### Terminal Commands

```bash
# Run the text preprocessing demo
python examples/demo_text.py

# Output: reviews_cleaned.csv
```

---

### 3. Time Series Data Preprocessing

#### Python API Usage

```python
from autoprepml import TimeSeriesPrepML
import pandas as pd

# Load time series data
df = pd.read_csv('sales.csv')

# Initialize with timestamp and value columns
prep = TimeSeriesPrepML(
    df, 
    timestamp_column='date',    # Column with dates
    value_column='sales'        # Column with values
)

# Step 1: Detect issues
issues = prep.detect_issues()
print(f"Duplicate timestamps: {issues['duplicate_timestamps']}")
print(f"Detected gaps: {issues['detected_gaps']}")
print(f"Missing values: {issues['missing_values']}")

# Step 2: Sort chronologically
prep.sort_by_time()

# Step 3: Handle duplicate timestamps
prep.remove_duplicate_timestamps(aggregate='mean')  # or 'sum', 'max', 'min'

# Step 4: Fill missing timestamps
prep.fill_missing_timestamps(freq='D')  # D=daily, W=weekly, M=monthly, H=hourly

# Step 5: Interpolate missing values
prep.interpolate_missing(method='linear')  # or 'ffill', 'bfill'

# Step 6: Detect outliers
prep.detect_outliers(method='zscore', threshold=3.0)  # or method='iqr'

# Step 7: Add time features (for ML models)
prep.add_time_features()
# Adds: year, month, day, dayofweek, hour, quarter, is_weekend

# Step 8: Add lag features
prep.add_lag_features(lags=[1, 7, 30])  # 1-day, 7-day, 30-day lags

# Step 9: Add rolling window features
prep.add_rolling_features(
    windows=[7, 30],           # 7-day and 30-day windows
    functions=['mean', 'std']  # Mean and std deviation
)

# Step 10: Resample to different frequency (optional)
# prep.resample(freq='W', agg_func='sum')  # Weekly aggregation

# Get enhanced data
enhanced_df = prep.df
enhanced_df.to_csv('sales_enhanced.csv', index=False)

# Generate report
report = prep.report()
print(report)
```

#### Terminal Commands

```bash
# Run the time series preprocessing demo
python examples/demo_timeseries.py

# Output: sales_cleaned.csv
```

---

### 4. Graph Data Preprocessing

#### Python API Usage

```python
from autoprepml import GraphPrepML
import pandas as pd

# Load nodes and edges
nodes_df = pd.read_csv('users.csv')       # user_id, name, etc.
edges_df = pd.read_csv('connections.csv') # from_user, to_user, etc.

# Initialize graph preprocessor
prep = GraphPrepML(
    nodes_df=nodes_df,
    edges_df=edges_df,
    node_id_col='user_id',      # Column with node IDs
    source_col='from_user',     # Column with source node
    target_col='to_user'        # Column with target node
)

# Step 1: Detect issues
issues = prep.detect_issues()
print(f"Duplicate nodes: {issues['nodes']['duplicate_node_ids']}")
print(f"Dangling edges: {issues['edges']['dangling_edges']}")
print(f"Self-loops: {issues['edges']['self_loops']}")

# Step 2: Validate and clean nodes
prep.validate_node_ids()  # Removes duplicates and missing IDs

# Step 3: Validate and clean edges
prep.validate_edges(
    remove_self_loops=True,   # Remove edges from node to itself
    remove_dangling=True      # Remove edges to non-existent nodes
)

# Step 4: Remove duplicate edges
prep.remove_duplicate_edges(keep='first')

# Step 5: Add node features
prep.add_node_features()
# Adds: out_degree, in_degree, total_degree, is_isolated

# Step 6: Add edge features
prep.add_edge_features()

# Step 7: Identify connected components
prep.identify_components()
# Adds: component_id to nodes

# Step 8: Get graph statistics
stats = prep.get_graph_stats()
print(f"Nodes: {stats['num_nodes']}")
print(f"Edges: {stats['num_edges']}")
print(f"Density: {stats['density']:.4f}")
print(f"Average degree: {stats['avg_degree']:.2f}")

# Step 9: Convert to different formats
edge_list = prep.to_edge_list()              # List of (source, target) tuples
adj_dict = prep.to_adjacency_dict()          # Dict of adjacencies

# Get cleaned data
clean_nodes = prep.nodes_df
clean_edges = prep.edges_df

# Save
clean_nodes.to_csv('users_cleaned.csv', index=False)
clean_edges.to_csv('connections_cleaned.csv', index=False)
```

#### Terminal Commands

```bash
# Run the graph preprocessing demo
python examples/demo_graph.py

# Output: social_network_nodes_cleaned.csv, social_network_edges_cleaned.csv
```

---

## 🎯 Running All Demos

### Run Individual Demos

```bash
# Navigate to project directory
cd autoprepml

# Tabular data demo
python examples/demo_script.py

# Text/NLP demo
python examples/demo_text.py

# Time series demo
python examples/demo_timeseries.py

# Graph data demo
python examples/demo_graph.py

# All data types in one demo
python examples/demo_all.py
```

### Run All Tests

```bash
# Run all 103 tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=autoprepml

# Run specific test file
pytest tests/test_text.py -v
pytest tests/test_timeseries.py -v
pytest tests/test_graph.py -v
```

---

## ⚙️ Configuration Files

### Create config.yaml

```yaml
# config.yaml
cleaning:
  missing_strategy: auto    # auto, mean, median, mode, drop
  outlier_method: iforest   # iforest, zscore
  scale_method: standard    # standard, minmax
  encode_method: label      # label, onehot
  balance_method: oversample # oversample, undersample
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

```bash
# Via CLI
autoprepml --input data.csv --output cleaned.csv --config config.yaml
```

```python
# Via Python API
prep = AutoPrepML(df, config_path='config.yaml')
clean_df, target = prep.clean()
```

---

## 🐛 Troubleshooting

### Problem: "Module not found" error

**Solution:**
```bash
pip install -e .
```

### Problem: "autoprepml command not found"

**Solution:**
```bash
pip uninstall autoprepml
pip install -e .
```

### Problem: Tests failing

**Solution:**
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Problem: Import error in Python

**Solution:**
```python
# Check installation
import sys
print(sys.path)

# Reinstall
pip install -e . --force-reinstall
```

---

## 📚 Next Steps

1. **Read Documentation**: Check `docs/` folder for detailed guides
2. **Run Examples**: Try all demo scripts in `examples/` folder
3. **Read API Reference**: See `docs/api_reference.md` for all functions
4. **Join Community**: Star the repo and join discussions

---

## 📞 Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/mdshoaibuddinchanda/autoprepml/issues)
- **Discussions**: [Ask questions](https://github.com/mdshoaibuddinchanda/autoprepml/discussions)
- **Documentation**: Check `docs/` folder

---

**Happy Data Preprocessing!** 🎉
