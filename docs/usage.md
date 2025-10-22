# Usage Guide

## Installation

Install AutoPrepML using pip:

```bash
pip install autoprepml
```

For development installation:

```bash
git clone https://github.com/yourusername/autoprepml.git
cd autoprepml
pip install -e ".[dev]"
```

## Basic Usage

### Python API

#### 1. Initialize AutoPrepML

```python
import pandas as pd
from autoprepml import AutoPrepML

df = pd.read_csv('your_data.csv')
prep = AutoPrepML(df)
```

#### 2. Detect Issues

```python
# Run detection
results = prep.detect(target_col='label')

# Check results
print(results['missing_values'])
print(results['outliers'])
print(results['class_imbalance'])
```

#### 3. Clean Data

```python
# Automatic cleaning
clean_df, report = prep.clean(task='classification', target_col='label')

# Save cleaned data
clean_df.to_csv('cleaned_data.csv', index=False)
```

#### 4. Generate Reports

```python
# Save HTML report
prep.save_report('report.html')

# Or JSON report
prep.save_report('report.json')
```

### CLI Usage

#### Basic Command

```bash
autoprepml --input data.csv --output cleaned.csv
```

#### With Options

```bash
autoprepml \
  --input train.csv \
  --output clean_train.csv \
  --task classification \
  --target label \
  --report report.html
```

#### Detection Only

```bash
autoprepml --input data.csv --output temp.csv --detect-only --report analysis.html
```

## Configuration Files

### Creating a Config File

Create `autoprepml.yaml`:

```yaml
cleaning:
  missing_strategy: auto
  numeric_strategy: median
  categorical_strategy: mode
  outlier_method: iforest
  outlier_contamination: 0.05
  remove_outliers: false
  scale_method: standard
  encode_method: label
  balance_method: oversample

detection:
  outlier_method: iforest
  contamination: 0.05
  zscore_threshold: 3.0
  imbalance_threshold: 0.3

reporting:
  format: html
  include_plots: true
  output_dir: ./reports
```

### Using Config Files

```python
prep = AutoPrepML(df, config_path='autoprepml.yaml')
clean_df, report = prep.clean()
```

Or via CLI:

```bash
autoprepml --input data.csv --output cleaned.csv --config autoprepml.yaml
```

## Advanced Usage

### Custom Preprocessing Pipeline

```python
from autoprepml import AutoPrepML, detection, cleaning

# Initialize
df = pd.read_csv('data.csv')
prep = AutoPrepML(df)

# Step 1: Detect issues
prep.detect()

# Step 2: Custom cleaning
df_clean = df.copy()
df_clean = cleaning.impute_missing(df_clean, strategy='median')
df_clean = cleaning.encode_categorical(df_clean, method='onehot')
df_clean = cleaning.scale_features(df_clean, method='minmax')

# Step 3: Manual verification
print(df_clean.info())
```

### Working with Specific Modules

```python
from autoprepml import detection, cleaning, visualization

# Detection only
missing_info = detection.detect_missing(df)
outlier_info = detection.detect_outliers(df, method='zscore', threshold=2.5)

# Cleaning only
df_imputed = cleaning.impute_missing(df, strategy='mean')
df_scaled = cleaning.scale_features(df_imputed, method='standard')

# Visualization only
plot_base64 = visualization.plot_missing(df)
```

## Common Workflows

### Classification Task

```python
prep = AutoPrepML(df)
clean_df, report = prep.clean(
    task='classification',
    target_col='label'
)
# Automatically handles:
# - Missing value imputation
# - Categorical encoding
# - Feature scaling
# - Class balancing (if imbalanced)
```

### Regression Task

```python
prep = AutoPrepML(df)
clean_df, report = prep.clean(
    task='regression',
    target_col='price'
)
# Automatically handles:
# - Missing value imputation
# - Categorical encoding
# - Feature scaling
# - Outlier detection (optional removal)
```

### Custom Config for Financial Data

```yaml
cleaning:
  missing_strategy: auto
  numeric_strategy: mean  # Use mean for financial metrics
  remove_outliers: true   # Remove extreme values
  outlier_contamination: 0.01  # More conservative
  scale_method: minmax    # Scale to 0-1 range

detection:
  zscore_threshold: 4.0   # More tolerant of variance
```

## Troubleshooting

### Issue: "No module named 'autoprepml'"

```bash
pip install autoprepml
```

### Issue: Memory errors with large datasets

Process in chunks:

```python
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    prep = AutoPrepML(chunk)
    clean_chunk, _ = prep.clean()
    clean_chunk.to_csv('output.csv', mode='a', header=False, index=False)
```

### Issue: Matplotlib backend errors

Set backend before importing:

```python
import matplotlib
matplotlib.use('Agg')
from autoprepml import AutoPrepML
```
