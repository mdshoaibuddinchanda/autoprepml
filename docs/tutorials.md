# Tutorials

Real-world examples and use cases for AutoPrepML.

## Tutorial 1: Classification Task - Customer Churn Prediction

### Scenario

You have a customer dataset and need to predict churn. The data has missing values, outliers, and class imbalance.

### Step-by-Step

```python
import pandas as pd
from autoprepml import AutoPrepML

# Load data
df = pd.read_csv('customer_data.csv')

# Initialize
prep = AutoPrepML(df)

# Detect issues
results = prep.detect(target_col='churned')
print(f"Missing values: {len(results['missing_values'])} columns")
print(f"Class imbalance: {results['class_imbalance']['is_imbalanced']}")

# Clean for classification
clean_df, report = prep.clean(task='classification', target_col='churned')

# Save results
clean_df.to_csv('customer_data_clean.csv', index=False)
prep.save_report('churn_report.html')
```

**What happens:**
1. Missing values → Imputed (median for numeric, mode for categorical)
2. Categorical features → Label encoded
3. Numeric features → StandardScaler normalization
4. Class imbalance → Oversampling minority class

---

## Tutorial 2: Regression Task - House Price Prediction

### Scenario

Predict house prices with a dataset containing outliers and missing values.

### Step-by-Step

```python
from autoprepml import AutoPrepML

df = pd.read_csv('house_prices.csv')

# Use custom config
config = {
    'cleaning': {
        'missing_strategy': 'mean',  # Use mean for prices
        'remove_outliers': True,     # Remove extreme values
        'scale_method': 'minmax'     # Scale to 0-1 range
    }
}

prep = AutoPrepML(df, config=config)
clean_df, report = prep.clean(task='regression', target_col='price')
```

**Key differences:**
- No class balancing (regression task)
- Outliers removed (optional for regression)
- Mean imputation for price-related columns

---

## Tutorial 3: Custom Preprocessing Pipeline

### Scenario

You need fine-grained control over each preprocessing step.

### Step-by-Step

```python
from autoprepml import detection, cleaning, visualization

# Step 1: Manual detection
missing = detection.detect_missing(df)
outliers = detection.detect_outliers(df, method='zscore', threshold=2.5)

# Step 2: Selective cleaning
df_clean = df.copy()

# Only impute specific columns
df_clean[['age', 'income']] = cleaning.impute_missing(
    df_clean[['age', 'income']], 
    strategy='median'
)

# Scale only numeric features except ID columns
df_clean = cleaning.scale_features(
    df_clean, 
    method='standard',
    exclude_cols=['customer_id', 'order_id']
)

# One-hot encode specific categorical columns
df_clean = cleaning.encode_categorical(
    df_clean, 
    method='onehot',
    exclude_cols=['target']
)

# Step 3: Generate custom visualizations
plots = {
    'missing': visualization.plot_missing(df),
    'correlation': visualization.plot_correlation(df_clean)
}
```

---

## Tutorial 4: Batch Processing with CLI

### Scenario

Process multiple CSV files in a directory.

### Bash Script

```bash
#!/bin/bash
# batch_preprocess.sh

for file in data/*.csv; do
    filename=$(basename "$file" .csv)
    echo "Processing $filename..."
    
    autoprepml \
        --input "$file" \
        --output "cleaned/$filename_clean.csv" \
        --report "reports/$filename_report.html" \
        --task classification \
        --target label
done

echo "Batch processing complete!"
```

---

## Tutorial 5: Using Configuration Files

### Scenario

Maintain consistent preprocessing across team members.

### Config File (`preprocessing_config.yaml`)

```yaml
cleaning:
  missing_strategy: auto
  numeric_strategy: median
  categorical_strategy: mode
  outlier_method: iforest
  outlier_contamination: 0.03  # More conservative
  remove_outliers: false       # Keep outliers
  scale_method: standard
  encode_method: label
  balance_method: oversample

detection:
  outlier_method: iforest
  contamination: 0.03
  zscore_threshold: 3.5
  imbalance_threshold: 0.2  # More tolerant

reporting:
  format: html
  include_plots: true
  output_dir: ./team_reports

logging:
  enabled: true
  level: INFO
  file: preprocessing.log
```

### Usage

```python
# Everyone on the team uses same config
prep = AutoPrepML(df, config_path='preprocessing_config.yaml')
clean_df, report = prep.clean(task='classification', target_col='label')
```

---

## Tutorial 6: Handling Large Datasets

### Scenario

Your CSV file is too large to fit in memory.

### Chunked Processing

```python
import pandas as pd
from autoprepml import AutoPrepML

chunk_size = 50000
output_file = 'large_data_cleaned.csv'
first_chunk = True

for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    # Process each chunk
    prep = AutoPrepML(chunk)
    clean_chunk, _ = prep.clean()
    
    # Append to output file
    mode = 'w' if first_chunk else 'a'
    clean_chunk.to_csv(output_file, mode=mode, header=first_chunk, index=False)
    first_chunk = False

print(f"Processed large file in chunks. Output: {output_file}")
```

---

## Tutorial 7: Integration with ML Workflow

### Scenario

Integrate AutoPrepML into your full ML pipeline.

```python
from autoprepml import AutoPrepML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load and preprocess
df = pd.read_csv('data.csv')
prep = AutoPrepML(df)
clean_df, report = prep.clean(task='classification', target_col='target')

# 2. Train/test split
X = clean_df.drop('target', axis=1)
y = clean_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 5. Save preprocessing report
prep.save_report('preprocessing_report.html')
```

---

## Best Practices

### 1. Always Save Reports

```python
prep.save_report('report.html')
```

Reports provide transparency and help debug issues.

### 2. Use Config Files for Production

```python
prep = AutoPrepML(df, config_path='production_config.yaml')
```

Ensures reproducibility across runs.

### 3. Validate Cleaned Data

```python
clean_df, report = prep.clean()
assert clean_df.isnull().sum().sum() == 0, "Still have missing values!"
```

Always verify cleaning was successful.

### 4. Version Your Configs

Store config files in version control alongside code.

### 5. Monitor Preprocessing Logs

```python
import logging
logging.basicConfig(filename='prep.log', level=logging.INFO)
```

Track what transformations were applied.

---

## Common Pitfalls

### ❌ Don't: Apply same config to all datasets

```python
# Bad - using same config for different datasets
prep1 = AutoPrepML(medical_data, config_path='config.yaml')
prep2 = AutoPrepML(financial_data, config_path='config.yaml')  # Same config!
```

### ✅ Do: Customize config per domain

```python
prep1 = AutoPrepML(medical_data, config_path='medical_config.yaml')
prep2 = AutoPrepML(financial_data, config_path='finance_config.yaml')
```

### ❌ Don't: Ignore detection results

```python
# Bad - clean without checking what's wrong
prep = AutoPrepML(df)
clean_df, _ = prep.clean()
```

### ✅ Do: Review detection first

```python
prep = AutoPrepML(df)
results = prep.detect()
print(results)  # Review before cleaning
clean_df, report = prep.clean()
```

---

## Need Help?

- Check the [API Reference](api_reference.md)
- See [Usage Guide](usage.md)
- Open an [issue on GitHub](https://github.com/yourusername/autoprepml/issues)
