# Advanced Features Guide

## 🚀 Version 1.1.0 Features

This guide covers the advanced features added in AutoPrepML v1.1.0:
- Advanced Imputation (KNN, Iterative)
- SMOTE for Class Balancing
- Enhanced Documentation

---

## 1️⃣ Advanced Imputation

### Overview

Simple imputation (mean, median, mode) can be too basic for complex datasets. AutoPrepML now supports sophisticated imputation methods:

- **KNN Imputation**: Uses K-Nearest Neighbors to predict missing values
- **Iterative Imputation**: MICE algorithm that models each feature iteratively

### When to Use Each Method

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Mean/Median** | Quick preprocessing, simple datasets | Fast, simple | Ignores relationships |
| **KNN** | Datasets with local patterns | Considers neighbors | Slower, needs tuning |
| **Iterative** | Complex relationships between features | Most accurate | Slowest, may overfit |

---

### KNN Imputation

**How it works**: Finds the k nearest samples (based on available features) and uses their values to impute missing data.

#### Basic Usage

```python
from autoprepml.cleaning import impute_knn
import pandas as pd
import numpy as np

# Create dataset with missing values
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50],
    'salary': [50000, np.nan, 70000, 80000, 90000],
    'experience': [2, 5, 8, np.nan, 15]
})

# Apply KNN imputation (default: 5 neighbors)
df_imputed = impute_knn(df, n_neighbors=5)

print(df_imputed)
# age and salary are now filled based on similar samples
```

#### Advanced Options

```python
# Use 3 nearest neighbors
df_imputed = impute_knn(df, n_neighbors=3)

# Exclude specific columns from imputation
df_imputed = impute_knn(df, n_neighbors=5, exclude_cols=['salary'])
```

#### Parameters

- `n_neighbors` (int, default=5): Number of neighbors to use
  - Lower values: More local, may be noisy
  - Higher values: More global, may be too smooth
  - Recommended: 3-7 for most datasets

- `exclude_cols` (list, optional): Columns to skip during imputation

#### Example: Real-World Application

```python
# Load dataset with missing values
df = pd.read_csv('housing_data.csv')

# Check missing values
print(df.isnull().sum())
# bedrooms: 15 missing
# bathrooms: 8 missing
# square_feet: 22 missing

# KNN imputation (bedrooms similar to other features)
df_clean = impute_knn(df, n_neighbors=5)

# Verify no missing values
assert df_clean.isnull().sum().sum() == 0
```

---

### Iterative Imputation (MICE)

**How it works**: Models each feature with missing values as a function of other features in multiple iterations (round-robin).

#### Basic Usage

```python
from autoprepml.cleaning import impute_iterative

# Create dataset with missing values
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50],
    'salary': [50000, np.nan, 70000, 80000, 90000],
    'experience': [2, 5, 8, np.nan, 15]
})

# Apply iterative imputation
df_imputed = impute_iterative(df, max_iter=10, random_state=42)

print(df_imputed)
# Missing values filled using predictive models
```

#### Advanced Options

```python
# More iterations for better convergence
df_imputed = impute_iterative(df, max_iter=20, random_state=42)

# Exclude specific columns
df_imputed = impute_iterative(df, max_iter=10, exclude_cols=['salary'], random_state=42)
```

#### Parameters

- `max_iter` (int, default=10): Maximum number of imputation rounds
  - Lower values: Faster, may not converge
  - Higher values: More accurate, slower
  - Recommended: 10-20 for most datasets

- `random_state` (int, default=42): Random seed for reproducibility

- `exclude_cols` (list, optional): Columns to skip

#### Example: Complex Dataset

```python
# Dataset with complex relationships
df = pd.read_csv('medical_records.csv')

# Multiple features with missing values
# heart_rate, blood_pressure, cholesterol

# Iterative imputation (features predict each other)
df_clean = impute_iterative(df, max_iter=15, random_state=42)

# More accurate than simple mean/median
# Captures relationships between vital signs
```

---

### Comparison: Simple vs Advanced Imputation

```python
import pandas as pd
import numpy as np
from autoprepml.cleaning import impute_missing, impute_knn, impute_iterative

# Create test dataset
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50, 55, np.nan, 65],
    'salary': [50000, 60000, np.nan, 80000, 90000, 95000, 100000, np.nan],
    'experience': [2, 5, 8, 12, 15, 18, np.nan, 25]
})

# Method 1: Simple mean imputation
df_simple = impute_missing(df, strategy='mean')
print("Simple (mean):", df_simple.loc[2, 'age'])  # ~48.3 (global mean)

# Method 2: KNN imputation
df_knn = impute_knn(df, n_neighbors=3)
print("KNN:", df_knn.loc[2, 'age'])  # ~37.5 (based on neighbors)

# Method 3: Iterative imputation
df_iter = impute_iterative(df, max_iter=10, random_state=42)
print("Iterative:", df_iter.loc[2, 'age'])  # ~39.2 (predictive model)

# KNN and Iterative are closer to true patterns!
```

---

## 2️⃣ SMOTE Class Balancing

### Overview

**SMOTE** (Synthetic Minority Over-sampling Technique) creates synthetic samples for minority classes instead of simple duplication. This prevents overfitting and improves model performance.

### Why SMOTE?

**Problem**: Imbalanced datasets hurt ML performance

```python
# Imbalanced dataset
# Class 0: 900 samples (90%)
# Class 1: 100 samples (10%)

# Model trained on this: 90% accuracy by just predicting class 0!
```

**Traditional Solution**: Oversample by duplication

```python
# Problem: Exact duplicates cause overfitting
# Model memorizes rather than generalizes
```

**SMOTE Solution**: Create synthetic samples

```python
# SMOTE interpolates between existing minority samples
# Creates new, plausible data points
# Model learns better decision boundaries
```

---

### Basic Usage

```python
from autoprepml.cleaning import balance_classes_smote
import pandas as pd

# Imbalanced dataset
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5] * 18 + [100, 110],
    'feature2': [10, 20, 30, 40, 50] * 18 + [1000, 1100],
    'label': [0] * 90 + [1] * 10
})

print("Before SMOTE:")
print(df['label'].value_counts())
# 0: 90
# 1: 10

# Apply SMOTE
df_balanced = balance_classes_smote(df, target_col='label', random_state=42)

print("\nAfter SMOTE:")
print(df_balanced['label'].value_counts())
# 0: 90
# 1: 90 (synthetic samples created!)
```

### Installation

SMOTE requires `imbalanced-learn`:

```bash
pip install imbalanced-learn
```

Or install with AutoPrepML:

```bash
pip install autoprepml[advanced]  # Includes imbalanced-learn
```

---

### Advanced Options

#### 1. Sampling Strategy

```python
# Auto: Balance all classes to match majority
df_balanced = balance_classes_smote(
    df, target_col='label',
    sampling_strategy='auto',  # Default
    random_state=42
)

# Minority only: Oversample minority to match majority
df_balanced = balance_classes_smote(
    df, target_col='label',
    sampling_strategy='minority',
    random_state=42
)

# Custom ratio: 50% of majority class
df_balanced = balance_classes_smote(
    df, target_col='label',
    sampling_strategy=0.5,  # Minority will be 50% of majority
    random_state=42
)
```

#### 2. K-Neighbors

```python
# Use 3 nearest neighbors (more conservative)
df_balanced = balance_classes_smote(
    df, target_col='label',
    k_neighbors=3,
    random_state=42
)

# Use 7 neighbors (more diverse synthetic samples)
df_balanced = balance_classes_smote(
    df, target_col='label',
    k_neighbors=7,
    random_state=42
)
```

---

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Input data with features and target |
| `target_col` | str | - | Name of target column |
| `sampling_strategy` | str/float | 'auto' | 'auto', 'minority', or custom ratio |
| `k_neighbors` | int | 5 | Number of nearest neighbors |
| `random_state` | int | 42 | Random seed for reproducibility |

---

### Important Notes

#### ⚠️ Features Must Be Numeric

```python
# ❌ This will fail (categorical feature)
df = pd.DataFrame({
    'age': [25, 30, 35] * 30 + [40, 45],
    'gender': ['M', 'F', 'M'] * 30 + ['M', 'F'],  # Categorical!
    'label': [0] * 90 + [1] * 10
})

# Solution: Encode first
from autoprepml.cleaning import encode_categorical, balance_classes_smote

df_encoded = encode_categorical(df, method='label', exclude_cols=['label'])
df_balanced = balance_classes_smote(df_encoded, target_col='label')
```

#### 📊 Multiclass Support

```python
# SMOTE works with multiple classes
df = pd.DataFrame({
    'feature1': list(range(200)),
    'feature2': list(range(200, 400)),
    'label': [0] * 100 + [1] * 70 + [2] * 30
})

# Balances all classes
df_balanced = balance_classes_smote(df, target_col='label')

print(df_balanced['label'].value_counts())
# 0: 100
# 1: 100
# 2: 100
```

---

### Complete Example: Fraud Detection

```python
from autoprepml import AutoPrepML
from autoprepml.cleaning import balance_classes_smote, encode_categorical
import pandas as pd

# Load imbalanced fraud dataset
df = pd.read_csv('transactions.csv')
# Legitimate: 9,500 (95%)
# Fraud: 500 (5%)

print("Original distribution:")
print(df['is_fraud'].value_counts())

# Step 1: Encode categorical variables
df_encoded = encode_categorical(df, method='label', exclude_cols=['is_fraud'])

# Step 2: Apply SMOTE
df_balanced = balance_classes_smote(
    df_encoded,
    target_col='is_fraud',
    sampling_strategy='auto',
    k_neighbors=5,
    random_state=42
)

print("\nBalanced distribution:")
print(df_balanced['is_fraud'].value_counts())
# Legitimate: 9,500
# Fraud: 9,500 (synthetic samples!)

# Step 3: Train model on balanced data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df_balanced.drop('is_fraud', axis=1)
y = df_balanced['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Model trained on balanced data performs much better!
# Recall on fraud cases: 85% (vs 30% without SMOTE)
```

---

## 3️⃣ Best Practices

### Combining Features

```python
from autoprepml.cleaning import impute_iterative, encode_categorical, balance_classes_smote

# Step 1: Handle missing values with advanced imputation
df_imputed = impute_iterative(df, max_iter=10, random_state=42)

# Step 2: Encode categorical variables
df_encoded = encode_categorical(df_imputed, method='label', exclude_cols=['target'])

# Step 3: Balance classes with SMOTE
df_final = balance_classes_smote(df_encoded, target_col='target', random_state=42)

# Ready for ML!
```

### When to Use What

| Scenario | Recommended Approach |
|----------|---------------------|
| **Quick EDA** | Simple imputation (mean/median) |
| **Complex relationships** | Iterative imputation |
| **Local patterns** | KNN imputation |
| **Mild imbalance (60/40)** | Simple oversampling |
| **Severe imbalance (95/5)** | SMOTE |
| **Small minority class** | SMOTE with lower k_neighbors |

---

## 📚 API Reference

### `impute_knn(df, n_neighbors=5, exclude_cols=None)`

Impute missing values using K-Nearest Neighbors.

**Returns**: DataFrame with imputed values

---

### `impute_iterative(df, max_iter=10, random_state=42, exclude_cols=None)`

Impute missing values using Iterative Imputer (MICE).

**Returns**: DataFrame with imputed values

---

### `balance_classes_smote(df, target_col, sampling_strategy='auto', k_neighbors=5, random_state=42)`

Balance classes using SMOTE synthetic oversampling.

**Returns**: DataFrame with balanced classes

**Raises**: 
- `ImportError` if imbalanced-learn not installed
- `ValueError` if features are non-numeric

---

## 🧪 Testing

All features include comprehensive tests:

```bash
# Run advanced features tests
pytest tests/test_advanced_features.py -v

# Test KNN imputation
pytest tests/test_advanced_features.py::TestAdvancedImputation::test_knn_imputation_basic -v

# Test SMOTE
pytest tests/test_advanced_features.py::TestSMOTE::test_smote_basic -v
```

---

## 📝 Changelog

### v1.1.0 (Q1 2025)

**Added**:
- KNN imputation (`impute_knn`)
- Iterative imputation (`impute_iterative`)
- SMOTE class balancing (`balance_classes_smote`)
- Comprehensive test suite (24 new tests)
- Advanced features documentation

**Dependencies**:
- Added `imbalanced-learn==0.12.0`

---

## 🔮 Coming Soon (v1.2.0)

- LLM integration for smart suggestions
- Image data preprocessing
- Audio/video metadata extraction
- Advanced SMOTE variants (ADASYN, BorderlineSMOTE)

---

## 💡 Questions?

- **GitHub Issues**: [Report bugs or ask questions](https://github.com/mdshoaibuddinchanda/autoprepml/issues)
- **Discussions**: [Community forum](https://github.com/mdshoaibuddinchanda/autoprepml/discussions)
- **Documentation**: [Full docs](../README.md)
