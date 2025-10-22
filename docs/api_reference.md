# API Reference

Complete API documentation for AutoPrepML.

## Core Module

### `AutoPrepML`

Main class for data preprocessing pipeline.

#### Constructor

```python
AutoPrepML(df: pd.DataFrame, config: Optional[Dict] = None, config_path: Optional[str] = None)
```

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame to preprocess
- `config` (dict, optional): Configuration dictionary
- `config_path` (str, optional): Path to YAML/JSON config file

**Attributes:**
- `original_df`: Original input DataFrame (copy)
- `df`: Current DataFrame state
- `cleaned_df`: Cleaned DataFrame after processing
- `log`: List of processing steps
- `detection_results`: Detection results dictionary
- `plots`: Generated plots dictionary

#### Methods

##### `detect(target_col: Optional[str] = None) -> Dict`

Run all detection functions.

**Returns:** Dictionary with detection results

##### `clean(task: Optional[str] = None, target_col: Optional[str] = None, auto: bool = True) -> Tuple[pd.DataFrame, Dict]`

Clean the dataset automatically.

**Parameters:**
- `task`: 'classification', 'regression', or None
- `target_col`: Name of target column
- `auto`: Apply all cleaning steps automatically

**Returns:** Tuple of (cleaned_df, report_dict)

##### `summary() -> Dict`

Get quick dataset summary.

**Returns:** Dictionary with shape, columns, dtypes, missing values

##### `report(include_plots: bool = True) -> Dict`

Generate comprehensive preprocessing report.

**Returns:** Complete report dictionary

##### `save_report(output_path: str) -> None`

Save report to file (.json or .html).

---

## Detection Module

### Functions

#### `detect_missing(df: pd.DataFrame) -> Dict[str, Any]`

Detect missing values in DataFrame.

**Returns:** Dict with column names and missing statistics

#### `detect_outliers(df: pd.DataFrame, method: str = 'iforest', contamination: float = 0.05, threshold: float = 3.0) -> Dict`

Detect outliers in numeric columns.

**Parameters:**
- `method`: 'iforest' or 'zscore'
- `contamination`: Expected outlier proportion (for iforest)
- `threshold`: Z-score threshold (for zscore)

**Returns:** Dict with outlier count and indices

#### `detect_imbalance(df: pd.DataFrame, target_col: str, threshold: float = 0.3) -> Dict`

Detect class imbalance in target column.

**Returns:** Dict with class distribution and imbalance metrics

#### `detect_all(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict`

Run all detection functions.

**Returns:** Complete detection results

---

## Cleaning Module

### Functions

#### `impute_missing(df: pd.DataFrame, strategy: str = 'auto', numeric_strategy: str = 'median', categorical_strategy: str = 'mode') -> pd.DataFrame`

Impute missing values.

**Parameters:**
- `strategy`: 'auto', 'median', 'mean', 'mode', 'drop'
- `numeric_strategy`: Strategy for numeric columns
- `categorical_strategy`: Strategy for categorical columns

**Returns:** DataFrame with imputed values

#### `scale_features(df: pd.DataFrame, method: str = 'standard', exclude_cols: list = None) -> pd.DataFrame`

Scale numeric features.

**Parameters:**
- `method`: 'standard' or 'minmax'
- `exclude_cols`: Columns to exclude from scaling

**Returns:** DataFrame with scaled features

#### `encode_categorical(df: pd.DataFrame, method: str = 'label', exclude_cols: list = None) -> pd.DataFrame`

Encode categorical features.

**Parameters:**
- `method`: 'label' or 'onehot'
- `exclude_cols`: Columns to exclude from encoding

**Returns:** DataFrame with encoded features

#### `balance_classes(df: pd.DataFrame, target_col: str, method: str = 'oversample') -> pd.DataFrame`

Balance class distribution.

**Parameters:**
- `method`: 'oversample' or 'undersample'

**Returns:** DataFrame with balanced classes

#### `remove_outliers(df: pd.DataFrame, outlier_indices: list) -> pd.DataFrame`

Remove rows identified as outliers.

**Returns:** DataFrame with outliers removed

---

## Visualization Module

### Functions

#### `plot_missing(df: pd.DataFrame, figsize: tuple = (10, 6)) -> str`

Generate bar plot of missing values.

**Returns:** Base64-encoded PNG image string

#### `plot_outliers(df: pd.DataFrame, outlier_indices: list = None, figsize: tuple = (12, 6)) -> str`

Generate box plots for outlier detection.

**Returns:** Base64-encoded PNG image string

#### `plot_distributions(df: pd.DataFrame, figsize: tuple = (14, 10)) -> str`

Generate histograms for numeric columns.

**Returns:** Base64-encoded PNG image string

#### `plot_correlation(df: pd.DataFrame, figsize: tuple = (10, 8)) -> str`

Generate correlation heatmap.

**Returns:** Base64-encoded PNG image string

#### `generate_all_plots(df: pd.DataFrame, outlier_indices: list = None) -> dict`

Generate all visualization plots.

**Returns:** Dict with plot names and base64 images

---

## Configuration Module

### Functions

#### `load_config(config_path: Optional[str] = None) -> Dict`

Load configuration from YAML or JSON file.

**Returns:** Configuration dictionary

#### `save_config(config: Dict, output_path: str) -> None`

Save configuration to YAML file.

#### `get_default_config() -> Dict`

Return default configuration.

---

## Reporting Module

### Functions

#### `generate_json_report(report: Dict) -> str`

Generate JSON report from report dictionary.

**Returns:** JSON string

#### `generate_html_report(report: Dict) -> str`

Generate HTML report from report dictionary.

**Returns:** HTML string

---

## LLM Suggestions Module

### Functions

#### `suggest_fix(df: pd.DataFrame, column: Optional[str] = None, issue_type: str = 'missing') -> str`

Generate AI-powered suggestions for data cleaning (placeholder).

**Returns:** Suggestion text string

#### `explain_cleaning_step(action: str, details: Dict) -> str`

Generate natural language explanation of cleaning step.

**Returns:** Human-readable explanation

---

## Configuration Schema

Default configuration structure:

```python
{
    'cleaning': {
        'missing_strategy': 'auto',
        'numeric_strategy': 'median',
        'categorical_strategy': 'mode',
        'outlier_method': 'iforest',
        'outlier_contamination': 0.05,
        'remove_outliers': False,
        'scale_method': 'standard',
        'encode_method': 'label',
        'balance_method': 'oversample'
    },
    'detection': {
        'outlier_method': 'iforest',
        'contamination': 0.05,
        'zscore_threshold': 3.0,
        'imbalance_threshold': 0.3
    },
    'reporting': {
        'format': 'html',
        'include_plots': True,
        'plot_style': 'seaborn',
        'output_dir': './reports'
    },
    'logging': {
        'enabled': True,
        'level': 'INFO',
        'file': 'autoprepml.log'
    }
}
```
