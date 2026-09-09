# API Reference

This page summarises the public API exposed by AutoPrepML 1.4.0. Type signatures are representative; consult the package source and docstrings for the complete contract.

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

## Modality and analysis classes

The package also exports the following high level classes:

* `TextPrepML`: clean, tokenize, and profile text columns.
* `TimeSeriesPrepML`: validate timestamps, fill gaps, and create temporal features.
* `GraphPrepML`: validate nodes and edges, calculate graph features, and report graph statistics.
* `ImagePrepML`: inspect, clean, transform, split, and report image datasets.
* `AutoEDA`: compute statistical summaries, correlations, distributions, outliers, and generated insights.
* `AutoFeatureEngine`: create, select, and rank engineered features.
* `InteractiveDashboard`: create Plotly visualisations and Streamlit applications.

### Chunk and streaming execution

* `iter_chunks(source, chunksize=10000)`: read DataFrame, CSV, JSON, JSONL, or Parquet input as bounded chunks.
* `iter_processed_chunks(source, processor, chunksize=10000, n_jobs=1, backend='thread')`: process chunks in deterministic order with bounded parallel work.
* `process_chunks(...)`: materialise processed chunks as one DataFrame.
* `stream_process(...)` and `write_stream(...)`: process and persist chunks without materialising the complete output.

### Storage adapters

`LocalStorageAdapter` provides atomic local table writes. `InMemoryStorageAdapter`
is useful for tests and notebooks. `FsspecStorageAdapter` is optional and
supports fsspec-backed URLs when the corresponding filesystem package is
installed. Implement `StorageAdapter` to add another backend.

### Experiment and model integrations

`LocalExperimentTracker` stores run parameters, numeric metrics, and copied
artifacts as JSON manifests. `MLflowExperimentTracker` is loaded lazily and is
available when MLflow is installed. `make_preprocessing_pipeline` and
`make_model_pipeline` build scikit-learn pipelines that fit preprocessing
state on training data only.

### Normalization

`TabularNormalizer(method='standard', columns=None)` is a fitted sklearn
transformer for numeric DataFrame columns. Supported methods are `standard`,
`minmax`, `robust`, and `maxabs`. Call `fit_transform` on training rows and
`transform` on later rows. Missing and non-finite values are rejected so that
imputation and schema validation cannot be skipped accidentally.

`fit_image_statistics(images)` computes training-only per-channel statistics.
`normalize_image_array(images, mode='zero_one', mean=None, std=None)` supports
`zero_one`, `minus_one_one`, `standard`, and `none`. Standard image mode uses
explicit per-channel training statistics. `denormalize_image_array` converts
normalized arrays back to uint8 pixels for image persistence.

See the [advanced features guide](ADVANCED_FEATURES.md), [usage guide](usage.md), and [tutorials](tutorials.md) for examples.

### Time series feature safety

`TimeSeriesPrepML.add_rolling_features(windows=None, functions=None, forecast_safe=True)` shifts the source series by one row by default. This prevents a forecasting feature from including the value it is intended to predict. Set `forecast_safe=False` when the rolling statistic is descriptive and contemporaneous by design. Supported functions are `mean`, `std`, `min`, and `max`.

### Graph direction

`GraphPrepML(..., directed=True)` treats `(source, target)` and `(target, source)` as different edges. Set `directed=False` for an undirected graph; duplicate detection, edge counts, adjacency conversion, degree features, and density then use undirected semantics consistently. The default remains directed for compatibility.

### Image augmentation

`ImagePrepML.clean(augment=True, augmentation_config=...)` supports deterministic NumPy transforms without an additional augmentation dependency. Configuration keys are `horizontal_flip`, `vertical_flip`, `rotations` (90-degree increments), and `include_original`.

`ImagePrepML` also accepts `normalization_mode`, `normalization_mean`, and
`normalization_std`. The default remains `zero_one` for compatibility.


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
- `method`: 'standard', 'minmax', 'robust', or 'maxabs'
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


## Configuration Module

### Functions

#### `load_config(config_path: Optional[str] = None) -> Dict`

Load configuration from YAML or JSON file.

**Returns:** Configuration dictionary

#### `save_config(config: Dict, output_path: str) -> None`

Save configuration to a YAML or JSON file. The format is selected from the output extension.

#### `get_default_config() -> Dict`

Return default configuration.


## Reporting Module

### Functions

#### `generate_json_report(report: Dict) -> str`

Generate JSON report from report dictionary.

**Returns:** JSON string

#### `generate_html_report(report: Dict) -> str`

Generate HTML report from report dictionary.

**Returns:** HTML string


## LLM Suggestions Module

### Functions

#### `suggest_fix(df: pd.DataFrame, column: Optional[str] = None, issue_type: str = 'missing') -> str`

Generate model assisted suggestions for a data quality issue. Provider credentials and optional dependencies are required.

**Returns:** Suggestion text string

#### `explain_cleaning_step(action: str, details: Dict) -> str`

Generate natural language explanation of cleaning step.

**Returns:** Human-readable explanation


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
