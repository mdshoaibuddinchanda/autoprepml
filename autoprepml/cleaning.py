"""Cleaning and transformation functions for AutoPrepML"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


def impute_missing(df: pd.DataFrame, strategy: str = 'auto', 
                   numeric_strategy: str = 'median',
                   categorical_strategy: str = 'mode') -> pd.DataFrame:
    """Impute missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'auto' (smart detection), 'median', 'mean', 'mode', 'drop'
        numeric_strategy: Strategy for numeric columns when strategy='auto'
        categorical_strategy: Strategy for categorical columns when strategy='auto'
        
    Returns:
        DataFrame with imputed values
    """
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() == 0:
            continue
        
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
            continue
        
        # Auto-detect strategy based on dtype
        if strategy == 'auto':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                current_strategy = numeric_strategy
            else:
                current_strategy = categorical_strategy
        else:
            current_strategy = strategy
        
        # Apply imputation
        if current_strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
            fill_value = df_clean[col].median()
        elif current_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
            fill_value = df_clean[col].mean()
        elif current_strategy == 'mode':
            mode_values = df_clean[col].mode()
            fill_value = mode_values.iloc[0] if len(mode_values) > 0 else ''
        else:
            fill_value = 0 if pd.api.types.is_numeric_dtype(df_clean[col]) else ''
        
        df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean


def impute_knn(df: pd.DataFrame, n_neighbors: int = 5, 
               exclude_cols: list = None) -> pd.DataFrame:
    """Impute missing values using K-Nearest Neighbors.
    
    KNN imputation fills missing values by finding the k nearest samples
    based on available features and using their values for imputation.
    More sophisticated than simple mean/median.
    
    Args:
        df: Input DataFrame
        n_neighbors: Number of neighboring samples to use for imputation
        exclude_cols: List of column names to exclude from imputation
        
    Returns:
        DataFrame with KNN-imputed values
        
    Example:
        >>> df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]})
        >>> imputed_df = impute_knn(df, n_neighbors=2)
    """
    df_clean = df.copy()
    exclude_cols = exclude_cols or []
    
    # Get numeric columns only (KNN works on numeric data)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_impute = [col for col in numeric_cols if col not in exclude_cols]
    
    if not cols_to_impute:
        return df_clean
    
    # Check if there are missing values to impute
    if df_clean[cols_to_impute].isnull().sum().sum() == 0:
        return df_clean
    
    # Apply KNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform')
    df_clean[cols_to_impute] = imputer.fit_transform(df_clean[cols_to_impute])
    
    return df_clean


def impute_iterative(df: pd.DataFrame, max_iter: int = 10, 
                     random_state: int = 42,
                     exclude_cols: list = None) -> pd.DataFrame:
    """Impute missing values using Iterative Imputer (MICE algorithm).
    
    Iterative imputation models each feature with missing values as a function
    of other features in a round-robin fashion. More accurate than simple
    imputation for datasets with complex relationships.
    
    Args:
        df: Input DataFrame
        max_iter: Maximum number of imputation rounds
        random_state: Random seed for reproducibility
        exclude_cols: List of column names to exclude from imputation
        
    Returns:
        DataFrame with iteratively imputed values
        
    Example:
        >>> df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]})
        >>> imputed_df = impute_iterative(df, max_iter=10)
    """
    df_clean = df.copy()
    exclude_cols = exclude_cols or []
    
    # Get numeric columns only
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_impute = [col for col in numeric_cols if col not in exclude_cols]
    
    if not cols_to_impute:
        return df_clean
    
    # Check if there are missing values to impute
    if df_clean[cols_to_impute].isnull().sum().sum() == 0:
        return df_clean
    
    # Apply iterative imputation
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    df_clean[cols_to_impute] = imputer.fit_transform(df_clean[cols_to_impute])
    
    return df_clean


def scale_features(df: pd.DataFrame, method: str = 'standard', 
                   exclude_cols: list = None) -> pd.DataFrame:
    """Scale numeric features.
    
    Args:
        df: Input DataFrame
        method: 'standard' (z-score normalization) or 'minmax' (0-1 scaling)
        exclude_cols: List of column names to exclude from scaling
        
    Returns:
        DataFrame with scaled numeric features
    """
    df_scaled = df.copy()
    exclude_cols = exclude_cols or []
    
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    if not cols_to_scale:
        return df_scaled
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    
    return df_scaled


def encode_categorical(df: pd.DataFrame, method: str = 'label', 
                      exclude_cols: list = None) -> pd.DataFrame:
    """Encode categorical features.
    
    Args:
        df: Input DataFrame
        method: 'label' (label encoding) or 'onehot' (one-hot encoding)
        exclude_cols: List of column names to exclude from encoding
        
    Returns:
        DataFrame with encoded categorical features
    """
    df_encoded = df.copy()
    exclude_cols = exclude_cols or []
    
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_to_encode = [col for col in categorical_cols if col not in exclude_cols]
    
    if not cols_to_encode:
        return df_encoded
    
    if method == 'label':
        for col in cols_to_encode:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    elif method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode, drop_first=True)
    
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return df_encoded


def balance_classes(df: pd.DataFrame, target_col: str, method: str = 'oversample') -> pd.DataFrame:
    """Balance class distribution in target column.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        method: 'oversample' (duplicate minority), 'undersample' (reduce majority)
        
    Returns:
        DataFrame with balanced classes
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    df_balanced = df.copy()
    value_counts = df_balanced[target_col].value_counts()
    
    if method == 'oversample':
        max_count = value_counts.max()
        dfs = []
        for class_value in value_counts.index:
            class_df = df_balanced[df_balanced[target_col] == class_value]
            count_diff = max_count - len(class_df)
            if count_diff > 0:
                oversampled = class_df.sample(n=count_diff, replace=True, random_state=42)
                dfs.extend([class_df, oversampled])
            else:
                dfs.append(class_df)
        df_balanced = pd.concat(dfs, ignore_index=True)
    elif method == 'undersample':
        min_count = value_counts.min()
        dfs = [
            df_balanced[df_balanced[target_col] == class_value].sample(n=min_count, random_state=42)
            for class_value in value_counts.index
        ]
        df_balanced = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


def balance_classes_smote(df: pd.DataFrame, target_col: str,
                          sampling_strategy: str = 'auto',
                          k_neighbors: int = 5,
                          random_state: int = 42) -> pd.DataFrame:
    """Balance classes using SMOTE (Synthetic Minority Over-sampling Technique).
    
    SMOTE creates synthetic samples by interpolating between existing minority
    class samples. More sophisticated than simple duplication - generates new,
    plausible data points rather than exact copies.
    
    Requires: pip install imbalanced-learn
    
    Args:
        df: Input DataFrame with numeric features
        target_col: Name of target column
        sampling_strategy: Strategy for resampling:
            - 'auto': resample all classes except majority to match majority
            - 'minority': resample only the minority class
            - float: ratio of minority to majority (e.g., 0.5 = half of majority)
        k_neighbors: Number of nearest neighbors for SMOTE algorithm
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with balanced classes using synthetic samples
        
    Example:
        >>> # Imbalanced dataset: 90 class 0, 10 class 1
        >>> df = pd.DataFrame({'A': range(100), 'B': range(100), 'label': [0]*90 + [1]*10})
        >>> balanced_df = balance_classes_smote(df, target_col='label')
        >>> # Result: ~90 class 0, ~90 class 1 (with synthetic samples)
        
    Raises:
        ImportError: If imbalanced-learn is not installed
        ValueError: If DataFrame contains non-numeric features
    """
    if not SMOTE_AVAILABLE:
        raise ImportError(
            "SMOTE requires imbalanced-learn library. Install it with:\n"
            "pip install imbalanced-learn"
        )

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Check if all features are numeric
    if (non_numeric := X.select_dtypes(exclude=[np.number]).columns.tolist()):
        raise ValueError(
            f"SMOTE requires all features to be numeric. "
            f"Found non-numeric columns: {non_numeric}. "
            f"Please encode categorical variables first using encode_categorical()."
        )

    # Apply SMOTE
    try:
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Combine features and target
        df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
        df_balanced[target_col] = y_resampled

        return df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    except ValueError as e:
        # Handle cases where SMOTE can't be applied (e.g., too few samples)
        raise ValueError(f"SMOTE balancing failed: {str(e)}") from e


def remove_outliers(df: pd.DataFrame, outlier_indices: list) -> pd.DataFrame:
    """Remove rows identified as outliers.
    
    Args:
        df: Input DataFrame
        outlier_indices: List of row indices to remove
        
    Returns:
        DataFrame with outliers removed
    """
    return df.drop(index=outlier_indices).reset_index(drop=True)
