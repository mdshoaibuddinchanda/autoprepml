"""Cleaning and transformation functions for AutoPrepML"""
from typing import Dict, Any, Literal, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


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
    
    match method:
        case 'oversample':
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
    
        case 'undersample':
            min_count = value_counts.min()
            dfs = [
                df_balanced[df_balanced[target_col] == class_value].sample(n=min_count, random_state=42)
                for class_value in value_counts.index
            ]
            df_balanced = pd.concat(dfs, ignore_index=True)
    
        case _:
            raise ValueError(f"Unknown balancing method: {method}")
    
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


def remove_outliers(df: pd.DataFrame, outlier_indices: list) -> pd.DataFrame:
    """Remove rows identified as outliers.
    
    Args:
        df: Input DataFrame
        outlier_indices: List of row indices to remove
        
    Returns:
        DataFrame with outliers removed
    """
    return df.drop(index=outlier_indices).reset_index(drop=True)
