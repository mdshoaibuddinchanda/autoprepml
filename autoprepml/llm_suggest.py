"""Optional LLM integration for AutoPrepML - provides AI-powered suggestions"""
from typing import Optional, Dict, Any
import pandas as pd


def suggest_fix(df: pd.DataFrame, column: Optional[str] = None, 
                issue_type: str = 'missing') -> str:
    """Generate LLM-powered suggestions for data cleaning.
    
    NOTE: This is a placeholder implementation. To enable LLM features:
    1. Install openai or ollama packages
    2. Set API keys in environment variables
    3. Uncomment and configure LLM client below
    
    Args:
        df: Input DataFrame
        column: Optional specific column to analyze
        issue_type: Type of issue ('missing', 'outlier', 'imbalance')
        
    Returns:
        Suggestion text string
    """
    # Placeholder implementation
    suggestions = {
        'missing': f"For missing values in column '{column}': Consider using median imputation for numeric data or mode for categorical. Check if missingness is random (MCAR) or systematic (MAR/MNAR).",
        'outlier': f"For outliers in column '{column}': Review if outliers are data errors or genuine extreme values. Consider robust scaling or capping at percentiles.",
        'imbalance': f"For class imbalance in column '{column}': Try SMOTE for oversampling minority class, or use class weights in your model. Stratified sampling is recommended."
    }
    
    return suggestions.get(issue_type, "No suggestion available for this issue type.")


def explain_cleaning_step(action: str, details: Dict[str, Any]) -> str:
    """Generate natural language explanation of a cleaning step.
    
    Args:
        action: Action type (e.g., 'imputed_missing', 'scaled_features')
        details: Dictionary with action details
        
    Returns:
        Human-readable explanation
    """
    explanations = {
        'imputed_missing': f"Filled missing values using {details.get('strategy', 'auto')} strategy. This replaces NaN values with statistically appropriate substitutes.",
        'removed_outliers': f"Removed {details.get('count', 0)} outlier rows that deviated significantly from normal data patterns.",
        'encoded_categorical': f"Converted categorical variables to numeric format using {details.get('method', 'label')} encoding for ML compatibility.",
        'scaled_features': f"Normalized numeric features using {details.get('method', 'standard')} scaling to ensure all features have similar ranges.",
        'balanced_classes': f"Adjusted class distribution using {details.get('method', 'oversample')} to create a balanced training set."
    }
    
    return explanations.get(action, f"Applied {action} transformation.")


# Future LLM integration example (commented out):
"""
import os
from openai import OpenAI

def suggest_fix_with_llm(df: pd.DataFrame, column: str, issue_type: str) -> str:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Build context
    col_info = {
        'dtype': str(df[column].dtype),
        'missing_pct': (df[column].isnull().sum() / len(df) * 100),
        'unique_values': df[column].nunique(),
        'sample_values': df[column].dropna().head(5).tolist()
    }
    
    prompt = f'''
    Given a DataFrame column with the following characteristics:
    - Data type: {col_info['dtype']}
    - Missing percentage: {col_info['missing_pct']:.2f}%
    - Unique values: {col_info['unique_values']}
    - Sample values: {col_info['sample_values']}
    
    Issue type: {issue_type}
    
    Provide a concise recommendation for handling this data quality issue.
    '''
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    
    return response.choices[0].message.content
"""
