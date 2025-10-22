"""Visualization functions for AutoPrepML"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


def plot_missing(df: pd.DataFrame, figsize: tuple = (10, 6)) -> str:
    """Generate bar plot of missing values by column.
    
    Args:
        df: Input DataFrame
        figsize: Figure size tuple (width, height)
        
    Returns:
        Base64-encoded PNG image string
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No missing values found', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
    else:
        fig, ax = plt.subplots(figsize=figsize)
        missing.plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return img_str


def plot_outliers(df: pd.DataFrame, outlier_indices: list = None, 
                  figsize: tuple = (12, 6)) -> str:
    """Generate box plots for numeric columns highlighting outliers.
    
    Args:
        df: Input DataFrame
        outlier_indices: List of outlier row indices to highlight
        figsize: Figure size tuple
        
    Returns:
        Base64-encoded PNG image string
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No numeric columns found', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
    else:
        n_cols = min(len(numeric_cols), 5)  # Limit to 5 columns
        cols_to_plot = numeric_cols[:n_cols]
        
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(cols_to_plot):
            axes[i].boxplot(df[col].dropna(), vert=True)
            axes[i].set_title(col, fontsize=10)
            axes[i].grid(axis='y', alpha=0.3)
        
        fig.suptitle('Outlier Detection - Box Plots', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return img_str


def plot_distributions(df: pd.DataFrame, figsize: tuple = (14, 10)) -> str:
    """Generate histograms for numeric columns.
    
    Args:
        df: Input DataFrame
        figsize: Figure size tuple
        
    Returns:
        Base64-encoded PNG image string
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No numeric columns found', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
    else:
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        # Flatten axes to always have 1D array
        axes = np.array(axes, ndmin=1).flatten()
        
        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_cols, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return img_str


def plot_correlation(df: pd.DataFrame, figsize: tuple = (10, 8)) -> str:
    """Generate correlation heatmap for numeric columns.
    
    Args:
        df: Input DataFrame
        figsize: Figure size tuple
        
    Returns:
        Base64-encoded PNG image string
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Need at least 2 numeric columns for correlation', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
    else:
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return img_str


def generate_all_plots(df: pd.DataFrame, outlier_indices: list = None) -> dict:
    """Generate all visualization plots.
    
    Args:
        df: Input DataFrame
        outlier_indices: Optional list of outlier indices
        
    Returns:
        Dictionary with plot names as keys and base64 image strings as values
    """
    return {
        'missing_plot': plot_missing(df),
        'outlier_plot': plot_outliers(df, outlier_indices),
        'distribution_plot': plot_distributions(df),
        'correlation_plot': plot_correlation(df)
    }
