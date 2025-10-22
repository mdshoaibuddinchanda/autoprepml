"""
AutoPrepML Demo Script
======================

This script demonstrates how to use AutoPrepML to preprocess data
using the Iris dataset as an example.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from autoprepml import AutoPrepML

def main():
    print("=" * 60)
    print("AutoPrepML Demo - Iris Dataset")
    print("=" * 60)
    
    # Load Iris dataset
    print("\n1. Loading Iris dataset...")
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    # Add some artificial missing values and outliers for demonstration
    print("\n2. Adding artificial data issues for demonstration...")
    df_dirty = df.copy()
    
    # Add missing values (10% random)
    np.random.seed(42)
    mask = np.random.rand(len(df_dirty), len(df_dirty.columns)) < 0.1
    df_dirty = df_dirty.mask(mask)
    
    # Add outliers
    df_dirty.loc[0, 'sepal length (cm)'] = 100.0
    df_dirty.loc[1, 'sepal width (cm)'] = -50.0
    
    print(f"   Original shape: {df.shape}")
    print(f"   Missing values added: {df_dirty.isnull().sum().sum()}")
    
    # Initialize AutoPrepML
    print("\n3. Initializing AutoPrepML...")
    prep = AutoPrepML(df_dirty)
    
    # Get summary
    print("\n4. Dataset Summary:")
    summary = prep.summary()
    print(f"   Shape: {summary['shape']}")
    print(f"   Numeric columns: {len(summary['numeric_columns'])}")
    print(f"   Categorical columns: {len(summary['categorical_columns'])}")
    
    # Detect issues
    print("\n5. Detecting data issues...")
    detection_results = prep.detect(target_col='target')
    
    print(f"   Missing values detected in {len(detection_results['missing_values'])} columns")
    print(f"   Outliers detected: {detection_results['outliers']['outlier_count']} rows")
    
    if detection_results.get('class_imbalance'):
        imbalance = detection_results['class_imbalance']
        print(f"   Class balance: {'Imbalanced' if imbalance['is_imbalanced'] else 'Balanced'}")
    
    # Clean data
    print("\n6. Cleaning data...")
    clean_df, report = prep.clean(task='classification', target_col='target')
    
    print(f"   Cleaned shape: {clean_df.shape}")
    print(f"   Missing values after cleaning: {clean_df.isnull().sum().sum()}")
    
    # Save results
    print("\n7. Saving results...")
    clean_df.to_csv('iris_cleaned.csv', index=False)
    prep.save_report('iris_report.html')
    
    print("   ✓ Cleaned data saved to: iris_cleaned.csv")
    print("   ✓ Report saved to: iris_report.html")
    
    print("\n" + "=" * 60)
    print("Demo complete! Check the report file for visualizations.")
    print("=" * 60)


if __name__ == '__main__':
    main()
