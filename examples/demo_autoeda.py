"""
Demo: AutoEDA - Automated Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
from autoprepml import AutoEDA

# Create sample dataset
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'loan_amount': np.random.uniform(1000, 50000, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples),
    'default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
}

# Introduce some missing values
data['income'][np.random.choice(n_samples, 50, replace=False)] = np.nan
data['credit_score'][np.random.choice(n_samples, 30, replace=False)] = np.nan

# Create DataFrame
df = pd.DataFrame(data)

print("=" * 80)
print("AutoEDA Demo - Automated Exploratory Data Analysis")
print("=" * 80)

# Initialize AutoEDA
eda = AutoEDA(df)

# Run full analysis
print("\n📊 Running comprehensive EDA analysis...")
results = eda.analyze()

print("\n✅ Analysis complete!")

# Display automated insights
print("\n" + "=" * 80)
print("🔍 Automated Insights")
print("=" * 80)
insights = eda.get_insights()
for insight in insights:
    print(f"  {insight}")

# Get compact summary
print("\n" + "=" * 80)
print("📋 Quick Summary")
print("=" * 80)
summary = eda.get_summary()
print(f"  Dataset shape: {summary['dataset_shape']}")
print(f"  Numeric columns: {summary['n_numeric_columns']}")
print(f"  Categorical columns: {summary['n_categorical_columns']}")
print(f"  Missing values: {summary['total_missing_values']}")
print(f"  Duplicate rows: {summary['duplicate_rows']}")
print(f"  Columns with outliers: {summary['columns_with_outliers']}")
print(f"  High correlations found: {summary['high_correlations']}")

# Display missing values details
if results['missing_values']['missing_by_column']:
    print("\n" + "=" * 80)
    print("⚠️  Missing Values Details")
    print("=" * 80)
    for item in results['missing_values']['missing_by_column']:
        print(f"  {item['column']}: {item['missing_count']} ({item['missing_percentage']:.1f}%)")

# Display high correlations
if results.get('correlations', {}).get('high_correlations'):
    print("\n" + "=" * 80)
    print("🔗 High Correlations")
    print("=" * 80)
    for corr in results['correlations']['high_correlations']:
        print(f"  {corr['column1']} ↔️ {corr['column2']}: {corr['correlation']:.3f} ({corr['strength']})")

# Display outliers
if results.get('outliers'):
    print("\n" + "=" * 80)
    print("📈 Outliers Detected")
    print("=" * 80)
    for col, info in results['outliers'].items():
        print(f"  {col}: {info['iqr_outliers']} outliers ({info['iqr_percentage']:.1f}%)")

# Generate HTML report
print("\n" + "=" * 80)
print("📄 Generating Reports")
print("=" * 80)

eda.generate_report('eda_report.html', title='Loan Default Dataset - EDA Report')
print("  ✅ HTML report saved: eda_report.html")

# Export to JSON
eda.to_json('eda_results.json')
print("  ✅ JSON results saved: eda_results.json")

print("\n" + "=" * 80)
print("✨ Demo Complete!")
print("=" * 80)
print("\nNext steps:")
print("  1. Open 'eda_report.html' in your browser to see the interactive report")
print("  2. Check 'eda_results.json' for programmatic access to all analysis results")
print("  3. Use the insights to guide your preprocessing decisions")
