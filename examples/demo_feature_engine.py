"""
Demo: AutoFeatureEngine - Automated Feature Engineering
"""
import pandas as pd
import numpy as np
from autoprepml import AutoFeatureEngine, auto_feature_engineering

# Create sample dataset
np.random.seed(42)
n_samples = 500

data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'debt': np.random.uniform(0, 30000, n_samples),
    'years_employed': np.random.randint(0, 40, n_samples),
    'num_accounts': np.random.randint(1, 10, n_samples),
    'target': np.random.choice([0, 1], n_samples)
}

df = pd.DataFrame(data)

print("=" * 80)
print("AutoFeatureEngine Demo - Automated Feature Engineering")
print("=" * 80)

print(f"\n📊 Original Dataset:")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Initialize Feature Engine
fe = AutoFeatureEngine(df, target_column='target')

# 1. Create Interaction Features
print("\n" + "=" * 80)
print("🔧 Creating Interaction Features")
print("=" * 80)
fe.create_interactions(max_interactions=5)
print(f"  ✅ Created {fe.get_feature_log()[-1]['n_features_created']} interaction features")

# 2. Create Ratio Features
print("\n🔧 Creating Ratio Features")
print("=" * 80)
fe.create_ratio_features(max_ratios=5)
print(f"  ✅ Created {fe.get_feature_log()[-1]['n_features_created']} ratio features")

# 3. Create Aggregation Features
print("\n🔧 Creating Aggregation Features")
print("=" * 80)
fe.create_aggregation_features(operations=['sum', 'mean', 'std', 'min', 'max'])
print(f"  ✅ Created {fe.get_feature_log()[-1]['n_features_created']} aggregation features")

# 4. Create Polynomial Features
print("\n🔧 Creating Polynomial Features")
print("=" * 80)
fe.create_polynomial_features(degree=2, interaction_only=True)
print(f"  ✅ Created {fe.get_feature_log()[-1]['n_features_created']} polynomial features")

# 5. Create Binned Features
print("\n🔧 Creating Binned Features")
print("=" * 80)
fe.create_binned_features(n_bins=5, strategy='quantile')
print(f"  ✅ Created {fe.get_feature_log()[-1]['n_features_created']} binned features")

# Get summary
print("\n" + "=" * 80)
print("📋 Feature Engineering Summary")
print("=" * 80)
summary = fe.get_summary()
print(f"  Original features: {summary['original_features']}")
print(f"  Current features: {summary['current_features']}")
print(f"  Features created: {summary['features_created']}")
print(f"  Operations performed: {summary['operations_performed']}")

print("\n  Operations breakdown:")
for op, count in summary['operations_breakdown'].items():
    print(f"    - {op}: {count}x")

# Get engineered dataset
df_engineered = fe.get_features()
print(f"\n  ✅ Final dataset shape: {df_engineered.shape}")

# Feature importance
print("\n" + "=" * 80)
print("📊 Top 10 Features by Importance")
print("=" * 80)
importance = fe.get_feature_importance(task='classification')
print(importance.head(10).to_string(index=False))

# Feature selection (select top k features)
print("\n" + "=" * 80)
print("🎯 Feature Selection")
print("=" * 80)
print(f"  Before selection: {len(df_engineered.columns) - 1} features (excluding target)")

fe_selected = AutoFeatureEngine(df_engineered, target_column='target')
fe_selected.select_features(method='mutual_info', k=15, task='classification')
df_selected = fe_selected.get_features()

print(f"  After selection: {len(df_selected.columns) - 1} features (excluding target)")
print(f"\n  Selected features: {[col for col in df_selected.columns if col != 'target']}")

# Convenience function demo
print("\n" + "=" * 80)
print("⚡ Quick Feature Engineering (Convenience Function)")
print("=" * 80)

# Reset to original data
df_quick, quick_summary = auto_feature_engineering(
    df,
    target_column='target',
    max_features=30,
    include_polynomials=True,
    include_interactions=True,
    include_ratios=False,
    include_aggregations=True
)

print(f"  ✅ Created {quick_summary['features_created']} features automatically")
print(f"  Final shape: {df_quick.shape}")

# Save engineered features
df_selected.to_csv('engineered_features.csv', index=False)
print("\n" + "=" * 80)
print("💾 Saving Results")
print("=" * 80)
print("  ✅ Engineered features saved to: engineered_features.csv")

# Feature log
print("\n" + "=" * 80)
print("📝 Feature Engineering Log")
print("=" * 80)
for i, log_entry in enumerate(fe.get_feature_log(), 1):
    print(f"  Step {i}: {log_entry['operation']}")
    print(f"    Features created: {log_entry.get('n_features_created', 'N/A')}")

print("\n" + "=" * 80)
print("✨ Demo Complete!")
print("=" * 80)
print("\nKey takeaways:")
print("  - Created multiple types of features automatically")
print("  - Used feature importance to identify best features")
print("  - Selected top k features for modeling")
print("  - Can use convenience function for quick feature engineering")
print("\nNext steps:")
print("  - Use engineered features for model training")
print("  - Experiment with different feature combinations")
print("  - Try datetime feature extraction if you have temporal data")
