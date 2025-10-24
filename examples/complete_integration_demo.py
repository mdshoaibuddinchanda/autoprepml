"""
Complete Example: AutoPrepML with LLM Integration (v1.2.0)

This example demonstrates the full integration of AutoPrepML with LLM-powered
suggestions for data preprocessing.
"""

import pandas as pd
import numpy as np
from autoprepml import AutoPrepML, LLMSuggestor

# Create sample dataset with various data quality issues
print("=" * 70)
print("AutoPrepML v1.2.0 - Complete Integration Example")
print("=" * 70)

# Sample data with issues
data = {
    'age': [25, 30, np.nan, 45, 50, 35, 28, np.nan, 42, 55],
    'salary': [50000, 60000, np.nan, 80000, 90000, 70000, 55000, 65000, np.nan, 95000],
    'department': ['HR', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance'],
    'experience': [2, 5, 3, 15, 20, 10, 3, 7, 12, 18],
    'label': [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print("\n📊 Original Dataset:")
print(df)
print(f"\nShape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# ============================================================================
# Example 1: Basic AutoPrepML Usage (No LLM)
# ============================================================================
print("\n" + "="*70)
print("Example 1: Basic Usage (Traditional)")
print("="*70)

prep_basic = AutoPrepML(df)

# Detect issues
detection_results = prep_basic.detect(target_col='label')
print(f"\n✓ Detected {len(detection_results['missing_values'])} columns with missing values")
print(f"✓ Detected {detection_results['outliers']['outlier_count']} outliers")

# Clean the data
clean_df_basic, report_basic = prep_basic.clean(
    task='classification',
    target_col='label'
)

print(f"\n✓ Cleaned dataset shape: {clean_df_basic.shape}")
print("✓ All preprocessing steps logged")

# ============================================================================
# Example 2: With Advanced Features (v1.1.0)
# ============================================================================
print("\n" + "="*70)
print("Example 2: Advanced Features (KNN Imputation, SMOTE)")
print("="*70)

prep_advanced = AutoPrepML(df)
prep_advanced.detect(target_col='label')

clean_df_advanced, report_advanced = prep_advanced.clean(
    task='classification',
    target_col='label',
    use_advanced=True,
    imputation_method='knn',  # Advanced KNN imputation
    balance_method='smote'     # SMOTE for class balancing
)

print(f"\n✓ Used KNN imputation for missing values")
print("✓ Applied SMOTE for class balancing")
print(f"✓ Final shape: {clean_df_advanced.shape}")

# ============================================================================
# Example 3: With LLM Integration (v1.2.0)
# ============================================================================
print("\n" + "="*70)
print("Example 3: AI-Powered Suggestions (LLM Integration)")
print("="*70)

# Option A: Use AutoPrepML with LLM enabled
print("\n🤖 Initializing with LLM support (Ollama - Local)...")

try:
    # Using Ollama (local, no API key needed)
    prep_llm = AutoPrepML(
        df, 
        enable_llm=True, 
        llm_provider='ollama'
    )
    
    print("✓ LLM support initialized!")
    
    # Get AI analysis of the dataset
    print("\n📊 Getting AI analysis of dataset...")
    analysis = prep_llm.analyze_with_llm(
        task='classification',
        target_col='label'
    )
    print("\n" + "-"*70)
    print("AI Analysis:")
    print("-"*70)
    print(analysis)
    
    # Get suggestions for specific issues
    print("\n" + "-"*70)
    print("Getting suggestions for missing values in 'age' column...")
    print("-"*70)
    age_suggestions = prep_llm.get_llm_suggestions(
        column='age',
        issue_type='missing'
    )
    print(age_suggestions)
    
    # Get feature engineering suggestions
    print("\n" + "-"*70)
    print("Getting feature engineering suggestions...")
    print("-"*70)
    feature_ideas = prep_llm.get_feature_suggestions(
        task='classification',
        target_col='label'
    )
    print("\nSuggested Features:")
    for i, feature in enumerate(feature_ideas, 1):
        print(f"  {i}. {feature}")
    
    # Clean with advanced methods
    print("\n" + "-"*70)
    print("Cleaning with advanced methods...")
    print("-"*70)
    clean_df_llm, report_llm = prep_llm.clean(
        task='classification',
        target_col='label',
        use_advanced=True,
        imputation_method='iterative',
        balance_method='smote'
    )
    
    # Explain what was done
    print("\n" + "-"*70)
    print("Explaining preprocessing steps...")
    print("-"*70)
    explanation = prep_llm.explain_step(
        'imputed_missing',
        {'strategy': 'iterative', 'method': 'MICE'}
    )
    print(explanation)
    
except Exception as e:
    print(f"\n⚠️  LLM features not available: {e}")
    print("   To use LLM features:")
    print("   1. Install: pip install autoprepml[llm]")
    print("   2. For Ollama (local): Install from https://ollama.ai")
    print("   3. For cloud providers: Configure API keys with 'autoprepml-config'")

# ============================================================================
# Example 4: Direct LLM Suggestor Usage
# ============================================================================
print("\n" + "="*70)
print("Example 4: Direct LLM Suggestor Usage")
print("="*70)

try:
    # Direct usage of LLM Suggestor
    suggestor = LLMSuggestor(provider='ollama', model='llama2')
    
    print("\n✓ LLM Suggestor initialized (Ollama/llama2)")
    
    # Get comprehensive dataset analysis
    print("\n📊 Dataset Analysis:")
    print("-"*70)
    analysis = suggestor.analyze_dataframe(
        df,
        task='classification',
        target_col='label'
    )
    print(analysis)
    
    # Get specific suggestions
    print("\n💡 Suggestions for handling missing salary values:")
    print("-"*70)
    salary_suggestions = suggestor.suggest_fix(
        df,
        column='salary',
        issue_type='missing'
    )
    print(salary_suggestions)
    
    # Feature engineering ideas
    print("\n🔧 Feature Engineering Ideas:")
    print("-"*70)
    features = suggestor.suggest_features(
        df,
        task='classification',
        target_col='label'
    )
    for i, feature in enumerate(features, 1):
        print(f"  {i}. {feature}")
        
except Exception as e:
    print(f"\n⚠️  Could not initialize LLM Suggestor: {e}")
    print("   Make sure Ollama is running: ollama serve")

# ============================================================================
# Example 5: Using Different LLM Providers
# ============================================================================
print("\n" + "="*70)
print("Example 5: Using Different LLM Providers")
print("="*70)

print("\n💡 You can use different LLM providers:")
print("\n1. OpenAI (GPT-4):")
print("   suggestor = LLMSuggestor(provider='openai', api_key='sk-...')")
print("   # Or configure: autoprepml-config --set openai")

print("\n2. Anthropic (Claude):")
print("   suggestor = LLMSuggestor(provider='anthropic', api_key='sk-ant-...')")
print("   # Or configure: autoprepml-config --set anthropic")

print("\n3. Google (Gemini):")
print("   suggestor = LLMSuggestor(provider='google', api_key='...')")
print("   # Or configure: autoprepml-config --set google")

print("\n4. Ollama (Local - No API key!):")
print("   suggestor = LLMSuggestor(provider='ollama', model='llama2')")
print("   # Just need Ollama installed and running")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("Summary")
print("="*70)

print("\n✅ AutoPrepML v1.2.0 Features Demonstrated:")
print("  • Basic preprocessing (v1.0)")
print("  • Advanced imputation: KNN, Iterative/MICE (v1.1.0)")
print("  • SMOTE class balancing (v1.1.0)")
print("  • LLM-powered suggestions (v1.2.0)")
print("  • Multi-provider LLM support (v1.2.0)")
print("  • Comprehensive API integration")

print("\n📚 Next Steps:")
print("  • Configure API keys: autoprepml-config")
print("  • Read docs: docs/LLM_CONFIGURATION.md")
print("  • Explore examples: examples/")
print("  • Run tests: pytest tests/")

print("\n" + "="*70)
print("✨ All systems operational!")
print("="*70)
