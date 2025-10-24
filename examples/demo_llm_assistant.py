"""
Demo: Enhanced LLM Cleaning Assistant
Requires: OPENAI_API_KEY or similar environment variable set

Note: This demo uses LLM API which requires:
- API key configuration (autoprepml-config --set openai)
- Or environment variable (OPENAI_API_KEY, GOOGLE_API_KEY, etc.)
- Or you can use Ollama for local/free LLM (no API key needed)
"""
import pandas as pd
import numpy as np
from autoprepml import LLMSuggestor, suggest_column_rename, generate_data_documentation

# Create sample dataset with poorly named columns
np.random.seed(42)
n_samples = 100

data = {
    'col1': np.random.randint(18, 80, n_samples),  # Actually age
    'var_2': np.random.normal(50000, 20000, n_samples),  # Actually annual income
    'x': np.random.randint(300, 850, n_samples),  # Actually credit score
    'data_field_3': np.random.uniform(1000, 50000, n_samples),  # Actually loan amount
    'cat1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),  # Actually education
    'status': np.random.choice(['1', '2', '3'], n_samples),  # Actually employment
    'y': np.random.choice([0, 1], n_samples)  # Actually default (target)
}

# Add missing values
data['var_2'][np.random.choice(n_samples, 15, replace=False)] = np.nan
data['x'][np.random.choice(n_samples, 10, replace=False)] = np.nan

df = pd.DataFrame(data)

print("=" * 80)
print("Enhanced LLM Cleaning Assistant Demo")
print("=" * 80)

print("\n📊 Sample Dataset with Poor Column Names:")
print(df.head(10))

# Check for API key
import os
provider = 'ollama'  # Default to local Ollama (free, no API key)
if os.getenv('OPENAI_API_KEY'):
    provider = 'openai'
elif os.getenv('GOOGLE_API_KEY'):
    provider = 'google'
elif os.getenv('ANTHROPIC_API_KEY'):
    provider = 'anthropic'

print(f"\n🤖 Using LLM Provider: {provider}")
if provider == 'ollama':
    print("   Note: Using local Ollama (ensure Ollama is running)")
    print("   Install: https://ollama.ai")
else:
    print(f"   Note: Using {provider.upper()} API")

try:
    # Initialize LLM Suggestor
    suggestor = LLMSuggestor(provider=provider)
    
    # 1. Suggest Column Renames
    print("\n" + "=" * 80)
    print("🏷️  Column Renaming Suggestions")
    print("=" * 80)
    
    print("\nAnalyzing column names and suggesting improvements...")
    if rename_suggestions := suggestor.suggest_all_column_renames(df):
        print("\nSuggested column renames:")
        for old_name, new_name in rename_suggestions.items():
            print(f"  '{old_name}' → '{new_name}'")
        
        # Apply renames
        df_renamed = df.rename(columns=rename_suggestions)
        print("\n✅ Columns renamed successfully!")
        print(f"\nNew column names: {list(df_renamed.columns)}")
    else:
        print("\n  No column renames suggested")
        df_renamed = df
    
    # 2. Explain Data Quality Issues
    print("\n" + "=" * 80)
    print("🔍 Data Quality Analysis")
    print("=" * 80)
    
    print("\nAnalyzing data quality issues...")
    quality_explanation = suggestor.explain_data_quality_issues(df_renamed)
    print("\n" + quality_explanation)
    
    # 3. Generate Data Documentation
    print("\n" + "=" * 80)
    print("📖 Generating Data Documentation")
    print("=" * 80)
    
    print("\nCreating comprehensive documentation...")
    documentation = suggestor.generate_data_documentation(df_renamed)
    print("\n" + documentation)
    
    # Save documentation
    with open('data_documentation.md', 'w') as f:
        f.write(documentation)
    print("\n✅ Documentation saved to: data_documentation.md")
    
    # 4. Suggest Preprocessing Pipeline
    print("\n" + "=" * 80)
    print("⚙️  Preprocessing Pipeline Suggestions")
    print("=" * 80)
    
    print("\nGenerating preprocessing pipeline for classification task...")
    pipeline_suggestion = suggestor.suggest_preprocessing_pipeline(df_renamed, task='classification')
    print("\n" + pipeline_suggestion)
    
    # 5. Get Specific Fix Suggestions
    print("\n" + "=" * 80)
    print("💡 Specific Fix Suggestions")
    print("=" * 80)
    
    # Find a column with missing values
    missing_col = df_renamed.columns[df_renamed.isnull().sum() > 0][0]
    print(f"\nGetting suggestions for missing values in '{missing_col}'...")
    
    fix_suggestion = suggestor.suggest_fix(df_renamed, column=missing_col, issue_type='missing')
    print(f"\nSuggestion:\n{fix_suggestion}")
    
    # 6. Analyze specific columns
    print("\n" + "=" * 80)
    print("📊 Column-Specific Analysis")
    print("=" * 80)
    
    numeric_col = df_renamed.select_dtypes(include=[np.number]).columns[0]
    print(f"\nAnalyzing '{numeric_col}'...")
    
    analysis = suggestor.analyze_dataframe(df_renamed, target_column=numeric_col)
    print(f"\n{analysis}")
    
    print("\n" + "=" * 80)
    print("✨ Demo Complete!")
    print("=" * 80)
    
    print("\nKey Features Demonstrated:")
    print("  ✓ Intelligent column renaming based on data content")
    print("  ✓ Automated data quality issue detection and explanation")
    print("  ✓ Comprehensive documentation generation")
    print("  ✓ Preprocessing pipeline recommendations")
    print("  ✓ Specific fix suggestions for data issues")
    print("  ✓ Column-specific analysis and insights")
    
    print("\nGenerated Files:")
    print("  📄 data_documentation.md - Comprehensive dataset documentation")
    
    print("\nNext Steps:")
    print("  1. Review the suggested column names")
    print("  2. Follow the preprocessing pipeline recommendations")
    print("  3. Use the documentation for team collaboration")
    print("  4. Apply specific fixes to problematic columns")
    
except ImportError as e:
    print(f"\n❌ Error: {e}")
    print("\nTo use LLM features, install required packages:")
    print("  pip install autoprepml[llm]")
    print("\nOr for local/free option:")
    print("  1. Install Ollama: https://ollama.ai")
    print("  2. Pull a model: ollama pull llama3.2")
    print("  3. Start Ollama service")
    
except Exception as e:
    print(f"\n⚠️  Could not connect to LLM service: {e}")
    print("\nOptions:")
    print("  1. Set API key: autoprepml-config --set openai")
    print("  2. Use local Ollama (free): ollama pull llama3.2")
    print("  3. Set environment variable: export OPENAI_API_KEY=...")

print("\n" + "=" * 80)
print("💡 Tips for Production Use")
print("=" * 80)
print("\n1. Column Renaming:")
print("   - Review suggestions before applying")
print("   - Maintain consistency across datasets")
print("   - Use for improving code readability")

print("\n2. Data Documentation:")
print("   - Auto-generate for new datasets")
print("   - Keep documentation version-controlled")
print("   - Share with data science teams")

print("\n3. Quality Analysis:")
print("   - Use insights for data collection improvements")
print("   - Track quality metrics over time")
print("   - Automate quality reporting")

print("\n4. Preprocessing Pipelines:")
print("   - Use as starting point for experimentation")
print("   - Adapt to specific domain requirements")
print("   - Validate with cross-validation")
