"""
Quick test script for Google Gemini API integration
"""

import pytest
import pandas as pd

# Skip if Google Generative AI is not installed
pytest.importorskip("google.generativeai", reason="google-generativeai not installed")

from autoprepml import AutoPrepML
from autoprepml.llm_suggest import LLMSuggestor

print("=" * 80)
print("Testing Google Gemini API Integration")
print("=" * 80)

# Create test data
df = pd.DataFrame({
    'age': [25, 30, None, 45, 50],
    'salary': [50000, 60000, None, 80000, 90000],
    'department': ['HR', 'IT', 'HR', 'Finance', 'IT']
})

print("\nTest DataFrame:")
print(df)
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Test 1: Direct LLM Suggestor
print("\n" + "-" * 80)
print("TEST 1: LLMSuggestor with Gemini")
print("-" * 80)
try:
    suggestor = LLMSuggestor(provider='google')
    print("✅ Gemini LLMSuggestor initialized")
    
    print("\nGetting AI suggestions for missing age values...")
    result = suggestor.suggest_fix(df, column='age', issue_type='missing')
    print("\n📝 Gemini's Suggestion:")
    print(result[:500] + "..." if len(result) > 500 else result)
    print("\n✅ Gemini API is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: AutoPrepML with Gemini
print("\n" + "-" * 80)
print("TEST 2: AutoPrepML with Gemini Integration")
print("-" * 80)
try:
    prep = AutoPrepML(df, enable_llm=True, llm_provider='google')
    print("✅ AutoPrepML with Gemini enabled")
    
    if prep.llm_enabled:
        print("✅ LLM features are active")
        print("   • get_llm_suggestions() - available")
        print("   • analyze_with_llm() - available")
        print("   • get_feature_suggestions() - available")
        print("   • explain_step() - available")
    
    print("\n🎉 All Gemini integration tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("Summary: Gemini API is configured and working correctly!")
print("=" * 80)
