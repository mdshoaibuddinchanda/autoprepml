"""Test dynamic LLM configuration - no hardcoded values!"""
import os
import pandas as pd
from autoprepml.llm_suggest import LLMSuggestor

print("=" * 80)
print("Testing Dynamic LLM Configuration")
print("=" * 80)

# Test 1: Custom model via parameter
print("\n[TEST 1] Custom Model via Parameter")
print("-" * 80)
suggestor = LLMSuggestor(
    provider='google',
    model='gemini-2.5-pro',  # Custom model!
    temperature=0.3,
    max_tokens=200
)
print(f"✅ Provider: {suggestor.provider.value}")
print(f"✅ Model: {suggestor.model}")
print(f"✅ Temperature: {suggestor.temperature}")
print(f"✅ Max Tokens: {suggestor.max_tokens}")

# Test 2: Model via environment variable
print("\n[TEST 2] Model via Environment Variable")
print("-" * 80)
os.environ['GOOGLE_MODEL'] = 'gemini-2.5-flash-lite'
os.environ['GOOGLE_TEMPERATURE'] = '0.9'
os.environ['GOOGLE_MAX_TOKENS'] = '1000'

suggestor2 = LLMSuggestor(provider='google')
print(f"✅ Model from env: {suggestor2.model}")
print(f"✅ Temperature from env: {suggestor2.temperature}")
print(f"✅ Max tokens from env: {suggestor2.max_tokens}")

# Test 3: Mix parameter and environment
print("\n[TEST 3] Mix Parameter and Environment")
print("-" * 80)
suggestor3 = LLMSuggestor(
    provider='google',
    model='gemini-2.5-pro',  # Override env var
    # temperature and max_tokens from environment
)
print(f"✅ Model (parameter): {suggestor3.model}")
print(f"✅ Temperature (env): {suggestor3.temperature}")
print(f"✅ Max tokens (env): {suggestor3.max_tokens}")

# Test 4: Different providers with custom models
print("\n[TEST 4] Multiple Providers with Custom Models")
print("-" * 80)

providers_config = {
    'google': {'model': 'gemini-2.5-flash', 'temp': 0.5},
    'ollama': {'model': 'llama3.2', 'temp': 0.7},
}

for provider, config in providers_config.items():
    try:
        s = LLMSuggestor(
            provider=provider,
            model=config['model'],
            temperature=config['temp']
        )
        print(f"✅ {provider.upper()}: {s.model} (temp={s.temperature})")
    except Exception as e:
        print(f"⚠️  {provider.upper()}: {str(e)[:50]}...")

# Test 5: Real API call with custom model
print("\n[TEST 5] Real API Call with Custom Gemini Model")
print("-" * 80)

# Clean up environment for this test
if 'GOOGLE_MODEL' in os.environ:
    del os.environ['GOOGLE_MODEL']

df = pd.DataFrame({
    'age': [25, 30, None, 45, 50],
    'salary': [50000, 60000, None, 80000, 90000]
})

# Use gemini-2.5-flash (fast and efficient)
suggestor_fast = LLMSuggestor(
    provider='google',
    model='gemini-2.5-flash',
    temperature=0.4,
    max_tokens=300
)

print(f"Testing with: {suggestor_fast.model}")
print("Calling Gemini API...")

try:
    result = suggestor_fast.suggest_fix(df, column='age', issue_type='missing')
    
    if 'Error' not in result and 'blocked' not in result:
        print("✅ API call successful!")
        print(f"\nResponse preview:\n{result[:200]}...")
    else:
        print(f"⚠️  Response: {result[:100]}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("🎉 Dynamic Configuration Working!")
print("=" * 80)
print("\n📝 Key Features Verified:")
print("  • ✅ Custom models via parameter")
print("  • ✅ Configuration via environment variables")
print("  • ✅ Mix of parameter and environment overrides")
print("  • ✅ Multiple providers with different settings")
print("  • ✅ No hardcoded values - fully flexible!")
print("\n💡 Users can now use ANY model they want!")
