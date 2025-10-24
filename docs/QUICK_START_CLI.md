# AutoPrepML CLI Configuration - Quick Start

## Installation

```bash
pip install autoprepml[llm]
```

## First Time Setup

After installing AutoPrepML, you'll be prompted to configure your LLM provider when you first try to use LLM features:

```
⚠️  Warning: No API key found for openai
   Set it with: autoprepml-config --set openai
   Or set environment variable: OPENAI_API_KEY
```

## Quick Configuration

### Method 1: Interactive Wizard (Recommended for Beginners)

```bash
autoprepml-config
```

You'll see:
```
🎯 AutoPrepML Configuration Wizard
============================================================

Which LLM provider would you like to configure?

1. OpenAI
2. Anthropic (Claude)
3. Google (Gemini)
4. Ollama (Local)
5. Skip / Configure later

Enter your choice (1-5):
```

### Method 2: Direct Configuration (Recommended for Advanced Users)

```bash
# Configure OpenAI
autoprepml-config --set openai
# You'll be prompted to enter your API key

# Configure Anthropic
autoprepml-config --set anthropic

# Configure Google Gemini
autoprepml-config --set google
```

### Method 3: Environment Variables (Recommended for Production)

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY="sk-proj-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:GOOGLE_API_KEY="..."
```

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## Verify Configuration

```bash
# List all configured providers
autoprepml-config --list
```

Output:
```
🔑 AutoPrepML API Key Configuration
============================================================
✅ OpenAI               (saved):    sk-proj-...xYz123
❌ Anthropic (Claude)   Not configured
❌ Google (Gemini)      Not configured
ℹ️  Ollama (Local)      (local):    No API key needed

💡 Tip: Use 'autoprepml-config --set <provider>' to configure API keys
============================================================
```

## Check Specific Provider

```bash
# Check if OpenAI is configured
autoprepml-config --check openai
```

Output:
```
✅ OpenAI API key is configured: sk-proj-...xYz123
```

## Remove API Key

```bash
autoprepml-config --remove openai
```

## Using Ollama (Local LLM - No API Key Needed!)

Ollama runs locally on your machine - perfect for privacy and cost savings!

### 1. Install Ollama

Visit https://ollama.ai/ and download for your OS.

### 2. Pull a Model

```bash
# Pull Llama 2 (7B model)
ollama pull llama2

# Or try other models
ollama pull mistral
ollama pull codellama
ollama pull phi
```

### 3. Use in Python

```python
from autoprepml.llm_suggest import LLMSuggestor

# No API key needed!
suggestor = LLMSuggestor(provider='ollama', model='llama2')

# Use it immediately
suggestions = suggestor.suggest_fix(df, column='age', issue_type='missing')
print(suggestions)
```

## Complete Example

```python
import pandas as pd
from autoprepml.llm_suggest import LLMSuggestor

# Load data
df = pd.DataFrame({
    'age': [25, 30, None, 45, 50],
    'salary': [50000, 60000, None, 80000, 90000],
    'department': ['HR', 'IT', 'HR', None, 'IT']
})

# Initialize (will auto-load API key from config)
suggestor = LLMSuggestor(provider='openai')

# Get suggestions for missing values
print("=== Missing Value Suggestions ===")
age_suggestions = suggestor.suggest_fix(df, column='age', issue_type='missing')
print(age_suggestions)

# Analyze entire dataset
print("\n=== Dataset Analysis ===")
analysis = suggestor.analyze_dataframe(df, task='regression', target_col='salary')
print(analysis)

# Get feature engineering ideas
print("\n=== Feature Suggestions ===")
features = suggestor.suggest_features(df, task='regression', target_col='salary')
for feature in features:
    print(f"  • {feature}")

# Explain a cleaning step
print("\n=== Step Explanation ===")
explanation = suggestor.explain_cleaning_step(
    step_name='imputed_missing',
    step_details={'strategy': 'median', 'columns': ['age', 'salary']}
)
print(explanation)
```

## Where are API Keys Stored?

API keys are stored securely in:

- **Windows**: `C:\Users\<YourName>\.autoprepml\config.json`
- **Linux/Mac**: `~/.autoprepml/config.json`

Example `config.json`:
```json
{
  "api_keys": {
    "openai": "sk-proj-...",
    "anthropic": "sk-ant-..."
  }
}
```

## Security Best Practices

1. ✅ **Never commit** `config.json` to version control
2. ✅ **Use environment variables** for production/CI/CD
3. ✅ **Rotate keys regularly** via provider dashboards
4. ✅ **Use separate keys** for dev and production
5. ✅ **Consider Ollama** for sensitive data (runs locally)

## Troubleshooting

### "No API key found" Warning

**Problem**:
```
⚠️  Warning: No API key found for openai
```

**Solution**:
```bash
autoprepml-config --set openai
# Then enter your API key when prompted
```

### API Key Not Working

**Check configuration**:
```bash
autoprepml-config --check openai
```

**Verify in Python**:
```python
from autoprepml.config_manager import AutoPrepMLConfig

key = AutoPrepMLConfig.get_api_key('openai')
if key:
    print(f"Key found: {key[:10]}...")
else:
    print("No key configured!")
```

### Permission Denied on Config File

**Windows**:
```powershell
# Check permissions
icacls C:\Users\<YourName>\.autoprepml\config.json
```

**Linux/Mac**:
```bash
# Fix permissions
chmod 600 ~/.autoprepml/config.json
```

## Next Steps

- 📖 Read the [Full LLM Configuration Guide](./LLM_CONFIGURATION.md)
- 🚀 Explore [Advanced Features](./ADVANCED_FEATURES.md)
- 💻 Check out [Code Examples](../examples/)
- 📚 Review the [API Reference](./API_REFERENCE.md)

## Get API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/settings/keys
- **Google Gemini**: https://makersuite.google.com/app/apikey
- **Ollama**: https://ollama.ai/ (No key needed - runs locally!)

## Support

Having issues? We're here to help!

- 🐛 **Bug Reports**: https://github.com/mdshoaibuddinchanda/autoprepml/issues
- 📧 **Email**: mdshoaibuddinchanda@gmail.com
- 💬 **Discussions**: https://github.com/mdshoaibuddinchanda/autoprepml/discussions
