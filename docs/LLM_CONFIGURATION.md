# AutoPrepML LLM Configuration Guide

## Overview

AutoPrepML supports AI-powered preprocessing suggestions using Large Language Models (LLMs) from multiple providers. This guide explains how to configure API keys and use the LLM features.

## Quick Start

After installing AutoPrepML, configure your LLM provider:

```bash
# Interactive configuration wizard
autoprepml-config

# Or configure a specific provider
autoprepml-config --set openai
```

## Supported Providers

### 1. **OpenAI (GPT-4, GPT-3.5)**
- **Get API Key**: https://platform.openai.com/api-keys
- **Models**: `gpt-4`, `gpt-3.5-turbo`
- **Configure**:
  ```bash
  autoprepml-config --set openai
  ```

### 2. **Anthropic (Claude)**
- **Get API Key**: https://console.anthropic.com/settings/keys
- **Models**: `claude-3-sonnet-20240229`, `claude-3-opus-20240229`
- **Configure**:
  ```bash
  autoprepml-config --set anthropic
  ```

### 3. **Google (Gemini)**
- **Get API Key**: https://makersuite.google.com/app/apikey
- **Models**: `gemini-pro`, `gemini-ultra`
- **Configure**:
  ```bash
  autoprepml-config --set google
  ```

### 4. **Ollama (Local LLMs)** ⭐ No API Key Needed!
- **Install**: https://ollama.ai/
- **Models**: `llama2`, `mistral`, `codellama`, `phi`
- **Setup**:
  ```bash
  # Install Ollama (see https://ollama.ai)
  # Pull a model
  ollama pull llama2
  
  # No configuration needed!
  ```

## CLI Commands

### Configure API Keys

```bash
# Interactive wizard
autoprepml-config

# Set a specific provider
autoprepml-config --set openai
autoprepml-config --set anthropic
autoprepml-config --set google

# List all configured keys (masked)
autoprepml-config --list

# Check if a provider is configured
autoprepml-config --check openai

# Remove an API key
autoprepml-config --remove openai

# Show package info
autoprepml-config --info
```

### Example Output

```
🔑 AutoPrepML API Key Configuration
============================================================
✅ OpenAI               (saved):    sk-proj-...xYz123
✅ Anthropic (Claude)   (from env): sk-ant-a...456def
❌ Google (Gemini)      Not configured
ℹ️  Ollama (Local)      (local):    No API key needed

💡 Tip: Use 'autoprepml-config --set <provider>' to configure API keys
============================================================
```

## Configuration Methods

API keys can be set in three ways (in order of priority):

### 1. Direct Parameter (Highest Priority)
```python
from autoprepml.llm_suggest import LLMSuggestor

suggestor = LLMSuggestor(provider='openai', api_key='sk-...')
```

### 2. Configuration File
```bash
# Stored in ~/.autoprepml/config.json
autoprepml-config --set openai
```

### 3. Environment Variables (Lowest Priority)
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:GOOGLE_API_KEY="..."

# Linux/Mac
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## Usage in Code

### Basic Usage

```python
from autoprepml.llm_suggest import LLMSuggestor
import pandas as pd

# Load your data
df = pd.read_csv('data.csv')

# Initialize (will auto-load API key from config/env)
suggestor = LLMSuggestor(provider='openai')

# Get suggestions for missing values
suggestions = suggestor.suggest_fix(
    df, 
    column='age', 
    issue_type='missing'
)
print(suggestions)

# Analyze entire dataset
analysis = suggestor.analyze_dataframe(
    df, 
    task='classification', 
    target_col='label'
)
print(analysis)
```

### Using Different Providers

```python
# OpenAI GPT-4
suggestor_gpt4 = LLMSuggestor(provider='openai', model='gpt-4')

# Anthropic Claude-3
suggestor_claude = LLMSuggestor(provider='anthropic')

# Google Gemini
suggestor_gemini = LLMSuggestor(provider='google')

# Local Ollama (no API key needed!)
suggestor_local = LLMSuggestor(provider='ollama', model='llama2')
```

### Explain Cleaning Steps

```python
# Get natural language explanation
explanation = suggestor.explain_cleaning_step(
    step_name='imputed_missing',
    step_details={'strategy': 'median', 'columns': ['age', 'salary']}
)
print(explanation)
```

### Feature Engineering Suggestions

```python
# Get AI-powered feature suggestions
features = suggestor.suggest_features(
    df, 
    task='regression', 
    target_col='price'
)
for feature in features:
    print(f"- {feature}")
```

## Configuration File Location

Configuration is stored in:
- **Windows**: `C:\Users\<username>\.autoprepml\config.json`
- **Linux/Mac**: `~/.autoprepml/config.json`

Example config file:
```json
{
  "api_keys": {
    "openai": "sk-proj-...",
    "anthropic": "sk-ant-..."
  }
}
```

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for production deployments
3. **Rotate keys regularly** via provider dashboards
4. **Use separate keys** for development and production
5. **Consider Ollama** for privacy-sensitive data (runs locally)

## Troubleshooting

### "No API key found" Warning

If you see:
```
⚠️  Warning: No API key found for openai
   Set it with: autoprepml-config --set openai
   Or set environment variable: OPENAI_API_KEY
```

**Solution**: Configure the API key using one of the methods above.

### Check Configuration

```bash
# Verify your setup
autoprepml-config --list

# Check specific provider
autoprepml-config --check openai
```

### Test Connection

```python
from autoprepml.llm_suggest import LLMSuggestor
import pandas as pd

# Test with simple data
df = pd.DataFrame({'age': [25, 30, None, 45]})

try:
    suggestor = LLMSuggestor(provider='openai')
    result = suggestor.suggest_fix(df, 'age', 'missing')
    print("✅ Connection successful!")
    print(result)
except Exception as e:
    print(f"❌ Error: {e}")
```

## Cost Considerations

| Provider | Free Tier | Pricing |
|----------|-----------|---------|
| **Ollama** | ✅ Unlimited (local) | Free |
| **OpenAI** | Limited trial credits | Pay-per-token |
| **Anthropic** | Limited trial | Pay-per-token |
| **Google Gemini** | Free tier available | Pay-per-token |

**💡 Tip**: Use Ollama for development/testing to avoid API costs!

## Next Steps

- Read the [Advanced Features Guide](../docs/ADVANCED_FEATURES.md)
- Explore [LLM Integration Examples](../examples/llm_examples.py)
- Check the [API Reference](../docs/API_REFERENCE.md)

## Support

- **Issues**: https://github.com/mdshoaibuddinchanda/autoprepml/issues
- **Documentation**: https://github.com/mdshoaibuddinchanda/autoprepml
- **Email**: mdshoaibuddinchanda@gmail.com
