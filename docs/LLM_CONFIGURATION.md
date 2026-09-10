# AutoPrepML LLM Configuration Guide

## Overview

AutoPrepML provides optional model assisted preprocessing suggestions through multiple providers. This guide explains credential configuration, provider selection, and safe use of the LLM features.

## Quick Start

After installing AutoPrepML, configure your LLM provider:

```bash
# Interactive configuration wizard
autoprepml-config

# Or configure a specific provider
autoprepml-config --set openai
```

## Supported Providers

### 1. **OpenAI**
- **Get an API key**: https://platform.openai.com/api-keys
- **Models**: Provide any model identifier available to your account, such as `gpt-4o-mini`.
- **Configure**:
  ```bash
  autoprepml-config --set openai
  ```

### 2. **Anthropic Claude**
- **Get an API key**: https://console.anthropic.com/settings/keys
- **Models**: Provide any model identifier available to your account.
- **Configure**:
  ```bash
  autoprepml-config --set anthropic
  ```

### 3. **Google Gemini**
- **Get an API key**: https://aistudio.google.com/apikey
- **Models**: Provide any model identifier supported by the installed `google-genai` SDK. See the dynamic configuration guide for discovery examples.
- **Configure**:
  ```bash
  autoprepml-config --set google
  ```

### 4. **Ollama local models**
- **Install**: https://ollama.ai/
- **Models**: `llama2`, `mistral`, `codellama`, `phi`
- **Setup**:
  ```bash
  # Install Ollama (see https://ollama.ai)
  # Pull a model
  ollama pull llama2
  
  # No API key is required.
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
AutoPrepML API key configuration
============================================================
OpenAI               (saved):     sk-proj-...xYz123
Anthropic (Claude)   (from env):  sk-ant-a...456def
Google (Gemini)      Not configured
Ollama (local)       (local):     No API key needed

Tip: use `autoprepml-config --set <provider>` to configure a provider.
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
# OpenAI
suggestor_openai = LLMSuggestor(provider='openai', model='gpt-4o-mini')

# Anthropic Claude
suggestor_claude = LLMSuggestor(provider='anthropic')

# Google Gemini
suggestor_gemini = LLMSuggestor(provider='google')

# Local Ollama; no API key is required
suggestor_local = LLMSuggestor(provider='ollama', model='llama3.2')
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
    print(feature)
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
5. **Consider Ollama** for privacy sensitive data because it runs locally.

By default, AutoPrepML sends aggregate dataset metadata to an LLM rather than raw samples. Only include raw samples when the `include_samples` option or `AUTOPREPML_LLM_INCLUDE_SAMPLES` environment variable is explicitly enabled and the data policy permits it.

## Structured recommendations

`analyze_dataframe` and `suggest_features` parse JSON responses when a provider
returns structured output. The parser validates the quality score and list or
object shapes before returning them. Invalid output is returned as an advisory
`raw_response` with a `validation_error`; it is never treated as executable
pipeline configuration.

Applications that accept recommendations at a stricter boundary can validate
their own payloads explicitly:

```python
from autoprepml import validate_analysis_recommendation, validate_feature_suggestions

analysis = validate_analysis_recommendation({
    "quality_score": 8,
    "critical_issues": [],
    "pipeline_steps": [],
    "feature_suggestions": [],
    "warnings": [],
})
features = validate_feature_suggestions([{"name": "customer_age_bucket"}])
```

These helpers validate shape and ranges only. A human or application policy
must still review whether a suggested transformation is appropriate for the
dataset and task.

## Troubleshooting

### "No API key found" Warning

If you see:
```
Warning: no API key found for openai
Set it with: `autoprepml-config --set openai`
Or set the `OPENAI_API_KEY` environment variable
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
    print("Connection successful")
    print(result)
except Exception as e:
    print(f"Error: {e}")
```

## Cost Considerations

| Provider | Free Tier | Pricing |
|----------|-----------|---------|
| **Ollama** | Unlimited locally | No provider charge |
| **OpenAI** | Limited trial credits | Pay-per-token |
| **Anthropic** | Limited trial | Pay-per-token |
| **Google Gemini** | Free tier available | Pay-per-token |

For development and testing, Ollama can keep data on the local machine and avoid provider charges.

## Next Steps

- Read the [Advanced Features Guide](ADVANCED_FEATURES.md)
- Explore the [LLM integration notebook](https://github.com/mdshoaibuddinchanda/autoprepml/blob/main/examples/notebooks/07_llm_integration.ipynb)
- Check the [API Reference](api_reference.md)

## Support

- **Issues**: https://github.com/mdshoaibuddinchanda/autoprepml/issues
- **Documentation**: https://github.com/mdshoaibuddinchanda/autoprepml
- **Email**: mdshoaibuddinchanda@gmail.com
