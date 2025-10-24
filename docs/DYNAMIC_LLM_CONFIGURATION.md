# Dynamic LLM Configuration Guide

## 🎯 Overview

AutoPrepML's LLM integration is **fully dynamic** - no hardcoded values! You can customize every aspect through:
1. **Direct parameters** (in code)
2. **Environment variables** (for flexibility)
3. **Config file** (for persistence)

## 🔧 Configuration Priority

```
1. Direct Parameter (highest priority)
   ↓
2. Environment Variable
   ↓
3. Config File (~/.autoprepml/config.json)
   ↓
4. Default Value (lowest priority)
```

## 📋 Available Configuration Options

### 1. API Keys

**Methods:**
```bash
# Method 1: CLI configuration (persistent)
autoprepml-config --set google

# Method 2: Environment variable (session)
export GOOGLE_API_KEY="your-key-here"

# Method 3: Direct in code (temporary)
LLMSuggestor(provider='google', api_key='your-key-here')
```

**Environment Variables:**
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key  
- `GOOGLE_API_KEY` - Google Gemini API key
- `OLLAMA_API_KEY` - Not needed (local)

---

### 2. Model Selection (Fully Dynamic!)

**Any valid model for each provider is supported!**

#### OpenAI Models
```python
# Use any OpenAI model
LLMSuggestor(provider='openai', model='gpt-4o')
LLMSuggestor(provider='openai', model='gpt-4o-mini')
LLMSuggestor(provider='openai', model='gpt-4-turbo')
LLMSuggestor(provider='openai', model='gpt-3.5-turbo')
LLMSuggestor(provider='openai', model='o1-preview')
```

**Via Environment Variable:**
```bash
export OPENAI_MODEL="gpt-4o-mini"
# Now all LLMSuggestor instances use this model by default
```

**Change Default:**
```bash
export OPENAI_DEFAULT_MODEL="gpt-4-turbo"
```

#### Anthropic Models
```python
# Use any Claude model
LLMSuggestor(provider='anthropic', model='claude-3-5-sonnet-20241022')
LLMSuggestor(provider='anthropic', model='claude-3-5-haiku-20241022')
LLMSuggestor(provider='anthropic', model='claude-3-opus-20240229')
LLMSuggestor(provider='anthropic', model='claude-3-sonnet-20240229')
```

**Via Environment Variable:**
```bash
export ANTHROPIC_MODEL="claude-3-5-sonnet-20241022"
```

#### Google Gemini Models
```python
# Use ANY Gemini model (automatically detects available models)
LLMSuggestor(provider='google', model='gemini-2.5-pro')
LLMSuggestor(provider='google', model='gemini-2.5-flash')
LLMSuggestor(provider='google', model='gemini-2.5-flash-lite')
LLMSuggestor(provider='google', model='gemini-2.0-flash-exp')
LLMSuggestor(provider='google', model='gemini-flash-latest')
LLMSuggestor(provider='google', model='gemini-pro-latest')
```

**Via Environment Variable:**
```bash
export GOOGLE_MODEL="gemini-2.5-pro"
```

**List Available Models:**
```python
import google.generativeai as genai
from autoprepml.config_manager import AutoPrepMLConfig

genai.configure(api_key=AutoPrepMLConfig.get_api_key('google'))
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✅ {model.name}")
```

#### Ollama Models (Local)
```python
# Use ANY model you've pulled with Ollama
LLMSuggestor(provider='ollama', model='llama3.2')
LLMSuggestor(provider='ollama', model='llama3.2:70b')
LLMSuggestor(provider='ollama', model='mistral')
LLMSuggestor(provider='ollama', model='mixtral')
LLMSuggestor(provider='ollama', model='codellama')
LLMSuggestor(provider='ollama', model='phi3')
LLMSuggestor(provider='ollama', model='qwen2.5')
```

**Via Environment Variable:**
```bash
export OLLAMA_MODEL="mixtral"
```

**List Available Models:**
```bash
ollama list
```

---

### 3. Temperature (Creativity Control)

**Range:** 0.0 (deterministic) to 1.0 (creative)

```python
# Direct parameter
LLMSuggestor(provider='google', temperature=0.3)  # More focused
LLMSuggestor(provider='google', temperature=0.9)  # More creative
```

**Via Environment Variable:**
```bash
# Per-provider temperature
export GOOGLE_TEMPERATURE="0.5"
export OPENAI_TEMPERATURE="0.8"
export ANTHROPIC_TEMPERATURE="0.7"
export OLLAMA_TEMPERATURE="0.6"
```

---

### 4. Max Tokens (Response Length)

**Control output length dynamically**

```python
# Direct parameter
LLMSuggestor(provider='google', max_tokens=1000)  # Longer responses
LLMSuggestor(provider='google', max_tokens=200)   # Shorter responses
```

**Via Environment Variable:**
```bash
export GOOGLE_MAX_TOKENS="1000"
export OPENAI_MAX_TOKENS="500"
export ANTHROPIC_MAX_TOKENS="2000"
export OLLAMA_MAX_TOKENS="800"
```

---

### 5. Base URL (Custom Endpoints)

**Use custom endpoints, proxies, or self-hosted models**

```python
# OpenAI-compatible endpoints
LLMSuggestor(
    provider='openai',
    base_url='https://your-proxy.com/v1'
)

# Custom Ollama server
LLMSuggestor(
    provider='ollama',
    base_url='http://192.168.1.100:11434'
)
```

**Via Environment Variable:**
```bash
export OPENAI_BASE_URL="https://your-proxy.com/v1"
export OLLAMA_BASE_URL="http://remote-server:11434"
export ANTHROPIC_BASE_URL="https://custom-endpoint.com"
```

---

### 6. Google Safety Settings

**Control content filtering dynamically**

```python
# Safety settings are read from environment
# No need to modify code!
```

**Via Environment Variable:**
```bash
# Options: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE
export GOOGLE_SAFETY_LEVEL="BLOCK_ONLY_HIGH"
```

---

## 🚀 Complete Examples

### Example 1: Use Latest Gemini Pro with Custom Settings

```python
from autoprepml.llm_suggest import LLMSuggestor
import pandas as pd

# Method A: Direct parameters
suggestor = LLMSuggestor(
    provider='google',
    model='gemini-2.5-pro',        # Latest pro model
    temperature=0.3,                # More focused
    max_tokens=1500                 # Longer responses
)

df = pd.read_csv('data.csv')
analysis = suggestor.analyze_dataframe(df, task='classification')
```

### Example 2: Environment-Based Configuration

```bash
# Set once, use everywhere
export GOOGLE_API_KEY="your-key"
export GOOGLE_MODEL="gemini-2.5-flash"
export GOOGLE_TEMPERATURE="0.6"
export GOOGLE_MAX_TOKENS="800"
export GOOGLE_SAFETY_LEVEL="BLOCK_NONE"
```

```python
# Code stays clean - all config from environment!
suggestor = LLMSuggestor(provider='google')
# Uses: gemini-2.5-flash, temp=0.6, max_tokens=800
```

### Example 3: Mix and Match

```python
# Override just what you need
suggestor = LLMSuggestor(
    provider='google',
    model='gemini-2.5-pro',        # Override model
    # temperature and max_tokens from environment
)
```

### Example 4: Multiple Providers with Different Settings

```python
# OpenAI with strict temperature
openai_suggestor = LLMSuggestor(
    provider='openai',
    model='gpt-4o',
    temperature=0.2
)

# Gemini with higher creativity
gemini_suggestor = LLMSuggestor(
    provider='google',
    model='gemini-2.5-pro',
    temperature=0.8
)

# Local Ollama with custom server
ollama_suggestor = LLMSuggestor(
    provider='ollama',
    model='llama3.2:70b',
    base_url='http://gpu-server:11434'
)
```

### Example 5: AutoPrepML Integration

```python
from autoprepml import AutoPrepML

# All LLM settings work in AutoPrepML too!
prep = AutoPrepML(
    df,
    enable_llm=True,
    llm_provider='google',
    llm_model='gemini-2.5-pro',          # Custom model
    llm_temperature=0.4,                  # Custom temperature
    llm_max_tokens=1000                   # Custom max tokens
)

# Get AI-powered suggestions
suggestions = prep.get_llm_suggestions(column='age', issue_type='missing')
```

---

## 🔍 Checking Current Configuration

### View All Settings

```python
from autoprepml.llm_suggest import LLMSuggestor

suggestor = LLMSuggestor(provider='google')

print(f"Provider: {suggestor.provider.value}")
print(f"Model: {suggestor.model}")
print(f"Temperature: {suggestor.temperature}")
print(f"Max Tokens: {suggestor.max_tokens}")
print(f"Base URL: {suggestor.base_url}")
print(f"API Key: {'✅ Set' if suggestor.api_key else '❌ Not set'}")
```

### Check API Keys

```bash
# Check all configured keys
autoprepml-config --list

# Check specific provider
autoprepml-config --check google
```

---

## 💡 Best Practices

### 1. Development vs Production

**Development:**
```bash
# Use faster, cheaper models
export GOOGLE_MODEL="gemini-2.5-flash"
export OPENAI_MODEL="gpt-4o-mini"
```

**Production:**
```bash
# Use more powerful models
export GOOGLE_MODEL="gemini-2.5-pro"
export OPENAI_MODEL="gpt-4o"
```

### 2. Cost Optimization

```bash
# Lower max_tokens = lower cost
export GOOGLE_MAX_TOKENS="300"
export OPENAI_MAX_TOKENS="400"
```

### 3. Quality vs Speed

```python
# High quality (slower, more expensive)
suggestor = LLMSuggestor(
    provider='google',
    model='gemini-2.5-pro',
    temperature=0.2,
    max_tokens=2000
)

# Fast responses (faster, cheaper)
suggestor = LLMSuggestor(
    provider='google',
    model='gemini-2.5-flash-lite',
    temperature=0.7,
    max_tokens=300
)
```

### 4. Local Development

```bash
# Use Ollama for free local inference
export OLLAMA_MODEL="llama3.2"
export OLLAMA_TEMPERATURE="0.7"
```

```python
suggestor = LLMSuggestor(provider='ollama')
# No API costs, fully offline!
```

---

## 🛠️ Troubleshooting

### Issue: Model not found

**Solution:** Check available models for your provider

```python
# For Google
import google.generativeai as genai
genai.configure(api_key="your-key")
for m in genai.list_models():
    print(m.name)

# For Ollama
# Run: ollama list
```

### Issue: Environment variable not working

**Solution:** Check if it's set

```bash
# Windows PowerShell
$env:GOOGLE_MODEL

# Linux/Mac
echo $GOOGLE_MODEL

# If empty, set it:
export GOOGLE_MODEL="gemini-2.5-flash"  # Linux/Mac
$env:GOOGLE_MODEL="gemini-2.5-flash"    # PowerShell
```

### Issue: Want to reset to defaults

**Solution:** Unset environment variables

```bash
# Linux/Mac
unset GOOGLE_MODEL
unset GOOGLE_TEMPERATURE

# Windows PowerShell
Remove-Item Env:\GOOGLE_MODEL
Remove-Item Env:\GOOGLE_TEMPERATURE
```

---

## 📊 Summary Table

| Configuration | Parameter | Environment Variable | Config File | Default |
|---------------|-----------|---------------------|-------------|---------|
| **API Key** | `api_key` | `<PROVIDER>_API_KEY` | Yes | None |
| **Model** | `model` | `<PROVIDER>_MODEL` | No | Provider default |
| **Temperature** | `temperature` | `<PROVIDER>_TEMPERATURE` | No | 0.7 |
| **Max Tokens** | `max_tokens` | `<PROVIDER>_MAX_TOKENS` | No | 500 |
| **Base URL** | `base_url` | `<PROVIDER>_BASE_URL` | No | Provider default |
| **Safety Level** | N/A | `GOOGLE_SAFETY_LEVEL` | No | BLOCK_NONE |
| **Default Model** | N/A | `<PROVIDER>_DEFAULT_MODEL` | No | Hardcoded |

---

## 🎯 Quick Reference

```bash
# Set API key (persistent)
autoprepml-config --set google

# Set model (session)
export GOOGLE_MODEL="gemini-2.5-pro"

# Set temperature (session)
export GOOGLE_TEMPERATURE="0.5"

# Set max tokens (session)
export GOOGLE_MAX_TOKENS="1000"

# Check configuration
autoprepml-config --list

# Use in code
python your_script.py  # Uses all env vars automatically!
```

---

**🚀 No Hardcoded Values - Full Control - Maximum Flexibility!**
