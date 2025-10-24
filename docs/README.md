# AutoPrepML Documentation

**Version 1.3.0** | [PyPI](https://pypi.org/project/autoprepml/) | [GitHub](https://github.com/mdshoaibuddinchanda/autoprepml) | [Issues](https://github.com/mdshoaibuddinchanda/autoprepml/issues)

## 📚 Documentation Overview

### Getting Started
- **[Quick Start CLI](QUICK_START_CLI.md)** - Command-line usage and examples
- **[Usage Guide](usage.md)** - Python API usage and patterns
- **[Tutorials](tutorials.md)** - Step-by-step guides for common tasks

### Features & Modules
- **[Advanced Features](ADVANCED_FEATURES.md)** - AutoEDA, AutoFeatureEngine, Interactive Dashboards
- **[API Reference](api_reference.md)** - Complete API documentation

### Configuration
- **[LLM Configuration](LLM_CONFIGURATION.md)** - OpenAI, Anthropic, Google, Ollama setup
- **[Dynamic LLM Config](DYNAMIC_LLM_CONFIGURATION.md)** - Runtime configuration management

### Release Notes
- **[v1.3.0](releases/RELEASE_v1.3.0.md)** - AutoEDA, AutoFeatureEngine, 227 tests
- **[v1.2.0](releases/RELEASE_v1.2.0.md)** - LLM Integration, Dashboard improvements
- **[Full Changelog](../CHANGELOG.md)** - Complete version history

## 🚀 Quick Links

### Installation
```bash
pip install autoprepml
```

### Basic Usage
```python
from autoprepml import AutoPrepML

# Automatic preprocessing
prep = AutoPrepML(df)
clean_df = prep.clean()
prep.report(output_path='report.html')
```

### CLI Usage
```bash
# Preprocess any data type
autoprepml preprocess data.csv --output cleaned.csv

# Generate report
autoprepml report data.csv --output report.html
```

## 📊 Supported Data Types

- **Tabular** - CSV, Excel, Parquet, databases
- **Text** - NLP preprocessing, tokenization, feature extraction
- **Time Series** - Temporal analysis, resampling, lag features
- **Graph** - Network data, node/edge validation
- **Image** - Computer vision preprocessing, augmentation

## 🧪 Testing & CI/CD

- **227 tests** passing across Python 3.10, 3.11, 3.12
- **95%+ code coverage**
- **Optimized CI/CD** with intelligent caching
- **Cross-platform** - Ubuntu, Windows, macOS

## 🤝 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) and [PUBLISHING.md](../PUBLISHING.md) for development guidelines.

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details.

---

**Need help?** [Open an issue](https://github.com/mdshoaibuddinchanda/autoprepml/issues) or check the [examples](../examples/)!
