# AutoPrepML Documentation

Version 1.3.0. See the [PyPI package](https://pypi.org/project/autoprepml/), [GitHub repository](https://github.com/mdshoaibuddinchanda/autoprepml), and [issue tracker](https://github.com/mdshoaibuddinchanda/autoprepml/issues).

## Documentation Overview

### Getting Started
- **[Quick Start CLI](QUICK_START_CLI.md)**: Command-line usage and examples
- **[Usage Guide](usage.md)**: Python API usage and patterns
- **[Tutorials](tutorials.md)**: Step by step guides for common tasks

### Features & Modules
- **[Advanced Features](ADVANCED_FEATURES.md)**: AutoEDA, AutoFeatureEngine, Interactive Dashboards
- **[API Reference](api_reference.md)**: Complete API documentation

### Configuration
- **[LLM Configuration](LLM_CONFIGURATION.md)**: OpenAI, Anthropic, Google, Ollama setup
- **[Dynamic LLM Config](DYNAMIC_LLM_CONFIGURATION.md)**: Runtime configuration management

### Release Notes
- **[v1.3.0](releases/RELEASE_v1.3.0.md)**: AutoEDA, AutoFeatureEngine, and interactive dashboards
- **[v1.2.0](releases/RELEASE_v1.2.0.md)**: LLM Integration, Dashboard improvements
- **[Full Changelog](../CHANGELOG.md)**: Complete version history

## Quick Links

### Installation
```bash
pip install autoprepml
```

### Basic Usage
```python
from autoprepml import AutoPrepML

# Automatic preprocessing
prep = AutoPrepML(df)
clean_df, report = prep.clean()
prep.save_report('report.html')
```

### CLI Usage
```bash
autoprepml --input data.csv --output cleaned.csv --report report.html
```

## Supported Data Types

- **Tabular**: CSV, Excel, Parquet, databases
- **Text**: NLP preprocessing, tokenization, feature extraction
- **Time Series**: Temporal analysis, resampling, lag features
- **Graph**: Network data, node/edge validation
- **Image**: Computer vision preprocessing, augmentation

## Testing & CI/CD

- **323 tests** passing locally
- **88 percent local line coverage**, with a 75 percent CI threshold
- **Blocking CI checks** for linting, tests, security, packaging, and documentation
- **Cross platform** validation on Ubuntu, Windows, and macOS

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## License

The project is released under the MIT License. See [LICENSE](../LICENSE) for details.

For help, [open an issue](https://github.com/mdshoaibuddinchanda/autoprepml/issues) or review the [examples](../examples/).
