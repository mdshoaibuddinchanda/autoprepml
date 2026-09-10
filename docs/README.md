# AutoPrepML Documentation

The repository currently targets the 1.5.0 development release. PyPI 1.4.0
remains the latest published version while the v1.5 architecture is developed.
See the [PyPI package](https://pypi.org/project/autoprepml/), [GitHub repository](https://github.com/mdshoaibuddinchanda/autoprepml), and [issue tracker](https://github.com/mdshoaibuddinchanda/autoprepml/issues).

## Project tags

| Tag | Scope |
| --- | --- |
| `machine-learning` | Data preparation and model-readiness workflows |
| `data-preprocessing` | Detection, cleaning, validation, and transformation |
| `multi-modal` | Tabular, text, time-series, graph, and image data |
| `python-3.9-to-3.14` | Supported and tested interpreter range |
| `production-readiness` | Contracts, lineage, reproducibility, and CI quality gates |

The documentation is organised as a guided path from installation and examples
to the fitted v1.5 architecture, integrations, and release procedures.

## Documentation Overview

### Getting Started
- **[Quick Start CLI](QUICK_START_CLI.md)**: Command-line usage and examples
- **[CLI reference](cli.md)**: Fitted DataPlan command workflows
- **[Usage Guide](usage.md)**: Python API usage and patterns
- **[Tutorials](tutorials.md)**: Step by step guides for common tasks

### Features & Modules
- **[Advanced Features](ADVANCED_FEATURES.md)**: AutoEDA, AutoFeatureEngine, Interactive Dashboards
- **[API Reference](api_reference.md)**: Complete API documentation

### Configuration
- **[LLM Configuration](LLM_CONFIGURATION.md)**: OpenAI, Anthropic, Google, Ollama setup
- **[Dynamic LLM Config](DYNAMIC_LLM_CONFIGURATION.md)**: Runtime configuration management

### v1.5 architecture

- **[Getting started](getting_started.md)**: Recommended fitted workflow
- **[DataPlan](concepts/data_plan.md)**: Reusable leakage-safe preprocessing state
- **[Data contracts](concepts/contracts.md)**: Structured schema compatibility checks
- **[Leakage prevention](concepts/leakage.md)**: Train-only fitting and resampling
- **[Reproducibility](concepts/reproducibility.md)**: Fingerprints and repeatability metadata
- **[Transformation lineage](concepts/lineage.md)**: Machine-readable manifests
- **[Limitations](limitations.md)**: Explicit scope and artifact trust guidance
- **[Migration to v1.5](migration_v1.5.md)**: Move from whole-frame cleaning to a fitted plan
- **[Benchmark harness](https://github.com/mdshoaibuddinchanda/autoprepml/tree/main/benchmarks)**: Reproduce synthetic fitted-plan and chunked throughput measurements

### Release Notes
- **[v1.5.0 development plan](releases/RELEASE_v1.5.0.md)**: DataPlan, contracts, fingerprints, lineage, and serialization
- **[v1.4.1](releases/RELEASE_v1.4.1.md)**: Production preprocessing safety, normalization, and documentation cleanup
- **[v1.4.0](releases/RELEASE_v1.4.0.md)**: Chunked execution, storage, streaming, experiment tracking, and sklearn pipelines
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
- **Image**: Computer vision preprocessing with deterministic flips and rotations

## Testing and CI/CD

The [root README testing section](https://github.com/mdshoaibuddinchanda/autoprepml/blob/main/README.md#testing) is the canonical source for the current quality baseline. Continuous integration runs the test and coverage suite on Python 3.9 through 3.14 with an 80 percent branch-aware gate, adds Windows and macOS validation on Python 3.12, and blocks linting, security, packaging, and documentation regressions.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## License

The project is released under the MIT License. See [LICENSE](../LICENSE) for details.

For help, [open an issue](https://github.com/mdshoaibuddinchanda/autoprepml/issues) or review the [examples](../examples/).
