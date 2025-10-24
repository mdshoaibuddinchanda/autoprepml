# Changelog

All notable changes to AutoPrepML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.0] - 2025-10-24

### Added - AutoEDA Module
- **AutoEDA Class**: Automated exploratory data analysis
  - `analyze()` - Comprehensive EDA with configurable components
  - Statistical summaries (mean, std, quartiles, skewness, kurtosis)
  - Missing value analysis with counts and percentages
  - Correlation matrix computation with high correlation detection
  - Distribution analysis (skewness, kurtosis, quartiles)
  - Outlier detection using IQR and Z-score methods
  - Categorical variable analysis (cardinality, mode, value counts)
  - Automated insights generation in natural language
- **Report Generation**:
  - `generate_report()` - Interactive HTML reports with visualizations
  - `to_json()` - Export analysis results to JSON format
- **Full Integration**: Added to `autoprepml` package exports
- **Example Script**: `examples/demo_autoeda.py` with complete workflow
- **Test Suite**: 40+ comprehensive tests for AutoEDA module

### Added - AutoFeatureEngine Module
- **AutoFeatureEngine Class**: Intelligent feature engineering
  - `create_polynomial_features()` - Polynomial and interaction terms
  - `create_interactions()` - Pairwise multiplication features
  - `create_ratio_features()` - Division-based features
  - `create_binned_features()` - Discretization (uniform, quantile, kmeans)
  - `create_aggregation_features()` - Row-wise aggregations (sum, mean, std, min, max)
  - `create_datetime_features()` - Extract temporal components
  - `select_features()` - Feature selection (mutual_info, f_test)
  - `get_feature_importance()` - Rank features by importance
- **Convenience Function**: `auto_feature_engineering()` for quick feature creation
- **Full Integration**: Added to `autoprepml` package exports
- **Example Script**: `examples/demo_feature_engine.py` with all methods
- **Test Suite**: 45+ comprehensive tests for feature engineering

### Added - Interactive Dashboard Module
- **InteractiveDashboard Class**: Visualization and app generation
  - `create_dashboard()` - Multi-subplot Plotly dashboards
  - `create_correlation_heatmap()` - Interactive correlation matrix
  - `create_missing_data_plot()` - Missing value visualization
  - `generate_streamlit_app()` - Full Streamlit app generation
- **Convenience Functions**:
  - `create_plotly_dashboard()` - Quick dashboard creation
  - `create_correlation_heatmap()` - Standalone heatmap
  - `create_missing_data_plot()` - Standalone missing data viz
  - `generate_streamlit_app()` - Standalone app generator
- **Streamlit App Features**:
  - File upload functionality
  - Overview tab (shape, dtypes, memory usage)
  - EDA tab (distributions, correlations, missing values)
  - Preprocessing tab (missing value handling, encoding)
  - Feature engineering tab (interactions, polynomial, binning)
- **Full Integration**: Added to `autoprepml` package exports
- **Example Script**: `examples/demo_dashboard.py` with all visualizations
- **Test Suite**: 35+ comprehensive tests for dashboard module

### Added - Enhanced LLM Assistant
- **New LLM Methods**: Expanded AI-powered capabilities
  - `suggest_column_rename()` - Intelligent column name suggestions
  - `suggest_all_column_renames()` - Batch rename all columns
  - `explain_data_quality_issues()` - Natural language quality explanations
  - `generate_data_documentation()` - Auto-generate Markdown documentation
  - `suggest_preprocessing_pipeline()` - Complete pipeline recommendations
- **Convenience Functions**:
  - `suggest_column_rename()` - Quick column rename
  - `generate_data_documentation()` - Quick doc generation
- **Full Integration**: Enhanced existing LLMSuggestor class
- **Example Script**: `examples/demo_llm_assistant.py` with all features
- **Use Cases**:
  - Automated dataset documentation
  - Intelligent column naming for readability
  - Data quality insights in plain English
  - Preprocessing workflow recommendations

### Added - Dependencies
- **Visualization Libraries**:
  - `plotly>=5.0.0` - Interactive visualizations
  - `streamlit>=1.0.0` - Web app framework (optional)
- **Installation Options**:
  - `pip install autoprepml[viz]` - With visualization support
  - `pip install autoprepml[all]` - Complete installation

### Changed
- Updated package version to 1.3.0
- Updated README with v1.3.0 features section
- Updated test count badge (158 tests passing)
- Enhanced `__init__.py` with 14 new exports

### Documentation
- Added comprehensive v1.3.0 features section to README
- Created 4 new example scripts with complete workflows
- Added inline documentation for all new modules
- Updated Quick Navigation table with v1.3.0 links

### Testing
- Added `tests/test_autoeda.py` with 40+ tests
- Added `tests/test_feature_engine.py` with 45+ tests
- Added `tests/test_dashboard.py` with 35+ tests
- Total test count: 158 tests (all passing)
- Test coverage maintained at 95%+

## [1.2.0] - 2025-10-24

### Added - Image Preprocessing Module
- **ImagePrepML Class**: Complete image data preprocessing
  - Support for multiple formats: PNG, JPG, JPEG, BMP, GIF, TIFF, WEBP
  - Automatic issue detection (corruption, size mismatch, color mode)
  - Batch processing and resizing
  - Color mode conversion (RGB, RGBA, Grayscale)
  - Pixel normalization to [0, 1]
  - Dataset splitting (train/val/test)
  - Duplicate detection
  - Quality checks (file size, dimensions)
- **Image Statistics**: Comprehensive dataset analysis
  - Size distribution
  - Color mode distribution  
  - File size statistics
  - Memory usage tracking
- **Export Functions**: Multiple output formats
  - Save processed images to directory
  - HTML report generation with visualization
  - NumPy array export for ML pipelines
- **Convenience Function**: `preprocess_images()` for quick processing
- **Full Integration**: Added to `autoprepml` package exports
- **Example Script**: `examples/demo_image.py` with complete workflow
- **Test Suite**: 17 comprehensive tests for image module

## [1.0.1] - 2025-10-24 (v1.2.0 Features)

### Added - LLM Integration (v1.2.0)
- **Multi-Provider LLM Support**: AI-powered preprocessing suggestions
  - OpenAI (GPT-4, GPT-3.5-turbo)
  - Anthropic (Claude-3-Sonnet, Claude-3-Opus)
  - Google (Gemini Pro)
  - Ollama (Local LLMs: llama2, mistral, codellama, phi)
- **LLMSuggestor Class**: Unified interface for all LLM providers
  - `suggest_fix()` - AI suggestions for data quality issues
  - `analyze_dataframe()` - Comprehensive dataset analysis
  - `explain_cleaning_step()` - Natural language explanations
  - `suggest_features()` - Feature engineering recommendations
- **Configuration Management**: Secure API key storage
  - `AutoPrepMLConfig` class for key management
  - Config stored in `~/.autoprepml/config.json`
  - 3-tier priority: Parameter > Config File > Environment Variables
- **CLI Configuration Tool**: `autoprepml-config` command
  - Interactive wizard mode
  - Commands: --set, --list, --check, --remove, --info
  - Support for all 4 LLM providers
- **Core Integration**: LLM support in AutoPrepML class
  - `enable_llm` parameter for initialization
  - `get_llm_suggestions()` - Get AI suggestions
  - `analyze_with_llm()` - AI dataset analysis
  - `get_feature_suggestions()` - Feature ideas
  - `explain_step()` - Natural language explanations
- **Optional Dependencies**: LLM packages as extras
  - Install with: `pip install autoprepml[llm]`
  - Graceful degradation when not installed

### Added - Advanced Features (v1.1.0)
- **Advanced Imputation Methods**:
  - `impute_knn()` - K-Nearest Neighbors imputation
  - `impute_iterative()` - Iterative/MICE imputation
- **SMOTE Class Balancing**:
  - `balance_classes_smote()` - Synthetic Minority Over-sampling
  - Configurable sampling strategies
  - Support for multiclass problems
- **Core Integration**: Advanced methods in AutoPrepML
  - `use_advanced` parameter in clean()
  - `imputation_method` parameter (simple/knn/iterative)
  - `balance_method` parameter (oversample/undersample/smote)

### Added - Documentation
- **LLM Configuration Guide** (`docs/LLM_CONFIGURATION.md`)
  - Comprehensive setup for all 4 providers
  - Security best practices
  - Troubleshooting guide
  - 300+ lines of documentation
- **Quick Start CLI Guide** (`docs/QUICK_START_CLI.md`)
  - Installation and setup
  - Configuration examples
  - Usage patterns
  - 250+ lines of documentation
- **Advanced Features Guide** (`docs/ADVANCED_FEATURES.md`)
  - KNN and Iterative imputation tutorials
  - SMOTE usage guide
  - Real-world examples
  - API reference
  - 500+ lines of documentation
- **Integration Audit** (`INTEGRATION_AUDIT.md`)
  - Complete integration verification
  - Feature matrix
  - API consistency checks
- **V1.2.0 Roadmap** (`V1.2.0_ROADMAP.md`)
  - Implementation plan for v1.2.0 features
  - Priority matrix
  - Timeline and dependencies

### Added - Testing
- **21 LLM Integration Tests** (`test_llm_integration.py`)
  - Provider initialization tests
  - Data profiling tests
  - Error handling tests
  - Integration tests (optional with API keys)
- **14 Configuration Manager Tests** (`test_config_manager.py`)
  - API key management tests
  - Priority order tests
  - Security tests
- **24 Advanced Features Tests** (`test_advanced_features.py`)
  - KNN imputation tests
  - Iterative imputation tests
  - SMOTE balancing tests
- **Total Test Count**: 159 tests (125 passing, 7 skipped, 27 requiring optional dependencies)

### Added - Examples
- **Complete Integration Demo** (`examples/complete_integration_demo.py`)
  - Demonstrates all v1.2.0 features
  - Shows basic → advanced → LLM workflow
  - Multiple provider examples
  - 300+ lines with comprehensive examples

### Changed
- **Version**: Updated to 1.0.1
- **README.md**: Enhanced with LLM integration section
  - Added installation with `[llm]` extras
  - Added AI-powered usage examples
  - Updated navigation table
  - Added cross-references to new docs
- **Package Exports** (`__init__.py`):
  - Exported `LLMSuggestor`, `LLMProvider`
  - Exported `AutoPrepMLConfig`
  - Updated author information
- **Setup Files** (`setup.py`, `pyproject.toml`):
  - Added `autoprepml-config` entry point
  - Added optional LLM dependencies
  - Synchronized versions to 1.0.1
- **Core Module** (`core.py`):
  - Extended with LLM support
  - Extended with advanced feature support
  - Added 4 new public methods
  - Backward compatible with v1.0

### Fixed
- **Test Compatibility**: Updated `test_llm_suggest.py` for new API
- **F-string Formatting**: Fixed in `config_manager.py`
- **Error Handling**: Added column existence check in `_get_column_info()`
- **Pytest Markers**: Registered custom `integration` marker

### Documentation Updates
- All version numbers synchronized to 1.0.1
- All examples updated to show new features
- All imports verified and working
- Cross-references between documents updated
- Installation instructions enhanced
- Navigation improved with new sections

### Security
- API keys stored securely in user home directory
- Keys masked in all output displays
- Support for environment variables
- Optional dependencies prevent bloat

## [1.0.0] - 2025-01-23

### Added
- **Multi-Modal Support**: Complete preprocessing for 4 data types
  - Tabular data (AutoPrepML)
  - Text/NLP data (TextPrepML)
  - Time series data (TimeSeriesPrepML)
  - Graph data (GraphPrepML)
- **Comprehensive Testing**: 103 tests with 95%+ coverage
- **HTML/JSON Reports**: Visual reports for all data types
- **CLI Interface**: Command-line tool with multiple options
- **Configuration System**: YAML/JSON config support
- **Demo Scripts**: Working examples for all data types
- **Documentation**: Complete API reference and tutorials

### Changed
- Updated Python requirement to 3.10+ (from 3.8+)
- Pinned all dependencies to exact versions for stability
- Restructured README for better navigation (reduced from 989 to 601 lines)
- Updated GitHub Actions to latest versions (v4/v5)

### Fixed
- Python 3.8 compatibility issues (replaced match/case)
- 16 ruff linting errors
- Test assertion failures
- GitHub Actions deprecation warnings

### Documentation
- Added flow diagram for data preprocessing
- Created quick navigation table
- Consolidated CLI reference table
- Added examples directory overview table
- Restructured roadmap with version targets
- Reduced redundancy in contact/license sections

## [0.1.0] - 2024-12-15

### Added
- Initial release with tabular data support
- Basic detection and cleaning functions
- JSON/HTML report generation
- CLI support
- 41 unit tests

---

## Version Naming Convention

- **Major (X.0.0)**: Breaking changes, new data type support
- **Minor (1.X.0)**: New features, backward compatible
- **Patch (1.0.X)**: Bug fixes, documentation updates

## Upcoming Releases

See [README.md Roadmap](README.md#️-roadmap) for planned features.
