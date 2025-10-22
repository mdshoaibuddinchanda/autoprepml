# Changelog

All notable changes to AutoPrepML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
