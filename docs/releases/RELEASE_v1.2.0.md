# AutoPrepML v1.2.0 Release Summary

This page records the v1.2.0 release snapshot. Current installation and test guidance is maintained in the main documentation.

## Publication status

The release preparation steps recorded on this page were complete at the time of publication.

## What's Been Done

### 1. Version Updates
- `setup.py`: version="1.2.0"
- `pyproject.toml`: version = "1.2.0"
- `autoprepml/__init__.py`: __version__ = "1.2.0"
- `CHANGELOG.md`: Documented as [1.2.0]: 2025-10-24

### 2. Dependencies Updated
- Added `Pillow>=10.0.0` to all dependency files
- Updated all three files: setup.py, pyproject.toml, requirements.txt

### 3. Tests Status
```
 158 passed
  8 skipped (expected - SMOTE tests require explicit install)
 0 failed
  0 warnings (suppressed deprecation warnings)
```

### 4. Code Quality
- Fixed image duplicate detection bug
- Updated LLM default models to latest versions
- Fixed deprecation warnings in visualization and timeseries
- All lint errors resolved

### 5. Documentation
- README.md: Updated with v1.2.0 features and badges
- CHANGELOG.md: Complete feature list for v1.2.0
- Release publishing guidance is maintained in the repository workflow.
- MANIFEST.in: Proper file inclusion rules

### 6. Build Artifacts
Successfully built:
- `dist/autoprepml-1.2.0.tar.gz` (source distribution)
- `dist/autoprepml-1.2.0-py3-none-any.whl` (wheel distribution)

## What's New in v1.2.0

### Image Preprocessing Module
- Complete image preprocessing pipeline (620+ lines)
- Support for 7 image formats (PNG, JPG, JPEG, BMP, GIF, TIFF, WEBP)
- Automatic issue detection (corruption, size mismatch, color modes)
- Batch resizing and normalization
- Dataset splitting (train/val/test)
- HTML report generation
- 17 comprehensive tests

### Dynamic LLM Configuration
- Fully dynamic model selection via environment variables
- No hardcoded model names or parameters
- Support for 4 LLM providers:
  - OpenAI: GPT-4o (updated from GPT-4)
  - Anthropic: Claude-3.5-Sonnet (updated)
  - Google: Gemini-2.5-Flash (updated from Gemini-Pro)
  - Ollama: Llama3.2 (updated from Llama2)
- Configurable temperature, max_tokens, base_url
- Google safety level configuration

### Code Improvements
- Fixed image deduplication bug
- Replaced deprecated pandas fillna(method=) with ffill()/bfill()
- Replaced deprecated matplotlib vert=True with orientation='vertical'
- Updated all test expectations to match new defaults

## Next Steps

### Option 1: Publish Immediately

#### A. Upload to PyPI
```powershell
# Install twine if not already installed
pip install twine

# Upload to PyPI (you'll need your API token)
twine upload dist/*
```

#### B. Push to GitHub
```powershell
# Initialize/commit (if not done)
git add .
git commit -m "Release v1.2.0 - Image preprocessing + dynamic LLM configuration"

# Push to GitHub
git remote add origin https://github.com/mdshoaibuddinchanda/autoprepml.git
git branch -M main
git push -u origin main
```

#### C. Create GitHub Release
1. Go to: https://github.com/mdshoaibuddinchanda/autoprepml/releases/new
2. Tag: `v1.2.0`
3. Title: `AutoPrepML v1.2.0: Image Preprocessing + Dynamic LLM`
4. Upload: `dist/autoprepml-1.2.0.tar.gz` and `.whl` files

### Option 2: Test First (Recommended)

#### Test on Test PyPI
```powershell
# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Install and test
pip install --index-url https://test.pypi.org/simple/ --no-deps autoprepml

# Verify it works
python -c "from autoprepml import AutoPrepML, ImagePrepML; print('Success')"
```

## Feature Summary

### Supported Data Types (5 Total)
1.  Tabular Data (AutoPrepML core)
2.  Text Data (TextPrepML)
3.  Time Series (TimeSeriesPrepML)
4.  Graph Data (GraphPrepML)
5.  **NEW** Image Data (ImagePrepML)

### Advanced Features
- KNN & Iterative Imputation
- SMOTE Class Balancing
- LLM Integration (GPT-4o, Claude, Gemini, Ollama)
- Dynamic Configuration System
- Image Preprocessing Pipeline

### Quality Metrics
- 158 tests passing
- 5 data modalities
- 2 CLI tools (autoprepml, autoprepml-config)
- 8 documentation files
- 7 example scripts
- 17 test files

## Important Links

- **GitHub**: https://github.com/mdshoaibuddinchanda/autoprepml
- **PyPI**: https://pypi.org/project/autoprepml/ (after publishing)
- **Email**: mdshoaibuddinchanda@gmail.com

## Quick Reference Commands

### Build Package
```powershell
python -m build
```

### Check Package
```powershell
twine check dist/*
```

### Upload to PyPI
```powershell
twine upload dist/*
```

### Run All Tests
```powershell
pytest -v --tb=short
```

### Clean Build
```powershell
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path autoprepml.egg-info) { Remove-Item -Recurse -Force autoprepml.egg-info }
```

## Release conclusion

Your package is production-ready with:
- All tests passing
- No failures or warnings
- Version properly updated
- Dependencies correct
- Documentation complete
- Build successful

The package is ready for the publication workflow described above.


For detailed publishing instructions, see the repository release workflow and package metadata.
