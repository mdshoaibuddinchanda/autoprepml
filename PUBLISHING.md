# Publishing AutoPrepML to GitHub and PyPI

This guide explains how to publish AutoPrepML v1.2.0 to GitHub and PyPI.

## Prerequisites

### 1. GitHub Setup
- Ensure you have a GitHub account
- Create a repository at: https://github.com/mdshoaibuddinchanda/autoprepml
- Install Git: https://git-scm.com/downloads

### 2. PyPI Setup
- Create PyPI account: https://pypi.org/account/register/
- Create API token: https://pypi.org/manage/account/token/
- Install build tools:
```bash
pip install --upgrade build twine
```

## Step 1: Prepare the Repository

### Clean Build Artifacts
```powershell
# Remove old build artifacts
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path autoprepml.egg-info) { Remove-Item -Recurse -Force autoprepml.egg-info }
```

### Verify All Tests Pass
```powershell
pytest -v --tb=short -W ignore::DeprecationWarning
```

Expected: **158 passed, 8 skipped** (SMOTE tests require imbalanced-learn)

## Step 2: Build the Distribution

### Create Source and Wheel Distributions
```powershell
python -m build
```

This creates:
- `dist/autoprepml-1.2.0.tar.gz` (source distribution)
- `dist/autoprepml-1.2.0-py3-none-any.whl` (wheel distribution)

### Verify the Build
```powershell
twine check dist/*
```

Expected output: `Checking dist\autoprepml-1.2.0.tar.gz: PASSED`

## Step 3: Publish to PyPI

### Option A: Test PyPI First (Recommended)
```powershell
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Install from Test PyPI to verify
pip install --index-url https://test.pypi.org/simple/ --no-deps autoprepml

# Test the installation
python -c "from autoprepml import AutoPrepML; print('Success!')"
```

### Option B: Publish to Production PyPI
```powershell
# Upload to PyPI
twine upload dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your PyPI API token (starts with `pypi-`)

### Post-Publication Verification
```powershell
# Install from PyPI
pip install autoprepml

# Verify installation
python -c "from autoprepml import AutoPrepML, ImagePrepML; print('v1.2.0 installed!')"
```

## Step 4: Push to GitHub

### Initialize Git Repository (if not already done)
```powershell
git init
git add .
git commit -m "Release v1.2.0 - Image preprocessing + dynamic LLM configuration"
```

### Connect to GitHub Remote
```powershell
git remote add origin https://github.com/mdshoaibuddinchanda/autoprepml.git
git branch -M main
```

### Push to GitHub
```powershell
git push -u origin main
```

### Create a GitHub Release

1. Go to: https://github.com/mdshoaibuddinchanda/autoprepml/releases/new
2. Tag version: `v1.2.0`
3. Release title: `AutoPrepML v1.2.0 - Image Preprocessing + Dynamic LLM`
4. Description:
```markdown
## What's New in v1.2.0

### 🖼️ Image Preprocessing Module
- Complete image data preprocessing pipeline
- Support for PNG, JPG, JPEG, BMP, GIF, TIFF, WEBP
- Automatic issue detection (corruption, size mismatch, color mode)
- Batch resizing and normalization
- Dataset splitting (train/val/test)
- HTML report generation

### 🤖 Dynamic LLM Configuration
- Fully configurable model selection via environment variables
- Support for OpenAI (GPT-4o), Anthropic (Claude-3.5), Google (Gemini-2.5), Ollama
- No hardcoded model names
- Enhanced security with configurable safety settings

### 📦 Installation
```bash
pip install autoprepml
```

### 🚀 Quick Start
```python
from autoprepml import ImagePrepML

# Image preprocessing
prep = ImagePrepML(image_dir='./images', target_size=(224, 224))
prep.detect()
processed = prep.clean()
prep.save_report('report.html')
```

### 📊 Test Coverage
- 158 tests passing
- 5 data modalities supported
- Production ready

### 🔗 Links
- Documentation: https://github.com/mdshoaibuddinchanda/autoprepml#readme
- PyPI: https://pypi.org/project/autoprepml/
- Issues: https://github.com/mdshoaibuddinchanda/autoprepml/issues
```

5. Attach files:
   - `dist/autoprepml-1.2.0.tar.gz`
   - `dist/autoprepml-1.2.0-py3-none-any.whl`

6. Click "Publish release"

## Step 5: Update README Badges

After publishing, update the PyPI badge in README.md:
```markdown
[![PyPI version](https://img.shields.io/pypi/v/autoprepml.svg)](https://pypi.org/project/autoprepml/)
[![Downloads](https://pepy.tech/badge/autoprepml)](https://pepy.tech/project/autoprepml)
```

## Verification Checklist

- [ ] All 158 tests passing
- [ ] Version updated to 1.2.0 in:
  - [ ] `setup.py`
  - [ ] `pyproject.toml`
  - [ ] `autoprepml/__init__.py`
  - [ ] `CHANGELOG.md`
- [ ] CHANGELOG.md updated with v1.2.0 features
- [ ] README.md updated with new features
- [ ] MANIFEST.in includes all necessary files
- [ ] requirements.txt includes Pillow
- [ ] Build succeeds: `python -m build`
- [ ] Twine check passes: `twine check dist/*`
- [ ] Uploaded to PyPI
- [ ] Pushed to GitHub
- [ ] GitHub release created
- [ ] Installation verified: `pip install autoprepml`

## Troubleshooting

### Build Fails
```powershell
# Clean everything and rebuild
Remove-Item -Recurse -Force dist, build, *.egg-info
python -m build
```

### Upload Fails (403 Error)
- Verify API token is correct
- Check package name is not taken
- Ensure version 1.2.0 hasn't been uploaded before

### Git Push Fails
```powershell
# If remote already exists
git remote remove origin
git remote add origin https://github.com/mdshoaibuddinchanda/autoprepml.git
git push -u origin main --force  # Use with caution
```

### Tests Fail
```powershell
# Run specific test
pytest tests/test_image.py -v

# Run with more details
pytest -vv --tb=long
```

## Version Bump for Future Releases

For the next release, update version in:
1. `setup.py`: `version="x.x.x"`
2. `pyproject.toml`: `version = "x.x.x"`
3. `autoprepml/__init__.py`: `__version__ = "x.x.x"`
4. `CHANGELOG.md`: Add new `## [x.x.x] - YYYY-MM-DD` section

## Support

- Issues: https://github.com/mdshoaibuddinchanda/autoprepml/issues
- Email: mdshoaibuddinchanda@gmail.com
- PyPI: https://pypi.org/project/autoprepml/
