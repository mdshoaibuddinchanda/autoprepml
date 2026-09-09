#!/bin/bash
# Script to release AutoPrepML to PyPI

set -e

echo "AutoPrepML Release Script"
echo "========================="

# Check if version is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 0.1.0"
    exit 1
fi

VERSION=$1

PROJECT_VERSION=$(python -c 'import re; text=open("pyproject.toml", encoding="utf-8").read(); print(re.search(r"^version = \"([^\"]+)\"$", text, re.MULTILINE).group(1))')
if [ "$VERSION" != "$PROJECT_VERSION" ]; then
    echo "Version mismatch: pyproject.toml contains $PROJECT_VERSION, requested $VERSION"
    exit 1
fi

echo "Releasing version: $VERSION"

# Run tests first
echo "Running tests..."
pytest tests/ -v

# Build package
echo "Building package..."
python -m build

# Validate metadata before any network mutation.
python -m twine check dist/*

if [ -z "${TWINE_PASSWORD:-}" ] && [ -z "${TWINE_REPOSITORY_URL:-}" ]; then
    echo "Set a fresh scoped TWINE_PASSWORD or configure a trusted publishing repository before upload."
    exit 1
fi

# Upload to PyPI
echo "Uploading to PyPI..."
python -m twine upload --non-interactive dist/*

echo "Release complete! Version $VERSION published to PyPI."
