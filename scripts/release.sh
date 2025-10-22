#!/bin/bash
# Script to release AutoPrepML to PyPI

set -e

echo "AutoPrepML Release Script"
echo "========================="

# Check if version is provided
if [ -z "$1" ]; then
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 0.1.0"
    exit 1
fi

VERSION=$1

echo "Releasing version: $VERSION"

# Run tests first
echo "Running tests..."
pytest tests/ -v

# Build package
echo "Building package..."
python -m build

# Upload to PyPI
echo "Uploading to PyPI..."
python -m twine upload dist/*

echo "Release complete! Version $VERSION published to PyPI."
