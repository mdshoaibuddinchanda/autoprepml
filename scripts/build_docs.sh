#!/bin/bash
# Script to build documentation

echo "Building AutoPrepML documentation..."

cd docs
mkdocs build
echo "Documentation built successfully. Check site/ folder."
