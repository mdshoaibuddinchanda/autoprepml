#!/bin/bash
# Script to build documentation

set -euo pipefail

echo "Building AutoPrepML documentation..."

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdocs build --strict
echo "Documentation built successfully. Check site/ folder."
