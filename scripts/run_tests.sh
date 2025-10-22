#!/bin/bash
# Script to run all tests with coverage

echo "Running AutoPrepML test suite..."

pytest tests/ -v --cov=autoprepml --cov-report=html --cov-report=term

echo "Test complete. Coverage report generated in htmlcov/index.html"
