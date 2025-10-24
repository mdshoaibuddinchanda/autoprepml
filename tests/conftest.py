"""Pytest configuration and fixtures."""
import pytest

# Register markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "viz: tests that require visualization libraries (plotly, streamlit)"
    )
    config.addinivalue_line(
        "markers", "llm: tests that require LLM providers"
    )


# Skip markers for optional dependencies
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests based on missing dependencies."""
    # Check for visualization libraries
    try:
        import plotly
        import streamlit
        has_viz = True
    except ImportError:
        has_viz = False
    
    # Check for LLM libraries
    try:
        import openai
        has_llm = True
    except ImportError:
        has_llm = False
    
    skip_viz = pytest.mark.skip(reason="plotly or streamlit not installed")
    skip_llm = pytest.mark.skip(reason="LLM libraries not installed")
    
    for item in items:
        if "viz" in item.keywords and not has_viz:
            item.add_marker(skip_viz)
        if "llm" in item.keywords and not has_llm:
            item.add_marker(skip_llm)
