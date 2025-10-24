"""AutoPrepML - AI-Assisted Multi-Modal Data Preprocessing Pipeline

A Python library for automatic detection, cleaning, and reporting of common
data quality issues in machine learning pipelines with LLM-powered suggestions.
"""

__version__ = "1.3.0"
__author__ = "MD Shoaibuddin Chanda"
__license__ = "MIT"

from .core import AutoPrepML
from .text import TextPrepML
from .timeseries import TimeSeriesPrepML
from .graph import GraphPrepML
from .image import ImagePrepML
from .autoeda import AutoEDA
from .feature_engine import AutoFeatureEngine, auto_feature_engineering
from .dashboard import InteractiveDashboard, create_plotly_dashboard, generate_streamlit_app
from . import detection
from . import cleaning
from . import visualization
from . import reports
from . import config
from . import llm_suggest
from .config_manager import AutoPrepMLConfig
from .llm_suggest import (
    LLMSuggestor, 
    LLMProvider, 
    suggest_column_rename,
    generate_data_documentation
)

__all__ = [
    # Core preprocessing
    "AutoPrepML",
    "TextPrepML",
    "TimeSeriesPrepML",
    "GraphPrepML",
    "ImagePrepML",
    
    # New v1.3.0 features
    "AutoEDA",
    "AutoFeatureEngine",
    "auto_feature_engineering",
    "InteractiveDashboard",
    "create_plotly_dashboard",
    "generate_streamlit_app",
    
    # LLM features
    "LLMSuggestor",
    "LLMProvider",
    "AutoPrepMLConfig",
    "suggest_column_rename",
    "generate_data_documentation",
    
    # Modules
    "detection",
    "cleaning",
    "visualization",
    "reports",
    "config",
    "llm_suggest",
]
