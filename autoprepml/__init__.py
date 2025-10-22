"""AutoPrepML - Automated Data Preprocessing Pipeline

A Python library for automatic detection, cleaning, and reporting of common
data quality issues in machine learning pipelines.
"""

__version__ = "0.1.0"
__author__ = "AutoPrepML Contributors"
__license__ = "MIT"

from .core import AutoPrepML
from .text import TextPrepML
from .timeseries import TimeSeriesPrepML
from .graph import GraphPrepML
from . import detection
from . import cleaning
from . import visualization
from . import reports
from . import config
from . import llm_suggest

__all__ = [
    "AutoPrepML",
    "TextPrepML",
    "TimeSeriesPrepML",
    "GraphPrepML",
    "detection",
    "cleaning",
    "visualization",
    "reports",
    "config",
    "llm_suggest",
]
