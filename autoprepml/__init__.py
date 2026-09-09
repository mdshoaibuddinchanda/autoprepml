"""AutoPrepML - AI-Assisted Multi-Modal Data Preprocessing Pipeline

A Python library for automatic detection, cleaning, and reporting of common
data quality issues in machine learning pipelines with LLM-powered suggestions.
"""

__version__ = "1.4.0"
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
    generate_data_documentation,
)
from .batch import iter_chunks, iter_processed_chunks, process_chunks
from .storage import (
    FsspecStorageAdapter,
    InMemoryStorageAdapter,
    LocalStorageAdapter,
    StorageAdapter,
    get_storage_adapter,
)
from .streaming import stream_process, write_stream
from .pipeline import make_model_pipeline, make_preprocessing_pipeline
from .experiments import ExperimentRun, LocalExperimentTracker, MLflowExperimentTracker
from .normalization import (
    TabularNormalizer,
    denormalize_image_array,
    fit_image_statistics,
    fit_tabular_normalizer,
    normalize_image_array,
)
from .config import validate_config

__all__ = [
    # Core preprocessing
    "AutoPrepML",
    "TextPrepML",
    "TimeSeriesPrepML",
    "GraphPrepML",
    "ImagePrepML",
    # v1.3.0 and v1.4.0 features
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
    # Execution, storage, and integrations
    "iter_chunks",
    "iter_processed_chunks",
    "process_chunks",
    "StorageAdapter",
    "LocalStorageAdapter",
    "InMemoryStorageAdapter",
    "FsspecStorageAdapter",
    "get_storage_adapter",
    "stream_process",
    "write_stream",
    "make_preprocessing_pipeline",
    "make_model_pipeline",
    "ExperimentRun",
    "LocalExperimentTracker",
    "MLflowExperimentTracker",
    "TabularNormalizer",
    "fit_tabular_normalizer",
    "fit_image_statistics",
    "normalize_image_array",
    "denormalize_image_array",
    "validate_config",
    # Modules
    "detection",
    "cleaning",
    "visualization",
    "reports",
    "config",
    "llm_suggest",
]
