"""Configuration management for AutoPrepML"""

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG = {
    "cleaning": {
        "missing_strategy": "auto",
        "numeric_strategy": "median",
        "categorical_strategy": "mode",
        "outlier_method": "iforest",
        "outlier_contamination": 0.05,
        "remove_outliers": False,
        "scale_method": "standard",
        # One-hot encoding is the safe default for nominal feature columns;
        # label encoding imposes an artificial order and remains available
        # when a caller explicitly selects it.
        "encode_method": "onehot",
        "balance_method": "oversample",
    },
    "detection": {
        "outlier_method": "iforest",
        "contamination": 0.05,
        "zscore_threshold": 3.0,
        "imbalance_threshold": 0.3,
    },
    "reporting": {
        "format": "html",
        "include_plots": True,
        "plot_style": "seaborn",
        "output_dir": "./reports",
    },
    "logging": {"enabled": True, "level": "INFO", "file": "autoprepml.log"},
}


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge defaults and validate known production configuration values.

    Unknown top-level sections remain available for extensions, while known
    controls fail early instead of producing a partially transformed dataset.
    """
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary")
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for section, values in config.items():
        if section in merged and not isinstance(values, dict):
            raise ValueError(f"Configuration section '{section}' must be an object")
        if section in merged:
            merged[section].update(copy.deepcopy(values))
        else:
            merged[section] = copy.deepcopy(values)

    cleaning = merged["cleaning"]
    enum_values = {
        "missing_strategy": {"auto", "median", "mean", "mode", "drop"},
        "numeric_strategy": {"median", "mean", "mode"},
        "categorical_strategy": {"mode"},
        "outlier_method": {"iforest", "zscore"},
        "scale_method": {"standard", "minmax", "robust", "maxabs"},
        "encode_method": {"onehot", "label"},
        "balance_method": {"oversample", "undersample", "smote"},
    }
    for key, allowed in enum_values.items():
        if cleaning.get(key) not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"cleaning.{key} must be one of: {choices}")
    for key in ("remove_outliers",):
        if not isinstance(cleaning.get(key), bool):
            raise TypeError(f"cleaning.{key} must be a boolean")

    detection = merged["detection"]
    for key in ("outlier_method",):
        if detection.get(key) not in {"iforest", "zscore"}:
            raise ValueError(f"detection.{key} must be iforest or zscore")
    for key in ("contamination", "imbalance_threshold"):
        value = detection.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ValueError(f"detection.{key} must be a number between 0 and 1")
    threshold = detection.get("zscore_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold <= 0:
        raise ValueError("detection.zscore_threshold must be a positive number")
    return merged


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file. If None, returns default config.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError("Config file must be YAML or JSON")

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            user_config = (
                yaml.safe_load(config_file)
                if suffix in {".yaml", ".yml"}
                else json.load(config_file)
            )
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid configuration file: {config_path}") from exc

    if user_config is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    if not isinstance(user_config, dict):
        raise ValueError("Configuration root must be an object")

    return validate_config(user_config)


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config file
    """
    config = validate_config(config)

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError("Config file must be YAML or JSON")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=f"{output_path.suffix}.tmp",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as config_file:
            if suffix == ".json":
                json.dump(config, config_file, indent=2)
            else:
                yaml.safe_dump(config, config_file, default_flow_style=False, sort_keys=False)
            config_file.write("\n")
            config_file.flush()
            os.fsync(config_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def get_default_config() -> Dict[str, Any]:
    """Return a copy of the default configuration.

    Returns:
        Default configuration dictionary
    """
    return copy.deepcopy(DEFAULT_CONFIG)
