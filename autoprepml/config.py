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
        "encode_method": "label",
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

    # Merge with defaults (user config overrides defaults) without sharing nested state.
    config = copy.deepcopy(DEFAULT_CONFIG)
    for section, values in user_config.items():
        if section in config and not isinstance(values, dict):
            raise ValueError(f"Configuration section '{section}' must be an object")
        if section in config and isinstance(config[section], dict):
            config[section].update(copy.deepcopy(values))
        else:
            config[section] = copy.deepcopy(values)

    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config file
    """
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary")

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
