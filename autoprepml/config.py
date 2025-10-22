"""Configuration management for AutoPrepML"""
import os
import json
import yaml
from typing import Dict, Any, Optional


DEFAULT_CONFIG = {
    'cleaning': {
        'missing_strategy': 'auto',
        'numeric_strategy': 'median',
        'categorical_strategy': 'mode',
        'outlier_method': 'iforest',
        'outlier_contamination': 0.05,
        'remove_outliers': False,
        'scale_method': 'standard',
        'encode_method': 'label',
        'balance_method': 'oversample'
    },
    'detection': {
        'outlier_method': 'iforest',
        'contamination': 0.05,
        'zscore_threshold': 3.0,
        'imbalance_threshold': 0.3
    },
    'reporting': {
        'format': 'html',
        'include_plots': True,
        'plot_style': 'seaborn',
        'output_dir': './reports'
    },
    'logging': {
        'enabled': True,
        'level': 'INFO',
        'file': 'autoprepml.log'
    }
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file. If None, returns default config.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        return DEFAULT_CONFIG.copy()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            user_config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            user_config = json.load(f)
        else:
            raise ValueError("Config file must be YAML or JSON")
    
    # Merge with defaults (user config overrides defaults)
    config = DEFAULT_CONFIG.copy()
    if user_config:
        for section, values in user_config.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(values)
            else:
                config[section] = values
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        if output_path.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Dict[str, Any]:
    """Return a copy of the default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return DEFAULT_CONFIG.copy()
