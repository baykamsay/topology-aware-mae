"""
Configuration utilities for loading YAML configs.
"""

import os
import yaml
import logging
from collections.abc import Mapping

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} does not exist.")
        return None

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {config_path}\n{e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the config: {e}")
        return None
    
def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modifies `source` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, Mapping) and key in source and isinstance(source.get(key, {}), Mapping):
            source[key] = deep_update(source.get(key, {}), value)
        else:
            source[key] = value
    return source

if __name__ == '__main__':
    # Example usage
    # Create a dummy config for testing relative to this script's location
    script_dir = os.path.dirname(__file__)
    dummy_config_path = os.path.join(script_dir, '..', 'configs', 'pretrain', 'dummy_test.yaml')

    dummy_content = {
        'model': {
            'name': 'test_model',
            'patch_size': 16,
        },
        'data': {
            'input_size': 112,
            'path': '/path/to/data',
        },
        'training': {
            'lr': 1e-4,
            'batch_size': 32,
        }
    }

    try:
        with open(dummy_config_path, 'w') as f:
            yaml.dump(dummy_content, f)
        
        print(f"Attempting to load dummy config from {dummy_config_path}")
        loaded_cfg = load_config(dummy_config_path)

        if loaded_cfg:
            print("Loaded configuration:")
            import json
            print(json.dumps(loaded_cfg, indent=2))
            print("\nAccessing values:")
            print(f"Model Name: {loaded_cfg.get('model', {}).get('name')}")
            print(f"Input Size: {loaded_cfg.get('data', {}).get('input_size')}")
            print(f"Learning Rate: {loaded_cfg.get('training', {}).get('lr')}")
        else:
            print("\nFailed to load dummy configuration.")
    
    finally:
        # Clean up the dummy config file after testing
        if os.path.exists(dummy_config_path):
            os.remove(dummy_config_path)
            print(f"Removed dummy config file: {dummy_config_path}")