"""
Base pretraining script.
"""

import argparse
import os
import sys
import pprint

# Adjust the path to include the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.config import load_config

def get_args_parser():
    parser = argparse.ArgumentParser('Topo MAE Pre-training', add_help=False)

    # Configuration file path
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        default=os.path.join(project_root, "configs", "pretrain", "base_pretrain.yaml"),
        help="Path to the configuration YAML file",
    )

    return parser

def main(args):
    # Load the configuration
    config = load_config(args.config)
    if config is None:
        print("Failed to load configuration. Exiting.")
        sys.exit(1)
    
    # Set up directories for logging and checkpoints
    os.makedirs(config.get('logging', {}).get('log_dir', './output/logs/pretrain'), exist_ok=True)
    os.makedirs(config.get('logging', {}).get('output_dir', './output/checkpoints/pretrain'), exist_ok=True)
    
    # Print the loaded configuration
    print("Loaded configuration:")
    pprint.pprint(config)

    # --- Placeholder for the rest of the pretraining script ---
    print("Starting pretraining...")
    print("Pretraining completed successfully.")
    # --- End of placeholder ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Topo MAE Pre-training', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)