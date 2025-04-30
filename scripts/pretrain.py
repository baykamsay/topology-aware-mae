"""
Base pretraining script.
"""

import argparse
import os
import sys
import pprint
import logging
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Adjust the path to include the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.config import load_config
from models import convnextv2, fcmae

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    # ---- Start Dataset and DataLoader setup ----
    logger.info("Setting up dataset and DataLoader...")

    # Define Transforms
    input_size = config.get('data', {}).get('input_size', 128)
    # ImageNet default mean and std
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    # Basic pre-training transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.1, 0.5), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=[0, 90, 180, 270], interpolation=InterpolationMode.NEAREST, expand=False),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    # Validation transforms
    val_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.1, 0.5), ratio=(1, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    # Create datasets
    train_data_path = config.get('data', {}).get('data_path')
    val_data_path = config.get('data', {}).get('eval_data_path')
    
    if not train_data_path:
        logger.error("Data path for training is not specified in the configuration.")
        sys.exit(1)

    try:
        train_dataset = datasets.ImageFolder(train_data_path, transform=train_transform)
        logger.info(f"Training dataset loaded from: {train_data_path} using ImageFolder. Found {len(train_dataset)} samples.")
    except Exception as e:
        logger.error(f"Error loading training dataset: {e}")
        sys.exit(1)

    val_dataset = None
    if val_data_path:
        try:
            val_dataset = datasets.ImageFolder(val_data_path, transform=val_transform)
            logger.info(f"Validation dataset loaded from: {val_data_path} using ImageFolder. Found {len(val_dataset)} samples.")
        except Exception as e:
            logger.warning(f"Error loading validation dataset: {e}")
            sys.exit(1) # or possibly continue if validation is optional
    
    # Create DataLoaders
    num_workers = config.get('data', {}).get('num_workers', 4)
    batch_size = config.get('training', {}).get('batch_size', 32)
    pin_memory = config.get('data', {}).get('pin_mem', True)

    # Use standard samplers for now
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset) if val_dataset else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    logger.info(f"Training DataLoader created with batch size {batch_size} and {num_workers} workers.")

    val_loader = None
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        logger.info(f"Validation DataLoader created with batch size {batch_size} and {num_workers} workers.")
    else:
        logger.info("No validation dataset provided, skipping DataLoader creation for validation.")
    # ---- End Dataset and DataLoader setup ----

    model_config = config.get('model', {})
    model_name = model_config.get('name', 'convnextv2_pico')
    model_params = {k: v for k, v in model_config.items() if k != "name"}
    # try:
    #     model = convnextv2.__dict__[model_name]()
    #     logger.info(f"Model {model_name} loaded successfully.")
    #     print(f"Model architecture: {model}")
    # except KeyError:
    #     logger.error(f"Model {model_name} is not defined in the models module.")
    #     sys.exit(1)
    # except Exception as e:
    #     logger.error(f"Error loading model {model_name}: {e}")
    #     sys.exit(1)

    try:
        model = fcmae.__dict__[model_name](**model_params)
        logger.info(f"Model {model_name} loaded successfully.")
        print(f"Model architecture: {model}")
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        sys.exit(1)

    # --- Placeholder for the rest of the pretraining script ---
    print("Starting pretraining...")
    print("Pretraining completed successfully.")
    # --- End of placeholder ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Topo MAE Pre-training', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)