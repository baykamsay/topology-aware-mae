"""
Base pretraining script.
"""

import argparse
import datetime
import numpy as np
import time
import json
import os
import sys
import pprint
import logging
from pathlib import Path


import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Adjust the path to include the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import utils.distributed as dist
import utils.logging as logging_utils
import utils.optimizer as optim_utils
import utils.scaler as scaler_utils
import utils.checkpointing as checkpointing
from utils.config import load_config
from models import fcmae
from engine.pretrain import train_one_epoch

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

    # Get configs
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    logging_config = config.get('logging', {})
    distributed_config = config.get('distributed', {})

    # TODO Set up distributed training
    # dist.init_distributed_mode(distributed_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # don't use distributed mode for now

    # Set random seed for reproducibility
    seed = training_config.get('seed', 42) + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True # Enable benchmark mode for faster training on GPUs with fixed input sizes

    # ---- Start Dataset and DataLoader setup ----
    logger.info("Setting up dataset and DataLoader...")

    # Define Transforms
    input_size = data_config.get('input_size', 128)
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
    train_data_path = data_config.get('data_path')
    val_data_path = data_config.get('eval_data_path')
    
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
    

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    # Create DataLoaders
    num_workers = data_config.get('num_workers', 4)
    batch_size = training_config.get('batch_size', 32)
    pin_memory = data_config.get('pin_mem', True)

    # Use standard samplers for now TODO change to DistributedSampler
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset) if val_dataset else None

    # if global_rank == 0 and logging_config.get('enable_wandb', False):
    #     log_writer = logging_utils.WandbLogger(logging_config)
    # else:
    log_writer = None

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

    model_name = model_config.get('name', 'convnextv2_pico')
    model_params = {k: v for k, v in model_config.items() if k != "name"}

    try:
        model = fcmae.__dict__[model_name](**model_params)
        model.to(device)
        logger.info(f"Model {model_name} loaded successfully.")
        if global_rank == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'Number of trainable parameters: {n_params / 1e6:.2f} M')
            # if enable_wandb: wandb.config.update({"trainable_params_M": n_params / 1e6})
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        sys.exit(1)
    
    model_without_ddp = model

    effective_batch_size = batch_size * training_config.get('update_freq', 1) * num_tasks
    num_training_steps_per_epoch = len(train_dataset) // effective_batch_size

    if 'lr' not in training_config:
        training_config['lr'] = training_config['blr'] * effective_batch_size / 256.0
    
    logger.info(f"Effective batch size: {effective_batch_size}, Actual learning rate: {training_config['lr']}")

    if num_tasks > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[global_rank % torch.cuda.device_count()], find_unused_parameters=False) # Adjust device_ids if needed
        model_without_ddp = model.module
        logger.info("Model wrapped with DistributedDataParallel.")

    # Fix after here, just for testing
    # --- Optimizer ---
    optimizer = optim_utils.create_optimizer(training_config, model_without_ddp)
    logger.info(f"Optimizer created: {optimizer}")

    # --- Loss Scaler ---
    loss_scaler = scaler_utils.NativeScalerWithGradNormCount()
    logger.info("Loss scaler created.")

    checkpointing.auto_load_model(
        args=config.get('logging', {}), model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    
    print(f"Start training for {training_config['epochs']} epochs")
    start_time = time.time()
    for epoch in range(0, training_config['epochs']):
        if False:
            train_loader.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_steps()
        train_stats = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, loss_scaler,
            config=config
        )
        # if args.output_dir and args.save_ckpt:
        #     if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
        #         checkpointing.save_model(
        #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #             loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_params}
        # if args.output_dir and dist.is_main_process():
        #     if log_writer is not None:
        #         log_writer.flush()
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Topo MAE Pre-training', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)