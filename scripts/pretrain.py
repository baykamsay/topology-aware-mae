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
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Adjust the path to include the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.config import load_config
from models import fcmae

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

def get_param_groups(model, weight_decay):
    """
    Assigns different weight decay to different parameter groups.
    Biases, LayerNorm/GroupNorm weights & biases are not decayed.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check for parameters that should not be decayed
        if len(param.shape) == 1 or name.endswith(".bias") or \
           isinstance(param, (torch.nn.modules.normalization.LayerNorm, 
                               torch.nn.modules.normalization.GroupNorm)) or \
           ".norm." in name or ".norm" in name.lower() or \
           name.endswith(".gamma") or name.endswith(".beta"): # For custom GRN params
            no_decay.append(param)
            # print(f"No decay for: {name}")
        else:
            decay.append(param)
            # print(f"Decay for: {name}")
            
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.}
    ]

def main(args):
    # Load the configuration
    config = load_config(args.config)
    if config is None:
        print("Failed to load configuration. Exiting.")
        sys.exit(1)
    
    # Set up directories for logging and checkpoints
    os.makedirs(config.get('logging', {}).get('output_dir', './output/pretrain'), exist_ok=True)
    
    # Print the loaded configuration
    print("Loaded configuration:")
    pprint.pprint(config)

    # ---- Start setting up device ----
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available. Using CPU.")
    # ---- End setting up device ----

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
        transforms.RandomResizedCrop(input_size, scale=(0.2, 0.5), ratio=(1, 1), interpolation=transforms.InterpolationMode.BICUBIC),
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
    pin_memory = True

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

    # ---- Start Model setup ----
    logger.info("Setting up model...")
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'convnextv2_pico')
    model_params = {k: v for k, v in model_config.items() if k != "name"}
    model_params['loss_config'] = config.get('loss', {'name': 'mse'})

    try:
        model = fcmae.__dict__[model_name](**model_params)
        model.to(device)
        logger.info(f"Model {model_name} loaded successfully.")
        print(f"Model architecture: {model}")
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        sys.exit(1)
    # ---- End Model setup ----

    # ---- Start Optimizer and Scheduler setup ----
    logger.info("Setting up optimizer and scheduler...")
    training_config = config.get('training', {})
    optimizer_name = training_config.get('optimizer', 'adamw').lower()
    lr = training_config.get('learning_rate', 1.5e-4)
    min_lr = training_config.get('min_lr', 0.0)
    weight_decay = training_config.get('weight_decay', 0.05)
    beta1 = training_config.get('beta1', 0.9)
    beta2 = training_config.get('beta2', 0.95)

    if weight_decay > 0:
        param_groups = get_param_groups(model, weight_decay)
        logger.info(f"Applied custom weight decay: {weight_decay} for decay group, 0 for no_decay group.")
    else:
        param_groups = model.parameters() # No weight decay applied or handled by optimizer default
        logger.info("No custom weight decay grouping applied (weight_decay is 0 or not positive).")

    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            betas=(beta1, beta2)
        )
    else:
        logger.error(f"Optimizer {optimizer_name} not supported.")
        sys.exit(1)
    
    num_decay_params = sum(p.numel() for p in param_groups[0]['params'] if p.requires_grad) if isinstance(param_groups, list) and len(param_groups)>0 else 0
    num_no_decay_params = sum(p.numel() for p in param_groups[1]['params'] if p.requires_grad) if isinstance(param_groups, list) and len(param_groups)>1 else 0
    logger.info(f"Number of decayed parameters: {num_decay_params}")
    logger.info(f"Number of non-decayed parameters: {num_no_decay_params}")
    logger.info(f"Optimizer: {optimizer_name} initialized with base LR: {lr}")

    # Learning Rate Scheduler (Cosine Annealing with Warmup)
    logger.info("Setting up learning rate scheduler with warmup...")
    warmup_epochs = training_config.get('warmup_epochs', 40)
    total_epochs = training_config.get('epochs', 800)
    min_lr = training_config.get('min_lr', 0.0)
    
    if warmup_epochs > 0:
        # Scheduler for the warmup phase
        # It will linearly increase the LR from base_lr * start_factor to base_lr * end_factor
        # To go from min_lr to base_lr:
        warmup_start_factor = min_lr / lr if min_lr > 0 else 1e-7

        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=warmup_start_factor, # Start LR = base_lr * start_factor
            end_factor=1.0,         # End LR = base_lr * end_factor
            total_iters=warmup_epochs # Number of epochs for warmup
        )
        logger.info(f"Warmup scheduler: LinearLR for {warmup_epochs} epochs, from LR {lr*warmup_start_factor:.2e} to {lr*1.0:.2e}.")

        # Scheduler for the decay phase (post-warmup)
        # T_max is the number of epochs for one cycle of cosine annealing
        epochs_after_warmup = total_epochs - warmup_epochs
        scheduler_cosine_decay = CosineAnnealingLR(
            optimizer,
            T_max=epochs_after_warmup if epochs_after_warmup > 0 else 1, # Ensure T_max is positive
            eta_min=min_lr      # Minimum learning rate
        )
        logger.info(f"Cosine decay scheduler: CosineAnnealingLR for {epochs_after_warmup} epochs, to min_lr {min_lr:.2e}.")

        # Combine them sequentially
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine_decay],
            milestones=[warmup_epochs] # Epoch at which to switch from scheduler_warmup to scheduler_cosine_decay
        )
    else: # No warmup, just cosine decay from the start
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=min_lr
        )
        logger.info(f"LR Scheduler: CosineAnnealingLR for {total_epochs} epochs, to min_lr {min_lr:.2e} (no warmup).")
    
    logger.info(f"Base LR for schedulers: {lr}, Min LR: {min_lr}")
    # ---- End Optimizer and LR Scheduler Setup ----

    # ---- Start Mixed Precision Setup ----
    use_mixed_precision = training_config.get('mixed_precision', True)
    grad_scaler = None
    if use_mixed_precision and device.type == 'cuda':
        grad_scaler = torch.amp.GradScaler('cuda')
        logger.info("Mixed precision training enabled with GradScaler.")
    elif use_mixed_precision and device.type == 'cpu':
        logger.warning("Mixed precision is enabled in config, but device is CPU. It will not be used.")
    # ---- End Mixed Precision Setup ----

    # ---- Placeholder for Checkpoint Loading ----
    start_epoch = 0
    # Implement this later
    # ---- End Placeholder for Checkpoint Loading ----


    # ---- Start Training Loop ----
    logger.info(f"Starting pretraining for {total_epochs} epochs...")
    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False) # Requires tqdm

        for batch_idx, (samples, _) in enumerate(train_loader): # We don't use labels for MAE
            samples = samples.to(device, non_blocking=True)
            
            # Zero gradients for each batch
            optimizer.zero_grad()

            # Mixed precision forward pass
            if use_mixed_precision and grad_scaler:
                with torch.amp.autocast('cuda'):
                    loss, _, _ = model(samples, mask_ratio=model_config.get('mask_ratio', 0.6))
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else: # Standard precision
                loss, _, _ = model(samples, mask_ratio=model_config.get('mask_ratio', 0.6))
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()

            if (batch_idx + 1) % config.get('logging', {}).get('log_interval', 50) == 0: # Log every N batches
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch: {epoch+1}/{total_epochs} | Batch: {batch_idx+1}/{len(train_loader)} | Training Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

        epoch_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr'] # Get LR after scheduler step (if any per epoch)
        logger.info(f"===> Epoch {epoch+1}/{total_epochs} | Average Training Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}")

        # Step the LR scheduler (typically after each epoch)
        lr_scheduler.step()

        # ---- Placeholder for Validation ----
        # if (epoch + 1) % config.get('logging', {}).get('val_interval', 1) == 0:
        #     if val_loader:
        #         # validation_loss = validate(val_loader, model, device, config)
        #         # logger.info(f"Epoch {epoch+1} Validation Loss: {validation_loss:.4f}")
        #         pass # We will implement this
        #     else:
        #         logger.info(f"Epoch {epoch+1}: No validation loader available, skipping validation.")
        # ---- End Placeholder for Validation ----

        # ---- Placeholder for Checkpointing ----
        # if (epoch + 1) % config.get('logging', {}).get('val_interval', 1) == 0: # Tying to val_interval as requested
            # save_checkpoint(...)
            # pass # We will implement this
        # ---- End Placeholder for Checkpointing ----
        
        # ---- Placeholder for W&B Logging & Visualization ----
        # if wandb_logger is not None:
            # Log metrics
            # if (epoch + 1) % config.get('logging', {}).get('vis_interval', 20) == 0:
                # Log visualizations
        # ---- End Placeholder for W&B Logging & Visualization ----

    # --- End of Training Loop ---
    logger.info("Pretraining completed successfully.")
    # --- End of placeholder ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Topo MAE Pre-training', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)