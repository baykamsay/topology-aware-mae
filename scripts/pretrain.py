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
import shutil
import random
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler

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

def validate(val_loader, model, device, config, use_mixed_precision, epoch):
    """
    Performs validation on the validation dataset.

    Args:
        val_loader (DataLoader): DataLoader for the validation set.
        model (nn.Module): The model to evaluate.
        device (torch.device): The device to run evaluation on.
        model_config_params (dict): Model-specific configuration parameters (e.g., mask_ratio).
        use_mixed_precision (bool): Whether to use mixed precision for evaluation.
        epoch (int): Current epoch number (for logging).
        total_epochs (int): Total number of epochs (for logging).

    Returns:
        float: Average validation loss.
    """
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    num_val_batches = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch_idx, (samples, _) in enumerate(val_loader):
            samples = samples.to(device, non_blocking=True)
            
            # Forward pass
            if use_mixed_precision and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    # The model's forward method returns: loss, predicted_patches, mask
                    loss, _, _ = model(samples, mask_ratio=config.get('mask_ratio', 0.6))
            else:
                loss, _, _ = model(samples, mask_ratio=config.get('mask_ratio', 0.6))
            
            total_val_loss += loss.item()
            num_val_batches += 1
            if (batch_idx + 1) % 20 == 0: # Log progress every 20 val batches
                 logger.debug(f"Validation Epoch {epoch+1} - Batch {batch_idx+1}/{len(val_loader)}")


    if num_val_batches == 0:
        logger.warning("Validation loader was empty. Returning 0 validation loss.")
        return 0.0
        
    avg_val_loss = total_val_loss / num_val_batches
    return avg_val_loss

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

def save_checkpoint(state, is_best, output_dir, filename="checkpoint.pth"):
    """
    Saves a training checkpoint.

    Args:
        state (dict): Contains model, optimizer, scheduler states, epoch, best_val_loss.
        is_best (bool): True if this checkpoint has the best validation loss so far.
        output_dir (str): Directory where checkpoints will be saved.
        filename (str): Base name for the checkpoint file.
    """
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

    if is_best:
        best_filepath = os.path.join(output_dir, "best_checkpoint.pth")
        shutil.copyfile(filepath, best_filepath)
        logger.info(f"Best checkpoint updated to {best_filepath} (Val Loss: {state.get('best_val_loss', 'N/A'):.4f})")

    # # Always save a 'latest' checkpoint for easy resume
    # latest_filepath = os.path.join(output_dir, "latest_checkpoint.pth")
    # shutil.copyfile(filepath, latest_filepath) # Or save directly to latest_checkpoint.pth
    # logger.info(f"Latest checkpoint saved to {latest_filepath}")

def main(args):
    # Load the configuration
    config = load_config(args.config)
    if config is None:
        print("Failed to load configuration. Exiting.")
        sys.exit(1)
    
    training_config = config.get('training', {})
    log_config = config.get('logging', {})
    output_dir = log_config.get('output_dir', './output/pretrain/test_run')
    
    # Set up directories for logging and checkpoints
    os.makedirs(output_dir, exist_ok=True)
    
    # Print the loaded configuration
    print("Loaded configuration:")
    pprint.pprint(config)

    # ---- Start setting up the seed ----
    seed = training_config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # monai.utils.set_determinism(seed=config.TRAIN.SEED) # type: ignore

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(seed)
    # ---- End setting up the seed ----

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
        transforms.Resize(375, interpolation=transforms.InterpolationMode.BICUBIC, max_size=750),
        transforms.CenterCrop(input_size),
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
        worker_init_fn=seed_worker,
        generator=g,
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
            worker_init_fn=seed_worker,
            generator=g,
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

    grad_acc_steps = config.get('training', {}).get('gradient_accumulation_steps', 1)
    if grad_acc_steps < 1:
        grad_acc_steps = 1
        logger.warning("gradient_accumulation_steps was less than 1, set to 1.")

    num_training_steps_per_epoch = len(train_loader) // grad_acc_steps
    total_epochs = config.get('training', {}).get('epochs', 800)
    total_training_steps = num_training_steps_per_epoch * total_epochs
    
    warmup_epochs = config.get('training', {}).get('warmup_epochs', 40)
    # Calculate warmup steps, ensuring it doesn't exceed total steps if epochs are very few
    warmup_steps = min(num_training_steps_per_epoch * warmup_epochs, total_training_steps)

    min_lr = training_config.get('min_lr', 0.0)
    
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_training_steps, # Total number of training steps for one cycle
        lr_min=min_lr,      # Minimum learning rate to decay to
        warmup_t=warmup_steps,          # Number of warmup steps
        warmup_lr_init=0.0,             # Start warmup from LR 0.0 (or a very small value like 1e-6 * lr)
        warmup_prefix=True,             # Ensures warmup is handled correctly before decay
        cycle_limit=1,                  # Number of cycles (1 for no restarts)
        t_in_epochs=False,              # t_initial and warmup_t are in steps, not epochs
    )
    logger.info(f"LR Scheduler: timm.CosineLRScheduler initialized.")
    logger.info(f"  Total training steps: {total_training_steps}")
    logger.info(f"  Warmup steps: {warmup_steps} (equivalent to {warmup_epochs} epochs of {num_training_steps_per_epoch} grad steps)")
    logger.info(f"  Base LR: {lr}, Min LR: {min_lr}, Warmup Init LR: 0.0")
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

    # ---- Start Checkpoint Loading ----
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    checkpoint_to_load_path = os.path.join(output_dir, "latest_checkpoint.pth")

    if os.path.isfile(checkpoint_to_load_path):
        logger.info(f"Loading checkpoint: {checkpoint_to_load_path}")
        try:
            checkpoint = torch.load(checkpoint_to_load_path, map_location=device) # Load to current device
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            global_step = checkpoint.get('global_step', start_epoch * num_training_steps_per_epoch) # Restore global_step
            
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            
            if use_mixed_precision and grad_scaler and 'grad_scaler_state_dict' in checkpoint:
                grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
            
            logger.info(f"Successfully loaded checkpoint. Resuming training from epoch {start_epoch}. Best Val Loss: {best_val_loss:.4f}")

        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_to_load_path}: {e}. Starting from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
            # Potentially exit
    else:
        logger.info("No checkpoint found or specified to resume from. Starting training from scratch.")
    # ---- End Checkpoint Loading ----

    # ---- Start Training Loop ----
    logger.info(f"Starting pretraining for {total_epochs} epochs...")
    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad() # Initialize gradients to zero at the start of each epoch / before accumulation window
        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False) # Requires tqdm

        for batch_idx, (samples, _) in enumerate(train_loader): # We don't use labels for MAE
            samples = samples.to(device, non_blocking=True)

            # Mixed precision forward pass
            if use_mixed_precision and grad_scaler:
                with torch.amp.autocast('cuda'):
                    loss, _, _ = model(samples, mask_ratio=model_config.get('mask_ratio', 0.6))
                loss = loss / grad_acc_steps # Scale loss for gradient accumulation
                grad_scaler.scale(loss).backward()
            else: # Standard precision
                loss, _, _ = model(samples, mask_ratio=model_config.get('mask_ratio', 0.6))
                loss = loss / grad_acc_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_acc_steps == 0 or (batch_idx + 1) == len(train_loader):
                if use_mixed_precision and grad_scaler:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad() # Zero gradients after optimizer step

                # Step the LR scheduler (per iteration/optimizer step)
                global_step += 1
                lr_scheduler.step_update(num_updates=global_step) # num_updates is the global step count
            
            running_loss += loss.item() * grad_acc_steps # Unscale loss for logging

            if (batch_idx + 1) % (config.get('logging', {}).get('log_interval', 50) * grad_acc_steps) == 0:
                current_batch_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch: {epoch+1}/{total_epochs} | Batch: {batch_idx+1}/{len(train_loader)} (Step: {global_step}) | Training Loss: {loss.item() * grad_acc_steps:.4f} | LR: {current_batch_lr:.6e}")

        epoch_loss = running_loss / len(train_loader)
        current_epoch_end_lr = optimizer.param_groups[0]['lr']
        logger.info(f"===> Epoch {epoch+1}/{total_epochs} | Average Training Loss: {epoch_loss:.4f} | End of Epoch LR: {current_epoch_end_lr:.6e}")

        # Step the LR scheduler (typically after each epoch)
        lr_scheduler.step(epoch + 1)

        # ---- Validation Step ----
        if (epoch + 1) % log_config.get('val_interval', 1) == 0:
            if val_loader:
                logger.info(f"--- Starting Validation for Epoch {epoch+1} ---")
                validation_loss = validate(
                    val_loader, 
                    model, 
                    device, 
                    model_config,
                    use_mixed_precision,
                    epoch
                )
                logger.info(f"====> Epoch {epoch+1} | Validation Loss: {validation_loss:.4f} ====")

                is_best = validation_loss < best_val_loss
                if is_best:
                    best_val_loss = validation_loss
                    logger.info(f"Best validation loss updated to {best_val_loss:.4f} at epoch {epoch+1}.")
            else:
                is_best = False
                logger.info(f"Epoch {epoch+1}: No validation loader available, skipping validation.")
        

            # ---- Save checkpoint ----
            checkpoint_state = {
                'epoch': epoch + 1, # Next epoch to start from
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config # Save config for reference, optional
            }
            if use_mixed_precision and grad_scaler:
                checkpoint_state['grad_scaler_state_dict'] = grad_scaler.state_dict()

            # Save checkpopint for latest_checkpoint.pth and best_checkpoint.pth
            save_checkpoint(checkpoint_state, is_best, output_dir, filename="latest_checkpoint.pth")
            # ---- Save checkpoint ----
        
        # ---- End Validation Step ----

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