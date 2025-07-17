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
import wandb
from timm.scheduler.cosine_lr import CosineLRScheduler
import threading

# Adjust the path to include the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.config import load_config, deep_update
from utils.visualizations import log_mae_visualizations
from models.convnextv2 import fcmae
from models.cnn_mae import cnn_mae
from models.vit_mae import vit_mae

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

    # Optional arguments
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Whether to run a hyperparameter sweep",
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
    individual_loss_totals = {}  # Dictionary to accumulate individual losses

    with torch.no_grad():  # Disable gradient calculations
        for batch_idx, (samples, _) in enumerate(val_loader):
            samples = samples.to(device, non_blocking=True)
            
            # Forward pass
            if use_mixed_precision and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    # The model's forward method returns: loss, predicted_patches, mask
                    loss, _, _, individual_losses = model(samples, mask_ratio=config.get('mask_ratio', 0.6), epoch=epoch)
            else:
                loss, _, _, individual_losses = model(samples, mask_ratio=config.get('mask_ratio', 0.6), epoch=epoch)
            
            total_val_loss += loss.item()
            num_val_batches += 1

            # Accumulate individual losses
            if individual_losses:
                for loss_name, loss_value in individual_losses.items():
                    if loss_name not in individual_loss_totals:
                        individual_loss_totals[loss_name] = 0.0
                    individual_loss_totals[loss_name] += loss_value.item()

            if (batch_idx + 1) % 20 == 0: # Log progress every 20 val batches
                 logger.debug(f"Validation Epoch {epoch+1} - Batch {batch_idx+1}/{len(val_loader)}")


    if num_val_batches == 0:
        logger.warning("Validation loader was empty. Returning 0 validation loss.")
        return 0.0, {}
        
    avg_val_loss = total_val_loss / num_val_batches

    # Calculate averages for individual losses
    avg_individual_losses = {}
    if individual_loss_totals:
        avg_individual_losses = {
            loss_name: total_loss / num_val_batches 
            for loss_name, total_loss in individual_loss_totals.items()
        }

    return avg_val_loss, avg_individual_losses

def unfreeze_encoder(model, optimizer, lr, weight_decay, beta1, beta2):
    """
    Unfreeze encoder parameters and add them to the optimizer.
    
    Args:
        model: The segmentation model
        optimizer: Current optimizer (will add new param group)
        lr: Learning rate for the encoder parameters
        weight_decay: Weight decay for the encoder
        beta1, beta2: Beta parameters for AdamW
    
    Returns:
        bool: True if unfreezing was successful
    """
    if not hasattr(model, 'encoder'):
        logger.error("Model does not have an encoder attribute. Cannot unfreeze.")
        return False
    
    # Unfreeze encoder parameters
    encoder_params = []
    for param in model.encoder.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            encoder_params.append(param)
    
    if not encoder_params:
        logger.warning("No frozen encoder parameters found to unfreeze.")
        return False
    
    # Add encoder parameters to optimizer as a new param group
    optimizer.add_param_group({
        'params': encoder_params,
        'lr': lr,
        'weight_decay': weight_decay,
        'betas': (beta1, beta2) if hasattr(optimizer, 'param_groups') and 'betas' in optimizer.param_groups[0] else None
    })
    
    logger.info(f"Successfully unfroze {len(encoder_params)} encoder parameters and added them to optimizer with LR: {lr}")
    return True

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

_save_thread = None

def save_checkpoint(state, is_best, output_dir, filename="checkpoint.pth"):
    """
    Saves a training checkpoint asynchronously in a background thread.

    Args:
        state (dict): Contains model, optimizer, scheduler states, epoch, best_val_loss.
        is_best (bool): True if this checkpoint has the best validation loss so far.
        output_dir (str): Directory where checkpoints will be saved.
        filename (str): Base name for the checkpoint file.
    """
    global _save_thread
    if _save_thread is not None:
        _save_thread.join() # Wait for previous save to finish

    # Move tensors to CPU to avoid race conditions with main training thread
    state_cpu = {
        'epoch': state['epoch'],
        'global_step': state['global_step'],
        'model_state_dict': {k: v.cpu() for k, v in state['model_state_dict'].items()},
        'optimizer_state_dict': state['optimizer_state_dict'], # AdamW state is on CPU by default
        'scheduler_state_dict': state['scheduler_state_dict'],
        'best_val_loss': state['best_val_loss'],
        'encoder_frozen': state['encoder_frozen'],
        'config': state.get('config')
    }
    if 'grad_scaler_state_dict' in state:
        state_cpu['grad_scaler_state_dict'] = state['grad_scaler_state_dict']

    def _save_job():
        filepath = os.path.join(output_dir, filename)
        torch.save(state_cpu, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

        # if is_best:
        #     best_filepath = os.path.join(output_dir, "best_checkpoint.pth")
        #     shutil.copyfile(filepath, best_filepath)
        #     logger.info(f"Best checkpoint updated to {best_filepath} (Val Loss: {state.get('best_val_loss', 'N/A'):.4f})")

    _save_thread = threading.Thread(target=_save_job)
    _save_thread.start()
    logger.info("Started saving checkpoint in background.")


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
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True) # For exact reproducibility, bad for performance
    # monai.utils.set_determinism(seed=config.TRAIN.SEED) # type: ignore

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(seed)
    # ---- End setting up the seed ----

    # ---- W&B Initialization ----
    wandb_logger = None
    if log_config.get('enable_wandb', False):
        try:
            # if running as a part of a sweep
            if args.sweep:
                wandb_logger = wandb.init()  # Initialize W&B without a project name for sweeps

                logger.info(f"W&B logger initialized. Project: {wandb.run.project}, Run Name: {wandb.run.name}")

                # Get sweep config and merge with main config
                sweep_config = dict(wandb.config)  # Get the sweep config
                config = deep_update(config, sweep_config)  # Merge sweep config into main config

                training_config = config.get('training', {})
                log_config = config.get('logging', {})
            else:
                # Initialize W&B with project name and other configurations
                wandb_logger = wandb.init(
                    project=log_config.get('wandb_project', 'topo-conv-mae-pretrain'),
                    name=log_config.get('wandb_run_name', None), # Optional run name
                    config=config, # Log the entire config
                    # dir=output_dir, # Optional: Save wandb files in the run's output directory
                    resume='allow', # Allow resuming previous runs if id is reused (useful with checkpointing)
                    id=log_config.get('wandb_run_id', None) # Optionally set an ID for explicit resuming, can be generated or from checkpoint
                )
                logger.info(f"W&B logger initialized. Project: {wandb.run.project}, Run Name: {wandb.run.name}")
                # Optionally watch the model (can increase overhead)
                # wandb.watch(model, log='gradients', log_freq=1000) 
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}. Disabling W&B logging.")
            wandb_logger = None # Ensure it's None if init fails
    else:
        logger.info("W&B logging is disabled in the configuration.")
    # ---- End W&B Initialization ----

    # ---- Start setting up device ----
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.deterministic = True # Switch for better performance
        torch.backends.cudnn.benchmark = False
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available. Using CPU.")
    # ---- End setting up device ----

    # ---- Start Dataset and DataLoader setup ----
    logger.info("Setting up dataset and DataLoader...")

    # Define Transforms
    input_size = config.get('data', {}).get('input_size', 112)
    augmentation_config = config.get('data', {}).get('augmentation', {})
    # min_scale = config.get('data', {}).get('min_scale', 0.1)
    # max_scale = config.get('data', {}).get('max_scale', 0.5)
    # ImageNet default mean and std
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    # img_mean = [0.33627802, 0.33987136, 0.29782979]
    # img_std = [0.19191039, 0.18239774, 0.18225507]

    if augmentation_config.get('name') == 'RandomResizedCrop':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size,
                                         scale=augmentation_config.get('scale', [0.2, 1.]),
                                         ratio=augmentation_config.get('ratio', [0.75, 1.3333]),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(degrees=[0, 90, 180, 270], interpolation=InterpolationMode.NEAREST, expand=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])
    elif augmentation_config.get('name') == 'RandomCrop':
        # Basic pre-training transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(degrees=[0, 90, 180, 270], interpolation=InterpolationMode.NEAREST, expand=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])
    elif augmentation_config.get('name') == 'CenterCrop':
        # CenterCrop for testing purposes
        train_transform = transforms.Compose([
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])
    else:
        logger.error(f"Unsupported augmentation type: {augmentation_config.get('name')}. Supported types: RandomResizedCrop, RandomCrop.")
        sys.exit(1)

    # whole_size = 420 if input_size == 112 else 840
    crop_size = input_size * 2 if input_size < 63 else input_size
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
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
        persistent_workers=True
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
            persistent_workers=True
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
    model_params['img_size'] = config.get('data', {}).get('input_size', 112)
    model_params['device'] = device

    try:
        if model_name.startswith('cnnmae'):
            model = cnn_mae.__dict__[model_name](**model_params)
        elif model_name.startswith('convnextv2'):
            model = fcmae.__dict__[model_name](**model_params)
        elif model_name.startswith('mae_vit'):
            model = vit_mae.__dict__[model_name](**model_params)
        else:
            logger.error(f"Model {model_name} not supported.")
            sys.exit(1)
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

    # Freeze encoder parameters
    if training_config.get('frozen_encoder_epochs', 0) > 0 and hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info("Froze encoder parameters.")

        encoder_frozen = True
        
        # Parameters to optimize are only from the decoder
        if hasattr(model, 'decoder'):
            parameters_to_optimize = []
            parameters_to_optimize.extend(list(model.decoder.parameters()))
            logger.info("Optimizing only decoder parameters.")
            if hasattr(model, 'proj'):
                parameters_to_optimize.extend(list(model.proj.parameters()))
            if hasattr(model, 'mask_token'):
                parameters_to_optimize.extend(model.mask_token)
        elif hasattr(model, 'decoder_stages'):
            parameters_to_optimize = []
            # If model has decoder_stages, optimize those
            parameters_to_optimize.extend(list(model.decoder_stages.parameters()))
            logger.info("Optimizing decoder stages parameters.")
            # If model has a final conv layer, optimize that too
            if hasattr(model, 'final_conv'):
                parameters_to_optimize.extend(list(model.final_conv.parameters()))
        else:
            logger.warning("Model does not have a 'decoder' attribute. Optimizing all parameters.")
            parameters_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    else:
        logger.info("Encoder parameters are not frozen. Optimizing all model parameters.")
        parameters_to_optimize = model.parameters()
        encoder_frozen = False

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
    warmup_lr_init = training_config.get('warmup_lr_init', 0.0)
    
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_training_steps, # Total number of training steps for one cycle
        lr_min=min_lr,      # Minimum learning rate to decay to
        warmup_t=warmup_steps,          # Number of warmup steps
        warmup_lr_init=warmup_lr_init,             # Start warmup from LR 0.0 (or a very small value like 1e-6 * lr)
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
            encoder_frozen = checkpoint.get('encoder_frozen', True)
            
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
        # Check to unfreeze encoder if needed
        if encoder_frozen and epoch >= training_config.get('frozen_encoder_epochs', 0):
            logger.info(f"Unfreezing encoder parameters at epoch {epoch+1}.")
            
            encoder_lr = optimizer.param_groups[0]['lr'] # Get current decoder LR
            success = unfreeze_encoder(model, optimizer, encoder_lr, weight_decay, beta1, beta2)
            if success:
                encoder_frozen = False
                # Update scheduler if needed
        
        model.train()
        running_loss = 0.0
        optimizer.zero_grad() # Initialize gradients to zero at the start of each epoch / before accumulation window
        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False) # Requires tqdm

        for batch_idx, (samples, _) in enumerate(train_loader): # We don't use labels for MAE
            samples = samples.to(device, non_blocking=True)

            # Mixed precision forward pass
            if use_mixed_precision and grad_scaler:
                with torch.amp.autocast('cuda'):
                    loss, _, _, individual_losses = model(samples, mask_ratio=model_config.get('mask_ratio', 0.6), epoch=epoch)
                loss = loss / grad_acc_steps # Scale loss for gradient accumulation
                grad_scaler.scale(loss).backward()
            else: # Standard precision
                loss, _, _, individual_losses = model(samples, mask_ratio=model_config.get('mask_ratio', 0.6), epoch=epoch)
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
                log_msg = f"Epoch: {epoch+1}/{total_epochs} | Batch: {batch_idx+1}/{len(train_loader)} (Step: {global_step}) | Training Loss: {loss.item() * grad_acc_steps:.4f}"
                if individual_losses:
                    individual_loss_str = " | ".join([f"{loss_name}: {loss_value.item():.4f}" for loss_name, loss_value in individual_losses.items()])
                    log_msg += f" | {individual_loss_str}"
                log_msg += f" | LR: {current_batch_lr:.6e}"
                logger.info(log_msg)

                # ---- Start W&B Logging ----
                if wandb_logger:
                    wandb_dict = {
                        "train/lr": current_batch_lr,
                        "train/loss": loss.item() * grad_acc_steps,
                    }

                    if individual_losses:
                        for loss_name, loss_value in individual_losses.items():
                            wandb_dict[f"train/{loss_name}"] = loss_value.item()
                    wandb.log(wandb_dict, step=global_step)
                # ---- End W&B Logging ----

        epoch_loss = running_loss / len(train_loader)
        current_epoch_end_lr = optimizer.param_groups[0]['lr']
        logger.info(f"===> Epoch {epoch+1}/{total_epochs} | Average Training Loss: {epoch_loss:.4f} | End of Epoch LR: {current_epoch_end_lr:.6e}")

        # Step the LR scheduler (typically after each epoch)
        lr_scheduler.step(epoch + 1)

        if wandb_logger:
            log_data = {
                "val/epoch": epoch + 1, # Log current epoch number (1-indexed)
            }

        # ---- Validation Step ----
        if (epoch + 1) % log_config.get('val_interval', 1) == 0:
            if val_loader:
                logger.info(f"--- Starting Validation for Epoch {epoch+1} ---")
                validation_loss, validation_individual_loss = validate(
                    val_loader, 
                    model, 
                    device, 
                    model_config,
                    use_mixed_precision,
                    epoch
                )
                log_msg = f"====> Epoch {epoch+1} | Validation Loss: {validation_loss:.4f}"
                if validation_individual_loss:
                    individual_loss_str = " | ".join([f"{loss_name}: {loss_value:.4f}" for loss_name, loss_value in validation_individual_loss.items()])
                    log_msg += f" | {individual_loss_str}"
                log_msg += f" ===="
                logger.info(log_msg)

                is_best = validation_loss < best_val_loss
                if is_best:
                    best_val_loss = validation_loss
                    logger.info(f"Best validation loss updated to {best_val_loss:.4f} at epoch {epoch+1}.")
                if wandb_logger:
                    log_data["val/loss"] = validation_loss
                    if validation_individual_loss:
                        for loss_name, loss_value in validation_individual_loss.items():
                            log_data[f"val/{loss_name}"] = loss_value
            else:
                is_best = False
                logger.info(f"Epoch {epoch+1}: No validation loader available, skipping validation.")
        

            # ---- Save checkpoint ----
            if log_config.get('checkpointing', True):
                checkpoint_state = {
                    'epoch': epoch + 1, # Next epoch to start from
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'encoder_frozen': encoder_frozen, # Save encoder frozen state
                    'config': config # Save config for reference, optional
                }
                if use_mixed_precision and grad_scaler:
                    checkpoint_state['grad_scaler_state_dict'] = grad_scaler.state_dict()

                # Save checkpopint for latest_checkpoint.pth and best_checkpoint.pth
                save_checkpoint(checkpoint_state, is_best, output_dir, filename="latest_checkpoint.pth")
            # ---- Save checkpoint ----
        
        # ---- End Validation Step ----

        # ---- W&B Metric Logging ----
        if wandb_logger:
            wandb.log(log_data, step=global_step) # Log metrics against global_step
            # Alternatively log against epoch: wandb.log(log_data, step=epoch + 1) 
            # Logging against global_step is often preferred for step-based LR schedules

        # ---- End W&B Metric Logging ----

        # ---- W&B Visualization Logging ----
        vis_interval = log_config.get('vis_interval', 20)
        if wandb_logger and log_config.get('include_vis', False) and val_loader: # Use val_loader for visualization samples
           if (epoch + 1) % vis_interval == 0:
               logger.info(f"--- Generating Visualizations for Epoch {epoch+1} ---")
               log_mae_visualizations(
                   model, 
                   val_loader, # Using validation loader for consistent visualization samples
                   device, 
                   config, 
                   epoch,
                   global_step, 
                   wandb_logger,
                   num_images=8 # Or get from config
               )
        # ---- End W&B Visualization ----

    # ---- End of Training Loop ----
    logger.info("Pretraining completed successfully.")
    
    # Ensure the final checkpoint is saved before exiting
    global _save_thread
    if _save_thread is not None:
        _save_thread.join()
        logger.info("Final checkpoint saving operation finished.")

    if wandb_logger:
        wandb_logger.finish() # Finish the W&B run

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Topo MAE Pre-training', parents=[get_args_parser()])
    args, _ = parser.parse_known_args()

    main(args)