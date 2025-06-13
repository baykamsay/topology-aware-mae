"""
Base segmentation script.
"""

import argparse
import os
import sys
import shutil
import logging
import pprint
import torch
import random
import numpy as np
import wandb
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler

# Adjust the path to include the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from datasets.segmentation_dataset import RoadsSegmentationDataset
from utils.config import load_config
from utils.visualizations import log_segmentation_visualizations
from models.cnn_mae import cnn_seg
from losses.segmentation import get_segmentation_loss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_args_parser():
    parser = argparse.ArgumentParser('Topo MAE Segmentation', add_help=False)

    # Configuration file path
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        default=os.path.join(project_root, "configs", "pretrain", "base_pretrain.yaml"),
        help="Path to the configuration YAML file",
    )

    return parser

class JointTransforms:
    def __init__(self, h_flip_prob=0.5, v_flip_prob=0.5, rotation_degrees=0):
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.rotation_degrees = rotation_degrees

    def __call__(self, image, mask): # image and mask are PIL.Image objects
        if random.random() < self.h_flip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if random.random() < self.v_flip_prob:
            image = F.vflip(image)
            mask = F.vflip(mask)

        if self.rotation_degrees > 0:
            # angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            angle = random.choice([0, 90, 180, 270]) # For simplicity, use fixed angles
            image = F.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = F.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST, fill=0) # fill mask background with 0

        return image, mask

# For test purposes, define metrics separately
def calculate_dice_score(preds, targets, smooth=1e-6):
    """Calculates Dice score for a batch."""
    # preds are logits from model, targets are binary masks
    preds_probs = torch.sigmoid(preds)
    preds_binary = (preds_probs > 0.5).float()

    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    dice = (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    return dice.item()

def validate(val_loader, model, criterion, device, config, use_mixed_precision, epoch):
    """
    Performs validation on the validation dataset.
    """
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    total_dice_score = 0.0
    num_val_batches = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward pass
            if use_mixed_precision and device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else: # No mixed precision
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            total_val_loss += loss.item()
            dice_score_batch = calculate_dice_score(outputs, masks) # Using the helper
            total_dice_score += dice_score_batch
            num_val_batches += 1

            if (batch_idx + 1) % config.get('logging', {}).get('log_interval_val', 50) == 0: # Optional: batch-level val logging
                 logger.debug(f"Validation Epoch {epoch+1} - Batch {batch_idx+1}/{len(val_loader)} | Batch Loss: {loss.item():.4f} | Batch Dice: {dice_score_batch:.4f}")

    if num_val_batches == 0:
        logger.warning("Validation loader was empty. Returning 0 validation loss and 0 Dice score.")
        return 0.0, {"dice_score": 0.0}
        
    avg_val_loss = total_val_loss / num_val_batches
    avg_dice_score = total_dice_score / num_val_batches

    return avg_val_loss, {"dice_score": avg_dice_score}

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
        logger.info(f"Best checkpoint updated to {best_filepath} (Val Metric: {state.get('best_val_metric', 'N/A'):.4f})")

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
    data_config = config.get('data', {})

    # Set up directories for logging and checkpoints
    output_dir = log_config.get('output_dir', './output/segmentation/test')
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

    # ---- W&B Initialization ----
    wandb_logger = None
    if log_config.get('enable_wandb', False):
        try:
            wandb_logger = wandb.init(
                project=log_config.get('wandb_project', 'topo-conv-mae-segmentation'),
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
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available. Using CPU.")
    # ---- End setting up device ----

    # ---- Start Dataset and DataLoader setup ----
    img_mean = [0.33627802, 0.33987136, 0.29782979]
    img_std = [0.19191039, 0.18239774, 0.18225507]
    input_size = data_config.get('input_size', 56) 

    joint_transform = JointTransforms(
        h_flip_prob=0.5,
        v_flip_prob=0.5,
        rotation_degrees=180
    )

    train_image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
        # Additional augmentations here
    ])

    # Possibly add mask transformations if needed

    val_image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    base_data_path = data_config.get('data_path') # ./data/finetune/roads_mini
    num_workers = data_config.get('num_workers', 4)
    batch_size = config.get('training', {}).get('batch_size', 32) # from test.yaml

    train_dataset_path = os.path.join(base_data_path, 'train')
    val_dataset_path = os.path.join(base_data_path, 'val')
    
    
    train_dataset = RoadsSegmentationDataset(
        root_dir=train_dataset_path,
        joint_transform=joint_transform,
        image_transform=train_image_transform 
        # mask_transform=train_mask_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_dataset = RoadsSegmentationDataset(
        root_dir=val_dataset_path,
        image_transform=val_image_transform
        # mask_transform=val_mask_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    
    logger.info(f"Training DataLoader created with {len(train_dataset)} samples.")
    if val_loader:
        logger.info(f"Validation DataLoader created with {len(val_dataset)} samples.")
    # ---- End Dataset and DataLoader setup ----

    # ---- Start Model setup ----
    logger.info("Setting up model...")
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'cnnseg_small')
    model_params = {k: v for k, v in model_config.items() if k != "name"}

    # Add other models here
    try:
        model = cnn_seg.__dict__[model_name](**model_params)
        model.to(device)
        logger.info(f"Model {model_name} loaded successfully.")
        print(f"Model architecture: {model}")
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        sys.exit(1)
    # ---- End Model setup ----

    # ---- Start Loss function setup ----
    loss_config = config.get('loss', {})
    criterion = get_segmentation_loss(loss_config)
    if criterion is None:
        logger.error("Could not create loss function. Exiting.")
        sys.exit(1)
    logger.info(f"Loss function {loss_config.get('name')} initialized.")
    # ---- End Loss function setup ----

    # ---- Start Optimizer and Scheduler setup ----
    logger.info("Setting up optimizer and scheduler...")
    optimizer_name = training_config.get('optimizer', 'adamw').lower()
    lr = training_config.get('learning_rate', 1e-3)
    min_lr = training_config.get('min_lr', 0.0)
    weight_decay = training_config.get('weight_decay', 0.05)
    beta1 = training_config.get('beta1', 0.9)
    beta2 = training_config.get('beta2', 0.95)

    # Might apply weight decay differently or use different LRs
    # Freeze encoder parameters
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info("Froze encoder parameters.")
        
        # Parameters to optimize are only from the decoder
        if hasattr(model, 'decoder'):
            parameters_to_optimize = model.decoder.parameters()
            logger.info("Optimizing only decoder parameters.")
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
        logger.warning("Model does not have an 'encoder' attribute. Optimizing all parameters.")
        parameters_to_optimize = model.parameters()

    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            parameters_to_optimize,
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
        )
    elif optimizer_name == 'sgd': # Example if you want to add SGD
        optimizer = optim.SGD(
            parameters_to_optimize,
            lr=lr,
            momentum=training_config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        logger.error(f"Optimizer {optimizer_name} not supported.")
        sys.exit(1)

    logger.info(f"Optimizer: {optimizer_name} initialized with base LR: {lr}, Weight Decay: {weight_decay}")

    # LR Scheduler CosineAnnealingLR
    logger.info("Setting up learning rate scheduler with warmup...")

    grad_acc_steps = config.get('training', {}).get('gradient_accumulation_steps', 1)
    if grad_acc_steps < 1:
        grad_acc_steps = 1
        logger.warning("gradient_accumulation_steps was less than 1, set to 1.")

    num_training_steps_per_epoch = len(train_loader) // grad_acc_steps 
    total_epochs = training_config.get('epochs', 100)
    total_training_steps = num_training_steps_per_epoch * total_epochs
    warmup_epochs = training_config.get('warmup_epochs', 5)
    warmup_steps = min(num_training_steps_per_epoch * warmup_epochs, total_training_steps)
    min_lr = training_config.get('min_lr', 0.0)
    cycle_limit = training_config.get('cycle_limit', 1)
    cycle_mult = training_config.get('cycle_mult', 1.0)
    cycle_decay = training_config.get('cycle_decay', 0.5)

    
    if cycle_limit > 1 and cycle_mult == 1.0:
        # All cycles have the same length
        first_cycle_epochs = total_epochs / cycle_limit
    else:
        # If cycles change length, or only one cycle, t_initial usually spans all steps unless modified by cycle_mult
        # For simplicity with timm's scheduler, t_initial is often the length of the first cycle.
        # If cycle_mult > 1, subsequent cycles are t_initial * cycle_mult, t_initial * cycle_mult^2, etc.
        # Let's define t_initial as the steps for the first cycle.
        # If we want N cycles over total_epochs:
        # If cycle_mult = 1, each cycle is total_epochs / N epochs.
        # If cycle_mult != 1, the calculation is more complex to fit total_epochs.
        # A common approach: set t_initial to (total_steps / cycle_limit) if cycle_mult is 1,
        # or make t_initial the length of the first desired cycle.
        first_cycle_epochs = total_epochs / cycle_limit if cycle_mult == 1.0 else total_epochs / sum([cycle_mult**i for i in range(cycle_limit)]) if cycle_mult !=1 else total_epochs

    t_initial_steps = int(num_training_steps_per_epoch * first_cycle_epochs)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=t_initial_steps,             # Number of steps in the first cycle
        lr_min=min_lr,
        warmup_t=warmup_steps,
        warmup_lr_init=training_config.get('warmup_lr_init', 0.0), # Often 0.0 or a very small fraction of base LR
        warmup_prefix=True,

        # Parameters for warm restarts
        cycle_limit=cycle_limit,
        cycle_mul=cycle_mult,       # Factor to multiply t_initial by after each cycle
        cycle_decay=cycle_decay,     # Factor to multiply LR by after each cycle

        t_in_epochs=False, # Make sure t_initial and warmup_t are in steps
    )
    logger.info(f"LR Scheduler: timm.CosineLRScheduler initialized with warm restarts.")
    logger.info(f"  First cycle steps (t_initial): {t_initial_steps} (~{first_cycle_epochs:.2f} epochs)")
    logger.info(f"  Warmup steps: {warmup_steps} (~{warmup_epochs} epochs)")
    logger.info(f"  Cycle limit: {cycle_limit}, Cycle mult: {cycle_mult}, Cycle LR decay: {cycle_decay}")
    # ---- End Optimizer and Scheduler setup ----

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
    best_val_metric = float('-inf') # If using a metric

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
    logger.info(f"Starting finetuning for {total_epochs} epochs...")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad() # Reset gradients at the start of each epoch
        
        # Optional: progress bar (e.g., from tqdm)
        # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} Training", leave=False)

        for batch_idx, (images, masks) in enumerate(train_loader): # Or progress_bar
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Mixed precision forward pass
            if use_mixed_precision and grad_scaler: # grad_scaler and use_mixed_precision should be defined
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                loss = loss / grad_acc_steps
                grad_scaler.scale(loss).backward()
            else: # Standard precision
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / grad_acc_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_acc_steps == 0 or (batch_idx + 1) == len(train_loader):
                if use_mixed_precision and grad_scaler:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()

                global_step += 1
                lr_scheduler.step_update(num_updates=global_step) # num_updates is the global step count
            
            running_loss += loss.item() * grad_acc_steps # Unscale loss for logging

            # Logging training progress (batch-level)
            if (batch_idx + 1) % (config.get('logging', {}).get('log_interval', 50) * grad_acc_steps) == 0:
                current_batch_lr = optimizer.param_groups[0]['lr']
                log_msg = f"Epoch: {epoch+1}/{total_epochs} | Batch: {batch_idx+1}/{len(train_loader)} (Step: {global_step}) | " \
                        f"Train Loss: {loss.item() * grad_acc_steps:.4f} | LR: {current_batch_lr:.6e}"
                logger.info(log_msg)
                if wandb_logger:
                    wandb.log({
                        "train/loss": loss.item() * grad_acc_steps,
                        "train/lr": current_batch_lr
                    }, step=global_step)

        epoch_loss = running_loss / len(train_loader)
        current_epoch_end_lr = optimizer.param_groups[0]['lr']
        logger.info(f"===> Epoch {epoch+1}/{total_epochs} | Average Training Loss: {epoch_loss:.4f} | End of Epoch LR: {current_epoch_end_lr:.6e}")
        if wandb_logger:
            wandb.log({"train/epoch_loss": epoch_loss}, step=global_step) # TODO maybe keep

        lr_scheduler.step(epoch + 1)

        # ---- Validation Step ----
        if (epoch + 1) % log_config.get('val_interval', 1) == 0:
            if val_loader:
                logger.info(f"--- Starting Validation for Epoch {epoch+1} ---")
                validation_loss, metrics = validate(
                    val_loader, 
                    model, 
                    criterion,
                    device, 
                    model_config,
                    use_mixed_precision,
                    epoch
                )
                log_msg = f"====> Epoch {epoch+1} | Validation Loss: {validation_loss:.4f}"
                if metrics:
                    individual_loss_str = " | ".join([f"{loss_name}: {loss_value:.4f}" for loss_name, loss_value in metrics.items()])
                    log_msg += f" | {individual_loss_str}"
                log_msg += f" ===="
                logger.info(log_msg)

                is_best = metrics["dice_score"] > best_val_metric
                if is_best:
                    best_val_metric = metrics["dice_score"]
                    logger.info(f"Best validation metric updated to {best_val_metric:.4f} at epoch {epoch+1}.")
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
                    'best_val_metric': best_val_metric,
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
            log_data = {
                "val/epoch": epoch + 1, # Log current epoch number (1-indexed)
            }
            if validation_loss is not None: # Only log validation loss if validation was run
                log_data["val/loss"] = validation_loss

            if metrics:
                for loss_name, loss_value in metrics.items():
                    log_data[f"val/{loss_name}"] = loss_value
            
            wandb.log(log_data, step=global_step) # Log metrics against global_step
            # Alternatively log against epoch: wandb.log(log_data, step=epoch + 1) 
            # Logging against global_step is often preferred for step-based LR schedules

        # ---- End W&B Metric Logging ----

        # ---- W&B Visualization Logging ----
        vis_interval = log_config.get('vis_interval', 20)
        if wandb_logger and log_config.get('include_vis', False) and val_loader: # Use val_loader for visualization samples
           if (epoch + 1) % vis_interval == 0:
               logger.info(f"--- Generating Visualizations for Epoch {epoch+1} ---")
               log_segmentation_visualizations(
                   model, 
                   val_loader, 
                   device, 
                   config,
                   use_mixed_precision,
                   epoch,
                   global_step,
                   wandb_logger,
                   num_images=8
               )

        # ---- End W&B Visualization ----

    # ---- End Training Loop ----
    logger.info("Finetuning completed successfully.")
    if wandb_logger:
        wandb_logger.finish() # Finish the W&B run

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Topo MAE Segmentation', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)