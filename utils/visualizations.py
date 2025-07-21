"""
Visualization utilities for generating images.
"""

import torch
import torchvision
import wandb
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Function to handle denormalization and image preparation for W&B
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # Clone to avoid modifying the original tensor
    tensor = tensor.clone()
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    tensor.mul_(std).add_(mean)
    # Clamp to valid image range [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

# Function to create and log visualizations
def log_mae_visualizations(model, loader, device, config, epoch, global_step, wandb_logger, num_images=8,):
    """
    Generates and logs MAE reconstruction visualizations to W&B.
    """
    model.eval() # Ensure model is in eval mode
    
    # Get a fixed batch or a random batch from the loader
    try:
        samples, _ = next(iter(loader))
    except StopIteration:
        logger.warning("Visualization loader is empty, cannot generate visualizations.")
        return
        
    samples = samples.to(device, non_blocking=True)
    
    # Limit number of images to visualize
    if samples.shape[0] > num_images:
        samples = samples[:num_images]
        
    images_to_log = []
    # captions = []

    with torch.no_grad():
        # Get model predictions: loss, pred_patches, binary_mask (0=keep, 1=remove)
        loss, pred_patches, binary_mask, individual_losses = model(samples, mask_ratio=config.get('model', {}).get('mask_ratio', 0.6), epoch=-1)

        # Reshape binary_mask from (N, L) to (N, 1, H_patch, W_patch)
        num_patches_side = int(binary_mask.shape[1]**0.5)
        binary_mask_img = binary_mask.reshape(-1, 1, num_patches_side, num_patches_side)
        
        # Upsample mask to full image resolution
        # Scale factor is patch_size
        patch_size = config.get('model', {}).get('patch_size', 16)
        img_mask_full = torch.nn.functional.interpolate(
            binary_mask_img.float(), 
            scale_factor=patch_size, 
            mode='nearest' # Use nearest neighbor to keep mask binary per patch region
        ) # Shape: (N, 1, H, W)
        
        # Unpatchify the prediction
        try:
            if len(pred_patches.shape) == 4:
                n, c, _, _ = pred_patches.shape
                pred_patches = pred_patches.reshape(n, c, -1)
                pred_patches = torch.einsum('ncl->nlc', pred_patches)
            
            # Fix norm_pix_loss visualization
            if config.get('loss', {}).get('norm_pix_loss', False):
                # If norm_pix_loss is True, we need to normalize pred_patches from target patches
                samples_patches = model.patchify(samples)
                mean = samples_patches.mean(dim=-1, keepdim=True)
                var = samples_patches.var(dim=-1, keepdim=True)
                pred_patches = pred_patches * (var + 1.e-6)**0.5 + mean

            reconstructed_imgs = model.unpatchify(pred_patches)
        except AttributeError:
            # If unpatchify needs to be called differently or implemented as utility
            logger.error("Model does not have unpatchify method. Cannot create reconstructed images.")

        # Denormalize original and reconstructed images
        original_imgs_denorm = denormalize(samples)
        reconstructed_imgs_denorm = denormalize(reconstructed_imgs)

        # Create masked images (overlay mask on original)
        # Where img_mask_full is 1 (masked), set original image pixel to gray/black/etc.
        masked_imgs_vis = original_imgs_denorm * (1 - img_mask_full) + (img_mask_full * 0.5) # Gray out masked patches

    # Prepare for logging (convert individual images in batch to wandb.Image)
    for i in range(samples.shape[0]):
        # Combine Original, Masked, Reconstructed side-by-side using torchvision.utils.make_grid
        grid = torchvision.utils.make_grid(
            [original_imgs_denorm[i], masked_imgs_vis[i], reconstructed_imgs_denorm[i]],
            nrow=3, padding=2, normalize=False # Already denormalized
        )
        images_to_log.append(wandb.Image(grid, caption=f"Epoch {epoch+1} - Img {i+1} (Orig | Masked | Recon)"))
        # Alternatively, log separately:
        # images_to_log.append(wandb.Image(original_imgs_denorm[i], caption=f"Epoch {epoch+1} - Original {i+1}"))
        # images_to_log.append(wandb.Image(masked_imgs_vis[i], caption=f"Epoch {epoch+1} - Masked {i+1}"))
        # images_to_log.append(wandb.Image(reconstructed_imgs_denorm[i], caption=f"Epoch {epoch+1} - Reconstructed {i+1}"))
        
    # Log the list of images
    # wandb_logger.log({"Reconstructions": images_to_log}, step=global_step) # Log against global step
    # logger.info(f"Logged {len(images_to_log)} reconstruction visualizations to W&B for epoch {epoch+1}.")

    model.train() # Set model back to training mode
    return images_to_log


def log_segmentation_visualizations(
    model,
    loader,
    # criterion,
    device,
    config,
    use_mixed_precision,
    epoch,
    global_step,
    wandb_logger,
    num_images=8
):
    """
    Visualize segmentation results in Weights & Biases with separate visualizations
    for original images, ground truth masks, predictions, and overlays.
    """
    model.eval()

    # Get a fixed batch or a random batch from the loader
    try:
        images, masks = next(iter(loader))
    except StopIteration:
        logger.warning("Visualization loader is empty, cannot generate visualizations.")
        return
    
    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)

    # Limit number of images to visualize
    if images.shape[0] > num_images:
        images = images[:num_images]
        masks = masks[:num_images]
    
    with torch.no_grad():
        if use_mixed_precision and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                predictions = model(images)
                # loss = criterion(predictions, masks)
        else:
            predictions = model(images)
            # loss = criterion(predictions, masks)
    
        # Ensure masks and predictions are 4D tensors
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        if predictions.ndim == 3:
            predictions = predictions.unsqueeze(1)
        
        images = denormalize(images)
    
    # Move tensors to CPU for numpy conversion
    images = images.cpu()
    masks = masks.cpu()
    predictions = predictions.cpu()
    
    # Create sigmoid version for visualization (raw model outputs)
    raw_predictions = None
    if hasattr(predictions, 'dtype') and predictions.dtype == torch.bool:
        # This means we got thresholded predictions, we don't have the raw scores
        print("Received boolean predictions, no raw scores available")
    elif predictions.max() <= 1.0 and predictions.min() >= 0.0:
        # Store copy of the original raw predictions before converting to binary
        raw_predictions = predictions.clone()
        # Now convert predictions to binary for visualization
        predictions = (predictions > 0.5).type(torch.float32)
    
    # Convert tensors to numpy arrays
    images_np = images.numpy()
    masks_np = masks.numpy()
    predictions_np = predictions.numpy()
    raw_np = raw_predictions.numpy() if raw_predictions is not None else None
    
    # Lists to store different visualization types
    original_images = []
    mask_images = []
    pred_images = []
    overlay_images = []
    raw_pred_images = []
    
    for i in range(num_images):
        # Get components
        img = images_np[i].transpose(1, 2, 0)  # (H, W, 3)
        mask = masks_np[i, 0]  # (H, W)
        pred = predictions_np[i, 0]  # (H, W)
        
        # Convert to uint8 for wandb
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Create colored mask images (white = road, black = background)
        mask_colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        # For boolean arrays, use direct indexing
        if mask.dtype == np.bool_:
            mask_colored[mask] = [255, 255, 255]
        else:
            mask_colored[mask > 0.5] = [255, 255, 255]  # White for roads in ground truth
        
        pred_colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        # For boolean arrays, use direct indexing
        if pred.dtype == np.bool_:
            pred_colored[pred] = [255, 255, 255]
        else:
            pred_colored[pred > 0.5] = [255, 255, 255]  # White for roads in prediction
        
        # Create overlay image
        overlay = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        overlay = (img * 255).astype(np.uint8).copy()  # Start with the original image
        
        # Color coding:
        # True positive (green): mask=1, pred=1
        # False positive (red): mask=0, pred=1
        # False negative (blue): mask=1, pred=0
        
        # Create a separate overlay image for clearer visualization
        overlay_colored = np.zeros_like(overlay)
        
        # True positive (green)
        tp_mask = np.logical_and(mask > 0.5, pred > 0.5)
        overlay_colored[tp_mask] = [0, 255, 0]  # Green
        
        # False positive (red)
        fp_mask = np.logical_and(mask <= 0.5, pred > 0.5)
        overlay_colored[fp_mask] = [255, 0, 0]  # Red
        
        # False negative (blue)
        fn_mask = np.logical_and(mask > 0.5, pred <= 0.5)
        overlay_colored[fn_mask] = [0, 0, 255]  # Blue
        
        # Add a semi-transparent version to the original image for context
        alpha = 0.5
        for c in range(3):
            overlay[..., c] = np.where(
                tp_mask | fp_mask | fn_mask,
                (1-alpha) * overlay[..., c] + alpha * overlay_colored[..., c],
                overlay[..., c]
            )
        
        # If we have raw predictions, visualize them as a heatmap
        if raw_np is not None:
            raw_pred = raw_np[i, 0]  # (H, W)
            
            # Create a heatmap (blue to red: 0.0 to 1.0)
            # Use a color map: dark blue (0.0) to light blue to green to yellow to red (1.0)
            raw_colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            
            # Rescale to 0-255
            raw_scaled = (raw_pred * 255).astype(np.uint8)
            
            # Simple blue channel for values closer to 0, red channel for values closer to 1
            for y in range(raw_colored.shape[0]):
                for x in range(raw_colored.shape[1]):
                    val = raw_pred[y, x]
                    if val < 0.2:
                        # Dark blue for very low values
                        raw_colored[y, x] = [0, 0, int(val * 255 * 5)]
                    elif val < 0.4:
                        # Light blue for low-mid values
                        raw_colored[y, x] = [0, int((val - 0.2) * 255 * 5), 255]
                    elif val < 0.6:
                        # Green for mid values
                        raw_colored[y, x] = [0, 255, int((0.6 - val) * 255 * 5)]
                    elif val < 0.8:
                        # Yellow for mid-high values
                        raw_colored[y, x] = [int((val - 0.6) * 255 * 5), 255, 0]
                    else:
                        # Red for high values
                        raw_colored[y, x] = [255, int((1.0 - val) * 255 * 5), 0]
            
            raw_pred_images.append(wandb.Image(raw_colored, caption=f"Raw Prediction {i} (color=value)"))
        
        # Add images to respective lists
        original_images.append(wandb.Image(img_uint8, caption=f"Original {i}"))
        mask_images.append(wandb.Image(mask_colored, caption=f"Ground Truth {i}"))
        pred_images.append(wandb.Image(pred_colored, caption=f"Prediction {i}"))
        overlay_images.append(wandb.Image(overlay, caption=f"Overlay {i}"))
    
    # Log to wandb
    if wandb.run is not None:
        log_dict = {
            f"Original": original_images,
            f"Ground_truth": mask_images,
            f"Prediction": pred_images,
            f"Overlay": overlay_images
        }
        
        if raw_pred_images:
            log_dict[f"Raw_prediction"] = raw_pred_images
        
        wandb.log(log_dict, step=global_step)