"""
Visualization utilities for generating images.
"""

import torch
import torchvision
import wandb
import logging

logger = logging.getLogger(__name__)

# Function to handle denormalization and image preparation for W&B
def denormalize(tensor, mean=[0.33627802, 0.33987136, 0.29782979], std=[0.19191039, 0.18239774, 0.18225507]):
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
        loss, pred_patches, binary_mask = model(samples, mask_ratio=config.get('model', {}).get('mask_ratio', 0.6))

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
            if hasattr(model, 'norm_pix_loss') and model.norm_pix_loss:
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
    wandb_logger.log({"Reconstructions": images_to_log}, step=global_step) # Log against global step
    logger.info(f"Logged {len(images_to_log)} reconstruction visualizations to W&B for epoch {epoch+1}.")

    model.train() # Set model back to training mode