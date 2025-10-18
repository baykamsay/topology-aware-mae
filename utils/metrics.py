"""
Metric functions for evaluating model performance.
"""

import torch
from topolosses.losses.betti_matching import BettiMatchingLoss

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

def calculate_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Compute IoU (Intersection over Union) for segmentation.
    
    Args:
        predictions: Predicted masks of shape (B, 1, H, W) or (B, H, W)
        targets: Target masks of shape (B, 1, H, W) or (B, H, W)
        smooth: Smoothing factor to avoid division by zero
        threshold: Threshold for binarizing predictions if not already binary
        
    Returns:
        IoU score
    """
    # Ensure predictions and targets are binary
    if predictions.max() > 1.0 or predictions.min() < 0.0:
        predictions = torch.sigmoid(predictions)
    
    if predictions.max() <= 1.0 and predictions.min() >= 0.0 and threshold < 1.0:
        predictions = (predictions > threshold).float()
    
    # Ensure 4D tensors
    if predictions.ndim == 3:
        predictions = predictions.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    
    # Reshape to (B, -1)
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, -1)
    targets = targets.reshape(batch_size, -1)
    
    # Compute IoU
    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    # Return mean IoU
    return iou.mean()

def calculate_betti_matching_seg(preds, targets):
    """
    Calculate the Betti Matching loss
    """
    superlevel_betti = BettiMatchingLoss(
        filtration_type="superlevel",
        num_processes=16,
        push_unmatched_to_1_0=False,
        sigmoid=True,
        sphere=False,
        use_base_loss=False
    )
    sublevel_betti = BettiMatchingLoss(
        filtration_type="sublevel",
        num_processes=16,
        push_unmatched_to_1_0=False,
        sigmoid=True,
        sphere=False,
        use_base_loss=False
    )

    superlevel_loss = superlevel_betti(preds, targets)
    sublevel_loss = sublevel_betti(preds, targets)

    return {
        "bm_error": superlevel_loss.item() + sublevel_loss.item(),
        "bm_1_error": superlevel_loss.item(),
        "bm_0_error": sublevel_loss.item(),
    }

def calculate_segmentation_metrics(preds, targets):
    """
    Calculate segmentation metrics: Dice score and IoU.
    
    Args:
        preds: Predicted masks (logits or binary) of shape (B, C, H, W)
        targets: Target masks (binary) of shape (B, C, H, W)
        
    Returns:
        dict: Contains 'dice_score' and 'iou'
    """
    # Assuming preds and targets are already in the correct shape
    dice_score = calculate_dice_score(preds, targets)
    iou = calculate_iou(preds, targets)
    betti_matchings = calculate_betti_matching_seg(preds, targets)

    return {
        "dice_score": dice_score,
        "iou": iou,
        **betti_matchings
    }