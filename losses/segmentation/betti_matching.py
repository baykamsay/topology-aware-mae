"""
Betti Matching Loss for Segmentation
"""

import os
import torch
import torch.nn as nn
from topolosses.losses.betti_matching import BettiMatchingLoss
from .dice import DiceLoss

class BettiMatchingWithDiceLoss(nn.Module):
    """
    Betti Matching Loss with Dice Loss for segmentation.
    """
    
    def __init__(self,
                 alpha=0.5,  # Weight for Betti Matching Loss
                 alpha_warmup_epochs=0,  # Number of epochs to warm up alpha
                 filtration='superlevel', 
                 push_unmatched_to_1_0=True, 
                 barcode_length_threshold=0.0,  # ignore barcodes with length < threshold
                 topology_weights=(1., 1.),  # weights for the topology classes: [matched, unmatched]
                 sphere=False):
        super().__init__()
        self.alpha = alpha
        self.alpha_warmup_epochs = alpha_warmup_epochs

        try:
            num_processes = int(os.getenv('SLURM_CPUS_PER_TASK', '16'))
            if num_processes <= 0: # Ensure positive
                num_processes = 16
        except ValueError:
            num_processes = 16 # Fallback if conversion fails

        self.BMLoss = BettiMatchingLoss(
            filtration_type=filtration,
            num_processes=num_processes,
            push_unmatched_to_1_0=push_unmatched_to_1_0, 
            barcode_length_threshold=barcode_length_threshold, 
            topology_weights=topology_weights, 
            sphere=sphere,
            include_background=True,
            alpha=1.,
            sigmoid=True,
            use_base_loss=False)
        self.diceLoss = DiceLoss(smooth=1e-6)
        

    def forward(self, inputs, targets, epoch=-1):
        """
        Args:
            inputs (torch.Tensor): Raw logits from the model (N, C, H, W).
            targets (torch.Tensor): Ground truth binary masks (N, C, H, W), values 0 or 1.
        """
        bm_loss = self.BMLoss(inputs, targets)
        dice_loss, _ = self.diceLoss(inputs, targets)

        alpha = self.alpha

        if epoch >= 0 and epoch < self.alpha_warmup_epochs:
            alpha = 0

        loss = dice_loss + alpha * bm_loss
        return loss, {
            "bm_loss": bm_loss,
            "dice_loss": dice_loss
        }



        
