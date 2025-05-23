"""
Betti Matching Loss
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from topolosses.losses.betti_matching import BettiMatchingLoss
from .mse import MaskedMSELoss

class BettiMatchingWithMSELoss(nn.Module):
    """
    Betti Matching Loss with Mean Squared Error (MSE) for pretraining.
    """
    
    def __init__(self,
                 model,
                 norm_pix_loss=False,
                 alpha=0.5, 
                 filtration='superlevel', 
                 push_unmatched_to_1_0=True, 
                 barcode_length_threshold=0.0, # ignore barcodes with length < threshold, set to a small value
                 topology_weights=(1., 1.), # weights for the topology classes in the following order: [matched, unmatched]. Possibly give matched (roads) higher weight
                 sphere=False,):
        super().__init__()
        self.patchify = model.patchify
        self.unpatchify = model.unpatchify
        self.norm_pix_loss = norm_pix_loss
        self.alpha = alpha

        # Initialize losses
        self.BMLoss = BettiMatchingLoss(
            filtration_type=filtration,
            num_processes=16,
            push_unmatched_to_1_0=push_unmatched_to_1_0, 
            barcode_length_threshold=barcode_length_threshold, 
            topology_weights=topology_weights, 
            sphere=sphere,
            include_background=True,
            alpha=1.,
            use_base_loss=False)
        # self.MSELoss = MaskedMSELoss(model, norm_pix_loss=False)
    
    def calculate_mse_loss(self, target, pred, mask):
        """
        target: [N, L, p*p*3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # [N, L], mean loss per patch over P*P*C dimensions

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """

        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)

        # Normalize target patches
        target = self.patchify(imgs)

        target_norm = target
        pred_denorm = pred
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target_norm = (target - mean) / (var + 1.e-6)**.5
            pred_denorm = pred * (var + 1.e-6)**0.5 + mean

        # Compute masked MSE loss
        mse_loss = self.calculate_mse_loss(target_norm, pred, mask)

        # Compute Betti Matching loss
        # Add unmasked patches from the original image here
        pred_img = self.unpatchify(pred_denorm)

        pred_img = TF.rgb_to_grayscale(pred_img, num_output_channels=1)
        target_img = TF.rgb_to_grayscale(imgs, num_output_channels=1)

        bm_loss = self.BMLoss(pred_img, target_img)

        # Combine losses
        loss = mse_loss + self.alpha * bm_loss
        return loss, {
            "bm_loss": bm_loss,
            "mse_loss": mse_loss
        }