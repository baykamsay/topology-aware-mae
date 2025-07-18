"""
Betti Matching Loss
"""

import os
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
                 alpha_warmup_epochs=0,
                 alpha_mse_treshold=0.0,
                 filtration='superlevel', 
                 push_unmatched_to_1_0=True, 
                 barcode_length_threshold=0.0, # ignore barcodes with length < threshold, set to a small value
                 topology_weights=(1., 1.), # weights for the topology classes in the following order: [matched, unmatched]. Possibly give matched (roads) higher weight
                 sphere=False,
                 calculate_channels_separately=False,
                 num_processes=16, # Number of processes for Betti Matching
                 **kwargs):
        super().__init__()
        self.patchify = model.patchify
        self.unpatchify = model.unpatchify
        self.norm_pix_loss = norm_pix_loss
        self.alpha = alpha
        self.alpha_warmup_epochs = alpha_warmup_epochs
        self.alpha_mse_treshold = alpha_mse_treshold
        self.calculate_channels_separately = calculate_channels_separately

        # Initialize losses
        self.BMLoss = BettiMatchingLoss(
            filtration_type=filtration,
            num_processes=num_processes,
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

    def forward(self, imgs, pred, mask, epoch):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        epoch: current training epoch
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
        # Combine masked patches from prediction with unmasked patches from target
        mask_expanded = mask.unsqueeze(-1)
        pred_patches = mask_expanded * pred_denorm + (1 - mask_expanded) * target
        pred_img = self.unpatchify(pred_patches)

        if self.calculate_channels_separately:
            # Calculate Betti Matching loss for each channel separately
            bm_loss = 0.0
            for c in range(pred_img.shape[1]):
                pred_channel = pred_img[:, c, :, :].unsqueeze(1)
                target_channel = imgs[:, c, :, :].unsqueeze(1)
                bm_loss += self.BMLoss(pred_channel, target_channel)
            bm_loss /= pred_img.shape[1]  # Average over channels
        else:
            pred_img = TF.rgb_to_grayscale(pred_img, num_output_channels=1)
            target_img = TF.rgb_to_grayscale(imgs, num_output_channels=1)

            bm_loss = self.BMLoss(pred_img, target_img)

        # Combine losses
        alpha = self.alpha
        if epoch >= 0 and epoch < self.alpha_warmup_epochs:
            alpha = 0

        if self.alpha_mse_treshold > 1e-6 and mse_loss.item() > self.alpha_mse_treshold:
            alpha = 0

        loss = mse_loss + alpha * bm_loss
        return loss, {
            "bm_loss": bm_loss,
            "mse_loss": mse_loss
        }