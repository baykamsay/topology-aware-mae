"""
MSE loss function for pretraining.
"""

import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    """
    Masked Mean Squared Error loss.
    Only computes the loss on masked tokens.
    """
    
    def __init__(self, model, norm_pix_loss=False):
        super().__init__()
        self.patchify = model.patchify
        self.norm_pix_loss = norm_pix_loss

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

        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # [N, L], mean loss per patch over P*P*C dimensions

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss