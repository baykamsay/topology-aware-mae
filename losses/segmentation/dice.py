import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Dice Loss for segmentation.
        Args:
            smooth (float): A small value to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Raw logits from the model (N, C, H, W). 
                                   For binary segmentation, C=1.
            targets (torch.Tensor): Ground truth binary masks (N, 1, H, W), values 0 or 1.
        """
        # Apply sigmoid to inputs to get probabilities
        inputs_probs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs_flat = inputs_probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        loss = 1 - dice_coeff
        return loss