"""
MSE loss function for pretraining.
"""

import torch

def mse_loss(pred, target):
    """
    Mean Squared Error (MSE) loss function.
    
    Args:
        pred (torch.Tensor): Predicted patches (N, L, P*P*C).
        target (torch.Tensor): Target patches (N, L, P*P*C).
    
    Returns:
        torch.Tensor: Computed MSE loss per patch.
    """

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1) # [N, L], mean loss per patch over P*P*C dimensions
    return loss