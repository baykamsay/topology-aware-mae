"""
Utility modules for the models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGRN(nn.Module):
    """ GRN layer that handles masking for dense simulation of sparse conv.
        Based on the provided JAX implementation concept.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim)) # Parameters for affine transform
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x, mask=None):
        # Assumes input format is (N, H, W, C) - channels_last and mask is (N, C, H, W)
        inputs = x # Store original input for residual connection & final scaling

        if mask is not None:
            mask_hwc = mask.permute(0, 2, 3, 1)
            x = x * (1. - mask_hwc)

        # Calculate norm over spatial dimensions (H, W)
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        # Calculate mean norm across channels (C)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)

        # Apply GRN: Scale original input by Nx, add bias, and add residual
        # Note: Scaling the *original* input 'inputs' by Nx, as per JAX code example
        return self.gamma * (inputs * Nx) + self.beta + inputs

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    Adapted from https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/utils.py
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Adapted from https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/utils.py
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x