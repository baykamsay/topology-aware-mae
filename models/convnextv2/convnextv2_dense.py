"""
Dense simulation of Sparse ConvNeXtV2 for FCMAE encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
from .utils import LayerNorm, DenseGRN

class Block(nn.Module):
    """ Dense ConvNeXtV2 Block simulating sparse convolutions with masking.
        Based on the provided JAX implementation concept.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # Depthwise convolution (groups=dim makes it depthwise)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        # Use standard LayerNorm, assumes channels_last format input after permute
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        # Pointwise/1x1 convolutions implemented using linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        # Use the mask-aware DenseGRN
        self.grn = DenseGRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # DropPath for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor (N, C, H, W)
            mask (torch.Tensor, optional): Binary mask (N, H, W, 1) or (N, 1, H, W).
                                            1 indicates masked (ignored), 0 indicates visible.
                                            Defaults to None (no masking).
        """
        input = x

        # --- Simulate sparse convolution ---
        if mask is not None:
            x = x * (1. - mask)

        x = self.dwconv(x)

        if mask is not None:
            x = x * (1. - mask)
        # --- End Simulation ---

        # Permute for LayerNorm and Linear layers
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x, mask=mask) # Pass mask to DenseGRN
        x = self.pwconv2(x)

        # Permute back to channels_first
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        # Apply DropPath to the residual connection
        x = input + self.drop_path(x)
        return x
    
class MaskedDownsample(nn.Module):
    """ Downsampling with a convolutional layer that supports masking to simulate sparse convolutions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, mask=None):
        x = self.norm(x)

        if mask is not None:
            x = x * (1. - mask)
        x = self.conv(x)

        # If mask is provided, downsample it as well
        if mask is not None:
            mask = F.max_pool2d(mask, kernel_size=2, stride=2)
            x = x * (1. - mask)

        # Return the mask too for the next layers
        return x, mask

class DenseConvNeXtV2(nn.Module):
    """ Dense ConvNeXtV2 simulating the sparse version for FCMAE encoder.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
    """
    def __init__(self,
                 in_chans=3,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 stem_size=4):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.num_stages = len(depths)

        # Stem and downsampling layers
        self.downsample_layers = nn.ModuleList()
        # Stem: Conv2d (patchify) + LayerNorm (channels_first)
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=stem_size, stride=stem_size),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        # Intermediate downsampling layers: LayerNorm (channels_first) + Conv2d (stride=2)
        for i in range(self.num_stages - 1):
            downsample_layer = MaskedDownsample(
                in_channels=dims[i],
                out_channels=dims[i + 1]
            )
            self.downsample_layers.append(downsample_layer)

        # Feature resolution stages, each consisting of multiple Blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            stage_blocks = nn.ModuleList([
                Block(dim=dims[i], drop_path=dp_rates[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(stage_blocks)
            cur += depths[i]

        # No final norm or head - this is just the encoder backbone
        self.apply(self._init_weights) # Apply weight initialization

    def _init_weights(self, m):
        """ Initialize weights """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)): # Includes our custom LayerNorm if it inherits nn.Module
             if m.bias is not None:
                 nn.init.constant_(m.bias, 0)
             if m.weight is not None:
                 nn.init.constant_(m.weight, 1.0)

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)

    def forward(self, x, patch_mask):
        """ Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input images (N, C, H, W).
            patch_mask (torch.Tensor): Binary mask for *patches* (N, L),
                                       where L is num_patches. 0=keep, 1=remove/mask.

        Returns:
            torch.Tensor: Encoded features (dense tensor).
        """
        mask = self.upsample_mask(patch_mask, 2**(self.num_stages-1)) 
        mask = mask.unsqueeze(1).type_as(x)

        # Stem (Patchify)
        x = self.downsample_layers[0](x) # (N, C_stem, H/4, W/4)

        # Apply mask *after* the stem convolution (similar to original sparse impl.)
        x = x * (1. - mask)

        # Main stages
        for i in range(self.num_stages):
            # Apply downsampling (except for the first stage)
            if i > 0:
                x, mask = self.downsample_layers[i](x, mask=mask) # Downsample and get new mask

            # Apply blocks within the stage, passing the mask
            for block in self.stages[i]:
                x = block(x, mask=mask) # Pass the correctly sized mask to each block

        # Output is the dense feature map from the last stage
        return x