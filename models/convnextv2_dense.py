"""
Dense simulation of Sparse ConvNeXtV2 for FCMAE encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm, DenseGRN # Use DenseGRN

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
        # 1. Apply mask before conv (zero out masked features)
        if mask is not None:
            # Ensure mask is in NCHW format if needed, or broadcastable
            if mask.shape[1] != x.shape[1] and mask.shape[1] == 1: # (N, 1, H, W) -> (N, C, H, W)
                 mask_nchw = mask.repeat(1, x.shape[1], 1, 1)
            else:
                 mask_nchw = mask # Assume it's already (N, C, H, W) or similar
            x = x * (1. - mask_nchw)

        x = self.dwconv(x)

        # 2. Apply mask after conv (zero out features *at* masked locations)
        if mask is not None:
            x = x * (1. - mask_nchw)
        # --- End Simulation ---

        # Permute for LayerNorm and Linear layers
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        # Prepare mask for channel-last format if needed for GRN
        mask_nhwc = None
        if mask is not None:
             # Ensure mask is broadcastable for channels_last: (N, H, W, 1)
             if mask.shape[1] == 1: # N1HW format original
                 mask_nhwc = mask.permute(0, 2, 3, 1) # N1HW -> NHW1
             elif mask.dim() == 4 and mask.shape[-1] == 1: # Already NHW1
                 mask_nhwc = mask
             # Add more sophisticated checks/reshaping if needed based on how mask is passed

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x, mask=mask_nhwc) # Pass mask to DenseGRN
        x = self.pwconv2(x)

        # Permute back to channels_first
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        # Apply DropPath to the residual connection
        x = input + self.drop_path(x)
        return x


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
                 drop_path_rate=0.):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.num_stages = len(depths)

        # Stem and downsampling layers (using standard dense layers)
        self.downsample_layers = nn.ModuleList()
        # Stem: Conv2d (patchify) + LayerNorm (channels_first)
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        # Intermediate downsampling layers: LayerNorm (channels_first) + Conv2d (stride=2)
        for i in range(self.num_stages - 1):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
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


    def forward(self, x, patch_mask):
        """ Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input images (N, C, H, W).
            patch_mask (torch.Tensor): Binary mask for *patches* (N, L),
                                       where L is num_patches. 0=keep, 1=remove/mask.

        Returns:
            torch.Tensor: Encoded features (dense tensor).
        """
        # Stem (Patchify)
        x = self.downsample_layers[0](x) # (N, C_stem, H/4, W/4)

        # Upsample patch mask to match the spatial resolution after stem
        # patch_mask is (N, L), needs to become (N, H/4, W/4) or similar
        N, L = patch_mask.shape
        H_patch, W_patch = x.shape[-2], x.shape[-1] # Spatial dim after stem
        if L != H_patch * W_patch:
             # This indicates an issue with config (patch_size, input_size) vs stem stride
             raise ValueError(f"Number of patches ({L}) does not match feature map size ({H_patch}x{W_patch}) after stem.")

        # Reshape patch_mask to spatial format (N, 1, H/4, W/4) - 1 means mask
        mask = patch_mask.reshape(N, 1, H_patch, W_patch).float() # Ensure float for multiplication

        # Apply mask *after* the stem convolution (similar to original sparse impl.)
        x = x * (1. - mask)

        # Main stages
        for i in range(self.num_stages):
            # Apply downsampling (except for the first stage)
            if i > 0:
                x = self.downsample_layers[i](x)
                # Downsample the mask if needed (e.g., max pooling or avg pooling)
                # For stride=2 conv, mask spatial dim halves
                mask = F.max_pool2d(mask, kernel_size=2, stride=2)
                # Alternatively: F.avg_pool2d(mask, kernel_size=2, stride=2) might preserve ratios better

            # Apply blocks within the stage, passing the mask
            for block in self.stages[i]:
                x = block(x, mask=mask) # Pass the correctly sized mask to each block

        # Output is the dense feature map from the last stage
        return x