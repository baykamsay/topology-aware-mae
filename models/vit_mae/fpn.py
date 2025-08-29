"""
Feature Pyramid Network (FPN) implementation for segmentation.
"""
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO Check if this code is correct
class SegmentationFPN(nn.Module):
    """
    Feature Pyramid Network for segmentation, adapted from the implementation in the paper
    "Benchmarking Detection Transfer Learning with Vision Transformers".
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int] = [256, 256, 256, 256],
        grid_size: int = 14,  # Default for 224x224 images with 16x16 patches
        use_batchnorm: bool = True,
    ):
        """
        Initialize the Feature Pyramid Network.
        
        Args:
            in_channels: Number of channels in the input features
            out_channels: List of output channels for each pyramid level
            grid_size: Size of the grid (number of patches in one dimension)
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        
        # Create lateral convolutions to reduce channel dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels[i], kernel_size=1)
            for i in range(4)
        ])
        
        # Create top-down pathway with upsampling and blending
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels[i], out_channels[i], kernel_size=3, padding=1)
            for i in range(4)
        ])
        
        # Normalization layers if requested
        if use_batchnorm:
            self.lateral_norms = nn.ModuleList([
                nn.BatchNorm2d(out_channels[i])
                for i in range(4)
            ])
            self.fpn_norms = nn.ModuleList([
                nn.BatchNorm2d(out_channels[i])
                for i in range(4)
            ])
        else:
            self.lateral_norms = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.fpn_norms = nn.ModuleList([nn.Identity() for _ in range(4)])
        
        # Activation function
        self.activation = nn.GELU()
        
        # Resolution modifiers for different scales
        self.up_scale_4x = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels) if use_batchnorm else nn.Identity(),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )
        
        self.up_scale_2x = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels) if use_batchnorm else nn.Identity(),
            nn.GELU(),
        )
        
        self.down_scale_2x = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through the Feature Pyramid Network.
        
        Args:
            features: List of ViT features [P1, P2, P3, P4] from transformer blocks
                P1 is from block d/4, P2 from 2d/4, P3 from 3d/4, P4 from d
                Each feature has shape (B, N, C) where N is the number of tokens
            
        Returns:
            List of FPN features [C1, C2, C3, C4] with strides 4, 8, 16, 32
        """
        # Convert ViT features to 2D feature maps
        feature_maps = []
        for i, feature in enumerate(features):
            # For ViT features, reshape them from (B, N, C) to (B, C, H, W)
            B, N, C = feature.shape
            
            # Remove CLS token if present
            if N - 1 == self.grid_size ** 2:
                feature = feature[:, 1:, :]
                N = N - 1
            
            # Reshape to 2D feature map
            H = W = int(N ** 0.5)
            feature_map = feature.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            feature_maps.append(feature_map)
        
        # Apply resolution modifiers to create multi-scale features
        p1 = self.up_scale_4x(feature_maps[0])  # d/4 block -> stride 4
        p2 = self.up_scale_2x(feature_maps[1])  # 2d/4 block -> stride 8
        p3 = feature_maps[2]  # 3d/4 block -> stride 16
        p4 = self.down_scale_2x(feature_maps[3])  # d block -> stride 32
        
        modified_features = [p1, p2, p3, p4]
        
        # Apply lateral convolutions
        laterals = []
        for i, feature in enumerate(modified_features):
            lateral = self.lateral_convs[i](feature)
            lateral = self.lateral_norms[i](lateral)
            lateral = self.activation(lateral)
            laterals.append(lateral)
        
        # Top-down pathway (highest resolution to lowest)
        fpn_features = [laterals[3]]  # Start with the coarsest level (P4)
        
        for i in range(2, -1, -1):  # From 2 to 0
            # Upsample the previous FPN feature
            upsampled = F.interpolate(
                fpn_features[-1], 
                size=laterals[i].shape[-2:],
                mode='bilinear', 
                align_corners=False
            )
            
            # Add the lateral connection
            feature = laterals[i] + upsampled
            
            # Apply convolution
            feature = self.fpn_convs[i](feature)
            feature = self.fpn_norms[i](feature)
            feature = self.activation(feature)
            
            fpn_features.append(feature)
        
        # Reverse to get features from fine to coarse - matching strides 4, 8, 16, 32
        fpn_features.reverse()
        
        return fpn_features