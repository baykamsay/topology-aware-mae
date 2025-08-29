"""
Segmentation decoder with FPN features.
"""
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import vit

from .fpn import SegmentationFPN

# TODO Check if this code is correct, I don't remember it.
class SegmentationUpBlock(nn.Module):
    """
    Upsampling block for segmentation decoder.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        use_batchnorm: bool = True,
        use_skip: bool = True,
        skip_channels: int = 0,
    ):
        """
        Initialize the upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            scale_factor: Factor to scale the feature map by
            use_batchnorm: Whether to use batch normalization
            use_skip: Whether to use skip connections
            skip_channels: Number of channels in the skip connection
        """
        super().__init__()
        self.use_skip = use_skip and skip_channels > 0
        self.scale_factor = scale_factor
        
        # Calculate input channels for the conv blocks
        conv_in_channels = in_channels + skip_channels if self.use_skip else in_channels
        
        # Convolutional blocks after upsampling and skip connection
        self.conv1 = nn.Conv2d(
            conv_in_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.activation1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.activation2 = nn.GELU()
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the upsampling block.
        
        Args:
            x: Input tensor
            skip: Skip connection tensor (optional)
            
        Returns:
            Upsampled feature map
        """
        # Bilinear upsampling
        if self.scale_factor > 1:
            x = F.interpolate(
                x, 
                scale_factor=self.scale_factor, 
                mode='bilinear', 
                align_corners=False
            )
        
        # Concatenate with skip connection if provided
        if self.use_skip and skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        # Apply convolutional blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        
        return x


class SegmentationHead(nn.Module):
    """
    Segmentation decoder with FPN features.
    """
    def __init__(
        self,
        in_channels: List[int],
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        num_classes: int = 1,
        use_batchnorm: bool = True,
    ):
        """
        Initialize the segmentation decoder.
        
        Args:
            in_channels: List of input channels for each FPN level
            decoder_channels: List of decoder channel dimensions
            num_classes: Number of output classes
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        self.in_channels = in_channels
        self.decoder_channels = decoder_channels
        
        # Initial convolution to process the deepest FPN feature
        self.initial_conv = nn.Conv2d(
            in_channels[3], decoder_channels[0], kernel_size=3, padding=1
        )
        self.initial_bn = nn.BatchNorm2d(decoder_channels[0]) if use_batchnorm else nn.Identity()
        self.initial_act = nn.GELU()
        
        # Upsampling blocks with skip connections
        self.up_blocks = nn.ModuleList([
            SegmentationUpBlock(
                in_channels=decoder_channels[i],
                out_channels=decoder_channels[i+1],
                scale_factor=2,
                use_batchnorm=use_batchnorm,
                use_skip=True,
                skip_channels=in_channels[2-i] if i < 3 else 0
            )
            for i in range(4)
        ])
        
        # Final classification head
        self.final_conv = nn.Conv2d(
            decoder_channels[-1], num_classes, kernel_size=1
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the segmentation decoder.
        
        Args:
            features: List of FPN features [C1, C2, C3, C4] with strides 4, 8, 16, 32
                Each feature has shape (B, C, H, W)
            
        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        # Process the deepest feature with initial convolution
        x = self.initial_conv(features[3])  # Start with C4 (stride 32)
        x = self.initial_bn(x)
        x = self.initial_act(x)
        
        # Apply upsampling blocks with skip connections
        skip_features = [features[2], features[1], features[0], None]  # [C3, C2, C1, None]
        
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, skip_features[i])
        
        # Final classification head
        x = self.final_conv(x)
        
        return x


class SegmentationViT(nn.Module):
    """
    Complete segmentation model with MAE encoder, FPN, and segmentation head.
    """
    def __init__(
        self,
        model_name: str,
        encoder_config: Dict[str, Any],
        encoder_output_dim: int = 768,
        fpn_channels: List[int] = [256, 256, 256, 256],
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        num_classes: int = 1,
        use_batchnorm: bool = True,
        pretrained_path: Optional[str] = None,
        **vit_kwargs: Any
    ):
        """
        Initialize the segmentation model.
        
        Args:
            encoder: Pretrained MAE encoder
            encoder_output_dim: Dimension of encoder output (embedding dimension)
            fpn_channels: List of channel dimensions for FPN output at each scale
            decoder_channels: List of decoder channel dimensions
            num_classes: Number of output classes
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        encoder = vit.__dict__[model_name](**vit_kwargs)
        self.encoder = encoder
        self.encoder_output_dim = encoder_output_dim
        
        # Calculate grid size from the encoder's patch embedding
        if hasattr(encoder, 'patch_embed'):
            self.grid_size = int(encoder.patch_embed.num_patches**0.5)
        else:
            self.grid_size = 14  # Default for 224x224 images with 16x16 patches
        
        # Create FPN
        self.fpn = SegmentationFPN(
            in_channels=encoder_output_dim,
            out_channels=fpn_channels,
            grid_size=self.grid_size,
            use_batchnorm=use_batchnorm,
        )
        
        # Create segmentation head
        self.segmentation_head = SegmentationHead(
            in_channels=fpn_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            use_batchnorm=use_batchnorm,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        # Save original image size for resizing the output
        input_size = x.shape[2:]
        
        # Extract features from the encoder
        # Include intermediate features
        _, intermediate_features = self.encoder(x, return_intermediate=True)
        
        # Process features through FPN
        fpn_features = self.fpn(intermediate_features)
        
        # Apply segmentation head
        logits = self.segmentation_head(fpn_features)
        
        # Resize output to match input size if needed ?? TODO
        if logits.shape[2:] != input_size:
            logits = F.interpolate(
                logits,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
        
        return logits