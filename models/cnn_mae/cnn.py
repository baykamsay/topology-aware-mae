"""
Basic CNN building blocks.
"""
import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    """
    Standard Convolutional Block with optional Batch Normalization and Residual Connection.

    Structure: Conv -> BN -> Activation -> Conv -> BN -> (Residual +) -> Activation
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_batchnorm: bool = True,
        use_residual: bool = True,
        activation: nn.Module = nn.GELU,
    ):
        """
        Initializes the CNN block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for convolutions. Defaults to 3.
            stride: Stride for the first convolution. Defaults to 1.
            use_batchnorm: Whether to use Batch Normalization. Defaults to True.
            use_residual: Whether to use a residual connection. Only applied if
                          in_channels == out_channels and stride == 1. Defaults to True.
            activation: Activation function module. Defaults to nn.GELU.
        """
        super().__init__()
        # Residual connection is only possible if dimensions match and no downsampling
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        padding = kernel_size // 2  # Same padding

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm  # Bias is not needed if BatchNorm follows
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.act1 = activation()

        # Second convolutional layer (always stride 1)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=not use_batchnorm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.act2 = activation() # Activation after potential residual add

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN block.

        Args:
            x: Input tensor (B, C_in, H, W).

        Returns:
            Output tensor (B, C_out, H', W').
        """
        identity = x

        # First layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        # Second layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Add residual connection if applicable
        if self.use_residual:
            out = out + identity # Add residual before final activation

        # Final activation
        out = self.act2(out)

        return out

