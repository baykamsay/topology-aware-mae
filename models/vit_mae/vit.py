"""
Vision Transformer (ViT) implementation.
Adapted from the timm library for compatibility with the latest PyTorch.
"""

from functools import partial
from typing import Optional, List, Tuple, Union, Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.layers import DropPath, to_2tuple
from timm.models.vision_transformer import Mlp

from .utils import get_2d_sincos_pos_embed


class Attention(nn.Module):
    """
    Multi-head Attention module with optional drop path and QKV bias options.
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = False, 
        attn_drop: float = 0., 
        proj_drop: float = 0.,
    ):
        """
        Initialize the attention module.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to add bias to the QKV projection
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention module.
        
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer block with attention, MLP, layer normalization, and optional drop path.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        use_checkpoint: bool = False
    ):
        """
        Initialize the transformer block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            qkv_bias: Whether to add bias to the QKV projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            act_layer: Activation layer
            norm_layer: Normalization layer
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer block.
        
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Output tensor of shape (B, N, C)
        """
        if self.use_checkpoint and self.training:
            x = x + checkpoint(self._forward_attn, x)
            x = x + checkpoint(self._forward_mlp, x)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    def _forward_attn(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function for checkpointing attention."""
        return self.drop_path(self.attn(self.norm1(x)))
    
    def _forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function for checkpointing MLP."""
        return self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding module.
    """
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
    ):
        """
        Initialize the patch embedding module.
        
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input channels
            embed_dim: Embedding dimension
            norm_layer: Normalization layer
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the patch embedding module.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Check image size - flexible handling for different sizes
        if hasattr(self, 'img_size') and isinstance(self.img_size, (list, tuple)):
            expected_H, expected_W = self.img_size
            if H != expected_H or W != expected_W:
                print(f"Warning: Input image size ({H}*{W}) doesn't exactly match model ({expected_H}*{expected_W}). " 
                      f"This might affect model performance.")
        
        # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        # (B, embed_dim, H', W') -> (B, embed_dim, H'*W')
        x = x.flatten(2)
        # (B, embed_dim, H'*W') -> (B, H'*W', embed_dim)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.
    """
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        global_pool: bool = False,
        use_checkpoint: bool = False,
    ):
        """
        Initialize the Vision Transformer model.
        
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input channels
            num_classes: Number of classes for classification head
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            qkv_bias: Whether to add bias to the QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            norm_layer: Normalization layer
            global_pool: Whether to use global average pooling for classification
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.depth = depth
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim) if not global_pool else nn.Identity()
        
        # Classifier head(s)
        if global_pool:
            self.fc_norm = norm_layer(embed_dim)
            
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Identity()
            
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize the weights of the model.
        """
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize cls token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize position embedding with fixed sine-cosine embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize the weights of the model.
        
        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following the original implementation of ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor, return_intermediate: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the feature extractor with optional intermediate feature extraction.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_intermediate: If True, return intermediate features
            
        Returns:
            If return_intermediate is False, returns the final feature tensor.
            If return_intermediate is True, returns a tuple of (final features, list of intermediate features).
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Extract intermediate features at fixed intervals
        intermediate_features = []
        interval = self.depth // 4
        
        # Apply transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            
            # Extract features at fixed intervals (d/4, 2d/4, 3d/4, d)
            if return_intermediate and (i + 1) % interval == 0:
                intermediate_features.append(self.norm(x.clone()))
        
        # Apply final normalization
        x = self.norm(x)
        
        if self.global_pool:
            x = x[:, 1:].mean(dim=1)  # Global pool without cls token
            outcome = self.fc_norm(x)
        else:
            # Return all tokens including the cls token
            outcome = x  # Return all tokens
        
        if return_intermediate:
            return outcome, intermediate_features
        else:
            return outcome

    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_intermediate: If True, return intermediate features
            
        Returns:
            If return_intermediate is False, returns the final output.
            If return_intermediate is True, returns a tuple of (final output, list of intermediate features).
        """
        if return_intermediate:
            features, intermediate = self.forward_features(x, return_intermediate=True)
            if self.num_classes > 0:
                features = self.head(features)
            return features, intermediate
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x

    def no_weight_decay(self) -> List[str]:
        """
        Get the names of parameters that should not use weight decay.
        
        Returns:
            List of parameter names
        """
        return ['pos_embed', 'cls_token']


def vit_base_patch16(**kwargs):
    """
    ViT-Base model with patch size 16x16.
    """
    model_kwargs = {
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
    }
    # Override defaults with any kwargs provided
    model_kwargs.update(kwargs)
    model = VisionTransformer(**model_kwargs)
    return model


def vit_large_patch16(**kwargs):
    """
    ViT-Large model with patch size 16x16.
    """
    model_kwargs = {
        'patch_size': 16,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
    }
    # Override defaults with any kwargs provided
    model_kwargs.update(kwargs)
    model = VisionTransformer(**model_kwargs)
    return model


def vit_huge_patch14(**kwargs):
    """
    ViT-Huge model with patch size 14x14.
    """
    model_kwargs = {
        'patch_size': 14,
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 16,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
    }
    # Override defaults with any kwargs provided
    model_kwargs.update(kwargs)
    model = VisionTransformer(**model_kwargs)
    return model