"""
Masked Autoencoder (MAE) model implementation.
Based on the paper: "Masked Autoencoders Are Scalable Vision Learners"
"""
from functools import partial
from typing import Tuple, Optional, Dict, Any, List, Union

import torch
import torch.nn as nn

from .vit import VisionTransformer, PatchEmbed, Block
from .utils import get_2d_sincos_pos_embed
from losses.pretrain import get_loss_function

class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with Vision Transformer backbone.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.,
        norm_layer: nn.Module = nn.LayerNorm,
        use_checkpoint: bool = False,
        loss_config={'name': 'mse'},
        device=None,
        pretrained_path=None,
        mask_ratio: float = 0.75
    ):
        """
        Initialize the Masked Autoencoder model.
        
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input channels
            embed_dim: Embedding dimension for encoder
            depth: Number of transformer blocks in encoder
            num_heads: Number of attention heads in encoder
            decoder_embed_dim: Embedding dimension for decoder
            decoder_depth: Number of transformer blocks in decoder
            decoder_num_heads: Number of attention heads in decoder
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            norm_layer: Normalization layer
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # Additions
        # Validation generator
        self.val_gen = torch.Generator(device=device)
        self.val_gen.manual_seed(27)  # Set seed for reproducible masks

        # Loss function
        self.loss_func = get_loss_function(loss_config, self)
        if self.loss_func is None:
            raise ValueError(f"Loss function {loss_config['name']} not found.")
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)  # decoder to pixel
        # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.depth = depth
        
        self.initialize_weights()

        if pretrained_path:
            self.load_pretrained_encoder(pretrained_path)

    def initialize_weights(self):
        """
        Initialize the weights of the model.
        """
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize cls token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Calculate grid size based on the patch embedding
        # This ensures proper compatibility with different image sizes
        grid_size = int(self.patch_embed.num_patches**0.5)

        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            grid_size,
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            grid_size,
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize the weights of the model.
        
        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following the original implementation
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def load_pretrained_encoder(self, pretrained_path): # TODO
        """
        Load pretrained encoder weights from a given path.
        """
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)
            
            self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained encoder from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained encoder: {e}")
    
    def eval(self):
        super().eval()
        self.reset_val_seed(27)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.
        
        Args:
            imgs: Input images of shape (N, 3, H, W)
            
        Returns:
            Patches of shape (N, L, patch_size^2 * 3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images.
        
        Args:
            x: Patches of shape (N, L, patch_size^2 * 3)
            
        Returns:
            Images of shape (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        
        # Ensure the number of patches can be reshaped into a square grid
        assert h * w == x.shape[1], f"Number of patches {x.shape[1]} must be a perfect square"
        
        # Ensure patch size matches what we expect
        assert x.shape[2] == p**2 * 3, f"Patch dimension {x.shape[2]} doesn't match expected size {p**2 * 3}"
        
        # Reshape to [B, h, w, p, p, 3]
        try:
            x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        except RuntimeError as e:
            print(f"Error reshaping patches: {e}")
            print(f"x.shape: {x.shape}, h: {h}, w: {w}, p: {p}")
            raise
            
        # Permute and reshape to image
        try:
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
            return imgs
        except RuntimeError as e:
            print(f"Error in einsum/reshaping to image: {e}")
            print(f"After first reshape, x.shape: {x.shape}")
            raise
    
    def reset_val_seed(self, seed=27):
        """
        Reset the validation generator seed.
        This is for reproducibility during validation.
        """
        self.val_gen.manual_seed(seed)

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform per-sample random masking by per-sample shuffling.
        
        Args:
            x: Input tokens of shape (N, L, D)
            mask_ratio: Proportion of tokens to mask
            
        Returns:
            Tuple of:
                - Subset of tokens that are kept, shape (N, L*(1-mask_ratio), D)
                - Mask of shape (N, L) where 0 is keep, 1 is remove
                - Restore indices of shape (N, L)
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        if self.training:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        else:
            # use the validation generator for reproducibility
            noise = torch.rand(N, L, device=x.device, generator=self.val_gen)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float, return_intermediate: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the encoder with masking.
        
        Args:
            x: Input images of shape (N, 3, H, W)
            mask_ratio: Proportion of tokens to mask
            return_intermediate: Whether to return intermediate features
            
        Returns:
            Tuple of:
                - Encoded features
                - Mask
                - Restore indices
                - (Optional) List of intermediate features
        """
        # Embed patches
        x = self.patch_embed(x)

        # Add position embeddings (without cls token)
        x = x + self.pos_embed[:, 1:, :]

        # Masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Extract intermediate features at fixed intervals if requested
        intermediate_features = []
        interval = self.depth // 4
        
        # Apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            
            # Extract features at fixed intervals (d/4, 2d/4, 3d/4, d)
            if return_intermediate and (i + 1) % interval == 0:
                intermediate_features.append(self.norm(x.clone()))

        # Apply final normalization
        x = self.norm(x)
        
        if return_intermediate:
            return x, mask, ids_restore, intermediate_features
        else:
            return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            x: Encoded features
            ids_restore: Indices for restoring the original sequence
            
        Returns:
            Decoded features
        """
        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add position embeddings
        x = x + self.decoder_pos_embed

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75, return_all_tokens: bool = False, return_intermediate: bool = False, epoch: int = -1) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass of the model.
        
        Args:
            imgs: Input images of shape (N, 3, H, W)
            mask_ratio: Proportion of tokens to mask
            return_all_tokens: Whether to return all tokens or just the masked ones
            return_intermediate: Whether to return intermediate features
            
        Returns:
            If return_intermediate is False:
                Tuple of (pred, mask, target)
            If return_intermediate is True:
                Tuple of (pred, mask, target, intermediate_features)
        """
        if return_intermediate: # TODO fix
            latent, mask, ids_restore, intermediate_features = self.forward_encoder(
                imgs, mask_ratio, return_intermediate=True
            )
        else:
            latent, mask, ids_restore = self.forward_encoder(
                imgs, mask_ratio, return_intermediate=False
            )
            
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]

        loss, individual_losses = self.loss_func(imgs, pred, mask, epoch)
        
        return loss, pred, mask, individual_losses
    
    def no_weight_decay(self) -> Dict[str, Any]:
        """
        Get the names of parameters that should not use weight decay.
        
        Returns:
            List of parameter names
        """
        return {'pos_embed', 'decoder_pos_embed', 'cls_token', 'mask_token'}
    
    def get_encoder(self) -> VisionTransformer:
        """
        Extract the encoder part of the model for fine-tuning.
        
        Returns:
            Encoder model with intermediate feature extraction capabilities
        """
        # Create a VisionTransformer with the same parameters as the encoder
        encoder = VisionTransformer(
            img_size=self.patch_embed.img_size[0],
            patch_size=self.patch_embed.patch_size[0],
            in_chans=3,
            embed_dim=self.pos_embed.shape[-1],
            depth=len(self.blocks),
            num_heads=self.blocks[0].attn.num_heads,
            mlp_ratio=4.0,
            global_pool=False,  # No global pooling
            num_classes=0,  # No classification head
        )
        
        # Copy the weights from the encoder part of the MAE
        encoder.patch_embed = self.patch_embed
        encoder.cls_token = self.cls_token
        encoder.pos_embed = self.pos_embed
        encoder.blocks = self.blocks
        encoder.norm = self.norm
        
        return encoder


def mae_vit_base_patch16(**kwargs):
    """
    MAE with ViT-Base backbone.
    """
    model_kwargs = {
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'decoder_embed_dim': 512,
        'decoder_depth': 8,
        'decoder_num_heads': 16,
        'mlp_ratio': 4,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
    }
    # Override defaults with any kwargs provided
    model_kwargs.update(kwargs)
    model = MaskedAutoencoderViT(**model_kwargs)
    return model


def mae_vit_large_patch16(**kwargs):
    """
    MAE with ViT-Large backbone.
    """
    model_kwargs = {
        'patch_size': 16,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'decoder_embed_dim': 512,
        'decoder_depth': 8,
        'decoder_num_heads': 16,
        'mlp_ratio': 4,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
    }
    # Override defaults with any kwargs provided
    model_kwargs.update(kwargs)
    model = MaskedAutoencoderViT(**model_kwargs)
    return model


def mae_vit_huge_patch14(**kwargs):
    """
    MAE with ViT-Huge backbone.
    """
    model_kwargs = {
        'patch_size': 14,
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 16,
        'decoder_embed_dim': 512,
        'decoder_depth': 8,
        'decoder_num_heads': 16,
        'mlp_ratio': 4,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
    }
    # Override defaults with any kwargs provided
    model_kwargs.update(kwargs)
    model = MaskedAutoencoderViT(**model_kwargs)
    return model