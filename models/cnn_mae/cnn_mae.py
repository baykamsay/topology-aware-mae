"""
Masked Autoencoder (MAE) with CNN backbone.
Includes fix for reshape error in forward_decoder and num_patches mismatch.
"""
from typing import Tuple, List, Optional, Union, Dict, Any
import math # Import math for sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.pretrain import get_loss_function

# Import the basic CNN block
from .cnn import CNNBlock

# Re-introduce CNNEncoder
class CNNEncoder(nn.Module):
    """
    CNN encoder for the Masked Autoencoder.
    Designed to output intermediate features suitable for U-Net skip connections.
    """
    def __init__(
        self,
        img_size: int = 56,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 64, # Base dimension, increases with depth
        depths: List[int] = [2, 2, 2], # Number of blocks at each stage (excluding stem)
        dims: Optional[List[int]] = None, # Output channels for each stage
        use_batchnorm: bool = True,
        use_checkpointing: bool = False, # Not implemented yet
    ):
        """
        Initialize the CNN encoder.
        """
        super().__init__()

        if not isinstance(img_size, (tuple, list)):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_stages = len(depths)

        # Calculate feature dimensions at each stage if not provided
        if dims is None:
            self.dims = [embed_dim * (2**i) for i in range(self.num_stages + 1)]
        else:
            if len(dims) != self.num_stages + 1:
                raise ValueError(f"Length of dims ({len(dims)}) must be len(depths) + 1 ({self.num_stages + 1})")
            self.dims = dims

        # Initial convolution (stem)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, self.dims[0], kernel_size=patch_size, stride=1, padding=0, bias=not use_batchnorm),
            nn.BatchNorm2d(self.dims[0]) if use_batchnorm else nn.Identity(),
            nn.GELU()
        )

        # Build the encoder stages
        self.stages = nn.ModuleList()
        current_dim = self.dims[0]
        h_cur, w_cur = img_size[0] // 1, img_size[1] // 1 # Size after stem
        for i in range(self.num_stages):
            stage_blocks = []
            # First block performs downsampling (stride 2)
            stage_blocks.append(
                CNNBlock(
                    in_channels=current_dim,
                    out_channels=self.dims[i+1],
                    kernel_size=3,
                    stride=2, # Downsamples H, W
                    use_batchnorm=use_batchnorm,
                    use_residual=False
                )
            )
            # Calculate size after downsampling block
            h_cur = math.floor((h_cur + 2 * 1 - 3) / 2 + 1) # K=3, S=2, P=1
            w_cur = math.floor((w_cur + 2 * 1 - 3) / 2 + 1)

            current_dim = self.dims[i+1]

            # Add remaining blocks for this stage (stride 1)
            for _ in range(depths[i] - 1):
                stage_blocks.append(
                    CNNBlock(
                        in_channels=current_dim,
                        out_channels=current_dim,
                        kernel_size=3,
                        stride=1, # Maintains H, W
                        use_batchnorm=use_batchnorm,
                        use_residual=True
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))

        # Final normalization
        self.norm = nn.BatchNorm2d(self.dims[-1]) if use_batchnorm else nn.Identity()

        # Calculate num_patches based on INPUT image size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        # Store final feature map size separately
        self.feature_h = h_cur
        self.feature_w = w_cur
        print(f"[CNNEncoder Init] img_size={img_size}, patch_size={patch_size}, num_stages={self.num_stages}")
        print(f"[CNNEncoder Init] Final feature_h={self.feature_h}, feature_w={self.feature_w}")
        print(f"[CNNEncoder Init] num_patches (from input image) = {self.num_patches}")


        # Store intermediate feature dimensions for U-Net
        self.intermediate_dims = self.dims[:-1]

    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the CNN encoder.
        """
        # Initial convolution (stem)
        x = self.stem(x)
        intermediate_features = [x] if return_intermediate else None

        # Apply CNN stages
        for stage in self.stages:
            x = stage(x)
            if return_intermediate:
                intermediate_features.append(x)

        # Final normalization
        x = self.norm(x)

        if return_intermediate:
            return x, intermediate_features
        else:
            return x

class CNNDecoder(nn.Module):
    """
    CNN decoder for the Masked Autoencoder.
    Predicts patch values from latent features.
    """
    def __init__(
        self,
        patch_size: int = 4,
        num_patches: int = 196, # Number of patches based on input image size
        in_chans: int = 3,
        decoder_dim: int = 256, # Base dimension for the decoder
        decoder_depth: int = 3, # Number of decoder blocks
        use_batchnorm: bool = True
    ):
        """
        Initialize the CNN decoder.
        """
        super().__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = num_patches # Use the correct L
        self.decoder_dim = decoder_dim

        # Decoder blocks (simple CNN blocks without up/down sampling)
        self.decoder_blocks = nn.ModuleList([
            CNNBlock(
                in_channels=decoder_dim,
                out_channels=decoder_dim,
                kernel_size=3,
                stride=1,
                use_batchnorm=use_batchnorm,
                use_residual=True
            )
            for _ in range(decoder_depth)
        ])

        self.norm = nn.BatchNorm2d(decoder_dim) if use_batchnorm else nn.Identity()

        # Final prediction layer
        # Input: (N, L, decoder_dim)
        # Output: (N, L, patch_size*patch_size*in_chans)
        self.final_pred = nn.Linear(decoder_dim, patch_size * patch_size * in_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN decoder.

        Args:
            x: Latent features including mask tokens, reshaped to spatial format.
               Expected shape (N, decoder_dim, H_feat, W_feat).
               H_feat * W_feat must equal self.num_patches.

        Returns:
            Predicted patches of shape (N, L, patch_size*patch_size*in_chans).
        """
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x) # Maintains spatial dimensions (H_feat, W_feat)

        x = self.norm(x)

        # Reshape for linear layer: (N, D, H, W) -> (N, H*W, D) = (N, L, D)
        N, D, H, W = x.shape
        L = H * W
        # Sanity check - L must match the number of patches expected (e.g., 196)
        if L != self.num_patches:
             # This error should not happen if the input `x` was reshaped correctly before calling this forward
             raise ValueError(f"CNNDecoder forward Error: Input spatial size {H}x{W}={L} does not match expected num_patches {self.num_patches}")

        x = x.permute(0, 2, 3, 1).reshape(N, L, D) # (N, L, decoder_dim)

        # Final prediction
        x = self.final_pred(x) # (N, L, patch_size*patch_size*in_chans)

        return x


class MaskedAutoencoderCNN(nn.Module):
    """
    Masked Autoencoder with CNN backbone.
    Outputs patch-based predictions for external loss calculation.
    """
    def __init__(
        self,
        img_size: int = 56,
        patch_size: int = 4,
        in_chans: int = 3,
        # Encoder params
        encoder_embed_dim: int = 64,
        encoder_depths: List[int] = [2, 2, 6], # Example for 'small'
        encoder_dims: Optional[List[int]] = None,
        # Decoder params
        decoder_dim: int = 256, # Embedding dim in decoder blocks
        decoder_depth: int = 3, # Number of blocks in decoder
        # Shared params
        use_batchnorm: bool = True,
        use_checkpointing: bool = False, # Placeholder
        mask_ratio: float = 0.75,
        # Loss
        # norm_pix_loss: bool = False,
        loss_config: Dict[str, Any] = {'name': 'mse'}
    ):
        """
        Initialize the CNN-based Masked Autoencoder.
        """
        super().__init__()

        if not isinstance(img_size, (tuple, list)):
             img_size = (img_size, img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        # self.norm_pix_loss = norm_pix_loss

        # loss function
        self.loss_func = get_loss_function(loss_config, self)
        if self.loss_func is None:
            raise ValueError(f"Loss function {loss_config['name']} not found.")

        # --- Encoder ---
        self.encoder = CNNEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depths=encoder_depths,
            dims=encoder_dims,
            use_batchnorm=use_batchnorm,
            use_checkpointing=use_checkpointing
        )
        # Get encoder output dim and CORRECT number of patches (L)
        self.encoder_output_dim = self.encoder.dims[-1]
        self.num_patches = self.encoder.num_patches # L = (H/p) * (W/p)

        # --- Projection (Optional) ---
        if self.encoder_output_dim != decoder_dim:
            self.decoder_embed = nn.Conv2d(self.encoder_output_dim, decoder_dim, kernel_size=1)
            print(f"Using Conv2d projection from encoder dim {self.encoder_output_dim} to decoder dim {decoder_dim}")
        else:
            self.decoder_embed = nn.Identity()
            print(f"Encoder output dim {self.encoder_output_dim} matches decoder dim {decoder_dim}. Using Identity projection.")

        # --- Mask Token ---
        self.mask_token_value = nn.Parameter(torch.zeros(1, decoder_dim))

        # --- Decoder ---
        # Pass the correct num_patches (e.g., 196)
        self.decoder = CNNDecoder(
            patch_size=patch_size,
            num_patches=self.num_patches, # Use L based on input image
            in_chans=in_chans,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            use_batchnorm=use_batchnorm
        )

        # Initialize weights
        self._init_weights()

        # For compatibility: store last mask/predictions if needed by get_reconstructed_images
        self.last_mask: Optional[torch.Tensor] = None
        self.last_pred_patches: Optional[torch.Tensor] = None
        self.last_target_patches: Optional[torch.Tensor] = None


    def _init_weights(self):
        """Initialize the weights."""
        nn.init.normal_(self.mask_token_value, std=0.02)
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m):
         """Initialize individual layers."""
         if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             if m.bias is not None:
                 nn.init.zeros_(m.bias)
         elif isinstance(m, nn.BatchNorm2d):
             nn.init.ones_(m.weight)
             nn.init.zeros_(m.bias)
         elif isinstance(m, nn.Linear):
              nn.init.xavier_uniform_(m.weight)
              if m.bias is not None:
                  nn.init.zeros_(m.bias)
         elif isinstance(m, nn.Parameter):
              if m is not self.mask_token_value:
                   nn.init.normal_(m, std=0.02)


    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches (N, L, D_patch).
        """
        p = self.patch_size
        h, w = imgs.shape[2], imgs.shape[3]
        # if h % p != 0 or w % p != 0:
        #      # Allow non-divisible sizes, F.unfold handles padding implicitly if needed
        #      print(f"Warning [patchify]: Image dimensions ({h}x{w}) not divisible by patch size ({p}).")

        # Use unfold to extract patches efficiently
        patches = F.unfold(imgs, kernel_size=p, stride=p).transpose(1, 2)

        # Verify L matches the expected num_patches from init
        if patches.shape[1] != self.num_patches:
             print(f"CRITICAL WARNING [patchify]: Actual number of patches {patches.shape[1]} differs from expected {self.num_patches}. This WILL cause errors.")
        return patches


    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images (N, C, H, W).
        """
        p = self.patch_size
        h_img, w_img = self.img_size
        num_patches_expected = self.num_patches
        patch_dim_expected = p * p * self.in_chans

        N, L, D = x.shape
        if L != num_patches_expected:
             raise ValueError(f"Cannot unpatchify: Number of patches L={L} does not match expected {num_patches_expected}")
        if D != patch_dim_expected:
             raise ValueError(f"Patch dimension {D} doesn't match expected size {patch_dim_expected}")

        # Use fold to reconstruct the image
        x = x.transpose(1, 2)
        imgs = F.fold(x, output_size=(h_img, w_img), kernel_size=p, stride=p)
        return imgs


    def random_masking_spatial(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform per-sample random masking on spatial feature maps (N, D, H_feat, W_feat).
        Generates mask and indices based on the *original image patch count* (self.num_patches).
        Returns the *full* feature map along with mask and indices.

        Args:
            x: Input feature map of shape (N, D, H_feat, W_feat).
            mask_ratio: Proportion of spatial locations (patches) to mask.

        Returns:
            Tuple of:
                - x: The input feature map `x` unchanged (N, D, H_feat, W_feat).
                - mask: Binary mask indicating masked locations (N, L) where L = self.num_patches, 1=masked.
                - ids_restore: Indices to restore the original sequence (N, L).
        """
        N, D, H, W = x.shape
        L_feat = H * W # Number of spatial locations in the feature map

        # --- Use self.num_patches (e.g., 196) for mask generation ---
        L = self.num_patches
        len_keep = int(L * (1 - mask_ratio))

        # Generate noise and sort based on L (196)
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask (1 is masked) based on L (196)
        mask = torch.ones([N, L], device=x.device, dtype=torch.int)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore) # Unshuffle mask

        # --- REMOVED WARNING ---
        if L_feat != L:
             print(f"Warning: L_feat ({L_feat}) != L ({L}). Masking strategy might be suboptimal.")

        # --- Return the full features, mask, and restore indices ---
        # The decoder will handle using the mask.
        return x, mask, ids_restore # Shape: (N, D, H, W), (N, L), (N, L)


    def forward_encoder(self, x: torch.Tensor, mask_ratio: float, return_intermediate: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the encoder, applies projection, and prepares data for decoder.

        Args:
            x: Input images of shape (N, C, H, W).
            mask_ratio: Proportion of patches to mask.
            return_intermediate: Whether to return intermediate features for U-Net.

        Returns:
            Tuple containing:
                - latent_full_proj: Full projected features (N, D_dec, H_feat, W_feat).
                - mask: Binary mask for all patches (N, L), 1=masked. L based on input image.
                - ids_restore: Indices to restore the full sequence (N, L). L based on input image.
                - (Optional) intermediate_features: List of raw features from encoder for U-Net skips.
        """
        # 1. Full Encoder Pass
        if return_intermediate:
            latent_full, intermediate_features = self.encoder(x, return_intermediate=True)
        else:
            latent_full = self.encoder(x, return_intermediate=False)
            intermediate_features = None
        # latent_full shape: (N, D_enc, H_feat, W_feat)

        # 2. Apply decoder embedding projection (if needed)
        latent_full_proj = self.decoder_embed(latent_full) # (N, D_dec, H_feat, W_feat)

        # 3. Perform spatial masking based on L = num_patches (e.g., 196)
        # Gets the full features back, plus mask and indices based on L=196.
        latent_full_proj_ignored, mask, ids_restore = self.random_masking_spatial(latent_full_proj, mask_ratio)
        # latent_full_proj_ignored is same as latent_full_proj
        # mask shape: (N, L)
        # ids_restore shape: (N, L)

        if return_intermediate:
            # Return the full projected features, mask, indices, and intermediate skips
            return latent_full_proj, mask, ids_restore, intermediate_features
        else:
            return latent_full_proj, mask, ids_restore


    def forward_decoder(self, x_full_proj: torch.Tensor, mask: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder. Handles mask token insertion.

        Args:
            x_full_proj: Full encoded & projected features (N, D_dec, H_feat, W_feat).
            mask: Binary mask for all patches (N, L), 1=masked. L based on input image.
            ids_restore: Indices for restoring the original sequence (N, L). (Not strictly needed here anymore).

        Returns:
            Predicted patches (N, L, patch_dim).
        """
        N, D_decoder, H_feat, W_feat = x_full_proj.shape
        L_feat = H_feat * W_feat
        L = mask.shape[1] # Use L from mask (e.g., 196)

        # Reshape projected features to sequence format (N, L_feat, D_dec)
        x_flat_proj = x_full_proj.permute(0, 2, 3, 1).reshape(N, L_feat, D_decoder)

        # --- Create the full sequence input for the decoder ---
        # Strategy: Upsample feature map to match L, then insert mask tokens.
        if L_feat != L:
             # Upsample spatially using nearest neighbor interpolation
             scale_factor = math.sqrt(L / L_feat)
             # Ensure scale_factor is integer if possible, or handle float scale factors
             # For L=196, L_feat=4, scale_factor=sqrt(49)=7
             x_upsampled = F.interpolate(x_full_proj, scale_factor=scale_factor, mode='nearest')
             # Reshape to sequence
             x_flat_upsampled = x_upsampled.permute(0, 2, 3, 1).reshape(N, L, D_decoder)
        else:
             # No upsampling needed
             x_flat_upsampled = x_flat_proj # This case should have L_feat = L

        # Create mask token tensor: (N, L, D_decoder)
        mask_bool = mask.bool() # (N, L), True where masked
        mask_tokens = self.mask_token_value.expand(N, L, -1) # (N, L, D_decoder)

        # Replace features with mask token where mask is True
        mask_bool_expanded = mask_bool.unsqueeze(-1) # (N, L, 1)
        x_decoder_input_flat = torch.where(mask_bool_expanded, mask_tokens, x_flat_upsampled)
        # Shape: (N, L, D_decoder)

        # --- Reshape for CNN Decoder ---
        # Calculate H, W corresponding to L (e.g., 14x14 for L=196)
        side_len = int(math.sqrt(L))
        if side_len * side_len != L:
             raise ValueError(f"Cannot reshape for CNN decoder: L={L} is not a perfect square.")
        H_dec, W_dec = side_len, side_len

        x_decoder_input_spatial = x_decoder_input_flat.reshape(N, H_dec, W_dec, D_decoder).permute(0, 3, 1, 2).contiguous()
        # Shape: (N, D_decoder, H_dec, W_dec)

        # --- Pass through CNN Decoder ---
        pred_patches = self.decoder(x_decoder_input_spatial) # Output: (N, L, patch_dim)

        return pred_patches

    # def forward_loss(self, imgs, pred, mask):
    #     """
    #     imgs: [N, 3, H, W]
    #     pred: [N, L, p*p*3]
    #     mask: [N, L], 0 is keep, 1 is remove
    #     """
    #     if len(pred.shape) == 4:
    #         n, c, _, _ = pred.shape
    #         pred = pred.reshape(n, c, -1)
    #         pred = torch.einsum('ncl->nlc', pred)

    #     target = self.patchify(imgs)

    #     if self.norm_pix_loss:
    #         mean = target.mean(dim=-1, keepdim=True)
    #         var = target.var(dim=-1, keepdim=True)
    #         target = (target - mean) / (var + 1.e-6)**.5

    #     loss = self.loss_func(pred, target)

    #     loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    #     return loss

    def forward(self, imgs: torch.Tensor, mask_ratio: Optional[float] = None, epoch: int = -1, return_intermediate: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass of the CNN-MAE model. Output matches ViT MAE format.
        """
        effective_mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio

        # --- Encoder + Mask Generation ---
        encoder_output = self.forward_encoder(imgs, effective_mask_ratio, return_intermediate)

        if return_intermediate:
            latent_full_proj, mask, ids_restore, intermediate_features = encoder_output
        else:
            latent_full_proj, mask, ids_restore = encoder_output
            intermediate_features = None
        # latent_full_proj shape: (N, D_dec, H_feat, W_feat)
        # mask shape: (N, L)
        # ids_restore shape: (N, L)

        # --- Decoder ---
        # Pass the full projected features and the mask to the decoder
        pred_patches = self.forward_decoder(latent_full_proj, mask, ids_restore) # (N, L, D_patch)

        # --- Target ---
        target_patches = self.patchify(imgs) # (N, L, D_patch)

        # --- Store for get_reconstructed_images ---
        self.last_mask = mask.clone().detach()
        self.last_pred_patches = pred_patches.clone().detach()
        self.last_target_patches = target_patches.clone().detach()

        loss, individual_losses = self.loss_func(imgs, pred_patches, mask, epoch)

        return loss, pred_patches, mask, individual_losses

    def no_weight_decay(self) -> Dict[str, Any]:
        """
        Get the names of parameters that should not use weight decay (e.g., biases, norms).
        Returns:
            Set of parameter names.
        """
        no_decay = set()
        for name, param in self.named_parameters():
            # Check for 1D parameters (biases, norm scales/shifts) or specific names
            if param.ndim <= 1 or 'bias' in name or 'bn' in name or 'norm' in name or 'mask_token_value' in name:
                no_decay.add(name)
        # print(f"No weight decay for: {no_decay}")
        return no_decay

    def get_encoder(self) -> CNNEncoder:
        """
        Extract the encoder part of the model for fine-tuning (e.g., segmentation).

        Returns:
            Encoder model (CNNEncoder instance).
        """
        return self.encoder


# --- Model Instantiation Functions --- (Keep these as they are)

def cnnmae_tiny(**kwargs):
    """MAE with CNN-Tiny backbone."""
    model_kwargs = {
        'patch_size': 4,
        'encoder_embed_dim': 64,
        'encoder_depths': [2, 2, 2], # ~Tiny depth
        'decoder_dim': 128,
        'decoder_depth': 2, # Example decoder depth
        'use_batchnorm': True,
    }
    model_kwargs.update(kwargs) # Override defaults
    model = MaskedAutoencoderCNN(**model_kwargs)
    return model

def cnnmae_small(**kwargs):
    """MAE with CNN-Small backbone."""
    model_kwargs = {
        'patch_size': 4,
        'encoder_embed_dim': 64, # Using 64 from config now
        'encoder_depths': [2, 2, 6], # ~Small depth
        'decoder_dim': 256, # Example decoder dim
        'decoder_depth': 3, # Example decoder depth
        'use_batchnorm': True,
    }
    model_kwargs.update(kwargs) # Override defaults
    # Ensure img_size is passed if not in kwargs
    if 'img_size' not in model_kwargs:
         model_kwargs['img_size'] = 56 # Default if not provided
    model = MaskedAutoencoderCNN(**model_kwargs)
    return model

def cnnmae_base(**kwargs):
    """MAE with CNN-Base backbone."""
    model_kwargs = {
        'patch_size': 4,
        'encoder_embed_dim': 128, # ~Base dim
        'encoder_depths': [2, 2, 18], # ~Base depth
        'decoder_dim': 512, # Example decoder dim
        'decoder_depth': 4, # Example decoder depth
        'use_batchnorm': True,
    }
    model_kwargs.update(kwargs) # Override defaults
    if 'img_size' not in model_kwargs:
         model_kwargs['img_size'] = 56 # Default if not provided
    model = MaskedAutoencoderCNN(**model_kwargs)
    return model
