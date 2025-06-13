"""
Segmentation model using CNNEncoder from cnn_mae.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .cnn import CNNBlock
from .cnn_mae import CNNEncoder

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True):
        super().__init__()
        # Upsampling: choose one - ConvTranspose2d or Upsample + Conv
        # Using Upsample + Conv is often preferred to avoid checkerboard artifacts
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Channels after upsampling is still in_channels. After concat with skip: in_channels + skip_channels
        self.convs = CNNBlock(in_channels + skip_channels, out_channels, kernel_size=3, stride=1, use_batchnorm=use_batchnorm)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        # Pad if spatial dimensions are slightly off due to convolutions
        # This can happen if padding wasn't "same" everywhere or if input size isn't perfectly divisible.
        # For now, assuming dimensions will match or skip_connection is larger/equal and we can crop skip.
        if x.shape[2:] < skip_connection.shape[2:]:
            # A simple cropping strategy if skip is larger. More robust handling might be needed.
            diffY = skip_connection.size()[2] - x.size()[2]
            diffX = skip_connection.size()[3] - x.size()[3]
            skip_connection = skip_connection[:, :, diffY // 2 : skip_connection.size()[2] - (diffY - diffY // 2),
                                                 diffX // 2 : skip_connection.size()[3] - (diffX - diffX // 2)]
        elif x.shape[2:] > skip_connection.shape[2:]:
            # If x is larger, interpolate skip_connection to match x size
            skip_connection = F.interpolate(skip_connection, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip_connection], dim=1)
        x = self.convs(x)
        return x

class CNNForSegmentation(nn.Module):
    def __init__(self, img_size=56, patch_size=4, in_chans=3,
                 encoder_embed_dim=64, encoder_depths=[2, 2, 6], encoder_dims=None,
                 use_batchnorm=True, num_classes=1, pretrained_path=None):
        super().__init__()

        self.num_classes = num_classes
        self.encoder = CNNEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depths=encoder_depths,
            dims=encoder_dims, # Output channels for each stage of encoder
            use_batchnorm=use_batchnorm
        )

        # --- U-Net Decoder ---
        # Encoder dims: [stem_dim, stage1_dim, stage2_dim, final_encoder_stage_dim]
        # Example: if encoder_dims = [64, 128, 256, 512] for a 3-stage encoder (+stem)
        # Skip connections will be from stem_dim, stage1_dim, stage2_dim
        # The input to the first decoder block is final_encoder_stage_dim (512)

        self.decoder_stages = nn.ModuleList()
        
        # Encoder's intermediate_dims are [stem_dim, stage1_dim, ...]
        # encoder.dims gives all output dims including the final one.
        # Let's use encoder.dims: dims = [dim_stem, dim_stage0, dim_stage1, dim_stage2_output] for depths=[d0,d1,d2]
        # Skip connections are encoder.dims[:-1] in reverse order.
        # Input to first decoder block is encoder.dims[-1]
        
        # Last encoder output channels
        current_decoder_in_channels = self.encoder.dims[-1]
        
        # Iterate through encoder stages in reverse to build decoder
        # (from deepest skip connection up to the stem's output)
        num_encoder_skip_stages = len(self.encoder.dims) - 1 # Number of skip connections available

        for i in range(num_encoder_skip_stages):
            skip_connection_channels = self.encoder.dims[-(i + 2)] # Access skip dims in reverse: encoder.dims[-2], encoder.dims[-3], ...
            
            # Define out_channels for this decoder block.
            # A common strategy is to halve the channels, e.g., make it same as skip_connection_channels or a predefined list
            decoder_block_out_channels = skip_connection_channels # Example: try to match the skip dim
            
            self.decoder_stages.append(
                DecoderBlock(current_decoder_in_channels, skip_connection_channels, decoder_block_out_channels, use_batchnorm=use_batchnorm)
            )
            current_decoder_in_channels = decoder_block_out_channels

        # Final convolution to get to num_classes
        # The input channels to this layer is the output of the last decoder block
        # (which processed the stem's skip connection)
        self.final_conv = nn.Conv2d(current_decoder_in_channels, num_classes, kernel_size=1)

        if pretrained_path:
            self.load_pretrained_encoder(pretrained_path)

    def load_pretrained_encoder(self, pretrained_path):
        try:
            # Assuming the MAE checkpoint is from MaskedAutoencoderCNN
            # and it saves the entire model's state_dict.
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            print(f"Loading pretrained MAE checkpoint from: {pretrained_path}")

            mae_state_dict = checkpoint.get('model_state_dict', checkpoint) # Handle checkpoints that might just be the state_dict

            encoder_state_dict = {}
            for k, v in mae_state_dict.items():
                if k.startswith('encoder.'):
                    encoder_state_dict[k.replace('encoder.', '', 1)] = v # replace only the first occurrence
            
            if encoder_state_dict:
                msg = self.encoder.load_state_dict(encoder_state_dict, strict=False)
                print(f"Encoder weights loaded with message: {msg}")
                if msg.missing_keys:
                    print(f"Missing keys in encoder during load: {msg.missing_keys}")
                if msg.unexpected_keys:
                    print(f"Unexpected keys in encoder during load: {msg.unexpected_keys}")
            else:
                print("ERROR: No 'encoder.*' weights found in the checkpoint.")

        except Exception as e:
            print(f"Error loading pretrained encoder weights: {e}. Training from scratch.")

    def forward(self, x):
        # Encoder pass, requesting intermediate features for skip connections
        encoder_final_output, skip_connections = self.encoder(x, return_intermediate=True)
        
        # Decoder pass
        current_features = skip_connections[-1] # Ignore batch norm output, start with the last encoder output
        
        # Iterate through decoder stages and corresponding skip connections (in reverse order of how they were captured)
        # skip_connections = [stem_feat, stage0_feat, stage1_feat]
        # We need to feed them in reverse: stage1_feat, then stage0_feat, then stem_feat
        for i in range(len(self.decoder_stages)):
            skip = skip_connections[-(i + 2)] # Skip last skip
            current_features = self.decoder_stages[i](current_features, skip)
            
        logits = self.final_conv(current_features) # Output will be (N, num_classes, H, W)
        
        return logits

# Factory function for this segmentation model
def cnnseg_small(pretrained_path=None, img_size=56, num_classes=1, **kwargs):
    """
    CNN Segmentation model with a 'small' encoder configuration.
    """
    model_kwargs = {
        'img_size': img_size,
        'patch_size': 4, # As used in CNNEncoder stem for initial patch-like conv
        'encoder_embed_dim': 64, # Base dim for CNN encoder
        'encoder_depths': kwargs.get('encoder_depths', [2, 2, 6]), # Default small config
        'encoder_dims': kwargs.get('encoder_dims', [64, 128, 256, 512]), # Example dims matching typical progression
        'use_batchnorm': kwargs.get('use_batchnorm', True),
        'num_classes': num_classes,
        'pretrained_path': pretrained_path
    }
    # Allow overrides from kwargs
    model_kwargs.update(kwargs)

    # Adjust encoder_dims if only embed_dim and depths are given (as in original CNNEncoder)
    if 'encoder_dims' not in kwargs and 'encoder_embed_dim' in model_kwargs and 'encoder_depths' in model_kwargs:
        num_stages = len(model_kwargs['encoder_depths'])
        model_kwargs['encoder_dims'] = [model_kwargs['encoder_embed_dim'] * (2**i) for i in range(num_stages + 1)]

    return CNNForSegmentation(**model_kwargs)
