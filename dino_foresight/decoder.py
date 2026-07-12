"""Lightweight pixel decoder for visualizing predicted latent features.

Takes predicted frozen encoder features and reconstructs pixel images.
This is only needed for qualitative evaluation (PSNR/SSIM/LPIPS).
The core prediction happens entirely in latent space.

Architecture: Small ViT-based decoder that maps patch tokens -> pixels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PixelDecoder(nn.Module):
    """ViT-based decoder: patch features -> pixel images.

    Uses a lightweight transformer + pixel shuffle upsampling.

    Args:
        feat_dim: Dimension of input features
        patch_size: Patch size of the encoder (14 for DINOv2, 16 for V-JEPA)
        img_size: Target image size
        in_channels: Number of image channels (3 for RGB)
        hidden_dim: Hidden dimension of decoder transformer
        num_layers: Number of decoder layers
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        feat_dim: int,
        patch_size: int = 14,
        img_size: int = 224,
        in_channels: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.n_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        # Project features to hidden dim
        self.input_proj = nn.Linear(feat_dim, hidden_dim)

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, hidden_dim) * 0.02
        )

        # Decoder transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Patch to pixels: each patch -> patch_size^2 * in_channels pixels
        self.patch_to_pixels = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Decode patch features to pixel images.

        Args:
            feats: (B, T, N_patches, D) patch features

        Returns:
            images: (B, T, C, H, W) reconstructed images in [-1, 1]
        """
        B, T, N, D = feats.shape
        x = feats.reshape(B * T, N, D)

        # Project + position embedding
        x = self.input_proj(x) + self.pos_embed

        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # Patch to pixels
        pixels = self.patch_to_pixels(x)  # (B*T, N, patch_size^2 * C)
        pixels = pixels.reshape(
            B * T, self.grid_size, self.grid_size,
            self.in_channels, self.patch_size, self.patch_size
        )
        # Rearrange to (B*T, C, H, W)
        pixels = pixels.permute(0, 3, 1, 4, 2, 5).contiguous()
        pixels = pixels.reshape(B * T, self.in_channels, self.img_size, self.img_size)

        return pixels.reshape(B, T, self.in_channels, self.img_size, self.img_size)


class ConvDecoder(nn.Module):
    """Convolutional decoder: patch features -> pixel images.

    Simpler and faster than ViT decoder. Reshapes patch tokens to a
    spatial grid, then uses convTranspose2d to upsample.

    Args:
        feat_dim: Dimension of input features
        patch_size: Patch size of the encoder
        img_size: Target image size
        in_channels: Number of image channels
    """

    def __init__(
        self,
        feat_dim: int,
        patch_size: int = 14,
        img_size: int = 224,
        in_channels: int = 3,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.grid_size = img_size // patch_size

        # Project features to a higher dim for conv decoding
        hidden = 256
        self.proj = nn.Linear(feat_dim, hidden)

        # Conv decoder: (hidden, grid, grid) -> (3, img_size, img_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 2x upsample
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 2x upsample
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, in_channels, 3, padding=1),
            nn.Tanh(),
        )

        # Calculate how much we need to resize after conv decoder
        self.upscale_factor = img_size / (self.grid_size * 4)
        self.use_interp = abs(self.upscale_factor - int(self.upscale_factor)) > 0.01

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Decode patch features to pixel images.

        Args:
            feats: (B, T, N_patches, D) patch features

        Returns:
            images: (B, T, C, H, W) reconstructed images in [-1, 1]
        """
        B, T, N, D = feats.shape
        x = self.proj(feats)  # (B, T, N, hidden)
        x = x.reshape(B * T, self.grid_size, self.grid_size, -1)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B*T, hidden, grid, grid)

        x = self.decoder(x)  # (B*T, C, grid*4, grid*4)

        if self.use_interp:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        return x.reshape(B, T, self.in_channels, self.img_size, self.img_size)
