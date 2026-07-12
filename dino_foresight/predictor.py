"""Masked Feature Transformer for future frame prediction in latent space.

Following DINO-Foresight (NeurIPS 2025):
- Decomposed temporal + spatial attention (not full self-attention)
- Future frame tokens are masked during training, predicted from context
- SmoothL1 loss on masked positions only

Architecture:
    Input: (B, N_total, N_patches, D)  -- features for all frames
    1. Project tokens to hidden dim
    2. Add spatial + temporal position embeddings
    3. Mask future frame tokens with learnable [MASK] vector
    4. L layers of: Temporal MSA -> Spatial MSA -> MLP
    5. Project back to feature dim
    Output: (B, N_future, N_patches, D) -- predicted future features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TemporalAttention(nn.Module):
    """Multi-head self-attention across the temporal dimension.

    Each spatial position attends to the same spatial position across all frames.
    Input: (B, N_frames, N_patches, D)
    Output: (B, N_frames, N_patches, D)
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        # Reshape: (B*N, T, D) -- treat each spatial position independently
        x = x.reshape(B * N, T, D)

        qkv = self.qkv(x).reshape(B * N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*N, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*N, heads, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B * N, T, D)
        out = self.proj(out)
        out = self.dropout(out)

        return out.reshape(B, T, N, D)


class SpatialAttention(nn.Module):
    """Multi-head self-attention within each frame (spatial dimension).

    Each frame's patches attend to each other independently.
    Input: (B, N_frames, N_patches, D)
    Output: (B, N_frames, N_patches, D)
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        # Reshape: (B*T, N, D) -- treat each frame independently
        x = x.reshape(B * T, N, D)

        qkv = self.qkv(x).reshape(B * T, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*T, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*T, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B * T, N, D)
        out = self.proj(out)
        out = self.dropout(out)

        return out.reshape(B, T, N, D)


class TransformerLayer(nn.Module):
    """One layer of the masked feature transformer: Temporal -> Spatial -> MLP."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.spatial_attn = SpatialAttention(dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.temporal_attn(self.norm1(x))
        x = x + self.spatial_attn(self.norm2(x))
        x = x + self.mlp(self.norm3(x))
        return x


class MaskedFeatureTransformer(nn.Module):
    """Predicts future frame features from past frame features.

    Args:
        feat_dim: Dimension of input features from frozen encoder
        hidden_dim: Hidden dimension of the transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        n_patches: Number of spatial patches per frame
        n_context: Number of context (past) frames
        n_future: Number of future frames to predict
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 8,
        n_patches: int = 256,
        n_context: int = 4,
        n_future: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.n_patches = n_patches
        self.n_context = n_context
        self.n_future = n_future
        self.n_total = n_context + n_future

        # Token projection
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, feat_dim)

        # Learnable mask token for future frames
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, hidden_dim) * 0.02)

        # Position embeddings
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, n_patches, hidden_dim) * 0.02
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, self.n_total, 1, hidden_dim) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, context_feats: torch.Tensor) -> torch.Tensor:
        """Predict future frame features from context features.

        Args:
            context_feats: (B, N_context, N_patches, D) frozen encoder features for past frames

        Returns:
            predicted_future_feats: (B, N_future, N_patches, D) predicted future features
        """
        B, T_c, N, D = context_feats.shape

        # Project to hidden dim
        context_tokens = self.input_proj(context_feats)  # (B, T_c, N, hidden)

        # Create mask tokens for future frames
        mask_tokens = self.mask_token.expand(B, self.n_future, N, -1)  # (B, T_f, N, hidden)

        # Concatenate context + mask tokens
        x = torch.cat([context_tokens, mask_tokens], dim=1)  # (B, T_total, N, hidden)

        # Add position embeddings
        x = x + self.spatial_pos_embed + self.temporal_pos_embed

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Project back to feature dim
        x = self.output_proj(x)  # (B, T_total, N, D)

        # Return only future frame predictions
        return x[:, T_c:, ...]  # (B, T_future, N, D)

    def forward_autoregressive(self, context_feats: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Autoregressively predict multiple future steps.

        Args:
            context_feats: (B, N_context, N_patches, D) frozen encoder features
            n_steps: Number of future steps to predict

        Returns:
            all_preds: (B, n_steps, N_patches, D) predicted features for all future steps
        """
        B, T_c, N, D = context_feats.shape
        all_preds = []

        current_context = context_feats
        for step in range(n_steps):
            pred = self.forward(current_context)  # (B, 1, N, D)
            all_preds.append(pred)

            # Append prediction to context, keep last n_context frames
            current_context = torch.cat([current_context[:, 1:, ...], pred], dim=1)

        return torch.cat(all_preds, dim=1)  # (B, n_steps, N, D)
