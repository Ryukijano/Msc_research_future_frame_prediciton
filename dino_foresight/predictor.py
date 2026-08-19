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
from typing import Optional

from dino_foresight.motion import PastMotionEncoder


class TemporalAttention(nn.Module):
    """Multi-head self-attention across the temporal dimension.

    Each spatial position attends to the same spatial position across all frames.
    Uses causal masking so frame t can only attend to frames <= t (no future leakage).
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

        # Causal mask: frame i can only attend to frames 0..i (lower triangular)
        # This prevents context frames from seeing future/mask tokens
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )  # True = mask out (above diagonal)
        attn = attn.masked_fill(causal_mask, float('-inf'))

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
        residual: If True, predict delta on last-frame features (Copy-Last at zero init)
        use_motion: If True, add past-only motion tokens to future queries
        motion_from: "feat", "rgb", or "both"
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
        dropout: float = 0.2,
        residual: bool = True,
        use_motion: bool = False,
        motion_from: str = "feat",
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.n_patches = n_patches
        self.n_context = n_context
        self.n_future = n_future
        self.n_total = n_context + n_future
        self.residual = residual
        self.use_motion = use_motion
        self.motion_from = motion_from

        # Token projection
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, feat_dim)
        if residual:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

        # Learnable mask token for future frames (used when residual=False)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, hidden_dim) * 0.02)

        self.motion_encoder = None
        if use_motion:
            self.motion_encoder = PastMotionEncoder(
                feat_dim=feat_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_rgb=motion_from in ("rgb", "both"),
            )

        # Position embeddings — allocate enough slots for teacher forcing + autoregressive
        # During training: context = n_context + n_future - 1, predict 1 => total = n_context + n_future
        # During AR eval: context = n_context, predict n_steps => total = n_context + n_steps
        # Allocate generously to avoid slicing errors
        max_seq_len = n_context + max(n_future, 20)
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, n_patches, hidden_dim) * 0.02
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, 1, hidden_dim) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        context_feats: torch.Tensor,
        n_predict: int = 1,
        frames_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict future frame features from context features.

        Args:
            context_feats: (B, N_context, N_patches, D) frozen encoder features for past frames
            n_predict: Number of future frames to predict (default 1 for single-step training)
            frames_ctx: optional (B, T_c, C, H, W) past RGB, used only for rgb/both motion

        Returns:
            predicted_future_feats: (B, n_predict, N_patches, D) predicted future features
        """
        B, T_c, N, D = context_feats.shape

        # Project to hidden dim
        context_tokens = self.input_proj(context_feats)  # (B, T_c, N, hidden)

        if self.residual:
            # DINO-Foresight half_half_previous: future queries start as last-frame tokens
            last_tokens = context_tokens[:, -1:, :, :]
            future_queries = last_tokens.expand(-1, n_predict, -1, -1).contiguous()
            if self.motion_encoder is not None:
                motion = self.motion_encoder(context_feats, frames_ctx)
                future_queries = future_queries + motion.unsqueeze(1)
        else:
            future_queries = self.mask_token.expand(B, n_predict, N, -1)

        x = torch.cat([context_tokens, future_queries], dim=1)

        # Add position embeddings (handle variable n_predict by slicing)
        x = x + self.spatial_pos_embed + self.temporal_pos_embed[:, :T_c + n_predict, :, :]

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Project back to feature dim
        x = self.output_proj(x)  # (B, T_c + n_predict, N, D)

        # Return only future frame predictions
        pred_future = x[:, T_c:, ...]  # (B, n_predict, N, D)

        # TDV-style residual: ẑ_{t+1} = z_t + delta. Zero-init output_proj => Copy-Last.
        if self.residual:
            pred_future = pred_future + context_feats[:, -1:, ...].detach()

        return pred_future  # (B, n_predict, N, D)

    def forward_autoregressive(
        self,
        context_feats: torch.Tensor,
        n_steps: int,
        frames_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressively predict multiple future steps.

        Args:
            context_feats: (B, N_context, N_patches, D) frozen encoder features
            n_steps: Number of future steps to predict
            frames_ctx: optional past RGB; used only on the first step

        Returns:
            all_preds: (B, n_steps, N_patches, D) predicted features for all future steps
        """
        all_preds = []

        current_context = context_feats
        current_frames = frames_ctx
        for _ in range(n_steps):
            pred = self.forward(
                current_context, n_predict=1, frames_ctx=current_frames
            )
            pred_step = pred[:, :1, ...]
            all_preds.append(pred_step)
            current_context = torch.cat([current_context[:, 1:, ...], pred_step], dim=1)
            current_frames = None

        return torch.cat(all_preds, dim=1)


def build_predictor(feat_dim: int, n_patches: int, cfg: dict, **overrides) -> MaskedFeatureTransformer:
    """Build a predictor from a train-arg dict or checkpoint args."""
    kwargs = dict(
        feat_dim=feat_dim,
        hidden_dim=int(cfg.get("hidden_dim", 768)),
        num_layers=int(cfg.get("num_layers", 8)),
        num_heads=int(cfg.get("num_heads", 8)),
        n_patches=n_patches,
        n_context=int(cfg.get("n_context", 4)),
        n_future=int(cfg.get("n_future", 1)),
        dropout=float(cfg.get("dropout", 0.2)),
        residual=bool(cfg.get("residual", False)),
        use_motion=bool(cfg.get("use_motion", False)),
        motion_from=str(cfg.get("motion_from", "feat")),
    )
    kwargs.update(overrides)
    return MaskedFeatureTransformer(**kwargs)

