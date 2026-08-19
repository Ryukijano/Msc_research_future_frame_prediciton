"""Past-only motion encoder for residual feature prediction.

The predictor models
    ẑ_{t+1} = z_t + f_motion(past)
so that a zero-initialized head is exactly Copy-Last at epoch 0.

Motion is computed from context only:
  feat: z_t - z_{t-1}  (PCA-DINOv2 tokens)
  rgb:  grayscale (x_t - x_{t-1}) pooled onto the patch grid

Never uses x_{t+1} or z_{t+1}. The last linear is zero-initialized.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PastMotionEncoder(nn.Module):
    """Encode past-only motion as hidden-dim tokens added to future queries."""

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        use_rgb: bool = False,
    ):
        super().__init__()
        self.use_rgb = use_rgb
        in_dim = feat_dim + (1 if use_rgb else 0)
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        context_feats: torch.Tensor,
        frames_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return motion tokens (B, P, hidden_dim).

        Args:
            context_feats: (B, T, P, D) past encoder features
            frames_ctx: optional (B, T, C, H, W) RGB context (past only)
        """
        B, T, P, D = context_feats.shape
        hidden = self.fc2.out_features
        if T < 2:
            return torch.zeros(
                B, P, hidden, device=context_feats.device, dtype=context_feats.dtype
            )

        feat_delta = context_feats[:, -1] - context_feats[:, -2]
        parts = [feat_delta]
        if self.use_rgb:
            parts.append(_rgb_delta_tokens(frames_ctx, P, feat_delta))
        h = torch.cat(parts, dim=-1)
        h = self.norm(h)
        h = self.drop(F.gelu(self.fc1(h)))
        return self.fc2(h)


def _rgb_delta_tokens(
    frames_ctx: Optional[torch.Tensor],
    n_patches: int,
    feat_delta: torch.Tensor,
) -> torch.Tensor:
    """Pool grayscale RGB delta onto the patch grid: (B, P, 1)."""
    B, P, _ = feat_delta.shape
    zeros = torch.zeros(B, P, 1, device=feat_delta.device, dtype=feat_delta.dtype)
    if frames_ctx is None or frames_ctx.shape[1] < 2:
        return zeros

    rgb_delta = frames_ctx[:, -1] - frames_ctx[:, -2]
    gray = rgb_delta.mean(dim=1, keepdim=True)
    grid = int(math.sqrt(n_patches))
    pooled = F.adaptive_avg_pool2d(gray, (grid, grid))
    tokens = pooled.flatten(2).transpose(1, 2)
    if tokens.shape[1] != n_patches:
        tokens = F.interpolate(
            tokens.transpose(1, 2), size=n_patches, mode="nearest"
        ).transpose(1, 2)
    return tokens.to(dtype=feat_delta.dtype)
