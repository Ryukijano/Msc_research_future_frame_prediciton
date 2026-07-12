"""Frozen foundation encoder wrappers for surgical video prediction.

Supports DINOv2 and V-JEPA 2.1 as frozen feature extractors.
Both produce spatially-structured patch tokens suitable for future feature prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 encoder producing spatial patch features.

    Args:
        model_name: DINOv2 variant (vitb14, vitl14, viTs14)
        img_size: Input image size (square)
        multi_layer: If True, extract and concat features from multiple layers
        pca_dim: If set, apply PCA projection to reduce feature dimension
    """

    def __init__(
        self,
        model_name: str = "vitb14",
        img_size: int = 224,
        multi_layer: bool = True,
        pca_dim: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.multi_layer = multi_layer
        self.pca_dim = pca_dim

        # Ensure model_name has the dinov2_ prefix
        if not model_name.startswith("dinov2_"):
            model_name = f"dinov2_{model_name}"
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.patch_size = self.model.patch_size
        self.n_patches = (img_size // self.patch_size) ** 2
        self.feat_dim = self.model.embed_dim

        if multi_layer:
            self.layer_indices = [3, 6, 9, 11]
            self.feat_dim = self.model.embed_dim * len(self.layer_indices)
        else:
            self.layer_indices = [11]

        self.pca_proj = None
        if pca_dim is not None:
            self.pca_proj = nn.Linear(self.feat_dim, pca_dim, bias=False)
            self.pca_proj.weight.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial patch features.

        Args:
            x: (B, C, H, W) input images

        Returns:
            features: (B, N_patches, D) spatial patch tokens
        """
        B = x.shape[0]

        if self.multi_layer:
            features = self.model.get_intermediate_layers(
                x, n=self.layer_indices, reshape=True
            )
            tokens = []
            for feat in features:
                B_, D, H, W = feat.shape
                tokens.append(feat.reshape(B_, D, H * W).permute(0, 2, 1))
            features = torch.cat(tokens, dim=-1)
        else:
            features = self.model.forward_features(x)
            features = features[:, 1:, :]

        if self.pca_proj is not None:
            features = self.pca_proj(features)

        return features

    @torch.no_grad()
    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features as spatial maps (for PCA visualization).

        Args:
            x: (B, C, H, W) input images

        Returns:
            features: (B, D, H', W') spatial feature maps
        """
        if self.multi_layer:
            features = self.model.get_intermediate_layers(
                x, n=self.layer_indices, reshape=True
            )
            return torch.cat(features, dim=1)  # (B, D*L, H', W')
        else:
            features = self.model.get_intermediate_layers(x, n=[11], reshape=True)
            return features[0]

    def fit_pca(self, dataloader, device, n_samples=1000):
        """Fit PCA projection on a subset of the data."""
        if self.pca_dim is None:
            return

        all_feats = []
        count = 0
        for batch in dataloader:
            frames = batch[0] if isinstance(batch, (list, tuple)) else batch
            frames = frames.to(device)
            B, T, C, H, W = frames.shape
            frames = frames.reshape(B * T, C, H, W)
            feats = self.forward(frames)
            all_feats.append(feats)
            count += feats.shape[0]
            if count >= n_samples:
                break

        all_feats = torch.cat(all_feats, dim=0)
        all_feats = all_feats.reshape(-1, all_feats.shape[-1])
        mean = all_feats.mean(dim=0, keepdim=True)
        centered = all_feats - mean
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        components = Vh[: self.pca_dim, :]
        self.pca_proj.weight.data = components.to(self.pca_proj.weight.device)


class VJEPA2Encoder(nn.Module):
    """Frozen V-JEPA 2.1 encoder producing spatial patch features.

    V-JEPA 2.1 is a video encoder — for single-frame extraction, we duplicate
    the image to 16 frames and average over the temporal dimension.

    Args:
        model_name: V-JEPA 2.1 variant (vjepa2_1_vit_base_384, vjepa2_1_vit_large_384)
        img_size: Input image size (must be 384 for V-JEPA 2.1)
        n_frames: Number of duplicated frames for single-image mode
    """

    def __init__(
        self,
        model_name: str = "vjepa2_1_vit_base_384",
        img_size: int = 384,
        n_frames: int = 16,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.n_frames = n_frames

        self.model = torch.hub.load("facebookresearch/vjepa2", model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.patch_size = 16
        self.n_patches = (img_size // self.patch_size) ** 2
        self.feat_dim = self.model.embed_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial patch features by duplicating image to video.

        Args:
            x: (B, C, H, W) input images

        Returns:
            features: (B, N_patches, D) spatial patch tokens
        """
        B, C, H, W = x.shape
        x_video = x.unsqueeze(1).expand(B, self.n_frames, C, H, W)
        x_video = x_video.reshape(B, self.n_frames, C, H, W)

        feats = self.model(x_video)
        features = feats.mean(dim=1)

        return features

    @torch.no_grad()
    def forward_video(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-frame features from a video clip.

        Args:
            x: (B, T, C, H, W) input video

        Returns:
            features: (B, T, N_patches, D) per-frame patch tokens
        """
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        x_dup = x_flat.unsqueeze(1).expand(B * T, self.n_frames, C, H, W)
        x_dup = x_dup.reshape(B * T, self.n_frames, C, H, W)

        feats = self.model(x_dup)
        feats = feats.mean(dim=1)
        feats = feats.reshape(B, T, feats.shape[1], feats.shape[2])

        return feats


def build_encoder(encoder_type: str = "dinov2", **kwargs) -> nn.Module:
    """Build a frozen encoder by type."""
    if encoder_type == "dinov2":
        return DINOv2Encoder(**kwargs)
    elif encoder_type == "vjepa2":
        return VJEPA2Encoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
