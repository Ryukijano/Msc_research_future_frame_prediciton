"""Evaluation metrics for future frame prediction.

PSNR, SSIM, and LPIPS computation for comparing predicted frames to ground truth.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
import math


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Compute PSNR between predicted and target images.

    Args:
        pred: (B, C, H, W) predicted images (normalized to [-1, 1])
        target: (B, C, H, W) ground truth images (normalized to [-1, 1])
        data_range: Data range (2.0 for [-1, 1], 1.0 for [0, 1])

    Returns:
        psnr: (B,) PSNR values in dB
    """
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    psnr_val = 10 * torch.log10(data_range ** 2 / (mse + 1e-10))
    return psnr_val


def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """Create 2D Gaussian kernel for SSIM."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    return kernel


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Compute SSIM between predicted and target images.

    Args:
        pred: (B, C, H, W) predicted images
        target: (B, C, H, W) ground truth images
        window_size: Size of Gaussian window
        sigma: Standard deviation of Gaussian
        data_range: Data range

    Returns:
        ssim: (B,) SSIM values
    """
    C = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma).to(pred.device)
    kernel = kernel.expand(C, 1, window_size, window_size)

    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean(dim=[1, 2, 3])


class LPIPSMetric:
    """LPIPS perceptual similarity metric using VGG features.

    Lightweight implementation using torchvision VGG16.
    """

    def __init__(self, device: str = "cuda"):
        import torchvision.models as models

        self.device = device
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        vgg = vgg.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False

        # Use features from different layers
        self.layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        self.vgg = vgg

        # Normalization for VGG
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS between predicted and target images.

        Args:
            pred: (B, C, H, W) predicted images (normalized to [-1, 1])
            target: (B, C, H, W) ground truth images (normalized to [-1, 1])

        Returns:
            lpips: (B,) LPIPS values (lower = more similar)
        """
        # Denormalize from [-1, 1] to [0, 1], then renormalize for VGG
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        # Extract features
        def extract_features(x):
            features = []
            for i, layer in enumerate(self.vgg):
                x = layer(x)
                if i in self.layers:
                    features.append(x)
            return features

        pred_feats = extract_features(pred)
        target_feats = extract_features(target)

        # Compute L2 distance between features
        dist = torch.zeros(pred.shape[0], device=pred.device)
        for pf, tf in zip(pred_feats, target_feats):
            dist += F.mse_loss(pf, tf, reduction="none").mean(dim=[1, 2, 3])

        return dist / len(self.layers)


def evaluate_predictions(
    pred_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    lpips_fn: Optional[LPIPSMetric] = None,
    device: str = "cuda",
) -> dict:
    """Evaluate predicted frames against ground truth.

    Args:
        pred_frames: (B, T, C, H, W) predicted frames in [-1, 1]
        gt_frames: (B, T, C, H, W) ground truth frames in [-1, 1]
        lpips_fn: Optional LPIPS metric function
        device: Device for computation

    Returns:
        dict with per-timestep and average metrics
    """
    B, T, C, H, W = pred_frames.shape
    pred_flat = pred_frames.reshape(B * T, C, H, W).to(device)
    gt_flat = gt_frames.reshape(B * T, C, H, W).to(device)

    psnr_vals = psnr(pred_flat, gt_flat).reshape(B, T)
    ssim_vals = ssim(pred_flat, gt_flat).reshape(B, T)

    results = {
        "psnr_per_step": psnr_vals.mean(dim=0).cpu().numpy(),
        "ssim_per_step": ssim_vals.mean(dim=0).cpu().numpy(),
        "psnr_avg": psnr_vals.mean().item(),
        "ssim_avg": ssim_vals.mean().item(),
    }

    if lpips_fn is not None:
        lpips_vals = lpips_fn(pred_flat, gt_flat).reshape(B, T)
        results["lpips_per_step"] = lpips_vals.mean(dim=0).cpu().numpy()
        results["lpips_avg"] = lpips_vals.mean().item()

    return results
