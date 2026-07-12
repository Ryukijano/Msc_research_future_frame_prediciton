"""Inference and visualization script for DINO-Foresight.

Generates predicted future frames from a trained model and saves
side-by-side comparison GIFs/images with ground truth.

Usage:
    python -m dino_foresight.inference \
        --checkpoint ./outputs/dino_foresight/best_model.pth \
        --data_dir /path/to/jigsaws \
        --n_pred_steps 20
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import glob

from dino_foresight.encoders import DINOv2Encoder, VJEPA2Encoder
from dino_foresight.predictor import MaskedFeatureTransformer
from dino_foresight.decoder import ConvDecoder, PixelDecoder
from dino_foresight.metrics import psnr, ssim, LPIPSMetric


def parse_args():
    parser = argparse.ArgumentParser(description="DINO-Foresight Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./inference_results")
    parser.add_argument("--n_pred_steps", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=10, help="Number of test samples to visualize")
    return parser.parse_args()


def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize from ImageNet normalization to [0, 1]."""
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(x.device)
    return (x * std + mean).clamp(0, 1)


def save_comparison(pred, gt, path, n_pred_steps):
    """Save side-by-side comparison image.
    
    Args:
        pred: (T, C, H, W) predicted frames in [0, 1]
        gt: (T, C, H, W) ground truth frames in [0, 1]
        path: Output path
        n_pred_steps: Number of prediction steps
    """
    T, C, H, W = pred.shape
    # Create grid: top row = GT, bottom row = prediction
    gap = 4
    grid = Image.new("RGB", (T * (W + gap) + gap, 2 * (H + gap) + gap), (255, 255, 255))
    
    for t in range(T):
        gt_img = transforms.ToPILImage()(gt[t].cpu())
        pred_img = transforms.ToPILImage()(pred[t].cpu())
        grid.paste(gt_img, (gap + t * (W + gap), gap))
        grid.paste(pred_img, (gap + t * (W + gap), gap + H + gap))
    
    grid.save(path)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"Encoder: {ckpt_args.get('encoder', 'dinov2')}")

    # Rebuild encoder
    encoder_type = ckpt_args.get("encoder", "dinov2")
    img_size = ckpt_args.get("img_size", 224)
    
    if encoder_type == "dinov2":
        encoder = DINOv2Encoder(
            model_name=ckpt_args.get("encoder_model", "vitb14"),
            img_size=img_size,
            multi_layer=ckpt_args.get("multi_layer", True),
        ).to(device)
    else:
        encoder = VJEPA2Encoder(
            model_name=ckpt_args.get("encoder_model", "vjepa2_1_vit_base_384"),
            img_size=img_size,
        ).to(device)
    
    feat_dim = encoder.feat_dim
    n_patches = encoder.n_patches
    patch_size = encoder.patch_size

    # Rebuild predictor
    predictor = MaskedFeatureTransformer(
        feat_dim=feat_dim,
        hidden_dim=ckpt_args.get("hidden_dim", 768),
        num_layers=ckpt_args.get("num_layers", 8),
        num_heads=ckpt_args.get("num_heads", 8),
        n_patches=n_patches,
        n_context=ckpt_args.get("n_context", 4),
        n_future=ckpt_args.get("n_future", 1),
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    # Rebuild decoder
    decoder = None
    if "decoder" in ckpt:
        decoder_type = ckpt_args.get("decoder_type", "conv")
        if decoder_type == "conv":
            decoder = ConvDecoder(feat_dim, patch_size, img_size).to(device)
        else:
            decoder = PixelDecoder(feat_dim, patch_size, img_size).to(device)
        decoder.load_state_dict(ckpt["decoder"])
        decoder.eval()

    if decoder is None:
        print("WARNING: No decoder in checkpoint — cannot generate pixel predictions.")
        return

    # Load test data
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dir = Path(args.data_dir) / "test"
    video_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])[:args.n_samples]

    lpips_fn = LPIPSMetric(device=str(device))

    all_psnr = []
    all_ssim = []
    all_lpips = []

    for vidx, vdir in enumerate(tqdm(video_dirs, desc="Inference")):
        frame_paths = sorted(glob.glob(str(vdir / "*.png")))
        if len(frame_paths) < args.n_pred_steps + 4:
            continue

        # Load frames
        frames = [transform(Image.open(fp).convert("RGB")) for fp in frame_paths]
        clip = torch.stack(frames).unsqueeze(0).to(device)  # (1, T, C, H, W)

        n_context = predictor.n_context
        past_frames = clip[:, :n_context]
        gt_future = clip[:, n_context:n_context + args.n_pred_steps]

        # Extract context features
        B, T_c, C, H, W = past_frames.shape
        past_flat = past_frames.reshape(B * T_c, C, H, W)
        with torch.no_grad():
            context_feats = encoder(past_flat).reshape(B, T_c, n_patches, feat_dim)

        # Autoregressive prediction
        pred_feats_list = []
        current_context = context_feats
        for step in range(args.n_pred_steps):
            with torch.no_grad():
                pred = predictor(current_context)
            pred_feats_list.append(pred)
            current_context = torch.cat([current_context[:, 1:], pred], dim=1)

        pred_feats = torch.cat(pred_feats_list, dim=1)

        # Decode to pixels
        with torch.no_grad():
            pred_pixels = decoder(pred_feats)  # (1, T, C, H, W)

        # Compute metrics
        for step in range(min(args.n_pred_steps, gt_future.shape[1])):
            p = psnr(pred_pixels[:, step], gt_future[:, step]).item()
            s = ssim(pred_pixels[:, step], gt_future[:, step]).item()
            l = lpips_fn(pred_pixels[:, step], gt_future[:, step]).item()
            all_psnr.append(p)
            all_ssim.append(s)
            all_lpips.append(l)

        # Save visualization
        pred_vis = denormalize(pred_pixels[0])
        gt_vis = denormalize(gt_future[0])
        save_comparison(
            pred_vis[:gt_future.shape[1]],
            gt_vis,
            os.path.join(args.output_dir, f"sample_{vidx:03d}_{vdir.name}.png"),
            args.n_pred_steps,
        )

    # Print results
    print(f"\n=== Inference Results ({len(video_dirs)} samples, {args.n_pred_steps} steps) ===")
    print(f"PSNR:  {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f}")
    print(f"SSIM:  {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    print(f"LPIPS: {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}")
    print(f"\nVisualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
