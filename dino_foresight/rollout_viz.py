#!/usr/bin/env python3
"""RGB rollout visualizations: GT vs RGB Copy-Last vs Track A2 (decoded).

Feature-space wins do not guarantee visible RGB gains — this script decodes
predicted DINOv2-PCA features with the Phase 1 RAE decoder and saves
side-by-side panels + GIFs on LOUO test clips.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dino_foresight.data import build_dataloaders
from dino_foresight.decoder import RAEDecoder
from dino_foresight.encoders import DINOv2Encoder
from dino_foresight.metrics import LPIPSMetric, psnr, ssim
from dino_foresight.predictor import build_predictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictor_ckpt", type=str, required=True)
    p.add_argument("--decoder_ckpt", type=str, required=True)
    p.add_argument("--pca_ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--n_context", type=int, default=4)
    p.add_argument("--n_pred_steps", type=int, default=10)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--max_clips_metrics", type=int, default=0,
                   help="If >0, cap clips for full-set RGB metrics")
    return p.parse_args()


def denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_gif(frames: list, path: Path, duration_ms: int = 200) -> None:
  frames[0].save(
      path,
      save_all=True,
      append_images=frames[1:],
      duration=duration_ms,
      loop=0,
  )


@torch.no_grad()
def rollout_one(
    predictor,
    decoder,
    encoder,
    past_frames: torch.Tensor,
    future_frames: torch.Tensor,
    n_steps: int,
    device: torch.device,
):
    """Returns dict of pixel tensors (T, C, H, W) on CPU in [0,1]."""
    B, T_c, C, H, W = past_frames.shape
    past = past_frames.to(device)
    gt = future_frames[:, :n_steps].to(device)

    flat = past.reshape(B * T_c, C, H, W)
    ctx = encoder(flat).reshape(B, T_c, encoder.n_patches, encoder.feat_dim)

    pred_feats = predictor.forward_autoregressive(ctx, n_steps=n_steps)
    pred_px = torch.clamp(decoder(pred_feats.float()), -1, 1)
    pred_px = denorm(pred_px[0])

    rgb_copy = past[0, -1:].expand(n_steps, -1, -1, -1)
    rgb_copy = denorm(rgb_copy)

    gt_vis = denorm(gt[0])
    past_vis = denorm(past[0])

    return {
        "past": past_vis.cpu(),
        "gt": gt_vis.cpu(),
        "rgb_copy": rgb_copy.cpu(),
        "pred": pred_px.cpu(),
    }


def save_panel(rollout: dict, path: Path, title: str, n_context: int) -> None:
    past = rollout["past"]
    gt = rollout["gt"]
    copy = rollout["rgb_copy"]
    pred = rollout["pred"]
    n_steps = gt.shape[0]
    n_cols = max(n_context, n_steps)

    fig, axes = plt.subplots(4, n_cols, figsize=(2.2 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(4, 1)

    row_titles = ["Past", "Ground truth", "RGB Copy-Last", "Track A2 decoded"]
    for c in range(n_cols):
        for r in range(4):
            axes[r, c].axis("off")
        if c < n_context:
            axes[0, c].imshow(past[c].permute(1, 2, 0).numpy())
            axes[0, c].set_title(f"t-{n_context - c}", fontsize=8)
        if c < n_steps:
            axes[1, c].imshow(gt[c].permute(1, 2, 0).numpy())
            axes[1, c].set_title(f"GT +{c + 1}", fontsize=8)
            axes[2, c].imshow(copy[c].permute(1, 2, 0).numpy())
            axes[2, c].set_title(f"Copy +{c + 1}", fontsize=8)
            axes[3, c].imshow(pred[c].permute(1, 2, 0).numpy())
            axes[3, c].set_title(f"A2 +{c + 1}", fontsize=8)

    for r, label in enumerate(row_titles):
        axes[r, 0].set_ylabel(label, fontsize=10, fontweight="bold")

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.predictor_ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    encoder = DINOv2Encoder(
        model_name=ckpt_args.get("encoder_model", "vitb14_reg"),
        img_size=args.img_size,
        multi_layer=True,
        pca_dim=int(ckpt_args.get("pca_dim", 1152)),
    ).to(device)
    encoder.load_pca(args.pca_ckpt, device)
    encoder.eval()

    predictor = build_predictor(
        feat_dim=encoder.feat_dim,
        n_patches=encoder.n_patches,
        cfg=ckpt_args,
        n_context=args.n_context,
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    dec_ckpt = torch.load(args.decoder_ckpt, map_location=device, weights_only=False)
    dec_args = dec_ckpt.get("args", {})
    decoder = RAEDecoder(
        encoder.feat_dim,
        encoder.patch_size,
        args.img_size,
        hidden_dim=int(dec_args.get("decoder_hidden_dim", 1152)),
        num_layers=int(dec_args.get("decoder_layers", 8)),
        num_heads=int(dec_args.get("decoder_heads", 12)),
    ).to(device)
    decoder.load_state_dict(dec_ckpt.get("decoder", dec_ckpt))
    decoder.eval()

    _, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        n_context=args.n_context,
        n_future=args.n_pred_steps,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=2,
        stride=args.stride,
        data_format="bair",
        rebuild_split=False,
        val_max_clips=0,
        seed=42,
    )
    loader = test_loader if args.split == "test" else val_loader

    lpips_fn = LPIPSMetric(device=str(device))
    sample_dir = out / "samples"
    gif_dir = out / "gifs"
    sample_dir.mkdir(exist_ok=True)
    gif_dir.mkdir(exist_ok=True)

    metrics = {
        "copy_psnr": [], "copy_ssim": [], "copy_lpips": [],
        "pred_psnr": [], "pred_ssim": [], "pred_lpips": [],
        "pred_psnr_per_step": [[] for _ in range(args.n_pred_steps)],
        "copy_psnr_per_step": [[] for _ in range(args.n_pred_steps)],
    }

    n_saved = 0
    clips = 0
    pbar = tqdm(loader, desc="rollout viz")
    for batch_idx, (past_frames, future_frames) in enumerate(pbar):
        if args.max_clips_metrics > 0 and clips >= args.max_clips_metrics:
            if n_saved >= args.n_samples:
                break
        if future_frames.shape[1] < args.n_pred_steps:
            continue

        rollout = rollout_one(
            predictor, decoder, encoder,
            past_frames, future_frames, args.n_pred_steps, device,
        )

        for t in range(args.n_pred_steps):
            gt_t = rollout["gt"][t].unsqueeze(0).to(device)
            cp_t = rollout["rgb_copy"][t].unsqueeze(0).to(device)
            pr_t = rollout["pred"][t].unsqueeze(0).to(device)
            metrics["copy_psnr"].append(psnr(cp_t, gt_t).item())
            metrics["copy_ssim"].append(ssim(cp_t, gt_t).item())
            metrics["copy_lpips"].append(lpips_fn(cp_t, gt_t).item())
            metrics["pred_psnr"].append(psnr(pr_t, gt_t).item())
            metrics["pred_ssim"].append(ssim(pr_t, gt_t).item())
            metrics["pred_lpips"].append(lpips_fn(pr_t, gt_t).item())
            metrics["copy_psnr_per_step"][t].append(metrics["copy_psnr"][-1])
            metrics["pred_psnr_per_step"][t].append(metrics["pred_psnr"][-1])

        if n_saved < args.n_samples:
            name = f"sample_{n_saved:02d}_batch{batch_idx}"
            save_panel(
                rollout,
                sample_dir / f"{name}.png",
                f"{args.split} rollout stride={args.stride} — {name}",
                args.n_context,
            )
            gt_gif = [tensor_to_pil(rollout["gt"][t]) for t in range(args.n_pred_steps)]
            cp_gif = [tensor_to_pil(rollout["rgb_copy"][t]) for t in range(args.n_pred_steps)]
            pr_gif = [tensor_to_pil(rollout["pred"][t]) for t in range(args.n_pred_steps)]
            save_gif(gt_gif, gif_dir / f"{name}_gt.gif")
            save_gif(cp_gif, gif_dir / f"{name}_copy_last.gif")
            save_gif(pr_gif, gif_dir / f"{name}_tracka2.gif")
            n_saved += 1

        clips += 1
        pbar.set_postfix(saved=n_saved, clips=clips)

    summary = {
        "split": args.split,
        "stride": args.stride,
        "n_pred_steps": args.n_pred_steps,
        "n_clips": clips,
        "rgb_copy_last": {
            "psnr_avg": float(np.mean(metrics["copy_psnr"])),
            "ssim_avg": float(np.mean(metrics["copy_ssim"])),
            "lpips_avg": float(np.mean(metrics["copy_lpips"])),
            "psnr_per_step": [float(np.mean(s)) for s in metrics["copy_psnr_per_step"]],
        },
        "tracka2_decoded": {
            "psnr_avg": float(np.mean(metrics["pred_psnr"])),
            "ssim_avg": float(np.mean(metrics["pred_ssim"])),
            "lpips_avg": float(np.mean(metrics["pred_lpips"])),
            "psnr_per_step": [float(np.mean(s)) for s in metrics["pred_psnr_per_step"]],
        },
        "delta_psnr_copy_minus_pred": float(
            np.mean(metrics["copy_psnr"]) - np.mean(metrics["pred_psnr"])
        ),
        "note": (
            "RGB Copy-Last repeats the last context RGB frame. "
            "Track A2 decodes autoregressive feature predictions. "
            "Decoder was NOT trained with Track A2 — pixel gaps may be small."
        ),
    }

    with open(out / "rgb_rollout_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    steps = list(range(args.n_pred_steps))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, summary["rgb_copy_last"]["psnr_per_step"], "o-", label="RGB Copy-Last", linewidth=2)
    ax.plot(steps, summary["tracka2_decoded"]["psnr_per_step"], "s-", label="Track A2 decoded", linewidth=2)
    ax.set_xlabel("Future step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"RGB rollout PSNR — {args.split}, stride={args.stride}", fontweight="bold")
    ax.legend(prop={"weight": "bold"})
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "rgb_psnr_per_step.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(json.dumps(summary, indent=2))
    print(f"\nSaved panels to {sample_dir}/")
    print(f"Saved GIFs to {gif_dir}/")


if __name__ == "__main__":
    main()
