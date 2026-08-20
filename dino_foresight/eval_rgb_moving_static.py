#!/usr/bin/env python3
"""RGB PSNR/SSIM/LPIPS with moving vs static patch breakdown.

Decodes feature predictions with the Phase 1 RAE decoder and compares
RGB Copy-Last vs predictor on LOUO test. Moving/static splits use
stop-grad ||z_{t+1}-z_t|| per patch (same gate as Track A2 training).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dino_foresight.data import build_dataloaders
from dino_foresight.decoder import RAEDecoder
from dino_foresight.encoders import DINOv2Encoder
from dino_foresight.metrics import LPIPSMetric, psnr, ssim
from dino_foresight.predictor import build_predictor
from dino_foresight.rollout_viz import compute_rgb_predictions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictor_ckpt", type=str, required=True)
    p.add_argument("--decoder_ckpt", type=str, required=True)
    p.add_argument("--pca_ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--label", type=str, default="tracka2")
    p.add_argument("--n_context", type=int, default=4)
    p.add_argument("--n_pred_steps", type=int, default=1)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_clips", type=int, default=0)
    return p.parse_args()


def patch_psnr(pred_px, gt_px, patch_mask, grid):
    """PSNR on spatial patches selected by boolean mask (B, N_patches)."""
    B, C, H, W = pred_px.shape
    ph = pw = H // grid
    vals = []
    for b in range(B):
        idx = patch_mask[b].nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        for pi in idx:
            row, col = int(pi // grid), int(pi % grid)
            y0, x0 = row * ph, col * pw
            p = pred_px[b, :, y0:y0 + ph, x0:x0 + pw]
            g = gt_px[b, :, y0:y0 + ph, x0:x0 + pw]
            vals.append(psnr(p.unsqueeze(0), g.unsqueeze(0)).item())
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def eval_horizon(predictor, decoder, encoder, loader, device, n_steps, lpips_fn, max_clips):
    copy_psnr_all, pred_psnr_all = [], []
    copy_mov, pred_mov = [], []
    copy_sta, pred_sta = [], []
    copy_ssim_all, pred_ssim_all = [], []
    copy_lpips_all, pred_lpips_all = [], []
    clips = 0
    grid = int(encoder.n_patches ** 0.5)

    for past_frames, future_frames in tqdm(loader, desc=f"rgb t={n_steps}"):
        if max_clips > 0 and clips >= max_clips:
            break
        if future_frames.shape[1] < n_steps:
            continue
        B, T_c, C, H, W = past_frames.shape
        past = past_frames.to(device)
        future = future_frames[:, :n_steps].to(device)

        gt_n, copy_n, pred_n = compute_rgb_predictions(
            predictor, decoder, encoder, past_frames, future_frames, n_steps, device,
        )

        all_f = torch.cat([past, future], dim=1)
        flat = all_f.reshape(B * (T_c + n_steps), C, H, W)
        feats = encoder(flat).reshape(B, T_c + n_steps, encoder.n_patches, encoder.feat_dim)

        for t in range(n_steps):
            gt_t = gt_n[:, t]
            cp_t = copy_n[:, t]
            pr_t = pred_n[:, t]
            copy_psnr_all.append(psnr(cp_t, gt_t).mean().item())
            pred_psnr_all.append(psnr(pr_t, gt_t).mean().item())
            copy_ssim_all.append(ssim(cp_t, gt_t).mean().item())
            pred_ssim_all.append(ssim(pr_t, gt_t).mean().item())
            copy_lpips_all.append(lpips_fn(cp_t, gt_t).mean().item())
            pred_lpips_all.append(lpips_fn(pr_t, gt_t).mean().item())

            z_prev = feats[:, T_c + t - 1] if t > 0 else feats[:, T_c - 1]
            mag = (feats[:, T_c + t] - z_prev).norm(dim=-1)
            moving = mag > mag.median(dim=1, keepdim=True).values
            static = ~moving
            copy_mov.append(patch_psnr(cp_t, gt_t, moving, grid))
            pred_mov.append(patch_psnr(pr_t, gt_t, moving, grid))
            copy_sta.append(patch_psnr(cp_t, gt_t, static, grid))
            pred_sta.append(patch_psnr(pr_t, gt_t, static, grid))

        clips += B

    def mean(xs):
        xs = [x for x in xs if not np.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "n_clips": clips,
        "copy_last": {
            "psnr_avg": mean(copy_psnr_all),
            "ssim_avg": mean(copy_ssim_all),
            "lpips_avg": mean(copy_lpips_all),
            "psnr_moving_patches": mean(copy_mov),
            "psnr_static_patches": mean(copy_sta),
        },
        "predictor": {
            "psnr_avg": mean(pred_psnr_all),
            "ssim_avg": mean(pred_ssim_all),
            "lpips_avg": mean(pred_lpips_all),
            "psnr_moving_patches": mean(pred_mov),
            "psnr_static_patches": mean(pred_sta),
        },
        "delta_psnr": mean(copy_psnr_all) - mean(pred_psnr_all),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = torch.load(args.predictor_ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    encoder = DINOv2Encoder(
        model_name=ckpt_args.get("encoder_model", "vitb14_reg"),
        img_size=224,
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
        encoder.feat_dim, encoder.patch_size, 224,
        hidden_dim=int(dec_args.get("decoder_hidden_dim", 1152)),
        num_layers=int(dec_args.get("decoder_layers", 8)),
        num_heads=int(dec_args.get("decoder_heads", 12)),
    ).to(device)
    decoder.load_state_dict(dec_ckpt.get("decoder", dec_ckpt))
    decoder.eval()

    lpips_fn = LPIPSMetric(device=str(device))
    horizons = [1, 5, 10]
    all_results = {}

    for n_steps in horizons:
        _, _, test_loader = build_dataloaders(
            data_dir=args.data_dir,
            n_context=args.n_context,
            n_future=n_steps,
            img_size=224,
            batch_size=args.batch_size,
            num_workers=4,
            stride=args.stride,
            data_format="bair",
            rebuild_split=False,
            val_max_clips=0,
            seed=42,
        )
        key = f"stride{args.stride}_t{n_steps}"
        res = eval_horizon(
            predictor, decoder, encoder, test_loader, device,
            n_steps, lpips_fn, args.max_clips,
        )
        all_results[key] = res
        c, p = res["copy_last"], res["predictor"]
        print(
            f"{key}: Copy PSNR={c['psnr_avg']:.2f} Pred={p['psnr_avg']:.2f} "
            f"Δ={res['delta_psnr']:.2f} | moving copy/pred={c['psnr_moving_patches']:.2f}/{p['psnr_moving_patches']:.2f}"
        )

    out = Path(args.output_dir) / f"rgb_moving_static_{args.label}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    lines = [
        f"# RGB moving/static — {args.label}",
        "",
        "| Horizon | Copy PSNR | Pred PSNR | Δ | Copy mov | Pred mov | Copy static | Pred static |",
        "|---------|-----------|-----------|---|----------|----------|-------------|-------------|",
    ]
    for key, res in all_results.items():
        t = key.split("_t")[-1]
        c, p = res["copy_last"], res["predictor"]
        lines.append(
            f"| t={t} | {c['psnr_avg']:.2f} | {p['psnr_avg']:.2f} | {res['delta_psnr']:.2f} | "
            f"{c['psnr_moving_patches']:.2f} | {p['psnr_moving_patches']:.2f} | "
            f"{c['psnr_static_patches']:.2f} | {p['psnr_static_patches']:.2f} |"
        )
    md = Path(args.output_dir) / f"rgb_moving_static_{args.label}.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}\nWrote {md}")


if __name__ == "__main__":
    main()
