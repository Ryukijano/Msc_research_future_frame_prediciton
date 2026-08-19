#!/usr/bin/env python3
"""Standalone evaluation script for DINO-Foresight on JIGSAWS.

Loads a trained predictor + decoder and computes PSNR/SSIM/LPIPS at
specified future horizons and frame strides.

Example:
    python -m dino_foresight.evaluate \
        --predictor_ckpt outputs/jigsaws_masterplan/predictor/best_model.pth \
        --decoder_ckpt outputs/jigsaws_masterplan/phase1_decoder/best_val_decoder.pth \
        --data_dir VPTR_jigsaws_working/jigsaws_suturing/bair_format_dir_louo \
        --output_dir outputs/jigsaws_masterplan/eval \
        --n_context 4 --n_pred_steps 5,10,20 --stride 1
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dino_foresight.data import build_dataloaders
from dino_foresight.encoders import DINOv2Encoder
from dino_foresight.predictor import build_predictor
from dino_foresight.decoder import RAEDecoder
from dino_foresight.train import setup_distributed, evaluate
from dino_foresight.metrics import LPIPSMetric, psnr, ssim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor_ckpt", type=str, default=None,
                        help="Path to trained predictor checkpoint (omit for decoder-only eval)")
    parser.add_argument("--decoder_ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="dinov2")
    parser.add_argument("--encoder_model", type=str, default="vitb14_reg")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_context", type=int, default=4)
    parser.add_argument("--n_pred_steps", type=str, default="1,5,10,20",
                        help="Comma-separated future horizons to evaluate")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--multi_layer", action="store_true", default=True)
    parser.add_argument("--pca_dim", type=int, default=1152)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--pca_ckpt", type=str, default=None,
                        help="Saved PCA from training (do not refit)")
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--decoder_hidden_dim", type=int, default=1152)
    parser.add_argument("--decoder_layers", type=int, default=8)
    parser.add_argument("--decoder_heads", type=int, default=12)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--max_clips", type=int, default=0,
                        help="If >0, limit clips per horizon")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def evaluate_decoder(decoder, encoder, dataloader, device, lpips_fn, is_main=True, max_clips=None):
    """Evaluate an RAE decoder by reconstructing ground-truth frames."""
    decoder.eval()
    encoder.eval()

    psnr_vals = []
    ssim_vals = []
    lpips_vals = [] if lpips_fn is not None else None
    clips_evaluated = 0
    pbar = tqdm(dataloader, desc="Decoder eval", disable=not is_main)
    for past_frames, future_frames in pbar:
        if max_clips is not None and clips_evaluated >= max_clips:
            break
        all_frames = torch.cat([past_frames, future_frames], dim=1).to(device)
        B, T, C, H, W = all_frames.shape
        all_flat = all_frames.reshape(B * T, C, H, W)
        feats = encoder(all_flat).reshape(B, T, encoder.n_patches, encoder.feat_dim)

        dec = decoder.module if hasattr(decoder, "module") else decoder
        pred_pixels = torch.clamp(dec(feats.float()), -1, 1)

        pred_flat = pred_pixels.reshape(B * T, C, H, W)
        gt_flat = all_frames.reshape(B * T, C, H, W)
        psnr_vals.append(psnr(pred_flat, gt_flat).mean().item())
        ssim_vals.append(ssim(pred_flat, gt_flat).mean().item())
        if lpips_fn is not None:
            lpips_vals.append(lpips_fn(pred_flat, gt_flat).mean().item())
        clips_evaluated += B

    psnr_avg = float(np.mean(psnr_vals))
    ssim_avg = float(np.mean(ssim_vals))
    results = {
        "psnr_avg": psnr_avg,
        "ssim_avg": ssim_avg,
        "psnr_per_step": [psnr_avg],
        "ssim_per_step": [ssim_avg],
    }
    if lpips_vals is not None:
        results["lpips_avg"] = float(np.mean(lpips_vals))
        results["lpips_per_step"] = [results["lpips_avg"]]
    return results


def build_model(args, device):
    encoder = DINOv2Encoder(
        model_name=args.encoder_model,
        img_size=args.img_size,
        multi_layer=args.multi_layer,
        pca_dim=args.pca_dim,
    ).to(device)

    decoder = RAEDecoder(
        encoder.feat_dim, encoder.patch_size, args.img_size,
        hidden_dim=args.decoder_hidden_dim,
        num_layers=args.decoder_layers,
        num_heads=args.decoder_heads,
    ).to(device)
    dec_ckpt = load_checkpoint(args.decoder_ckpt, device)
    decoder.load_state_dict(dec_ckpt.get("decoder", dec_ckpt))

    predictor = None
    if args.predictor_ckpt:
        pred_ckpt = load_checkpoint(args.predictor_ckpt, device)
        ckpt_args = pred_ckpt.get("args", {})
        predictor = build_predictor(
            feat_dim=encoder.feat_dim,
            n_patches=encoder.n_patches,
            cfg=ckpt_args,
            hidden_dim=ckpt_args.get("hidden_dim", args.hidden_dim),
            num_layers=ckpt_args.get("num_layers", args.num_layers),
            num_heads=ckpt_args.get("num_heads", args.num_heads),
            n_context=ckpt_args.get("n_context", args.n_context),
        ).to(device)
        predictor.load_state_dict(pred_ckpt.get("predictor", pred_ckpt))

    return encoder, predictor, decoder


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed() if args.ddp else (0, 1, 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    os.makedirs(args.output_dir, exist_ok=True)

    encoder, predictor, decoder = build_model(args, device)

    pca_path = args.pca_ckpt
    if pca_path is None and args.predictor_ckpt:
        candidate = Path(args.predictor_ckpt).parent / "pca.pth"
        if candidate.exists():
            pca_path = str(candidate)
    if hasattr(encoder, "pca_proj") and encoder.pca_proj is not None:
        if pca_path:
            print(f"Loading PCA from {pca_path} (not refitting)")
            encoder.load_pca(pca_path, device)
        elif is_main:
            train_loader, _, _ = build_dataloaders(
                data_dir=args.data_dir,
                n_context=args.n_context,
                n_future=1,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                stride=args.stride,
                data_format="bair",
                rebuild_split=False,
                val_max_clips=0,
                seed=args.seed if hasattr(args, "seed") else 42,
            )
            print("WARNING: no pca.pth found; fitting a new PCA (eval will not match training)")
            encoder.fit_pca(train_loader, device, n_samples=2000)

    if args.ddp and world_size > 1:
        from torch.nn.parallel import DistributedDataParallel
        if predictor is not None:
            predictor = DistributedDataParallel(predictor, device_ids=[local_rank])
        decoder = DistributedDataParallel(decoder, device_ids=[local_rank])

    lpips_fn = LPIPSMetric(device=str(device)) if is_main else None

    horizons = [int(x) for x in args.n_pred_steps.split(",")]
    all_results = {}

    for n_steps in horizons:
        _, _, test_loader = build_dataloaders(
            data_dir=args.data_dir,
            n_context=args.n_context,
            n_future=n_steps,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            stride=args.stride,
            data_format="bair",
            rebuild_split=False,
            val_max_clips=0,
            seed=42,
        )

        if is_main:
            print(f"\n=== Horizon t={n_steps} (stride={args.stride}) ===")
            print(f"Test clips: {len(test_loader.dataset)}")

        if predictor is None:
            results = evaluate_decoder(
                decoder=decoder,
                encoder=encoder,
                dataloader=test_loader,
                device=device,
                lpips_fn=lpips_fn,
                is_main=is_main,
                max_clips=args.max_clips if args.max_clips > 0 else None,
            )
        else:
            results = evaluate(
                predictor=predictor,
                decoder=decoder,
                encoder=encoder,
                dataloader=test_loader,
                device=device,
                n_pred_steps=n_steps,
                lpips_fn=lpips_fn,
                is_main=is_main,
                max_clips=args.max_clips if args.max_clips > 0 else None,
            )

        if is_main:
            summary = {
                "psnr_avg": float(results.get("psnr_avg", 0.0)),
                "ssim_avg": float(results.get("ssim_avg", 0.0)),
                "lpips_avg": float(results.get("lpips_avg", 0.0)) if "lpips_avg" in results else None,
                "psnr_per_step": [float(x) for x in results.get("psnr_per_step", [])],
                "ssim_per_step": [float(x) for x in results.get("ssim_per_step", [])],
                "lpips_per_step": [float(x) for x in results.get("lpips_per_step", [])] if "lpips_per_step" in results else None,
                "n_clips": len(test_loader.dataset),
            }
            all_results[f"stride{args.stride}_t{n_steps}"] = summary
            print(f"  PSNR: {summary['psnr_avg']:.2f}, SSIM: {summary['ssim_avg']:.4f}, "
                  f"LPIPS: {summary['lpips_avg']:.4f}" if summary['lpips_avg'] is not None else "")

    if is_main:
        out_path = Path(args.output_dir) / f"results_stride{args.stride}.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
