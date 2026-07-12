"""Training script for DINO-Foresight surgical video prediction.

Vision-only future frame prediction using frozen foundation encoders.
No action conditioning, no gesture labels — pure visual prediction.

Usage:
    # Single GPU
    python -m dino_foresight.train --encoder dinov2 --data_dir /path/to/jigsaws

    # Multi-GPU DDP
    torchrun --standalone --nproc_per_node=3 -m dino_foresight.train \
        --encoder dinov2 --data_dir /path/to/jigsaws --ddp

    # V-JEPA 2.1 encoder
    python -m dino_foresight.train --encoder vjepa2 --data_dir /path/to/jigsaws
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# NCCL env vars for L40S (PCIe, no NVLink)
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

from dino_foresight.encoders import build_encoder, DINOv2Encoder, VJEPA2Encoder
from dino_foresight.predictor import MaskedFeatureTransformer
from dino_foresight.decoder import ConvDecoder, PixelDecoder
from dino_foresight.data import build_dataloaders, JigsawsBAIRDataset, JigsawsNPZDataset
from dino_foresight.metrics import psnr, ssim, LPIPSMetric, evaluate_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="DINO-Foresight Surgical Video Prediction")
    parser.add_argument("--encoder", type=str, default="dinov2", choices=["dinov2", "vjepa2"])
    parser.add_argument("--encoder_model", type=str, default=None,
                        help="Specific model variant (e.g. vitb14, vjepa2_1_vit_base_384)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to JIGSAWS data (BAIR format)")
    parser.add_argument("--data_format", type=str, default="bair", choices=["bair", "npz"])
    parser.add_argument("--npz_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/dino_foresight")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_context", type=int, default=4)
    parser.add_argument("--n_future", type=int, default=1)
    parser.add_argument("--n_pred_steps", type=int, default=20,
                        help="Number of autoregressive steps for evaluation")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6.4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2,
                        help="Frame stride (2 = every other frame, matching TPG-VAE)")
    parser.add_argument("--multi_layer", action="store_true", default=True,
                        help="Use multi-layer features from encoder")
    parser.add_argument("--pca_dim", type=int, default=None,
                        help="PCA dimensionality reduction (None = no PCA)")
    parser.add_argument("--train_decoder", action="store_true", default=True,
                        help="Also train a pixel decoder for visualization")
    parser.add_argument("--decoder_type", type=str, default="conv", choices=["conv", "vit"])
    parser.add_argument("--lambda_pixel", type=float, default=1.0,
                        help="Weight for pixel reconstruction loss")
    parser.add_argument("--lambda_feat", type=float, default=1.0,
                        help="Weight for feature prediction loss")
    parser.add_argument("--ddp", action="store_true", help="Enable DDP multi-GPU")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="surgical-future-frame-prediction")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_distributed():
    """Setup DDP if running with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Cosine annealing with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if is_main:
        print(f"=== DINO-Foresight Surgical Video Prediction ===")
        print(f"Encoder:     {args.encoder}")
        print(f"Data:        {args.data_dir}")
        print(f"Output:      {args.output_dir}")
        print(f"Img size:    {args.img_size}")
        print(f"Context:     {args.n_context} frames")
        print(f"Future:      {args.n_future} frames")
        print(f"Batch:       {args.batch_size} x {world_size} GPUs")
        print(f"Epochs:      {args.epochs}")
        print(f"LR:          {args.lr}")
        print(f"Device:      {device}")
        print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        n_context=args.n_context,
        n_future=args.n_future,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stride=args.stride,
        data_format=args.data_format,
        npz_path=args.npz_path,
    )

    if is_main:
        print(f"Train clips: {len(train_loader.dataset)}")
        print(f"Val clips:   {len(val_loader.dataset)}")
        print(f"Test clips:  {len(test_loader.dataset)}")
        print()

    # Build frozen encoder
    if args.encoder == "dinov2":
        encoder = DINOv2Encoder(
            model_name=args.encoder_model or "vitb14",
            img_size=args.img_size,
            multi_layer=args.multi_layer,
            pca_dim=args.pca_dim,
        ).to(device)
    else:
        encoder = VJEPA2Encoder(
            model_name=args.encoder_model or "vjepa2_1_vit_base_384",
            img_size=args.img_size,
        ).to(device)

    feat_dim = encoder.feat_dim
    n_patches = encoder.n_patches
    patch_size = encoder.patch_size

    if is_main:
        print(f"Encoder:     {args.encoder} (feat_dim={feat_dim}, n_patches={n_patches}, patch={patch_size})")

    # Build predictor
    predictor = MaskedFeatureTransformer(
        feat_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        n_patches=n_patches,
        n_context=args.n_context,
        n_future=args.n_future,
    ).to(device)

    # Build decoder (optional, for pixel-space evaluation)
    decoder = None
    if args.train_decoder:
        if args.decoder_type == "conv":
            decoder = ConvDecoder(feat_dim, patch_size, args.img_size).to(device)
        else:
            decoder = PixelDecoder(feat_dim, patch_size, args.img_size).to(device)

    # DDP wrapping
    if world_size > 1:
        predictor = nn.parallel.DistributedDataParallel(
            predictor, device_ids=[local_rank], find_unused_parameters=True
        )
        if decoder is not None:
            decoder = nn.parallel.DistributedDataParallel(
                decoder, device_ids=[local_rank], find_unused_parameters=True
            )

    # Optimizer — only train predictor and decoder (encoder is frozen)
    params = list(predictor.parameters())
    if decoder is not None:
        params += list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = get_lr_scheduler(
        optimizer, args.warmup_epochs, args.epochs, len(train_loader)
    )

    # W&B
    wandb_run = None
    if args.wandb and is_main:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"{args.encoder}_{int(time.time())}",
        )

    # LPIPS metric
    lpips_fn = LPIPSMetric(device=str(device)) if is_main else None

    # Training loop
    best_psnr = 0.0
    for epoch in range(args.epochs):
        predictor.train()
        if decoder is not None:
            decoder.train()

        epoch_losses = {"feat": [], "pixel": [], "total": []}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main)

        for batch_idx, (past_frames, future_frames) in enumerate(pbar):
            past_frames = past_frames.to(device)  # (B, T_c, C, H, W)
            future_frames = future_frames.to(device)  # (B, T_f, C, H, W)

            B, T_c, C, H, W = past_frames.shape
            T_f = future_frames.shape[1]

            # Extract frozen features for all frames
            with torch.no_grad():
                all_frames = torch.cat([past_frames, future_frames], dim=1)  # (B, T_c+T_f, C, H, W)
                all_frames_flat = all_frames.reshape(B * (T_c + T_f), C, H, W)
                all_feats = encoder(all_frames_flat)  # (B*(T_c+T_f), N, D)
                all_feats = all_feats.reshape(B, T_c + T_f, n_patches, feat_dim)

                context_feats = all_feats[:, :T_c].detach()  # (B, T_c, N, D)
                future_feats = all_feats[:, T_c:].detach()  # (B, T_f, N, D) — target

            # Predict future features
            pred_future_feats = predictor(context_feats)  # (B, T_f, N, D)

            # Feature prediction loss (SmoothL1)
            feat_loss = F.smooth_l1_loss(pred_future_feats, future_feats, beta=0.1)

            total_loss = args.lambda_feat * feat_loss

            # Optional pixel reconstruction loss
            pixel_loss = torch.tensor(0.0, device=device)
            if decoder is not None:
                pred_pixels = decoder(pred_future_feats)  # (B, T_f, C, H, W)
                pixel_loss = F.l1_loss(pred_pixels, future_frames)
                total_loss = total_loss + args.lambda_pixel * pixel_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses["feat"].append(feat_loss.item())
            epoch_losses["pixel"].append(pixel_loss.item())
            epoch_losses["total"].append(total_loss.item())

            if is_main and batch_idx % 50 == 0:
                pbar.set_postfix({
                    "feat": f"{feat_loss.item():.4f}",
                    "pix": f"{pixel_loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        # Log epoch metrics
        if is_main:
            avg_feat = np.mean(epoch_losses["feat"])
            avg_pixel = np.mean(epoch_losses["pixel"])
            avg_total = np.mean(epoch_losses["total"])
            print(f"Epoch {epoch}: feat={avg_feat:.4f}, pixel={avg_pixel:.4f}, total={avg_total:.4f}")

            if wandb_run:
                wandb_run.log({
                    "epoch": epoch,
                    "train/feat_loss": avg_feat,
                    "train/pixel_loss": avg_pixel,
                    "train/total_loss": avg_total,
                    "train/lr": scheduler.get_last_lr()[0],
                })

        # Evaluation
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            eval_results = evaluate(
                predictor=predictor,
                decoder=decoder,
                encoder=encoder,
                dataloader=test_loader,
                device=device,
                n_pred_steps=args.n_pred_steps,
                lpips_fn=lpips_fn,
                is_main=is_main,
            )

            if is_main:
                print(f"  Eval: PSNR={eval_results['psnr_avg']:.2f}, SSIM={eval_results['ssim_avg']:.4f}", end="")
                if "lpips_avg" in eval_results:
                    print(f", LPIPS={eval_results['lpips_avg']:.4f}", end="")
                print()

                if wandb_run:
                    wandb_run.log({
                        "epoch": epoch,
                        "val/psnr": eval_results["psnr_avg"],
                        "val/ssim": eval_results["ssim_avg"],
                    })
                    if "lpips_avg" in eval_results:
                        wandb_run.log({"val/lpips": eval_results["lpips_avg"]})

                # Save best model
                if eval_results["psnr_avg"] > best_psnr:
                    best_psnr = eval_results["psnr_avg"]
                    save_checkpoint(
                        predictor=predictor,
                        decoder=decoder,
                        optimizer=optimizer,
                        epoch=epoch,
                        args=args,
                        path=os.path.join(args.output_dir, "best_model.pth"),
                    )
                    print(f"  *** New best PSNR: {best_psnr:.2f} ***")

        # Periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            if is_main:
                save_checkpoint(
                    predictor=predictor,
                    decoder=decoder,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    path=os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pth"),
                )

    # Final save
    if is_main:
        save_checkpoint(
            predictor=predictor,
            decoder=decoder,
            optimizer=optimizer,
            epoch=args.epochs - 1,
            args=args,
            path=os.path.join(args.output_dir, "final_model.pth"),
        )
        print(f"\nTraining complete! Best PSNR: {best_psnr:.2f}")
        print(f"Checkpoints saved to: {args.output_dir}")

        if wandb_run:
            wandb_run.finish()


@torch.no_grad()
def evaluate(predictor, decoder, encoder, dataloader, device, n_pred_steps, lpips_fn, is_main=True):
    """Evaluate the model on the test set with autoregressive prediction."""
    predictor.eval()
    if decoder is not None:
        decoder.eval()

    all_psnr = []
    all_ssim = []
    all_lpips = []

    pbar = tqdm(dataloader, desc="Evaluating", disable=not is_main)
    for past_frames, future_frames in pbar:
        past_frames = past_frames.to(device)
        future_frames = future_frames.to(device)

        B, T_c, C, H, W = past_frames.shape
        T_f = future_frames.shape[1]
        n_steps = min(n_pred_steps, T_f)

        # Extract context features
        past_flat = past_frames.reshape(B * T_c, C, H, W)
        context_feats = encoder(past_flat).reshape(B, T_c, encoder.n_patches, encoder.feat_dim)

        # Autoregressive prediction
        pred_feats_list = []
        current_context = context_feats

        for step in range(n_steps):
            if hasattr(predictor, "module"):
                pred = predictor.module(current_context)
            else:
                pred = predictor(current_context)
            pred_feats_list.append(pred)

            # Slide window: drop oldest, append prediction
            current_context = torch.cat([current_context[:, 1:], pred], dim=1)

        pred_feats = torch.cat(pred_feats_list, dim=1)  # (B, n_steps, N, D)

        # Decode to pixels if decoder available
        if decoder is not None:
            if hasattr(decoder, "module"):
                pred_pixels = decoder.module(pred_feats)
            else:
                pred_pixels = decoder(pred_feats)
        else:
            # Without decoder, we can only evaluate in feature space
            # For now, skip pixel metrics
            continue

        # Compute metrics per step
        for step in range(n_steps):
            pred_step = pred_pixels[:, step]
            gt_step = future_frames[:, step]
            all_psnr.append(psnr(pred_step, gt_step).mean().item())
            all_ssim.append(ssim(pred_step, gt_step).mean().item())
            if lpips_fn is not None:
                all_lpips.append(lpips_fn(pred_step, gt_step).mean().item())

    results = {
        "psnr_avg": np.mean(all_psnr) if all_psnr else 0.0,
        "ssim_avg": np.mean(all_ssim) if all_ssim else 0.0,
    }
    if all_lpips:
        results["lpips_avg"] = np.mean(all_lpips)

    return results


def save_checkpoint(predictor, decoder, optimizer, epoch, args, path):
    """Save model checkpoint."""
    state = {
        "epoch": epoch,
        "predictor": (predictor.module if hasattr(predictor, "module") else predictor).state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    if decoder is not None:
        state["decoder"] = (decoder.module if hasattr(decoder, "module") else decoder).state_dict()
    torch.save(state, path)


if __name__ == "__main__":
    main()
