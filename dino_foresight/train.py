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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image
try:
    import wandb
except ImportError:
    wandb = None

# NCCL env vars for L40S (PCIe, no NVLink)
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

from dino_foresight.encoders import build_encoder, DINOv2Encoder, VJEPA2Encoder
from dino_foresight.predictor import MaskedFeatureTransformer
from dino_foresight.decoder import ConvDecoder, PixelDecoder, RAEDecoder, PerceptualLoss
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
    parser.add_argument("--n_future", type=int, default=1,
                        help="Number of future frames to predict during training (1 = single-step)")
    parser.add_argument("--n_pred_steps", type=int, default=1,
                        help="Number of steps for evaluation (1 = single-step, >1 = autoregressive)")
    parser.add_argument("--hidden_dim", type=int, default=1152)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay for AdamW (0.05 for small dataset regularization)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs (300 for small dataset)")
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=6,
                        help="Frame stride (6 = every 6th frame, larger motion magnitude for slow surgical video)")
    parser.add_argument("--multi_layer", action="store_true", default=True,
                        help="Use multi-layer features from encoder (default: True, layers 2,5,8,11 concatenated = 3072-dim)")
    parser.add_argument("--no_multi_layer", dest="multi_layer", action="store_false",
                        help="Use single-layer features (768-dim, no PCA)")
    parser.add_argument("--pca_dim", type=int, default=1152,
                        help="PCA dimensionality reduction (1152 = match DINO-Foresight paper, 0 = no PCA)")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="TDV-style residual prediction: predict delta from last context frame (default: off for slow surgical motion)")
    parser.add_argument("--no_residual", dest="residual", action="store_false",
                        help="Disable residual prediction (predict absolute features, default)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Predictor dropout")
    parser.add_argument("--use_motion", action="store_true", default=False,
                        help="Add past-only motion (z_t - z_{t-1}, optional RGB delta) to future queries")
    parser.add_argument("--motion_from", type=str, default="feat", choices=["feat", "rgb", "both"],
                        help="Past motion source (never uses the future frame)")
    parser.add_argument("--motion_weight", action="store_true", default=False,
                        help="Weight SmoothL1 by stop-grad ||z_{t+1}-z_t|| so moving patches dominate")
    parser.add_argument("--motion_weight_floor", type=float, default=0.25,
                        help="Minimum motion weight so static clips still train")
    parser.add_argument("--pca_ckpt", type=str, default=None,
                        help="Load an existing PCA projection instead of fitting (reuse Track A pca.pth)")
    parser.add_argument("--init_eval", action="store_true", default=True,
                        help="Run val Copy-Last gate before epoch 0")
    parser.add_argument("--no_init_eval", dest="init_eval", action="store_false")
    parser.add_argument("--abort_if_skip_broken", action="store_true", default=True,
                        help="Exit if residual init val is not Copy-Last")
    parser.add_argument("--no_abort_if_skip_broken", dest="abort_if_skip_broken", action="store_false")
    parser.add_argument("--skip_gate_tol", type=float, default=0.08,
                        help="Abort if |init_val - copy_last| exceeds this SmoothL1 gap")
    parser.add_argument("--loss_type", type=str, default="cosine_l2", choices=["smooth_l1", "cosine_l2", "cosine", "l2"],
                        help="Feature loss type: cosine_l2 (default), cosine, l2, or smooth_l1 (legacy)")
    parser.add_argument("--lambda_cosine", type=float, default=1.0,
                        help="Weight for cosine similarity loss component")
    parser.add_argument("--lambda_l2", type=float, default=0.5,
                        help="Weight for L2 (MSE) loss component")
    parser.add_argument("--n_pred_steps_train", type=int, default=3,
                        help="Number of future steps to predict during training (multi-step). 1=single-step")
    parser.add_argument("--scheduled_sampling", action="store_true", default=True,
                        help="Use scheduled sampling: decay probability of using GT context vs predicted context")
    parser.add_argument("--no_scheduled_sampling", dest="scheduled_sampling", action="store_false",
                        help="Disable scheduled sampling (always teacher forcing)")
    parser.add_argument("--ss_decay_epochs", type=int, default=50,
                        help="Number of epochs over which to decay teacher forcing from 1.0 to 0.5")
    parser.add_argument("--vicreg_lambda", type=float, default=0.0,
                        help="Weight for VICReg regularizer (variance + covariance) on predicted features. 0=disabled (default)")
    parser.add_argument("--vicreg_var_coeff", type=float, default=1.0,
                        help="VICReg variance coefficient")
    parser.add_argument("--vicreg_cov_coeff", type=float, default=1.0,
                        help="VICReg covariance coefficient")
    parser.add_argument("--joint_finetune", action="store_true", default=False,
                        help="Phase 3: jointly fine-tune predictor + decoder with small decoder LR")
    parser.add_argument("--train_decoder", action="store_true", default=False,
                        help="Also train a pixel decoder (disabled by default, use --pretrain_decoder for Phase 1)")
    parser.add_argument("--decoder_type", type=str, default="rae", choices=["conv", "vit", "rae"],
                        help="rae=Representation Autoencoder ViT decoder (best), vit=small ViT, conv=legacy")
    parser.add_argument("--lambda_pixel", type=float, default=0.0,
                        help="Weight for pixel reconstruction loss (0=disabled, decoder bottleneck makes this noisy)")
    parser.add_argument("--lambda_lpips", type=float, default=0.0,
                        help="Weight for LPIPS perceptual loss (0=disabled by default)")
    parser.add_argument("--lambda_feat", type=float, default=1.0,
                        help="Weight for feature prediction loss")
    # Phase 1: decoder pretraining
    parser.add_argument("--pretrain_decoder", action="store_true", default=False,
                        help="Phase 1: only train RAE decoder on pixel reconstruction (no predictor)")
    parser.add_argument("--decoder_ckpt", type=str, default=None,
                        help="Path to pretrained decoder checkpoint to load")
    parser.add_argument("--predictor_ckpt", type=str, default=None,
                        help="Path to pretrained predictor checkpoint (for Phase 3 joint fine-tuning)")
    # RAE decoder config
    parser.add_argument("--decoder_hidden_dim", type=int, default=1152,
                        help="Decoder hidden dim (match feature dim: 1152 for multi-layer PCA, 768 for single-layer)")
    parser.add_argument("--decoder_layers", type=int, default=8)
    parser.add_argument("--decoder_heads", type=int, default=12)
    parser.add_argument("--decoder_lr", type=float, default=1e-4,
                        help="Separate LR for decoder (lower than predictor)")
    parser.add_argument("--decoder_warmup_epochs", type=int, default=5)
    parser.add_argument("--ddp", action="store_true", help="Enable DDP multi-GPU")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="surgical-future-frame-prediction")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_max_clips", type=int, default=200,
                        help="Max clips for evaluation (subset for speed)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="Compute val feature loss every N epochs (overfitting detection)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without val loss improvement)")
    parser.add_argument("--rebuild_split", action="store_true", default=False,
                        help="Merge train+val dirs and re-split 80/20 (fixes inverted data splits)")
    parser.add_argument("--val_max_clips", type=int, default=500,
                        help="Max val clips for validation (subsample for speed, 0 = use all)")
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


def motion_frames_ctx(args, past_frames):
    """RGB context for the motion encoder; None when motion is feature-only."""
    if args.use_motion and args.motion_from in ("rgb", "both"):
        return past_frames
    return None


def patch_motion_weights(context_feats, target_feats, floor=0.25):
    """Stop-grad per-patch weights from ||z_{t+k} - z_{t+k-1}||, floored so static clips train."""
    z_t = context_feats[:, -1]
    prev = torch.cat([z_t.unsqueeze(1), target_feats[:, :-1]], dim=1)
    mag = (target_feats - prev).detach().norm(dim=-1)
    return (mag / (mag.mean() + 1e-6)).clamp(min=floor)


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
        print(f"Future:      {args.n_future} frames (train: {max(args.n_future, args.n_pred_steps_train) if not args.pretrain_decoder else args.n_future})")
        print(f"Pred steps:  train={args.n_pred_steps_train}, eval={args.n_pred_steps}")
        print(f"Batch:       {args.batch_size} x {world_size} GPUs")
        print(f"Epochs:      {args.epochs}")
        print(f"LR:          {args.lr}")
        print(f"Residual:    {args.residual}  motion={args.use_motion}/{args.motion_from}  dropout={args.dropout}")
        print(f"Device:      {device}")
        print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build dataloaders — ensure enough future frames for multi-step training
    n_future_train = max(args.n_future, args.n_pred_steps_train) if not args.pretrain_decoder else args.n_future
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        n_context=args.n_context,
        n_future=n_future_train,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stride=args.stride,
        data_format=args.data_format,
        npz_path=args.npz_path,
        rebuild_split=args.rebuild_split,
        val_max_clips=args.val_max_clips,
        seed=args.seed,
    )

    # Separate eval loader with more future frames for long-horizon evaluation
    if args.n_pred_steps > args.n_future:
        _, _, eval_test_loader = build_dataloaders(
            data_dir=args.data_dir,
            n_context=args.n_context,
            n_future=args.n_pred_steps,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            stride=args.stride,
            data_format=args.data_format,
            npz_path=args.npz_path,
            rebuild_split=args.rebuild_split,
            val_max_clips=0,  # use full test set for eval
            seed=args.seed,
        )
    else:
        eval_test_loader = test_loader

    if is_main:
        print(f"Train clips: {len(train_loader.dataset)}")
        print(f"Val clips:   {len(val_loader.dataset)}")
        print(f"Test clips:  {len(test_loader.dataset)}")
        print(f"Eval clips:  {len(eval_test_loader.dataset)} (n_future={args.n_pred_steps})")
        print()

    # Build frozen encoder
    if args.encoder == "dinov2":
        encoder = DINOv2Encoder(
            model_name=args.encoder_model or "vitb14_reg",
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

    # Fit or load PCA. Reuse Track A pca.pth so eval stays on the same basis.
    pca_path = os.path.join(args.output_dir, "pca.pth")
    if hasattr(encoder, "pca_proj") and encoder.pca_proj is not None:
        if args.pca_ckpt:
            encoder.load_pca(args.pca_ckpt, device)
            if is_main:
                torch.save(
                    {
                        "pca_proj": encoder.pca_proj.weight.detach().cpu(),
                        "pca_mean": encoder.pca_mean.detach().cpu(),
                        "pca_dim": int(args.pca_dim),
                    },
                    pca_path,
                )
                print(f"  Reused PCA from {args.pca_ckpt} (copied to {pca_path})")
        else:
            if is_main:
                print("Fitting PCA on training data...")
                encoder.fit_pca(train_loader, device, n_samples=2000)
                torch.save(
                    {
                        "pca_proj": encoder.pca_proj.weight.detach().cpu(),
                        "pca_mean": encoder.pca_mean.detach().cpu(),
                        "pca_dim": int(args.pca_dim),
                    },
                    pca_path,
                )
                print(f"  Saved PCA to {pca_path}")
            if world_size > 1:
                torch.distributed.barrier()
            if not is_main:
                encoder.load_pca(pca_path, device)

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
        dropout=args.dropout,
        residual=args.residual,
        use_motion=args.use_motion,
        motion_from=args.motion_from,
    ).to(device)

    # Build decoder
    decoder = None
    if args.train_decoder or args.pretrain_decoder or args.decoder_ckpt:
        if args.decoder_type == "rae":
            decoder = RAEDecoder(
                feat_dim, patch_size, args.img_size,
                hidden_dim=args.decoder_hidden_dim,
                num_layers=args.decoder_layers,
                num_heads=args.decoder_heads,
            ).to(device)
        elif args.decoder_type == "conv":
            decoder = ConvDecoder(feat_dim, patch_size, args.img_size).to(device)
        else:
            decoder = PixelDecoder(feat_dim, patch_size, args.img_size).to(device)

    # Load pretrained decoder if specified
    if args.decoder_ckpt and decoder is not None:
        ckpt = torch.load(args.decoder_ckpt, map_location=device, weights_only=False)
        decoder_state = ckpt.get("decoder", ckpt)
        if hasattr(decoder, "module"):
            decoder.module.load_state_dict(decoder_state)
        else:
            decoder.load_state_dict(decoder_state)
        if is_main:
            print(f"Loaded pretrained decoder from {args.decoder_ckpt}")

    # Load pretrained predictor if specified (Phase 3 joint fine-tuning)
    if args.predictor_ckpt:
        ckpt = torch.load(args.predictor_ckpt, map_location=device, weights_only=False)
        pred_state = ckpt.get("predictor", ckpt)
        if hasattr(predictor, "module"):
            predictor.module.load_state_dict(pred_state)
        else:
            predictor.load_state_dict(pred_state)
        if is_main:
            print(f"Loaded pretrained predictor from {args.predictor_ckpt}")

    # Perceptual loss (for RAE decoder training or Phase 2/3 LPIPS)
    perceptual_loss = None
    if args.decoder_type == "rae" and (args.train_decoder or args.pretrain_decoder or args.lambda_lpips > 0):
        perceptual_loss = PerceptualLoss(device=str(device)).to(device)
        if is_main:
            print(f"Decoder:     RAE (hidden={args.decoder_hidden_dim}, layers={args.decoder_layers}, heads={args.decoder_heads})")
            print(f"Loss:        feat={args.lambda_feat}, pixel_l1={args.lambda_pixel}, lpips={args.lambda_lpips}")

    # DDP wrapping
    if world_size > 1:
        if not args.pretrain_decoder:
            predictor = nn.parallel.DistributedDataParallel(
                predictor, device_ids=[local_rank], find_unused_parameters=True
            )
        if decoder is not None:
            decoder = nn.parallel.DistributedDataParallel(
                decoder, device_ids=[local_rank], find_unused_parameters=True
            )

    # Optimizer — separate LR for predictor and decoder
    if args.pretrain_decoder:
        # Phase 1: only train decoder
        params = list(decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=args.decoder_lr, weight_decay=args.weight_decay)
        scheduler = get_lr_scheduler(
            optimizer, args.decoder_warmup_epochs, args.epochs, len(train_loader)
        )
    else:
        # Phase 2/3: train predictor (+ optionally decoder)
        param_groups = [
            {"params": list(predictor.parameters()), "lr": args.lr},
        ]
        train_decoder_in_phase2 = (args.lambda_pixel > 0 or args.lambda_lpips > 0) and decoder is not None
        if args.joint_finetune and decoder is not None:
            # Phase 3: joint fine-tune predictor + decoder
            param_groups.append({"params": list(decoder.parameters()), "lr": args.decoder_lr})
            if is_main:
                print(f"Joint fine-tuning: predictor LR={args.lr}, decoder LR={args.decoder_lr}")
        elif train_decoder_in_phase2:
            param_groups.append({"params": list(decoder.parameters()), "lr": args.decoder_lr})
        else:
            # Freeze decoder
            if decoder is not None:
                for p in decoder.parameters():
                    p.requires_grad = False
                decoder.eval()
                if is_main:
                    print("Decoder frozen (feature-only predictor training)")
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
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

    # Log initial PCA visualization to W&B
    if is_main and wandb_run:
        log_pca_visualization(encoder, train_loader, device, wandb_run, args.img_size)

    # Mixed precision: BF16 (same exponent range as FP32, no overflow/underflow)
    # FP16 causes NaN in cosine similarity due to norm overflow on random init predictions
    scaler = torch.amp.GradScaler('cuda', enabled=False)

    # Epoch-0 gate: residual + zero-init output_proj must match Copy-Last (~0.45 SmoothL1)
    if not args.pretrain_decoder and args.init_eval:
        init_stats = compute_val_loss(
            predictor, encoder, val_loader, device, is_main,
            loss_type=args.loss_type,
            lambda_cosine=args.lambda_cosine,
            lambda_l2=args.lambda_l2,
            return_details=True,
            frames_ctx_from_rgb=bool(args.use_motion and args.motion_from in ("rgb", "both")),
        )
        if is_main:
            print(
                "Init val (Copy-Last gate): "
                f"pred={init_stats['feat_loss']:.4f}  copy={init_stats['copy_last']:.4f}  "
                f"moving pred/copy={init_stats['moving_pred']:.4f}/{init_stats['moving_copy']:.4f}"
            )
            if wandb_run:
                wandb_run.log({
                    "epoch": -1,
                    "val/feat_loss": init_stats["feat_loss"],
                    "val/copy_last": init_stats["copy_last"],
                    "val/moving_pred": init_stats["moving_pred"],
                    "val/moving_copy": init_stats["moving_copy"],
                    "val/static_pred": init_stats["static_pred"],
                    "val/static_copy": init_stats["static_copy"],
                })
            skip_gap = abs(init_stats["feat_loss"] - init_stats["copy_last"])
            if args.residual and args.abort_if_skip_broken and skip_gap > args.skip_gate_tol:
                print(
                    f"ABORT: residual skip is not Copy-Last "
                    f"(|pred-copy|={skip_gap:.4f} > {args.skip_gate_tol}). "
                    "Not spending a GPU day on a broken skip."
                )
                if wandb_run:
                    wandb_run.finish()
                sys.exit(2)

    # Training loop
    best_psnr = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(args.epochs):
        if not args.pretrain_decoder:
            predictor.train()
        if decoder is not None and (args.joint_finetune or args.lambda_pixel > 0 or args.lambda_lpips > 0):
            decoder.train()
        elif decoder is not None:
            decoder.eval()  # frozen decoder stays in eval mode

        epoch_losses = {
            "feat": [], "feat_unweighted": [], "copy": [],
            "moving_pred": [], "moving_copy": [],
            "pixel": [], "lpips": [], "total": [],
        }
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main)

        for batch_idx, (past_frames, future_frames) in enumerate(pbar):
            past_frames = past_frames.to(device)
            future_frames = future_frames.to(device)

            B, T_c, C, H, W = past_frames.shape
            T_f = future_frames.shape[1]

            if args.pretrain_decoder:
                # Phase 1: train decoder to reconstruct pixels from frozen DINOv2 features
                with torch.no_grad():
                    all_frames = torch.cat([past_frames, future_frames], dim=1)
                    all_frames_flat = all_frames.reshape(B * (T_c + T_f), C, H, W)
                    all_feats = encoder(all_frames_flat)
                    all_feats = all_feats.reshape(B, T_c + T_f, n_patches, feat_dim)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred_pixels = decoder(all_feats)  # reconstruct all frames
                    target_pixels = torch.cat([past_frames, future_frames], dim=1)

                    pixel_loss = F.l1_loss(pred_pixels.float(), target_pixels)
                    lpips_loss = torch.tensor(0.0, device=device)
                    if perceptual_loss is not None:
                        pp = pred_pixels.reshape(B * (T_c + T_f), C, H, W).float()
                        tp = target_pixels.reshape(B * (T_c + T_f), C, H, W)
                        lpips_loss = perceptual_loss(pp, tp)

                    total_loss = args.lambda_pixel * pixel_loss + args.lambda_lpips * lpips_loss

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_losses["pixel"].append(pixel_loss.item())
                epoch_losses["lpips"].append(lpips_loss.item())
                epoch_losses["total"].append(total_loss.item())

                if is_main and batch_idx % 50 == 0:
                    pbar.set_postfix({
                        "pix": f"{pixel_loss.item():.4f}",
                        "lpips": f"{lpips_loss.item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })

            else:
                # Phase 2/3: multi-step predictor training with cosine+L2 loss
                n_steps_train = args.n_pred_steps_train if not args.pretrain_decoder else 1

                # Encode all frames (context + future) with frozen encoder
                with torch.no_grad():
                    all_frames = torch.cat([past_frames, future_frames], dim=1)
                    all_frames_flat = all_frames.reshape(B * (T_c + T_f), C, H, W)
                    all_feats = encoder(all_frames_flat)
                    all_feats = all_feats.reshape(B, T_c + T_f, n_patches, feat_dim)

                    # Context = first T_c frames, targets = next n_steps_train frames
                    context_feats = all_feats[:, :T_c].detach()  # (B, T_c, N, D)
                    target_feats = all_feats[:, T_c:T_c + n_steps_train].detach()  # (B, n_steps, N, D)

                # Scheduled sampling: decide whether to use GT or predicted context for multi-step
                use_teacher_forcing = True
                if args.scheduled_sampling and n_steps_train > 1:
                    # Linear decay from 1.0 to 0.5 over ss_decay_epochs
                    ss_prob = max(0.5, 1.0 - epoch / max(args.ss_decay_epochs, 1))
                    use_teacher_forcing = (np.random.rand() < ss_prob)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred_model = predictor.module if hasattr(predictor, "module") else predictor
                    frames_ctx = motion_frames_ctx(args, past_frames)

                    if n_steps_train == 1 or use_teacher_forcing:
                        pred_future_feats = predictor(
                            context_feats, n_predict=n_steps_train, frames_ctx=frames_ctx
                        )
                    else:
                        pred_future_feats = pred_model.forward_autoregressive(
                            context_feats, n_steps=n_steps_train, frames_ctx=frames_ctx
                        )

                    motion_w = None
                    if args.motion_weight:
                        motion_w = patch_motion_weights(
                            context_feats, target_feats, floor=args.motion_weight_floor
                        )
                    feat_unweighted = feature_loss(
                        pred_future_feats, target_feats,
                        loss_type=args.loss_type,
                        lambda_cosine=args.lambda_cosine,
                        lambda_l2=args.lambda_l2,
                    )
                    feat_loss = feature_loss(
                        pred_future_feats, target_feats,
                        loss_type=args.loss_type,
                        lambda_cosine=args.lambda_cosine,
                        lambda_l2=args.lambda_l2,
                        motion_weights=motion_w,
                    )
                    total_loss = args.lambda_feat * feat_loss

                    # VICReg regularizer (prevents collapse to mean prediction)
                    vicreg_loss_val = torch.tensor(0.0, device=device)
                    if args.vicreg_lambda > 0:
                        vicreg_loss_val = vicreg_loss(
                            pred_future_feats,
                            var_coeff=args.vicreg_var_coeff,
                            cov_coeff=args.vicreg_cov_coeff,
                        )
                        total_loss = total_loss + args.vicreg_lambda * vicreg_loss_val

                    # Pixel + LPIPS loss (perceptual signal to predictor)
                    pixel_loss = torch.tensor(0.0, device=device)
                    lpips_loss = torch.tensor(0.0, device=device)
                    if args.lambda_pixel > 0 and decoder is not None:
                        # Decode last predicted step for pixel loss
                        pred_pixels = decoder(pred_future_feats[:, -1:].float())
                        target_pixels = future_frames[:, n_steps_train - 1:n_steps_train] \
                            if T_f >= n_steps_train else future_frames[:, -1:]
                        pixel_loss = F.l1_loss(pred_pixels, target_pixels)
                        total_loss = total_loss + args.lambda_pixel * pixel_loss
                    if args.lambda_lpips > 0 and perceptual_loss is not None:
                        pp = pred_pixels.reshape(B, C, H, W)
                        tp = target_pixels.reshape(B, C, H, W)
                        lpips_loss = perceptual_loss(pp, tp)
                        total_loss = total_loss + args.lambda_lpips * lpips_loss

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                pred_params = predictor.module if hasattr(predictor, "module") else predictor
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(pred_params.parameters(), max_norm=1.0)
                if args.joint_finetune and decoder is not None:
                    dec_params = decoder.module if hasattr(decoder, "module") else decoder
                    torch.nn.utils.clip_grad_norm_(dec_params.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_losses["feat"].append(feat_loss.item())
                epoch_losses["feat_unweighted"].append(feat_unweighted.item())
                epoch_losses["pixel"].append(pixel_loss.item())
                epoch_losses["lpips"].append(lpips_loss.item())
                epoch_losses["total"].append(total_loss.item())
                with torch.no_grad():
                    copy_pred = context_feats[:, -1:].expand_as(target_feats)
                    copy_l = F.smooth_l1_loss(copy_pred, target_feats, beta=0.1)
                    epoch_losses["copy"].append(copy_l.item())
                    mag = (target_feats[:, 0] - context_feats[:, -1]).norm(dim=-1)
                    moving = mag > mag.median()
                    elem_p = F.smooth_l1_loss(
                        pred_future_feats[:, 0], target_feats[:, 0], beta=0.1, reduction="none"
                    ).mean(-1)
                    elem_c = F.smooth_l1_loss(
                        copy_pred[:, 0], target_feats[:, 0], beta=0.1, reduction="none"
                    ).mean(-1)
                    if moving.any():
                        epoch_losses["moving_pred"].append(elem_p[moving].mean().item())
                        epoch_losses["moving_copy"].append(elem_c[moving].mean().item())

                if is_main and batch_idx % 50 == 0:
                    pbar.set_postfix({
                        "feat": f"{feat_loss.item():.4f}",
                        "pix": f"{pixel_loss.item():.4f}",
                        "lpips": f"{lpips_loss.item():.4f}",
                        "vic": f"{vicreg_loss_val.item():.4f}" if args.vicreg_lambda > 0 else "",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })

        # Log epoch metrics
        if is_main:
            avg_feat = np.mean(epoch_losses["feat"]) if epoch_losses["feat"] else 0.0
            avg_feat_uw = np.mean(epoch_losses["feat_unweighted"]) if epoch_losses["feat_unweighted"] else avg_feat
            avg_copy = np.mean(epoch_losses["copy"]) if epoch_losses["copy"] else float("nan")
            avg_mv_p = np.mean(epoch_losses["moving_pred"]) if epoch_losses["moving_pred"] else float("nan")
            avg_mv_c = np.mean(epoch_losses["moving_copy"]) if epoch_losses["moving_copy"] else float("nan")
            avg_pixel = np.mean(epoch_losses["pixel"])
            avg_lpips = np.mean(epoch_losses["lpips"]) if epoch_losses["lpips"] else 0.0
            avg_total = np.mean(epoch_losses["total"])
            if args.pretrain_decoder:
                print(f"Epoch {epoch}: pixel={avg_pixel:.4f}, lpips={avg_lpips:.4f}, total={avg_total:.4f}")
            else:
                print(
                    f"Epoch {epoch}: feat={avg_feat:.4f} (unw={avg_feat_uw:.4f}) "
                    f"copy={avg_copy:.4f} moving pred/copy={avg_mv_p:.4f}/{avg_mv_c:.4f} "
                    f"pixel={avg_pixel:.4f}"
                )

            if wandb_run:
                log_dict = {
                    "epoch": epoch,
                    "train/pixel_loss": avg_pixel,
                    "train/lpips_loss": avg_lpips,
                    "train/total_loss": avg_total,
                    "train/lr": scheduler.get_last_lr()[0],
                }
                if not args.pretrain_decoder:
                    log_dict["train/feat_loss"] = avg_feat
                    log_dict["train/feat_unweighted"] = avg_feat_uw
                    log_dict["train/copy_last"] = avg_copy
                    log_dict["train/moving_pred"] = avg_mv_p
                    log_dict["train/moving_copy"] = avg_mv_c
                wandb_run.log(log_dict)

        # Validation feature loss (overfitting detection)
        if not args.pretrain_decoder and (epoch + 1) % args.val_interval == 0:
            val_stats = compute_val_loss(predictor, encoder, val_loader, device, is_main,
                                        loss_type=args.loss_type,
                                        lambda_cosine=args.lambda_cosine,
                                        lambda_l2=args.lambda_l2,
                                        return_details=True,
                                        frames_ctx_from_rgb=bool(
                                            args.use_motion and args.motion_from in ("rgb", "both")
                                        ))
            val_loss = val_stats["feat_loss"]
            if is_main:
                print(
                    f"  Val feat={val_loss:.4f} copy={val_stats['copy_last']:.4f} "
                    f"delta(copy-pred)={val_stats['copy_last'] - val_loss:.4f} "
                    f"(train: {avg_feat:.4f}, gap: {val_loss - avg_feat:.4f})"
                )
                if wandb_run:
                    wandb_run.log({
                        "epoch": epoch,
                        "val/feat_loss": val_loss,
                        "val/copy_last": val_stats["copy_last"],
                        "val/overfit_gap": val_loss - avg_feat,
                        "val/moving_pred": val_stats["moving_pred"],
                        "val/moving_copy": val_stats["moving_copy"],
                    })

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(
                        predictor=predictor, decoder=decoder, optimizer=optimizer,
                        epoch=epoch, args=args,
                        path=os.path.join(args.output_dir, "best_val_model.pth"),
                    )
                    print(f"  *** New best val loss: {best_val_loss:.4f} ***")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                        break

        # Phase 1 validation pixel loss
        if args.pretrain_decoder and (epoch + 1) % args.val_interval == 0:
            val_pixel = compute_val_pixel_loss(decoder, encoder, val_loader, device, is_main)
            if is_main:
                print(f"  Val pixel loss: {val_pixel:.4f} (train: {avg_pixel:.4f}, gap: {val_pixel - avg_pixel:.4f})")
                if wandb_run:
                    wandb_run.log({"epoch": epoch, "val/pixel_loss": val_pixel, "val/pixel_overfit_gap": val_pixel - avg_pixel})

                if val_pixel < best_val_loss:
                    best_val_loss = val_pixel
                    patience_counter = 0
                    dec = decoder.module if hasattr(decoder, "module") else decoder
                    torch.save({
                        "epoch": epoch, "decoder": dec.state_dict(), "args": vars(args),
                    }, os.path.join(args.output_dir, "best_val_decoder.pth"))
                    print(f"  *** New best val pixel loss: {best_val_loss:.4f} ***")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                        break

        # Evaluation (skip during Phase 1 decoder pretraining)
        if not args.pretrain_decoder and ((epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1):
            eval_results = evaluate(
                predictor=predictor,
                decoder=decoder,
                encoder=encoder,
                dataloader=eval_test_loader,
                device=device,
                n_pred_steps=args.n_pred_steps,
                lpips_fn=lpips_fn,
                is_main=is_main,
                max_clips=args.eval_max_clips,
            )

            if is_main:
                print(f"  Eval: PSNR={eval_results['psnr_avg']:.2f}, SSIM={eval_results['ssim_avg']:.4f}", end="")
                if "lpips_avg" in eval_results:
                    print(f", LPIPS={eval_results['lpips_avg']:.4f}", end="")
                print()

                if wandb_run:
                    log_dict = {
                        "epoch": epoch,
                        "val/psnr": eval_results["psnr_avg"],
                        "val/ssim": eval_results["ssim_avg"],
                    }
                    if "lpips_avg" in eval_results:
                        log_dict["val/lpips"] = eval_results["lpips_avg"]
                    # Per-timestep curves
                    if "psnr_per_step" in eval_results:
                        for t, v in enumerate(eval_results["psnr_per_step"]):
                            log_dict[f"val/psnr_step_{t}"] = float(v)
                        for t, v in enumerate(eval_results["ssim_per_step"]):
                            log_dict[f"val/ssim_step_{t}"] = float(v)
                        if "lpips_per_step" in eval_results:
                            for t, v in enumerate(eval_results["lpips_per_step"]):
                                log_dict[f"val/lpips_step_{t}"] = float(v)
                        # Log per-timestep curve plot
                        fig = plot_per_timestep_curves(eval_results)
                        log_dict["val/per_timestep_curves"] = wandb.Image(fig)
                        plt.close(fig)
                    # Log prediction visualizations every few evals
                    if (epoch + 1) % (args.eval_interval * 4) == 0 or epoch == args.epochs - 1:
                        log_sample_predictions(predictor, decoder, encoder, eval_test_loader, device, wandb_run, n_steps=min(args.n_pred_steps, 20), epoch=epoch)
                    wandb_run.log(log_dict)

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
                if args.pretrain_decoder:
                    # Phase 1: only save decoder
                    dec = decoder.module if hasattr(decoder, "module") else decoder
                    torch.save({
                        "epoch": epoch,
                        "decoder": dec.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                    }, os.path.join(args.output_dir, f"decoder_epoch{epoch+1}.pth"))
                else:
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
        if args.pretrain_decoder:
            dec = decoder.module if hasattr(decoder, "module") else decoder
            torch.save({
                "epoch": args.epochs - 1,
                "decoder": dec.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }, os.path.join(args.output_dir, "decoder_final.pth"))
            print(f"\nPhase 1 complete! Decoder saved to: {args.output_dir}")
        else:
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


def feature_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "cosine_l2",
                 lambda_cosine: float = 1.0, lambda_l2: float = 0.5,
                 motion_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute feature prediction loss matching DINOv2 feature geometry.

    DINOv2 features live on a hypersphere — direction matters more than magnitude.
    cosine_l2 combines cosine similarity (directional alignment) with L2 (magnitude).

    Args:
        pred: (B, T, N, D) predicted features
        target: (B, T, N, D) target features
        loss_type: "cosine_l2", "cosine", "l2", or "smooth_l1"
        lambda_cosine: weight for cosine component
        lambda_l2: weight for L2 component
    """
    if loss_type == "smooth_l1":
        elem = F.smooth_l1_loss(pred, target, beta=0.1, reduction="none").mean(dim=-1)
        return _reduce_token_loss(elem, motion_weights)

    if loss_type == "l2":
        elem = (pred - target).pow(2).mean(dim=-1)
        return _reduce_token_loss(elem, motion_weights)

    if loss_type == "cosine":
        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        return _reduce_token_loss(1.0 - cos_sim, motion_weights)

    # cosine_l2: combine directional + magnitude
    cos_sim = F.cosine_similarity(pred, target, dim=-1)  # (B, T, N)
    cos_loss = _reduce_token_loss(1.0 - cos_sim, motion_weights)
    l2_loss = _reduce_token_loss((pred - target).pow(2).mean(dim=-1), motion_weights)
    return lambda_cosine * cos_loss + lambda_l2 * l2_loss


def _reduce_token_loss(per_token: torch.Tensor, motion_weights: Optional[torch.Tensor]) -> torch.Tensor:
    """per_token: (B, T, N)."""
    if motion_weights is None:
        return per_token.mean()
    return (per_token * motion_weights).mean()


def vicreg_loss(pred: torch.Tensor, var_coeff: float = 1.0, cov_coeff: float = 1.0,
                target_std: float = 1.0) -> torch.Tensor:
    """VICReg regularizer on predicted features to prevent collapse.

    Variance: encourages per-dimension std > target_std (prevents constant predictions)
    Covariance: decorrelates feature dimensions (prevents all-dims-same collapse)

    Args:
        pred: (B, T, N, D) predicted features
        var_coeff: weight for variance term
        cov_coeff: weight for covariance term
        target_std: minimum target std per dimension
    """
    B, T, N, D = pred.shape
    z = pred.reshape(-1, D)
    z = z.float()

    # Variance: per-dimension std should be > target_std
    z_std = z.std(dim=0)
    var_loss = F.relu(target_std - z_std).mean()

    # Covariance: off-diagonal of cov matrix should be 0
    z_centered = z - z.mean(dim=0, keepdim=True)
    n = z.shape[0]
    cov = (z_centered.T @ z_centered) / max(n - 1, 1)
    cov_loss = (cov - torch.diag(torch.diagonal(cov))).pow(2).sum() / D

    return var_coeff * var_loss + cov_coeff * cov_loss


@torch.no_grad()
def compute_val_loss(predictor, encoder, val_loader, device, is_main=True, max_clips=200,
                     loss_type="cosine_l2", lambda_cosine=1.0, lambda_l2=0.5,
                     return_details=False, frames_ctx_from_rgb=False):
    """Compute validation feature loss (no decoder needed, fast overfitting check).

    Val metric is always unweighted so it is comparable to Copy-Last SmoothL1.
    """
    predictor.eval()
    total_loss = 0.0
    copy_total = 0.0
    n_batches = 0
    clips = 0
    moving_pred_sum = 0.0
    moving_copy_sum = 0.0
    moving_n = 0
    static_pred_sum = 0.0
    static_copy_sum = 0.0
    static_n = 0
    pbar = tqdm(val_loader, desc="Val loss", disable=not is_main)
    for past_frames, future_frames in pbar:
        if max_clips is not None and clips >= max_clips:
            break
        past_frames = past_frames.to(device)
        future_frames = future_frames.to(device)
        B, T_c, C, H, W = past_frames.shape
        T_f = future_frames.shape[1]

        all_frames = torch.cat([past_frames, future_frames], dim=1)
        all_flat = all_frames.reshape(B * (T_c + T_f), C, H, W)
        all_feats = encoder(all_flat).reshape(B, T_c + T_f, encoder.n_patches, encoder.feat_dim)

        context_feats = all_feats[:, :T_c]
        target_feat = all_feats[:, T_c:T_c+1]
        frames_ctx = past_frames if frames_ctx_from_rgb else None

        pred_model = predictor.module if hasattr(predictor, "module") else predictor
        pred_feats = pred_model(context_feats, n_predict=1, frames_ctx=frames_ctx)
        copy_feats = context_feats[:, -1:]

        loss = feature_loss(pred_feats, target_feat, loss_type=loss_type,
                            lambda_cosine=lambda_cosine, lambda_l2=lambda_l2)
        copy_loss = F.smooth_l1_loss(copy_feats, target_feat, beta=0.1)
        total_loss += loss.item()
        copy_total += copy_loss.item()
        n_batches += 1
        clips += B

        mag = (target_feat[:, 0] - context_feats[:, -1]).norm(dim=-1)
        moving = mag > mag.median()
        elem_p = F.smooth_l1_loss(
            pred_feats[:, 0], target_feat[:, 0], beta=0.1, reduction="none"
        ).mean(-1)
        elem_c = F.smooth_l1_loss(
            copy_feats[:, 0], target_feat[:, 0], beta=0.1, reduction="none"
        ).mean(-1)
        if moving.any():
            moving_pred_sum += elem_p[moving].sum().item()
            moving_copy_sum += elem_c[moving].sum().item()
            moving_n += int(moving.sum().item())
        static = ~moving
        if static.any():
            static_pred_sum += elem_p[static].sum().item()
            static_copy_sum += elem_c[static].sum().item()
            static_n += int(static.sum().item())

    predictor.train()
    feat_loss = total_loss / max(n_batches, 1)
    if not return_details:
        return feat_loss
    return {
        "feat_loss": feat_loss,
        "copy_last": copy_total / max(n_batches, 1),
        "moving_pred": moving_pred_sum / max(moving_n, 1),
        "moving_copy": moving_copy_sum / max(moving_n, 1),
        "static_pred": static_pred_sum / max(static_n, 1),
        "static_copy": static_copy_sum / max(static_n, 1),
    }


@torch.no_grad()
def compute_val_pixel_loss(decoder, encoder, val_loader, device, is_main=True, max_clips=200):
    """Compute validation pixel reconstruction loss for Phase 1 decoder."""
    decoder.eval()
    total_loss = 0.0
    n_batches = 0
    clips = 0
    pbar = tqdm(val_loader, desc="Val pixel", disable=not is_main)
    for past_frames, future_frames in pbar:
        if max_clips is not None and clips >= max_clips:
            break
        past_frames = past_frames.to(device)
        future_frames = future_frames.to(device)
        B, T_c, C, H, W = past_frames.shape
        T_f = future_frames.shape[1]

        all_frames = torch.cat([past_frames, future_frames], dim=1)
        all_flat = all_frames.reshape(B * (T_c + T_f), C, H, W)
        all_feats = encoder(all_flat).reshape(B, T_c + T_f, encoder.n_patches, encoder.feat_dim)

        dec = decoder.module if hasattr(decoder, "module") else decoder
        pred_pixels = dec(all_feats)
        target_pixels = all_frames

        loss = F.l1_loss(pred_pixels, target_pixels)
        total_loss += loss.item()
        n_batches += 1
        clips += B

    decoder.train()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(predictor, decoder, encoder, dataloader, device, n_pred_steps, lpips_fn, is_main=True, max_clips=None):
    """Evaluate the model on the test set.
    
    If n_pred_steps=1: single-step eval (predict 1 frame from all context frames)
    If n_pred_steps>1: autoregressive unrolling
    """
    predictor.eval()
    if decoder is not None:
        decoder.eval()

    psnr_per_step = []
    ssim_per_step = []
    lpips_per_step = []
    n_total_steps = 0
    clips_evaluated = 0

    pbar = tqdm(dataloader, desc="Evaluating", disable=not is_main)
    for past_frames, future_frames in pbar:
        if max_clips is not None and clips_evaluated >= max_clips:
            break

        past_frames = past_frames.to(device)
        future_frames = future_frames.to(device)

        B, T_c, C, H, W = past_frames.shape
        T_f = future_frames.shape[1]
        n_steps = min(n_pred_steps, T_f)
        n_total_steps = n_steps

        if n_steps == 1:
            # Single-step eval: use all available frames except last as context
            all_frames = torch.cat([past_frames, future_frames], dim=1)
            all_flat = all_frames.reshape(B * (T_c + T_f), C, H, W)
            all_feats = encoder(all_flat).reshape(B, T_c + T_f, encoder.n_patches, encoder.feat_dim)
            
            # Use all frames except last as context, predict last frame
            context_feats = all_feats[:, :-1]
            target_feat = all_feats[:, -1:]
            
            pred_model = predictor.module if hasattr(predictor, "module") else predictor
            pred_feats = pred_model(context_feats, n_predict=1)
            
            # For metrics, compare decoded prediction vs last future frame
            if decoder is not None:
                dec = decoder.module if hasattr(decoder, "module") else decoder
                pred_pixels = torch.clamp(dec(pred_feats.float()), -1, 1)
                gt_pixels = future_frames[:, -1:]
            else:
                continue
            
            # Single step metrics
            p = psnr(pred_pixels[:, 0], gt_pixels[:, 0]).mean().item()
            s = ssim(pred_pixels[:, 0], gt_pixels[:, 0]).mean().item()
            if len(psnr_per_step) <= 0:
                psnr_per_step.append([])
                ssim_per_step.append([])
                if lpips_fn is not None:
                    lpips_per_step.append([])
            psnr_per_step[0].append(p)
            ssim_per_step[0].append(s)
            if lpips_fn is not None:
                l = lpips_fn(pred_pixels[:, 0], gt_pixels[:, 0]).mean().item()
                lpips_per_step[0].append(l)
        else:
            # Autoregressive eval (for multi-step evaluation)
            past_flat = past_frames.reshape(B * T_c, C, H, W)
            context_feats = encoder(past_flat).reshape(B, T_c, encoder.n_patches, encoder.feat_dim)

            pred_model = predictor.module if hasattr(predictor, "module") else predictor
            pred_feats = pred_model.forward_autoregressive(context_feats, n_steps=n_steps)

            if decoder is not None:
                dec = decoder.module if hasattr(decoder, "module") else decoder
                pred_pixels = torch.clamp(dec(pred_feats.float()), -1, 1)
            else:
                continue

            for step in range(n_steps):
                pred_step = pred_pixels[:, step]
                gt_step = future_frames[:, step]
                p = psnr(pred_step, gt_step).mean().item()
                s = ssim(pred_step, gt_step).mean().item()
                if len(psnr_per_step) <= step:
                    psnr_per_step.append([])
                    ssim_per_step.append([])
                    if lpips_fn is not None:
                        lpips_per_step.append([])
                psnr_per_step[step].append(p)
                ssim_per_step[step].append(s)
                if lpips_fn is not None:
                    l = lpips_fn(pred_step, gt_step).mean().item()
                    lpips_per_step[step].append(l)

        clips_evaluated += B

    results = {
        "psnr_avg": np.mean([v for sublist in psnr_per_step for v in sublist]) if psnr_per_step else 0.0,
        "ssim_avg": np.mean([v for sublist in ssim_per_step for v in sublist]) if ssim_per_step else 0.0,
        "psnr_per_step": np.array([np.mean(sublist) for sublist in psnr_per_step]) if psnr_per_step else np.array([]),
        "ssim_per_step": np.array([np.mean(sublist) for sublist in ssim_per_step]) if ssim_per_step else np.array([]),
    }
    if lpips_per_step:
        results["lpips_avg"] = np.mean([v for sublist in lpips_per_step for v in sublist])
        results["lpips_per_step"] = np.array([np.mean(sublist) for sublist in lpips_per_step])

    return results


def log_pca_visualization(encoder, train_loader, device, wandb_run, img_size):
    """Extract features from sample frames, compute PCA, log to W&B."""
    print("  Logging PCA visualization to W&B...")
    encoder.eval()
    all_feats = []
    all_frames = []
    count = 0
    with torch.no_grad():
        for past_frames, _ in train_loader:
            past_frames = past_frames.to(device)
            B, T, C, H, W = past_frames.shape
            frames_flat = past_frames.reshape(B * T, C, H, W)
            feats = encoder(frames_flat)
            all_feats.append(feats.cpu())
            all_frames.append(frames_flat.cpu())
            count += feats.shape[0]
            if count >= 200:
                break
    all_feats = torch.cat(all_feats, dim=0)  # (N, n_patches, D)
    all_frames = torch.cat(all_frames, dim=0)  # (N, C, H, W)
    N, n_patches, D = all_feats.shape
    grid_size = int(np.sqrt(n_patches))
    # PCA via SVD
    feats_flat = all_feats.reshape(N * n_patches, D).numpy()
    mean = feats_flat.mean(axis=0, keepdims=True)
    centered = feats_flat - mean
    U, S, Vh = np.linalg.svd(centered, full_matrices=False)
    variance_explained = (S ** 2) / (S ** 2).sum()
    cumvar = np.cumsum(variance_explained)
    # Project to 3D
    proj_3d = centered @ Vh[:3].T
    proj_3d = proj_3d.reshape(N, n_patches, 3)
    proj_min = proj_3d.min(axis=(0, 1), keepdims=True)
    proj_max = proj_3d.max(axis=(0, 1), keepdims=True)
    proj_3d_norm = (proj_3d - proj_min) / (proj_max - proj_min + 1e-8)
    # Plot: top row = original frames, bottom row = PCA
    n_vis = min(8, N)
    fig, axes = plt.subplots(2, n_vis, figsize=(3 * n_vis, 6))
    if n_vis == 1:
        axes = axes.reshape(2, 1)
    for i in range(n_vis):
        img = all_frames[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Frame {i}", fontsize=9)
        axes[0, i].axis("off")
        pca_img = proj_3d_norm[i].reshape(grid_size, grid_size, 3)
        axes[1, i].imshow(pca_img)
        axes[1, i].set_title("PCA 3-comp", fontsize=9)
        axes[1, i].axis("off")
    plt.suptitle(f"PCA of {encoder.__class__.__name__} Features on JIGSAWS", fontsize=13, fontweight="bold")
    plt.tight_layout()
    wandb_run.log({"analysis/pca_visualization": wandb.Image(fig), "epoch": 0})
    plt.close(fig)
    # Variance plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(range(min(50, len(variance_explained))), variance_explained[:50], color="steelblue")
    ax1.set_title("Individual Variance")
    ax1.set_xlabel("Component")
    ax2.plot(range(len(cumvar)), cumvar, color="steelblue", linewidth=2)
    ax2.axhline(y=0.90, color="orange", linestyle="--", alpha=0.7)
    ax2.axhline(y=0.95, color="green", linestyle="--", alpha=0.7)
    ax2.set_title("Cumulative Variance")
    ax2.set_xlabel("Components")
    plt.suptitle(f"{encoder.__class__.__name__} — PCA Variance Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    wandb_run.log({"analysis/variance_curve": wandb.Image(fig2), "epoch": 0})
    plt.close(fig2)
    print(f"  PCA: 90% var at {int(np.searchsorted(cumvar, 0.90))+1} dims, 95% at {int(np.searchsorted(cumvar, 0.95))+1} dims")


def plot_per_timestep_curves(eval_results):
    """Plot PSNR/SSIM/LPIPS per prediction timestep."""
    has_lpips = "lpips_per_step" in eval_results
    n_cols = 3 if has_lpips else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    steps = range(len(eval_results["psnr_per_step"]))
    axes[0].plot(steps, eval_results["psnr_per_step"], "o-", color="steelblue", linewidth=2)
    axes[0].set_title("PSNR per Timestep")
    axes[0].set_xlabel("Future Step")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, eval_results["ssim_per_step"], "s-", color="darkorange", linewidth=2)
    axes[1].set_title("SSIM per Timestep")
    axes[1].set_xlabel("Future Step")
    axes[1].set_ylabel("SSIM")
    axes[1].grid(True, alpha=0.3)
    if has_lpips:
        axes[2].plot(steps, eval_results["lpips_per_step"], "^-", color="firebrick", linewidth=2)
        axes[2].set_title("LPIPS per Timestep")
        axes[2].set_xlabel("Future Step")
        axes[2].set_ylabel("LPIPS (lower=better)")
        axes[2].grid(True, alpha=0.3)
    plt.suptitle("Per-Timestep Prediction Quality", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def log_sample_predictions(predictor, decoder, encoder, test_loader, device, wandb_run, n_steps=10, epoch=0):
    """Generate predictions on a few test samples and log to W&B."""
    if decoder is None:
        return
    print(f"  Logging sample predictions to W&B (epoch {epoch})...")
    predictor.eval()
    decoder.eval()
    encoder.eval()
    # Get a few test samples
    samples_logged = 0
    max_samples = 4
    with torch.no_grad():
        for past_frames, future_frames in test_loader:
            if samples_logged >= max_samples:
                break
            # Take only 2 samples per batch to avoid OOM
            past_frames = past_frames[:2].to(device)
            future_frames = future_frames[:2].to(device)
            B, T_c, C, H, W = past_frames.shape
            T_f = future_frames.shape[1]
            n_pred = min(n_steps, T_f)
            
            if n_pred == 1:
                # Single-step: use all frames except last as context
                all_frames = torch.cat([past_frames, future_frames], dim=1)
                all_flat = all_frames.reshape(B * (T_c + T_f), C, H, W)
                all_feats = encoder(all_flat).reshape(B, T_c + T_f, encoder.n_patches, encoder.feat_dim)
                context_feats = all_feats[:, :-1]
                pred_model = predictor.module if hasattr(predictor, "module") else predictor
                pred_feats = pred_model(context_feats, n_predict=1)
            else:
                # Autoregressive
                past_flat = past_frames.reshape(B * T_c, C, H, W)
                context_feats = encoder(past_flat).reshape(B, T_c, encoder.n_patches, encoder.feat_dim)
                pred_model = predictor.module if hasattr(predictor, "module") else predictor
                pred_feats = pred_model.forward_autoregressive(context_feats, n_steps=n_pred)
            
            # Decode
            dec = decoder.module if hasattr(decoder, "module") else decoder
            pred_pixels = dec(pred_feats.float())
            # Denormalize
            mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(device)
            std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(device)
            pred_vis = (pred_pixels * std_t + mean_t).clamp(0, 1)
            gt_vis = (future_frames * std_t + mean_t).clamp(0, 1)
            past_vis = (past_frames * std_t + mean_t).clamp(0, 1)
            # Create comparison figure for first sample in batch
            for b in range(min(B, max_samples - samples_logged)):
                fig, axes = plt.subplots(3, n_pred, figsize=(2 * n_pred, 6))
                if n_pred == 1:
                    axes = axes.reshape(3, 1)
                # Row 1: past (context) frames
                for t in range(T_c):
                    if t < n_pred:
                        axes[0, t].imshow(past_vis[b, t].cpu().permute(1, 2, 0).numpy())
                        axes[0, t].set_title(f"t-{T_c-t}", fontsize=8)
                        axes[0, t].axis("off")
                # Row 2: ground truth future
                for t in range(n_pred):
                    axes[1, t].imshow(gt_vis[b, t].cpu().permute(1, 2, 0).numpy())
                    axes[1, t].set_title(f"GT t+{t+1}", fontsize=8)
                    axes[1, t].axis("off")
                # Row 3: predicted future
                for t in range(n_pred):
                    axes[2, t].imshow(pred_vis[b, t].cpu().permute(1, 2, 0).numpy())
                    axes[2, t].set_title(f"Pred t+{t+1}", fontsize=8)
                    axes[2, t].axis("off")
                plt.suptitle(f"Sample {samples_logged} — Past | GT Future | Predicted Future", fontsize=12, fontweight="bold")
                plt.tight_layout()
                wandb_run.log({f"predictions/sample_{samples_logged}": wandb.Image(fig), "epoch": epoch})
                plt.close(fig)
                samples_logged += 1
            break  # One batch is enough


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
