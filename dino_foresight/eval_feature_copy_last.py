#!/usr/bin/env python3
"""Feature-space Copy-Last vs predictor on JIGSAWS LOUO.

Compares SmoothL1 / L2 / cosine of predicted DINOv2-PCA features against
copying the last context frame's features. No RGB decoder.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dino_foresight.data import build_dataloaders
from dino_foresight.encoders import DINOv2Encoder
from dino_foresight.predictor import build_predictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor_ckpt", type=str, required=True)
    parser.add_argument("--pca_ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--encoder_model", type=str, default="vitb14_reg")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_context", type=int, default=4)
    parser.add_argument("--n_pred_steps", type=str, default="1,5,10")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pca_dim", type=int, default=1152)
    parser.add_argument("--hidden_dim", type=int, default=1152)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "both"])
    parser.add_argument("--max_clips", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def feature_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """pred/target: (B, T, N, D). Returns batch-mean scalars."""
    smooth = F.smooth_l1_loss(pred, target, beta=0.1).item()
    l2 = F.mse_loss(pred, target).item()
    pred_f = pred.reshape(-1, pred.shape[-1])
    tgt_f = target.reshape(-1, target.shape[-1])
    cosine = F.cosine_similarity(pred_f, tgt_f, dim=-1).mean().item()
    return {"smooth_l1": smooth, "l2": l2, "cosine": cosine}


def per_step_smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> list:
    steps = []
    for t in range(pred.shape[1]):
        steps.append(F.smooth_l1_loss(pred[:, t], target[:, t], beta=0.1).item())
    return steps


@torch.no_grad()
def eval_split(encoder, predictor, loader, device, n_steps, max_clips):
    encoder.eval()
    predictor.eval()

    copy_vals = {"smooth_l1": [], "l2": [], "cosine": []}
    pred_vals = {"smooth_l1": [], "l2": [], "cosine": []}
    copy_steps = [[] for _ in range(n_steps)]
    pred_steps = [[] for _ in range(n_steps)]
    clips = 0

    pbar = tqdm(loader, desc=f"feature eval t={n_steps}")
    for past_frames, future_frames in pbar:
        if max_clips > 0 and clips >= max_clips:
            break
        past_frames = past_frames.to(device, non_blocking=True)
        future_frames = future_frames.to(device, non_blocking=True)
        B, T_c, C, H, W = past_frames.shape
        T_f = future_frames.shape[1]
        if T_f < n_steps:
            continue

        all_frames = torch.cat([past_frames, future_frames[:, :n_steps]], dim=1)
        flat = all_frames.reshape(B * (T_c + n_steps), C, H, W)
        feats = encoder(flat).reshape(B, T_c + n_steps, encoder.n_patches, encoder.feat_dim)
        context = feats[:, :T_c]
        target = feats[:, T_c:]

        copy_pred = context[:, -1:].expand(-1, n_steps, -1, -1)
        if n_steps == 1:
            model_pred = predictor(context, n_predict=1)
        else:
            model_pred = predictor.forward_autoregressive(context, n_steps=n_steps)

        cm = feature_metrics(copy_pred, target)
        pm = feature_metrics(model_pred, target)
        for k in copy_vals:
            copy_vals[k].append(cm[k])
            pred_vals[k].append(pm[k])

        for t, v in enumerate(per_step_smooth_l1(copy_pred, target)):
            copy_steps[t].append(v)
        for t, v in enumerate(per_step_smooth_l1(model_pred, target)):
            pred_steps[t].append(v)

        clips += B
        pbar.set_postfix({
            "copy": f"{cm['smooth_l1']:.4f}",
            "pred": f"{pm['smooth_l1']:.4f}",
        })

    def mean_dict(d):
        return {k: float(np.mean(v)) if v else float("nan") for k, v in d.items()}

    return {
        "n_clips": clips,
        "copy_last": {
            **mean_dict(copy_vals),
            "smooth_l1_per_step": [float(np.mean(s)) if s else float("nan") for s in copy_steps],
        },
        "predictor": {
            **mean_dict(pred_vals),
            "smooth_l1_per_step": [float(np.mean(s)) if s else float("nan") for s in pred_steps],
        },
    }


def write_markdown(path: Path, results: dict) -> None:
    lines = [
        "# Feature-space Copy-Last vs Track A predictor",
        "",
        "Lower SmoothL1 / L2 is better. Higher cosine is better.",
        "",
        "| Split | Horizon | Method | SmoothL1 | L2 | Cosine | n_clips |",
        "|-------|---------|--------|----------|----|--------|---------|",
    ]
    for split, by_h in results.items():
        for key, block in by_h.items():
            for method in ("copy_last", "predictor"):
                m = block[method]
                lines.append(
                    f"| {split} | {key} | {method} | "
                    f"{m['smooth_l1']:.4f} | {m['l2']:.4f} | {m['cosine']:.4f} | {block['n_clips']} |"
                )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.predictor_ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    hidden_dim = int(ckpt_args.get("hidden_dim", args.hidden_dim))
    num_layers = int(ckpt_args.get("num_layers", args.num_layers))
    num_heads = int(ckpt_args.get("num_heads", args.num_heads))
    n_context = int(ckpt_args.get("n_context", args.n_context))
    pca_dim = int(ckpt_args.get("pca_dim", args.pca_dim))

    encoder = DINOv2Encoder(
        model_name=args.encoder_model,
        img_size=args.img_size,
        multi_layer=True,
        pca_dim=pca_dim,
    ).to(device)
    encoder.load_pca(args.pca_ckpt, device)

    predictor = build_predictor(
        feat_dim=encoder.feat_dim,
        n_patches=encoder.n_patches,
        cfg=ckpt_args,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        n_context=n_context,
        n_future=1,
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    splits = ["val", "test"] if args.split == "both" else [args.split]
    horizons = [int(x) for x in args.n_pred_steps.split(",")]
    all_results = {}

    for split in splits:
        all_results[split] = {}
        for n_steps in horizons:
            loaders = build_dataloaders(
                data_dir=args.data_dir,
                n_context=n_context,
                n_future=n_steps,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                stride=args.stride,
                data_format="bair",
                rebuild_split=False,
                val_max_clips=0,
                seed=args.seed,
            )
            loader = loaders[1] if split == "val" else loaders[2]
            print(f"\n=== {split} t={n_steps} clips={len(loader.dataset)} ===")
            summary = eval_split(
                encoder, predictor, loader, device, n_steps, args.max_clips,
            )
            all_results[split][f"t{n_steps}"] = summary
            cl = summary["copy_last"]
            pr = summary["predictor"]
            print(
                f"  Copy-Last  SmoothL1={cl['smooth_l1']:.4f}  L2={cl['l2']:.4f}  cos={cl['cosine']:.4f}"
            )
            print(
                f"  Predictor  SmoothL1={pr['smooth_l1']:.4f}  L2={pr['l2']:.4f}  cos={pr['cosine']:.4f}"
            )
            delta = cl["smooth_l1"] - pr["smooth_l1"]
            print(f"  Delta SmoothL1 (copy - pred, >0 means predictor wins): {delta:.4f}")

    out_json = Path(args.output_dir) / "feature_copy_last_vs_predictor.json"
    out_md = Path(args.output_dir) / "feature_copy_last_vs_predictor.md"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    write_markdown(out_md, all_results)
    print(f"\nWrote {out_json}\nWrote {out_md}")


if __name__ == "__main__":
    main()
