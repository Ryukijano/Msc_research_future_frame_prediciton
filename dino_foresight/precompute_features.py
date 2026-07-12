"""Feature pre-extraction script for JIGSAWS frames.

Pre-computes frozen encoder features for all frames and saves to disk.
This speeds up training by avoiding repeated encoder forward passes.

Usage:
    python -m dino_foresight.precompute_features --encoder dinov2 --data_dir /path/to/jigsaws
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import glob
from PIL import Image
import torchvision.transforms as transforms

from dino_foresight.encoders import DINOv2Encoder, VJEPA2Encoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="dinov2", choices=["dinov2", "vjepa2"])
    parser.add_argument("--encoder_model", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./precomputed_features")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--multi_layer", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build encoder
    if args.encoder == "dinov2":
        encoder = DINOv2Encoder(
            model_name=args.encoder_model or "vitb14",
            img_size=args.img_size,
            multi_layer=args.multi_layer,
        ).to(device)
    else:
        encoder = VJEPA2Encoder(
            model_name=args.encoder_model or "vjepa2_1_vit_base_384",
            img_size=args.img_size,
        ).to(device)

    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each split
    for split in ["train", "val", "test"]:
        split_dir = Path(args.data_dir) / split
        if not split_dir.exists():
            print(f"Skipping {split} — directory not found: {split_dir}")
            continue

        video_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        out_split = Path(args.output_dir) / args.encoder / split
        out_split.mkdir(parents=True, exist_ok=True)

        for vdir in tqdm(video_dirs, desc=f"Processing {split}"):
            frame_paths = sorted(glob.glob(str(vdir / "*.png")))
            if not frame_paths:
                continue

            # Load and transform all frames
            frames = []
            for fp in frame_paths:
                img = Image.open(fp).convert("RGB")
                frames.append(transform(img))

            frames_tensor = torch.stack(frames).to(device)  # (N, C, H, W)

            # Extract features in batches
            all_feats = []
            for i in range(0, len(frames_tensor), args.batch_size):
                batch = frames_tensor[i : i + args.batch_size]
                with torch.no_grad():
                    feats = encoder(batch)  # (B, N_patches, D)
                all_feats.append(feats.cpu())

            all_feats = torch.cat(all_feats, dim=0)  # (N_frames, N_patches, D)

            # Save
            out_path = out_split / f"{vdir.name}.pt"
            torch.save(all_feats, out_path)

        print(f"Saved {len(video_dirs)} videos to {out_split}")


if __name__ == "__main__":
    main()
