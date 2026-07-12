"""JIGSAWS Suturing dataset loader for future frame prediction.

Supports two data formats:
1. BAIR-like format (from existing VPTR pipeline): directories of frames
2. NPZ format (preprocessed clips): single .npz file with clips array

The loader produces (past_clip, future_clip) pairs where:
- past_clip: (T_context, C, H, W) tensor
- future_clip: (T_future, C, H, W) tensor
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple
import torchvision.transforms as transforms
from PIL import Image


class JigsawsBAIRDataset(Dataset):
    """JIGSAWS Suturing dataset in BAIR-like directory format.

    Expected structure:
        data_dir/
            train/
                example_0/
                    frame_000.png
                    frame_001.png
                    ...
            test/
                example_1/
                    ...

    Args:
        data_dir: Root directory containing train/val/test splits
        split: "train", "val", or "test"
        n_context: Number of past frames
        n_future: Number of future frames to predict
        img_size: Target image size (square)
        stride: Frame stride for sampling (1 = every frame, 2 = every other)
        augment: Whether to apply data augmentation (horizontal flip)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        n_context: int = 4,
        n_future: int = 1,
        img_size: int = 224,
        stride: int = 1,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir) / split
        self.n_context = n_context
        self.n_future = n_future
        self.img_size = img_size
        self.stride = stride
        self.augment = augment and split == "train"

        # Collect all video directories
        self.video_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        if not self.video_dirs:
            raise FileNotFoundError(f"No video directories found in {self.data_dir}")

        # Build clip index: (video_idx, start_frame_idx) for all valid clips
        self.clips = []
        clip_len = (n_context + n_future - 1) * stride + 1
        for vidx, vdir in enumerate(self.video_dirs):
            frames = sorted(glob.glob(str(vdir / "*.png")))
            n_frames = len(frames)
            n_clips = max(0, n_frames - clip_len + 1)
            for cidx in range(n_clips):
                self.clips.append((vidx, cidx))

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.clips)

    def _load_frame(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vidx, cidx = self.clips[idx]
        vdir = self.video_dirs[vidx]
        frames = sorted(glob.glob(str(vdir / "*.png")))

        # Sample frame indices
        indices = list(range(cidx, cidx + (self.n_context + self.n_future) * self.stride, self.stride))

        # Load and transform frames
        imgs = [self._load_frame(frames[i]) for i in indices]

        # Augmentation: random horizontal flip
        if self.augment and np.random.rand() < 0.5:
            imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]

        # Apply transforms
        tensors = [self.transform(img) for img in imgs]
        clip = torch.stack(tensors, dim=0)  # (T, C, H, W)

        past_clip = clip[:self.n_context]
        future_clip = clip[self.n_context:]

        return past_clip, future_clip


class JigsawsNPZDataset(Dataset):
    """JIGSAWS Suturing dataset in preprocessed NPZ format.

    Expected NPZ structure:
        clips: (2, N_clips, 2) array — [[past_start, past_len], [future_start, future_len]]
        input_raw_data: (total_frames, C, H, W) array — all frames concatenated

    Args:
        npz_path: Path to .npz file
        n_context: Number of past frames (must match preprocessing)
        n_future: Number of future frames (must match preprocessing)
        img_size: Target image size (resize if different from stored)
        augment: Whether to apply data augmentation
    """

    def __init__(
        self,
        npz_path: str,
        n_context: int = 10,
        n_future: int = 10,
        img_size: int = 224,
        augment: bool = True,
    ):
        self.npz_path = Path(npz_path)
        self.n_context = n_context
        self.n_future = n_future
        self.img_size = img_size
        self.augment = augment

        # Load NPZ
        data = np.load(self.npz_path.absolute().as_posix())
        self.clips = data["clips"]  # (2, N, 2)
        self.frames = data["input_raw_data"]  # (total, C, H, W)
        self.n_clips = self.clips.shape[1]

        # Transforms
        self.to_pil = transforms.ToPILImage()
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.n_clips

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_idx = self.clips[:, idx, :]

        # Past clip
        psi, pei = clip_idx[0, 0], clip_idx[0, 0] + clip_idx[0, 1]
        past_np = self.frames[psi:pei]

        # Future clip
        fsi, fei = clip_idx[1, 0], clip_idx[1, 0] + clip_idx[1, 1]
        future_np = self.frames[fsi:fei]

        # Convert to PIL, transform, stack
        full_np = np.concatenate((past_np, future_np), axis=0)
        imgs = [self.to_pil(frame) for frame in full_np]

        if self.augment and np.random.rand() < 0.5:
            imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]

        tensors = [self.transform(img) for img in imgs]
        clip = torch.stack(tensors, dim=0)

        past_clip = clip[: self.n_context]
        future_clip = clip[self.n_context :]

        return past_clip, future_clip


def build_dataloaders(
    data_dir: str,
    n_context: int = 4,
    n_future: int = 1,
    img_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    stride: int = 1,
    data_format: str = "bair",
    npz_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders.

    Args:
        data_dir: Root data directory
        n_context: Number of context frames
        n_future: Number of future frames
        img_size: Image size
        batch_size: Batch size
        num_workers: Number of workers
        stride: Frame stride
        data_format: "bair" or "npz"
        npz_path: Path to NPZ file (if format is "npz")

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if data_format == "npz":
        assert npz_path is not None, "npz_path required for npz format"
        # For NPZ, we assume separate train/test files
        train_path = npz_path.replace(".npz", "_train.npz")
        test_path = npz_path.replace(".npz", "_test.npz")
        train_ds = JigsawsNPZDataset(train_path, n_context, n_future, img_size, augment=True)
        test_ds = JigsawsNPZDataset(test_path, n_context, n_future, img_size, augment=False)
        val_ds = test_ds  # Use test as val for simplicity
    else:
        train_ds = JigsawsBAIRDataset(data_dir, "train", n_context, n_future, img_size, stride, augment=True)
        val_ds = JigsawsBAIRDataset(data_dir, "val", n_context, n_future, img_size, stride, augment=False)
        test_ds = JigsawsBAIRDataset(data_dir, "test", n_context, n_future, img_size, stride, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
