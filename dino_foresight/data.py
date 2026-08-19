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
from typing import List, Optional, Sequence, Tuple
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


def apply_clip_color_jitter(
    imgs: Sequence[Image.Image],
    jitter: transforms.ColorJitter,
) -> List[Image.Image]:
    """Apply one sampled ColorJitter to every frame in a clip.

    torchvision.ColorJitter samples new brightness/contrast/hue per call, so
    `jitter(frame)` inside a loop invents fake inter-frame motion.
    """
    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = (
        transforms.ColorJitter.get_params(
            jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue
        )
    )
    out: List[Image.Image] = []
    for img in imgs:
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = TF.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = TF.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = TF.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = TF.adjust_hue(img, hue_factor)
        out.append(img)
    return out


def drop_context_frames(clip: torch.Tensor, n_context: int, max_drop: int = 2) -> torch.Tensor:
    """Zero 1–2 context frames, never the last context frame (z_t)."""
    eligible = np.arange(max(n_context - 1, 0))
    if len(eligible) == 0:
        return clip
    n_drop = int(np.random.randint(1, min(max_drop + 1, len(eligible) + 1)))
    drop_idx = np.random.choice(eligible, size=n_drop, replace=False)
    clip[drop_idx] = 0.0
    return clip


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
        # Account for temporal jitter: max stride = stride + stride_jitter
        self.clips = []
        max_stride = stride + max(1, stride - 1)
        clip_len = (n_context + n_future - 1) * max_stride + 1
        for vidx, vdir in enumerate(self.video_dirs):
            frames = sorted(glob.glob(str(vdir / "*.png")))
            n_frames = len(frames)
            n_clips = max(0, n_frames - clip_len + 1)
            for cidx in range(n_clips):
                self.clips.append((vidx, cidx))

        # Transforms (resize is done per-frame; crop applied during __getitem__)
        self.resize = transforms.Resize((img_size, img_size))
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.crop_pad = int(img_size * 0.1)  # 10% random crop range
        # Color jitter (applied consistently across frames in a clip)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        # Temporal augmentation: random stride jitter per clip
        self.stride_jitter = max(1, stride - 1)  # e.g. stride=2 -> jitter 1-3
        # Frame dropout: randomly mask 1-2 context frames (set to zero = black)
        self.frame_dropout_prob = 0.1

    def __len__(self):
        return len(self.clips)

    def _load_frame(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vidx, cidx = self.clips[idx]
        vdir = self.video_dirs[vidx]
        frames = sorted(glob.glob(str(vdir / "*.png")))

        # Sample frame indices
        # Temporal jitter: vary stride slightly per clip for diverse temporal deltas
        if self.augment and self.stride_jitter > 0:
            clip_stride = np.random.randint(1, self.stride + self.stride_jitter + 1)
        else:
            clip_stride = self.stride
        indices = list(range(cidx, cidx + (self.n_context + self.n_future) * clip_stride, clip_stride))

        # Load frames
        imgs = [self._load_frame(frames[i]) for i in indices]

        # Resize all frames to same size first
        imgs = [self.resize(img) for img in imgs]

        # Augmentation: random horizontal flip (applied consistently across frames)
        if self.augment and np.random.rand() < 0.5:
            imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]

        # Augmentation: random crop (same crop for all frames in clip)
        if self.augment and self.crop_pad > 0:
            w, h = imgs[0].size
            crop_w = w - self.crop_pad
            crop_h = h - self.crop_pad
            x0 = np.random.randint(0, self.crop_pad + 1)
            y0 = np.random.randint(0, self.crop_pad + 1)
            imgs = [img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((self.img_size, self.img_size)) for img in imgs]

        # Augmentation: color jitter (same params for all frames in clip)
        if self.augment:
            imgs = apply_clip_color_jitter(imgs, self.color_jitter)

        # Apply transforms
        tensors = [self.to_tensor(img) for img in imgs]
        clip = torch.stack(tensors, dim=0)  # (T, C, H, W)

        # Frame dropout: never drop the last context frame (needed for residual skip)
        if self.augment and np.random.rand() < self.frame_dropout_prob:
            clip = drop_context_frames(clip, self.n_context)

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
    rebuild_split: bool = False,
    val_max_clips: int = 0,
    seed: int = 42,
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
        rebuild_split: If True, merge train+val+test dirs and re-split 80/10/10
        val_max_clips: If >0, subsample val set to this many clips
        seed: Random seed for rebuild_split

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
        if rebuild_split:
            # Merge all video dirs from train+val+test, re-split 80/10/10
            all_video_dirs = []
            for split in ["train", "val", "test"]:
                split_dir = Path(data_dir) / split
                if split_dir.exists():
                    all_video_dirs.extend(
                        sorted([d for d in split_dir.iterdir() if d.is_dir()])
                    )
            rng = np.random.RandomState(seed)
            rng.shuffle(all_video_dirs)
            n_total = len(all_video_dirs)
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)
            train_dirs = all_video_dirs[:n_train]
            val_dirs = all_video_dirs[n_train:n_train + n_val]
            test_dirs = all_video_dirs[n_train + n_val:]
            print(f"  Rebuild split: {n_total} videos -> train={len(train_dirs)}, val={len(val_dirs)}, test={len(test_dirs)}")

            train_ds = _BAIRFromDirs(train_dirs, n_context, n_future, img_size, stride, augment=True)
            val_ds = _BAIRFromDirs(val_dirs, n_context, n_future, img_size, stride, augment=False)
            test_ds = _BAIRFromDirs(test_dirs, n_context, n_future, img_size, stride, augment=False)
        else:
            train_ds = JigsawsBAIRDataset(data_dir, "train", n_context, n_future, img_size, stride, augment=True)
            test_ds = JigsawsBAIRDataset(data_dir, "test", n_context, n_future, img_size, stride, augment=False)
            # Handle missing val split: use test as val (common for small datasets like JIGSAWS)
            val_dir = Path(data_dir) / "val"
            if val_dir.exists() and any(val_dir.iterdir()):
                val_ds = JigsawsBAIRDataset(data_dir, "val", n_context, n_future, img_size, stride, augment=False)
            else:
                val_ds = test_ds

    # Subsample val set if requested
    if val_max_clips > 0 and len(val_ds) > val_max_clips:
        from torch.utils.data import Subset
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(val_ds), val_max_clips, replace=False)
        val_ds = Subset(val_ds, indices.tolist())
        print(f"  Val subsampled to {len(val_ds)} clips")

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


class _BAIRFromDirs(Dataset):
    """Build a dataset from an explicit list of video directories (for rebuild_split)."""

    def __init__(self, video_dirs, n_context, n_future, img_size, stride, augment=True):
        self.video_dirs = video_dirs
        self.n_context = n_context
        self.n_future = n_future
        self.img_size = img_size
        self.stride = stride
        self.augment = augment

        self.clips = []
        max_stride = stride + max(1, stride - 1)
        clip_len = (n_context + n_future - 1) * max_stride + 1
        for vidx, vdir in enumerate(video_dirs):
            frames = sorted(glob.glob(str(vdir / "*.png")))
            n_frames = len(frames)
            n_clips = max(0, n_frames - clip_len + 1)
            for cidx in range(n_clips):
                self.clips.append((vidx, cidx))

        self.resize = transforms.Resize((img_size, img_size))
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.crop_pad = int(img_size * 0.1)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        self.stride_jitter = max(1, stride - 1)
        self.frame_dropout_prob = 0.1

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        vidx, cidx = self.clips[idx]
        vdir = self.video_dirs[vidx]
        frames = sorted(glob.glob(str(vdir / "*.png")))

        if self.augment and self.stride_jitter > 0:
            clip_stride = np.random.randint(1, self.stride + self.stride_jitter + 1)
        else:
            clip_stride = self.stride
        indices = list(range(cidx, cidx + (self.n_context + self.n_future) * clip_stride, clip_stride))

        imgs = [Image.open(frames[i]).convert("RGB") for i in indices]
        imgs = [self.resize(img) for img in imgs]

        if self.augment and np.random.rand() < 0.5:
            imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]

        if self.augment and self.crop_pad > 0:
            w, h = imgs[0].size
            crop_w = w - self.crop_pad
            crop_h = h - self.crop_pad
            x0 = np.random.randint(0, self.crop_pad + 1)
            y0 = np.random.randint(0, self.crop_pad + 1)
            imgs = [img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((self.img_size, self.img_size)) for img in imgs]

        if self.augment:
            imgs = apply_clip_color_jitter(imgs, self.color_jitter)

        tensors = [self.to_tensor(img) for img in imgs]
        clip = torch.stack(tensors, dim=0)

        if self.augment and np.random.rand() < self.frame_dropout_prob:
            clip = drop_context_frames(clip, self.n_context)

        return clip[:self.n_context], clip[self.n_context:]
