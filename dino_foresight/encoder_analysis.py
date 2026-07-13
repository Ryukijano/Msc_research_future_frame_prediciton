"""Encoder comparison and PCA analysis for surgical video prediction.

Tests DINOv2 and V-JEPA 2.1 on the A2 GPU:
1. Load each encoder, measure memory + inference time
2. Extract features from sample frames
3. PCA analysis: variance explained, top components
4. Feature similarity analysis (cosine similarity between patches)
5. Save PCA visualizations

Usage:
    module load miniforge
    conda activate endofm-lv
    python -m dino_foresight.encoder_analysis --data_dir /path/to/jigsaws
"""

import os
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import glob

os.environ.setdefault("HF_HOME", "/scratch/kcwp264/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/scratch/kcwp264/.cache/torch")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/scratch/kcwp264/Msc_research_future_frame_prediciton/VPTR_jigsaws_working/jigsaws_suturing/bair_format_dir/train",
                        help="Path to JIGSAWS frames. If None, uses synthetic test.")
    parser.add_argument("--output_dir", type=str, default="./encoder_analysis")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--encoders", type=str, nargs="+", default=["dinov2_vitb14", "dinov2_vits14", "dinov2_vitb14_reg", "tipsv2_b14", "lingbot_small"])
    return parser.parse_args()


def get_sample_frames(data_dir, n_samples, img_size, device):
    """Load sample frames from JIGSAWS or generate synthetic."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if data_dir and Path(data_dir).exists():
        frame_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            frame_paths.extend(glob.glob(str(Path(data_dir) / "**" / ext), recursive=True))

        if len(frame_paths) == 0:
            print(f"No frames found in {data_dir}, using synthetic data")
            return torch.randn(n_samples, 3, img_size, img_size).to(device)

        # Sample frames evenly across different video sequences for diverse PCA
        frame_paths = sorted(frame_paths)
        if len(frame_paths) > n_samples:
            # Stratified sample: pick evenly spaced frames across all sequences
            indices = np.linspace(0, len(frame_paths) - 1, n_samples, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]
        print(f"Loaded {len(frame_paths)} frames from {data_dir} at {img_size}x{img_size}")

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            frames.append(transform(img))
        return torch.stack(frames).to(device)
    else:
        print("No data dir specified, using synthetic data")
        return torch.randn(n_samples, 3, img_size, img_size).to(device)


def measure_encoder(encoder_name, frames, device, output_dir):
    """Test an encoder: load, measure memory, extract features, PCA analysis."""
    print(f"\n{'='*60}")
    print(f"Testing encoder: {encoder_name}")
    print(f"{'='*60}")

    # Parse encoder name
    if encoder_name.startswith("dinov2_"):
        # Full hubconf name like dinov2_vitb14, dinov2_vits14, dinov2_vitb14_reg
        from dino_foresight.encoders import DINOv2Encoder
        img_size = frames.shape[-1]
        encoder = DINOv2Encoder(model_name=encoder_name, img_size=img_size, multi_layer=True)
    elif encoder_name.startswith("vjepa2_"):
        model_name = encoder_name
        img_size = 384  # V-JEPA 2.1 requires 384
        from dino_foresight.encoders import VJEPA2Encoder
        encoder = VJEPA2Encoder(model_name=model_name, img_size=img_size)
        # Resize frames for V-JEPA
        frames = F.interpolate(frames, size=(img_size, img_size), mode="bilinear", align_corners=False)
    elif encoder_name.startswith("tipsv2_"):
        model_name = encoder_name.replace("tipsv2_", "")
        img_size = frames.shape[-1]
        from dino_foresight.encoders import TIPSv2Encoder
        encoder = TIPSv2Encoder(model_name=model_name, img_size=img_size)
    elif encoder_name.startswith("lingbot_"):
        variant = encoder_name.replace("lingbot_", "")
        img_size = frames.shape[-1]
        from dino_foresight.encoders import LingBotVisionEncoder
        encoder = LingBotVisionEncoder(model_name=variant, img_size=img_size)
    else:
        print(f"Unknown encoder: {encoder_name}")
        return None

    encoder = encoder.to(device)
    encoder.eval()

    # Memory before
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)

    # Inference timing
    times = []
    with torch.no_grad():
        # Warmup
        for i in range(min(3, len(frames))):
            _ = encoder(frames[i:i+1])

        torch.cuda.synchronize()
        for i in range(len(frames)):
            t0 = time.time()
            feats = encoder(frames[i:i+1])
            torch.cuda.synchronize()
            times.append(time.time() - t0)

    mem_after = torch.cuda.max_memory_allocated(device)
    mem_peak_mb = mem_after / 1024**2

    avg_time_ms = np.mean(times) * 1000
    std_time_ms = np.std(times) * 1000

    # Extract all features
    with torch.no_grad():
        all_feats = []
        batch_size = 8
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            feats = encoder(batch)
            all_feats.append(feats.cpu())
        all_feats = torch.cat(all_feats, dim=0)  # (N, n_patches, D)

    N, n_patches, D = all_feats.shape
    grid_size = int(np.sqrt(n_patches))

    print(f"  Feature shape:  ({N}, {n_patches}, {D})")
    print(f"  Grid size:      {grid_size}x{grid_size}")
    print(f"  Inference time: {avg_time_ms:.1f} ± {std_time_ms:.1f} ms/image")
    print(f"  Peak GPU mem:   {mem_peak_mb:.0f} MB")
    print(f"  Feature stats:  mean={all_feats.mean():.4f}, std={all_feats.std():.4f}")

    # PCA analysis
    print(f"\n  --- PCA Analysis ---")
    feats_flat = all_feats.reshape(N * n_patches, D).numpy()
    mean = feats_flat.mean(axis=0, keepdims=True)
    centered = feats_flat - mean

    # SVD-based PCA
    U, S, Vh = np.linalg.svd(centered, full_matrices=False)
    variance_explained = (S ** 2) / (S ** 2).sum()
    cumvar = np.cumsum(variance_explained)

    # Find dimensions needed for 90%, 95%, 99% variance
    dims_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    dims_99 = int(np.searchsorted(cumvar, 0.99)) + 1

    print(f"  Dims for 90% variance: {dims_90}/{D}")
    print(f"  Dims for 95% variance: {dims_95}/{D}")
    print(f"  Dims for 99% variance: {dims_99}/{D}")
    print(f"  Top-3 variance:        {variance_explained[:3]}")
    print(f"  Top-10 cumvar:         {cumvar[:10]}")

    # PCA projection to 3D for visualization
    proj_3d = centered @ Vh[:3].T  # (N*n_patches, 3)
    proj_3d = proj_3d.reshape(N, n_patches, 3)

    # Normalize to [0, 1] for visualization
    proj_min = proj_3d.min(axis=(0, 1), keepdims=True)
    proj_max = proj_3d.max(axis=(0, 1), keepdims=True)
    proj_3d_norm = (proj_3d - proj_min) / (proj_max - proj_min + 1e-8)

    # Save PCA visualizations for first 8 samples
    n_vis = min(8, N)
    fig, axes = plt.subplots(2, n_vis, figsize=(3 * n_vis, 6))
    if n_vis == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_vis):
        # Original image (denormalized)
        img = frames[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Frame {i}", fontsize=10)
        axes[0, i].axis("off")

        # PCA visualization
        pca_img = proj_3d_norm[i].reshape(grid_size, grid_size, 3)
        axes[1, i].imshow(pca_img)
        axes[1, i].set_title(f"PCA (3 comp)", fontsize=10)
        axes[1, i].axis("off")

    plt.suptitle(f"{encoder_name} — PCA Visualization (first 3 components)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = Path(output_dir) / f"pca_{encoder_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PCA visualization saved: {save_path}")

    # Variance explained plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(range(min(50, len(variance_explained))), variance_explained[:50], color="steelblue")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained")
    ax1.set_title(f"{encoder_name} — Individual Variance")
    ax1.axhline(y=1.0/D, color="r", linestyle="--", alpha=0.5, label=f"Uniform (1/D={1.0/D:.4f})")
    ax1.legend()

    ax2.plot(range(len(cumvar)), cumvar, color="steelblue", linewidth=2)
    ax2.axhline(y=0.90, color="orange", linestyle="--", alpha=0.7, label="90%")
    ax2.axhline(y=0.95, color="green", linestyle="--", alpha=0.7, label="95%")
    ax2.axhline(y=0.99, color="red", linestyle="--", alpha=0.7, label="99%")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained")
    ax2.set_title(f"{encoder_name} — Cumulative Variance")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    plt.suptitle(f"{encoder_name} — PCA Variance Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = Path(output_dir) / f"variance_{encoder_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Variance plot saved: {save_path}")

    # Feature similarity analysis (cosine similarity between patches within a frame)
    sim_matrix = torch.zeros(n_patches, n_patches)
    for i in range(min(5, N)):
        feats_i = all_feats[i]  # (n_patches, D)
        norms = feats_i.norm(dim=-1, keepdim=True)
        sim = (feats_i @ feats_i.T) / (norms * norms.T + 1e-8)
        sim_matrix += sim.cpu()
    sim_matrix /= min(5, N)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(sim_matrix.numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"{encoder_name} — Patch Cosine Similarity (avg over 5 frames)")
    ax.set_xlabel("Patch index")
    ax.set_ylabel("Patch index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_path = Path(output_dir) / f"patch_sim_{encoder_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Patch similarity saved: {save_path}")

    # Cleanup
    del encoder
    torch.cuda.empty_cache()

    return {
        "encoder": encoder_name,
        "feat_shape": f"({N}, {n_patches}, {D})",
        "grid_size": grid_size,
        "n_patches": n_patches,
        "feat_dim": D,
        "inference_ms": float(avg_time_ms),
        "inference_std_ms": float(std_time_ms),
        "peak_gpu_mb": float(mem_peak_mb),
        "feat_mean": float(all_feats.mean()),
        "feat_std": float(all_feats.std()),
        "dims_90pct": dims_90,
        "dims_95pct": dims_95,
        "dims_99pct": dims_99,
        "top3_variance": variance_explained[:3].tolist(),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB" if torch.cuda.is_available() else "")

    os.makedirs(args.output_dir, exist_ok=True)

    # Get sample frames
    frames = get_sample_frames(args.data_dir, args.n_samples, args.img_size, device)
    print(f"Loaded {frames.shape[0]} frames at {frames.shape[-1]}x{frames.shape[-2]}")

    # Test each encoder
    results = []
    for enc_name in args.encoders:
        try:
            result = measure_encoder(enc_name, frames, device, args.output_dir)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n  ERROR with {enc_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary comparison
    print(f"\n{'='*80}")
    print("ENCODER COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Encoder':<25} {'Dim':>6} {'Patches':>8} {'Time(ms)':>10} {'GPU(MB)':>10} {'90%':>6} {'95%':>6} {'99%':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['encoder']:<25} {r['feat_dim']:>6} {r['n_patches']:>8} "
              f"{r['inference_ms']:>10.1f} {r['peak_gpu_mb']:>10.0f} "
              f"{r['dims_90pct']:>6} {r['dims_95pct']:>6} {r['dims_99pct']:>6}")

    # Save results as JSON
    import json
    with open(Path(args.output_dir) / "encoder_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/encoder_comparison.json")

    # GR00T connection analysis
    print(f"\n{'='*80}")
    print("GR00T CONNECTION")
    print(f"{'='*80}")
    print("""
GR00T N1 uses Eagle-2 VLM (SigLIP-2 + SmolLM2) at 224×224 → 64 image tokens.
Key finding: uses middle-layer (12th) LLM embeddings, not final layer.

For surgical downstream tasks, the question is:
- Do DINOv2/V-JEPA features generalize better to surgical video?
- Which encoder's PCA structure is more semantically meaningful?
- Can these features serve as drop-in replacements for GR00T's visual backbone?

The PCA analysis above shows how much each dimension contributes.
Lower dims for 90% variance = more compact, potentially more generalizable.
More uniform variance = richer representation, less collapse.
""")


if __name__ == "__main__":
    main()
