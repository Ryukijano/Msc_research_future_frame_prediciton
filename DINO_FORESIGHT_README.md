# DINO-Foresight: Surgical Video Prediction with Frozen Foundation Encoders

> Branch: `dino-foresight-surgical` — Vision-only future frame prediction for JIGSAWS Suturing

## Overview

This branch implements a **vision-only** future frame prediction model for robotic surgical video, using frozen foundation vision encoders (DINOv2 / V-JEPA 2.1) instead of training encoders from scratch.

**Key idea**: Extract rich spatial features from a frozen pre-trained encoder, then train a lightweight masked feature transformer to predict future frame features in latent space. No action conditioning, no gesture labels — pure visual prediction.

This is inspired by:
- **DINO-Foresight** (NeurIPS 2025) — self-supervised future feature prediction with masked transformers
- **DINO-WM** (arXiv 2411.04983) — world models on pre-trained visual features
- **DINO-world** (arXiv 2507.19468) — generalist video world model in DINOv2 latent space

## Architecture

```
Past N frames (vision only)
    ↓
[Frozen Foundation Encoder]  ← DINOv2 ViT-B/14 or V-JEPA 2.1 ViT-B/16
    ↓
Patch features per frame (N × H' × W' × D)
    ↓
[Masked Feature Transformer]  ← temporal + spatial decomposed attention
    ↓ (SmoothL1 loss in latent space)
Predicted future frame features
    ↓
[Lightweight Conv Decoder]  ← for pixel-space visualization (optional)
    ↓
Predicted future frames → PSNR / SSIM / LPIPS evaluation
```

## Comparison with original VPTR/TPG-VAE

| Aspect | VPTR / TPG-VAE | DINO-Foresight |
|--------|----------------|----------------|
| Encoder | ResNet (from scratch) | DINOv2 / V-JEPA 2.1 (frozen, pre-trained) |
| Training signal | Pixel-space reconstruction | Latent-space feature prediction |
| Temporal model | Transformer on AE features | Masked transformer on frozen features |
| Conditioning | Gesture labels (TPG-VAE) | None (vision-only) |
| Parameters to train | Full model | Only transformer + decoder |
| Generalization | Limited (small dataset) | Benefits from large-scale pre-training |

## Project Structure

```
dino_foresight/
├── __init__.py
├── encoders.py              # Frozen DINOv2 + V-JEPA 2.1 wrappers
├── predictor.py             # Masked feature transformer (temporal + spatial attention)
├── decoder.py               # Lightweight pixel decoders (Conv + ViT)
├── data.py                  # JIGSAWS dataset loaders (BAIR + NPZ formats)
├── metrics.py               # PSNR, SSIM, LPIPS
├── train.py                 # Main training loop with W&B logging
├── inference.py             # Generate predictions + visualizations
├── precompute_features.py   # Pre-extract frozen encoder features for speed
└── requirements.txt
jobs/
└── dino-foresight.slurm     # AIRE HPC Slurm job script (3x L40S)
```

## Usage

### Single GPU

```bash
conda activate endofm-lv
python -m dino_foresight.train \
    --encoder dinov2 \
    --data_dir /path/to/jigsaws \
    --img_size 224 \
    --n_context 4 \
    --n_future 1 \
    --epochs 200 \
    --batch_size 16
```

### Multi-GPU DDP (3x L40S)

```bash
torchrun --standalone --nproc_per_node=3 \
    -m dino_foresight.train \
    --encoder dinov2 \
    --data_dir /path/to/jigsaws \
    --ddp \
    --batch_size 16
```

### Slurm (AIRE HPC)

```bash
sbatch jobs/dino-foresight.slurm
```

### Inference & Visualization

```bash
python -m dino_foresight.inference \
    --checkpoint ./outputs/dino_foresight/best_model.pth \
    --data_dir /path/to/jigsaws \
    --n_pred_steps 20
```

### Pre-compute Features (optional, speeds up training)

```bash
python -m dino_foresight.precompute_features \
    --encoder dinov2 \
    --data_dir /path/to/jigsaws
```

## Encoders

### DINOv2 (default)
- Model: `vitb14` (ViT-B/14, 80M params, 768-dim features)
- Pre-trained on 142M images (LVD-142M)
- Multi-layer feature extraction (layers 3, 6, 9, 11)
- Patch size 14, flexible input resolution

### V-JEPA 2.1
- Model: `vjepa2_1_vit_base_384` (ViT-B/16, 80M params)
- Pre-trained on 1M+ hours of internet video
- Video-native encoder with temporal understanding
- Patch size 16, fixed 384×384 input
- For single-frame extraction: duplicate image to 16 frames, average over time

## Dataset

JIGSAWS Suturing dataset in BAIR-like format:
```
data/
├── train/
│   ├── example_B001_capture1_0/
│   │   ├── frame_000.png
│   │   ├── frame_001.png
│   │   └── ...
│   └── ...
├── val/
└── test/
```

## Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio) — higher is better
- **SSIM** (Structural Similarity) — higher is better
- **LPIPS** (Learned Perceptual Image Patch Similarity) — lower is better

All metrics computed at t=5, t=10, t=15, t=20 prediction horizons (following TPG-VAE protocol).

## References

1. Efstathios Karypidis et al. "DINO-Foresight: Looking into the Future with DINO." NeurIPS 2025.
2. Zhou et al. "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning." arXiv 2411.04983.
3. Assran et al. "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning." arXiv 2506.09985.
4. Gao et al. "Future Frame Prediction for Robot-assisted Surgery." ICRA 2021. (TPG-VAE)
5. Gao et al. "JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS)." MICCAI 2014.
