#!/bin/bash
#SBATCH --job-name=dino-tracka2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/kcwp264/logs/dino-tracka2_%j.out
#SBATCH --error=/scratch/kcwp264/logs/dino-tracka2_%j.err

# Track A2 GPU sizing (do not bump to 2–3 GPUs or 48h):
# - Bottleneck is frozen DINOv2 on 5 frames/clip, not the 4×768 predictor.
# - Track A: 1×L40S, ~21 min/epoch, 41 epochs, 14h22m. A2 is the same encoder.
# - gpu partition often has 100+ pending jobs; extra GPUs cost queue time, not wall time.
# - DDP with global batch 16 changes the run vs Copy-Last (batch 8). Keep 1 GPU.
# - 18h cap: 40 × ~20 min ≈ 13h plus init val; patience 12 usually stops earlier.
# - Job aborts in ~25 min if epoch-0 val is not Copy-Last (~0.45 SmoothL1).

set -euo pipefail
source /scratch/kcwp264/.aire_scratch_env.sh

ENV_DIR="/scratch/kcwp264/conda/envs/endofm-lv"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/scratch/kcwp264/Msc_research_future_frame_prediciton:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

REPO="/scratch/kcwp264/Msc_research_future_frame_prediciton"
DATA_DIR="$REPO/VPTR_jigsaws_working/jigsaws_suturing/bair_format_dir_louo"
OUTPUT_DIR="$REPO/outputs/jigsaws_masterplan/predictor_tracka2"
PCA_CKPT="$REPO/outputs/jigsaws_masterplan/predictor_tracka/pca.pth"

mkdir -p "$OUTPUT_DIR" /scratch/kcwp264/logs
cd "$REPO"

echo "=== Track A2 residual + past-only motion (1x L40S) ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}  Date: $(date)"
nvidia-smi -L

"$ENV_DIR/bin/python" -m dino_foresight.train \
    --encoder dinov2 \
    --encoder_model vitb14_reg \
    --data_dir "$DATA_DIR" \
    --data_format bair \
    --output_dir "$OUTPUT_DIR" \
    --img_size 224 \
    --n_context 4 \
    --n_future 1 \
    --n_pred_steps 1 \
    --n_pred_steps_train 1 \
    --no_scheduled_sampling \
    --hidden_dim 768 \
    --num_layers 4 \
    --num_heads 8 \
    --dropout 0.1 \
    --batch_size 8 \
    --lr 8e-5 \
    --weight_decay 0.05 \
    --epochs 40 \
    --warmup_epochs 2 \
    --num_workers 4 \
    --stride 6 \
    --multi_layer \
    --pca_dim 1152 \
    --pca_ckpt "$PCA_CKPT" \
    --loss_type smooth_l1 \
    --lambda_feat 1.0 \
    --lambda_pixel 0.0 \
    --lambda_lpips 0.0 \
    --residual \
    --use_motion \
    --motion_from feat \
    --motion_weight \
    --eval_interval 999 \
    --eval_max_clips 8 \
    --val_interval 1 \
    --patience 12 \
    --save_interval 10 \
    --val_max_clips 500 \
    --wandb \
    --wandb_project surgical-future-frame-prediction \
    --seed 42

echo "=== Track A2 finished $(date) ==="
