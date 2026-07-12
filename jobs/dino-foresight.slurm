#!/bin/bash
#SBATCH --job-name=dino-foresight-surg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/kcwp264/logs/dino-foresight_%j.out
#SBATCH --error=/scratch/kcwp264/logs/dino-foresight_%j.err

# ============================================================
# DINO-Foresight Surgical Video Prediction — AIRE HPC L40S
# ============================================================

set -euo pipefail

# --- Environment ---
source /scratch/kcwp264/conda/etc/profile.d/conda.sh
conda activate endofm-lv
export LD_LIBRARY_PATH=/scratch/kcwp264/conda/envs/endofm-lv/lib:$LD_LIBRARY_PATH

# NCCL settings for L40S (PCIe, no NVLink)
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# W&B
export WANDB_DIR=/scratch/kcwp264/Msc_research_future_frame_prediciton/wandb

# --- Paths ---
REPO=/scratch/kcwp264/Msc_research_future_frame_prediciton
DATA_DIR=${JIGSAWS_DATA_DIR:-"/scratch/kcwp264/Msc_research_future_frame_prediciton/data/Suturing"}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/kcwp264/Msc_research_future_frame_prediciton/outputs/dino_foresight"}

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" /scratch/kcwp264/logs

cd "$REPO"

# --- Determine GPU count ---
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "=== DINO-Foresight Surgical Video Prediction ==="
echo "GPUs:       $NUM_GPUS"
echo "Data:       $DATA_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Date:       $(date)"
echo "================================================"

# --- Encoder selection ---
ENCODER=${ENCODER:-dinov2}
IMG_SIZE=${IMG_SIZE:-224}
EPOCHS=${EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-16}
N_CONTEXT=${N_CONTEXT:-4}
N_FUTURE=${N_FUTURE:-1}
LR=${LR:-6.4e-4}

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching multi-GPU DDP training ($NUM_GPUS GPUs)..."
    torchrun --standalone --nproc_per_node=$NUM_GPUS \
        -m dino_foresight.train \
        --encoder "$ENCODER" \
        --data_dir "$DATA_DIR" \
        --data_format bair \
        --output_dir "$OUTPUT_DIR" \
        --img_size "$IMG_SIZE" \
        --n_context "$N_CONTEXT" \
        --n_future "$N_FUTURE" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --ddp \
        --wandb
else
    echo "Launching single-GPU training..."
    python -m dino_foresight.train \
        --encoder "$ENCODER" \
        --data_dir "$DATA_DIR" \
        --data_format bair \
        --output_dir "$OUTPUT_DIR" \
        --img_size "$IMG_SIZE" \
        --n_context "$N_CONTEXT" \
        --n_future "$N_FUTURE" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --wandb
fi

echo "=== Training complete ==="
echo "Checkpoints: $OUTPUT_DIR"
