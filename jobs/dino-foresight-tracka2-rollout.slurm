#!/bin/bash
#SBATCH --job-name=dino-a2-rollout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/kcwp264/logs/dino-tracka2-rollout_%j.out
#SBATCH --error=/scratch/kcwp264/logs/dino-tracka2-rollout_%j.err

set -euo pipefail
source /scratch/kcwp264/.aire_scratch_env.sh

ENV_DIR="/scratch/kcwp264/conda/envs/endofm-lv"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/scratch/kcwp264/Msc_research_future_frame_prediciton:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

REPO="/scratch/kcwp264/Msc_research_future_frame_prediciton"
DATA_DIR="$REPO/VPTR_jigsaws_working/jigsaws_suturing/bair_format_dir_louo"
PRED_CKPT="$REPO/outputs/jigsaws_masterplan/predictor_tracka2/best_val_model.pth"
DEC_CKPT="$REPO/outputs/jigsaws_masterplan/phase1_decoder/best_val_decoder.pth"
PCA_CKPT="$REPO/outputs/jigsaws_masterplan/predictor_tracka2/pca.pth"
OUT_DIR="$REPO/outputs/jigsaws_masterplan/tracka2_rollout_viz"

mkdir -p "$OUT_DIR" /scratch/kcwp264/logs
cd "$REPO"

echo "=== Track A2 RGB rollout viz ==="
nvidia-smi -L

"$ENV_DIR/bin/python" -m dino_foresight.rollout_viz \
    --predictor_ckpt "$PRED_CKPT" \
    --decoder_ckpt "$DEC_CKPT" \
    --pca_ckpt "$PCA_CKPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR" \
    --split test \
    --n_context 4 \
    --n_pred_steps 10 \
    --stride 6 \
    --n_samples 8

echo "=== Done $(date) ==="
