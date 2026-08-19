#!/bin/bash
#SBATCH --job-name=dino-tracka2-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/kcwp264/logs/dino-tracka2-eval_%j.out
#SBATCH --error=/scratch/kcwp264/logs/dino-tracka2-eval_%j.err

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
CKPT_DIR="$REPO/outputs/jigsaws_masterplan/predictor_tracka2"
PCA_CKPT="$CKPT_DIR/pca.pth"
OUT_DIR="$REPO/outputs/jigsaws_masterplan/eval_feature_copy_last_tracka2"

mkdir -p "$OUT_DIR" /scratch/kcwp264/logs
cd "$REPO"

echo "=== Feature Copy-Last vs Track A2 ==="
echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}  Date: $(date)"
nvidia-smi -L

"$ENV_DIR/bin/python" -m dino_foresight.eval_feature_copy_last \
    --predictor_ckpt "$CKPT_DIR/best_val_model.pth" \
    --pca_ckpt "$PCA_CKPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR" \
    --n_context 4 \
    --n_pred_steps 1,5,10 \
    --stride 6 \
    --batch_size 8 \
    --num_workers 4 \
    --hidden_dim 768 \
    --num_layers 4 \
    --num_heads 8 \
    --split both

echo "=== Done $(date) ==="
