#!/bin/bash
#SBATCH --job-name=dino-a2-rgb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/kcwp264/logs/dino-tracka2-rgb_%j.out
#SBATCH --error=/scratch/kcwp264/logs/dino-tracka2-rgb_%j.err

# Step 1/3: proper RGB decode eval (frozen Phase-1 decoder) + moving/static breakdown.

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
RGB_OUT="$REPO/outputs/jigsaws_masterplan/eval_tracka2_rgb"

mkdir -p "$RGB_OUT" /scratch/kcwp264/logs
cd "$REPO"

echo "=== Track A2 RGB eval (frozen decoder) ==="
nvidia-smi -L

"$ENV_DIR/bin/python" -m dino_foresight.evaluate \
    --predictor_ckpt "$PRED_CKPT" \
    --decoder_ckpt "$DEC_CKPT" \
    --pca_ckpt "$PCA_CKPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$RGB_OUT" \
    --encoder_model vitb14_reg \
    --img_size 224 \
    --n_context 4 \
    --n_pred_steps 1,5,10 \
    --stride 6 \
    --batch_size 8 \
    --num_workers 4 \
    --multi_layer \
    --pca_dim 1152 \
    --hidden_dim 768 \
    --num_layers 4 \
    --num_heads 8 \
    --decoder_hidden_dim 1152 \
    --decoder_layers 8 \
    --decoder_heads 12

"$ENV_DIR/bin/python" -m dino_foresight.eval_rgb_moving_static \
    --predictor_ckpt "$PRED_CKPT" \
    --decoder_ckpt "$DEC_CKPT" \
    --pca_ckpt "$PCA_CKPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$RGB_OUT" \
    --label tracka2_frozen \
    --n_context 4 \
    --stride 6 \
    --batch_size 4

"$ENV_DIR/bin/python" -m dino_foresight.aggregate_results \
    --root "$REPO/outputs/jigsaws_masterplan" \
    --output "$REPO/outputs/jigsaws_masterplan/master_results.md"

echo "=== RGB eval done $(date) ==="
echo "Copy-Last RGB baseline (test stride6 t=1): PSNR 13.53 dB"
