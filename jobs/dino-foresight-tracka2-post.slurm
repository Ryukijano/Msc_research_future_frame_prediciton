#!/bin/bash
#SBATCH --job-name=dino-a2-post
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/kcwp264/logs/dino-tracka2-post_%j.out
#SBATCH --error=/scratch/kcwp264/logs/dino-tracka2-post_%j.err

# Step 3/3: post-Phase-3 feature eval, RGB eval, rollout viz, master tables.

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
PRED_CKPT="$REPO/outputs/jigsaws_masterplan/predictor_tracka2_p3/best_val_model.pth"
DEC_CKPT="$PRED_CKPT"
PCA_CKPT="$REPO/outputs/jigsaws_masterplan/predictor_tracka2_p3/pca.pth"
if [ ! -f "$PCA_CKPT" ]; then
  PCA_CKPT="$REPO/outputs/jigsaws_masterplan/predictor_tracka2/pca.pth"
fi
FEAT_OUT="$REPO/outputs/jigsaws_masterplan/eval_feature_copy_last_tracka2_p3"
RGB_OUT="$REPO/outputs/jigsaws_masterplan/eval_tracka2_p3_rgb"
ROLL_OUT="$REPO/outputs/jigsaws_masterplan/tracka2_p3_rollout_viz"

mkdir -p "$FEAT_OUT" "$RGB_OUT" "$ROLL_OUT" /scratch/kcwp264/logs
cd "$REPO"

echo "=== Track A2 post-Phase-3 eval + viz ==="
nvidia-smi -L

"$ENV_DIR/bin/python" -m dino_foresight.eval_feature_copy_last \
    --predictor_ckpt "$PRED_CKPT" \
    --pca_ckpt "$PCA_CKPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$FEAT_OUT" \
    --n_context 4 \
    --n_pred_steps 1,5,10 \
    --stride 6 \
    --batch_size 8 \
    --num_workers 4 \
    --hidden_dim 768 \
    --num_layers 4 \
    --num_heads 8 \
    --split both

"$ENV_DIR/bin/python" -m dino_foresight.evaluate \
    --predictor_ckpt "$PRED_CKPT" \
    --decoder_ckpt "$DEC_CKPT" \
    --pca_ckpt "$PCA_CKPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$RGB_OUT" \
    --encoder_model vitb14_reg \
    --n_context 4 \
    --n_pred_steps 1,5,10 \
    --stride 6 \
    --batch_size 8 \
    --num_workers 4 \
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
    --label tracka2_p3 \
    --stride 6 \
    --batch_size 4

"$ENV_DIR/bin/python" -m dino_foresight.rollout_viz \
    --predictor_ckpt "$PRED_CKPT" \
    --decoder_ckpt "$DEC_CKPT" \
    --pca_ckpt "$PCA_CKPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$ROLL_OUT" \
    --split test \
    --n_pred_steps 10 \
    --stride 6 \
    --n_samples 8 \
    --max_clips_metrics 0

"$ENV_DIR/bin/python" -m dino_foresight.aggregate_results \
    --root "$REPO/outputs/jigsaws_masterplan" \
    --output "$REPO/outputs/jigsaws_masterplan/master_results.md"

echo "=== Post-pipeline done $(date) ==="
echo "Rollout panels: $ROLL_OUT/samples/"
echo "Master tables: $REPO/outputs/jigsaws_masterplan/master_results.md"
