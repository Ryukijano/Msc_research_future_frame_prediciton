#!/bin/bash
#SBATCH --job-name=dino-a2-p3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/kcwp264/logs/dino-tracka2-p3_%j.out
#SBATCH --error=/scratch/kcwp264/logs/dino-tracka2-p3_%j.err

# Step 2/3: light Phase-3 joint finetune — align decoder to Track A2 predicted features.
# 1 GPU, ~16h cap (not 48h). Frozen DINOv2; small pixel+LPIPS on decoder.

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
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket

REPO="/scratch/kcwp264/Msc_research_future_frame_prediciton"
DATA_DIR="$REPO/VPTR_jigsaws_working/jigsaws_suturing/bair_format_dir_louo"
OUTPUT_DIR="$REPO/outputs/jigsaws_masterplan/predictor_tracka2_p3"
PRED_CKPT="$REPO/outputs/jigsaws_masterplan/predictor_tracka2/best_val_model.pth"
DEC_CKPT="$REPO/outputs/jigsaws_masterplan/phase1_decoder/best_val_decoder.pth"
PCA_CKPT="$REPO/outputs/jigsaws_masterplan/predictor_tracka2/pca.pth"

mkdir -p "$OUTPUT_DIR" /scratch/kcwp264/logs
cd "$REPO"

echo "=== Track A2 Phase 3 joint finetune ==="
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
    --lr 5e-5 \
    --decoder_lr 1e-5 \
    --weight_decay 0.05 \
    --epochs 30 \
    --warmup_epochs 2 \
    --num_workers 4 \
    --stride 6 \
    --multi_layer \
    --pca_dim 1152 \
    --pca_ckpt "$PCA_CKPT" \
    --loss_type smooth_l1 \
    --lambda_feat 1.0 \
    --lambda_pixel 0.25 \
    --lambda_lpips 0.25 \
    --residual \
    --use_motion \
    --motion_from feat \
    --motion_weight \
    --joint_finetune \
    --predictor_ckpt "$PRED_CKPT" \
    --decoder_ckpt "$DEC_CKPT" \
    --decoder_type rae \
    --decoder_hidden_dim 1152 \
    --decoder_layers 8 \
    --decoder_heads 12 \
    --eval_interval 999 \
    --eval_max_clips 8 \
    --val_interval 1 \
    --patience 10 \
    --save_interval 5 \
    --val_max_clips 500 \
    --wandb \
    --wandb_project surgical-future-frame-prediction \
    --seed 42

echo "=== Phase 3 done $(date) ==="
