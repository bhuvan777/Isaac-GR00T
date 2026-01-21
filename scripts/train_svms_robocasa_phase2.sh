#!/bin/bash
################################################################################
# SVMS-GR00T Phase 2 Training: Sheaf Activation + DiT Unfreezing
#
# This script continues from Phase 1 checkpoint and activates sheaf consistency
# while unfreezing the bottom 8 layers of the DiT for fine-tuning.
#
# Hardware: Optimized for RTX 32GB VRAM
# Duration: ~16 hours for 10k steps
# Memory: ~24GB (streams + sheaf + DiT bottom 8 layers)
#
# Usage:
#   bash scripts/train_svms_robocasa_phase2.sh
################################################################################

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Model and data paths
BASE_MODEL="nvidia/GR00T-N1.6-3B"
PHASE1_CHECKPOINT="./checkpoints_svms/phase1_poc/checkpoint-5000"  # TODO: Update with your Phase 1 checkpoint
DATASET_PATH="<REPLACE_WITH_YOUR_ROBOCASA_DATASET_PATH>"  # TODO: Set your dataset path
EMBODIMENT_TAG="ROBOCASA_PANDA_OMRON"

# Output directory
OUTPUT_DIR="./checkpoints_svms/phase2_sheaf_activation"
WANDB_PROJECT="svms-groot-robocasa"
WANDB_RUN_NAME="phase2_sheaf_activation"

# Training hyperparameters (Phase 2: Sheaf Activation + DiT Unfreezing)
MAX_STEPS=10000
GLOBAL_BATCH_SIZE=16  # Effective batch size
GRADIENT_ACCUM_STEPS=4  # Physical batch = 16/4 = 4 per GPU
LEARNING_RATE=5e-5  # Lower LR for fine-tuning
WEIGHT_DECAY=0.01

# SVMS parameters (Phase 2: Activate sheaf)
USE_SHEAF_STREAMS=true
LAMBDA_AUX=0.3  # Continue auxiliary supervision
LAMBDA_SHEAF_MAX=0.1  # Gradually ramp up sheaf
LAMBDA_SHEAF_MIN=0.01
SHEAF_SCHEDULE_MODE="adaptive"  # Gradual increase
SHEAF_DELAY_FRAC=0.2  # Start sheaf at 20% of training

# Freezing strategy (Phase 2: Unfreeze DiT bottom 8 layers)
FREEZE_VLM=true  # Keep VLM frozen (except top 4 layers as per N1.6)
FREEZE_DIT_TOP=true  # Keep DiT top layers frozen
UNFREEZE_DIT_BOTTOM_LAYERS=8  # Unfreeze bottom 8 layers

# Checkpointing
SAVE_STEPS=1000
SAVE_TOTAL_LIMIT=5
EVAL_STEPS=500

# Hardware
NUM_GPUS=1
CUDA_DEVICE=0

# =============================================================================
# Validation
# =============================================================================

echo "========================================================================"
echo "SVMS-GR00T Phase 2: Sheaf Activation + DiT Unfreezing"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Phase 1 checkpoint: $PHASE1_CHECKPOINT"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Steps: $MAX_STEPS"
echo "  Batch size: $GLOBAL_BATCH_SIZE (physical: $(($GLOBAL_BATCH_SIZE / $GRADIENT_ACCUM_STEPS)))"
echo "  LR: $LEARNING_RATE"
echo "  Aux weight: $LAMBDA_AUX"
echo "  Sheaf weight: $LAMBDA_SHEAF_MIN → $LAMBDA_SHEAF_MAX (adaptive)"
echo "  DiT layers unfrozen: bottom $UNFREEZE_DIT_BOTTOM_LAYERS"
echo ""

# Check Phase 1 checkpoint
if [[ "$PHASE1_CHECKPOINT" == *"TODO"* ]]; then
    echo "ERROR: Please set PHASE1_CHECKPOINT in the script!"
    echo "  Edit this script and replace with your Phase 1 checkpoint path."
    exit 1
fi

if [ ! -d "$PHASE1_CHECKPOINT" ]; then
    echo "WARNING: Phase 1 checkpoint does not exist: $PHASE1_CHECKPOINT"
    echo "  Make sure Phase 1 training completed successfully."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check dataset path
if [[ "$DATASET_PATH" == *"REPLACE"* ]]; then
    echo "ERROR: Please set DATASET_PATH in the script!"
    exit 1
fi

# =============================================================================
# Training Launch
# =============================================================================

echo "Starting Phase 2 training..."
echo "  Press Ctrl+C to stop"
echo ""

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE uv run python gr00t/experiment/launch_train.py \
    --base-model-path "$BASE_MODEL" \
    --resume-from-checkpoint "$PHASE1_CHECKPOINT" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --modality-config-path gr00t/configs/data/robocasa_modality_config.py \
    --output-dir "$OUTPUT_DIR" \
    --num-gpus $NUM_GPUS \
    \
    --max-steps $MAX_STEPS \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUM_STEPS \
    --learning-rate $LEARNING_RATE \
    --weight-decay $WEIGHT_DECAY \
    \
    --use-sheaf-streams \
    --lambda-aux $LAMBDA_AUX \
    --lambda-sheaf-max $LAMBDA_SHEAF_MAX \
    --lambda-sheaf-min $LAMBDA_SHEAF_MIN \
    --sheaf-schedule-mode $SHEAF_SCHEDULE_MODE \
    --sheaf-delay-until-diffusion $SHEAF_DELAY_FRAC \
    --use-aux-losses \
    \
    --unfreeze-dit-bottom-layers $UNFREEZE_DIT_BOTTOM_LAYERS \
    \
    --save-steps $SAVE_STEPS \
    --save-total-limit $SAVE_TOTAL_LIMIT \
    --eval-steps $EVAL_STEPS \
    \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    \
    --dataloader-num-workers 4 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# =============================================================================
# Post-Training Summary
# =============================================================================

echo ""
echo "========================================================================"
echo "Phase 2 Training Complete!"
echo "========================================================================"
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Training log: $OUTPUT_DIR/training.log"
echo ""
echo "Expected improvements over Phase 1:"
echo "  - Sheaf consistency loss should decrease"
echo "  - Better coherence between streams"
echo "  - Improved action prediction accuracy"
echo ""
echo "Next steps:"
echo "  1. Check W&B for sheaf loss convergence:"
echo "     - loss_sheaf should decrease to < 0.1"
echo "     - Stream weights should balance (33% each)"
echo ""
echo "  2. If Phase 2 training is successful, proceed to Phase 3:"
echo "     bash scripts/train_svms_robocasa_phase3.sh"
echo ""
echo "  3. To evaluate open-loop performance:"
echo "     uv run python gr00t/eval/open_loop_eval_sheaf.py \\"
echo "       --model-path $OUTPUT_DIR/checkpoint-10000 \\"
echo "       --dataset-path $DATASET_PATH \\"
echo "       --embodiment-tag $EMBODIMENT_TAG"
echo ""
echo "========================================================================"
