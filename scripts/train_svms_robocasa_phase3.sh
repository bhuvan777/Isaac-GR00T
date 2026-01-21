#!/bin/bash
################################################################################
# SVMS-GR00T Phase 3 Training: End-to-End Fine-Tuning
#
# This script performs final end-to-end fine-tuning of the full SVMS-GR00T
# model with all components unfrozen (streams, DiT, top VLM layers).
#
# Hardware: Optimized for RTX 32GB VRAM
# Duration: ~10 hours for 5k steps
# Memory: ~28GB (full model)
#
# Usage:
#   bash scripts/train_svms_robocasa_phase3.sh
################################################################################

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Model and data paths
BASE_MODEL="nvidia/GR00T-N1.6-3B"
PHASE2_CHECKPOINT="./checkpoints_svms/phase2_sheaf_activation/checkpoint-10000"  # TODO: Update with your Phase 2 checkpoint
DATASET_PATH="<REPLACE_WITH_YOUR_ROBOCASA_DATASET_PATH>"  # TODO: Set your dataset path
EMBODIMENT_TAG="ROBOCASA_PANDA_OMRON"

# Output directory
OUTPUT_DIR="./checkpoints_svms/phase3_end_to_end"
WANDB_PROJECT="svms-groot-robocasa"
WANDB_RUN_NAME="phase3_end_to_end_finetuning"

# Training hyperparameters (Phase 3: End-to-end)
MAX_STEPS=5000
GLOBAL_BATCH_SIZE=12  # Slightly smaller for full model
GRADIENT_ACCUM_STEPS=4  # Physical batch = 12/4 = 3 per GPU
LEARNING_RATE=1e-5  # Very low LR for stability
WEIGHT_DECAY=0.01

# SVMS parameters (Phase 3: Full integration)
USE_SHEAF_STREAMS=true
LAMBDA_AUX=0.2  # Reduce auxiliary supervision (streams already specialized)
LAMBDA_SHEAF_MAX=0.1  # Constant sheaf weight
SHEAF_SCHEDULE_MODE="constant"  # No ramping, full weight from start

# Freezing strategy (Phase 3: Unfreeze everything)
FREEZE_VLM=false  # Unfreeze top 4 VLM layers (as per GR00T N1.6 design)
FREEZE_DIT=false  # Unfreeze full DiT
UNFREEZE_DIT_BOTTOM_LAYERS=32  # All layers

# Checkpointing
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=5
EVAL_STEPS=250

# Hardware
NUM_GPUS=1
CUDA_DEVICE=0

# =============================================================================
# Validation
# =============================================================================

echo "========================================================================"
echo "SVMS-GR00T Phase 3: End-to-End Fine-Tuning"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Phase 2 checkpoint: $PHASE2_CHECKPOINT"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Steps: $MAX_STEPS"
echo "  Batch size: $GLOBAL_BATCH_SIZE (physical: $(($GLOBAL_BATCH_SIZE / $GRADIENT_ACCUM_STEPS)))"
echo "  LR: $LEARNING_RATE (very low for stability)"
echo "  Aux weight: $LAMBDA_AUX (reduced)"
echo "  Sheaf weight: $LAMBDA_SHEAF_MAX (constant)"
echo "  Model: FULL END-TO-END (all layers unfrozen)"
echo ""

# Check Phase 2 checkpoint
if [[ "$PHASE2_CHECKPOINT" == *"TODO"* ]]; then
    echo "ERROR: Please set PHASE2_CHECKPOINT in the script!"
    echo "  Edit this script and replace with your Phase 2 checkpoint path."
    exit 1
fi

if [ ! -d "$PHASE2_CHECKPOINT" ]; then
    echo "WARNING: Phase 2 checkpoint does not exist: $PHASE2_CHECKPOINT"
    echo "  Make sure Phase 2 training completed successfully."
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
# Memory Warning
# =============================================================================

echo "========================================================================"
echo "MEMORY WARNING"
echo "========================================================================"
echo ""
echo "Phase 3 trains the full model and requires ~28GB VRAM."
echo "If you encounter OOM errors:"
echo "  1. Reduce batch size to 8 (GLOBAL_BATCH_SIZE=8)"
echo "  2. Increase gradient accumulation (GRADIENT_ACCUM_STEPS=8)"
echo "  3. Use gradient checkpointing (add --gradient-checkpointing flag)"
echo ""
read -p "Continue with Phase 3? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# =============================================================================
# Training Launch
# =============================================================================

echo "Starting Phase 3 training..."
echo "  Press Ctrl+C to stop"
echo ""

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE uv run python gr00t/experiment/launch_train.py \
    --base-model-path "$BASE_MODEL" \
    --resume-from-checkpoint "$PHASE2_CHECKPOINT" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --output-dir "$OUTPUT_DIR" \
    --num-gpus $NUM_GPUS \
    \
    --max-steps $MAX_STEPS \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUM_STEPS \
    --learning-rate $LEARNING_RATE \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps 500 \
    \
    --use-sheaf-streams \
    --lambda-aux $LAMBDA_AUX \
    --lambda-sheaf-max $LAMBDA_SHEAF_MAX \
    --sheaf-schedule-mode $SHEAF_SCHEDULE_MODE \
    --use-aux-losses \
    \
    --no-freeze-vlm \
    --no-freeze-dit \
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
echo "Phase 3 Training Complete!"
echo "========================================================================"
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Training log: $OUTPUT_DIR/training.log"
echo ""
echo "SVMS-GR00T training complete! You now have:"
echo "  - Specialized streams (Phase 1)"
echo "  - Sheaf-consistent representations (Phase 2)"
echo "  - End-to-end optimized model (Phase 3)"
echo ""
echo "Next steps:"
echo ""
echo "  1. Evaluate on RoboCasa test set:"
echo "     uv run python gr00t/eval/open_loop_eval_sheaf.py \\"
echo "       --model-path $OUTPUT_DIR/checkpoint-5000 \\"
echo "       --dataset-path $DATASET_PATH \\"
echo "       --embodiment-tag $EMBODIMENT_TAG \\"
echo "       --split test"
echo ""
echo "  2. Compare against baseline GR00T:"
echo "     bash scripts/compare_baseline_svms.sh"
echo ""
echo "  3. Deploy for closed-loop evaluation:"
echo "     uv run python gr00t/eval/sim/robocasa/evaluate_robocasa.py \\"
echo "       --model-path $OUTPUT_DIR/checkpoint-5000 \\"
echo "       --task pick_and_place_coffee \\"
echo "       --num-episodes 50"
echo ""
echo "  4. Analyze stream specialization:"
echo "     uv run python gr00t/eval/analyze_stream_specialization.py \\"
echo "       --model-path $OUTPUT_DIR/checkpoint-5000 \\"
echo "       --dataset-path $DATASET_PATH"
echo ""
echo "========================================================================"
echo ""
echo "Expected performance (based on GSM8K results):"
echo "  - Baseline GR00T N1.6: ~65-70% success rate"
echo "  - SVMS-GR00T: ~75-82% success rate (estimated)"
echo "  - Gains from multi-stream reasoning and sheaf consistency"
echo ""
echo "Check W&B for detailed metrics and comparisons!"
echo "========================================================================"
