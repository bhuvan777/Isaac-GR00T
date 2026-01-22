#!/bin/bash
################################################################################
# SVMS-GR00T Phase 1 Training: Stream Specialization on SO100
#
# This script trains the sheaf-based multi-stream architecture on SO100 dataset.
# Phase 1 focuses on stream specialization with auxiliary supervision.
#
# Hardware: Optimized for RTX 32GB VRAM
# Duration: ~4 hours for 5k steps (SO100 is simpler than RoboCasa)
# Memory: ~18GB (streams only, DiT frozen)
#
# Usage:
#   bash scripts/train_svms_so100_phase1.sh
################################################################################

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Model and data paths
BASE_MODEL="nvidia/GR00T-N1.6-3B"
DATASET_PATH="examples/SO100/izuluaga/finish_sandwich"
MODALITY_CONFIG_PATH="examples/SO100/so100_config.py"
EMBODIMENT_TAG="NEW_EMBODIMENT"  # SO100 uses NEW_EMBODIMENT

# Output directory
OUTPUT_DIR="./checkpoints_svms/so100_phase1"
WANDB_PROJECT="svms-groot-so100"
WANDB_RUN_NAME="phase1_stream_specialization"

# Training hyperparameters (Phase 1: Stream Specialization)
MAX_STEPS=5000
GLOBAL_BATCH_SIZE=16  # Effective batch size across GPUs and accumulation
GRADIENT_ACCUM_STEPS=4  # Physical batch = 16/4 = 4 per GPU
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01

# SVMS parameters
USE_SHEAF_STREAMS=true
LAMBDA_AUX=0.5  # Strong auxiliary supervision in Phase 1
LAMBDA_SHEAF=0.0  # Sheaf OFF in Phase 1 (let streams diverge)

# Freezing strategy (Phase 1: Only train streams)
FREEZE_DIT=true  # Keep DiT frozen
FREEZE_VLM=true  # Keep VLM frozen (except top 4 layers as per N1.6)

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
echo "SVMS-GR00T Phase 1: Stream Specialization on SO100"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Steps: $MAX_STEPS"
echo "  Batch size: $GLOBAL_BATCH_SIZE (physical: $(($GLOBAL_BATCH_SIZE / $GRADIENT_ACCUM_STEPS)))"
echo "  LR: $LEARNING_RATE"
echo "  Aux weight: $LAMBDA_AUX"
echo "  Sheaf weight: $LAMBDA_SHEAF (OFF in Phase 1)"
echo ""

# Check dataset path
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset path does not exist: $DATASET_PATH"
    echo "  Please download SO100 dataset first."
    echo "  See: examples/SO100/README.md"
    exit 1
fi

# Check modality config
if [ ! -f "$MODALITY_CONFIG_PATH" ]; then
    echo "ERROR: Modality config not found: $MODALITY_CONFIG_PATH"
    exit 1
fi

# =============================================================================
# Training Launch
# =============================================================================

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting Phase 1 training..."
echo "  Press Ctrl+C to stop"
echo ""

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --modality-config-path "$MODALITY_CONFIG_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --output-dir "$OUTPUT_DIR" \
    --num-gpus $NUM_GPUS \
    \
    --max-steps $MAX_STEPS \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUM_STEPS \
    --learning-rate $LEARNING_RATE \
    --weight-decay $WEIGHT_DECAY \
    \
    --tune-projector False \
    --tune-diffusion-model False \
    \
    --use-sheaf-streams \
    --lambda-aux $LAMBDA_AUX \
    --lambda-sheaf-max $LAMBDA_SHEAF \
    \
    --save-steps $SAVE_STEPS \
    --save-total-limit $SAVE_TOTAL_LIMIT \
    \
    --use-wandb \
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
echo "Phase 1 Training Complete!"
echo "========================================================================"
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Training log: $OUTPUT_DIR/training.log"
echo ""
echo "Next steps:"
echo "  1. Check auxiliary accuracy in W&B:"
echo "     - aux_acc_A (Visual) should be > 70%"
echo "     - aux_acc_B (Temporal) should be > 70%"
echo "     - aux_acc_C (State) should be > 65%"
echo ""
echo "  2. If streams are well-specialized, proceed to Phase 2:"
echo "     - Increase sheaf regularization gradually"
echo "     - Start unfreezing DiT layers"
echo ""
echo "  3. To evaluate the model:"
echo "     - Use GR00T's evaluation tools on SO100 test set"
echo ""
echo "========================================================================"
