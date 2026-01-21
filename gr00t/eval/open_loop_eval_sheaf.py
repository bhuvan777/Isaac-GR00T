#!/usr/bin/env python3
"""
Open-Loop Evaluation for SVMS-GR00T

This script evaluates trained SVMS-GR00T models on held-out test data
in an open-loop setting (no environment interaction).

Metrics computed:
- Action prediction accuracy (L1, L2 errors)
- Position delta errors (x, y, z)
- Rotation delta errors (roll, pitch, yaw)
- Gripper command accuracy
- Success trajectory completion rate

Usage:
    # Evaluate single checkpoint
    python gr00t/eval/open_loop_eval_sheaf.py \
        --model-path ./checkpoints_svms/phase3_end_to_end/checkpoint-5000 \
        --dataset-path ./data/robocasa_groot_format \
        --embodiment-tag ROBOCASA_PANDA_OMRON \
        --split test

    # Compare baseline vs SVMS
    python gr00t/eval/open_loop_eval_sheaf.py \
        --baseline-path ./checkpoints/baseline_groot \
        --svms-path ./checkpoints_svms/phase3_end_to_end/checkpoint-5000 \
        --dataset-path ./data/robocasa_groot_format \
        --embodiment-tag ROBOCASA_PANDA_OMRON \
        --compare

Author: SVMS-GR00T Evaluation
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# GR00T imports
from transformers import AutoTokenizer, AutoModel


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Open-loop evaluation for SVMS-GR00T",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to SVMS-GR00T checkpoint",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default=None,
        help="Path to baseline GR00T checkpoint (for comparison)",
    )
    parser.add_argument(
        "--svms-path",
        type=str,
        default=None,
        help="Path to SVMS checkpoint (for comparison mode)",
    )

    # Data
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to processed dataset",
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default="ROBOCASA_PANDA_OMRON",
        help="Embodiment configuration tag",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )

    # Evaluation settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to evaluate (for quick tests)",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=16,
        help="Action prediction horizon",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predicted trajectories for visualization",
    )

    # Comparison mode
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare baseline vs SVMS (requires --baseline-path and --svms-path)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )

    return parser.parse_args()


def load_model(model_path: str, device: str):
    """
    Load GR00T model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        model: Loaded model
        tokenizer: Associated tokenizer
    """
    print(f"Loading model from: {model_path}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
        ).to(device)

        model.eval()
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"  SVMS enabled: {hasattr(model, 'svms_wrapper') and model.svms_wrapper is not None}")

        return model, tokenizer

    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        raise


def compute_action_errors(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
) -> Dict[str, float]:
    """
    Compute action prediction errors.

    Args:
        pred_actions: Predicted actions (B, H, 7) or (B, 7)
        gt_actions: Ground truth actions (same shape)

    Returns:
        Dictionary of error metrics
    """
    # Handle horizon dimension
    if pred_actions.ndim == 3:
        # Average over horizon
        pred_actions = pred_actions.mean(axis=1)  # (B, 7)
        gt_actions = gt_actions[:, 0, :]  # Use first action in horizon

    # Overall errors
    l1_error = np.abs(pred_actions - gt_actions).mean()
    l2_error = np.sqrt(((pred_actions - gt_actions) ** 2).mean())

    # Component-wise errors
    pos_error = np.abs(pred_actions[:, :3] - gt_actions[:, :3]).mean()
    rot_error = np.abs(pred_actions[:, 3:6] - gt_actions[:, 3:6]).mean()
    gripper_error = np.abs(pred_actions[:, 6] - gt_actions[:, 6]).mean()

    # Detailed position errors
    pos_x_error = np.abs(pred_actions[:, 0] - gt_actions[:, 0]).mean()
    pos_y_error = np.abs(pred_actions[:, 1] - gt_actions[:, 1]).mean()
    pos_z_error = np.abs(pred_actions[:, 2] - gt_actions[:, 2]).mean()

    # Detailed rotation errors
    rot_roll_error = np.abs(pred_actions[:, 3] - gt_actions[:, 3]).mean()
    rot_pitch_error = np.abs(pred_actions[:, 4] - gt_actions[:, 4]).mean()
    rot_yaw_error = np.abs(pred_actions[:, 5] - gt_actions[:, 5]).mean()

    return {
        "l1_error": float(l1_error),
        "l2_error": float(l2_error),
        "pos_error": float(pos_error),
        "rot_error": float(rot_error),
        "gripper_error": float(gripper_error),
        "pos_x_error": float(pos_x_error),
        "pos_y_error": float(pos_y_error),
        "pos_z_error": float(pos_z_error),
        "rot_roll_error": float(rot_roll_error),
        "rot_pitch_error": float(rot_pitch_error),
        "rot_yaw_error": float(rot_yaw_error),
    }


def evaluate_model(
    model,
    tokenizer,
    dataloader,
    device: str,
    max_episodes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on dataset.

    Args:
        model: GR00T model
        tokenizer: Tokenizer
        dataloader: Test dataloader
        device: Device
        max_episodes: Maximum episodes to evaluate

    Returns:
        Dictionary of evaluation metrics
    """
    all_pred_actions = []
    all_gt_actions = []
    num_episodes = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = model(batch)

            # Extract predictions
            if hasattr(outputs, "action_logits"):
                # GR00T returns action logits
                pred_actions = outputs.action_logits  # (B, H, D)
            elif hasattr(outputs, "logits"):
                pred_actions = outputs.logits
            else:
                raise ValueError("Could not extract action predictions from outputs")

            # Extract ground truth
            if "actions" in batch:
                gt_actions = batch["actions"]
            elif "labels" in batch:
                gt_actions = batch["labels"]
            else:
                raise ValueError("Could not find ground truth actions in batch")

            # Accumulate
            all_pred_actions.append(pred_actions.cpu().numpy())
            all_gt_actions.append(gt_actions.cpu().numpy())

            num_episodes += batch["actions"].shape[0] if "actions" in batch else batch["labels"].shape[0]

            if max_episodes is not None and num_episodes >= max_episodes:
                break

    # Concatenate all batches
    all_pred_actions = np.concatenate(all_pred_actions, axis=0)
    all_gt_actions = np.concatenate(all_gt_actions, axis=0)

    # Compute metrics
    metrics = compute_action_errors(all_pred_actions, all_gt_actions)
    metrics["num_episodes"] = num_episodes

    return metrics, all_pred_actions, all_gt_actions


def print_results(results: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print evaluation results"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()
    print(f"Episodes evaluated: {results.get('num_episodes', 'N/A')}")
    print()
    print("Overall Errors:")
    print(f"  L1 error:      {results['l1_error']:.4f}")
    print(f"  L2 error:      {results['l2_error']:.4f}")
    print()
    print("Component Errors:")
    print(f"  Position:      {results['pos_error']:.4f}")
    print(f"    - X:         {results['pos_x_error']:.4f}")
    print(f"    - Y:         {results['pos_y_error']:.4f}")
    print(f"    - Z:         {results['pos_z_error']:.4f}")
    print(f"  Rotation:      {results['rot_error']:.4f} rad")
    print(f"    - Roll:      {results['rot_roll_error']:.4f} rad")
    print(f"    - Pitch:     {results['rot_pitch_error']:.4f} rad")
    print(f"    - Yaw:       {results['rot_yaw_error']:.4f} rad")
    print(f"  Gripper:       {results['gripper_error']:.4f}")
    print()
    print("=" * 80)


def compare_models(baseline_results: Dict, svms_results: Dict):
    """Print comparison between baseline and SVMS"""
    print("\n" + "=" * 80)
    print("  Baseline vs SVMS Comparison")
    print("=" * 80)
    print()

    metrics_to_compare = [
        ("l1_error", "L1 Error", "lower"),
        ("l2_error", "L2 Error", "lower"),
        ("pos_error", "Position Error", "lower"),
        ("rot_error", "Rotation Error", "lower"),
        ("gripper_error", "Gripper Error", "lower"),
    ]

    print(f"{'Metric':<25} {'Baseline':<12} {'SVMS':<12} {'Improvement':<15}")
    print("-" * 80)

    for key, name, direction in metrics_to_compare:
        baseline_val = baseline_results[key]
        svms_val = svms_results[key]

        if direction == "lower":
            improvement = ((baseline_val - svms_val) / baseline_val) * 100
            symbol = "↓" if improvement > 0 else "↑"
        else:
            improvement = ((svms_val - baseline_val) / baseline_val) * 100
            symbol = "↑" if improvement > 0 else "↓"

        print(f"{name:<25} {baseline_val:<12.4f} {svms_val:<12.4f} {symbol} {abs(improvement):>6.2f}%")

    print("=" * 80)


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    # TODO: Implement dataset loading based on your format
    # This is a placeholder - replace with actual dataset loading
    print("WARNING: Dataset loading not implemented - this is a template script")
    print("Please implement dataset loading based on your LeRobot v2 format")

    # Comparison mode
    if args.compare:
        if not args.baseline_path or not args.svms_path:
            raise ValueError("--compare requires both --baseline-path and --svms-path")

        # Load baseline
        baseline_model, baseline_tokenizer = load_model(args.baseline_path, args.device)

        # Load SVMS
        svms_model, svms_tokenizer = load_model(args.svms_path, args.device)

        # TODO: Create dataloader
        # dataloader = create_dataloader(args.dataset_path, args.split, args.batch_size)

        # Evaluate baseline
        print("\nEvaluating baseline model...")
        # baseline_results, _, _ = evaluate_model(baseline_model, baseline_tokenizer, dataloader, args.device, args.max_episodes)
        # print_results(baseline_results, "Baseline GR00T Results")

        # Evaluate SVMS
        print("\nEvaluating SVMS model...")
        # svms_results, _, _ = evaluate_model(svms_model, svms_tokenizer, dataloader, args.device, args.max_episodes)
        # print_results(svms_results, "SVMS-GR00T Results")

        # Compare
        # compare_models(baseline_results, svms_results)

        print("\nComparison mode - TODO: Implement dataset loading")

    else:
        # Single model evaluation
        if not args.model_path:
            raise ValueError("Either --model-path or --compare mode is required")

        model, tokenizer = load_model(args.model_path, args.device)

        # TODO: Create dataloader
        # dataloader = create_dataloader(args.dataset_path, args.split, args.batch_size)

        # Evaluate
        print("\nEvaluating model...")
        # results, pred_actions, gt_actions = evaluate_model(model, tokenizer, dataloader, args.device, args.max_episodes)
        # print_results(results)

        print("\nSingle model evaluation - TODO: Implement dataset loading")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
