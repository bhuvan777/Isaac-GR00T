#!/usr/bin/env python3
"""
RoboCasa Dataset Preparation for GR00T N1.6

This script converts raw RoboCasa demonstrations (absolute coordinates) to
GR00T-compatible format (state-relative coordinates).

Key conversions:
- Absolute position [x,y,z] → Relative deltas [Δx,Δy,Δz]
- Quaternion rotations → Euler angles → Relative deltas
- Compute action chunks with horizon=16
- Generate normalization statistics
- Validate conversions

Usage:
    # Basic usage
    python scripts/prepare_robocasa_for_groot.py \
        --input /path/to/raw/robocasa/demos \
        --output ./data/robocasa_groot_format

    # With validation and visualization
    python scripts/prepare_robocasa_for_groot.py \
        --input /path/to/raw/robocasa/demos \
        --output ./data/robocasa_groot_format \
        --validate \
        --visualize-samples 5 \
        --action-horizon 16

    # Process specific task
    python scripts/prepare_robocasa_for_groot.py \
        --input /path/to/raw/robocasa/demos \
        --output ./data/robocasa_groot_format \
        --task pick_and_place_coffee \
        --max-episodes 100

Author: Adapted for GR00T N1.6 SVMS training
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gr00t.data.robocasa_dataset_processor import RoboCasaDatasetProcessor
from gr00t.configs.data.robocasa_modality_config import (
    ROBOCASA_PANDA_OMRON,
    update_normalization_stats,
    validate_config,
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert RoboCasa dataset to GR00T-compatible format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw RoboCasa dataset directory (contains .hdf5 files)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed GR00T-format dataset",
    )

    # Dataset configuration
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=16,
        help="Action horizon for chunked predictions (GR00T default: 16)",
    )
    parser.add_argument(
        "--use-relative-actions",
        action="store_true",
        default=True,
        help="Use state-relative action space (recommended for GR00T N1.6)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter to specific task (e.g., 'pick_and_place_coffee'). If None, process all tasks.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process (for testing). If None, process all.",
    )

    # Validation and debugging
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks after processing",
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=0,
        help="Number of sample trajectories to visualize (saved as PNG)",
    )
    parser.add_argument(
        "--compute-stats-only",
        action="store_true",
        help="Only compute normalization statistics without full processing",
    )

    # Output configuration
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["lerobot", "hdf5", "both"],
        default="lerobot",
        help="Output format: LeRobot v2, HDF5, or both",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress output dataset (saves disk space)",
    )

    # Processing options
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for processing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information",
    )

    return parser.parse_args()


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def check_input_directory(input_path: str) -> bool:
    """
    Check if input directory exists and contains RoboCasa data.

    Args:
        input_path: Path to input directory

    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(input_path):
        print(f"ERROR: Input directory does not exist: {input_path}")
        return False

    if not os.path.isdir(input_path):
        print(f"ERROR: Input path is not a directory: {input_path}")
        return False

    # Check for .hdf5 files
    hdf5_files = list(Path(input_path).rglob("*.hdf5"))
    if len(hdf5_files) == 0:
        print(f"WARNING: No .hdf5 files found in {input_path}")
        print("  Make sure your RoboCasa dataset is in the correct location.")
        return False

    print(f"✓ Found {len(hdf5_files)} .hdf5 files in input directory")
    return True


def create_output_directory(output_path: str):
    """Create output directory structure"""
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "episodes"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "visualizations"), exist_ok=True)
    print(f"✓ Created output directory: {output_path}")


def save_metadata(
    output_path: str,
    config: dict,
    stats: Optional[dict] = None,
    args: Optional[argparse.Namespace] = None,
):
    """
    Save metadata and configuration to output directory.

    Args:
        output_path: Output directory
        config: Modality configuration
        stats: Normalization statistics (if available)
        args: Command-line arguments
    """
    meta = {
        "embodiment": config["embodiment_name"],
        "robot_type": config["robot_type"],
        "gripper_type": config["gripper_type"],
        "state_dim": config["state_dim"],
        "action_dim": config["action_dim"],
        "action_horizon": config["action_horizon"],
        "action_space": config["action_space"],
        "cameras": list(config["cameras"].keys()),
        "data_format": config["metadata"]["data_format"],
    }

    if stats is not None:
        meta["normalization"] = stats

    if args is not None:
        meta["processing_args"] = {
            "input_path": args.input,
            "action_horizon": args.action_horizon,
            "use_relative_actions": args.use_relative_actions,
            "task_filter": args.task,
            "max_episodes": args.max_episodes,
        }

    meta_path = os.path.join(output_path, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✓ Saved metadata to: {meta_path}")


def main():
    args = parse_args()

    print_section("RoboCasa Dataset Preparation for GR00T N1.6")

    print("Configuration:")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Action horizon: {args.action_horizon}")
    print(f"  Relative actions: {args.use_relative_actions}")
    if args.task:
        print(f"  Task filter: {args.task}")
    if args.max_episodes:
        print(f"  Max episodes: {args.max_episodes}")
    print()

    # Step 1: Validate input
    print_section("Step 1: Validating Input")
    if not check_input_directory(args.input):
        sys.exit(1)

    # Step 2: Create output directory
    print_section("Step 2: Creating Output Directory")
    create_output_directory(args.output)

    # Step 3: Load configuration
    print_section("Step 3: Loading Configuration")
    config = ROBOCASA_PANDA_OMRON.copy()
    config["action_horizon"] = args.action_horizon

    try:
        validate_config(config)
    except ValueError as e:
        print(f"ERROR: Configuration validation failed: {e}")
        sys.exit(1)

    # Step 4: Initialize processor
    print_section("Step 4: Initializing Dataset Processor")
    processor = RoboCasaDatasetProcessor(
        action_horizon=args.action_horizon,
        use_relative_actions=args.use_relative_actions,
    )
    print("✓ Processor initialized")

    # Step 5: Compute normalization statistics
    print_section("Step 5: Computing Normalization Statistics")
    print("Scanning dataset to compute mean and std...")

    try:
        stats = processor.compute_normalization_stats(
            input_path=args.input,
            task_filter=args.task,
            max_episodes=args.max_episodes,
            num_workers=args.num_workers,
        )

        print("\n✓ Normalization statistics computed:")
        print(f"  State mean: {stats['state']['mean'][:3]}... (first 3 dims)")
        print(f"  State std:  {stats['state']['std'][:3]}... (first 3 dims)")
        print(f"  Action mean: {stats['action']['mean'][:3]}... (first 3 dims)")
        print(f"  Action std:  {stats['action']['std'][:3]}... (first 3 dims)")

        # Update config with computed stats
        config = update_normalization_stats(config, stats)

        # Save metadata with stats
        save_metadata(args.output, config, stats, args)

    except Exception as e:
        print(f"ERROR: Failed to compute normalization statistics: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if args.compute_stats_only:
        print("\n✓ Statistics computation complete (--compute-stats-only mode)")
        print(f"\nNormalization stats saved to: {args.output}/meta.json")
        print("\nTo use these stats, update gr00t/configs/data/robocasa_modality_config.py:")
        print(f"  normalization: {json.dumps(stats, indent=2)}")
        sys.exit(0)

    # Step 6: Process full dataset
    print_section("Step 6: Processing Dataset")
    print("Converting episodes to GR00T format...")
    print("This may take several minutes depending on dataset size.\n")

    try:
        processed_data = processor.process_full_dataset(
            input_path=args.input,
            output_path=args.output,
            task_filter=args.task,
            max_episodes=args.max_episodes,
            save_format=args.save_format,
            compress=args.compress,
            num_workers=args.num_workers,
            verbose=args.verbose,
        )

        print(f"\n✓ Successfully processed {len(processed_data)} episodes")

    except Exception as e:
        print(f"\nERROR: Dataset processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Step 7: Validation
    if args.validate:
        print_section("Step 7: Validating Processed Dataset")
        print("Running validation checks...")

        try:
            validation_results = processor.validate_dataset(processed_data)

            print("\n✓ Validation complete:")
            for key, value in validation_results.items():
                if isinstance(value, bool):
                    status = "PASS" if value else "FAIL"
                    print(f"  {key}: {status}")
                else:
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"\nWARNING: Validation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Step 8: Visualization
    if args.visualize_samples > 0:
        print_section("Step 8: Visualizing Sample Trajectories")
        print(f"Generating visualizations for {args.visualize_samples} samples...\n")

        viz_dir = os.path.join(args.output, "visualizations")
        num_visualized = min(args.visualize_samples, len(processed_data))

        for i in range(num_visualized):
            try:
                episode_data = processed_data[i]
                viz_path = os.path.join(viz_dir, f"episode_{i:04d}.png")

                processor.visualize_trajectory(
                    episode_data=episode_data,
                    save_path=viz_path,
                    title=f"Episode {i}",
                )

                print(f"  ✓ Saved visualization: {viz_path}")

            except Exception as e:
                print(f"  WARNING: Failed to visualize episode {i}: {e}")

        print(f"\n✓ Generated {num_visualized} visualizations")

    # Final summary
    print_section("Dataset Preparation Complete!")

    print("Summary:")
    print(f"  Total episodes processed: {len(processed_data)}")
    print(f"  Output directory: {args.output}")
    print(f"  Metadata: {args.output}/meta.json")
    if args.visualize_samples > 0:
        print(f"  Visualizations: {args.output}/visualizations/")
    print()

    print("Next steps:")
    print("  1. Update gr00t/configs/data/robocasa_modality_config.py with computed stats")
    print(f"     (see {args.output}/meta.json)")
    print()
    print("  2. Run Phase 1 training:")
    print("     bash scripts/train_svms_robocasa_phase1_poc.sh")
    print()
    print("  3. Make sure to set DATASET_PATH in the training script to:")
    print(f"     DATASET_PATH=\"{args.output}\"")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
