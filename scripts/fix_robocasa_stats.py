#!/usr/bin/env python3
"""
Fix RoboCasa stats.json by adding relative_action statistics.

The actions in the dataset are already relative (as confirmed by modality.json),
but the stats.json file doesn't have a "relative_action" section that GR00T expects.

This script extracts the relevant action dimensions and creates the relative_action
statistics based on the modality configuration.

Usage:
    python scripts/fix_robocasa_stats.py --dataset-path /path/to/dataset
"""

import argparse
import json
from pathlib import Path


def extract_action_stats(action_stats: dict, start_idx: int, end_idx: int) -> dict:
    """Extract statistics for a slice of the action vector."""
    result = {}
    for stat_name in ["mean", "std", "min", "max", "q01", "q99"]:
        if stat_name in action_stats:
            result[stat_name] = action_stats[stat_name][start_idx:end_idx]
    return result


def main():
    parser = argparse.ArgumentParser(description="Fix RoboCasa stats.json for GR00T training")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset directory (e.g., single_panda_gripper.CloseDoubleDoor)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    stats_file = dataset_path / "meta" / "stats.json"

    if not stats_file.exists():
        print(f"❌ Error: stats.json not found at {stats_file}")
        return 1

    print(f"📂 Loading stats from: {stats_file}")
    with open(stats_file, "r") as f:
        stats = json.load(f)

    if "action" not in stats:
        print("❌ Error: 'action' key not found in stats.json")
        return 1

    if "relative_action" in stats:
        print("⚠️  Warning: 'relative_action' already exists in stats.json")
        print("   Overwriting with new values...")

    # Based on modality.json from the dataset:
    # action indices:
    #   0-3: base_motion (not used in our config)
    #   4: control_mode (not used)
    #   5-7: end_effector_position (RELATIVE)
    #   8-10: end_effector_rotation (RELATIVE, axis-angle)
    #   11: gripper_close (ABSOLUTE/binary)
    #
    # Our ModalityConfig uses:
    #   - end_effector_position (indices 5-7)
    #   - end_effector_rotation (indices 8-10)
    #   - gripper_close (index 11)

    print("📊 Extracting relative action statistics...")
    action_stats = stats["action"]

    relative_action_stats = {
        "end_effector_position": extract_action_stats(action_stats, 5, 8),
        "end_effector_rotation": extract_action_stats(action_stats, 8, 11),
        "gripper_close": extract_action_stats(action_stats, 11, 12),
    }

    # Add the relative_action section
    stats["relative_action"] = relative_action_stats

    # Create backup
    backup_file = stats_file.with_suffix(".json.backup")
    print(f"💾 Creating backup at: {backup_file}")
    with open(backup_file, "w") as f:
        json.dump(stats, f, indent=4)

    # Save updated stats
    print(f"✏️  Writing updated stats to: {stats_file}")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4)

    print("\n✅ Success! Added relative_action statistics:")
    print(f"   - end_effector_position: 3D ({relative_action_stats['end_effector_position']['mean']})")
    print(f"   - end_effector_rotation: 3D ({relative_action_stats['end_effector_rotation']['mean']})")
    print(f"   - gripper_close: 1D ({relative_action_stats['gripper_close']['mean']})")
    print("\n🚀 You can now run training:")
    print(f"   bash scripts/train_svms_robocasa_phase1_poc.sh")

    return 0


if __name__ == "__main__":
    exit(main())
