#!/usr/bin/env python3
"""
Add RoboCasa statistics to the pretrained GR00T model's statistics.json file.

When fine-tuning from a pretrained model, the processor loads statistics from
the model checkpoint, not from the dataset. This script adds the RoboCasa
embodiment statistics to the model's statistics.json so training can proceed.

Usage:
    python scripts/add_robocasa_stats_to_model.py \
        --model-path nvidia/GR00T-N1.6-3B \
        --dataset-path /path/to/robocasa/dataset
"""

import argparse
import json
import os
from pathlib import Path


def find_model_statistics_file(model_path: str) -> Path:
    """Find the statistics.json file in the model directory or HF cache."""
    model_path = Path(model_path)

    # Case 1: Local model directory
    if model_path.exists() and model_path.is_dir():
        stats_file = model_path / "statistics.json"
        if stats_file.exists():
            return stats_file
        raise FileNotFoundError(f"statistics.json not found in {model_path}")

    # Case 2: HuggingFace model identifier - check cache
    # Check HF_HOME first, then default cache
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hf_cache = Path(hf_home) / "hub"
    else:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"

    # Convert model name to cache format: nvidia/GR00T-N1.6-3B -> models--nvidia--GR00T-N1.6-3B
    cache_name = "models--" + str(model_path).replace("/", "--")
    cache_dir = hf_cache / cache_name

    if cache_dir.exists():
        # Look for statistics.json in snapshots
        for snapshot_dir in (cache_dir / "snapshots").glob("*"):
            stats_file = snapshot_dir / "statistics.json"
            if stats_file.exists():
                return stats_file

    raise FileNotFoundError(
        f"Could not find statistics.json for model '{model_path}'. "
        f"Make sure the model is downloaded locally or cached by HuggingFace."
    )


def load_dataset_files(dataset_path: Path) -> tuple[dict, dict]:
    """Load both stats.json and modality.json from the dataset."""
    stats_file = dataset_path / "meta" / "stats.json"
    modality_file = dataset_path / "meta" / "modality.json"

    if not stats_file.exists():
        raise FileNotFoundError(f"Dataset stats not found at {stats_file}")
    if not modality_file.exists():
        raise FileNotFoundError(f"Dataset modality not found at {modality_file}")

    with open(stats_file, "r") as f:
        stats = json.load(f)
    with open(modality_file, "r") as f:
        modality = json.load(f)

    return stats, modality


def extract_stats_slice(flat_stats: dict, start: int, end: int) -> dict:
    """Extract a slice of statistics for a specific key."""
    result = {}
    for stat_name in ["mean", "std", "min", "max", "q01", "q99"]:
        if stat_name in flat_stats:
            result[stat_name] = flat_stats[stat_name][start:end]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Add RoboCasa statistics to pretrained model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path or HuggingFace identifier for the pretrained model (e.g., nvidia/GR00T-N1.6-3B)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the RoboCasa dataset directory",
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default="robocasa_panda_omron",
        help="Embodiment tag to use for the statistics",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Adding RoboCasa Statistics to Pretrained Model")
    print("=" * 80)
    print()

    # Find model statistics file
    print(f"🔍 Looking for model statistics...")
    try:
        model_stats_file = find_model_statistics_file(args.model_path)
        print(f"✓ Found: {model_stats_file}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1

    # Load model statistics
    print(f"\n📂 Loading model statistics...")
    with open(model_stats_file, "r") as f:
        model_stats = json.load(f)
    print(f"✓ Loaded {len(model_stats)} embodiments from model")

    # Check if embodiment already exists
    if args.embodiment_tag in model_stats:
        print(f"\n⚠️  Warning: '{args.embodiment_tag}' already exists in model statistics")
        print("   Overwriting...")

    # Load dataset files
    print(f"\n📂 Loading dataset statistics and modality mapping...")
    dataset_path = Path(args.dataset_path)
    try:
        dataset_stats, modality_config = load_dataset_files(dataset_path)
        print(f"✓ Loaded dataset files")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1

    # Verify required keys exist
    if "observation.state" not in dataset_stats:
        print(f"❌ Error: 'observation.state' not found in stats.json")
        return 1
    if "action" not in dataset_stats:
        print(f"❌ Error: 'action' not found in stats.json")
        return 1

    # Parse state statistics from flat array to per-key dicts
    print(f"\n📊 Parsing state statistics...")
    state_stats = {}
    for key, info in modality_config["state"].items():
        start = info["start"]
        end = info["end"]
        state_stats[key] = extract_stats_slice(dataset_stats["observation.state"], start, end)
        print(f"   ✓ {key}: indices [{start}:{end}]")

    # Parse action statistics from flat array to per-key dicts
    print(f"\n📊 Parsing action statistics...")
    action_stats = {}
    for key, info in modality_config["action"].items():
        start = info["start"]
        end = info["end"]
        action_stats[key] = extract_stats_slice(dataset_stats["action"], start, end)
        print(f"   ✓ {key}: indices [{start}:{end}]")

    # For relative_action, we use the same keys as action
    # Map the action keys to our modality config keys
    print(f"\n📊 Preparing relative action statistics...")
    relative_action_stats = {
        "end_effector_position": extract_stats_slice(dataset_stats["action"], 5, 8),
        "end_effector_rotation": extract_stats_slice(dataset_stats["action"], 8, 11),
        "gripper_close": extract_stats_slice(dataset_stats["action"], 11, 12),
    }
    for key in relative_action_stats:
        print(f"   ✓ {key}")

    # Prepare embodiment statistics in the correct format
    print(f"\n📦 Assembling embodiment statistics for '{args.embodiment_tag}'...")
    embodiment_stats = {
        "state": state_stats,
        "action": action_stats,
        "relative_action": relative_action_stats,
    }

    # Add to model statistics
    model_stats[args.embodiment_tag] = embodiment_stats

    # Create backup
    backup_file = model_stats_file.with_suffix(".json.backup")
    print(f"\n💾 Creating backup at: {backup_file}")
    with open(backup_file, "w") as f:
        json.dump(model_stats, f, indent=2)

    # Save updated statistics
    print(f"✏️  Writing updated statistics to: {model_stats_file}")
    with open(model_stats_file, "w") as f:
        json.dump(model_stats, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ Success!")
    print("=" * 80)
    print(f"\nAdded statistics for embodiment: {args.embodiment_tag}")
    print(f"  - state: {len(state_stats)} keys")
    print(f"  - action: {len(action_stats)} keys")
    print(f"  - relative_action: {len(relative_action_stats)} keys")
    print(f"\nModel now supports {len(model_stats)} embodiments:")
    for tag in sorted(model_stats.keys()):
        print(f"  - {tag}")

    print(f"\n🚀 You can now run training:")
    print(f"   bash scripts/train_svms_robocasa_phase1_poc.sh")

    return 0


if __name__ == "__main__":
    exit(main())
