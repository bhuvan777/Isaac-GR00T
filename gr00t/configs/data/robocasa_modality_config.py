"""
RoboCasa Modality Configuration for GR00T N1.6

Defines embodiment-specific configurations for RoboCasa Panda robot with Omron gripper.

This module provides two configs:
1. ROBOCASA_PANDA_OMRON_CONFIG: The proper ModalityConfig structure for GR00T training
   - Contains video, state, action, and language modality configurations
   - Used by the training system (automatically registered)

2. ROBOCASA_PANDA_OMRON_METADATA: Metadata and documentation (legacy dict format)
   - Contains camera specs, normalization params, action space details
   - Used for reference and documentation

Usage:
    # The config is automatically registered when this module is imported
    # You don't need to manually import it, just pass the embodiment tag:
    --embodiment-tag ROBOCASA_PANDA_OMRON
    --modality-config-path gr00t/configs/data/robocasa_modality_config.py
"""

import numpy as np

# =============================================================================
# RoboCasa Panda + Omron Gripper Configuration
# =============================================================================

ROBOCASA_PANDA_OMRON = {
    # =========================================================================
    # Embodiment Information
    # =========================================================================
    "embodiment_name": "ROBOCASA_PANDA_OMRON",
    "robot_type": "Franka Panda",
    "gripper_type": "Omron 2-finger parallel jaw",
    "workspace": "Kitchen countertop manipulation",

    # =========================================================================
    # State Configuration
    # =========================================================================
    "state_dim": 14,  # Total state dimension
    "state_components": {
        "end_effector_position_relative": 3,  # [Δx, Δy, Δz] in meters
        "end_effector_rotation_relative": 3,  # [Δroll, Δpitch, Δyaw] in radians
        "joint_position": 7,                  # 7-DOF arm joint angles in radians
        "gripper_position": 1,                # Gripper state [0=open, 1=closed]
    },

    "state_keys": [
        "end_effector_position_relative",
        "end_effector_rotation_relative",
        "joint_position",
        "gripper_position",
    ],

    # State ranges (for clipping/validation)
    "state_ranges": {
        "end_effector_position_relative": {
            "min": [-0.5, -0.5, -0.5],  # Max 0.5m per step
            "max": [0.5, 0.5, 0.5],
        },
        "end_effector_rotation_relative": {
            "min": [-np.pi/2, -np.pi/2, -np.pi/2],  # Max 90° per step
            "max": [np.pi/2, np.pi/2, np.pi/2],
        },
        "joint_position": {
            "min": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],  # Panda joint limits
            "max": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        },
        "gripper_position": {
            "min": [0.0],  # Fully open
            "max": [1.0],  # Fully closed
        },
    },

    # =========================================================================
    # Action Configuration
    # =========================================================================
    "action_dim": 7,  # [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]
    "action_horizon": 16,  # Number of future steps to predict
    "action_space": "relative",  # KEY: GR00T N1.6 uses relative actions!

    "action_components": {
        "position_delta": 3,       # [Δx, Δy, Δz]
        "rotation_delta": 3,       # [Δroll, Δpitch, Δyaw]
        "gripper_delta": 1,        # Δgripper
    },

    # Action ranges (for clipping)
    "action_ranges": {
        "position_delta": {
            "min": [-0.5, -0.5, -0.5],
            "max": [0.5, 0.5, 0.5],
        },
        "rotation_delta": {
            "min": [-np.pi/2, -np.pi/2, -np.pi/2],
            "max": [np.pi/2, np.pi/2, np.pi/2],
        },
        "gripper_delta": {
            "min": [-1.0],
            "max": [1.0],
        },
    },

    # =========================================================================
    # Camera Configuration
    # =========================================================================
    "cameras": {
        "wrist": {
            "key": "robot0_eye_in_hand",         # RoboCasa observation key
            "resolution": [480, 640],             # [height, width]
            "crop_size": None,                    # None = use full image
            "shortest_edge": 256,                 # For GR00T's flexible resolution
            "crop_fraction": 0.95,
        },
        "front": {
            "key": "robot0_frontview",
            "resolution": [480, 640],
            "crop_size": None,
            "shortest_edge": 256,
            "crop_fraction": 0.95,
        },
    },

    "primary_camera": "wrist",  # Used if only single camera
    "use_multiple_cameras": True,

    # =========================================================================
    # Normalization Parameters
    # =========================================================================
    # NOTE: These are PLACEHOLDER values!
    # Actual values should be computed from your dataset using:
    #   python scripts/prepare_robocasa_for_groot.py --compute-stats
    #
    # After processing, update these with the computed values from meta.json

    "normalization": {
        "state": {
            "mean": [
                # EEF position deltas (usually ~0 for relative)
                0.0, 0.0, 0.0,
                # EEF rotation deltas (usually ~0 for relative)
                0.0, 0.0, 0.0,
                # Joint positions (centered around typical config)
                0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0,
                # Gripper (usually mid-range)
                0.5,
            ],
            "std": [
                # EEF position deltas
                0.05, 0.05, 0.05,
                # EEF rotation deltas
                0.1, 0.1, 0.1,
                # Joint positions
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                # Gripper
                0.3,
            ],
        },
        "action": {
            "mean": [
                # Actions are deltas → typically centered at 0
                0.0, 0.0, 0.0,  # Position deltas
                0.0, 0.0, 0.0,  # Rotation deltas
                0.0,            # Gripper delta
            ],
            "std": [
                # Position deltas (smaller for precise tasks)
                0.02, 0.02, 0.02,
                # Rotation deltas
                0.05, 0.05, 0.05,
                # Gripper delta
                0.2,
            ],
        },
    },

    # =========================================================================
    # Training Configuration
    # =========================================================================
    "training": {
        "max_seq_len": 1024,               # Maximum sequence length
        "chunk_size": 100,                 # Episode chunk size for data loading
        "action_horizon": 16,              # Horizon for action prediction
        "execution_horizon": 8,            # How many actions to execute (rollout)

        # Data augmentation
        "use_augmentation": True,
        "color_jitter": {
            "brightness": 0.3,
            "contrast": 0.4,
            "saturation": 0.5,
            "hue": 0.08,
        },
        "random_rotation": None,           # Degrees (None = disabled)
        "crop_fraction": 0.95,
    },

    # =========================================================================
    # Task-Specific Settings
    # =========================================================================
    "tasks": {
        # RoboCasa foundational skills
        "atomic_tasks": [
            "pick_and_place",
            "open_drawer",
            "close_drawer",
            "turn_knob",
            "press_button",
            "insert",
            "wipe",
            "pour",
        ],

        # Composite tasks
        "composite_tasks": [
            "restock_supplies",
            "brew_coffee",
            "prepare_meal",
        ],
    },

    # =========================================================================
    # Metadata
    # =========================================================================
    "metadata": {
        "dataset_source": "RoboCasa",
        "robot_platform": "Franka Panda",
        "gripper": "Omron 2F85",
        "control_frequency": 20,  # Hz
        "action_space_type": "state_relative",  # KEY for GR00T N1.6
        "data_format": "LeRobot_v2",
        "conversion_script": "scripts/prepare_robocasa_for_groot.py",
    },
}


# =============================================================================
# Alternative Configurations
# =============================================================================

# Simplified config with fewer cameras (for faster iteration)
ROBOCASA_PANDA_OMRON_SINGLE_CAM = {
    **ROBOCASA_PANDA_OMRON,
    "embodiment_name": "ROBOCASA_PANDA_OMRON_SINGLE_CAM",
    "use_multiple_cameras": False,
    "cameras": {
        "wrist": ROBOCASA_PANDA_OMRON["cameras"]["wrist"],
    },
}

# Config for absolute actions (for comparison/ablation)
ROBOCASA_PANDA_OMRON_ABSOLUTE = {
    **ROBOCASA_PANDA_OMRON,
    "embodiment_name": "ROBOCASA_PANDA_OMRON_ABSOLUTE",
    "action_space": "absolute",
    "action_components": {
        "position_absolute": 3,    # [x, y, z]
        "rotation_absolute": 4,    # [qx, qy, qz, qw]
        "gripper_absolute": 1,     # gripper state
    },
    "action_dim": 8,  # 3 + 4 + 1
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_robocasa_config(variant: str = "default") -> dict:
    """
    Get RoboCasa configuration by variant name.

    Args:
        variant: One of ['default', 'single_cam', 'absolute']

    Returns:
        config: Configuration dictionary
    """
    configs = {
        "default": ROBOCASA_PANDA_OMRON,
        "single_cam": ROBOCASA_PANDA_OMRON_SINGLE_CAM,
        "absolute": ROBOCASA_PANDA_OMRON_ABSOLUTE,
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(configs.keys())}")

    return configs[variant]


def update_normalization_stats(config: dict, stats: dict) -> dict:
    """
    Update configuration with computed normalization statistics.

    Args:
        config: Configuration dictionary
        stats: Normalization stats from dataset processing

    Returns:
        updated_config: Configuration with updated normalization
    """
    config = config.copy()
    config["normalization"] = stats
    return config


def validate_config(config: dict) -> bool:
    """
    Validate configuration for consistency.

    Args:
        config: Configuration dictionary

    Returns:
        is_valid: True if configuration is valid

    Raises:
        ValueError: If configuration has issues
    """
    # Check state dimension
    expected_state_dim = sum(config["state_components"].values())
    if config["state_dim"] != expected_state_dim:
        raise ValueError(
            f"State dim mismatch: config says {config['state_dim']} "
            f"but components sum to {expected_state_dim}"
        )

    # Check action dimension
    expected_action_dim = sum(config["action_components"].values())
    if config["action_dim"] != expected_action_dim:
        raise ValueError(
            f"Action dim mismatch: config says {config['action_dim']} "
            f"but components sum to {expected_action_dim}"
        )

    # Check normalization shapes
    if len(config["normalization"]["state"]["mean"]) != config["state_dim"]:
        raise ValueError("State normalization mean has wrong dimension")

    if len(config["normalization"]["action"]["mean"]) != config["action_dim"]:
        raise ValueError("Action normalization mean has wrong dimension")

    print("✓ Configuration is valid")
    return True


# =============================================================================
# Module-level validation
# =============================================================================

# =============================================================================
# Proper ModalityConfig Structure for GR00T
# =============================================================================

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# This is the structure that GR00T expects for modality configs
ROBOCASA_PANDA_OMRON_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],  # Single frame observation
        modality_keys=[
            "left_view",     # observation.images.left_view
            "right_view",    # observation.images.right_view
            "wrist_view",    # observation.images.wrist_view
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],  # Current state only
        modality_keys=[
            "end_effector_position_relative",  # 3D position delta
            "end_effector_rotation_relative",  # Quaternion rotation delta (4D)
            "joint_position",                   # 7-DOF joint angles
            "gripper_qpos",                    # 2D gripper position
        ],
        # Use sin/cos encoding for joint angles (cyclical)
        sin_cos_embedding_keys=["joint_position"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),  # 16-step action horizon
        modality_keys=[
            "end_effector_position",    # 3D position delta
            "end_effector_rotation",    # 3D axis-angle rotation delta
            "gripper_close",            # Binary gripper command
        ],
        action_configs=[
            # end_effector_position (relative/delta)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # end_effector_rotation (relative/delta, axis-angle format)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROTVEC,  # axis-angle = rotation vector
            ),
            # gripper_close (binary: 0=open, 1=close)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

# Keep the metadata for reference (rename original)
ROBOCASA_PANDA_OMRON_METADATA = ROBOCASA_PANDA_OMRON

# =============================================================================
# Registration
# =============================================================================

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag

# Register the proper config structure (not the metadata)
register_modality_config(ROBOCASA_PANDA_OMRON_CONFIG, embodiment_tag=EmbodimentTag.ROBOCASA_PANDA_OMRON)


if __name__ == "__main__":
    # Validate metadata config
    print("Validating ROBOCASA_PANDA_OMRON_METADATA configuration...")
    validate_config(ROBOCASA_PANDA_OMRON_METADATA)

    print("\nMetadata configuration summary:")
    print(f"  State dim: {ROBOCASA_PANDA_OMRON_METADATA['state_dim']}")
    print(f"  Action dim: {ROBOCASA_PANDA_OMRON_METADATA['action_dim']}")
    print(f"  Action space: {ROBOCASA_PANDA_OMRON_METADATA['action_space']}")
    print(f"  Action horizon: {ROBOCASA_PANDA_OMRON_METADATA['action_horizon']}")
    print(f"  Cameras: {list(ROBOCASA_PANDA_OMRON_METADATA['cameras'].keys())}")

    print("\n" + "="*70)
    print("GR00T ModalityConfig structure:")
    print("="*70)
    print(f"  Video keys: {ROBOCASA_PANDA_OMRON_CONFIG['video'].modality_keys}")
    print(f"  State keys: {ROBOCASA_PANDA_OMRON_CONFIG['state'].modality_keys}")
    print(f"  Action keys: {ROBOCASA_PANDA_OMRON_CONFIG['action'].modality_keys}")
    print(f"  Action horizon: {len(ROBOCASA_PANDA_OMRON_CONFIG['action'].delta_indices)}")
    print(f"  Language keys: {ROBOCASA_PANDA_OMRON_CONFIG['language'].modality_keys}")
    print("\n✓ All checks passed!")
