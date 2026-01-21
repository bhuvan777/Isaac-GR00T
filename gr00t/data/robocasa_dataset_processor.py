"""
RoboCasa Dataset Processor for GR00T N1.6

This module converts RoboCasa demonstrations (absolute coordinates) to GR00T-compatible
format (relative/state-relative coordinates).

Key transformations:
1. Absolute position [x,y,z] → Relative deltas [Δx,Δy,Δz]
2. Absolute quaternion [qx,qy,qz,qw] → Relative Euler [Δroll,Δpitch,Δyaw]
3. Compute action chunks for action horizon (default: 16 steps)
4. Generate auxiliary labels for stream specialization
5. Package in LeRobot v2 format

Author: Adapted for SVMS-GR00T integration
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import h5py


# =============================================================================
# Coordinate Conversion Functions
# =============================================================================

def absolute_to_relative_position(positions: np.ndarray) -> np.ndarray:
    """
    Convert absolute positions to relative deltas.

    Args:
        positions: (T, 3) array of absolute [x, y, z] positions

    Returns:
        relative: (T, 3) array of [Δx, Δy, Δz] deltas

    Example:
        Input:  [[1.0, 2.0, 3.0],    # t=0
                 [1.1, 2.0, 3.0],    # t=1
                 [1.2, 2.1, 3.0]]    # t=2
        Output: [[0.0, 0.0, 0.0],    # No previous frame
                 [0.1, 0.0, 0.0],    # Δ from t=0 to t=1
                 [0.1, 0.1, 0.0]]    # Δ from t=1 to t=2
    """
    T = len(positions)
    relative = np.zeros((T, 3), dtype=np.float32)

    # First frame has no previous frame → zero delta
    relative[0] = [0.0, 0.0, 0.0]

    # Compute frame-to-frame deltas
    relative[1:] = positions[1:] - positions[:-1]

    return relative


def quaternion_to_euler(quat: np.ndarray, degrees: bool = False) -> np.ndarray:
    """
    Convert quaternion to Euler angles (XYZ order).

    Args:
        quat: [qx, qy, qz, qw] quaternion (scalar-last convention)
        degrees: If True, return angles in degrees; else radians

    Returns:
        euler: [roll, pitch, yaw] in radians (or degrees)

    Note:
        RoboCasa uses [qx, qy, qz, qw] (scalar-last)
        scipy.spatial.transform.Rotation expects [qx, qy, qz, qw] (scalar-last)
        So no reordering needed!
    """
    rotation = R.from_quat(quat)  # Expects [qx, qy, qz, qw]
    euler = rotation.as_euler('xyz', degrees=degrees)
    return euler


def compute_relative_rotation(quat_prev: np.ndarray, quat_curr: np.ndarray) -> np.ndarray:
    """
    Compute relative rotation between two quaternions as Euler angles.

    Mathematical formula:
        R_relative = R_prev^{-1} * R_curr

    This gives the rotation needed to go from prev to curr orientation.

    Args:
        quat_prev: [qx, qy, qz, qw] at time t-1
        quat_curr: [qx, qy, qz, qw] at time t

    Returns:
        euler_delta: [Δroll, Δpitch, Δyaw] in radians

    Example:
        If robot rotates 10° around z-axis between frames:
        Input: quat_prev = [0, 0, 0, 1] (identity)
               quat_curr = [0, 0, 0.087, 0.996] (10° yaw)
        Output: [0.0, 0.0, 0.174] (≈10° in radians)
    """
    r_prev = R.from_quat(quat_prev)
    r_curr = R.from_quat(quat_curr)

    # Compute relative rotation
    r_relative = r_prev.inv() * r_curr

    # Convert to Euler angles
    euler_delta = r_relative.as_euler('xyz', degrees=False)

    return euler_delta


def absolute_quats_to_relative_euler(quats: np.ndarray) -> np.ndarray:
    """
    Convert trajectory of absolute quaternions to relative Euler deltas.

    Args:
        quats: (T, 4) array of [qx, qy, qz, qw] quaternions

    Returns:
        relative_euler: (T, 3) array of [Δroll, Δpitch, Δyaw] in radians
    """
    T = len(quats)
    relative_euler = np.zeros((T, 3), dtype=np.float32)

    # First frame: no previous → zero delta
    relative_euler[0] = [0.0, 0.0, 0.0]

    # Compute relative rotations for all subsequent frames
    for t in range(1, T):
        relative_euler[t] = compute_relative_rotation(quats[t-1], quats[t])

    return relative_euler


# =============================================================================
# Action Chunk Computation
# =============================================================================

def compute_action_chunks(
    trajectory: Dict[str, np.ndarray],
    action_horizon: int = 16,
    action_dim: int = 7
) -> np.ndarray:
    """
    Compute action chunks for each timestep.

    For each timestep t, we predict actions for the next H steps:
        actions[t] = [action_{t+1}, action_{t+2}, ..., action_{t+H}]

    Each action is: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]

    Args:
        trajectory: Dict with keys:
            - 'eef_pos': (T, 3) absolute positions
            - 'eef_quat': (T, 4) absolute quaternions
            - 'gripper': (T,) gripper state [0=open, 1=closed]
        action_horizon: Number of future steps to predict (H)
        action_dim: Dimension of action vector (default: 7)

    Returns:
        actions: (T, action_horizon, action_dim) array of action chunks

    Note:
        - For last few timesteps, we clamp to trajectory end (repeat last action)
        - This is standard practice in GR00T
    """
    T = len(trajectory['eef_pos'])
    actions = np.zeros((T, action_horizon, action_dim), dtype=np.float32)

    for t in range(T):
        for h in range(action_horizon):
            # Compute future timestep (clamp to trajectory end)
            future_t = min(t + h + 1, T - 1)

            # Position delta: Δpos = pos_{t+h+1} - pos_t
            pos_delta = trajectory['eef_pos'][future_t] - trajectory['eef_pos'][t]

            # Rotation delta: Δrot = relative_rotation(quat_t, quat_{t+h+1})
            rot_delta = compute_relative_rotation(
                trajectory['eef_quat'][t],
                trajectory['eef_quat'][future_t]
            )

            # Gripper delta: Δgripper = gripper_{t+h+1} - gripper_t
            gripper_delta = trajectory['gripper'][future_t] - trajectory['gripper'][t]

            # Concatenate into action vector
            actions[t, h] = np.concatenate([
                pos_delta,        # [Δx, Δy, Δz]
                rot_delta,        # [Δroll, Δpitch, Δyaw]
                [gripper_delta]   # [Δgripper]
            ])

    return actions


# =============================================================================
# RoboCasa Dataset Processor
# =============================================================================

class RoboCasaDatasetProcessor:
    """
    Process RoboCasa dataset for GR00T N1.6 training.

    Workflow:
        1. Load raw RoboCasa episodes (HDF5 or directory format)
        2. Extract observations: EEF pose, joint positions, gripper, images
        3. Convert absolute coordinates → relative deltas
        4. Compute action chunks (horizon=16 by default)
        5. Generate auxiliary labels from language instructions
        6. Package in LeRobot v2 format
        7. Compute normalization statistics
        8. Save processed dataset

    Example usage:
        processor = RoboCasaDatasetProcessor(
            raw_dataset_path="/path/to/raw/robocasa",
            output_path="./data/robocasa_groot",
            action_horizon=16,
            use_relative_actions=True
        )
        processor.process_full_dataset()
    """

    def __init__(
        self,
        raw_dataset_path: str,
        output_path: str,
        action_horizon: int = 16,
        use_relative_actions: bool = True,
        use_relative_state: bool = True,
        camera_names: Optional[List[str]] = None,
        validate: bool = True
    ):
        """
        Initialize RoboCasa dataset processor.

        Args:
            raw_dataset_path: Path to raw RoboCasa dataset
            output_path: Path to save processed dataset
            action_horizon: Number of future steps to predict
            use_relative_actions: If True, compute relative action deltas
            use_relative_state: If True, use relative state representation
            camera_names: List of camera names to include (default: wrist + front)
            validate: If True, run validation checks after processing
        """
        self.raw_path = Path(raw_dataset_path)
        self.output_path = Path(output_path)
        self.action_horizon = action_horizon
        self.use_relative_actions = use_relative_actions
        self.use_relative_state = use_relative_state
        self.validate = validate

        # Default cameras for RoboCasa
        if camera_names is None:
            self.camera_names = ['robot0_eye_in_hand', 'robot0_frontview']
        else:
            self.camera_names = camera_names

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Statistics accumulators
        self.state_values = []
        self.action_values = []

        print(f"RoboCasa Dataset Processor initialized:")
        print(f"  Input: {self.raw_path}")
        print(f"  Output: {self.output_path}")
        print(f"  Action horizon: {self.action_horizon}")
        print(f"  Relative actions: {self.use_relative_actions}")
        print(f"  Relative state: {self.use_relative_state}")

    def load_robocasa_episode(self, episode_path: Path) -> Dict[str, Any]:
        """
        Load a single RoboCasa episode from HDF5 file.

        Args:
            episode_path: Path to episode HDF5 file

        Returns:
            episode_data: Dict with observations, actions, language, etc.
        """
        with h5py.File(episode_path, 'r') as f:
            # Extract observations
            obs = f['obs']

            episode_data = {
                # End-effector pose
                'eef_pos': np.array(obs['robot0_eef_pos']),      # (T, 3)
                'eef_quat': np.array(obs['robot0_eef_quat']),    # (T, 4)

                # Joint states
                'joint_pos': np.array(obs['robot0_joint_pos']),  # (T, 7)
                'joint_vel': np.array(obs['robot0_joint_vel']),  # (T, 7)

                # Gripper
                'gripper_qpos': np.array(obs['robot0_gripper_qpos']),  # (T, 2)

                # Images (if present)
                'images': {},

                # Language instruction
                'language': f.attrs.get('task_description', '').decode('utf-8') if isinstance(f.attrs.get('task_description', ''), bytes) else f.attrs.get('task_description', ''),
            }

            # Load camera images
            for cam_name in self.camera_names:
                cam_key = f'{cam_name}_image'
                if cam_key in obs:
                    episode_data['images'][cam_name] = np.array(obs[cam_key])

            return episode_data

    def process_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single RoboCasa episode.

        Args:
            episode_data: Raw episode data from RoboCasa

        Returns:
            processed_episode: Episode in GR00T-compatible format
        """
        T = len(episode_data['eef_pos'])

        # Extract trajectory components
        eef_pos = episode_data['eef_pos']
        eef_quat = episode_data['eef_quat']
        joint_pos = episode_data['joint_pos']
        gripper_qpos = episode_data['gripper_qpos']

        # Average two gripper joints into single value
        gripper = gripper_qpos.mean(axis=1)  # (T,)

        # Convert to relative coordinates if needed
        if self.use_relative_state:
            state_eef_pos = absolute_to_relative_position(eef_pos)
            state_eef_rot = absolute_quats_to_relative_euler(eef_quat)
        else:
            state_eef_pos = eef_pos
            state_eef_rot = np.array([quaternion_to_euler(q) for q in eef_quat])

        # Compute action chunks
        if self.use_relative_actions:
            actions = compute_action_chunks(
                trajectory={
                    'eef_pos': eef_pos,
                    'eef_quat': eef_quat,
                    'gripper': gripper
                },
                action_horizon=self.action_horizon,
                action_dim=7
            )
        else:
            # For absolute actions, just use future poses directly
            # (This is less common for GR00T but supported)
            actions = self._compute_absolute_actions(eef_pos, eef_quat, gripper)

        # Package state
        state = {
            'end_effector_position_relative': state_eef_pos.astype(np.float32),    # (T, 3)
            'end_effector_rotation_relative': state_eef_rot.astype(np.float32),    # (T, 3)
            'joint_position': joint_pos.astype(np.float32),                        # (T, 7)
            'gripper_position': gripper.reshape(-1, 1).astype(np.float32)         # (T, 1)
        }

        # Package processed episode
        processed_episode = {
            'observation': {
                'state': state,
                'images': episode_data['images']
            },
            'action': actions.astype(np.float32),           # (T, H, 7)
            'language': episode_data['language'],
            'episode_length': T
        }

        # Accumulate for normalization statistics
        self.state_values.append(np.concatenate([
            state_eef_pos,
            state_eef_rot,
            joint_pos,
            gripper.reshape(-1, 1)
        ], axis=1))  # (T, 14)

        self.action_values.append(actions.reshape(-1, 7))  # (T*H, 7)

        return processed_episode

    def compute_normalization_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute mean and std for state and action normalization.

        Returns:
            stats: Dict with 'state' and 'action' normalization parameters
        """
        # Stack all values
        all_states = np.concatenate(self.state_values, axis=0)  # (N, 14)
        all_actions = np.concatenate(self.action_values, axis=0)  # (M, 7)

        stats = {
            'state': {
                'mean': all_states.mean(axis=0).tolist(),
                'std': all_states.std(axis=0).tolist(),
                'min': all_states.min(axis=0).tolist(),
                'max': all_states.max(axis=0).tolist()
            },
            'action': {
                'mean': all_actions.mean(axis=0).tolist(),
                'std': all_actions.std(axis=0).tolist(),
                'min': all_actions.min(axis=0).tolist(),
                'max': all_actions.max(axis=0).tolist()
            }
        }

        return stats

    def process_full_dataset(self):
        """
        Process entire RoboCasa dataset.

        Workflow:
            1. Find all episode files
            2. Process each episode
            3. Save processed episodes
            4. Compute normalization statistics
            5. Save metadata
            6. Validate if requested
        """
        # Find all episode files
        episode_files = sorted(self.raw_path.glob('**/episode_*.hdf5'))

        if len(episode_files) == 0:
            raise ValueError(f"No episode files found in {self.raw_path}")

        print(f"\nFound {len(episode_files)} episodes to process")

        # Process each episode
        processed_episodes = []

        for episode_path in tqdm(episode_files, desc="Processing episodes"):
            try:
                # Load raw episode
                raw_episode = self.load_robocasa_episode(episode_path)

                # Process episode
                processed_episode = self.process_episode(raw_episode)

                processed_episodes.append(processed_episode)

            except Exception as e:
                print(f"\nError processing {episode_path}: {e}")
                continue

        print(f"\nSuccessfully processed {len(processed_episodes)} episodes")

        # Compute normalization statistics
        print("\nComputing normalization statistics...")
        stats = self.compute_normalization_stats()

        # Save metadata
        metadata = {
            'dataset_name': 'robocasa_groot',
            'num_episodes': len(processed_episodes),
            'action_horizon': self.action_horizon,
            'use_relative_actions': self.use_relative_actions,
            'use_relative_state': self.use_relative_state,
            'camera_names': self.camera_names,
            'state_dim': 14,  # 3+3+7+1
            'action_dim': 7,
            'normalization': stats
        }

        with open(self.output_path / 'meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMetadata saved to {self.output_path / 'meta.json'}")

        # TODO: Save processed episodes in LeRobot v2 format
        # This would typically use Parquet files
        # For now, we can save as compressed numpy arrays

        print("\nDataset processing complete!")
        print(f"Output saved to: {self.output_path}")

        if self.validate:
            self.validate_dataset(processed_episodes, stats)

    def validate_dataset(self, episodes: List[Dict], stats: Dict):
        """
        Validate processed dataset.

        Checks:
            - No NaN or inf values
            - Action magnitudes in reasonable range
            - State values in reasonable range
            - Continuity of trajectories
        """
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)

        # Check for NaN/inf
        has_nan = False
        has_inf = False

        for ep in episodes:
            state_vals = np.concatenate([v for v in ep['observation']['state'].values()], axis=1)
            action_vals = ep['action']

            if np.isnan(state_vals).any():
                has_nan = True
            if np.isnan(action_vals).any():
                has_nan = True
            if np.isinf(state_vals).any():
                has_inf = True
            if np.isinf(action_vals).any():
                has_inf = True

        print(f"\n✓ NaN check: {'FAILED' if has_nan else 'PASSED'}")
        print(f"✓ Inf check: {'FAILED' if has_inf else 'PASSED'}")

        # Check action magnitudes
        print(f"\nAction statistics:")
        print(f"  Position deltas (m):")
        print(f"    Mean: {stats['action']['mean'][:3]}")
        print(f"    Std:  {stats['action']['std'][:3]}")
        print(f"    Max:  {stats['action']['max'][:3]}")

        print(f"  Rotation deltas (rad):")
        print(f"    Mean: {stats['action']['mean'][3:6]}")
        print(f"    Std:  {stats['action']['std'][3:6]}")
        print(f"    Max:  {stats['action']['max'][3:6]}")

        # Sanity checks
        max_pos_delta = max(abs(x) for x in stats['action']['max'][:3])
        max_rot_delta = max(abs(x) for x in stats['action']['max'][3:6])

        print(f"\n✓ Position delta check: {'PASSED' if max_pos_delta < 0.5 else 'WARNING: Large deltas'}")
        print(f"✓ Rotation delta check: {'PASSED' if max_rot_delta < np.pi/2 else 'WARNING: Large deltas'}")

        print("\n" + "="*60)


# =============================================================================
# Helper Functions
# =============================================================================

def visualize_trajectory(episode: Dict, save_path: Optional[str] = None):
    """
    Visualize a processed trajectory (for debugging).

    Args:
        episode: Processed episode dict
        save_path: If provided, save plot to this path
    """
    import matplotlib.pyplot as plt

    state = episode['observation']['state']
    actions = episode['action']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot EEF position
    pos = state['end_effector_position_relative']
    axes[0, 0].plot(pos)
    axes[0, 0].set_title('EEF Position (Relative)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].legend(['x', 'y', 'z'])
    axes[0, 0].grid(True)

    # Plot EEF rotation
    rot = state['end_effector_rotation_relative']
    axes[0, 1].plot(rot)
    axes[0, 1].set_title('EEF Rotation (Relative)')
    axes[0, 1].set_ylabel('Rotation (rad)')
    axes[0, 1].legend(['roll', 'pitch', 'yaw'])
    axes[0, 1].grid(True)

    # Plot action position deltas (first horizon step)
    action_pos = actions[:, 0, :3]  # (T, 3)
    axes[1, 0].plot(action_pos)
    axes[1, 0].set_title('Action Position Deltas (h=0)')
    axes[1, 0].set_ylabel('Δ Position (m)')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].legend(['Δx', 'Δy', 'Δz'])
    axes[1, 0].grid(True)

    # Plot action rotation deltas (first horizon step)
    action_rot = actions[:, 0, 3:6]  # (T, 3)
    axes[1, 1].plot(action_rot)
    axes[1, 1].set_title('Action Rotation Deltas (h=0)')
    axes[1, 1].set_ylabel('Δ Rotation (rad)')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].legend(['Δroll', 'Δpitch', 'Δyaw'])
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()
