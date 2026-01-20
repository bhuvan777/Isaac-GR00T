"""
RoboCasa Auxiliary Label Generator for Stream Specialization

Creates token-level binary labels to supervise stream specialization:
- Stream A (Visual): Objects, spatial relations, visual attributes
- Stream B (Temporal): Actions, sequences, causal reasoning
- Stream C (State): Robot state, object states, physical properties

These labels provide explicit supervision for each stream to learn its specialty,
accelerating training and improving interpretability.

Author: Adapted for RoboCasa kitchen manipulation tasks
"""

import re
from typing import Tuple, List
import torch


# =============================================================================
# Stream A: Visual Scene Reasoning Keywords
# =============================================================================

ROBOCASA_OBJECTS = {
    # Containers & storage
    'pot', 'pan', 'kettle', 'bowl', 'plate', 'cup', 'mug', 'glass', 'bottle',
    'cabinet', 'drawer', 'shelf', 'counter', 'table', 'tray',

    # Appliances
    'stove', 'oven', 'microwave', 'sink', 'faucet', 'dishwasher',
    'refrigerator', 'fridge', 'toaster', 'blender',

    # Manipulable objects
    'knob', 'handle', 'button', 'lever', 'door', 'lid',
    'spoon', 'fork', 'knife', 'spatula', 'ladle', 'whisk',

    # Food items
    'apple', 'banana', 'bread', 'cheese', 'egg', 'milk', 'water',
    'vegetable', 'fruit', 'ingredient', 'food',

    # Other
    'towel', 'cloth', 'sponge', 'cutting', 'board',
}

SPATIAL_RELATIONS = {
    # Position
    'on', 'in', 'inside', 'above', 'below', 'under', 'over',
    'next', 'near', 'beside', 'adjacent', 'around',
    'left', 'right', 'front', 'back', 'behind',
    'top', 'bottom', 'middle', 'center', 'edge', 'corner',

    # Direction
    'toward', 'away', 'into', 'out', 'through',
    'up', 'down', 'across', 'along',
}

VISUAL_ATTRIBUTES = {
    # Colors
    'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray',

    # Size
    'large', 'small', 'big', 'tiny', 'huge', 'medium',
    'tall', 'short', 'wide', 'narrow', 'thick', 'thin',

    # Shape
    'round', 'square', 'rectangular', 'circular', 'flat', 'curved',

    # Material (visual)
    'metal', 'wood', 'plastic', 'glass', 'ceramic',

    # State (visually apparent)
    'empty', 'full', 'filled', 'clean', 'dirty',
}


# =============================================================================
# Stream B: Temporal Planning & Sequencing Keywords
# =============================================================================

ACTION_VERBS = {
    # Manipulation primitives
    'grasp', 'grab', 'hold', 'pick', 'lift', 'carry',
    'place', 'put', 'set', 'drop', 'release', 'let',
    'move', 'transfer', 'relocate', 'transport',
    'reach', 'approach', 'retract', 'withdraw',

    # Interaction
    'open', 'close', 'shut',
    'push', 'pull', 'slide',
    'turn', 'twist', 'rotate', 'spin',
    'press', 'click', 'tap', 'touch',
    'insert', 'remove', 'extract', 'take',

    # Kitchen-specific actions
    'pour', 'scoop', 'stir', 'mix', 'blend',
    'cut', 'slice', 'chop', 'spread',
    'cook', 'heat', 'boil', 'bake',
    'wash', 'rinse', 'clean', 'wipe',
}

TEMPORAL_MARKERS = {
    # Sequence
    'first', 'second', 'third', 'fourth', 'fifth',
    'next', 'then', 'after', 'before', 'finally', 'last',
    'initially', 'subsequently', 'eventually',

    # Duration
    'while', 'during', 'until', 'when', 'once',
    'start', 'begin', 'finish', 'end', 'complete',

    # Repetition
    'again', 'repeat', 'continue', 'keep', 'maintain',
}

CAUSAL_KEYWORDS = {
    # Causation
    'because', 'since', 'as', 'due', 'owing',
    'cause', 'result', 'effect', 'lead', 'produce',

    # Purpose
    'to', 'for', 'so', 'in order', 'order to',
    'goal', 'aim', 'purpose', 'objective',

    # Reasoning
    'therefore', 'thus', 'hence', 'consequently',
    'if', 'then', 'unless', 'otherwise', 'else',
}

PLANNING_KEYWORDS = {
    # Decomposition
    'step', 'stage', 'phase', 'part', 'component',
    'subtask', 'task', 'procedure', 'process', 'method',

    # Planning
    'plan', 'strategy', 'approach', 'sequence', 'order',
    'prepare', 'setup', 'arrange', 'organize',
}


# =============================================================================
# Stream C: State Tracking & Physical Reasoning Keywords
# =============================================================================

OBJECT_STATES = {
    # Binary states
    'open', 'opened', 'closed', 'shut',
    'on', 'off', 'activated', 'deactivated',
    'locked', 'unlocked',
    'empty', 'full', 'filled',

    # Thermal states
    'hot', 'cold', 'warm', 'cool', 'frozen', 'melted',
    'boiling', 'simmering', 'heating', 'cooling',

    # Processing states
    'cooked', 'raw', 'uncooked', 'done', 'ready',
    'chopped', 'sliced', 'mixed', 'blended',

    # Liquid states
    'poured', 'spilled', 'dripping', 'flowing', 'leaking',
    'wet', 'dry', 'damp', 'soaked',

    # Cleanliness
    'clean', 'dirty', 'washed', 'unwashed', 'rinsed',
}

PHYSICAL_PROPERTIES = {
    # Mass & weight
    'heavy', 'light', 'weigh', 'weight', 'mass',

    # Mechanical
    'rigid', 'flexible', 'stiff', 'soft', 'hard',
    'fragile', 'sturdy', 'stable', 'unstable',
    'balanced', 'unbalanced', 'tilted',

    # Friction & contact
    'slippery', 'sticky', 'rough', 'smooth',
    'grip', 'grasp', 'contact', 'touch', 'force',

    # Dynamics
    'moving', 'static', 'stationary', 'still',
    'falling', 'dropping', 'rising', 'sinking',
}

ROBOT_STATE_KEYWORDS = {
    # Robot components
    'gripper', 'finger', 'hand', 'arm', 'wrist', 'elbow', 'shoulder',
    'joint', 'link', 'base', 'end-effector', 'eef',

    # Robot state descriptors
    'position', 'pose', 'orientation', 'rotation',
    'angle', 'velocity', 'speed', 'acceleration',
    'force', 'torque', 'pressure',

    # Control terms
    'control', 'command', 'actuate', 'move', 'stop',
    'limit', 'constraint', 'boundary', 'range',
}


# =============================================================================
# Auxiliary Label Generation Functions
# =============================================================================

def is_number_token(token_text: str) -> bool:
    """Check if token contains numerical content"""
    # Check for digits
    if re.search(r'\d', token_text):
        return True

    # Check for numerical words
    numerical_words = {'zero', 'one', 'two', 'three', 'four', 'five',
                      'six', 'seven', 'eight', 'nine', 'ten'}
    if token_text.lower().strip() in numerical_words:
        return True

    return False


def is_visual_token(token_text: str) -> bool:
    """
    Check if token is related to visual scene reasoning.

    Visual tokens include:
    - Object names
    - Spatial relationships
    - Visual attributes (color, size, shape)
    - Demonstrative pronouns referring to visual entities
    """
    token_lower = token_text.lower().strip()

    # Remove punctuation for matching
    clean_token = re.sub(r'[^\w]', '', token_lower)

    if not clean_token:
        return False

    # Check object names
    if clean_token in ROBOCASA_OBJECTS:
        return True

    # Check spatial relations
    if clean_token in SPATIAL_RELATIONS:
        return True

    # Check visual attributes
    if clean_token in VISUAL_ATTRIBUTES:
        return True

    # Check for demonstratives (visual references)
    demonstratives = {'this', 'that', 'these', 'those', 'it', 'here', 'there'}
    if clean_token in demonstratives:
        return True

    # Partial matches for compound words
    for obj in ROBOCASA_OBJECTS:
        if obj in clean_token or clean_token in obj:
            return True

    return False


def is_temporal_token(token_text: str) -> bool:
    """
    Check if token is related to temporal planning & sequencing.

    Temporal tokens include:
    - Action verbs
    - Sequence markers
    - Causal connectives
    - Planning keywords
    """
    token_lower = token_text.lower().strip()
    clean_token = re.sub(r'[^\w]', '', token_lower)

    if not clean_token:
        return False

    # Check action verbs (including conjugations)
    for verb in ACTION_VERBS:
        if verb in clean_token or clean_token in verb:
            return True

    # Check temporal markers
    if clean_token in TEMPORAL_MARKERS:
        return True

    # Check causal keywords
    for keyword in CAUSAL_KEYWORDS:
        if clean_token == keyword or (len(clean_token) > 3 and keyword in clean_token):
            return True

    # Check planning keywords
    if clean_token in PLANNING_KEYWORDS:
        return True

    return False


def is_state_token(token_text: str) -> bool:
    """
    Check if token is related to state tracking & physical reasoning.

    State tokens include:
    - Object state descriptors
    - Physical properties
    - Robot state keywords
    - Numerical values (positions, velocities, etc.)
    """
    token_lower = token_text.lower().strip()
    clean_token = re.sub(r'[^\w]', '', token_lower)

    if not clean_token:
        return False

    # Numerical values often represent state
    if is_number_token(token_text):
        return True

    # Check object states
    if clean_token in OBJECT_STATES:
        return True

    # Check physical properties
    if clean_token in PHYSICAL_PROPERTIES:
        return True

    # Check robot state keywords
    if clean_token in ROBOT_STATE_KEYWORDS:
        return True

    # Units and measurements
    units = {'degree', 'degrees', 'meter', 'meters', 'cm', 'mm',
            'second', 'seconds', 'newton', 'newtons', 'kg', 'gram'}
    if clean_token in units:
        return True

    return False


def create_auxiliary_labels(
    tokens: List[str],
    tokenizer = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create auxiliary labels for token-level stream specialization.

    Args:
        tokens: List of text tokens (decoded from token IDs)
        tokenizer: Tokenizer (optional, for additional processing)

    Returns:
        labels_A: Binary labels for Stream A (visual reasoning) - shape (T,)
        labels_B: Binary labels for Stream B (temporal planning) - shape (T,)
        labels_C: Binary labels for Stream C (state tracking) - shape (T,)
    """
    seq_len = len(tokens)
    labels_A = torch.zeros(seq_len, dtype=torch.float32)
    labels_B = torch.zeros(seq_len, dtype=torch.float32)
    labels_C = torch.zeros(seq_len, dtype=torch.float32)

    for i, token_text in enumerate(tokens):
        # Stream A: Visual scene reasoning
        if is_visual_token(token_text):
            labels_A[i] = 1.0

        # Stream B: Temporal planning
        if is_temporal_token(token_text):
            labels_B[i] = 1.0

        # Stream C: State tracking
        if is_state_token(token_text):
            labels_C[i] = 1.0

    return labels_A, labels_B, labels_C


def create_auxiliary_labels_from_ids(
    input_ids: torch.Tensor,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create auxiliary labels from token IDs (using tokenizer for decoding).

    Args:
        input_ids: Token IDs tensor (T,) or (B, T)
        tokenizer: HuggingFace tokenizer for decoding

    Returns:
        labels_A, labels_B, labels_C: Binary labels (same shape as input_ids)
    """
    # Handle batched or single sequence
    is_batched = input_ids.ndim == 2

    if is_batched:
        B, T = input_ids.shape
        all_labels_A = []
        all_labels_B = []
        all_labels_C = []

        for b in range(B):
            # Decode tokens for this sequence
            tokens = [tokenizer.decode([tid.item()], skip_special_tokens=False)
                     for tid in input_ids[b]]

            labels_A, labels_B, labels_C = create_auxiliary_labels(tokens, tokenizer)
            all_labels_A.append(labels_A)
            all_labels_B.append(labels_B)
            all_labels_C.append(labels_C)

        return (
            torch.stack(all_labels_A),  # (B, T)
            torch.stack(all_labels_B),  # (B, T)
            torch.stack(all_labels_C),  # (B, T)
        )
    else:
        # Single sequence
        tokens = [tokenizer.decode([tid.item()], skip_special_tokens=False)
                 for tid in input_ids]
        return create_auxiliary_labels(tokens, tokenizer)


def analyze_label_coverage(
    labels_A: torch.Tensor,
    labels_B: torch.Tensor,
    labels_C: torch.Tensor,
) -> dict:
    """
    Analyze coverage statistics for auxiliary labels.

    Useful for debugging and understanding label distribution.

    Args:
        labels_A, labels_B, labels_C: Auxiliary label tensors

    Returns:
        Dictionary with coverage statistics
    """
    stats = {
        "visual_coverage": labels_A.mean().item(),
        "temporal_coverage": labels_B.mean().item(),
        "state_coverage": labels_C.mean().item(),
        "total_tokens": labels_A.numel(),
        "visual_tokens": labels_A.sum().item(),
        "temporal_tokens": labels_B.sum().item(),
        "state_tokens": labels_C.sum().item(),
    }

    # Overlap analysis
    overlap_AB = (labels_A * labels_B).sum().item()
    overlap_BC = (labels_B * labels_C).sum().item()
    overlap_AC = (labels_A * labels_C).sum().item()
    overlap_ABC = (labels_A * labels_B * labels_C).sum().item()

    stats.update({
        "overlap_AB": overlap_AB,
        "overlap_BC": overlap_BC,
        "overlap_AC": overlap_AC,
        "overlap_ABC": overlap_ABC,
    })

    return stats


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: RoboCasa task instruction
    instruction = "First grasp the red mug on the counter, then place it inside the microwave and close the door."

    # Tokenize (simplified - actual usage would use HF tokenizer)
    tokens = instruction.lower().split()

    labels_A, labels_B, labels_C = create_auxiliary_labels(tokens)

    print("Instruction:", instruction)
    print("\nToken-level labels:")
    print(f"{'Token':<15} {'Visual':<8} {'Temporal':<10} {'State':<8}")
    print("-" * 45)
    for i, token in enumerate(tokens):
        print(f"{token:<15} {int(labels_A[i]):<8} {int(labels_B[i]):<10} {int(labels_C[i]):<8}")

    print("\nCoverage statistics:")
    stats = analyze_label_coverage(labels_A, labels_B, labels_C)
    for key, value in stats.items():
        print(f"  {key}: {value}")
