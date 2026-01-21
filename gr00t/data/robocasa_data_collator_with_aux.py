"""
RoboCasa Data Collator with Auxiliary Label Generation

This data collator extends the standard GR00T data collator to automatically
generate auxiliary labels for SVMS stream specialization during training.

The collator:
1. Batches episodes from the dataset
2. Extracts instruction tokens from VLM inputs
3. Generates auxiliary labels (Stream A, B, C) on-the-fly
4. Adds labels to the batch for SVMS training

Usage:
    from gr00t.data.robocasa_data_collator_with_aux import RoboCasaDataCollatorWithAux

    collator = RoboCasaDataCollatorWithAux(
        tokenizer=vlm_tokenizer,
        generate_aux_labels=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator,
    )

Author: Adapted for GR00T N1.6 SVMS training
"""

import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from gr00t.data.robocasa_auxiliary_labels import create_auxiliary_labels_from_ids


@dataclass
class RoboCasaDataCollatorWithAux:
    """
    Data collator that generates auxiliary labels for SVMS training.

    This collator wraps the standard batching process and adds auxiliary label
    generation based on instruction tokens.

    Args:
        tokenizer: VLM tokenizer for decoding token IDs
        generate_aux_labels: Whether to generate auxiliary labels
        pad_token_id: Padding token ID (default: 0)
        max_seq_len: Maximum sequence length for padding
    """

    tokenizer: Any
    generate_aux_labels: bool = True
    pad_token_id: int = 0
    max_seq_len: Optional[int] = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of episodes and generate auxiliary labels.

        Args:
            batch: List of episode dictionaries from dataset

        Returns:
            Batched dictionary with auxiliary labels added
        """
        # Stack batch tensors
        batched = self._stack_batch(batch)

        # Generate auxiliary labels if enabled
        if self.generate_aux_labels and "input_ids" in batched:
            aux_labels = self._generate_auxiliary_labels(batched["input_ids"])
            batched.update(aux_labels)

        return batched

    def _stack_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Stack individual episodes into a batched dictionary.

        Args:
            batch: List of episode dictionaries

        Returns:
            Batched dictionary with stacked tensors
        """
        if len(batch) == 0:
            return {}

        # Get all keys from first item
        keys = batch[0].keys()

        batched = {}
        for key in keys:
            values = [item[key] for item in batch]

            # Handle different data types
            if isinstance(values[0], torch.Tensor):
                # Stack tensors
                batched[key] = self._stack_tensors(values, key)
            elif isinstance(values[0], (int, float)):
                # Convert scalars to tensor
                batched[key] = torch.tensor(values)
            elif isinstance(values[0], str):
                # Keep strings as list
                batched[key] = values
            elif isinstance(values[0], dict):
                # Recursively batch nested dicts
                batched[key] = self._stack_batch(values)
            else:
                # Keep as list for other types
                batched[key] = values

        return batched

    def _stack_tensors(
        self, tensors: List[torch.Tensor], key: str
    ) -> torch.Tensor:
        """
        Stack a list of tensors with optional padding.

        Args:
            tensors: List of tensors to stack
            key: Tensor key (used to determine padding behavior)

        Returns:
            Stacked tensor (B, ...)
        """
        # Check if all tensors have same shape
        shapes = [t.shape for t in tensors]
        if all(s == shapes[0] for s in shapes):
            # Same shape -> simple stack
            return torch.stack(tensors, dim=0)

        # Different shapes -> need padding
        # This typically happens with variable-length sequences
        return self._pad_and_stack(tensors, key)

    def _pad_and_stack(
        self, tensors: List[torch.Tensor], key: str
    ) -> torch.Tensor:
        """
        Pad tensors to same length and stack.

        Args:
            tensors: List of tensors with potentially different first dimensions
            key: Tensor key (determines padding value)

        Returns:
            Padded and stacked tensor (B, max_len, ...)
        """
        # Determine max length
        max_len = max(t.shape[0] for t in tensors)
        if self.max_seq_len is not None:
            max_len = min(max_len, self.max_seq_len)

        # Determine padding value
        if "input_ids" in key or "tokens" in key:
            pad_value = self.pad_token_id
        elif "attention_mask" in key or "mask" in key:
            pad_value = 0
        else:
            pad_value = 0

        # Pad each tensor
        padded = []
        for t in tensors:
            # Truncate if needed
            if t.shape[0] > max_len:
                t = t[:max_len]

            # Pad if needed
            if t.shape[0] < max_len:
                pad_size = [max_len - t.shape[0]] + list(t.shape[1:])
                padding = torch.full(
                    pad_size, pad_value, dtype=t.dtype, device=t.device
                )
                t = torch.cat([t, padding], dim=0)

            padded.append(t)

        return torch.stack(padded, dim=0)

    def _generate_auxiliary_labels(
        self, input_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate auxiliary labels for SVMS from input token IDs.

        Args:
            input_ids: Batched input token IDs (B, T)

        Returns:
            Dictionary with aux_labels_A, aux_labels_B, aux_labels_C
        """
        # Generate labels using the auxiliary label module
        labels_A, labels_B, labels_C = create_auxiliary_labels_from_ids(
            input_ids=input_ids,
            tokenizer=self.tokenizer,
        )

        return {
            "aux_labels_A": labels_A,  # Visual reasoning labels
            "aux_labels_B": labels_B,  # Temporal planning labels
            "aux_labels_C": labels_C,  # State tracking labels
        }


# =============================================================================
# Simplified Alternative: Generate Labels in Trainer
# =============================================================================
# If integrating a custom collator is complex, you can generate auxiliary
# labels directly in the trainer's training_step:


def generate_aux_labels_in_trainer(batch: Dict[str, torch.Tensor], tokenizer) -> Dict[str, torch.Tensor]:
    """
    Helper function to generate auxiliary labels in trainer.

    Add this to your trainer's training_step:

    ```python
    def training_step(self, batch, batch_idx):
        # Generate auxiliary labels
        aux_labels = generate_aux_labels_in_trainer(batch, self.tokenizer)
        batch.update(aux_labels)

        # Continue with normal forward pass
        outputs = self.model(batch)
        ...
    ```

    Args:
        batch: Batch dictionary from dataloader
        tokenizer: VLM tokenizer

    Returns:
        Dictionary with auxiliary labels
    """
    if "input_ids" not in batch:
        # No text input -> return dummy labels
        B, T = batch.get("attention_mask", torch.zeros(1, 1)).shape
        return {
            "aux_labels_A": torch.zeros(B, T),
            "aux_labels_B": torch.zeros(B, T),
            "aux_labels_C": torch.zeros(B, T),
        }

    labels_A, labels_B, labels_C = create_auxiliary_labels_from_ids(
        input_ids=batch["input_ids"],
        tokenizer=tokenizer,
    )

    return {
        "aux_labels_A": labels_A,
        "aux_labels_B": labels_B,
        "aux_labels_C": labels_C,
    }


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    print("RoboCasa Data Collator with Auxiliary Labels")
    print("=" * 80)
    print()

    print("Option 1: Use Custom Collator")
    print("-" * 80)
    print("""
from gr00t.data.robocasa_data_collator_with_aux import RoboCasaDataCollatorWithAux
from torch.utils.data import DataLoader

# Create collator
collator = RoboCasaDataCollatorWithAux(
    tokenizer=vlm_tokenizer,
    generate_aux_labels=True,
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=collator,
    num_workers=4,
)

# Training loop
for batch in dataloader:
    # batch now includes aux_labels_A, aux_labels_B, aux_labels_C
    outputs = model(batch)
    ...
""")

    print("\nOption 2: Generate Labels in Trainer (Simpler)")
    print("-" * 80)
    print("""
from gr00t.data.robocasa_data_collator_with_aux import generate_aux_labels_in_trainer

class GR00TTrainer(Trainer):
    def training_step(self, batch, batch_idx):
        # Generate auxiliary labels on-the-fly
        aux_labels = generate_aux_labels_in_trainer(batch, self.tokenizer)
        batch.update(aux_labels)

        # Normal forward pass
        outputs = self.model(batch)

        # Compute losses (including SVMS losses)
        loss = self.compute_loss(outputs, batch)
        return loss
""")

    print("\nRecommendation:")
    print("-" * 80)
    print("""
For quick integration, use Option 2 (generate in trainer).

Advantages:
- Minimal changes to existing code
- Easy to debug
- Flexible (can enable/disable per phase)

Disadvantages:
- Slightly slower (generates labels on GPU)
- Couples data processing with training logic

For production, use Option 1 (custom collator).

Advantages:
- Cleaner separation of concerns
- Slightly faster (generates on CPU during loading)
- Reusable across different trainers

Disadvantages:
- Requires modifying dataloader setup
- More complex to integrate
""")

    print("\n" + "=" * 80)
    print("For GR00T SVMS training, we recommend starting with Option 2,")
    print("then migrating to Option 1 once the training pipeline is stable.")
    print("=" * 80)
