"""Custom Trainer with simple profiling utilities.

This subclass of HuggingFace's ``Trainer`` measures:
1. Data loading latency (time between the end of the previous ``training_step`` and
   the start of the current ``training_step``).
2. Forward-pass latency (time spent inside the base ``training_step`` implementation,
   which essentially wraps the model's forward / loss computation).

The statistics are logged via ``self.log`` every ``profile_log_interval`` steps and
also sent to the standard ``logging`` logger.  This is *not* meant to be a fully
fledged profiler – it is a quick, lightweight way to confirm whether the training
pipeline is bottlenecked by data loading or by the model's computation.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any, Optional

import torch
from transformers.trainer import TRAINER_STATE_NAME, Trainer, TrainerState, get_last_checkpoint
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

# Import for SVMS auxiliary label generation
try:
    from gr00t.data.robocasa_data_collator_with_aux import generate_aux_labels_in_trainer
    HAS_AUX_LABELS = True
except ImportError:
    HAS_AUX_LABELS = False
    logging.warning("Could not import auxiliary label generator. SVMS training may not work properly.")


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


class _BatchIterator:
    """Lightweight iterator that yields pre-collated batches."""

    def __init__(self, buffer, bs, collator, total_steps):
        self._buffer = buffer
        self._bs = bs
        self._collate = collator
        self._total_steps = total_steps
        self._produced = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self._total_steps

    def __next__(self):
        if self._produced >= self._total_steps:
            raise StopIteration

        # Fast path – single lock acquisition inside ``sample_batch``.
        batch_samples = self._buffer.sample_batch(self._bs)  # type: ignore[attr-defined]
        self._produced += 1
        return self._collate(batch_samples)


class _PrefetchIterator:
    def __init__(self, buffer, bs, collate_fn, total_steps):
        self.buffer = buffer
        self.bs = bs
        self.collate = collate_fn
        self.total = total_steps
        self.produced = 0

        self._q = queue.Queue(maxsize=4)
        self._stop = False

        # Start background worker
        self._worker = threading.Thread(target=self._fill)
        self._worker.daemon = True
        self._worker.start()

    def _fill(self):
        while not self._stop:
            if self.produced + self._q.qsize() >= self.total:
                break
            # block if queue is full
            samples = self.buffer.sample_batch(self.bs)
            batch = self.collate(samples)
            self._q.put(batch)

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.produced >= self.total:
            self._stop = True
            # in case worker is blocked on put()
            raise StopIteration
        batch = self._q.get()  # this will block until the next batch is ready
        self.produced += 1
        return batch


def _batch_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, action_offset: Optional[int] = None
) -> torch.Tensor:  # noqa: D401
    """Compute token-level accuracy, ignoring ``-100`` label positions.

    Args:
        preds: Predicted token ids of shape ``(batch, seq_len)``.
        labels: Ground-truth label ids with the same shape as ``preds``.

    Returns:
        Scalar tensor with the fraction of correctly predicted labels in the
        current batch.
    """
    # casual prediction
    # Shift so that tokens < n predict n
    # https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L60
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = (preds == labels) & mask

    # Avoid division by zero for empty masks (should not happen in practice)
    denom = mask.sum().clamp(min=1)
    accuracy = correct.sum().float() / denom.float()
    return accuracy


# Global variables for batched evaluation metrics
_eval_accuracy_accumulated_correct = 0
_eval_accuracy_accumulated_total = 0


def compute_eval_accuracy(
    eval_pred: EvalPrediction, compute_result: bool, action_offset: Optional[int] = None
):
    logits = eval_pred.predictions[0]
    if action_offset is not None:
        logits = logits[..., action_offset:]
    preds = logits.argmax(axis=-1)
    labels = eval_pred.label_ids

    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = ((preds == labels) & mask).sum()
    total = mask.sum()

    global _eval_accuracy_accumulated_correct, _eval_accuracy_accumulated_total
    _eval_accuracy_accumulated_correct += correct
    _eval_accuracy_accumulated_total += total

    if compute_result:
        accuracy = _eval_accuracy_accumulated_correct / max(_eval_accuracy_accumulated_total, 1)
        _eval_accuracy_accumulated_correct = 0
        _eval_accuracy_accumulated_total = 0
        return {"eval_accuracy": accuracy}
    else:
        return {}


class Gr00tTrainer(Trainer):
    """Trainer that bypasses torch dataloader and makes data collator async."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:  # noqa: D401 – simple description above
        """Initialize the trainer.

        Args:
            *args: Positional arguments forwarded to ``Trainer``.
        """
        self.action_offset = kwargs.pop("action_offset", None)
        self.multiprocessing_context = kwargs.pop("multiprocessing_context", "fork")

        # SVMS-specific parameters
        self.use_svms = kwargs.pop("use_svms", False)
        self.lambda_aux = kwargs.pop("lambda_aux", 0.3)
        self.lambda_sheaf_max = kwargs.pop("lambda_sheaf_max", 0.1)
        self.lambda_sheaf_min = kwargs.pop("lambda_sheaf_min", 0.01)
        self.sheaf_schedule_mode = kwargs.pop("sheaf_schedule_mode", "adaptive")
        self.sheaf_delay_until_diffusion = kwargs.pop("sheaf_delay_until_diffusion", 0.4)
        self.aux_warmup_steps = kwargs.pop("aux_warmup_steps", 5000)

        super().__init__(
            *args,
            **kwargs,
            # compute_metrics=partial(compute_eval_accuracy, action_offset=self.action_offset),
        )

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Hide epoch from logged metrics as it's misleading for Iterable datasets.
        epoch = self.state.epoch
        self.state.epoch = None
        super().log(logs, start_time=start_time)
        self.state.epoch = epoch

    # ------------------------------------------------------------------
    # SVMS-specific helper methods
    # ------------------------------------------------------------------

    def _compute_lambda_sheaf(self, step: int) -> float:
        """
        Compute sheaf consistency weight using adaptive scheduling.

        Args:
            step: Current training step

        Returns:
            Lambda value for sheaf loss
        """
        if not self.use_svms:
            return 0.0

        if self.sheaf_schedule_mode == "off":
            return 0.0

        # Delay sheaf until main training is progressing
        total_steps = self.state.max_steps if self.state.max_steps > 0 else 10000
        delay_step = int(total_steps * self.sheaf_delay_until_diffusion)

        if step < delay_step:
            return 0.0

        # Ramp up sheaf weight
        if self.sheaf_schedule_mode == "adaptive":
            # Gradual increase from min to max
            progress = (step - delay_step) / (total_steps - delay_step)
            progress = min(progress, 1.0)
            lambda_sheaf = self.lambda_sheaf_min + progress * (
                self.lambda_sheaf_max - self.lambda_sheaf_min
            )
        elif self.sheaf_schedule_mode == "constant":
            lambda_sheaf = self.lambda_sheaf_max
        else:
            lambda_sheaf = 0.0

        return lambda_sheaf

    def _compute_lambda_aux(self, step: int) -> float:
        """
        Compute auxiliary loss weight with warmup.

        Args:
            step: Current training step

        Returns:
            Lambda value for auxiliary losses
        """
        if not self.use_svms:
            return 0.0

        if step >= self.aux_warmup_steps:
            return self.lambda_aux

        # Linear warmup
        warmup_frac = step / self.aux_warmup_steps
        return self.lambda_aux * warmup_frac

    def get_train_dataloader(self):  # noqa: D401
        """Return a iterable dataloader without skipping the data during resume, but reseed the dataset instead."""

        # Fall back to default behaviour if not using the custom buffer.
        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            new_seed = self.train_dataset.seed + curr_global_step
            self.train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )
        # Use persistent workers for sharded dataset if num_workers is greater than 0
        persistent_workers = self.args.dataloader_num_workers > 0

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
        }

        # multiprocessing_context can only be used with num_workers > 0
        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context

        return torch.utils.data.DataLoader(self.train_dataset, **dataloader_params)

    def train(
        self,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                logging.warning(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            logging.info(f"Resuming from checkpoint {resume_from_checkpoint}")
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )

        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    # ------------------------------------------------------------------
    # Loss / accuracy computation override
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        """Compute loss *and* log token-level accuracy every training step.

        We delegate the heavy-lifting (including label smoothing, custom loss
        functions, etc.) to the parent ``Trainer.compute_loss`` implementation
        by calling it with ``return_outputs=True``.  After obtaining the loss
        *and* model outputs, we calculate accuracy and push it to the logger.

        SVMS Extension:
        - Generates auxiliary labels if not present in batch
        - Adds SVMS losses (sheaf consistency + auxiliary supervision)
        - Logs stream-specific metrics
        """

        # --------------------------------------------------------------
        # SVMS: Generate auxiliary labels if needed
        # --------------------------------------------------------------
        if self.use_svms and HAS_AUX_LABELS and model.training:
            # Check if auxiliary labels are already in batch
            if "aux_labels_A" not in inputs:
                # Generate on-the-fly
                try:
                    aux_labels = generate_aux_labels_in_trainer(
                        inputs,
                        self.tokenizer if hasattr(self, "tokenizer") else None
                    )
                    inputs.update(aux_labels)
                except Exception as e:
                    logging.warning(f"Failed to generate auxiliary labels: {e}")

            # Pass training step to model for router temperature scheduling
            inputs["training_step"] = self.state.global_step

        # Use parent implementation to preserve built-in functionality.
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        # --------------------------------------------------------------
        # SVMS: Add sheaf and auxiliary losses
        # --------------------------------------------------------------
        if self.use_svms and model.training and hasattr(outputs, "svms_outputs"):
            svms_outputs = outputs.svms_outputs

            # Compute scheduling weights
            lambda_sheaf = self._compute_lambda_sheaf(self.state.global_step)
            lambda_aux = self._compute_lambda_aux(self.state.global_step)

            # Sheaf consistency loss
            if lambda_sheaf > 0 and "loss_sheaf" in svms_outputs:
                loss_sheaf = svms_outputs["loss_sheaf"]
                loss = loss + lambda_sheaf * loss_sheaf

            # Auxiliary supervision losses
            if lambda_aux > 0:
                aux_losses = []
                for stream in ["A", "B", "C"]:
                    loss_key = f"loss_aux_{stream}"
                    if loss_key in svms_outputs:
                        aux_losses.append(svms_outputs[loss_key])

                if len(aux_losses) > 0:
                    loss_aux_total = sum(aux_losses) / len(aux_losses)
                    loss = loss + lambda_aux * loss_aux_total

        # Record last loss for testing purposes.
        self.loss = loss

        # --------------------------------------------------------------
        # Accuracy calculation
        # --------------------------------------------------------------
        if (
            self.state.global_step % self.args.logging_steps == 0
            and model.training
            and "labels" in inputs
        ):
            if self.action_offset is not None:
                preds = outputs.logits.detach()[:, :, self.action_offset :].argmax(dim=-1).cpu()
            else:
                preds = outputs.logits.detach().argmax(dim=-1).cpu()
            with torch.no_grad():
                acc_local = _batch_accuracy(
                    preds, inputs["labels"].to(device=preds.device), self.action_offset
                )
            acc_tensor = torch.tensor(acc_local.item(), device=loss.device)
            acc_mean = self._nested_gather(acc_tensor).mean().item()

            logs = {"train_accuracy": acc_mean}

            # --------------------------------------------------------------
            # SVMS: Log stream-specific metrics
            # --------------------------------------------------------------
            if self.use_svms and hasattr(outputs, "svms_outputs"):
                svms_outputs = outputs.svms_outputs

                # Log loss components
                if "loss_sheaf" in svms_outputs:
                    logs["loss_sheaf"] = svms_outputs["loss_sheaf"].item()
                    logs["lambda_sheaf"] = self._compute_lambda_sheaf(self.state.global_step)

                # Log auxiliary accuracies
                for stream in ["A", "B", "C"]:
                    acc_key = f"aux_acc_{stream}"
                    if acc_key in svms_outputs:
                        logs[acc_key] = svms_outputs[acc_key].item()

                # Log router weights
                if "router_weights" in svms_outputs:
                    weights = svms_outputs["router_weights"]  # (B, T, 3)
                    mean_weights = weights.mean(dim=[0, 1])  # (3,)
                    logs["router_weight_A"] = mean_weights[0].item()
                    logs["router_weight_B"] = mean_weights[1].item()
                    logs["router_weight_C"] = mean_weights[2].item()

                logs["lambda_aux"] = self._compute_lambda_aux(self.state.global_step)

            if self.args.local_rank in (-1, 0):
                self.log(logs)

        return (loss, outputs) if return_outputs else loss
