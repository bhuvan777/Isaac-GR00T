"""
Sheaf-based Multi-Stream (SVMS) components for GR00T N1.6

This module implements specialized processing streams with sheaf consistency
for robotic manipulation tasks. Based on sheaf theory from algebraic topology,
adapted for neural representations in vision-language-action models.

Components:
- StreamHead: Specialized processing heads for different reasoning modalities
- LowRankAdapter: Restriction maps for sheaf overlaps
- SheafConsistency: Consistency loss and iterative correction
- StreamRouter: Adaptive token-level routing
- SVMSWrapper: Main wrapper integrating all components

Author: Adapted from GSM8K sheaf implementation
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamHead(nn.Module):
    """
    Stream processing head with residual MLP architecture.

    Each stream specializes in a different aspect of the task:
    - Stream A: Visual scene reasoning (objects, spatial relations, affordances)
    - Stream B: Temporal planning (action sequences, causal reasoning)
    - Stream C: State tracking (proprioception, object states, physics)

    Args:
        d_in: Input dimension (from VLM backbone)
        d_stream: Output stream dimension
        dropout: Dropout probability
    """

    def __init__(self, d_in: int, d_stream: int, dropout: float = 0.1):
        super().__init__()
        self.d_in = d_in
        self.d_stream = d_stream

        # Projection layer
        self.proj = nn.Linear(d_in, d_stream)

        # Residual MLP
        self.h1 = nn.Linear(d_stream, d_stream * 2)
        self.h2 = nn.Linear(d_stream * 2, d_stream)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_stream)

        # Initialize with small weights for stable training
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.h1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.h2.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.h1.bias)
        nn.init.zeros_(self.h2.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            h: Input features (B, T, d_in)

        Returns:
            Stream features (B, T, d_stream)
        """
        z = self.proj(h)                  # (B, T, d_stream)
        y = F.gelu(self.h1(z))           # (B, T, d_stream * 2)
        y = self.h2(self.drop(y))        # (B, T, d_stream)
        return self.norm(z + y)          # Residual + normalization


class LowRankAdapter(nn.Module):
    """
    Low-rank adapter for sheaf restriction maps.

    Projects stream representations to overlap spaces using bottleneck architecture.
    This is the "restriction map" ρ in sheaf theory terminology.

    For streams sA and sB sharing overlap O_AB:
    - R_AB: sA → O_AB (restriction from A to AB overlap)
    - R_BA: sB → O_AB (restriction from B to AB overlap)

    Sheaf condition: R_AB(sA) ≈ R_BA(sB) (agreement in overlap)

    Args:
        d_stream: Stream dimension
        d_overlap: Overlap space dimension
        rank: Bottleneck dimension (for low-rank factorization)
    """

    def __init__(self, d_stream: int, d_overlap: int, rank: int = 128):
        super().__init__()
        self.d_stream = d_stream
        self.d_overlap = d_overlap
        self.rank = rank

        # Low-rank factorization: stream → rank → overlap
        self.down = nn.Linear(d_stream, rank, bias=False)
        self.up = nn.Linear(rank, d_overlap, bias=False)

        # Initialize with larger weights to prevent collapse
        nn.init.xavier_uniform_(self.down.weight, gain=1.0)
        nn.init.xavier_uniform_(self.up.weight, gain=1.0)

        # Scale up initial outputs
        with torch.no_grad():
            self.up.weight.mul_(2.0)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Project stream to overlap space.

        Args:
            s: Stream features (B, T, d_stream)

        Returns:
            Overlap features (B, T, d_overlap)
        """
        return self.up(self.down(s))


class SheafConsistency(nn.Module):
    """
    Sheaf consistency loss and iterative correction module.

    Enforces the sheaf condition: projections to shared overlap spaces must agree.
    Uses iterative refinement (sheaf Laplacian) to bring representations into consensus.

    Mathematical formulation:
    - Loss: ||R_AB(sA) - R_BA(sB)||² + ||R_BC(sB) - R_CB(sC)||²
    - Correction: Iterative gradient descent in overlap space
    - Anti-collapse: Regularize adapter magnitudes to prevent trivial solutions

    Args:
        d_overlap: Dimension of overlap spaces
    """

    def __init__(self, d_overlap: int):
        super().__init__()
        self.d_overlap = d_overlap

        # Learnable correction step size (initialized to 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _pool_time(x: torch.Tensor) -> torch.Tensor:
        """Pool over time dimension for sequence-level consistency"""
        return x.mean(dim=1)  # (B, T, D) → (B, D)

    def loss_and_correct(
        self,
        sA: torch.Tensor,
        sB: torch.Tensor,
        sC: torch.Tensor,
        R_AB: LowRankAdapter,
        R_BA: LowRankAdapter,
        R_BC: LowRankAdapter,
        R_CB: LowRankAdapter,
        unroll_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute sheaf consistency loss and optionally apply iterative correction.

        Args:
            sA, sB, sC: Stream features (B, T, d_stream)
            R_AB, R_BA, R_BC, R_CB: Restriction map adapters
            unroll_steps: Number of correction iterations (0 = loss only, no correction)

        Returns:
            loss: Sheaf consistency loss
            residual: Per-sample residual (B,) for monitoring
            (sA, sB, sC): Corrected stream features
        """
        # Project to overlap spaces and pool over time
        a_to_b = self._pool_time(R_AB(sA))  # (B, d_overlap)
        b_to_a = self._pool_time(R_BA(sB))  # (B, d_overlap)
        b_to_c = self._pool_time(R_BC(sB))  # (B, d_overlap)
        c_to_b = self._pool_time(R_CB(sC))  # (B, d_overlap)

        # Sheaf loss: disagreement in overlaps
        loss_ab = F.mse_loss(a_to_b, b_to_a)
        loss_bc = F.mse_loss(b_to_c, c_to_b)
        loss = loss_ab + loss_bc

        # Anti-collapse regularization
        # Penalize near-zero adapter outputs (prevents trivial solution)
        adapter_magnitude = (
            a_to_b.pow(2).mean() + b_to_a.pow(2).mean() +
            b_to_c.pow(2).mean() + c_to_b.pow(2).mean()
        ) / 4.0

        target_magnitude = torch.tensor(0.5, device=adapter_magnitude.device, dtype=adapter_magnitude.dtype)
        magnitude_loss = F.mse_loss(adapter_magnitude, target_magnitude)
        loss = loss + 0.1 * magnitude_loss

        # Compute per-sample residual for monitoring
        residual = (
            (a_to_b - b_to_a).pow(2).sum(dim=-1) +  # (B,)
            (b_to_c - c_to_b).pow(2).sum(dim=-1)    # (B,)
        )

        # Iterative correction (sheaf Laplacian)
        if unroll_steps > 0:
            alpha = torch.sigmoid(self.alpha)  # Constrain to (0, 1)

            for _ in range(unroll_steps):
                # Gradient descent step toward consensus
                a_to_b = a_to_b - alpha * (a_to_b - b_to_a)
                b_to_a = b_to_a - alpha * (b_to_a - a_to_b)
                b_to_c = b_to_c - alpha * (b_to_c - c_to_b)
                c_to_b = c_to_b - alpha * (c_to_b - b_to_c)

            # Compute correction magnitude
            delta_A_norm = (a_to_b - self._pool_time(R_AB(sA))).norm(dim=-1, keepdim=True)  # (B, 1)
            delta_C_norm = (c_to_b - self._pool_time(R_CB(sC))).norm(dim=-1, keepdim=True)  # (B, 1)
            delta_B_norm = 0.5 * (
                (b_to_a - self._pool_time(R_BA(sB))).norm(dim=-1, keepdim=True) +
                (b_to_c - self._pool_time(R_BC(sB))).norm(dim=-1, keepdim=True)
            )  # (B, 1)

            # Apply small uniform correction to streams
            # Broadcast to (B, T, d_stream) and apply subtle perturbation
            sA = sA * (1.0 - delta_A_norm.unsqueeze(1) * 0.001)
            sB = sB * (1.0 - delta_B_norm.unsqueeze(1) * 0.001)
            sC = sC * (1.0 - delta_C_norm.unsqueeze(1) * 0.001)

        return loss, residual, (sA, sB, sC)


class StreamRouter(nn.Module):
    """
    Adaptive router that computes token-level weights for each stream.

    The router learns which stream should be emphasized based on content:
    - Visual tokens → higher weight on Stream A
    - Temporal/planning tokens → higher weight on Stream B
    - State/proprioception tokens → higher weight on Stream C

    Uses temperature-scaled softmax for smooth interpolation between streams.

    Args:
        d_stream: Dimension of each stream
        n_streams: Number of streams (default: 3)
        hidden_dim: Hidden dimension for router MLP
    """

    def __init__(self, d_stream: int, n_streams: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.n_streams = n_streams
        self.d_stream = d_stream

        # Router MLP: concatenated streams → logits
        self.router = nn.Sequential(
            nn.Linear(n_streams * d_stream, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_streams),
        )

        # Initialize
        for layer in self.router:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        sA: torch.Tensor,
        sB: torch.Tensor,
        sC: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute routing weights for each stream.

        Args:
            sA, sB, sC: Stream features (B, T, d_stream)
            attention_mask: Mask for padding tokens (B, T)
            temperature: Softmax temperature (higher = softer routing)

        Returns:
            Routing weights (B, n_streams) - per-sequence weights
        """
        # Concatenate streams
        S_concat = torch.cat([sA, sB, sC], dim=-1)  # (B, T, 3*d_stream)

        # Pool over time (masking padding tokens)
        mask = attention_mask.unsqueeze(-1).to(S_concat.dtype)  # (B, T, 1)
        S_pooled = (S_concat * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # (B, 3*d_stream)

        # Compute router logits
        logits = self.router(S_pooled)  # (B, n_streams)

        # Temperature-scaled softmax
        weights = F.softmax(logits / max(temperature, 1e-6), dim=-1)  # (B, n_streams)

        return weights


class SVMSWrapper(nn.Module):
    """
    Sheaf-based Multi-Stream wrapper for GR00T N1.6.

    Integrates all SVMS components:
    - 3 specialized stream heads
    - 4 restriction map adapters (for 2 overlaps: A↔B and B↔C)
    - Sheaf consistency module
    - Adaptive router
    - Stream merge layer
    - Auxiliary classification heads (for explicit supervision)

    This wrapper sits between the VLM backbone and the DiT action head,
    refining backbone features through specialized processing.

    Args:
        d_vlm: Dimension of VLM backbone features
        d_stream: Dimension of each stream
        d_overlap: Dimension of overlap spaces
        adapter_rank: Bottleneck rank for adapters
        dropout: Dropout probability
        use_aux_losses: Whether to use auxiliary supervision
    """

    def __init__(
        self,
        d_vlm: int,
        d_stream: int = 768,
        d_overlap: int = 384,
        adapter_rank: int = 128,
        dropout: float = 0.1,
        use_aux_losses: bool = True,
    ):
        super().__init__()
        self.d_vlm = d_vlm
        self.d_stream = d_stream
        self.d_overlap = d_overlap
        self.use_aux_losses = use_aux_losses

        # Stream heads
        self.stream_A = StreamHead(d_vlm, d_stream, dropout)  # Visual
        self.stream_B = StreamHead(d_vlm, d_stream, dropout)  # Temporal
        self.stream_C = StreamHead(d_vlm, d_stream, dropout)  # State

        # Sheaf restriction maps
        self.R_AB = LowRankAdapter(d_stream, d_overlap, adapter_rank)
        self.R_BA = LowRankAdapter(d_stream, d_overlap, adapter_rank)
        self.R_BC = LowRankAdapter(d_stream, d_overlap, adapter_rank)
        self.R_CB = LowRankAdapter(d_stream, d_overlap, adapter_rank)

        # Sheaf consistency module
        self.sheaf = SheafConsistency(d_overlap)

        # Router
        self.router = StreamRouter(d_stream, n_streams=3)

        # Merge layer: 3 streams → VLM dimension
        self.merge = nn.Linear(3 * d_stream, d_vlm)
        self.pre_merge_norm = nn.LayerNorm(3 * d_stream)

        # Initialize merge
        nn.init.normal_(self.merge.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.merge.bias)

        # Auxiliary classification heads (for stream specialization)
        if use_aux_losses:
            self.aux_head_A = nn.Linear(d_stream, 1)  # Binary: is visual token?
            self.aux_head_B = nn.Linear(d_stream, 1)  # Binary: is temporal token?
            self.aux_head_C = nn.Linear(d_stream, 1)  # Binary: is state token?

            nn.init.xavier_uniform_(self.aux_head_A.weight)
            nn.init.xavier_uniform_(self.aux_head_B.weight)
            nn.init.xavier_uniform_(self.aux_head_C.weight)
            nn.init.zeros_(self.aux_head_A.bias)
            nn.init.zeros_(self.aux_head_B.bias)
            nn.init.zeros_(self.aux_head_C.bias)

    def forward(
        self,
        backbone_features: torch.Tensor,
        attention_mask: torch.Tensor,
        sheaf_unroll_steps: int = 1,
        router_temperature: float = 1.0,
        aux_labels_A: Optional[torch.Tensor] = None,
        aux_labels_B: Optional[torch.Tensor] = None,
        aux_labels_C: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SVMS wrapper.

        Args:
            backbone_features: VLM features (B, T, d_vlm)
            attention_mask: Mask for padding (B, T)
            sheaf_unroll_steps: Number of sheaf correction steps
            router_temperature: Softmax temperature for routing
            aux_labels_A/B/C: Auxiliary labels for stream specialization (B, T)

        Returns:
            Dictionary containing:
                - refined_features: Merged stream features (B, T, d_vlm)
                - sheaf_loss: Sheaf consistency loss
                - sheaf_residual: Per-sample residual
                - router_weights: Routing weights (B, n_streams)
                - aux_loss: Auxiliary classification loss (if enabled)
                - aux_acc_A/B/C: Auxiliary accuracy per stream
        """
        B, T, _ = backbone_features.shape
        device = backbone_features.device

        # Apply stream heads
        sA = self.stream_A(backbone_features)  # (B, T, d_stream)
        sB = self.stream_B(backbone_features)  # (B, T, d_stream)
        sC = self.stream_C(backbone_features)  # (B, T, d_stream)

        # Sheaf consistency + correction
        sheaf_loss, sheaf_residual, (sA, sB, sC) = self.sheaf.loss_and_correct(
            sA, sB, sC,
            self.R_AB, self.R_BA, self.R_BC, self.R_CB,
            unroll_steps=sheaf_unroll_steps
        )

        # Router weights
        router_weights = self.router(sA, sB, sC, attention_mask, router_temperature)  # (B, 3)

        # Apply routing to streams
        w4 = router_weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, 3, 1)
        sA_weighted = sA * w4[:, :, 0, :]  # (B, T, d_stream)
        sB_weighted = sB * w4[:, :, 1, :]  # (B, T, d_stream)
        sC_weighted = sC * w4[:, :, 2, :]  # (B, T, d_stream)

        # Concatenate and merge
        S_concat = torch.cat([sA_weighted, sB_weighted, sC_weighted], dim=-1)  # (B, T, 3*d_stream)
        S_normed = self.pre_merge_norm(S_concat)
        refined_features = self.merge(S_normed)  # (B, T, d_vlm)

        # Small residual from backbone (bootstrapping)
        refined_features = refined_features + 0.01 * backbone_features

        # Auxiliary losses (stream specialization)
        aux_loss = torch.tensor(0.0, device=device)
        aux_acc_A = aux_acc_B = aux_acc_C = 0.0

        if self.use_aux_losses and aux_labels_A is not None:
            aux_logits_A = self.aux_head_A(sA).squeeze(-1)  # (B, T)
            aux_logits_B = self.aux_head_B(sB).squeeze(-1)  # (B, T)
            aux_logits_C = self.aux_head_C(sC).squeeze(-1)  # (B, T)

            # Binary cross-entropy loss
            aux_loss_A = F.binary_cross_entropy_with_logits(aux_logits_A, aux_labels_A, reduction="mean")
            aux_loss_B = F.binary_cross_entropy_with_logits(aux_logits_B, aux_labels_B, reduction="mean")
            aux_loss_C = F.binary_cross_entropy_with_logits(aux_logits_C, aux_labels_C, reduction="mean")
            aux_loss = aux_loss_A + aux_loss_B + aux_loss_C

            # Compute accuracy (for monitoring)
            with torch.no_grad():
                aux_preds_A = (torch.sigmoid(aux_logits_A) > 0.5).float()
                aux_preds_B = (torch.sigmoid(aux_logits_B) > 0.5).float()
                aux_preds_C = (torch.sigmoid(aux_logits_C) > 0.5).float()
                aux_acc_A = (aux_preds_A == aux_labels_A).float().mean().item()
                aux_acc_B = (aux_preds_B == aux_labels_B).float().mean().item()
                aux_acc_C = (aux_preds_C == aux_labels_C).float().mean().item()

        return {
            "refined_features": refined_features,
            "sheaf_loss": sheaf_loss,
            "sheaf_residual": sheaf_residual.mean(),  # Average over batch
            "router_weights": router_weights,
            "aux_loss": aux_loss,
            "aux_acc_A": aux_acc_A,
            "aux_acc_B": aux_acc_B,
            "aux_acc_C": aux_acc_C,
        }
