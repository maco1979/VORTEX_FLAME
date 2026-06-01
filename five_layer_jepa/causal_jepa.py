"""
Causal-JEPA (C-JEPA) — Object-Centric Causal World Model
=========================================================
Based on: "Causal-JEPA: Learning World Models through Object-Level Latent Interventions"
Authors: Heejeong Nam, Quentin Le Lidec, Lucas Maes, Yann LeCun, Randall Balestriero
Paper: arXiv:2602.11389 (Feb 2026)
Code: https://github.com/galilai-group/cjepa

Status: Architecture complete. Training pending GPU availability (P0 todo).

Core Innovation: Extends masked JEPA from patch-level to object-level.
  - Object-level masking forces the model to infer an object's state from OTHER objects
  - This induces latent interventions with counterfactual-like effects
  - Prevents shortcut solutions (e.g., copying an object's own history)
  - Makes interaction reasoning ESSENTIAL, not optional

Key Results:
  - Counterfactual reasoning: +20% absolute accuracy (CLEVRER)
  - Planning efficiency: 1% of patch-model tokens, 8x faster MPC (Push-T)
  - Formal proof: object-level masking induces causal inductive bias

Architecture Overview:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 1. Frozen Object-Centric Encoder (VideoSAUR / SAVi)                │
  │    pixels → N object slots (each 128-dim)                          │
  │                                                                    │
  │ 2. Object-Level Masking (Causal Intervention)                      │
  │    Randomly mask entire object history → force cross-object推理     │
  │                                                                    │
  │ 3. Causal Predictor (Transformer)                                  │
  │    Observable slots + auxiliaries → predict masked & future slots   │
  │                                                                    │
  │ 4. Auxiliary Conditioning (structured, not naive concat)           │
  │    Cross-attention injection of auxiliary variables                │
  │                                                                    │
  │ 5. Joint Loss: Masked Recovery + Forward Prediction + VICReg       │
  └──────────────────────────────────────────────────────────────────────┘

Integration with VORTEX_FLAME 5-Layer JEPA:
  - C-JEPA enhances each layer's encoder with object-centric slot extraction
  - C-JEPA replaces the simple MLP predictor with a causal Transformer predictor
  - C-JEPA adds object-level masking as a training strategy
  - The 5 modalities (V/A/PHYS/ART/DESIGN) each get their own slot vocabulary
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class JEPAModality(Enum):
    VISUAL = "visual"
    AUDIO = "audio"
    PHYSICS = "physics"
    ART = "art"
    DESIGN = "design"


class CausalInterventionType(Enum):
    NONE = "none"
    OBJECT_MASK = "object_mask"
    TEMPORAL_MASK = "temporal_mask"
    JOINT_MASK = "joint_mask"


@dataclass
class ObjectSlot:
    slot_id: int
    embedding: torch.Tensor
    is_masked: bool = False
    is_observable: bool = True
    object_type: str = "unknown"


@dataclass
class CausalInterventionResult:
    masked_slot_ids: List[int]
    observable_slot_ids: List[int]
    intervention_type: CausalInterventionType
    predicted_slots: Dict[int, torch.Tensor]
    prediction_confidence: Dict[int, float]


@dataclass
class CJEPATrainingBatch:
    slots_history: torch.Tensor
    slots_future: torch.Tensor
    auxiliaries: Optional[torch.Tensor] = None
    slot_mask: Optional[torch.Tensor] = None


class SlotAttention(nn.Module):
    """
    Slot Attention mechanism (Locatello et al., ICML 2020).
    Iteratively binds features to a fixed number of object slots.

    This is the core mechanism that converts pixel/feature representations
    into object-centric slots — the foundation of C-JEPA's efficiency.

    Architecture:
      input_features (B, N_input, D) → N_slots object slots (B, N_slots, slot_dim)

    The iterative attention mechanism:
      1. Initialize slots from learnable distribution or Gaussian
      2. For each iteration:
         a. Compute attention weights: slots × features → attention map
         b. Normalize attention (softmax over slots for competition)
         c. Weighted sum of features → slot updates
         d. Update slots via GRU + FFN residual
    """

    def __init__(
        self,
        num_slots: int = 7,
        input_dim: int = 512,
        slot_dim: int = 128,
        num_iterations: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations

        self.project_q = nn.Linear(slot_dim, hidden_dim)
        self.project_k = nn.Linear(input_dim, hidden_dim)
        self.project_v = nn.Linear(input_dim, slot_dim)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )

        self.slots_init = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        nn.init.xavier_uniform_(self.slots_init)

    def forward(
        self, inputs: torch.Tensor, num_slots: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_input, _ = inputs.shape
        n_slots = num_slots or self.num_slots

        inputs = self.norm_input(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        slots = self.slots_init.expand(B, -1, -1)
        if n_slots != self.num_slots:
            slots = slots[:, :n_slots, :]

        attn_weights = None
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)

            attn_logits = torch.einsum("bsd,bnd->bsn", q, k) / math.sqrt(q.shape[-1])
            attn_weights = F.softmax(attn_logits, dim=1)

            attn_weights_normed = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.einsum("bsn,bnd->bsd", attn_weights_normed, v)

            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(B, n_slots, self.slot_dim)

            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_weights


class ObjectSlotEncoder(nn.Module):
    """
    Encodes raw features into object-centric slot representations.
    Combines a feature extractor with Slot Attention to produce
    a compact set of object slots.

    This replaces the simple ContextEncoder in the original 5-Layer JEPA.
    Instead of: LLM hidden_states → Linear → embed_dim
    We do:      features → FeatureProj → SlotAttention → N_slots × slot_dim

    The slot representation is dramatically more efficient:
    - Patch-based: ~196 patches × 768 dim = 150,528 features
    - Object-based: ~7 slots × 128 dim = 896 features (1.02% of patch-based!)
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_slots: int = 7,
        slot_dim: int = 128,
        num_iterations: int = 3,
        feature_proj_dim: int = 512,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, feature_proj_dim),
            nn.LayerNorm(feature_proj_dim),
            nn.GELU(),
        )

        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            input_dim=feature_proj_dim,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
        )

        self.identity_encoding = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.temporal_encoding = None

    def forward(
        self,
        features: torch.Tensor,
        temporal_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if features.dim() == 4:
            B, T, N, D = features.shape
            features_flat = features.reshape(B * T, N, D)
            projected = self.feature_proj(features_flat)
            slots, attn_weights = self.slot_attention(projected)
            slots = slots.reshape(B, T, -1, self.slot_dim)
            attn_weights = attn_weights.reshape(B, T, -1, N)
        else:
            B, N, D = features.shape
            projected = self.feature_proj(features)
            slots, attn_weights = self.slot_attention(projected)
            slots = slots.unsqueeze(1)
            attn_weights = attn_weights.unsqueeze(1)

        id_enc = self.identity_encoding[:, :slots.shape[-2], :]
        slots = slots + id_enc

        return slots, attn_weights


class ObjectLevelMasker(nn.Module):
    """
    C-JEPA's core innovation: Object-level masking as causal intervention.

    Instead of randomly masking individual patches (I-JEPA), we mask ENTIRE
    objects across their temporal history. This forces the predictor to infer
    a masked object's state from OTHER objects — inducing counterfactual
    reasoning.

    Masking Strategies:
    1. Random Object Mask: randomly select K objects to mask
    2. Temporal Object Mask: mask an object's history (past T frames)
    3. Joint Mask: combine object + temporal masking

    The key insight from the paper (formally proven):
      Object-level masking ≈ latent intervention in a structural causal model
      → induces causal inductive bias
      → prevents shortcut solutions (e.g., just copying own history)
    """

    def __init__(
        self,
        num_slots: int = 7,
        mask_ratio: float = 0.5,
        min_masked: int = 1,
        max_masked: Optional[int] = None,
        temporal_mask_prob: float = 0.3,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.mask_ratio = mask_ratio
        self.min_masked = min_masked
        self.max_masked = max_masked or num_slots - 1
        self.temporal_mask_prob = temporal_mask_prob

        self.mask_token = nn.Parameter(torch.zeros(128))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        slots: torch.Tensor,
        force_num_masked: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        B, T, N, D = slots.shape

        if force_num_masked is not None:
            num_masked = force_num_masked
        else:
            num_masked = max(self.min_masked, int(N * self.mask_ratio))
            num_masked = min(num_masked, self.max_masked)

        masked_slots = slots.clone()
        slot_mask = torch.zeros(B, T, N, dtype=torch.bool, device=slots.device)
        masked_indices_batch = []

        for b in range(B):
            perm = torch.randperm(N, device=slots.device)
            masked_idx = perm[:num_masked].tolist()
            masked_indices_batch.append(masked_idx)

            for idx in masked_idx:
                slot_mask[b, :, idx] = True
                if torch.rand(1).item() < self.temporal_mask_prob:
                    masked_slots[b, :, idx] = self.mask_token
                else:
                    t_split = torch.randint(1, T, (1,)).item()
                    masked_slots[b, :t_split, idx] = self.mask_token

        return masked_slots, slot_mask, masked_indices_batch


class AuxiliaryConditioner(nn.Module):
    """
    C-JEPA's auxiliary conditioning strategy.

    Instead of naively concatenating auxiliary variables with slot representations
    (which limits expressiveness), C-JEPA uses cross-attention to inject auxiliary
    information into the predictor.

    Auxiliary variables can include:
    - Action vectors (robot actions, style parameters, etc.)
    - Environmental context (gravity, friction, lighting)
    - Task-specific signals (goal position, target style)

    Architecture:
      Slots → Self-Attention (Q from slots)
      Aux   → Cross-Attention (K,V from aux, Q from slots)
      Combined → FFN

    This is more expressive than concatenation because:
    1. Each slot can attend to different aspects of the auxiliary info
    2. The attention weights reveal which auxiliary info matters for which object
    3. No information bottleneck from dimensionality reduction
    """

    def __init__(
        self,
        slot_dim: int = 128,
        aux_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.aux_dim = aux_dim

        self.aux_proj = nn.Linear(aux_dim, slot_dim)

        self.cross_attn_layers = nn.ModuleList()
        self.self_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.self_attn_layers.append(
                nn.MultiheadAttention(slot_dim, num_heads, batch_first=True)
            )
            self.cross_attn_layers.append(
                nn.MultiheadAttention(slot_dim, num_heads, batch_first=True)
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(slot_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, slot_dim),
                )
            )
            self.norms.append(nn.ModuleList([
                nn.LayerNorm(slot_dim),
                nn.LayerNorm(slot_dim),
                nn.LayerNorm(slot_dim),
            ]))

    def forward(
        self,
        slots: torch.Tensor,
        auxiliaries: torch.Tensor,
    ) -> torch.Tensor:
        if auxiliaries.dim() == 2:
            auxiliaries = auxiliaries.unsqueeze(1)

        aux = self.aux_proj(auxiliaries)

        output = slots
        for i in range(len(self.self_attn_layers)):
            norm1, norm2, norm3 = self.norms[i]

            self_attn_out, _ = self.self_attn_layers[i](
                norm1(output), norm1(output), norm1(output)
            )
            output = output + self_attn_out

            cross_attn_out, _ = self.cross_attn_layers[i](
                norm2(output), aux, aux
            )
            output = output + cross_attn_out

            ffn_out = self.ffn_layers[i](norm3(output))
            output = output + ffn_out

        return output


class CausalPredictor(nn.Module):
    """
    C-JEPA's causal predictor — a Transformer that predicts masked and future
    object slots from observable slots and auxiliary variables.

    This replaces the simple MLP predictor (JEPAActionPredictor) in the
    original 5-Layer JEPA. The key differences:

    1. Slot-level prediction (not single embedding vector)
    2. Cross-object attention (objects attend to each other)
    3. Object-level masking forces causal reasoning
    4. Auxiliary conditioning via cross-attention

    Training Objective (Joint):
      L = L_masked_recovery + λ₁ * L_forward_prediction + λ₂ * L_vicreg

    Where:
      L_masked_recovery  = VICReg(predicted_masked_slots, target_masked_slots)
      L_forward_prediction = VICReg(predicted_future_slots, target_future_slots)
      L_vicreg           = variance + invariance + covariance regularization
    """

    def __init__(
        self,
        slot_dim: int = 128,
        num_slots: int = 7,
        num_heads: int = 4,
        num_layers: int = 4,
        hidden_dim: int = 256,
        aux_dim: int = 64,
        history_len: int = 6,
        future_len: int = 4,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.history_len = history_len
        self.future_len = future_len

        self.input_proj = nn.Linear(slot_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, slot_dim)

        self.temporal_pos = nn.Parameter(
            torch.randn(1, history_len + future_len, hidden_dim) * 0.02
        )
        self.slot_pos = nn.Parameter(
            torch.randn(1, num_slots, hidden_dim) * 0.02
        )

        self.auxiliary_conditioner = AuxiliaryConditioner(
            slot_dim=hidden_dim,
            aux_dim=aux_dim,
            num_heads=num_heads,
            num_layers=2,
            hidden_dim=hidden_dim * 2,
        )

        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_layers.append(nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                ),
                "norm1": nn.LayerNorm(hidden_dim),
                "norm2": nn.LayerNorm(hidden_dim),
            }))

        self.future_query = nn.Parameter(
            torch.randn(1, future_len, num_slots, hidden_dim) * 0.02
        )

    def forward(
        self,
        history_slots: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None,
        auxiliaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, D = history_slots.shape

        x = self.input_proj(history_slots)

        x = x + self.temporal_pos[:, :T, :].unsqueeze(2)
        x = x + self.slot_pos[:, :N, :].unsqueeze(1)

        x_flat = x.reshape(B, T * N, -1)

        for layer in self.transformer_layers:
            normed = layer["norm1"](x_flat)
            attn_out, _ = layer["self_attn"](normed, normed, normed)
            x_flat = x_flat + attn_out

            ffn_out = layer["ffn"](layer["norm2"](x_flat))
            x_flat = x_flat + ffn_out

        x = x_flat.reshape(B, T, N, -1)

        if auxiliaries is not None:
            x_reshaped = x.reshape(B * T, N, -1)
            if auxiliaries.dim() == 2:
                aux_expanded = auxiliaries.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            else:
                aux_expanded = auxiliaries.reshape(B * T, -1)
            x_reshaped = self.auxiliary_conditioner(x_reshaped, aux_expanded)
            x = x_reshaped.reshape(B, T, N, -1)

        recovered_slots = self.output_proj(x)

        future_q = self.future_query.expand(B, -1, -1, -1).reshape(B, self.future_len * N, -1)
        future_t_pos = self.temporal_pos[:, T:T + self.future_len, :].unsqueeze(2).expand(-1, -1, N, -1).reshape(1, self.future_len * N, -1)
        future_s_pos = self.slot_pos[:, :N, :].unsqueeze(1).expand(-1, self.future_len, -1, -1).reshape(1, self.future_len * N, -1)
        future_q = future_q + future_t_pos + future_s_pos

        for layer in self.transformer_layers:
            normed_q = layer["norm1"](future_q)
            normed_kv = layer["norm1"](x_flat)
            attn_out, _ = layer["self_attn"](normed_q, normed_kv, normed_kv)
            future_q = future_q + attn_out

            ffn_out = layer["ffn"](layer["norm2"](future_q))
            future_q = future_q + ffn_out

        future_slots = self.output_proj(future_q.reshape(B, self.future_len, N, -1))

        return recovered_slots, future_slots


class CausalVICRegLoss(nn.Module):
    """
    Extended VICReg loss for C-JEPA with causal structure.

    Adds a causal interaction regularization term that encourages
    the predicted slots to capture inter-object dependencies,
    not just individual object dynamics.

    L_causal = λ_causal * Σ_{i≠j} |cov(z_i, z_j) - cov_target(z_i, z_j)|

    This ensures the predictor doesn't just learn independent
    per-object dynamics but captures the causal interactions.
    """

    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        causal_weight: float = 5.0,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.causal_weight = causal_weight

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if z_pred.dim() == 3:
            B, N, D = z_pred.shape
            z_pred_2d = z_pred.reshape(B * N, D)
            z_target_2d = z_target.reshape(B * N, D)
        else:
            z_pred_2d = z_pred
            z_target_2d = z_target
            B = z_pred.shape[0]
            N = 1
            D = z_pred.shape[-1]

        z_pred_centered = z_pred_2d - z_pred_2d.mean(dim=0)
        z_target_centered = z_target_2d - z_target_2d.mean(dim=0)

        sim_loss = F.mse_loss(z_pred_2d, z_target_2d)

        std_pred = torch.sqrt(z_pred_2d.var(dim=0) + 1e-4)
        std_target = torch.sqrt(z_target_2d.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))

        B_eff = z_pred_2d.shape[0]
        cov_pred = (z_pred_centered.T @ z_pred_centered) / (B_eff - 1)
        cov_target = (z_target_centered.T @ z_target_centered) / (B_eff - 1)
        num_features = cov_pred.shape[0]
        eye_mask = ~torch.eye(num_features, device=cov_pred.device, dtype=torch.bool)
        off_diag_pred = cov_pred[eye_mask]
        off_diag_target = cov_target[eye_mask]
        cov_loss = (off_diag_pred.pow(2).sum() + off_diag_target.pow(2).sum()) / B_eff

        vicreg = self.sim_weight * sim_loss + self.var_weight * var_loss + self.cov_weight * cov_loss

        causal_loss = torch.tensor(0.0, device=z_pred.device)
        if z_pred.dim() == 3 and N > 1:
            z_pred_c3d = z_pred - z_pred.mean(dim=0)
            z_target_c3d = z_target - z_target.mean(dim=0)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        cov_pred_ij = (z_pred_c3d[:, i] * z_pred_c3d[:, j]).mean()
                        cov_target_ij = (z_target_c3d[:, i] * z_target_c3d[:, j]).mean()
                        causal_loss = causal_loss + (cov_pred_ij - cov_target_ij).pow(2)
            causal_loss = causal_loss / (N * (N - 1))

        total = vicreg + self.causal_weight * causal_loss

        return {
            "total": total,
            "vicreg": vicreg,
            "sim_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "causal_loss": causal_loss,
        }


class CJEPACuriosityReward(nn.Module):
    """
    Causal curiosity reward for C-JEPA.

    Extends the standard curiosity reward with causal discovery bonus:
    - Standard curiosity: ||z_predicted - z_actual||^2 (prediction error)
    - Causal bonus: reward for correctly predicting masked objects
      (high prediction error on masked objects → high causal discovery potential)

    This is particularly powerful for the RL fusion in VORTEX_FLAME:
    - PHYS-JEPA: discovers new physical interaction laws
    - V-JEPA: discovers novel visual object relationships
    - A-JEPA: discovers harmonic/melodic interaction patterns
    """

    def __init__(
        self,
        eta_prediction: float = 0.1,
        eta_causal: float = 0.3,
        reward_scale: float = 1.0,
    ):
        super().__init__()
        self.eta_prediction = eta_prediction
        self.eta_causal = eta_causal
        self.reward_scale = reward_scale
        self.register_buffer("running_mean", torch.tensor(0.0))
        self.register_buffer("running_var", torch.tensor(1.0))

    @torch.no_grad()
    def compute_reward(
        self,
        z_predicted: torch.Tensor,
        z_actual: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prediction_reward = self.eta_prediction * F.mse_loss(
            z_predicted, z_actual, reduction="none"
        ).mean(dim=-1)

        causal_reward = torch.tensor(0.0, device=z_predicted.device)
        if slot_mask is not None and slot_mask.any():
            masked_pred = z_predicted[slot_mask]
            masked_actual = z_actual[slot_mask]
            if masked_pred.numel() > 0:
                causal_reward = self.eta_causal * F.mse_loss(
                    masked_pred, masked_actual, reduction="none"
                ).mean(dim=-1)

        raw_reward = prediction_reward + causal_reward

        batch_mean = raw_reward.mean()
        batch_var = raw_reward.var()
        self.running_mean = 0.99 * self.running_mean + 0.01 * batch_mean
        self.running_var = 0.99 * self.running_var + 0.01 * batch_var

        normalized = (raw_reward - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
        return normalized * self.reward_scale


class CJEPALayer(nn.Module):
    """
    A single C-JEPA layer for one modality.

    Replaces JEPAEngine in the original 5-Layer JEPA with:
    1. ObjectSlotEncoder instead of ContextEncoder
    2. ObjectLevelMasker for causal intervention
    3. CausalPredictor instead of JEPAActionPredictor
    4. AuxiliaryConditioner for structured auxiliary injection
    5. CausalVICRegLoss instead of plain VICRegLoss
    6. CJEPACuriosityReward for causal discovery

    Training Pipeline:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Step 1: Feature Extraction                                      │
    │   raw_features → ObjectSlotEncoder → slots (B, T, N, D)        │
    │                                                                  │
    │ Step 2: Object-Level Masking (Causal Intervention)              │
    │   slots → ObjectLevelMasker → masked_slots + slot_mask          │
    │                                                                  │
    │ Step 3: Causal Prediction                                       │
    │   masked_slots + auxiliaries → CausalPredictor                  │
    │     → recovered_slots (masked recovery)                         │
    │     → future_slots (forward prediction)                         │
    │                                                                  │
    │ Step 4: Loss Computation                                        │
    │   CausalVICReg(recovered_masked, target_masked) → L_recovery    │
    │   CausalVICReg(predicted_future, target_future) → L_forward     │
    │   Total = L_recovery + λ * L_forward                            │
    │                                                                  │
    │ Step 5: Curiosity Reward                                        │
    │   CJEPACuriosityReward → intrinsic RL reward                    │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_slots: int = 7,
        slot_dim: int = 128,
        num_iterations: int = 3,
        mask_ratio: float = 0.5,
        num_predictor_heads: int = 4,
        num_predictor_layers: int = 4,
        predictor_hidden_dim: int = 256,
        aux_dim: int = 64,
        history_len: int = 6,
        future_len: int = 4,
        ema_decay: float = 0.996,
        curiosity_eta: float = 0.1,
        temporal_mask_prob: float = 1.0,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.history_len = history_len
        self.future_len = future_len
        self.ema_decay = ema_decay

        self.per_slot_input_proj = nn.ModuleList([
            nn.Linear(input_dim, input_dim)
            for _ in range(num_slots)
        ])

        for lin in self.per_slot_input_proj:
            with torch.no_grad():
                nn.init.normal_(lin.weight, std=0.001)
                lin.weight.data.add_(torch.eye(input_dim))
                nn.init.zeros_(lin.bias)

        self.context_encoder = ObjectSlotEncoder(
            input_dim=input_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
        )

        self.target_encoder = ObjectSlotEncoder(
            input_dim=input_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
        )
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.masker = ObjectLevelMasker(
            num_slots=num_slots,
            mask_ratio=mask_ratio,
            temporal_mask_prob=temporal_mask_prob,
        )

        self.predictor = CausalPredictor(
            slot_dim=slot_dim,
            num_slots=num_slots,
            num_heads=num_predictor_heads,
            num_layers=num_predictor_layers,
            hidden_dim=predictor_hidden_dim,
            aux_dim=aux_dim,
            history_len=history_len,
            future_len=future_len,
        )

        self.loss_fn = CausalVICRegLoss()
        self.curiosity = CJEPACuriosityReward(eta_prediction=curiosity_eta)

        self._last_prediction = None
        self._last_target = None

    @torch.no_grad()
    def update_ema(self):
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_tgt.data.mul_(self.ema_decay).add_(p_ctx.data, alpha=1 - self.ema_decay)

    def encode(
        self, features: torch.Tensor, use_target: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder = self.target_encoder if use_target else self.context_encoder
        return encoder(features)

    def project_to_slots(self, features_bb: torch.Tensor) -> torch.Tensor:
        """
        Convert a single feature vector per timestep into num_slots
        differentiated input tokens for Slot Attention to compete over.

        Each slot gets a distinct learnable projection of the same audio
        feature, creating the competitive pressure that forces slots to
        specialize in different acoustic attributes (pitch, timbre, rhythm).

        Args:
            features_bb: (B, T, D) — one D-dim feature per timestep

        Returns:
            (B, T, num_slots, D) — num_slots differentiated token views
        """
        B, T, D = features_bb.shape
        slot_outputs = []
        for i in range(self.num_slots):
            slot_outputs.append(self.per_slot_input_proj[i](features_bb))
        return torch.stack(slot_outputs, dim=2)

    def train_step(
        self,
        features_history: torch.Tensor,
        features_future: torch.Tensor,
        auxiliaries: Optional[torch.Tensor] = None,
        force_num_masked: Optional[int] = None,
    ) -> Dict[str, float]:
        with torch.no_grad():
            target_slots_history, _ = self.target_encoder(features_history)
            target_slots_future, _ = self.target_encoder(features_future)

        context_slots_history, _ = self.context_encoder(features_history)

        masked_slots, slot_mask, masked_indices = self.masker(
            context_slots_history, force_num_masked=force_num_masked
        )

        recovered_slots, predicted_future_slots = self.predictor(
            masked_slots, slot_mask=slot_mask, auxiliaries=auxiliaries
        )

        recovery_loss_dict = self.loss_fn(
            recovered_slots.reshape(-1, self.slot_dim),
            target_slots_history.reshape(-1, self.slot_dim),
            slot_mask=slot_mask.reshape(-1),
        )

        T_pred = predicted_future_slots.shape[1]
        T_target = target_slots_future.shape[1]
        T_match = min(T_pred, T_target)
        forward_loss_dict = self.loss_fn(
            predicted_future_slots[:, :T_match].reshape(-1, self.slot_dim),
            target_slots_future[:, :T_match].reshape(-1, self.slot_dim),
        )

        total_loss = recovery_loss_dict["total"] + 0.5 * forward_loss_dict["total"]

        self.update_ema()

        self._last_prediction = predicted_future_slots.detach()
        self._last_target = target_slots_future.detach()

        with torch.no_grad():
            reward_mask = slot_mask.reshape(-1) if slot_mask is not None and slot_mask.numel() == predicted_future_slots[:, :T_match].reshape(-1, self.slot_dim).shape[0] else None
            curiosity_reward = self.curiosity.compute_reward(
                predicted_future_slots[:, :T_match].reshape(-1, self.slot_dim),
                target_slots_future[:, :T_match].reshape(-1, self.slot_dim),
                slot_mask=reward_mask,
            )

        return {
            "total_loss": total_loss.item(),
            "recovery_loss": recovery_loss_dict["total"].item(),
            "forward_loss": forward_loss_dict["total"].item(),
            "causal_loss": recovery_loss_dict["causal_loss"].item(),
            "curiosity_reward": curiosity_reward.mean().item(),
            "num_masked_slots": sum(len(m) for m in masked_indices) / len(masked_indices),
        }

    @torch.no_grad()
    def predict_future(
        self,
        features: torch.Tensor,
        auxiliaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slots, attn_weights = self.context_encoder(features)
        _, future_slots = self.predictor(
            slots.unsqueeze(1) if slots.dim() == 3 else slots,
            slot_mask=None,
            auxiliaries=auxiliaries,
        )
        return future_slots, attn_weights

    @torch.no_grad()
    def counterfactual_predict(
        self,
        features: torch.Tensor,
        intervene_slot_id: int,
        intervene_embedding: torch.Tensor,
        auxiliaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slots, attn_weights = self.context_encoder(features)

        if slots.dim() == 3:
            slots = slots.unsqueeze(1)

        intervened_slots = slots.clone()
        intervened_slots[:, :, intervene_slot_id] = intervene_embedding

        _, future_slots = self.predictor(
            intervened_slots,
            slot_mask=None,
            auxiliaries=auxiliaries,
        )
        return future_slots, attn_weights

    @torch.no_grad()
    def plan(
        self,
        features: torch.Tensor,
        candidate_actions: List[torch.Tensor],
        goal_slots: torch.Tensor,
    ) -> Dict[str, Any]:
        slots, _ = self.context_encoder(features)
        if slots.dim() == 3:
            slots = slots.unsqueeze(1)

        scores = []
        trajectories = []

        for action in candidate_actions:
            if action.dim() == 1:
                action = action.unsqueeze(0).expand(slots.shape[0], -1)
            _, future_slots = self.predictor(slots, slot_mask=None, auxiliaries=action)

            last_step = future_slots[:, -1]
            if last_step.dim() == 3:
                last_step = last_step.reshape(-1)
            goal_flat = goal_slots.reshape(-1)
            min_len = min(last_step.shape[0], goal_flat.shape[0])
            distance = F.mse_loss(last_step[:min_len], goal_flat[:min_len])
            scores.append(-distance.item())
            trajectories.append(future_slots)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return {
            "best_action_index": best_idx,
            "action_scores": scores,
            "predicted_trajectory": trajectories[best_idx],
            "best_score": scores[best_idx],
        }


class CVJEPA(CJEPALayer):
    """
    Causal Visual JEPA — Object-centric visual understanding.

    Replaces VJEPA with object-level causal reasoning.
    Input: Visual features (from DINOv2 or similar backbone)
    Slots: Visual objects (entities, regions, objects in scene)
    Actions: Style/generation parameters, camera movements

    Key advantage over VJEPA:
    - 1% of token consumption vs patch-based models
    - Causal understanding of object interactions (collisions, occlusions)
    - Counterfactual visual reasoning ("what if this object were removed?")
    """

    def __init__(self, input_dim: int = 768, **kwargs):
        super().__init__(input_dim=input_dim, num_slots=7, slot_dim=128, **kwargs)


class CAJEPA(CJEPALayer):
    """
    Causal Audio JEPA — Object-centric audio/music understanding.

    Replaces AJEPA with object-level causal reasoning.
    Input: Audio spectrogram features
    Slots: Audio objects (instruments, voices, sound sources)
    Actions: Musical parameters (tempo, key, velocity)

    Key advantage: Discovers harmonic/melodic causal interactions
    between instruments, not just individual instrument dynamics.
    """

    def __init__(self, input_dim: int = 512, **kwargs):
        super().__init__(input_dim=input_dim, num_slots=5, slot_dim=128, **kwargs)


class CPHYSJEPA(CJEPALayer):
    """
    Causal Physics JEPA — Object-centric physical reasoning.

    Replaces PHYSJEPA with object-level causal reasoning.
    Input: Physical system features (from LLM or simulation)
    Slots: Physical objects (particles, bodies, fields)
    Actions: Physical parameters (force, angle, mass, velocity)

    Key advantage: Discovers causal physical laws through
    object-level interventions, not just correlations.
    This is the closest to the original C-JEPA paper's vision.
    """

    def __init__(self, input_dim: int = 512, **kwargs):
        super().__init__(input_dim=input_dim, num_slots=7, slot_dim=128, **kwargs)


class CARTJEPA(CJEPALayer):
    """
    Causal Art JEPA — Object-centric aesthetic understanding.

    Replaces ARTJEPA with object-level causal reasoning.
    Input: Artwork features (from LLM or vision model)
    Slots: Art objects (composition elements, color regions, shapes)
    Actions: Aesthetic parameters (composition, color harmony, contrast)

    Key advantage: Understands causal relationships between
    composition elements (e.g., how moving one element affects
    the visual balance of the whole).
    """

    def __init__(self, input_dim: int = 768, **kwargs):
        super().__init__(input_dim=input_dim, num_slots=8, slot_dim=128, **kwargs)


class CDESIGNJEPA(CJEPALayer):
    """
    Causal Design JEPA — Object-centric design logic verification.

    Replaces DESIGNJEPA with object-level causal reasoning.
    Input: Design specification features
    Slots: Design objects (components, layouts, UI elements)
    Actions: Design parameters (layout, spacing, hierarchy)

    Key advantage: Verifies causal consistency of design decisions
    (e.g., how changing one component's size affects the layout).
    """

    def __init__(self, input_dim: int = 512, **kwargs):
        super().__init__(input_dim=input_dim, num_slots=6, slot_dim=128, **kwargs)


class CFINJEPA(CJEPALayer):
    """
    Causal Financial JEPA — Object-centric financial time-series understanding.

    Input: OHLCV bar features (from FinancialFeatureProjector)
    Slots: Financial objects (trend, momentum, volatility, volume, S/R, regime)
    Actions: Trading parameters (position size, entry/exit, hedging)

    Key advantage: Discovers causal relationships between market
    factors (e.g., how momentum shift affects regime transition),
    enabling counterfactual analysis like "what if volume doubled?"
    """

    def __init__(self, input_dim: int = 256, **kwargs):
        super().__init__(input_dim=input_dim, num_slots=6, slot_dim=128, **kwargs)


class CCODEJEPA(CJEPALayer):
    """
    Causal Code JEPA — Object-centric code understanding.

    Input: Code AST + execution trace features
    Slots: Code objects (control_flow, data_structures, api_calls,
            error_handling, side_effects, type_system)
    Actions: Code transformation parameters (refactor, optimize, fix)

    Key advantage: Understands causal dependencies between code
    components (e.g., how changing one function affects callers).
    """

    def __init__(self, input_dim: int = 384, **kwargs):
        super().__init__(input_dim=input_dim, num_slots=7, slot_dim=128, **kwargs)


class CBIOJEPA(CJEPALayer):
    """
    Causal Biology JEPA — Object-centric biological system understanding.

    Input: Genomic/proteomic features
    Slots: Bio objects (gene_expression, protein_structure, metabolic_pathway,
            cell_signal, phenotype_trait, evolutionary_pressure)
    Actions: Biological parameters (gene knockout, drug dose, environment)

    Key advantage: Discovers causal gene regulatory networks through
    object-level interventions (e.g., what if gene X is knocked out?).
    """

    def __init__(self, input_dim: int = 512, **kwargs):
        super().__init__(input_dim=input_dim, num_slots=6, slot_dim=128, **kwargs)


class CGEOJEPA(CJEPALayer):
    """
    Causal Geography JEPA — Object-centric geographic/ecological understanding.

    Input: Geospatial features (satellite, climate, terrain)
    Slots: Geo objects (terrain, vegetation, water, climate, human_activity, geology)
    Actions: Environmental parameters (land use change, climate scenario)

    Key advantage: Models causal chains in ecosystems
    (e.g., how deforestation affects rainfall patterns).
    """

    def __init__(self, input_dim: int = 384, **kwargs):
        super().__init__(input_dim=input_dim, num_slots=6, slot_dim=128, **kwargs)


class CLAWJEPA(CJEPALayer):
    """
    Causal Law JEPA — Object-centric legal reasoning.

    Input: Legal text features (statutes, case law, regulations)
    Slots: Law objects (legal_rule, precedent, jurisdiction, temporal_validity,
            exception_clause, interpretation_method)
    Actions: Legal parameters (jurisdiction change, time period, fact pattern)

    Key advantage: Discovers causal logic in legal reasoning
    (e.g., how a precedent in one jurisdiction affects another).
    """

    def __init__(self, input_dim: int = 256, **kwargs):
        super().__init__(input_dim=input_dim, num_slots=6, slot_dim=128, **kwargs)


class CJEPAManager:
    """
    Manages all 5 C-JEPA layers and coordinates their training/inference.

    This replaces JEPAManager with the causal-enhanced architecture.

    Training Schedule:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Phase 1: Pretrain Object Slot Encoders (unsupervised)           │
    │   - Train Slot Attention on modality-specific data              │
    │   - VICReg loss only, no causal masking                         │
    │   - Target encoder EMA ramp from 0.996 → 1.0                   │
    │                                                                  │
    │ Phase 2: Train with Object-Level Masking (causal)               │
    │   - Enable ObjectLevelMasker                                    │
    │   - CausalPredictor learns inter-object dynamics                │
    │   - CausalVICReg loss with causal interaction term              │
    │                                                                  │
    │ Phase 3: RL Fine-tuning (causal curiosity-driven)               │
    │   - Freeze slot encoders                                        │
    │   - CJEPACuriosityReward drives exploration                     │
    │   - Causal discovery bonus for novel interactions               │
    │                                                                  │
    │ Phase 4: Soul Integration with Causal Reasoning                 │
    │   - C-JEPA embeddings feed into soul LoRA adapters              │
    │   - Counterfactual prediction enables "what-if" soul reasoning  │
    │   - Causal anomaly detection triggers soul mode switches        │
    └──────────────────────────────────────────────────────────────────┘

    Efficiency Comparison:
    ┌──────────────┬─────────────────────┬──────────────────────┐
    │ Metric       │ Patch-based JEPA    │ C-JEPA (Object-based)│
    ├──────────────┼─────────────────────┼──────────────────────┤
    │ Tokens       │ ~196 × 768 = 150K   │ ~7 × 128 = 896      │
    │ Token Ratio  │ 100%                │ 1.02%                │
    │ MPC Speed    │ 5763s (50 traj)     │ 673s (50 traj)       │
    │ Speedup      │ 1x                  │ 8.6x                 │
    │ Counterfactual│ ~40% accuracy      │ ~60% accuracy        │
    └──────────────┴─────────────────────┴──────────────────────┘
    """

    def __init__(
        self,
        llm_hidden_dim: int = 4096,
        action_dim: int = 64,
        slot_dim: int = 128,
    ):
        self.layers = {
            JEPAModality.VISUAL: CVJEPA(input_dim=768),
            JEPAModality.AUDIO: CAJEPA(input_dim=512),
            JEPAModality.PHYSICS: CPHYSJEPA(input_dim=llm_hidden_dim),
            JEPAModality.ART: CARTJEPA(input_dim=768),
            JEPAModality.DESIGN: CDESIGNJEPA(input_dim=llm_hidden_dim),
        }
        self._training_phase = 1
        self._num_masked_slots = {m: 0 for m in self.layers}

    def get_layer(self, modality: JEPAModality) -> CJEPALayer:
        return self.layers[modality]

    def set_training_phase(self, phase: int):
        if phase not in (1, 2, 3, 4):
            raise ValueError(f"Invalid phase {phase}, must be 1-4")
        self._training_phase = phase
        if phase == 1:
            self._num_masked_slots = {m: 0 for m in self.layers}
        elif phase >= 2:
            for m in self.layers:
                num_slots = self.layers[m].num_slots
                self._num_masked_slots[m] = max(1, num_slots // 2)

    def set_num_masked_slots(self, modality: JEPAModality, num: int):
        self._num_masked_slots[modality] = num

    def train_step_all(
        self,
        batch: Dict[JEPAModality, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for modality, (features_history, features_future, auxiliaries) in batch.items():
            layer = self.layers[modality]
            force_masked = self._num_masked_slots.get(modality, 0) if self._training_phase >= 2 else 0
            results[modality.value] = layer.train_step(
                features_history,
                features_future,
                auxiliaries=auxiliaries,
                force_num_masked=force_masked if force_masked > 0 else None,
            )
        return results

    def counterfactual_query(
        self,
        modality: JEPAModality,
        features: torch.Tensor,
        intervene_slot_id: int,
        intervene_embedding: torch.Tensor,
        auxiliaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers[modality]
        return layer.counterfactual_predict(
            features, intervene_slot_id, intervene_embedding, auxiliaries
        )

    def compute_combined_curiosity_reward(
        self,
        batch: Dict[JEPAModality, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        weights: Optional[Dict[JEPAModality, float]] = None,
    ) -> torch.Tensor:
        if weights is None:
            weights = {m: 1.0 / len(self.layers) for m in self.layers}

        total_reward = None
        for modality, (features_history, features_future, auxiliaries) in batch.items():
            layer = self.layers[modality]
            with torch.no_grad():
                context_slots, _ = layer.context_encoder(features_history)
                target_slots, _ = layer.target_encoder(features_future)

                if context_slots.dim() == 3:
                    context_slots = context_slots.unsqueeze(1)
                if target_slots.dim() == 3:
                    target_slots = target_slots.unsqueeze(1)

                masked_slots, slot_mask, _ = layer.masker(context_slots)
                _, predicted_future = layer.predictor(
                    masked_slots, slot_mask=slot_mask, auxiliaries=auxiliaries
                )

                reward = layer.curiosity.compute_reward(
                    predicted_future.reshape(-1, layer.slot_dim),
                    target_slots.reshape(-1, layer.slot_dim),
                    slot_mask=slot_mask.reshape(-1) if slot_mask is not None else None,
                )
                weighted = reward * weights.get(modality, 0.2)
                total_reward = weighted if total_reward is None else total_reward + weighted

        return total_reward

    def get_efficiency_stats(self) -> Dict[str, Any]:
        stats = {}
        for modality, layer in self.layers.items():
            patch_tokens = 196 * 768
            object_tokens = layer.num_slots * layer.slot_dim
            ratio = object_tokens / patch_tokens * 100
            stats[modality.value] = {
                "num_slots": layer.num_slots,
                "slot_dim": layer.slot_dim,
                "total_features": object_tokens,
                "vs_patch_ratio_pct": round(ratio, 2),
                "estimated_mpc_speedup": round(patch_tokens / object_tokens, 1),
            }
        return stats
