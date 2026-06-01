"""
Causal-JEPA v2 — Action-Conditioned + SIGReg Upgrade
=====================================================
Upgrades to the C-JEPA architecture (ICML 2026) based on latest research:

1. SIGReg Loss (LeWorldModel, LeCun et al. 2026.03):
   - Replaces 4-term CausalVICReg with 2-term Gaussian regularization
   - Drop: sim_loss (MSE), causal_loss (inter-slot covariance matching)
   - Keep: var_loss (variance) + cov_loss (decorrelation)
   - 2 terms vs 4 terms, 1 hyperparameter vs 4
   - Empirically more stable on small-scale training

2. Action-Conditioned CausalPredictor (V-JEPA 2, Meta 2025.06):
   - Adds action embedding input → CausalPredictor(f(slots, action)) = next_slots
   - Action is prepended as a special token to the slot sequence
   - Enables: "JEPA understands → JEPA controls" (world model → device control)
   - Backward compatible: when action=None, behaves like original CausalPredictor

Reference papers:
  - LeWorldModel: arXiv:2603.19312 (stable end-to-end JEPA from pixels, single GPU)
  - V-JEPA 2: arXiv:2506.09985 (zero-shot robot control from video pretraining)
  - C-JEPA: arXiv:2602.11389 (object-level latent masking, ICML 2026)

Usage:
  from five_layer_jepa.causal_jepa_v2 import SIGRegLoss, ActionCAJEPA
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGRegLoss(nn.Module):
    """
    SIGReg: Simplified Invariant Gaussian Regularizer
    From LeWorldModel (LeCun, Maes, Le Lidec et al., 2026.03)

    Key insight: CausalVICReg's 4-term loss can be simplified to 2 terms
    without loss of representation quality, and often with better stability.

    CausalVICReg (4 terms, 4 hyperparams):
      L = λ_sim * L_sim + λ_var * L_var + λ_cov * L_cov + λ_causal * L_causal

    SIGReg (2 terms, 2 hyperparams):
      L = λ_var * L_var + λ_cov * L_cov
      The predictor network implicitly provides invariance (no explicit sim_loss needed)

    Variance term: encourages each dimension to have unit variance (target_std=1.0)
      → Prevents representation collapse to constant
    Covariance term: decorrelates dimensions (off-diagonal of covariance matrix)
      → Prevents dimensional collapse, maximizes info capacity

    Why it works better:
      1. No sim_loss means the predictor has more freedom to explore representations
      2. Fewer competing loss terms = less gradient interference
      3. Gaussian target distribution is a natural prior for continuous representations
      4. Empirically: 48x faster planning vs DINOv2-world-model, competitive on control

    Risk: Without causal_loss, inter-slot causal interaction tracking may degrade.
          Mitigation: Action-Conditioned predictor naturally captures interactions
          via cross-object attention over time-conditioned sequences.
    """

    def __init__(self, var_weight: float = 25.0, cov_weight: float = 1.0, target_std: float = 1.0):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.target_std = target_std

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z.dim() == 3:
            B, N, D = z.shape
            z = z.reshape(B * N, D)

        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(self.target_std - std))

        z_centered = z - z.mean(dim=0)
        N_eff = z_centered.shape[0]
        cov = (z_centered.T @ z_centered) / (N_eff - 1)

        D = cov.shape[0]
        eye_mask = ~torch.eye(D, device=cov.device, dtype=torch.bool)
        off_diag = cov[eye_mask]
        cov_loss = off_diag.pow(2).sum() / D

        total = self.var_weight * var_loss + self.cov_weight * cov_loss

        return {
            "total": total,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }


class SIGRegWithPredictionLoss(nn.Module):
    """
    SIGReg + optional prediction accuracy term.

    Some setups benefit from keeping a weak similarity signal.
    This variant adds a light sim_loss (weight=1.0) to SIGReg's var+cov.
    """

    def __init__(
        self,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        sim_weight: float = 1.0,
        target_std: float = 1.0,
    ):
        super().__init__()
        self.sigreg = SIGRegLoss(var_weight=var_weight, cov_weight=cov_weight, target_std=target_std)
        self.sim_weight = sim_weight

    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        sigreg_out = self.sigreg(z_pred)
        sim_loss = F.mse_loss(z_pred.reshape(-1, z_pred.shape[-1]), z_target.reshape(-1, z_target.shape[-1]))
        total = sigreg_out["total"] + self.sim_weight * sim_loss
        return {**sigreg_out, "sim_loss": sim_loss, "total": total}


class ActionConditionedCausalPredictor(nn.Module):
    """
    Action-Conditioned Causal Predictor — inspired by V-JEPA 2-AC.

    Extends CausalPredictor (ICML 2026) with an action embedding input.
    Action tokens are prepended to the slot sequence before the Transformer.

    Architecture:
      Input:  slots_history (B,T,N,D) + action (B,action_dim)
      Processing:
        1. ActionEmbedding(action) → action_token (B,1,D_hidden)
        2. Input = cat([action_token, flatten(slots_history)], dim=seq)
        3. Transformer with self-attention over all tokens
        4. Split output: action_out (discarded), recovered_slots, future_slots

    When action=None: falls back to standard CausalPredictor behavior.

    Integration with DeviceGateway:
      device_gateway.ActionSpec → encode → action vector (64-dim)
      → ActionConditionedCausalPredictor(slots, action) → predicted next state
      → VORTEX can evaluate action consequences before execution
    """

    def __init__(
        self,
        slot_dim: int = 128,
        num_slots: int = 7,
        num_heads: int = 4,
        num_layers: int = 4,
        hidden_dim: int = 256,
        aux_dim: int = 64,
        action_dim: int = 64,
        history_len: int = 6,
        future_len: int = 4,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.history_len = history_len
        self.future_len = future_len
        self.action_dim = action_dim

        self.input_proj = nn.Linear(slot_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, slot_dim)

        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.temporal_pos = nn.Parameter(torch.randn(1, history_len + future_len, hidden_dim) * 0.02)
        self.slot_pos = nn.Parameter(torch.randn(1, num_slots, hidden_dim) * 0.02)

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

        self.future_query = nn.Parameter(torch.randn(1, future_len, num_slots, hidden_dim) * 0.02)

    def forward(
        self,
        history_slots: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None,
        auxiliaries: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, D = history_slots.shape

        x = self.input_proj(history_slots)
        x = x + self.temporal_pos[:, :T, :].unsqueeze(2)
        x = x + self.slot_pos[:, :N, :].unsqueeze(1)
        x_flat = x.reshape(B, T * N, -1)

        if action is not None:
            action_token = self.action_embed(action)
            if action_token.dim() == 2:
                action_token = action_token.unsqueeze(1)
            x_flat = torch.cat([action_token, x_flat], dim=1)

        for layer in self.transformer_layers:
            normed = layer["norm1"](x_flat)
            attn_out, _ = layer["self_attn"](normed, normed, normed)
            x_flat = x_flat + attn_out
            ffn_out = layer["ffn"](layer["norm2"](x_flat))
            x_flat = x_flat + ffn_out

        if action is not None:
            x_flat = x_flat[:, 1:, :]

        x = x_flat.reshape(B, T, N, -1)

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


class ActionCAJEPA(nn.Module):
    """
    Action-Conditioned CAJEPA — extends causal audio JEPA with action control.

    Reuses ObjectSlotEncoder and ObjectLevelMasker from causal_jepa.py.
    Adds ActionConditionedCausalPredictor for action-conditioned world modeling.

    This is the VORTEX_FLAME bridge between "understanding the world" (JEPA)
    and "controlling the world" (DeviceGateway):
      - Audio input → CAJEPA encodes world state as slots
      - Device action → ActionConditionedCausalPredictor predicts consequence
      - VORTEX evaluates predicted state against safety rules
      - If safe → execute via DeviceGateway

    Training modes:
      1. action=None: standard C-JEPA training (backward compatible)
      2. action=random: explore action space for curiosity-driven learning
      3. action=device_gateway: action-conditioned world model training
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_slots: int = 5,
        slot_dim: int = 128,
        num_iterations: int = 3,
        mask_ratio: float = 0.5,
        num_predictor_heads: int = 4,
        num_predictor_layers: int = 4,
        predictor_hidden_dim: int = 256,
        aux_dim: int = 64,
        action_dim: int = 64,
        history_len: int = 6,
        future_len: int = 4,
        ema_decay: float = 0.996,
        use_sigreg: bool = True,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.history_len = history_len
        self.future_len = future_len
        self.ema_decay = ema_decay
        self.use_sigreg = use_sigreg

        from five_layer_jepa.causal_jepa import ObjectSlotEncoder, ObjectLevelMasker
        self.context_encoder = ObjectSlotEncoder(
            input_dim=input_dim, num_slots=num_slots, slot_dim=slot_dim, num_iterations=num_iterations,
        )
        self.target_encoder = ObjectSlotEncoder(
            input_dim=input_dim, num_slots=num_slots, slot_dim=slot_dim, num_iterations=num_iterations,
        )
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.masker = ObjectLevelMasker(num_slots=num_slots, mask_ratio=mask_ratio)

        self.predictor = ActionConditionedCausalPredictor(
            slot_dim=slot_dim, num_slots=num_slots,
            num_heads=num_predictor_heads, num_layers=num_predictor_layers,
            hidden_dim=predictor_hidden_dim, aux_dim=aux_dim,
            action_dim=action_dim, history_len=history_len, future_len=future_len,
        )

        if use_sigreg:
            self.loss_fn = SIGRegLoss(var_weight=25.0, cov_weight=1.0, target_std=1.0)
        else:
            from five_layer_jepa.causal_jepa import CausalVICRegLoss
            self.loss_fn = CausalVICRegLoss()

    @torch.no_grad()
    def update_ema(self):
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(self.ema_decay).add_(p_ctx.data, alpha=1 - self.ema_decay)

    def encode(self, features: torch.Tensor, use_target: bool = False):
        encoder = self.target_encoder if use_target else self.context_encoder
        return encoder(features)

    def train_step(
        self,
        features_history: torch.Tensor,
        features_future: torch.Tensor,
        auxiliaries: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        force_num_masked: Optional[int] = None,
    ) -> Dict[str, float]:
        with torch.no_grad():
            target_slots_history, _ = self.target_encoder(features_history)
            target_slots_future, _ = self.target_encoder(features_future)

        context_slots_history, _ = self.context_encoder(features_history)

        masked_slots, slot_mask, masked_indices = self.masker(
            context_slots_history, force_num_masked=force_num_masked,
        )

        recovered_slots, predicted_future_slots = self.predictor(
            masked_slots, slot_mask=slot_mask, auxiliaries=auxiliaries, action=action,
        )

        T_pred = predicted_future_slots.shape[1]
        T_target = target_slots_future.shape[1]
        T_match = min(T_pred, T_target)

        if self.use_sigreg:
            recovery_loss_dict = self.loss_fn(recovered_slots.reshape(-1, self.slot_dim))
            forward_loss_dict = self.loss_fn(predicted_future_slots[:, :T_match].reshape(-1, self.slot_dim))
        else:
            recovery_loss_dict = self.loss_fn(
                recovered_slots.reshape(-1, self.slot_dim),
                target_slots_history.reshape(-1, self.slot_dim),
            )
            forward_loss_dict = self.loss_fn(
                predicted_future_slots[:, :T_match].reshape(-1, self.slot_dim),
                target_slots_future[:, :T_match].reshape(-1, self.slot_dim),
            )

        total_loss = recovery_loss_dict["total"] + 0.5 * forward_loss_dict["total"]

        self.update_ema()

        return {
            "total_loss": total_loss.item(),
            "recovery_loss": recovery_loss_dict["total"].item(),
            "forward_loss": forward_loss_dict["total"].item(),
            "var_loss": recovery_loss_dict.get("var_loss", 0.0),
            "cov_loss": recovery_loss_dict.get("cov_loss", 0.0),
            "num_masked_slots": sum(len(m) for m in masked_indices) / max(1, len(masked_indices)),
        }

    @torch.no_grad()
    def predict_consequence(self, history_features: torch.Tensor, action: torch.Tensor):
        slots, _ = self.context_encoder(history_features)
        _, future_slots = self.predictor(slots, action=action)
        return future_slots
