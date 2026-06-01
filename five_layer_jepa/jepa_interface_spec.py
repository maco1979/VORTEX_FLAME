﻿﻿﻿"""
5-Layer JEPA World Model — Unified Interface Specification
===========================================================
Based on Meta's I-JEPA (CVPR 2023) and V-JEPA (ICLR 2024).

Status: Architecture complete. Training pending GPU availability (P0 todo).

Core Philosophy: Predict in representation space, not pixel space.

RL+LLM Fusion Architecture:
  LLM (Mistral-7B) provides world knowledge → JEPA encodes into modality-specific
  representations → Predictor learns temporal dynamics → Prediction error becomes
  intrinsic RL reward (curiosity-driven exploration)

Training Data Flow:
  1. LLM forward pass → hidden_states (4096-dim)
  2. Context Encoder: hidden_states → modality embedding (384/256/512/768-dim)
  3. Predictor: z_t + action → z_{t+1} (representation space prediction)
  4. Loss: VICReg-style (variance + invariance + covariance) between predicted and actual
  5. RL Reward: -||z_predicted - z_actual||^2 (curiosity bonus for novel states)

Layers:
  V-JEPA     — Visual understanding (embed_dim=384)
  A-JEPA     — Audio/Music perception (embed_dim=256)
  PHYS-JEPA  — Physical quantity prediction (embed_dim=512)
  ART-JEPA   — Aesthetic evaluation (embed_dim=768)
  DESIGN-JEPA — Design logic verification (embed_dim=512)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAModality(Enum):
    VISUAL = "visual"
    AUDIO = "audio"
    PHYSICS = "physics"
    ART = "art"
    DESIGN = "design"


@dataclass
class JEPAResult:
    embedding: List[float]
    semantic_label: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JEPAPredictionResult:
    predicted_embedding: List[float]
    horizon: int
    uncertainty: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JEPAAnomalyResult:
    is_anomaly: bool
    anomaly_score: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JEPAPlanResult:
    best_action_index: int
    action_scores: List[float]
    predicted_trajectory: List[List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JEPAVerifyResult:
    is_valid: bool
    quality_score: float
    issues: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JEPATrajectory:
    observations: List[Any]
    actions: List[List[float]]
    rewards: List[float]
    embeddings: Optional[List[List[float]]] = None


class VICRegLoss(nn.Module):
    """
    VICReg loss from Bardes et al. (ICLR 2022).
    Prevents representation collapse without negative pairs.

    Three components:
    - Variance: Keeps representations spread out (h_i - mean)^2
    - Invariance: Pulls positive pairs together ||z_i - z_j||^2
    - Covariance: Decorrelates dimensions (off-diagonal of cov matrix)
    """

    def __init__(self, sim_weight: float = 25.0, var_weight: float = 25.0, cov_weight: float = 1.0):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        batch_size = z_pred.shape[0]
        z_pred = z_pred - z_pred.mean(dim=0)
        z_target = z_target - z_target.mean(dim=0)

        sim_loss = F.mse_loss(z_pred, z_target)

        std_pred = torch.sqrt(z_pred.var(dim=0) + 1e-4)
        std_target = torch.sqrt(z_target.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))

        cov_pred = (z_pred.T @ z_pred) / (batch_size - 1)
        cov_target = (z_target.T @ z_target) / (batch_size - 1)
        num_features = cov_pred.shape[0]
        mask = ~torch.eye(num_features, device=cov_pred.device, dtype=torch.bool)
        off_diag_pred = cov_pred[mask]
        off_diag_target = cov_target[mask]
        cov_loss = (off_diag_pred.pow(2).sum() + off_diag_target.pow(2).sum()) / batch_size

        return self.sim_weight * sim_loss + self.var_weight * var_loss + self.cov_weight * cov_loss


class ContextEncoder(nn.Module):
    """
    Encodes LLM hidden states into modality-specific representations.

    Architecture: Linear(4096 → 4*embed_dim) → LayerNorm → GELU → Linear(4*embed_dim → embed_dim)

    This is the bridge between LLM world knowledge and JEPA representation space.
    The LLM provides the 'understanding', the Context Encoder projects it into
    the modality-specific representation where JEPA prediction happens.
    """

    def __init__(self, llm_hidden_dim: int = 4096, embed_dim: int = 384):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(llm_hidden_dim, 4 * embed_dim),
            nn.LayerNorm(4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)


class TargetEncoder(nn.Module):
    """
    EMA-updated target encoder for self-supervised learning.
    Slower-moving encoder provides stable targets for the predictor.

    Architecture: Same as ContextEncoder but updated via EMA.
    EMA decay: 0.996 (linearly ramped from 0.996 to 1.0 during training)
    """

    def __init__(self, llm_hidden_dim: int = 4096, embed_dim: int = 384):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(llm_hidden_dim, 4 * embed_dim),
            nn.LayerNorm(4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_ema(self, source_encoder: ContextEncoder, momentum: float = 0.996):
        for param_s, param_t in zip(source_encoder.parameters(), self.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)


class JEPAActionPredictor(nn.Module):
    """
    Predicts future representation given current representation and action.

    Architecture:
      z_t (embed_dim) + action (action_dim)
        → Concat → Linear → LayerNorm → GELU → Linear → z_{t+1} (embed_dim)

    This is the core RL fusion point: the predictor learns the dynamics model
    (transition function) in representation space. The prediction error becomes
    the intrinsic reward signal for curiosity-driven exploration.
    """

    def __init__(self, embed_dim: int = 384, action_dim: int = 64, hidden_dim: int = 768):
        super().__init__()
        input_dim = embed_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z_current: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_current, action], dim=-1)
        return self.net(x)


class CuriosityRewardModule(nn.Module):
    """
    Intrinsic reward based on JEPA prediction error.

    Reward = η * ||z_predicted - z_actual||^2

    High prediction error → novel/unexpected state → high curiosity reward
    Low prediction error → familiar state → low curiosity reward

    This implements the RL fusion: JEPA's prediction error drives exploration,
    while the LLM's world knowledge provides the representation foundation.

    Reward normalization: Running mean/std to keep rewards in reasonable range.
    """

    def __init__(self, eta: float = 0.1, reward_scale: float = 1.0):
        super().__init__()
        self.eta = eta
        self.reward_scale = reward_scale
        self.register_buffer("running_mean", torch.tensor(0.0))
        self.register_buffer("running_var", torch.tensor(1.0))
        self.register_buffer("count", torch.tensor(0.0))

    @torch.no_grad()
    def compute_reward(self, z_predicted: torch.Tensor, z_actual: torch.Tensor) -> torch.Tensor:
        raw_reward = self.eta * F.mse_loss(z_predicted, z_actual, reduction="none").mean(dim=-1)

        self.count += 1
        batch_mean = raw_reward.mean()
        batch_var = raw_reward.var()
        self.running_mean = 0.99 * self.running_mean + 0.01 * batch_mean
        self.running_var = 0.99 * self.running_var + 0.01 * batch_var

        normalized = (raw_reward - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
        return normalized * self.reward_scale


class BaseJEPA(ABC):
    """Abstract base class for all JEPA layers."""

    EMBED_DIM: int = 384
    MODALITY: JEPAModality = JEPAModality.VISUAL

    @abstractmethod
    def understand(self, input_data: Any) -> JEPAResult:
        pass

    @abstractmethod
    def predict(self, current_input: Any = None,
                action: Optional[List[float]] = None,
                horizon: int = 1) -> JEPAPredictionResult:
        pass

    @abstractmethod
    def detect_anomaly(self, actual_input: Any) -> JEPAAnomalyResult:
        pass

    @abstractmethod
    def plan(self, goal_embedding: List[float],
             candidate_actions: List[List[float]],
             horizon: int = 3) -> JEPAPlanResult:
        pass

    @abstractmethod
    def verify_generation(self, generated_output: Any,
                          context: str = "") -> JEPAVerifyResult:
        pass


class JEPAEngine(BaseJEPA):
    """
    Concrete JEPA engine implementing the RL+LLM fusion architecture.

    Training Pipeline:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Step 1: LLM Forward Pass                                       │
    │   input_tokens → Mistral-7B → hidden_states (B, L, 4096)      │
    │                                                                 │
    │ Step 2: Context Encoding (online)                               │
    │   hidden_states → ContextEncoder → z_context (B, embed_dim)    │
    │                                                                 │
    │ Step 3: Target Encoding (EMA, no grad)                          │
    │   hidden_states_target → TargetEncoder → z_target (B, embed)   │
    │                                                                 │
    │ Step 4: Prediction                                              │
    │   z_context + action → ActionPredictor → z_predicted (B, embed)│
    │                                                                 │
    │ Step 5: Loss Computation                                        │
    │   VICReg(z_predicted, z_target) → jepe_loss                    │
    │                                                                 │
    │ Step 6: RL Reward (for downstream RL training)                  │
    │   curiosity_reward = η * ||z_predicted - z_target||^2          │
    └─────────────────────────────────────────────────────────────────┘

    Data Flow for Each Layer:
    ┌──────────┬──────────────────────┬────────────────────────────────┐
    │ Layer    │ Input Source         │ Training Data                  │
    ├──────────┼──────────────────────┼────────────────────────────────┤
    │ V-JEPA   │ ComfyUI/viewport img │ Image pairs (t, t+Δt) + action│
    │ A-JEPA   │ Ableton/Audio input  │ Audio frames + MIDI actions    │
    │ PHYS-JEPA│ Einstein/Galileo Q&A │ Physics sim states + actions   │
    │ ART-JEPA │ Monet/VanGogh gen    │ Art pairs (style/content)      │
    │ DESIGN   │ DaVinci/OpenDesign   │ Design iterations + changes     │
    └──────────┴──────────────────────┴────────────────────────────────┘
    """

    def __init__(
        self,
        embed_dim: int = 384,
        llm_hidden_dim: int = 4096,
        action_dim: int = 64,
        predictor_hidden_dim: int = 768,
        ema_decay: float = 0.996,
        curiosity_eta: float = 0.1,
    ):
        self.embed_dim = embed_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.action_dim = action_dim

        self.context_encoder = ContextEncoder(llm_hidden_dim, embed_dim)
        self.target_encoder = TargetEncoder(llm_hidden_dim, embed_dim)
        self.predictor = JEPAActionPredictor(embed_dim, action_dim, predictor_hidden_dim)
        self.vicreg_loss = VICRegLoss()
        self.curiosity = CuriosityRewardModule(eta=curiosity_eta)

        self.ema_decay = ema_decay
        self._last_prediction: Optional[torch.Tensor] = None
        self._last_target: Optional[torch.Tensor] = None

    def encode(self, hidden_states: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        if use_target:
            with torch.no_grad():
                return self.target_encoder(hidden_states)
        return self.context_encoder(hidden_states)

    def train_step(
        self,
        hidden_states: torch.Tensor,
        hidden_states_target: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, float]:
        z_context = self.context_encoder(hidden_states)
        with torch.no_grad():
            z_target = self.target_encoder(hidden_states_target)
        z_predicted = self.predictor(z_context, actions)

        loss = self.vicreg_loss(z_predicted, z_target)

        self.target_encoder.update_ema(self.context_encoder, self.ema_decay)

        self._last_prediction = z_predicted.detach()
        self._last_target = z_target.detach()

        with torch.no_grad():
            curiosity_reward = self.curiosity.compute_reward(z_predicted, z_target)

        return {
            "loss": loss.item(),
            "curiosity_reward_mean": curiosity_reward.mean().item(),
            "prediction_error": F.mse_loss(z_predicted, z_target).item(),
        }

    def compute_intrinsic_reward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_target: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            z_context = self.context_encoder(hidden_states)
            z_target = self.target_encoder(hidden_states_target)
            z_predicted = self.predictor(z_context, actions)
        return self.curiosity.compute_reward(z_predicted, z_target)

    def understand(self, input_data: Any) -> JEPAResult:
        if isinstance(input_data, torch.Tensor):
            with torch.no_grad():
                z = self.context_encoder(input_data)
            embedding = z.squeeze().tolist()
            confidence = float(torch.norm(z).item())
            return JEPAResult(
                embedding=embedding,
                semantic_label=f"{self.MODALITY.value}_state",
                confidence=min(confidence / 10.0, 1.0),
            )
        return JEPAResult(embedding=[], semantic_label="unknown", confidence=0.0)

    def predict(self, current_input: Any = None,
                action: Optional[List[float]] = None,
                horizon: int = 1) -> JEPAPredictionResult:
        if current_input is None or action is None:
            return JEPAPredictionResult(
                predicted_embedding=[], horizon=horizon, uncertainty=1.0
            )

        with torch.no_grad():
            if isinstance(current_input, torch.Tensor):
                z_current = self.context_encoder(current_input)
            else:
                z_current = torch.zeros(1, self.embed_dim)

            action_tensor = torch.tensor([action], dtype=torch.float32)
            z_predicted = self.predictor(z_current, action_tensor)

            uncertainty = 0.0
            trajectory = [z_predicted.squeeze().tolist()]
            z_next = z_predicted
            for h in range(1, horizon):
                z_next = self.predictor(z_next, action_tensor)
                trajectory.append(z_next.squeeze().tolist())
                uncertainty += float(F.mse_loss(z_next, z_predicted).item())

        return JEPAPredictionResult(
            predicted_embedding=trajectory[-1],
            horizon=horizon,
            uncertainty=uncertainty / max(horizon, 1),
            metadata={"trajectory": trajectory},
        )

    def detect_anomaly(self, actual_input: Any) -> JEPAAnomalyResult:
        if self._last_prediction is None or self._last_target is None:
            return JEPAAnomalyResult(is_anomaly=False, anomaly_score=0.0, explanation="No prior prediction")

        if isinstance(actual_input, torch.Tensor):
            with torch.no_grad():
                z_actual = self.context_encoder(actual_input)
            error = float(F.mse_loss(z_actual, self._last_prediction).item())
        else:
            error = 1.0

        is_anomaly = error > 2.0
        return JEPAAnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=error,
            explanation=f"Prediction error {error:.4f} {'exceeds' if is_anomaly else 'within'} threshold 2.0",
        )

    def plan(self, goal_embedding: List[float],
             candidate_actions: List[List[float]],
             horizon: int = 3) -> JEPAPlanResult:
        if not candidate_actions:
            return JEPAPlanResult(best_action_index=0, action_scores=[], predicted_trajectory=[])

        z_goal = torch.tensor([goal_embedding], dtype=torch.float32)
        scores = []
        trajectories = []

        for action in candidate_actions:
            action_tensor = torch.tensor([action], dtype=torch.float32)
            with torch.no_grad():
                z_current = z_goal.clone()
                traj = [z_current.squeeze().tolist()]
                for _ in range(horizon):
                    z_next = self.predictor(z_current, action_tensor)
                    traj.append(z_next.squeeze().tolist())
                    z_current = z_next

            distance = float(F.mse_loss(z_current, z_goal).item())
            scores.append(-distance)
            trajectories.append(traj)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return JEPAPlanResult(
            best_action_index=best_idx,
            action_scores=scores,
            predicted_trajectory=trajectories[best_idx],
        )

    def verify_generation(self, generated_output: Any,
                          context: str = "") -> JEPAVerifyResult:
        if not isinstance(generated_output, torch.Tensor):
            return JEPAVerifyResult(is_valid=False, quality_score=0.0, issues=["Invalid input type"], suggestions=[])

        with torch.no_grad():
            z = self.context_encoder(generated_output)
            quality = float(torch.norm(z).item()) / 10.0
            quality = min(quality, 1.0)

        issues = []
        suggestions = []
        if quality < 0.3:
            issues.append("Low representation quality")
            suggestions.append("Input may be out of distribution")
        if self._last_prediction is not None:
            error = float(F.mse_loss(z, self._last_prediction).item())
            if error > 3.0:
                issues.append(f"High prediction divergence: {error:.4f}")
                suggestions.append("Generated output diverges from expected trajectory")

        return JEPAVerifyResult(
            is_valid=quality >= 0.3 and len(issues) == 0,
            quality_score=quality,
            issues=issues,
            suggestions=suggestions,
        )


class VJEPA(JEPAEngine):
    """
    Visual JEPA — Image/video understanding.

    Input: LLM hidden states from image-conditioned prompts
    Training data: Image pairs from ComfyUI (txt2img/img2img) + viewport screenshots
    Action space: Style/generation parameters (seed, steps, cfg_scale, etc.)

    RL Fusion: Curiosity reward drives exploration of novel visual styles,
    enabling Beethoven/Monet/VanGogh souls to discover new aesthetic territories
    without human-labeled style data.
    """
    EMBED_DIM = 384
    MODALITY = JEPAModality.VISUAL

    def __init__(self, **kwargs):
        super().__init__(embed_dim=384, **kwargs)


class AJEPA(JEPAEngine):
    """
    Audio JEPA — Music/acoustics perception.

    Input: LLM hidden states from audio-conditioned prompts
    Training data: Audio frames from Ableton + MIDI actions
    Action space: Musical parameters (tempo, key, velocity, note_on/off)

    RL Fusion: Curiosity reward drives exploration of novel harmonic/melodic
    patterns, enabling Beethoven soul to compose beyond training distribution.
    """
    EMBED_DIM = 256
    MODALITY = JEPAModality.AUDIO

    def __init__(self, **kwargs):
        super().__init__(embed_dim=256, **kwargs)


class PHYSJEPA(JEPAEngine):
    """
    Physics JEPA — Physical quantity prediction.

    Input: LLM hidden states from physics problem descriptions
    Training data: Physics simulation states + experimental outcomes
    Action space: Physical parameters (force, angle, mass, velocity)

    RL Fusion: Curiosity reward drives discovery of novel physical phenomena,
    enabling Einstein/Galileo souls to propose hypotheses beyond known physics.
    This is the closest to Silver's vision: self-discovery of physical laws.
    """
    EMBED_DIM = 512
    MODALITY = JEPAModality.PHYSICS

    def __init__(self, **kwargs):
        super().__init__(embed_dim=512, **kwargs)


class ARTJEPA(JEPAEngine):
    """
    Art JEPA — Aesthetic evaluation.

    Input: LLM hidden states from art descriptions/critiques
    Training data: Artwork pairs with style/content variations
    Action space: Aesthetic parameters (composition, color_harmony, contrast)

    RL Fusion: Curiosity reward drives exploration of aesthetic space,
    enabling Monet/VanGogh souls to discover novel artistic expressions
    that transcend their training data's aesthetic distribution.
    """
    EMBED_DIM = 768
    MODALITY = JEPAModality.ART

    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, **kwargs)


class DESIGNJEPA(JEPAEngine):
    """
    Design JEPA — Design logic verification.

    Input: LLM hidden states from design specifications
    Training data: Design iterations (wireframe → mockup → prototype)
    Action space: Design parameters (layout, spacing, hierarchy, color)

    RL Fusion: Curiosity reward drives exploration of design space,
    enabling DaVinci soul to discover novel design patterns that satisfy
    functional constraints while being aesthetically innovative.
    """
    EMBED_DIM = 512
    MODALITY = JEPAModality.DESIGN

    def __init__(self, **kwargs):
        super().__init__(embed_dim=512, **kwargs)


class JEPAManager:
    """
    Manages all 5 JEPA layers and coordinates their training/inference.

    Training Schedule:
    ┌──────────────────────────────────────────────────────────────┐
    │ Phase 1: Pretrain Context Encoders (unsupervised)           │
    │   - Each layer trains independently on modality data        │
    │   - VICReg loss only, no RL component                       │
    │   - Target encoder EMA ramp from 0.996 → 1.0               │
    │                                                              │
    │ Phase 2: Train Predictors with actions (supervised)         │
    │   - Add action-conditioned prediction                       │
    │   - Predictor learns dynamics model                         │
    │   - Curiosity reward module initialized                    │
    │                                                              │
    │ Phase 3: RL Fine-tuning (curiosity-driven)                  │
    │   - Freeze context/target encoders                          │
    │   - Use curiosity reward as intrinsic motivation            │
    │   - Fine-tune predictor for exploration                     │
    │   - Combined loss: jepe_loss + λ * policy_gradient_loss     │
    │                                                              │
    │ Phase 4: Soul Integration                                   │
    │   - JEPA embeddings feed into soul LoRA adapters            │
    │   - Anomaly detection triggers soul mode switches           │
    │   - Planning module guides soul action selection            │
    └──────────────────────────────────────────────────────────────┘

    Data Flow Summary:
    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │ LLM Hidden  │───▶│ Context      │───▶│ Predictor    │───▶│ Curiosity    │
    │ States      │    │ Encoder      │    │ (z_t + a →   │    │ Reward       │
    │ (4096-dim)  │    │ (→ embed_dim)│    │  z_{t+1})    │    │ (η*||Δz||²) │
    └─────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                     │
                           ┌──────────────┐    ┌──────────────┐       │
                           │ Target       │◀───│ EMA Update   │       ▼
                           │ Encoder      │    │ (0.996 decay)│   ┌──────────────┐
                           │ (no grad)    │    └──────────────┘   │ Soul LoRA    │
                           └──────┬───────┘                      │ Fine-tune    │
                                  │                               │ (RL reward)  │
                                  ▼                               └──────────────┘
                           ┌──────────────┐
                           │ VICReg Loss  │
                           │ (var+inv+cov)│
                           └──────────────┘
    """

    def __init__(self, llm_hidden_dim: int = 4096, action_dim: int = 64):
        self.layers: Dict[JEPAModality, JEPAEngine] = {
            JEPAModality.VISUAL: VJEPA(llm_hidden_dim=llm_hidden_dim, action_dim=action_dim),
            JEPAModality.AUDIO: AJEPA(llm_hidden_dim=llm_hidden_dim, action_dim=action_dim),
            JEPAModality.PHYSICS: PHYSJEPA(llm_hidden_dim=llm_hidden_dim, action_dim=action_dim),
            JEPAModality.ART: ARTJEPA(llm_hidden_dim=llm_hidden_dim, action_dim=action_dim),
            JEPAModality.DESIGN: DESIGNJEPA(llm_hidden_dim=llm_hidden_dim, action_dim=action_dim),
        }
        self._training_phase = 1

    def get_layer(self, modality: JEPAModality) -> JEPAEngine:
        return self.layers[modality]

    def train_step_all(
        self,
        batch: Dict[JEPAModality, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for modality, (h_online, h_target, actions) in batch.items():
            layer = self.layers[modality]
            if self._training_phase == 1:
                zero_actions = torch.zeros_like(actions)
                results[modality.value] = layer.train_step(h_online, h_target, zero_actions)
            else:
                results[modality.value] = layer.train_step(h_online, h_target, actions)
        return results

    def set_training_phase(self, phase: int):
        if phase not in (1, 2, 3, 4):
            raise ValueError(f"Invalid phase {phase}, must be 1-4")
        self._training_phase = phase

    def compute_combined_intrinsic_reward(
        self,
        batch: Dict[JEPAModality, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        weights: Optional[Dict[JEPAModality, float]] = None,
    ) -> torch.Tensor:
        if weights is None:
            weights = {m: 1.0 / len(self.layers) for m in self.layers}

        total_reward: Optional[torch.Tensor] = None
        for modality, (h_online, h_target, actions) in batch.items():
            layer = self.layers[modality]
            reward = layer.compute_intrinsic_reward(h_online, h_target, actions)
            weighted = reward * weights.get(modality, 0.2)
            total_reward = weighted if total_reward is None else total_reward + weighted

        if total_reward is None:
            return torch.tensor(0.0)
        return total_reward

    def augment_with_code_intelligence(
        self,
        hidden_states: torch.Tensor,
        code_context_embedding: List[float],
        modality: JEPAModality,
    ) -> torch.Tensor:
        """
        Augment LLM hidden states with code intelligence embeddings.

        Code Intelligence Integration:
          code_intelligence.py → CodeIntelligenceManager.to_embedding()
            → code_context_embedding (syntax + semantic features)
            → concatenated with LLM hidden_states
            → projected back to llm_hidden_dim via augmentation layer

        This enables PHYS-JEPA and DESIGN-JEPA to leverage structured code
        knowledge (call graphs, domain concepts, impact analysis) as additional
        context for more accurate representation predictions.

        Args:
            hidden_states: LLM hidden states (B, llm_hidden_dim)
            code_context_embedding: Code intelligence embedding vector
            modality: Target JEPA modality layer

        Returns:
            Augmented hidden states (B, llm_hidden_dim)
        """
        if not code_context_embedding:
            return hidden_states

        code_tensor = torch.tensor(
            [code_context_embedding], dtype=hidden_states.dtype, device=hidden_states.device
        )

        if code_tensor.shape[-1] != hidden_states.shape[-1]:
            min_dim = min(code_tensor.shape[-1], hidden_states.shape[-1])
            padded = torch.zeros(
                1, hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device
            )
            padded[0, :min_dim] = code_tensor[0, :min_dim]
            code_tensor = padded

        alpha = 0.15
        return hidden_states * (1 - alpha) + code_tensor * alpha
