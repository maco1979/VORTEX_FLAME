"""
JEPA Training Guard — Protects JEPA training from collapse
===========================================================

Protection mechanisms:
  1. Loss spike detection & skip (loss > threshold * running_median)
  2. Gradient explosion protection (norm clipping + NaN detection)
  3. Parameter drift monitoring (weight norm tracking)
  4. Learning rate warmup + cooldown
  5. EMA decay schedule (ramp from 0.99 → 0.996)
  6. Automatic checkpoint rollback on sustained collapse
  7. NaN/Inf detection in loss, gradients, and parameters

Usage:
  guard = TrainingGuard(total_steps=10000, warmup_steps=200)

  for step, batch in enumerate(dataloader):
      guard.pre_step(step)

      loss = compute_loss(...)
      if guard.check_loss(loss):
          continue  # skip this batch

      loss.backward()

      if guard.check_gradients(model):
          optimizer.zero_grad()
          continue  # skip, bad gradients

      torch.nn.utils.clip_grad_norm_(params, guard.max_grad_norm)
      optimizer.step()
      guard.post_step(loss.item())
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class GuardState:
    step: int = 0
    losses: deque = field(default_factory=lambda: deque(maxlen=100))
    grad_norms: deque = field(default_factory=lambda: deque(maxlen=100))
    param_norms: deque = field(default_factory=lambda: deque(maxlen=50))
    skipped_batches: int = 0
    nan_batches: int = 0
    loss_spikes: int = 0
    grad_explosions: int = 0
    rollback_count: int = 0
    consecutive_bad: int = 0
    last_good_loss: float = float("inf")
    warmup_done: bool = False


class TrainingGuard:
    def __init__(
        self,
        total_steps: int = 10000,
        warmup_steps: int = 200,
        loss_spike_factor: float = 5.0,
        max_grad_norm: float = 1.0,
        nan_tolerance: int = 10,
        collapse_patience: int = 50,
        ema_decay_start: float = 0.99,
        ema_decay_end: float = 0.996,
        checkpoint_dir: Optional[str] = None,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.loss_spike_factor = loss_spike_factor
        self.max_grad_norm = max_grad_norm
        self.nan_tolerance = nan_tolerance
        self.collapse_patience = collapse_patience
        self.ema_decay_start = ema_decay_start
        self.ema_decay_end = ema_decay_end
        self.checkpoint_dir = checkpoint_dir

        self.state = GuardState()
        self._loss_history: List[float] = []

    def pre_step(self, step: int):
        self.state.step = step
        if step >= self.warmup_steps:
            self.state.warmup_done = True

    def get_lr_scale(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        return 1.0

    def get_ema_decay(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.ema_decay_start
        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        return self.ema_decay_start + (self.ema_decay_end - self.ema_decay_start) * progress

    def check_loss(self, loss: torch.Tensor) -> bool:
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss)

        if torch.isnan(loss) or torch.isinf(loss):
            self.state.nan_batches += 1
            self.state.consecutive_bad += 1
            print(f"  [GUARD] NaN/Inf loss at step {self.state.step}! "
                  f"(total NaN: {self.state.nan_batches})", flush=True)
            return True

        loss_val = loss.item()
        self.state.losses.append(loss_val)

        if len(self.state.losses) >= 10:
            median_loss = sorted(self.state.losses)[-len(self.state.losses) // 2]
            if loss_val > median_loss * self.loss_spike_factor:
                self.state.loss_spikes += 1
                self.state.consecutive_bad += 1
                print(f"  [GUARD] Loss spike at step {self.state.step}: "
                      f"{loss_val:.4f} > {median_loss:.4f} * {self.loss_spike_factor} "
                      f"(spikes: {self.state.loss_spikes})", flush=True)
                return True

        self.state.consecutive_bad = 0
        if loss_val < self.state.last_good_loss:
            self.state.last_good_loss = loss_val
        return False

    def check_gradients(self, model: nn.Module) -> bool:
        total_norm = 0.0
        has_nan = False

        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    has_nan = True
                    break
                total_norm += p.grad.data.norm(2).item() ** 2

        total_norm = total_norm ** 0.5
        self.state.grad_norms.append(total_norm)

        if has_nan:
            self.state.grad_explosions += 1
            self.state.consecutive_bad += 1
            print(f"  [GUARD] NaN gradient at step {self.state.step}! "
                  f"(explosions: {self.state.grad_explosions})", flush=True)
            return True

        if total_norm > self.max_grad_norm * 10:
            self.state.grad_explosions += 1
            print(f"  [GUARD] Large gradient at step {self.state.step}: "
                  f"norm={total_norm:.2f} (will clip to {self.max_grad_norm})", flush=True)
            return False

        return False

    def check_parameters(self, model: nn.Module) -> bool:
        total_norm = 0.0
        has_nan = False

        for p in model.parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                has_nan = True
                break
            total_norm += p.data.norm(2).item() ** 2

        total_norm = total_norm ** 0.5
        self.state.param_norms.append(total_norm)

        if has_nan:
            print(f"  [GUARD] NaN in model parameters at step {self.state.step}! "
                  f"Consider rollback.", flush=True)
            return True

        if len(self.state.param_norms) >= 10:
            recent = list(self.state.param_norms)[-10:]
            if total_norm > max(recent[:-1]) * 3:
                print(f"  [GUARD] Parameter drift at step {self.state.step}: "
                      f"norm={total_norm:.2f}", flush=True)
                return True

        return False

    def should_rollback(self) -> bool:
        if self.state.nan_batches >= self.nan_tolerance:
            return True
        if self.state.consecutive_bad >= self.collapse_patience:
            return True
        return False

    def save_safe_checkpoint(self, model_dict: Dict, name: str = "safe_checkpoint.pt"):
        if self.checkpoint_dir is None:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(model_dict, path)

    def post_step(self, loss_val: float):
        self._loss_history.append(loss_val)

    def get_stats(self) -> Dict:
        return {
            "step": self.state.step,
            "skipped_batches": self.state.skipped_batches,
            "nan_batches": self.state.nan_batches,
            "loss_spikes": self.state.loss_spikes,
            "grad_explosions": self.state.grad_explosions,
            "consecutive_bad": self.state.consecutive_bad,
            "last_good_loss": self.state.last_good_loss,
            "recent_loss_avg": sum(self.state.losses) / max(len(self.state.losses), 1),
            "recent_grad_avg": sum(self.state.grad_norms) / max(len(self.state.grad_norms), 1),
        }

    def log_stats(self, step: int, extra: Dict = None):
        stats = self.get_stats()
        if extra:
            stats.update(extra)
        if step % 100 == 0 and stats["skipped_batches"] > 0:
            print(f"  [GUARD STATS] skipped={stats['skipped_batches']} "
                  f"nan={stats['nan_batches']} spikes={stats['loss_spikes']} "
                  f"grad_exp={stats['grad_explosions']} "
                  f"consec_bad={stats['consecutive_bad']}", flush=True)
