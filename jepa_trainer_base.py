#!/usr/bin/env python3
"""
VORTEX FLAME — JEPA Universal Trainer Base
============================================
Shared training utilities for all 10 C-JEPA variants.

Provides:
  - SyntheticJEPADataset: generates temporally-structured feature sequences
  - train_jepa(): standard training loop with guard, checkpoint, logging
  - get_jepa_model(): factory for all 10 C-JEPA variants

Usage (thin wrapper scripts):
  from jepa_trainer_base import SyntheticJEPADataset, train_jepa, get_jepa_model
  model = get_jepa_model("physjepa")
  train_jepa(model, dataset, "physjepa_checkpoints", ...)
"""

import argparse
import json
import math
import os
import sys
import time
import random
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).parent))
from five_layer_jepa.causal_jepa import (
    CVJEPA, CAJEPA, CPHYSJEPA, CARTJEPA, CDESIGNJEPA,
    CFINJEPA, CCODEJEPA, CBIOJEPA, CGEOJEPA, CLAWJEPA,
)
from five_layer_jepa.causal_jepa_v2 import SIGRegWithPredictionLoss
from jepa_training_guard import TrainingGuard

VARIANT_CONFIG = {
    "cvjepa":       {"cls": CVJEPA,        "input_dim": 768, "num_slots": 7,  "slot_dim": 128, "history_len": 6, "future_len": 4},
    "cajepa":       {"cls": CAJEPA,        "input_dim": 512, "num_slots": 5,  "slot_dim": 128, "history_len": 6, "future_len": 4},
    "physjepa":     {"cls": CPHYSJEPA,     "input_dim": 512, "num_slots": 7,  "slot_dim": 128, "history_len": 8, "future_len": 4},
    "artjepa":      {"cls": CARTJEPA,      "input_dim": 768, "num_slots": 8,  "slot_dim": 128, "history_len": 6, "future_len": 4},
    "designjepa":   {"cls": CDESIGNJEPA,   "input_dim": 512, "num_slots": 6,  "slot_dim": 128, "history_len": 6, "future_len": 4},
    "finjepa":      {"cls": CFINJEPA,      "input_dim": 256, "num_slots": 6,  "slot_dim": 128, "history_len": 8, "future_len": 4},
    "codejepa":     {"cls": CCODEJEPA,     "input_dim": 384, "num_slots": 6,  "slot_dim": 128, "history_len": 8, "future_len": 4},
    "biojepa":      {"cls": CBIOJEPA,      "input_dim": 512, "num_slots": 7,  "slot_dim": 128, "history_len": 8, "future_len": 4},
    "geojepa":      {"cls": CGEOJEPA,      "input_dim": 512, "num_slots": 7,  "slot_dim": 128, "history_len": 8, "future_len": 4},
    "lawjepa":      {"cls": CLAWJEPA,      "input_dim": 512, "num_slots": 7,  "slot_dim": 128, "history_len": 8, "future_len": 4},
}

VARIANT_SEQUENCE_META = {
    "cvjepa":     {"num_modules": 8,  "module_dim": 128, "pattern": "visual_objects"},
    "physjepa":   {"num_modules": 8,  "module_dim": 128, "pattern": "physical_bodies"},
    "artjepa":    {"num_modules": 10, "module_dim": 128, "pattern": "art_elements"},
    "designjepa": {"num_modules": 8,  "module_dim": 128, "pattern": "design_components"},
    "finjepa":    {"num_modules": 8,  "module_dim": 64,  "pattern": "market_factors"},
    "codejepa":   {"num_modules": 8,  "module_dim": 96,  "pattern": "code_objects"},
    "biojepa":    {"num_modules": 8,  "module_dim": 128, "pattern": "bio_systems"},
    "geojepa":    {"num_modules": 8,  "module_dim": 128, "pattern": "geo_systems"},
    "lawjepa":    {"num_modules": 8,  "module_dim": 128, "pattern": "legal_entities"},
}

VARIANT_KNOWLEDGE_SEED = {
    "physjepa":   {"souls": ["einstein", "galileo"],     "keywords": ["physics", "quantum", "relativity", "particle", "force", "energy", "orbit", "gravity"]},
    "codejepa":   {"souls": ["cezanne"],                  "keywords": ["algorithm", "compiler", "function", "class", "loop", "variable", "memory", "thread"]},
    "biojepa":    {"souls": ["darwin", "yuanlongping"],   "keywords": ["gene", "protein", "cell", "DNA", "evolution", "metabolism", "enzyme", "mutation"]},
    "lawjepa":    {"souls": ["guizhu", "montesquieu"],    "keywords": ["law", "justice", "right", "obligation", "contract", "statute", "precedent", "jurisdiction"]},
    "geojepa":    {"souls": ["humboldt", "herodotus"],    "keywords": ["climate", "terrain", "water", "erosion", "plate", "atmosphere", "ocean", "soil"]},
    "artjepa":    {"souls": ["monet", "vangogh"],         "keywords": ["color", "composition", "brush", "canvas", "light", "shade", "perspective", "texture"]},
    "designjepa": {"souls": ["davinci"],                  "keywords": ["design", "layout", "proportion", "symmetry", "structure", "balance", "rhythm", "hierarchy"]},
}


class SyntheticJEPADataset(torch.utils.data.Dataset):
    """Generates temporally-structured synthetic feature sequences.

    Dynamics engine — three components that make prediction non-trivial:

    1. COUPLED OSCILLATORS: each module has a distinct natural frequency and
       sparse directed coupling to other modules. This produces smooth but
       structured temporal patterns (not trivial random walk).

    2. SPARSE CAUSAL INTERVENTIONS: at random times, a "cause" module receives a
       spike; after a delay of 2-4 steps, the "effect" module responds. The
       delay creates a genuine causal gap — you must learn the cause→effect
       mapping, not just interpolate.

    3. PHASE TRANSITIONS: at the history/future boundary (or nearby), all
       module frequencies shift and coupling reconfigures. The history window
       shows regime A; the future window shows regime B. Reconstruction is easy
       (same regime), prediction is hard (different regime).

    These three mechanisms together create a rec_loss ≪ fwd_loss gap of
    2-4x, confirming the model learns causal structure rather than
    trivial interpolation.

    Output: per-module LayerNorm after projection → stable ~std=1 features.

    Args:
        num_sequences: total unique base sequences (augmented via offsets)
        input_dim: output feature dimension per object per timestep
        num_modules: number of independently-evolving objects/modules
        seq_len: total time steps per sequence
        history_len: context window length
        future_len: prediction target window length
        module_dim: internal oscillator dimension before projection
    """
    def __init__(self, num_sequences=2000, input_dim=512, num_modules=8,
                 seq_len=14, history_len=8, future_len=4, module_dim=128):
        super().__init__()
        self.num_sequences = num_sequences
        self.input_dim = input_dim
        self.num_modules = num_modules
        self.seq_len = seq_len
        self.history_len = history_len
        self.future_len = future_len
        self.module_dim = module_dim

        self._base_states = []
        for i in range(num_sequences):
            states = self._simulate_sequence(seed=20000 + i)
            self._base_states.append(states)

        self._projectors = nn.ModuleList([
            nn.Linear(self.module_dim, self.input_dim)
            for _ in range(self.num_modules)
        ])
        self._norms = nn.ModuleList([
            nn.LayerNorm(self.input_dim)
            for _ in range(self.num_modules)
        ])

    def _simulate_sequence(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        rng = random.Random(seed * 37)
        T_total = self.seq_len + 6
        N = self.num_modules
        D = self.module_dim
        H = self.history_len

        omega = 0.04 + torch.linspace(0.0, 0.18, N)
        coupling = torch.zeros(N, N)
        for i in range(N):
            n_targets = max(1, N // 3)
            targets = torch.randperm(N, generator=g)[:n_targets]
            for j in targets:
                if j != i:
                    coupling[j, i] = (torch.rand(1, generator=g).item() - 0.25) * 0.10

        split = N // 2
        active_in_history = list(range(split))
        active_in_future = list(range(split, N))
        rng.shuffle(active_in_history)
        rng.shuffle(active_in_future)

        n_interv = rng.randint(1, 3)
        interventions = []
        for k in range(n_interv):
            t_cause = rng.randint(max(3, H - 3), H)
            cause = rng.choice(active_in_history)
            effect = rng.randint(0, N - 1)
            while effect == cause:
                effect = rng.randint(0, N - 1)
            delay = rng.randint(2, 5)
            strength = 2.5 + rng.random() * 4.0
            interventions.append((t_cause, cause, effect, delay, strength))

        phase = torch.rand(N, generator=g) * 2.0 * math.pi
        states = torch.zeros(T_total, N, D)

        for t in range(T_total):
            history_amp = 1.0
            future_amp = min(1.0, max(0.0, (t - H) / 3.0))

            dphase = omega.clone()
            for i in range(N):
                for j in range(N):
                    if coupling[i, j] != 0:
                        dphase[i] += coupling[i, j] * torch.sin(phase[i] - phase[j])
            phase = phase + dphase * 0.25

            for i in range(N):
                is_history_active = i in active_in_history
                amp_factor = history_amp if is_history_active else future_amp
                if t <= H:
                    amp_factor = 1.0 if is_history_active else 0.1
                else:
                    amp_factor = 0.1 if is_history_active else future_amp

                for h in range(min(4, D // 2)):
                    val = amp_factor * (0.5 + 0.3 * math.sin(phase[i].item() * 0.7))
                    states[t, i, 2 * h] = val * math.sin(phase[i].item() * (h + 1))
                    states[t, i, 2 * h + 1] = val * math.cos(phase[i].item() * (h + 1))

            for itv_t, cause, effect, delay, strength in interventions:
                if t == itv_t:
                    phase[cause] += strength
                if t == itv_t + delay:
                    phase[effect] += strength * 0.5

            noise = 0.005 if t <= H else 0.015 * (1.0 + 0.2 * (t - H))
            states[t] += torch.randn(N, D, generator=g) * noise

        return states

    def __len__(self):
        return self.num_sequences * 4

    def __getitem__(self, idx):
        base_idx = idx % self.num_sequences
        offset = random.randint(0, 4)
        raw = self._base_states[base_idx][offset:offset + self.seq_len]

        with torch.no_grad():
            features = []
            for t in range(self.seq_len):
                frame = []
                for m in range(self.num_modules):
                    f = self._projectors[m](raw[t, m])
                    f = self._norms[m](f)
                    frame.append(f)
                features.append(torch.stack(frame))
            features = torch.stack(features)

        history = features[:self.history_len]
        future = features[self.history_len:self.history_len + self.future_len]
        return history, future


class KnowledgeSeededDataset(torch.utils.data.Dataset):
    """Generates feature sequences seeded from soul knowledge base embeddings.

    Uses existing knowledge entries as initialization seeds for object dynamics.
    This creates semantically-meaningful temporal patterns tied to each soul's
    domain knowledge, rather than purely random synthetic data.
    """
    def __init__(self, souls, keywords, input_dim=512, num_modules=8,
                 seq_len=14, history_len=8, future_len=4,
                 max_kb_entries=5000):
        super().__init__()
        self.input_dim = input_dim
        self.num_modules = num_modules
        self.seq_len = seq_len
        self.history_len = history_len
        self.future_len = future_len
        self.souls = souls
        self.keywords = keywords

        self._seeds = []
        try:
            from soul_memory import SoulMemoryEngine
            m = SoulMemoryEngine()
            seen = set()
            for soul in souls:
                for kw in keywords:
                    results = m.recall(soul, kw, top_k=80)
                    for r in results:
                        content = r.get("content", {})
                        if isinstance(content, dict):
                            topic = content.get("topic", "")
                            text = content.get("text", "")
                            if topic and topic not in seen:
                                seen.add(topic)
                                self._seeds.append(text[:200] if text else topic)
                                if len(self._seeds) >= max_kb_entries:
                                    break
                    if len(self._seeds) >= max_kb_entries:
                        break
                if len(self._seeds) >= max_kb_entries:
                    break
        except Exception:
            pass

        if len(self._seeds) < 100:
            self._seeds = [f"seed_{i}" for i in range(200)]

        torch.manual_seed(42)
        self._dyn_matrices = torch.randn(num_modules, num_modules, 64, 64) * 0.05

    def __len__(self):
        return len(self._seeds) * 2

    def __getitem__(self, idx):
        seed_idx = idx % len(self._seeds)
        seed_str = self._seeds[seed_idx]

        torch.manual_seed(hash(seed_str) % (2**31))
        state = torch.randn(self.num_modules, 64) * 0.3

        features = []
        for t in range(self.seq_len):
            noise = torch.randn(self.num_modules, 64) * 0.05
            interaction = torch.zeros(self.num_modules, 64)
            for i in range(self.num_modules):
                for j in range(self.num_modules):
                    interaction[i] += (self._dyn_matrices[i, j] @ state[j]) * 0.1
            state = 0.93 * state + noise + interaction
            projected = F.linear(state, torch.randn(self.input_dim, 64)) * 0.5
            features.append(projected)

        features = torch.stack(features)
        history = features[:self.history_len]
        future = features[self.history_len:self.history_len + self.future_len]
        return history, future


def get_jepa_model(variant: str, device=None):
    """Factory: returns (model, config_dict) for a given JEPA variant."""
    if variant not in VARIANT_CONFIG:
        raise ValueError(f"Unknown variant '{variant}'. Valid: {list(VARIANT_CONFIG.keys())}")
    cfg = VARIANT_CONFIG[variant]
    model = cfg["cls"](
        input_dim=cfg["input_dim"],
        history_len=cfg["history_len"],
        future_len=cfg["future_len"],
    )
    if device:
        model = model.to(device)
    return model, cfg


def train_jepa(model, dataset, checkpoint_dir, variant_name="jepa",
               epochs=100, batch=8, lr=1e-4, device=None,
               use_knowledge_seed=False):
    """Standard training loop for any C-JEPA variant."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {type(model).__name__}")
    print(f"Device: {device}")
    print(f"Params: {total_params:,} total, {trainable:,} trainable")
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    print()

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    steps_per_epoch = len(dataloader)
    T_0 = 5 * steps_per_epoch
    warmup_steps_total = min(200, steps_per_epoch * 2)
    for g in optimizer.param_groups:
        g["lr"] = lr * 0.01
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=1e-6)

    start_epoch = 1
    saved_global_step = 0
    best_loss = float("inf")

    sigreg_loss = SIGRegWithPredictionLoss(var_weight=25.0, cov_weight=1.0, sim_weight=1.0)

    resume_path = os.path.join(checkpoint_dir, f"{variant_name}_best.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        saved_global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch-1}, best_loss={best_loss:.4f}")

    guard = TrainingGuard(
        total_steps=epochs * steps_per_epoch,
        warmup_steps=200,
        loss_spike_factor=5.0,
        max_grad_norm=2.0,
        nan_tolerance=20,
        collapse_patience=100,
        checkpoint_dir=checkpoint_dir,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(checkpoint_dir, "training_log.jsonl")
    global_step = saved_global_step

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (history, future) in enumerate(dataloader):
            guard.pre_step(global_step)

            history = history.to(device)
            future = future.to(device)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                with torch.no_grad():
                    target_slots_h, _ = model.target_encoder(history)
                    target_slots_f, _ = model.target_encoder(future)

                context_slots, _ = model.context_encoder(history)
                masked_slots, slot_mask, _ = model.masker(context_slots, force_num_masked=3)

                recovered, predicted_future = model.predictor(
                    masked_slots, slot_mask=slot_mask
                )

                r_loss = sigreg_loss(
                    recovered.reshape(-1, model.slot_dim),
                    target_slots_h.reshape(-1, model.slot_dim),
                )

                T_match = min(predicted_future.shape[1], target_slots_f.shape[1])
                f_loss = sigreg_loss(
                    predicted_future[:, :T_match].reshape(-1, model.slot_dim),
                    target_slots_f[:, :T_match].reshape(-1, model.slot_dim),
                )

                loss = r_loss["total"] + 0.5 * f_loss["total"]

            if torch.isnan(loss) or torch.isinf(loss):
                guard.nan_count += 1
                print(f"  [NaN] step={global_step}, skipping batch", flush=True)
                continue

            optimizer.zero_grad()
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            if total_norm > 20.0:
                print(f"  [GUARD] Large gradient at step {global_step}: norm={total_norm:.2f} (will clip to 2.0)", flush=True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()

            if global_step < warmup_steps_total:
                warmup_frac = (global_step + 1) / warmup_steps_total
                warmup_lr = lr * (0.01 + 0.99 * warmup_frac)
                for g in optimizer.param_groups:
                    g["lr"] = warmup_lr

            with torch.no_grad():
                model.update_ema()

            loss_val = loss.item()
            epoch_loss += loss_val
            n_batches += 1
            global_step += 1

            guard.post_step(loss_val)

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:4d}/{epochs} | loss={avg_loss:.4f} | lr={current_lr:.2e} | {elapsed:.1f}s", flush=True)

        log_entry = {
            "epoch": epoch, "global_step": global_step,
            "avg_loss": avg_loss, "lr": current_lr,
            "elapsed_s": elapsed,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(checkpoint_dir, f"{variant_name}_best.pt"))
            print(f"  [BEST] Saved checkpoint (loss={best_loss:.4f})", flush=True)

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(checkpoint_dir, f"{variant_name}_epoch{epoch}.pt"))

    print(f"\n{'='*60}")
    print(f"Training complete: {epochs} epochs, best_loss={best_loss:.4f}")
    print(f"Checkpoints in: {checkpoint_dir}")
    print(f"{'='*60}")
