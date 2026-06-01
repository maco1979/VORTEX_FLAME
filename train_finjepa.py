#!/usr/bin/env python3
"""
FIN-JEPA (Financial JEPA) Training Script for VORTEX_FLAME
===========================================================

Trains C-FINJEPA on financial time-series data (K-line OHLCV).

Architecture:
  OHLCV bars → FinancialFeatureProjector → C-FINJEPA slots
  → object-level masking → causal prediction

Financial "objects" in slots:
  slot 0: trend (趋势方向)
  slot 1: momentum (动量强度)
  slot 2: volatility (波动率)
  slot 3: volume_profile (量能形态)
  slot 4: support_resistance (支撑阻力)
  slot 5: regime (市场状态: bull/bear/range)

Key differences from A-JEPA:
  - Input: OHLCV bars (5 features) instead of mel spectrograms
  - Auxiliary: macro indicators (interest rate, VIX, etc.)
  - Prediction: next N bars' slot representation
  - Counterfactual: "what if interest rate changed?"

Usage:
  python train_finjepa.py --epochs 100 --batch 32 --lr 1e-4
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))
from five_layer_jepa.causal_jepa import CFINJEPA
from jepa_training_guard import TrainingGuard

DATA_DIR = r"D:\VORTEX_FLAME\fin_jepa_data"
CHECKPOINT_DIR = r"D:\VORTEX_FLAME\finjepa_checkpoints"

BAR_FEATURES = 5
BARS_PER_SEGMENT = 20
HISTORY_SEGMENTS = 8
FUTURE_SEGMENTS = 4
MIN_SEGMENTS = HISTORY_SEGMENTS + FUTURE_SEGMENTS


class FinancialFeatureProjector(nn.Module):
    def __init__(self, bar_features=5, bars_per_segment=20, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(bar_features, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1)
        dummy = torch.zeros(1, bar_features, bars_per_segment)
        with torch.no_grad():
            x = F.gelu(self.conv1(dummy))
            x = F.gelu(self.conv2(x))
            x = F.gelu(self.conv3(x))
        flat_dim = x.reshape(1, -1).shape[1]
        self.proj = nn.Linear(flat_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, bar_segment):
        if bar_segment.dim() == 2:
            bar_segment = bar_segment.unsqueeze(0)
        x = F.gelu(self.conv1(bar_segment))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = self.norm(self.proj(x))
        return x


class FinJEPADataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, history_len=8, future_len=4):
        self.history_len = history_len
        self.future_len = future_len
        self.min_segments = history_len + future_len

        self.symbol_data = []
        manifest_path = os.path.join(data_dir, "manifest.json")

        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            print(f"Loading {manifest['total_symbols']} symbols from manifest...", flush=True)

            for sym_info in manifest["symbols"]:
                fpath = os.path.join(data_dir, sym_info["file"])
                try:
                    data = torch.load(fpath, map_location="cpu", weights_only=False)
                    ohlcv = data["data"]
                    n_bars = ohlcv.shape[0]
                    n_segments = n_bars // BARS_PER_SEGMENT
                    if n_segments >= self.min_segments:
                        self.symbol_data.append({
                            "ohlcv": ohlcv,
                            "code": data.get("code", sym_info["code"]),
                            "name": data.get("name", sym_info["name"]),
                            "market": data.get("market", sym_info["market"]),
                            "n_segments": n_segments,
                        })
                except Exception as e:
                    print(f"  [WARN] Cannot load {fpath}: {e}", flush=True)
        else:
            print(f"No manifest found at {manifest_path}, scanning directory...", flush=True)
            if os.path.exists(data_dir):
                for fname in os.listdir(data_dir):
                    if not fname.endswith(".pt"):
                        continue
                    fpath = os.path.join(data_dir, fname)
                    try:
                        data = torch.load(fpath, map_location="cpu", weights_only=False)
                        ohlcv = data["data"]
                        n_bars = ohlcv.shape[0]
                        n_segments = n_bars // BARS_PER_SEGMENT
                        if n_segments >= self.min_segments:
                            self.symbol_data.append({
                                "ohlcv": ohlcv,
                                "code": data.get("code", "?"),
                                "name": data.get("name", "?"),
                                "market": data.get("market", "?"),
                                "n_segments": n_segments,
                            })
                    except Exception:
                        pass

        total_bars = sum(s["ohlcv"].shape[0] for s in self.symbol_data)
        markets = {}
        for s in self.symbol_data:
            m = s["market"]
            markets[m] = markets.get(m, 0) + 1
        market_str = ", ".join(f"{k}:{v}" for k, v in markets.items())
        print(f"Dataset: {len(self.symbol_data)} symbols, {total_bars} bars ({market_str})", flush=True)

    def __len__(self):
        return len(self.symbol_data) * 16

    def _segment_ohlcv(self, ohlcv):
        n_bars = ohlcv.shape[0]
        n_complete = (n_bars // BARS_PER_SEGMENT) * BARS_PER_SEGMENT
        ohlcv = ohlcv[:n_complete]
        n_seg = n_complete // BARS_PER_SEGMENT
        segments = ohlcv.reshape(n_seg, BARS_PER_SEGMENT, BAR_FEATURES)
        segments = segments.permute(0, 2, 1)
        return segments

    def __getitem__(self, idx):
        sym_idx = idx % len(self.symbol_data)
        info = self.symbol_data[sym_idx]

        segments = self._segment_ohlcv(info["ohlcv"])
        actual_n = segments.shape[0]
        max_offset = max(0, actual_n - self.min_segments)
        offset = random.randint(0, max_offset) if max_offset > 0 else 0

        history = segments[offset:offset + self.history_len]
        future = segments[offset + self.history_len:offset + self.history_len + self.future_len]

        if history.shape[0] < self.history_len:
            pad = torch.zeros(self.history_len - history.shape[0], BAR_FEATURES, BARS_PER_SEGMENT)
            history = torch.cat([history, pad], dim=0)
        if future.shape[0] < self.future_len:
            pad = torch.zeros(self.future_len - future.shape[0], BAR_FEATURES, BARS_PER_SEGMENT)
            future = torch.cat([future, pad], dim=0)

        return history, future


def train(args):
    dataset = FinJEPADataset(DATA_DIR, HISTORY_SEGMENTS, FUTURE_SEGMENTS)
    if len(dataset.symbol_data) == 0:
        print("ERROR: No financial data found! Run collect_fin_data.py first.")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    projector = FinancialFeatureProjector(
        bar_features=BAR_FEATURES, bars_per_segment=BARS_PER_SEGMENT, output_dim=256
    ).to(device)
    finjepa = CFINJEPA(input_dim=256, history_len=HISTORY_SEGMENTS, future_len=FUTURE_SEGMENTS).to(device)

    all_params = list(projector.parameters()) + list(finjepa.parameters())
    total_params = sum(p.numel() for p in all_params)
    trainable = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"Params: {total_params:,} total, {trainable:,} trainable")

    optimizer = Adam(all_params, lr=args.lr, weight_decay=1e-5)
    start_epoch = 1
    best_loss = float("inf")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    resume_path = os.path.join(CHECKPOINT_DIR, "finjepa_best.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        projector.load_state_dict(ckpt["projector_state"])
        finjepa.load_state_dict(ckpt["finjepa_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        best_loss = ckpt.get("best_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch-1}, best_loss={best_loss:.4f}")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))

    guard = TrainingGuard(
        total_steps=args.epochs * len(dataloader),
        warmup_steps=500,
        loss_spike_factor=5.0,
        max_grad_norm=5.0,
        nan_tolerance=20,
        collapse_patience=50,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    log_path = os.path.join(CHECKPOINT_DIR, "training_log.jsonl")
    global_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        finjepa.train()
        projector.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (history_bars, future_bars) in enumerate(dataloader):
            guard.pre_step(global_step)

            history_bars = history_bars.to(device)
            future_bars = future_bars.to(device)

            B, T_hist, F_dim, T_bars = history_bars.shape
            _, T_fut, _, _ = future_bars.shape

            history_flat = history_bars.reshape(B * T_hist, F_dim, T_bars)
            history_features = projector(history_flat)
            history_features = history_features.reshape(B, T_hist, -1)
            history_features = history_features.unsqueeze(2).expand(-1, -1, 6, -1)

            future_flat = future_bars.reshape(B * T_fut, F_dim, T_bars)
            future_features = projector(future_flat)
            future_features = future_features.reshape(B, T_fut, -1)
            future_features = future_features.unsqueeze(2).expand(-1, -1, 6, -1)

            with torch.no_grad():
                target_slots_history, _ = finjepa.target_encoder(history_features)
                target_slots_future, _ = finjepa.target_encoder(future_features)

            context_slots_history, _ = finjepa.context_encoder(history_features)
            masked_slots, slot_mask, _ = finjepa.masker(context_slots_history)
            recovered_slots, predicted_future = finjepa.predictor(masked_slots)

            recovery_loss = finjepa.loss_fn(
                recovered_slots.reshape(-1, finjepa.slot_dim),
                target_slots_history.reshape(-1, finjepa.slot_dim),
            )
            T_pred = predicted_future.shape[1]
            T_target = target_slots_future.shape[1]
            T_match = min(T_pred, T_target)
            forward_loss = finjepa.loss_fn(
                predicted_future[:, :T_match].reshape(-1, finjepa.slot_dim),
                target_slots_future[:, :T_match].reshape(-1, finjepa.slot_dim),
            )
            total_loss = recovery_loss["total"] + 0.5 * forward_loss["total"]

            if guard.check_loss(total_loss):
                guard.state.skipped_batches += 1
                global_step += 1
                continue

            optimizer.zero_grad()
            total_loss.backward()

            if guard.check_gradients(finjepa) or guard.check_gradients(projector):
                optimizer.zero_grad()
                guard.state.skipped_batches += 1
                global_step += 1
                continue

            torch.nn.utils.clip_grad_norm_(all_params, guard.max_grad_norm)

            lr_scale = guard.get_lr_scale(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale

            optimizer.step()
            scheduler.step()

            ema_decay = guard.get_ema_decay(global_step)
            finjepa.ema_decay = ema_decay
            finjepa.update_ema()

            guard.post_step(total_loss.item())
            guard.log_stats(global_step)

            epoch_loss += total_loss.item()
            n_batches += 1
            global_step += 1

            if batch_idx % 20 == 0:
                vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                elapsed = time.time() - t0
                print(f"  E{epoch} B{batch_idx}/{len(dataloader)} "
                      f"loss={total_loss.item():.4f} "
                      f"rec={recovery_loss['total'].item():.4f} "
                      f"fwd={forward_loss['total'].item():.4f} "
                      f"VRAM={vram:.1f}GB "
                      f"{elapsed:.0f}s", flush=True)

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"\n=== Epoch {epoch}/{args.epochs} avg_loss={avg_loss:.4f} ({elapsed:.0f}s) ===", flush=True)

        log_entry = {
            "epoch": epoch, "global_step": global_step, "avg_loss": avg_loss,
            "lr": scheduler.get_last_lr()[0], "elapsed_s": elapsed,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "finjepa_state": finjepa.state_dict(),
                "projector_state": projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(CHECKPOINT_DIR, "finjepa_best.pt"))
            print(f"  NEW BEST: {best_loss:.4f}")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "finjepa_state": finjepa.state_dict(),
                "projector_state": projector.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(CHECKPOINT_DIR, f"finjepa_epoch{epoch}.pt"))

    torch.save({
        "projector_state": projector.state_dict(),
        "finjepa_state": finjepa.state_dict(),
        "config": {"bar_features": BAR_FEATURES, "bars_per_segment": BARS_PER_SEGMENT,
                    "num_slots": 6, "slot_dim": 128},
    }, os.path.join(CHECKPOINT_DIR, "finjepa_final.pt"))
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    train(parser.parse_args())
