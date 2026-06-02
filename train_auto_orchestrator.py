#!/usr/bin/env python3
"""
VORTEX FLAME — 自动化增量训练与回测编排器
============================================

阶段递增策略:
  每次增加 10% 背景音乐 + 全部 ESC-50 训练folds → 训练 → 回测 → 判定收敛
  音乐永不重复, 直至全部用尽或模型收敛。

收敛判定 (三重门):
  G1: Linear Probe 5-fold CV 连续 3 个阶段提升 < 1%
  G2: JEPA loss 连续 3 个阶段波动 < 5%
  G3: 音乐数据已全部用完

用法:
  python train_auto_orchestrator.py --epochs_per_stage 30 --batch 8 --lr 1e-4
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import random
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))

from sample_augment import build_sample_augment, batch_overlay_short_samples, short_sample_contrastive_pairs
from bozak_augment import build_audio_circuit_augment

from train_ajepa_multiclass import (
    AudioFeatureProjector,
    MulticlassAJEPADataset,
    supervised_contrastive_loss,
    evaluate_linear_probe,
    load_esc50_metadata,
    scan_audio_files,
    CAJEPA,
    SIGRegWithPredictionLoss,
    AUDIO_DIRS,
    CACHE_DIR,
    CHECKPOINT_DIR,
    MC_CHECKPOINT_DIR,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    SEGMENT_FRAMES,
    HISTORY_SEGMENTS,
    FUTURE_SEGMENTS,
    MIN_SEGMENTS,
    NUM_CLASSES,
    MUSIC_LABEL,
    MUSIC_RATIO,
    ESC50_TARGET_DURATION,
    ESC50_AUDIO_DIR,
    ESC50_META,
)

from jepa_training_guard import TrainingGuard
from five_layer_jepa.causal_jepa import CAJEPA as _CAJEPA
from five_layer_jepa.causal_jepa_v2 import SIGRegWithPredictionLoss as _SIGREG

STAGE_LOG_DIR = r"D:\VORTEX_FLAME\stage_logs"
STAGE_CHECKPOINT_DIR = r"D:\VORTEX_FLAME\stage_checkpoints"
STAGE_INCREMENT = 0.10
CONVERGENCE_ACC_DELTA = 0.01
CONVERGENCE_LOSS_DELTA = 0.05
CONVERGENCE_WINDOW = 3
BACKTEST_FOLDS = [1, 2, 3, 4, 5]


def _cache_key(filepath):
    return hashlib.md5(filepath.encode()).hexdigest()[:16]


class MusicFileTracker:
    def __init__(self, log_dir, force_reset=False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.log_dir / "music_tracker.json"

        if self.state_path.exists() and not force_reset:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.used_hashes = set(state.get("used_hashes", []))
            self.stage_history = state.get("stage_history", {})
            self.total_available = state.get("total_available", 0)
        else:
            self.used_hashes = set()
            self.stage_history = {}
            self.total_available = 0

    def scan_available_music(self, audio_files):
        available = []
        for fp in audio_files:
            key = _cache_key(fp)
            cache_path = os.path.join(CACHE_DIR, "music", f"{key}.pt")
            if not os.path.exists(cache_path):
                continue
            h = hashlib.md5(fp.encode()).hexdigest()
            if h not in self.used_hashes:
                available.append(fp)
        self.total_available = len(available) + len(self.used_hashes)
        return available

    def allocate_stage(self, audio_files, stage_idx, ratio=STAGE_INCREMENT):
        fresh = self.scan_available_music(audio_files)
        if not fresh:
            return [], 0.0

        n_allocate = max(1, int(len(audio_files) * ratio))
        shuffled = list(fresh)
        random.shuffle(shuffled)
        allocated = shuffled[:n_allocate]

        new_hashes = []
        for fp in allocated:
            h = hashlib.md5(fp.encode()).hexdigest()
            self.used_hashes.add(h)
            new_hashes.append(h)

        self.stage_history[str(stage_idx)] = {
            "count": len(allocated),
            "hashes": new_hashes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

        used_pct = len(self.used_hashes) / max(self.total_available, 1) * 100
        return allocated, used_pct

    def _save(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({
                "used_hashes": sorted(self.used_hashes),
                "stage_history": self.stage_history,
                "total_available": self.total_available,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2, ensure_ascii=False)


class StageResult:
    def __init__(self, stage_idx, music_count, music_used_pct):
        self.stage_idx = stage_idx
        self.music_count = music_count
        self.music_used_pct = music_used_pct
        self.epochs_run = 0
        self.best_jepa_loss = float("inf")
        self.best_cont_loss = float("inf")
        self.best_total_loss = float("inf")
        self.final_jepa_loss = 0.0
        self.final_cont_loss = 0.0
        self.final_total_loss = 0.0
        self.fold_accuracies = {}
        self.mean_accuracy = 0.0
        self.elapsed_s = 0.0
        self.status = "pending"
        self.error_msg = ""
        self.loss_history = []
        self.checkpoint_path = ""

    def to_dict(self):
        return {
            "stage_idx": self.stage_idx,
            "music_count": self.music_count,
            "music_used_pct": round(self.music_used_pct, 2),
            "epochs_run": self.epochs_run,
            "best_jepa_loss": round(self.best_jepa_loss, 6),
            "best_cont_loss": round(self.best_cont_loss, 6),
            "best_total_loss": round(self.best_total_loss, 6),
            "final_jepa_loss": round(self.final_jepa_loss, 6),
            "final_cont_loss": round(self.final_cont_loss, 6),
            "final_total_loss": round(self.final_total_loss, 6),
            "fold_accuracies": {str(k): round(v, 4) for k, v in self.fold_accuracies.items()},
            "mean_accuracy": round(self.mean_accuracy, 4),
            "elapsed_s": round(self.elapsed_s, 1),
            "status": self.status,
            "error_msg": self.error_msg,
            "checkpoint_path": self.checkpoint_path,
        }


class ConvergenceMonitor:
    def __init__(self, window=CONVERGENCE_WINDOW, acc_delta=CONVERGENCE_ACC_DELTA,
                 loss_delta=CONVERGENCE_LOSS_DELTA):
        self.window = window
        self.acc_delta = acc_delta
        self.loss_delta = loss_delta
        self.acc_history = []
        self.loss_history = []

    def update(self, mean_accuracy, final_total_loss):
        self.acc_history.append(mean_accuracy)
        self.loss_history.append(final_total_loss)

    def check_convergence(self, music_exhausted=False):
        if music_exhausted:
            return True, "MUSIC_EXHAUSTED"

        if len(self.acc_history) < self.window:
            return False, "waiting"

        recent_acc = self.acc_history[-self.window:]
        max_acc = max(recent_acc)
        min_acc = min(recent_acc)
        acc_range = max_acc - min_acc

        recent_loss = self.loss_history[-self.window:]
        avg_loss = sum(recent_loss) / len(recent_loss)
        loss_range = (max(recent_loss) - min(recent_loss)) / max(avg_loss, 1e-8)

        reasons = []

        g1 = acc_range < self.acc_delta
        if g1:
            reasons.append(f"acc_plateau({acc_range:.4f}<{self.acc_delta})")

        g2 = loss_range < self.loss_delta
        if g2:
            reasons.append(f"loss_stable({loss_range:.4f}<{self.loss_delta})")

        if g1 and g2:
            return True, "+".join(reasons)
        elif g1:
            return True, "+".join(reasons)
        elif g2 and acc_range < self.acc_delta * 2:
            return True, "soft:" + "+".join(reasons)

        return False, f"not_converged(acc_range={acc_range:.4f}, loss_range={loss_range:.4f})"


def run_stage_training(stage_music_files, esc50_entries, stage_idx, args, resume_ckpt=None, sample_bank=None, circuit_augment=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(args, "cuda", False) and torch.cuda.is_available():
        device = torch.device(f"cuda:{getattr(args, 'device', 0)}")

    validation_folds = {5}
    dataset = MulticlassAJEPADataset(
        stage_music_files, esc50_entries, HISTORY_SEGMENTS, FUTURE_SEGMENTS,
        esc50_validation_folds=validation_folds,
        music_ratio=1.0,
    )
    if len(dataset) == 0:
        raise RuntimeError("No valid training data for stage")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True,
    )

    projector = AudioFeatureProjector(n_mels=N_MELS, segment_frames=SEGMENT_FRAMES, output_dim=512).to(device)
    ajepa = _CAJEPA(input_dim=512).to(device)

    all_params = list(projector.parameters()) + list(ajepa.parameters())
    total_params = sum(p.numel() for p in all_params)
    print(f"Params: {total_params:,} total", flush=True)

    slot_proj_ids = set(id(p) for p in ajepa.per_slot_input_proj.parameters())
    slot_proj_params = list(ajepa.per_slot_input_proj.parameters())
    other_params = [p for p in all_params if id(p) not in slot_proj_ids]
    optimizer = Adam([
        {"params": other_params, "lr": args.lr},
        {"params": slot_proj_params, "lr": args.lr / 5},
    ], weight_decay=1e-5)
    sigreg_loss = _SIGREG(var_weight=5.0, cov_weight=1.0, sim_weight=1.0)
    start_epoch = 1
    best_loss = float("inf")

    total_train_steps = args.epochs_per_stage * len(dataloader)
    steps_per_epoch = len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=1e-6)

    saved_global_step = 0
    if resume_ckpt and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        missing_proj, _ = projector.load_state_dict(ckpt.get("projector_state", {}), strict=False)
        missing_ajepa, _ = ajepa.load_state_dict(ckpt.get("ajepa_state", {}), strict=False)
        if missing_proj:
            print(f"  projector missing keys: {missing_proj}", flush=True)
        if missing_ajepa:
            print(f"  ajepa missing keys: {missing_ajepa}", flush=True)
        try:
            optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
        except ValueError:
            print("  optimizer state incompatible, starting fresh", flush=True)
        saved_global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        start_epoch = 1
        print(f"Resumed from checkpoint (step={saved_global_step}, best_loss={best_loss:.4f})", flush=True)

    os.makedirs(STAGE_CHECKPOINT_DIR, exist_ok=True)
    stage_ckpt_path = os.path.join(STAGE_CHECKPOINT_DIR, f"stage_{stage_idx:03d}_best.pt")
    log_path = os.path.join(STAGE_CHECKPOINT_DIR, f"stage_{stage_idx:03d}_log.jsonl")

    guard = TrainingGuard(
        total_steps=args.epochs_per_stage * len(dataloader),
        warmup_steps=200,
        loss_spike_factor=5.0,
        max_grad_norm=5.0,
        nan_tolerance=20,
        collapse_patience=100,
        checkpoint_dir=STAGE_CHECKPOINT_DIR,
    )

    global_step = saved_global_step
    recent_losses = []
    contrastive_weight = getattr(args, "contrastive_weight", 0.3)

    result = StageResult(stage_idx, len(stage_music_files), 0.0)
    result.epochs_run = 0
    t_stage_start = time.time()

    print(f"\n=== Stage {stage_idx}: Training ({args.epochs_per_stage} epochs) ===", flush=True)
    print(f"  Music files: {len(stage_music_files)}, ESC-50: {len(esc50_entries)}", flush=True)
    print(f"  Batches/epoch: {len(dataloader)}", flush=True)
    if sample_bank is not None:
        print(f"  Overlay augment: Ableton短采样 ({len(sample_bank._index)} files)", flush=True)
    if circuit_augment is not None:
        print(f"  Circuit augment: BOZAK-AR4/Euphonia/9500BW (p=0.5)", flush=True)

    circuit_count = {"bozak": 0, "euphonia": 0, "model9500bw": 0, "skip": 0}
    training_start_time = time.time()

    for epoch in range(start_epoch, args.epochs_per_stage + 1):
        ajepa.train()
        projector.train()
        epoch_jepa_loss = 0.0
        epoch_cont_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (history_mel, future_mel, labels) in enumerate(dataloader):
            guard.pre_step(global_step)

            history_mel = history_mel.to(device)
            future_mel = future_mel.to(device)
            labels = labels.to(device)

            if sample_bank is not None and len(sample_bank._index) > 0:
                history_mel, future_mel = batch_overlay_short_samples(
                    history_mel, future_mel, sample_bank, p=0.5, max_samples=3,
                )

            if circuit_augment is not None:
                B, Th, NM, F = history_mel.shape
                h_flat = history_mel.reshape(B * Th, NM, F)
                h_aug, h_info = circuit_augment(h_flat)
                history_mel = h_aug.reshape(B, Th, NM, F)
                Bf, Tf, _, _ = future_mel.shape
                f_flat = future_mel.reshape(Bf * Tf, NM, F)
                f_aug, f_info = circuit_augment(f_flat)
                future_mel = f_aug.reshape(Bf, Tf, NM, F)
                circ = h_info.get("circuit", "skip") if h_info.get("augmented") else "skip"
                circuit_count[circ] = circuit_count.get(circ, 0) + 2

            B, T_hist, N_M, F = history_mel.shape

            history_flat = history_mel.reshape(B * T_hist, N_M, F)
            history_features = projector(history_flat).reshape(B, T_hist, -1)
            history_for_jepa = ajepa.project_to_slots(history_features)

            future_flat = future_mel.reshape(B * future_mel.shape[1], N_M, F)
            future_features = projector(future_flat).reshape(B, future_mel.shape[1], -1)
            future_for_jepa = ajepa.project_to_slots(future_features)

            with torch.no_grad():
                target_slots_history, _ = ajepa.target_encoder(history_for_jepa)
                target_slots_future, _ = ajepa.target_encoder(future_for_jepa)

            context_slots_history, _ = ajepa.context_encoder(history_for_jepa)
            masked_slots, slot_mask, _ = ajepa.masker(context_slots_history)
            recovered_slots, predicted_future = ajepa.predictor(masked_slots)

            recovery_loss = sigreg_loss(
                recovered_slots.reshape(-1, ajepa.slot_dim),
                target_slots_history.reshape(-1, ajepa.slot_dim),
            )
            T_pred = predicted_future.shape[1]
            T_target = target_slots_future.shape[1]
            T_match = min(T_pred, T_target)
            forward_loss = sigreg_loss(
                predicted_future[:, :T_match].reshape(-1, ajepa.slot_dim),
                target_slots_future[:, :T_match].reshape(-1, ajepa.slot_dim),
            )
            jepa_loss = recovery_loss["total"] + 0.5 * forward_loss["total"]

            clip_embeds = history_features.mean(dim=1)
            cont_loss = supervised_contrastive_loss(clip_embeds, labels, temperature=0.07)

            total_loss = jepa_loss + contrastive_weight * cont_loss

            if sample_bank is not None and len(sample_bank._index) > 0 and batch_idx % 8 == 0:
                anchors, positives = short_sample_contrastive_pairs(
                    sample_bank, n_pairs=min(B, 16), device=device,
                )
                if anchors is not None and anchors.shape[0] >= 2:
                    anc_feats = projector(anchors)
                    pos_feats = projector(positives)
                    all_feats = torch.cat([anc_feats, pos_feats], dim=0)
                    all_labels = torch.arange(len(anc_feats), device=device).repeat(2)
                    short_cont = torch.nn.functional.cross_entropy(
                        (all_feats / all_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)) @ all_feats.T / 0.07,
                        all_labels,
                    )
                    total_loss = total_loss + contrastive_weight * 0.15 * short_cont

            if guard.check_loss(total_loss):
                guard.state.skipped_batches += 1
                global_step += 1
                continue

            optimizer.zero_grad()
            total_loss.backward()

            if guard.check_gradients(ajepa) or guard.check_gradients(projector):
                optimizer.zero_grad()
                guard.state.skipped_batches += 1
                global_step += 1
                continue

            torch.nn.utils.clip_grad_norm_(all_params, guard.max_grad_norm)
            optimizer.step()
            scheduler.step()

            ema_decay = guard.get_ema_decay(global_step)
            ajepa.ema_decay = ema_decay
            ajepa.update_ema()

            guard.post_step(total_loss.item())

            epoch_jepa_loss += jepa_loss.item()
            epoch_cont_loss += cont_loss.item()
            n_batches += 1
            global_step += 1

            if batch_idx % 20 == 0:
                elapsed = time.time() - t0
                print(f"  S{stage_idx} E{epoch} B{batch_idx}/{len(dataloader)} "
                      f"loss={total_loss.item():.4f} jepa={jepa_loss.item():.4f} "
                      f"cont={cont_loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                      f"{elapsed:.0f}s", flush=True)

        avg_jepa = epoch_jepa_loss / max(n_batches, 1)
        avg_cont = epoch_cont_loss / max(n_batches, 1)
        avg_loss = avg_jepa + contrastive_weight * avg_cont
        elapsed = time.time() - t0
        print(f"\n=== S{stage_idx} E{epoch}/{args.epochs_per_stage} "
              f"jepa={avg_jepa:.4f} cont={avg_cont:.4f} total={avg_loss:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.0f}s) ===", flush=True)

        log_entry = {
            "stage": stage_idx, "epoch": epoch, "global_step": global_step,
            "avg_jepa": avg_jepa, "avg_cont": avg_cont, "avg_loss": avg_loss,
            "lr": scheduler.get_last_lr()[0], "elapsed_s": elapsed,
        }
        result.loss_history.append(log_entry)
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            result.best_jepa_loss = avg_jepa
            result.best_cont_loss = avg_cont
            result.best_total_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "stage": stage_idx,
                "ajepa_state": ajepa.state_dict(),
                "projector_state": projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss,
            }, stage_ckpt_path)
            result.checkpoint_path = stage_ckpt_path

        if epoch % 10 == 0:
            total_circ = sum(circuit_count.values())
            circ_lines = []
            for name in ["bozak", "euphonia", "model9500bw", "skip"]:
                cnt = circuit_count.get(name, 0)
                pct = cnt / max(total_circ, 1) * 100
                circ_lines.append(f"    {name}: {cnt} ({pct:.1f}%)")
            print(f"\n  === EPOCH {epoch} SUMMARY ({len(result.loss_history)} epochs) ===", flush=True)
            print(f"  Circuit分布:", flush=True)
            for line in circ_lines:
                print(line, flush=True)
            if len(result.loss_history) >= 2:
                losses = [e["avg_loss"] for e in result.loss_history]
                le = len(losses)
                head_n = min(le, 5)
                tail_n = min(le, 5)
                head = " → ".join(f"{losses[i]:.2f}" for i in range(head_n))
                if le > head_n + tail_n:
                    tail = " → ".join(f"{losses[le - tail_n + i]:.2f}" for i in range(tail_n))
                    loss_line = f"loss: {head} → ... → {tail}"
                else:
                    loss_line = f"loss: {' → '.join(f'{l:.2f}' for l in losses)}"
                print(f"  {loss_line}", flush=True)
                jepa_l = [e["avg_jepa"] for e in result.loss_history]
                cont_l = [e["avg_cont"] for e in result.loss_history]
                print(f"  jepa: {jepa_l[0]:.2f} → {jepa_l[-1]:.2f}  cont: {cont_l[0]:.2f} → {cont_l[-1]:.2f}", flush=True)
                delta = losses[0] - losses[-1]
                print(f"  Δloss: -{delta:.2f} ({delta/losses[0]*100:.1f}% of initial)", flush=True)
            elapsed_total = time.time() - training_start_time
            print(f"  总耗时: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)", flush=True)
            print(f"  {'='*45}", flush=True)

        recent_losses.append(avg_loss)
        if len(recent_losses) > 5:
            recent_losses.pop(0)
        if len(recent_losses) >= 5:
            mean5 = sum(recent_losses) / 5
            range_pct = (max(recent_losses) - min(recent_losses)) / mean5 * 100 if mean5 > 0 else 0
            if range_pct < 3.0:
                print(f"\n  STAGE {stage_idx} EARLY STOP: loss plateaued (range={range_pct:.1f}%)", flush=True)
                break

        result.epochs_run += 1

    result.final_jepa_loss = avg_jepa
    result.final_cont_loss = avg_cont
    result.final_total_loss = avg_loss
    result.elapsed_s = time.time() - t_stage_start

    print(f"\n=== Stage {stage_idx} Backtest (Linear Probe 5-fold CV) ===", flush=True)
    acc = evaluate_linear_probe(projector, device)
    result.mean_accuracy = acc

    result.status = "completed"
    return result, stage_ckpt_path, projector, ajepa


def run_backtest(projector, device):
    print("  Running backtest...", flush=True)
    acc = evaluate_linear_probe(projector, device)
    return acc


def main():
    parser = argparse.ArgumentParser(description="VORTEX FLAME 自动化增量训练编排器")
    parser.add_argument("--epochs_per_stage", type=int, default=30,
                       help="每个阶段的训练 epoch 数")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--contrastive_weight", type=float, default=0.8, help="InfoNCE权重 (越高频率分辨越强)")
    parser.add_argument("--use_ableton_samples", action="store_true",
                       help="叠加Ableton短采样增强 + 短采样对比学习")
    parser.add_argument("--use_circuit_augment", action="store_true",
                       help="三重模拟电路增强 (BOZAK AR-4 + Euphonia + MODEL9500BW)")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_stages", type=int, default=10,
                       help="最大阶段数(安全上限)")
    parser.add_argument("--force_reset", action="store_true",
                       help="清除音乐追踪记录,从头开始")
    parser.add_argument("--dry_run", action="store_true",
                       help="仅扫描数据不训练")
    args = parser.parse_args()

    os.makedirs(STAGE_LOG_DIR, exist_ok=True)
    os.makedirs(STAGE_CHECKPOINT_DIR, exist_ok=True)

    print("=" * 60)
    print("VORTEX FLAME — 自动化增量训练与回测")
    print("=" * 60)
    print(f"Stage increment: {STAGE_INCREMENT*100:.0f}% per stage")
    print(f"Epochs per stage: {args.epochs_per_stage}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Contrastive weight: {args.contrastive_weight}")
    print(f"Convergence window: {CONVERGENCE_WINDOW} stages")
    print(f"Convergence acc delta: {CONVERGENCE_ACC_DELTA}")
    print(f"Max stages: {args.max_stages}")
    print()

    print("[1/5] Scanning audio files...", flush=True)
    all_music_files = scan_audio_files()
    print(f"  Total music files found: {len(all_music_files)}")

    esc50_entries = load_esc50_metadata()
    train_esc50 = [e for e in esc50_entries if e["fold"] != 5]
    print(f"  ESC-50 train entries (folds 1-4): {len(train_esc50)}")
    print(f"  ESC-50 test entries (fold 5): {len([e for e in esc50_entries if e['fold'] == 5])}")

    print("\n[2/5] Initializing Music Tracker...", flush=True)
    tracker = MusicFileTracker(STAGE_LOG_DIR, force_reset=args.force_reset)
    fresh_music = tracker.scan_available_music(all_music_files)
    print(f"  Fresh (uncached/used): {len(fresh_music)}")
    print(f"  Already used: {len(tracker.used_hashes)}")

    if args.dry_run:
        print("\n=== DRY RUN — No training ===")
        for stage_idx in range(1, args.max_stages + 1):
            allocated, used_pct = tracker.allocate_stage(all_music_files, stage_idx)
            print(f"  Stage {stage_idx}: {len(allocated)} music files, {used_pct:.1f}% used")
        return

    print("\n[3/5] Starting Orchestrator Loop...", flush=True)

    sample_bank = None
    if getattr(args, "use_ableton_samples", False):
        print("  Initializing Ableton短采样库...", flush=True)
        sample_bank = build_sample_augment(max_cache=2000)

    circuit_augment = None
    if getattr(args, "use_circuit_augment", False):
        print("  Initializing 三重模拟电路增强 (BOZAK/Euphonia/9500BW)...", flush=True)
        circuit_augment = build_audio_circuit_augment(n_mels=N_MELS, sample_rate=SAMPLE_RATE, p_augment=0.5)

    monitor = ConvergenceMonitor()
    all_stage_results = []
    resume_ckpt = None
    final_projector = None
    final_ajepa = None

    next_stage = 1
    if not args.force_reset:
        import glob as _glob, re as _re
        ckpts = sorted(_glob.glob(os.path.join(STAGE_CHECKPOINT_DIR, "stage_*_best.pt")))
        if ckpts:
            resume_ckpt = ckpts[-1]
            m = _re.search(r'stage_(\d+)_best', os.path.basename(ckpts[-1]))
            if m:
                next_stage = int(m.group(1)) + 1
            print(f"  Auto-resume: {os.path.basename(resume_ckpt)} -> Stage {next_stage}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")

    for stage_idx in range(next_stage, args.max_stages + 1):
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}/{args.max_stages}")
        print(f"{'='*60}")

        allocated, used_pct = tracker.allocate_stage(all_music_files, stage_idx)
        print(f"  Allocated music: {len(allocated)}, total used: {used_pct:.1f}%")

        music_exhausted = len(allocated) == 0
        if music_exhausted:
            print("  All music files exhausted. Triggering convergence.")
            break

        try:
            stage_result, stage_ckpt, proj, ajepa_model = run_stage_training(
                allocated, esc50_entries, stage_idx, args, resume_ckpt=resume_ckpt,
                sample_bank=sample_bank,
                circuit_augment=circuit_augment,
            )
            resume_ckpt = stage_ckpt
            final_projector = proj
            final_ajepa = ajepa_model
        except Exception as e:
            print(f"\n  *** STAGE {stage_idx} FAILED: {e} ***")
            traceback.print_exc()
            stage_result = StageResult(stage_idx, len(allocated), used_pct)
            stage_result.status = "failed"
            stage_result.error_msg = str(e)
            all_stage_results.append(stage_result)
            break

        stage_result.music_used_pct = used_pct
        all_stage_results.append(stage_result)

        monitor.update(stage_result.mean_accuracy, stage_result.final_total_loss)

        print(f"\n--- Stage {stage_idx} Summary ---")
        print(f"  Epochs run: {stage_result.epochs_run}")
        print(f"  Best loss: jepa={stage_result.best_jepa_loss:.4f} "
              f"cont={stage_result.best_cont_loss:.4f} total={stage_result.best_total_loss:.4f}")
        print(f"  Final loss: jepa={stage_result.final_jepa_loss:.4f} "
              f"cont={stage_result.final_cont_loss:.4f} total={stage_result.final_total_loss:.4f}")
        print(f"  Backtest (5-fold CV): {stage_result.mean_accuracy*100:.1f}%")
        print(f"  Music: {stage_result.music_count} files, {stage_result.music_used_pct:.1f}% used")
        print(f"  Elapsed: {stage_result.elapsed_s:.0f}s")

        converged, reason = monitor.check_convergence(music_exhausted=False)
        if converged:
            print(f"\n  *** CONVERGED: {reason} ***")
            break
        else:
            print(f"  Status: {reason}")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    summary_path = os.path.join(STAGE_CHECKPOINT_DIR, "orchestrator_summary.json")
    summary_data = []
    for r in all_stage_results:
        summary_data.append(r.to_dict())
        print(f"\n  Stage {r.stage_idx}: {r.status}")
        print(f"    Music: {r.music_count} files ({r.music_used_pct:.1f}% used)")
        print(f"    Best loss: total={r.best_total_loss:.4f}")
        print(f"    Backtest accuracy: {r.mean_accuracy*100:.1f}%")
        print(f"    Epochs: {r.epochs_run}, Elapsed: {r.elapsed_s:.0f}s")
        if r.error_msg:
            print(f"    Error: {r.error_msg}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")

    acc_values = [r.mean_accuracy for r in all_stage_results if r.status == "completed"]
    if acc_values:
        print(f"\nAccuracy trend: {' → '.join(f'{a*100:.1f}%' for a in acc_values)}")
        best_stage = max(range(len(acc_values)), key=lambda i: acc_values[i])
        print(f"Best stage: {best_stage+1} ({acc_values[best_stage]*100:.1f}%)")


if __name__ == "__main__":
    main()
