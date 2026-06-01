#!/usr/bin/env python3
"""
A-JEPA Multiclass Training — JEPA + Supervised Contrastive Learning
====================================================================

Trains AudioFeatureProjector + CAJEPA on mixed data:
  - Deep House music (class 50: "music") — JEPA temporal prediction
  - ESC-50 (classes 0-49) — supervised contrastive clustering

Pipeline:
  audio -> MelSpectrogram -> segments -> AudioFeatureProjector
  -> CAJEPA (JEPA loss) + ContrastiveHead (InfoNCE with labels)

After pretraining: linear probe evaluates 51-class accuracy.

Usage:
  python train_ajepa_multiclass.py --epochs 50 --batch 8 --contrastive_weight 0.3
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
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))
from five_layer_jepa.causal_jepa import CAJEPA
from five_layer_jepa.causal_jepa_v2 import SIGRegWithPredictionLoss
from jepa_training_guard import TrainingGuard

AUDIO_DIRS = [
    r"E:\E盘数据\DEEP HOUSE 集成版",
    r"E:\VORTEX_FLAME_歌词工厂\人声训练包\原始音频\原创音乐",
    r"E:\VORTEX_FLAME_歌词工厂\人声训练包\降噪后",
    r"D:\temple_music",
    r"D:\AppData_New\Local\Mixed In Key",
    r"C:\ProgramData\Native Instruments\Traktor Pro 4\Factory Sounds",
    r"C:\Users\42235\Downloads",
]

ESC50_AUDIO_DIR = r"E:\AI_Data\ESC-50\ESC-50-master\audio"
ESC50_META = r"E:\AI_Data\ESC-50\ESC-50-master\meta\esc50.csv"

CHECKPOINT_DIR = r"D:\VORTEX_FLAME\ajepa_checkpoints"
MC_CHECKPOINT_DIR = r"D:\VORTEX_FLAME\ajepa_multiclass_checkpoints"

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SEGMENT_FRAMES = 256
HISTORY_SEGMENTS = 6
FUTURE_SEGMENTS = 4
MIN_SEGMENTS = HISTORY_SEGMENTS + FUTURE_SEGMENTS
NUM_CLASSES = 51
MUSIC_LABEL = 50

ESC50_TARGET_DURATION = 65.0
CACHE_DIR = r"E:\AI_Data\mel_cache"
MUSIC_RATIO = 0.1

def _cache_key(filepath):
    return hashlib.md5(filepath.encode()).hexdigest()[:16]


class AudioFeatureProjector(nn.Module):
    def __init__(self, n_mels=128, segment_frames=256, output_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        dummy = torch.zeros(1, 1, n_mels, segment_frames)
        with torch.no_grad():
            dummy = F.gelu(self.conv1(dummy))
            dummy = F.gelu(self.conv2(dummy))
            dummy = F.gelu(self.conv3(dummy))
        flat_dim = dummy.reshape(1, -1).shape[1]
        self.proj = nn.Linear(flat_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, mel_segment):
        if mel_segment.dim() == 3:
            mel_segment = mel_segment.unsqueeze(1)
        x = F.gelu(self.bn1(self.conv1(mel_segment)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.conv3(x))
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = self.norm(self.proj(x))
        return x


def scan_audio_files():
    audio_exts = {".mp3", ".flac", ".wav", ".m4a", ".ogg"}
    seen_names = set()
    all_files = []
    dupes = 0
    for audio_dir in AUDIO_DIRS:
        if not os.path.exists(audio_dir):
            continue
        for root, dirs, files in os.walk(audio_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in audio_exts:
                    fp = os.path.join(root, f)
                    try:
                        sz = os.path.getsize(fp)
                        if sz > 500000:
                            with open(fp, "rb") as fh:
                                fh.read(100)
                            key = f.lower()
                            if key in seen_names:
                                dupes += 1
                                continue
                            seen_names.add(key)
                            all_files.append(fp)
                    except Exception:
                        pass
    print(f"Found {len(all_files)} readable audio files (>500KB)", flush=True)
    if dupes:
        print(f"  Skipped {dupes} duplicate filenames", flush=True)
    return all_files


def load_esc50_metadata():
    esc50_entries = []
    with open(ESC50_META, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            folder = int(row["fold"])
            target = int(row["target"])
            category = row["category"]
            filepath = os.path.join(ESC50_AUDIO_DIR, filename)
            if os.path.exists(filepath):
                esc50_entries.append({
                    "path": filepath,
                    "fold": folder,
                    "label": target,
                    "category": category,
                })
    return esc50_entries


class MulticlassAJEPADataset(torch.utils.data.Dataset):
    _fallback_count = 0

    @classmethod
    def get_fallback_rate(cls, total_samples):
        if total_samples <= 0:
            return 0.0
        return cls._fallback_count / total_samples

    @classmethod
    def reset_fallback_counter(cls):
        cls._fallback_count = 0

    def __init__(self, music_files, esc50_entries, history_len=6, future_len=4,
                 esc50_validation_folds=None, music_ratio=MUSIC_RATIO):
        self.history_len = history_len
        self.future_len = future_len
        self.min_segments = history_len + future_len

        if music_ratio < 1.0 and len(music_files) > 0:
            n_sample = max(1, int(len(music_files) * music_ratio))
            random.shuffle(music_files)
            music_files = music_files[:n_sample]
            print(f"Sampled {n_sample} music files ({music_ratio*100:.0f}% of total)", flush=True)

        self.entries = []
        self._index_music_from_cache(music_files)
        self._index_esc50_from_cache(esc50_entries, esc50_validation_folds)

        n_music = sum(1 for e in self.entries if e["label"] == MUSIC_LABEL)
        n_esc50 = sum(1 for e in self.entries if e["label"] != MUSIC_LABEL)
        print(f"Dataset: {len(self.entries)} songs ({n_music} music, {n_esc50} ESC-50), "
              f"{sum(e['duration'] for e in self.entries)/3600:.1f}h", flush=True)

    def _index_music_from_cache(self, music_files):
        print(f"Indexing {len(music_files)} music files (cached)...", flush=True)
        for fp in music_files:
            key = _cache_key(fp)
            cache_path = os.path.join(CACHE_DIR, "music", f"{key}.pt")
            if not os.path.exists(cache_path):
                continue
            try:
                mel_db = torch.load(cache_path, map_location="cpu", weights_only=True)
                n_frames = mel_db.shape[1]
                n_segments = n_frames // SEGMENT_FRAMES
                if n_segments >= self.min_segments:
                    duration = n_frames * HOP_LENGTH / SAMPLE_RATE
                    self.entries.append({
                        "cache_key": key, "is_esc50": False,
                        "n_segments": n_segments, "duration": duration,
                        "label": MUSIC_LABEL,
                    })
            except Exception:
                pass

    def _index_esc50_from_cache(self, esc50_entries, validation_folds):
        if validation_folds is None:
            validation_folds = set()
        train_entries = [e for e in esc50_entries if e["fold"] not in validation_folds]

        print(f"Indexing {len(train_entries)} ESC-50 files (cached)...", flush=True)
        for entry in train_entries:
            key = _cache_key(entry["path"])
            cache_path = os.path.join(CACHE_DIR, "esc50", f"{key}.pt")
            if not os.path.exists(cache_path):
                continue
            try:
                mel_db = torch.load(cache_path, map_location="cpu", weights_only=True)
                n_frames = mel_db.shape[1]
                n_segments = n_frames // SEGMENT_FRAMES
                if n_segments >= self.min_segments:
                    self.entries.append({
                        "cache_key": key, "is_esc50": True,
                        "n_segments": n_segments,
                        "duration": ESC50_TARGET_DURATION,
                        "label": entry["label"], "category": entry["category"],
                    })
            except Exception:
                pass

    def __len__(self):
        return len(self.entries) * 30

    def _load_segments_from_cache(self, cache_key, is_esc50):
        folder = "esc50" if is_esc50 else "music"
        cache_path = os.path.join(CACHE_DIR, folder, f"{cache_key}.pt")
        try:
            mel_db = torch.load(cache_path, map_location="cpu", weights_only=True)
        except Exception:
            MulticlassAJEPADataset._fallback_count += 1
            return torch.randn(self.min_segments, N_MELS, SEGMENT_FRAMES) * 0.1

        n_frames = mel_db.shape[1]
        n_segments = n_frames // SEGMENT_FRAMES
        if n_segments < self.min_segments:
            MulticlassAJEPADataset._fallback_count += 1
            return torch.randn(self.min_segments, N_MELS, SEGMENT_FRAMES) * 0.1

        mel_db = mel_db[:, :n_segments * SEGMENT_FRAMES]
        segments = mel_db.reshape(N_MELS, n_segments, SEGMENT_FRAMES).permute(1, 0, 2)
        return segments

    def __getitem__(self, idx):
        entry_idx = idx % len(self.entries)
        info = self.entries[entry_idx]
        segments = self._load_segments_from_cache(info["cache_key"], info["is_esc50"])

        if segments is None:
            MulticlassAJEPADataset._fallback_count += 1
            segments = torch.randn(self.min_segments, N_MELS, SEGMENT_FRAMES) * 0.1

        actual_n = segments.shape[0]
        max_offset = max(0, actual_n - self.min_segments)
        offset = random.randint(0, max_offset) if max_offset > 0 else 0
        history = segments[offset:offset + self.history_len]
        future = segments[offset + self.history_len:offset + self.history_len + self.future_len]

        return history, future, info["label"]


def supervised_contrastive_loss(features, labels, temperature=0.07):
    features = F.normalize(features, dim=1)
    sim_matrix = torch.mm(features, features.T) / temperature

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    mask.fill_diagonal_(0)
    pos_sim = (sim_matrix * mask).sum(dim=1)
    pos_count = mask.sum(dim=1).clamp(min=1)
    pos_sim = pos_sim / pos_count

    exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(sim_matrix.shape[0], device=features.device))
    denominator = exp_sim.sum(dim=1).clamp(min=1e-8)

    loss = -pos_sim + torch.log(denominator)
    return loss.mean()


def _compute_grad_norms(projector, ajepa):
    norms = {}
    for name, p in projector.named_parameters():
        if p.grad is not None and p.ndim >= 2:
            norms[f"proj.{name}"] = round(p.grad.norm(2).item(), 4)
    total_proj = round(sum(v for v in norms.values()), 4)
    norms["proj._total"] = total_proj
    for name, p in ajepa.named_parameters():
        if p.grad is not None and p.ndim >= 2:
            norms[f"ajepa.{name}"] = round(p.grad.norm(2).item(), 4)
    total_ajepa = round(sum(v for k, v in norms.items() if k.startswith("ajepa.")), 4)
    norms["ajepa._total"] = total_ajepa
    all_params = list(projector.parameters()) + list(ajepa.parameters())
    total_norm = round(sum(p.grad.norm(2).item() ** 2 for p in all_params if p.grad is not None) ** 0.5, 4)
    norms["_all"] = total_norm
    return norms


def _write_grad_log(path, entry):
    try:
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def evaluate_linear_probe(projector, device, n_classes=NUM_CLASSES):
    projector.eval()

    X_all = []
    y_all = []

    esc50_entries = load_esc50_metadata()
    for entry in esc50_entries:
        key = _cache_key(entry["path"])
        cache_path = os.path.join(CACHE_DIR, "esc50", f"{key}.pt")
        if not os.path.exists(cache_path):
            continue
        try:
            mel_db = torch.load(cache_path, map_location="cpu", weights_only=True)
            n_frames = mel_db.shape[1]
            if n_frames < SEGMENT_FRAMES:
                pad = torch.zeros(N_MELS, SEGMENT_FRAMES - n_frames)
                mel_db = torch.cat([mel_db, pad], dim=1)
                n_frames = SEGMENT_FRAMES
            n_segments = n_frames // SEGMENT_FRAMES
            mel_db = mel_db[:, :n_segments * SEGMENT_FRAMES]
            segments = mel_db.reshape(N_MELS, n_segments, SEGMENT_FRAMES).permute(1, 0, 2)
            with torch.no_grad():
                feats = projector(segments.to(device))
                clip_embed = feats.mean(dim=0)
            X_all.append(clip_embed.cpu())
            y_all.append(entry["label"])
        except Exception:
            pass

    if not X_all:
        print("  Linear Probe: no features extracted", flush=True)
        projector.train()
        return 0.0

    X = torch.stack(X_all)
    y = torch.tensor(y_all, dtype=torch.long)

    folds = {}
    for entry in esc50_entries:
        folds[entry["path"]] = entry["fold"]

    fold_accuracies = []
    for fold_idx in range(1, 6):
        test_mask = torch.tensor([folds.get(esc50_entries[i]["path"], 0) == fold_idx
                                   for i in range(len(y))])
        if test_mask.sum() == 0:
            continue
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        probe = nn.Linear(512, n_classes).to(device)
        opt = Adam(probe.parameters(), lr=0.001)
        ce = nn.CrossEntropyLoss()

        X_train_d = X_train.to(device)
        y_train_d = y_train.to(device)
        X_test_d = X_test.to(device)
        y_test_d = y_test.to(device)

        for _ in range(200):
            opt.zero_grad()
            loss = ce(probe(X_train_d), y_train_d)
            loss.backward()
            opt.step()

        with torch.no_grad():
            preds = probe(X_test_d).argmax(dim=1)
            acc = (preds == y_test_d).float().mean().item()
        fold_accuracies.append(acc)
        print(f"  Fold {fold_idx}: {acc*100:.1f}%", flush=True)

    mean_acc = sum(fold_accuracies) / len(fold_accuracies) if fold_accuracies else 0
    print(f"  Linear Probe 5-fold CV: {mean_acc*100:.1f}%", flush=True)
    return mean_acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(args, "cuda", False) and torch.cuda.is_available():
        device = torch.device(f"cuda:{getattr(args, 'device', 0)}")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("  WARNING: Running on CPU")

    music_files = scan_audio_files()
    esc50_entries = load_esc50_metadata()
    print(f"ESC-50: {len(esc50_entries)} labeled files, {len(set(e['label'] for e in esc50_entries))} classes")

    validation_folds = {5}
    dataset = MulticlassAJEPADataset(
        music_files, esc50_entries, HISTORY_SEGMENTS, FUTURE_SEGMENTS,
        esc50_validation_folds=validation_folds,
    )
    if len(dataset) == 0:
        print("ERROR: No valid training data!")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True,
    )

    projector = AudioFeatureProjector(n_mels=N_MELS, segment_frames=SEGMENT_FRAMES, output_dim=512).to(device)
    ajepa = CAJEPA(input_dim=512).to(device)

    all_params = list(projector.parameters()) + list(ajepa.parameters())
    total_params = sum(p.numel() for p in all_params)
    trainable = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"Params: {total_params:,} total, {trainable:,} trainable")

    slot_proj_ids = set(id(p) for p in ajepa.per_slot_input_proj.parameters())
    slot_proj_params = list(ajepa.per_slot_input_proj.parameters())
    other_params = [p for p in all_params if id(p) not in slot_proj_ids]
    optimizer = Adam([
        {"params": other_params, "lr": args.lr},
        {"params": slot_proj_params, "lr": args.lr / 5},
    ], weight_decay=1e-5)
    sigreg_loss = SIGRegWithPredictionLoss(var_weight=5.0, cov_weight=1.0, sim_weight=1.0)
    start_epoch = 1
    best_loss = float("inf")

    total_train_steps = args.epochs * len(dataloader)
    steps_per_epoch = len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=1e-6)

    saved_global_step = 0
    mc_resume_path = os.path.join(MC_CHECKPOINT_DIR, "ajepa_mc_best.pt")
    old_resume_path = os.path.join(CHECKPOINT_DIR, "ajepa_best.pt")
    resume_path = mc_resume_path if os.path.exists(mc_resume_path) else old_resume_path

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        missing_proj, _ = projector.load_state_dict(ckpt.get("projector_state", ckpt.get("projector_state", {})), strict=False)
        if "ajepa_state" in ckpt:
            missing_ajepa, _ = ajepa.load_state_dict(ckpt["ajepa_state"], strict=False)
            if missing_ajepa:
                print(f"  Note: ajepa missing keys (new architecture): {missing_ajepa}")
        if missing_proj:
            print(f"  Note: projector missing keys: {missing_proj}")
        if "optimizer_state" in ckpt and resume_path == mc_resume_path:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except ValueError:
                print("  Note: optimizer state incompatible, starting optimizer fresh")
        if "scheduler_state" in ckpt and resume_path == mc_resume_path:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            except Exception:
                pass
        saved_global_step = ckpt.get("global_step", 0) if resume_path == mc_resume_path else 0
        best_loss = ckpt.get("best_loss", float("inf")) if resume_path == mc_resume_path else float("inf")
        if resume_path == mc_resume_path:
            start_epoch = ckpt.get("epoch", 0) + 1
        else:
            start_epoch = 1
            print("  Note: loaded projector weights from old checkpoint, starting multiclass training from epoch 1")
        if args.lr != 1e-4 and resume_path == mc_resume_path:
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 0:
                    param_group["lr"] = args.lr
                elif i == 1:
                    param_group["lr"] = args.lr / 5
            scheduler = CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=1e-6)
            scheduler.last_epoch = saved_global_step
            print(f"  Note: lr=[{args.lr:.0e}, {args.lr/5:.0e}] scheduler reset at step {saved_global_step}")
        print(f"Resumed from epoch {start_epoch-1}, step={saved_global_step}, best_loss={best_loss:.4f}")

    guard = TrainingGuard(
        total_steps=args.epochs * len(dataloader),
        warmup_steps=200,
        loss_spike_factor=5.0,
        max_grad_norm=5.0,
        nan_tolerance=20,
        collapse_patience=100,
        checkpoint_dir=MC_CHECKPOINT_DIR,
    )

    os.makedirs(MC_CHECKPOINT_DIR, exist_ok=True)
    log_path = os.path.join(MC_CHECKPOINT_DIR, "training_log.jsonl")
    global_step = saved_global_step
    recent_losses = []
    contrastive_weight = getattr(args, "contrastive_weight", 0.3)

    print(f"\n=== Training: JEPA + Contrastive (weight={contrastive_weight}) ===")
    print(f"Classes: 0-49 (ESC-50 sound events), 50 (music)")
    print(f"Batches/epoch: {len(dataloader)}, Steps/epoch: {steps_per_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
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

            B, T_hist, N_M, F = history_mel.shape
            _, T_fut, _, _ = future_mel.shape

            history_flat = history_mel.reshape(B * T_hist, N_M, F)
            history_features = projector(history_flat)
            history_features_bb = history_features.reshape(B, T_hist, -1)

            history_features_for_jepa = ajepa.project_to_slots(history_features_bb)

            future_flat = future_mel.reshape(B * T_fut, N_M, F)
            future_features = projector(future_flat)
            future_features_bb = future_features.reshape(B, T_fut, -1)
            future_features_for_jepa = ajepa.project_to_slots(future_features_bb)

            with torch.no_grad():
                target_slots_history, _ = ajepa.target_encoder(history_features_for_jepa)
                target_slots_future, _ = ajepa.target_encoder(future_features_for_jepa)

            context_slots_history, _ = ajepa.context_encoder(history_features_for_jepa)
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

            clip_embeds = history_features_bb.mean(dim=1)
            cont_loss = supervised_contrastive_loss(clip_embeds, labels, temperature=0.07)

            total_loss = jepa_loss + contrastive_weight * cont_loss

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

            if batch_idx % 10 == 0:
                vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                elapsed = time.time() - t0
                print(f"  E{epoch} B{batch_idx}/{len(dataloader)} "
                      f"loss={total_loss.item():.4f} "
                      f"jepa={jepa_loss.item():.4f} "
                      f"cont={cont_loss.item():.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} "
                      f"VRAM={vram:.1f}GB {elapsed:.0f}s", flush=True)

        avg_jepa = epoch_jepa_loss / max(n_batches, 1)
        avg_cont = epoch_cont_loss / max(n_batches, 1)
        avg_loss = avg_jepa + contrastive_weight * avg_cont
        elapsed = time.time() - t0
        total_samples = n_batches * args.batch
        fallback_count = MulticlassAJEPADataset._fallback_count
        fallback_rate = MulticlassAJEPADataset.get_fallback_rate(total_samples)
        print(f"\n=== E{epoch}/{args.epochs} "
              f"jepa={avg_jepa:.4f} cont={avg_cont:.4f} total={avg_loss:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.0f}s) ===", flush=True)
        if fallback_count > 0:
            print(f"  [FALLBACK] {fallback_count} corrupted ({fallback_rate*100:.2f}%)", flush=True)
        MulticlassAJEPADataset.reset_fallback_counter()

        if epoch % 5 == 0:
            print("  Running linear probe evaluation...", flush=True)
            acc = evaluate_linear_probe(projector, device)
            print(f"  Linear probe accuracy: {acc*100:.1f}%", flush=True)

        recent_losses.append(avg_loss)
        if len(recent_losses) > 5:
            recent_losses.pop(0)
        if len(recent_losses) >= 5:
            mean5 = sum(recent_losses) / 5
            range_pct = (max(recent_losses) - min(recent_losses)) / mean5 * 100 if mean5 > 0 else 0
            if range_pct < 3.0:
                print(f"\n  EARLY STOP: loss plateaued (range={range_pct:.1f}%)", flush=True)
                break

        log_entry = {
            "epoch": epoch, "global_step": global_step,
            "avg_jepa": avg_jepa, "avg_cont": avg_cont, "avg_loss": avg_loss,
            "lr": scheduler.get_last_lr()[0], "elapsed_s": elapsed,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "ajepa_state": ajepa.state_dict(),
                "projector_state": projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(MC_CHECKPOINT_DIR, "ajepa_mc_best.pt"))
            print(f"  NEW BEST: {best_loss:.4f}")

    print("\n=== Final Linear Probe Evaluation ===")
    final_acc = evaluate_linear_probe(projector, device)
    print(f"\nFINAL: {final_acc*100:.1f}% 51-class accuracy via linear probe")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--contrastive_weight", type=float, default=0.3)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    train(args)
