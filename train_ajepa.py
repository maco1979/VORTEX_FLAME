#!/usr/bin/env python3
"""
A-JEPA (Audio JEPA) Training Script for VORTEX FLAME
=====================================================

Trains CAJEPA (Causal Audio JEPA) on real audio data.
Real-time mel spectrogram extraction from local audio files.

Pipeline:
  mp3/wav -> MelSpectrogram -> chunk segments -> AudioFeatureProjector
  -> CAJEPA slots -> object-level masking -> causal prediction

Usage:
  python train_ajepa.py --epochs 50 --batch 8
"""

import argparse
import json
import os
import sys
import time
import random
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
    os.getenv("DEEP_HOUSE", r"E:\E盘数据\DEEP HOUSE 集成版"),
    os.getenv("VOCAL_RAW", r"E:\VORTEX_FLAME_歌词工厂\人声训练包\原始音频\原创音乐"),
    os.getenv("VOCAL_DENOISED", r"E:\VORTEX_FLAME_歌词工厂\人声训练包\降噪后"),
    os.getenv("TEMPLE_MUSIC", r"D:\temple_music"),
    os.getenv("MIXED_IN_KEY", r"D:\AppData_New\Local\Mixed In Key"),
    os.getenv("TRAKTOR_FACTORY", r"C:\ProgramData\Native Instruments\Traktor Pro 4\Factory Sounds"),
    os.path.expanduser("~/Downloads"),
]

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ajepa_checkpoints")

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SEGMENT_FRAMES = 256
HISTORY_SEGMENTS = 6
FUTURE_SEGMENTS = 4
MIN_SEGMENTS = HISTORY_SEGMENTS + FUTURE_SEGMENTS


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
        for root, _, files in os.walk(audio_dir):
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


class AJEPADataset(torch.utils.data.Dataset):
    _fallback_count = 0

    @classmethod
    def get_fallback_rate(cls, total_samples: int) -> float:
        if total_samples <= 0:
            return 0.0
        return cls._fallback_count / total_samples

    @classmethod
    def reset_fallback_counter(cls):
        cls._fallback_count = 0

    def __init__(self, audio_files, history_len=6, future_len=4):
        import torchaudio

        self.history_len = history_len
        self.future_len = future_len
        self.min_segments = history_len + future_len

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

        print(f"Indexing {len(audio_files)} audio files...", flush=True)
        self.song_data = []
        for fp in audio_files:
            try:
                info = torchaudio.info(fp)  # pyright: ignore[reportAttributeAccessIssue]
                duration = info.num_frames / info.sample_rate
                n_frames_mel = int(duration * SAMPLE_RATE / HOP_LENGTH)
                n_segments = n_frames_mel // SEGMENT_FRAMES
                if n_segments >= self.min_segments:
                    self.song_data.append({"path": fp, "n_segments": n_segments, "duration": duration})
            except Exception:
                pass

        total_hrs = sum(s["duration"] for s in self.song_data) / 3600
        print(f"Dataset: {len(self.song_data)} songs, {total_hrs:.1f}h", flush=True)

        bad_paths = set()
        from contextlib import redirect_stderr
        for s in self.song_data:
            try:
                with open(os.devnull, "w") as fnull:
                    with redirect_stderr(fnull):
                        torchaudio.load(s["path"], num_frames=48000)
            except Exception:
                bad_paths.add(s["path"])
        if bad_paths:
            self.song_data = [s for s in self.song_data if s["path"] not in bad_paths]
            print(f"Filtered {len(bad_paths)} un-decodable files (corrupted body despite valid header)", flush=True)
            for bp in bad_paths:
                print(f"  BAD: {bp}", flush=True)

    def __len__(self):
        return len(self.song_data) * 50

    def _load_mel(self, filepath):
        import torchaudio
        from contextlib import redirect_stderr
        try:
            with open(os.devnull, "w") as fnull:
                with redirect_stderr(fnull):
                    wav, sr = torchaudio.load(filepath)
        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e)[:120]
            if "Unspecified internal error" in err_msg or "resync" in err_msg:
                print(f"  [BAD_FILE] DECODE_ERROR path={filepath} reason=mpg123_decode_failed", flush=True)
            elif "No such file" in err_msg or "not found" in err_msg.lower():
                print(f"  [BAD_FILE] MISSING path={filepath}", flush=True)
            else:
                print(f"  [BAD_FILE] LOAD_ERROR path={filepath} type={err_type} reason={err_msg}", flush=True)
            return None
        try:
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            with torch.no_grad():
                mel = self.mel_transform(wav)
                mel_db = self.amp_to_db(mel)
            mel_db = mel_db.squeeze(0)
            n_frames = mel_db.shape[1]
            n_segments = n_frames // SEGMENT_FRAMES
            if n_segments < self.min_segments:
                return None
            mel_db = mel_db[:, :n_segments * SEGMENT_FRAMES]
            segments = mel_db.reshape(N_MELS, n_segments, SEGMENT_FRAMES).permute(1, 0, 2)
            return segments
        except Exception as e:
            err_type = type(e).__name__
            print(f"  [BAD_FILE] MEL_ERROR path={filepath} type={err_type} reason={str(e)[:120]}", flush=True)
            return None

    def __getitem__(self, idx):
        song_idx = idx % len(self.song_data)
        info = self.song_data[song_idx]

        segments = self._load_mel(info["path"])
        if segments is None:
            AJEPADataset._fallback_count += 1
            segments = torch.randn(self.min_segments, N_MELS, SEGMENT_FRAMES) * 0.1

        actual_n = segments.shape[0]
        max_offset = max(0, actual_n - self.min_segments)
        offset = random.randint(0, max_offset) if max_offset > 0 else 0
        history = segments[offset:offset + self.history_len]
        future = segments[offset + self.history_len:offset + self.history_len + self.future_len]

        return history, future


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
    import json as _json
    try:
        with open(path, "a") as f:
            f.write(_json.dumps(entry) + "\n")
    except Exception:
        pass


def train(args):
    audio_files = scan_audio_files()
    if not audio_files:
        print("ERROR: No audio files found!")
        return

    dataset = AJEPADataset(audio_files, HISTORY_SEGMENTS, FUTURE_SEGMENTS)
    if len(dataset) == 0:
        print("ERROR: No valid training data!")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(args, "cuda", False) and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(getattr(args, "device", 0)))
    print(f"Device: {device}")
    if device.type != "cuda":
        print("  ⚠ WARNING: Running on CPU — training will be slow.")

    projector = AudioFeatureProjector(n_mels=N_MELS, segment_frames=SEGMENT_FRAMES, output_dim=512).to(device)
    ajepa = CAJEPA(input_dim=512).to(device)

    all_params = list(projector.parameters()) + list(ajepa.parameters())
    total_params = sum(p.numel() for p in all_params)
    trainable = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"Params: {total_params:,} total, {trainable:,} trainable")

    optimizer = Adam(all_params, lr=args.lr, weight_decay=1e-5)
    sigreg_loss = SIGRegWithPredictionLoss(var_weight=5.0, cov_weight=1.0, sim_weight=1.0)
    start_epoch = 1
    best_loss = float("inf")

    total_train_steps = args.epochs * len(dataloader)
    steps_per_epoch = len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=1e-6)

    saved_global_step = 0
    resume_path = os.path.join(CHECKPOINT_DIR, "ajepa_best.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        missing_proj, unexpected_proj = projector.load_state_dict(ckpt["projector_state"], strict=False)
        ajepa.load_state_dict(ckpt["ajepa_state"])
        if missing_proj:
            print(f"  Note: projector missing keys (new BN layers will train from scratch): {missing_proj}")
        if "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except ValueError:
                print("  Note: optimizer state incompatible (model changed), starting optimizer fresh")
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        saved_global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch-1}, step={saved_global_step}, best_loss={best_loss:.4f}")

        if "scheduler_state" not in ckpt:
            print("OLD CHECKPOINT: scheduler state missing, applying warmup")
            warmup_epochs = 3
            warmup_steps = warmup_epochs * steps_per_epoch
            for g in optimizer.param_groups:
                g['lr'] = args.lr * 0.1
            lr_history = [args.lr * 0.1 + (args.lr - args.lr * 0.1) * (i / max(warmup_steps - 1, 1)) for i in range(warmup_steps)]
            from torch.optim.lr_scheduler import LambdaLR  # pyright: ignore[reportUnusedImport]
            warmup_lambda = lambda step: lr_history[min(step, len(lr_history) - 1)] / args.lr if step < len(lr_history) else 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
            for _ in range(warmup_steps):
                scheduler.step()
            print(f"  Warmup complete over {warmup_steps} steps, LR→{optimizer.param_groups[0]['lr']:.2e}")
            scheduler = CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=1e-6)

    guard = TrainingGuard(
        total_steps=args.epochs * len(dataloader),
        warmup_steps=200,
        loss_spike_factor=5.0,
        max_grad_norm=5.0,
        nan_tolerance=20,
        collapse_patience=100,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log_path = os.path.join(CHECKPOINT_DIR, "training_log.jsonl")
    global_step = saved_global_step
    recent_losses = []

    for epoch in range(start_epoch, args.epochs + 1):
        ajepa.train()
        projector.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (history_mel, future_mel) in enumerate(dataloader):
            guard.pre_step(global_step)

            history_mel = history_mel.to(device)
            future_mel = future_mel.to(device)

            B, T_hist, N_M, F = history_mel.shape
            _, T_fut, _, _ = future_mel.shape

            history_flat = history_mel.reshape(B * T_hist, N_M, F)
            history_features = projector(history_flat)
            history_features = history_features.reshape(B, T_hist, -1)
            history_features = history_features.unsqueeze(2).expand(-1, -1, 8, -1)

            future_flat = future_mel.reshape(B * T_fut, N_M, F)
            future_features = projector(future_flat)
            future_features = future_features.reshape(B, T_fut, -1)
            future_features = future_features.unsqueeze(2).expand(-1, -1, 8, -1)

            with torch.no_grad():
                target_slots_history, _ = ajepa.target_encoder(history_features)
                target_slots_future, _ = ajepa.target_encoder(future_features)

            context_slots_history, _ = ajepa.context_encoder(history_features)
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
            total_loss = recovery_loss["total"] + 0.5 * forward_loss["total"]

            if guard.check_loss(total_loss):
                guard.state.skipped_batches += 1
                global_step += 1
                continue

            optimizer.zero_grad()
            total_loss.backward()

            grad_norms = _compute_grad_norms(projector, ajepa)
            grad_log_entry = {
                "global_step": global_step, "epoch": epoch, "batch": batch_idx,
                "total_loss": total_loss.item(), "lr": optimizer.param_groups[0]["lr"],
                "grad_norms": grad_norms,
            }
            _write_grad_log(os.path.join(CHECKPOINT_DIR, "gradient_log.jsonl"), grad_log_entry)

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
            guard.log_stats(global_step)

            epoch_loss += total_loss.item()
            n_batches += 1
            global_step += 1

            if batch_idx % 10 == 0:
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
        total_samples = n_batches * args.batch
        fallback_count = AJEPADataset._fallback_count
        fallback_rate = AJEPADataset.get_fallback_rate(total_samples)
        print(f"\n=== Epoch {epoch}/{args.epochs} avg_loss={avg_loss:.4f} ({elapsed:.0f}s) ===", flush=True)
        if fallback_count > 0:
            print(f"  [FALLBACK] {fallback_count} corrupted samples → random noise ({fallback_rate*100:.2f}% of {total_samples} total)", flush=True)
        AJEPADataset.reset_fallback_counter()

        recent_losses.append(avg_loss)
        if len(recent_losses) > 5:
            recent_losses.pop(0)
        if len(recent_losses) >= 5:
            mean5 = sum(recent_losses) / 5
            range_pct = (max(recent_losses) - min(recent_losses)) / mean5 * 100 if mean5 > 0 else 0
            if range_pct < 3.0:
                print(f"\n  EARLY STOP: loss plateaued (range={range_pct:.1f}% over last 5 epochs)", flush=True)
                break

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
                "global_step": global_step,
                "ajepa_state": ajepa.state_dict(),
                "projector_state": projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(CHECKPOINT_DIR, "ajepa_best.pt"))
            _backup_path = os.path.join(CHECKPOINT_DIR, f"ajepa_best_backup_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "ajepa_state": ajepa.state_dict(),
                "projector_state": projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss,
            }, _backup_path)
            print(f"  NEW BEST: {best_loss:.4f} (backup saved)")

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "ajepa_state": ajepa.state_dict(),
                "projector_state": projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss,
            }, os.path.join(CHECKPOINT_DIR, f"ajepa_epoch{epoch}.pt"))

    torch.save({
        "projector_state": projector.state_dict(),
        "ajepa_encoder_state": ajepa.context_encoder.state_dict(),
        "config": {"n_mels": N_MELS, "segment_frames": SEGMENT_FRAMES,
                   "input_dim": 512, "num_slots": 5, "slot_dim": 128},
    }, os.path.join(CHECKPOINT_DIR, "ajepa_audio_embedder.pt"))
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="A-JEPA Training for VORTEX FLAME")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cuda", action="store_true", default=False, help="Force CUDA device")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
