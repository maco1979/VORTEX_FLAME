import os
import sys
import glob
import json
import random
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_ajepa_multiclass import (
    AudioFeatureProjector,
    load_esc50_metadata,
    N_MELS,
    SEGMENT_FRAMES,
    NUM_CLASSES,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "mel_cache")
STAGE_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "stage_checkpoints")

random.seed(42)
torch.manual_seed(42)


def _cache_key(path):
    import hashlib
    return hashlib.md5(path.encode()).hexdigest()


def build_augment_pipeline():
    circuit_aug = None
    sample_bank = None

    try:
        from bozak_augment import build_audio_circuit_augment
        circuit_aug = build_audio_circuit_augment(n_mels=N_MELS, sample_rate=22050, p_augment=0.5)
        print(f"  [OK] Circuit augment loaded (BOZAK/Euphonia/9500BW)")
    except Exception as e:
        print(f"  [SKIP] Circuit augment: {e}")

    try:
        from sample_augment import build_sample_augment
        sample_bank = build_sample_augment(max_cache=2000)
        print(f"  [OK] Ableton sample bank loaded ({len(sample_bank._mel_cache)} cached)")
    except Exception as e:
        print(f"  [SKIP] Sample bank: {e}")

    return circuit_aug, sample_bank


def apply_augmentation_to_segments(segments, circuit_aug, sample_bank, device):
    N_M, n_seg, SegF = segments.shape

    if sample_bank is not None and len(sample_bank._mel_cache) > 0:
        from sample_augment import batch_overlay_short_samples
        batch = segments.unsqueeze(0)
        batch_aug, _ = batch_overlay_short_samples(
            batch, batch, sample_bank, p=0.5, max_samples=2,
        )
        segments = batch_aug.squeeze(0)

    if circuit_aug is not None:
        aug_out, _ = circuit_aug(segments, force_augment=True)
        segments = aug_out

    return segments


def extract_features(projector, mel_segments, device):
    projector.eval()
    with torch.no_grad():
        feats = projector(mel_segments.to(device))
        clip_embed = feats.mean(dim=0)
    return clip_embed.cpu()


def load_all_esc50_mel_segments():
    all_entries = []
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
            all_entries.append((segments, entry["label"], entry["fold"], entry["path"]))
        except Exception:
            pass
    return all_entries


def linear_probe_5fold_cv(X, y, folds_list, device, n_classes=NUM_CLASSES, n_iters=200):
    fold_accuracies = []
    for fold_idx in range(1, 6):
        test_mask = torch.tensor([f == fold_idx for f in folds_list])
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

        for _ in range(n_iters):
            opt.zero_grad()
            loss = ce(probe(X_train_d), y_train_d)
            loss.backward()
            opt.step()

        with torch.no_grad():
            preds = probe(X_test_d).argmax(dim=1)
            acc = (preds == y_test_d).float().mean().item()
        fold_accuracies.append(acc)
        print(f"    Fold {fold_idx}: {acc*100:.1f}%", flush=True)

    mean_acc = sum(fold_accuracies) / len(fold_accuracies) if fold_accuracies else 0
    return mean_acc, fold_accuracies


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    ckpts = sorted(glob.glob(os.path.join(STAGE_CHECKPOINT_DIR, "stage_*_best.pt")))
    if not ckpts:
        print("[ERROR] No checkpoints found!")
        return

    target_ckpt = ckpts[-1]
    print(f"[1] Loading checkpoint: {os.path.basename(target_ckpt)}")
    ckpt = torch.load(target_ckpt, map_location=device, weights_only=False)

    projector = AudioFeatureProjector(
        n_mels=N_MELS, segment_frames=SEGMENT_FRAMES, output_dim=512,
    ).to(device)

    missing, unexpected = projector.load_state_dict(ckpt.get("projector_state", {}), strict=False)
    if missing:
        print(f"  Projector missing keys: {missing}")
    if unexpected:
        print(f"  Projector unexpected keys: {unexpected}")

    print(f"  Stage: {ckpt.get('stage', '?')}, Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Best training loss: {ckpt.get('best_loss', float('inf')):.4f}")
    print()

    print("[2] Loading ESC-50 Mel segments...", flush=True)
    all_entries = load_all_esc50_mel_segments()
    print(f"  Loaded {len(all_entries)} clips from {NUM_CLASSES} classes")
    print()

    print("[3] Building augmentation pipeline...", flush=True)
    circuit_aug, sample_bank = build_augment_pipeline()
    if circuit_aug:
        circuit_aug = circuit_aug.to(device)
    print()

    print("=" * 60)
    print("TEST A: CLEAN (NO augmentation — raw Mel)")
    print("=" * 60)
    print()

    X_clean = []
    y_clean = []
    folds_clean = []

    projector.eval()
    for segments, label, fold, path in all_entries:
        clip_f = extract_features(projector, segments, device)
        X_clean.append(clip_f)
        y_clean.append(label)
        folds_clean.append(fold)

    X_clean = torch.stack(X_clean)
    y_clean = torch.tensor(y_clean, dtype=torch.long)

    print(f"  Extracted {len(X_clean)} clean feature vectors ({X_clean.shape[1]}-dim)")
    acc_clean, folds_clean_acc = linear_probe_5fold_cv(X_clean, y_clean, folds_clean, device)
    print(f"\n  >>> CLEAN 5-fold CV: {acc_clean*100:.1f}% <<<")
    print()

    print("=" * 60)
    print("TEST B: AUGMENTED (circuit + Ableton overlay)")
    print("=" * 60)
    print()

    X_aug = []
    y_aug = []
    folds_aug = []

    projector.eval()
    n_total = len(all_entries)
    for idx, (segments, label, fold, path) in enumerate(all_entries):
        seg_aug = apply_augmentation_to_segments(segments, circuit_aug, sample_bank, device)
        clip_f = extract_features(projector, seg_aug, device)
        X_aug.append(clip_f)
        y_aug.append(label)
        folds_aug.append(fold)
        if (idx + 1) % 400 == 0:
            print(f"  Processing {idx+1}/{n_total}...", flush=True)

    X_aug = torch.stack(X_aug)
    y_aug = torch.tensor(y_aug, dtype=torch.long)

    print(f"  Extracted {len(X_aug)} augmented feature vectors ({X_aug.shape[1]}-dim)")
    acc_aug, folds_aug_acc = linear_probe_5fold_cv(X_aug, y_aug, folds_aug, device)
    print(f"\n  >>> AUGMENTED 5-fold CV: {acc_aug*100:.1f}% <<<")
    print()

    print("=" * 60)
    print("A/B COMPARISON")
    print("=" * 60)
    print(f"  Clean (raw Mel):               {acc_clean*100:.1f}%")
    print(f"  Augmented (circuit + overlay): {acc_aug*100:.1f}%")
    delta = acc_clean - acc_aug
    if delta > 0:
        print(f"  Delta: CLEAN wins by +{delta*100:.1f}%")
        print(f"  → Model is robust: augmentation does not hurt performance")
    else:
        print(f"  Delta: AUGMENTED wins by +{-delta*100:.1f}%")
        print(f"  → Augmentation helps! Model uses circuit cues")

    print()
    for i in range(5):
        c_acc = folds_clean_acc[i] * 100 if i < len(folds_clean_acc) else 0
        a_acc = folds_aug_acc[i] * 100 if i < len(folds_aug_acc) else 0
        print(f"  Fold {i+1}:  clean {c_acc:.1f}%  |  augmented {a_acc:.1f}%  |  Δ {c_acc - a_acc:+.1f}%")

    print()
    print("=" * 60)
    print("CONTEXT: Training History")
    print("=" * 60)
    summary_path = os.path.join(STAGE_CHECKPOINT_DIR, "orchestrator_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        for r in summary:
            if r.get("status") in ("completed", None):
                acc = r.get("mean_accuracy", 0)
                loss = r.get("best_total_loss", float("inf"))
                print(f"  S{r['stage_idx']}: acc={acc*100:.1f}%  loss={loss:.4f}  ({r.get('music_used_pct',0):.0f}% music)")
        print()
        print(f"  NOTE: orchestrator backtest uses ESC-50 val set (fold 5 split)")
        print(f"  This clean/aug eval uses FULL 5-fold CV on all 2000 clips")

    print()
    print("=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    print(f"  Clean accuracy:   {acc_clean*100:.1f}%")
    print(f"  Augmented accuracy: {acc_aug*100:.1f}%")
    print(f"  Gap: {abs(delta)*100:.1f}%")
    if abs(delta) < 2.0:
        print(f"  >> Gap < 2%: augmentation is NEUTRAL — model ignores circuit artifacts")
    elif acc_clean > acc_aug:
        print(f"  >> Clean better: model may rely on genuine audio features")
    else:
        print(f"  >> Augmented better: circuit simulation provides useful inductive bias")
    print()


if __name__ == "__main__":
    main()
