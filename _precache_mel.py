#!/usr/bin/env python3
"""Precompute Mel spectrograms for all audio files and cache as .pt"""
import os
import sys
import hashlib
import math
import csv
import time
import torch
import torchaudio

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
CACHE_DIR = r"E:\AI_Data\mel_cache"

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SEGMENT_FRAMES = 256
MIN_SEGMENTS = 10
MIN_SAMPLES = MIN_SEGMENTS * SEGMENT_FRAMES * HOP_LENGTH  # ~1.3M samples
ESC50_TARGET_DURATION = 65.0
MUSIC_LOAD_SEC = MIN_SEGMENTS * SEGMENT_FRAMES * HOP_LENGTH / SAMPLE_RATE + 5  # ~65s


def cache_key(filepath):
    return hashlib.md5(filepath.encode()).hexdigest()[:16]


def load_fragment(fp, target_sec):
    """Load up to target_sec seconds from a file, silent errors."""
    from contextlib import redirect_stderr
    try:
        with open(os.devnull, "w") as fnull:
            with redirect_stderr(fnull):
                wav, sr = torchaudio.load(fp, num_frames=int(target_sec * SAMPLE_RATE))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav
    except Exception:
        return None


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "music"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "esc50"), exist_ok=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    audio_exts = {".mp3", ".flac", ".wav", ".m4a", ".ogg"}

    # --- Scan music files ---
    music_files = []
    seen_names = set()
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
                                continue
                            seen_names.add(key)
                            music_files.append(fp)
                    except Exception:
                        pass

    print(f"Precaching {len(music_files)} music files ({MUSIC_LOAD_SEC:.0f}s fragment each)...")
    music_success = 0
    t0 = time.time()
    for i, fp in enumerate(music_files):
        key = cache_key(fp)
        cache_path = os.path.join(CACHE_DIR, "music", f"{key}.pt")
        if os.path.exists(cache_path):
            music_success += 1
            continue
        wav = load_fragment(fp, MUSIC_LOAD_SEC)
        if wav is None:
            continue
        with torch.no_grad():
            mel = mel_transform(wav)
            mel_db = amp_to_db(mel).squeeze(0)
        n_frames = mel_db.shape[1]
        if n_frames < MIN_SEGMENTS * SEGMENT_FRAMES:
            pad = torch.zeros(N_MELS, MIN_SEGMENTS * SEGMENT_FRAMES - n_frames)
            mel_db = torch.cat([mel_db, pad], dim=1)
        torch.save(mel_db, cache_path)
        music_success += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(music_files) - i - 1) / rate if rate > 0 else 0
            print(f"  music: {music_success}/{i+1} ({rate:.1f} files/s, ETA {eta/60:.0f}min)", flush=True)
    print(f"Music done: {music_success}/{len(music_files)} cached", flush=True)

    # --- ESC-50 files ---
    esc50_entries = []
    with open(ESC50_META, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fp = os.path.join(ESC50_AUDIO_DIR, row["filename"])
            if os.path.exists(fp):
                esc50_entries.append((fp, int(row["fold"]), int(row["target"]), row["category"]))

    print(f"Precaching {len(esc50_entries)} ESC-50 files...")
    esc50_success = 0
    t0 = time.time()
    for i, (fp, fold, label, cat) in enumerate(esc50_entries):
        key = cache_key(fp)
        cache_path = os.path.join(CACHE_DIR, "esc50", f"{key}.pt")
        if os.path.exists(cache_path):
            esc50_success += 1
            continue
        wav = load_fragment(fp, ESC50_TARGET_DURATION)
        if wav is None:
            continue
        wav = wav.squeeze(0)
        target_samples = int(ESC50_TARGET_DURATION * SAMPLE_RATE)
        if wav.shape[0] < target_samples:
            repeats = math.ceil(target_samples / wav.shape[0])
            wav = wav.repeat(repeats)[:target_samples]
        with torch.no_grad():
            mel = mel_transform(wav.unsqueeze(0))
            mel_db = amp_to_db(mel).squeeze(0)
        torch.save(mel_db, cache_path)
        esc50_success += 1
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(esc50_entries) - i - 1) / rate if rate > 0 else 0
            print(f"  esc50: {esc50_success}/{len(esc50_entries)} ({rate:.1f} files/s, ETA {eta/60:.0f}min)", flush=True)
    print(f"ESC-50 done: {esc50_success}/{len(esc50_entries)} cached", flush=True)

    print(f"\nTotal cached: {music_success} music + {esc50_success} esc50", flush=True)
    print("DONE", flush=True)
