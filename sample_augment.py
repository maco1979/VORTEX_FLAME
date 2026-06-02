"从Ableton短采样提取的频谱多样性注入器 — 预缓存Mel到内存 + 向量化"
import os
import random
import torch
import torchaudio
import torch.nn as nn
from pathlib import Path

SAMPLE_DIR = os.getenv("ALP_OUTPUT", "./ableton_samples")
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SEGMENT_FRAMES = 256


class ShortSampleBank:
    def __init__(
        self,
        sample_dir=SAMPLE_DIR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        sample_rate=SAMPLE_RATE,
        segment_frames=SEGMENT_FRAMES,
        max_load=2000,
    ):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.segment_frames = segment_frames
        self.max_load = max_load

        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2,
        )
        self._amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self._mel_cache = []
        self._preload_all(sample_dir)
        print(f"[ShortSampleBank] 预缓存 {len(self._mel_cache)} 个Mel频谱 (总{sum(m.shape[2] for m in self._mel_cache)}帧)")
        self._index = list(range(len(self._mel_cache)))

    def _preload_all(self, sample_dir):
        import glob
        exts = ("*.ogg", "*.wav", "*.aif", "*.aiff", "*.mp3", "*.flac")
        all_files = []
        for ext in exts:
            all_files.extend(glob.glob(os.path.join(sample_dir, ext)))

        random.shuffle(all_files)
        count = 0
        for fp in all_files:
            if count >= self.max_load:
                break
            try:
                waveform, sr = torchaudio.load(fp)
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                waveform = waveform.mean(dim=0, keepdim=True)
                mel = self._amplitude_to_db(self._mel_transform(waveform))
                n_frames = mel.shape[2]
                if n_frames >= 4:
                    self._mel_cache.append(mel)
                    count += 1
            except Exception:
                pass

    def random_sample_mel(self, max_duration_frames=None, device=None):
        if not self._mel_cache:
            return None
        mel = random.choice(self._mel_cache)
        n_frames = mel.shape[2]
        if max_duration_frames and n_frames > max_duration_frames:
            start = random.randint(0, n_frames - max_duration_frames)
            mel = mel[:, :, start:start + max_duration_frames]
        if device is not None:
            mel = mel.to(device)
        return mel


def overlay_short_samples(mel, sample_bank, p=0.5, max_samples=3, snr_db_range=(-12, -3)):
    if random.random() > p:
        return mel
    if sample_bank._mel_cache is None or len(getattr(sample_bank, '_mel_cache', [])) == 0:
        return mel

    unsqueezed = False
    was_4d = False
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
        unsqueezed = True
    elif mel.dim() == 4:
        B, T_hist, N, F_mel = mel.shape
        mel = mel.reshape(B * T_hist, N, F_mel)
        was_4d = True
        T_hist_val = T_hist

    B, N, T = mel.shape
    device = mel.device

    for b in range(B):
        n_overlay = random.randint(1, max_samples)
        for _ in range(n_overlay):
            short = sample_bank.random_sample_mel(max_duration_frames=T, device=device)
            if short is None:
                continue

            sf = short.shape[2]
            if sf == 0:
                continue
            t_start = random.randint(0, max(1, T - sf))
            t_end = min(T, t_start + sf)
            seg_len = t_end - t_start
            if seg_len <= 0:
                continue

            snr_db = random.uniform(*snr_db_range)
            scale = 10 ** (snr_db / 20)
            short_slice = short[:, :, :seg_len]
            if short_slice.shape[0] == 1:
                short_slice = short_slice.squeeze(0)
            mel[b, :, t_start:t_end] += short_slice * scale

    if unsqueezed:
        mel = mel.squeeze(0)
    elif was_4d:
        mel = mel.reshape(B // T_hist_val, T_hist_val, N, T)

    return mel


def batch_overlay_short_samples(history_mel, future_mel, sample_bank, p=0.5, max_samples=3):
    B, Th, N, F = history_mel.shape
    h_flat = history_mel.reshape(B * Th, N, F)
    h_aug = overlay_short_samples(h_flat, sample_bank, p=p, max_samples=max_samples)
    if h_aug.dim() == 2:
        h_aug = h_aug.unsqueeze(0)
    history_mel = h_aug.reshape(B, Th, N, F)

    Bf, Tf, _, _ = future_mel.shape
    f_flat = future_mel.reshape(Bf * Tf, N, F)
    f_aug = overlay_short_samples(f_flat, sample_bank, p=p, max_samples=max_samples)
    if f_aug.dim() == 2:
        f_aug = f_aug.unsqueeze(0)
    future_mel = f_aug.reshape(Bf, Tf, N, F)

    return history_mel, future_mel


def short_sample_contrastive_pairs(sample_bank, n_pairs=32, device=None):
    anchors = []
    positives = []
    for _ in range(n_pairs):
        s = sample_bank.random_sample_mel(max_duration_frames=SEGMENT_FRAMES, device=device)
        if s is None:
            continue
        mel = s.squeeze(0)
        if mel.dim() == 2:
            import torch.nn.functional as F
            if mel.shape[1] < SEGMENT_FRAMES:
                mel = F.pad(mel, (0, SEGMENT_FRAMES - mel.shape[1]))
            else:
                mel = mel[:, :SEGMENT_FRAMES]

        anchors.append(mel)
        noise = torch.randn_like(mel) * 0.02
        positives.append(mel + noise)

    if not anchors:
        return None, None

    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    return anchors, positives


def build_sample_augment(max_cache=2000):
    return ShortSampleBank(max_load=max_cache)
