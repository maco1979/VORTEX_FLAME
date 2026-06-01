"""
CVJEPA Real Video Training — 3DPW Dataset
============================================
Trains CVJEPA on real 3D human motion videos from the 3DPW dataset.

Pipeline:
    3DPW frames (zip) → Lightweight ConvNet → 768-d features
    → CVJEPA (7 slots, 128d) → SIGReg causal prediction

Architecture:
    RealVideoFeatureExtractor: 4-layer ConvNet (112x112→768d)
    3DPWDataset: on-the-fly frame loading from zip, random sequence sampling
    CVJEPA: object-level causal world model

Usage:
    python train_cvjepa_real.py --epochs 50 --batch 4 --lr 1e-4
    python train_cvjepa_real.py --cpu --epochs 10 --batch 2
"""

import argparse, json, os, sys, time, random, math, zipfile
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from five_layer_jepa.causal_jepa import CVJEPA
from five_layer_jepa.causal_jepa_v2 import SIGRegWithPredictionLoss
from jepa_training_guard import TrainingGuard

DATA_DIR = Path("D:/VORTEX_FLAME/_data/3DPW")
CHECKPOINT_DIR = Path("D:/VORTEX_FLAME/cvjepa_real_checkpoints")

INPUT_DIM = 768
NUM_SLOTS = 7
SLOT_DIM = 128
HISTORY_LEN = 6
FUTURE_LEN = 4
SEQ_LEN = HISTORY_LEN + FUTURE_LEN
FRAME_SIZE = 112


class RealVideoFeatureExtractor(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.norm(self.proj(x))
        if 'T' in dir():
            x = x.reshape(B, T, -1)
        return x


class ThreeDPWDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, history_len=6, future_len=4, frame_size=112):
        self.data_dir = Path(data_dir)
        self.history_len = history_len
        self.future_len = future_len
        self.seq_len = history_len + future_len
        self.frame_size = frame_size

        self.transform = transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        zip_path = self.data_dir / "imageFiles.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"3DPW data not found at {zip_path}")

        self._zf = zipfile.ZipFile(zip_path, 'r')
        self._build_index()

        total_windows = sum(max(0, n - self.seq_len) for n in self._seq_frames.values())
        print(f"3DPW Dataset: {len(self._seq_names)} sequences, {total_windows:,} valid windows")

    def _build_index(self):
        seq_frames = defaultdict(list)
        for name in self._zf.namelist():
            if not name.endswith('.jpg'):
                continue
            parts = name.split('/')
            if len(parts) >= 3:
                seq_name = parts[1]
                try:
                    frame_num = int(parts[2].replace('image_', '').replace('.jpg', ''))
                except ValueError:
                    continue
                seq_frames[seq_name].append((frame_num, name))

        self._seq_frames = {}
        self._seq_names = []
        for seq_name, frames in seq_frames.items():
            frames.sort()
            if len(frames) >= self.seq_len:
                self._seq_frames[seq_name] = len(frames)
                self._seq_names.append(seq_name)

    def __len__(self):
        return len(self._seq_names) * 64

    def _load_frame(self, path_in_zip):
        data = self._zf.read(path_in_zip)
        img = Image.open(BytesIO(data)).convert('RGB')
        return self.transform(img)

    def __getitem__(self, idx):
        seq_name = self._seq_names[idx % len(self._seq_names)]
        max_frames = self._seq_frames[seq_name]
        max_offset = max(0, max_frames - self.seq_len)
        offset = random.randint(0, max_offset)

        frames = []
        for i in range(self.seq_len):
            frame_num = offset + i
            path = f"imageFiles/{seq_name}/image_{frame_num:05d}.jpg"
            frames.append(self._load_frame(path))

        frames = torch.stack(frames, dim=0)
        history = frames[:self.history_len]
        future = frames[self.history_len:self.history_len + self.future_len]
        return history, future, seq_name


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    extractor = RealVideoFeatureExtractor(output_dim=INPUT_DIM).to(device)
    cvjepa = CVJEPA(input_dim=INPUT_DIM).to(device)

    total_params = sum(p.numel() for p in list(extractor.parameters()) + list(cvjepa.parameters()))
    trainable = sum(p.numel() for p in list(extractor.parameters()) + list(cvjepa.parameters()) if p.requires_grad)
    print(f"Params: {total_params:,} total, {trainable:,} trainable")
    print(f"  Extractor: {sum(p.numel() for p in extractor.parameters()):,}")
    print(f"  CVJEPA:    {sum(p.numel() for p in cvjepa.parameters()):,}")

    dataset = ThreeDPWDataset(DATA_DIR, HISTORY_LEN, FUTURE_LEN, FRAME_SIZE)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda", drop_last=True,
    )

    sigreg = SIGRegWithPredictionLoss(var_weight=25.0, cov_weight=1.0, sim_weight=1.0)
    all_params = list(extractor.parameters()) + list(cvjepa.parameters())
    optimizer = Adam(all_params, lr=args.lr, weight_decay=1e-5)

    steps_per_epoch = len(dataloader)
    T_0 = 5 * steps_per_epoch
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=1e-6)

    guard = TrainingGuard(
        total_steps=args.epochs * len(dataloader),
        warmup_steps=200,
        loss_spike_factor=5.0,
        max_grad_norm=2.0,
        nan_tolerance=20,
        collapse_patience=100,
        checkpoint_dir=str(CHECKPOINT_DIR),
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log_path = CHECKPOINT_DIR / "training_log.jsonl"
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        cvjepa.train()
        extractor.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (history_raw, future_raw, seq_names) in enumerate(dataloader):
            guard.pre_step(global_step)

            B, T_hist, C, H, W = history_raw.shape
            _, T_fut, _, _, _ = future_raw.shape

            history_features = extractor(history_raw)
            future_features = extractor(future_raw)

            hf = history_features.unsqueeze(2).expand(-1, -1, NUM_SLOTS, -1)
            ff = future_features.unsqueeze(2).expand(-1, -1, NUM_SLOTS, -1)

            with torch.no_grad():
                target_hist, _ = cvjepa.target_encoder(hf)
                target_fut, _ = cvjepa.target_encoder(ff)

            ctx_hist, _ = cvjepa.context_encoder(hf)
            masked, slot_mask, _ = cvjepa.masker(ctx_hist)
            recovered, predicted = cvjepa.predictor(masked)

            rec_loss = sigreg(
                recovered.reshape(-1, SLOT_DIM),
                target_hist.reshape(-1, SLOT_DIM),
            )
            T_match = min(predicted.shape[1], target_fut.shape[1])
            fwd_loss = sigreg(
                predicted[:, :T_match].reshape(-1, SLOT_DIM),
                target_fut[:, :T_match].reshape(-1, SLOT_DIM),
            )
            total_loss = rec_loss["total"] + 0.5 * fwd_loss["total"]

            if guard.check_loss(total_loss):
                guard.state.skipped_batches += 1
                global_step += 1
                continue

            optimizer.zero_grad()
            total_loss.backward()

            if guard.check_gradients(cvjepa) or guard.check_gradients(extractor):
                optimizer.zero_grad()
                guard.state.skipped_batches += 1
                global_step += 1
                continue

            torch.nn.utils.clip_grad_norm_(all_params, 2.0)
            optimizer.step()
            scheduler.step()

            cvjepa.update_ema()

            epoch_loss += total_loss.item()
            global_step += 1

            if batch_idx % 100 == 0 and batch_idx > 0:
                avg = epoch_loss / (batch_idx + 1)
                elapsed = time.time() - t0
                print(f"  E{epoch} B{batch_idx}/{steps_per_epoch} "
                      f"loss={avg:.4f} rec={rec_loss['total'].item():.4f} "
                      f"fwd={fwd_loss['total'].item():.4f} "
                      f"gap={fwd_loss['total'].item()/max(rec_loss['total'].item(),0.01):.2f}x "
                      f"{elapsed:.0f}s", flush=True)

        avg_loss = epoch_loss / max(steps_per_epoch, 1)
        elapsed = time.time() - t0
        print(f"E{epoch}/{args.epochs} avg_loss={avg_loss:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} {elapsed:.0f}s", flush=True)

        with open(log_path, "a") as f:
            log_entry = json.dumps({
                "epoch": epoch, "global_step": global_step,
                "avg_loss": avg_loss, "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": elapsed,
            })
            f.write(log_entry + "\n")

        if epoch % 10 == 0:
            ckpt = {
                "epoch": epoch, "global_step": global_step,
                "extractor_state": extractor.state_dict(),
                "cvjepa_state": cvjepa.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(ckpt, CHECKPOINT_DIR / f"cvjepa_real_e{epoch:03d}.pt")
            print(f"  Checkpoint saved: cvjepa_real_e{epoch:03d}.pt", flush=True)

    final_ckpt = {
        "extractor_state": extractor.state_dict(),
        "cvjepa_state": cvjepa.state_dict(),
    }
    torch.save(final_ckpt, CHECKPOINT_DIR / "cvjepa_real_best.pt")
    print(f"\nTraining complete. Best model: {CHECKPOINT_DIR / 'cvjepa_real_best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)
