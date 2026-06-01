"""
CVJEPA CPU Training — Causal Visual JEPA on Synthetic Data
=============================================================
Trains the CVJEPA causal visual world model entirely on CPU using
synthetic visual feature sequences. No GPU required — optimized for
~1 sample/second on consumer CPU.

Architecture:
    Synthetic visual features (768d, simulating DINOv2 backbone)
    → CVJEPA (7 object slots, 128d each)
    → Object-level masking → causal prediction → CausalVICRegLoss

This is the VORTEX FLAME Base Layer training — building the causal
visual world model that will later supervise LLM outputs.

Usage:
    python train_cvjepa_cpu.py --epochs 100 --batch 8 --lr 1e-4
    python train_cvjepa_cpu.py --epochs 500 --batch 4 --lr 5e-5 --checkpoint_dir ./checkpoints
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from five_layer_jepa.causal_jepa import CVJEPA
from jepa_training_guard import TrainingGuard

CHECKPOINT_DIR = Path("D:/VORTEX_FLAME/cvjepa_checkpoints")
DEFAULT_INPUT_DIM = 768
DEFAULT_NUM_SLOTS = 7
DEFAULT_SLOT_DIM = 128
SEQUENCE_LENGTH = 10
HISTORY_LEN = 6
FUTURE_LEN = 4


class SyntheticVisualDataset:
    def __init__(self, num_sequences: int = 1000, input_dim: int = 768, seq_len: int = 10):
        self.num_sequences = num_sequences
        self.input_dim = input_dim
        self.seq_len = seq_len
        self._data = None

    def _generate(self):
        base_patterns = torch.randn(20, self.input_dim)
        data = []
        for i in range(self.num_sequences):
            seq = []
            state = torch.randn(self.input_dim) * 0.5
            for t in range(self.seq_len):
                pattern_idx = (i + t) % 20
                state = 0.7 * state + 0.3 * base_patterns[pattern_idx] + 0.1 * torch.randn(self.input_dim)
                state = state / (state.norm() + 1e-8) * 3.0
                seq.append(state.clone())
            data.append(torch.stack(seq))
        self._data = torch.stack(data)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self._data is None:
            self._generate()
        full_seq = self._data[idx]  # type: ignore[reportOptionalSubscript]
        return full_seq[:HISTORY_LEN], full_seq[HISTORY_LEN:HISTORY_LEN + FUTURE_LEN]


def create_cvjepa_model(input_dim: int = DEFAULT_INPUT_DIM) -> CVJEPA:
    return CVJEPA(
        input_dim=input_dim,
        mask_ratio=0.5,
        history_len=HISTORY_LEN,
        future_len=FUTURE_LEN,
        num_predictor_layers=3,
        num_predictor_heads=4,
        ema_decay=0.996,
    )


def train_epoch(model, dataloader, scheduler, epoch: int, guard: TrainingGuard, total_batches: int) -> dict:
    model.train()
    total_loss = 0.0
    total_recovery = 0.0
    total_forward = 0.0
    n_batches = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (history, future) in enumerate(dataloader):
            history = history.float().unsqueeze(2)
            future = future.float().unsqueeze(2)

            global_step = (epoch - 1) * total_batches + batch_idx
            guard.pre_step(global_step)

            loss_dict = model.train_step(history, future)

            loss_val = float(loss_dict.get("total_loss", 0))
            if guard.check_loss(torch.tensor(loss_val)):
                pass

            model.update_ema()
            guard.check_parameters(model)
            guard.post_step(loss_val)

            if scheduler is not None:
                scheduler.step()

            total_loss += loss_val
            total_recovery += float(loss_dict.get("recovery_loss", 0))
            total_forward += float(loss_dict.get("forward_loss", 0))
            n_batches += 1

    elapsed = time.time() - t0
    return {
        "epoch": epoch,
        "avg_loss": total_loss / max(n_batches, 1),
        "avg_recovery": total_recovery / max(n_batches, 1),
        "avg_forward": total_forward / max(n_batches, 1),
        "batches": n_batches,
        "time_sec": round(elapsed, 1),
        "lr": 0.0,
    }


def save_checkpoint(model, epoch: int, metrics: dict, checkpoint_dir: Path):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"cvjepa_e{epoch:04d}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }, path)
    latest = checkpoint_dir / "cvjepa_latest.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }, latest)
    print(f"  Saved checkpoint: {path}")

    history_path = checkpoint_dir / "training_history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def main():
    parser = argparse.ArgumentParser(description="CVJEPA CPU Training")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sequences", type=int, default=2000, help="Synthetic sequences to generate")
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR), help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--input_dim", type=int, default=DEFAULT_INPUT_DIM)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"=== CVJEPA CPU Training ===")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch}, LR: {args.lr}")
    print(f"  Input dim: {args.input_dim}, Slots: {DEFAULT_NUM_SLOTS} (fixed), Slot dim: {DEFAULT_SLOT_DIM} (fixed)")
    print(f"  Sequences: {args.sequences} (synthetic)")
    print()

    model = create_cvjepa_model(args.input_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {type(model.context_encoder).__name__}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print()

    dataset = SyntheticVisualDataset(num_sequences=args.sequences, input_dim=args.input_dim)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)  # type: ignore[reportArgumentType]

    checkpoint_dir = Path(args.checkpoint_dir)
    total_steps = args.epochs * len(dataloader)
    guard = TrainingGuard(
        total_steps=total_steps,
        warmup_steps=min(50, total_steps // 4),
        checkpoint_dir=str(checkpoint_dir),
    )
    best_loss = float("inf")
    total_batches_per_epoch = len(dataloader)

    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, dataloader, None, epoch, guard, total_batches_per_epoch)

        status = " "
        if metrics["avg_loss"] < best_loss:
            best_loss = metrics["avg_loss"]
            status = "★"

        print(f"Epoch {epoch:4d}/{args.epochs} | "
              f"Loss: {metrics['avg_loss']:.4f} | Rec: {metrics['avg_recovery']:.4f} | "
              f"Fwd: {metrics['avg_forward']:.4f} | "
              f"Time: {metrics['time_sec']:5.1f}s | LR: {metrics['lr']:.2e}{status}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(model, epoch, metrics, checkpoint_dir)

    final_path = checkpoint_dir / "cvjepa_final.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": args.input_dim,
            "num_slots": DEFAULT_NUM_SLOTS,
            "slot_dim": DEFAULT_SLOT_DIM,
        },
        "best_loss": best_loss,
    }, final_path)

    print(f"\n=== Training Complete ===")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Final model: {final_path}")


if __name__ == "__main__":
    main()
