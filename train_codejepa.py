#!/usr/bin/env python3
"""
CCODEJEPA Training — Causal Code JEPA
=======================================
Trains the code world model on synthetic code execution traces.

Architecture:
  Simulated code object features (AST nodes, execution states)
  → CCODEJEPA (6 object slots, 128d each)
  → Object-level masking → causal prediction → CausalVICRegLoss

Serves: cezanne (代码/软件工程)

Usage:
  python train_codejepa.py --epochs 100 --batch 8 --lr 1e-4
"""
import argparse, os, sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from jepa_trainer_base import SyntheticJEPADataset, train_jepa, get_jepa_model, VARIANT_CONFIG

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "codejepa_checkpoints")

def main():
    parser = argparse.ArgumentParser(description="CCODEJEPA Training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sequences", type=int, default=3000)
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    model, cfg = get_jepa_model("codejepa",
        device=torch.device("cpu") if args.cpu else None)
    dataset = SyntheticJEPADataset(
        num_sequences=args.sequences,
        input_dim=cfg["input_dim"],
        num_modules=8,
        seq_len=cfg["history_len"] + cfg["future_len"] + 2,
        history_len=cfg["history_len"],
        future_len=cfg["future_len"],
        module_dim=96,
    )

    train_jepa(model, dataset, args.checkpoint_dir,
               variant_name="codejepa",
               epochs=args.epochs, batch=args.batch, lr=args.lr,
               device="cpu" if args.cpu else None)

if __name__ == "__main__":
    main()
