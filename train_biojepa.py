#!/usr/bin/env python3
"""
CBIOJEPA Training — Causal Biology JEPA
=========================================
Trains the biology world model on synthetic gene/protein dynamics.

Architecture:
  Simulated biological system features (gene expression, protein states)
  → CBIOJEPA (7 object slots, 128d each)
  → Object-level masking → causal prediction → CausalVICRegLoss

Serves: darwin (生物/进化), yuanlongping (农业/作物)

Usage:
  python train_biojepa.py --epochs 100 --batch 8 --lr 1e-4
"""
import argparse, os, sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from jepa_trainer_base import SyntheticJEPADataset, train_jepa, get_jepa_model, VARIANT_CONFIG

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "biojepa_checkpoints")

def main():
    parser = argparse.ArgumentParser(description="CBIOJEPA Training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sequences", type=int, default=3000)
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    model, cfg = get_jepa_model("biojepa",
        device=torch.device("cpu") if args.cpu else None)
    dataset = SyntheticJEPADataset(
        num_sequences=args.sequences,
        input_dim=cfg["input_dim"],
        num_modules=8,
        seq_len=cfg["history_len"] + cfg["future_len"] + 2,
        history_len=cfg["history_len"],
        future_len=cfg["future_len"],
    )

    train_jepa(model, dataset, args.checkpoint_dir,
               variant_name="biojepa",
               epochs=args.epochs, batch=args.batch, lr=args.lr,
               device="cpu" if args.cpu else None)

if __name__ == "__main__":
    main()
