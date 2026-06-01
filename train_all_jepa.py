#!/usr/bin/env python3
"""
VORTEX FLAME — Batch JEPA Trainer
====================================
One-click training for ALL 9 C-JEPA variants sequentially.

Trains each variant on synthetic temporal data (Phase 1: pre-training),
saves checkpoints, and logs progress.

Usage:
  python train_all_jepa.py --epochs 100 --batch 8 --lr 1e-4
  python train_all_jepa.py --epochs 100 --batch 4 --lr 5e-5  (conservative)
  python train_all_jepa.py --list                              (show what will run)

Order (by P0→P1 priority):
  1. physjepa   (einstein + galileo, 物理)       P0
  2. codejepa   (cezanne, 代码)                  P0
  3. biojepa    (darwin + yuanlongping, 生物)    P0
  4. lawjepa    (guizhu + montesquieu, 法律)     P1
  5. geojepa    (humboldt + herodotus, 地理)     P1
  6. artjepa    (monet + vangogh, 艺术)          P1
  7. designjepa (davinci, 设计)                  P1
  8. finjepa    (strategy, 金融) —— 已有脚本但数据可用              8. finjepa    (strategy, 金融) —— 已有脚本但数据可用同步跑
  9. cvjepa     (davinci + herodotus, 视觉) —— 已有脚本但数据可用    9. cvjepa     (davinci + herodotus, 视觉) —— 已有脚本但数据可用同步跑

Note: CAJEPA (beethoven) is already running long-term training — not included here.
      train_ajepa.py is running in a separate terminal.

Architecture per variant (fixed config):
  - ObjectSlotEncoder → ObjectLevelMasker → CausalPredictor
  - Loss: SIGRegWithPredictionLoss (2026 LeJEPA style: MSE + SIGReg)
  - EMA: target encoder momentum 0.996
  - ~6M params, ~30min/100 epochs on V100-16GB
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent

TRAIN_ORDER = [
    {"name": "physjepa",   "script": "train_physjepa.py",   "souls": "einstein, galileo",       "desc": "物理世界模型"},
    {"name": "codejepa",   "script": "train_codejepa.py",   "souls": "cezanne",                  "desc": "代码世界模型"},
    {"name": "biojepa",    "script": "train_biojepa.py",    "souls": "darwin, yuanlongping",     "desc": "生物世界模型"},
    {"name": "lawjepa",    "script": "train_lawjepa.py",    "souls": "guizhu, montesquieu",      "desc": "法律世界模型"},
    {"name": "geojepa",    "script": "train_geojepa.py",    "souls": "humboldt, herodotus",      "desc": "地理世界模型"},
    {"name": "artjepa",    "script": "train_artjepa.py",    "souls": "monet, vangogh",           "desc": "艺术世界模型"},
    {"name": "designjepa", "script": "train_designjepa.py", "souls": "davinci",                  "desc": "设计世界模型"},
    {"name": "finjepa",    "script": "train_finjepa.py",    "souls": "strategy",                 "desc": "金融世界模型"},
    {"name": "cvjepa",     "script": "train_cvjepa_cpu.py", "souls": "davinci, herodotus",       "desc": "视觉世界模型"},
]

DT_FORMAT = "%Y-%m-%d %H:%M:%S"

def log(msg):
    ts = time.strftime(DT_FORMAT)
    print(f"[{ts}] {msg}", flush=True)

def train_one(info, epochs, batch, lr, sequences, cpu):
    log(f"{'='*60}")
    log(f"START  {info['name']:>12} | {info['desc']} | souls={info['souls']}")
    log(f"       epochs={epochs} batch={batch} lr={lr} sequences={sequences} cpu={cpu}")
    log(f"{'='*60}")

    t0 = time.time()
    cmd = [sys.executable, info["script"],
           "--epochs", str(epochs),
           "--batch", str(batch),
           "--lr", str(lr),
           "--sequences", str(sequences)]
    if cpu:
        cmd.append("--cpu")
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=False, text=True)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL (code={result.returncode})"

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    log(f"DONE   {info['name']:>12} | {status} | {mins}m{secs}s")
    log("")
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Batch C-JEPA Trainer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sequences", type=int, default=3000)
    parser.add_argument("--list", action="store_true", help="List training order and exit")
    parser.add_argument("--start-from", type=str, default="",
                        help="Start from a specific variant name (skip earlier ones)")
    parser.add_argument("--only", type=str, default="",
                        help="Only train this one variant (comma-separated for multiple)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training for all variants")
    args = parser.parse_args()

    if args.list:
        print("\nC-JEPA Batch Training Order:\n")
        for i, info in enumerate(TRAIN_ORDER, 1):
            print(f"  {i:>2}. {info['name']:>12} → {info['desc']:>20} | souls: {info['souls']}")
        print(f"\n  Total: {len(TRAIN_ORDER)} variants")
        print(f"  Per-variant: ~{args.epochs} epochs, ~6M params each")
        print(f"  Estimated total: ~{len(TRAIN_ORDER) * 0.5:.0f}h on V100-16GB\n")
        return

    if args.only:
        targets = [t.strip() for t in args.only.split(",")]
        order = [info for info in TRAIN_ORDER if info["name"] in targets]
        if not order:
            print(f"Error: no matching variants for --only '{args.only}'")
            print(f"Valid names: {[t['name'] for t in TRAIN_ORDER]}")
            sys.exit(1)
        log(f"ONLY mode: {[t['name'] for t in order]}")
    else:
        order = list(TRAIN_ORDER)
        if args.start_from:
            try:
                idx = next(i for i, t in enumerate(order) if t["name"] == args.start_from)
                order = order[idx:]
            except StopIteration:
                print(f"Error: --start-from '{args.start_from}' not found")
                sys.exit(1)

    log(f"BATCH TRAINING: {len(order)} variants")
    log(f"Config: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
    total_start = time.time()

    ok = 0
    fail = 0
    for info in order:
        success = train_one(info, args.epochs, args.batch, args.lr, args.sequences, args.cpu)
        if success:
            ok += 1
        else:
            fail += 1

    total_elapsed = time.time() - total_start
    total_mins = int(total_elapsed // 60)
    log(f"{'='*60}")
    log(f"BATCH COMPLETE: {ok} OK, {fail} FAILED out of {len(order)}")
    log(f"Total time: {total_mins}m ({total_elapsed/3600:.1f}h)")
    log(f"{'='*60}")

if __name__ == "__main__":
    main()
