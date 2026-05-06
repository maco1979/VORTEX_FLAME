---
name: "soul-deep-expansion"
description: "Universal soul deep expansion training (24L->150L). Invoke when expanding any soul model depth, starting deep training, or chaining 48L->96L->150L."
---

# Soul Deep Expansion - Universal Training Protocol

## Overview

Universal method to expand any soul model from 24L to 150L using slice training.
One script works for ALL 9 souls. Auto GPU profile detection + parameter adaptation.

## Quick Start

```bash
# Einstein to 150L on GPU 0 (auto-chains 48L->96L->150L)
python train_soul_deep.py --soul einstein --target 150 --gpu 0 --auto-chain

# Beethoven to 48L on GPU 1 (auto-detects 1060, adjusts batch/grad_accum)
python train_soul_deep.py --soul beethoven --target 48 --gpu 1

# Cezanne to 48L on GPU 1
python train_soul_deep.py --soul cezanne --target 48 --gpu 1
```

## GPU Auto-Detection

Script auto-detects GPU model and adjusts parameters:

| GPU Profile | Role | Max Depth | 48L Config | 96L Config | 150L Config |
|-------------|------|-----------|-----------|-----------|-------------|
| RTX 3060 12GB | deep_expansion | 150L | batch=8, accum=2, slice=8L | batch=8, accum=2, slice=8L | batch=6, accum=4, slice=6L |
| GTX 1060 6GB | base_training | 48L | batch=2, accum=4, slice=8L | N/A | N/A |

Auto-adjusts target if GPU can't handle requested depth (e.g., 150L on 1060 auto-reduces to 48L).

## Dual GPU Schedule (6 Rounds)

```
Round  | GPU 0 (3060 12GB)              | GPU 1 (1060 6GB)
-------|--------------------------------|-------------------------------
  1    | Einstein 24L->150L             | Beethoven 24L base
  2    | DaVinci 24L->150L              | Cezanne 24L->48L
  3    | Guizhu 24L->130L               | FKJ 24L->48L
  4    | Einstein 150L refine           | Monet 24L->48L
  5    | DaVinci 150L refine            | Strategy 24L->48L
  6    | Guizhu 130L refine             | VanGogh 24L->48L
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| --soul | required | Soul name (einstein, beethoven, cezanne, davinci, fkj, guizhu, monet, strategy, vangogh) |
| --target | 150 | Target depth: 48, 96, or 150 |
| --gpu | 0 | GPU device index |
| --auto-chain | false | Auto chain to next stage after completion |
| --hidden-dim | 768 | Hidden dimension |
| --num-heads | 12 | Number of attention heads |
| --ffn-dim | 3072 | FFN dimension |

## Expansion Stages

| Stage | Depth | Params | Slice Size | Slices | Batch | LR | Continuous Epochs |
|-------|-------|--------|-----------|--------|-------|-----|-------------------|
| 1 | 24L->48L | 348M | 8L | 3 | 8 | 5e-5 | 200 |
| 2 | 48L->96L | 696M | 8L | 6 | 8 | 5e-5 | 300 |
| 3 | 96L->150L | 1.1B | 6L | 9 | 6 | 3e-5 | 500 |

## Training Protocol (per stage)

```
Phase 1: Slice Training
  - Freeze all layers except current slice
  - Train each slice for 10 epochs
  - Save checkpoint after each slice

Phase 2: Fusion Training
  - Unfreeze all layers
  - Soul layers (L0-11) use LR x 0.1
  - New layers use full LR
  - Train 50 epochs
  - Save checkpoint every 2 epochs

Phase 3: Continuous Training
  - Continue training until Loss < 1.5
  - Save checkpoint every 10 epochs
  - Auto-stop when target reached

Phase 4: AB Test
  - Compare current depth vs previous depth
  - Log improvement percentage
  - Determine if expansion was successful
```

## VRAM Requirements

| Model | Full Model FP16 | Slice 8L Training | Slice 6L Training | 3060 OK | 1060 OK |
|-------|----------------|-------------------|-------------------|---------|---------|
| 48L | ~696MB | ~2950MB | - | YES | YES (batch=2) |
| 96L | ~1392MB | ~3700MB | - | YES | NO |
| 150L | ~2200MB | - | ~1850MB | YES | NO |

## Directory Structure

```
D:\VORTEX_FLAME\
  train_soul_deep.py                    # Universal training script
  checkpoints_{soul}_48l/latest.pt      # 48L checkpoint
  checkpoints_{soul}_96l/latest.pt      # 96L checkpoint
  checkpoints_{soul}_150l/latest.pt     # 150L checkpoint
  soul_training_data/{soul}/{soul}_hq_10k.json  # Training data
  {soul}_deep_log.txt                   # Training log
```

## Auto-Resume

Script automatically detects current depth by checking checkpoint directories.
If checkpoint exists, resumes from that phase. No manual intervention needed.

## Key Rules

1. SOUL_LAYERS=12 always frozen during slice training (soul protection)
2. Soul layers use LR x 0.1 during fusion/continuous (fine-tune, not overwrite)
3. Target Loss = 1.5 for all souls
4. Each soul needs >= 10000 training entries for best results
5. GPU Preloaded Dataset: all data in VRAM, zero disk reads during training
6. Slice size shrinks from 8L to 6L at 150L stage (VRAM constraint)
7. GPU auto-profile: 3060 does deep (150L), 1060 does base (48L max)
8. If target > GPU max_depth, auto-adjusts target down
