# VORTEX_FLAME JEPA Architecture Specification

> **Version**: 2.0 | **Date**: 2026-05-30 | **Updated by**: ICML 2026 C-JEPA + V-JEPA 2 + LeWorldModel integration

## 1. JEPA Lineage & Provenance

VORTEX_FLAME's C-JEPA implementation follows the Joint Embedding Predictive Architecture paradigm proposed by Yann LeCun (2022), with direct lineage to the ICML 2026 C-JEPA paper.

```text
LeCun's JEPA Vision (2022)
    ├── I-JEPA (2023): Image patch-level masked prediction, ImageNet 84.2%
    ├── V-JEPA (2024): Video spatiotemporal masked prediction
    ├── V-JEPA 2 (2025.06): 1M hours video + action-conditioned, zero-shot robot control
    ├── C-JEPA (ICML 2026): Object-level masked prediction + causal intervention
    │   └── arXiv:2602.11389, LeCun/Nam/Maes/Lidec/Balestriero
    ├── LeWorldModel (2026.03): End-to-end 15M params, 2-loss, single GPU
    │   └── arXiv:2603.19312
    └── IA-JEPA (2026.05): Interaction-aware masking for physics reasoning
```

## 2. Core Architecture

### 2.1 Object-Centric Slot Attention

```
Input features (B, N_input, D) → SlotAttention iterations → Object slots (B, N_slots, 128)
```

The encoder binds input features to a fixed number of object slots via iterative attention:
1. Initialize slots from learnable Gaussian distribution
2. Compute slot-to-feature attention weights
3. Softmax-normalize across slots (competition forces slots to specialize)
4. Weighted sum of features → GRU-updated slots
5. Repeat 3 iterations (configurable)

### 2.2 Object-Level Masking (Causal Intervention)

Instead of masking patches (I-JEPA), we mask **entire objects across time**:

```text
Observable slots:     [o1, o4, o5]  
Masked slots:          [o2, o3]
                ↓
Predictor must infer state of o2, o3 from o1, o4, o5
                ↓
This induces latent intervention = counterfactual reasoning
```

**Why this works** (formally proven in C-JEPA paper):
- Masking an entire object = do(o=blank) in Pearl's do-calculus
- Predictor must use cross-object relationships, not self-dynamics
- Prevents shortcut solutions (copying own history)
- Interaction reasoning becomes NECESSARY, not optional

### 2.3 CausalPredictor (Transformer Backbone)

```text
Input:  Masked slot history (B, T, N, slot_dim=128)
        + Temporal positional encoding
        + Slot identity encoding

Layers: 4× Transformer blocks
        - Self-Attention (Multihead, 4 heads)
        - FFN (256→1024→256, GELU)
        - LayerNorm pre-norm

Output: Recovered masked slots + Predicted future slots
```

### 2.4 ActionConditionedCausalPredictor (v2 Upgrade)

Inspired by V-JEPA 2-AC (zero-shot robot control):

```text
Input:  Slot history (B, T, N, 128) + Action (B, 64)
Process: Action → ActionEmbedding (2-layer MLP) → action_token (B, 1, 256)
         cat([action_token, flatten(slots)]) → Transformer
Output: Recovered slots + Future slots (same as CausalPredictor)

When action=None: falls back to standard CausalPredictor behavior
Backward compatible.
```

**Integration with DeviceGateway:**
```
device_gateway.ActionSpec.encode() → action_vector (64-dim)
    → ActionConditionedCausalPredictor(slots, action) → predicted_next_state
    → VORTEX evaluates safety before execution
```

## 3. Loss Functions

### 3.1 CausalVICRegLoss (ICML 2026 Standard)

4 terms, 4 hyperparameters:

| Term | Weight | Purpose |
|------|--------|---------|
| sim_loss | 25.0 | Prediction accuracy (MSE pred vs target) |
| var_loss | 25.0 | Prevents collapse to constant (target std=1.0) |
| cov_loss | 1.0 | Decorrelates dimensions (maximizes info capacity) |
| causal_loss | 5.0 | Tracks inter-slot causal covariance structure |

Total: L = 25*sim + 25*var + 1*cov + 5*causal

### 3.2 SIGRegLoss (v2 Upgrade, from LeWorldModel)

2 terms, 2 hyperparameters:

| Term | Weight | Purpose |
|------|--------|---------|
| var_loss | 25.0 | Variance regularization (target std=1.0) |
| cov_loss | 1.0 | Covariance decorrelation |

Total: L = 25*var + 1*cov

**Key insight**: The predictor network implicitly provides invariance.
No explicit sim_loss needed → less gradient interference.

**Trade-off**: Causal interaction tracking (causal_loss) is dropped.
Mitigation: Action-Conditioned predictor captures interactions naturally.

### 3.3 Comparison Results (from compare_losses.py)

| Loss | Total (random init) | Hyperparams | Grad norm | Speed |
|------|---------------------|-------------|-----------|-------|
| CausalVICReg | ~53.4 | 4 | ~0.53 | baseline |
| SIGReg | ~1.7 | 2 | ~0.13 | 1.3x faster |
| SIGReg+Sim | ~3.7 | 3 | ~0.13 | 1.1x faster |

SIGReg produces ~31x lower total loss on random data and 3.9x smaller gradients.
On real training: loss values converge differently, experiment needed.

## 4. 10 C-JEPA Domain Variants

Each variant inherits from CJEPALayer, specialized by domain:

| Variant | Domain | Slots | Soul | Input Dim |
|---------|--------|-------|------|-----------|
| CAJEPA | Audio/Music | 5 (drums, bass, vocals, melody, harmony) | Beethoven | 512 |
| CVJEPA | Visual/Art | 8 (composition elements) | VanGogh | 768 |
| CPHYSJEPA | Physics | 7 (bodies, fields, particles) | Einstein | 512 |
| CARTJEPA | Art/Aesthetics | 8 (shapes, colors, textures) | Monet | 768 |
| CDESIGNJEPA | Engineering | 6 (components, layouts) | DaVinci | 512 |
| CFINJEPA | Financial | 6 (trend, momentum, volatility, volume, S/R, regime) | Strategy | 256 |
| CCODEJEPA | Code/AST | 7 (control_flow, data_structures, api_calls, etc.) | Cezanne | 384 |
| CBIOJEPA | Biology | 6 (genes, proteins, pathways, traits) | Darwin | 512 |
| CGEOJEPA | Geography | 6 (climate, terrain, hydrology) | Humboldt | 512 |
| CLAWJEPA | Law/Compliance | 6 (rules, cases, precedents) | Montesquieu | 512 |

All variants share: SlotAttention + CausalPredictor + ObjectLevelMasker.

## 5. Training Protocol

### 5.1 Current Training (A-JEPA Phase 1)

```
Command: python train_ajepa.py --epochs 100 --batch 8 --lr 5e-5
Status:  Epoch 54/100, Loss ~46.5, VRAM 0.2GB
Data:    E:\ large audio dataset (DEEP HOUSE, temple_music, etc.)
Feature: MelSpectrogram (22050Hz, 128 mel, 2048 FFT, 512 hop)
Model:   CAJEPA (5 slots × 128 dim), AudioFeatureProjector (Conv2D backbone)
Loss:    CausalVICReg (4-term)
Guard:   jepa_training_guard.py (spike/gradient/NaN/collapse detection)
Checkpoints: D:\VORTEX_FLAME\ajepa_checkpoints\
```

### 5.2 Phase 2 Plan (Post-Phase-1)

```text
Objective: Object-level masking + causal prediction fine-tuning
Loss:      SIGReg (2-term) or CausalVICReg (4-term) — TBD by experiment
Experiment: compare_losses.py (SIGReg vs CausalVICReg comparison)
Key test:  Counterfactual reasoning accuracy on music data
```

### 5.3 Phase 3 Plan: Action-Conditioned

```text
Objective: Add action embedding → JEPA can predict action consequences
Model:     ActionCAJEPA (causal_jepa_v2.py)
Reference: V-JEPA 2-AC (62 hours robot data → zero-shot control)
VORTEX use: Device action → JEPA predicts consequence → safety check → execute
```

### 5.4 Future: RL Curiosity Fine-tuning

```text
Objective: CJEPACuriosityReward for intrinsic motivation
Mechanism: Prediction error on masked objects = curiosity signal
Use case:  Self-supervised discovery of new interaction patterns
```

## 6. Integration Points

### 6.1 Dual Pathway Bridge

```
Path A (RAG): soul_memory query → text facts → text embedding (384-dim)
Path B (C-JEPA): world_embedding_cache query → causal graph → world embedding (128-dim)
Fusion: DualPathwayBridge Cross-Attention (text ↔ world) → fused (512-dim) → LLM
```

### 6.2 Device Gateway

```
LLM decision → VORTEX (permission + translation) → DeviceGateway → Backend
DeviceGateway: abstract interface, 24 registered devices (NullBackend placeholder)
ActionConditionedCausalPredictor: evaluate action consequences before execution
Safety: RuleEngine hard-check + JEPA physics prediction = dual validation
```

### 6.3 Soul Memory

```
soul_memory: 3 independent storages per soul (conversation, knowledge, todo)
Retrieval: BM25 keyword + FAISS semantic hybrid search
Total entries: ~150.2K (150K original + 159 workflow + 30 domain v1 + 34 domain v2)
```

## 7. File Reference

| File | Purpose |
|------|---------|
| `five_layer_jepa/causal_jepa.py` | C-JEPA core: SlotAttention, CausalPredictor, CausalVICRegLoss, CJEPALayer |
| `five_layer_jepa/causal_jepa_v2.py` | v2 upgrades: SIGRegLoss, ActionConditionedCausalPredictor, ActionCAJEPA |
| `train_ajepa.py` | A-JEPA training script (currently running, Epoch 54/100) |
| `compare_losses.py` | SIGReg vs CausalVICReg comparison experiment |
| `jepa_training_guard.py` | Training protection (spike/gradient/NaN/collapse detection) |
| `world_embedding_cache.py` | World-Embedding cache (SQLite + FTS5, 128-dim vectors) |
| `causal_knowledge_extractor.py` | Text → ObjectGraph → C-JEPA samples |
| `device_gateway.py` | Device abstraction layer (24 devices, 9 categories) |
| `dual_pathway_bridge.py` | Cross-Attention text↔world fusion |
| `jepa_soul_bridge.py` | JEPA ↔ Soul Memory ↔ LLM bridge |
| `index_extended_domains.py` | Domain knowledge indexer v1 (30 entries) |
| `index_knowledge_v2.py` | Private KB rules indexer v2 (34 entries) |
| `extract_extended_domain_causal.py` | Domain causal extraction v1 (7 graphs) |
| `extract_causal_v2.py` | Private KB causal extraction v2 (6 graphs) |
| `extract_workflow_causal.py` | Workflow engine causal extraction (6 graphs) |
| `index_workflow_kb.py` | Workflow engine knowledge indexer (159 entries) |

## 8. Key Papers

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|-----------------|
| Causal-JEPA | ICML 2026 | 2026 | Object-level masking, causal intervention, +20% counterfactual |
| V-JEPA 2 | arXiv 2506.09985 | 2025 | Action-conditioned JEPA, zero-shot robot control |
| LeWorldModel | arXiv 2603.19312 | 2026 | End-to-end 15M JEPA, SIGReg, single GPU |
| VL-JEPA | arXiv 2512.10942 | 2026 | Vision-language JEPA, predicts embeddings not tokens |
| EB-JEPA | ICLR 2026 WS | 2026 | Open-source modular JEPA library |
| IA-JEPA | arXiv 2605.15466 | 2026 | Interaction-aware masking, physics reasoning |
| WavJEPA | arXiv 2509.23238 | 2025 | Raw waveform JEPA, end-to-end audio |
| Stable World Model | ICLR 2026 WS | 2026 | Reproducible world model research framework |
