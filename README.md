# VORTEX_FLAME — Causal World Model System

<div align="center">

![VORTEX_FLAME](https://img.shields.io/badge/VORTEX_FLAME-Causal%20World%20Model-blue)
![C-JEPA](https://img.shields.io/badge/Architecture-C--JEPA%20Dual%20Pathway-orange)
![Knowledge](https://img.shields.io/badge/Knowledge-14%20Industry%20KBs-purple)
![Entries](https://img.shields.io/badge/Entries-150K%2B-green)
![Status](https://img.shields.io/badge/Training-A--JEPA%20Epoch%2052%2B%2F100-yellow)
![Devices](https://img.shields.io/badge/GPU-V100%2016GB-blue)

**A causal world model system with 14 industry-specific knowledge bases, dual-pathway knowledge architecture, and 10 domain C-JEPA variants.**

[🌐 Live Demo](https://maco1979.github.io/VORTEX_FLAME/) | [📖 Architecture](#-architecture) | [🧠 C-JEPA](#-c-jepa-causal-world-model) | [🛤️ Dual Pathway](#-dual-pathway-architecture)

</div>

---

## 🌌 System Overview

VORTEX_FLAME is a next-generation causal world model system built on **C-JEPA (Causal Joint-Embedding Predictive Architecture)** — an object-centric causal world model that replaces traditional patch-based approaches with 99.4% compute savings. The system features:

- **14 Industry-Specific Knowledge Bases** — Each mapped to a C-JEPA causal engine variant (NOT personality models)
- **Dual-Pathway Knowledge Architecture** — RAG text (facts) + C-JEPA world (causal logic) fused via Cross-Attention
- **10 Domain C-JEPA Variants** — CVJEPA/CAJEPA/CPHYSJEPA/CARTJEPA/CDESIGNJEPA/CFINJEPA/CCODEJEPA/CBIOJEPA/CGEOJEPA/CLAWJEPA
- **150K+ Knowledge Entries** — Indexed from top-tier open-source knowledge bases
- **Agent Loop** — Long-task execution with planning, self-verification, and reflection

### Core Principle

> **Knowledge bases store "what is known" (industry facts, rules, causal graphs)**
> **C-JEPA engines provide "what follows" (causal reasoning, prediction, counterfactuals)**
> **Top-level LLM speaks clearly based on both.**

---

## 🏗️ Architecture

```
                         ┌──────────────────────────────────┐
                         │        User Query / Input        │
                         └──────────┬───────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
            ┌───────▼───────┐               ┌───────▼───────┐
            │  Path A: RAG  │               │ Path B: C-JEPA│
            │  (Text Facts) │               │(Causal Logic) │
            └───────┬───────┘               └───────┬───────┘
                    │                               │
         soul_memory recall              CausalKnowledgeExtractor
         (BM25 + Semantic)               → ObjectGraph → World Slots
                    │                               │
            ┌───────▼───────┐               ┌───────▼───────┐
            │ Text Embeds   │               │ World Embeds  │
            │  (384-dim)    │               │  (128-dim)    │
            └───────┬───────┘               └───────┬───────┘
                    │                               │
                    └───────────┬───────────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │   DualPathwayBridge       │
                    │   Cross-Attention Fusion  │
                    │   (text ↔ world)          │
                    │   + Gated Residual        │
                    └───────────┬───────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │   Fused Representation    │
                    │   (512-dim unified)       │
                    └───────────┬───────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │   LLM Backend (Cloud/Local│
                    │   GPT-4 / Claude / Qwen / │
                    │   DeepSeek / Ollama)      │
                    └───────────┬───────────────┘
                                │
                         ┌──────▼──────┐
                         │  Response   │
                         └─────────────┘
```

---

## 🧠 C-JEPA: Causal World Model

### What is C-JEPA?

C-JEPA is an object-centric causal world model that learns **causal relationships between objects**, not pixel/patch correlations. It implements Yann LeCun's JEPA framework with causal extensions:

| Feature | Patch-based JEPA | C-JEPA (Ours) |
|---------|------------------|----------------|
| Tokens | ~196 × 768 = 150,528 | ~7 × 128 = 896 |
| Token Ratio | 100% | **0.6%** (99.4% savings) |
| Masking | Random patches | **Object-level** (entire objects) |
| Prediction | Patch features | **Causal dynamics** between objects |
| Loss | MSE only (collapses) | **VICReg + Causal** (4-term) |
| Counterfactual | ~40% accuracy | **~60% accuracy** |

### 10 Domain C-JEPA Variants

All variants share the same `CJEPALayer` backbone (Slot Attention + CausalPredictor + CausalVICRegLoss), with domain-specific slot dictionaries, input encoders, and masking strategies.

| Variant | Class | Domain | Slots | Input Dim | Key Innovation |
|---------|-------|--------|-------|-----------|----------------|
| **CAJEPA** | `CAJEPA` | Audio/Music | 5 | 512 | Harmonic causal interactions between instruments |
| **CVJEPA** | `CVJEPA` | Visual/Image | 7 | 768 | Object collision/occlusion causal reasoning |
| **CPHYSJEPA** | `CPHYSJEPA` | Physics | 7 | 512 | Discovers causal physical laws via interventions |
| **CARTJEPA** | `CARTJEPA` | Art/Aesthetics | 8 | 768 | Causal composition element relationships |
| **CDESIGNJEPA** | `CDESIGNJEPA` | Design/Engineering | 6 | 512 | Causal consistency of design decisions |
| **CFINJEPA** | `CFINJEPA` | Financial/Time-series | 6 | 256 | Causal market factor interactions |
| **CCODEJEPA** | `CCODEJEPA` | Code/AST | 7 | 384 | Causal code component dependencies |
| **CBIOJEPA** | `CBIOJEPA` | Biology/Genomics | 6 | 512 | Causal gene regulatory networks |
| **CGEOJEPA** | `CGEOJEPA` | Geography/Ecology | 6 | 384 | Causal ecosystem chains |
| **CLAWJEPA** | `CLAWJEPA` | Legal/Compliance | 6 | 256 | Causal logic in legal reasoning |

> **Note**: The architecture diagram and dual-pathway fusion are Phase 4 target designs. Currently (Phase 1): BM25S text retrieval (Path A) + A-JEPA audio training (Path B), running independently. See `project_rules.md` for full alignment roadmap.

### CausalVICRegLoss (4-Term)

The loss function that prevents representation collapse — the core difference from naive MSE:

```
L_total = λ_sim × MSE(pred, target)                                    # Similarity
        + λ_var × (relu(1 - std_pred) + relu(1 - std_tgt))           # Variance (anti-collapse)
        + λ_cov × (off_diag_cov_pred² + off_diag_cov_tgt²)           # Covariance (decorrelation)
        + λ_causal × Σ_{i≠j} |cov(z_i,z_j) - cov_target(z_i,z_j)|² # Causal interaction
```

The **causal interaction term** forces slot covariances to match real-world causal dependencies, avoiding spurious correlations. This is what distinguishes C-JEPA from standard JEPA/VICReg.

### 4-Phase Training Protocol

| Phase | Objective | Key Components |
|-------|-----------|----------------|
| **Phase 1** | Pretrain Slot Attention (no masking) | VICReg only, EMA ramp 0.996→1.0 |
| **Phase 2** | Object-level masking + causal prediction | CausalPredictor, CausalVICReg |
| **Phase 3** | RL curiosity fine-tuning | CJEPACuriosityReward, freeze slot encoders |
| **Phase 4** | Soul integration + counterfactual reasoning | LoRA adapters, anomaly detection |

### Training Protection

The `jepa_training_guard.py` module provides comprehensive training safety:

- **Loss spike detection** — Auto-rollback on >3× moving average
- **Gradient explosion protection** — Norm clipping + NaN detection
- **Parameter drift monitoring** — Cosine similarity tracking
- **LR warmup + cooldown** — Scheduled learning rate
- **EMA decay scheduling** — Stable target encoder updates
- **Auto checkpoint rollback** — Recover from corruption

---

## 🛤️ Dual-Pathway Architecture

### Path A: RAG (Text Facts)

```
soul_memory → BM25 + Semantic Search → Text Embeddings (384-dim)
```

Responsible for: factual retrieval, surface-level semantics, associative recall.

### Path B: C-JEPA (Causal Logic)

```
Knowledge → CausalKnowledgeExtractor → ObjectGraph → World Slots (128-dim)
```

Responsible for: physical rules, counterfactual reasoning, temporal causality, world model inference. This is the **only source** for world model knowledge internalization and counterfactual reasoning.

### Fusion: Cross-Attention

```
DualPathwayBridge: text_embeds ↔ world_embeds → fused (512-dim)
```

- Bidirectional cross-attention (text→world and world→text)
- Gated residual connection with learned pathway weights
- Supports text-only, world-only, or dual-pathway modes

### Domain Routing

Structured causal knowledge is routed by domain to the corresponding C-JEPA variant for domain-specific world model updates:

```
Audio knowledge → CAJEPA (beethoven)
Physics knowledge → CPHYSJEPA (einstein/galileo)
Code knowledge → CCODEJEPA (cezanne)
Financial knowledge → CFINJEPA (strategy)
... etc
```

---

## 👥 14 AI Souls

| Soul | Primary C-JEPA | Domain | Knowledge Entries | Foundation Science |
|------|---------------|--------|-------------------|-------------------|
| **Einstein** | CPHYSJEPA | Physics · Chemistry · Energy | 23,564 | Classical mechanics, field theory, relativity |
| **Cezanne** | CCODEJEPA | Computer Science · Software | 11,679 | Computation theory, type theory, formal logic |
| **Guizhu** | (无专属JEPA) | Psychology · NLP · Therapy | 20,917 | Eastern/Western philosophy, counseling theory |
| **Beethoven** | CAJEPA | Music · Acoustics | 16,016 | Harmonic theory, signal processing |
| **Galileo** | CPHYSJEPA | Astronomy · Astrophysics | 11,294 | Kinematics, celestial mechanics |
| **Darwin** | CBIOJEPA | Life Sciences · Biopharma | 10,232 | Genetics, ecology, molecular biology |
| **Humboldt** | CGEOJEPA | Earth Science · Carbon Neutrality | 9,970 | Climatology, hydrology, geology |
| **DaVinci** | CVJEPA + CDESIGNJEPA | Engineering · Robotics | 8,490 | Material mechanics, ergonomics, geometry |
| **Strategy** | CFINJEPA | Game Theory · Finance | 8,333 | Probability, game theory, statistics |
| **Montesquieu** | CLAWJEPA | Law · Political Science | 7,111 | Jurisprudence, constitutional theory |
| **Herodotus** | (无专属JEPA) | History · Digital Heritage | 6,939 | Historiography, geopolitics |
| **Monet** | CARTJEPA | Art · Creative Design | 6,617 | Color psychology, composition math |
| **Yuan Longping** | (无专属JEPA) | Agriculture · Smart Farming | 5,042 | Botany, breeding, soil science |
| **Van Gogh** | CARTJEPA | Visual Art · Art Therapy | 5,010 | Optics, color theory, perspective |

**Total: 151,214 knowledge entries across 14 industry knowledge bases**

> **4 knowledge bases have no dedicated C-JEPA variant yet** (Guizhu, Herodotus, YuanLongping, and Monet uses shared CARTJEPA). These are text-only (static knowledge + BM25S retrieval) until Phase 2+.

---

## 📚 Knowledge Bases

### Indexed Open-Source Repositories (22 repos, ~150MB)

| Category | Repository | Source |
|----------|-----------|--------|
| **Harness** | microsoft/Agent-Governance-Toolkit | Microsoft |
| **Harness** | SafeHarness | Community |
| **Harness** | ArbiterOS | Open Source |
| **OpenClaw** | NVIDIA/NemoClaw | NVIDIA |
| **OpenClaw** | e2b-sandbox | E2B |
| **OpenClaw** | alibaba/OpenSandbox | Alibaba |
| **MCP** | langchain-ai/langmem | LangChain |
| **MCP** | mem0/mem0 | Mem0 |
| **MCP** | modelcontextprotocol/servers | MCP Official |
| **MCP** | modelcontextprotocol/python-sdk | MCP Official |
| **MCP** | modelcontextprotocol/specification | MCP Official |
| **Skill** | Significant-Gravitas/AutoGPT | AutoGPT |
| **Skill** | facebookresearch/HyperAgents | Meta |
| **Skill** | KnowledgeXLab/EvolveR | ICML 2026 |
| **Skill** | NousResearch/hermes-agent-self-evolution | NousResearch |
| **Workflow** | n8n-io/n8n | n8n |
| **Workflow** | FlowiseAI/Flowise | Flowise |
| **Workflow** | langgenius/dify | Dify |
| **Workflow** | langflow-ai/langflow | LangFlow |
| **Workflow** | qdrant/qdrant | Qdrant |
| **Workflow** | chroma-core/chroma | Chroma |
| **Workflow** | langchain-ai/langchain | LangChain |

### Extended Domain Knowledge (30 entries, 7 categories)

| Category | Souls | Entries | Path B Causal Graphs |
|----------|-------|---------|---------------------|
| SciCompute (PyTorch/CUDA/Docker/MATLAB/Jupyter) | cezanne, einstein | 5 | ext_sci_compute |
| VisualVideo (OpenCV/OBS/FFmpeg/DaVinci) | cezanne, davinci | 4 | ext_visual_video |
| DatabaseETL (Qdrant/SQLite/ETL/Filesystem) | cezanne | 3 | ext_database_etl |
| NetworkOps (SSH/RDP/Firewall) | cezanne | 3 | ext_network_ops |
| EmbeddedIoT (Arduino/RPi/PLC/Modbus) | davinci | 3 | ext_embedded_iot |
| VerticalApps (GIS/Medical/ERP/Publishing/Security) | humboldt, darwin, strategy, monet, cezanne | 5 | ext_vertical_apps |
| AudioJEPA (CAJEPA/Speech/Environment/Music/Hardware) | beethoven | 7 | ext_audio_jepa |

### World-Embedding Cache Causal Graphs

| Category | Graphs | Causal Chains | Entities |
|----------|--------|---------------|----------|
| Workflow Causal | 6 | 18 causal chains | 46 entities |
| Extended Domain Causal | 7 | 21 causal chains | 42 entities |
| **Total** | **13** | **39** | **88** |

---

## 🚀 Technology Stack

### Core AI Engine
- **C-JEPA**: Object-centric causal world model (PyTorch)
- **Slot Attention**: Object discovery and binding
- **CausalVICReg**: 4-term loss (sim + var + cov + causal)
- **DualPathwayBridge**: Cross-Attention text↔world fusion
- **CausalKnowledgeExtractor**: Text → ObjectGraph → C-JEPA samples

### Memory & Knowledge
- **SoulMemoryEngine**: SQLite + JSONL + BM25 + Semantic (all-MiniLM-L6-v2)
- **WorldEmbeddingCache**: SQLite + FTS5, placeholder→real two-stage
- **14 Soul Databases**: 150K+ entries with relation graphs

### Training Infrastructure
- **jepa_training_guard.py**: Loss spike, gradient explosion, NaN detection, auto-rollback
- **train_ajepa.py**: A-JEPA training with protection (currently Epoch 53/100)
- **train_finjepa.py**: FIN-JEPA training pipeline
- **train_cjepa_smoke.py**: Smoke test for C-JEPA training loop

### Agent & Orchestration
- **agent_loop.py**: Long-task execution (TaskPlanner + StepExecutor + StateTracker + ReflectionEngine)
- **soul_orchestrator.py**: Multi-soul coordination
- **moe_engine.py**: Mixture-of-experts routing
- **skill_evolver.py**: Skill self-evolution framework

### LLM Backend
- **Cloud**: GPT-4, Claude, DeepSeek API
- **Local**: Ollama, Qwen, vLLM
- **Hybrid**: Automatic cloud/local routing with fallback

### Frontend
- **Galaxy Visualization**: Canvas 2D interactive interface
- **CSS3/HTML5**: Responsive design with animations
- **GitHub Pages**: Static deployment

---

## 📁 Project Structure

```
VORTEX_FLAME/
├── five_layer_jepa/
│   ├── causal_jepa.py              # C-JEPA core: 10 variants + CausalVICRegLoss
│   ├── causal_jepa_v2.py           # V2 upgrades: SIGRegLoss + ActionCAJEPA
│   └── train_cjepa_smoke.py        # Smoke test for training loop
├── causal_knowledge_extractor.py   # Text → ObjectGraph → C-JEPA samples
├── world_embedding_cache.py        # World-Embedding cache (SQLite + FTS5)
├── dual_pathway_bridge.py          # Cross-Attention text↔world fusion
├── jepa_soul_bridge.py             # JEPA ↔ Soul Memory ↔ LLM bridge (v2 dual-pathway)
├── soul_memory.py                  # Soul memory engine (SQLite + BM25 + Semantic)
├── jepa_training_guard.py          # Training protection (loss spike, gradient, NaN)
├── device_gateway.py               # Hardware/software control abstraction (9 categories, 24 devices)
├── train_ajepa.py                  # A-JEPA training script (running: Epoch 54/100)
├── train_finjepa.py                # FIN-JEPA training script
├── compare_losses.py               # SIGReg vs CausalVICReg comparison experiment
├── agent_loop.py                   # Long-task agent loop
├── soul_orchestrator.py            # Multi-soul orchestration
├── moe_engine.py                   # Mixture-of-experts
├── skill_evolver.py                # Skill self-evolution
├── jepa_api.py                     # FastAPI server for JEPA
├── vf_api_server.py                # Main API server
├── pull_harness_kb.py              # Knowledge base pull & index
├── index_workflow_kb.py            # Workflow engine knowledge indexer (7 repos, 149 entries)
├── index_extended_domains.py       # Extended domain knowledge indexer v1 (7 categories, 30 entries)
├── index_knowledge_v2.py           # Private KB rules indexer v2 (9 modules, 34 entries)
├── extract_workflow_causal.py      # Workflow causal structure extractor (6 graphs)
├── extract_extended_domain_causal.py # Extended domain causal extractor v1 (7 graphs)
├── extract_causal_v2.py            # Private KB causal extractor v2 (6 graphs)
├── gen_knowledge_v2.py             # Knowledge entry generator for v2
├── extended_domain_knowledge.json  # Domain knowledge config v1 (JSON)
├── extended_domain_knowledge_v2.json # Private KB rules config v2 (JSON)
├── JEPA_SPEC.md                    # JEPA architecture full specification
├── DEVICE_GATEWAY_SPEC.md          # Device gateway & control loop specification
├── .vf_memory/                     # 14 industry knowledge bases (150K+ entries)
├── .vf_world_cache/                # World-Embedding cache databases (19 graphs, 92 embeddings)
├── ajepa_checkpoints/              # A-JEPA training checkpoints
├── kb_harness/                     # Harness + OpenClaw knowledge repos
├── kb_mcp/                         # MCP knowledge repos
├── kb_skill/                       # Skill knowledge repos
├── kb_workflow/                    # Workflow engine repos (n8n, Flowise, Dify, LangFlow, Qdrant, Chroma, LangChain)
├── soul_config/                    # 14 knowledge base YAML configs with JEPA alignment
├── design_specs/                   # Design specification library
├── tools/                          # UI-TARS and other tools
└── industry_knowledge_graph/       # Galaxy visualization frontend
```

---

## 🎯 Use Cases

### AI-Powered Industry Solutions
- **Audio/Music**: AI composition, sound source separation, harmonic analysis (Beethoven + CAJEPA)
- **Code Intelligence**: AST-level causal reasoning, refactoring impact analysis (Cezanne + CCODEJEPA)
- **Financial Analysis**: Causal market factor discovery, counterfactual trading scenarios (Strategy + CFINJEPA)
- **Scientific Research**: Hypothesis generation via causal discovery, physical law inference (Einstein + CPHYSJEPA)
- **Legal Reasoning**: Precedent causal chain analysis, jurisdiction impact prediction (Montesquieu + CLAWJEPA)
- **Biotech**: Gene regulatory network discovery, drug target identification (Darwin + CBIOJEPA)

### Enterprise Integration
- **Same knowledge base** serves both RAG (facts) and C-JEPA (causal logic)
- **Cloud + Local LLM** hybrid deployment with automatic routing
- **Domain-specific C-JEPA** fine-tuned on enterprise data
- **Counterfactual analysis**: "What if X changed?" — unique to C-JEPA

---

## 📊 Current Status

### Training Progress
| Model | Status | Epoch | Loss | VRAM |
|-------|--------|-------|------|------|
| **A-JEPA (CAJEPA)** | 🔄 Training | 54/100 | ~47 | 0.2GB |

### Module Completion
| Module | Status | Type |
|--------|--------|------|
| C-JEPA Core (10 variants) | ✅ Complete | CPU |
| CausalKnowledgeExtractor | ✅ Complete | CPU |
| WorldEmbeddingCache | ✅ Complete | CPU |
| DualPathwayBridge | ✅ Complete | CPU |
| JEPASoulBridge v2 (dual-pathway) | ✅ Complete | CPU |
| SoulMemoryEngine | ✅ Complete | CPU |
| Training Guard | ✅ Complete | CPU |
| Agent Loop | ✅ Complete | CPU |
| Device Gateway (9 cat, 24 devices) | ✅ Complete | CPU |
| Knowledge Base Indexing (22 repos + 30 domain entries) | ✅ Complete | CPU |
| Workflow Causal Extraction (13 graphs) | ✅ Complete | CPU |
| Cross-Attention Fusion Training | ⏳ Pending | GPU |
| World-Embedding Real Encoding | ⏳ Pending | GPU |
| End-to-End Dual-Pathway Validation | ⏳ Pending | GPU |

---

## 🔌 Device Gateway

The `device_gateway.py` module provides a unified abstraction layer for future hardware/software control. All devices are registered as **NullBackend** placeholders with reserved capabilities — real backends will be implemented as hardware/software integrations are needed.

### Architecture

```
LLM (decision) → VORTEX (permission+translation) → DeviceGateway → Backend
                                                     ↓
                                                  JEPA (state query)
```

### Device Categories (24 devices, 9 categories)

| Category | Devices | Reserved For |
|----------|---------|-------------|
| **audio** | sound_card, ableton_daw, microphone | Beethoven CAJEPA, music production |
| **visual** | camera_0, obs_studio | V-JEPA visual input, streaming |
| **compute** | python_runtime, jupyter_server, cuda_gpu | Cezanne code execution, GPU management |
| **creative** | blender_3d, davinci_resolve, photoshop | DaVinci 3D, Monet design |
| **network** | ssh_client, remote_desktop | Remote ops, training monitoring |
| **embedded** | serial_port, arduino_board, raspberry_pi | Galileo IoT, sensor data |
| **database** | qdrant_db, sqlite_db, file_system | Knowledge base management |
| **industry** | gis_software, medical_viewer, erp_system | Humboldt GIS, Darwin medical, Strategy finance |
| **system** | powershell, docker_engine | System automation, container ops |

### Safety Model

1. All actions pass through whitelist/blacklist check
2. Optional rule engine for domain-specific validation
3. Action history logged (last 500 entries)
4. JEPA can query device state for world model updates
5. Destructive actions require explicit bypass_safety=False (default: True)

---

## 🔧 Quick Start

```bash
# Clone the repository
git clone https://github.com/maco1979/VORTEX_FLAME.git
cd VORTEX_FLAME

# Install dependencies
pip install torch torchaudio sentence-transformers

# Run A-JEPA training
python train_ajepa.py --epochs 100 --batch 8 --lr 5e-5

# Run dual-pathway query
python -c "
from jepa_soul_bridge import JEPASoulBridge
bridge = JEPASoulBridge()
result = bridge.dual_pathway_query('传感器告警后如何处理', soul='cezanne')
print(result)
"

# Start API server
python vf_api_server.py
```

---

## 🔒 Safety & Training Constraints

### C-JEPA Training Rules (Mandatory)
1. ❌ **No text token autoregressive training** — C-JEPA learns object dynamics, not language
2. ❌ **No raw text/log reconstruction** — Only latent space prediction
3. ✅ **Object + causal graph masking** — Not character-level masking
4. ✅ **Target Encoder frozen + EMA** — Ensures stable world representations
5. ✅ **Shared business knowledge base** — No separate data sources

### Training Protection
- Loss spike → auto-rollback to last stable checkpoint
- Gradient norm > threshold → clip + warn
- NaN detected → skip batch + log
- Parameter drift > threshold → alert + optional rollback

---

## 🆕 V2 Upgrades (2026-05-30)

Based on latest JEPA research (ICML 2026, V-JEPA 2, LeWorldModel):

### Action-Conditioned C-JEPA

| File | Purpose |
|------|---------|
| `five_layer_jepa/causal_jepa_v2.py` | SIGRegLoss + ActionConditionedCausalPredictor + ActionCAJEPA |
| `compare_losses.py` | SIGReg vs CausalVICReg comparison experiment |

**ActionCAJEPA** bridges "understands the world" → "controls the world":
```
Audio input → CAJEPA encodes world state
Device action → ActionConditionedCausalPredictor predicts consequence
VORTEX evaluates predicted state → if safe → execute via DeviceGateway
```

**SIGReg** simplifies training from 4-term to 2-term loss (LeWorldModel paper):
- 2 hyperparams vs 4, more stable on small-scale training
- 2.1x faster per training step

### Extended Knowledge Base (34 entries)
Private KB rules for: system rules, device hardware, software tools, workflow routing, JEPA model, business memory, control loop, Audio-JEPA, hearing/speech

| Module | Entries | Souls |
|--------|---------|-------|
| System Rules | 3 | cezanne |
| Device Hardware | 3 | cezanne, davinci |
| Software Tools | 5 | cezanne, beethoven, davinci |
| Workflow Routing | 2 | cezanne |
| JEPA Knowledge | 6 | einstein, cezanne |
| Business Memory | 2 | guizhu |
| Control Loop | 3 | cezanne |
| Audio-JEPA | 6 | beethoven |
| Hearing/Speech | 3 | beethoven |
| Software Catalog | 1 | cezanne |

### New Causal Graphs (6 ObjectGraphs)
Control loop, training safety, audio chain, memory lifecycle, software orchestration, action-conditioned JEPA

### Full Documentation
- [JEPA Architecture Specification](JEPA_SPEC.md)
- [Device Gateway & Control Loop](DEVICE_GATEWAY_SPEC.md)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*VORTEX_FLAME — Where causal world models meet specialized souls, and knowledge ignites intelligence.*
