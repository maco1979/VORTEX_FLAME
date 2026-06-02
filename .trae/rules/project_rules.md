# VORTEX FLAME — AI Agent Development Rules & Onboarding

> **ANY AI MODEL OR DEVELOPER MUST READ THIS ENTIRE DOCUMENT BEFORE MAKING ANY CODE CHANGES.**
> 任何AI模型或开发者在做出任何代码更改之前必须完整阅读本文档。

---

## ⛔ STOP — READ THESE FIRST (MANDATORY)

Before writing a single line of code, you MUST load these files into context:

### 1. Project Memory (THE source of truth)
```
D:\VORTEX_FLAME\VORTEX_FLAME_项目记忆_v7.txt
```
Contains: All 40 parts of project history, every architecture decision, training BUG fixes, dataset audits, alignment standards, download status, knowledge base stats.

### 2. Todo List (Current priorities)
```
D:\VORTEX_FLAME\VORTEX_FLAME_待办清单_v2.txt
```
Contains: P0/P1/P2 priorities, what's done, what's downloading, what's pending.

### 3. 14 Soul Knowledge Bases (SQLite)
```text
D:\VORTEX_FLAME\.vf_memory\{soul}.db
```
14 independent SQLite databases. Unified schema:
`memories` table (13 columns), `memories_fts` (BM25S),
`relations`, `profiles`. Must query these for domain knowledge
before making decisions in any soul's domain.

**Actual 13-Column Schema** (from `soul_memory.py:274`):
```sql
entry_id TEXT PRIMARY KEY,
soul TEXT NOT NULL,
category TEXT NOT NULL,
content TEXT NOT NULL,
document_date TEXT NOT NULL,
event_date TEXT,
relations TEXT DEFAULT '[]',
access_count INTEGER DEFAULT 0,
last_accessed TEXT,
importance REAL DEFAULT 0.5,
tags TEXT DEFAULT '[]',
embedding BLOB,           ← 1536‑byte placeholder (384‑dim float32, all‑zero in Phase 1)
created_at TEXT DEFAULT (datetime('now'))
```

### 4. MCP Servers Config
```
%APPDATA%\Trae CN\User\mcp.json
```
All available MCP tools. Use them — don't write code that duplicates MCP capabilities.

### 5. Skills Registry
```
D:\VORTEX_FLAME\skill_registry_auto.py
D:\VORTEX_FLAME\skill_cache\registry_cache.json
```
Available skills. Invoke via the Skill tool before domain-specific work.

---

## 🏗️ PROJECT ARCHITECTURE

### Core Identity
VORTEX FLAME is a **causal world model system** with:
- **14 industry-specific knowledge bases** — each mapped to one of 10 C-JEPA causal engine variants (NOT personality models, NOT trained LoRA weights)
- **Dual-pathway knowledge**: RAG text facts (Path A) + C-JEPA causal logic (Path B)
- **10 domain C-JEPA variants**: CVJEPA/CAJEPA/CPHYSJEPA/CARTJEPA/CDESIGNJEPA/CFINJEPA/CCODEJEPA/CBIOJEPA/CGEOJEPA/CLAWJEPA
- **151,214 knowledge entries** across 14 SQLite DBs (industry knowledge bases, not model weights)
- **Currently training**: CAJEPA (Audio Causal JEPA, via train_ajepa.py) on V100 GPU

**CRITICAL DISTINCTION — Souls are NOT models:**
- A "soul" = an industry-specific knowledge base (SQLite DB + BM25S index + causal graph)
- Souls do NOT have trained model weights (no LoRA, no fine-tuned parameters)
- The 14 soul knowledge bases serve as domain context for the top-level LLM (cloud + local)
- The 10 C-JEPA variants provide causal reasoning engines that consume these knowledge bases
- Einstein soul ≠ an Einstein personality model; Einstein soul = physics/chemistry industry knowledge base

### ⛔ ARCHITECTURE RED LINES — SINGLE SOURCE OF TRUTH

**These 8 rules are the definitive answer to "what is this system?". Any document that repeats these must match exactly or reference this section.**

**Layer 1: Knowledge Bases (14)**
```
Soul = Industry Knowledge Base (SQLite + BM25S + Causal Graph)
       NOT a model. NOT a personality. Stores "what is known".
```

**Layer 2: C-JEPA Base (10 variants) — The World Model**
```
C-JEPA = BASE layer. Encodes physical rules. Provides causal reasoning.
         C-JEPA checks: "Is this causally consistent with reality?"
CVJEPA/CAJEPA/CPHYSJEPA/CARTJEPA/CDESIGNJEPA/CFINJEPA/CCODEJEPA/CBIOJEPA/CGEOJEPA/CLAWJEPA
```

**Layer 3: LLM Top Layer — User Interaction**
```
LLM = TOP layer. Generates text. Interacts with user. Executes tasks.
      LLM does NOT understand the world. LLM predicts P(next_token|context).
      Ollama: hermes3:8b / qwen2.5:7b / qwen2.5:3b / mistral:7b (SoulModelRouter)
      Cloud: API models for adversarial review (待配置)
```

**Layer 4: Adversarial Loop — BUG → Zero**
```
LLM generates → C-JEPA validates causality → inconsistency found (BUG) → LLM fixes → C-JEPA re-validates → loop until BUG=0
```

**Core Iron Laws:**
1. **底座≠LLM**: C-JEPA is the base (world model), LLM is the top layer (user interaction). Invert this = everything breaks.
2. **LLM幻觉无法根治**: LLM does P(next_token|context), not physical reasoning. Must be supervised by C-JEPA.
3. **C-JEPA编码物理规则, LLM生成语言表达**: They are not interchangeable. C-JEPA = ground truth of causality.
4. **知识库=行业领域事实**: Not personality. Not model weights. SQLite + BM25S only.
5. **10 C-JEPA变体=10个因果引擎**: Each covers a different domain's causal structure.
6. **双通路**: Path A(RAG facts) + Path B(C-JEPA causal logic), fused via Cross-Attention.
7. **不做LoRA SFT/DPO训练**: New architecture is pure inference. Knowledge is in knowledge bases, not in model weights.
8. **多知识库隔离**: Independent directories, data, logs. No cross-KB weight sharing.

### 14 KB → C-JEPA Mapping (Definitive)
| KB | C-JEPA | KB | C-JEPA |
|----|--------|----|--------|
| Einstein | CPHYSJEPA | Galileo | CPHYSJEPA |
| Cezanne | CCODEJEPA | Strategy | CFINJEPA |
| Beethoven | CAJEPA | Montesquieu | CLAWJEPA |
| Darwin | CBIOJEPA | Guizhu | CLAWJEPA |
| Humboldt | CGEOJEPA | Herodotus | CVJEPA+CGEOJEPA |
| DaVinci | CVJEPA+CDESIGNJEPA | YuanLongping | CBIOJEPA |
| Monet | CARTJEPA | VanGogh | CARTJEPA |

### Key Files (DO NOT MODIFY WITHOUT UNDERSTANDING THEM)
| File | Purpose | Risk |
|------|---------|------|
| `train_ajepa.py` | CAJEPA training script | HIGH — currently running, LR BUG already fixed |
| `_import_hf_to_v4.py` | HF dataset → soul DB encoder | MEDIUM — idempotent, 8 encoding functions, SHA256 dedup |
| `jepa_training_guard.py` | 7 protection mechanisms | HIGH — protects running training |
| `soul_memory.py` | Soul knowledge base engine | HIGH — all 14 DBs depend on it |
| `five_layer_jepa/causal_jepa.py` | CAJEPA v1 architecture | HIGH — 1260 lines, ObjectSlotEncoder |
| `five_layer_jepa/causal_jepa_v2.py` | CAJEPA v2 with SIGReg | MEDIUM — Phase 2 candidate |
| `_check_alignment.py` | Schema/embedding verification | LOW — diagnostic tool |
| `gen_knowledge_v3.py` | V3 knowledge generator | LOW — already executed |

### 14 Industry Knowledge Bases Mapping
| Knowledge Base | DB File | Industry Domain | C-JEPA Variant | Current Entries |
|----------------|---------|-----------------|-----------------|-----------------|
| Humboldt | humboldt.db | Earth/Climate | CGEOJEPA | 9,970 (v4=7) |
| Einstein | einstein.db | Physics/Chemistry | CPHYSJEPA | 23,564 (v4=4) |
| Beethoven | beethoven.db | Music/Audio | CAJEPA | 16,016 (v4=2) |
| Darwin | darwin.db | Biology/Genome | CBIOJEPA | 10,232 (v4=3) |
| Cezanne | cezanne.db | Logic/Code | CCODEJEPA | 11,679 (v4=3) |
| Monet | monet.db | Aesthetics/Creative | CARTJEPA | 6,617 (v4=0) |
| Galileo | galileo.db | Astronomy/Math | CPHYSJEPA | 11,294 (v4=3) |
| DaVinci | davinci.db | Engineering/Vision | CVJEPA+CDESIGNJEPA | 8,490 (v4=1) |
| Strategy | strategy.db | Strategy/Economy | CFINJEPA | 8,333 (v4=3) |
| Guizhu | guizhu.db | Philosophy/Dialogue | CLAWJEPA | 20,917 (v4=0) |
| YuanLongping | yuanlongping.db | Agriculture/Food | CBIOJEPA+CGEOJEPA | 5,042 (v4=0) |
| Herodotus | herodotus.db | History/Archive | CGEOJEPA | 6,939 (v4=0) |
| VanGogh | vangogh.db | Art/Visual | CARTJEPA | 5,010 (v4=1) |
| Montesquieu | montesquieu.db | Law/Governance | CLAWJEPA | 7,111 (v4=0) |
| **TOTAL** | | | **151,214 (v4=27)** |

---

## 🔬 ALIGNMENT STANDARDS (v5 FINAL — 0 VULNERABILITIES)

After 5 rounds of adversarial review, this is the definitive standard. **NO further debate needed.**

### Physical Facts (from real code, not assumptions)
```
1536‑byte_placeholder_blob = 384‑dim float32 (OpenAI ada-002 legacy, stored in `embedding` BLOB column)
A‑JEPA AudioFeatureProjector output = 512 dims = 2048 bytes (train_ajepa.py:58)
A‑JEPA 5 slots × 128 dim concatenated = 640 dims = 2560 bytes
PHYSICAL TRUTH: JEPA vectors (2048/2560 bytes) > old BLOB container (1536 bytes)
The old `embedding` column CANNOT hold real JEPA vectors. "Reuse BLOB" is physically impossible.
```

### Six Iron Laws

**Law 1: Old 1536‑byte BLOB — placeholder only, never expand, never reuse**
- Phase1: all zeros in `embedding` column, Phase2+: read-only
- NEVER truncate/compress JEPA 512/640 vectors into this container
- Truncation = JEPA slot destruction = alignment meaningless

**Law 2: New native vector column required (Phase 2)**
- New column: `ajepa_embedding BLOB` (2048-2560 bytes for true JEPA vectors)
- Old column preserved, read-only, excluded from all computation
- Migration: idempotent ALTER script (SQLite has no cross-DB transactions, use per-DB idempotent check)
- `PRAGMA table_info` → if column not exists → `ALTER TABLE ADD COLUMN`

**Law 3: NEVER write bare "1536"**
- Must always annotate: `1536‑byte BLOB (384‑dim float32, in \"embedding\" column)`
- JEPA: `ajepa_512dim (2048‑byte)` or `ajepa_640dim_slotcat (2560‑byte)`
- Enforced in ALL code, logs, schema comments

**Law 4: cross_modal_link_id — optional, weak binding, NULL isolation**
- Allows NULL (most knowledge has no cross-modal binding)
- No unique index constraint
- NULL → fully independent BM25S retrieval per modality
- Has link_id → associated BM25S retrieval via ID

**Law 5: No "text vector space" exists currently**
- Text side: only static curated JSON + BM25S bag-of-words
- No LLM training, no dynamic text vectors
- Current alignment: cross_modal_link_id + BM25S ONLY
- Phase 1-3: NO vector similarity, NO vector projection

**Law 6: Phase 1 Schema frozen**
- Current 13-column unified schema (see above), no additions in Phase 1
- New columns (`ajepa_embedding`, `embedding_version`, `train_phase`, `native_dim`) — all deferred to Phase 2

### Phase Roadmap (Phase 1→4 with allow/forbid lists)

**Phase 1 (CURRENT — pure self-supervised, zero alignment)**
- ✅ CAJEPA mel→slot self-supervised training
- ✅ BM25S full-text retrieval
- ✅ `embedding` column: all-zero placeholder BLOB (Phase 1 only)
- ✅ BM25S full-text retrieval
- ✅ Static knowledge base curation
- ❌ Any cross-modal projection/CLIP/contrastive/joint training
- ❌ Vector similarity computation (zero vectors are meaningless)
- ❌ Schema modification
- ❌ Truncation/compression of JEPA vectors
- ❌ Bare "1536" in any code/log

**Phase 2 (Cold alignment — new column + BM25S-only association)**
- ✅ Idempotent migration: ADD COLUMN ajepa_embedding BLOB (14 souls)
- ✅ Old `embedding` column → read-only, excluded from computation
- ✅ Cross-modal ONLY via cross_modal_link_id + BM25S
- ❌ Hot alignment / joint training / bidirectional / CLIP
- ❌ JEPA vector projection to text space (text has no vector space yet)

**Phase 3 (Hot alignment — small-scale contrastive fine-tuning)**
- ✅ Text-side vectors generated on-demand → then contrastive learning
- ✅ Bidirectional projection calibration
- ❌ Full retraining of JEPA backbone

**Phase 4 (Full unification — vector + full-text hybrid retrieval)**
- ✅ Cosine similarity cross-modal retrieval
- ✅ Vector + full-text hybrid retrieval
- ✅ Agent integration

### Three-Tier Vector Validation (Phase 2+)
1. Pure placeholder all-zeros (phase=1, version=placeholder_v0)
2. True JEPA native vector (phase=2, version=ajepa_v1, 512/640 dims)
3. Corrupt → excluded from all retrieval/computation

embedding_type enum: `placeholder_zero_384` | `ajepa_512` | `ajepa_640`

### True JEPA Vector Activation Rule (Phase 2+)
A record is considered to have a **real JEPA vector** if and only if BOTH conditions hold:
1. `ajepa_embedding` column IS NOT NULL
2. `embedding_type` is set to `ajepa_512dim` or `ajepa_640dim`

This dual-condition gate prevents: new column written but type not synced → inconsistent validation across 14 souls.

### Old Placeholder BLOB Expulsion Rule (Phase 2+)
Phase 2 and all subsequent phases: ALL business logic (retrieval, validation, alignment computation, agent reasoning) MUST completely ignore the old `embedding` column. Only the new `ajepa_embedding` native vector column participates in any computation.
- Old `embedding`: permanently read-only, zero business value post-Phase 1
- This is an explicit rule, not an implicit assumption — write it into migration scripts, validation code, and retrieval logic

### 10 Global Forbidden Operations
1. ❌ Bare "1536" anywhere
2. ❌ Treating 1536‑byte as 1536‑dim
3. ❌ Truncating/compressing/down-projecting JEPA vectors into old BLOB
4. ❌ Any cross-modal projection/CLIP/contrastive/joint training in Phase 1
5. ❌ Cosine similarity / vector retrieval on zero vectors
6. ❌ Expanding or reusing old BLOB column
7. ❌ Assuming a training LLM with vector output exists
8. ❌ Enforcing global audio-text pairing
9. ❌ Extending schema in Phase 1
10. ❌ Unique index on cross_modal_link_id

### 🛡️ IRON LAW 14: Anti-Distillation Protection (反蒸馏铁律 — added 2026-05-31)

**This is a NON-NEGOTIABLE, PERMANENT rule. No future AI model, developer, or automated process may override it.**

**任何云端模型、第三方API、外部服务，禁止反向蒸馏本项目的以下资产：**

#### 14.1 Protected Assets (受保护资产)

The following categories of intellectual property are PERMANENTLY protected from reverse distillation, extraction, replication, or unauthorized transfer:

| Category | Scope | Examples |
|----------|-------|----------|
| **知识库** | 14个灵魂SQLite数据库的全部内容 | `.vf_memory/*.db` 中 151,214 条知识条目、BM25S索引、关系图谱、用户画像 |
| **架构** | 因果多灵魂架构 + 双通路知识系统 | 14-soul causal architecture, RAG(Path A) + C-JEPA(Path B), ObjectSlotEncoder, SIGReg |
| **机制** | 所有自研算法与闭环机制 | 知识反馈闭环引擎、供应链自洽算法、处方周期轮换、体质-季节-食材关联、跨模态对齐六铁律 |
| **技能** | 所有灵魂技能包及其训练数据 | 14 soul skill packs, `soul_training_data/*/` 10k HQ datasets, BPE tokenizers, model configs |
| **MCP** | 所有MCP服务器配置与集成方案 | `mcp.json` 配置, Blender/GitHub/Code-Sandbox/Ableton/HF/KnowledgeGraph 集成方案 |
| **训练数据** | 所有HF数据集编码产物 | 174GB HF datasets → v4编码(27条), `_import_hf_to_v4.py` 8种编码函数 |
| **衍生项目** | 基于本项目的所有衍生系统 | SmartHealth营养顾问系统、知识反馈闭环、产品图自动设计工作流 |

#### 14.2 Prohibited Actions (禁止行为)

The following actions are PERMANENTLY FORBIDDEN for any cloud model, third-party API, or external service:

1. ❌ **知识蒸馏** — 将知识库内容、架构设计、算法逻辑作为prompt/context发送给任何云端模型（GPT/Claude/Gemini等）用于训练、微调或知识提取
2. ❌ **架构提取** — 将项目架构图、模块关系、数据流图、训练pipeline以任何形式传输给外部模型用于复制
3. ❌ **机制复制** — 将自研算法（知识反馈闭环、供应链自洽、C-JEPA变体）的源码或伪代码提供给第三方用于实现相同功能
4. ❌ **技能逆向** — 反向工程灵魂技能包的训练数据、模型权重、tokenizer用于构建等效系统
5. ❌ **MCP配置泄露** — 将MCP服务器URL、API密钥、集成方案暴露给任何未授权的外部服务
6. ❌ **批量导出** — 通过API/脚本/数据库查询批量导出知识库内容到任何外部存储或云端
7. ❌ **代理蒸馏** — 通过AI代理（包括本项目的AI agent）间接将受保护资产传输给云端模型
8. ❌ **子项目泄露** — 将SmartHealth等衍生项目的业务逻辑、用户反馈数据、处方算法传输给云端

#### 14.3 Allowed Actions (允许行为)

The following are the ONLY permitted interactions with external systems:

1. ✅ **本地推理** — 使用本地部署的模型（Mistral-7B、Gemma等）进行推理，不发送知识库原始内容
2. ✅ **匿名统计** — 发送不含知识内容的聚合统计数据（如"今日评估次数"）
3. ✅ **用户授权分享** — 用户明确同意后分享其个人评估结果（不含知识库原始条目）
4. ✅ **开源组件使用** — 使用开源库（FAISS/SQLite/FastAPI等），这些库本身不受本铁律约束
5. ✅ **模型下载** — 从HuggingFace下载公开模型和数据集，下载行为本身不违反本铁律

#### 14.4 Enforcement Mechanism (执行机制)

1. **代码层面**：所有涉及外部API调用的代码必须经过`_check_distillation_risk()`审查：
   - 检查发送内容是否包含知识库原始条目
   - 检查prompt/context是否包含架构设计细节
   - 检查是否有批量数据导出行为
   - 违反则拒绝执行并记录日志

2. **配置层面**：`engine.json`中所有外部API endpoint必须标注`"local_only": true`或`"cloud_safe": true`（需人工审核）

3. **审计层面**：每次AI agent会话结束后，自动检查是否有受保护资产被传输到外部（日志审计）

4. **法律层面**：本铁律构成项目知识产权保护的技术措施，受《计算机软件保护条例》和《反不正当竞争法》保护。违反本铁律的行为同时违反法律。

#### 14.5 Scope of Application (适用范围)

- **适用于**：所有AI模型（包括本项目的AI agent）、所有开发者、所有自动化脚本、所有MCP工具调用
- **不适用于**：用户主动要求且明确授权的单次操作（需用户书面确认）
- **永久有效**：本铁律无过期时间，不可被后续任何规则覆盖或削弱
- **优先级**：本铁律优先级高于所有其他规则，包括Standard Rules 1-10和Audit Rules 11-13

---

## 📡 AVAILABLE TOOLS (USE THEM)

### MCP Servers (active)
| Server | Description |
|--------|-------------|
| `blender` | 3D rendering/scene manipulation |
| `GitHub` | Repository management, issues, PRs, code search |
| `Code-Sandbox MCP` | Remote code execution (python/nodejs/go/...) |
| `ableton-mcp` | Ableton Live DAW control |
| `huggingface-skills` | HF model/dataset download and management |
| `Knowledge Graph Memory` | Knowledge graph persistence |
| `hotnews` | Hot news retrieval |
| `dsa-handbook` | DSA algorithm handbook |
| `blackbox-shield` | Security audit / protection |
| `delivery-audit` | Delivery audit |

### Skills (AI assistant capabilities)
Available in Skill tool. Always invoke relevant skills before domain-specific work:
- `cezanne-soul-skill` — Code & logic
- `beethoven-soul-skill` — Music & acoustics
- `monet-soul-skill` — Creative & aesthetics
- `strategy-soul-skill` — Game theory & decision
- `guizhu-soul-skill` — Philosophy & dialogue
- `scientific-rigor-enforcer` — Physics/math/logic enforcement
- `multidimensional-thinking` — Multi-dimensional strategic design
- `vortex-flame-boot` — Project boot/memory loading
- `hf-cli` — HuggingFace Hub operations
- `TRAE-code-review` — Code review
- `TRAE-security-review` — Security scanning
- `TRAE-debugger` — Runtime debugging with instrumentation
- `soul-deep-expansion` — Universal soul deep expansion training (24L→150L)
- `node-dispatcher-quantum-reasoning` — Quantum reasoning framework for AI systems
- `skill-creator` — Create new skills

---

## 🚨 CURRENT LIVE STATE (READ BEFORE ANY ACTION)

### Running Processes — DO NOT INTERRUPT
| Terminal | Process | Status |
|----------|---------|--------|
| Terminal #5 | `python train_ajepa.py --epochs 100 --batch 8 --lr 1e-4` | 🔄 RUNNING (V100 GPU) |
| Terminal #3 | `python vf_api_server.py` (API server, port 8765) | 🔄 RUNNING |

### Phase 2 Entry Gate ⛔ — READ BEFORE ANY P2 ACTION
```
⛔ Phase 2 CANNOT start until ALL of these are true:
  1. CAJEPA epoch 100 completed (currently epoch ~52+/100, running as train_ajepa.py)
  2. Best model saved to ajepa_best.pt
  3. Training guard confirms no NaN/collapse for last 10 epochs
  4. User explicitly confirms "start Phase 2"

DO NOT: add ajepa_embedding columns, modify schema, or start alignment
before ALL 4 conditions are met. Violating this gate = data corruption risk.
```

### Training Status
- Epoch ~52+/100 (recovering from LR=0 BUG, fixed with CosineAnnealingWarmRestarts)
- Best loss: 40.86 (epoch 50, restored from ajepa_best.pt)
- VRAM: 0.2GB / 16GB V100
- TrainingGuard active: gradient clipping at 1.0, NaN detection, collapse detection
- **DO NOT kill Terminal #5 without explicit user request**

### HF Datasets — All Downloaded & Encoded ✅
| Dataset | Soul | Size | Encoded |
|---------|------|------|---------|
| ClimDetect | Humboldt | 8.7GB | ✅ v4=7 |
| MolLangBench | Einstein | 1.6GB | ✅ v4=4 |
| LogiGLUE | Cezanne | 0.7GB | ✅ v4=3 |
| ProteinConformers | Darwin | 6.6GB | ✅ v4=3 |
| MagnaTagATune | Beethoven | 2.7GB | ✅ v4=2 |
| GQA | DaVinci + VanGogh | 11.1GB | ✅ v4=1+1 |
| STAR | Galileo | 57GB | ✅ v4=3 |
| SEC EDGAR | Strategy | 85.5GB | ✅ v4=3 |
| **TOTAL** | **9 knowledge bases** | **~174GB** | **v4=27** |

### CAJEPA Parameters (DO NOT CHANGE in Phase 1)
```python
SAMPLE_RATE = 22050, N_FFT = 2048, HOP_LENGTH = 512, N_MELS = 128
SEGMENT_FRAMES = 256, HISTORY_SEGMENTS = 6, FUTURE_SEGMENTS = 4
batch=8, lr=1e-4, scheduler=CosineAnnealingWarmRestarts(T_0=5)
Total params: 7,214,161 (6,552,657 trainable)
```

### E Drive Space
- Total: 1.86TB, Used: ~1270GB, Free: ~590GB

---

## 🔧 DEVELOPMENT RULES

### Standard Rules
1. **Read memory files FIRST** — all decisions are documented in the memory v7 and todo v2 files
2. **Query soul DBs for domain knowledge** — use `.vf_memory/{soul}.db` SQLite before making domain decisions
3. **Use MCP tools** — don't write code for things MCP already handles (Blender, GitHub, Code-Sandbox, Ableton)
4. **Invoke skills** — use the Skill tool for domain-specific reasoning
5. **One variable at a time** — never change multiple hyperparameters simultaneously during training
6. **Check terminal status** — verify what's running before launching new processes
7. **E drive is sandboxed** — can read/download but can't delete files on E drive
8. **Log everything** — training state, download status, encoding results must be logged
9. **Update memory files** — after significant changes, append to memory v7 and update todo v2
10. **Don't touch Phase 1 training** — no hyperparameter changes until epoch 100 completes

### 🛡️ Audit Integrity Rules (Iron Laws — added 2026-05-30 after CLAWJEPA grep incident)

These three rules exist because a self-written audit grep with `head_limit=20` truncated the 10th JEPA variant (CLAWJEPA at line 1092), leading to a false conclusion that caused deletion of 3 valid README references. **Toolchain bias can corrupt anyone's judgment, including the author of the audit code.**

**Rule 11: Disk byte-level verification is the ONLY final authority**
- Every audit conclusion MUST be validated by direct disk read: `open(path, 'rb').read()` or `SELECT COUNT(*)` — never trust grep output, pagination limits, cache views, or memory snapshots alone.
- `grep -n pattern file | head -N` is a tool output, not ground truth. The disk bytes are immutable; tool output is filtered.

**Rule 12: Document mutations require bidirectional confirmation**
- Before deleting or modifying any document reference (README, rules, memory), you MUST:
  1. Full-scan the source code on disk (no head_limit, no pagination) to confirm whether the referenced entity exists
  2. If deleting: confirm the entity does NOT exist in ANY source file (byte-level scan)
  3. If adding: confirm the entity DOES exist in source code (byte-level scan)
- Never delete a document reference based solely on a tool search that returned empty.

**Rule 13: Cross-validation is mandatory redundancy**
- Single-audit-chain conclusions are vulnerable to tool bias. Always cross-validate with at least ONE independent method:
  - Method A: structured tool (grep, SQL query, py_compile)
  - Method B: raw disk read (Python `open().read()`, `os.stat()`, direct byte scan)
  - If A and B disagree: B wins. Always.
- Human + AI + raw disk scan = the minimum viable cross-validation triad.
