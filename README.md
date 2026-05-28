# VORTEX FLAME — Multi-Soul AI System

> Architecture concepts, soul definitions, and pipeline design.
> Core algorithms are proprietary and not included in this repository.

---

## What is VORTEX FLAME?

VORTEX FLAME is a **multi-soul AI system** that embodies 15 expert souls (plus Gemini Munger = 16 souls), each specializing in a distinct domain of human knowledge. Built on a Mistral-7B base with LoRA fine-tuning, it creates a collective intelligence through:

- **Domain-specialized LoRA adapters** — each soul is a lightweight expert layer
- **MoE (Mixture of Experts) routing** — dynamic soul selection per task
- **5-Layer JEPA World Model** — multimodal understanding across visual/audio/physics/art/design
- **OMC Pipeline Role Layer** — 12 development pipeline roles mapped to 15 souls via functional modes
- **3 Execution Engines** — Team / Ultrapilot / Ralph modes for different collaboration patterns
- **Subagent Parallel Execution** — up to 3 subagents working simultaneously with Git Worktree isolation

---

## Soul Architecture

### 15 Souls by Tier

| Tier | Soul | Domain | LoRA r | Functional Modes |
|------|------|--------|--------|-----------------|
| A | Einstein | Physics, Quantum Mechanics, Innovation | 16 | review/generate/validate |
| A | Cezanne | Code, Logic, Algorithm, Systems | 16 | review/fix/generate/validate |
| A | Galileo | Astronomy, Astrophysics, Orbital Mechanics | 16 | review/generate/validate |
| A | Darwin | Biology, Genetics, Evolution | 16 | review/generate/validate |
| B | DaVinci | Engineering, Architecture, Design | 16 | generate/deploy |
| B | Strategy (Nash) | Game Theory, Strategy, Decision Making | 16 | review/generate |
| B | SkillAuthor | Skill Generation, Evolution, Audit | 16 | skill_write/skill_modify/skill_diff/skill_audit_prep |
| C | Humboldt | Geography, Ecology, Earth Science | 8 | review/generate |
| C | YuanLongping | Agriculture, Genetics, Food Science | 8 | review/generate |
| C | Montesquieu | Law, Political Science, Logic | 8 | review/generate |
| D | Guizhu | Philosophy, Logic, Dialogue | 8 | review/fix/generate/validate |
| D | Herodotus | History, Causality, Civilization | 8 | generate |
| E | Beethoven (+FKJ) | Music, Acoustics, Language Composition | 8 | generate/improvise |
| E | Monet | Aesthetics, Creative Writing, Art Therapy | 8 | generate |
| E | VanGogh | Emotion, Visual Art, Color Science | 8 | generate |

### Training Pipeline (4 Stages)

| Stage | Method | Data Volume | Goal |
|-------|--------|-------------|------|
| S1 | SFT (MathInstruct foundation) | ≤4K | Base capability |
| S2 | SFT (Domain deepening) | ≤4K | Domain expertise |
| S3 | SFT (Cross-domain generalization) | ≤4K | Cross-domain reasoning |
| S4 | DPO Self-play | ≤4K | Preference alignment |

---

## MoE Engine Architecture

```
Input
  ↓
Shared Base (Mistral-7B, 4bit NF4, frozen)
  ↓ layers 0-29
Expert Layers (last N layers, FP16, trainable)
  ↓ only active expert participates
LoRA on Expert (r=16/8)
  ↓
Output (full model forward pass)
```

---

## 5-Layer JEPA World Model

Based on Meta's I-JEPA (CVPR 2023) and V-JEPA (ICLR 2024) philosophy: **predict in representation space, not pixel space**.

| Layer | Modality | Embed Dim | Purpose |
|-------|----------|-----------|----------|
| V-JEPA | Visual | 384 | Image/video understanding |
| A-JEPA | Audio | 256 | Music/acoustics perception |
| PHYS-JEPA | Physics | 512 | Physical quantity prediction |
| ART-JEPA | Art | 768 | Aesthetic evaluation |
| DESIGN-JEPA | Design | 512 | Design logic verification |

Each JEPA implements 5 core methods: `understand()`, `predict()`, `detect_anomaly()`, `plan()`, `verify_generation()`

See `five_layer_jepa/jepa_interface_spec.py` for the full interface specification.

---

## OMC Pipeline Role Layer

12 development pipeline roles mapped to 15 souls via functional modes (review/fix/generate/validate):

| Role | Soul | Mode | OMC Equivalent |
|------|------|------|---------------|
| analyst | Guizhu | review | analyst (opus) |
| planner | Strategy | review | planner (opus) |
| architect | DaVinci | generate | architect (opus) |
| executor | Cezanne | generate | executor (sonnet) |
| debugger | Cezanne | fix | debugger (sonnet) |
| test_engineer | Cezanne | validate | test-engineer (sonnet) |
| code_reviewer | Guizhu | review | code-reviewer (opus) |
| security_reviewer | Cezanne | review | security-reviewer (sonnet) |
| verifier | Cezanne | validate | verifier (sonnet) |
| writer | Herodotus | generate | writer (haiku) |
| designer | Monet | generate | designer (sonnet) |
| critic | Guizhu | review | critic (opus) |

---

## 3 Execution Engines

| Mode | Trigger | Flow | Parallelism |
|------|---------|------|------------|
| **Team** | "team/协作开发/autopilot" | plan→prd→exec→verify→fix | Serial stages + Subagent parallel in exec |
| **Ultrapilot** | "parallel/ulw/ultrapilot" | Up to 3 souls working simultaneously | 3-way parallel |
| **Ralph** | "validate/ralph/精益求精" | verify→fix→review loop until quality gate passes | Iterative loop (max 5) |

### Subagent Parallel Execution

Team mode's `team_exec` stage supports up to 3 subagents working in parallel:
- **Task splitting**: `split_task_by_module()` — keyword-based module decomposition
- **Worktree isolation**: Each subagent gets an independent git worktree (`vf/{stage}/{sub_id}/{hash8}`)
- **Merge strategy**: `--no-ff` merge to main branch, conflicts reported (not auto-resolved)
- **Cleanup**: `WorktreeManager.cleanup_all()` for emergency cleanup

---

## LoRA Depth Routing

Replaces OMC's Haiku/Sonnet/Opus three-tier model routing with LoRA parameter depth:

| Tier | lora_r | OMC Equivalent | Token Saving | Tasks | Souls |
|------|--------|---------------|-------------|-------|-------|
| light | 8 | haiku | 40-50% | Format checks, docs, search | Herodotus/Humboldt/YuanLongping/SkillAuthor |
| standard | 16 | sonnet | 0% | Code, debug, test, design | Cezanne/Strategy/Monet/Beethoven/VanGogh |
| heavy | 32 | opus | -20% (more precise, slower) | Architecture, physics, philosophy | DaVinci/Einstein/Guizhu |

**Core principle**: Task pattern determines tier, soul follows tier.

---

## Skill Self-Evolution

Based on SkillEvolver (Tsinghua+BJTU) and EmbodiSkill (NJU+Microsoft+Tsinghua AIR):

1. **Role Separation** — SkillAuthor soul writes skills, execution souls only read
2. **Contrastive Update** — Compare success/failure trajectories, patch only differences
3. **Independent Audit** — Guizhu soul 5-dimension review (overfitting/ambiguity/non-executability/contradiction/coverage_gap)
4. **Body+Errata Split** — core_rules (stable) + errata (dynamic), errors only recorded in errata

---

## Security Architecture

### 7 Training Red Lines (T001-T007)

| ID | Rule | Violation |
|----|------|----------|
| T001 | No cross-soul LoRA weight sharing | Training terminated |
| T002 | No new training while GPU occupied | Launch refused |
| T003 | No training data modification | Warning + rollback |
| T004 | No LoRA merge into base model | Save refused |
| T005 | No hardcoded hyperparameters | Use defaults |
| T006 | No skipping NaN detection | Force insertion |
| T007 | No cross-soul resume loading | Load refused |

### Agent Security 4 Layers

| Risk | Defense Layer |
|------|--------------|
| Autonomous execution runaway | ActionGuard (action whitelist) |
| Prompt injection/jailbreak | PromptInjectionDetector (16 patterns) |
| Data leakage | AgentAuditTrail (full operation logging) |
| Supply chain poisoning | NetworkWhitelist (domain whitelist/blacklist) |

---

## MCP Servers (16)

| Server | Category | Core Capability |
|--------|----------|---------------|
| soul-memory | core | Conversation, knowledge base, recall, todo |
| comfyui | visual | txt2img, img2img, 7 soul style presets |
| browse | testing | Navigate, screenshot, click, fill |
| osint | osint | User search, profile, compliance check |
| rag | knowledge | Create KB, add documents, query |
| open-design | design | 31 skills, PPT generation, aesthetic scoring |
| codex-enhance | soul | Syntax check, type check, security scan |
| soul-pipeline | pipeline | Get soul config, plan training |
| nla | interpretability | Extract activations, train SAE |
| blackbox-shield | security | Obfuscate, checksum, scan |
| ableton | music | MIDI, track operations |
| delivery-audit | security | Delivery audit |
| dsa | core | Algorithms |
| animejs | visual | Animation generation |
| voice | voice | Whisper + edge-tts |
| ui-tars | automation | Screenshot, analyze, execute, run_task |

---

## License

MIT License — See [LICENSE](LICENSE) for details.

**Note**: This repository contains architecture concepts and interface specifications only. Core algorithm implementations (soul routing, MoE engine, training pipeline, security guardian, etc.) are proprietary and not included.
