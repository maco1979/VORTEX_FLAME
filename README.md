# VORTEX FLAME

Multi-Soul AI Architecture — 13 specialized AI souls trained via self-play, orbiting a central knowledge galaxy.

## Architecture

```
                    ┌─────────────┐
                    │  Black Hole │  ← Central Knowledge Core
                    │  (Galaxy)   │
                    └──────┬──────┘
               ┌───────────┼───────────┐
          Arm 1│           │           │Arm 2
               │           │           │
    Cezanne ── Einstein ── Galileo ── Darwin
       │          │           │         │
    DaVinci ── Strategy ── Humboldt ── Yuan
       │          │           │         │
    Guizhu ── Montesquieu ── Herodotus
       │          │
    Beethoven ── Monet ── VanGogh
```

## 13 AI Souls

| Tier | Soul | Domain | Status |
|------|------|--------|--------|
| A | Cezanne | Computer Science · Software Engineering | Self-Play Iter 2 |
| A | Einstein | Physics · Chemistry · Energy | 5-Stage Complete |
| A | Galileo | Astronomy · Astrophysics | Data Ready |
| A | Darwin | Life Sciences · Biopharma | Data Ready |
| B | DaVinci | Engineering · Robotics | Stage 1 |
| B | Strategy | Game Theory · Finance | Data Ready |
| C | Humboldt | Earth Science · Carbon Neutrality | Data Ready |
| C | Yuan Longping | Agriculture · Smart Farming | Data Ready |
| D | Herodotus | History · Digital Heritage | Data Ready |
| D | Guizhu | Psychology · NLP · Therapy | Stage 1 |
| D | Montesquieu | Law · Political Science | Data Ready |
| E | Beethoven | Music · Acoustics (FKJ merged) | Stage 1 |
| E | Monet | Art · Creative Design | Data Ready |
| E | Van Gogh | Visual Art · Art Therapy | Data Ready |

## Training Pipeline

- **Base Model**: Mistral-7B (4-bit frozen, QLoRA fine-tuning)
- **Self-Play Engine**: AlphaZero-style self-play with 7 categories (Debug, Logic, Algorithm, Systems, Complexity, Network, Memory)
- **Knowledge Base**: FAISS-indexed domain knowledge (908MB for Cezanne)
- **8B Cezanne PRO**: 4-stage training (S1→S2→S3a→S3b), 82% regression pass rate

## Galaxy Visualization

Interactive galaxy visualization at `industry_knowledge_graph/hologram.html`:
- Central black hole with accretion disk (based on NASA JWST Sgr A* imagery)
- 2 spiral arms with density wave structure (logarithmic spiral)
- 13 soul installations as organic light forms with dynamic glow
- 4-layer knowledge graph (Basic Science → Tech Science → Industry → Specific Domain)
- Double-click any soul to explore its knowledge graph

## Safety Red Lines

- Base model permanently frozen (4-bit)
- NF4 permanently banned
- LoRA never merged into base
- Strict serial training
- No domestic base models (Qwen/DeepSeek/Baichuan/Yi/ChatGLM)

## Tech Stack

- PyTorch + QLoRA + vLLM
- FAISS + sentence-transformers
- TRAE IDE + GitNexus
- Canvas 2D + Python HTTP Server

## Hardware

- GPU0: RTX 3060 12GB (training)
- GPU1: GTX 1060 6GB (standby)
- iGPU: Intel HD 630 (display only)

---

*VORTEX FLAME — Where souls orbit knowledge, and knowledge ignites intelligence.*
