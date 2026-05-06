#!/usr/bin/env python3
"""
Categorize and write dataset metadata to shared memory store.
Categories:
  - Einstein: MathNet (olympiad math), WithinUsAI Einstein_Tesla series
  - FKJ/Beethoven: Ableton God Producer Dataset
  - Cezanne: WithinUsAI coding datasets (PythonGOD, Genesis, GOD_Coder)
  - Global: WithinUsAI research concepts, MathNet benchmark info
"""
import sys, os, json, time
sys.path.insert(0, "/mnt/d/VORTEX_FLAME")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from long_memory import write_knowledge, close_knowledge_handles, build_knowledge_index

def write_einstein():
    soul = "einstein"
    datasets = [
        {
            "category": "dataset_catalog",
            "content": "MathNet (ShadenA/MathNet) - MIT ICLR 2026: 27,817 Olympiad-level math problems from 58 countries, 17 languages, 1985-2025. Topics: Geometry (plane/solid/differential), Algebra (polynomials/inequalities/functional equations/sequences/linear algebra/abstract algebra), Number Theory (divisibility/primes/modular arithmetic/Diophantine/quadratic residues/p-adic), Combinatorics (counting/graph theory/extremal/pigeonhole/invariants/games/coloring/generating functions), Calculus/Analysis, Probability & Statistics. Each problem has: problem_markdown (LaTeX), solutions_markdown (expert solutions), topics_flat (hierarchical taxonomy), country, competition, language, problem_type (proof only/answer only/proof and answer), final_answer. 5,148 problems with figures (7,541 images). SOTA performance: Gemini-3.1-Pro 78.4%, GPT-5 69.3% on MathNet-Solve-Test. Embedding models struggle with equivalence retrieval (Recall@1 < 5%). Expert RAG lifts DeepSeek-V3.2-Speciale to 97.3%. HF: ShadenA/MathNet, License: CC BY 4.0",
            "metadata": {"source": "ShadenA/MathNet", "size": "27817", "type": "olympiad_math", "paper": "ICLR 2026", "url": "https://huggingface.co/datasets/ShadenA/MathNet"}
        },
        {
            "category": "dataset_catalog",
            "content": "WithinUsAI/Einstein_Telsa_Inventor_100k - 100K rows of Einstein/Tesla inventor-style reasoning data. Related variants: Einstein_Telsa_Inventor_rationalize_100k (rationalized reasoning), Einstein_Telsa_Inventor_reasoning_100k (chain-of-thought), Einstein_Telsa_Inventor_Thinking_100k (thinking-style), Einstein_Telsa_Inventor_MOE_100k (mixture-of-experts format). All by WithinUsAI. HF: WithinUsAI/Einstein_Telsa_Inventor_*",
            "metadata": {"source": "WithinUsAI", "size": "100k each", "type": "inventor_reasoning", "variants": 5}
        },
    ]
    for d in datasets:
        write_knowledge(soul, d["category"], d["content"], d["metadata"])
    print(f"  Einstein: {len(datasets)} entries written", flush=True)

def write_fkj_beethoven():
    soul = "fkj"
    datasets = [
        {
            "category": "dataset_catalog",
            "content": "WithinUsAI/ableton_god_producer_dataset - 9K+ examples training LLMs to be god-level Ableton Live 12 producers. Categories: drum_programming, bass_design, mixing_mastering, sound_design, arrangement, remixing. Styles: Trap, Drill, Lo-Fi Hip Hop, Future Bass, Techno, UK Garage, Phonk, Jungle, House, Hyperpop, Ambient Electronic, IDM, Drum & Bass. Features: god-level reasoning with step-by-step explanations, exact Ableton tool calling commands (Load Drum Rack, set velocity, etc.), specific parameter values (BPM, key, synth settings for Serum/Massive X/Vital/Operator), no placeholders. Recommended fine-tuning: Qwen2.5-72B or Llama-3.3-70B with ORPO + Tool Calling, 8192+ context. HF: WithinUsAI/ableton_god_producer_dataset, License: Apache 2.0",
            "metadata": {"source": "WithinUsAI", "size": "9k+", "type": "music_production", "tool": "Ableton Live 12", "url": "https://huggingface.co/datasets/WithinUsAI/ableton_god_producer_dataset"}
        },
    ]
    for d in datasets:
        write_knowledge(soul, d["category"], d["content"], d["metadata"])
    print(f"  FKJ: {len(datasets)} entries written", flush=True)

    soul2 = "beethoven"
    for d in datasets:
        write_knowledge(soul2, d["category"], d["content"], d["metadata"])
    print(f"  Beethoven: {len(datasets)} entries written", flush=True)

def write_cezanne():
    soul = "cezanne"
    datasets = [
        {
            "category": "dataset_catalog",
            "content": "WithinUsAI/python_GOD_coder_100k - 100K high-density Python coding examples. Part of the GOD Coder series. Related: GOD_Coder_100k (general coding), Elite_GOD_Coder_100k (curated elite), Omega_Genesis_Coder_100k (genesis+coding), Royal_Ghost_Coder_500k (500K scale), Legend_Python_CoderV.1, HyperScholar-OmniPython-50K. HF: WithinUsAI/python_GOD_coder_100k",
            "metadata": {"source": "WithinUsAI", "size": "100k", "type": "python_coding"}
        },
        {
            "category": "dataset_catalog",
            "content": "WithinUsAI/Genesis_AI_Code_Series - Progressive dataset scaling: 1K Demo -> 10K -> 50K -> 100K. Designed for frontier coding agent training. Agentic reasoning traces, tool-using workflows, code generation + verification, diff-based patching, test-driven reasoning. HF: WithinUsAI/Genesis_AI_Code_10k, Genesis_AI_Code_50k, Genesis_AI_Code_100k",
            "metadata": {"source": "WithinUsAI", "size": "1k-100k", "type": "agentic_coding", "progressive": True}
        },
        {
            "category": "dataset_catalog",
            "content": "WithinUsAI/GPT5.5_thinking_max_distill_god_seed_25K - 25K distilled reasoning from GPT-5.5 with max thinking. Related: Grok4.4_heavy_max_distill_god_seed_25k (Grok-4.4 distill), Opus4.7_thinking_max_distill_god_seed_25k (Opus-4.7 distill). All 25K rows with deep reasoning traces. HF: WithinUsAI/*_distill_god_seed_25k",
            "metadata": {"source": "WithinUsAI", "size": "25k each", "type": "reasoning_distill", "models": ["GPT5.5", "Grok4.4", "Opus4.7"]}
        },
        {
            "category": "dataset_catalog",
            "content": "WithinUsAI/self_evolving_self_debugging_250_implementations - 250 self-evolving, self-debugging code implementations. Demonstrates recursive self-improvement in code generation. Related: self_training_implementations_250.jsonl, hf_real_training_implementations_250.jsonl, Self_Q_and_A_dataset. HF: WithinUsAI/self_evolving_self_debugging_*",
            "metadata": {"source": "WithinUsAI", "size": "250", "type": "self_evolving_code"}
        },
    ]
    for d in datasets:
        write_knowledge(soul, d["category"], d["content"], d["metadata"])
    print(f"  Cezanne: {len(datasets)} entries written", flush=True)

def write_global():
    soul = "global"
    entries = [
        {
            "category": "dataset_catalog",
            "content": "WithinUsAI Research Organization - Independent AI research org focused on: (1) Recursive Intelligence Systems - TRM-style models, external memory indexing, self-reinforcing computation loops, Noogenesis.Concordia.Mind.XI architecture. (2) Agentic AI & Code Systems - tool-using workflows, code generation+verification, diff-based patching, test-driven reasoning. (3) High-Signal Dataset Engineering - datasets as training environments, not just corpora. (4) Model Engineering - fine-tuning, weight merging, MoE model merging, cross-model capability transfer. (5) Efficient Deployment - GGUF/llama.cpp, low-cost inference, multi-GPU/TPU. Flagship: GODs.Ghost.Codex.XI (recursive architecture), PythonGOD-25k, Genesis AI Code Series, MoE Efficient Coders. Vision: Developmental Autopoiesis - AI that continuously evolves through recursion, memory, and self-generated experience. Shift from static training -> continuous adaptation, single-pass inference -> recursive cognition loops, scaling parameters -> designing learning systems. HF: WithinUsAI",
            "metadata": {"source": "WithinUsAI", "type": "research_org", "url": "https://huggingface.co/WithinUsAI"}
        },
        {
            "category": "dataset_catalog",
            "content": "WithinUsAI/With_In_Memory_250k - 250K entries for memory-augmented AI systems. Related: CitationGround-1M (1M citation-grounded entries for retrieval), OpenToolTrace-X (60 tool-use traces). HF: WithinUsAI/With_In_Memory_250k, CitationGround-1M, OpenToolTrace-X",
            "metadata": {"source": "WithinUsAI", "size": "250k/1M/60", "type": "memory_retrieval_tools"}
        },
        {
            "category": "dataset_catalog",
            "content": "WithinUsAI/high_priest_gnosticism_100k & high_priest_Catholicism_100k - 100K entries each for religious/philosophical knowledge systems. Gnosticism and Catholicism perspectives. Could be relevant for philosophy/contemplation souls. HF: WithinUsAI/high_priest_*",
            "metadata": {"source": "WithinUsAI", "size": "100k each", "type": "philosophy_religion"}
        },
        {
            "category": "dataset_catalog",
            "content": "MathNet Benchmark Tasks - Three evaluation tasks: (I) Problem Solving - generative models on Olympiad problems graded against expert solutions. (II) Math-Aware Retrieval - embedding models' ability to retrieve mathematically equivalent/structurally similar problems. (III) Retrieval-Augmented Problem Solving - how retrieval quality affects reasoning. Key findings: SOTA reasoners still challenged (Gemini-3.1-Pro 78.4%, GPT-5 69.3%). Embedding models struggle with equivalence retrieval (Recall@1 < 5%). Expert RAG lifts DeepSeek-V3.2-Speciale to 97.3%. MathNet comparison vs GSM8K (8.5K grade school), MATH (12.5K high school), OlympiadBench (6.1K), OlympicArena (3.2K), OlymMATH (200). MathNet is largest: 30.7K problems, 17 languages, multimodal. Paper: mathnet.mit.edu, ICLR 2026",
            "metadata": {"source": "MathNet", "type": "benchmark", "paper": "ICLR 2026"}
        },
    ]
    for e in entries:
        write_knowledge(soul, e["category"], e["content"], e["metadata"])
    print(f"  Global: {len(entries)} entries written", flush=True)

def main():
    print("\n=== Writing Dataset Catalog to Shared Memory ===", flush=True)
    write_einstein()
    write_fkj_beethoven()
    write_cezanne()
    write_global()

    close_knowledge_handles()

    print("\n=== Building Knowledge Indexes ===", flush=True)
    for soul in ["einstein", "fkj", "beethoven", "cezanne", "global"]:
        try:
            count = build_knowledge_index(soul)
            print(f"  {soul}: {count} entries indexed", flush=True)
        except Exception as e:
            print(f"  {soul}: index build failed - {e}", flush=True)

    print("\n=== Done! ===", flush=True)

if __name__ == "__main__":
    main()
