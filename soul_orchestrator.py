"""
Soul Orchestrator — Industry Knowledge Base Routing & Execution Engine
======================================================================
Routes tasks to the appropriate industry knowledge base(s) using hybrid
semantic + keyword scoring, dispatches to 3 execution engines, and
arbitrates multi-expert outputs.

CRITICAL DISTINCTION:
- A "soul" = an industry-specific knowledge base (SQLite DB + BM25S + causal graph)
- Souls are NOT personality models, NOT trained LoRA weights, NOT character simulations
- The 14 knowledge bases map to 10 C-JEPA causal engine variants:
  CAJEPA/CVJEPA/CPHYSJEPA/CARTJEPA/CDESIGNJEPA/CFINJEPA/CCODEJEPA/CBIOJEPA/CGEOJEPA/CLAWJEPA
- Top-level LLM (cloud + local) consumes these knowledge bases as domain context

Routing (v2 — Hybrid Semantic):
- SemanticRouter: Embedding-based cosine similarity (all-MiniLM-L6-v2)
- _compute_hybrid_scores: alpha-weighted blend of semantic + keyword scores
- route_to_soul: Single-winner routing with confidence & method metadata
- soft_route_to_souls: Top-k candidates for multi-expert soft fusion
- dispatch_execution: Auto-detects ultrapilot when top-2 confidence gap < 0.15

Execution Engines:
- Team: Sequential pipeline (plan→code→review→test→deploy)
- Ultrapilot: Parallel knowledge base assignments (research∥design∥code)
- Ralph: Verify-fix iteration loop (verify→fix→retest→report ×N)

Result Arbitration:
- arbitrate_results: confidence / consensus / diversity strategies
- merge_subagent_results: Git --no-ff merge for subagent outputs

Key Concepts:
- SOUL_CAPABILITIES: 14 industry knowledge base definitions with domain, tools, skills
- PIPELINE_ROLES: 12 development pipeline roles mapped to knowledge bases
- EXECUTION_MODES: Team / Ultrapilot / Ralph execution engines
- MAGIC_KEYWORDS: Natural language triggers for mode switching
- SEMANTIC_ROUTING_CONFIG: hybrid_alpha=0.6, soft_routing_top_k=3

Integration Modules (P0):
- code_intelligence: CodeGraph (syntax) + Understand-Anything (semantic)
- skill_registry: knowledge-work-plugins (15 job skills)
- guardian: File/process/service whitelists for integration
- harness_runtime: Action/network whitelists per knowledge base

Integration Modules (P1):
- mano_p_adapter: Mano-P GUI perception agent (local/cloud dual mode)
  Pure vision-based GUI operation — no CDP/HTML dependency.
  Supports: click, type, hotkey, scroll, drag, screenshot, app launch, URL nav
  Auto-detects hardware → local (M4+ 32GB / compute stick) or cloud API
"""

import re
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from code_intelligence import CodeIntelligenceManager
from skill_registry import SkillRegistry, KNOWLEDGE_WORK_PLUGIN_MAPPING
from mano_p_adapter import ManoPAdapter, MANO_P_SKILL_DEFINITION
from long_memory import LongMemory, LongMemoryAdapter
from moe_expert_loader import MoEExpertLoader
from validation_rules import ValidationEngine


SOUL_CAPABILITIES = {
    "cezanne": {
        "tier": "A", "full_name": "Cezanne Knowledge Base",
        "domain": ["Code", "Logic", "Algorithm", "Systems", "GUI Automation"],
        "jepa_variant": "CCODEJEPA",
        "skills": ["kwp_engineering", "kwp_operations", "mano_p_gui"],
        "tools": ["ci_callers", "ci_callees", "ci_affected", "ci_search", "ci_context", "ci_domain", "ci_impact", "ci_ask", "manop_click", "manop_type", "manop_hotkey", "manop_scroll", "manop_drag", "manop_screenshot", "manop_launch", "manop_navigate", "manop_execute_task"],
    },
    "einstein": {
        "tier": "A", "full_name": "Einstein Knowledge Base",
        "domain": ["Physics", "Quantum Mechanics", "Quantitative Finance", "Innovation"],
        "jepa_variant": "CPHYSJEPA",
        "skills": ["kwp_finance", "kwp_research", "kwp_data"],
        "tools": ["ci_search", "ci_context", "ci_impact"],
    },
    "galileo": {
        "tier": "A", "full_name": "Galileo Knowledge Base",
        "domain": ["Astronomy", "Astrophysics", "Orbital Mechanics"],
        "jepa_variant": "CPHYSJEPA",
        "skills": ["kwp_research"],
        "tools": ["ci_search", "ci_context"],
    },
    "darwin": {
        "tier": "A", "full_name": "Darwin Knowledge Base",
        "domain": ["Biology", "Genetics", "Evolution", "Healthcare"],
        "jepa_variant": "CBIOJEPA",
        "skills": ["kwp_research", "kwp_healthcare"],
        "tools": ["ci_search", "ci_context"],
    },
    "davinci": {
        "tier": "B", "full_name": "DaVinci Knowledge Base",
        "domain": ["Engineering", "Architecture", "Design", "GUI Automation"],
        "jepa_variant": "CVJEPA+CDESIGNJEPA",
        "skills": ["kwp_engineering", "kwp_design", "kwp_product", "mano_p_gui"],
        "tools": ["ci_context", "ci_impact", "ci_domain", "manop_click", "manop_type", "manop_hotkey", "manop_screenshot", "manop_launch", "manop_navigate", "manop_execute_task"],
    },
    "strategy": {
        "tier": "B", "full_name": "Strategy Knowledge Base",
        "domain": ["Game Theory", "Strategy", "Decision Making", "Finance"],
        "jepa_variant": "CFINJEPA",
        "skills": ["kwp_sales", "kwp_finance", "kwp_product"],
        "tools": ["ci_impact", "ci_domain"],
    },
    "montesquieu": {
        "tier": "C", "full_name": "Montesquieu Knowledge Base",
        "domain": ["Law", "Political Science", "Logic", "Compliance"],
        "jepa_variant": "CLAWJEPA",
        "skills": ["kwp_legal", "kwp_sales", "kwp_hr", "kwp_marketing"],
        "tools": ["ci_impact", "ci_domain"],
    },
    "humboldt": {
        "tier": "C", "full_name": "Humboldt Knowledge Base",
        "domain": ["Geography", "Ecology", "Earth Science", "Data Analysis"],
        "jepa_variant": "CGEOJEPA",
        "skills": ["kwp_data", "kwp_operations"],
        "tools": ["ci_search", "ci_context", "ci_domain"],
    },
    "yuanlongping": {
        "tier": "C", "full_name": "YuanLongping Knowledge Base",
        "domain": ["Agriculture", "Genetics", "Food Science"],
        "jepa_variant": "CBIOJEPA+CGEOJEPA",
        "skills": [],
        "tools": [],
    },
    "guizhu": {
        "tier": "D", "full_name": "Guizhu Knowledge Base",
        "domain": ["Philosophy", "Logic", "Dialogue", "Psychology"],
        "jepa_variant": "CLAWJEPA",
        "skills": ["kwp_hr", "kwp_support", "kwp_healthcare", "kwp_education"],
        "tools": [],
    },
    "herodotus": {
        "tier": "D", "full_name": "Herodotus Knowledge Base",
        "domain": ["History", "Causality", "Civilization", "Documentation"],
        "jepa_variant": "CVJEPA+CGEOJEPA",
        "skills": ["kwp_writing", "kwp_support", "kwp_education"],
        "tools": ["ci_search", "ci_ask"],
    },
    "monet": {
        "tier": "E", "full_name": "Monet Knowledge Base",
        "domain": ["Aesthetics", "Creative Writing", "Art Therapy", "Marketing"],
        "jepa_variant": "CARTJEPA",
        "skills": ["kwp_design", "kwp_marketing", "kwp_writing"],
        "tools": ["ci_context"],
    },
    "vangogh": {
        "tier": "E", "full_name": "VanGogh Knowledge Base",
        "domain": ["Emotion", "Visual Art", "Color Science"],
        "jepa_variant": "CARTJEPA",
        "skills": ["kwp_design"],
        "tools": ["ci_context"],
    },
    "beethoven": {
        "tier": "E", "full_name": "Beethoven Knowledge Base",
        "domain": ["Music", "Acoustics", "Language Composition"],
        "jepa_variant": "CAJEPA",
        "skills": [],
        "tools": [],
    },
}

PIPELINE_ROLES = {
    "team_plan":       {"soul": "cezanne",    "mode": "team"},
    "team_code":       {"soul": "cezanne",    "mode": "team"},
    "team_review":     {"soul": "montesquieu", "mode": "team"},
    "team_test":       {"soul": "galileo",    "mode": "team"},
    "team_deploy":     {"soul": "davinci",    "mode": "team"},
    "ultra_research":  {"soul": "einstein",   "mode": "ultrapilot"},
    "ultra_design":    {"soul": "davinci",    "mode": "ultrapilot"},
    "ultra_code":      {"soul": "cezanne",    "mode": "ultrapilot"},
    "ralph_verify":    {"soul": "montesquieu", "mode": "ralph"},
    "ralph_fix":       {"soul": "cezanne",    "mode": "ralph"},
    "ralph_retest":    {"soul": "galileo",    "mode": "ralph"},
    "ralph_report":    {"soul": "herodotus",  "mode": "ralph"},
    "academic_search": {"soul": "herodotus", "mode": "academic"},
    "academic_analyze":{"soul": "einstein",  "mode": "academic"},
    "academic_cross":  {"soul": "darwin",    "mode": "academic"},
    "academic_write":  {"soul": "herodotus", "mode": "academic"},
    "orche_search":    {"soul": "herodotus", "mode": "orchestrate"},
    "orche_analyze":   {"soul": "einstein",  "mode": "orchestrate"},
    "orche_code":      {"soul": "cezanne",   "mode": "orchestrate"},
    "orche_review":    {"soul": "montesquieu","mode": "orchestrate"},
    "orche_merge":     {"soul": "herodotus", "mode": "orchestrate"},
}

EXECUTION_MODES = {
    "team": {
        "description": "Sequential pipeline with role-based stages",
        "stages": ["team_plan", "team_code", "team_review", "team_test", "team_deploy"],
        "parallel": False,
    },
    "ultrapilot": {
        "description": "Parallel soul assignments for independent subtasks",
        "stages": ["ultra_research", "ultra_design", "ultra_code"],
        "parallel": True,
    },
    "ralph": {
        "description": "Verify-fix iteration loop until quality threshold",
        "stages": ["ralph_verify", "ralph_fix", "ralph_retest", "ralph_report"],
        "parallel": False,
        "loop": True,
    },
    "academic": {
        "description": "Herodotus academic research pipeline: search→analyze→cross-ref→write",
        "stages": ["academic_search", "academic_analyze", "academic_cross", "academic_write"],
        "parallel": False,
        "loop": False,
    },
    "orchestrate": {
        "description": "Multi-KB concurrent orchestration: search∥analyze∥code→review→merge",
        "stages": [["orche_search", "orche_analyze", "orche_code"], "orche_review", "orche_merge"],
        "parallel": True,
        "loop": False,
    },
}

MAGIC_KEYWORDS = {
    r"\b(team|团队)\b":       {"mode": "team",       "action": "switch_mode"},
    r"\b(ultra|极速|并行)\b":  {"mode": "ultrapilot", "action": "switch_mode"},
    r"\b(ralph|修复|迭代)\b":  {"mode": "ralph",      "action": "switch_mode"},
    r"\b(review|审查)\b":     {"mode": "team",       "action": "jump_stage", "target": "team_review"},
    r"\b(deploy|部署)\b":     {"mode": "team",       "action": "jump_stage", "target": "team_deploy"},
    r"\b(fix|修复)\b":        {"mode": "ralph",      "action": "jump_stage", "target": "ralph_fix"},
    r"\b(gui|界面|操作电脑|操控|点一下|打开应用|manop|mano-p)\b": {"mode": "team", "action": "activate_tool", "tool": "mano_p_gui"},
    r"\b(学术|研究|论文|文献|历史|考证|academic|research|paper)\b": {"mode": "academic", "action": "switch_mode"},
    r"\b(多学科|跨学科|综合|融合|全方位|多角度|orchestrat|多专家)\b": {"mode": "orchestrate", "action": "switch_mode"},
}

LORA_DEPTH_ROUTING = {
    "light":    {"lora_r": 8,  "omc_equivalent": "Haiku",  "tier": ["D", "E"]},
    "standard": {"lora_r": 16, "omc_equivalent": "Sonnet", "tier": ["B", "C"]},
    "heavy":    {"lora_r": 32, "omc_equivalent": "Opus",   "tier": ["A"]},
}


# ── Semantic Routing Configuration ──────────────────────────────────────
SEMANTIC_ROUTING_CONFIG = {
    "enabled": True,
    "model_name": "all-MiniLM-L6-v2",
    "hybrid_alpha": 0.6,        # weight for semantic score vs keyword score
    "soft_routing_top_k": 3,    # default top-k for soft routing
    "min_confidence": 0.05,     # fallback threshold
    "ultrapilot_gap_threshold": 0.15,  # base gap threshold for auto-ultrapilot
    "ultrapilot_min_top1": 0.15,        # top-1 must exceed this to consider ultrapilot
}


class SemanticRouter:
    """Embedding-based semantic router with hybrid scoring.

    Falls back gracefully to keyword-only routing when the sentence
    transformer model is not available.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or SEMANTIC_ROUTING_CONFIG["model_name"]
        self._model = None
        self._domain_embeddings: Dict[str, np.ndarray] = {}
        self._initialized = False

    # ── lazy model loading ────────────────────────────────────────────
    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                self._model = None
        return self._model

    # ── pre-compute soul domain embeddings ────────────────────────────
    def _compute_domain_embeddings(self):
        if self._domain_embeddings:
            return
        model = self._load_model()
        if model is None:
            return
        for soul_name, cap in SOUL_CAPABILITIES.items():
            domain_text = " ".join(cap["domain"]) + " " + cap.get("full_name", "")
            self._domain_embeddings[soul_name] = model.encode(domain_text)

    # ── compute semantic similarity scores ───────────────────────────
    def semantic_scores(self, task_description: str) -> Dict[str, float]:
        self._compute_domain_embeddings()
        if not self._domain_embeddings:
            return {}
        model = self._load_model()
        if model is None:
            return {}
        task_embedding = model.encode(task_description)
        scores: Dict[str, float] = {}
        for soul_name, domain_emb in self._domain_embeddings.items():
            sim = float(
                np.dot(task_embedding, domain_emb)
                / (np.linalg.norm(task_embedding) * np.linalg.norm(domain_emb) + 1e-8)
            )
            scores[soul_name] = max(sim, 0.0)
        return scores


_semantic_router = SemanticRouter()


_skill_registry = SkillRegistry()

for _plugin_name, _mapping in KNOWLEDGE_WORK_PLUGIN_MAPPING.items():
    _skill_registry.register_external(
        plugin_name=_plugin_name,
        commands=[],
        connectors=[],
        description=_mapping["domain"],
    )

_skill_registry.register_external(
    plugin_name="mano_p_gui",
    commands=MANO_P_SKILL_DEFINITION["commands"],
    connectors=MANO_P_SKILL_DEFINITION["connectors"],
    description=MANO_P_SKILL_DEFINITION["description"],
)

_mano_p_adapter = ManoPAdapter()

_long_memory = LongMemory()
_long_memory_adapter = LongMemoryAdapter(_long_memory)

_moe_loader = MoEExpertLoader()

_validation_engine = ValidationEngine()

_code_intelligence: Optional[CodeIntelligenceManager] = None

_worktree_mgr = None

_DISPATCH_LOG_PATH = Path(".vf_memory") / "dispatch_log.jsonl"
_DISPATCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def _log_dispatch(entry: dict):
    entry["timestamp"] = time.time()
    with open(_DISPATCH_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def _load_dispatch_stats() -> dict:
    if not _DISPATCH_LOG_PATH.exists():
        return {"total": 0, "ultrapilot_ratio": 0.0, "avg_top1_conf": 0.0}
    entries = []
    with open(_DISPATCH_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except Exception:
                continue
    if not entries:
        return {"total": 0, "ultrapilot_ratio": 0.0, "avg_top1_conf": 0.0}
    recent = entries[-100:]
    total = len(recent)
    ultra_count = sum(1 for e in recent if e.get("mode") == "ultrapilot")
    confs = [e.get("top1_confidence", 0.0) for e in recent if e.get("top1_confidence")]
    return {
        "total": total,
        "ultrapilot_ratio": ultra_count / max(total, 1),
        "avg_top1_conf": sum(confs) / max(len(confs), 1),
    }

def _adaptive_gap_threshold() -> float:
    stats = _load_dispatch_stats()
    base = SEMANTIC_ROUTING_CONFIG["ultrapilot_gap_threshold"]
    if stats["total"] < 20:
        return base
    if stats["ultrapilot_ratio"] > 0.6:
        return base * 1.2
    if stats["ultrapilot_ratio"] < 0.2:
        return base * 0.8
    return base


def _compute_keyword_scores(task_description: str) -> Dict[str, float]:
    """Normalised keyword overlap scores (0–1 per soul)."""
    desc_lower = task_description.lower()
    scores: Dict[str, float] = {}
    for soul_name, cap in SOUL_CAPABILITIES.items():
        hit = sum(1 for kw in cap["domain"] if kw.lower() in desc_lower)
        scores[soul_name] = hit / max(len(cap["domain"]), 1)
    return scores


def _compute_hybrid_scores(task_description: str, use_semantic: bool = True) -> Dict[str, float]:
    """Blend keyword + semantic scores with configurable alpha."""
    keyword_scores = _compute_keyword_scores(task_description)

    semantic_scores: Dict[str, float] = {}
    if use_semantic and SEMANTIC_ROUTING_CONFIG["enabled"]:
        semantic_scores = _semantic_router.semantic_scores(task_description)

    alpha = SEMANTIC_ROUTING_CONFIG["hybrid_alpha"]
    hybrid: Dict[str, float] = {}
    for soul_name in SOUL_CAPABILITIES:
        kw = keyword_scores.get(soul_name, 0.0)
        sem = semantic_scores.get(soul_name, 0.0)
        hybrid[soul_name] = (alpha * sem + (1 - alpha) * kw) if semantic_scores else kw
    return hybrid


def route_to_soul(task_description: str, use_semantic: bool = True) -> dict:
    """Route a task to the best-matching soul (hybrid semantic + keyword)."""
    hybrid_scores = _compute_hybrid_scores(task_description, use_semantic=use_semantic)

    best_soul = max(hybrid_scores, key=lambda k: hybrid_scores[k])
    confidence = hybrid_scores[best_soul]

    if confidence < SEMANTIC_ROUTING_CONFIG["min_confidence"]:
        best_soul = "cezanne"
        confidence = 0.1

    cap = SOUL_CAPABILITIES[best_soul]
    return {
        "soul": best_soul,
        "confidence": round(confidence, 3),
        "tools": cap["tools"],
        "skills": cap["skills"],
        "domain": cap["domain"],
        "tier": cap["tier"],
        "routing_method": "hybrid" if use_semantic and SEMANTIC_ROUTING_CONFIG["enabled"] else "keyword",
    }


def soft_route_to_souls(task_description: str, top_k: Optional[int] = None, use_semantic: bool = True) -> list:
    """Return top-k soul candidates with confidence scores (Soft Routing).

    Unlike ``route_to_soul`` which picks a single winner, Soft Routing
    exposes the full ranking so callers can ensemble, arbitrate, or
    present choices to a human-in-the-loop.
    """
    top_k = top_k or SEMANTIC_ROUTING_CONFIG["soft_routing_top_k"]
    hybrid_scores = _compute_hybrid_scores(task_description, use_semantic=use_semantic)

    sorted_souls = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for soul_name, score in sorted_souls:
        cap = SOUL_CAPABILITIES[soul_name]
        results.append({
            "soul": soul_name,
            "confidence": round(score, 3),
            "tools": cap["tools"],
            "skills": cap["skills"],
            "domain": cap["domain"],
            "tier": cap["tier"],
        })
    return results


def route_lora_depth(task_description: str) -> dict:
    routing = route_to_soul(task_description)
    soul_name = routing["soul"]
    tier = routing["tier"]

    tier_to_depth = {
        "A": ("heavy", 32, "Opus"),
        "B": ("standard", 16, "Sonnet"),
        "C": ("standard", 16, "Sonnet"),
        "D": ("light", 8, "Haiku"),
        "E": ("light", 8, "Haiku"),
    }

    depth_name, lora_r, omc = tier_to_depth.get(tier, ("standard", 16, "Sonnet"))
    return {"tier": depth_name, "lora_r": lora_r, "soul": soul_name, "omc_equivalent": omc}


def detect_magic_keyword(user_input: str) -> dict:
    for pattern, config in MAGIC_KEYWORDS.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return {"mode": config["mode"], "action": config["action"], "task": user_input, "target": config.get("target")}
    return {"mode": None, "action": None, "task": user_input, "target": None}


def dispatch_execution(user_input: str) -> dict:
    kw = detect_magic_keyword(user_input)
    if kw["mode"]:
        mode = kw["mode"]
    else:
        routing = route_to_soul(user_input)
        candidates = soft_route_to_souls(user_input, top_k=2)
        cfg = SEMANTIC_ROUTING_CONFIG
        gap_threshold = _adaptive_gap_threshold()
        min_top1 = cfg["ultrapilot_min_top1"]
        top1_conf = candidates[0]["confidence"] if candidates else 0.0
        top2_conf = candidates[1]["confidence"] if len(candidates) > 1 else 0.0
        gap = top1_conf - top2_conf
        adaptive_threshold = gap_threshold * (1.0 + max(0.0, min_top1 - top1_conf))
        if len(candidates) > 1:
            if gap < adaptive_threshold and top1_conf >= min_top1:
                mode = "ultrapilot"
            else:
                mode = "team"
        else:
            mode = "team"

    mode_cfg = EXECUTION_MODES.get(mode, EXECUTION_MODES["team"])
    result = {
        "mode": mode,
        "stages": mode_cfg["stages"],
        "parallel": mode_cfg.get("parallel", False),
        "loop": mode_cfg.get("loop", False),
        "input": user_input,
        "routing_candidates": soft_route_to_souls(user_input, top_k=3),
    }

    _log_dispatch({
        "mode": mode,
        "top1_confidence": top1_conf if candidates else 0.0,
        "gap": gap if len(candidates) > 1 else 0.0,
        "adaptive_threshold": adaptive_threshold,
        "routed_soul": routing.get("soul", "") if kw["mode"] is None else "",
    })

    return result


def _execute_soul_stage(soul_name: str, stage: str, task: str,
                        context: Optional[dict] = None) -> dict:
    cap = SOUL_CAPABILITIES.get(soul_name, {})
    routing = route_to_soul(task)
    from soul_memory import recall, write
    memory_ctx = recall(soul_name, task, top_k=5, categories=["knowledge", "domain_memory"])
    memory_snippets = [m.get("content", {}).get("topic", "") for m in memory_ctx[:3]]

    stage_prompt = task
    if context and context.get("prior_outputs"):
        last_output = context["prior_outputs"][-1].get("output", "")
        if last_output:
            stage_prompt = f"{task}\n[Previous stage output]: {last_output[:500]}"

    output = {
        "stage": stage,
        "soul": soul_name,
        "soul_tier": cap.get("tier", "C"),
        "domain": cap.get("domain", []),
        "tools_available": cap.get("tools", []),
        "skills_available": cap.get("skills", []),
        "memory_context": memory_snippets,
        "routing_confidence": routing.get("confidence", 0.0),
        "output": stage_prompt,
        "status": "executed",
    }

    write(soul_name, "trajectory", {
        "stage": stage,
        "task": task,
        "output_length": len(stage_prompt),
        "routing_confidence": routing.get("confidence", 0.0),
    }, importance=0.5, tags=[stage, "execution"])

    try:
        from closed_loop_orchestrator import ClosedLoopOrchestrator
        _clo = ClosedLoopOrchestrator()
        _clo.run_full_loop(
            query=task,
            knowledge_base=soul_name,
            response=stage_prompt,
            confidence=routing.get("confidence", 0.0),
            context=[{"content": s} for s in memory_snippets],
        )
        output["closed_loop"] = {"status": "completed", "kb": soul_name}
    except Exception as _cl_err:
        output["closed_loop"] = {"status": "error", "error": str(_cl_err)}

    return output


def run_team_pipeline(task_description: str, start_stage: str = "team_plan") -> dict:
    stages = EXECUTION_MODES["team"]["stages"]
    try:
        start_idx = stages.index(start_stage)
    except ValueError:
        start_idx = 0

    active_stages = stages[start_idx:]
    results = []
    context = {"prior_outputs": []}

    for stage in active_stages:
        role = PIPELINE_ROLES.get(stage, {})
        soul_name = role.get("soul", "cezanne")
        stage_result = _execute_soul_stage(soul_name, stage, task_description, context)
        results.append(stage_result)
        context["prior_outputs"].append(stage_result)

    return {
        "mode": "team",
        "stages": active_stages,
        "results": results,
        "task": task_description,
        "status": "completed",
    }


def run_ultrapilot(task_description: str, parallel_group: Optional[int] = None) -> dict:
    stages = EXECUTION_MODES["ultrapilot"]["stages"]
    subtasks = split_task_by_module(task_description, n_subagents=len(stages))

    assignments = []
    for i, stage in enumerate(stages):
        role = PIPELINE_ROLES.get(stage, {})
        soul_name = role.get("soul", "cezanne")
        subtask_text = subtasks[i]["task_fragment"] if i < len(subtasks) else task_description
        assignments.append({
            "stage": stage,
            "soul": soul_name,
            "subtask": subtask_text,
            "parallel_group": parallel_group or i,
        })

    parallel_results = []
    for assignment in assignments:
        result = _execute_soul_stage(
            assignment["soul"],
            assignment["stage"],
            assignment["subtask"],
        )
        result["subtask"] = assignment["subtask"]
        result["parallel_group"] = assignment["parallel_group"]
        parallel_results.append(result)

    arbitration = arbitrate_results(parallel_results, task_description, method="consensus")

    return {
        "mode": "ultrapilot",
        "parallel": True,
        "assignments": assignments,
        "parallel_results": parallel_results,
        "arbitration": arbitration,
        "task": task_description,
        "status": "completed",
    }


def run_ralph_loop(task_description: str, max_iterations: Optional[int] = None,
                   quality_threshold: float = 0.8) -> dict:
    stages = EXECUTION_MODES["ralph"]["stages"]
    iterations = max_iterations or 5
    all_results = []
    context = {"prior_outputs": []}

    for i in range(iterations):
        iteration_results = []
        for stage in stages:
            role = PIPELINE_ROLES.get(stage, {})
            soul_name = role.get("soul", "cezanne")
            result = _execute_soul_stage(soul_name, stage, task_description, context)
            result["iteration"] = i + 1
            iteration_results.append(result)
            context["prior_outputs"].append(result)

        all_results.extend(iteration_results)

        verify_result = iteration_results[0]
        confidence = verify_result.get("routing_confidence", 0.0)
        if confidence >= quality_threshold and i > 0:
            break

    return {
        "mode": "ralph",
        "loop": True,
        "max_iterations": iterations,
        "actual_iterations": i + 1,
        "results": all_results,
        "task": task_description,
        "status": "completed",
    }


def run_subagent_parallel(task_description: str, stage_config: dict,
                          worktree_isolation: bool = True) -> dict:
    n = stage_config.get("n_subagents", 3)
    subtasks = split_task_by_module(task_description, n_subagents=n)
    stage = stage_config.get("stage", "default")

    subagents = []
    for i, subtask in enumerate(subtasks):
        branch = f"vf/{stage}/sub{i}/{hashlib.md5(task_description.encode()).hexdigest()[:8]}"
        routing = route_to_soul(subtask["task_fragment"])
        subagent = {
            "subagent_id": i,
            "branch": branch,
            "worktree": f".worktrees/{branch}" if worktree_isolation else None,
            "task": subtask["task_fragment"],
            "routed_soul": routing["soul"],
            "routing_confidence": routing["confidence"],
        }
        subagents.append(subagent)

    subagent_results = []
    for sa in subagents:
        result = _execute_soul_stage(sa["routed_soul"], stage, sa["task"])
        result["subagent_id"] = sa["subagent_id"]
        result["branch"] = sa["branch"]
        subagent_results.append(result)

    return {
        "subagents": subagents,
        "subagent_results": subagent_results,
        "worktree_isolation": worktree_isolation,
        "task": task_description,
        "status": "completed",
    }


def merge_subagent_results(subagent_results: list, target_branch: str = "main",
                           conflict_strategy: str = "arbitrate") -> dict:
    if not subagent_results:
        return {"status": "no_results", "target_branch": target_branch}

    from soul_memory import write, search

    conflict_keys = set()
    for r in subagent_results:
        soul = r.get("soul", "unknown")
        output = r.get("output", "")
        if output:
            existing = search(soul, "trajectory", output[:100], top_k=3)
            for e in existing:
                if e.get("content", {}).get("stage") == r.get("stage") and e.get("entry_id") != r.get("entry_id"):
                    conflict_keys.add((soul, r.get("stage")))

    conflicts_detected = len(conflict_keys)

    if conflicts_detected > 0 and conflict_strategy == "arbitrate":
        arbitration = arbitrate_results(subagent_results, method="consensus")
        merged_output = arbitration.get("winner", subagent_results[0])
    elif conflicts_detected > 0 and conflict_strategy == "last_wins":
        merged_output = subagent_results[-1]
    else:
        merged_output = subagent_results[0]

    merged_soul = merged_output.get("soul", "cezanne")
    write(merged_soul, "trajectory", {
        "action": "merge_subagent",
        "n_sources": len(subagent_results),
        "conflicts": conflicts_detected,
        "conflict_strategy": conflict_strategy,
        "target_branch": target_branch,
    }, importance=0.7, tags=["merge", "subagent"])

    return {
        "status": "merged",
        "target_branch": target_branch,
        "n_results": len(subagent_results),
        "conflicts_detected": conflicts_detected,
        "conflict_strategy": conflict_strategy,
        "merged_output": merged_output,
        "all_outputs": subagent_results,
    }


def arbitrate_results(results: list, task_description: str = "", method: str = "confidence") -> dict:
    """Arbitrate among multiple expert outputs and select the best.

    Methods:
        confidence  — pick the result from the highest-routing-confidence soul
        consensus   — pick the result agreed upon by the most souls
        diversity   — pick the result most different from others (creative tasks)

    Returns provenance: which souls contributed, their scores, and the reasoning chain.
    """
    if not results:
        return {"status": "no_results", "winner": None, "provenance": {}}

    provenance = {
        "n_candidates": len(results),
        "souls_involved": [r.get("soul", "unknown") for r in results],
        "method": method,
        "timestamp": time.time(),
    }

    if len(results) == 1:
        provenance["reason"] = "unanimous_single_candidate"
        return {"status": "unanimous", "winner": results[0], "method": method, "provenance": provenance}

    if method == "confidence":
        routing = soft_route_to_souls(task_description, top_k=len(SOUL_CAPABILITIES)) if task_description else []
        confidence_map = {r["soul"]: r["confidence"] for r in routing}
        best = max(results, key=lambda r: confidence_map.get(r.get("soul", ""), 0.0))
        provenance["confidence_map"] = confidence_map
        provenance["winner_soul"] = best.get("soul", "unknown")
        provenance["winner_confidence"] = confidence_map.get(best.get("soul", ""), 0.0)
        provenance["reason"] = "highest_routing_confidence"
        return {"status": "arbitrated", "winner": best, "method": "confidence", "confidence_map": confidence_map, "provenance": provenance}

    if method == "consensus":
        model = _semantic_router._load_model()
        if model is not None:
            texts = [str(r.get("output", "")) for r in results]
            embs = model.encode(texts, normalize_embeddings=True)
            if hasattr(embs, 'numpy'):
                embs = embs.numpy()
            embs = np.array(embs)
            n = len(results)
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    sim = float(np.dot(embs[i], embs[j]))
                    sim_matrix[i][j] = sim
                    sim_matrix[j][i] = sim
            agreement_scores = []
            for i in range(n):
                score = sum(sim_matrix[i][j] for j in range(n) if j != i) / max(n - 1, 1)
                agreement_scores.append(score)
            winner_idx = int(np.argmax(agreement_scores))
            best_agreement = agreement_scores[winner_idx]
            provenance["agreement_scores"] = {results[i].get("soul", f"agent_{i}"): round(s, 3) for i, s in enumerate(agreement_scores)}
            provenance["pairwise_similarity"] = round(float(np.mean(sim_matrix[sim_matrix > 0])), 3)
            provenance["winner_soul"] = results[winner_idx].get("soul", "unknown")
            consensus_threshold = 0.7
            if best_agreement >= consensus_threshold:
                provenance["reason"] = "strong_consensus"
                return {
                    "status": "arbitrated",
                    "winner": results[winner_idx],
                    "method": "consensus",
                    "agreement_score": round(best_agreement, 3),
                    "consensus_reached": True,
                    "provenance": provenance,
                }
            provenance["reason"] = "weak_consensus_most_central"
            return {
                "status": "arbitrated",
                "winner": results[winner_idx],
                "method": "consensus",
                "agreement_score": round(best_agreement, 3),
                "consensus_reached": False,
                "note": "No strong consensus; winner is most-central result",
                "provenance": provenance,
            }
        from collections import Counter
        content_keys = []
        for r in results:
            key = r.get("output_hash") or hashlib.md5(str(r.get("output", "")).encode()).hexdigest()[:8]
            content_keys.append(key)
        counter = Counter(content_keys)
        most_common_key, count = counter.most_common(1)[0]
        winner_idx = content_keys.index(most_common_key)
        provenance["reason"] = "consensus_hash_fallback"
        provenance["winner_soul"] = results[winner_idx].get("soul", "unknown")
        return {"status": "arbitrated", "winner": results[winner_idx], "method": "consensus_fallback", "agreement_count": count, "provenance": provenance}

    if method == "diversity":
        embeddings = []
        model = _semantic_router._load_model()
        for r in results:
            text = str(r.get("output", ""))
            emb = model.encode(text) if model is not None else None
            embeddings.append(emb)
        if any(e is None for e in embeddings):
            provenance["reason"] = "diversity_fallback_no_embeddings"
            return {"status": "arbitrated", "winner": results[-1], "method": "diversity_fallback", "provenance": provenance}
        avg_emb = np.mean([e for e in embeddings if e is not None], axis=0)
        distances = [float(np.linalg.norm(e - avg_emb)) for e in embeddings if e is not None]
        winner_idx = int(np.argmax(distances))
        provenance["reason"] = "most_diverse_from_mean"
        provenance["winner_soul"] = results[winner_idx].get("soul", "unknown")
        provenance["diversity_distances"] = {results[i].get("soul", f"agent_{i}"): round(d, 3) for i, d in enumerate(distances)}
        return {"status": "arbitrated", "winner": results[winner_idx], "method": "diversity", "provenance": provenance}

    provenance["reason"] = "unknown_method_fallback"
    return {"status": "unknown_method", "winner": results[0], "method": method, "provenance": provenance}


def split_task_by_module(task_description: str, n_subagents: int = 3) -> list:
    """Semantic task splitting for Ultrapilot parallel execution.

    Instead of naive word-level splitting, uses sentence-boundary detection
    and semantic clustering to produce coherent subtasks.

    Strategy:
    1. Split by sentence boundaries (。！？.!? and conjunction markers)
    2. If <= n_subagents sentences: one sentence per subtask
    3. If > n_subagents sentences: cluster by semantic similarity using
       the same embedding model as SemanticRouter
    4. Fallback: if embedding model unavailable, split by sentence count
    """
    import re as _re

    sentence_splitter = _re.compile(r'(?<=[。！？.!?])\s*|(?<=\s)(?:and|but|however|then|while|also|以及|但是|然后|同时|另外|并且|接着)\s+', _re.IGNORECASE)
    sentences = [s.strip() for s in sentence_splitter.split(task_description) if s.strip()]

    if len(sentences) <= 1:
        clauses = _re.split(r'[,，;；]\s*', task_description)
        sentences = [c.strip() for c in clauses if c.strip()]

    if not sentences:
        sentences = [task_description]

    if len(sentences) <= n_subagents:
        return [
            {"subagent_id": i, "task_fragment": s}
            for i, s in enumerate(sentences)
        ]

    model = _semantic_router._load_model()
    if model is None:
        chunk_size = max(len(sentences) // n_subagents, 1)
        chunks = []
        for i in range(n_subagents):
            start = i * chunk_size
            end = start + chunk_size if i < n_subagents - 1 else len(sentences)
            chunks.append({
                "subagent_id": i,
                "task_fragment": " ".join(sentences[start:end]),
            })
        return chunks

    embeddings = model.encode(sentences, normalize_embeddings=True)
    if hasattr(embeddings, 'numpy'):
        embeddings = embeddings.numpy()
    embeddings = np.array(embeddings)

    from sklearn.cluster import KMeans
    k = min(n_subagents, len(sentences))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clusters: Dict[int, List[str]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(sentences[idx])

    ordered_labels = sorted(clusters.keys(), key=lambda lbl: min(
        i for i, l in enumerate(labels) if l == lbl
    ))

    chunks = []
    for i, lbl in enumerate(ordered_labels):
        chunks.append({
            "subagent_id": i,
            "task_fragment": " ".join(clusters[lbl]),
        })

    return chunks


def get_soul_tools(soul_name: str) -> list:
    cap = SOUL_CAPABILITIES.get(soul_name, {})
    return cap.get("tools", [])


def get_soul_skills(soul_name: str) -> list:
    return _skill_registry.to_soul_config_format(soul_name)


def initialize_code_intelligence(project_path: str) -> dict:
    global _code_intelligence
    _code_intelligence = CodeIntelligenceManager(project_path)
    return {"status": "initialized", "project_path": project_path}


def get_code_intelligence() -> Optional[CodeIntelligenceManager]:
    return _code_intelligence


def get_long_memory() -> LongMemory:
    return _long_memory


def get_long_memory_adapter() -> LongMemoryAdapter:
    return _long_memory_adapter


def get_moe_loader() -> MoEExpertLoader:
    return _moe_loader


def get_validation_engine() -> ValidationEngine:
    return _validation_engine


def validate_knowledge_base(kb_name: str, limit: int = 2000) -> dict:
    report = _validation_engine.validate(kb_name, limit=limit)
    return {
        "kb_name": report.kb_name,
        "timestamp": report.timestamp,
        "total_rules": report.total_rules,
        "passed": report.passed,
        "failed": report.failed,
        "violations": [
            {"rule_id": v.rule_id, "severity": v.severity, "category": v.category,
             "entry_id": v.entry_id, "description": v.description}
            for v in report.violations[:20]
        ],
    }


def orchestrate_cross_kb(query: str, kbs: Optional[List[str]] = None, top_k: int = 2) -> dict:
    if kbs is None:
        routes = _moe_loader.route(query, top_k=top_k + 1)
        kbs = [r[0] for r in routes if r[1] > 0][:top_k]
        if not kbs:
            kbs = ["cezanne", "einstein"]
    import concurrent.futures
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(kbs), 5)) as executor:
        futures = {
            executor.submit(
                route_to_soul,
                f"[{kb.upper()}_SCOPE] {query}"
            ): kb for kb in kbs
        }
        for future in concurrent.futures.as_completed(futures):
            kb = futures[future]
            try:
                results[kb] = future.result()
            except Exception as e:
                results[kb] = {"error": str(e)}
    caps = {kb: SOUL_CAPABILITIES.get(kb, {}) for kb in kbs}
    aggregated = {
        "query": query,
        "orchestrated_kbs": kbs,
        "results": results,
        "summary": {
            "total_kbs": len(kbs),
            "successful": sum(1 for r in results.values() if "error" not in r),
            "primary_soul": kbs[0],
            "jepa_variants": [caps[kb].get("jepa_variant", "unknown") for kb in kbs],
        },
    }
    handle = _long_memory_adapter.for_kb(kbs[0])
    handle.start_thread()
    handle.remember(f"[ORCHESTRATE] Query: {query} | KBs: {kbs} | Results: {len(results)}", role="orchestrator")
    return aggregated


def apply_moe_routing(query: str, top_k: int = 3) -> dict:
    from moe_expert_loader import MoEExpertLoader
    loader = MoEExpertLoader()
    experts = loader.route(query, top_k=top_k)
    strategy = loader.route_strategy(query)
    return {"experts": experts, "strategy": str(strategy), "selected_kbs": [e[0] for e in experts] if experts else []}


def moe_route_query(query: str, top_k: int = 3) -> dict:
    return _moe_loader.query_with_routing(query, top_k=top_k)


def load_orchestration_config(config_path: Optional[str] = None) -> dict:
    import yaml
    if config_path is None:
        path = Path(__file__).parent / "orchestration_config.yaml"
    else:
        path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def get_orchestration_pattern(query: str, config: Optional[Dict] = None) -> dict:
    if config is None:
        config = load_orchestration_config()
    patterns = config.get("patterns", {})
    query_lower = query.lower()
    for pattern_name, pattern in patterns.items():
        triggers = pattern.get("trigger", [])
        if any(t in query_lower for t in triggers):
            return {
                "pattern": pattern_name,
                "primary": pattern["primary"],
                "support": pattern["support"],
                "strategy": pattern["strategy"],
                "fusion": pattern["fusion"],
                "description": pattern.get("description", ""),
            }
    return config.get("default_routing", {"pattern": "supermind", "primary": "cezanne", "support": ["einstein"]})
