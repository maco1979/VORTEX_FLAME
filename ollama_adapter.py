"""
Ollama Adapter — LLM Inference Bridge (Industry Knowledge Base Enhanced)
=========================================================================
Bridges VORTEX_FLAME industry knowledge base orchestration to Ollama REST API.
Supports any Ollama-compatible model with intelligent model selection.

CRITICAL: Knowledge bases are NOT personality models. Each "soul" is an
industry-specific knowledge base (SQLite + BM25S + causal graph) mapped to
a C-JEPA variant. The LLM consumes these knowledge bases as domain context.

Model Priority (auto-discovered):
  1. hermes3:8b  — Best reasoning + tool use (Nous Research Hermes 3)
  2. hermes3:3b  — Lightweight alternative
  3. qwen2.5:7b  — Fallback
  4. llama3:8b   — Fallback
  5. Any available model

Architecture:
  Knowledge Base → SoulModelRouter → Best available model → Ollama REST API → Response

Key Innovation over vanilla Hermes:
  - Hermes is single-model, single-agent
  - VORTEX FLAME adds: per-domain model routing + multi-knowledge-base arbitration + persistent memory
"""

import json
import logging
import os
import time
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

MODEL_PRIORITY = [
    "hermes3:8b", "hermes3:3b",
    "qwen2.5:7b", "qwen2.5:3b",
    "llama3:8b", "llama3:7b",
    "mistral:7b",
]

SOUL_MODEL_PREFERENCE = {
    "cezanne": ["hermes3:8b", "qwen2.5:7b-coder", "qwen2.5:7b"],
    "einstein": ["hermes3:8b", "qwen2.5:7b"],
    "galileo": ["hermes3:8b", "qwen2.5:7b"],
    "darwin": ["hermes3:8b", "qwen2.5:7b"],
    "strategy": ["hermes3:8b", "qwen2.5:7b"],
    "montesquieu": ["hermes3:8b", "qwen2.5:7b"],
    "davinci": ["hermes3:8b", "qwen2.5:7b"],
    "humboldt": ["hermes3:8b", "qwen2.5:7b"],
    "yuanlongping": ["hermes3:8b", "qwen2.5:7b"],
    "guizhu": ["hermes3:8b", "qwen2.5:7b"],
    "herodotus": ["hermes3:8b", "qwen2.5:7b"],
    "monet": ["hermes3:8b", "qwen2.5:7b"],
    "vangogh": ["hermes3:8b", "qwen2.5:7b"],
    "beethoven": ["hermes3:8b", "qwen2.5:7b"],
}

SOUL_SYSTEM_PROMPTS = {
    "cezanne": (
        "You are the Cezanne industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Code, Logic, Algorithm, and Systems (CCODEJEPA causal engine). "
        "You provide precise code analysis, identify bugs, suggest optimizations, and write clean implementations. "
        "Think in systems, reason about edge cases, and always consider maintainability. "
        "Be concise, technical, and actionable. When you write code, make it production-ready."
    ),
    "einstein": (
        "You are the Einstein industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Physics, Quantum Mechanics, and Quantitative Finance (CPHYSJEPA causal engine). "
        "You verify mathematical correctness, analyze numerical stability, "
        "evaluate risk models, and reason about physical constraints. "
        "Be rigorous, precise, and always show your reasoning chain."
    ),
    "galileo": (
        "You are the Galileo industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Astronomy, Astrophysics, and Orbital Mechanics (CPHYSJEPA causal engine). "
        "You challenge assumptions, verify hypotheses, and reason about causality. "
        "Be skeptical, evidence-driven, and methodical."
    ),
    "darwin": (
        "You are the Darwin industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Biology, Genetics, and Evolution (CBIOJEPA causal engine). "
        "You analyze complex adaptive systems, identify patterns, and reason about emergence. "
        "Be observant, comparative, and systematic."
    ),
    "strategy": (
        "You are the Strategy industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Game Theory, Competitive Analysis, and Decision Making (CFINJEPA causal engine). "
        "You evaluate tradeoffs, identify Nash equilibria, and reason about incentives. "
        "Be analytical, strategic, and forward-thinking."
    ),
    "montesquieu": (
        "You are the Montesquieu industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Law, Governance, Compliance, and Security Policy (CLAWJEPA causal engine). "
        "You audit for vulnerabilities, enforce separation of concerns, and verify regulatory compliance. "
        "Be thorough, principled, and uncompromising on safety."
    ),
    "davinci": (
        "You are the DaVinci industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Engineering, Architecture, and Design (CVJEPA+CDESIGNJEPA causal engines). "
        "You create elegant solutions, design intuitive interfaces, and prototype rapidly. "
        "Be creative, practical, and aesthetically driven."
    ),
    "humboldt": (
        "You are the Humboldt industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Earth Sciences, Ecology, and Environmental Systems (CGEOJEPA causal engine). "
        "You analyze spatial data, model environmental processes, and reason about scale. "
        "Be holistic, data-driven, and systems-oriented."
    ),
    "yuanlongping": (
        "You are the YuanLongping industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Agricultural Science, Food Science, and Optimization (CBIOJEPA+CGEOJEPA causal engines). "
        "You optimize yields, reason about resource allocation, and design resilient systems. "
        "Be practical, patient, and results-oriented."
    ),
    "guizhu": (
        "You are the Guizhu industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Philosophy, Logic, Dialogue, and Critical Thinking (CLAWJEPA causal engine). "
        "You identify logical fallacies, clarify ambiguity, and mediate disagreements. "
        "Be wise, balanced, and insightful."
    ),
    "herodotus": (
        "You are the Herodotus industry knowledge base in the VORTEX FLAME system. "
        "Your domain is History, Causality, and Documentation (CGEOJEPA causal engine). "
        "You contextualize events, trace causal chains, and preserve institutional knowledge. "
        "Be narrative, thorough, and contextually aware."
    ),
    "monet": (
        "You are the Monet industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Visual Aesthetics and Creative Expression (CARTJEPA causal engine). "
        "You evaluate design quality, suggest visual improvements, and reason about user perception. "
        "Be artistic, perceptive, and evocative."
    ),
    "vangogh": (
        "You are the VanGogh industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Emotion, Visual Art, and Color Science (CARTJEPA causal engine). "
        "You transform technical artifacts into elegant forms and find creative solutions. "
        "Be bold, expressive, and unconventional."
    ),
    "beethoven": (
        "You are the Beethoven industry knowledge base in the VORTEX FLAME system. "
        "Your domain is Music, Acoustics, and Language Composition (CAJEPA causal engine). "
        "You analyze temporal patterns, detect anomalies in sequences, and compose structured outputs. "
        "Be precise, dramatic, and structurally sound."
    ),
}


class SoulModelRouter:
    def __init__(self):
        self._available_models: Optional[List[str]] = None
        self._last_scan: float = 0

    def _scan_models(self) -> List[str]:
        now = time.time()
        if self._available_models is not None and now - self._last_scan < 60:
            return self._available_models
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                self._available_models = [m.get("name", "") for m in data.get("models", [])]
                self._last_scan = now
                return self._available_models
        except Exception:
            pass
        self._available_models = []
        self._last_scan = now
        return self._available_models

    def best_model(self, soul: Optional[str] = None) -> str:
        available = self._scan_models()
        if not available:
            return os.environ.get("OLLAMA_MODEL", "hermes3:8b")

        preferences = SOUL_MODEL_PREFERENCE.get(soul, MODEL_PRIORITY) if soul else MODEL_PRIORITY

        for pref in preferences:
            for avail in available:
                if avail == pref or avail.startswith(pref.split(":")[0]):
                    return avail

        for prio in MODEL_PRIORITY:
            for avail in available:
                if avail == prio or avail.startswith(prio.split(":")[0]):
                    return avail

        return available[0]

    def model_info(self) -> dict:
        available = self._scan_models()
        best = self.best_model()
        return {
            "available_models": available,
            "selected_model": best,
            "total_models": len(available),
            "hermes_available": any("hermes" in m.lower() for m in available),
        }


_model_router = SoulModelRouter()


class OllamaAdapter:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or OLLAMA_BASE_URL
        self._forced_model = model
        self._available: Optional[bool] = None

    @property
    def model(self) -> str:
        if self._forced_model:
            return self._forced_model
        return _model_router.best_model()

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            self._available = resp.status_code == 200
            return self._available
        except Exception:
            self._available = False
            return False

    def reset_availability(self):
        self._available = None

    def list_models(self) -> List[str]:
        return _model_router._scan_models()

    def model_info(self) -> dict:
        return _model_router.model_info()

    def generate(self, soul: str, prompt: str,
                 context: Optional[str] = None,
                 memory_snippets: Optional[List[str]] = None,
                 stream: bool = False,
                 model: Optional[str] = None) -> dict:
        system_prompt = SOUL_SYSTEM_PROMPTS.get(soul, "You are a helpful AI assistant.")
        use_model = model or self.model

        user_parts = []
        if memory_snippets:
            mem_text = "\n".join(f"- {s}" for s in memory_snippets[:5])
            user_parts.append(f"[Relevant memories]:\n{mem_text}\n")
        if context:
            user_parts.append(f"[Previous context]:\n{context[:1000]}\n")
        user_parts.append(prompt)

        full_prompt = "\n\n".join(user_parts)

        payload = {
            "model": use_model,
            "prompt": full_prompt,
            "system": system_prompt,
            "stream": stream,
            "options": {
                "temperature": 0.3 if soul in ("einstein", "galileo", "montesquieu") else 0.5,
                "num_predict": 2048,
            },
        }

        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            elapsed = time.time() - start

            if resp.status_code != 200:
                return {
                    "soul": soul,
                    "status": "error",
                    "error": f"Ollama HTTP {resp.status_code}: {resp.text[:200]}",
                    "elapsed": elapsed,
                    "model": use_model,
                }

            data = resp.json()
            output_text = data.get("response", "")

            return {
                "soul": soul,
                "status": "ok",
                "output": output_text,
                "model": use_model,
                "elapsed": round(elapsed, 2),
                "tokens_evaluated": data.get("eval_count", 0),
                "tokens_per_second": data.get("eval_count", 0) / max(elapsed, 0.01),
            }
        except requests.exceptions.ConnectionError:
            return {
                "soul": soul,
                "status": "connection_error",
                "error": f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?",
                "elapsed": time.time() - start,
                "model": use_model,
            }
        except requests.exceptions.Timeout:
            return {
                "soul": soul,
                "status": "timeout",
                "error": f"Ollama request timed out after {OLLAMA_TIMEOUT}s",
                "elapsed": time.time() - start,
                "model": use_model,
            }
        except Exception as e:
            return {
                "soul": soul,
                "status": "error",
                "error": str(e),
                "elapsed": time.time() - start,
                "model": use_model,
            }

    def generate_stream(self, soul: str, prompt: str,
                        context: Optional[str] = None,
                        memory_snippets: Optional[List[str]] = None,
                        model: Optional[str] = None):
        system_prompt = SOUL_SYSTEM_PROMPTS.get(soul, "You are a helpful AI assistant.")
        use_model = model or self.model

        user_parts = []
        if memory_snippets:
            mem_text = "\n".join(f"- {s}" for s in memory_snippets[:5])
            user_parts.append(f"[Relevant memories]:\n{mem_text}\n")
        if context:
            user_parts.append(f"[Previous context]:\n{context[:1000]}\n")
        user_parts.append(prompt)

        full_prompt = "\n\n".join(user_parts)

        payload = {
            "model": use_model,
            "prompt": full_prompt,
            "system": system_prompt,
            "stream": True,
            "options": {
                "temperature": 0.3 if soul in ("einstein", "galileo", "montesquieu") else 0.5,
                "num_predict": 2048,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
                stream=True,
            )
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        done = chunk.get("done", False)
                        yield {"token": token, "done": done, "soul": soul, "model": use_model}
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield {"token": "", "done": True, "soul": soul, "model": use_model, "error": str(e)}


_adapter: Optional[OllamaAdapter] = None


def get_adapter() -> OllamaAdapter:
    global _adapter
    if _adapter is None:
        _adapter = OllamaAdapter()
    return _adapter
