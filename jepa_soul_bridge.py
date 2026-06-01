"""
JEPA-Soul Bridge — Connects JEPA embeddings to soul_memory and LLM inference
=============================================================================

Architecture (Dual-Pathway v2):
  Path A (RAG):     raw_input → soul_memory text retrieval → text embeddings
  Path B (C-JEPA):  raw_input → JEPA encode → world slot embeddings
  Fusion:           Cross-Attention(text_embeds, world_embeds) → fused representation
  Output:           fused → LLM generate → response

Supports all JEPA modalities:
  - A-JEPA (audio/music)
  - V-JEPA (visual/image)
  - FIN-JEPA (financial/time-series)
  - CODE-JEPA (code/AST)
  - PHYS-JEPA (physics)
  - ART-JEPA (aesthetics)
  - DESIGN-JEPA (design)
  - BIO-JEPA (biology/genomics)
  - GEO-JEPA (geography/ecology)
  - LAW-JEPA (legal/compliance)

Usage:
  bridge = JEPASoulBridge()
  result = bridge.process(
      raw_input=audio_clip,
      modality="audio",
      soul="beethoven",
      query="What key is this in?",
  )

  # Dual-pathway query (new):
  result = bridge.dual_pathway_query(
      query="设备故障后如何处理",
      soul="cezanne",
  )
"""

import os
import sys
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(os.path.dirname(__file__)))
from soul_memory import SoulMemoryEngine


class JEPAModality(Enum):
    VISUAL = "visual"
    AUDIO = "audio"
    PHYSICS = "physics"
    ART = "art"
    DESIGN = "design"
    FINANCIAL = "financial"
    CODE = "code"
    BIOLOGY = "biology"
    GEOGRAPHY = "geography"
    LAW = "law"


MODALITY_SLOT_NAMES = {
    JEPAModality.AUDIO: [
        "drums_percussion", "bass", "vocals", "melody_lead", "harmony_pads", "effects_ambient", "silence_gap"
    ],
    JEPAModality.VISUAL: [
        "foreground_subject", "background_scene", "text_overlay", "color_palette", "composition", "lighting", "depth_layer"
    ],
    JEPAModality.FINANCIAL: [
        "trend_direction", "momentum", "volatility", "volume_profile", "support_resistance", "market_regime"
    ],
    JEPAModality.CODE: [
        "control_flow", "data_structures", "api_calls", "error_handling", "side_effects", "type_system", "concurrency"
    ],
    JEPAModality.PHYSICS: [
        "kinematics", "forces", "energy", "fields", "conservation_laws", "boundary_conditions"
    ],
    JEPAModality.ART: [
        "composition", "color_harmony", "texture", "style_period", "emotional_tone", "symbolism", "technique"
    ],
    JEPAModality.DESIGN: [
        "layout_grid", "typography", "spacing_rhythm", "visual_hierarchy", "interaction_pattern", "accessibility"
    ],
    JEPAModality.BIOLOGY: [
        "gene_expression", "protein_structure", "metabolic_pathway", "cell_signal", "phenotype_trait", "evolutionary_pressure"
    ],
    JEPAModality.GEOGRAPHY: [
        "terrain_elevation", "vegetation_cover", "water_systems", "climate_pattern", "human_activity", "geological_structure"
    ],
    JEPAModality.LAW: [
        "legal_rule", "precedent_case", "jurisdiction_scope", "temporal_validity", "exception_clause", "interpretation_method"
    ],
}

MODALITY_SOUL_MAP = {
    JEPAModality.AUDIO: ["beethoven"],
    JEPAModality.VISUAL: ["vangogh", "monet"],
    JEPAModality.FINANCIAL: ["strategy", "einstein"],
    JEPAModality.CODE: ["cezanne"],
    JEPAModality.PHYSICS: ["einstein", "galileo"],
    JEPAModality.ART: ["monet", "vangogh"],
    JEPAModality.DESIGN: ["davinci"],
    JEPAModality.BIOLOGY: ["darwin", "yuanlongping"],
    JEPAModality.GEOGRAPHY: ["humboldt"],
    JEPAModality.LAW: ["montesquieu"],
}


@dataclass
class SlotDescription:
    slot_id: int
    name: str
    activation: float
    summary: str


@dataclass
class BridgeResult:
    modality: JEPAModality
    slot_descriptions: List[SlotDescription]
    knowledge_context: List[Dict]
    llm_response: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class SlotToTextDecoder:
    def decode(self, slots: torch.Tensor, modality: JEPAModality) -> List[SlotDescription]:
        slot_names = MODALITY_SLOT_NAMES.get(modality, [f"slot_{i}" for i in range(slots.shape[-2])])
        descriptions = []

        if slots.dim() == 4:
            slots = slots.mean(dim=(0, 1))
        elif slots.dim() == 3:
            slots = slots.mean(dim=0)

        n_slots = min(slots.shape[0], len(slot_names))

        for i in range(n_slots):
            slot_vec = slots[i]
            activation = float(slot_vec.norm().item())
            name = slot_names[i]

            summary = self._summarize_slot(name, activation, slot_vec, modality)
            descriptions.append(SlotDescription(
                slot_id=i, name=name, activation=activation, summary=summary
            ))

        descriptions.sort(key=lambda d: d.activation, reverse=True)
        return descriptions

    def _summarize_slot(self, name: str, activation: float, vec: torch.Tensor,
                        modality: JEPAModality) -> str:
        if activation < 0.5:
            return f"{name}: inactive"
        elif activation < 1.5:
            return f"{name}: moderate"
        elif activation < 3.0:
            return f"{name}: strong"
        else:
            return f"{name}: dominant"

    def to_query_text(self, descriptions: List[SlotDescription], modality: JEPAModality) -> str:
        active = [d for d in descriptions if d.activation >= 0.5]
        if not active:
            return f"{modality.value} input with no significant features detected"

        parts = [f"{d.summary}" for d in active[:5]]
        return f"{modality.value} analysis: " + ", ".join(parts)


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "", max_tokens: int = 2048,
                 temperature: float = 0.7) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...


class CloudBackend(LLMBackend):
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini"):
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system: str = "", max_tokens: int = 2048,
                 temperature: float = 0.7) -> str:
        if not self.is_available():
            return "[Cloud LLM unavailable: no API key]"

        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Cloud LLM error: {e}]"


class LocalBackend(LLMBackend):
    def __init__(self, model_path: Optional[str] = None, backend: str = "llama_cpp"):
        self.model_path = model_path
        self.backend = backend
        self._model = None

    def is_available(self) -> bool:
        return self.model_path is not None and os.path.exists(self.model_path) if self.model_path else False

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.is_available():
            return None
        try:
            if self.backend == "llama_cpp":
                from llama_cpp import Llama
                self._model = Llama(model_path=self.model_path, n_ctx=4096, verbose=False)
            return self._model
        except Exception:
            return None

    def generate(self, prompt: str, system: str = "", max_tokens: int = 2048,
                 temperature: float = 0.7) -> str:
        model = self._load_model()
        if model is None:
            return "[Local LLM unavailable: model not loaded]"

        try:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            output = model(full_prompt, max_tokens=max_tokens, temperature=temperature)
            return output["choices"][0]["text"].strip()
        except Exception as e:
            return f"[Local LLM error: {e}]"


class HybridBackend(LLMBackend):
    def __init__(self, cloud: CloudBackend, local: LocalBackend,
                 complexity_threshold: float = 0.5):
        self.cloud = cloud
        self.local = local
        self.complexity_threshold = complexity_threshold

    def is_available(self) -> bool:
        return self.cloud.is_available() or self.local.is_available()

    def _estimate_complexity(self, prompt: str) -> float:
        score = 0.0
        if len(prompt) > 1000:
            score += 0.2
        if any(kw in prompt.lower() for kw in ["analyze", "compare", "evaluate", "reason"]):
            score += 0.2
        if any(kw in prompt.lower() for kw in ["counterfactual", "what if", "hypothetical"]):
            score += 0.3
        if prompt.count("\n") > 10:
            score += 0.1
        return min(score, 1.0)

    def generate(self, prompt: str, system: str = "", max_tokens: int = 2048,
                 temperature: float = 0.7) -> str:
        complexity = self._estimate_complexity(prompt)

        if complexity >= self.complexity_threshold and self.cloud.is_available():
            return self.cloud.generate(prompt, system, max_tokens, temperature)
        elif self.local.is_available():
            return self.local.generate(prompt, system, max_tokens, temperature)
        elif self.cloud.is_available():
            return self.cloud.generate(prompt, system, max_tokens, temperature)
        else:
            return "[No LLM backend available]"


class JEPASoulBridge:
    def __init__(self, memory_dir: Optional[str] = None, llm_backend: Optional[LLMBackend] = None):
        self.memory = SoulMemoryEngine(memory_dir)
        self.slot_decoder = SlotToTextDecoder()
        self.llm = llm_backend or CloudBackend()
        self._jepa_models: Dict[JEPAModality, Any] = {}
        self._projectors: Dict[JEPAModality, Any] = {}
        self._dual_pathway_bridge = None
        self._world_cache = None
        self._causal_extractor = None

    def register_jepa(self, modality: JEPAModality, jepa_model, projector=None):
        self._jepa_models[modality] = jepa_model
        if projector is not None:
            self._projectors[modality] = projector

    def encode(self, raw_input: Any, modality: JEPAModality) -> torch.Tensor:
        if modality not in self._jepa_models:
            raise ValueError(f"No JEPA model registered for {modality.value}")

        jepa = self._jepa_models[modality]
        projector = self._projectors.get(modality)

        if projector is not None and isinstance(raw_input, torch.Tensor):
            with torch.no_grad():
                if raw_input.dim() == 3:
                    B, T, F = raw_input.shape
                    flat = raw_input.reshape(B * T, F) if raw_input.dim() == 3 else raw_input
                    features = projector(flat)
                    features = features.reshape(B, T, -1)
                else:
                    features = projector(raw_input)
                raw_input = features

        with torch.no_grad():
            slots, _ = jepa.context_encoder(raw_input)
        return slots

    def process(
        self,
        raw_input: Any,
        modality: JEPAModality,
        soul: Optional[str] = None,
        query: str = "",
        top_k: int = 5,
        use_llm: bool = True,
    ) -> BridgeResult:
        if soul is None:
            souls = MODALITY_SOUL_MAP.get(modality, ["cezanne"])
            soul = souls[0]

        slots = self.encode(raw_input, modality)
        descriptions = self.slot_decoder.decode(slots, modality)
        query_text = self.slot_decoder.to_query_text(descriptions, modality)

        search_query = f"{query} {query_text}" if query else query_text
        knowledge = self.memory.recall(soul, search_query, top_k=top_k)

        llm_response = None
        if use_llm and self.llm.is_available():
            prompt = self._assemble_prompt(modality, soul, descriptions, knowledge, query)
            system = self._system_prompt(soul, modality)
            llm_response = self.llm.generate(prompt, system)

        return BridgeResult(
            modality=modality,
            slot_descriptions=descriptions,
            knowledge_context=knowledge,
            llm_response=llm_response,
            metadata={"soul": soul, "query": query, "n_slots": len(descriptions)},
        )

    def _system_prompt(self, soul: str, modality: JEPAModality) -> str:
        return (f"You are the {soul} soul in the VORTEX_FLAME system, "
                f"specializing in {modality.value} analysis. "
                f"Use the provided JEPA slot analysis and knowledge base context "
                f"to give accurate, domain-specific responses.")

    def _assemble_prompt(
        self,
        modality: JEPAModality,
        soul: str,
        descriptions: List[SlotDescription],
        knowledge: List[Dict],
        query: str,
    ) -> str:
        parts = [f"[{modality.value.upper()} JEPA Analysis]"]
        for d in descriptions[:5]:
            parts.append(f"  - {d.summary} (activation: {d.activation:.2f})")

        if knowledge:
            parts.append(f"\n[Knowledge Base ({len(knowledge)} entries)]")
            for i, entry in enumerate(knowledge[:3]):
                content = entry.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        pass
                topic = content.get("topic", "") if isinstance(content, dict) else str(content)[:100]
                parts.append(f"  {i+1}. {topic}")

        if query:
            parts.append(f"\n[User Query] {query}")

        parts.append("\nPlease provide your analysis based on the JEPA perception and knowledge context above.")
        return "\n".join(parts)

    def counterfactual(
        self,
        raw_input: Any,
        modality: JEPAModality,
        intervene_slot_id: int,
        intervene_embedding: torch.Tensor,
        soul: Optional[str] = None,
        query: str = "What would change?",
    ) -> BridgeResult:
        if soul is None:
            souls = MODALITY_SOUL_MAP.get(modality, ["cezanne"])
            soul = souls[0]

        if modality not in self._jepa_models:
            raise ValueError(f"No JEPA model for {modality.value}")

        jepa = self._jepa_models[modality]
        projector = self._projectors.get(modality)

        if projector is not None and isinstance(raw_input, torch.Tensor):
            with torch.no_grad():
                features = projector(raw_input)
                raw_input = features

        with torch.no_grad():
            future_slots, attn = jepa.counterfactual_predict(
                raw_input, intervene_slot_id, intervene_embedding
            )

        descriptions = self.slot_decoder.decode(future_slots, modality)
        query_text = self.slot_decoder.to_query_text(descriptions, modality)
        search_query = f"counterfactual {query} {query_text}"
        knowledge = self.memory.recall(soul, search_query, top_k=3)

        llm_response = None
        if self.llm.is_available():
            slot_names = MODALITY_SLOT_NAMES.get(modality, [])
            intervened_name = slot_names[intervene_slot_id] if intervene_slot_id < len(slot_names) else f"slot_{intervene_slot_id}"
            prompt = (f"[Counterfactual Analysis - {modality.value}]\n"
                      f"Intervened on: {intervened_name} (slot {intervene_slot_id})\n"
                      f"Predicted outcome:\n")
            for d in descriptions[:5]:
                prompt += f"  - {d.summary}\n"
            prompt += f"\n[Query] {query}"
            system = self._system_prompt(soul, modality)
            llm_response = self.llm.generate(prompt, system)

        return BridgeResult(
            modality=modality,
            slot_descriptions=descriptions,
            knowledge_context=knowledge,
            llm_response=llm_response,
            metadata={"soul": soul, "counterfactual": True,
                        "intervened_slot": intervene_slot_id},
        )

    def _get_dual_pathway_bridge(self):
        if self._dual_pathway_bridge is None:
            from dual_pathway_bridge import DualPathwayBridge
            self._dual_pathway_bridge = DualPathwayBridge(
                d_text=384,
                d_world=128,
                d_fused=256,
                d_output=512,
            )
        return self._dual_pathway_bridge

    def _get_world_cache(self):
        if self._world_cache is None:
            from world_embedding_cache import WorldEmbeddingCache
            self._world_cache = WorldEmbeddingCache()
        return self._world_cache

    def _get_causal_extractor(self):
        if self._causal_extractor is None:
            from causal_knowledge_extractor import CausalKnowledgeExtractor
            self._causal_extractor = CausalKnowledgeExtractor(use_embedding=True)
        return self._causal_extractor

    def dual_pathway_query(
        self,
        query: str,
        soul: str = "cezanne",
        modality: Optional[JEPAModality] = None,
        raw_input: Any = None,
        top_k_text: int = 3,
        top_k_world: int = 5,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Dual-pathway query: RAG text + C-JEPA world → Cross-Attention fusion → LLM.

        Path A: soul_memory RAG → text embeddings (facts)
        Path B: World-Embedding cache → causal embeddings (logic)
        Fusion: Cross-Attention(text, world) → unified representation
        Output: LLM generates response from fused knowledge
        """
        text_results = self.memory.recall(soul, query, top_k=top_k_text)

        world_cache = self._get_world_cache()
        world_results = world_cache.search(soul, query, top_k=top_k_world)

        text_embeds = self._prepare_text_embeds_from_results(text_results)
        world_embeds = self._prepare_world_embeds_from_results(world_results, soul)

        fusion_info = None
        if text_embeds is not None and world_embeds is not None:
            bridge = self._get_dual_pathway_bridge()
            fusion_result = bridge(text_embeds, world_embeds, return_pathway_weights=True)
            fusion_info = {
                "fused_shape": list(fusion_result["fused"].shape),
                "pathway_weights": fusion_result.get("pathway_weights"),
            }
        elif text_embeds is not None:
            fusion_info = {"mode": "text_only", "shape": list(text_embeds.shape)}
        elif world_embeds is not None:
            fusion_info = {"mode": "world_only", "shape": list(world_embeds.shape)}

        slot_descriptions = []
        if modality and raw_input is not None and modality in self._jepa_models:
            slots = self.encode(raw_input, modality)
            slot_descriptions = self.slot_decoder.decode(slots, modality)

        llm_response = None
        if use_llm and self.llm.is_available():
            prompt = self._assemble_dual_pathway_prompt(
                query, soul, text_results, world_results, slot_descriptions, modality
            )
            system = self._system_prompt(soul, modality or JEPAModality.CODE)
            llm_response = self.llm.generate(prompt, system)

        return {
            "query": query,
            "soul": soul,
            "modality": modality.value if modality else None,
            "text_context_count": len(text_results),
            "world_context_count": len(world_results),
            "fusion": fusion_info,
            "slot_descriptions": [d.summary for d in slot_descriptions[:5]] if slot_descriptions else [],
            "llm_response": llm_response,
        }

    def index_knowledge_causal(
        self,
        dirpath: str,
        soul: str = "cezanne",
        category: str = "knowledge",
        max_files: int = 500,
    ) -> Dict[str, Any]:
        """
        Index a directory into both soul_memory (text RAG) and World-Embedding cache (causal).

        This is the unified indexing entry point for the dual-pathway architecture.
        """
        from causal_knowledge_extractor import CausalKnowledgeIndexer

        indexer = CausalKnowledgeIndexer()
        causal_stats = indexer.index_directory(dirpath, soul=soul, category=category, max_files=max_files)

        return {
            "causal_indexing": causal_stats,
            "soul": soul,
            "dirpath": dirpath,
        }

    def _prepare_text_embeds_from_results(
        self, results: List[Dict]
    ) -> Optional[torch.Tensor]:
        if not results:
            return None
        try:
            from soul_memory import EmbeddingProvider
            provider = EmbeddingProvider.get()
            if not provider.available:
                return None
            embeddings = []
            for entry in results:
                content = entry.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = {"text": content}
                text = content.get("text", content.get("topic", ""))
                if text:
                    emb = provider.encode(text)
                    if emb:
                        embeddings.append(np.frombuffer(emb, dtype=np.float32))
            if embeddings:
                return torch.tensor(np.stack(embeddings), dtype=torch.float32).unsqueeze(0)
        except Exception:
            pass
        return None

    def _prepare_world_embeds_from_results(
        self, results: list, soul: str
    ) -> Optional[torch.Tensor]:
        if not results:
            return None
        try:
            cache = self._get_world_cache()
            embeddings = []
            for r in results:
                entry_id = r.entry_id if hasattr(r, "entry_id") else r.get("entry_id", "")
                emb = cache.get_embedding(soul, entry_id)
                if emb is not None:
                    embeddings.append(emb)
            if embeddings:
                return torch.tensor(np.stack(embeddings), dtype=torch.float32).unsqueeze(0)
        except Exception:
            pass
        return None

    def _assemble_dual_pathway_prompt(
        self,
        query: str,
        soul: str,
        text_results: List[Dict],
        world_results: list,
        slot_descriptions: List[SlotDescription],
        modality: Optional[JEPAModality] = None,
    ) -> str:
        parts = []

        if modality:
            parts.append(f"[{modality.value.upper()} JEPA Perception]")
            for d in slot_descriptions[:5]:
                parts.append(f"  - {d.summary} (activation: {d.activation:.2f})")
            parts.append("")

        if text_results:
            parts.append("[Text Knowledge (RAG Path A)]")
            for i, entry in enumerate(text_results[:3]):
                content = entry.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = {"text": content}
                topic = content.get("topic", content.get("text", ""))[:200]
                parts.append(f"  {i+1}. {topic}")
            parts.append("")

        if world_results:
            parts.append("[Causal World Knowledge (C-JEPA Path B)]")
            for i, r in enumerate(world_results[:3]):
                if hasattr(r, "objects"):
                    parts.append(f"  {i+1}. Objects: {r.objects}")
                    if hasattr(r, "causal_summary") and r.causal_summary:
                        parts.append(f"     Causal: {r.causal_summary[:100]}")
                elif isinstance(r, dict):
                    parts.append(f"  {i+1}. {r.get('objects', [])}")
            parts.append("")

        parts.append(f"[User Query] {query}")
        parts.append("\nProvide analysis based on both factual knowledge (Path A) and causal logic (Path B).")

        return "\n".join(parts)
