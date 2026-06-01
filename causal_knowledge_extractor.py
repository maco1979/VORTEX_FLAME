"""
Causal Knowledge Extractor — Knowledge → Structured Causal Objects Pipeline
============================================================================
Bridges the gap between raw text knowledge and C-JEPA's object-centric input.

The dual-pathway architecture requires:
  Path A (RAG): text → chunks → text embedding → vector DB → Transformer
  Path B (C-JEPA): text → entities/relations/causal/temporal → object graph → C-JEPA

This module implements Path B's preprocessing:
  1. Extract entities (objects) from text
  2. Extract relations between entities
  3. Extract causal chains (cause → effect)
  4. Extract temporal sequences (event ordering)
  5. Build structured ObjectGraph for C-JEPA consumption
  6. Generate C-JEPA training samples with object-level masking

No GPU required. Uses rule-based NLP + sentence-transformers for semantic matching.
"""

import json
import logging
import os
import re
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

logger = logging.getLogger(__name__)

CAUSAL_KEYWORDS_ZH = [
    "导致", "引起", "造成", "使得", "引发", "促使", "带来", "产生",
    "因为", "由于", "原因是", "归因于", "源于",
    "所以", "因此", "于是", "从而", "进而",
    "如果", "假如", "假设", "若", "一旦",
    "那么", "则", "就会", "便会",
    "防止", "避免", "阻止", "抑制",
    "触发", "激活", "启动", "驱动",
]

CAUSAL_KEYWORDS_EN = [
    "causes", "leads to", "results in", "triggers", "induces", "produces",
    "because", "due to", "as a result of", "caused by", "attributed to",
    "therefore", "consequently", "hence", "thus", "so",
    "if", "when", "whenever", "once", "provided that",
    "then", "will", "would", "could",
    "prevents", "avoids", "blocks", "inhibits", "suppresses",
    "enables", "activates", "initiates", "drives",
]

TEMPORAL_KEYWORDS_ZH = [
    "然后", "接着", "随后", "之后", "下一步", "最后", "最终",
    "首先", "其次", "再次", "接着", "最后",
    "之前", "之前", "先", "提前",
    "同时", "并行", "与此同时",
    "直到", "直到...才", "在...之前", "在...之后",
]

TEMPORAL_KEYWORDS_EN = [
    "then", "next", "after", "afterwards", "subsequently", "finally", "ultimately",
    "first", "secondly", "thirdly", "next", "lastly",
    "before", "prior to", "earlier", "previously",
    "meanwhile", "simultaneously", "at the same time", "concurrently",
    "until", "by the time", "once", "as soon as",
]

ENTITY_PATTERNS = [
    re.compile(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'),
    re.compile(r'[\u4e00-\u9fff]{2,8}'),
]

CAUSAL_PATTERNS = [
    re.compile(r'(.+?)(?:导致|引起|造成|使得|引发|促使|带来|产生)(.+?)[。；\n]'),
    re.compile(r'(.+?)(?:因为|由于)(.+?)[，,](?:所以|因此|于是)(.+?)[。；\n]'),
    re.compile(r'如果(.+?)[，,](?:那么|则|就)(.+?)[。；\n]'),
    re.compile(r'(.+?)(?:causes?|leads?\s+to|results?\s+in|triggers?)(.+?)[.;\n]'),
    re.compile(r'(?:if|when|whenever)\s+(.+?)[,.]?\s*(?:then|will|would|could)\s+(.+?)[.;\n]'),
    re.compile(r'(.+?)(?:because|due\s+to|as\s+a\s+result\s+of)\s+(.+?)[.;\n]'),
]

TEMPORAL_PATTERNS = [
    re.compile(r'(.+?)[，,]?(?:然后|接着|随后|之后|下一步)(.+?)[。；\n]'),
    re.compile(r'(?:首先|第一步)[，,]?(.+?)[；;]?(?:其次|第二步)[，,]?(.+?)[；;]?(?:最后|第三步)[，,]?(.+?)[。；\n]'),
    re.compile(r'(.+?)[,.]?\s*(?:then|next|afterwards?|subsequently)\s*,?\s*(.+?)[.;\n]'),
    re.compile(r'(?:first|initially)\s*,?\s*(.+?)[.;]?\s*(?:then|next)\s*,?\s*(.+?)[.;]?\s*(?:finally|lastly)\s*,?\s*(.+?)[.;\n]'),
]


@dataclass
class CausalEntity:
    entity_id: str
    name: str
    entity_type: str = "object"
    attributes: Dict[str, str] = field(default_factory=dict)
    source_text: str = ""
    confidence: float = 1.0


@dataclass
class CausalRelation:
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str = "interacts"
    confidence: float = 1.0
    source_text: str = ""


@dataclass
class CausalChain:
    chain_id: str
    cause: str
    effect: str
    mechanism: str = ""
    counterfactual: str = ""
    confidence: float = 1.0
    source_text: str = ""


@dataclass
class TemporalStep:
    step_id: str
    step_order: int
    description: str
    entity_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ObjectGraph:
    graph_id: str
    entities: List[CausalEntity] = field(default_factory=list)
    relations: List[CausalRelation] = field(default_factory=list)
    causal_chains: List[CausalChain] = field(default_factory=list)
    temporal_sequence: List[TemporalStep] = field(default_factory=list)
    source_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cjepa_sample(self) -> Dict[str, Any]:
        objects = [e.name for e in self.entities]
        attributes = {e.name: e.attributes for e in self.entities if e.attributes}
        temporal = [s.description for s in sorted(self.temporal_sequence, key=lambda s: s.step_order)]
        causal_graph = {}
        for chain in self.causal_chains:
            causal_graph = {
                "cause": chain.cause,
                "effect": chain.effect,
                "counterfactual": chain.counterfactual or f"若{chain.cause}不发生，则{chain.effect}不发生",
            }
            break
        return {
            "objects": objects,
            "attributes": attributes,
            "temporal_sequence": temporal,
            "causal_graph": causal_graph,
            "context_mask": [],
            "target_embedding": None,
            "graph_id": self.graph_id,
            "source_text": self.source_text[:500],
        }

    def to_dict(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "entities": [asdict(e) for e in self.entities],
            "relations": [asdict(r) for r in self.relations],
            "causal_chains": [asdict(c) for c in self.causal_chains],
            "temporal_sequence": [asdict(s) for s in self.temporal_sequence],
            "source_text": self.source_text[:500],
            "metadata": self.metadata,
        }


class CausalKnowledgeExtractor:
    """
    Extracts structured causal knowledge from text for C-JEPA consumption.

    Pipeline:
      Raw text → Sentence split → Entity extraction → Relation extraction
      → Causal chain extraction → Temporal ordering → ObjectGraph

    No GPU required. Uses pattern matching + keyword heuristics.
    For production, can be upgraded with LLM-based extraction.
    """

    def __init__(self, use_embedding: bool = True):
        self._entity_cache: Dict[str, CausalEntity] = {}
        self._relation_counter = 0
        self._chain_counter = 0
        self._step_counter = 0
        self._graph_counter = 0
        self._embedding_provider = None
        if use_embedding:
            try:
                from soul_memory import EmbeddingProvider
                self._embedding_provider = EmbeddingProvider.get()
            except Exception:
                logger.warning("Embedding provider unavailable, using pure rule-based extraction")

    def extract_from_text(self, text: str, source_id: str = "") -> ObjectGraph:
        sentences = self._split_sentences(text)
        entities = self._extract_entities(sentences, text)
        relations = self._extract_relations(sentences, entities, text)
        causal_chains = self._extract_causal_chains(sentences, text)
        temporal_seq = self._extract_temporal_sequence(sentences, text)

        self._graph_counter += 1
        graph_id = source_id or f"graph_{self._graph_counter}"

        return ObjectGraph(
            graph_id=graph_id,
            entities=entities,
            relations=relations,
            causal_chains=causal_chains,
            temporal_sequence=temporal_seq,
            source_text=text[:2000],
            metadata={
                "num_sentences": len(sentences),
                "num_entities": len(entities),
                "num_causal_chains": len(causal_chains),
                "num_temporal_steps": len(temporal_seq),
            },
        )

    def extract_from_file(self, filepath: str) -> ObjectGraph:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.extract_from_text(text, source_id=path.stem)

    def extract_from_directory(
        self,
        dirpath: str,
        extensions: Optional[List[str]] = None,
        max_files: int = 1000,
    ) -> List[ObjectGraph]:
        exts = set(extensions or [".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml"])
        graphs = []
        root = Path(dirpath)
        if not root.exists():
            return graphs

        for fp in sorted(root.rglob("*")):
            if fp.suffix.lower() not in exts:
                continue
            if len(graphs) >= max_files:
                break
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
                if len(text.strip()) < 20:
                    continue
                rel_path = str(fp.relative_to(root))
                graph = self.extract_from_text(text, source_id=rel_path)
                graphs.append(graph)
            except Exception as e:
                logger.warning(f"Failed to extract from {fp}: {e}")

        return graphs

    def extract_from_soul_memory(
        self,
        soul: str,
        category: str = "knowledge",
        query: str = "",
        top_k: int = 50,
    ) -> List[ObjectGraph]:
        try:
            from soul_memory import SoulMemoryEngine
            engine = SoulMemoryEngine()
            if query:
                entries = engine.search(soul, category, query, top_k=top_k)
            else:
                entries = engine.search(soul, category, "", top_k=top_k)
        except Exception as e:
            logger.warning(f"Failed to search soul_memory: {e}")
            return []

        graphs = []
        for entry in entries:
            content = entry.get("content", {})
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except Exception:
                    content = {"text": content}
            text = content.get("text", content.get("topic", ""))
            if not text or len(text.strip()) < 20:
                continue
            entry_id = entry.get("entry_id", f"mem_{len(graphs)}")
            graph = self.extract_from_text(text, source_id=entry_id)
            graphs.append(graph)

        return graphs

    def generate_cjepa_training_samples(
        self,
        graphs: List[ObjectGraph],
        mask_ratios: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        mask_ratios = mask_ratios or [0.3, 0.5, 0.7]
        samples = []

        for graph in graphs:
            base_sample = graph.to_cjepa_sample()
            objects = base_sample["objects"]
            if len(objects) < 2:
                continue

            for ratio in mask_ratios:
                num_mask = max(1, int(len(objects) * ratio))
                num_mask = min(num_mask, len(objects) - 1)

                import random
                masked_indices = random.sample(range(len(objects)), num_mask)
                masked_objects = [objects[i] for i in masked_indices]
                observable_objects = [objects[i] for i in range(len(objects)) if i not in masked_indices]

                sample = {
                    **base_sample,
                    "context_mask": masked_objects,
                    "observable_objects": observable_objects,
                    "mask_ratio": ratio,
                    "num_masked": num_mask,
                }
                samples.append(sample)

        return samples

    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r'\n{2,}', '\n', text)
        parts = re.split(r'[。！？；\n]+', text)
        parts_en = []
        for part in parts:
            sub = re.split(r'(?<=[.!?;])\s+', part)
            parts_en.extend(sub)
        sentences = [s.strip() for s in parts_en if len(s.strip()) > 5]
        return sentences

    def _extract_entities(self, sentences: List[str], full_text: str) -> List[CausalEntity]:
        entity_freq: Dict[str, int] = defaultdict(int)
        entity_contexts: Dict[str, List[str]] = defaultdict(list)

        for sent in sentences:
            found = set()
            for pattern in ENTITY_PATTERNS:
                for match in pattern.finditer(sent):
                    name = match.group().strip()
                    if len(name) >= 2 and not self._is_stopword(name):
                        found.add(name)
            for name in found:
                entity_freq[name] += 1
                entity_contexts[name].append(sent)

        entities = []
        for name, freq in sorted(entity_freq.items(), key=lambda x: -x[1]):
            if freq < 1:
                continue
            entity_id = f"ent_{hashlib.md5(name.encode()).hexdigest()[:8]}"
            entity_type = self._classify_entity_type(name, entity_contexts[name])
            attributes = self._extract_attributes(name, entity_contexts[name][:3])
            entities.append(CausalEntity(
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                attributes=attributes,
                source_text=entity_contexts[name][0][:200] if entity_contexts[name] else "",
                confidence=min(1.0, freq / 5.0),
            ))
            self._entity_cache[name] = entities[-1]

        return entities[:30]

    def _extract_relations(
        self,
        sentences: List[str],
        entities: List[CausalEntity],
        full_text: str,
    ) -> List[CausalRelation]:
        entity_names = {e.name for e in entities}
        relations = []

        for sent in sentences:
            found_in_sent = [name for name in entity_names if name in sent]
            if len(found_in_sent) < 2:
                continue

            for i in range(len(found_in_sent)):
                for j in range(i + 1, len(found_in_sent)):
                    rel_type = self._classify_relation(found_in_sent[i], found_in_sent[j], sent)
                    self._relation_counter += 1
                    relations.append(CausalRelation(
                        relation_id=f"rel_{self._relation_counter}",
                        source_id=self._entity_cache[found_in_sent[i]].entity_id if found_in_sent[i] in self._entity_cache else found_in_sent[i],
                        target_id=self._entity_cache[found_in_sent[j]].entity_id if found_in_sent[j] in self._entity_cache else found_in_sent[j],
                        relation_type=rel_type,
                        confidence=0.7,
                        source_text=sent[:200],
                    ))

        return relations[:50]

    def _extract_causal_chains(self, sentences: List[str], full_text: str) -> List[CausalChain]:
        chains = []

        for pattern in CAUSAL_PATTERNS:
            for match in pattern.finditer(full_text):
                groups = match.groups()
                if len(groups) >= 2:
                    cause = groups[0].strip()
                    effect = groups[-1].strip()
                    if len(cause) < 3 or len(effect) < 3:
                        continue
                    self._chain_counter += 1
                    counterfactual = self._generate_counterfactual(cause, effect)
                    chains.append(CausalChain(
                        chain_id=f"chain_{self._chain_counter}",
                        cause=cause,
                        effect=effect,
                        mechanism=groups[1].strip() if len(groups) > 2 else "",
                        counterfactual=counterfactual,
                        confidence=0.8,
                        source_text=match.group()[:200],
                    ))

        for sent in sentences:
            has_cause = any(kw in sent for kw in CAUSAL_KEYWORDS_ZH + CAUSAL_KEYWORDS_EN)
            if has_cause and not any(c.source_text in sent for c in chains):
                parts = re.split(r'(?:导致|引起|造成|使得|因此|所以|causes?|leads?\s+to)', sent)
                if len(parts) >= 2:
                    self._chain_counter += 1
                    chains.append(CausalChain(
                        chain_id=f"chain_{self._chain_counter}",
                        cause=parts[0].strip(),
                        effect=parts[-1].strip(),
                        counterfactual=self._generate_counterfactual(parts[0].strip(), parts[-1].strip()),
                        confidence=0.6,
                        source_text=sent[:200],
                    ))

        return chains[:30]

    def _extract_temporal_sequence(self, sentences: List[str], full_text: str) -> List[TemporalStep]:
        steps = []

        for pattern in TEMPORAL_PATTERNS:
            for match in pattern.finditer(full_text):
                groups = match.groups()
                for order, desc in enumerate(groups):
                    desc = desc.strip()
                    if len(desc) < 3:
                        continue
                    self._step_counter += 1
                    entity_ids = []
                    for name, ent in self._entity_cache.items():
                        if name in desc:
                            entity_ids.append(ent.entity_id)
                    steps.append(TemporalStep(
                        step_id=f"step_{self._step_counter}",
                        step_order=order,
                        description=desc,
                        entity_ids=entity_ids,
                        confidence=0.7,
                    ))

        if not steps:
            for i, sent in enumerate(sentences[:10]):
                has_temporal = any(kw in sent for kw in TEMPORAL_KEYWORDS_ZH + TEMPORAL_KEYWORDS_EN)
                if has_temporal:
                    self._step_counter += 1
                    entity_ids = []
                    for name, ent in self._entity_cache.items():
                        if name in sent:
                            entity_ids.append(ent.entity_id)
                    steps.append(TemporalStep(
                        step_id=f"step_{self._step_counter}",
                        step_order=i,
                        description=sent[:200],
                        entity_ids=entity_ids,
                        confidence=0.5,
                    ))

        return steps[:20]

    def _is_stopword(self, name: str) -> bool:
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can",
            "this", "that", "these", "those", "it", "its",
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这",
            "import", "from", "class", "def", "return", "if", "else",
            "for", "while", "with", "try", "except", "raise", "None",
        }
        return name.lower() in stopwords

    def _classify_entity_type(self, name: str, contexts: List[str]) -> str:
        context = " ".join(contexts[:3]).lower()
        if any(kw in context for kw in ["设备", "机器", "系统", "device", "system", "machine"]):
            return "device"
        if any(kw in context for kw in ["人", "操作员", "用户", "user", "operator", "person"]):
            return "agent"
        if any(kw in context for kw in ["信号", "数据", "信息", "signal", "data", "information"]):
            return "signal"
        if any(kw in context for kw in ["规则", "流程", "步骤", "rule", "process", "step"]):
            return "rule"
        if any(kw in context for kw in ["事件", "告警", "故障", "event", "alert", "fault"]):
            return "event"
        return "object"

    def _extract_attributes(self, name: str, contexts: List[str]) -> Dict[str, str]:
        attrs = {}
        for ctx in contexts:
            patterns = [
                re.compile(f'{re.escape(name)}(?:是|为|:|＝|=)\\s*([^，,。.；;\\n]+)'),
                re.compile(f'{re.escape(name)}\\s*(?:is|are|has|have)\\s+([^,.;\\n]+)'),
            ]
            for pat in patterns:
                m = pat.search(ctx)
                if m:
                    val = m.group(1).strip()
                    if len(val) < 50:
                        attrs["property"] = val
                        break
        return attrs

    def _classify_relation(self, source: str, target: str, context: str) -> str:
        ctx = context.lower()
        if any(kw in ctx for kw in ["导致", "引起", "造成", "cause", "lead to", "result in"]):
            return "causes"
        if any(kw in ctx for kw in ["依赖", "需要", "depend", "require", "need"]):
            return "depends_on"
        if any(kw in ctx for kw in ["包含", "组成", "contain", "consist", "compose"]):
            return "contains"
        if any(kw in ctx for kw in ["控制", "管理", "control", "manage", "govern"]):
            return "controls"
        if any(kw in ctx for kw in ["交互", "通信", "interact", "communicate"]):
            return "interacts"
        return "related"

    def _generate_counterfactual(self, cause: str, effect: str) -> str:
        if any('\u4e00' <= c <= '\u9fff' for c in cause):
            return f"若{cause}不发生，则{effect}可能不会发生"
        return f"If {cause} did not happen, {effect} might not occur"


class CausalKnowledgeIndexer:
    """
    Batch-indexes knowledge sources into structured ObjectGraphs
    and stores them in soul_memory for C-JEPA consumption.

    Usage:
      indexer = CausalKnowledgeIndexer()
      stats = indexer.index_directory("D:/VORTEX_FLAME/kb_harness", soul="cezanne")
      stats = indexer.index_soul_memory(soul="cezanne", category="knowledge")
    """

    def __init__(self, memory_dir: Optional[str] = None):
        self.extractor = CausalKnowledgeExtractor(use_embedding=True)
        try:
            from soul_memory import SoulMemoryEngine
            self.memory = SoulMemoryEngine(memory_dir)
        except Exception:
            self.memory = None
            logger.warning("soul_memory unavailable, indexing will be extraction-only")

    def index_directory(
        self,
        dirpath: str,
        soul: str = "cezanne",
        category: str = "knowledge",
        extensions: Optional[List[str]] = None,
        max_files: int = 500,
    ) -> Dict[str, Any]:
        graphs = self.extractor.extract_from_directory(dirpath, extensions, max_files)
        return self._store_graphs(graphs, soul, category)

    def index_soul_memory(
        self,
        soul: str,
        category: str = "knowledge",
        query: str = "",
        top_k: int = 100,
    ) -> Dict[str, Any]:
        graphs = self.extractor.extract_from_soul_memory(soul, category, query, top_k)
        return self._store_graphs(graphs, soul, "domain_memory")

    def index_text(self, text: str, soul: str = "cezanne", source_id: str = "") -> Dict[str, Any]:
        graph = self.extractor.extract_from_text(text, source_id)
        return self._store_graphs([graph], soul, "knowledge")

    def _store_graphs(self, graphs: List[ObjectGraph], soul: str, category: str) -> Dict[str, Any]:
        stats = {
            "total_graphs": len(graphs),
            "total_entities": 0,
            "total_relations": 0,
            "total_causal_chains": 0,
            "total_temporal_steps": 0,
            "stored": 0,
            "errors": 0,
        }

        for graph in graphs:
            try:
                stats["total_entities"] += len(graph.entities)
                stats["total_relations"] += len(graph.relations)
                stats["total_causal_chains"] += len(graph.causal_chains)
                stats["total_temporal_steps"] += len(graph.temporal_sequence)

                if self.memory is None:
                    continue

                content = {
                    "topic": f"causal_graph:{graph.graph_id}",
                    "text": graph.source_text[:1000],
                    "object_graph": graph.to_dict(),
                    "cjepa_sample": graph.to_cjepa_sample(),
                }

                tags = ["causal_graph", "cjepa_input"]
                for chain in graph.causal_chains:
                    tags.append(f"causal:{chain.cause[:30]}")
                for entity in graph.entities[:5]:
                    tags.append(f"entity:{entity.name[:30]}")

                self.memory.write(
                    soul=soul,
                    category=category,
                    content=content,
                    importance=0.7 + 0.1 * min(len(graph.causal_chains), 3),
                    tags=tags,
                )
                stats["stored"] += 1
            except Exception as e:
                logger.warning(f"Failed to store graph {graph.graph_id}: {e}")
                stats["errors"] += 1

        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    extractor = CausalKnowledgeExtractor()

    test_text = """
    传感器采集数据后，如果数据超限，则触发告警信号。
    告警信号导致设备进入异常状态，操作员需要确认告警。
    操作员确认后，系统执行停机流程。首先切断电源，然后释放压力，最后记录故障日志。
    如果操作员未在5分钟内确认，系统自动执行紧急停机。
    陶瓷杯从1米高度掉落会导致破碎，因为冲击力超过材料强度。
    """

    graph = extractor.extract_from_text(test_text, source_id="test_demo")
    print(f"\n=== Extracted ObjectGraph ===")
    print(f"Entities: {[e.name for e in graph.entities]}")
    print(f"Relations: {[(r.source_id, r.relation_type, r.target_id) for r in graph.relations]}")
    print(f"Causal chains: {[(c.cause[:30], '→', c.effect[:30]) for c in graph.causal_chains]}")
    print(f"Temporal steps: {[s.description[:40] for s in graph.temporal_sequence]}")

    sample = graph.to_cjepa_sample()
    print(f"\n=== C-JEPA Training Sample ===")
    print(json.dumps(sample, ensure_ascii=False, indent=2))

    samples = extractor.generate_cjepa_training_samples([graph])
    print(f"\nGenerated {len(samples)} training samples with object-level masking")
    for i, s in enumerate(samples[:3]):
        print(f"  Sample {i}: mask_ratio={s['mask_ratio']}, masked={s['context_mask']}, observable={s['observable_objects']}")

    print("\n=== Indexing KB directories ===")
    indexer = CausalKnowledgeIndexer()

    kb_dirs = [
        ("D:/VORTEX_FLAME/kb_harness", "cezanne"),
        ("D:/VORTEX_FLAME/kb_mcp", "guizhu"),
        ("D:/VORTEX_FLAME/kb_skill", "beethoven"),
    ]

    for dirpath, soul in kb_dirs:
        if os.path.exists(dirpath):
            stats = indexer.index_directory(dirpath, soul=soul, max_files=200)
            print(f"\n  [{dirpath}] → soul={soul}")
            print(f"  Graphs: {stats['total_graphs']}, Entities: {stats['total_entities']}, "
                  f"Causal: {stats['total_causal_chains']}, Stored: {stats['stored']}")
        else:
            print(f"\n  [{dirpath}] not found, skipping")
