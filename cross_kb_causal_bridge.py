"""
Cross-Knowledge-Base Causal Bridge
===================================
Enables causal reasoning across multiple industry knowledge bases.
When a query spans multiple domains (e.g., "agricultural supply chain + weather"),
this bridge coordinates C-JEPA variants across knowledge bases to produce
unified causal predictions.

Architecture:
  Query → Multi-KB Route → C-JEPA Per-KB Prediction → Cross-KB Causal Merge → Response

This addresses the gap identified from SmartHealth integration:
  SmartHealth's knowledge_feedback.py writes to VORTEX_FLAME KBs, but
  there was no mechanism for VORTEX_FLAME to reason ACROSS KBs causally.

No GPU required — this is CPU-based causal graph traversal and merging.
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VF_MEMORY_DIR = os.environ.get("VF_MEMORY_DIR", r"D:\VORTEX_FLAME\.vf_memory")

CJEPVA_VARIANT_MAP = {
    "cezanne": "CCODEJEPA",
    "einstein": "CPHYSJEPA",
    "galileo": "CPHYSJEPA",
    "darwin": "CBIOJEPA",
    "strategy": "CFINJEPA",
    "montesquieu": "CLAWJEPA",
    "davinci": "CVJEPA",      # also CDESIGNJEPA secondary
    "humboldt": "CGEOJEPA",
    "yuanlongping": "CBIOJEPA",
    "guizhu": "CLAWJEPA",
    "herodotus": "CVJEPA",      # also CGEOJEPA secondary
    "monet": "CARTJEPA",
    "vangogh": "CARTJEPA",
    "beethoven": "CAJEPA",
}

SHARED_JEPA_GROUPS = {
    "CPHYSJEPA": ["einstein", "galileo"],
    "CBIOJEPA": ["darwin", "yuanlongping"],
    "CGEOJEPA": ["humboldt", "herodotus"],
    "CARTJEPA": ["monet", "vangogh"],
    "CLAWJEPA": ["montesquieu", "guizhu"],
}


@dataclass
class CausalNode:
    node_id: str
    knowledge_base: str
    jepa_variant: str
    concept: str
    evidence_count: int = 0
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class CausalEdge:
    source_id: str
    target_id: str
    relation: str
    confidence: float = 0.0
    cross_kb: bool = False
    evidence: List[str] = field(default_factory=list)


@dataclass
class CrossKBPrediction:
    prediction_id: str
    query: str
    involved_kbs: List[str]
    involved_jepa_variants: List[str]
    causal_chain: List[Dict]
    confidence: float = 0.0
    novelty: float = 0.0
    timestamp: str = ""


class CrossKBCausalBridge:
    """
    Bridges causal reasoning across multiple industry knowledge bases.

    When a query touches multiple domains, this bridge:
    1. Identifies which KBs are relevant
    2. Extracts causal nodes from each KB
    3. Finds cross-KB causal links
    4. Merges predictions from multiple C-JEPA variants
    5. Produces a unified causal prediction
    """

    def __init__(self, memory_dir: Optional[str] = None):
        self.memory_dir = memory_dir or VF_MEMORY_DIR
        self._causal_graph_nodes: Dict[str, CausalNode] = {}
        self._causal_graph_edges: List[CausalEdge] = []
        self._prediction_history: List[CrossKBPrediction] = []

    def identify_relevant_kbs(self, query: str) -> List[Tuple[str, float]]:
        kb_scores = defaultdict(float)
        query_lower = query.lower()
        query_words = set(query_lower.split())

        domain_keywords = {
            "cezanne": ["code", "algorithm", "system", "debug", "deploy", "software", "programming"],
            "einstein": ["physics", "quantum", "math", "simulation", "energy", "force", "calculation"],
            "galileo": ["astronomy", "orbit", "telescope", "planet", "space", "observation"],
            "darwin": ["biology", "evolution", "genetics", "species", "mutation", "adaptation"],
            "strategy": ["game", "market", "competition", "strategy", "nash", "equilibrium", "finance"],
            "montesquieu": ["law", "compliance", "governance", "regulation", "policy", "security"],
            "davinci": ["design", "engineering", "architecture", "prototype", "interface", "visual"],
            "humboldt": ["earth", "climate", "geography", "ecology", "environment", "spatial"],
            "yuanlongping": ["agriculture", "crop", "food", "nutrition", "farming", "yield"],
            "guizhu": ["philosophy", "logic", "dialogue", "reasoning", "ethics", "argument"],
            "herodotus": ["history", "event", "causality", "documentation", "chronicle"],
            "monet": ["art", "aesthetic", "visual", "design", "color", "composition"],
            "vangogh": ["emotion", "creative", "art", "color", "expression", "therapy"],
            "beethoven": ["music", "acoustic", "audio", "sound", "rhythm", "composition"],
        }

        for kb, keywords in domain_keywords.items():
            for kw in keywords:
                if kw in query_lower:
                    kb_scores[kb] += 2.0
            overlap = len(query_words & set(keywords))
            if overlap > 0:
                kb_scores[kb] += overlap * 0.5

        scored = sorted(kb_scores.items(), key=lambda x: x[1], reverse=True)
        return [(kb, score) for kb, score in scored if score > 0]

    def extract_causal_nodes(self, knowledge_base: str, query: str, top_k: int = 5) -> List[CausalNode]:
        db_path = os.path.join(self.memory_dir, f"{knowledge_base}.db")
        if not os.path.exists(db_path):
            return []

        nodes = []
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT entry_id, category, content, importance, tags
                FROM memories
                WHERE category IN ('knowledge', 'domain_memory', 'feedback')
                ORDER BY importance DESC
                LIMIT ?
            """, (top_k * 3,))
            rows = cur.fetchall()
            conn.close()

            for row in rows:
                entry_id, category, content_str, importance, tags_str = row
                try:
                    content = json.loads(content_str) if isinstance(content_str, str) else {}
                except (json.JSONDecodeError, TypeError):
                    content = {"raw": str(content_str)[:200]}

                topic = content.get("topic", "")
                detail = content.get("detail", "")
                query_lower = query.lower()
                relevance = 0.0
                for word in query_lower.split():
                    if word in topic.lower() or word in detail.lower():
                        relevance += 1.0

                if relevance > 0 or importance > 0.7:
                    node = CausalNode(
                        node_id=entry_id,
                        knowledge_base=knowledge_base,
                        jepa_variant=CJEPVA_VARIANT_MAP.get(knowledge_base, "CAJEPA"),
                        concept=topic[:100],
                        evidence_count=1,
                        confidence=min(1.0, (relevance + importance) / 2),
                        metadata={"category": category, "detail_preview": detail[:100]},
                    )
                    nodes.append(node)
                    self._causal_graph_nodes[node.node_id] = node

        except Exception as e:
            logger.error(f"Failed to extract nodes from {knowledge_base}: {e}")

        nodes.sort(key=lambda n: n.confidence, reverse=True)
        return nodes[:top_k]

    def find_cross_kb_links(self, nodes: List[CausalNode]) -> List[CausalEdge]:
        edges = []
        kb_groups = defaultdict(list)
        for node in nodes:
            kb_groups[node.knowledge_base].append(node)

        for jepa_variant, kb_list in SHARED_JEPA_GROUPS.items():
            variant_nodes = [n for n in nodes if n.jepa_variant == jepa_variant]
            if len(variant_nodes) < 2:
                continue
            for i in range(len(variant_nodes)):
                for j in range(i + 1, len(variant_nodes)):
                    n1, n2 = variant_nodes[i], variant_nodes[j]
                    if n1.knowledge_base != n2.knowledge_base:
                        edge = CausalEdge(
                            source_id=n1.node_id,
                            target_id=n2.node_id,
                            relation=f"shared_{jepa_variant}_causal",
                            confidence=min(n1.confidence, n2.confidence) * 0.8,
                            cross_kb=True,
                            evidence=[f"{n1.knowledge_base}:{n1.concept}", f"{n2.knowledge_base}:{n2.concept}"],
                        )
                        edges.append(edge)

        return edges

    def predict_cross_kb(self, query: str, top_kbs: int = 3) -> CrossKBPrediction:
        relevant_kbs = self.identify_relevant_kbs(query)[:top_kbs]
        if not relevant_kbs:
            return CrossKBPrediction(
                prediction_id="empty",
                query=query,
                involved_kbs=[],
                involved_jepa_variants=[],
                causal_chain=[],
                confidence=0.0,
                novelty=0.0,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

        all_nodes = []
        for kb, score in relevant_kbs:
            nodes = self.extract_causal_nodes(kb, query, top_k=5)
            for node in nodes:
                node.confidence = min(1.0, node.confidence * (score / max(score, 1.0)))
            all_nodes.extend(nodes)

        cross_edges = self.find_cross_kb_links(all_nodes)

        involved_kbs = list(set(n.knowledge_base for n in all_nodes))
        involved_variants = list(set(n.jepa_variant for n in all_nodes))

        causal_chain = []
        for edge in cross_edges:
            src = self._causal_graph_nodes.get(edge.source_id)
            tgt = self._causal_graph_nodes.get(edge.target_id)
            if src and tgt:
                causal_chain.append({
                    "from": {"kb": src.knowledge_base, "concept": src.concept, "variant": src.jepa_variant},
                    "to": {"kb": tgt.knowledge_base, "concept": tgt.concept, "variant": tgt.jepa_variant},
                    "relation": edge.relation,
                    "confidence": edge.confidence,
                    "cross_kb": edge.cross_kb,
                })

        for node in all_nodes:
            causal_chain.append({
                "node": {"kb": node.knowledge_base, "concept": node.concept, "variant": node.jepa_variant},
                "confidence": node.confidence,
            })

        avg_confidence = (
            sum(n.confidence for n in all_nodes) / len(all_nodes) if all_nodes else 0.0
        )
        novelty = min(1.0, len(cross_edges) * 0.2 + len(involved_kbs) * 0.1)

        pred_id = hashlib.sha256(
            f"{query}_{involved_kbs}_{time.time()}".encode()
        ).hexdigest()[:16]

        prediction = CrossKBPrediction(
            prediction_id=pred_id,
            query=query,
            involved_kbs=involved_kbs,
            involved_jepa_variants=involved_variants,
            causal_chain=causal_chain,
            confidence=avg_confidence,
            novelty=novelty,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        self._prediction_history.append(prediction)
        return prediction

    def get_prediction_history(self, limit: int = 20) -> List[Dict]:
        recent = self._prediction_history[-limit:]
        return [
            {
                "prediction_id": p.prediction_id,
                "query": p.query[:80],
                "involved_kbs": p.involved_kbs,
                "involved_variants": p.involved_jepa_variants,
                "chain_length": len(p.causal_chain),
                "confidence": p.confidence,
                "novelty": p.novelty,
                "timestamp": p.timestamp,
            }
            for p in recent
        ]
