"""
Soul Self-Trainer — Post-Inference Knowledge Extraction & Auto-Deposition
=========================================================================
Enables all 14 souls to autonomously extract, validate, and deposit
new knowledge from their inference sessions into their persistent
memory stores.

Architecture:
  after_inference() hook → extract → validate → conflict-check → deposit

Domain Strategies:
  - Science (einstein, galileo, darwin): high autonomy, logic self-consistency
  - Engineering (cezanne, davinci): code verifiability, structural patterns
  - Strategy (strategy, montesquieu): semi-auto, law/fact immutability
  - Nature (humboldt, yuanlongping): data-source annotation required
  - Humanities (guizhu, herodotus): semi-auto, historical fact protection
  - Art (monet, vangogh, beethoven): style/preference only, no core knowledge mutation

Integration:
  - soul_orchestrator._execute_soul_stage() → SoulSelfTrainer.after_inference()
  - soul_memory.write() / detect_contradictions() as foundation
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    FULL_AUTO = "full_auto"
    SEMI_AUTO = "semi_auto"
    STYLE_ONLY = "style_only"


@dataclass
class DomainPolicy:
    mode: TrainingMode
    confidence_threshold: float
    allow_knowledge_write: bool
    allow_style_write: bool
    immutable_keywords: List[str]
    validation_prompt: str


DOMAIN_POLICIES: Dict[str, DomainPolicy] = {
    "einstein": DomainPolicy(
        mode=TrainingMode.FULL_AUTO,
        confidence_threshold=0.7,
        allow_knowledge_write=True,
        allow_style_write=True,
        immutable_keywords=[],
        validation_prompt="Verify logical consistency and mathematical correctness",
    ),
    "galileo": DomainPolicy(
        mode=TrainingMode.FULL_AUTO,
        confidence_threshold=0.7,
        allow_knowledge_write=True,
        allow_style_write=True,
        immutable_keywords=[],
        validation_prompt="Verify astronomical data and orbital mechanics",
    ),
    "darwin": DomainPolicy(
        mode=TrainingMode.FULL_AUTO,
        confidence_threshold=0.7,
        allow_knowledge_write=True,
        allow_style_write=True,
        immutable_keywords=[],
        validation_prompt="Verify biological taxonomy and evolutionary claims",
    ),
    "cezanne": DomainPolicy(
        mode=TrainingMode.FULL_AUTO,
        confidence_threshold=0.6,
        allow_knowledge_write=True,
        allow_style_write=True,
        immutable_keywords=[],
        validation_prompt="Verify code correctness and system architecture validity",
    ),
    "davinci": DomainPolicy(
        mode=TrainingMode.FULL_AUTO,
        confidence_threshold=0.6,
        allow_knowledge_write=True,
        allow_style_write=True,
        immutable_keywords=[],
        validation_prompt="Verify engineering feasibility and design consistency",
    ),
    "strategy": DomainPolicy(
        mode=TrainingMode.SEMI_AUTO,
        confidence_threshold=0.8,
        allow_knowledge_write=True,
        allow_style_write=True,
        immutable_keywords=["law", "regulation", "statute", "treaty"],
        validation_prompt="Verify game-theoretic reasoning and financial data",
    ),
    "montesquieu": DomainPolicy(
        mode=TrainingMode.SEMI_AUTO,
        confidence_threshold=0.8,
        allow_knowledge_write=True,
        allow_style_write=True,
        immutable_keywords=["constitution", "law", "amendment", "statute", "code"],
        validation_prompt="Verify legal citations and constitutional references",
    ),
    "humboldt": DomainPolicy(
        mode=TrainingMode.FULL_AUTO,
        confidence_threshold=0.7,
        allow_knowledge_write=True,
        allow_style_write=True,
        immutable_keywords=[],
        validation_prompt="Verify geographic and ecological data with source annotation",
    ),
    "yuanlongping": DomainPolicy(
        mode=TrainingMode.FULL_AUTO,
        confidence_threshold=0.7,
        allow_knowledge_write=True,
        allow_style_write=True,
        immutable_keywords=[],
        validation_prompt="Verify agricultural data and genetic information",
    ),
    "guizhu": DomainPolicy(
        mode=TrainingMode.SEMI_AUTO,
        confidence_threshold=0.8,
        allow_knowledge_write=True,
        allow_style_write=True,
        immutable_keywords=["historical fact", "date", "event"],
        validation_prompt="Verify philosophical reasoning, protect historical facts",
    ),
    "herodotus": DomainPolicy(
        mode=TrainingMode.SEMI_AUTO,
        confidence_threshold=0.8,
        allow_knowledge_write=True,
        allow_style_write=True,
        immutable_keywords=["date", "battle", "treaty", "dynasty", "reign"],
        validation_prompt="Verify historical chronology and factual accuracy",
    ),
    "monet": DomainPolicy(
        mode=TrainingMode.STYLE_ONLY,
        confidence_threshold=0.5,
        allow_knowledge_write=False,
        allow_style_write=True,
        immutable_keywords=[],
        validation_prompt="Record aesthetic preferences and creative style observations",
    ),
    "vangogh": DomainPolicy(
        mode=TrainingMode.STYLE_ONLY,
        confidence_threshold=0.5,
        allow_knowledge_write=False,
        allow_style_write=True,
        immutable_keywords=[],
        validation_prompt="Record emotional color associations and artistic techniques",
    ),
    "beethoven": DomainPolicy(
        mode=TrainingMode.STYLE_ONLY,
        confidence_threshold=0.5,
        allow_knowledge_write=False,
        allow_style_write=True,
        immutable_keywords=[],
        validation_prompt="Record musical preferences, acoustic observations, composition techniques",
    ),
}

KNOWLEDGE_INDICATORS = [
    "therefore", "thus", "consequently", "it follows", "we can conclude",
    "the result is", "this means", "in other words", "specifically",
    "the key insight", "importantly", "notably", "the solution",
    "the answer", "the correct", "the formula", "the principle",
    "the rule", "the pattern", "the relationship", "the structure",
]

STYLE_INDICATORS = [
    "I prefer", "I like", "my style", "aesthetically", "visually",
    "sounds better", "feels right", "in my experience", "I find",
    "tastes like", "reminds me of", "evokes", "inspires",
    "beautiful", "elegant", "harmonious", "resonant",
]

FACT_PATTERN = re.compile(
    r"(?:(?:the|a|an)\s+\w+\s+(?:is|are|was|were|equals|=|≈|≡)\s+[^.]+\.?"
    r"|(?:therefore|thus|consequently|so|hence)\s*,?\s*[^.]+\.?"
    r"|(?:this means|this implies|this shows|this indicates)\s+[^.]+\.?)",
    re.IGNORECASE,
)

DEFINITION_PATTERN = re.compile(
    r"(?:defined as|refers to|means|denotes|is known as|is called|is a|is an|are)\s+[^.]+\.?",
    re.IGNORECASE,
)

PROCEDURE_PATTERN = re.compile(
    r"(?:step \d|first|then|next|finally|to do this|how to)\s*[:\-]?\s*[^.]+\.?",
    re.IGNORECASE,
)


@dataclass
class ExtractedKnowledge:
    knowledge_type: str
    content: str
    confidence: float
    source_query: str
    source_response: str
    tags: List[str] = field(default_factory=list)


class KnowledgeExtractor:

    def extract(self, query: str, response: str, soul: str) -> List[ExtractedKnowledge]:
        policy = DOMAIN_POLICIES.get(soul)
        if policy is None:
            return []

        items: List[ExtractedKnowledge] = []

        items.extend(self._extract_facts(query, response, policy))
        items.extend(self._extract_definitions(query, response, policy))
        items.extend(self._extract_procedures(query, response, policy))
        items.extend(self._extract_style(query, response, policy))

        return items

    def _extract_facts(self, query: str, response: str, policy: DomainPolicy) -> List[ExtractedKnowledge]:
        if policy.mode == TrainingMode.STYLE_ONLY and not policy.allow_knowledge_write:
            return []

        results = []
        for match in FACT_PATTERN.finditer(response):
            fact = match.group(0).strip()
            if len(fact) < 15:
                continue
            confidence = self._compute_confidence(fact, response, policy)
            results.append(ExtractedKnowledge(
                knowledge_type="fact",
                content=fact,
                confidence=confidence,
                source_query=query,
                source_response=response[:200],
                tags=["auto_extracted", "fact"],
            ))
        return results

    def _extract_definitions(self, query: str, response: str, policy: DomainPolicy) -> List[ExtractedKnowledge]:
        if policy.mode == TrainingMode.STYLE_ONLY and not policy.allow_knowledge_write:
            return []

        results = []
        for match in DEFINITION_PATTERN.finditer(response):
            definition = match.group(0).strip()
            if len(definition) < 15:
                continue
            confidence = self._compute_confidence(definition, response, policy)
            results.append(ExtractedKnowledge(
                knowledge_type="definition",
                content=definition,
                confidence=confidence,
                source_query=query,
                source_response=response[:200],
                tags=["auto_extracted", "definition"],
            ))
        return results

    def _extract_procedures(self, query: str, response: str, policy: DomainPolicy) -> List[ExtractedKnowledge]:
        if policy.mode == TrainingMode.STYLE_ONLY and not policy.allow_knowledge_write:
            return []

        results = []
        for match in PROCEDURE_PATTERN.finditer(response):
            proc = match.group(0).strip()
            if len(proc) < 15:
                continue
            confidence = self._compute_confidence(proc, response, policy)
            results.append(ExtractedKnowledge(
                knowledge_type="procedure",
                content=proc,
                confidence=confidence,
                source_query=query,
                source_response=response[:200],
                tags=["auto_extracted", "procedure"],
            ))
        return results

    def _extract_style(self, query: str, response: str, policy: DomainPolicy) -> List[ExtractedKnowledge]:
        if not policy.allow_style_write:
            return []

        results = []
        response_lower = response.lower()
        for indicator in STYLE_INDICATORS:
            idx = response_lower.find(indicator.lower())
            if idx >= 0:
                start = max(0, idx - 20)
                end = min(len(response), idx + len(indicator) + 100)
                style_obs = response[start:end].strip()
                results.append(ExtractedKnowledge(
                    knowledge_type="style_preference",
                    content=style_obs,
                    confidence=0.5,
                    source_query=query,
                    source_response=response[:200],
                    tags=["auto_extracted", "style", "preference"],
                ))
        return results

    def _compute_confidence(self, extracted: str, full_response: str, policy: DomainPolicy) -> float:
        base = 0.55
        extracted_lower = extracted.lower()

        for indicator in KNOWLEDGE_INDICATORS:
            if indicator in extracted_lower:
                base += 0.1

        if any(kw in extracted_lower for kw in policy.immutable_keywords):
            base -= 0.3

        if len(extracted) > 50:
            base += 0.05

        has_number = bool(re.search(r'\d+\.?\d*', extracted))
        if has_number:
            base += 0.15

        causal_words = ["because", "since", "due to", "result of", "caused by", "leads to"]
        if any(w in extracted_lower for w in causal_words):
            base += 0.1

        if extracted_lower.startswith(("therefore", "thus", "consequently", "hence", "so ")):
            base += 0.1

        return min(max(base, 0.0), 1.0)


class QualityGate:

    def __init__(self, min_confidence: float = 0.5, max_per_session: int = 10):
        self.min_confidence = min_confidence
        self.max_per_session = max_per_session

    def filter(self, items: List[ExtractedKnowledge], policy: DomainPolicy) -> List[ExtractedKnowledge]:
        filtered = []
        for item in items:
            if item.confidence < policy.confidence_threshold:
                continue
            if item.confidence < self.min_confidence:
                continue
            content_lower = item.content.lower()
            is_immutable = any(kw in content_lower for kw in policy.immutable_keywords)
            if is_immutable and item.knowledge_type != "style_preference":
                continue
            filtered.append(item)

        return filtered[:self.max_per_session]


class SoulSelfTrainer:
    def __init__(self):
        self._extractor = KnowledgeExtractor()
        self._gate = QualityGate(min_confidence=0.5, max_per_session=10)
        self._session_stats: Dict[str, Dict] = {}

    def after_inference(self, soul: str, query: str, response: str,
                        stage: str = "", routing_confidence: float = 0.0) -> dict:
        policy = DOMAIN_POLICIES.get(soul)
        if policy is None:
            return {"status": "skipped", "reason": f"no policy for soul '{soul}'"}

        raw_items = self._extractor.extract(query, response, soul)
        if not raw_items:
            self._update_stats(soul, extracted=0, deposited=0, queued=0)
            return {"status": "no_knowledge", "extracted": 0}

        filtered = self._gate.filter(raw_items, policy)

        deposited = 0
        queued = 0
        errors = 0

        for item in filtered:
            try:
                result = self._deposit(soul, item, policy)
                if result == "deposited":
                    deposited += 1
                elif result == "queued":
                    queued += 1
            except Exception as e:
                errors += 1
                logger.warning(f"SoulSelfTrainer deposit error [{soul}]: {e}")

        self._update_stats(soul, extracted=len(raw_items), deposited=deposited, queued=queued)

        return {
            "status": "processed",
            "soul": soul,
            "policy_mode": policy.mode.value,
            "extracted_raw": len(raw_items),
            "filtered": len(filtered),
            "deposited": deposited,
            "queued_as_todo": queued,
            "errors": errors,
        }

    def _deposit(self, soul: str, item: ExtractedKnowledge, policy: DomainPolicy) -> str:
        from soul_memory import write, detect_contradictions

        if item.knowledge_type == "style_preference":
            content = {
                "topic": f"Style preference: {item.content[:60]}",
                "type": "style_preference",
                "observation": item.content,
                "source_query": item.source_query[:100],
                "confidence": item.confidence,
            }
            write(
                soul=soul,
                category="self_training",
                content=content,
                importance=0.4,
                tags=item.tags + ["style"],
            )
            return "deposited"

        if not policy.allow_knowledge_write:
            return "queued"

        content = {
            "topic": f"{item.knowledge_type.title()}: {item.content[:80]}",
            "type": item.knowledge_type,
            "content": item.content,
            "source_query": item.source_query[:100],
            "confidence": item.confidence,
            "validation_hint": policy.validation_prompt,
        }

        contradictions = detect_contradictions(soul, "knowledge", content)

        if contradictions and policy.mode == TrainingMode.SEMI_AUTO:
            todo_content = {
                "topic": f"Verify: {item.content[:60]}",
                "type": "validation_task",
                "knowledge_to_verify": item.content,
                "contradictions_found": len(contradictions),
                "validation_prompt": policy.validation_prompt,
                "source_query": item.source_query[:100],
            }
            write(
                soul=soul,
                category="todo",
                content=todo_content,
                importance=0.7,
                tags=["validation", "auto_queued", item.knowledge_type],
            )
            return "queued"

        if contradictions and policy.mode == TrainingMode.FULL_AUTO:
            write(
                soul=soul,
                category="knowledge",
                content=content,
                importance=item.confidence,
                tags=item.tags + ["auto_deposited", "has_contradiction"],
            )
            return "deposited"

        write(
            soul=soul,
            category="knowledge",
            content=content,
            importance=item.confidence,
            tags=item.tags + ["auto_deposited"],
        )
        return "deposited"

    def _update_stats(self, soul: str, extracted: int, deposited: int, queued: int):
        if soul not in self._session_stats:
            self._session_stats[soul] = {
                "total_sessions": 0, "total_extracted": 0,
                "total_deposited": 0, "total_queued": 0,
            }
        s = self._session_stats[soul]
        s["total_sessions"] += 1
        s["total_extracted"] += extracted
        s["total_deposited"] += deposited
        s["total_queued"] += queued

    def get_stats(self, soul: Optional[str] = None) -> dict:
        if soul:
            return self._session_stats.get(soul, {})
        return dict(self._session_stats)

    def get_policy(self, soul: str) -> Optional[DomainPolicy]:
        return DOMAIN_POLICIES.get(soul)

    def list_policies(self) -> Dict[str, Dict]:
        return {
            soul: {
                "mode": p.mode.value,
                "confidence_threshold": p.confidence_threshold,
                "allow_knowledge_write": p.allow_knowledge_write,
                "allow_style_write": p.allow_style_write,
                "immutable_count": len(p.immutable_keywords),
            }
            for soul, p in DOMAIN_POLICIES.items()
        }


_trainer = SoulSelfTrainer()


def after_inference(soul: str, query: str, response: str,
                    stage: str = "", routing_confidence: float = 0.0) -> dict:
    return _trainer.after_inference(soul, query, response, stage, routing_confidence)


def get_stats(soul: Optional[str] = None) -> dict:
    return _trainer.get_stats(soul)


def get_policy(soul: str) -> Optional[DomainPolicy]:
    return _trainer.get_policy(soul)


def list_policies() -> Dict[str, Dict]:
    return _trainer.list_policies()
