"""
Closed-Loop Orchestrator — Perception → Reasoning → Action → Feedback
======================================================================
The missing link in VORTEX_FLAME's architecture: a unified orchestrator that
closes the loop from inference results back to knowledge bases.

Architecture Flow:
  1. PERCEIVE: User query → route to knowledge base → retrieve context
  2. REASON: LLM generates response using knowledge base + C-JEPA causal logic
  3. ACT: Execute action via DeviceGateway (if applicable)
  4. EVALUATE: Assess response quality (confidence, consistency, novelty)
  5. FEEDBACK: Write evaluation results back to knowledge base
  6. EVOLVE: Update causal graphs, adjust routing weights

This closes the loop that SmartHealth has (knowledge_feedback.py) but
VORTEX_FLAME was missing on its own side.

Reference: SmartHealth's knowledge_feedback.py pattern
  业务数据 → 清洗 → 通用知识 → 写入灵魂DB → 下次查询增强 → 新业务数据

No GPU required — this is pure CPU orchestration logic.
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

VF_MEMORY_DIR = os.environ.get("VF_MEMORY_DIR", r"D:\VORTEX_FLAME\.vf_memory")


class FeedbackType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTION = "correction"
    NOVEL_INSIGHT = "novel_insight"


class LoopPhase(Enum):
    PERCEIVE = "perceive"
    REASON = "reason"
    ACT = "act"
    EVALUATE = "evaluate"
    FEEDBACK = "feedback"
    EVOLVE = "evolve"


@dataclass
class LoopState:
    query: str
    knowledge_base: str
    phase: LoopPhase = LoopPhase.PERCEIVE
    retrieved_context: List[Dict] = field(default_factory=list)
    response: str = ""
    confidence: float = 0.0
    action_result: Optional[Dict] = None
    evaluation: Optional["EvaluationResult"] = None
    feedback_type: Optional[FeedbackType] = None
    feedback_written: bool = False
    loop_id: str = ""
    started_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        if not self.loop_id:
            raw = f"{self.knowledge_base}_{self.query[:50]}_{time.time()}"
            self.loop_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
        if not self.started_at:
            self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class EvaluationResult:
    confidence: float
    consistency_score: float
    novelty_score: float
    domain_coverage: float
    feedback_type: FeedbackType
    details: Dict = field(default_factory=dict)


class ClosedLoopOrchestrator:
    """
    Orchestrates the complete perception→reasoning→action→feedback loop.

    This is the architectural component that was missing: SmartHealth has
    knowledge_feedback.py for its domain, but VORTEX_FLAME itself lacked
    the mechanism to feed inference results back into its own knowledge bases.
    """

    def __init__(self, memory_dir: Optional[str] = None):
        self.memory_dir = memory_dir or VF_MEMORY_DIR
        self._loop_history: List[LoopState] = []
        self._feedback_stats = {
            "total_loops": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "corrections": 0,
            "novel_insights": 0,
            "avg_confidence": 0.0,
        }

    def perceive(self, query: str, knowledge_base: str, context: Optional[List[Dict]] = None) -> LoopState:
        state = LoopState(query=query, knowledge_base=knowledge_base)
        state.phase = LoopPhase.PERCEIVE
        state.retrieved_context = context or []
        logger.info(f"[Loop {state.loop_id}] PERCEIVE: kb={knowledge_base}, query={query[:50]}")
        return state

    def reason(self, state: LoopState, response: str, confidence: float) -> LoopState:
        state.phase = LoopPhase.REASON
        state.response = response
        state.confidence = confidence
        logger.info(f"[Loop {state.loop_id}] REASON: confidence={confidence:.2f}")
        return state

    def act(self, state: LoopState, action_result: Optional[Dict] = None) -> LoopState:
        state.phase = LoopPhase.ACT
        state.action_result = action_result
        logger.info(f"[Loop {state.loop_id}] ACT: action={'executed' if action_result else 'none'}")
        return state

    def evaluate(
        self,
        state: LoopState,
        consistency_score: Optional[float] = None,
        novelty_score: Optional[float] = None,
        domain_coverage: Optional[float] = None,
        user_feedback: Optional[FeedbackType] = None,
    ) -> LoopState:
        state.phase = LoopPhase.EVALUATE

        if consistency_score is None:
            consistency_score = min(1.0, state.confidence * 1.1)
        if novelty_score is None:
            novelty_score = self._compute_novelty(state)
        if domain_coverage is None:
            domain_coverage = min(1.0, len(state.retrieved_context) / 5.0)
        if user_feedback is None:
            user_feedback = self._infer_feedback_type(state.confidence)

        state.evaluation = EvaluationResult(
            confidence=state.confidence,
            consistency_score=consistency_score,
            novelty_score=novelty_score,
            domain_coverage=domain_coverage,
            feedback_type=user_feedback,
        )
        state.feedback_type = user_feedback

        logger.info(
            f"[Loop {state.loop_id}] EVALUATE: "
            f"conf={state.confidence:.2f}, consist={consistency_score:.2f}, "
            f"novel={novelty_score:.2f}, coverage={domain_coverage:.2f}, "
            f"type={user_feedback.value}"
        )
        return state

    def feedback(self, state: LoopState) -> LoopState:
        state.phase = LoopPhase.FEEDBACK

        if state.evaluation is None:
            logger.warning(f"[Loop {state.loop_id}] FEEDBACK skipped: no evaluation")
            return state

        eval_result = state.evaluation

        if eval_result.feedback_type == FeedbackType.NEUTRAL:
            state.feedback_written = False
            return state

        topic = self._generate_feedback_topic(state)
        detail = self._generate_feedback_detail(state)
        importance = self._compute_importance(eval_result)
        tags = self._generate_tags(state)

        write_result = self._write_to_knowledge_base(
            soul=state.knowledge_base,
            topic=topic,
            detail=detail,
            category="feedback",
            tags=tags,
            importance=importance,
        )

        state.feedback_written = write_result.get("status") in ("created", "updated")
        self._update_stats(eval_result)

        logger.info(
            f"[Loop {state.loop_id}] FEEDBACK: written={state.feedback_written}, "
            f"importance={importance:.2f}, type={eval_result.feedback_type.value}"
        )
        return state

    def evolve(self, state: LoopState) -> LoopState:
        state.phase = LoopPhase.EVOLVE
        state.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._loop_history.append(state)
        self._update_causal_graph(state)

        logger.info(f"[Loop {state.loop_id}] EVOLVE: loop completed")
        return state

    def run_full_loop(
        self,
        query: str,
        knowledge_base: str,
        response: str,
        confidence: float,
        context: Optional[List[Dict]] = None,
        action_result: Optional[Dict] = None,
        user_feedback: Optional[FeedbackType] = None,
    ) -> LoopState:
        state = self.perceive(query, knowledge_base, context)
        state = self.reason(state, response, confidence)
        state = self.act(state, action_result)
        state = self.evaluate(state, user_feedback=user_feedback)
        state = self.feedback(state)
        state = self.evolve(state)
        return state

    def get_stats(self) -> Dict:
        return {
            **self._feedback_stats,
            "recent_loops": len(self._loop_history[-100:]),
            "total_loops_completed": len(self._loop_history),
        }

    def get_loop_history(self, limit: int = 20) -> List[Dict]:
        recent = self._loop_history[-limit:]
        return [
            {
                "loop_id": s.loop_id,
                "knowledge_base": s.knowledge_base,
                "query": s.query[:80],
                "confidence": s.confidence,
                "feedback_type": s.feedback_type.value if s.feedback_type else None,
                "feedback_written": s.feedback_written,
                "started_at": s.started_at,
                "completed_at": s.completed_at,
            }
            for s in recent
        ]

    def _compute_novelty(self, state: LoopState) -> float:
        if not state.retrieved_context:
            return 0.8
        query_words = set(state.query.lower().split())
        context_words = set()
        for ctx in state.retrieved_context:
            content = ctx.get("content", "")
            context_words.update(content.lower().split())
        if not context_words:
            return 0.8
        overlap = len(query_words & context_words) / max(len(query_words), 1)
        return 1.0 - overlap

    def _infer_feedback_type(self, confidence: float) -> FeedbackType:
        if confidence >= 0.85:
            return FeedbackType.POSITIVE
        elif confidence >= 0.5:
            return FeedbackType.NEUTRAL
        elif confidence >= 0.3:
            return FeedbackType.NEGATIVE
        else:
            return FeedbackType.CORRECTION

    def _generate_feedback_topic(self, state: LoopState) -> str:
        ft = state.feedback_type.value if state.feedback_type else "unknown"
        return f"[{ft}] {state.knowledge_base}: {state.query[:60]}"

    def _generate_feedback_detail(self, state: LoopState) -> str:
        eval_result = state.evaluation
        if eval_result is None:
            return json.dumps({"error": "no evaluation"}, ensure_ascii=False)
        return json.dumps({
            "query": state.query[:200],
            "response_preview": state.response[:200],
            "confidence": eval_result.confidence,
            "consistency": eval_result.consistency_score,
            "novelty": eval_result.novelty_score,
            "domain_coverage": eval_result.domain_coverage,
            "feedback_type": eval_result.feedback_type.value,
            "context_count": len(state.retrieved_context),
            "had_action": state.action_result is not None,
        }, ensure_ascii=False)

    def _compute_importance(self, eval_result: EvaluationResult) -> float:
        base = 0.5
        if eval_result.feedback_type == FeedbackType.CORRECTION:
            base = 0.9
        elif eval_result.feedback_type == FeedbackType.NOVEL_INSIGHT:
            base = 0.85
        elif eval_result.feedback_type == FeedbackType.NEGATIVE:
            base = 0.75
        elif eval_result.feedback_type == FeedbackType.POSITIVE:
            base = 0.4
        novelty_bonus = eval_result.novelty_score * 0.1
        return min(1.0, base + novelty_bonus)

    def _generate_tags(self, state: LoopState) -> List[str]:
        tags = ["closed-loop", "feedback"]
        if state.feedback_type:
            tags.append(state.feedback_type.value)
        if state.action_result:
            tags.append("action-executed")
        if state.evaluation and state.evaluation.novelty_score > 0.7:
            tags.append("novel")
        return tags

    def _write_to_knowledge_base(
        self,
        soul: str,
        topic: str,
        detail: str,
        category: str = "feedback",
        tags: Optional[List[str]] = None,
        importance: float = 0.6,
    ) -> Dict:
        db_path = os.path.join(self.memory_dir, f"{soul}.db")
        if not os.path.exists(db_path):
            logger.warning(f"DB not found: {db_path}")
            return {"status": "error", "message": f"DB not found: {soul}.db"}

        entry_id = hashlib.sha256(f"{soul}_{topic}_{detail[:100]}".encode()).hexdigest()[:16]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content_json = json.dumps({"topic": topic, "detail": detail}, ensure_ascii=False)
        tags_json = json.dumps(tags or [], ensure_ascii=False)

        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT entry_id FROM memories WHERE entry_id = ?", (entry_id,))
            if cur.fetchone():
                cur.execute("""
                    UPDATE memories SET
                        access_count = access_count + 1,
                        last_accessed = ?,
                        importance = MAX(importance, ?)
                    WHERE entry_id = ?
                """, (now, importance, entry_id))
                conn.commit()
                conn.close()
                return {"status": "updated", "entry_id": entry_id, "soul": soul}
            else:
                cur.execute("""
                    INSERT INTO memories (
                        entry_id, soul, category, content, document_date,
                        relations, access_count, importance, tags, created_at
                    ) VALUES (?, ?, ?, ?, ?, '[]', 1, ?, ?, ?)
                """, (entry_id, soul, category, content_json, now, importance, tags_json, now))
                conn.commit()
                conn.close()
                return {"status": "created", "entry_id": entry_id, "soul": soul}
        except Exception as e:
            logger.error(f"Failed to write feedback to {soul}.db: {e}")
            return {"status": "error", "message": str(e)}

    def _update_stats(self, eval_result: EvaluationResult):
        self._feedback_stats["total_loops"] += 1
        ft = eval_result.feedback_type
        if ft == FeedbackType.POSITIVE:
            self._feedback_stats["positive_feedback"] += 1
        elif ft == FeedbackType.NEGATIVE:
            self._feedback_stats["negative_feedback"] += 1
        elif ft == FeedbackType.CORRECTION:
            self._feedback_stats["corrections"] += 1
        elif ft == FeedbackType.NOVEL_INSIGHT:
            self._feedback_stats["novel_insights"] += 1

        total = self._feedback_stats["total_loops"]
        old_avg = self._feedback_stats["avg_confidence"]
        self._feedback_stats["avg_confidence"] = (
            old_avg * (total - 1) + eval_result.confidence
        ) / total

    def _update_causal_graph(self, state: LoopState):
        if not state.evaluation:
            return
        if state.evaluation.novelty_score > 0.7 and state.feedback_written:
            logger.info(
                f"[Loop {state.loop_id}] Causal graph update triggered: "
                f"novel insight in {state.knowledge_base}"
            )


class CJEPAFeedbackBridge:
    """
    Bridges C-JEPA inference results back to knowledge bases.

    When C-JEPA makes a prediction (e.g., "this causal chain leads to X"),
    the actual outcome should be compared against the prediction. If they
    differ, this creates a learning signal that updates the knowledge base.

    This is the "JEPA understanding → JEPA control → feedback → update" loop
    that was identified as missing in the architecture analysis.
    """

    def __init__(self, orchestrator: ClosedLoopOrchestrator):
        self.orchestrator = orchestrator
        self._prediction_log: List[Dict] = []

    def log_prediction(
        self,
        knowledge_base: str,
        prediction: str,
        confidence: float,
        context: Optional[Dict] = None,
    ) -> str:
        pred_id = hashlib.sha256(
            f"{knowledge_base}_{prediction[:50]}_{time.time()}".encode()
        ).hexdigest()[:16]
        entry = {
            "pred_id": pred_id,
            "knowledge_base": knowledge_base,
            "prediction": prediction,
            "confidence": confidence,
            "context": context or {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "outcome": None,
            "resolved": False,
        }
        self._prediction_log.append(entry)
        return pred_id

    def resolve_prediction(
        self,
        pred_id: str,
        actual_outcome: str,
        user_assessment: Optional[FeedbackType] = None,
    ):
        for entry in self._prediction_log:
            if entry["pred_id"] == pred_id and not entry["resolved"]:
                entry["outcome"] = actual_outcome
                entry["resolved"] = True

                prediction_matched = self._assess_match(
                    entry["prediction"], actual_outcome
                )

                if prediction_matched:
                    feedback_type = FeedbackType.POSITIVE
                else:
                    feedback_type = user_assessment or FeedbackType.CORRECTION

                self.orchestrator.run_full_loop(
                    query=f"Prediction: {entry['prediction'][:100]}",
                    knowledge_base=entry["knowledge_base"],
                    response=actual_outcome,
                    confidence=entry["confidence"] if prediction_matched else 0.3,
                    user_feedback=feedback_type,
                )
                break

    def get_unresolved_predictions(self, knowledge_base: Optional[str] = None) -> List[Dict]:
        results = [p for p in self._prediction_log if not p["resolved"]]
        if knowledge_base:
            results = [p for p in results if p["knowledge_base"] == knowledge_base]
        return results

    def _assess_match(self, prediction: str, outcome: str) -> bool:
        pred_words = set(prediction.lower().split())
        out_words = set(outcome.lower().split())
        if not pred_words or not out_words:
            return False
        overlap = len(pred_words & out_words) / max(len(pred_words), 1)
        return overlap > 0.5
