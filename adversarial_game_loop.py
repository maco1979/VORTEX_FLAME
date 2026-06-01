#!/usr/bin/env python3
"""
VORTEX FLAME Iterative Review-Fix Loop — LLM Execute, JEPA Verify
===================================================================

Formalizes the multi-round adversarial methodology:
  DeepSeek(执行) → 豆包(审查) → DeepSeek(修复) → 豆包(再审查) → ... → 0 BUG

Mapped to VORTEX FLAME:
  LLM(执行) → JEPA(感知审查) → LLM(修复) → JEPA(再审查) → ... → 0 BUG

Core Loop:
  ┌──────────────────────────────────────────────┐
  │  1. EXECUTE:  LLM produces output            │
  │  2. REVIEW:   JEPA + KB verify the output     │
   │  3. BUGS?     YES → LLM fixes → goto 2       │
  │              NO  → DONE ✓                      │
  └──────────────────────────────────────────────┘

Termination:
  - JEPA review passes (0 bugs found)
  - Max iterations reached (forced acceptance)
  - Consecutive clean reviews >= threshold

This is NOT a debate. It's an iterative quality gate:
  - LLM is the worker (produces artifacts)
  - JEPA is the inspector (finds defects)
  - Knowledge base is the spec (ground truth)

Usage:
  from adversarial_game_loop import IterativeReviewLoop

  loop = IterativeReviewLoop(memory_dir=".vf_memory")
  result = loop.execute(
      task="Implement gradient clipping for transformer training",
      soul="cezanne",
      modality="code",
  )
"""

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(os.path.dirname(__file__)))

from jepa_soul_bridge import (
    JEPASoulBridge, JEPAModality, BridgeResult, SlotDescription,
    LLMBackend, CloudBackend, LocalBackend, HybridBackend,
    MODALITY_SOUL_MAP, MODALITY_SLOT_NAMES,
)
from soul_memory import SoulMemoryEngine


class IterationPhase(Enum):
    EXECUTE = "execute"
    REVIEW = "review"
    FIX = "fix"
    DONE = "done"


class BugSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    STYLE = "style"


@dataclass
class Bug:
    bug_id: str
    description: str
    severity: BugSeverity
    location: str
    suggested_fix: str
    verified_by: str


@dataclass
class IterationState:
    iteration: int
    phase: IterationPhase
    output: Optional[str] = None
    bugs_found: List[Bug] = field(default_factory=list)
    bugs_fixed: List[str] = field(default_factory=list)
    review_passed: bool = False
    slot_evidence: List[SlotDescription] = field(default_factory=list)
    kb_evidence: List[Dict] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class TaskState:
    task_id: str
    task: str
    soul: str
    modality: Optional[JEPAModality]
    raw_input: Any
    iterations: List[IterationState] = field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 10
    final_output: Optional[str] = None
    total_bugs_found: int = 0
    total_bugs_fixed: int = 0
    clean_reviews: int = 0
    required_clean_reviews: int = 2
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    task: str
    soul: str
    final_output: str
    iterations_used: int
    total_bugs_found: int
    total_bugs_fixed: int
    passed: bool
    elapsed_seconds: float
    trajectory: List[Dict]


class LLMExecutor:
    """
    LLM-based executor: produces artifacts and fixes bugs.
    This is the "DeepSeek" role — it does the actual work.
    """

    def __init__(self, llm: LLMBackend, memory: SoulMemoryEngine):
        self.llm = llm
        self.memory = memory

    def execute(self, task: str, soul: str, context: str = "") -> str:
        knowledge = self.memory.recall(soul, task, top_k=5)
        kb_context = self._format_knowledge(knowledge)

        if self.llm.is_available():
            return self._llm_execute(task, soul, kb_context, context)
        return self._template_execute(task, soul, kb_context)

    def fix(self, task: str, soul: str, output: str,
            bugs: List[Bug], context: str = "") -> str:
        if not bugs:
            return output

        if self.llm.is_available():
            return self._llm_fix(task, soul, output, bugs, context)

        return self._template_fix(output, bugs)

    def _llm_execute(self, task: str, soul: str,
                     kb_context: str, context: str) -> str:
        prompt = f"""You are the {soul} soul executing a task.

Task: {task}

Knowledge base context:
{kb_context[:2000]}

{f'Additional context: {context[:500]}' if context else ''}

Produce a complete, correct solution. Be specific and precise."""

        system = (f"You are the {soul} soul. Execute tasks with precision. "
                  f"Output concrete, verifiable results.")
        return self.llm.generate(prompt, system, max_tokens=2048, temperature=0.3)

    def _template_execute(self, task: str, soul: str,
                          kb_context: str) -> str:
        return (f"[{soul} execution] Task: {task}\n"
                f"Knowledge context: {kb_context[:500]}\n"
                f"Note: LLM unavailable, using template execution.")

    def _llm_fix(self, task: str, soul: str, output: str,
                 bugs: List[Bug], context: str) -> str:
        bug_descriptions = []
        for b in bugs:
            bug_descriptions.append(
                f"  [{b.severity.value}] {b.description}\n"
                f"    Location: {b.location}\n"
                f"    Suggested fix: {b.suggested_fix}"
            )

        prompt = f"""You are the {soul} soul fixing bugs in your previous output.

Task: {task}

Previous output:
{output[:3000]}

Bugs found by JEPA reviewer:
{chr(10).join(bug_descriptions)}

{f'Additional context: {context[:500]}' if context else ''}

Fix ALL the bugs listed above. Output the corrected version in full."""

        system = (f"You are the {soul} soul. Fix bugs precisely. "
                  f"Address every issue the reviewer found.")
        return self.llm.generate(prompt, system, max_tokens=2048, temperature=0.2)

    def _template_fix(self, output: str, bugs: List[Bug]) -> str:
        fix_notes = []
        for b in bugs:
            fix_notes.append(f"  [FIXED] {b.description}: {b.suggested_fix}")
        return f"{output}\n\n[Applied fixes]:\n" + "\n".join(fix_notes)

    def _format_knowledge(self, knowledge: List[Dict]) -> str:
        if not knowledge:
            return "No relevant knowledge found."
        parts = []
        for i, entry in enumerate(knowledge[:5]):
            content = entry.get("content", {})
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except Exception:
                    content = {"text": content}
            if isinstance(content, dict):
                topic = content.get("topic", "")
                detail = content.get("detail", "")
                parts.append(f"  {i+1}. {topic}: {detail[:200]}")
            else:
                parts.append(f"  {i+1}. {str(content)[:200]}")
        return "\n".join(parts)


class JEPACReviewer:
    """
    JEPA + Knowledge Base reviewer: inspects LLM output for defects.
    This is the "豆包" role — it reviews and finds bugs.

    Review sources:
      1. JEPA slot analysis (if JEPA model available for modality)
      2. BM25S knowledge base fact-checking
      3. Rule-based static analysis (always available)
      4. LLM-assisted review (if LLM available)
    """

    def __init__(self, bridge: JEPASoulBridge, llm: LLMBackend,
                 memory: SoulMemoryEngine):
        self.bridge = bridge
        self.llm = llm
        self.memory = memory

    def review(self, task: str, output: str, soul: str,
               modality: Optional[JEPAModality] = None,
               raw_input: Any = None) -> Tuple[List[Bug], List[SlotDescription], List[Dict]]:
        bugs = []
        slot_descs = []
        kb_evidence = []

        kb_evidence = self.memory.recall(soul, task, top_k=5)
        kb_bugs = self._kb_fact_check(task, output, soul, kb_evidence)
        bugs.extend(kb_bugs)

        if modality and raw_input is not None and modality in self.bridge._jepa_models:
            jepa_bugs, slot_descs = self._jepa_review(task, output, modality, soul, raw_input)
            bugs.extend(jepa_bugs)

        rule_bugs = self._rule_review(task, output, soul)
        bugs.extend(rule_bugs)

        if self.llm.is_available() and len(bugs) < 3:
            llm_bugs = self._llm_review(task, output, soul)
            bugs.extend(llm_bugs)

        seen = set()
        unique_bugs = []
        for b in bugs:
            key = (b.description[:50], b.severity)
            if key not in seen:
                seen.add(key)
                unique_bugs.append(b)

        unique_bugs.sort(key=lambda b: {
            BugSeverity.CRITICAL: 0, BugSeverity.HIGH: 1,
            BugSeverity.MEDIUM: 2, BugSeverity.LOW: 3, BugSeverity.STYLE: 4,
        }.get(b.severity, 5))

        return unique_bugs, slot_descs, kb_evidence

    def _kb_fact_check(self, task: str, output: str, soul: str,
                       kb_evidence: List[Dict]) -> List[Bug]:
        bugs = []
        if not kb_evidence:
            bugs.append(Bug(
                bug_id=f"kb_{0:02d}",
                description="No knowledge base evidence found for this task domain",
                severity=BugSeverity.MEDIUM,
                location="output:general",
                suggested_fix="Verify claims against domain knowledge or add relevant KB entries",
                verified_by="kb_search",
            ))
        return bugs

    def _jepa_review(self, task: str, output: str, modality: JEPAModality,
                     soul: str, raw_input: Any) -> Tuple[List[Bug], List[SlotDescription]]:
        bugs = []
        try:
            result = self.bridge.process(
                raw_input=raw_input,
                modality=modality,
                soul=soul,
                query=f"verify: {task}",
                use_llm=False,
            )
            slot_descs = result.slot_descriptions

            active = [s for s in slot_descs if s.activation >= 0.5]
            if not active:
                bugs.append(Bug(
                    bug_id="jepa_00",
                    description="JEPA perception finds no significant features — output may not align with input",
                    severity=BugSeverity.HIGH,
                    location="jepa:perception",
                    suggested_fix="Verify the output corresponds to the actual input features",
                    verified_by="jepa_slots",
                ))

            dominant = active[0] if active else None
            if dominant and dominant.activation > 5.0:
                bugs.append(Bug(
                    bug_id="jepa_01",
                    description=f"JEPA slot '{dominant.name}' over-activated ({dominant.activation:.1f}) — possible hallucination or overfitting",
                    severity=BugSeverity.MEDIUM,
                    location=f"jepa:slot_{dominant.name}",
                    suggested_fix="Check if the dominant feature is genuinely present or an artifact",
                    verified_by="jepa_slots",
                ))

            return bugs, slot_descs

        except Exception as e:
            bugs.append(Bug(
                bug_id="jepa_err",
                description=f"JEPA review failed: {str(e)[:100]}",
                severity=BugSeverity.LOW,
                location="jepa:runtime",
                suggested_fix="JEPA model may not be loaded for this modality",
                verified_by="jepa_error",
            ))
            return bugs, []

    def _rule_review(self, task: str, output: str, soul: str) -> List[Bug]:
        bugs = []

        if len(output) < 20:
            bugs.append(Bug(
                bug_id="rule_00",
                description="Output is too short to be a meaningful solution",
                severity=BugSeverity.HIGH,
                location="output:length",
                suggested_fix="Provide a more complete and detailed solution",
                verified_by="rule_check",
            ))

        if "TODO" in output or "FIXME" in output or "HACK" in output:
            bugs.append(Bug(
                bug_id="rule_01",
                description="Output contains unresolved TODO/FIXME/HACK markers",
                severity=BugSeverity.MEDIUM,
                location="output:markers",
                suggested_fix="Resolve all placeholder markers before submission",
                verified_by="rule_check",
            ))

        if "error" in output.lower() and "handle" not in output.lower():
            bugs.append(Bug(
                bug_id="rule_02",
                description="Output mentions 'error' but lacks error handling",
                severity=BugSeverity.MEDIUM,
                location="output:error_handling",
                suggested_fix="Add proper error handling for mentioned error cases",
                verified_by="rule_check",
            ))

        critical_keywords = ["NaN", "infinity", "segfault", "overflow", "underflow"]
        in_code_block = False
        code_lines = []
        text_lines = []
        for line in output.split("\n"):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                code_lines.append(line)
            else:
                text_lines.append(line)
        code_text = "\n".join(code_lines)
        text_lower = "\n".join(text_lines).lower()
        for kw in critical_keywords:
            kw_in_code = kw.lower() in code_text.lower()
            kw_in_explanation = kw.lower() in text_lower
            if kw_in_code:
                bugs.append(Bug(
                    bug_id=f"rule_{kw}",
                    description=f"Code contains critical issue indicator: {kw}",
                    severity=BugSeverity.CRITICAL,
                    location=f"output:code:{kw}",
                    suggested_fix=f"Investigate and fix the {kw} issue in code",
                    verified_by="rule_check",
                ))
                break
            elif kw_in_explanation:
                bugs.append(Bug(
                    bug_id=f"rule_{kw}_mention",
                    description=f"Output mentions '{kw}' in explanatory text — verify this is intentional, not a latent bug",
                    severity=BugSeverity.LOW,
                    location=f"output:text:{kw}",
                    suggested_fix=f"Confirm '{kw}' is used descriptively, not indicating an actual defect",
                    verified_by="rule_check",
                ))
                break

        if not bugs:
            bugs.append(Bug(
                bug_id="rule_pass",
                description="Rule-based review passed — no obvious defects found",
                severity=BugSeverity.STYLE,
                location="output:general",
                suggested_fix="Consider deeper JEPA or KB verification",
                verified_by="rule_check",
            ))

        return bugs

    def _llm_review(self, task: str, output: str, soul: str) -> List[Bug]:
        prompt = f"""Review this {soul} soul output for defects.

Task: {task}

Output to review:
{output[:2000]}

Find specific bugs or issues. For each, state:
1. What the bug is
2. Severity: critical/high/medium/low
3. Where it is
4. How to fix it

Respond in JSON:
{{
  "bugs": [
    {{
      "description": "...",
      "severity": "high",
      "location": "...",
      "suggested_fix": "..."
    }}
  ]
}}

If no bugs found, return empty bugs array."""

        response = self.llm.generate(
            prompt,
            system="You are a code reviewer. Find real bugs only. Output valid JSON.",
            temperature=0.1,
        )

        try:
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())
            bugs = []
            for i, b in enumerate(data.get("bugs", [])[:5]):
                sev_str = b.get("severity", "medium").lower()
                try:
                    severity = BugSeverity(sev_str)
                except ValueError:
                    severity = BugSeverity.MEDIUM
                bugs.append(Bug(
                    bug_id=f"llm_{i:02d}",
                    description=b.get("description", "Unspecified issue"),
                    severity=severity,
                    location=b.get("location", "unknown"),
                    suggested_fix=b.get("suggested_fix", "Review and fix"),
                    verified_by="llm_review",
                ))
            return bugs
        except Exception:
            return []


class IterativeReviewLoop:
    """
    Main orchestrator: LLM executes, JEPA reviews, iterate until clean.

    This is the formalization of the DeepSeek→豆包 iterative methodology:
      1. LLM produces output
      2. JEPA + KB reviews for bugs
      3. If bugs found → LLM fixes → goto 2
      4. If clean review → DONE

    The loop terminates when:
      - JEPA review finds 0 actionable bugs (clean pass)
      - Consecutive clean reviews >= required_clean_reviews
      - Max iterations reached (forced acceptance with warning)
    """

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        llm_backend: Optional[LLMBackend] = None,
        bridge: Optional[JEPASoulBridge] = None,
        max_iterations: int = 10,
        required_clean_reviews: int = 2,
    ):
        self.llm = llm_backend or CloudBackend()
        self.memory = SoulMemoryEngine(memory_dir)
        self.bridge = bridge or JEPASoulBridge(
            memory_dir=memory_dir, llm_backend=self.llm
        )

        self.executor = LLMExecutor(self.llm, self.memory)
        self.reviewer = JEPACReviewer(self.bridge, self.llm, self.memory)

        self.max_iterations = max_iterations
        self.required_clean_reviews = required_clean_reviews
        self._active_tasks: Dict[str, TaskState] = {}

    def execute(
        self,
        task: str,
        soul: str = "cezanne",
        modality: Optional[str] = None,
        raw_input: Any = None,
        max_iterations: Optional[int] = None,
        context: Optional[Dict] = None,
    ) -> TaskResult:
        task_id = str(uuid.uuid4())[:8]
        jepa_modality = None
        if modality:
            try:
                jepa_modality = JEPAModality(modality)
            except ValueError:
                jepa_modality = None

        state = TaskState(
            task_id=task_id,
            task=task,
            soul=soul,
            modality=jepa_modality,
            raw_input=raw_input,
            max_iterations=max_iterations or self.max_iterations,
            start_time=time.time(),
            metadata=context or {},
        )

        self._active_tasks[task_id] = state

        print(f"\n{'='*60}")
        print(f"ITERATIVE REVIEW LOOP [{task_id}]")
        print(f"Task: {task[:80]}")
        print(f"Soul: {soul} | Modality: {modality or 'text'}")
        print(f"Max iterations: {state.max_iterations}")
        print(f"{'='*60}")

        current_output = ""
        context_str = ""

        for iteration in range(state.max_iterations):
            state.current_iteration = iteration
            iter_state = IterationState(
                iteration=iteration,
                phase=IterationPhase.EXECUTE,
                timestamp=time.time(),
            )

            print(f"\n--- Iteration {iteration + 1}/{state.max_iterations} ---")

            if iteration == 0:
                print(f"  EXECUTE: LLM producing initial output...")
                current_output = self.executor.execute(task, soul, context_str)
                iter_state.output = current_output
                iter_state.phase = IterationPhase.REVIEW
            else:
                prev_bugs = state.iterations[-1].bugs_found if state.iterations else []
                actionable = [b for b in prev_bugs if b.severity in (
                    BugSeverity.CRITICAL, BugSeverity.HIGH, BugSeverity.MEDIUM
                )]
                if not actionable:
                    print(f"  No actionable bugs — checking if clean review confirmed...")
                    state.clean_reviews += 1
                    if state.clean_reviews >= state.required_clean_reviews:
                        print(f"  ✓ {state.clean_reviews} consecutive clean reviews — DONE")
                        break
                    print(f"  Need {state.required_clean_reviews - state.clean_reviews} more clean reviews")
                    iter_state.phase = IterationPhase.REVIEW
                else:
                    print(f"  FIX: LLM fixing {len(actionable)} bugs...")
                    current_output = self.executor.fix(
                        task, soul, current_output, actionable, context_str
                    )
                    iter_state.output = current_output
                    iter_state.phase = IterationPhase.REVIEW
                    iter_state.bugs_fixed = [b.bug_id for b in actionable]

            print(f"  REVIEW: JEPA + KB inspecting output...")
            bugs, slot_descs, kb_evidence = self.reviewer.review(
                task, current_output, soul, jepa_modality, raw_input
            )
            iter_state.bugs_found = bugs
            iter_state.slot_evidence = slot_descs
            iter_state.kb_evidence = kb_evidence

            actionable = [b for b in bugs if b.severity in (
                BugSeverity.CRITICAL, BugSeverity.HIGH, BugSeverity.MEDIUM
            )]
            style_only = [b for b in bugs if b.severity in (BugSeverity.LOW, BugSeverity.STYLE)]

            state.total_bugs_found += len(actionable)
            state.total_bugs_fixed += len(iter_state.bugs_fixed)

            if not actionable:
                iter_state.review_passed = True
                state.clean_reviews += 1
                print(f"  ✓ Review PASSED (clean: {state.clean_reviews}/{state.required_clean_reviews})")
                if style_only:
                    print(f"  ℹ Style notes: {len(style_only)} (non-blocking)")
                if state.clean_reviews >= state.required_clean_reviews:
                    iter_state.phase = IterationPhase.DONE
                    state.iterations.append(iter_state)
                    print(f"  ✓✓ Confirmed clean — task complete!")
                    break
            else:
                state.clean_reviews = 0
                print(f"  ✗ Review FAILED: {len(actionable)} bugs found")
                for b in actionable[:5]:
                    print(f"    [{b.severity.value}] {b.description[:60]}")
                    print(f"      → {b.suggested_fix[:60]}")

            context_str = self._build_context(state, current_output, bugs)
            state.iterations.append(iter_state)

        else:
            print(f"\n  ⚠ Max iterations ({state.max_iterations}) reached — forced acceptance")

        state.final_output = current_output
        state.end_time = time.time()

        self._save_trajectory(state)

        passed = state.clean_reviews >= state.required_clean_reviews

        result = TaskResult(
            task_id=task_id,
            task=task,
            soul=soul,
            final_output=current_output,
            iterations_used=len(state.iterations),
            total_bugs_found=state.total_bugs_found,
            total_bugs_fixed=state.total_bugs_fixed,
            passed=passed,
            elapsed_seconds=state.end_time - state.start_time,
            trajectory=self._serialize_trajectory(state),
        )

        print(f"\n{'='*60}")
        print(f"ITERATIVE REVIEW LOOP [{task_id}] COMPLETE")
        print(f"Passed: {'YES' if passed else 'NO'} | Iterations: {result.iterations_used}")
        print(f"Bugs found: {result.total_bugs_found} | Bugs fixed: {result.total_bugs_fixed}")
        print(f"Time: {result.elapsed_seconds:.1f}s")
        print(f"{'='*60}\n")

        del self._active_tasks[task_id]
        return result

    def _build_context(self, state: TaskState, current_output: str,
                       bugs: List[Bug]) -> str:
        parts = []
        if state.iterations:
            prev = state.iterations[-1]
            fixed = [b.bug_id for b in prev.bugs_found if b.severity in (
                BugSeverity.CRITICAL, BugSeverity.HIGH, BugSeverity.MEDIUM
            )]
            if fixed:
                parts.append(f"Previously fixed bugs: {', '.join(fixed)}")

        remaining = [b for b in bugs if b.severity in (
            BugSeverity.CRITICAL, BugSeverity.HIGH, BugSeverity.MEDIUM
        )]
        if remaining:
            parts.append(f"Remaining bugs to fix: {len(remaining)}")
            for b in remaining[:3]:
                parts.append(f"  - [{b.severity.value}] {b.description[:80]}")

        return "\n".join(parts)

    def _save_trajectory(self, state: TaskState):
        try:
            self.memory.write(
                state.soul,
                "trajectory",
                {
                    "task_id": state.task_id,
                    "task": state.task,
                    "iterations": len(state.iterations),
                    "bugs_found": state.total_bugs_found,
                    "bugs_fixed": state.total_bugs_fixed,
                    "passed": state.clean_reviews >= state.required_clean_reviews,
                    "elapsed_seconds": state.end_time - state.start_time,
                },
            )
        except Exception as e:
            print(f"  ⚠ Failed to save trajectory to memory: {e}")
            try:
                traj_dir = os.path.join(
                    self.memory.memory_dir or ".", "review_trajectories"
                )
                os.makedirs(traj_dir, exist_ok=True)
                traj_path = os.path.join(traj_dir, f"review_{state.task_id}.json")
                with open(traj_path, "w", encoding="utf-8") as f:
                    json.dump(
                        self._serialize_trajectory(state), f,
                        ensure_ascii=False, indent=2,
                    )
            except Exception as e2:
                print(f"  ⚠ Fallback save also failed: {e2}")

    def _serialize_trajectory(self, state: TaskState) -> List[Dict]:
        trajectory = []
        for it in state.iterations:
            entry = {
                "iteration": it.iteration,
                "phase": it.phase.value,
                "review_passed": it.review_passed,
                "bugs_found": [
                    {
                        "id": b.bug_id,
                        "severity": b.severity.value,
                        "description": b.description[:200],
                        "location": b.location,
                        "verified_by": b.verified_by,
                    }
                    for b in it.bugs_found
                ],
                "bugs_fixed": it.bugs_fixed,
                "n_slot_evidence": len(it.slot_evidence),
                "n_kb_evidence": len(it.kb_evidence),
            }
            trajectory.append(entry)
        return trajectory

    def get_active_tasks(self) -> Dict[str, Dict]:
        return {
            tid: {
                "task": t.task[:80],
                "soul": t.soul,
                "iteration": t.current_iteration,
                "max_iterations": t.max_iterations,
            }
            for tid, t in self._active_tasks.items()
        }


class AdversarialAgentLoop:
    """
    Integration layer: embeds IterativeReviewLoop inside AgentLoop.

    When AgentLoop encounters a complex step, it can invoke the
    iterative review loop instead of a simple LLM call, making
    the agent's output more robust through LLM→JEPA→LLM iteration.

    Usage (inside agent_loop.py):
        from adversarial_game_loop import AdversarialAgentLoop

        adv = AdversarialAgentLoop(memory_dir="...", llm_backend=cloud)
        result = adv.execute_step_with_review(
            step=step, working_memory=wm, artifacts=artifacts,
        )
    """

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        llm_backend: Optional[LLMBackend] = None,
        bridge: Optional[JEPASoulBridge] = None,
        max_iterations: int = 5,
        complexity_threshold: float = 0.5,
    ):
        self.review_loop = IterativeReviewLoop(
            memory_dir=memory_dir,
            llm_backend=llm_backend,
            bridge=bridge,
            max_iterations=max_iterations,
            required_clean_reviews=2,
        )
        self.complexity_threshold = complexity_threshold

    def should_review(self, step_description: str, context: Dict = None) -> bool:
        complexity = self._estimate_complexity(step_description, context or {})
        return complexity >= self.complexity_threshold

    def _estimate_complexity(self, description: str, context: Dict) -> float:
        score = 0.0
        complex_keywords = [
            "implement", "debug", "fix", "optimize", "refactor",
            "analyze", "design", "architect", "verify", "validate",
            "diagnose", "troubleshoot", "ensure", "guarantee",
        ]
        desc_lower = description.lower()
        for kw in complex_keywords:
            if kw in desc_lower:
                score += 0.12

        if len(description) > 200:
            score += 0.1
        if context.get("requires_verification"):
            score += 0.2
        if context.get("safety_critical"):
            score += 0.3

        return min(score, 1.0)

    def execute_step_with_review(
        self,
        step,
        working_memory=None,
        artifacts: Dict = None,
    ) -> Any:
        soul = getattr(step, 'soul', 'cezanne')
        modality = getattr(step, 'modality', None)
        modality_str = modality.value if modality else None
        description = getattr(step, 'description', str(step))

        raw_input = None
        if artifacts:
            import torch
            for v in artifacts.values():
                if isinstance(v, torch.Tensor):
                    raw_input = v
                    break

        result = self.review_loop.execute(
            task=description,
            soul=soul,
            modality=modality_str,
            raw_input=raw_input,
            max_iterations=5,
        )

        if working_memory:
            working_memory.add(
                "iterative_review",
                f"Review [{result.task_id}]: {'PASSED' if result.passed else 'FAILED'} "
                f"in {result.iterations_used} iterations. "
                f"Bugs: {result.total_bugs_found} found, {result.total_bugs_fixed} fixed.",
            )

        return result.final_output


if __name__ == "__main__":
    print("VORTEX FLAME Iterative Review Loop")
    print("=" * 40)
    print()
    print("Core: LLM(Execute) → JEPA(Review) → LLM(Fix) → ... → 0 BUG")
    print()
    print("Quick test (Cezanne soul, code task, no JEPA model):")

    loop = IterativeReviewLoop(
        memory_dir=".vf_memory",
        max_iterations=3,
        required_clean_reviews=1,
    )
    result = loop.execute(
        task="Explain how gradient clipping prevents training instability",
        soul="cezanne",
        modality=None,
        raw_input=None,
    )

    print(f"\nFinal result:")
    print(f"  Passed: {result.passed}")
    print(f"  Iterations: {result.iterations_used}")
    print(f"  Bugs: {result.total_bugs_found} found, {result.total_bugs_fixed} fixed")
    print(f"  Output: {result.final_output[:200]}")
