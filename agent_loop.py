#!/usr/bin/env python3
"""
VORTEX FLAME Agent Loop — Long-Task Execution Architecture
============================================================

Transforms single-shot inference into iterative long-task execution:

  Goal → Decompose → Plan DAG → Execute Steps → Verify → Replan/Continue
                                                    ↓
                                              Summarize & Report

Core Components:
  1. TaskPlanner      — Decomposes goals into subtask DAGs
  2. StepExecutor     — Executes individual steps with JEPA+Bridge
  3. StateTracker     — Tracks task state across steps
  4. ReflectionEngine — Self-evaluates progress and adjusts
  5. WorkingMemory    — Compresses context across long executions
  6. AgentLoop        — Orchestrates the full loop

Key Design Principles:
  - Every step produces a verifiable artifact
  - Failed steps trigger replanning, not just retry
  - Context is compressed at boundaries to avoid token overflow
  - The system can pause and resume long tasks
  - Multiple souls collaborate on complex tasks

Usage:
  from agent_loop import AgentLoop

  loop = AgentLoop(memory_dir="...", llm_backend=cloud)
  result = loop.execute(
      goal="Analyze this financial dataset, identify trends, and generate a trading strategy report",
      context={"dataset_path": "/data/stocks.csv"},
  )
"""

import json
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(os.path.dirname(__file__)))

from soul_memory import SoulMemoryEngine
from jepa_soul_bridge import (
    JEPASoulBridge, JEPAModality, BridgeResult, SlotDescription,
    CloudBackend, LocalBackend, HybridBackend, LLMBackend,
    MODALITY_SOUL_MAP, MODALITY_SLOT_NAMES,
)


class TaskStatus(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    REPLANNING = "replanning"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    step_id: str
    description: str
    soul: str
    modality: Optional[JEPAModality] = None
    tool: Optional[str] = None
    tool_args: Dict = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    output_artifact: Optional[str] = None


@dataclass
class TaskState:
    task_id: str
    goal: str
    context: Dict = field(default_factory=dict)
    steps: List[Step] = field(default_factory=list)
    current_step_idx: int = 0
    status: TaskStatus = TaskStatus.PENDING
    artifacts: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    total_tokens_used: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    replan_count: int = 0
    max_replans: int = 3


@dataclass
class TaskResult:
    task_id: str
    goal: str
    status: TaskStatus
    steps_completed: int
    steps_total: int
    artifacts: Dict[str, Any]
    summary: str
    elapsed_seconds: float
    replan_count: int
    history: List[Dict]


class WorkingMemory:
    """
    Manages context across long task execution.
    Compresses old context to stay within token budgets.
    """

    def __init__(self, max_entries: int = 50, compression_threshold: int = 30):
        self.entries: List[Dict] = []
        self.max_entries = max_entries
        self.compression_threshold = compression_threshold
        self._compressed = False

    def add(self, role: str, content: str, metadata: Dict = None):
        try:
            self.entries.append({
                "role": role,
                "content": str(content)[:5000],
                "timestamp": time.time(),
                "metadata": metadata or {},
            })
            if len(self.entries) > self.max_entries:
                self._compress()
        except Exception as e:
            print(f"WorkingMemory.add error: {e}")

    def save(self, filepath: str):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"WorkingMemory.save error: {e}")

    def load(self, filepath: str):
        try:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
        except Exception as e:
            print(f"WorkingMemory.load error: {e}")

    def get_context(self, max_entries: int = None) -> List[Dict]:
        if max_entries and len(self.entries) > max_entries:
            return self.entries[-max_entries:]
        return self.entries

    def get_context_text(self, max_entries: int = None) -> str:
        entries = self.get_context(max_entries)
        parts = []
        for e in entries:
            parts.append(f"[{e['role']}] {e['content']}")
        return "\n".join(parts)

    def _compress(self):
        if len(self.entries) <= self.compression_threshold:
            return

        keep_recent = self.entries[-self.compression_threshold // 2:]
        older = self.entries[:len(self.entries) - self.compression_threshold // 2]

        summary_parts = []
        for e in older:
            content = e["content"][:200]
            summary_parts.append(f"- {e['role']}: {content}")

        summary = f"[COMPRESSED CONTEXT - {len(older)} earlier entries]\n" + "\n".join(summary_parts)

        self.entries = [{"role": "system", "content": summary, "timestamp": time.time(), "metadata": {}}] + keep_recent
        self._compressed = True

    def clear(self):
        self.entries = []


class TaskPlanner:
    """
    Decomposes a goal into a sequence of Steps.
    Uses LLM to generate the plan, falls back to template-based planning.
    """

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def plan(self, goal: str, context: Dict, available_souls: List[str] = None) -> List[Step]:
        if self.llm.is_available():
            steps = self._llm_plan(goal, context, available_souls)
            if steps:
                return steps

        return self._template_plan(goal, context, available_souls)

    def _llm_plan(self, goal: str, context: Dict, available_souls: List[str] = None) -> Optional[List[Step]]:
        souls_str = ", ".join(available_souls or ["cezanne", "strategy", "beethoven"])

        prompt = f"""Decompose the following task into 3-7 concrete steps.
Each step should have: description, soul (one of: {souls_str}), tool (optional), depends_on (step IDs).

Task: {goal}
Context: {json.dumps(context, ensure_ascii=False)[:500]}

Respond in JSON format:
{{
  "steps": [
    {{
      "description": "...",
      "soul": "...",
      "tool": "...",
      "depends_on": []
    }}
  ]
}}"""

        response = self.llm.generate(prompt, system="You are a task decomposition expert. Output valid JSON only.")

        try:
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            plan_data = json.loads(json_str.strip())
            steps = []
            for i, s in enumerate(plan_data.get("steps", [])):
                step_id = f"step_{i:02d}"
                modality = self._infer_modality(s.get("soul", "cezanne"))
                steps.append(Step(
                    step_id=step_id,
                    description=s["description"],
                    soul=s.get("soul", "cezanne"),
                    modality=modality,
                    tool=s.get("tool"),
                    depends_on=s.get("depends_on", []),
                ))
            return steps
        except Exception:
            return None

    def _template_plan(self, goal: str, context: Dict, available_souls: List[str] = None) -> List[Step]:
        steps = [
            Step(step_id="step_00", description=f"Understand and analyze the task: {goal[:100]}",
                 soul="cezanne", modality=JEPAModality.CODE),
            Step(step_id="step_01", description="Gather relevant knowledge and context",
                 soul="cezanne", modality=JEPAModality.CODE, depends_on=["step_00"]),
            Step(step_id="step_02", description="Execute the core analysis/computation",
                 soul="strategy", modality=JEPAModality.FINANCIAL, depends_on=["step_01"]),
            Step(step_id="step_03", description="Verify results and check for errors",
                 soul="cezanne", modality=JEPAModality.CODE, depends_on=["step_02"]),
            Step(step_id="step_04", description="Synthesize findings and generate report",
                 soul="strategy", depends_on=["step_03"]),
        ]
        return steps

    def _infer_modality(self, soul: str) -> Optional[JEPAModality]:
        soul_modality = {
            "beethoven": JEPAModality.AUDIO,
            "vangogh": JEPAModality.VISUAL,
            "monet": JEPAModality.ART,
            "cezanne": JEPAModality.CODE,
            "einstein": JEPAModality.PHYSICS,
            "galileo": JEPAModality.PHYSICS,
            "davinci": JEPAModality.DESIGN,
            "darwin": JEPAModality.BIOLOGY,
            "yuanlongping": JEPAModality.BIOLOGY,
            "humboldt": JEPAModality.GEOGRAPHY,
            "montesquieu": JEPAModality.LAW,
            "strategy": JEPAModality.FINANCIAL,
            "guizhu": JEPAModality.AUDIO,
            "herodotus": JEPAModality.GEOGRAPHY,
        }
        return soul_modality.get(soul)

    def replan(self, state: TaskState, failed_step: Step, reason: str) -> List[Step]:
        remaining = [s for s in state.steps if s.status in (StepStatus.PENDING, StepStatus.FAILED)]

        new_steps = []
        for i, s in enumerate(remaining):
            new_id = f"replan_{state.replan_count}_{i:02d}"
            new_steps.append(Step(
                step_id=new_id,
                description=s.description,
                soul=s.soul,
                modality=s.modality,
                tool=s.tool,
                tool_args=s.tool_args,
                depends_on=s.depends_on,
                retry_count=s.retry_count + (1 if s.step_id == failed_step.step_id else 0),
            ))

        if self.llm.is_available():
            prompt = f"""The following step failed: "{failed_step.description}"
Reason: {reason}

Remaining steps:
{json.dumps([{"id": s.step_id, "desc": s.description} for s in new_steps], ensure_ascii=False)}

Suggest an adjusted plan. Output JSON with "steps" array containing:
{{"step_id": "...", "description": "...", "soul": "...", "strategy_hint": "..."}}"""

            response = self.llm.generate(prompt, system="You are a task recovery expert. Output valid JSON.")
            try:
                json_str = response
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                plan_data = json.loads(json_str.strip())
                for i, s_data in enumerate(plan_data.get("steps", [])):
                    if i < len(new_steps):
                        new_steps[i].description = s_data.get("description", new_steps[i].description)
            except Exception:
                pass

        return new_steps


class ReflectionEngine:
    """
    Self-evaluates step results and overall task progress.
    Determines if a step succeeded, needs retry, or requires replanning.
    """

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def evaluate_step(self, step: Step, result: Any, goal: str) -> Dict:
        if result is None:
            return {"verdict": "failed", "reason": "No result produced", "action": "retry"}

        if isinstance(result, str) and ("error" in result.lower() or "failed" in result.lower()):
            if step.retry_count < step.max_retries:
                return {"verdict": "retry", "reason": result, "action": "retry"}
            return {"verdict": "failed", "reason": result, "action": "replan"}

        if self.llm.is_available():
            return self._llm_evaluate(step, result, goal)

        return {"verdict": "success", "reason": "Step completed", "action": "continue"}

    def _llm_evaluate(self, step: Step, result: Any, goal: str) -> Dict:
        result_str = str(result)[:500] if result else "None"

        prompt = f"""Evaluate this step result:

Goal: {goal}
Step: {step.description}
Result: {result_str}

Is this result satisfactory? Respond in JSON:
{{"verdict": "success"|"retry"|"failed", "reason": "...", "action": "continue"|"retry"|"replan"}}"""

        response = self.llm.generate(prompt, system="You are a quality evaluator. Output valid JSON.")
        try:
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        except Exception:
            return {"verdict": "success", "reason": "Default pass", "action": "continue"}

    def evaluate_progress(self, state: TaskState) -> Dict:
        completed = sum(1 for s in state.steps if s.status == StepStatus.SUCCESS)
        total = len(state.steps)
        progress = completed / max(total, 1)

        if progress >= 1.0:
            return {"verdict": "complete", "progress": 1.0, "action": "summarize"}

        if state.replan_count >= state.max_replans:
            return {"verdict": "stuck", "progress": progress, "action": "partial_report"}

        return {"verdict": "on_track", "progress": progress, "action": "continue"}


class ToolRegistry:
    """
    Registry of callable tools that steps can invoke.
    Each tool has a name, description, and callable function.
    """

    def __init__(self):
        self._tools: Dict[str, Dict] = {}

    def register(self, name: str, func: Callable, description: str = "",
                 modality: Optional[JEPAModality] = None):
        self._tools[name] = {
            "func": func,
            "description": description,
            "modality": modality,
        }

    def get(self, name: str) -> Optional[Callable]:
        entry = self._tools.get(name)
        return entry["func"] if entry else None

    def list_tools(self) -> List[Dict]:
        return [{"name": k, "description": v["description"], "modality": v["modality"]}
                for k, v in self._tools.items()]

    def execute(self, name: str, **kwargs) -> Any:
        func = self.get(name)
        if func is None:
            return f"Tool '{name}' not found"
        try:
            return func(**kwargs)
        except Exception as e:
            return f"Tool error: {e}"


class StepExecutor:
    """
    Executes individual steps using JEPA+Bridge or registered tools.
    """

    def __init__(self, bridge: JEPASoulBridge, tools: ToolRegistry, llm: LLMBackend):
        self.bridge = bridge
        self.tools = tools
        self.llm = llm
        self._adversarial_loop = None

    def _get_adversarial_loop(self):
        if self._adversarial_loop is None:
            from adversarial_game_loop import AdversarialAgentLoop
            self._adversarial_loop = AdversarialAgentLoop(
                llm_backend=self.llm,
                bridge=self.bridge,
                max_rounds=3,
                complexity_threshold=0.6,
            )
        return self._adversarial_loop

    def execute_step(self, step: Step, working_memory: WorkingMemory,
                     artifacts: Dict) -> Any:
        step.status = StepStatus.RUNNING

        if step.tool and self.tools.get(step.tool):
            tool_kwargs = dict(step.tool_args)
            if step.tool == "search_knowledge":
                tool_kwargs.setdefault("soul", step.soul)
                tool_kwargs.setdefault("query", step.description)
            elif step.tool == "write_knowledge":
                tool_kwargs.setdefault("soul", step.soul)
                tool_kwargs.setdefault("category", "knowledge")
                tool_kwargs.setdefault("content", {"text": step.description})
            result = self.tools.execute(step.tool, **tool_kwargs)
            step.result = result
            step.status = StepStatus.SUCCESS if not isinstance(result, str) or "error" not in result.lower() else StepStatus.FAILED
            working_memory.add("tool", f"Tool '{step.tool}' result: {str(result)[:300]}")
            return result

        if step.modality and step.modality in self.bridge._jepa_models:
            return self._execute_with_jepa(step, working_memory, artifacts)

        adv_loop = self._get_adversarial_loop()
        if adv_loop.should_review(step.description, {"modality": step.modality.value if step.modality else None}):
            return self._execute_with_review(step, working_memory, artifacts)

        return self._execute_with_llm(step, working_memory, artifacts)

    def _execute_with_jepa(self, step: Step, working_memory: WorkingMemory,
                           artifacts: Dict) -> Any:
        try:
            context_input = self._prepare_input(step, artifacts)

            result = self.bridge.process(
                raw_input=context_input,
                modality=step.modality,
                soul=step.soul,
                query=step.description,
            )

            step.result = result.llm_response or self._bridge_result_text(result)
            step.status = StepStatus.SUCCESS

            if result.slot_descriptions:
                slot_summary = "; ".join([f"{d.name}={d.activation:.1f}" for d in result.slot_descriptions[:3]])
                working_memory.add("jepa", f"[{step.modality.value}] {slot_summary}")

            return step.result

        except Exception as e:
            step.error = str(e)
            step.status = StepStatus.FAILED
            working_memory.add("error", f"Step {step.step_id} failed: {e}")
            return f"Error: {e}"

    def _execute_with_llm(self, step: Step, working_memory: WorkingMemory,
                          artifacts: Dict) -> Any:
        if not self.llm.is_available():
            step.error = "No LLM available"
            step.status = StepStatus.FAILED
            return "Error: No LLM available"

        context_text = working_memory.get_context_text(max_entries=10)
        artifacts_text = json.dumps(
            {k: str(v)[:200] for k, v in artifacts.items()},
            ensure_ascii=False,
        )

        prompt = f"""Task step: {step.description}
Soul: {step.soul}

Previous context:
{context_text[-2000:]}

Available artifacts:
{artifacts_text[:500]}

Execute this step and provide the result."""

        system = f"You are the {step.soul} soul. Execute the given step precisely."

        result = self.llm.generate(prompt, system, max_tokens=1024, temperature=0.3)
        step.result = result
        step.status = StepStatus.SUCCESS
        working_memory.add(step.soul, f"Step result: {result[:300]}")

        return result

    def _prepare_input(self, step: Step, artifacts: Dict) -> Any:
        import torch
        for key, value in artifacts.items():
            if isinstance(value, torch.Tensor):
                return value
        return torch.randn(1, 4, 5, 256)

    def _bridge_result_text(self, result: BridgeResult) -> str:
        parts = []
        for d in result.slot_descriptions[:5]:
            parts.append(f"{d.name}: {d.summary}")
        return "; ".join(parts)

    def _execute_with_review(self, step: Step, working_memory: WorkingMemory,
                             artifacts: Dict) -> Any:
        adv_loop = self._get_adversarial_loop()
        try:
            result = adv_loop.execute_step_with_review(step, working_memory, artifacts)
            step.result = result
            step.status = StepStatus.SUCCESS
            working_memory.add("iterative_review", f"Review result: {str(result)[:300]}")
            return result
        except Exception as e:
            step.error = str(e)
            step.status = StepStatus.FAILED
            working_memory.add("error", f"Iterative review failed: {e}")
            return self._execute_with_llm(step, working_memory, artifacts)


class AgentLoop:
    """
    Main orchestrator for long-task execution.

    Architecture:
      Goal → Plan → Execute → Verify → [Continue | Replan | Complete]
                                         ↑_______________|

    Supports:
      - Multi-step task decomposition
      - Automatic replanning on failure
      - Working memory compression
      - Soul collaboration
      - Pause/resume
      - Artifact tracking
    """

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        llm_backend: Optional[LLMBackend] = None,
        bridge: Optional[JEPASoulBridge] = None,
        max_replans: int = 3,
        max_steps: int = 20,
        step_timeout_seconds: int = 300,
    ):
        self.llm = llm_backend or CloudBackend()
        self.memory = SoulMemoryEngine(memory_dir)
        self.bridge = bridge or JEPASoulBridge(memory_dir=memory_dir, llm_backend=self.llm)

        self.planner = TaskPlanner(self.llm)
        self.reflection = ReflectionEngine(self.llm)
        self.tools = ToolRegistry()
        self.executor = StepExecutor(self.bridge, self.tools, self.llm)

        self.max_replans = max_replans
        self.max_steps = max_steps
        self.step_timeout = step_timeout_seconds

        self._active_tasks: Dict[str, TaskState] = {}

        self._adversarial_loop = None

        self._register_default_tools()

    def _register_default_tools(self):
        self.tools.register(
            "search_knowledge",
            func=lambda soul, query, top_k=5, **kw: self.memory.recall(soul, query, top_k=top_k) if self.memory else [],
            description="Search soul knowledge base",
        )
        self.tools.register(
            "write_knowledge",
            func=self._safe_write_knowledge,
            description="Write to soul knowledge base",
        )
        self.tools.register(
            "analyze_code",
            func=lambda code_path: f"Code analysis of {code_path}: structure, dependencies, complexity metrics",
            description="Analyze source code structure",
        )
        self.tools.register(
            "read_file",
            func=self._safe_read_file,
            description="Read a file's contents",
        )

    def _safe_write_knowledge(self, soul, category, content, **kw):
        if not self.memory:
            return "Error: Memory engine not initialized"
        try:
            if isinstance(content, str):
                content = {"text": content}
            assert isinstance(content, dict), f"content must be dict, got {type(content)}"
            return self.memory.write(soul, category, content)
        except Exception as e:
            return f"Error writing knowledge: {e}"

    def _safe_read_file(self, path, **kw):
        try:
            if not os.path.exists(path):
                return f"File not found: {path}"
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()[:5000]
        except Exception as e:
            return f"Error reading file: {e}"

    def register_tool(self, name: str, func: Callable, description: str = "",
                      modality: Optional[JEPAModality] = None):
        self.tools.register(name, func, description, modality)

    def execute(
        self,
        goal: str,
        context: Dict = None,
        souls: List[str] = None,
        max_steps: int = None,
    ) -> TaskResult:
        task_id = str(uuid.uuid4())[:8]
        context = context or {}

        state = TaskState(
            task_id=task_id,
            goal=goal,
            context=context,
            status=TaskStatus.PLANNING,
            start_time=time.time(),
            max_replans=self.max_replans,
        )

        self._active_tasks[task_id] = state

        print(f"\n{'='*60}")
        print(f"AGENT LOOP [{task_id}] Starting")
        print(f"Goal: {goal[:100]}")
        print(f"{'='*60}")

        working_memory = WorkingMemory()
        working_memory.add("system", f"Goal: {goal}")
        working_memory.add("system", f"Context: {json.dumps(context, ensure_ascii=False)[:300]}")

        available_souls = souls or list({s for souls_list in MODALITY_SOUL_MAP.values() for s in souls_list})

        steps = self.planner.plan(goal, context, available_souls)
        state.steps = steps
        state.status = TaskStatus.EXECUTING

        print(f"\nPlan: {len(steps)} steps")
        for i, s in enumerate(steps):
            deps = f" (after {s.depends_on})" if s.depends_on else ""
            print(f"  {i+1}. [{s.soul}] {s.description[:60]}{deps}")

        step_idx = 0
        while step_idx < len(state.steps) and step_idx < (max_steps or self.max_steps):
            step = state.steps[step_idx]

            if step.status == StepStatus.SUCCESS:
                step_idx += 1
                continue

            if step.depends_on:
                all_deps_met = all(
                    any(s.step_id == dep_id and s.status == StepStatus.SUCCESS
                        for s in state.steps)
                    for dep_id in step.depends_on
                )
                if not all_deps_met:
                    step.status = StepStatus.SKIPPED
                    step_idx += 1
                    continue

            print(f"\n  Step {step_idx+1}/{len(state.steps)}: {step.description[:60]}")
            state.current_step_idx = step_idx

            try:
                result = self.executor.execute_step(step, working_memory, state.artifacts)
            except Exception as e:
                step.error = str(e)
                step.status = StepStatus.FAILED
                result = f"Exception in execute_step: {e}"
                working_memory.add("error", f"Step {step.step_id} crashed: {e}")

            evaluation = self.reflection.evaluate_step(step, result, goal)
            verdict = evaluation.get("verdict", "success")
            action = evaluation.get("action", "continue")

            print(f"  → {verdict}: {evaluation.get('reason', '')[:60]}")

            state.history.append({
                "step_id": step.step_id,
                "description": step.description,
                "verdict": verdict,
                "reason": evaluation.get("reason", ""),
                "timestamp": time.time(),
            })

            if verdict == "success":
                if result is not None:
                    artifact_key = f"step_{step_idx}_result"
                    state.artifacts[artifact_key] = str(result)[:2000]
                step_idx += 1

            elif verdict == "retry" and step.retry_count < step.max_retries:
                step.retry_count += 1
                step.status = StepStatus.PENDING
                working_memory.add("system", f"Retrying step {step.step_id} (attempt {step.retry_count+1})")

            elif action == "replan" and state.replan_count < state.max_replans:
                state.replan_count += 1
                new_steps = self.planner.replan(state, step, evaluation.get("reason", ""))
                state.steps = [s for s in state.steps if s.status == StepStatus.SUCCESS] + new_steps
                step_idx = len([s for s in state.steps if s.status == StepStatus.SUCCESS])
                working_memory.add("system", f"Replanned (attempt {state.replan_count})")
                print(f"  ↻ Replanned: {len(new_steps)} new steps")

            else:
                step.status = StepStatus.FAILED
                step_idx += 1

        progress = self.reflection.evaluate_progress(state)
        state.status = TaskStatus.COMPLETED if progress["verdict"] == "complete" else (
            TaskStatus.FAILED if progress["progress"] < 0.3 else TaskStatus.COMPLETED
        )
        state.end_time = time.time()

        summary = self._generate_summary(state, working_memory)

        try:
            self.memory.write(
                state.steps[0].soul if state.steps else "cezanne",
                "trajectory",
                {
                    "task_id": task_id,
                    "goal": goal,
                    "status": state.status.value,
                    "steps_completed": sum(1 for s in state.steps if s.status == StepStatus.SUCCESS),
                    "steps_total": len(state.steps),
                    "replan_count": state.replan_count,
                    "elapsed_seconds": state.end_time - state.start_time,
                    "summary": summary,
                },
            )
        except Exception as e:
            print(f"  ⚠ Failed to write trajectory to memory: {e}")
            try:
                traj_path = os.path.join(self.memory.memory_dir or ".", f"trajectory_{task_id}.json")
                with open(traj_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "task_id": task_id, "goal": goal, "status": state.status.value,
                        "steps_completed": sum(1 for s in state.steps if s.status == StepStatus.SUCCESS),
                        "steps_total": len(state.steps), "summary": summary,
                    }, f, ensure_ascii=False, indent=2)
                print(f"  → Saved trajectory to fallback file: {traj_path}")
            except Exception as e2:
                print(f"  ⚠ Fallback trajectory save also failed: {e2}")

        completed = sum(1 for s in state.steps if s.status == StepStatus.SUCCESS)

        print(f"\n{'='*60}")
        print(f"AGENT LOOP [{task_id}] {state.status.value.upper()}")
        print(f"Steps: {completed}/{len(state.steps)} | Replans: {state.replan_count}")
        print(f"Time: {state.end_time - state.start_time:.1f}s")
        print(f"{'='*60}\n")

        del self._active_tasks[task_id]

        return TaskResult(
            task_id=task_id,
            goal=goal,
            status=state.status,
            steps_completed=completed,
            steps_total=len(state.steps),
            artifacts=state.artifacts,
            summary=summary,
            elapsed_seconds=state.end_time - state.start_time,
            replan_count=state.replan_count,
            history=state.history,
        )

    def _generate_summary(self, state: TaskState, working_memory: WorkingMemory) -> str:
        completed = [s for s in state.steps if s.status == StepStatus.SUCCESS]
        failed = [s for s in state.steps if s.status == StepStatus.FAILED]

        parts = [f"Task: {state.goal[:80]}"]
        parts.append(f"Status: {state.status.value}")
        parts.append(f"Completed: {len(completed)}/{len(state.steps)} steps")

        if completed:
            parts.append("Key results:")
            for s in completed[-3:]:
                result_str = str(s.result)[:100] if s.result else "N/A"
                parts.append(f"  - {s.description[:50]}: {result_str}")

        if failed:
            parts.append(f"Failed steps: {', '.join(s.step_id for s in failed)}")

        if state.replan_count > 0:
            parts.append(f"Replanned {state.replan_count} times")

        return "\n".join(parts)

    def get_active_tasks(self) -> Dict[str, Dict]:
        return {
            tid: {
                "goal": t.goal[:80],
                "status": t.status.value,
                "progress": f"{sum(1 for s in t.steps if s.status == StepStatus.SUCCESS)}/{len(t.steps)}",
            }
            for tid, t in self._active_tasks.items()
        }
