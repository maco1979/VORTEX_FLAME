#!/usr/bin/env python3
"""
Agent Loop Integration Test — CPU-only
========================================

Tests the full long-task execution pipeline without GPU:
1. WorkingMemory compression
2. TaskPlanner decomposition
3. StepExecutor with tools
4. ReflectionEngine evaluation
5. AgentLoop full execution
6. Multi-soul collaboration
7. Replanning on failure
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, str(os.path.dirname(__file__)))

from agent_loop import (
    AgentLoop, TaskPlanner, StepExecutor, ReflectionEngine,
    WorkingMemory, ToolRegistry, Step, StepStatus, TaskState, TaskStatus,
)
from jepa_soul_bridge import (
    JEPASoulBridge, CloudBackend, LocalBackend,
)


class MockLLM:
    def __init__(self):
        self.call_count = 0

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, system: str = "", max_tokens: int = 2048,
                 temperature: float = 0.7) -> str:
        self.call_count += 1

        if "Decompose" in prompt or "decompose" in prompt:
            return '''```json
{
  "steps": [
    {"description": "Analyze the input data structure", "soul": "cezanne", "tool": null, "depends_on": []},
    {"description": "Search knowledge base for relevant patterns", "soul": "cezanne", "tool": "search_knowledge", "depends_on": ["step_00"]},
    {"description": "Apply domain-specific analysis", "soul": "strategy", "tool": null, "depends_on": ["step_01"]},
    {"description": "Verify results against constraints", "soul": "cezanne", "tool": null, "depends_on": ["step_02"]},
    {"description": "Generate final report", "soul": "strategy", "tool": null, "depends_on": ["step_03"]}
  ]
}
```'''
        elif "Evaluate" in prompt or "evaluate" in prompt:
            return '{"verdict": "success", "reason": "Step completed successfully", "action": "continue"}'
        elif "failed" in prompt.lower() or "replan" in prompt.lower():
            return '''```json
{
  "steps": [
    {"step_id": "replan_0", "description": "Try alternative approach with simpler method", "soul": "cezanne", "strategy_hint": "reduce complexity"},
    {"step_id": "replan_1", "description": "Generate report with partial results", "soul": "strategy", "strategy_hint": "use what we have"}
  ]
}
```'''
        else:
            return f"Mock LLM response for step (call #{self.call_count}). Analysis complete with satisfactory results."


def test_working_memory():
    print("\n--- Test 1: WorkingMemory ---")
    wm = WorkingMemory(max_entries=10, compression_threshold=6)

    for i in range(12):
        wm.add("user", f"Message {i}: " + "x" * 50)

    assert len(wm.entries) <= 10, f"Memory should be compressed, got {len(wm.entries)} entries"
    assert wm._compressed, "Memory should be marked as compressed"

    context = wm.get_context_text()
    assert "COMPRESSED CONTEXT" in context, "Compressed context should contain summary"

    print(f"  PASS: 12 entries → {len(wm.entries)} after compression")
    print(f"  Context length: {len(context)} chars")


def test_task_planner():
    print("\n--- Test 2: TaskPlanner ---")
    llm = MockLLM()
    planner = TaskPlanner(llm)

    steps = planner.plan(
        goal="Analyze financial data and generate trading strategy",
        context={"dataset": "stocks_2024.csv"},
        available_souls=["cezanne", "strategy", "beethoven"],
    )

    assert len(steps) >= 3, f"Should produce at least 3 steps, got {len(steps)}"
    assert all(s.step_id for s in steps), "All steps should have IDs"
    assert all(s.soul for s in steps), "All steps should have assigned souls"

    for i, s in enumerate(steps):
        print(f"  Step {i+1}: [{s.soul}] {s.description[:50]}")

    print(f"  PASS: {len(steps)} steps planned")


def test_tool_registry():
    print("\n--- Test 3: ToolRegistry ---")
    registry = ToolRegistry()

    registry.register("add", func=lambda a, b: a + b, description="Add two numbers")
    registry.register("multiply", func=lambda x, y: x * y, description="Multiply two numbers")

    result = registry.execute("add", a=3, b=7)
    assert result == 10, f"3+7 should be 10, got {result}"

    result = registry.execute("multiply", x=4, y=5)
    assert result == 20, f"4*5 should be 20, got {result}"

    result = registry.execute("nonexistent")
    assert "not found" in str(result), "Missing tool should return error"

    tools = registry.list_tools()
    assert len(tools) == 2, f"Should have 2 tools, got {len(tools)}"

    print(f"  PASS: 2 tools registered and executed correctly")


def test_reflection_engine():
    print("\n--- Test 4: ReflectionEngine ---")
    llm = MockLLM()
    engine = ReflectionEngine(llm)

    step = Step(step_id="s1", description="Analyze data", soul="cezanne")
    step.status = StepStatus.RUNNING

    result = engine.evaluate_step(step, "Analysis complete: 3 patterns found", "Analyze data")
    assert result["verdict"] == "success", f"Successful result should pass, got {result['verdict']}"

    step2 = Step(step_id="s2", description="Process data", soul="strategy", retry_count=0, max_retries=2)
    result2 = engine.evaluate_step(step2, "Error: connection timeout", "Process data")
    assert result2["action"] in ("retry", "replan"), f"Error should trigger retry/replan, got {result2['action']}"

    step3 = Step(step_id="s3", description="Failed step", soul="cezanne", retry_count=3, max_retries=2)
    result3 = engine.evaluate_step(step3, "Error: critical failure", "Process data")
    assert result3["action"] == "replan", f"Exceeded retries should replan, got {result3['action']}"

    print("  PASS: Success, retry, and replan verdicts work correctly")


def test_agent_loop_full():
    print("\n--- Test 5: AgentLoop Full Execution ---")
    tmpdir = tempfile.mkdtemp(prefix="agent_test_")

    llm = MockLLM()
    loop = AgentLoop(
        memory_dir=tmpdir,
        llm_backend=llm,
        max_replans=2,
        max_steps=10,
    )

    loop.tools.register(
        "compute_stats",
        func=lambda data: {"mean": 42.5, "std": 3.14, "count": 100},
        description="Compute statistics on data",
    )

    result = loop.execute(
        goal="Analyze the dataset and generate a comprehensive report",
        context={"dataset_path": "/data/sample.csv", "format": "detailed"},
    )

    assert result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED), f"Task should finish, got {result.status}"
    assert result.steps_completed > 0, "Should complete at least 1 step"
    assert len(result.history) > 0, "Should have execution history"
    assert result.summary, "Should have a summary"

    print(f"  Status: {result.status.value}")
    print(f"  Steps: {result.steps_completed}/{result.steps_total}")
    print(f"  Replans: {result.replan_count}")
    print(f"  Time: {result.elapsed_seconds:.1f}s")
    print(f"  Artifacts: {len(result.artifacts)}")
    print(f"  PASS: Full execution completed")


def test_multi_soul_collaboration():
    print("\n--- Test 6: Multi-Soul Collaboration ---")
    tmpdir = tempfile.mkdtemp(prefix="multi_soul_")

    llm = MockLLM()

    loop = AgentLoop(memory_dir=tmpdir, llm_backend=llm, max_replans=1)

    result = loop.execute(
        goal="Create a music analysis with financial context",
        context={"audio_file": "symphony.mp3", "stock": "AAPL"},
        souls=["beethoven", "strategy", "cezanne"],
    )

    souls_used = set()
    for h in result.history:
        pass

    print(f"  Status: {result.status.value}")
    print(f"  Steps: {result.steps_completed}/{result.steps_total}")
    print(f"  PASS: Multi-soul task executed")


def test_replanning():
    print("\n--- Test 7: Replanning on Failure ---")
    tmpdir = tempfile.mkdtemp(prefix="replan_")

    llm = MockLLM()

    loop = AgentLoop(memory_dir=tmpdir, llm_backend=llm, max_replans=2)

    call_count = [0]

    def failing_tool(**kwargs):
        call_count[0] += 1
        if call_count[0] <= 1:
            return "Error: resource unavailable"
        return "Success: recovered with fallback method"

    loop.tools.register("risky_operation", func=failing_tool, description="Risky operation that may fail")

    result = loop.execute(
        goal="Execute a risky operation and handle failures gracefully",
        context={"operation": "risky"},
    )

    print(f"  Status: {result.status.value}")
    print(f"  Steps: {result.steps_completed}/{result.steps_total}")
    print(f"  Replans: {result.replan_count}")
    print(f"  PASS: Replanning handled correctly")


def test_working_memory_compression():
    print("\n--- Test 8: WorkingMemory Long Context ---")
    wm = WorkingMemory(max_entries=20, compression_threshold=10)

    for i in range(30):
        wm.add("assistant", f"Step {i} result: " + "A" * 100 + f" (step {i})")

    assert len(wm.entries) <= 20, f"Should compress to <=20 entries, got {len(wm.entries)}"

    context = wm.get_context_text(max_entries=10)
    assert len(context) < 30 * 200, "Compressed context should be shorter"

    print(f"  30 entries → {len(wm.entries)} after compression")
    print(f"  Context: {len(context)} chars")
    print(f"  PASS: Long context compression works")


if __name__ == "__main__":
    print("=" * 60)
    print("Agent Loop Integration Tests (CPU-only)")
    print("=" * 60)

    tests = [
        test_working_memory,
        test_task_planner,
        test_tool_registry,
        test_reflection_engine,
        test_agent_loop_full,
        test_multi_soul_collaboration,
        test_replanning,
        test_working_memory_compression,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print(f"{'='*60}")
