"""
VORTEX Code — Multi-Expert Code Intelligence CLI
==================================================
Terminal-native multi-expert AI assistant.

Usage:
  vortex ask "explain this function"          # Ask any question
  vortex review src/engine.py                 # Multi-expert code review
  vortex fix src/engine.py --issue L42        # Ralph iterative fix
  vortex plan "build a REST API"              # Ultrapilot parallel design
  vortex memory show                          # View soul memories
  vortex memory search "VaR calculation"      # Search memories
  vortex status                               # System status
  vortex souls                                # List available souls

Design Philosophy:
  Claude Code:  terminal → black box → answer
  VORTEX Code:  terminal → visible routing → multi-expert → arbitration → answer

  Each expert's output is color-coded and labeled.
  Routing confidence and arbitration strategy are always visible.
"""

import argparse
import json
import os
import sys
import time

SOUL_COLORS = {
    "cezanne": "\033[94m",
    "einstein": "\033[95m",
    "galileo": "\033[96m",
    "darwin": "\033[92m",
    "strategy": "\033[93m",
    "montesquieu": "\033[91m",
    "davinci": "\033[97m",
    "humboldt": "\033[33m",
    "yuanlongping": "\033[32m",
    "guizhu": "\033[35m",
    "herodotus": "\033[36m",
    "monet": "\033[95m",
    "vangogh": "\033[93m",
    "beethoven": "\033[96m",
}

SOUL_ICONS = {
    "cezanne": "\U0001f3a8",
    "einstein": "\u269b\ufe0f",
    "galileo": "\U0001f52d",
    "darwin": "\U0001fab2",
    "strategy": "\U0001f3af",
    "montesquieu": "\u2696\ufe0f",
    "davinci": "\U0001f527",
    "humboldt": "\U0001f30d",
    "yuanlongping": "\U0001f33e",
    "guizhu": "\U0001f54a\ufe0f",
    "herodotus": "\U0001f4d6",
    "monet": "\U0001f5bc\ufe0f",
    "vangogh": "\u2728",
    "beethoven": "\U0001f3b5",
}

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"


def _color_soul(soul: str) -> str:
    color = SOUL_COLORS.get(soul, "")
    icon = SOUL_ICONS.get(soul, "\u2753")
    return f"{color}{icon} {soul.title()}{RESET}"


def _print_routing(routing_result: dict):
    candidates = routing_result.get("candidates", [])
    if not candidates:
        print(f"  {YELLOW}\u26a0 No strong routing match, using default{RESET}")
        return

    print(f"\n  {BOLD}\U0001f500 Routing:{RESET}")
    for c in candidates[:4]:
        soul = c.get("soul", "?")
        conf = c.get("confidence", 0.0)
        bar_len = int(conf * 20)
        bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
        selected = " \u2190 selected" if c.get("selected") else ""
        color = SOUL_COLORS.get(soul, "")
        print(f"    {color}{soul.title():12s}{RESET} {bar} {conf:.2f}{selected}")
    print()


def _print_expert_output(soul: str, output: str):
    color = SOUL_COLORS.get(soul, "")
    icon = SOUL_ICONS.get(soul, "\u2753")
    print(f"\n  {color}{icon} {soul.title()}:{RESET}")
    for line in output.split("\n"):
        print(f"    {color}\u2502{RESET} {line}")
    print()


def _print_arbitration(arb: dict):
    method = arb.get("method", "unknown")
    provenance = arb.get("provenance", {})
    reason = provenance.get("reason", "")
    winner = arb.get("winner", {})
    winner_soul = winner.get("soul", "?") if isinstance(winner, dict) else "?"

    print(f"\n  {BOLD}\u2696\ufe0f Arbitration:{RESET} {method} | {reason}")
    if winner_soul != "?":
        color = SOUL_COLORS.get(winner_soul, "")
        print(f"    Winner: {color}{winner_soul.title()}{RESET}")
    print()


def _print_memory_hits(hits: list):
    if not hits:
        return
    print(f"  {DIM}\U0001f4d6 Memory hits:{RESET}")
    for h in hits[:3]:
        topic = h.get("content", {}).get("topic", "") if isinstance(h, dict) else str(h)
        if topic:
            print(f"    {DIM}\u2022 {topic[:60]}{RESET}")
    print()


def cmd_ask(args):
    from soul_orchestrator import route_to_soul, soft_route_to_souls
    from soul_memory import recall
    from ollama_adapter import get_adapter

    query = args.query
    if not query:
        print(f"{RED}Error: Please provide a query. Example: vortex ask \"explain this function\"{RESET}")
        return

    adapter = get_adapter()

    if not adapter.is_available():
        print(f"{RED}Error: Ollama is not running at {adapter.base_url}{RESET}")
        print(f"  Start Ollama first: {CYAN}ollama serve{RESET}")
        print(f"  Then pull a model:  {CYAN}ollama pull qwen2.5:7b{RESET}")
        return

    routing = soft_route_to_souls(query, top_k=2)
    candidates = routing.get("candidates", [])

    if not candidates:
        candidates = [{"soul": "cezanne", "confidence": 0.5, "selected": True}]

    primary = candidates[0]["soul"]
    _print_routing(routing)

    memory_ctx = recall(primary, query, top_k=3, categories=["knowledge", "domain_memory", "conversation"])
    memory_snippets = []
    for m in memory_ctx[:3]:
        content = m.get("content", {})
        if isinstance(content, dict):
            memory_snippets.append(content.get("topic", ""))
        else:
            memory_snippets.append(str(content)[:80])

    _print_memory_hits(memory_ctx)

    print(f"  {DIM}\u23f3 Thinking...{RESET}")

    if len(candidates) >= 2 and candidates[0].get("confidence", 0) - candidates[1].get("confidence", 0) < 0.15:
        soul_a = candidates[0]["soul"]
        soul_b = candidates[1]["soul"]
        mem_a = [m.get("content", {}).get("topic", "") for m in recall(soul_a, query, top_k=2) if isinstance(m.get("content", {}), dict)]
        mem_b = [m.get("content", {}).get("topic", "") for m in recall(soul_b, query, top_k=2) if isinstance(m.get("content", {}), dict)]

        result_a = adapter.generate(soul_a, query, memory_snippets=mem_a)
        result_b = adapter.generate(soul_b, query, memory_snippets=mem_b)

        if result_a["status"] == "ok":
            _print_expert_output(soul_a, result_a["output"])
        else:
            print(f"  {RED}{soul_a.title()} error: {result_a.get('error', 'unknown')}{RESET}")

        if result_b["status"] == "ok":
            _print_expert_output(soul_b, result_b["output"])
        else:
            print(f"  {RED}{soul_b.title()} error: {result_b.get('error', 'unknown')}{RESET}")

        print(f"  {DIM}\u2696\ufe0f Dual-expert mode (gap < 0.15){RESET}")
    else:
        result = adapter.generate(primary, query, memory_snippets=memory_snippets)
        if result["status"] == "ok":
            _print_expert_output(primary, result["output"])
        else:
            print(f"  {RED}Error: {result.get('error', 'unknown')}{RESET}")

    elapsed = result.get("elapsed", 0) if 'result' in dir() else 0
    tps = result.get("tokens_per_second", 0) if 'result' in dir() and result.get("tokens_per_second") else 0
    print(f"  {DIM}\u23f1 {elapsed:.1f}s | {tps:.1f} tok/s | {adapter.model}{RESET}")


def cmd_review(args):
    filepath = args.file
    if not filepath:
        print(f"{RED}Error: Please provide a file path. Example: vortex review src/engine.py{RESET}")
        return

    if not os.path.exists(filepath):
        print(f"{RED}Error: File not found: {filepath}{RESET}")
        return

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        code = f.read()

    if len(code) > 8000:
        code = code[:8000] + f"\n... [truncated, showing first 8000 of {len(code)} chars]"

    query = f"Review this code for bugs, edge cases, and improvements:\n\n```\n{code}\n```"
    args.query = query
    cmd_ask(args)


def cmd_fix(args):
    from ollama_adapter import get_adapter
    from soul_memory import recall

    filepath = args.file
    if not filepath or not os.path.exists(filepath):
        print(f"{RED}Error: Please provide a valid file path.{RESET}")
        return

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        code = f.read()

    issue = args.issue or "general quality"
    adapter = get_adapter()

    if not adapter.is_available():
        print(f"{RED}Error: Ollama is not running.{RESET}")
        return

    print(f"  {BOLD}\U0001f527 Ralph Mode: Iterative Fix{RESET}")
    print(f"  Target: {filepath}")
    print(f"  Issue: {issue}")
    print()

    max_iterations = 3
    current_code = code

    for i in range(max_iterations):
        print(f"  {CYAN}--- Iteration {i+1}/{max_iterations} ---{RESET}")

        prompt = (
            f"Fix the following issue in this code: {issue}\n\n"
            f"Current code:\n```\n{current_code[:4000]}\n```\n\n"
            f"Provide the COMPLETE fixed code. Do not use placeholders."
        )

        result = adapter.generate("cezanne", prompt)

        if result["status"] != "ok":
            print(f"  {RED}Error: {result.get('error')}{RESET}")
            break

        _print_expert_output("cezanne", result["output"])

        verify_prompt = (
            f"Verify this code fix is correct. The issue was: {issue}\n\n"
            f"Fixed code:\n```\n{result['output'][:3000]}\n```\n\n"
            f"Respond with PASS or FAIL and explain why."
        )

        verify = adapter.generate("einstein", verify_prompt)
        if verify["status"] == "ok":
            verdict = verify["output"][:200]
            if "PASS" in verdict.upper():
                print(f"  {GREEN}\u2705 Verification PASSED{RESET}")
                print(f"    {DIM}{verdict}{RESET}")
                break
            else:
                print(f"  {YELLOW}\u26a0 Verification needs improvement{RESET}")
                print(f"    {DIM}{verdict}{RESET}")
                current_code = result["output"]
        else:
            print(f"  {RED}Verify error: {verify.get('error')}{RESET}")
            break

    print(f"\n  {DIM}Ralph mode complete after {i+1} iteration(s){RESET}")


def cmd_plan(args):
    from soul_orchestrator import soft_route_to_souls
    from soul_memory import recall
    from ollama_adapter import get_adapter

    task = args.task
    if not task:
        print(f"{RED}Error: Please provide a task description.{RESET}")
        return

    adapter = get_adapter()
    if not adapter.is_available():
        print(f"{RED}Error: Ollama is not running.{RESET}")
        return

    print(f"  {BOLD}\U0001f680 Ultrapilot Mode: Parallel Planning{RESET}")
    print(f"  Task: {task}\n")

    routing = soft_route_to_souls(task, top_k=3)
    candidates = routing.get("candidates", [])

    if len(candidates) < 2:
        candidates = [
            {"soul": "cezanne", "confidence": 0.7},
            {"soul": "einstein", "confidence": 0.5},
        ]

    _print_routing(routing)

    results = []
    for c in candidates[:3]:
        soul = c["soul"]
        mem = [m.get("content", {}).get("topic", "") for m in recall(soul, task, top_k=2) if isinstance(m.get("content", {}), dict)]

        prompt = f"Design a plan for: {task}\n\nProvide a structured, step-by-step plan with key decisions and tradeoffs."
        result = adapter.generate(soul, prompt, memory_snippets=mem)

        if result["status"] == "ok":
            _print_expert_output(soul, result["output"])
            results.append({"soul": soul, "output": result["output"], "confidence": c["confidence"]})

    if len(results) >= 2:
        print(f"  {BOLD}\u2696\ufe0f Synthesis: Combine the above plans into a unified approach.{RESET}")


def cmd_status(args):
    from ollama_adapter import get_adapter, _model_router
    from soul_memory import _engine

    print(f"\n  {BOLD}VORTEX Code Status{RESET}")
    print(f"  {'='*40}")

    adapter = get_adapter()
    if adapter.is_available():
        models = adapter.list_models()
        info = _model_router.model_info()
        hermes_flag = f"{GREEN}(Hermes){RESET}" if info.get("hermes_available") else f"{YELLOW}(no Hermes){RESET}"
        print(f"  {GREEN}\u2705 Ollama: Connected{RESET} ({adapter.base_url})")
        print(f"    Model: {adapter.model} {hermes_flag}")
        print(f"    Available: {', '.join(models[:5])}")
    else:
        print(f"  {RED}\u274c Ollama: Not connected{RESET} ({adapter.base_url})")
        print(f"    Start with: ollama serve")
        print(f"    Pull Hermes: {CYAN}ollama pull hermes3:8b{RESET}")

    print()
    from soul_orchestrator import SOUL_CAPABILITIES
    all_souls = list(SOUL_CAPABILITIES.keys())

    print(f"  {BOLD}Souls:{RESET}")
    for s in all_souls:
        color = SOUL_COLORS.get(s, "")
        icon = SOUL_ICONS.get(s, "")
        cap = SOUL_CAPABILITIES.get(s, {})
        tier = cap.get("tier", "?")
        boundary = cap.get("boundary", {})
        available = boundary.get("可用", boundary.get("available", []))
        mcp_tools = cap.get("mcp_tools", [])
        try:
            count = _engine._get_db(s).execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
            mem_str = f" | {count} memories"
        except Exception:
            mem_str = ""
        tool_str = f" | MCP:{len(mcp_tools)}" if mcp_tools else ""
        print(f"    {color}{icon} {s.title():12s}{RESET} \u2705 Tier-{tier}{tool_str}{mem_str}")
        if available:
            avail_str = ", ".join(available[:4]) if isinstance(available, list) else str(available)[:60]
            print(f"      {DIM}可用: {avail_str}{RESET}")

    print()
    print(f"  {BOLD}Execution Modes:{RESET}")
    print(f"    Team       \u2705 Sequential pipeline (5 stages)")
    print(f"    Ultrapilot \u2705 Parallel multi-expert (3 stages)")
    print(f"    Ralph      \u2705 Iterative verify-fix loop")

    print()


def cmd_souls(args):
    from soul_orchestrator import SOUL_CAPABILITIES

    print(f"\n  {BOLD}VORTEX FLAME — 14 Souls{RESET}")
    print(f"  {'='*50}\n")

    for soul, cap in SOUL_CAPABILITIES.items():
        color = SOUL_COLORS.get(soul, "")
        icon = SOUL_ICONS.get(soul, "")
        tier = cap.get("tier", "?")
        domain = ", ".join(cap.get("domain", []))
        boundary = cap.get("boundary", {})
        available = boundary.get("可用", boundary.get("available", []))
        mcp_tools = cap.get("mcp_tools", [])
        status = f"{GREEN}\u2705 Active{RESET}"

        print(f"  {color}{icon} {soul.title()}{RESET} [Tier {tier}]")
        print(f"    Domain: {domain}")
        if mcp_tools:
            print(f"    MCP Tools: {', '.join(mcp_tools)}")
        if available:
            avail_str = ", ".join(available) if isinstance(available, list) else str(available)[:80]
            print(f"    Available: {avail_str}")
        print(f"    Status: {status}")
        print()


def cmd_memory(args):
    from soul_memory import recall, search

    subcmd = args.memory_cmd or "show"

    if subcmd == "show":
        soul = args.soul or "cezanne"
        memories = recall(soul, "", top_k=10)
        if not memories:
            print(f"  {DIM}No memories found for {soul}{RESET}")
            return

        print(f"\n  {BOLD}Recent memories for {soul.title()}:{RESET}\n")
        for m in memories[:10]:
            content = m.get("content", {})
            if isinstance(content, dict):
                topic = content.get("topic", "")
                cat = m.get("category", "")
                imp = m.get("importance", 0)
                print(f"    {CYAN}[{cat}]{RESET} {topic[:50]} (importance: {imp})")
            else:
                print(f"    {str(content)[:60]}")
        print()

    elif subcmd == "search":
        query = args.query or ""
        soul = args.soul or "cezanne"
        if not query:
            print(f"{RED}Error: Please provide a search query.{RESET}")
            return

        results = search(soul, "knowledge", query, top_k=5)
        if not results:
            results = search(soul, "domain_memory", query, top_k=5)

        print(f"\n  {BOLD}Search results for \"{query}\" ({soul.title()}):{RESET}\n")
        for r in results[:5]:
            content = r.get("content", {})
            if isinstance(content, dict):
                topic = content.get("topic", "")
                summary = content.get("summary", "")[:80]
                print(f"    {CYAN}\u2022{RESET} {topic}")
                if summary:
                    print(f"      {DIM}{summary}{RESET}")
        print()


def cmd_skill(args):
    from skill_evolver import SkillEvolver
    from soul_memory import recall
    from ollama_adapter import get_adapter, _model_router

    subcmd = args.skill_cmd or "list"

    if subcmd == "list":
        try:
            evolver = SkillEvolver()
            skills = evolver.list_skills()
            if not skills:
                print(f"  {DIM}No evolved skills yet.{RESET}")
                print(f"  {CYAN}Use 'vortex skill evolve \"task description\"' to auto-generate skills.{RESET}")
                return
            print(f"\n  {BOLD}Evolved Skills:{RESET}\n")
            for s in skills[:10]:
                name = s.get("name", "?")
                status = s.get("status", "?")
                source = s.get("source", "?")
                print(f"    {GREEN}\u2022{RESET} {name} [{status}] (source: {source})")
            print()
        except Exception as e:
            print(f"  {DIM}Skill list unavailable: {e}{RESET}")

    elif subcmd == "evolve":
        task = args.task or ""
        if not task:
            print(f"{RED}Error: Provide a task description for skill evolution.{RESET}")
            print(f"  Example: vortex skill evolve \"deploy to kubernetes\"")
            return

        adapter = get_adapter()
        if not adapter.is_available():
            print(f"{RED}Error: Ollama not running.{RESET}")
            return

        print(f"  {BOLD}\U0001f9ec Skill Evolution{RESET}")
        print(f"  Task: {task}\n")

        prompt = (
            f"Based on this task, generate a reusable skill script that could automate similar tasks in the future.\n"
            f"Task: {task}\n\n"
            f"Output a JSON object with:\n"
            f'- "name": short skill name (snake_case)\n'
            f'- "description": what it does\n'
            f'- "steps": array of step descriptions\n'
            f'- "tools_needed": array of tool names\n'
            f'- "trigger_patterns": array of regex patterns that would trigger this skill\n\n'
            f"Respond ONLY with valid JSON, no other text."
        )

        result = adapter.generate("cezanne", prompt)
        if result["status"] == "ok":
            try:
                skill_data = json.loads(result["output"])
                name = skill_data.get("name", "unnamed_skill")
                desc = skill_data.get("description", "")
                steps = skill_data.get("steps", [])
                triggers = skill_data.get("trigger_patterns", [])

                print(f"  {GREEN}\u2705 Skill Generated:{RESET} {name}")
                print(f"    {desc}")
                if steps:
                    print(f"    Steps:")
                    for i, step in enumerate(steps[:5], 1):
                        print(f"      {i}. {step}")
                if triggers:
                    print(f"    Triggers: {', '.join(triggers[:3])}")

                try:
                    evolver = SkillEvolver()
                    evolver.register_evolved_skill(name, skill_data)
                    print(f"  {GREEN}\u2705 Skill registered in evolution system{RESET}")
                except Exception as e:
                    print(f"  {YELLOW}\u26a0 Registration failed: {e}{RESET}")
            except json.JSONDecodeError:
                print(f"  {YELLOW}\u26a0 LLM output not valid JSON, raw output:{RESET}")
                print(f"    {result['output'][:200]}")
        else:
            print(f"  {RED}Error: {result.get('error')}{RESET}")

    elif subcmd == "suggest":
        adapter = get_adapter()
        if not adapter.is_available():
            print(f"{RED}Error: Ollama not running.{RESET}")
            return

        recent_tasks = []
        try:
            for soul in ["cezanne", "einstein", "davinci"]:
                mems = recall(soul, "", top_k=5, categories=["trajectory"])
                for m in mems[:3]:
                    content = m.get("content", {})
                    if isinstance(content, dict) and content.get("task"):
                        recent_tasks.append(content["task"])
        except Exception:
            pass

        if not recent_tasks:
            print(f"  {DIM}No recent task history for skill suggestions.{RESET}")
            return

        prompt = (
            f"Based on these recent tasks, suggest 3 skills that could be auto-generated to automate recurring work:\n\n"
            + "\n".join(f"- {t}" for t in recent_tasks[:10])
            + "\n\nFor each skill, provide: name, description, and trigger pattern. Be concise."
        )

        result = adapter.generate("cezanne", prompt)
        if result["status"] == "ok":
            print(f"\n  {BOLD}\U0001f4a1 Skill Suggestions:{RESET}\n")
            for line in result["output"].split("\n"):
                if line.strip():
                    print(f"    {line}")
            print()
        else:
            print(f"  {RED}Error: {result.get('error')}{RESET}")


def cmd_hermes(args):
    from ollama_adapter import get_adapter, _model_router

    print(f"\n  {BOLD}HERMES Integration Status{RESET}")
    print(f"  {'='*40}\n")

    adapter = get_adapter()
    info = _model_router.model_info()

    if adapter.is_available():
        print(f"  {GREEN}\u2705 Ollama: Connected{RESET}")

        if info.get("hermes_available"):
            print(f"  {GREEN}\u2705 Hermes Model: Available{RESET}")
            print(f"    Selected: {_model_router.best_model()}")
            for m in info.get("available_models", []):
                if "hermes" in m.lower():
                    print(f"    {GREEN}\u2022{RESET} {m}")
        else:
            print(f"  {YELLOW}\u26a0 Hermes Model: Not installed{RESET}")
            print(f"    Install with: {CYAN}ollama pull hermes3:8b{RESET}")
            print(f"    Or lighter:   {CYAN}ollama pull hermes3:3b{RESET}")

        print(f"\n  {BOLD}Available Models:{RESET}")
        for m in info.get("available_models", []):
            is_best = m == info.get("selected_model")
            marker = f"{GREEN}\u2190 active{RESET}" if is_best else ""
            print(f"    \u2022 {m} {marker}")

        print(f"\n  {BOLD}Soul → Model Mapping:{RESET}")
        for soul in ["cezanne", "einstein", "strategy", "davinci", "guizhu"]:
            best = _model_router.best_model(soul)
            color = SOUL_COLORS.get(soul, "")
            icon = SOUL_ICONS.get(soul, "")
            print(f"    {color}{icon} {soul.title():12s}{RESET} → {best}")
    else:
        print(f"  {RED}\u274c Ollama: Not connected{RESET}")
        print(f"    Start: {CYAN}ollama serve{RESET}")
        print(f"    Then:  {CYAN}ollama pull hermes3:8b{RESET}")

    print(f"\n  {BOLD}VORTEX FLAME vs Vanilla Hermes:{RESET}")
    print(f"    Hermes:   Single model → Single agent → No routing")
    print(f"    VORTEX:   14 Souls → Semantic routing → Multi-expert arbitration")
    print(f"    Edge:     {GREEN}Per-soul model selection + Memory + Skill evolution{RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="vortex",
        description="VORTEX Code — Multi-Expert Code Intelligence CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_ask = subparsers.add_parser("ask", help="Ask a question to multi-expert system")
    p_ask.add_argument("query", nargs="?", default="", help="Your question")
    p_ask.add_argument("--soul", default=None, help="Force specific soul (e.g., cezanne, einstein)")

    p_review = subparsers.add_parser("review", help="Multi-expert code review")
    p_review.add_argument("file", nargs="?", default="", help="File to review")
    p_review.add_argument("--soul", default=None, help="Force specific soul")

    p_fix = subparsers.add_parser("fix", help="Ralph iterative fix mode")
    p_fix.add_argument("file", nargs="?", default="", help="File to fix")
    p_fix.add_argument("--issue", default=None, help="Specific issue to fix")

    p_plan = subparsers.add_parser("plan", help="Ultrapilot parallel planning")
    p_plan.add_argument("task", nargs="?", default="", help="Task description")

    p_status = subparsers.add_parser("status", help="Show system status")

    p_souls = subparsers.add_parser("souls", help="List all souls")

    p_mem = subparsers.add_parser("memory", help="Memory management")
    p_mem.add_argument("memory_cmd", nargs="?", default="show", choices=["show", "search"])
    p_mem.add_argument("--soul", default="cezanne", help="Soul name")
    p_mem.add_argument("--query", default=None, help="Search query")

    p_skill = subparsers.add_parser("skill", help="Skill evolution (Hermes-style self-evolution)")
    p_skill.add_argument("skill_cmd", nargs="?", default="list", choices=["list", "evolve", "suggest"])
    p_skill.add_argument("--task", default=None, help="Task description for skill evolution")

    p_hermes = subparsers.add_parser("hermes", help="HERMES integration status")

    args = parser.parse_args()

    if args.command == "ask":
        cmd_ask(args)
    elif args.command == "review":
        cmd_review(args)
    elif args.command == "fix":
        cmd_fix(args)
    elif args.command == "plan":
        cmd_plan(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "souls":
        cmd_souls(args)
    elif args.command == "memory":
        cmd_memory(args)
    elif args.command == "skill":
        cmd_skill(args)
    elif args.command == "hermes":
        cmd_hermes(args)
    else:
        parser.print_help()
        print(f"\n  {CYAN}Quick start:{RESET}")
        print(f"    vortex status          # Check system status")
        print(f"    vortex hermes          # HERMES integration status")
        print(f"    vortex ask \"question\"  # Ask multi-expert system")
        print(f"    vortex review file.py  # Code review")
        print(f"    vortex fix file.py     # Iterative fix")
        print(f"    vortex plan \"task\"     # Parallel planning")
        print(f"    vortex skill list      # View evolved skills")
        print(f"    vortex skill evolve \"task\" # Auto-generate skill")


if __name__ == "__main__":
    main()
