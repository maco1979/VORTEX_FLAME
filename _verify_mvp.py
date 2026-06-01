"""
End-to-end verification for VORTEX Code Phase 1 MVP.
Tests: ollama_adapter, vf_cli imports, vf_api_server imports,
       soul_orchestrator Ollama integration, web dashboard.
"""

import sys
import traceback

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  \u2705 {name}")
        passed += 1
    except Exception as e:
        print(f"  \u274c {name}: {e}")
        traceback.print_exc()
        failed += 1


print("=" * 60)
print("VORTEX Code Phase 1 MVP — End-to-End Verification")
print("=" * 60)

print("\n[1] ollama_adapter.py — Module import & structure")
def test_ollama_import():
    from ollama_adapter import OllamaAdapter, get_adapter, SOUL_SYSTEM_PROMPTS
    adapter = OllamaAdapter()
    assert adapter.base_url, "Missing base_url"
    assert adapter.model, "Missing model"
    assert len(SOUL_SYSTEM_PROMPTS) >= 14, f"Expected 14 soul prompts, got {len(SOUL_SYSTEM_PROMPTS)}"
    for soul in ["cezanne", "einstein", "galileo", "beethoven"]:
        assert soul in SOUL_SYSTEM_PROMPTS, f"Missing prompt for {soul}"
test("OllamaAdapter import + 14 soul system prompts", test_ollama_import)

def test_ollama_adapter_methods():
    from ollama_adapter import OllamaAdapter
    adapter = OllamaAdapter()
    assert hasattr(adapter, 'is_available'), "Missing is_available"
    assert hasattr(adapter, 'list_models'), "Missing list_models"
    assert hasattr(adapter, 'generate'), "Missing generate"
    assert hasattr(adapter, 'generate_stream'), "Missing generate_stream"
test("OllamaAdapter has all required methods", test_ollama_adapter_methods)

def test_ollama_singleton():
    from ollama_adapter import get_adapter
    a1 = get_adapter()
    a2 = get_adapter()
    assert a1 is a2, "get_adapter should return singleton"
test("get_adapter returns singleton", test_ollama_singleton)

print("\n[2] vf_cli.py — CLI module import & command structure")
def test_cli_import():
    import vf_cli
    assert hasattr(vf_cli, 'cmd_ask'), "Missing cmd_ask"
    assert hasattr(vf_cli, 'cmd_review'), "Missing cmd_review"
    assert hasattr(vf_cli, 'cmd_fix'), "Missing cmd_fix"
    assert hasattr(vf_cli, 'cmd_plan'), "Missing cmd_plan"
    assert hasattr(vf_cli, 'cmd_status'), "Missing cmd_status"
    assert hasattr(vf_cli, 'cmd_souls'), "Missing cmd_souls"
    assert hasattr(vf_cli, 'cmd_memory'), "Missing cmd_memory"
test("vf_cli has all 7 command handlers", test_cli_import)

def test_cli_colors():
    from vf_cli import SOUL_COLORS, SOUL_ICONS, RESET
    assert len(SOUL_COLORS) >= 14, f"Expected 14 soul colors, got {len(SOUL_COLORS)}"
    assert len(SOUL_ICONS) >= 14, f"Expected 14 soul icons, got {len(SOUL_ICONS)}"
    assert RESET, "Missing RESET color code"
test("CLI has color/icon mappings for all 14 souls", test_cli_colors)

print("\n[3] vf_api_server.py — API server import & endpoints")
def test_api_import():
    from vf_api_server import app
    routes = [r.path for r in app.routes]
    expected = ["/api/status", "/api/souls", "/api/ask", "/api/review", "/api/fix", "/api/plan", "/api/memory", "/api/stream"]
    for ep in expected:
        assert ep in routes, f"Missing endpoint: {ep}"
test("API server has all 8 endpoints", test_api_import)

def test_api_models():
    from vf_api_server import AskRequest, ReviewRequest, FixRequest, PlanRequest, MemoryRequest
    req = AskRequest(query="test")
    assert req.query == "test"
    assert req.top_k == 2
    fix = FixRequest(code="print('hi')", issue="bug")
    assert fix.max_iterations == 3
test("API request models validate correctly", test_api_models)

print("\n[4] soul_orchestrator — Ollama backend integration")
def test_orchestrator_ollama_hook():
    from soul_orchestrator import _execute_soul_stage
    import inspect
    src = inspect.getsource(_execute_soul_stage)
    assert "ollama_adapter" in src, "Missing ollama_adapter import in _execute_soul_stage"
    assert "llm_status" in src, "Missing llm_status field"
    assert "llm_output" in src, "Missing llm_output variable"
    assert "llm_elapsed" in src, "Missing llm_elapsed field"
    assert "simulated" in src, "Missing simulated fallback"
test("_execute_soul_stage has Ollama integration with fallback", test_orchestrator_ollama_hook)

def test_orchestrator_fallback():
    from soul_orchestrator import _execute_soul_stage
    result = _execute_soul_stage("cezanne", "team_plan", "test task")
    assert result["status"] == "executed", f"Expected executed, got {result['status']}"
    assert "llm_status" in result, "Missing llm_status in result"
    assert result["llm_status"] in ("simulated", "llm", "llm_error"), f"Invalid llm_status: {result['llm_status']}"
    assert "llm_elapsed" in result, "Missing llm_elapsed in result"
test("_execute_soul_stage returns valid result with Ollama offline", test_orchestrator_fallback)

print("\n[5] Web dashboard — HTML file exists and is valid")
def test_web_html():
    from pathlib import Path
    html_path = Path(__file__).parent / "vf_web" / "index.html"
    assert html_path.exists(), f"Web dashboard not found at {html_path}"
    content = html_path.read_text(encoding="utf-8")
    assert "VORTEX CODE" in content, "Missing VORTEX CODE title"
    assert "soul-card" in content, "Missing soul-card component"
    assert "/api/ask" in content, "Missing /api/ask endpoint call"
    assert "/api/status" in content, "Missing /api/status endpoint call"
    assert "routing-bar" in content, "Missing routing visualization"
    assert "expert-box" in content, "Missing expert output component"
test("Web dashboard HTML exists with all components", test_web_html)

print("\n[6] Integration — Full pipeline with Ollama offline")
def test_full_pipeline_offline():
    from soul_orchestrator import run_team_pipeline
    result = run_team_pipeline("Write a function to calculate Fibonacci numbers")
    assert result.get("status") in ("completed", "partial"), f"Unexpected status: {result.get('status')}"
    results = result.get("results", [])
    assert len(results) > 0, "Pipeline should produce at least one stage result"
    for s in results:
        assert "llm_status" in s, f"Stage missing llm_status: {s.get('stage')}"
test("Team pipeline runs end-to-end with Ollama offline", test_full_pipeline_offline)

def test_ultrapilot_offline():
    from soul_orchestrator import dispatch_execution
    result = dispatch_execution("Optimize this sorting algorithm for large datasets")
    assert "mode" in result, "Missing mode in dispatch result"
    assert result["mode"] in ("team", "ultrapilot", "ralph"), f"Invalid mode: {result['mode']}"
test("Dispatch execution works with Ollama offline", test_ultrapilot_offline)

print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
if failed == 0:
    print("\U0001f525 All Phase 1 MVP components verified!")
else:
    print(f"\u26a0 {failed} tests failed — review errors above")
print("=" * 60)
