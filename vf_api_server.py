"""
VORTEX Code — API Server
==========================
FastAPI-based API gateway for VORTEX Code.

Endpoints:
  POST /api/ask       — Ask multi-expert system
  POST /api/review    — Multi-expert code review
  POST /api/fix       — Ralph iterative fix
  POST /api/plan      — Ultrapilot parallel planning
  GET  /api/status    — System status
  GET  /api/souls     — List souls
  GET  /api/memory    — Query soul memories
  GET  /api/stream    — SSE streaming for real-time output
  GET  /              — Web dashboard
"""

import json
import time
from pathlib import Path
from typing import Optional

from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="VORTEX Code API", version="0.1.0")

WEB_DIR = Path(__file__).parent / "vf_web"


class AskRequest(BaseModel):
    query: str
    soul: Optional[str] = None
    top_k: int = 2
    stream: bool = False


class ReviewRequest(BaseModel):
    code: str
    filename: Optional[str] = None
    soul: Optional[str] = None


class FixRequest(BaseModel):
    code: str
    issue: str = "general quality"
    max_iterations: int = 3


class PlanRequest(BaseModel):
    task: str
    top_k: int = 3


class MemoryRequest(BaseModel):
    soul: str = "cezanne"
    query: str = ""
    category: Optional[str] = None
    top_k: int = 10


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = WEB_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>VORTEX Code</h1><p>Web dashboard not found. Use CLI: vortex status</p>")


@app.get("/api/status")
async def status():
    from ollama_adapter import get_adapter, _model_router
    from soul_memory import _engine

    adapter = get_adapter()
    ollama_ok = adapter.is_available()
    models = adapter.list_models() if ollama_ok else []
    model_info = _model_router.model_info() if ollama_ok else {}

    soul_stats = {}
    for soul in ["cezanne", "einstein", "galileo", "darwin", "davinci",
                  "strategy", "montesquieu", "humboldt", "yuanlongping",
                  "guizhu", "herodotus", "monet", "vangogh", "beethoven"]:
        try:
            count = _engine._get_db(soul).execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
            soul_stats[soul] = {"active": True, "memories": count}
        except Exception:
            soul_stats[soul] = {"active": True, "memories": 0}

    return {
        "ollama": {"connected": ollama_ok, "url": adapter.base_url, "model": adapter.model, "models": models},
        "hermes": {
            "available": model_info.get("hermes_available", False),
            "selected_model": model_info.get("selected_model", ""),
            "soul_model_map": {s: _model_router.best_model(s) for s in ["cezanne", "einstein", "strategy", "davinci", "guizhu"]},
        },
        "souls": soul_stats,
        "version": "0.2.0",
        "timestamp": time.time(),
    }


@app.get("/api/souls")
async def list_souls():
    from soul_orchestrator import SOUL_CAPABILITIES
    return {"souls": SOUL_CAPABILITIES, "active": list(SOUL_CAPABILITIES.keys()), "total": len(SOUL_CAPABILITIES)}


@app.post("/api/ask")
async def ask(req: AskRequest):
    from soul_orchestrator import soft_route_to_souls, SOUL_CAPABILITIES
    from soul_memory import recall
    from ollama_adapter import get_adapter

    adapter = get_adapter()

    if req.soul:
        routing = {"candidates": [{"soul": req.soul, "confidence": 1.0, "selected": True}]}
    else:
        routing = soft_route_to_souls(req.query, top_k=req.top_k)

    candidates = routing.get("candidates", [])  # type: ignore[union-attr]
    if not candidates:
        candidates = [{"soul": "cezanne", "confidence": 0.5, "selected": True}]

    ollama_available = adapter.is_available()

    results = []
    for c in candidates[:req.top_k]:
        soul = c["soul"]
        mem = recall(soul, req.query, top_k=3, categories=["knowledge", "domain_memory", "conversation"])
        snippets = [m.get("content", {}).get("topic", "") for m in mem[:3] if isinstance(m.get("content", {}), dict)]

        if ollama_available:
            result = adapter.generate(soul, req.query, memory_snippets=snippets)
            results.append({
                "soul": soul,
                "confidence": c["confidence"],
                "output": result.get("output", ""),
                "status": result["status"],
                "elapsed": result.get("elapsed", 0),
                "tokens_per_second": result.get("tokens_per_second", 0),
                "memory_hits": len([m for m in mem if m]),
            })
        else:
            cap = SOUL_CAPABILITIES.get(soul, {})
            domain = ", ".join(cap.get("domain", []))
            boundary = cap.get("boundary", {})
            available = boundary.get("可用", [])
            avail_str = ", ".join(available[:5]) if isinstance(available, list) else str(available)[:100]
            results.append({
                "soul": soul,
                "confidence": c["confidence"],
                "output": f"[离线模式] Ollama未连接。{soul.title()}灵魂领域：{domain}。可用能力：{avail_str}。请启动Ollama获取完整推理能力。",
                "status": "offline",
                "elapsed": 0,
                "tokens_per_second": 0,
                "memory_hits": len([m for m in mem if m]),
            })

    return {
        "query": req.query,
        "routing": routing,
        "results": results,
        "ollama_connected": ollama_available,
        "mode": "dual_expert" if len(results) >= 2 and abs(candidates[0]["confidence"] - candidates[1]["confidence"]) < 0.15 else "single_expert",
    }


@app.post("/api/review")
async def review(req: ReviewRequest):
    from soul_orchestrator import soft_route_to_souls
    from soul_memory import recall
    from ollama_adapter import get_adapter

    adapter = get_adapter()
    if not adapter.is_available():
        raise HTTPException(status_code=503, detail="Ollama not connected")

    code = req.code[:8000]
    query = f"Review this code for bugs, edge cases, and improvements:\n\n```\n{code}\n```"

    if req.soul:
        routing = {"candidates": [{"soul": req.soul, "confidence": 1.0}]}
    else:
        routing = soft_route_to_souls(query, top_k=2)

    candidates = routing.get("candidates", [])  # type: ignore[union-attr]
    if not candidates:
        candidates = [{"soul": "cezanne", "confidence": 0.7}, {"soul": "einstein", "confidence": 0.5}]

    results = []
    for c in candidates[:2]:
        soul = c["soul"]
        mem = recall(soul, "code review", top_k=2)
        snippets = [m.get("content", {}).get("topic", "") for m in mem[:2] if isinstance(m.get("content", {}), dict)]
        result = adapter.generate(soul, query, memory_snippets=snippets)
        results.append({
            "soul": soul,
            "confidence": c["confidence"],
            "output": result.get("output", ""),
            "status": result["status"],
            "elapsed": result.get("elapsed", 0),
        })

    return {"filename": req.filename, "routing": routing, "results": results}


@app.post("/api/fix")
async def fix(req: FixRequest):
    from ollama_adapter import get_adapter

    adapter = get_adapter()
    if not adapter.is_available():
        raise HTTPException(status_code=503, detail="Ollama not connected")

    iterations = []
    current_code = req.code[:4000]

    for i in range(req.max_iterations):
        prompt = (
            f"Fix the following issue: {req.issue}\n\n"
            f"Current code:\n```\n{current_code}\n```\n\n"
            f"Provide the COMPLETE fixed code."
        )

        fix_result = adapter.generate("cezanne", prompt)
        if fix_result["status"] != "ok":
            iterations.append({"iteration": i + 1, "status": "error", "error": fix_result.get("error")})
            break

        verify_prompt = (
            f"Verify this code fix is correct. Issue was: {req.issue}\n\n"
            f"Fixed code:\n```\n{fix_result['output'][:3000]}\n```\n\n"
            f"Respond PASS or FAIL."
        )
        verify_result = adapter.generate("einstein", verify_prompt)

        passed = verify_result["status"] == "ok" and "PASS" in verify_result.get("output", "").upper()

        iterations.append({
            "iteration": i + 1,
            "fix_output": fix_result["output"],
            "verify_output": verify_result.get("output", ""),
            "passed": passed,
            "elapsed": fix_result.get("elapsed", 0) + verify_result.get("elapsed", 0),
        })

        if passed:
            break
        current_code = fix_result["output"][:4000]

    return {"issue": req.issue, "iterations": iterations, "total_iterations": len(iterations)}


@app.post("/api/plan")
async def plan(req: PlanRequest):
    from soul_orchestrator import soft_route_to_souls
    from soul_memory import recall
    from ollama_adapter import get_adapter

    adapter = get_adapter()
    if not adapter.is_available():
        raise HTTPException(status_code=503, detail="Ollama not connected")

    routing = soft_route_to_souls(req.task, top_k=req.top_k)
    candidates = routing.get("candidates", [])  # type: ignore[union-attr]
    if not candidates:
        candidates = [{"soul": "cezanne", "confidence": 0.7}, {"soul": "einstein", "confidence": 0.5}]

    results = []
    for c in candidates[:3]:
        soul = c["soul"]
        mem = recall(soul, req.task, top_k=2)
        snippets = [m.get("content", {}).get("topic", "") for m in mem[:2] if isinstance(m.get("content", {}), dict)]
        prompt = f"Design a plan for: {req.task}\n\nProvide a structured, step-by-step plan."
        result = adapter.generate(soul, prompt, memory_snippets=snippets)
        results.append({
            "soul": soul,
            "confidence": c["confidence"],
            "output": result.get("output", ""),
            "status": result["status"],
        })

    return {"task": req.task, "routing": routing, "results": results}


@app.get("/api/memory")
async def memory_query(soul: str = "cezanne", query: str = "", category: Optional[str] = None, top_k: int = 10):
    from soul_memory import recall, search

    if query:
        cat = category or "knowledge"
        results = search(soul, cat, query, top_k=top_k)
    else:
        results = recall(soul, query, top_k=top_k)

    return {"soul": soul, "query": query, "results": results, "count": len(results)}


class SkillEvolveRequest(BaseModel):
    task: str
    soul: str = "cezanne"


@app.get("/api/skills")
async def skills_list():
    try:
        from skill_evolver import SkillEvolver
        evolver = SkillEvolver()
        skills = evolver.list_skills()
        return {"skills": skills, "count": len(skills)}
    except Exception as e:
        return {"skills": [], "count": 0, "error": str(e)}


@app.post("/api/skills/evolve")
async def skills_evolve(req: SkillEvolveRequest):
    from ollama_adapter import get_adapter

    adapter = get_adapter()
    if not adapter.is_available():
        raise HTTPException(status_code=503, detail="Ollama not connected")

    prompt = (
        f"Based on this task, generate a reusable skill script.\n"
        f"Task: {req.task}\n\n"
        f'Output a JSON object with: "name", "description", "steps", "tools_needed", "trigger_patterns".\n'
        f"Respond ONLY with valid JSON."
    )

    result = adapter.generate(req.soul, prompt)

    if result["status"] != "ok":
        raise HTTPException(status_code=500, detail=result.get("error", "LLM error"))

    try:
        skill_data = json.loads(result["output"])
    except json.JSONDecodeError:
        return {"status": "parse_error", "raw_output": result["output"][:500]}

    try:
        from skill_evolver import SkillEvolver
        evolver = SkillEvolver()
        evolver.register_evolved_skill(skill_data.get("name", "auto_skill"), skill_data)
    except Exception as e:
        return {"status": "registered_with_warning", "skill": skill_data, "warning": str(e)}

    return {"status": "ok", "skill": skill_data}


@app.get("/api/skills/suggest")
async def skills_suggest():
    from ollama_adapter import get_adapter
    from soul_memory import recall

    adapter = get_adapter()
    if not adapter.is_available():
        raise HTTPException(status_code=503, detail="Ollama not connected")

    recent_tasks = []
    for soul in ["cezanne", "einstein", "davinci"]:
        try:
            mems = recall(soul, "", top_k=5, categories=["trajectory"])
            for m in mems[:3]:
                content = m.get("content", {})
                if isinstance(content, dict) and content.get("task"):
                    recent_tasks.append(content["task"])
        except Exception:
            pass

    if not recent_tasks:
        return {"suggestions": [], "reason": "no_recent_tasks"}

    prompt = (
        f"Based on these recent tasks, suggest 3 skills that could be auto-generated:\n\n"
        + "\n".join(f"- {t}" for t in recent_tasks[:10])
        + "\n\nFor each: name, description, trigger pattern. Be concise."
    )

    result = adapter.generate("cezanne", prompt)
    return {"suggestions_raw": result.get("output", ""), "recent_tasks_count": len(recent_tasks)}


@app.get("/api/hermes")
async def hermes_status():
    from ollama_adapter import get_adapter, _model_router

    adapter = get_adapter()
    info = _model_router.model_info()

    soul_map = {}
    for soul in ["cezanne", "einstein", "strategy", "davinci", "guizhu",
                 "galileo", "darwin", "montesquieu", "humboldt", "yuanlongping",
                 "herodotus", "monet", "vangogh", "beethoven"]:
        soul_map[soul] = _model_router.best_model(soul)

    return {
        "ollama_connected": adapter.is_available(),
        "hermes_available": info.get("hermes_available", False),
        "selected_model": info.get("selected_model", ""),
        "available_models": info.get("available_models", []),
        "soul_model_map": soul_map,
        "vortex_advantage": [
            "14 industry knowledge bases with C-JEPA causal engines",
            "Per-domain model selection (Hermes single-model)",
            "Multi-knowledge-base arbitration (Hermes single-agent)",
            "Persistent memory per knowledge base (Hermes flat memory)",
            "Skill evolution with 5-dimension audit (Hermes basic GEPA)",
        ],
    }


@app.post("/api/stream")
async def stream_ask(req: AskRequest):
    from soul_orchestrator import soft_route_to_souls
    from soul_memory import recall
    from ollama_adapter import get_adapter

    adapter = get_adapter()
    if not adapter.is_available():
        raise HTTPException(status_code=503, detail="Ollama not connected")

    routing = soft_route_to_souls(req.query, top_k=1)
    candidates = routing.get("candidates", [{"soul": "cezanne", "confidence": 0.5}])  # type: ignore[union-attr]
    soul = req.soul or candidates[0]["soul"]

    mem = recall(soul, req.query, top_k=3)
    snippets = [m.get("content", {}).get("topic", "") for m in mem[:3] if isinstance(m.get("content", {}), dict)]

    def event_stream():
        yield f"data: {json.dumps({'type': 'routing', 'soul': soul, 'confidence': candidates[0].get('confidence', 0)})}\n\n"
        yield f"data: {json.dumps({'type': 'memory', 'hits': len(mem)})}\n\n"

        for chunk in adapter.generate_stream(soul, req.query, memory_snippets=snippets):
            if chunk.get("token"):
                yield f"data: {json.dumps({'type': 'token', 'soul': soul, 'content': chunk['token']})}\n\n"
            if chunk.get("done"):
                yield f"data: {json.dumps({'type': 'done', 'soul': soul})}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/mcp/tools")
async def mcp_tools_list(soul: Optional[str] = None, category: Optional[str] = None):
    from mcp_sandbox_server import get_server
    server = get_server()
    return {"tools": server.list_tools(soul=soul or "", category=category or "")}


@app.get("/api/mcp/tools/{tool_id}")
async def mcp_tool_detail(tool_id: str):
    from mcp_sandbox_server import get_server
    server = get_server()
    tool = server.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")
    return tool


class MCPExecuteRequest(BaseModel):
    tool_id: str
    soul: str
    arguments: dict = {}


@app.post("/api/mcp/execute")
async def mcp_execute(req: MCPExecuteRequest):
    from mcp_sandbox_server import get_server
    server = get_server()
    result = server.execute(req.tool_id, req.soul, req.arguments)
    return asdict(result) if hasattr(result, "__dataclass_fields__") else result


@app.get("/api/aigen/tools")
async def aigen_tools_list(soul: Optional[str] = None):
    from ai_gen_adapter import get_adapter
    adapter = get_adapter()
    return {"tools": adapter.get_tools_for_soul(soul) if soul else adapter.list_tools()}


class AIGenRequest(BaseModel):
    prompt: str
    soul: str = "monet"
    gen_type: str = "image"
    params: dict = {}


@app.post("/api/aigen/generate")
async def aigen_generate(req: AIGenRequest):
    from ai_gen_adapter import get_adapter
    adapter = get_adapter()
    if req.gen_type == "image":
        return adapter.txt2img(req.prompt, soul=req.soul, **req.params)
    elif req.gen_type == "music":
        return adapter.txt2music(req.prompt, soul=req.soul, **req.params)
    elif req.gen_type == "video":
        return adapter.txt2video(req.prompt, soul=req.soul, **req.params)
    elif req.gen_type == "voice":
        return adapter.txt2voice(req.prompt, soul=req.soul, **req.params)
    elif req.gen_type == "3d":
        return adapter.txt23d(req.prompt, soul=req.soul, **req.params)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown gen_type: {req.gen_type}")


@app.get("/api/aigen/status")
async def aigen_status():
    from ai_gen_adapter import get_adapter
    adapter = get_adapter()
    return adapter.status()


@app.get("/api/opendesign/skills")
async def opendesign_skills(soul: Optional[str] = None):
    from open_design_adapter import get_adapter
    adapter = get_adapter()
    return {"skills": adapter.list_skills(soul=soul or "")}


@app.get("/api/opendesign/systems")
async def opendesign_systems():
    from open_design_adapter import get_adapter
    adapter = get_adapter()
    return {"design_systems": adapter.list_design_systems()}


class OpenDesignPPTRequest(BaseModel):
    prompt: str
    soul: str = "monet"
    design_system: str = "guizang_ppt"
    title: str = ""
    outline: str = ""
    export_format: str = "html"


@app.post("/api/opendesign/generate_ppt")
async def opendesign_generate_ppt(req: OpenDesignPPTRequest):
    from open_design_adapter import get_adapter
    adapter = get_adapter()
    return adapter.generate_ppt(req.prompt, soul=req.soul,
                                design_system=req.design_system,
                                title=req.title, outline=req.outline,
                                export_format=req.export_format)


class OpenDesignScoreRequest(BaseModel):
    content: str
    soul: str = "monet"


@app.post("/api/opendesign/aesthetic_score")
async def opendesign_aesthetic_score(req: OpenDesignScoreRequest):
    from open_design_adapter import get_adapter
    adapter = get_adapter()
    return adapter.aesthetic_score(req.content, soul=req.soul)


class OpenDesignColorRequest(BaseModel):
    prompt: str
    soul: str = "monet"
    style: str = "linear"
    n_colors: int = 5


@app.post("/api/opendesign/color_palette")
async def opendesign_color_palette(req: OpenDesignColorRequest):
    from open_design_adapter import get_adapter
    adapter = get_adapter()
    return adapter.color_palette(req.prompt, soul=req.soul,
                                 style=req.style, n_colors=req.n_colors)


@app.get("/api/opendesign/status")
async def opendesign_status():
    from open_design_adapter import get_adapter
    adapter = get_adapter()
    return adapter.status()


@app.get("/api/mcp/registry")
async def mcp_registry(category: Optional[str] = None, soul: Optional[str] = None, status: Optional[str] = None):
    from mcp_server_registry import list_servers
    servers = list_servers(category=category or "", soul=soul or "", status=status or "")
    return {"servers": [{"id": s.server_id, "name_zh": s.name_zh, "category": s.category.value,
                          "status": s.status.value, "tools": s.tools,
                          "soul_mapping": s.soul_mapping} for s in servers],
            "total": len(servers)}


@app.get("/api/skills/registry/stats")
async def skill_registry_stats():
    from skill_registry_auto import get_registry
    reg = get_registry()
    return reg.stats()


@app.get("/api/skills/registry/search")
async def skill_registry_search(q: str = "", category: str = "", tier: str = "",
                                 soul: str = "", tag: str = "", status: str = "",
                                 source: str = "", verified_only: bool = False,
                                 sort_by: str = "stars", limit: int = 50):
    from skill_registry_auto import get_registry
    reg = get_registry()
    return {"results": reg.search(q, category, tier, soul, tag, status, source, verified_only, sort_by, limit)}


@app.get("/api/skills/registry/{skill_id}")
async def skill_registry_get(skill_id: str):
    from skill_registry_auto import get_registry
    reg = get_registry()
    entry = reg.get(skill_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Skill not found")
    return entry


@app.post("/api/skills/registry/add")
async def skill_registry_add(entry_data: dict):
    from skill_registry_auto import get_registry
    reg = get_registry()
    return reg.add_skill(entry_data)


@app.delete("/api/skills/registry/{skill_id}")
async def skill_registry_remove(skill_id: str):
    from skill_registry_auto import get_registry
    reg = get_registry()
    if not reg.remove_skill(skill_id):
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"status": "removed"}


@app.post("/api/skills/registry/refresh")
async def skill_registry_refresh(source: str = "all"):
    from skill_registry_auto import get_registry
    reg = get_registry()
    return reg.refresh_from_sources(source)


@app.get("/api/skills/registry/categories")
async def skill_registry_categories():
    from skill_registry_auto import get_registry
    reg = get_registry()
    return {"categories": reg.get_categories()}


@app.get("/api/files/ls")
async def files_list(path: str = "."):
    p = (Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path).resolve()
    if not p.exists():
        raise HTTPException(404, f"Path not found: {p}")
    if p.is_file():
        return {"path": str(p), "is_file": True, "size": p.stat().st_size}
    entries = []
    for entry in sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
        try:
            st = entry.stat()
            entries.append({"name": entry.name, "is_dir": entry.is_dir(), "size": st.st_size if entry.is_file() else 0, "mtime": st.st_mtime})
        except OSError:
            pass
    return {"path": str(p), "is_dir": True, "entries": entries}


@app.get("/api/files/read")
async def files_read(path: str, start: int = 0, end: int = -1):
    p = (Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path).resolve()
    if not p.is_file():
        raise HTTPException(404, f"File not found: {p}")
    if not str(p).startswith(str(PROJECT_ROOT)):
        raise HTTPException(403, "Access denied outside project")
    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"path": str(p), "binary": True, "size": p.stat().st_size, "content": "[Binary file - cannot display]"}
    total = len(content)
    if end == -1 or end > total:
        end = total
    return {"path": str(p), "content": content[start:end], "total_lines": content.count("\n") + 1, "size": p.stat().st_size, "language": _guess_language(p.name)}


@app.post("/api/files/write")
async def files_write(req: dict):
    path = req.get("path", "")
    content = req.get("content", "")
    if not path or not content and content != "":
        raise HTTPException(400, "path and content required")
    p = (Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path).resolve()
    if not str(p).startswith(str(PROJECT_ROOT)):
        raise HTTPException(403, "Access denied outside project")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"path": str(p), "written": len(content), "status": "ok"}


@app.get("/api/files/search")
async def files_search(q: str = "", path: str = ".", max_results: int = 20):
    p = (Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path).resolve()
    if not p.exists():
        raise HTTPException(404, f"Path not found: {p}")
    results = []
    exts = {".py", ".js", ".ts", ".tsx", ".html", ".css", ".json", ".yaml", ".yml", ".md", ".txt", ".sh", ".ps1"}
    for fp in p.rglob("*"):
        if fp.suffix not in exts or fp.stat().st_size > 500_000:
            continue
        if any(part.startswith(".") and part != "." for part in fp.parts):
            continue
        if any(skip in fp.parts for skip in ["__pycache__", "node_modules", ".git", "kb_workflow", "kb_mcp", "kb_skill"]):
            continue
        try:
            text = fp.read_text(encoding="utf-8")
            if q.lower() in text.lower():
                lines = text.split("\n")
                for i, line in enumerate(lines):
                    if q.lower() in line.lower():
                        results.append({"file": str(fp.relative_to(p)), "line": i + 1, "content": line.strip()[:200]})
                        if len(results) >= max_results:
                            break
                if len(results) >= max_results:
                    break
        except Exception:
            pass
    return {"query": q, "results": results[:max_results], "total": len(results)}


def _guess_language(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return { "py": "python", "js": "javascript", "ts": "typescript", "tsx": "typescript", "html": "html",
             "css": "css", "json": "json", "yaml": "yaml", "yml": "yaml", "md": "markdown", "sh": "bash",
             "ps1": "powershell", "sql": "sql", "java": "java", "cpp": "cpp", "c": "c", "rs": "rust",
             "go": "go", "rb": "ruby", "php": "php", "swift": "swift", "kt": "kotlin", "xml": "xml",
             "svg": "xml", "txt": "plaintext", "cfg": "ini", "ini": "ini", "toml": "toml" }.get(ext, "plaintext")


PROJECT_ROOT = Path(__file__).parent.resolve()


class ModifyRequest(BaseModel):
    file_path: str
    instruction: str
    soul: Optional[str] = "cezanne"
    use_knowledge: bool = True
    top_k: int = 5
    mode: str = "replace"


@app.post("/api/modify")
async def api_modify(req: ModifyRequest):
    p = (Path(req.file_path) if Path(req.file_path).is_absolute() else PROJECT_ROOT / req.file_path).resolve()
    if not p.exists():
        raise HTTPException(404, f"File not found: {p}")
    try:
        original = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File is not valid UTF-8 text")

    kb_context = ""
    if req.use_knowledge:
        try:
            soul_entries = _get_soul_knowledge(req.soul or "cezanne", req.instruction, top_k=req.top_k)
            kb_context = "\n".join(
                f"[{e.get('category','')}] {e.get('content_text','')[:300]}"
                for e in soul_entries
            )[:4000]
        except Exception as e:
            kb_context = f"[KB查询失败: {e}]"

    system_prompt = f"""You are the {req.soul} soul of VORTEX FLAME, an expert HTML/CSS code modifier.
You modify HTML/CSS files according to instructions. Output ONLY the complete modified file content.
Do NOT include explanations, markdown fences, or any text outside the file content.
Current knowledge: {kb_context}"""

    user_prompt = f"""Modify this file according to the instruction:

INSTRUCTION: {req.instruction}

ORIGINAL FILE:
{original[:15000]}

Return the COMPLETE modified file."""

    try:
        import requests as _rq
        resp = _rq.post(
            "http://localhost:11434/api/generate",
            json={"model": "hermes3:8b", "system": system_prompt, "prompt": user_prompt, "stream": False,
                  "options": {"temperature": 0.3, "num_predict": 16384}},
            timeout=120,
        )
        if resp.status_code == 200:
            modified = resp.json().get("response", original)
            if req.mode == "replace":
                p.write_text(modified, encoding="utf-8")
            return {"status": "ok", "mode": req.mode, "file": str(p), "original_length": len(original),
                    "modified_length": len(modified), "kb_entries": 0 if not kb_context else 1}
        else:
            raise HTTPException(502, f"Ollama returned {resp.status_code}")
    except Exception as e:
        raise HTTPException(500, f"Modification failed: {e}")


def _get_soul_knowledge(soul: str, query: str, top_k: int = 5):
    db_path = PROJECT_ROOT / ".vf_memory" / f"{soul}.db"
    if not db_path.exists():
        return []
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT category, content_text, importance FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, top_k)
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        try:
            rows = conn.execute(
                "SELECT category, content_text, importance FROM memories WHERE soul=? ORDER BY importance DESC LIMIT ?",
                (soul, top_k)
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
    finally:
        conn.close()


class OrchestrateRequest(BaseModel):
    query: str
    kbs: Optional[list] = None
    top_k: int = 3
    mode: str = "auto"


@app.post("/api/orchestrate")
async def orchestrate(req: OrchestrateRequest):
    from soul_orchestrator import orchestrate_cross_kb, soft_route_to_souls, apply_moe_routing

    if req.mode == "moe":
        route_result = apply_moe_routing(req.query, top_k=req.top_k)
        kbs = route_result["selected_kbs"][:3] if route_result["selected_kbs"] else ["cezanne", "einstein"]
    elif req.kbs:
        kbs = req.kbs
    else:
        routing = soft_route_to_souls(req.query, top_k=req.top_k)
        kbs = [c["soul"] for c in routing if c.get("confidence", 0) > 0.05][:3] or ["cezanne"]

    orchestrated = orchestrate_cross_kb(req.query, kbs=kbs, top_k=2)

    return {
        "query": req.query,
        "mode": req.mode,
        "orchestrated_kbs": orchestrated["orchestrated_kbs"],
        "results": orchestrated["results"],
        "summary": orchestrated["summary"],
    }


@app.get("/api/orchestrate/kbs")
async def list_orchestrate_kbs():
    from soul_orchestrator import SOUL_CAPABILITIES
    from moe_expert_loader import MoEExpertLoader
    loader = MoEExpertLoader()
    kbs = []
    for s, cap in SOUL_CAPABILITIES.items():
        kbs.append({
            "id": s,
            "name": cap.get("full_name", s),
            "domain": cap.get("domain", []),
            "tier": cap.get("tier", "?"),
            "jepa_variant": cap.get("jepa_variant", "unknown"),
            "tools": cap.get("tools", [])[:5],
            "skills": cap.get("skills", [])[:5],
        })
    return {"kbs": kbs, "total": len(kbs), "router": loader.list_experts() if hasattr(loader, "list_experts") else []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)

app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
