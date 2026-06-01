"""
VORTEX FLAME Plugin Registry — Definitive Plugin & Tool Listing
=================================================================
Single source of truth for every plugin, tool, MCP server, and skill
available in the VORTEX FLAME system, with KB mappings.

Categories:
  PLUGIN_MAP: 16 knowledge-work-plugins → KB routing
  MCP_SERVER_MAP: MCP servers → KB + tools
  TOOL_MAP: Every tool in the system → owning KBs + category
  EXECUTION_CAPABILITY_MAP: Real-world actions → KB + tool + MCP

Usage:
  from vortex_plugin_list import PLUGIN_MAP, TOOL_MAP, resolve_tool
  kb = resolve_tool("ci_search")  # → ["cezanne", "einstein", ...]
"""

from typing import Dict, List, Optional

PLUGIN_MAP: Dict[str, dict] = {
    "sales":        {"kbs": ["montesquieu", "strategy"],   "domain": "Sales strategy, CRM, pipeline management"},
    "finance":      {"kbs": ["einstein", "strategy"],       "domain": "Financial analysis, forecasting, risk"},
    "legal":        {"kbs": ["montesquieu"],                "domain": "Legal compliance, contract review"},
    "engineering":  {"kbs": ["cezanne", "davinci"],         "domain": "Software engineering, architecture"},
    "data":         {"kbs": ["humboldt", "einstein"],       "domain": "Data analysis, visualization, statistics"},
    "design":       {"kbs": ["davinci", "monet"],           "domain": "UI/UX design, prototyping"},
    "marketing":    {"kbs": ["monet", "montesquieu"],       "domain": "Marketing strategy, content, brand"},
    "hr":           {"kbs": ["guizhu", "montesquieu"],      "domain": "HR management, recruitment"},
    "product":      {"kbs": ["davinci", "strategy"],         "domain": "Product management, roadmap"},
    "research":     {"kbs": ["einstein", "galileo", "darwin"], "domain": "Research methodology, literature"},
    "support":      {"kbs": ["guizhu", "herodotus"],        "domain": "Customer support, troubleshooting"},
    "operations":   {"kbs": ["humboldt", "cezanne"],        "domain": "Ops management, process optimization"},
    "writing":      {"kbs": ["herodotus", "monet"],         "domain": "Technical writing, documentation"},
    "education":    {"kbs": ["herodotus", "guizhu"],        "domain": "Education design, curriculum"},
    "healthcare":   {"kbs": ["darwin", "guizhu"],           "domain": "Healthcare analytics, clinical"},
    "mano_p_gui":   {"kbs": ["cezanne", "davinci"],         "domain": "GUI perception, computer control"},
}

MCP_SERVER_MAP: Dict[str, dict] = {
    "filesystem":   {"kbs": ["cezanne", "herodotus"], "tools": ["read_file", "write_file", "list_dir", "search_files"]},
    "sqlite":       {"kbs": ["humboldt", "einstein"], "tools": ["read_query", "write_query", "create_table", "list_tables"]},
    "fetch":        {"kbs": ["herodotus", "einstein"], "tools": ["fetch_url", "fetch_markdown"]},
    "github":       {"kbs": ["cezanne"],               "tools": ["create_repo", "push_files", "search_code", "create_pr"]},
    "blender":      {"kbs": ["davinci", "monet"],      "tools": ["scene_info", "object_info", "execute_code", "screenshot"]},
    "memory":       {"kbs": ["herodotus", "guizhu"],   "tools": ["create_entities", "create_relations", "search_nodes"]},
    "brave_search": {"kbs": ["einstein", "herodotus"], "tools": ["web_search", "local_search"]},
    "puppeteer":    {"kbs": ["cezanne"],               "tools": ["navigate", "screenshot", "click", "type", "evaluate"]},
    "sequential":   {"kbs": ["cezanne", "strategy"],   "tools": ["sequential_thinking"]},
    "context7":     {"kbs": ["cezanne"],               "tools": ["resolve_library_id", "get_library_docs"]},
    "playwright":   {"kbs": ["cezanne", "davinci"],    "tools": ["browser_navigate", "browser_click", "browser_snapshot"]},
    "vscode":       {"kbs": ["cezanne"],               "tools": ["open_file", "search_code", "run_command", "list_directory"]},
}

TOOL_MAP: Dict[str, dict] = {
    "ci_search":    {"kbs": ["cezanne", "einstein", "galileo", "darwin", "humboldt", "herodotus", "beethoven", "yuanlongping"], "category": "intelligence", "desc": "Search codebase by text/pattern"},
    "ci_context":   {"kbs": ["cezanne", "einstein", "galileo", "darwin", "davinci", "humboldt", "herodotus", "monet", "vangogh", "beethoven", "yuanlongping"], "category": "intelligence", "desc": "Get context around a code element"},
    "ci_impact":    {"kbs": ["cezanne", "einstein", "davinci", "strategy", "montesquieu"], "category": "intelligence", "desc": "Analyze impact of a code change"},
    "ci_domain":    {"kbs": ["davinci", "strategy", "montesquieu", "humboldt", "yuanlongping"], "category": "intelligence", "desc": "Domain-specific code analysis"},
    "ci_callers":   {"kbs": ["cezanne"], "category": "intelligence", "desc": "Find all callers of a function"},
    "ci_callees":   {"kbs": ["cezanne"], "category": "intelligence", "desc": "Find all callees of a function"},
    "ci_affected":  {"kbs": ["cezanne"], "category": "intelligence", "desc": "Find code affected by a change"},
    "ci_ask":       {"kbs": ["cezanne", "herodotus", "beethoven"], "category": "intelligence", "desc": "Ask a question about the codebase"},
    "manop_click":  {"kbs": ["cezanne", "davinci"], "category": "action", "desc": "Click at coordinates or on element"},
    "manop_type":   {"kbs": ["cezanne", "davinci"], "category": "action", "desc": "Type text at cursor"},
    "manop_hotkey": {"kbs": ["cezanne", "davinci"], "category": "action", "desc": "Press keyboard shortcut"},
    "manop_scroll": {"kbs": ["cezanne", "davinci"], "category": "action", "desc": "Scroll viewport"},
    "manop_drag":   {"kbs": ["cezanne", "davinci"], "category": "action", "desc": "Drag from A to B"},
    "manop_screenshot": {"kbs": ["cezanne", "davinci"], "category": "action", "desc": "Capture screen region"},
    "manop_launch": {"kbs": ["cezanne", "davinci"], "category": "action", "desc": "Launch application"},
    "manop_navigate": {"kbs": ["cezanne", "davinci"], "category": "action", "desc": "Navigate to URL"},
    "manop_execute_task": {"kbs": ["cezanne", "davinci"], "category": "action", "desc": "Execute multi-step GUI task"},
}

EXECUTION_CAPABILITY_MAP: Dict[str, dict] = {
    "blender_render":   {"kb": "davinci",   "mcp": "blender",    "tool": "execute_code",    "desc": "Programmatic 3D rendering"},
    "ffmpeg_audio":     {"kb": "beethoven", "mcp": "filesystem",  "tool": "run_command",     "desc": "Audio processing pipeline"},
    "doc_generate":     {"kb": "herodotus", "mcp": "filesystem",  "tool": "write_file",      "desc": "Document generation"},
    "code_review":      {"kb": "cezanne",   "mcp": "github",      "tool": "create_pr_review","desc": "Automated code review"},
    "web_search":       {"kb": "einstein",  "mcp": "brave_search","tool": "web_search",      "desc": "Live web search"},
    "web_scrape":       {"kb": "herodotus", "mcp": "fetch",       "tool": "fetch_markdown",  "desc": "Web content extraction"},
    "browser_control":  {"kb": "cezanne",   "mcp": "playwright",  "tool": "browser_navigate","desc": "Browser automation"},
    "kb_query":         {"kb": "einstein",  "mcp": "sqlite",      "tool": "read_query",     "desc": "Knowledge base SQL query"},
    "gui_operate":      {"kb": "davinci",   "mcp": "mano_p",      "tool": "manop_click",    "desc": "GUI operation"},
}

KB_ALL_TOOLS: Dict[str, List[str]] = {
    "cezanne":      ["ci_search", "ci_context", "ci_callers", "ci_callees", "ci_affected", "ci_domain", "ci_impact", "ci_ask",
                     "manop_click", "manop_type", "manop_hotkey", "manop_scroll", "manop_drag", "manop_screenshot", "manop_launch", "manop_navigate", "manop_execute_task"],
    "einstein":     ["ci_search", "ci_context", "ci_impact"],
    "galileo":      ["ci_search", "ci_context"],
    "darwin":       ["ci_search", "ci_context"],
    "davinci":      ["ci_context", "ci_impact", "ci_domain",
                     "manop_click", "manop_type", "manop_hotkey", "manop_scroll", "manop_drag", "manop_screenshot", "manop_launch", "manop_navigate", "manop_execute_task"],
    "strategy":     ["ci_impact", "ci_domain"],
    "montesquieu":  ["ci_impact", "ci_domain"],
    "humboldt":     ["ci_search", "ci_context", "ci_domain"],
    "yuanlongping": ["ci_search", "ci_context", "ci_domain"],
    "guizhu":       [],
    "herodotus":    ["ci_search", "ci_ask"],
    "monet":        ["ci_context"],
    "vangogh":      ["ci_context"],
    "beethoven":    ["ci_search", "ci_context", "ci_ask"],
}

KB_ALL_SKILLS: Dict[str, List[str]] = {
    "cezanne":      ["kwp_engineering", "kwp_operations", "mano_p_gui"],
    "einstein":     ["kwp_finance", "kwp_research", "kwp_data"],
    "galileo":      ["kwp_research"],
    "darwin":       ["kwp_research", "kwp_healthcare"],
    "davinci":      ["kwp_engineering", "kwp_design", "kwp_product", "mano_p_gui"],
    "strategy":     ["kwp_sales", "kwp_finance", "kwp_product"],
    "montesquieu":  ["kwp_legal", "kwp_sales", "kwp_hr", "kwp_marketing"],
    "humboldt":     ["kwp_data", "kwp_operations"],
    "yuanlongping": ["kwp_research", "kwp_data", "kwp_operations"],
    "guizhu":       ["kwp_hr", "kwp_support", "kwp_healthcare", "kwp_education"],
    "herodotus":    ["kwp_writing", "kwp_support", "kwp_education"],
    "monet":        ["kwp_design", "kwp_marketing", "kwp_writing"],
    "vangogh":      ["kwp_design"],
    "beethoven":    ["kwp_writing", "kwp_design", "kwp_education"],
}


def resolve_tool(tool_name: str) -> List[str]:
    entry = TOOL_MAP.get(tool_name)
    return entry["kbs"] if entry else []


def resolve_skill(skill_name: str) -> List[str]:
    kbs = []
    for kb, skills in KB_ALL_SKILLS.items():
        if skill_name in skills:
            kbs.append(kb)
    return kbs


def resolve_kb(kb_name: str) -> dict:
    return {
        "tools": KB_ALL_TOOLS.get(kb_name, []),
        "skills": KB_ALL_SKILLS.get(kb_name, []),
        "mcps": [mcp for mcp, cfg in MCP_SERVER_MAP.items() if kb_name in cfg["kbs"]],
        "plugins": [p for p, cfg in PLUGIN_MAP.items() if kb_name in cfg["kbs"]],
    }


def list_all_tools() -> List[str]:
    return sorted(TOOL_MAP.keys())


def list_all_mcps() -> List[str]:
    return sorted(MCP_SERVER_MAP.keys())


def stats() -> dict:
    return {
        "total_plugins": len(PLUGIN_MAP),
        "total_mcp_servers": len(MCP_SERVER_MAP),
        "total_tools": len(TOOL_MAP),
        "total_capabilities": len(EXECUTION_CAPABILITY_MAP),
        "total_kbs": len(KB_ALL_TOOLS),
        "kbs_with_tools": sum(1 for t in KB_ALL_TOOLS.values() if t),
        "kbs_with_skills": sum(1 for s in KB_ALL_SKILLS.values() if s),
    }
