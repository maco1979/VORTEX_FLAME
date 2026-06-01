#!/usr/bin/env python3
"""
VORTEX FLAME Knowledge Base Puller
====================================

Pulls top-tier Harness + OpenClaw knowledge repos and indexes them
into soul_memory for RAG retrieval and JEPA training.

Knowledge Sources:
  OpenClaw (Agent Sandbox):
    1. NVIDIA OpenClaw:  https://github.com/nvidia/openclaw
    2. E2B Sandbox:      https://github.com/e2b-dev/e2b
    3. Alibaba OpenSandbox: https://github.com/alibaba/opensandbox

  Harness Runtime (Agent Governance):
    1. Microsoft Agent Governance: https://github.com/microsoft/agent-governance
    2. SafeHarness (CAS):          https://github.com/AgentSecurityLab/SafeHarness
    3. ArbiterOS (CUHK):           https://github.com/ArbiterOS/arbiter-core
    4. OWASP Agent Top10:          https://owasp.org/www-project-agent-security-top-10/

Output:
  - Cloned repos in D:\VORTEX_FLAME\kb_harness\
  - Indexed entries in soul_memory (cezanne + montesquieu souls)
  - AST features for CODE-JEPA training
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(os.path.dirname(__file__)))

KB_ROOT = r"D:\VORTEX_FLAME\kb_harness"
MCP_ROOT = r"D:\VORTEX_FLAME\kb_mcp"
SKILL_ROOT = r"D:\VORTEX_FLAME\kb_skill"

REPOS = {
    "openclaw": [
        {"name": "nvidia_openclaw", "url": "https://github.com/NVIDIA/NemoClaw",
         "soul": "cezanne", "category": "code_memory",
         "tags": ["sandbox", "seccomp", "isolation", "agent-safety", "nvidia"]},
        {"name": "e2b_sandbox", "url": "https://github.com/e2b-dev/e2b",
         "soul": "cezanne", "category": "code_memory",
         "tags": ["sandbox", "firecracker", "vm-isolation", "agent-execution"]},
        {"name": "alibaba_opensandbox", "url": "https://github.com/alibaba/opensandbox",
         "soul": "cezanne", "category": "code_memory",
         "tags": ["sandbox", "container", "seccomp", "cgroup"]},
    ],
    "harness": [
        {"name": "microsoft_agent_governance", "url": "https://github.com/microsoft/agent-governance-toolkit",
         "soul": "montesquieu", "category": "knowledge",
         "tags": ["governance", "owasp", "permission", "interception", "agent-security"]},
        {"name": "safeharness", "url": "https://github.com/liu-yang-maker/SafeHarness",
         "soul": "montesquieu", "category": "knowledge",
         "tags": ["harness", "runtime-governance", "memory-poisoning", "cascade-risk"]},
        {"name": "arbiteros", "url": "https://github.com/cure-lab/ArbiterOS",
         "soul": "montesquieu", "category": "knowledge",
         "tags": ["arbiter", "permission-matrix", "multi-agent", "least-privilege"]},
    ],
}

MCP_REPOS = [
    {"name": "langmem", "url": "https://github.com/langchain-ai/langmem",
     "soul": "guizhu", "category": "knowledge",
     "tags": ["mcp", "memory", "langchain", "agent-memory", "long-short-term", "decay"]},
    {"name": "mem0", "url": "https://github.com/mem0ai/mem0",
     "soul": "guizhu", "category": "knowledge",
     "tags": ["mcp", "memory", "semantic-memory", "conversation-memory", "conflict-resolution"]},
    {"name": "mcp_servers", "url": "https://github.com/modelcontextprotocol/servers",
     "soul": "cezanne", "category": "code_memory",
     "tags": ["mcp", "protocol", "server", "tool-integration", "context-protocol"]},
    {"name": "mcp_python_sdk", "url": "https://github.com/modelcontextprotocol/python-sdk",
     "soul": "cezanne", "category": "code_memory",
     "tags": ["mcp", "sdk", "python", "protocol-implementation"]},
    {"name": "mcp_spec", "url": "https://github.com/modelcontextprotocol/specification",
     "soul": "guizhu", "category": "knowledge",
     "tags": ["mcp", "specification", "protocol", "standard"]},
]

SKILL_REPOS = [
    {"name": "autogpt", "url": "https://github.com/Significant-Gravitas/AutoGPT",
     "soul": "beethoven", "category": "skill",
     "tags": ["skill", "autogpt", "agent", "self-evolution", "skill-registry"]},
    {"name": "meta_hyperagents", "url": "https://github.com/facebookresearch/HyperAgents",
     "soul": "beethoven", "category": "skill",
     "tags": ["skill", "meta", "self-improvement", "hyperagents", "evolution"]},
    {"name": "evolver", "url": "https://github.com/KnowledgeXLab/EvolveR",
     "soul": "beethoven", "category": "skill",
     "tags": ["skill", "evolution", "icml2026", "experience-distillation", "rl"]},
    {"name": "hermes_evolution", "url": "https://github.com/NousResearch/hermes-agent-self-evolution",
     "soul": "beethoven", "category": "skill",
     "tags": ["skill", "hermes", "self-evolution", "gepa", "dspy", "prompt-evolution"]},
]

CODE_EXTENSIONS = {".py", ".js", ".ts", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp"}
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc", ".html"}
CONFIG_EXTENSIONS = {".yaml", ".yml", ".toml", ".json", ".cfg", ".ini"}

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "venv", ".venv", "dist", "build",
    ".tox", "egg-info", ".mypy_cache", ".pytest_cache", "target", "vendor",
}


def clone_repo(url: str, dest: str, depth: int = 1) -> bool:
    if os.path.exists(dest):
        print(f"  SKIP (exists): {dest}")
        return True

    try:
        cmd = ["git", "clone", "--depth", str(depth), url, dest]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print(f"  OK: cloned to {dest}")
            return True
        else:
            stderr = result.stderr.strip()
            if "not found" in stderr.lower() or "does not exist" in stderr.lower():
                print(f"  SKIP (repo not found): {url}")
            else:
                print(f"  FAIL: {stderr[:100]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: clone exceeded 120s")
        return False
    except FileNotFoundError:
        print(f"  ERROR: git not found, trying pip install gitpython...")
        return _clone_with_gitpython(url, dest, depth)


def _clone_with_gitpython(url: str, dest: str, depth: int = 1) -> bool:
    try:
        import git
        git.Repo.clone_from(url, dest, depth=depth)
        print(f"  OK (gitpython): cloned to {dest}")
        return True
    except Exception as e:
        print(f"  FAIL (gitpython): {e}")
        return False


def scan_repo(repo_path: str) -> dict:
    code_files = []
    doc_files = []
    config_files = []

    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            ext = Path(fname).suffix.lower()

            rel_path = os.path.relpath(fpath, repo_path)

            if ext in CODE_EXTENSIONS:
                code_files.append(rel_path)
            elif ext in DOC_EXTENSIONS:
                doc_files.append(rel_path)
            elif ext in CONFIG_EXTENSIONS:
                config_files.append(rel_path)

    return {
        "code_files": code_files,
        "doc_files": doc_files,
        "config_files": config_files,
        "total": len(code_files) + len(doc_files) + len(config_files),
    }


def read_file_safe(fpath: str, max_chars: int = 5000) -> str:
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def index_repo_to_memory(repo_info: dict, repo_path: str, memory_engine) -> dict:
    soul = repo_info["soul"]
    category = repo_info["category"]
    tags = repo_info.get("tags", [])
    repo_name = repo_info["name"]

    scan = scan_repo(repo_path)
    indexed = 0
    errors = 0

    doc_files = scan["doc_files"][:50]
    for rel_path in doc_files:
        fpath = os.path.join(repo_path, rel_path)
        content = read_file_safe(fpath, max_chars=8000)
        if len(content.strip()) < 50:
            continue

        title = Path(rel_path).stem.replace("-", " ").replace("_", " ")
        entry_content = {
            "topic": f"[{repo_name}] {title}",
            "source": repo_info["url"],
            "path": rel_path,
            "text": content[:3000],
            "tags": tags,
        }

        try:
            memory_engine.write(soul, category, entry_content, tags=tags)
            indexed += 1
        except Exception as e:
            errors += 1

    code_files = scan["code_files"][:30]
    for rel_path in code_files:
        fpath = os.path.join(repo_path, rel_path)
        content = read_file_safe(fpath, max_chars=5000)
        if len(content.strip()) < 30:
            continue

        entry_content = {
            "topic": f"[{repo_name}] {Path(rel_path).name}",
            "source": repo_info["url"],
            "path": rel_path,
            "text": content[:2000],
            "tags": tags + ["source-code"],
        }

        try:
            memory_engine.write(soul, "code_memory", entry_content, tags=tags + ["source-code"])
            indexed += 1
        except Exception as e:
            errors += 1

    return {"indexed": indexed, "errors": errors, "scan": scan}


def pull_owasp_content(output_dir: str, memory_engine) -> dict:
    owasp_url = "https://owasp.org/www-project-agent-security-top-10/"
    owasp_dir = os.path.join(output_dir, "owasp_agent_top10")
    os.makedirs(owasp_dir, exist_ok=True)

    indexed = 0

    try:
        import urllib.request
        req = urllib.request.Request(owasp_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        text = html
        for tag in ["script", "style", "nav", "header", "footer"]:
            while f"<{tag}" in text and f"</{tag}>" in text:
                start = text.find(f"<{tag}")
                end = text.find(f"</{tag}>") + len(f"</{tag}>")
                if start < end:
                    text = text[:start] + text[end:]
                else:
                    break

        import re
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > 100:
            entry_content = {
                "topic": "[OWASP] Agent Security Top 10",
                "source": owasp_url,
                "text": text[:5000],
                "tags": ["owasp", "agent-security", "threat-model", "harness", "baseline"],
            }
            memory_engine.write("montesquieu", "knowledge", entry_content,
                                tags=["owasp", "agent-security"])
            indexed += 1
            print(f"  OK: OWASP content indexed ({len(text)} chars)")

    except Exception as e:
        print(f"  FAIL: OWASP fetch failed: {e}")

    return {"indexed": indexed}


def _index_group(repos, root_dir, memory, group_name, skip_clone=False):
    total_indexed = 0
    total_repos = 0
    repo_stats = {}

    print(f"\n{'='*40}")
    print(f"  {group_name.upper()}")
    print(f"{'='*40}")
    os.makedirs(root_dir, exist_ok=True)

    for repo_info in repos:
        repo_name = repo_info["name"]
        repo_url = repo_info["url"]
        dest = os.path.join(root_dir, repo_name)

        print(f"\n  [{repo_name}] {repo_url}")

        if not skip_clone:
            cloned = clone_repo(repo_url, dest, depth=1)
        else:
            cloned = os.path.exists(dest) and os.path.isdir(dest)
            if cloned:
                print(f"  SKIP (index-only, exists): {dest}")
            else:
                print(f"  SKIP (not found): {dest}")

        if cloned and os.path.exists(dest):
            stats = index_repo_to_memory(repo_info, dest, memory)
            total_indexed += stats["indexed"]
            total_repos += 1
            repo_stats[repo_name] = {
                "cloned": True,
                "indexed": stats["indexed"],
                "errors": stats["errors"],
                "files": stats["scan"]["total"],
                "code": len(stats["scan"]["code_files"]),
                "docs": len(stats["scan"]["doc_files"]),
            }
            print(f"  Indexed: {stats['indexed']} entries "
                  f"({stats['scan']['total']} files: "
                  f"{len(stats['scan']['code_files'])} code, "
                  f"{len(stats['scan']['doc_files'])} docs)")
        else:
            repo_stats[repo_name] = {"cloned": False, "indexed": 0}
            print(f"  Not indexed (clone failed)")

    return total_indexed, total_repos, repo_stats


def main():
    print("=" * 60)
    print("VORTEX FLAME Knowledge Base Puller + Indexer")
    print("Harness + OpenClaw + MCP + Skill Engineering")
    print("=" * 60)

    from soul_memory import SoulMemoryEngine
    memory = SoulMemoryEngine()

    grand_indexed = 0
    grand_repos = 0
    all_stats = {}

    for group_name, repos in REPOS.items():
        group_dir = os.path.join(KB_ROOT, group_name)
        idx, cnt, stats = _index_group(repos, group_dir, memory, f"Harness/{group_name}", skip_clone=True)
        grand_indexed += idx
        grand_repos += cnt
        all_stats.update(stats)

    idx, cnt, stats = _index_group(MCP_REPOS, MCP_ROOT, memory, "MCP", skip_clone=True)
    grand_indexed += idx
    grand_repos += cnt
    all_stats.update(stats)

    idx, cnt, stats = _index_group(SKILL_REPOS, SKILL_ROOT, memory, "Skill", skip_clone=True)
    grand_indexed += idx
    grand_repos += cnt
    all_stats.update(stats)

    print(f"\n--- OWASP ---")
    owasp_stats = pull_owasp_content(KB_ROOT, memory)
    grand_indexed += owasp_stats["indexed"]

    stats_path = os.path.join(KB_ROOT, "pull_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_repos": grand_repos,
            "total_indexed": grand_indexed,
            "repos": all_stats,
            "owasp": owasp_stats,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: {grand_repos} repos, {grand_indexed} entries indexed")
    print(f"Stats: {stats_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
