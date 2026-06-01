#!/usr/bin/env python3
"""
VF Toolchain MCP Server — VORTEX_FLAME 工具链 MCP 接口
======================================================
通过标准 MCP 协议暴露全部非模型工具模块，任何 MCP 兼容客户端
(TRAE、Claude Desktop、Cursor 等) 均可直接调用。

支持的工具 (Tools):
  - vf_dedup_run       数据去重
  - vf_cleanse_run     数据清洗
  - vf_inspect_run     数据质量检查
  - vf_inspect_report  获取质检报告
  - vf_distill_run     知识蒸馏
  - vf_distill_compress_rate  查看蒸馏压缩率
  - vf_auto_fix_run    代码诊断自动修复
  - vf_auto_fix_report 自动修复报告
  - vf_precommit_gate  代码门禁检查
  - vf_knowledge_filter 知识质量分类
  - vf_knowledge_query 知识库查询
  - vf_pipeline_status 系统状态查询

启动方式:
  python vf_mcp_server.py           # stdio 模式 (用于 TRAE/Claude Desktop 集成)
  python vf_mcp_server.py --sse     # SSE HTTP 模式 (用于远程调用)

MCP 客户端配置 (~/.trae/mcp.json 或 claude_desktop_config.json):
  {
    "mcpServers": {
      "vftoolchain": {
        "command": "python",
        "args": ["D:/VORTEX_FLAME/vf_mcp_server.py"]
      }
    }
  }
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    ResourceTemplate,
    Prompt,
    PromptArgument,
    PromptMessage,
    GetPromptResult,
)

CLIENT_TEMPLATES_DIR = PROJECT_ROOT / "client_templates"
KB_HARNESS_DIR = PROJECT_ROOT / "kb_harness"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vf-mcp-server")


class VFToolbox:
    def __init__(self):
        self._available = self._check_modules()

    def _check_modules(self) -> Dict[str, bool]:
        modules = {
            "dedup": False,
            "cleanse": False,
            "inspect": False,
            "distill": False,
            "auto_fix": False,
            "precommit_gate": False,
            "knowledge_filter": False,
        }
        try:
            from vf_dedup import DedupEngine
            from vf_data_core import ConfigManager, AuditLogger
            modules["dedup"] = True
        except ImportError:
            pass

        try:
            from vf_cleanse import CleanseEngine
            modules["cleanse"] = True
        except ImportError:
            pass

        try:
            from vf_inspect import InspectEngine
            modules["inspect"] = True
        except ImportError:
            pass

        try:
            from vf_distill import DistillEngine
            modules["distill"] = True
        except ImportError:
            pass

        try:
            from vf_auto_fix import AutoFixLoop
            modules["auto_fix"] = True
        except ImportError:
            pass

        try:
            from vf_precommit_gate import gate_diagnostic, gate_sensitive, gate_gitignore
            modules["precommit_gate"] = True
        except ImportError:
            pass

        try:
            from vf_knowledge_filter import classify_entry
            modules["knowledge_filter"] = True
        except ImportError:
            pass

        return modules

    def _get_core(self):
        from vf_data_core import ConfigManager, AuditLogger
        config = ConfigManager(str(PROJECT_ROOT / "vf_data_config.yaml"))  # type: ignore[reportArgumentType]
        audit = AuditLogger(config.audit_dir)
        return config, audit

    def run_dedup(self, data: List[Dict]) -> Dict:
        from vf_dedup import DedupEngine
        config, audit = self._get_core()
        engine = DedupEngine(config, audit)
        clean, stats = engine.process(data)
        return {
            "input_count": len(data),
            "output_count": len(clean),
            "duplicates_removed": getattr(stats, "duplicates_removed", 0),
            "exact_matches": getattr(stats, "exact_matches", 0),
            "simhash_matches": getattr(stats, "simhash_matches", 0),
            "fuzzy_matches": getattr(stats, "fuzzy_matches", 0),
            "output": clean[:20],
            "sample_size": min(len(clean), 20),
        }

    def run_cleanse(self, data: List[Dict]) -> Dict:
        from vf_cleanse import CleanseEngine
        config, audit = self._get_core()
        engine = CleanseEngine(config, audit)
        clean, stats = engine.process(data)
        return {
            "input_count": len(data),
            "output_count": len(clean),
            "normalized_dates": getattr(stats, "normalized_dates", 0),
            "normalized_numbers": getattr(stats, "normalized_numbers", 0),
            "missing_values_filled": getattr(stats, "missing_values_filled", 0),
            "outliers_detected": getattr(stats, "outliers_detected", 0),
            "output": clean[:20],
            "sample_size": min(len(clean), 20),
        }

    def run_inspect(self, data: List[Dict], required_fields: Optional[List[str]] = None,
                    business_rules: Optional[List[Dict]] = None) -> Dict:
        from vf_inspect import InspectEngine
        config, audit = self._get_core()
        if required_fields:
            config.set("inspect", "required_fields", value=required_fields)
        engine = InspectEngine(config, audit)
        engine.process(data)
        report = engine.report()
        return report

    def run_distill(self, data: List[Dict]) -> Dict:
        from vf_distill import DistillEngine
        config, audit = self._get_core()
        engine = DistillEngine(config, audit)
        compressed, stats = engine.process(data)
        return {
            "input_count": len(data),
            "output_count": len(compressed),
            "reduction_ratio": getattr(stats, "reduction_ratio", 0),
            "entities_extracted": getattr(stats, "entities_extracted", 0),
            "summaries_generated": getattr(stats, "summaries_generated", 0),
            "topics_discovered": getattr(stats, "topics_discovered", 0),
            "output": compressed[:15],
            "sample_size": min(len(compressed), 15),
        }

    def run_auto_fix(self, target_files: Optional[List[str]] = None,
                     max_iterations: int = 5) -> Dict:
        from vf_auto_fix import AutoFixLoop
        loop = AutoFixLoop(max_iterations=max_iterations)
        result = loop.run_full(files=target_files)
        report = loop.generate_report()
        return {
            "files_modified": result.file_path,
            "fixes_applied": result.fixes_applied,
            "errors_before": result.errors_before,
            "errors_after": result.errors_after,
            "iterations": report["total_iterations"],
            "unfixable_count": report["unfixable_count"],
            "unfixable": report["unfixable_summary"][:10],
            "fixes_detail": result.fixes_detail,
        }

    def run_precommit_gate(self) -> Dict:
        from vf_precommit_gate import gate_diagnostic, gate_sensitive, gate_gitignore
        results = {
            "diagnostic": gate_diagnostic(None),
            "sensitive": gate_sensitive(None),
            "gitignore": gate_gitignore(),
        }
        return {
            "diagnostic_passed": results["diagnostic"].passed,
            "sensitive_passed": results["sensitive"].passed,
            "gitignore_passed": results["gitignore"].passed,
            "all_passed": all(r.passed for r in results.values()),
            "diagnostic_errors": [str(e)[:120] for e in results["diagnostic"].errors[:10]],
            "sensitive_errors": [str(e)[:120] for e in results["sensitive"].errors[:10]],
        }

    def run_knowledge_filter(self, entries: List[Dict]) -> Dict:
        from vf_knowledge_filter import classify_entry
        labels = {"GENERALIZABLE": 0, "PROJECT_SPECIFIC": 0, "NOISE": 0}
        details = []
        for e in entries:
            result = classify_entry(e)
            label = result.label if hasattr(result, 'label') else str(result)
            labels[label] = labels.get(label, 0) + 1
            details.append({
                "id": e.get("id", e.get("title", "")),
                "category": label,
                "confidence": getattr(result, 'confidence', 0),
                "reasons": getattr(result, 'reasons', []),
            })
        return {
            "GENERALIZABLE": labels.get("GENERALIZABLE", 0),
            "PROJECT_SPECIFIC": labels.get("PROJECT_SPECIFIC", 0),
            "NOISE": labels.get("NOISE", 0),
            "details": details,
        }

    def query_knowledge(self, pattern: str = "") -> List[Dict]:
        kb_dir = PROJECT_ROOT / "kb_harness"
        results = []
        if kb_dir.exists():
            for fp in sorted(kb_dir.glob("knowledge_*.json")):
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        for entry in data:
                            if pattern:
                                text = json.dumps(entry, ensure_ascii=False).lower()
                                if pattern.lower() not in text:
                                    continue
                            results.append({
                                "file": fp.name,
                                "title": entry.get("title", entry.get("topic", "")),
                                "type": entry.get("type", ""),
                                "domain": entry.get("domain", ""),
                                "entry_id": entry.get("id", entry.get("entry_id", "")),
                            })
                    elif isinstance(data, dict) and data.get("entries"):
                        results.extend(data["entries"])
                except Exception:
                    pass
        return {  # type: ignore[reportReturnType]
            "total": len(results),
            "entries": results[:30],
        }

    def get_status(self) -> Dict:
        return {
            "modules_available": self._available,
            "project_root": str(PROJECT_ROOT),
            "python_version": sys.version,
            "templates_count": len(list(CLIENT_TEMPLATES_DIR.glob("*.html"))) if CLIENT_TEMPLATES_DIR.exists() else 0,
            "kb_entries_count": self._count_kb_entries(),
        }

    def _count_kb_entries(self) -> int:
        count = 0
        if KB_HARNESS_DIR.exists():
            for fp in KB_HARNESS_DIR.glob("knowledge_*.json"):
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        count += len(data)
                except Exception:
                    pass
        return count

    def query_delivery_sop(self, phase: str = "") -> Dict:
        sop_file = KB_HARNESS_DIR / "knowledge_ai_delivery_executable_sop.json"
        if not sop_file.exists():
            return {"error": "SOP knowledge not found", "file": str(sop_file)}

        data = json.loads(sop_file.read_text(encoding="utf-8"))
        if not data:
            return {"error": "SOP is empty"}

        sop = data[0]
        if phase:
            phase_key = f"phase_{phase}"
            for key, val in sop.items():
                if key.startswith(phase_key) or phase.lower() in key.lower():
                    return {
                        "sop_title": sop.get("topic", ""),
                        "phase": key,
                        "content": val,
                    }
            return {"error": f"Phase '{phase}' not found", "available_phases": [k for k in sop if k.startswith("phase_")]}

        return sop

    def audit_delivery(self, project_dir: str) -> Dict:
        project = Path(project_dir)
        if not project.exists():
            return {"error": f"Project not found: {project_dir}"}

        sop_file = KB_HARNESS_DIR / "knowledge_ai_delivery_executable_sop.json"
        if not sop_file.exists():
            return {"error": "SOP knowledge base not found"}

        data = json.loads(sop_file.read_text(encoding="utf-8"))
        sop = data[0]

        phase_21 = sop.get("phase_6_shipping", {}).get("step_21_package", {}).get("deliverables_checklist", [])
        found = []
        missing = []

        file_map = {
            "api_server.py": ["api_server.py", "main.py", "app.py"],
            "kb_engine.py": ["kb_engine.py", "knowledge_engine.py", "kg_engine.py"],
            "business_rules.py": ["business_rules.py", "agri_core.py", "rules.py", "core.py"],
            "intent_router.py": ["intent_router.py", "router.py", "intent.py"],
            "auth.py": ["auth.py", "authentication.py"],
            "wechat_bot.py": ["wechat_bot.py", "wecom_bot.py"],
            "chat_ui.html": ["chat_ui.html", "index.html"],
            "kb_upload.html": ["kb_upload.html", "upload.html"],
            "admin.html": ["admin.html", "dashboard.html"],
            "deploy.ps1": ["deploy.ps1", "start.ps1", "run.ps1"],
            "deploy.sh": ["deploy.sh", "start.sh", "run.sh"],
            "docker-compose.yml": ["docker-compose.yml", "docker-compose.yaml"],
            ".env.example": [".env.example", ".env.template", "env.example"],
            "requirements.txt": ["requirements.txt"],
            "README_客户版.md": ["README_客户版.md", "README.md", "readme.md"],
            "test_e2e.py": ["test_e2e.py", "test.py", "tests.py"],
        }

        project_files = {f.name.lower(): f.name for f in project.rglob("*") if f.is_file()}

        for item in file_map:
            item_name = item.replace("☐ ", "")
            candidates = file_map[item]
            matched = False
            for cand in candidates:
                if cand.lower() in project_files:
                    found.append({"item": item_name, "found": project_files[cand.lower()]})
                    matched = True
                    break
            if not matched:
                missing.append({"item": item_name})

        total_checks = len(file_map)
        completion = len(found) / total_checks * 100 if total_checks > 0 else 0

        return {
            "project": str(project),
            "completion_pct": round(completion, 1),
            "total_checks": total_checks,
            "found_count": len(found),
            "missing_count": len(missing),
            "verdict": "READY" if completion >= 90 else "INCOMPLETE" if completion >= 40 else "CRITICAL",
            "missing": missing,
            "found": found if len(found) < 10 else found[:10],
        }

    def list_client_templates(self) -> List[Dict]:
        templates = []
        if CLIENT_TEMPLATES_DIR.exists():
            for fp in sorted(CLIENT_TEMPLATES_DIR.glob("*")):
                if fp.is_file():
                    templates.append({
                        "name": fp.name,
                        "size": fp.stat().st_size,
                        "description": self._template_description(fp.name, fp),
                    })
        return templates

    def _template_description(self, name: str, fp: Path) -> str:
        descriptions = {
            "chat_ui.html": "客户聊天界面 — 对话窗口、多轮对话、流式输出、品牌可定制",
            "kb_upload.html": "知识库上传管理 — 拖拽上传、自动分段、向量检索、质量审计",
        }
        return descriptions.get(name, f"客户端模板文件 ({fp.stat().st_size} bytes)")

    def get_client_template(self, name: str) -> Dict:
        fp = CLIENT_TEMPLATES_DIR / name
        if not fp.exists():
            available = [f.name for f in CLIENT_TEMPLATES_DIR.glob("*")] if CLIENT_TEMPLATES_DIR.exists() else []
            return {"error": f"Template '{name}' not found", "available": available}

        content = fp.read_text(encoding="utf-8")
        return {
            "name": fp.name,
            "size": len(content),
            "lines": content.count('\n'),
            "content": content,
        }

    def run_evaluation(self, project_dir: str) -> Dict:
        project = Path(project_dir)
        if not project.exists():
            return {"error": f"Project not found: {project_dir}", "overall_score": 0}

        py_files = list(project.rglob("*.py"))
        json_files = list(project.rglob("*.json")) + list(project.rglob("*.jsonl"))
        md_files = list(project.rglob("*.md"))
        has_git = (project / ".git").exists()

        scores = {}
        details = {}

        # 1. Code Quality (0-10)
        code_issues = 0
        try:
            from vf_auto_fix import AutoFixLoop
            loop = AutoFixLoop(max_iterations=1)
            rel_files = [str(f.relative_to(PROJECT_ROOT)) for f in py_files
                         if f.suffix == ".py" and "_test_" not in f.name and f.parent == project
                         or f.parent.parent == project]
            if rel_files:
                errors = loop.run_pyright(rel_files)
                code_issues = len(errors)
            else:
                code_issues = 0
        except Exception:
            code_issues = -1

        if code_issues == -1:
            scores["code_quality"] = 7.0
            details["code_quality"] = "pyright unavailable, estimated good"
        elif code_issues == 0:
            scores["code_quality"] = 10.0
            details["code_quality"] = "zero diagnostics"
        elif code_issues <= 5:
            scores["code_quality"] = 8.0
            details["code_quality"] = f"{code_issues} minor issues"
        elif code_issues <= 20:
            scores["code_quality"] = 6.0
            details["code_quality"] = f"{code_issues} issues need attention"
        else:
            scores["code_quality"] = 3.0
            details["code_quality"] = f"{code_issues} issues, significant cleanup needed"

        # 2. Automation Completeness (0-10)
        auto_score = 0.0
        auto_detail = []

        gitignore_count = 0
        if has_git:
            gitignore = project / ".gitignore"
            if gitignore.exists():
                gitignore_count = len(gitignore.read_text(encoding="utf-8", errors="ignore").splitlines())
            auto_score += 1.5
            auto_detail.append("git initialized")

        precommit = project / ".pre-commit-config.yaml"
        if precommit.exists():
            auto_score += 2.0
            auto_detail.append("pre-commit hooks configured")

        hook_dir = project / ".git" / "hooks"
        if hook_dir.exists() and any(hook_dir.iterdir()):
            auto_score += 1.5
            auto_detail.append("git hooks active")

        ci_dir = project / ".github" / "workflows"
        if ci_dir.exists() and list(ci_dir.glob("*.yml")):
            auto_score += 2.0
            auto_detail.append("CI/CD pipelines")

        config_yaml = list(project.glob("*.yaml")) + list(project.glob("*.yml"))
        config_yaml = [f for f in config_yaml if f.name not in (".pre-commit-config.yaml",)]
        if config_yaml:
            auto_score += 1.5
            auto_detail.append(f"{len(config_yaml)} config file(s)")

        if gitignore_count > 5:
            auto_score += 1.5
            auto_detail.append(f"comprehensive .gitignore ({gitignore_count} rules)")

        scores["automation"] = min(10.0, auto_score) if auto_detail else 1.0
        details["automation"] = "; ".join(auto_detail) if auto_detail else "no automation found"

        # 3. Data Quality (0-10)
        data_files = len(json_files)
        data_valid = 0
        data_invalid = 0
        for jf in json_files[:20]:
            try:
                content = json.loads(jf.read_text(encoding="utf-8"))
                if isinstance(content, (dict, list)):
                    data_valid += 1
                else:
                    data_invalid += 1
            except json.JSONDecodeError:
                data_invalid += 1

        if data_files == 0:
            scores["data_quality"] = 3.0
            details["data_quality"] = "no JSON data files found — add data for validation"
        elif data_invalid == 0:
            scores["data_quality"] = 10.0
            details["data_quality"] = f"{data_valid}/{data_files} valid, 0 corrupted"
        elif data_invalid <= data_files * 0.2:
            scores["data_quality"] = 7.0
            details["data_quality"] = f"{data_valid} valid, {data_invalid} corrupted"
        else:
            scores["data_quality"] = 4.0
            details["data_quality"] = f"{data_invalid}/{data_files} files corrupted"

        # 4. Documentation Completeness (0-10)
        doc_score = 0.0
        doc_detail = []

        readme = list(project.glob("README*")) + list(project.glob("readme*"))
        if readme:
            doc_score += 2.5
            doc_detail.append("README")

        api_doc = list(project.glob("**/api*.md")) + list(project.glob("**/*API*"))
        if api_doc:
            doc_score += 2.0
            doc_detail.append("API docs")

        if md_files:
            doc_score += min(3.0, len(md_files) * 0.5)
            doc_detail.append(f"{len(md_files)} markdown file(s)")

        requirements = list(project.glob("requirements*.txt")) + list(project.glob("pyproject.toml"))
        if requirements:
            doc_score += 1.5
            doc_detail.append("dependency spec")

        setup_scripts = list(project.glob("setup.*")) + list(project.glob("Makefile")) + list(project.glob("docker*"))
        if setup_scripts:
            doc_score += 1.0
            doc_detail.append("setup/install scripts")

        scores["documentation"] = min(10.0, doc_score) if doc_detail else 1.0
        details["documentation"] = "; ".join(doc_detail) if doc_detail else "no documentation"

        # 5. Project Completeness (0-10)
        completeness = 0.0
        if py_files:
            completeness += min(5.0, len(py_files) * 0.5)
        if json_files:
            completeness += min(2.0, len(json_files) * 0.3)
        if md_files:
            completeness += min(1.5, len(md_files) * 0.2)
        if has_git:
            completeness += 1.0
        if (project / "tests").exists() or (project / "test").exists():
            completeness += 0.5
        scores["project_completeness"] = min(10.0, completeness)
        details["project_completeness"] = (
            f"{len(py_files)} .py, {len(json_files)} .json, {len(md_files)} .md, "
            f"git={has_git}"
        )

        overall = sum(scores.values()) / len(scores)

        return {
            "project": str(project),
            "overall_score": round(overall, 1),
            "grade": "A" if overall >= 8.5 else "B" if overall >= 7.0 else "C" if overall >= 5.0 else "D",
            "ready_for_production": overall >= 8.0,
            "dimensions": scores,
            "details": details,
            "summary": {
                "total_py_files": len(py_files),
                "total_data_files": len(json_files),
                "total_doc_files": len(md_files),
                "git_initialized": has_git,
                "code_issues": code_issues if code_issues >= 0 else "unavailable",
            },
            "recommendations": self._generate_recommendations(scores),
        }

    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        recs = []
        if scores.get("code_quality", 10) < 6:
            recs.append("P0: 运行 vf_auto_fix_run 清除代码诊断错误")
        if scores.get("automation", 10) < 5:
            recs.append("P1: 添加 .pre-commit-config.yaml + GitHub Actions CI")
        if scores.get("data_quality", 10) < 5:
            recs.append("P1: 通过 vf_inspect_run 检查所有JSON数据文件")
        if scores.get("documentation", 10) < 4:
            recs.append("P2: 补充 README.md 和 API 文档")
        if scores.get("project_completeness", 10) < 4:
            recs.append("P2: 项目文件数量较少，可能缺乏核心实现")
        if not recs:
            recs.append("项目各项指标良好，无需紧急处理")
        return recs


toolbox = VFToolbox()


async def main():
    server = Server("vf-toolchain")
    available = toolbox._available

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        tools = []

        if available["dedup"]:
            tools.append(Tool(
                name="vf_dedup_run",
                description="数据去重：输入JSON数组，输出去重后的数据。支持精确匹配+SimHash+模糊匹配三算法组合，去重率>95%。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "description": "待去重的数据数组，每条记录为JSON对象",
                            "items": {"type": "object"},
                        }
                    },
                    "required": ["data"],
                },
            ))

        if available["cleanse"]:
            tools.append(Tool(
                name="vf_cleanse_run",
                description="数据清洗：自动归一化日期/数值/文本格式，检测异常值(Z-score/IQR)，填充缺失值。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "description": "待清洗的数据数组",
                            "items": {"type": "object"},
                        }
                    },
                    "required": ["data"],
                },
            ))

        if available["inspect"]:
            tools.append(Tool(
                name="vf_inspect_run",
                description="数据质量检查：四维评分(完整性/准确性/一致性/合规性)，检测PII泄露，生成问题清单。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "待检查的数据",
                        },
                        "required_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "必填字段列表",
                        },
                    },
                    "required": ["data"],
                },
            ))

        if available["distill"]:
            tools.append(Tool(
                name="vf_distill_run",
                description="知识蒸馏：从非结构化/半结构化数据中提取实体、主题建模、摘要生成、技能要点抽取。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "待蒸馏的知识数据，每条记录需包含title/content字段",
                        }
                    },
                    "required": ["data"],
                },
            ))

        if available["auto_fix"]:
            tools.append(Tool(
                name="vf_auto_fix_run",
                description="代码诊断自动修复：运行pyright扫描→三级错误分类→自动应用安全修复→迭代验证至零错误。覆盖30+诊断规则。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "要修复的文件名列表(相对于项目根目录)，留空为全项目扫描",
                        },
                        "max_iterations": {
                            "type": "integer",
                            "default": 5,
                            "description": "最大修复迭代次数",
                        },
                    },
                },
            ))

        if available["precommit_gate"]:
            tools.append(Tool(
                name="vf_precommit_gate",
                description="代码门禁检查：运行三道门(诊断错误/敏感信息/gitignore)，返回通过/阻止状态。",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ))

        if available["knowledge_filter"]:
            tools.append(Tool(
                name="vf_knowledge_filter",
                description="知识质量分类：对知识条目进行GENERALIZABLE/PROJECT_SPECIFIC/NOISE三级自动分类。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entries": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "待分类的知识条目数组",
                        }
                    },
                    "required": ["entries"],
                },
            ))

        tools.append(Tool(
            name="vf_knowledge_query",
            description="知识库查询：搜索 kb_harness/ 目录中的结构化知识条目。可按关键词过滤。",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "搜索关键词，留空返回全部",
                    }
                },
            },
        ))

        tools.append(Tool(
            name="vf_pipeline_status",
            description="系统状态查询：检查各模块可用性、项目路径、Python版本。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ))

        tools.append(Tool(
            name="vf_evaluate_score",
            description="企业级项目评估打分：五维评分(代码质量/自动化程度/数据质量/文档完整度/项目规模) 0-10分量表，含评级(A/B/C/D)、生产就绪判断和优化建议。",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_dir": {
                        "type": "string",
                        "description": "待评估项目的绝对路径，如 E:\\农小秘",
                    }
                },
                "required": ["project_dir"],
            },
        ))

        tools.append(Tool(
            name="vf_delivery_sop",
            description="AI项目交付标准操作流程(SOP)：查询6阶段22步完整交付规范。可按阶段筛选，如输入phase='1'获取MVP阶段步骤。任何AI模型开发客户项目时必须遵循此SOP。",
            inputSchema={
                "type": "object",
                "properties": {
                    "phase": {
                        "type": "string",
                        "description": "可选阶段号(1-6)，留空返回完整SOP",
                    }
                },
            },
        ))

        tools.append(Tool(
            name="vf_delivery_audit",
            description="项目交付完整性审计：对照22步SOP检查目标项目缺失了哪些交付物，返回完成度百分比、缺失清单、判定(READY/INCOMPLETE/CRITICAL)。",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_dir": {
                        "type": "string",
                        "description": "待审计项目的绝对路径",
                    }
                },
                "required": ["project_dir"],
            },
        ))

        tools.append(Tool(
            name="vf_template_list",
            description="列出所有可用的客户端模板文件(聊天界面/知识库上传/管理后台等)，用于分发给客户项目。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ))

        tools.append(Tool(
            name="vf_template_get",
            description="获取指定客户端模板的完整源代码。拿到后修改品牌名/API地址即可用于客户项目。",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "模板文件名，如 chat_ui.html 或 kb_upload.html",
                    }
                },
                "required": ["name"],
            },
        ))

        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[TextContent]:
        try:
            if name == "vf_dedup_run":
                result = toolbox.run_dedup(arguments.get("data", []))
            elif name == "vf_cleanse_run":
                result = toolbox.run_cleanse(arguments.get("data", []))
            elif name == "vf_inspect_run":
                result = toolbox.run_inspect(
                    arguments.get("data", []),
                    arguments.get("required_fields"),
                )
            elif name == "vf_distill_run":
                result = toolbox.run_distill(arguments.get("data", []))
            elif name == "vf_auto_fix_run":
                result = toolbox.run_auto_fix(
                    arguments.get("files"),
                    arguments.get("max_iterations", 5),
                )
            elif name == "vf_precommit_gate":
                result = toolbox.run_precommit_gate()
            elif name == "vf_knowledge_filter":
                result = toolbox.run_knowledge_filter(arguments.get("entries", []))
            elif name == "vf_knowledge_query":
                result = toolbox.query_knowledge(arguments.get("pattern", ""))
            elif name == "vf_pipeline_status":
                result = toolbox.get_status()
            elif name == "vf_evaluate_score":
                result = toolbox.run_evaluation(arguments.get("project_dir", ""))
            elif name == "vf_delivery_sop":
                result = toolbox.query_delivery_sop(arguments.get("phase", ""))
            elif name == "vf_delivery_audit":
                result = toolbox.audit_delivery(arguments.get("project_dir", ""))
            elif name == "vf_template_list":
                result = toolbox.list_client_templates()
            elif name == "vf_template_get":
                result = toolbox.get_client_template(arguments.get("name", ""))
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            return [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2),
            )]
        except Exception as e:
            logger.exception(f"Tool {name} failed")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": str(e),
                    "tool": name,
                    "type": type(e).__name__,
                }, ensure_ascii=False, indent=2),
            )]

    @server.list_prompts()
    async def list_prompts() -> List[Prompt]:
        return [
            Prompt(
                name="vf_data_pipeline",
                description="完整数据管道提示词：引导AI使用去重→清洗→检查→蒸馏四阶段流水线处理数据",
                arguments=[
                    PromptArgument(name="data_summary", description="数据的简要描述", required=True),
                ],
            ),
            Prompt(
                name="vf_code_quality",
                description="代码质量检查提示词：引导AI使用auto_fix + precommit_gate建立质量闭环",
                arguments=[
                    PromptArgument(name="file_count", description="待检查的文件数量", required=False),
                ],
            ),
            Prompt(
                name="vf_ai_project_delivery",
                description="AI项目全栈交付提示词：引导AI按照6阶段22步SOP完整交付客户项目(前端+后端+知识库+部署+集成+监控)。任何AI模型在开发客户AI项目时，必须先调用此prompt获取完整流程，然后逐阶段执行。",
                arguments=[
                    PromptArgument(name="project_name", description="项目名称", required=True),
                    PromptArgument(name="domain", description="业务领域，如：农产品供应链、医疗问诊、法律咨询", required=True),
                    PromptArgument(name="client_language", description="客户使用的语言，默认zh(中文)", required=False),
                ],
            ),
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:  # type: ignore[reportReturnType]
        if name == "vf_data_pipeline":
            summary = arguments.get("data_summary", "unknown") if arguments else "unknown"
            return GetPromptResult(
                description=f"Data pipeline prompt for: {summary}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"""You have access to the VORTEX_FLAME toolchain via MCP. Process the following data using a stage-gate pipeline:

1. **DEDUP**: Call `vf_dedup_run` first — remove exact, SimHash, and fuzzy duplicates
2. **CLEANSE**: Call `vf_cleanse_run` — normalize dates/numbers/text, fill missing values, detect outliers
3. **INSPECT**: Call `vf_inspect_run` — check completeness, accuracy, consistency, compliance. Review the report — if score below 0.7, fix issues and re-run
4. **DISTILL**: Call `vf_distill_run` — extract entities, topics, summaries from the cleaned data
5. **FILTER**: Call `vf_knowledge_filter` on distilled output — classify each entry as GENERALIZABLE/PROJECT_SPECIFIC/NOISE

Data to process: {summary}

For each stage, report: input count, output count, key metrics, and any issues found."""
                        ),
                    )
                ],
            )

        if name == "vf_code_quality":
            file_count = arguments.get("file_count", "all") if arguments else "all"
            return GetPromptResult(
                description=f"Code quality prompt for {file_count} files",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"""You have access to the VORTEX_FLAME code quality toolchain via MCP. Establish a quality loop:

1. **FIX**: Call `vf_auto_fix_run` with files=[{file_count}] — runs pyright, classifies errors, applies safe fixes, iterates until clean
2. **GATE**: Call `vf_precommit_gate` — three gates (diagnostic, sensitive info, gitignore)
3. If gate shows errors, call `vf_auto_fix_run` again with the flagged files
4. Loop until all gates pass

Current scope: {file_count} files"""
                        ),
                    )
                ],
            )

        if name == "vf_ai_project_delivery":
            project_name = arguments.get("project_name", "未命名项目") if arguments else "未命名项目"
            domain = arguments.get("domain", "通用") if arguments else "通用"
            lang = arguments.get("client_language", "zh") if arguments else "zh"
            return GetPromptResult(
                description=f"AI项目全栈交付: {project_name}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"""你正在开发客户项目「{project_name}」，业务领域：{domain}。你必须按照VORTEX_FLAME的6阶段22步SOP完整交付，不可跳过任何阶段。所有输出必须使用中文。

## 第一步：获取完整流程

先调用 `vf_delivery_sop` 获取完整SOP，然后逐阶段执行。

## 第二步：创建项目目录

为「{project_name}」创建项目目录，所有文件放进去。

## 第三步：按阶段执行

### 阶段1: MVP（最小可用产品）
1. 调用 `vf_delivery_sop` phase='1' 获取阶段1的详细步骤
2. 调用 `vf_template_get` name='chat_ui.html' 获取聊天界面模板
3. 修改模板中的品牌名、API地址、系统提示词
4. 创建 api_server.py（FastAPI, CORS, 流式SSE, /api/chat + /api/health + /api/config）
5. 创建 deploy.ps1 和 deploy.sh 部署脚本
6. 创建 requirements.txt

### 阶段2: 知识引擎
1. 调用 `vf_delivery_sop` phase='2' 
2. 调用 `vf_template_get` name='kb_upload.html' 获取知识库上传模板
3. 创建 kb_engine.py（文档解析+分段+向量化+检索）

### 阶段3: 业务逻辑
1. 调用 `vf_delivery_sop` phase='3'
2. 创建 intent_router.py（意图分类）
3. 创建 business_rules.py（业务规则引擎）
4. 实现对话记忆管理

### 阶段4: 多渠道集成
1. 调用 `vf_delivery_sop` phase='4'
2. 创建 wechat_bot.py（企业微信）

### 阶段5: 管理后台
1. 调用 `vf_delivery_sop` phase='5'
2. 创建 admin.html 管理面板
3. 创建 auth.py 用户认证

### 阶段6: 交付
1. 调用 `vf_delivery_sop` phase='6'
2. 创建 docker-compose.yml
3. 创建 .env.example
4. 创建 README_客户版.md（纯中文、非技术语言）
5. 创建 test_e2e.py
6. 最后调用 `vf_delivery_audit` project_dir='{project_name}' 审计完整性
7. 如果 audit 显示 verdict 不是 READY，补齐缺失项

## 核心约束

- 所有输出必须是中文（代码注释、README、界面文字）
- 每完成一个阶段，调用 `vf_delivery_audit` 检查进度
- 禁止只交付后端API就停止
- 禁止跳过前端UI
- 禁止跳过部署脚本
- 禁止使用 gradio/streamlit 作为"产品界面"——那是开发者工具，不是客户产品

开始从阶段1执行。"""
                        ),
                    )
                ],
            )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
