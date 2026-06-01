#!/usr/bin/env python3
"""
VF Knowledge Filter — 知识质量分类器
======================================
自动判定一段候选知识是否值得沉淀到 kb_harness/。

分类标签：
  GENERALIZABLE    — 跨项目通用，领域无关，可安全入库
  PROJECT_SPECIFIC — 仅在 VORTEX_FLAME 上下文中有意义
  NOISE            — 工程噪音，不应入库（类型注释/git操作/格式修复）

判定维度：
  1. DOMAIN        — 涉及深度学习/算法/系统工程 vs 仅本项目配置
  2. REUSABILITY   — 代码模式能否直接复制到其他项目？
  3. ABSTRACTION   — 是否已去除项目特定的文件路径/变量名？
  4. ACTIONABILITY  — 读完这条知识后能否立即动手操作？

用法：
  python vf_knowledge_filter.py <candidate.json>           # 分析并输出清洗后结果
  python vf_knowledge_filter.py --check <entry.json>       # 单条判定
  python vf_knowledge_filter.py --scan kb_harness/         # 扫描已有知识库质量
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent

GENERALIZABLE_DOMAINS = {
    "deep_learning.architecture",
    "deep_learning.training",
    "deep_learning.optimization",
    "knowledge_engineering.domain_modeling",
    "knowledge_engineering.data_architecture",
    "knowledge_engineering.knowledge_graph",
    "ai_engineering.agent_design",
    "ai_engineering.rag",
    "systems.distributed",
    "systems.mcp",
    "engineering.patterns",
}

PROJECT_SPECIFIC_DOMAINS = {
    "engineering.type_systems",
    "engineering.code_quality",
    "project.config",
    "project.deployment",
}

NOISE_DOMAINS = {
    "engineering.linting",
    "engineering.formatting",
    "git.operations",
}

GENERALIZABLE_KEYWORDS = [
    "gradient flow", "initialization", "architecture pattern",
    "loss function", "optimization", "attention mechanism",
    "embedding", "transformer", "encoder", "decoder",
    "knowledge graph", "entity resolution", "schema design",
    "agent capability", "tool use", "multi-agent",
    "retrieval strategy", "vector search", "semantic similarity",
    "data pipeline", "feature engineering",
    "causal", "probabilistic", "bayesian",
]

NOISE_KEYWORDS = [
    "pyright", "type ignore", "type: ignore", "basedpyright",
    "pylance", "mypy", "lint", "linter",
    "unused import", "unused variable", "formatting",
    "indentation", "whitespace", "line ending",
    "git add", "git commit", "git push", "git rm",
    "staging", "rebase", "merge conflict",
]

CLIENT_INDICATORS = [
    "客户", "甲方", "需求方", "项目方",
    "公司名称", "企业名称", "联系人",
    "合同", "报价", "发票",
    "公司地址", "联系电话",
]


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    reasons: List[str] = field(default_factory=list)


def classify_entry(entry: Dict[str, Any]) -> ClassificationResult:
    reasons = []
    noise_score = 0.0
    gen_score = 0.0

    entry_id = entry.get("id", "")
    domain = entry.get("domain", "")
    topic = entry.get("topic", "")
    problem = entry.get("problem") or entry.get("summary") or ""
    solution = entry.get("solution") or entry.get("design_pattern") or entry.get("code_pattern_description") or ""
    code_pattern = entry.get("code_pattern", "")

    if domain in NOISE_DOMAINS:
        noise_score += 0.6
        reasons.append(f"domain {domain} is classified as NOISE")

    if domain in PROJECT_SPECIFIC_DOMAINS:
        noise_score += 0.3
        reasons.append(f"domain {domain} is project-specific")

    if domain in GENERALIZABLE_DOMAINS:
        gen_score += 0.5
        reasons.append(f"domain {domain} is generalizable")

    content_text = " ".join([
        topic, problem, solution,
        str(code_pattern) if isinstance(code_pattern, str) else "",
    ]).lower()

    noise_hits = sum(1 for kw in NOISE_KEYWORDS if kw in content_text)
    gen_hits = sum(1 for kw in GENERALIZABLE_KEYWORDS if kw in content_text)

    if noise_hits > 0:
        noise_score += min(noise_hits * 0.15, 0.5)
        reasons.append(f"{noise_hits} noise keywords found")

    if gen_hits > 0:
        gen_score += min(gen_hits * 0.1, 0.5)
        reasons.append(f"{gen_hits} generalizable keywords found")

    for indicator in CLIENT_INDICATORS:
        if indicator in content_text:
            noise_score += 0.8
            reasons.append(f"contains client indicator: '{indicator}'")
            break

    if code_pattern and "pyright" not in code_pattern and "type:" not in code_pattern:
        gen_score += 0.2
        reasons.append("has clean code pattern")

    if not problem or len(problem) < 10:
        gen_score -= 0.15
        reasons.append("short or missing problem/summary")
    else:
        gen_score += 0.05

    if not solution or len(solution) < 10:
        gen_score -= 0.15
        reasons.append("short or missing solution/pattern")
    else:
        gen_score += 0.05

    if len(topic) > 5 and len(problem) > 20 and len(solution) > 20:
        gen_score += 0.15
        reasons.append("well-structured entry (topic + problem + solution)")

    vf_specific = sum(1 for kw in ["vortex_flame", "cajepa", "train_ajepa"] if kw in content_text)
    if vf_specific > 2:
        noise_score += 0.1
        reasons.append(f"{vf_specific} project-specific references")

    if noise_score > 0.5:
        return ClassificationResult("NOISE", min(noise_score, 1.0), reasons)
    elif noise_score > gen_score:
        return ClassificationResult("PROJECT_SPECIFIC", max(gen_score / (noise_score + gen_score + 0.01), 0.3), reasons)
    elif gen_score > 0.3:
        return ClassificationResult("GENERALIZABLE", min(gen_score, 1.0), reasons)
    else:
        return ClassificationResult("PROJECT_SPECIFIC", 0.4, reasons + ["insufficient signal for classification"])


def filter_knowledge_file(input_path: Path) -> Dict[str, Any]:
    if not input_path.exists():
        return {"error": f"file not found: {input_path}"}

    with open(input_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        entries = [entries]

    results = {
        "file": str(input_path),
        "total": len(entries),
        "generalizable": [],
        "project_specific": [],
        "noise": [],
        "summary": {"GENERALIZABLE": 0, "PROJECT_SPECIFIC": 0, "NOISE": 0},
    }

    for entry in entries:
        classification = classify_entry(entry)
        results["summary"][classification.label] += 1
        item = {
            "id": entry.get("id", "unknown"),
            "topic": entry.get("topic", "")[:80],
            "confidence": round(classification.confidence, 2),
            "reasons": classification.reasons,
        }
        if classification.label == "GENERALIZABLE":
            results["generalizable"].append(item)
        elif classification.label == "PROJECT_SPECIFIC":
            results["project_specific"].append(item)
        else:
            results["noise"].append(item)

    return results


def scan_knowledge_dir(kb_dir: Path) -> Dict[str, Any]:
    results = {"directory": str(kb_dir), "files": [], "total_summary": {"GENERALIZABLE": 0, "PROJECT_SPECIFIC": 0, "NOISE": 0}}

    for fp in sorted(kb_dir.glob("*.json")):
        if "filter_report" in fp.name:
            continue
        file_result = filter_knowledge_file(fp)
        results["files"].append(file_result)
        for label in ["GENERALIZABLE", "PROJECT_SPECIFIC", "NOISE"]:
            results["total_summary"][label] += file_result["summary"].get(label, 0)

    return results


def print_report(report: Dict[str, Any]):
    print("=" * 60)
    print("KNOWLEDGE QUALITY REPORT")
    print("=" * 60)

    if "file" in report:
        print(f"\nFile: {report['file']}")
        print(f"Total entries: {report['total']}")
        print(f"  GENERALIZABLE:    {report['summary']['GENERALIZABLE']}")
        print(f"  PROJECT_SPECIFIC: {report['summary']['PROJECT_SPECIFIC']}")
        print(f"  NOISE:            {report['summary']['NOISE']}")

        if report["generalizable"]:
            print(f"\n--- GENERALIZABLE ({len(report['generalizable'])}) ---")
            for item in report["generalizable"]:
                print(f"  ✅ [{item['id']}] {item['topic']} (conf={item['confidence']:.2f})")

        if report["project_specific"]:
            print(f"\n--- PROJECT_SPECIFIC ({len(report['project_specific'])}) ---")
            for item in report["project_specific"]:
                print(f"  ⚠ [{item['id']}] {item['topic']} (conf={item['confidence']:.2f})")

        if report["noise"]:
            print(f"\n--- NOISE ({len(report['noise'])}) ---")
            for item in report["noise"]:
                print(f"  🗑 [{item['id']}] {item['topic']} — {', '.join(item['reasons'][:2])}")

    elif "directory" in report:
        print(f"\nDirectory: {report['directory']}")
        ts = report["total_summary"]
        print(f"Total across all files: GENERALIZABLE={ts['GENERALIZABLE']}, PROJECT_SPECIFIC={ts['PROJECT_SPECIFIC']}, NOISE={ts['NOISE']}")
        for fr in report["files"]:
            if "error" in fr:
                continue
            print(f"\n  {fr['file']}: {fr['total']} entries → G={fr['summary']['GENERALIZABLE']} P={fr['summary']['PROJECT_SPECIFIC']} N={fr['summary']['NOISE']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VF Knowledge Filter")
    parser.add_argument("target", nargs="?", help="JSON file or directory to analyze")
    parser.add_argument("--check", type=str, help="check a single entry JSON string")
    parser.add_argument("--scan", type=str, help="scan a kb_harness directory")
    parser.add_argument("--output", type=str, help="save cleaned (generalizable only) to file")
    args = parser.parse_args()

    target_path = args.target or args.scan

    if args.check:
        try:
            entry = json.loads(args.check)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            sys.exit(1)
        result = classify_entry(entry)
        print(f"Classification: {result.label} (confidence={result.confidence:.2f})")
        for r in result.reasons:
            print(f"  → {r}")
        sys.exit(0)

    if not target_path:
        parser.print_help()
        sys.exit(1)

    tp = Path(target_path).resolve()

    if not tp.exists():
        print(f"Path not found: {tp}")
        sys.exit(1)

    if tp.is_dir():
        report = scan_knowledge_dir(tp)
    else:
        report = filter_knowledge_file(tp)

    print_report(report)

    if args.output and tp.is_file():
        with open(tp, "r", encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            entries = [entries]
        clean = [e for e in entries if classify_entry(e).label == "GENERALIZABLE"]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(clean)} GENERALIZABLE entries to {args.output}")
