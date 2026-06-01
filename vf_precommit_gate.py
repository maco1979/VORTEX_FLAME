#!/usr/bin/env python3
"""
VF Pre-Commit Gate — 代码门禁系统
====================================
在每次 git commit 之前自动执行三道检查，拦截脏代码进入仓库。

三道闸门：
  1. DIAGNOSTIC  — pyright 类型检查，Errors > 0 则拦截
  2. SENSITIVE   — 扫描暂存区文件，检测客户信息/密钥/敏感路径
  3. GITIGNORE   — 验证受保护目录未被意外跟踪

用法：
  python vf_precommit_gate.py           # 完整检查（拦截模式）
  python vf_precommit_gate.py --warn    # 仅警告不拦截
  python vf_precommit_gate.py --staged  # 仅检查 git staged 文件

退出码：
  0 = 全部通过
  1 = DIAGNOSTIC 拦截
  2 = SENSITIVE 拦截
  3 = GITIGNORE 拦截
  4 = 多个闸门拦截
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
PYRIGHT_CONFIG = PROJECT_ROOT / "pyrightconfig.json"

SKIP_FILES = {"_smoke_test.py", "_precache_mel.py"}
SKIP_PREFIXES = ("_test_", ".", "__")

GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"

PROTECTED_DIRS = [
    ".vf_memory",
    ".vf_world_cache",
    "soul_memory_store",
    "rag_knowledge_bases",
    "client_data",
    "客户提供需要改成通用知识库",
]

PROTECTED_FILE_PATTERNS = [
    "*.env",
    "*.key",
    "*.pem",
    "credentials*",
    "secrets*",
    "client_config*",
]

SENSITIVE_PATTERNS = [
    (re.compile(r"客户[A-Za-z]*[：:]"), "client_reference"),
    (re.compile(r"gh[pous]_[A-Za-z0-9]{36}"), "github_token"),
    (re.compile(r"sk-[A-Za-z0-9]{32,}"), "openai_key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_key"),
    (re.compile(r"-----BEGIN (RSA|EC|OPENSSH|DSA) PRIVATE KEY-----"), "private_key"),
    (re.compile(r"password\s*[=:]\s*[\"'][^\"']{4,}[\"']"), "hardcoded_password"),
    (re.compile(r"公司[：:]\s*\S"), "company_name_explicit"),
    (re.compile(r"联系人[：:]\s*\S"), "contact_person"),
    (re.compile(r"手机[号]?\s*[：:]\s*1[3-9]\d{9}"), "phone_number"),
    (re.compile(r"身份证[号]?\s*[：:]\s*\d{15,18}"), "id_number"),
    (re.compile(r"地址[：:]\s*\S{6,}"), "physical_address"),
]

NON_GENERALIZABLE_INDICATORS = [
    "client_name",
    "company_specific",
    "hardcoded_threshold",
    "one_off_fix",
    "pyright ignore",
    "type: ignore",
    "workaround",
    "todo_hack",
    "temporary",
]


class GateResult:
    def __init__(self, gate_name: str):
        self.name = gate_name
        self.passed = True
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "PASS" if self.passed else "BLOCKED"
        lines = [f"[{self.name}] {status}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


def get_changed_py_files() -> List[Path]:
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
    except Exception:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

    return [PROJECT_ROOT / f for f in lines if f.endswith(".py") and (PROJECT_ROOT / f).exists()]


def get_all_tracked_py_files() -> List[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "*.py"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
    except Exception:
        return list(PROJECT_ROOT.rglob("*.py"))

    return [PROJECT_ROOT / f for f in lines if (PROJECT_ROOT / f).exists()]


def gate_diagnostic(files: Optional[List[Path]] = None) -> GateResult:
    gate = GateResult("DIAGNOSTIC")

    targets = files or get_all_tracked_py_files()
    if not targets:
        gate.add_warning("no .py files found to check")
        return gate

    try:
        result = subprocess.run(
            ["pyright", "--outputjson", "--project", str(PYRIGHT_CONFIG)] + [str(f) for f in targets],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120,
        )
    except FileNotFoundError:
        gate.add_warning("pyright not installed, skipping static analysis")
        return gate
    except subprocess.TimeoutExpired:
        gate.add_error("pyright timed out (>120s)")
        return gate

    if result.returncode == 0:
        return gate

    try:
        data = json.loads(result.stdout)
        diagnostics = data.get("generalDiagnostics", [])
        errors = [d for d in diagnostics if d.get("severity") == "error"]
        if errors:
            for err in errors[:15]:
                fpath = err.get("file", "?")
                line = err.get("range", {}).get("start", {}).get("line", "?")
                msg = err.get("message", "?")[:100]
                gate.add_error(f"{fpath}:{line} — {msg}")
            if len(errors) > 15:
                gate.add_error(f"... and {len(errors) - 15} more errors ({len(errors)} total)")
    except (json.JSONDecodeError, KeyError):
        stderr_preview = result.stderr[:500] if result.stderr else ""
        gate.add_error(f"pyright failed with non-zero exit, output: {stderr_preview}")

    return gate


def gate_sensitive(files: Optional[List[Path]] = None) -> GateResult:
    gate = GateResult("SENSITIVE")

    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        staged = set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()
    except Exception:
        staged = set()

    targets = files or get_all_tracked_py_files()

    for fp in targets:
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for pattern, label in SENSITIVE_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                gate.add_error(f"{fp.relative_to(PROJECT_ROOT)}: detected {label} ({len(matches)} occurrences)")

    for fp_rel_str in staged:
        fp = PROJECT_ROOT / fp_rel_str
        if not fp.exists():
            continue
        for pdir in PROTECTED_DIRS:
            if fp_rel_str.startswith(pdir + "/") or fp_rel_str.startswith(pdir + "\\") or fp_rel_str == pdir:
                gate.add_error(f"{fp_rel_str} is in PROTECTED_DIR '{pdir}' but staged for commit")

    return gate


def gate_gitignore() -> GateResult:
    gate = GateResult("GITIGNORE")

    if not GITIGNORE_PATH.exists():
        gate.add_error(".gitignore not found")
        return gate

    ignore_content = GITIGNORE_PATH.read_text(encoding="utf-8")
    ignore_lines = ignore_content.split("\n")

    for pdir in PROTECTED_DIRS:
        found = any(line.strip().rstrip("/") == pdir for line in ignore_lines)
        if not found:
            gate.add_error(f"PROTECTED_DIR '{pdir}' not found in .gitignore")

    for pfile_pattern in PROTECTED_FILE_PATTERNS:
        found = any(line.strip() == pfile_pattern for line in ignore_lines)
        if not found:
            gate.add_warning(f"PROTECTED pattern '{pfile_pattern}' not found in .gitignore")

    return gate


def run_all_gates(files: Optional[List[Path]] = None, warn_only: bool = False,
                  auto_fix: bool = False) -> int:
    print("=" * 60)
    print("VF PRE-COMMIT GATE")
    print("=" * 60)

    results = [
        gate_diagnostic(files),
        gate_sensitive(files),
        gate_gitignore(),
    ]

    if auto_fix and results[0].errors:
        print("\n[GATE] Diagnostic errors found — triggering auto-fix loop...")
        try:
            import vf_auto_fix
            from vf_auto_fix import AutoFixLoop
            target_files = [str(f) for f in files] if files else None
            fix_loop = AutoFixLoop(max_iterations=5)
            fix_result = fix_loop.run_full(files=target_files)
            print(f"[GATE] Auto-fix: {fix_result.errors_before}→{fix_result.errors_after} ({fix_result.fixes_applied} fixes)")
            for detail in fix_result.fixes_detail[-3:]:
                print(f"  {detail}")

            results[0] = gate_diagnostic(files)
        except ImportError:
            print("[GATE] vf_auto_fix not available, continuing with manual review")
        except Exception as e:
            print(f"[GATE] Auto-fix failed: {e}")

    total_errors = 0
    for r in results:
        print(r.summary())
        print()
        total_errors += len(r.errors)

    print("-" * 60)
    if total_errors == 0:
        print("ALL GATES PASSED")
        return 0

    if warn_only:
        print(f"WARN: {total_errors} issue(s) found (warn-only mode)")
        return 0

    exit_codes = [1, 2, 3]
    code = exit_codes[0] if results[0].errors else 0
    code = max(code, exit_codes[1] if results[1].errors else 0)
    code = max(code, exit_codes[2] if results[2].errors else 0)
    if sum(1 for r in results if r.errors) > 1:
        code = 4

    print(f"BLOCKED: {total_errors} issue(s) — fix before commit")
    return code


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VF Pre-Commit Gate")
    parser.add_argument("--warn", action="store_true", help="warn only, don't block")
    parser.add_argument("--staged", action="store_true", help="only check staged files")
    parser.add_argument("--auto-fix", action="store_true", help="auto-fix diagnostic errors before gate (iterative loop)")
    parser.add_argument("--diagnostic-only", action="store_true", help="only run diagnostic gate")
    parser.add_argument("--sensitive-only", action="store_true", help="only run sensitive gate")
    args = parser.parse_args()

    files = get_changed_py_files() if args.staged else None

    if args.diagnostic_only:
        result = gate_diagnostic(files)
        print(result.summary())
        sys.exit(0 if result.passed else 1)
    elif args.sensitive_only:
        result = gate_sensitive(files)
        print(result.summary())
        sys.exit(0 if result.passed else 2)

    code = run_all_gates(files, warn_only=args.warn, auto_fix=getattr(args, 'auto_fix', False))
    sys.exit(code)
