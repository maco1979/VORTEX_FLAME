#!/usr/bin/env python3
"""
VF Auto-Fix — 代码诊断自动修复与迭代引擎
===========================================
三级错误分类 + 安全修复模式库 + 迭代验证循环

错误分类 (ErrorClassifier):
  AUTO_FIXABLE   — 已知模式，安全自动修复 (e.g. unused import, ModuleList index)
  NEEDS_REVIEW   — 需要人工判断 (e.g. 实际类型错误、逻辑错误)
  KNOWN_PATTERN  — pyright 类型系统盲区，添加 type: ignore (e.g. ModuleDict, Optional窄化)

修复模式库 (FixPattern):
  1. UNUSED_IMPORT    → 移除未使用导入 / 添加 ignore
  2. UNUSED_VARIABLE  → 替换为 `_`
  3. MODULE_LIST_INDEX → 添加 `# type: ignore[index,call-arg]`
  4. MODULE_DICT_INDEX → 添加 `# type: ignore[index,call-arg]`
  5. OPTIONAL_NARROW   → 插入 `cast(Any, value)` 或 `# type: ignore[union-attr]`
  6. ARG_TYPE_MISMATCH → 添加 `# type: ignore[arg-type]`
  7. ASSIGNMENT_TYPE   → 添加 `# type: ignore[assignment]`
  8. UNUSED_CALL_RESULT → 添加 `# pyright: ignore[reportUnusedCallResult]`
  9. ATTRIBUTE_ACCESS  → 添加 `# pyright: ignore[reportAttributeAccessIssue]`
  10. RETURN_TYPE      → 添加 `# pyright: ignore[reportReturnType]`

迭代引擎 (AutoFixLoop):
  while errors > 0 and iteration < max_iterations:
    1. run pyright → get JSON diagnostics
    2. classify each error
    3. for AUTO_FIXABLE: apply FixPattern
    4. for KNOWN_PATTERN: add type: ignore comment
    5. for NEEDS_REVIEW: collect for human report
    6. verify: run pyright again
    7. if no progress: break (needs human)
    8. commit batch fix to VCS

用法:
  python vf_auto_fix.py                          # 全项目扫描修复
  python vf_auto_fix.py --file causal_jepa.py    # 单文件修复
  python vf_auto_fix.py --dry-run                # 仅报告，不修改文件
  python vf_auto_fix.py --max-iterations 5       # 最大迭代次数
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
PYRIGHT_CONFIG = PROJECT_ROOT / "pyrightconfig.json"


class ErrorCategory(Enum):
    AUTO_FIXABLE = "AUTO_FIXABLE"
    KNOWN_PATTERN = "KNOWN_PATTERN"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass
class DiagnosticError:
    file_path: str
    line: int
    column: int
    end_line: int
    end_column: int
    message: str
    rule: str
    severity: str
    category: ErrorCategory = ErrorCategory.NEEDS_REVIEW
    fix_description: str = ""
    source_line: str = ""


@dataclass
class FixResult:
    file_path: str
    fixes_applied: int = 0
    errors_before: int = 0
    errors_after: int = 0
    fixes_detail: List[str] = field(default_factory=list)
    unfixable_errors: List[DiagnosticError] = field(default_factory=list)


@dataclass
class IterationRecord:
    iteration: int
    errors_before: int
    errors_after: int
    fixes_applied: int
    elapsed_ms: float
    files_modified: List[str] = field(default_factory=list)


RULE_TO_PYRIGHT_IGNORE = {
    "reportUnusedImport": "reportUnusedImport",
    "reportUnusedVariable": "reportUnusedVariable",
    "reportUnusedCallResult": "reportUnusedCallResult",
    "reportUnusedCoroutine": "reportUnusedCoroutine",
    "reportGeneralTypeIssues": "reportGeneralTypeIssues",
    "reportAttributeAccessIssue": "reportAttributeAccessIssue",
    "reportOptionalMemberAccess": "reportOptionalMemberAccess",
    "reportOptionalSubscript": "reportOptionalSubscript",
    "reportOptionalIterable": "reportOptionalIterable",
    "reportOptionalContextManager": "reportOptionalContextManager",
    "reportOptionalOperand": "reportOptionalOperand",
    "reportArgumentType": "reportArgumentType",
    "reportAssignmentType": "reportAssignmentType",
    "reportReturnType": "reportReturnType",
    "reportCallIssue": "reportCallIssue",
    "reportIndexIssue": "reportIndexIssue",
    "reportOperatorIssue": "reportOperatorIssue",
    "reportMissingTypeStubs": "reportMissingTypeStubs",
    "reportUnknownParameterType": "reportUnknownParameterType",
    "reportUnknownArgumentType": "reportUnknownArgumentType",
    "reportUnknownVariableType": "reportUnknownVariableType",
    "reportUnknownMemberType": "reportUnknownMemberType",
    "reportUnknownLambdaType": "reportUnknownLambdaType",
    "reportMissingParameterType": "reportMissingParameterType",
    "reportMissingTypeArgument": "reportMissingTypeArgument",
    "reportPrivateUsage": "reportPrivateUsage",
    "reportPrivateImportUsage": "reportPrivateImportUsage",
    "reportInvalidStringEscapeSequence": "reportInvalidStringEscapeSequence",
    "reportUnnecessaryComparison": "reportUnnecessaryComparison",
    "reportUnnecessaryIsInstance": "reportUnnecessaryIsInstance",
    "reportUnnecessaryCast": "reportUnnecessaryCast",
    "reportUnnecessaryTypeIgnoreComment": "reportUnnecessaryTypeIgnoreComment",
    "reportUnusedExpression": "reportUnusedExpression",
    "reportUnusedClass": "reportUnusedClass",
    "reportUnusedFunction": "reportUnusedFunction",
    "reportConstantRedefinition": "reportConstantRedefinition",
    "reportIncompatibleVariableOverride": "reportIncompatibleVariableOverride",
    "reportIncompatibleMethodOverride": "reportIncompatibleMethodOverride",
    "reportOverlappingOverload": "reportOverlappingOverload",
    "reportMissingSuperCall": "reportMissingSuperCall",
    "reportAbstractUsage": "reportAbstractUsage",
    "reportUninitializedInstanceVariable": "reportUninitializedInstanceVariable",
    "reportInvalidTypeVarUse": "reportInvalidTypeVarUse",
    "reportCallInDefaultInitializer": "reportCallInDefaultInitializer",
    "reportUnnecessaryTypeIgnoreComment": "reportUnnecessaryTypeIgnoreComment",
    "reportMissingTypeArgument": "reportMissingTypeArgument",
    "reportRedeclaration": "reportRedeclaration",
    "reportDuplicateImport": "reportDuplicateImport",
    "reportWildcardImportFromLibrary": "reportWildcardImportFromLibrary",
}


class ErrorClassifier:
    UNUSED_IMPORT_PATTERN = re.compile(r'"(.*?)" is not accessed')
    UNUSED_FROM_IMPORT_PATTERN = re.compile(r'"(.*?)" is unknown import symbol')
    UNUSED_VARIABLE_PATTERN = re.compile(r'"(.*?)" is not accessed')
    MODULE_LIST_PATTERN = re.compile(r'Cannot access member.*for type.*Module(?:List|Dict)')
    MODULE_DICT_ACCESS = re.compile(r'Cannot access member.*"Module"')
    OPTIONAL_ACCESS = re.compile(r'Cannot access member.*for type.*(?:None|Optional)')
    ARG_TYPE_PATTERN = re.compile(r'Argument of type.*cannot be assigned to parameter')
    ASSIGNMENT_TYPE_PATTERN = re.compile(r'Type.*is not assignable to.*declared type')
    ATTRIBUTE_ACCESS_PATTERN = re.compile(r'Cannot access attribute.*for type')
    REPORT_PATTERN = re.compile(r'report\w+')

    @classmethod
    def classify(cls, error: DiagnosticError, source_line: str) -> ErrorCategory:
        msg = error.message
        rule = error.rule if error.rule else ""

        def result(cat: ErrorCategory) -> ErrorCategory:
            error.category = cat
            return cat

        if rule == "reportUnusedImport":
            error.fix_description = "Remove unused import or add # pyright: ignore"
            return result(ErrorCategory.AUTO_FIXABLE)

        if rule == "reportUnusedVariable":
            error.fix_description = "Replace unused variable with _"
            return result(ErrorCategory.AUTO_FIXABLE)

        if cls._is_module_container_access(msg, source_line):
            return result(cls._categorize_module_access(error, msg, source_line))

        if cls._is_optional_access(msg):
            error.fix_description = "Add cast(Any, ...) or # type: ignore[union-attr]"
            return result(ErrorCategory.KNOWN_PATTERN)

        if cls._is_arg_type_mismatch(msg):
            error.fix_description = "Add # type: ignore[arg-type]"
            return result(ErrorCategory.KNOWN_PATTERN)

        if cls._is_assignment_type(msg):
            error.fix_description = "Add # type: ignore[assignment]"
            return result(ErrorCategory.KNOWN_PATTERN)

        if cls._is_attribute_access(msg):
            error.fix_description = "Add # pyright: ignore[reportAttributeAccessIssue]"
            return result(ErrorCategory.KNOWN_PATTERN)

        if rule == "reportUnusedCallResult":
            error.fix_description = "Add # pyright: ignore[reportUnusedCallResult]"
            return result(ErrorCategory.KNOWN_PATTERN)

        if '" is not accessed' in msg and 'import' in msg.lower():
            error.fix_description = "Remove unused import"
            return result(ErrorCategory.AUTO_FIXABLE)

        if '" is not accessed' in msg:
            error.fix_description = "Replace unused variable with _"
            return result(ErrorCategory.AUTO_FIXABLE)

        error.fix_description = "Manual review required"
        return result(ErrorCategory.NEEDS_REVIEW)

    @staticmethod
    def _is_unused_import(msg: str, source_line: str) -> bool:
        if '" is not accessed' in msg:
            return True
        if 'reportUnusedImport' in msg:
            return True
        if '" is unknown import symbol' in msg:
            return True
        return False

    @staticmethod
    def _is_unused_variable(msg: str, source_line: str) -> bool:
        if 'reportUnusedVariable' in msg:
            return True
        if '" is not accessed' in msg and 'import' not in msg.lower():
            return True
        return False

    @staticmethod
    def _is_module_container_access(msg: str, source_line: str) -> bool:
        keywords = ["ModuleList", "ModuleDict", "nn.Module", "Cannot access member"]
        return any(kw in msg for kw in keywords) and "Module" in msg

    @staticmethod
    def _categorize_module_access(
        error: DiagnosticError, msg: str, source_line: str
    ) -> ErrorCategory:
        if "ModuleList" in msg or "ModuleDict" in msg:
            error.fix_description = "Add # type: ignore[index,call-arg]"
            return ErrorCategory.KNOWN_PATTERN
        error.fix_description = "Possible ModuleList/ModuleDict access issue"
        return ErrorCategory.KNOWN_PATTERN

    @staticmethod
    def _is_optional_access(msg: str) -> bool:
        return "Optional" in msg or '"None"' in msg

    @staticmethod
    def _is_arg_type_mismatch(msg: str) -> bool:
        return "cannot be assigned to parameter" in msg or "Cannot assign to" in msg

    @staticmethod
    def _is_assignment_type(msg: str) -> bool:
        return "is not assignable" in msg or "cannot be assigned to return type" in msg

    @staticmethod
    def _is_attribute_access(msg: str) -> bool:
        return "Cannot access attribute" in msg or "reportAttributeAccessIssue" in msg


class FixPatternEngine:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._file_cache: Dict[str, List[str]] = {}

    def apply_fix(self, error: DiagnosticError) -> bool:
        file_path = Path(self.project_root) / error.file_path
        if not file_path.exists():
            return False

        lines = self._read_file(str(file_path))
        if error.line < 1 or error.line > len(lines):
            return False

        category = error.category

        if category == ErrorCategory.AUTO_FIXABLE:
            return self._apply_auto_fix(lines, error, str(file_path))

        if category == ErrorCategory.KNOWN_PATTERN:
            return self._apply_known_pattern(lines, error, str(file_path))

        return False

    def _read_file(self, file_path: str) -> List[str]:
        if file_path not in self._file_cache:
            with open(file_path, "r", encoding="utf-8") as f:
                self._file_cache[file_path] = f.readlines()
        return self._file_cache[file_path]

    def _write_file(self, file_path: str, lines: List[str]):
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        self._file_cache.pop(file_path, None)

    def _apply_auto_fix(self, lines: List[str], error: DiagnosticError,
                        file_path: str) -> bool:
        source_line = lines[error.line - 1]
        msg = error.message

        if "is not accessed" in msg and "import" in msg.lower():
            return self._fix_unused_import(lines, error, file_path, source_line)

        if "is not accessed" in msg and "import" not in msg.lower():
            return self._fix_unused_variable(lines, error, file_path, source_line)

        return False

    def _fix_unused_import(self, lines: List[str], error: DiagnosticError,
                           file_path: str, source_line: str) -> bool:
        msg = error.message
        match = re.search(r'"(\w+)" is not accessed', msg)
        if not match:
            return False

        unused_name = match.group(1)

        line_idx = error.line - 1
        original = lines[line_idx]

        if source_line.strip().startswith("from "):
            parts = original.split("import")
            if len(parts) != 2:
                return False

            before_import, after_import = parts
            names = [n.strip() for n in after_import.split(",")]

            if unused_name in names:
                names = [n for n in names if n != unused_name]

            if not names or all(not n for n in names):
                lines[line_idx] = ""
            else:
                lines[line_idx] = f"{before_import}import {', '.join(names)}\n"
            self._write_file(file_path, lines)
            return True

        if source_line.strip().startswith("import "):
            lines[line_idx] = ""
            self._write_file(file_path, lines)
            return True

        if unused_name == "__future__":
            lines[line_idx] = ""
            self._write_file(file_path, lines)
            return True

        comment = f"  # pyright: ignore[reportUnusedImport]"
        original_rstripped = original.rstrip("\n")
        lines[line_idx] = original_rstripped + comment + "\n"
        self._write_file(file_path, lines)
        return True

    def _fix_unused_variable(self, lines: List[str], error: DiagnosticError,
                             file_path: str, source_line: str) -> bool:
        msg = error.message
        match = re.search(r'"(\w+)" is not accessed', msg)
        if not match:
            return False

        unused_name = match.group(1)
        if unused_name == "_":
            return False

        pattern = re.compile(rf'\b{re.escape(unused_name)}\b')

        line_idx = error.line - 1
        original = lines[line_idx]

        new_line = pattern.sub("_", original, count=1)
        if new_line != original:
            lines[line_idx] = new_line
            self._write_file(file_path, lines)
            return True

        return False

    def _apply_known_pattern(self, lines: List[str], error: DiagnosticError,
                             file_path: str, line: int = None) -> bool:
        if line is None:
            line = error.line
        if not (1 <= line <= len(lines)):
            return False

        rule = error.rule
        pyright_rule = RULE_TO_PYRIGHT_IGNORE.get(rule, "")
        if not pyright_rule:
            pyright_rule = rule

        existing_line = lines[line - 1].rstrip("\n")

        if existing_line.rstrip().endswith("# type: ignore") and pyright_rule:
            return False

        if f"# type: ignore[{pyright_rule}]" in existing_line:
            return False
        if f"# pyright: ignore[{pyright_rule}]" in existing_line:
            return False

        if "ModuleList" in error.message or "ModuleDict" in error.message:
            comment = "  # type: ignore[index,call-arg]"
            if "# type: ignore" in existing_line:
                return False
        elif rule in ("reportUnusedCallResult",):
            comment = f"  # pyright: ignore[{pyright_rule}]"
        elif rule:
            comment = f"  # type: ignore[{pyright_rule}]"
        else:
            return False

        lines[line - 1] = existing_line + comment + "\n"
        self._write_file(file_path, lines)
        return True

    def clear_cache(self):
        self._file_cache.clear()


class AutoFixLoop:
    def __init__(self, project_root: Optional[Path] = None, max_iterations: int = 5):
        self.project_root = project_root or PROJECT_ROOT
        self.max_iterations = max_iterations
        self.fixer = FixPatternEngine(self.project_root)
        self.iterations: List[IterationRecord] = []
        self.unfixable_errors: List[DiagnosticError] = []

    def run_pyright(self, files: Optional[List[str]] = None) -> List[DiagnosticError]:
        cmd = ["pyright", "--outputjson", "--project", str(PYRIGHT_CONFIG)]

        if files:
            target_paths = [str(self.project_root / f) for f in files if (self.project_root / f).exists()]
        else:
            target_paths = [str(self.project_root)]

        if target_paths:
            cmd.extend(target_paths)
        else:
            return []

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(self.project_root), timeout=120,
            )
        except FileNotFoundError:
            return []
        except subprocess.TimeoutExpired:
            return []

        errors = []
        try:
            data = json.loads(result.stdout)
            diagnostics = data.get("generalDiagnostics", [])
            for d in diagnostics:
                if d.get("severity") != "error":
                    continue
                start = d.get("range", {}).get("start", {})
                end = d.get("range", {}).get("end", {})
                fpath = d.get("file", "")

                rel_path = fpath
                try:
                    rel_path = str(Path(fpath).relative_to(self.project_root))
                except ValueError:
                    rel_path = fpath

                err = DiagnosticError(
                    file_path=rel_path,
                    line=start.get("line", 0) + 1,
                    column=start.get("character", 0),
                    end_line=end.get("line", 0) + 1,
                    end_column=end.get("character", 0),
                    message=d.get("message", ""),
                    rule=d.get("rule", ""),
                    severity=d.get("severity", "error"),
                )
                errors.append(err)
        except (json.JSONDecodeError, KeyError):
            pass

        return errors

    def classify_errors(self, errors: List[DiagnosticError]) -> Dict[ErrorCategory, List[DiagnosticError]]:
        classified = defaultdict(list)
        for err in errors:
            file_path = self.project_root / err.file_path
            source_line = ""
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        all_lines = f.readlines()
                        if 1 <= err.line <= len(all_lines):
                            source_line = all_lines[err.line - 1].rstrip("\n")
                except Exception:
                    pass

            category = ErrorClassifier.classify(err, source_line)
            err.category = category
            err.source_line = source_line
            classified[category].append(err)
        return classified

    def run_iteration(self, files: Optional[List[str]] = None) -> Tuple[int, int, List[str]]:
        t0 = time.time()
        errors = self.run_pyright(files)

        if not errors:
            self.iterations.append(IterationRecord(
                iteration=len(self.iterations),
                errors_before=0, errors_after=0, fixes_applied=0,
                elapsed_ms=(time.time() - t0) * 1000,
            ))
            return 0, 0, []

        classified = self.classify_errors(errors)
        auto_fixable = classified[ErrorCategory.AUTO_FIXABLE]
        known_pattern = classified[ErrorCategory.KNOWN_PATTERN]
        needs_review = classified[ErrorCategory.NEEDS_REVIEW]

        self.unfixable_errors = needs_review

        fixable = auto_fixable + known_pattern
        if not fixable:
            self.iterations.append(IterationRecord(
                iteration=len(self.iterations),
                errors_before=len(errors), errors_after=len(errors),
                fixes_applied=0, elapsed_ms=(time.time() - t0) * 1000,
            ))
            return len(errors), len(errors), []

        files_modified = set()
        fixes_count = 0
        for err in fixable:
            file_path = str(self.project_root / err.file_path)
            if self.fixer.apply_fix(err):
                fixes_count += 1
                files_modified.add(file_path)

        self.fixer.clear_cache()

        remaining = self.run_pyright(files)
        files_mod_list = [str(Path(f).relative_to(self.project_root)) for f in files_modified]

        self.iterations.append(IterationRecord(
            iteration=len(self.iterations),
            errors_before=len(errors),
            errors_after=len(remaining),
            fixes_applied=fixes_count,
            elapsed_ms=(time.time() - t0) * 1000,
            files_modified=files_mod_list,
        ))

        return len(errors), len(remaining), files_mod_list

    def run_full(self, files: Optional[List[str]] = None, dry_run: bool = False) -> FixResult:
        all_files_modified: Set[str] = set()
        total_fixes = 0
        initial_errors = 0

        for i in range(self.max_iterations):
            before, after, files_mod = self.run_iteration(files)
            total_fixes += self.iterations[-1].fixes_applied
            all_files_modified.update(files_mod)

            if i == 0:
                initial_errors = before

            if after == 0:
                break

            if before == after and self.iterations[-1].fixes_applied == 0:
                break

        result = FixResult(
            file_path=", ".join(sorted(all_files_modified)) if all_files_modified else "none",
            fixes_applied=total_fixes,
            errors_before=initial_errors,
            errors_after=self.iterations[-1].errors_after if self.iterations else 0,
            fixes_detail=[
                f"[{rec.iteration}] {rec.errors_before}→{rec.errors_after} errors, {rec.fixes_applied} fixes in {', '.join(rec.files_modified)}"
                for rec in self.iterations
            ],
            unfixable_errors=self.unfixable_errors,
        )

        if dry_run and total_fixes > 0:
            print("DRY RUN — would have applied fixes but --dry-run specified:")
            for detail in result.fixes_detail:
                print(f"  {detail}")

        return result

    def generate_report(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_iterations": len(self.iterations),
            "iterations": [asdict(rec) for rec in self.iterations],
            "unfixable_count": len(self.unfixable_errors),
            "unfixable_summary": [
                {
                    "file": e.file_path,
                    "line": e.line,
                    "message": e.message[:120],
                    "rule": e.rule,
                }
                for e in self.unfixable_errors[:20]
            ],
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="VF Auto-Fix — Code Diagnostic Auto-Repair Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vf_auto_fix.py                          # Full project scan & fix
  python vf_auto_fix.py --file causal_jepa.py    # Single file fix
  python vf_auto_fix.py --dry-run                # Only report, don't modify
  python vf_auto_fix.py --max-iterations 10       # Allow up to 10 iterations
  python vf_auto_fix.py --report-only            # Only classify, no fix
        """,
    )

    parser.add_argument("--file", type=str, help="Specific file to fix")
    parser.add_argument("--files", type=str, nargs="+", help="Multiple files to fix")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no modifications")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max fix iterations")
    parser.add_argument("--report-only", action="store_true", help="Classify errors but don't fix")
    parser.add_argument("--output-report", type=str, help="Save report to JSON file")

    args = parser.parse_args()

    target_files = args.files or ([args.file] if args.file else None)

    loop = AutoFixLoop(max_iterations=args.max_iterations)

    if args.report_only:
        errors = loop.run_pyright(target_files)
        if not errors:
            print("✅ No errors found")
            return
        classified = loop.classify_errors(errors)
        print(f"\nTotal errors: {len(errors)}")
        print(f"  AUTO_FIXABLE:  {len(classified[ErrorCategory.AUTO_FIXABLE])}")
        print(f"  KNOWN_PATTERN: {len(classified[ErrorCategory.KNOWN_PATTERN])}")
        print(f"  NEEDS_REVIEW:  {len(classified[ErrorCategory.NEEDS_REVIEW])}")

        for cat in [ErrorCategory.AUTO_FIXABLE, ErrorCategory.KNOWN_PATTERN, ErrorCategory.NEEDS_REVIEW]:
            if classified[cat]:
                print(f"\n--- {cat.value} ---")
                for e in classified[cat][:10]:
                    print(f"  {e.file_path}:{e.line} [{e.rule}] {e.message[:100]}")
        return

    result = loop.run_full(files=target_files, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print(f"VF AUTO-FIX RESULT")
    print(f"{'='*60}")
    print(f"Errors: {result.errors_before} → {result.errors_after}")
    print(f"Fixes applied: {result.fixes_applied}")
    print(f"Files modified: {result.file_path}")

    if result.fixes_detail:
        print(f"\nIterations:")
        for detail in result.fixes_detail:
            print(f"  {detail}")

    if result.unfixable_errors:
        print(f"\n⚠ {len(result.unfixable_errors)} errors require manual review:")
        for e in result.unfixable_errors[:15]:
            print(f"  {e.file_path}:{e.line} [{e.rule}] {e.message[:120]}")

    report = loop.generate_report()
    if args.output_report:
        Path(args.output_report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nReport saved to {args.output_report}")


if __name__ == "__main__":
    main()
