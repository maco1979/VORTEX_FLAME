#!/usr/bin/env python3
"""Auto-fix engine unit test — direct ErrorClassifier + FixPatternEngine validation"""
import sys, os, json, tempfile, shutil
sys.path.insert(0, r"D:\VORTEX_FLAME")

from vf_auto_fix import (
    ErrorClassifier, ErrorCategory, DiagnosticError,
    FixPatternEngine, AutoFixLoop,
)
from pathlib import Path

FAILED = 0

def check(name, condition, detail=""):
    global FAILED
    if condition:
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name}: {detail}")

print("=== TEST 1: ErrorClassifier — Unused Import ===")
err = DiagnosticError(
    file_path="test.py", line=2, column=8, end_line=2, end_column=10,
    message='"os" is not accessed', rule="reportUnusedImport",
    severity="error",
)
cat = ErrorClassifier.classify(err, "import os")
check("Classified AUTO_FIXABLE", cat == ErrorCategory.AUTO_FIXABLE)
check("Fix description set", "Remove unused import" in err.fix_description)

print("\n=== TEST 2: ErrorClassifier — Unused Variable ===")
err2 = DiagnosticError(
    file_path="test.py", line=5, column=0, end_line=5, end_column=15,
    message='"unused_var" is not accessed', rule="reportUnusedVariable",
    severity="error",
)
cat2 = ErrorClassifier.classify(err2, "unused_var = 42")
check("Classified AUTO_FIXABLE", cat2 == ErrorCategory.AUTO_FIXABLE)

print("\n=== TEST 3: ErrorClassifier — ModuleList Type ===")
err3 = DiagnosticError(
    file_path="test.py", line=10, column=12, end_line=10, end_column=35,
    message='Cannot access member "forward" for type "ModuleList"',
    rule="reportAttributeAccessIssue", severity="error",
)
cat3 = ErrorClassifier.classify(err3, "self.layers[i](x)")
check("Classified KNOWN_PATTERN", cat3 == ErrorCategory.KNOWN_PATTERN)
check("Has type: ignore hint", "type: ignore" in err3.fix_description)

print("\n=== TEST 4: ErrorClassifier — Optional Access ===")
err4 = DiagnosticError(
    file_path="test.py", line=7, column=0, end_line=7, end_column=20,
    message='Cannot access attribute "x" for type "Optional[int]"',
    rule="reportOptionalMemberAccess", severity="error",
)
cat4 = ErrorClassifier.classify(err4, "result = val.x")
check("Classified KNOWN_PATTERN", cat4 == ErrorCategory.KNOWN_PATTERN)

print("\n=== TEST 5: FixPatternEngine — Unused Import Removal ===")
tmpdir = tempfile.mkdtemp()
try:
    test_file = Path(tmpdir) / "fix_test.py"
    test_file.write_text("import os\nimport sys\n\nx = 42\n", encoding="utf-8")

    fixer = FixPatternEngine(tmpdir)
    err_os = DiagnosticError(
        file_path="fix_test.py", line=1, column=8, end_line=1, end_column=10,
        message='"os" is not accessed', rule="reportUnusedImport",
        severity="error", category=ErrorCategory.AUTO_FIXABLE,
    )
    err_os.source_line = "import os"
    result = fixer.apply_fix(err_os)
    check("Unused import fix applied", result)

    content = test_file.read_text()
    check("os import removed", "import os" not in content)
    check("sys import kept", "import sys" in content)
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

print("\n=== TEST 6: FixPatternEngine — Unused Variable → _ ===")
tmpdir2 = tempfile.mkdtemp()
try:
    test_file = Path(tmpdir2) / "fix_test2.py"
    test_file.write_text("unused_var = 42\nused = 10\nprint(used)\n", encoding="utf-8")

    fixer = FixPatternEngine(tmpdir2)
    err_var = DiagnosticError(
        file_path="fix_test2.py", line=1, column=0, end_line=1, end_column=11,
        message='"unused_var" is not accessed', rule="reportUnusedVariable",
        severity="error", category=ErrorCategory.AUTO_FIXABLE,
    )
    err_var.source_line = "unused_var = 42"
    result = fixer.apply_fix(err_var)
    check("Unused variable → _ applied", result)

    content = test_file.read_text()
    check("unused_var replaced with _", "_ = 42" in content)
    check("used not changed", "used = 10" in content)
finally:
    shutil.rmtree(tmpdir2, ignore_errors=True)

print("\n=== TEST 7: FixPatternEngine — Known Pattern Add Ignore ===")
tmpdir3 = tempfile.mkdtemp()
try:
    test_file = Path(tmpdir3) / "fix_test3.py"
    test_file.write_text("self.layers[i](x)\nself.other()\n", encoding="utf-8")

    fixer = FixPatternEngine(tmpdir3)
    err_mod = DiagnosticError(
        file_path="fix_test3.py", line=1, column=12, end_line=1, end_column=42,
        message='Cannot access member "forward" for type "ModuleList"',
        rule="reportCallIssue", severity="error",
        category=ErrorCategory.KNOWN_PATTERN,
    )
    err_mod.source_line = "self.layers[i](x)"
    result = fixer.apply_fix(err_mod)
    check("ModuleList ignore comment added", result)

    content = test_file.read_text()
    check("Contains type: ignore", "# type: ignore" in content)
finally:
    shutil.rmtree(tmpdir3, ignore_errors=True)

print("\n=== TEST 8: Classify ALL error types via pattern matching ===")
pattern_matching_tests = [
    ("reportUnusedCallResult", "Result of call expression is not used", ErrorCategory.KNOWN_PATTERN),
    ("reportAssignmentType", 'Type "int" is not assignable to declared type "str"', ErrorCategory.KNOWN_PATTERN),
    ("reportAttributeAccessIssue", 'Cannot access attribute "x" for type "ModuleList"', ErrorCategory.KNOWN_PATTERN),
    ("reportOptionalMemberAccess", 'Cannot access member "x" for type "Optional[int]"', ErrorCategory.KNOWN_PATTERN),
    ("reportReturnType", 'Expression of type "float" cannot be assigned to return type "int"', ErrorCategory.KNOWN_PATTERN),
]
for rule, msg, expected_cat in pattern_matching_tests:
    err = DiagnosticError(
        file_path="x.py", line=1, column=0, end_line=1, end_column=10,
        message=msg, rule=rule, severity="error",
    )
    cat = ErrorClassifier.classify(err, "")
    check(f"{rule} → {expected_cat.value}", err.category == expected_cat,
          f"got {err.category.value}")

print(f"\n{'='*60}")
if FAILED == 0:
    print("ALL TESTS PASSED")
else:
    print(f"{FAILED} TEST(S) FAILED")
print("="*60)
