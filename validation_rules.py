"""
Knowledge Base Validation Rules — CI/CD for Knowledge Quality
================================================================
Machine-verifiable rules for Einstein (physics/chemistry) and
Cezanne (code/logic) knowledge bases. Each rule is a testable
predicate that runs against SQLite KB entries.

Concept: "Give knowledge the same rigor as code."
Every rule returns (pass: bool, violation: Optional[str]).

Einstein: 45 rules (physics/chem dimensional consistency + causal checks)
Cezanne: 60 rules (code correctness + algorithmic validity + logic)

Usage:
    engine = ValidationEngine()
    report = engine.validate("einstein")
    report = engine.validate("cezanne")
    full = engine.validate_all()
"""

import re
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime

KB_DIR = Path("D:/VORTEX_FLAME/.vf_memory")


@dataclass
class Violation:
    rule_id: str
    severity: str  # CRITICAL HIGH MEDIUM LOW
    category: str
    entry_id: str
    description: str
    suggestion: str


@dataclass
class AuditReport:
    kb_name: str
    timestamp: str
    total_rules: int
    passed: int
    failed: int
    violations: List[Violation] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== {self.kb_name} Audit Report ({self.timestamp}) ===",
            f"Rules: {self.passed}/{self.total_rules} passed, {self.failed} violations",
        ]
        if self.violations:
            for v in self.violations:
                lines.append(f"  [{v.severity}] {v.rule_id}: {v.description}")
        return "\n".join(lines)


class ValidationRule:
    def __init__(self, rule_id: str, category: str, severity: str,
                 description: str, check: Callable[[str, str], Tuple[bool, Optional[str]]]):
        self.rule_id = rule_id
        self.category = category
        self.severity = severity
        self.description = description
        self._check = check

    def check(self, entry_id: str, content: str) -> Tuple[bool, Optional[Violation]]:
        passed, detail = self._check(entry_id, content)
        if passed:
            return True, None
        return False, Violation(
            rule_id=self.rule_id, severity=self.severity,
            category=self.category, entry_id=entry_id,
            description=detail or self.description,
            suggestion=f"Review {self.category} consistency for {entry_id}"
        )


def _has_dimensions(content: str) -> bool:
    return bool(re.search(r'[mskgJNVAW]|meter|second|kilogram|joule|newton|volt|amp', content))

def _has_formula(content: str) -> bool:
    return bool(re.search(r'[=≈≠≤≥]|∑|∏|∫|∂|Δ|∇', content))

def _has_units(content: str) -> bool:
    return bool(re.search(r'[kcmμnp]?[mgsAVWJNHzPaK]|meter|second|hertz|pascal', content))

def _check_tabs(content: str) -> Tuple[bool, Optional[str]]:
    if '```python' not in content:
        return True, None
    idx = content.index('```python')
    end = content.find('```', idx + 10)
    if end == -1:
        return True, None
    block = content[idx + 10:end]
    if '\t' in block:
        return False, "Non-PEP8 indentation (tabs found in Python code block)"
    return True, None

def _check_docstring(content: str) -> Tuple[bool, Optional[str]]:
    if 'def ' not in content or '```python' not in content:
        return True, None
    after_first_def = content.split('def ')[-1]
    if '"""' in after_first_def[:200] or "'''" in after_first_def[:200]:
        return True, None
    return False, "Function without docstring"


EINSTEIN_RULES = [
    ValidationRule("E01", "dimensional", "CRITICAL",
        "F=ma consistency: force must equal mass×acceleration",
        lambda eid, c: (True, None) if not _has_formula(c) else
            ((False, "Force formula may be inconsistent") if re.search(r'F\s*[=≈]\s*(?!m\s*[×*]\s*a)', c) and 'F' in c else (True, None))),

    ValidationRule("E02", "dimensional", "HIGH",
        "Energy units check: E must be in joules (kg·m²/s²)",
        lambda eid, c: (False, "Energy unit mismatch") if re.search(r'energy|Energy|ENERGY', c) and _has_units(c) and not re.search(r'joule|J\b|kg.*m.*s', c) else (True, None)),

    ValidationRule("E03", "conservation", "CRITICAL",
        "Conservation of energy: energy cannot be created or destroyed in closed systems",
        lambda eid, c: (False, "Violates energy conservation") if re.search(r'(create|生成).*(energy|能量).*(from nothing|无中生有|凭空)', c) else (True, None)),

    ValidationRule("E04", "conservation", "CRITICAL",
        "Conservation of momentum: total momentum constant in isolated systems",
        lambda eid, c: (False, "Violates momentum conservation") if re.search(r'(momentum|动量).*(change|改变|without|没有)', c) and not re.search(r'external.force|外力|collision|碰撞', c) else (True, None)),

    ValidationRule("E05", "thermodynamics", "CRITICAL",
        "Second law: entropy of isolated system never decreases",
        lambda eid, c: (False, "Violates second law of thermodynamics") if re.search(r'(entropy|熵).*(decrease|减少|spontaneously|自发).*(isolated|封闭|孤立)', c) else (True, None)),

    ValidationRule("E06", "thermodynamics", "HIGH",
        "Absolute zero (0K = -273.15°C) is unattainable",
        lambda eid, c: (False, "Claims absolute zero is reachable") if re.search(r'(absolute.zero|绝对零度).*(reach|达到|achieve)', c) else (True, None)),

    ValidationRule("E07", "relativity", "CRITICAL",
        "Nothing exceeds speed of light c in vacuum",
        lambda eid, c: (False, "Faster-than-light claim without context") if re.search(r'(faster|faster.than|超过|超越).*(light|光速|c\b).*(information|信息|particle|粒子)', c) else (True, None)),

    ValidationRule("E08", "relativity", "HIGH",
        "E=mc² applies only to rest mass, not kinetic energy",
        lambda eid, c: (False, "E=mc² applied to non-rest-mass context") if re.search(r'E\s*=\s*mc.*kinetic|动能.*E\s*=\s*mc', c) else (True, None)),

    ValidationRule("E09", "quantum", "MEDIUM",
        "Heisenberg uncertainty: Δx·Δp ≥ ℏ/2",
        lambda eid, c: (False, "Violates uncertainty principle") if re.search(r'(simultaneously|同时).*(know|知道|measure|测量).*(position|位置).*(momentum|动量).*(exact|精确|precise)', c) else (True, None)),

    ValidationRule("E10", "quantum", "MEDIUM",
        "Superposition collapses upon measurement",
        lambda eid, c: (False, "Superposition claim without measurement context") if re.search(r'superposition|叠加态', c) and not re.search(r'(measure|测量|collapse|坍缩|observe|观察)', c) else (True, None)),

    ValidationRule("E11", "chemistry", "HIGH",
        "Charge conservation: total charge in reactions is constant",
        lambda eid, c: (False, "Charge imbalance in reaction") if re.search(r'(charge|电荷).*(create|创造|destroy|消灭)', c) else (True, None)),

    ValidationRule("E12", "chemistry", "HIGH",
        "Stoichiometry: atom counts must balance in equations",
        lambda eid, c: (False, "Unbalanced chemical equation") if re.search(r'→|->|\+', c) and _has_formula(c) and re.search(r'(unbalanced|不平衡)', c.lower()) else (True, None)),

    ValidationRule("E13", "wave", "MEDIUM",
        "Wave speed v = f·λ must hold for mechanical waves",
        lambda eid, c: (False, "Wave equation inconsistency") if re.search(r'freq.*wave|波.*频率', c) and re.search(r'v\s*[=≈]', c) and not re.search(r'[λf]|lambda', c) else (True, None)),

    ValidationRule("E14", "electromagnetic", "HIGH",
        "Maxwell's equations: changing B field creates E field, and vice versa",
        lambda eid, c: (False, "EM induction contradiction") if re.search(r'(magnetic|磁).*(change|变化).*(no|没有|without).*(electric|电)', c) else (True, None)),

    ValidationRule("E15", "nuclear", "HIGH",
        "Mass defect = nuclear binding energy / c²",
        lambda eid, c: (False, "Mass-energy mismatch in nuclear context") if re.search(r'(binding.energy|结合能).*E\s*=\s*mc', c) and not re.search(r'(mass.defect|质量亏损|Δm)', c) else (True, None)),

    ValidationRule("E16", "dimensional", "HIGH",
        "Velocity units must be m/s or equivalent",
        lambda eid, c: (False, "Velocity unit error") if re.search(r'velocity.*[=≈].*\d+\s*(kg|J|N|W)\b', c) else (True, None)),

    ValidationRule("E17", "dimensional", "HIGH",
        "Pressure units must be Pa (N/m²) or equivalent",
        lambda eid, c: (False, "Pressure unit error") if re.search(r'pressure.*[=≈].*\d+\s*(m/s|Hz|kg)\b', c) else (True, None)),

    ValidationRule("E18", "optics", "MEDIUM",
        "Snell's law: n₁sinθ₁ = n₂sinθ₂",
        lambda eid, c: (False, "Refraction without refractive index") if re.search(r'(refraction|折射)', c) and re.search(r'angle|角度', c) and not re.search(r'(refractive.index|折射率|n\d)', c) else (True, None)),

    ValidationRule("E19", "mechanics", "HIGH",
        "Newton's third law: action = -reaction",
        lambda eid, c: (False, "Missing reaction force") if re.search(r'(force|力).*(act|作用).*(object|物体)', c) and not re.search(r'(reaction|反作用|opposite|相反)', c) and re.search(r'(single|单独|only.one|只有)', c) else (True, None)),

    ValidationRule("E20", "mechanics", "MEDIUM",
        "Friction force f ≤ μ·N (static friction upper bound)",
        lambda eid, c: (False, "Friction exceeds normal force bound") if re.search(r'friction.*>\s*\d+\.?\d*\s*[×*]\s*normal', c) else (True, None)),

    ValidationRule("E21", "thermodynamics", "MEDIUM",
        "Carnot efficiency η = 1 - T_cold/T_hot (absolute temps)",
        lambda eid, c: (False, "Efficiency > Carnot limit claimed") if re.search(r'(efficiency|效率|η).*(100%|1\.0|perfect|完美)', c) and re.search(r'(engine|热机|heat)', c) else (True, None)),

    ValidationRule("E22", "quantum", "MEDIUM",
        "Pauli exclusion: no two fermions share all quantum numbers",
        lambda eid, c: (False, "Violates Pauli principle") if re.search(r'(electron|电子).*(same|相同).*(state|状态).*(all|所有).*(quantum|量子)', c) else (True, None)),

    ValidationRule("E23", "chemistry", "MEDIUM",
        "pH scale: 0-14 in water at 25°C (with rare exceptions noted)",
        lambda eid, c: (False, "pH outside valid range without explanation") if re.search(r'pH\s*[=>]?\s*(-?\d+\.?\d*)', c) else (True, None)),

    ValidationRule("E24", "chemistry", "MEDIUM",
        "Avogadro's number: N_A ≈ 6.022×10²³ mol⁻¹",
        lambda eid, c: (False, "Avogadro number is incorrect") if re.search(r'Avogadro|阿伏加德罗', c) and re.search(r'6\.02[2-3].*10\^?2[3-4]', c) else (True, None)),

    ValidationRule("E25", "gravity", "HIGH",
        "g ≈ 9.8 m/s² at Earth surface (not exactly 10 unless stated approximation)",
        lambda eid, c: (False, "g=10 without approximation note") if re.search(r'g\s*=\s*10\s*(m/s|meters)', c) and not re.search(r'approximation|近似|≈|~', c) else (True, None)),

    ValidationRule("E26", "knowledge_gap", "HIGH",
        "Entry should not state 'X is still unknown' without citation",
        lambda eid, c: (False, "Unsourced knowledge gap claim") if re.search(r'(unknown|未知|not.known|no.one.knows)', c) and not re.search(r'\[|ref|reference|citation|引用|来源|paper', c) else (True, None)),

    ValidationRule("E27", "knowledge_gap", "MEDIUM",
        "Entry with formula should include variable definitions",
        lambda eid, c: (False, "Formula without variable definitions") if _has_formula(c) and not re.search(r'where|其中|定义|denote|let\b', c) else (True, None)),

    ValidationRule("E28", "knowledge_gap", "MEDIUM",
        "Physics entry should mention applicable scale (quantum/classical/relativistic)",
        lambda eid, c: (False, "Missing scale context") if _has_formula(c) and not re.search(r'(quantum|classical|relativistic|量子|经典|相对论|micro|macro)', c) else (True, None)),

    ValidationRule("E29", "causality", "CRITICAL",
        "Cause must precede effect in temporal reasoning",
        lambda eid, c: (False, "Causal order violation") if re.search(r'(after|之后|then|然后).*(cause|cause|原因)', c) and re.search(r'(before|之前).*(effect|结果|影响)', c) else (True, None)),

    ValidationRule("E30", "causality", "HIGH",
        "Correlation ≠ causation: must not assert causation from correlation alone",
        lambda eid, c: (False, "Correlation presented as causation") if re.search(r'(correlat|相关).*(cause|导致|造成|caus)', c) and not re.search(r'(experiment|实验|trial|试验|control|对照)', c) else (True, None)),

    ValidationRule("E31", "math", "MEDIUM",
        "Division by zero must not appear without limit notation",
        lambda eid, c: (False, "Potential division by zero") if re.search(r'(divide|除以|/)\s*0\b', c) and not re.search(r'limit|lim|→.*0|approaches', c) else (True, None)),

    ValidationRule("E32", "math", "MEDIUM",
        "Infinity must be qualified (countable/uncountable/potential/actual)",
        lambda eid, c: (False, "Unqualified infinity") if re.search(r'infinite|无穷大|∞', c) and not re.search(r'(countable|uncountable|可数|不可数|potential|actual)', c) else (True, None)),

    ValidationRule("E33", "temporal", "LOW",
        "Date references should use standard notation (CE/BCE or AD/BC)",
        lambda eid, c: (False, "Non-standard date notation") if re.search(r'\d{3,4}\s*(年|year)', c) and not re.search(r'(BCE|CE|BC|AD|公元|前)', c) else (True, None)),

    ValidationRule("E34", "precision", "LOW",
        "Significant figures: computed values should not claim excessive precision",
        lambda eid, c: (False, "Excessive precision without uncertainty") if re.search(r'\d+\.\d{6,}', c) and not re.search(r'\±|±|error|uncertaint|误差|不确定', c) else (True, None)),

    ValidationRule("E35", "precision", "LOW",
        "Physical constants must use accepted values within 1%",
        lambda eid, c: (False, "Physical constant value outside accepted range") if re.search(r'Planck.*6\.6[3-9]', c) and not re.search(r'6\.626', c) else (True, None)),

    ValidationRule("E36", "circular", "HIGH",
        "Definition must not be circular (A defined as A)",
        lambda eid, c: (False, "Circular definition detected") if re.search(r'(is|是|定义|define).{0,50}(itself|自身|same|同)', c) else (True, None)),

    ValidationRule("E37", "boundary", "MEDIUM",
        "Physical law should specify boundary conditions or domain of applicability",
        lambda eid, c: (False, "Missing boundary conditions") if _has_formula(c) and not re.search(r'(boundary|边界|condition|条件|assume|假设|when|当)', c) else (True, None)),

    ValidationRule("E38", "units", "MEDIUM",
        "SI units should be used consistently (not mixed with imperial without conversion)",
        lambda eid, c: (False, "Mixed unit systems without conversion") if re.search(r'(feet|pounds|miles)', c) and re.search(r'(meters|kg|km)', c) and not re.search(r'convert|转换|→|≈', c) else (True, None)),

    ValidationRule("E39", "special_relativity", "HIGH",
        "Time dilation: moving clocks run slower, Δt' = γ·Δt",
        lambda eid, c: (False, "Time dilation direction inverted") if re.search(r'(moving.clock|运动的钟).*(faster|更快)', c) else (True, None)),

    ValidationRule("E40", "special_relativity", "MEDIUM",
        "Length contraction: L = L₀/γ, objects contract along motion direction",
        lambda eid, c: (False, "Length contraction direction incorrect") if re.search(r'(length.contract|长度收缩).*(perpendicular|垂直|transverse)', c) else (True, None)),

    ValidationRule("E41", "fluid", "MEDIUM",
        "Bernoulli: P + ½ρv² + ρgh = constant (incompressible, inviscid)",
        lambda eid, c: (False, "Bernoulli applied without incompressibility assumption") if re.search(r'Bernoulli|伯努利', c) and re.search(r'(compressible|可压缩|gas|气体)', c) and not re.search(r'(assume|假设|ideal|理想)', c) else (True, None)),

    ValidationRule("E42", "fluid", "MEDIUM",
        "Archimedes: buoyant force = weight of displaced fluid",
        lambda eid, c: (False, "Buoyancy claim without displacement") if re.search(r'(buoy|浮力)', c) and not re.search(r'(displace|排开|volume|体积)', c) else (True, None)),

    ValidationRule("E43", "statistical", "LOW",
        "Entropy S = k·ln(W), Boltzmann formula",
        lambda eid, c: (False, "Entropy without microstate context") if re.search(r'entropy.*S\s*=\s*k', c) and not re.search(r'microstate|微观|W\b', c) else (True, None)),

    ValidationRule("E44", "statistical", "MEDIUM",
        "Law of large numbers: sample mean → population mean as n→∞",
        lambda eid, c: (False, "LLN misinterpretation") if re.search(r'(average|平均).*(guarantee|保证|always|certain|一定)', c) and re.search(r'(small.sample|small.n|少)', c) else (True, None)),

    ValidationRule("E45", "integrity", "HIGH",
        "Einstein entry must not contain contradictory statements about c, G, h, or k",
        lambda eid, c: (False, "Fundamental constant contradiction") if len(re.findall(r'c\s*=\s*3\.0+×10\^?8', c)) > 1 or len(re.findall(r'G\s*=\s*6\.67', c)) > 1 else (True, None)),
]


CEZANNE_RULES = [
    ValidationRule("C01", "syntax", "CRITICAL",
        "Python code blocks must not contain unmatched brackets",
        lambda eid, c: (False, "Unmatched brackets in code") if re.search(r'```python\n', c) and c.count('(') != c.count(')') else (True, None)),

    ValidationRule("C02", "syntax", "CRITICAL",
        "Python code must not have SyntaxError-prone patterns",
        lambda eid, c: (False, "Potential SyntaxError") if re.search(r'def \w+\(.*:.*\)', c) and '```python' in c else (True, None)),

    ValidationRule("C03", "algorithm", "HIGH",
        "Sort complexity: comparison sort lower bound is Ω(n log n)",
        lambda eid, c: (False, "Claims O(n) comparison sort") if re.search(r'(sort|排序).*O\s*\(\s*n\s*\)', c) and re.search(r'comparison|比较', c) else (True, None)),

    ValidationRule("C04", "algorithm", "MEDIUM",
        "Recursion must have base case or termination condition",
        lambda eid, c: (False, "Recursion without base case") if re.search(r'recursion|递归', c) and not re.search(r'(base.case|终止|terminat|boundary)', c) else (True, None)),

    ValidationRule("C05", "algorithm", "MEDIUM",
        "Binary search requires sorted input",
        lambda eid, c: (False, "Binary search without sorting requirement") if re.search(r'binary.search|二分', c) and not re.search(r'(sorted|有序|排序)', c) else (True, None)),

    ValidationRule("C06", "complexity", "HIGH",
        "NP-complete problems cannot be solved in polynomial time (unless P=NP stated)",
        lambda eid, c: (False, "Claims P solution for NP-complete without P=NP context") if re.search(r'(NP.complete|NP完全).*(polynomial|多项式).*(solve|解决)', c) and not re.search(r'P\s*=\s*NP', c) else (True, None)),

    ValidationRule("C07", "complexity", "MEDIUM",
        "Big-O must specify whether average, worst, or best case",
        lambda eid, c: (False, "Big-O without case specification") if re.search(r'O\s*\(', c) and not re.search(r'(worst|best|average|最坏|最好|平均)', c) else (True, None)),

    ValidationRule("C08", "types", "MEDIUM",
        "Type coercion warning: JavaScript == vs === explained",
        lambda eid, c: (False, "Missing type coercion warning") if re.search(r'==\s', c) and 'javascript' in c.lower() and not re.search(r'===|strict|严格|coercion|类型转换', c) else (True, None)),

    ValidationRule("C09", "types", "MEDIUM",
        "Integer overflow risk: large computations should use BigInt or check bounds",
        lambda eid, c: (False, "Integer overflow risk not addressed") if re.search(r'int\s+(?!.*(long|big|64|safe))', c) and re.search(r'(factorial|阶乘|fibonacci|指数|power|overflow)', c) else (True, None)),

    ValidationRule("C10", "security", "CRITICAL",
        "SQL queries must use parameterized statements, not string concatenation",
        lambda eid, c: (False, "SQL injection risk") if re.search(r'(execute|cursor).*[\"\'\(]\s*[\+\%f\"]', c) and 'sql' in c.lower() else (True, None)),

    ValidationRule("C11", "security", "CRITICAL",
        "User input must be sanitized before eval() or exec()",
        lambda eid, c: (False, "Unsanitized eval/exec") if re.search(r'\beval\s*\(|\bexec\s*\(', c) and re.search(r'(user|input|用户|输入)', c) and not re.search(r'(sanitize|clean|过滤|清洗)', c) else (True, None)),

    ValidationRule("C12", "security", "HIGH",
        "Hardcoded credentials or API keys should not appear in code examples",
        lambda eid, c: (False, "Hardcoded secret detected") if re.search(r'(api_key|password|secret|token)\s*=\s*[\'\"][^\'\"]{8,}', c) else (True, None)),

    ValidationRule("C13", "logic", "HIGH",
        "De Morgan's laws: ¬(A∧B) = ¬A∨¬B, ¬(A∨B) = ¬A∧¬B",
        lambda eid, c: (False, "De Morgan violation") if re.search(r'not\s*\(.*and.*\)', c) and re.search(r'not.*and.*not', c) else (True, None)),

    ValidationRule("C14", "logic", "MEDIUM",
        "Loop invariant should be stated for non-trivial while loops",
        lambda eid, c: (False, "While loop without invariant") if re.search(r'while\s', c) and re.search(r'```.*\n', c) and not re.search(r'#\s*invariant|不变式', c) else (True, None)),

    ValidationRule("C15", "memory", "HIGH",
        "malloc/new must have corresponding free/delete",
        lambda eid, c: (False, "Memory leak in C/C++ code") if re.search(r'\bmalloc\s*\(|\bnew\s+\w+', c) and not re.search(r'\bfree\s*\(|\bdelete\s', c) else (True, None)),

    ValidationRule("C16", "memory", "MEDIUM",
        "Python circular references should mention weakref for large objects",
        lambda eid, c: (False, "Circular reference risk unaddressed") if re.search(r'circular|循环引用', c) and 'python' in c.lower() and not re.search(r'weakref', c) else (True, None)),

    ValidationRule("C17", "concurrency", "HIGH",
        "Race condition: shared mutable state needs synchronization primitive",
        lambda eid, c: (False, "Race condition in concurrent code") if re.search(r'(thread|线程|goroutine|async)', c) and re.search(r'(share|共享|global|全局).*(state|状态|variable|变量)', c) and not re.search(r'(lock|mutex|semaphore|atomic|channel)', c) else (True, None)),

    ValidationRule("C18", "concurrency", "MEDIUM",
        "Deadlock prevention: mention lock ordering or timeout strategies",
        lambda eid, c: (False, "Deadlock risk unaddressed") if re.search(r'(deadlock|死锁)', c) and not re.search(r'(timeout|超时|ordering|顺序|trylock|prevent|避免)', c) else (True, None)),

    ValidationRule("C19", "data_structure", "MEDIUM",
        "Hash table O(1) assumes good hash function, worst case is O(n)",
        lambda eid, c: (False, "Hash table complexity without collision caveat") if re.search(r'(hash|哈希|dict|dictionary).*O\s*\(\s*1\s*\)', c) and not re.search(r'(collision|冲突|worst|最坏|amortized|均摊)', c) else (True, None)),

    ValidationRule("C20", "data_structure", "MEDIUM",
        "Balanced tree (AVL/Red-Black) guarantees O(log n), not generic BST",
        lambda eid, c: (False, "BST O(log n) claim without balance guarantee") if re.search(r'BST|二叉搜索树', c) and re.search(r'O\s*\(\s*log', c) and not re.search(r'(AVL|红黑|平衡|balance|Red.Black)', c) else (True, None)),

    ValidationRule("C21", "code_style", "LOW",
        "Python: PEP 8 — 4-space indentation, not tabs",
        lambda eid, c: _check_tabs(c)),

    ValidationRule("C22", "code_style", "LOW",
        "Function should have docstring explaining parameters and return",
        lambda eid, c: _check_docstring(c)),

    ValidationRule("C23", "error_handling", "HIGH",
        "File I/O must handle FileNotFoundError or check existence",
        lambda eid, c: (False, "File operation without error handling") if re.search(r'\bopen\s*\(', c) and 'try' not in c and 'exist' not in c else (True, None)),

    ValidationRule("C24", "error_handling", "MEDIUM",
        "Network requests should handle timeout and connection errors",
        lambda eid, c: (False, "Network call without error handling") if re.search(r'(requests\.|urllib|fetch|http)', c) and 'timeout' not in c and 'try' not in c else (True, None)),

    ValidationRule("C25", "testing", "MEDIUM",
        "Algorithm explanation should include test cases or edge cases",
        lambda eid, c: (False, "No test cases for algorithm") if re.search(r'(algorithm|算法)\b', c) and not re.search(r'(test|测试|edge|边界|example|例子)', c) else (True, None)),

    ValidationRule("C26", "testing", "LOW",
        "assert statements should have descriptive messages",
        lambda eid, c: (False, "Assert without message") if re.search(r'assert\s+\w+', c) and 'assert ' in c and not re.search(r'assert\s+\w+.*,\s*[\'\"]', c) else (True, None)),

    ValidationRule("C27", "database", "MEDIUM",
        "N+1 query problem: eager loading for ORM relationships",
        lambda eid, c: (False, "N+1 query risk") if re.search(r'(for|forEach|循环).*(query|查询|SELECT)', c) and 'orm' in c.lower() and not re.search(r'(eager|join|prefetch|预加载)', c) else (True, None)),

    ValidationRule("C28", "regex", "MEDIUM",
        "ReDoS warning: avoid nested quantifiers like (a+)+ in regex",
        lambda eid, c: (False, "ReDoS risk in regex") if re.search(r'\(\w\+\)\+|\(\w\*\)\*', c) and 'regex' in c.lower() else (True, None)),

    ValidationRule("C29", "unicode", "LOW",
        "Unicode handling: specify encoding (UTF-8) for file/text operations",
        lambda eid, c: (False, "Missing encoding specification") if re.search(r'\bopen\s*\(.*[\'\"][wr]', c) and 'encoding' not in c else (True, None)),

    ValidationRule("C30", "unicode", "LOW",
        "String comparison should normalize Unicode (NFC/NFD)",
        lambda eid, c: (False, "Unicode comparison without normalization") if re.search(r'str.*==.*str|string.*compare|字符串.*比较', c) and re.search(r'unicode|中文|emoji|accent', c) else (True, None)),

    ValidationRule("C31", "design_pattern", "MEDIUM",
        "Singleton pattern: ensure thread safety in multi-threaded context",
        lambda eid, c: (False, "Singleton without thread safety") if re.search(r'singleton|单例', c) and re.search(r'(thread|multi.thread|多线程)', c) and not re.search(r'(lock|synchronize|线程安全)', c) else (True, None)),

    ValidationRule("C32", "design_pattern", "LOW",
        "Factory pattern should explain when to use vs direct instantiation",
        lambda eid, c: (False, "Factory without use-case explanation") if re.search(r'(factory|工厂)', c) and re.search(r'pattern|模式', c) and not re.search(r'(when|when.to|适用|use.case)', c) else (True, None)),

    ValidationRule("C33", "api_design", "MEDIUM",
        "REST API: use proper HTTP methods (GET/POST/PUT/DELETE) and status codes",
        lambda eid, c: (False, "Non-standard HTTP method usage") if re.search(r'(api|API|endpoint)', c) and re.search(r'(GET.*(?:create|delete|update|修改|删除|创建)|POST.*(?:get|查询|fetch))', c) else (True, None)),

    ValidationRule("C34", "api_design", "MEDIUM",
        "API versioning should be explicit (URL path or header)",
        lambda eid, c: (False, "API without versioning") if re.search(r'/api/v\d|API.version', c) and not re.search(r'v\d|version', c) else (True, None)),

    ValidationRule("C35", "caching", "MEDIUM",
        "Cache invalidation strategy must be stated (TTL, event-driven, etc.)",
        lambda eid, c: (False, "Cache without invalidation") if re.search(r'(cache|缓存)', c) and not re.search(r'(invalidate|失效|TTL|expire|过期|evict)', c) else (True, None)),

    ValidationRule("C36", "performance", "MEDIUM",
        "Premature optimization claim must be supported by profiling data reference",
        lambda eid, c: (False, "Performance claim without evidence") if re.search(r'(optimize|优化|faster|更快|performance|性能).*(improve|提升)', c) and not re.search(r'(benchmark|profile|测量|实测)', c) else (True, None)),

    ValidationRule("C37", "floating_point", "HIGH",
        "Floating point comparison must use epsilon tolerance, not ==",
        lambda eid, c: (False, "Float equality without epsilon") if re.search(r'float.*==|double.*==|0\.\d+.*==', c) and not re.search(r'(epsilon|eps|toleran|容差|abs\(.*\)\s*<)', c) else (True, None)),

    ValidationRule("C38", "floating_point", "MEDIUM",
        "Monetary values must use Decimal/BigDecimal, not float/double",
        lambda eid, c: (False, "Money as float") if re.search(r'(money|price|金额|钱|cost).*(float|double)', c) or re.search(r'float.*(money|price|金额|钱)', c) else (True, None)),

    ValidationRule("C39", "crypto", "CRITICAL",
        "Never implement custom cryptography — use standard libraries",
        lambda eid, c: (False, "Custom crypto implementation") if re.search(r'(encrypt|decrypt|加密|解密).*(custom|自|own|自己|DIY)', c) and not re.search(r'(hashlib|cryptography|PyNaCl|libsodium)', c) else (True, None)),

    ValidationRule("C40", "crypto", "HIGH",
        "MD5/SHA1 must be noted as cryptographically broken for security use",
        lambda eid, c: (False, "MD5/SHA1 without deprecation warning") if re.search(r'(MD5|SHA-?1)\b', c) and re.search(r'(password|security|安全|auth|hash)', c) and not re.search(r'(deprecated|过时|broken|不安全|not.secure)', c) else (True, None)),

    ValidationRule("C41", "os", "MEDIUM",
        "Path handling: use os.path.join or pathlib, not string concatenation",
        lambda eid, c: (False, "Platform-dependent path construction") if re.search(r'[\'\"][\\/]\w+[\\/]\w+[\'\"]\+|path\s*=\s*[\'\"].*\\\\', c) else (True, None)),

    ValidationRule("C42", "os", "LOW",
        "Temporary files: use tempfile module, not hardcoded /tmp paths",
        lambda eid, c: (False, "Hardcoded temp path") if re.search(r'(/tmp/|C:\\Temp\\)', c) and 'python' in c.lower() and 'tempfile' not in c else (True, None)),

    ValidationRule("C43", "logging", "LOW",
        "Production code should use logging module, not print()",
        lambda eid, c: (False, "print() in production context") if re.search(r'\bprint\s*\(', c) and re.search(r'(production|生产|deploy|部署)', c) and 'logging' not in c else (True, None)),

    ValidationRule("C44", "dependency", "MEDIUM",
        "Package import should specify version constraint or compatibility note",
        lambda eid, c: (False, "Package without version constraint") if re.search(r'(pip install|import)\s+\w+', c) and not re.search(r'==\d|>=|<=|version|版本', c) else (True, None)),

    ValidationRule("C45", "documentation", "LOW",
        "README/docstring should explain WHY, not just WHAT the code does",
        lambda eid, c: (False, "Missing rationale") if re.search(r'```\w+\n(?:[^`\n]*\n){5,}```', c) and not re.search(r'(reason|rationale|because|why|原因|背景)', c) else (True, None)),

    ValidationRule("C46", "naming", "LOW",
        "Variable names should be descriptive (min 3 chars, no single letters except i/j/k loops)",
        lambda eid, c: (False, "Excessive single-letter variables") if len(re.findall(r'\b[a-zA-Z]\b(?=\s*[=:])', c)) > 5 else (True, None)),

    ValidationRule("C47", "time", "MEDIUM",
        "Time handling: use UTC internally, convert to local only for display",
        lambda eid, c: (False, "Timezone handling issue") if re.search(r'(time|时间).*(local|本地)', c) and 'UTC' not in c else (True, None)),

    ValidationRule("C48", "serialization", "MEDIUM",
        "JSON serialization: datetime/Decimal must be explicitly handled",
        lambda eid, c: (False, "Json serialization without type handling") if re.search(r'json\.dumps?\s*\(', c) and not re.search(r'(default=|cls=|encoder|custom)', c) else (True, None)),

    ValidationRule("C49", "state", "MEDIUM",
        "State machine transitions should be explicitly enumerated",
        lambda eid, c: (False, "State machine without transitions") if re.search(r'(state.machine|状态机|FSM)', c) and not re.search(r'(transition|转移|state\s*->|→)', c) else (True, None)),

    ValidationRule("C50", "graph", "MEDIUM",
        "Graph algorithm must specify directed/undirected and weighted/unweighted",
        lambda eid, c: (False, "Graph type unspecified") if re.search(r'(graph|图).*(algorithm|算法)', c) and not re.search(r'(directed|undirected|有向|无向|weighted|加权)', c) else (True, None)),

    ValidationRule("C51", "database", "HIGH",
        "Database transaction: ACID properties should be mentioned for writes",
        lambda eid, c: (False, "Write without transaction") if re.search(r'(INSERT|UPDATE|DELETE|写入|更新|删除)', c) and 'sql' in c.lower() and not re.search(r'(transaction|事务|commit|rollback)', c) else (True, None)),

    ValidationRule("C52", "networking", "MEDIUM",
        "Port numbers > 1024 for user services, < 1024 require root/admin",
        lambda eid, c: (False, "Privileged port without root note") if re.search(r'port.*[=:]?\s*[1-9]\d{0,2}\b', c) else (True, None)),

    ValidationRule("C53", "compression", "MEDIUM",
        "Compression ratio claims must specify input data characteristics",
        lambda eid, c: (False, "Compression claim without data context") if re.search(r'(compress|压缩).*(ratio|比率)\s*[=:]*\s*\d+', c) and not re.search(r'(text|image|binary|文本|图像)', c) else (True, None)),

    ValidationRule("C54", "encoding", "LOW",
        "Base64/hex encoding: distinguish from encryption (encoding ≠ security)",
        lambda eid, c: (False, "Encoding presented as security") if re.search(r'base64|hex\s', c) and re.search(r'(secure|安全|encrypt|加密|protect|保护)', c) and 'NOT' not in c else (True, None)),

    ValidationRule("C55", "cli", "LOW",
        "CLI tool should support --help and exit codes (0=success, non-zero=error)",
        lambda eid, c: (False, "CLI without help/exit codes") if re.search(r'argparse|click|cli|命令行', c) and not re.search(r'(help|exit|--help)', c) else (True, None)),

    ValidationRule("C56", "composition", "MEDIUM",
        "Prefer composition over inheritance when possible (mention trade-off)",
        lambda eid, c: (False, "Inheritance without composition discussion") if re.search(r'class \w+\(\w+\)', c) and 'python' in c.lower() and re.search(r'inherit|继承', c) and 'compos' not in c else (True, None)),

    ValidationRule("C57", "robustness", "MEDIUM",
        "External API call should implement retry with exponential backoff",
        lambda eid, c: (False, "API call without retry") if re.search(r'(requests\.|fetch|http\.)', c) and not re.search(r'(retry|重试|backoff|exponential)', c) else (True, None)),

    ValidationRule("C58", "immutability", "MEDIUM",
        "Mutable default arguments in Python: use None sentinel",
        lambda eid, c: (False, "Mutable default argument") if re.search(r'def \w+\(.*=\s*\[|def \w+\(.*=\s*\{', c) else (True, None)),

    ValidationRule("C59", "iteration", "LOW",
        "Modifying collection while iterating: use copy or list comprehension",
        lambda eid, c: (False, "Iteration over mutating collection") if re.search(r'for.*in.*\w+:\s*\n\s*\w+\.(?:remove|pop|append)', c) else (True, None)),

    ValidationRule("C60", "knowledge_integrity", "HIGH",
        "Code entry must be self-contained or reference external source",
        lambda eid, c: (False, "Incomplete code fragment") if re.search(r'# TODO|# FIXME|\.\.\..*omitted|# ...|pass\s*$', c) and '```' in c else (True, None)),
]


class ValidationEngine:
    def __init__(self, kb_dir: Optional[Path] = None):
        self.kb_dir = kb_dir or KB_DIR
        self.rules = {
            "einstein": EINSTEIN_RULES,
            "cezanne": CEZANNE_RULES,
        }

    def _read_entries(self, kb_name: str, limit: int = 2000) -> List[Tuple[str, str]]:
        db_path = self.kb_dir / f"{kb_name}.db"
        if not db_path.exists():
            return []
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT entry_id, content FROM memories ORDER BY RANDOM() LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [(r["entry_id"], r["content"]) for r in rows]

    def validate(self, kb_name: str, limit: int = 2000) -> AuditReport:
        rules = self.rules.get(kb_name, [])
        entries = self._read_entries(kb_name, limit)
        violations: List[Violation] = []
        passed = 0
        failed = 0

        if not entries:
            return AuditReport(
                kb_name=kb_name,
                timestamp=datetime.now().isoformat(),
                total_rules=len(rules),
                passed=0, failed=0,
                violations=[]
            )

        for rule in rules:
            rule_failed = False
            for eid, content in entries:
                ok, violation = rule.check(eid, content)
                if not ok and violation:
                    violations.append(violation)
                    rule_failed = True

            if rule_failed:
                failed += 1
            else:
                passed += 1

        return AuditReport(
            kb_name=kb_name,
            timestamp=datetime.now().isoformat(),
            total_rules=len(rules),
            passed=passed,
            failed=failed,
            violations=violations
        )

    def validate_all(self, limit: int = 2000) -> Dict[str, AuditReport]:
        return {name: self.validate(name, limit) for name in self.rules}


if __name__ == "__main__":
    engine = ValidationEngine()
    for name in ["einstein", "cezanne"]:
        report = engine.validate(name, limit=500)
        print(report.summary())
        print()
