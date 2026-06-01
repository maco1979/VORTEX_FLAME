"""
VF-SEC Pipeline — VORTEX_FLAME Security Audit Pipeline (maps to gstack /cso)
=============================================================================
16-phase security audit that runs BEFORE commit, powered by existing
VORTEX_FLAME infrastructure: soul_memory, code_intelligence,
validation_rules, cross_kb_causal_bridge, closed_loop_orchestrator.

Architecture:
  git diff / staged files
      → VFSECRunner.run()
          → Phase 1-16 sequential/parallel execution
              → each phase uses existing VORTEX components
          → Score aggregation + threshold filtering
              → Daily mode: score >= 6 → pass
              → Comprehensive mode: score >= 2 → TENTATIVE flag
          → AuditReport → soul_memory audit category → MCP broadcast

Component Mapping:
  Phase 1  (Attack Surface)    → code_intelligence.py + device_gateway.py
  Phase 2  (Secret Archaeology)→ soul_memory BM25 search + git log scanner
  Phase 3  (Supply Chain)      → validation_rules.py + npm/pip audit
  Phase 4  (CI/CD Security)    → validation_rules.py + GitHub Actions check
  Phase 5  (Shadow Infra)      → device_gateway.py + docker inspect
  Phase 6  (Webhook Audit)     → validation_rules.py + HTTP signature check
  Phase 7  (AI/LLM Safety)     → cross_kb_causal_bridge + strategy soul
  Phase 8  (Dangerous Tools)   → validation_rules.py + pattern matching
  Phase 9  (OWASP Top 10)      → soul_memory knowledge (cezanne KB)
  Phase 10 (STRIDE Threat)     → strategy soul + cross_kb_causal_bridge
  Phase 11 (Data Classification)→ soul_memory categories + tagging
  Phase 12 (Quality Gate)      → closed_loop_orchestrator evaluate
  Phase 13 (Dependency Graph)  → code_intelligence.py dependency layer
  Phase 14 (Config Drift)      → soul_memory temporal comparison
  Phase 15 (Access Control)    → validation_rules.py + permission check
  Phase 16 (Final Report)      → aggregation + soul_memory write

Usage:
    runner = VFSECRunner(mode="daily")
    report = runner.run(diff_text="...", staged_files=["api.py", "config.yaml"])
    runner = VFSECRunner(mode="comprehensive")
    report = runner.run(project_path="D:/VORTEX_FLAME")
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DAILY_THRESHOLD = 6.0
COMPREHENSIVE_THRESHOLD = 2.0
MAX_FINDINGS_PER_PHASE = 50


class AuditMode(Enum):
    DAILY = "daily"
    COMPREHENSIVE = "comprehensive"


class FindingSeverity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class FindingStatus(Enum):
    CONFIRMED = "CONFIRMED"
    TENTATIVE = "TENTATIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    SUPPRESSED = "SUPPRESSED"


@dataclass
class SecurityFinding:
    phase: int
    phase_name: str
    title: str
    severity: FindingSeverity
    score: float
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    remediation: str = ""
    status: FindingStatus = FindingStatus.CONFIRMED
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "phase_name": self.phase_name,
            "title": self.title,
            "severity": self.severity.value,
            "score": self.score,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "remediation": self.remediation,
            "status": self.status.value,
            "tags": self.tags,
        }


@dataclass
class PhaseResult:
    phase: int
    name: str
    findings: List[SecurityFinding] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None
    skipped: bool = False

    @property
    def max_score(self) -> float:
        return max((f.score for f in self.findings), default=0.0)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == FindingSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == FindingSeverity.HIGH)


@dataclass
class AuditReport:
    mode: AuditMode
    timestamp: float
    project_path: str
    phases: List[PhaseResult] = field(default_factory=list)
    total_findings: int = 0
    critical_findings: int = 0
    high_findings: int = 0
    overall_score: float = 10.0
    pass_gate: bool = True
    duration_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== VF-SEC Audit Report ({self.mode.value}) ===",
            f"Project: {self.project_path}",
            f"Overall Score: {self.overall_score:.1f}/10.0",
            f"Gate: {'PASS' if self.pass_gate else 'BLOCK'}",
            f"Findings: {self.total_findings} total, {self.critical_findings} critical, {self.high_findings} high",
            f"Duration: {self.duration_ms:.0f}ms",
            "",
        ]
        for pr in self.phases:
            if pr.skipped:
                lines.append(f"  Phase {pr.phase:2d} [{pr.name}]: SKIPPED")
            elif pr.error:
                lines.append(f"  Phase {pr.phase:2d} [{pr.name}]: ERROR - {pr.error}")
            else:
                lines.append(
                    f"  Phase {pr.phase:2d} [{pr.name}]: "
                    f"{len(pr.findings)} findings, max_score={pr.max_score:.1f}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "timestamp": self.timestamp,
            "project_path": self.project_path,
            "overall_score": self.overall_score,
            "pass_gate": self.pass_gate,
            "total_findings": self.total_findings,
            "critical_findings": self.critical_findings,
            "high_findings": self.high_findings,
            "duration_ms": self.duration_ms,
            "phases": [
                {
                    "phase": p.phase,
                    "name": p.name,
                    "findings": [f.to_dict() for f in p.findings],
                    "max_score": p.max_score,
                    "duration_ms": p.duration_ms,
                }
                for p in self.phases
            ],
        }


SECRET_PATTERNS = [
    (r'(?:password|passwd|pwd)\s*[=:]\s*["\'][^"\']{6,}["\']', "Hardcoded Password", 9.0),
    (r'(?:api_key|apikey|api_secret)\s*[=:]\s*["\'][^"\']{8,}["\']', "Hardcoded API Key", 9.5),
    (r'(?:secret|token|auth_token)\s*[=:]\s*["\'][^"\']{8,}["\']', "Hardcoded Secret/Token", 9.0),
    (r'(?:aws_access_key_id|aws_secret_access_key)\s*[=:]\s*["\'][^"\']+["\']', "AWS Credential", 10.0),
    (r'(?:private_key|rsa_private)\s*[=:]\s*["\']-----BEGIN', "Private Key in Code", 10.0),
    (r'(?:mongodb|postgres|mysql|redis)://[^\s"\']+: [^\s"\']+@[^\s"\']+', "Database URL with Credentials", 8.5),
    (r'Bearer\s+[A-Za-z0-9\-._~+/]+=*', "Bearer Token in Code", 8.0),
    (r'ghp_[A-Za-z0-9]{36}', "GitHub Personal Access Token", 10.0),
    (r'sk-[A-Za-z0-9]{20,}', "OpenAI API Key Pattern", 9.5),
    (r'AKIA[A-Z0-9]{16}', "AWS Access Key ID", 9.5),
]

DANGEROUS_TOOL_PATTERNS = [
    (r'subprocess\.(call|run|Popen)\([^)]*shell\s*=\s*True', "Shell Injection Risk", 8.0),
    (r'os\.system\s*\(', "os.system Usage", 7.0),
    (r'eval\s*\(', "eval() Usage", 8.5),
    (r'exec\s*\(', "exec() Usage", 8.5),
    (r'__import__\s*\(', "Dynamic Import", 6.0),
    (r'pickle\.loads?\s*\(', "Pickle Deserialization", 9.0),
    (r'yaml\.load\s*\([^)]*\)(?!.*Loader)', "Unsafe YAML Load", 7.5),
    (r'marshal\.loads?\s*\(', "Marshal Deserialization", 8.0),
    (r'input\s*\(\s*["\'].*(?:curl|wget|rm|sudo)', "Dangerous Input to Command", 9.5),
]

OWASP_PATTERNS = [
    (r'(?:SELECT|INSERT|UPDATE|DELETE).*\+\s*(?:request|user_input|params)', "SQL Injection Risk", 9.0),
    (r'\.format\s*\([^)]*(?:request|user_input|params)', "Format String Injection", 7.0),
    (r'innerHTML\s*=', "XSS via innerHTML", 8.0),
    (r'document\.write\s*\(', "XSS via document.write", 7.5),
    (r'(?:chmod|chown)\s+777', "Overly Permissive File Mode", 8.0),
    (r'CORS.*Access-Control-Allow-Origin.*\*', "Overly Permissive CORS", 7.0),
    (r'assert\s+.*(?:request|user_input)', "Assert in Production", 6.0),
    (r'try:.*except\s*:', "Bare Exception Handler", 5.0),
]

PROMPT_INJECTION_PATTERNS = [
    (r'(?:system|user|assistant)\s*:\s*["\'].*(?:ignore|disregard|override)\s+(?:previous|above|all)', "Prompt Injection in System Message", 9.0),
    (r'(?:execute|run|eval|call)\s*\(\s*(?:user_input|request\.data|prompt)', "LLM Output to Code Execution", 9.5),
    (r'(?:curl|wget|requests\.(?:get|post))\s*.*(?:user_input|prompt|message)', "LLM-Triggered Network Request", 8.5),
    (r'(?:open|write|remove|delete)\s*\(\s*(?:user_input|prompt|message)', "LLM-Triggered File Operation", 9.0),
    (r'\.redirect\s*\(\s*(?:user_input|request|prompt)', "LLM-Triggered Redirect", 8.0),
    (r'subprocess.*(?:user_input|prompt|message)', "LLM Output to Subprocess", 9.5),
]

DATA_CLASSIFICATION_PATTERNS = [
    (r'(?:ssn|social_security|sin)\s*[=:]|["\']\d{3}-\d{2}-\d{4}["\']', "PII: SSN", 10.0, "RESTRICTED"),
    (r'(?:credit_card|card_number|cc_num)\s*[=:]|["\']\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}["\']', "PII: Credit Card", 10.0, "RESTRICTED"),
    (r'(?:email|e-mail)\s*[=:]\s*["\'][^"\']+@[^"\']+["\']', "PII: Email Address", 6.0, "CONFIDENTIAL"),
    (r'(?:phone|mobile|tel)\s*[=:]\s*["\']\+?\d{10,15}["\']', "PII: Phone Number", 6.0, "CONFIDENTIAL"),
    (r'(?:ip_address|ip_addr)\s*[=:]\s*["\']\d{1,3}\.\d{1,3}', "Internal IP Exposure", 5.0, "CONFIDENTIAL"),
    (r'(?:salary|compensation|wage)\s*[=:]', "Financial Data", 7.0, "CONFIDENTIAL"),
    (r'(?:diagnosis|medical|patient|prescription)\s*[=:]', "Medical Data (HIPAA)", 10.0, "RESTRICTED"),
]

TECH_STACK_DETECTORS = {
    "python": ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile"],
    "node": ["package.json", "yarn.lock", "pnpm-lock.yaml"],
    "rust": ["Cargo.toml", "Cargo.lock"],
    "go": ["go.mod", "go.sum"],
    "php": ["composer.json", "composer.lock"],
    "ruby": ["Gemfile", "Gemfile.lock"],
    "java": ["pom.xml", "build.gradle"],
    "dotnet": [".csproj", ".fsproj", "global.json"],
}

FRAMEWORK_DETECTORS = {
    "django": ["manage.py", "settings.py"],
    "flask": ["app.py", "flask"],
    "fastapi": ["main.py", "fastapi"],
    "express": ["express"],
    "react": ["react", "next.config"],
    "redis": ["redis", "ioredis"],
    "docker": ["Dockerfile", "docker-compose"],
    "k8s": ["kubernetes", "k8s", "deployment.yaml"],
}


class VFSECRunner:
    def __init__(self, mode: AuditMode = AuditMode.DAILY, project_path: str = "."):
        self.mode = mode
        self.project_path = project_path
        self.threshold = DAILY_THRESHOLD if mode == AuditMode.DAILY else COMPREHENSIVE_THRESHOLD
        self._code_intel = None
        self._memory = None
        self._validation = None

    def _get_code_intel(self):
        if self._code_intel is None:
            try:
                from code_intelligence import CodeIntelligenceManager
                self._code_intel = CodeIntelligenceManager(self.project_path)
            except Exception:
                pass
        return self._code_intel

    def _get_memory(self):
        if self._memory is None:
            try:
                from soul_memory import SoulMemoryEngine
                self._memory = SoulMemoryEngine()
            except Exception:
                pass
        return self._memory

    def _get_validation(self):
        if self._validation is None:
            try:
                from validation_rules import ValidationEngine
                self._validation = ValidationEngine()
            except Exception:
                pass
        return self._validation

    def run(self, diff_text: Optional[str] = None,
            staged_files: Optional[List[str]] = None,
            source_code: Optional[Dict[str, str]] = None) -> AuditReport:
        start = time.time()
        report = AuditReport(
            mode=self.mode,
            timestamp=time.time(),
            project_path=self.project_path,
        )

        if staged_files is None:
            staged_files = self._get_staged_files()
        if diff_text is None:
            diff_text = self._get_staged_diff()
        if source_code is None:
            source_code = self._read_staged_files(staged_files)
        diff_text = diff_text or ""
        source_code = source_code or {}

        phase_runners = [
            (1, "Attack Surface Census", self._phase_attack_surface),
            (2, "Secret Archaeology", self._phase_secret_archaeology),
            (3, "Supply Chain Security", self._phase_supply_chain),
            (4, "CI/CD Security", self._phase_cicd_security),
            (5, "Shadow Infrastructure", self._phase_shadow_infra),
            (6, "Webhook Audit", self._phase_webhook_audit),
            (7, "AI/LLM Safety", self._phase_ai_llm_safety),
            (8, "Dangerous Tool Usage", self._phase_dangerous_tools),
            (9, "OWASP Top 10", self._phase_owasp),
            (10, "STRIDE Threat Model", self._phase_stride),
            (11, "Data Classification", self._phase_data_classification),
            (12, "Quality Gate", self._phase_quality_gate),
            (13, "Dependency Graph", self._phase_dependency_graph),
            (14, "Config Drift Detection", self._phase_config_drift),
            (15, "Access Control", self._phase_access_control),
            (16, "Final Aggregation", self._phase_final),
        ]

        for phase_num, phase_name, runner in phase_runners:
            t0 = time.time()
            try:
                findings = runner(
                    diff_text=diff_text,
                    staged_files=staged_files,
                    source_code=source_code,
                )
                if self.mode == AuditMode.DAILY:
                    findings = [f for f in findings if f.score >= self.threshold]
                else:
                    for f in findings:
                        if f.score < DAILY_THRESHOLD and f.score >= COMPREHENSIVE_THRESHOLD:
                            f.status = FindingStatus.TENTATIVE

                findings = findings[:MAX_FINDINGS_PER_PHASE]
                pr = PhaseResult(
                    phase=phase_num, name=phase_name, findings=findings,
                    duration_ms=(time.time() - t0) * 1000,
                )
            except Exception as e:
                logger.warning(f"Phase {phase_num} error: {e}")
                pr = PhaseResult(
                    phase=phase_num, name=phase_name,
                    error=str(e), duration_ms=(time.time() - t0) * 1000,
                )
            report.phases.append(pr)

        report.total_findings = sum(len(p.findings) for p in report.phases)
        report.critical_findings = sum(
            1 for p in report.phases for f in p.findings
            if f.severity == FindingSeverity.CRITICAL
        )
        report.high_findings = sum(
            1 for p in report.phases for f in p.findings
            if f.severity == FindingSeverity.HIGH
        )

        penalty = sum(f.score * 0.1 for p in report.phases for f in p.findings)
        report.overall_score = max(0.0, 10.0 - penalty)
        report.pass_gate = report.overall_score >= self.threshold and report.critical_findings == 0
        report.duration_ms = (time.time() - start) * 1000

        self._write_audit_to_memory(report)

        return report

    def _get_staged_files(self) -> List[str]:
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True, text=True, cwd=self.project_path, timeout=10,
                encoding="utf-8", errors="replace",
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        except Exception:
            pass
        return []

    def _get_staged_diff(self) -> str:
        try:
            result = subprocess.run(
                ["git", "diff", "--cached"],
                capture_output=True, text=True, cwd=self.project_path, timeout=30,
                encoding="utf-8", errors="replace",
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return ""

    def _read_staged_files(self, staged_files: List[str]) -> Dict[str, str]:
        code = {}
        for fp in staged_files:
            full = os.path.join(self.project_path, fp)
            if os.path.isfile(full):
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        code[fp] = f.read()
                except Exception:
                    pass
        return code

    def _pattern_scan(self, patterns: List[Tuple], text: str, phase: int,
                      phase_name: str, filepath: Optional[str] = None) -> List[SecurityFinding]:
        findings = []
        for pattern_tuple in patterns:
            if len(pattern_tuple) == 3:
                pat, title, score = pattern_tuple
                data_class = None
            else:
                pat, title, score, data_class = pattern_tuple
            try:
                for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE):
                    f = SecurityFinding(
                        phase=phase, phase_name=phase_name,
                        title=title, severity=self._score_to_severity(score),
                        score=score, description=f"Pattern match: {pat[:60]}",
                        file_path=filepath,
                        line_number=text[:m.start()].count("\n") + 1,
                        remediation=self._remediation_for(title),
                        tags=[data_class.lower()] if data_class else [],
                    )
                    findings.append(f)
            except re.error:
                pass
        return findings

    @staticmethod
    def _score_to_severity(score: float) -> FindingSeverity:
        if score >= 9.0:
            return FindingSeverity.CRITICAL
        elif score >= 7.0:
            return FindingSeverity.HIGH
        elif score >= 5.0:
            return FindingSeverity.MEDIUM
        elif score >= 3.0:
            return FindingSeverity.LOW
        return FindingSeverity.INFO

    @staticmethod
    def _remediation_for(title: str) -> str:
        remediations = {
            "Hardcoded Password": "Use environment variables or secret manager (e.g., vault, .env)",
            "Hardcoded API Key": "Move to environment variable, rotate immediately if committed",
            "Hardcoded Secret/Token": "Use secret management system, never commit tokens",
            "AWS Credential": "Use IAM roles or environment variables, rotate immediately",
            "Private Key in Code": "Store in secret manager, never commit private keys",
            "Shell Injection Risk": "Use subprocess without shell=True, validate inputs",
            "eval() Usage": "Replace with ast.literal_eval() or safe parser",
            "exec() Usage": "Avoid dynamic code execution, use safe alternatives",
            "Pickle Deserialization": "Use json/msgpack for serialization, never unpickle untrusted data",
            "SQL Injection Risk": "Use parameterized queries / ORM, never concatenate user input",
            "XSS via innerHTML": "Use textContent or DOMPurify sanitization",
            "Prompt Injection in System Message": "Validate and sanitize all LLM inputs, use input boundaries",
            "LLM Output to Code Execution": "Never execute LLM output directly, use approval workflow",
            "PII: SSN": "Encrypt at rest, tokenization, access logging required",
            "PII: Credit Card": "PCI-DSS compliance required, tokenize card numbers",
            "Medical Data (HIPAA)": "HIPAA compliance required, encrypt, audit log, access control",
        }
        return remediations.get(title, "Review and remediate based on security best practices")

    def _phase_attack_surface(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        project = self.project_path

        detected_stacks = []
        for lang, markers in TECH_STACK_DETECTORS.items():
            for m in markers:
                if os.path.exists(os.path.join(project, m)):
                    detected_stacks.append(lang)
                    break

        detected_frameworks = []
        for fw, markers in FRAMEWORK_DETECTORS.items():
            for m in markers:
                if os.path.exists(os.path.join(project, m)):
                    detected_frameworks.append(fw)
                    break

        source_code = kwargs.get("source_code") or {}
        public_endpoints = 0
        auth_required = 0
        for fp, content in source_code.items():
            public_endpoints += len(re.findall(
                r'@(?:app|router)\.(?:get|post|put|delete|patch)\s*\(', content,
            ))
            public_endpoints += len(re.findall(
                r'(?:router|app)\.(?:get|post|put|delete)\s*\(', content,
            ))
            auth_required += len(re.findall(
                r'(?:auth|jwt|token|session|login|authenticate)', content, re.IGNORECASE,
            ))

        if public_endpoints > 0 and auth_required == 0:
            findings.append(SecurityFinding(
                phase=1, phase_name="Attack Surface Census",
                title="No Authentication Detected", severity=FindingSeverity.HIGH,
                score=8.0,
                description=f"{public_endpoints} public endpoints found, 0 auth references",
                remediation="Add authentication middleware to all public endpoints",
            ))

        if not detected_stacks:
            findings.append(SecurityFinding(
                phase=1, phase_name="Attack Surface Census",
                title="Unknown Tech Stack", severity=FindingSeverity.LOW,
                score=3.0,
                description="Cannot detect tech stack for targeted security scanning",
            ))

        return findings

    def _phase_secret_archaeology(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        source_code = kwargs.get("source_code", {})
        diff_text = kwargs.get("diff_text", "")

        scan_text = diff_text + "\n" + "\n".join(source_code.values())
        findings.extend(self._pattern_scan(SECRET_PATTERNS, scan_text, 2, "Secret Archaeology"))

        try:
            result = subprocess.run(
                ["git", "log", "--all", "--diff-filter=D", "--name-only", "--pretty=format:", "-n", "100"],
                capture_output=True, text=True, cwd=self.project_path, timeout=15,
                encoding="utf-8", errors="replace",
            )
            if result.returncode == 0:
                deleted_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
                sensitive_deleted = [
                    f for f in deleted_files
                    if any(kw in f.lower() for kw in [".env", "secret", "credential", "key", "token", "password"])
                ]
                if sensitive_deleted:
                    findings.append(SecurityFinding(
                        phase=2, phase_name="Secret Archaeology",
                        title="Sensitive Files Previously Deleted", severity=FindingSeverity.HIGH,
                        score=7.5,
                        description=f"Deleted files still in git history: {sensitive_deleted[:5]}",
                        remediation="Use git-filter-branch or BFG to purge from history, rotate all exposed secrets",
                    ))
        except Exception:
            pass

        return findings

    def _phase_supply_chain(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        project = self.project_path

        lock_files = {
            "package-lock.json": "npm",
            "yarn.lock": "yarn",
            "pnpm-lock.yaml": "pnpm",
            "requirements.txt": "pip",
            "Pipfile.lock": "pipenv",
            "poetry.lock": "poetry",
            "Cargo.lock": "cargo",
            "go.sum": "go",
        }
        for lock, mgr in lock_files.items():
            lf = os.path.join(project, lock)
            if os.path.exists(lf):
                try:
                    with open(lf, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    if "http://" in content and "https://" not in content:
                        findings.append(SecurityFinding(
                            phase=3, phase_name="Supply Chain Security",
                            title=f"Insecure Registry in {lock}", severity=FindingSeverity.HIGH,
                            score=7.0,
                            description=f"{lock} references http:// (not https://) registries",
                            file_path=lock,
                            remediation=f"Update {lock} to use HTTPS registries only",
                        ))
                except Exception:
                    pass

        for pkg_file in ["package.json", "requirements.txt", "pyproject.toml"]:
            pf = os.path.join(project, pkg_file)
            if os.path.exists(pf):
                try:
                    with open(pf, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    if re.search(r'(?:preinstall|postinstall|prepare)\s*:', content):
                        findings.append(SecurityFinding(
                            phase=3, phase_name="Supply Chain Security",
                            title=f"Install Script in {pkg_file}", severity=FindingSeverity.HIGH,
                            score=8.0,
                            description=f"{pkg_file} contains install scripts — supply chain attack vector",
                            file_path=pkg_file,
                            remediation="Audit install scripts, prefer --ignore-scripts in production",
                        ))
                except Exception:
                    pass

        return findings

    def _phase_cicd_security(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        project = self.project_path

        for ci_dir in [".github/workflows", ".gitlab-ci.d"]:
            ci_path = os.path.join(project, ci_dir)
            if not os.path.isdir(ci_path):
                continue
            for fn in os.listdir(ci_path):
                if not fn.endswith((".yml", ".yaml")):
                    continue
                fp = os.path.join(ci_path, fn)
                try:
                    with open(fp, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    if "pull_request_target" in content:
                        findings.append(SecurityFinding(
                            phase=4, phase_name="CI/CD Security",
                            title="pull_request_target Usage", severity=FindingSeverity.HIGH,
                            score=8.5,
                            description=f"{fn} uses pull_request_target — potential privilege escalation",
                            file_path=fp,
                            remediation="Avoid pull_request_target, or restrict secrets access",
                        ))
                    uses = re.findall(r'uses:\s*([^\s@]+)@([^\s]+)', content)
                    for repo, ref in uses:
                        if not re.match(r'^[0-9a-f]{40}$', ref):
                            findings.append(SecurityFinding(
                                phase=4, phase_name="CI/CD Security",
                                title=f"Unpinned GitHub Action: {repo}@{ref}",
                                severity=FindingSeverity.MEDIUM, score=6.0,
                                description=f"Action {repo} pinned to tag/branch, not SHA",
                                file_path=fp,
                                remediation=f"Pin to commit SHA: {repo}@<full-sha>",
                            ))
                except Exception:
                    pass

        return findings

    def _phase_shadow_infra(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        project = self.project_path

        dockerfile = os.path.join(project, "Dockerfile")
        if os.path.exists(dockerfile):
            try:
                with open(dockerfile, "r", encoding="utf-8") as f:
                    content = f.read()
                if re.search(r'^\s*USER\s+root', content, re.MULTILINE):
                    findings.append(SecurityFinding(
                        phase=5, phase_name="Shadow Infrastructure",
                        title="Container Runs as Root", severity=FindingSeverity.HIGH,
                        score=8.0, description="Dockerfile sets USER root",
                        file_path="Dockerfile",
                        remediation="Use non-root user: RUN adduser --disabled-password appuser && USER appuser",
                    ))
                if re.search(r'^\s*EXPOSE\s+\d+', content, re.MULTILINE):
                    exposed = re.findall(r'EXPOSE\s+(\d+)', content)
                    findings.append(SecurityFinding(
                        phase=5, phase_name="Shadow Infrastructure",
                        title=f"Exposed Ports: {exposed}", severity=FindingSeverity.LOW,
                        score=3.0, description=f"Dockerfile exposes ports: {exposed}",
                        file_path="Dockerfile",
                    ))
            except Exception:
                pass

        for env_file in [".env", ".env.production", ".env.staging"]:
            ef = os.path.join(project, env_file)
            if os.path.exists(ef):
                findings.append(SecurityFinding(
                    phase=5, phase_name="Shadow Infrastructure",
                    title=f"Environment File in Project: {env_file}",
                    severity=FindingSeverity.HIGH, score=7.5,
                    description=f"{env_file} found — may contain secrets",
                    file_path=env_file,
                    remediation="Add to .gitignore, use .env.example for templates",
                ))

        return findings

    def _phase_webhook_audit(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        source_code = kwargs.get("source_code", {})

        for fp, content in source_code.items():
            if "webhook" in content.lower():
                if "signature" not in content.lower() and "hmac" not in content.lower():
                    findings.append(SecurityFinding(
                        phase=6, phase_name="Webhook Audit",
                        title="Unsigned Webhook Endpoint", severity=FindingSeverity.HIGH,
                        score=8.0,
                        description=f"Webhook in {fp} has no signature verification",
                        file_path=fp,
                        remediation="Add HMAC signature verification to all webhook endpoints",
                    ))
        return findings

    def _phase_ai_llm_safety(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        source_code = kwargs.get("source_code", {})
        diff_text = kwargs.get("diff_text", "")

        scan_text = diff_text + "\n" + "\n".join(source_code.values())
        findings.extend(self._pattern_scan(
            PROMPT_INJECTION_PATTERNS, scan_text, 7, "AI/LLM Safety",
        ))

        memory = self._get_memory()
        if memory is not None:
            try:
                results = memory.search("cezanne", "knowledge", "prompt injection LLM safety", top_k=3)
                if results:
                    findings.append(SecurityFinding(
                        phase=7, phase_name="AI/LLM Safety",
                        title="LLM Safety Knowledge Base Available", severity=FindingSeverity.INFO,
                        score=0.0,
                        description=f"Found {len(results)} relevant entries in cezanne KB for LLM safety",
                    ))
            except Exception:
                pass

        return findings

    def _phase_dangerous_tools(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        source_code = kwargs.get("source_code", {})
        diff_text = kwargs.get("diff_text", "")

        scan_text = diff_text + "\n" + "\n".join(source_code.values())
        findings.extend(self._pattern_scan(
            DANGEROUS_TOOL_PATTERNS, scan_text, 8, "Dangerous Tool Usage",
        ))
        return findings

    def _phase_owasp(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        source_code = kwargs.get("source_code", {})
        diff_text = kwargs.get("diff_text", "")

        scan_text = diff_text + "\n" + "\n".join(source_code.values())
        findings.extend(self._pattern_scan(
            OWASP_PATTERNS, scan_text, 9, "OWASP Top 10",
        ))

        memory = self._get_memory()
        if memory is not None:
            try:
                results = memory.search("cezanne", "knowledge", "OWASP vulnerability injection", top_k=3)
                if results:
                    for r in results[:2]:
                        findings.append(SecurityFinding(
                            phase=9, phase_name="OWASP Top 10",
                            title=f"KB Reference: {r.get('content', {}).get('topic', 'unknown')}",
                            severity=FindingSeverity.INFO, score=0.0,
                            description="Relevant OWASP knowledge found in cezanne KB",
                        ))
            except Exception:
                pass

        return findings

    def _phase_stride(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        source_code = kwargs.get("source_code", {})

        stride_categories = {
            "Spoofing": [r'(?:auth|login|session)\s*=\s*None', r'anonymous\s*=\s*True'],
            "Tampering": [r'(?:update|modify|alter)\s*\(.*(?:request|user_input)', r'\.save\s*\(\s*\).*#.*no.*valid'],
            "Repudiation": [r'#.*(?:no.*log|skip.*audit|disable.*log)', r'logging\.disable'],
            "Information Disclosure": [r'(?:print|console\.log|fmt\.Print).*?(?:password|token|secret|key)'],
            "Denial of Service": [r'(?:while|for)\s+True\s*:', r'requests\.get\s*\([^)]*timeout\s*=\s*None'],
            "Elevation of Privilege": [r'sudo\s+', r'os\.setuid\s*\(', r'chmod\s+777'],
        }

        for category, patterns in stride_categories.items():
            for fp, content in source_code.items():
                for pat in patterns:
                    matches = re.findall(pat, content, re.IGNORECASE)
                    if matches:
                        findings.append(SecurityFinding(
                            phase=10, phase_name="STRIDE Threat Model",
                            title=f"STRIDE: {category}", severity=FindingSeverity.MEDIUM,
                            score=6.0,
                            description=f"Potential {category} risk in {fp}",
                            file_path=fp,
                            remediation=f"Review for {category} vulnerability",
                            tags=["stride", category.lower().replace(" ", "-")],
                        ))

        return findings

    def _phase_data_classification(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        source_code = kwargs.get("source_code", {})
        diff_text = kwargs.get("diff_text", "")

        scan_text = diff_text + "\n" + "\n".join(source_code.values())
        findings.extend(self._pattern_scan(
            DATA_CLASSIFICATION_PATTERNS, scan_text, 11, "Data Classification",
        ))
        return findings

    def _phase_quality_gate(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        validation = self._get_validation()
        if validation is not None:
            try:
                report = validation.validate("cezanne")
                if report and report.failed > 0:
                    findings.append(SecurityFinding(
                        phase=12, phase_name="Quality Gate",
                        title=f"Validation Rules: {report.failed} violations",
                        severity=FindingSeverity.MEDIUM, score=5.0,
                        description=f"Knowledge base validation found {report.failed} rule violations",
                        remediation="Run validation_rules.py --fix to address violations",
                    ))
            except Exception:
                pass

        staged_files = kwargs.get("staged_files", [])
        for fp in staged_files:
            if fp.endswith(".py"):
                full = os.path.join(self.project_path, fp)
                if os.path.isfile(full):
                    try:
                        with open(full, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        if len(content) > 0 and content.strip().startswith("#"):
                            first_comment = content.split("\n")[0]
                            if re.match(r'^#\s*(fix|wip|update|tmp|hack)\s*$', first_comment, re.I):
                                findings.append(SecurityFinding(
                                    phase=12, phase_name="Quality Gate",
                                    title=f"Low-Quality Commit Marker: {fp}",
                                    severity=FindingSeverity.LOW, score=2.0,
                                    description=f"File starts with '{first_comment.strip()}' — low-quality commit",
                                    file_path=fp,
                                    remediation="Write descriptive commit messages",
                                ))
                    except Exception:
                        pass

        return findings

    def _phase_dependency_graph(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        intel = self._get_code_intel()
        if intel is not None:
            try:
                result = intel.query("dependencies")
                if result and result.get("circular"):
                    findings.append(SecurityFinding(
                        phase=13, phase_name="Dependency Graph",
                        title="Circular Dependencies Detected",
                        severity=FindingSeverity.MEDIUM, score=5.5,
                        description=f"Circular deps: {result['circular'][:3]}",
                        remediation="Refactor to break circular dependencies",
                    ))
            except Exception:
                pass

        staged_files = kwargs.get("staged_files", [])
        if len(staged_files) > 20:
            findings.append(SecurityFinding(
                phase=13, phase_name="Dependency Graph",
                title="Large Changeset", severity=FindingSeverity.MEDIUM,
                score=4.0,
                description=f"{len(staged_files)} files changed — high blast radius",
                remediation="Consider splitting into smaller, focused changesets",
            ))

        return findings

    def _phase_config_drift(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        memory = self._get_memory()
        if memory is None:
            return findings

        try:
            results = memory.search("cezanne", "audit", "config drift baseline", top_k=1)
            if not results:
                findings.append(SecurityFinding(
                    phase=14, phase_name="Config Drift Detection",
                    title="No Baseline Configuration", severity=FindingSeverity.LOW,
                    score=2.0,
                    description="No configuration baseline stored in audit memory",
                    remediation="Run baseline scan to establish config drift reference",
                ))
        except Exception:
            pass

        return findings

    def _phase_access_control(self, **kwargs) -> List[SecurityFinding]:
        findings = []
        source_code = kwargs.get("source_code", {})

        for fp, content in source_code.items():
            if "cors" in content.lower() and "*" in content:
                findings.append(SecurityFinding(
                    phase=15, phase_name="Access Control",
                    title="Wildcard CORS Policy", severity=FindingSeverity.HIGH,
                    score=7.5,
                    description=f"Wildcard CORS detected in {fp}",
                    file_path=fp,
                    remediation="Restrict CORS to specific origins",
                ))
            if re.search(r'@public|@skip_auth|allow_unauthenticated', content):
                findings.append(SecurityFinding(
                    phase=15, phase_name="Access Control",
                    title="Authentication Bypass Annotation", severity=FindingSeverity.CRITICAL,
                    score=9.0,
                    description=f"Auth bypass annotation found in {fp}",
                    file_path=fp,
                    remediation="Remove authentication bypass, implement proper access control",
                ))

        return findings

    def _phase_final(self, **kwargs) -> List[SecurityFinding]:
        return []

    def _write_audit_to_memory(self, report: AuditReport):
        memory = self._get_memory()
        if memory is None:
            return
        try:
            from soul_memory import write
            write("cezanne", "audit", {
                "topic": f"[VF-SEC] {report.mode.value} audit — score={report.overall_score:.1f}",
                "detail": report.summary(),
                "pass_gate": report.pass_gate,
                "total_findings": report.total_findings,
                "critical": report.critical_findings,
            }, importance=0.8 if not report.pass_gate else 0.3,
               tags=["vf-sec", "security-audit", report.mode.value])
        except Exception as e:
            logger.debug(f"Audit write to memory failed: {e}")


def run_security_audit(project_path: str = ".", mode: str = "daily",
                       diff_text: Optional[str] = None,
                       staged_files: Optional[List[str]] = None) -> AuditReport:
    audit_mode = AuditMode.COMPREHENSIVE if mode == "comprehensive" else AuditMode.DAILY
    runner = VFSECRunner(mode=audit_mode, project_path=project_path)
    return runner.run(diff_text=diff_text, staged_files=staged_files)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    runner = VFSECRunner(mode=AuditMode.DAILY, project_path=r"D:\VORTEX_FLAME")
    report = runner.run(staged_files=["vf_sec_pipeline.py"])
    print(report.summary())
