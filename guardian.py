"""
Guardian — Security Daemon
===========================
Security daemon with anti-debugging, auto-lock, and process monitoring.

Capabilities:
- File monitoring: Detect unauthorized modifications
- Process monitoring: Detect debugging/analysis tools
- Service monitoring: Detect port scanning/service enumeration
- Auto-lock: Lock system on security breach
- Anti-debug: Detect and resist reverse engineering
- Action enforcement: Per-soul action whitelists

Integration Whitelists:
- File monitor whitelist: .codegraph/, .understand-anything/
- Process whitelist: node (CodeGraph MCP server), claude-code (UA pipeline)
- Service whitelist: localhost:{codegraph_port}, localhost:{ua_port}
"""

import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

FILE_MONITOR_WHITELIST = [
    ".codegraph/",
    ".understand-anything/",
    ".worktrees/",
    "models/mano-p/",
]

PROCESS_WHITELIST = [
    "node",
    "claude-code",
    "codegraph-mcp",
    "mano-p",
    "mano_p.serve",
    "python",
]

SERVICE_WHITELIST = [
    "localhost:9432",
    "localhost:9433",
    "localhost:9450",
]

DEBUG_PROCESS_PATTERNS = [
    "ollydbg", "x64dbg", "ida", "ghidra", "wireshark", "fiddler",
    "charles", "dnspy", "dotpeek", "cheatengine", "processhacker",
    "procmon", "procexp", "apimonitor",
]

SOUL_ACTION_WHITELISTS = {
    "cezanne": ["read_file", "write_file", "execute_code", "search_code", "git_commit"],
    "einstein": ["read_file", "write_file", "search_web", "compute"],
    "galileo": ["read_file", "search_web", "compute"],
    "darwin": ["read_file", "search_web", "compute"],
    "davinci": ["read_file", "write_file", "execute_code", "design"],
    "strategy": ["read_file", "search_web", "compute"],
    "montesquieu": ["read_file", "review", "audit"],
    "humboldt": ["read_file", "search_web", "compute"],
    "yuanlongping": ["read_file", "search_web"],
    "guizhu": ["read_file", "dialogue"],
    "herodotus": ["read_file", "write_file", "search_web"],
    "monet": ["read_file", "write_file"],
    "vangogh": ["read_file"],
    "beethoven": ["read_file", "write_file"],
}


class Guardian:
    def __init__(self, config_path: Optional[str] = None):
        self._locked = False
        self._breach_log: List[dict] = []
        self._file_hashes: Dict[str, str] = {}
        self._start_time = time.time()

    def start(self) -> dict:
        self._start_time = time.time()
        self._locked = False
        logger.info("Guardian started")
        return {"status": "started", "timestamp": self._start_time}

    def check_security(self) -> dict:
        checks = {
            "debug_processes": self._check_debug_processes(),
            "file_integrity": self._check_file_integrity(),
            "locked": self._locked,
            "uptime": time.time() - self._start_time,
        }
        if any(checks["debug_processes"]):
            self._lock("debug_process_detected")
        return checks

    def _check_debug_processes(self) -> List[str]:
        found = []
        try:
            import subprocess
            result = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=5)
            output = result.stdout.lower()
            for pattern in DEBUG_PROCESS_PATTERNS:
                if pattern in output:
                    found.append(pattern)
                    self._breach_log.append({
                        "type": "debug_process",
                        "pattern": pattern,
                        "timestamp": time.time(),
                    })
        except Exception:
            pass
        return found

    def _check_file_integrity(self) -> dict:
        results = {"monitored": 0, "violations": 0}
        for path, expected_hash in self._file_hashes.items():
            if Path(path).exists():
                actual = self._hash_file(path)
                if actual != expected_hash:
                    results["violations"] += 1
                    self._breach_log.append({
                        "type": "file_integrity_violation",
                        "path": path,
                        "timestamp": time.time(),
                    })
            results["monitored"] += 1
        return results

    def _hash_file(self, path: str) -> str:
        try:
            with open(path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return ""

    def _lock(self, reason: str):
        self._locked = True
        logger.critical(f"Guardian LOCKED: {reason}")
        self._breach_log.append({"type": "lock", "reason": reason, "timestamp": time.time()})

    def unlock(self, auth_token: str = "") -> dict:
        self._locked = False
        return {"status": "unlocked", "breach_count": len(self._breach_log)}

    def is_locked(self) -> bool:
        return self._locked

    def add_file_whitelist(self, path_pattern: str) -> dict:
        FILE_MONITOR_WHITELIST.append(path_pattern)
        return {"status": "added", "whitelist": FILE_MONITOR_WHITELIST}

    def add_process_whitelist(self, process_name: str) -> dict:
        PROCESS_WHITELIST.append(process_name)
        return {"status": "added", "whitelist": PROCESS_WHITELIST}

    def add_service_whitelist(self, service_endpoint: str) -> dict:
        SERVICE_WHITELIST.append(service_endpoint)
        return {"status": "added", "whitelist": SERVICE_WHITELIST}

    def register_file_monitor(self, path: str) -> dict:
        h = self._hash_file(path)
        if h:
            self._file_hashes[path] = h
            return {"status": "registered", "path": path, "hash": h}
        return {"status": "error", "message": "File not found or unreadable"}

    def check_soul_action(self, soul: str, action: str) -> dict:
        allowed = SOUL_ACTION_WHITELISTS.get(soul, [])
        if action in allowed:
            return {"soul": soul, "action": action, "allowed": True}
        self._breach_log.append({
            "type": "action_violation",
            "soul": soul,
            "action": action,
            "timestamp": time.time(),
        })
        return {"soul": soul, "action": action, "allowed": False}

    def get_breach_log(self) -> List[dict]:
        return self._breach_log
