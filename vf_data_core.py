#!/usr/bin/env python3
"""
VF Data Core — 企业级数据处理系统核心引擎
=============================================
提供所有数据处理模块的公共基础设施：

  1. ConfigManager    — YAML配置加载/验证/热更新
  2. AuditLogger      — 结构化JSONL审计日志(每次操作全追踪)
  3. TaskScheduler    — 时间/事件/手动三模式任务调度
  4. ResourceMonitor  — CPU/内存/磁盘资源监控与限流
  5. BaseProcessor    — 所有处理模块的抽象基类(process/validate/report)

架构原则：
  - 所有状态变更必须可审计 (who/when/what/before/after)
  - 关键路径有资源限流 (CPU≤80%, MEM≤70%)
  - 模块松耦合，仅依赖本文件定义的接口
  - 遵循ValidationRule/Violation/AuditReport模式(vf_validation_rules.py兼容)

用法:
  from vf_data_core import ConfigManager, AuditLogger, BaseProcessor, TaskScheduler
  cfg = ConfigManager("vf_data_config.yaml")
  logger = AuditLogger(cfg.audit_dir)
  processor = MyProcessor(cfg)
"""

import hashlib
import json
import logging
import os
import platform
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


class Severity(Enum):
    DEBUG = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()
    CRITICAL = auto()


class TaskTrigger(Enum):
    TIME = "time"
    EVENT = "event"
    MANUAL = "manual"


@dataclass
class AuditRecord:
    event_id: str
    timestamp: str
    module: str
    operation: str
    status: str
    duration_ms: float
    record_count: int
    details: Dict[str, Any] = field(default_factory=dict)
    user: str = "system"
    trace_id: str = ""


@dataclass
class ResourceSnapshot:
    cpu_percent: float
    mem_percent: float
    mem_used_gb: float
    mem_total_gb: float
    disk_free_gb: float
    timestamp: str = ""


@dataclass
class ProcessingStats:
    total_input: int = 0
    total_output: int = 0
    filtered: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    throughput_per_sec: float = 0.0


class AuditLogger:
    def __init__(self, audit_dir: Path, rotation_mb: int = 100):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.rotation_mb = rotation_mb
        self._lock = threading.Lock()
        self._current_file = self._get_log_path()
        self._trace_context = threading.local()

    def _get_log_path(self) -> Path:
        date_str = datetime.now().strftime("%Y%m%d")
        return self.audit_dir / f"audit_{date_str}.jsonl"

    def _maybe_rotate(self):
        if self._current_file.exists() and self._current_file.stat().st_size > self.rotation_mb * 1024 * 1024:
            ts = datetime.now().strftime("%H%M%S")
            rotated = self._current_file.with_suffix(f".{ts}.jsonl")
            self._current_file.rename(rotated)
            self._current_file = self._get_log_path()

    def set_trace(self, trace_id: str):
        self._trace_context.trace_id = trace_id

    def get_trace(self) -> str:
        return getattr(self._trace_context, "trace_id", str(uuid.uuid4())[:8])

    def log(self, module: str, operation: str, status: str = "OK",
            duration_ms: float = 0.0, record_count: int = 0,
            details: Optional[Dict[str, Any]] = None):
        trace_id = self.get_trace()
        event_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"

        record = AuditRecord(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            module=module,
            operation=operation,
            status=status,
            duration_ms=round(duration_ms, 2),
            record_count=record_count,
            details=details or {},
            trace_id=trace_id,
        )

        with self._lock:
            self._maybe_rotate()
            with open(self._current_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def query(self, module: Optional[str] = None, date_str: Optional[str] = None,
              limit: int = 100) -> List[Dict[str, Any]]:
        results = []
        if date_str:
            target = self.audit_dir / f"audit_{date_str}.jsonl"
        else:
            target = self._current_file

        if not target.exists():
            return results

        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if module is None or rec.get("module") == module:
                        results.append(rec)
                        if len(results) >= limit:
                            break
                except json.JSONDecodeError:
                    continue
        return results


class ResourceMonitor:
    def __init__(self, cpu_limit: float = 80.0, mem_limit: float = 70.0):
        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            pass

    def snapshot(self) -> ResourceSnapshot:
        if self._psutil:
            cpu = self._psutil.cpu_percent(interval=0.1)
            mem = self._psutil.virtual_memory()
            return ResourceSnapshot(
                cpu_percent=cpu,
                mem_percent=mem.percent,
                mem_used_gb=mem.used / (1024**3),
                mem_total_gb=mem.total / (1024**3),
                disk_free_gb=self._psutil.disk_usage(str(PROJECT_ROOT)).free / (1024**3),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        return ResourceSnapshot(
            cpu_percent=0.0,
            mem_percent=0.0,
            mem_used_gb=0.0,
            mem_total_gb=0.0,
            disk_free_gb=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def can_process(self, batch_size: int = 1000) -> Tuple[bool, Optional[str]]:
        snap = self.snapshot()
        if snap.cpu_percent > self.cpu_limit:
            return False, f"CPU {snap.cpu_percent:.1f}% > limit {self.cpu_limit}%"
        if snap.mem_percent > self.mem_limit:
            return False, f"MEM {snap.mem_percent:.1f}% > limit {self.mem_limit}%"
        return True, None

    def throttle_if_needed(self):
        snap = self.snapshot()
        if snap.cpu_percent > self.cpu_limit * 0.9:
            time.sleep(0.5)
        if snap.mem_percent > self.mem_limit * 0.9:
            time.sleep(0.3)


class ConfigManager:
    DEFAULT_CONFIG = {
        "system": {
            "name": "VF_DATA_PIPELINE",
            "version": "1.0.0",
            "max_batch_size": 100000,
            "max_workers": max(1, (os.cpu_count() or 4) - 1),
            "temp_dir": str(PROJECT_ROOT / ".vf_data_temp"),
            "audit_dir": str(PROJECT_ROOT / ".vf_data_audit"),
            "log_level": "INFO",
        },
        "resources": {
            "cpu_limit_pct": 80.0,
            "mem_limit_pct": 70.0,
            "disk_min_free_gb": 5.0,
        },
        "dedup": {
            "enabled": True,
            "methods": ["exact", "simhash", "fuzzy"],
            "simhash_fp_len": 64,
            "simhash_ngram": 3,
            "simhash_threshold": 3,
            "fuzzy_threshold": 0.85,
            "semantic_enabled": False,
            "semantic_threshold": 0.92,
            "conflict_resolution": "keep_first",
            "fields_weight": {},
        },
        "cleanse": {
            "enabled": True,
            "normalize_dates": True,
            "normalize_numbers": True,
            "normalize_text": True,
            "outlier_method": "iqr",
            "outlier_threshold": 3.0,
            "missing_strategy": "mode",
            "filter_patterns": [],
            "filter_keywords": [],
        },
        "inspect": {
            "enabled": True,
            "completeness_check": True,
            "accuracy_check": True,
            "consistency_check": True,
            "compliance_check": True,
            "required_fields": [],
            "business_rules": [],
            "compliance_rules": [],
            "report_format": "json",
        },
        "distill": {
            "enabled": True,
            "entity_extraction": True,
            "topic_modeling": True,
            "summarization": True,
            "skill_extraction": False,
            "target_reduction_ratio": 0.5,
            "min_core_value_retention": 0.90,
            "llm_enabled": False,
            "llm_model": "",
        },
        "scheduler": {
            "time_triggers": [],
            "event_triggers": [],
            "retry_max": 3,
            "retry_delay_s": 60,
            "timeout_s": 3600,
        },
        "notifications": {
            "enabled": False,
            "webhook_url": "",
            "email_to": "",
            "alert_on_error": True,
            "alert_on_completion": False,
        },
    }

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = Path(config_path) if config_path else (PROJECT_ROOT / "vf_data_config.yaml")
        self._data: Dict[str, Any] = dict(self.DEFAULT_CONFIG)
        self._last_modified: float = 0.0
        self._lock = threading.RLock()
        self.load()

    def load(self):
        with self._lock:
            if self._config_path.exists():
                try:
                    with open(self._config_path, "r", encoding="utf-8") as f:
                        user_config = yaml.safe_load(f) or {}
                    self._deep_merge(self._data, user_config)
                    self._last_modified = self._config_path.stat().st_mtime
                except Exception as e:
                    logging.warning(f"Config load failed: {e}, using defaults")

    def _deep_merge(self, base: dict, override: dict):
        for key, val in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                self._deep_merge(base[key], val)
            else:
                base[key] = val

    def reload_if_changed(self):
        if self._config_path.exists():
            mtime = self._config_path.stat().st_mtime
            if mtime > self._last_modified:
                self.load()
                return True
        return False

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def get(self, *path: str, default: Any = None) -> Any:
        node = self._data
        for key in path:
            if isinstance(node, dict):
                node = node.get(key)
            else:
                return default
        return node if node is not None else default

    def set(self, *path: str, value: Any):
        node = self._data
        for key in path[:-1]:
            if key not in node:
                node[key] = {}
            node = node[key]
        node[path[-1]] = value

    def save(self):
        with self._lock:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._data, f, default_flow_style=False, allow_unicode=True)

    @property
    def audit_dir(self) -> Path:
        return Path(self.get("system", "audit_dir", default=".vf_data_audit"))

    @property
    def max_batch_size(self) -> int:
        return self.get("system", "max_batch_size", default=100000)

    @property
    def max_workers(self) -> int:
        return self.get("system", "max_workers", default=4)

    def to_yaml(self) -> str:
        return yaml.dump(self._data, default_flow_style=False, allow_unicode=True)


class TaskScheduler:
    def __init__(self, config: ConfigManager, audit: AuditLogger):
        self.config = config
        self.audit = audit
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._event_queue: queue.Queue = queue.Queue()
        self._running = False
        self._timer_thread: Optional[threading.Thread] = None
        self._event_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def register(self, task_id: str, func: Callable, trigger: TaskTrigger,
                 schedule: Optional[str] = None, events: Optional[List[str]] = None,
                 priority: int = 0):
        with self._lock:
            self._tasks[task_id] = {
                "id": task_id,
                "func": func,
                "trigger": trigger,
                "schedule": schedule,
                "events": events or [],
                "priority": priority,
                "last_run": None,
                "run_count": 0,
                "error_count": 0,
            }

    def fire_event(self, event_name: str, payload: Optional[Dict] = None):
        self._event_queue.put({"name": event_name, "payload": payload or {}, "time": time.time()})

    def start(self):
        self._running = True
        self._event_thread = threading.Thread(target=self._event_loop, daemon=True, name="vf-event-loop")
        self._event_thread.start()
        if any(t["trigger"] == TaskTrigger.TIME for t in self._tasks.values()):
            self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True, name="vf-timer-loop")
            self._timer_thread.start()
        self.audit.log("scheduler", "start", "OK", details={"tasks": len(self._tasks)})

    def stop(self):
        self._running = False
        self.audit.log("scheduler", "stop", "OK")

    def _event_loop(self):
        while self._running:
            try:
                event = self._event_queue.get(timeout=1.0)
                for tid, task in self._tasks.items():
                    if task["trigger"] == TaskTrigger.EVENT and event["name"] in task["events"]:
                        self._execute(tid, task, event)
            except queue.Empty:
                continue
            except Exception:
                traceback.print_exc()

    def _timer_loop(self):
        while self._running:
            now = datetime.now()
            for tid, task in self._tasks.items():
                if task["trigger"] != TaskTrigger.TIME:
                    continue
                if not task["schedule"]:
                    continue
                try:
                    last = task["last_run"]
                    interval = self._parse_interval(task["schedule"])
                    if last is None or (now - last).total_seconds() >= interval:
                        self._execute(tid, task, None)
                except Exception:
                    traceback.print_exc()
            time.sleep(5)

    def _parse_interval(self, schedule: str) -> float:
        schedule = schedule.strip().lower()
        if schedule.endswith("s"):
            return float(schedule[:-1])
        elif schedule.endswith("m"):
            return float(schedule[:-1]) * 60
        elif schedule.endswith("h"):
            return float(schedule[:-1]) * 3600
        else:
            return float(schedule)

    def _execute(self, tid: str, task: Dict, event: Optional[Dict]):
        t0 = time.time()
        try:
            task["func"](event)
            task["last_run"] = datetime.now()
            task["run_count"] += 1
            self.audit.log("scheduler", f"task:{tid}", "OK",
                           duration_ms=(time.time() - t0) * 1000)
        except Exception as e:
            task["error_count"] += 1
            self.audit.log("scheduler", f"task:{tid}", f"ERROR:{e}",
                           duration_ms=(time.time() - t0) * 1000)

    def run_manual(self, task_id: str, payload: Optional[Dict] = None):
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")
        self._execute(task_id, task, payload)


class BaseProcessor(ABC):
    def __init__(self, config: ConfigManager, audit: AuditLogger, module_name: str):
        self.config = config
        self.audit = audit
        self.module_name = module_name
        self.stats = ProcessingStats()
        self._resource = ResourceMonitor(
            cpu_limit=config.get("resources", "cpu_limit_pct", default=80.0),
            mem_limit=config.get("resources", "mem_limit_pct", default=70.0),
        )

    @abstractmethod
    def process(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], ProcessingStats]:
        pass

    @abstractmethod
    def validate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

    def report(self) -> Dict[str, Any]:
        return {
            "module": self.module_name,
            "stats": asdict(self.stats),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _resource_check(self, record_count: int = 0) -> bool:
        if record_count > 0 and record_count < 1000:
            return True
        ok, msg = self._resource.can_process()
        if not ok:
            logging.warning(f"[{self.module_name}] Resource limit: {msg}")
            self._resource.throttle_if_needed()
            return False
        return True

    def _audit_operation(self, operation: str, status: str, duration_ms: float,
                         record_count: int, details: Optional[Dict] = None):
        self.audit.log(
            module=self.module_name,
            operation=operation,
            status=status,
            duration_ms=duration_ms,
            record_count=record_count,
            details=details,
        )


def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_content_hash(content: str, algorithm: str = "md5") -> str:
    return hashlib.new(algorithm, content.encode("utf-8")).hexdigest()


def is_valid_json(data: Any) -> bool:
    if isinstance(data, (dict, list)):
        return True
    if isinstance(data, str):
        try:
            json.loads(data)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    return False


def normalize_field_names(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for rec in records:
        normalized = {}
        for k, v in rec.items():
            nk = k.strip().lower().replace(" ", "_").replace("-", "_")
            normalized[nk] = v
        result.append(normalized)
    return result
