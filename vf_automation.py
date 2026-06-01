#!/usr/bin/env python3
"""
VF Automation — 跨环境事件驱动自动化引擎
============================================
构建于 vf_data_core.TaskScheduler 之上，提供完整的自动化闭环：

  1. TaskStateMachine    — 任务生命周期状态机 (PENDING→RUNNING→SUCCESS/FAILURE)
  2. WorkflowDAG         — DAG编排，支持依赖/并行/条件分支
  3. HookRegistry        — 状态钩子注册表，每阶段可绑定触发回调
  4. CrossEnvMonitor     — 跨环境监听器 (HTTP webhook / 本地进程 / 文件变更)
  5. AutomationOrchestrator — 总控引擎，串联所有组件

状态机:
  PENDING ──[start]──→ STARTED ──[run]──→ RUNNING
  RUNNING ──[success]──→ SUCCESS     (触发 on_success 钩子)
  RUNNING ──[failure]──→ FAILURE     (触发 on_failure 钩子 → 重试逻辑)
  RUNNING ──[cancel]───→ CANCELLED   (触发 on_cancel 钩子)
  FAILURE ──[retry]───→ PENDING      (重试次数 < max_retries)

跨环境触发:
  CLOUD:  HTTP POST webhook → 接收外部事件 → 触发工作流
  LOCAL:  psutil进程监控 → 检测PID状态变化 → 触发回调
  FILE:   watchdog文件监听 → .ready / .done 标记文件 → 触发工作流
  TIMER:  继承TaskScheduler → cron/interval → 周期触发

用法:
  from vf_automation import AutomationOrchestrator
  auto = AutomationOrchestrator(cfg, audit)
  auto.register_workflow("kb_pipeline", [dedup_task, cleanse_task, inspect_task])
  auto.start()  # 启动webhook服务器 + 文件监听 + 定时任务
  auto.fire_event("data.arrived", {"path": "/data/new_batch.json"})
"""

import copy
import functools
import hashlib
import http.server
import json
import logging
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from vf_data_core import (
    AuditLogger, ConfigManager, TaskScheduler, TaskTrigger,
    ResourceMonitor, ResourceSnapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False


class TaskState(Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"


VALID_TRANSITIONS = {
    TaskState.PENDING:    {TaskState.STARTED, TaskState.SKIPPED},
    TaskState.STARTED:    {TaskState.RUNNING, TaskState.CANCELLED},
    TaskState.RUNNING:    {TaskState.SUCCESS, TaskState.FAILURE, TaskState.CANCELLED},
    TaskState.FAILURE:    {TaskState.PENDING, TaskState.SKIPPED},
    TaskState.CANCELLED:  {TaskState.PENDING},
    TaskState.SKIPPED:    set(),
}


@dataclass
class TaskInstance:
    instance_id: str
    task_id: str
    state: TaskState = TaskState.PENDING
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    error_message: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    parent_instance_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowNode:
    task_id: str
    func: Callable
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    timeout_s: float = 3600.0
    retry_on_failure: bool = True
    priority: int = 0


@dataclass
class WorkflowRun:
    run_id: str
    workflow_name: str
    nodes: Dict[str, WorkflowNode]
    instances: Dict[str, TaskInstance] = field(default_factory=dict)
    status: TaskState = TaskState.PENDING
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class HookRegistry:
    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = defaultdict(list)

    def on(self, event: str, callback: Optional[Callable] = None):
        if callback is not None:
            self._hooks[event].append(callback)
            return callback

        def decorator(func):
            self._hooks[event].append(func)
            return func
        return decorator

    def off(self, event: str, callback: Callable):
        if event in self._hooks and callback in self._hooks[event]:
            self._hooks[event].remove(callback)

    def emit(self, event: str, **kwargs):
        results = []
        for cb in self._hooks.get(event, []):
            try:
                result = cb(**kwargs)
                results.append(result)
            except Exception:
                traceback.print_exc()
        return results

    def list_hooks(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self._hooks.items()}


class TaskStateMachine:
    def __init__(self, hooks: HookRegistry):
        self.hooks = hooks
        self._instances: Dict[str, TaskInstance] = {}
        self._lock = threading.RLock()

    def create_instance(self, task_id: str, payload: Optional[Dict] = None,
                        max_retries: int = 3) -> TaskInstance:
        inst = TaskInstance(
            instance_id=f"{task_id}-{uuid.uuid4().hex[:8]}",
            task_id=task_id,
            payload=payload or {},
            max_retries=max_retries,
        )
        with self._lock:
            self._instances[inst.instance_id] = inst
        self.hooks.emit("instance.created", instance=inst)
        return inst

    def transition(self, instance_id: str, new_state: TaskState,
                   error: str = "", result: Any = None) -> bool:
        with self._lock:
            inst = self._instances.get(instance_id)
            if inst is None:
                return False

            if new_state not in VALID_TRANSITIONS.get(inst.state, set()):
                logging.warning(
                    f"Invalid transition: {inst.task_id} {inst.state.value} → {new_state.value}"
                )
                return False

            old_state = inst.state
            inst.state = new_state

            if new_state == TaskState.STARTED:
                inst.started_at = datetime.now(timezone.utc).isoformat()
            elif new_state == TaskState.RUNNING:
                pass
            elif new_state == TaskState.SUCCESS:
                inst.finished_at = datetime.now(timezone.utc).isoformat()
                if inst.started_at:
                    start = datetime.fromisoformat(inst.started_at)
                    inst.duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                inst.result = result
            elif new_state == TaskState.FAILURE:
                inst.finished_at = datetime.now(timezone.utc).isoformat()
                inst.error_message = error

            self.hooks.emit(
                f"instance.{new_state.value.lower()}",
                instance=inst, old_state=old_state,
                error=error, result=result,
            )

            if new_state == TaskState.FAILURE and inst.retry_count < inst.max_retries:
                inst.retry_count += 1
                inst.state = TaskState.PENDING
                self.hooks.emit("instance.retrying", instance=inst)

            return True

    def get_instance(self, instance_id: str) -> Optional[TaskInstance]:
        return self._instances.get(instance_id)

    def list_instances(self, state: Optional[TaskState] = None) -> List[TaskInstance]:
        with self._lock:
            if state:
                return [i for i in self._instances.values() if i.state == state]
            return list(self._instances.values())

    def cancel(self, instance_id: str) -> bool:
        inst = self._instances.get(instance_id)
        if inst and inst.state in (TaskState.PENDING, TaskState.RUNNING):
            return self.transition(instance_id, TaskState.CANCELLED)
        return False


class WorkflowDAG:
    def __init__(self, name: str, hooks: HookRegistry, executor: ThreadPoolExecutor):
        self.name = name
        self.hooks = hooks
        self.executor = executor
        self._nodes: Dict[str, WorkflowNode] = {}
        self._lock = threading.RLock()

    def add_node(self, node: WorkflowNode):
        with self._lock:
            self._nodes[node.task_id] = node

    def remove_node(self, task_id: str):
        with self._lock:
            self._nodes.pop(task_id, None)
            for node in self._nodes.values():
                if task_id in node.depends_on:
                    node.depends_on.remove(task_id)

    def validate(self) -> Tuple[bool, Optional[str]]:
        with self._lock:
            task_ids = set(self._nodes.keys())
            for tid, node in self._nodes.items():
                for dep in node.depends_on:
                    if dep not in task_ids:
                        return False, f"Node '{tid}' depends on unknown '{dep}'"
            if self._has_cycle():
                return False, "DAG contains a cycle"
            return True, None

    def _has_cycle(self) -> bool:
        adj = {tid: node.depends_on for tid, node in self._nodes.items()}
        visited = set()
        rec_stack = set()

        def dfs(v):
            visited.add(v)
            rec_stack.add(v)
            for neighbor in adj.get(v, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(v)
            return False

        for v in adj:
            if v not in visited:
                if dfs(v):
                    return True
        return False

    def _topological_order(self) -> List[str]:
        adj = {tid: list(node.depends_on) for tid, node in self._nodes.items()}
        in_degree = {tid: 0 for tid in self._nodes}
        for deps in adj.values():
            for d in deps:
                in_degree[d] = in_degree.get(d, 0)

        adj_rev: Dict[str, List[str]] = defaultdict(list)
        for tid, deps in adj.items():
            for d in deps:
                adj_rev[d].append(tid)

        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        order = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for neighbor in adj_rev.get(v, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._nodes):
            remaining = set(self._nodes.keys()) - set(order)
            order.extend(remaining)
        return order

    def execute(self, state_machine: TaskStateMachine,
                initial_context: Optional[Dict] = None) -> WorkflowRun:
        ok, err = self.validate()
        if not ok:
            raise ValueError(f"Invalid DAG: {err}")

        run = WorkflowRun(
            run_id=f"{self.name}-{uuid.uuid4().hex[:8]}",
            workflow_name=self.name,
            nodes=dict(self._nodes),
            status=TaskState.STARTED,
            started_at=datetime.now(timezone.utc).isoformat(),
            context=initial_context or {},
        )
        self.hooks.emit("workflow.started", run=run)

        order = self._topological_order()

        for task_id in order:
            node = self._nodes[task_id]
            if node.condition and not node.condition(run.context):
                inst = state_machine.create_instance(task_id, max_retries=0)
                run.instances[task_id] = inst
                state_machine.transition(inst.instance_id, TaskState.SKIPPED)
                continue

            inst = state_machine.create_instance(
                task_id,
                payload={"context": copy.deepcopy(run.context)},
                max_retries=3 if node.retry_on_failure else 0,
            )
            run.instances[task_id] = inst
            state_machine.transition(inst.instance_id, TaskState.STARTED)
            state_machine.transition(inst.instance_id, TaskState.RUNNING)

            try:
                future = self.executor.submit(node.func, inst.payload)
                result = future.result(timeout=node.timeout_s)
                state_machine.transition(inst.instance_id, TaskState.SUCCESS, result=result)
                if isinstance(result, dict):
                    run.context[task_id] = result
            except Exception as e:
                state_machine.transition(inst.instance_id, TaskState.FAILURE, error=str(e))
                run.status = TaskState.FAILURE
                self.hooks.emit("workflow.failed", run=run, error=str(e))
                return run

        run.status = TaskState.SUCCESS
        run.finished_at = datetime.now(timezone.utc).isoformat()
        self.hooks.emit("workflow.completed", run=run)
        return run


class WebhookHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, orchestrator=None, **kwargs):
        self.orchestrator = orchestrator
        super().__init__(*args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": "invalid json"}).encode())
            return

        event_name = payload.get("event", self.path.lstrip("/"))
        self.orchestrator.fire_event(event_name, payload)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "received", "event": event_name}).encode())

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        payload = {"event": parsed.path.lstrip("/")}
        for k, v in params.items():
            payload[k] = v[0] if len(v) == 1 else v

        self.orchestrator.fire_event(payload["event"], payload)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "received"}).encode())

    def log_message(self, format, *args):
        pass


class CrossEnvMonitor:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.config = orchestrator.config
        self.audit = orchestrator.audit
        self._webhook_server: Optional[http.server.HTTPServer] = None
        self._file_observer: Optional[Any] = None
        self._process_watchers: Dict[int, threading.Thread] = {}
        self._running = False

    def start_webhook(self, host: str = "0.0.0.0", port: int = 9845):
        handler = functools.partial(WebhookHandler, orchestrator=self.orchestrator)
        self._webhook_server = http.server.HTTPServer((host, port), handler)
        t = threading.Thread(target=self._webhook_server.serve_forever,
                            daemon=True, name="vf-webhook")
        t.start()
        self.audit.log("automation.webhook", "start", "OK",
                       details={"host": host, "port": port})

    def start_file_watch(self, watch_paths: List[str], patterns: Optional[List[str]] = None):
        if not HAS_WATCHDOG:
            logging.warning("watchdog not installed, file watching disabled")
            return

        patterns = patterns or ["*.json", "*.csv", "*.ready", "*.done", "*.trigger"]

        class VFEventHandler(FileSystemEventHandler):
            def __init__(self, monitor):
                super().__init__()
                self.monitor = monitor

            def on_created(self, event):
                if event.is_directory:
                    return
                fpath = Path(event.src_path)
                if any(fpath.match(p) for p in patterns):
                    self.monitor._handle_file_event("created", fpath)

            def on_modified(self, event):
                if event.is_directory:
                    return
                fpath = Path(event.src_path)
                if any(fpath.match(p) for p in patterns):
                    self.monitor._handle_file_event("modified", fpath)

        handler = VFEventHandler(self)
        self._file_observer = Observer()
        for wp in watch_paths:
            wp_path = Path(wp)
            if wp_path.exists():
                self._file_observer.schedule(handler, str(wp_path), recursive=True)

        self._file_observer.start()
        self.audit.log("automation.file_watch", "start", "OK",
                       details={"paths": watch_paths, "patterns": patterns})

    def _handle_file_event(self, event_type: str, fpath: Path):
        suffix = fpath.suffix.lstrip(".")
        event_name = f"file.{event_type}.{suffix}"
        if fpath.suffix in (".ready", ".trigger"):
            event_name = f"pipeline.trigger.{fpath.stem}"

        self.orchestrator.fire_event(event_name, {
            "type": event_type,
            "path": str(fpath),
            "name": fpath.name,
            "size": fpath.stat().st_size if fpath.exists() else 0,
            "mtime": fpath.stat().st_mtime if fpath.exists() else 0,
        })
        self.audit.log("automation.file_event", event_name, "OK",
                       details={"path": str(fpath)})

    def watch_process(self, pid: int, on_exit: Optional[Callable] = None,
                      poll_interval: float = 5.0):
        if not HAS_PSUTIL:
            return

        def _watcher():
            while self._running:
                try:
                    proc = psutil.Process(pid)
                    if not proc.is_running():
                        self.orchestrator.fire_event(f"process.exit.{pid}", {
                            "pid": pid,
                            "exit_code": proc.wait() if proc else -1,
                        })
                        if on_exit:
                            on_exit(pid)
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self.orchestrator.fire_event(f"process.exit.{pid}", {"pid": pid})
                    if on_exit:
                        on_exit(pid)
                    break
                time.sleep(poll_interval)

        t = threading.Thread(target=_watcher, daemon=True, name=f"proc-watch-{pid}")
        t.start()
        self._process_watchers[pid] = t

    def stop(self):
        self._running = False
        if self._webhook_server:
            self._webhook_server.shutdown()
        if self._file_observer and HAS_WATCHDOG:
            self._file_observer.stop()
            self._file_observer.join(timeout=5)


class AutomationOrchestrator:
    def __init__(self, config: ConfigManager, audit: AuditLogger):
        self.config = config
        self.audit = audit
        self.hooks = HookRegistry()
        self.state_machine = TaskStateMachine(self.hooks)
        self.executor = ThreadPoolExecutor(
            max_workers=config.get("system", "max_workers", default=4),
            thread_name_prefix="vf-auto",
        )
        self.scheduler = TaskScheduler(config, audit)
        self.monitor = CrossEnvMonitor(self)
        self._workflows: Dict[str, WorkflowDAG] = {}
        self._event_queue: queue.Queue = queue.Queue()
        self._running = False
        self._event_loop_thread: Optional[threading.Thread] = None

        self._setup_default_hooks()

    def _setup_default_hooks(self):
        @self.hooks.on("instance.success")
        def _on_success(instance: TaskInstance, **kwargs):
            self.audit.log(
                "automation", f"task:{instance.task_id}",
                "SUCCESS",
                duration_ms=instance.duration_ms,
            )

        @self.hooks.on("instance.failure")
        def _on_failure(instance: TaskInstance, error: str, **kwargs):
            self.audit.log(
                "automation", f"task:{instance.task_id}",
                f"FAILURE:{error[:100]}",
                duration_ms=instance.duration_ms,
            )

        @self.hooks.on("instance.retrying")
        def _on_retry(instance: TaskInstance, **kwargs):
            self.audit.log(
                "automation", f"task:{instance.task_id}",
                f"RETRY:{instance.retry_count}/{instance.max_retries}",
            )

    def register_workflow(self, name: str, nodes: List[WorkflowNode]) -> WorkflowDAG:
        dag = WorkflowDAG(name, self.hooks, self.executor)
        for node in nodes:
            dag.add_node(node)
        ok, err = dag.validate()
        if not ok:
            raise ValueError(f"Workflow '{name}' validation failed: {err}")
        self._workflows[name] = dag
        self.audit.log("automation", f"workflow:register:{name}", "OK",
                       details={"nodes": len(nodes)})
        return dag

    def trigger_workflow(self, name: str, context: Optional[Dict] = None) -> Optional[WorkflowRun]:
        dag = self._workflows.get(name)
        if not dag:
            logging.error(f"Unknown workflow: {name}")
            return None
        self.audit.log("automation", f"workflow:trigger:{name}", "START")
        return dag.execute(self.state_machine, context)

    def fire_event(self, event_name: str, payload: Optional[Dict] = None):
        self._event_queue.put({
            "name": event_name,
            "payload": payload or {},
            "time": time.time(),
        })
        self.scheduler.fire_event(event_name, payload)
        self.hooks.emit(event_name, payload=payload or {})

    def on_event(self, event_name: str):
        """装饰器：注册事件处理器"""
        def decorator(func):
            self.hooks.on(event_name, func)
            return func
        return decorator

    def start(self, webhook_port: Optional[int] = None):
        self._running = True

        auto_cfg = self.config.get("automation", default={})
        wh_port = webhook_port or auto_cfg.get("webhook_port", 9845)
        wh_host = auto_cfg.get("webhook_host", "0.0.0.0")
        self.monitor.start_webhook(host=wh_host, port=wh_port)

        watch_paths = auto_cfg.get("watch_paths", [".vf_data_temp", "kb_harness"])
        if watch_paths:
            self.monitor.start_file_watch(watch_paths)

        self.scheduler.start()

        self._event_loop_thread = threading.Thread(
            target=self._event_loop, daemon=True, name="vf-event-loop"
        )
        self._event_loop_thread.start()

        self.audit.log("automation", "start", "OK",
                       details={"webhook_port": wh_port, "watch_paths": watch_paths})

    def stop(self):
        self._running = False
        self.monitor.stop()
        self.scheduler.stop()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.audit.log("automation", "stop", "OK")

    def _event_loop(self):
        while self._running:
            try:
                event = self._event_queue.get(timeout=2.0)
                event_name = event["name"]
                payload = event["payload"]

                for wf_name in self._workflows:
                    triggers = self.config.get("automation", "event_workflow_map", default={})
                    if event_name in triggers.get(wf_name, []):
                        self.trigger_workflow(wf_name, payload)

            except queue.Empty:
                continue
            except Exception:
                traceback.print_exc()

    def status(self) -> Dict[str, Any]:
        return {
            "automation": {
                "running": self._running,
                "webhook": self.monitor._webhook_server is not None,
                "file_watching": self.monitor._file_observer is not None,
            },
            "workflows": {
                name: {"nodes": list(dag._nodes.keys())}
                for name, dag in self._workflows.items()
            },
            "hooks": self.hooks.list_hooks(),
            "instances": {
                state.value: len(self.state_machine.list_instances(state))
                for state in TaskState
            },
        }

    def register_scheduled_task(self, task_id: str, func: Callable, interval: str):
        self.scheduler.register(task_id, func, TaskTrigger.TIME, schedule=interval)

    def register_event_task(self, task_id: str, func: Callable, events: List[str]):
        self.scheduler.register(task_id, func, TaskTrigger.EVENT, events=events)
