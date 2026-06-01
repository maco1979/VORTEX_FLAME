#!/usr/bin/env python3
"""
VF Auto-Fix Daemon — 文件保存自动触发修复守护进程
===================================================
桥接 VS Code pyright 诊断 ↔ vf_auto_fix.py 自动修复。

工作原理:
  while running:
    1. 监控项目中 .py 文件的修改事件
    2. 文件变更后等待 2秒（确保 VS Code 完成保存）
    3. 对变更文件运行 pyright 诊断
    4. 如果发现错误 → 调用 AutoFixLoop 自动修复
    5. 修复后文件内容变更 → VS Code 自动刷新诊断面板

这不是 VS Code 插件，安全、独立、可随时启停：
  python vf_auto_fix_daemon.py            # 前台运行，Ctrl+C 停止
  python vf_auto_fix_daemon.py --daemon   # 后台运行
  python vf_auto_fix_daemon.py --once     # 单次扫描（检查是否有待修复错误）

技术限制与 VS Code 集成说明:
  - VS Code Pylance 的分析结果无法直接从外部获取
  - 我们用 pyright CLI 独立运行诊断（与 VS Code Pylance 同源）
  - 自动修复后 VS Code 需要重新分析文件，会有1-3秒延迟
  - 如果 VS Code 中仍看到红色波浪线，手动点击文件或 Ctrl+S 触发刷新
"""

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from vf_auto_fix import AutoFixLoop, DiagnosticError, ErrorCategory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DAEMON] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vf-daemon")


class FileWatcher:
    def __init__(self, project_root: Path, poll_interval: float = 1.0):
        self.root = project_root
        self.poll_interval = poll_interval
        self._mtimes: dict[str, float] = {}
        self._excluded = {"__pycache__", ".git", ".vf_data_audit", ".vf_memory",
                          "kb_mcp", "kb_skill", "kb_workflow", "tools"}
        self._build_initial_snapshot()

    def _should_watch(self, path: Path) -> bool:
        parts = set(path.parts)
        if parts & self._excluded:
            return False
        if any(p.startswith("_test_") for p in path.parts):
            return False
        if any(p.startswith("_smoke_") or p.startswith("_precache_") for p in path.parts):
            return False
        return path.suffix == ".py" and path.exists()

    def _build_initial_snapshot(self):
        for py_file in self.root.rglob("*.py"):
            if self._should_watch(py_file):
                try:
                    self._mtimes[str(py_file)] = py_file.stat().st_mtime
                except OSError:
                    pass
        logger.info(f"Watching {len(self._mtimes)} .py files")

    def poll(self) -> Set[str]:
        changed = set()
        for py_file in self.root.rglob("*.py"):
            if not self._should_watch(py_file):
                continue
            key = str(py_file)
            try:
                new_mtime = py_file.stat().st_mtime
            except OSError:
                continue

            old_mtime = self._mtimes.get(key, 0)
            if new_mtime > old_mtime + 1.0:
                changed.add(key)
            self._mtimes[key] = new_mtime
        return changed


class AutoFixDaemon:
    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or PROJECT_ROOT
        self.watcher = FileWatcher(self.root)
        self.running = True
        self.total_fixes = 0
        self.total_scans = 0

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _resolve_file_path(self, abs_path: str) -> Optional[str]:
        try:
            rel = Path(abs_path).relative_to(self.root)
            return str(rel)
        except ValueError:
            return None

    def _process_changed_files(self, files: Set[str]) -> int:
        rel_files = []
        for f in files:
            rel = self._resolve_file_path(f)
            if rel:
                rel_files.append(rel)

        if not rel_files:
            return 0

        logger.info(f"Changed: {', '.join(rel_files[:5])}{'...' if len(rel_files) > 5 else ''}")

        loop = AutoFixLoop(project_root=self.root, max_iterations=3)
        result = loop.run_full(files=rel_files)

        if result.fixes_applied > 0:
            logger.info(f"Fixed {result.fixes_applied} issues: {result.errors_before}→{result.errors_after}")
            for detail in result.fixes_detail:
                logger.info(f"  {detail}")
        else:
            if result.errors_before > 0:
                remaining = result.unfixable_errors[:3]
                for e in remaining:
                    logger.info(f"  ⚠ unfixable [{e.rule}]: {e.message[:100]} ({e.file_path}:{e.line})")

        return result.fixes_applied

    def run_forever(self, quiet_mode: bool = True):
        logger.info("=" * 50)
        logger.info("VF AUTO-FIX DAEMON STARTED")
        logger.info(f"Project: {self.root}")
        logger.info("Mode: file watcher + auto-fix on change")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 50)

        while self.running:
            try:
                changed = self.watcher.poll()
                if changed:
                    time.sleep(2.0)  # wait for VS Code to finish writing
                    fixes = self._process_changed_files(changed)
                    self.total_fixes += fixes
                    self.total_scans += 1
                time.sleep(self.watcher.poll_interval)
            except Exception as e:
                logger.error(f"Poll error: {e}")
                time.sleep(5)

        logger.info(f"Stopped — {self.total_scans} scans, {self.total_fixes} fixes total")

    def run_once(self) -> int:
        loop = AutoFixLoop(project_root=self.root, max_iterations=5)
        result = loop.run_full()
        print(f"Once scan: {result.errors_before}→{result.errors_after} ({result.fixes_applied} fixes)")
        if result.fixes_applied == 0 and result.errors_before > 0:
            print("Unfixable errors (need manual review):")
            for e in result.unfixable_errors[:20]:
                print(f"  {e.file_path}:{e.line} [{e.rule}] {e.message[:120]}")

        report = loop.generate_report()
        report_path = self.root / ".vf_auto_fix_report.json"
        import json
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return result.fixes_applied


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VF Auto-Fix Daemon")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon (currently foreground only)")
    parser.add_argument("--once", action="store_true", help="Single scan only")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="File poll interval in seconds")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    daemon = AutoFixDaemon()
    daemon.watcher.poll_interval = args.poll_interval

    if args.once:
        fixes = daemon.run_once()
        sys.exit(0 if fixes >= 0 else 1)
    else:
        daemon.run_forever()


if __name__ == "__main__":
    main()
