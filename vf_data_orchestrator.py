#!/usr/bin/env python3
"""
VF Data Orchestrator — 企业级数据处理总控引擎
=================================================
整合所有模块提供统一入口:

模块依赖链:
  vf_data_core.py     → ConfigManager, AuditLogger, TaskScheduler, BaseProcessor
  vf_dedup.py         → DedupEngine (去重)
  vf_cleanse.py       → CleanseEngine (清洗)
  vf_inspect.py       → InspectEngine (检查)
  vf_distill.py       → DistillEngine (蒸馏)
  vf_automation.py    → AutomationOrchestrator (事件驱动自动化)
  vf_knowledge_filter.py → classify_entry (知识质量分类)

Pipeline:
  LOAD → DEDUP → CLEANSE → INSPECT → DISTILL → SAVE
           ↑        ↑         ↑         ↑
           └────────┴─────────┴─────────┘
                    AuditLogger (全链路审计)

自动化闭环:
  data.arrived ──→ [DEDUP] ──→ [CLEANSE] ──→ [INSPECT] ──→ [DISTILL] ──→ [SAVE]
                                                       ↓
                                                [KNOWNOWLEDGE_FILTER]
                                                       ↓
                                                [KB_HARNESS]

用法:
  python vf_data_orchestrator.py --pipeline full --input data.json --output cleaned.json
  python vf_data_orchestrator.py --pipeline quick --input data/
  python vf_data_orchestrator.py --serve  # 启动webhook服务器
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from vf_data_core import (
    ConfigManager, AuditLogger, TaskScheduler, TaskTrigger,
    BaseProcessor, ProcessingStats, ResourceMonitor,
    normalize_field_names,
)
from vf_dedup import DedupEngine
from vf_cleanse import CleanseEngine
from vf_inspect import InspectEngine
from vf_distill import DistillEngine
from vf_automation import (
    AutomationOrchestrator, WorkflowNode, TaskStateMachine, HookRegistry,
)
from vf_knowledge_filter import classify_entry

PROJECT_ROOT = Path(__file__).resolve().parent


class DataPipeline:
    def __init__(self, config_path: Optional[Path] = None):
        self.config = ConfigManager(config_path)
        self.audit = AuditLogger(self.config.audit_dir)

        self.dedup = DedupEngine(self.config, self.audit) if self.config.get("dedup", "enabled", default=True) else None
        self.cleanse = CleanseEngine(self.config, self.audit) if self.config.get("cleanse", "enabled", default=True) else None
        self.inspect = InspectEngine(self.config, self.audit) if self.config.get("inspect", "enabled", default=True) else None
        self.distill = DistillEngine(self.config, self.audit) if self.config.get("distill", "enabled", default=True) else None

        self.automation = AutomationOrchestrator(self.config, self.audit)
        self.resource = ResourceMonitor(
            cpu_limit=self.config.get("resources", "cpu_limit_pct", default=80.0),
            mem_limit=self.config.get("resources", "mem_limit_pct", default=70.0),
        )
        self._setup_workflows()

        self.audit.log("orchestrator", "init", "OK",
                       details={"modules": self.list_modules()})

    def _setup_workflows(self):
        full_nodes = [
            WorkflowNode(
                task_id="dedup", func=self._step_dedup,
                depends_on=[], timeout_s=600, retry_on_failure=True,
            ),
            WorkflowNode(
                task_id="cleanse", func=self._step_cleanse,
                depends_on=["dedup"], timeout_s=600, retry_on_failure=True,
            ),
            WorkflowNode(
                task_id="inspect", func=self._step_inspect,
                depends_on=["cleanse"], timeout_s=300, retry_on_failure=False,
            ),
            WorkflowNode(
                task_id="distill", func=self._step_distill,
                depends_on=["cleanse"], timeout_s=600, retry_on_failure=True,
            ),
        ]
        self.automation.register_workflow("full_pipeline", full_nodes)

        quick_nodes = [
            WorkflowNode(
                task_id="dedup", func=self._step_dedup,
                depends_on=[], timeout_s=300,
            ),
            WorkflowNode(
                task_id="inspect", func=self._step_inspect,
                depends_on=["dedup"], timeout_s=300,
            ),
        ]
        self.automation.register_workflow("quick_validate", quick_nodes)

    def _step_dedup(self, payload: Dict) -> Dict[str, Any]:
        data = payload.get("data", [])
        if not data:
            return {"output": [], "stats": {}}
        output, stats = self.dedup.process(data)
        return {
            "output": output,
            "stats": stats.__dict__ if hasattr(stats, '__dict__') else {},
            "count": len(output),
        }

    def _step_cleanse(self, payload: Dict) -> Dict[str, Any]:
        data = payload.get("data", []) or payload.get("context", {}).get("dedup", {}).get("output", [])
        if not data:
            return {"output": [], "stats": {}}
        output, stats = self.cleanse.process(data)
        return {
            "output": output,
            "stats": stats.__dict__ if hasattr(stats, '__dict__') else {},
            "count": len(output),
        }

    def _step_inspect(self, payload: Dict) -> Dict[str, Any]:
        data = payload.get("data", []) or payload.get("context", {}).get("cleanse", {}).get("output", [])
        if not data:
            return {"output": [], "report": {}}
        output, stats = self.inspect.process(data)
        return {
            "output": output,
            "report": self.inspect.report(),
            "issues": len(self.inspect._report.issues),
            "score": self.inspect._report.overall_score,
        }

    def _step_distill(self, payload: Dict) -> Dict[str, Any]:
        data = payload.get("data", []) or payload.get("context", {}).get("cleanse", {}).get("output", [])
        if not data:
            return {"output": [], "stats": {}}
        output, stats = self.distill.process(data)
        return {
            "output": output,
            "stats": stats.__dict__ if hasattr(stats, '__dict__') else {},
            "count": len(output),
        }

    def load_data(self, source: str, format_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        source_path = Path(source)
        t0 = time.time()
        data = []

        if source_path.is_dir():
            data = self._load_from_directory(source_path)
        elif source_path.suffix.lower() == ".json" or source_path.suffix.lower() == ".jsonl":
            data = self._load_json(source_path)
        elif source_path.suffix.lower() == ".csv":
            data = self._load_csv(source_path)
        elif source_path.suffix.lower() in (".yaml", ".yml"):
            data = self._load_yaml(source_path)
        else:
            try:
                data = self._load_json(source_path)
            except Exception:
                raise ValueError(f"Unsupported format: {source_path.suffix}")

        data = normalize_field_names(data)
        self.audit.log("orchestrator", "load", "OK",
                       duration_ms=(time.time() - t0) * 1000,
                       record_count=len(data),
                       details={"source": str(source), "count": len(data)})
        return data

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ["data", "records", "entries", "items", "results"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        if isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, str):
                    try:
                        result.append(json.loads(item))
                    except json.JSONDecodeError:
                        result.append({"content": item})
                else:
                    result.append(item)
            return result
        return []

    def _load_csv(self, path: Path) -> List[Dict[str, Any]]:
        import csv
        data = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned = {}
                for k, v in row.items():
                    if v is None or v.strip() == "":
                        cleaned[k.strip()] = None
                    else:
                        try:
                            cleaned[k.strip()] = int(v)
                        except ValueError:
                            try:
                                cleaned[k.strip()] = float(v)
                            except ValueError:
                                cleaned[k.strip()] = v.strip()
                data.append(cleaned)
        return data

    def _load_yaml(self, path: Path) -> List[Dict[str, Any]]:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    def _load_from_directory(self, dir_path: Path) -> List[Dict[str, Any]]:
        data = []
        for fp in sorted(dir_path.glob("*.json")) + sorted(dir_path.glob("*.jsonl")) + sorted(dir_path.glob("*.csv")):
            try:
                loaded = self.load_data(str(fp))
                data.extend(loaded)
            except Exception as e:
                logging.warning(f"Skipping {fp}: {e}")
        return data

    def save_data(self, data: List[Dict[str, Any]], target: str, format: str = "json"):
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == "jsonl":
            with open(target_path, "w", encoding="utf-8") as f:
                for rec in data:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        elif format == "csv":
            self._save_csv(data, target_path)

        self.audit.log("orchestrator", "save", "OK",
                       record_count=len(data),
                       details={"target": str(target), "format": format})

    def _save_csv(self, data: List[Dict[str, Any]], target_path: Path):
        if not data:
            return
        import csv
        all_fields = list(dict.fromkeys(
            f for rec in data for f in rec.keys() if not f.startswith("_")
        ))
        with open(target_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            for rec in data:
                row = {}
                for k in all_fields:
                    v = rec.get(k)
                    if isinstance(v, (dict, list)):
                        row[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        row[k] = v
                writer.writerow(row)

    def run_pipeline(self, name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        self.audit.set_trace(f"pipeline-{name}-{int(t0)}")

        self.automation._step_dedup = self._step_dedup
        self.automation._step_cleanse = self._step_cleanse
        self.automation._step_inspect = self._step_inspect
        self.automation._step_distill = self._step_distill

        context = {"data": data}
        run = self.automation.trigger_workflow(name, context)

        result = {
            "pipeline": name,
            "status": run.status.value if run else "FAILED",
            "duration_ms": (time.time() - t0) * 1000,
            "steps": {},
            "output": [],
        }

        if run:
            for task_id, inst in run.instances.items():
                result["steps"][task_id] = {
                    "state": inst.state.value,
                    "duration_ms": inst.duration_ms,
                    "error": inst.error_message[:100] if inst.error_message else "",
                }
                if inst.result and isinstance(inst.result, dict):
                    output_data = inst.result.get("output", [])
                    if output_data:
                        result["output"] = output_data

        self.audit.log("orchestrator", f"pipeline:{name}", run.status.value if run else "FAILED",
                       duration_ms=(time.time() - t0) * 1000,
                       record_count=len(result.get("output", [])))
        return result

    def start_server(self, webhook_port: int = 9845):
        self.automation.start(webhook_port=webhook_port)
        print(f"VF Data Pipeline server running on http://0.0.0.0:{webhook_port}")
        print("Registered workflows:", list(self.automation._workflows.keys()))
        print("Send POST to /data.arrived to trigger full_pipeline")
        print("Press Ctrl+C to stop")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.automation.stop()

    def list_modules(self) -> List[str]:
        modules = []
        if self.dedup:
            modules.append("dedup")
        if self.cleanse:
            modules.append("cleanse")
        if self.inspect:
            modules.append("inspect")
        if self.distill:
            modules.append("distill")
        modules.append("automation")
        return modules

    def status(self) -> Dict[str, Any]:
        return {
            "modules": self.list_modules(),
            "automation": self.automation.status(),
            "config_version": self.config.get("system", "version", default="1.0.0"),
            "resource": self.resource.snapshot().__dict__,
        }


def main():
    parser = argparse.ArgumentParser(
        description="VF Data Pipeline — Enterprise Data Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vf_data_orchestrator.py --pipeline full --input data.json --output cleaned.json
  python vf_data_orchestrator.py --pipeline quick --input ./data_dir/
  python vf_data_orchestrator.py --serve --port 9845
  python vf_data_orchestrator.py --status
        """,
    )

    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "vf_data_config.yaml"),
                        help="Path to config YAML")
    parser.add_argument("--pipeline", type=str, choices=["full", "quick", "dedup", "cleanse", "inspect", "distill"],
                        help="Pipeline to run")
    parser.add_argument("--input", type=str, help="Input file or directory")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--format", type=str, choices=["json", "jsonl", "csv"], default="json",
                        help="Output format")
    parser.add_argument("--serve", action="store_true", help="Start webhook server")
    parser.add_argument("--port", type=int, default=9845, help="Webhook server port")
    parser.add_argument("--status", action="store_true", help="Print system status")

    args = parser.parse_args()

    config_path = Path(args.config)
    pipeline = DataPipeline(config_path)

    if args.status:
        print(json.dumps(pipeline.status(), ensure_ascii=False, indent=2))
        return

    if args.serve:
        pipeline.start_server(webhook_port=args.port)
        return

    if args.pipeline and args.input:
        print(f"Loading data from: {args.input}")
        data = pipeline.load_data(args.input)
        print(f"Loaded {len(data)} records")

        result = pipeline.run_pipeline(args.pipeline, data)
        print(f"\nPipeline '{args.pipeline}': {result['status']} ({result['duration_ms']:.0f}ms)")

        for step_id, step_info in result["steps"].items():
            status_icon = "✅" if step_info["state"] == "SUCCESS" else "❌" if step_info["state"] == "FAILURE" else "⏭"
            print(f"  {status_icon} {step_id}: {step_info['state']} ({step_info['duration_ms']:.0f}ms)")

        if args.output and result["output"]:
            pipeline.save_data(result["output"], args.output, args.format)
            print(f"\nSaved {len(result['output'])} records to {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
