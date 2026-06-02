"""
Auto-Index Engine — Replaces Manual index_supplement_kb.py Injection
======================================================================
Scans AI_DATA/ periodically, detects new files, and auto-indexes
them into the appropriate soul knowledge bases.

Previously: index_supplement_kb.py → manual hardcoded knowledge injection
Now:        auto_index_engine.py → automatic file-system-driven indexing

Architecture:
    {AI_DATA}/* → FileDetector → ContentRouter → SoulMemoryEngine.write()

Routing Rules:
    MetObjects.csv           → monet, vangogh, davinci  (art metadata)
    Capybara/                → guizhu, einstein         (conversations)
    Capybara-code/           → cezanne                  (code reasoning)
    Opus-4.6-Reasoning/      → einstein, galileo        (math/science)
    rStar-Coder/             → cezanne                  (competitive code)
    *.mp4, *.webm            → cezanne                  (CVJEPA video index)
    3DPW/                    → cezanne                  (CVJEPA pose)
    SA-V/                    → cezanne                  (CVJEPA frames)
    *.txt, *.md              → cezanne                  (general text)

Usage:
    engine = AutoIndexEngine()
    report = engine.scan_and_index()                 # incremental
    report = engine.scan_and_index(full=True)         # full re-index
    report = engine.scan_and_index(dry_run=True)      # preview only

Journal:
    .vf_memory/index_journal.json  — tracks indexed file hashes + timestamps
"""

import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

AI_DATA_ROOT = os.getenv("AI_DATA", r"E:\AI_Data")
JOURNAL_PATH = Path(__file__).parent / ".vf_memory" / "index_journal.json"
JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)

SKIP_PATTERNS = {
    ".cache", ".git", "__pycache__", ".huggingface",
    ".gitattributes", "CACHEDIR.TAG", ".gitignore",
}

SKIP_EXTENSIONS = {
    ".metadata", ".incomplete", ".lock", ".tmp",
}

ROUTING_RULES = [
    {
        "path_patterns": ["MetObjects.csv", "MetObjects"],
        "souls": ["monet", "vangogh", "davinci"],
        "category": "domain_memory",
        "label": "Met Museum Art Metadata",
        "max_entries": 50000,
    },
    {
        "path_patterns": ["Capybara-code"],
        "souls": ["cezanne"],
        "category": "code_memory",
        "label": "Code Conversations",
        "max_entries": 20000,
    },
    {
        "path_patterns": ["Capybara", "capybara"],
        "souls": ["guizhu", "einstein"],
        "category": "conversation",
        "label": "Multi-turn Conversations",
        "max_entries": 20000,
    },
    {
        "path_patterns": ["Opus-4.6-Reasoning", "opus-4.6"],
        "souls": ["einstein", "galileo"],
        "category": "knowledge",
        "label": "Math Science Reasoning",
        "max_entries": 5000,
    },
    {
        "path_patterns": ["rStar-Coder", "rstar"],
        "souls": ["cezanne"],
        "category": "code_memory",
        "label": "Competitive Code Reasoning",
        "max_entries": 100000,
    },
    {
        "path_patterns": [r"3DPW", "imageFiles"],
        "souls": ["cezanne"],
        "category": "domain_memory",
        "label": "3D Human Pose Video",
        "max_entries": 200,
    },
    {
        "path_patterns": ["SA-V", "sav_"],
        "souls": ["cezanne"],
        "category": "domain_memory",
        "label": "Video Segmentation Dataset",
        "max_entries": 500,
    },
]

EXT_ROUTING = {
    ".csv": {"souls": ["cezanne"], "category": "knowledge", "label": "CSV Data"},
    ".json": {"souls": ["cezanne"], "category": "knowledge", "label": "JSON Data"},
    ".jsonl": {"souls": ["cezanne"], "category": "knowledge", "label": "JSONL Data"},
    ".parquet": {"souls": ["cezanne"], "category": "code_memory", "label": "Parquet Data"},
    ".arrow": {"souls": ["cezanne"], "category": "code_memory", "label": "Arrow Data"},
    ".mp4": {"souls": ["cezanne"], "category": "domain_memory", "label": "Video File"},
    ".webm": {"souls": ["cezanne"], "category": "domain_memory", "label": "Video File"},
    ".txt": {"souls": ["cezanne"], "category": "knowledge", "label": "Text File"},
    ".md": {"souls": ["cezanne"], "category": "knowledge", "label": "Markdown File"},
    ".yaml": {"souls": ["cezanne"], "category": "knowledge", "label": "YAML Config"},
    ".yml": {"souls": ["cezanne"], "category": "knowledge", "label": "YAML Config"},
    ".py": {"souls": ["cezanne"], "category": "code_memory", "label": "Python Source"},
    ".png": {"souls": ["monet"], "category": "domain_memory", "label": "PNG Image"},
    ".jpg": {"souls": ["monet"], "category": "domain_memory", "label": "JPEG Image"},
}

FILE_SIZE_LIMIT_BYTES = 500 * 1024 * 1024
READ_LIMIT_BYTES = 2 * 1024 * 1024


def _file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    size = os.path.getsize(filepath)
    hasher.update(str(size).encode())
    hasher.update(str(os.path.getmtime(filepath)).encode())
    hasher.update(filepath.encode())
    with open(filepath, "rb") as f:
        head = f.read(8192)
        hasher.update(head)
        if size > 16384:
            f.seek(size // 2)
            hasher.update(f.read(8192))
            f.seek(-8192, 2)
            hasher.update(f.read(8192))
    return hasher.hexdigest()


def _match_routing(filepath: str, filename_lower: str) -> Optional[dict]:
    for rule in ROUTING_RULES:
        for pat in rule["path_patterns"]:
            if pat.lower() in filepath.lower() or pat.lower() in filename_lower:
                return rule
    ext = os.path.splitext(filename_lower)[1]
    return EXT_ROUTING.get(ext)


def _read_preview(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read(READ_LIMIT_BYTES)
    except Exception:
        return f"[Binary file: {os.path.getsize(filepath)} bytes]"


def _summarize_csv(filepath: str) -> str:
    try:
        import csv
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = [next(reader, []) for _ in range(3)]
            rows = [r for r in rows if r]
        ncols = len(header)
        return f"CSV: {ncols} columns [{', '.join(header[:10])}...], {len(rows)} sample rows previewed"
    except Exception as e:
        return f"CSV parse error: {e}"


def _summarize_parquet(filepath: str) -> str:
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(filepath)
        cols = pf.schema.names
        nrows = pf.metadata.num_rows
        return f"Parquet: {nrows} rows, {len(cols)} columns [{', '.join(cols[:8])}...]"
    except ImportError:
        return f"Parquet file: {os.path.getsize(filepath)} bytes (pyarrow not installed for preview)"
    except Exception as e:
        return f"Parquet parse error: {e}"


def _summarize_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return _summarize_csv(filepath)
    elif ext == ".parquet":
        return _summarize_parquet(filepath)
    else:
        preview = _read_preview(filepath)
        if len(preview) > 2000:
            preview = preview[:2000] + "..."
        return preview


class AutoIndexEngine:
    def __init__(self, data_root: Optional[str] = None):
        self.data_root = data_root or AI_DATA_ROOT
        self._journal = self._load_journal()
        self._stats = {
            "total_files_scanned": 0,
            "new_files_indexed": 0,
            "skipped_files": 0,
            "errors": 0,
            "soul_entries": defaultdict(int),
        }

    def _load_journal(self) -> dict:
        if JOURNAL_PATH.exists():
            try:
                with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                logger.warning("Corrupted index journal, starting fresh")
        return {"version": 1, "entries": {}, "last_scan": None, "stats": {}}

    def _save_journal(self):
        self._journal["last_scan"] = time.time()
        self._journal["stats"] = dict(self._stats)
        with open(JOURNAL_PATH, "w", encoding="utf-8") as f:
            json.dump(self._journal, f, ensure_ascii=False, indent=2)

    def _should_skip(self, filepath: str, filename: str) -> bool:
        parts = Path(filepath).parts
        for part in parts:
            if part in SKIP_PATTERNS:
                return True
        ext = os.path.splitext(filename)[1].lower()
        if ext in SKIP_EXTENSIONS:
            return True
        try:
            size = os.path.getsize(filepath)
            if size > FILE_SIZE_LIMIT_BYTES:
                return True
            if size == 0:
                return True
        except OSError:
            return True
        if filename.startswith(".") and filename not in (".gitattributes",):
            return True
        return False

    def _is_indexed(self, filepath: str) -> bool:
        fh = _file_hash(filepath)
        entries = self._journal.get("entries", {})
        if filepath in entries:
            return entries[filepath].get("hash") == fh
        return False

    def _mark_indexed(self, filepath: str, souls: List[str], summary: str):
        fh = _file_hash(filepath)
        self._journal.setdefault("entries", {})[filepath] = {
            "hash": fh,
            "size": os.path.getsize(filepath),
            "souls": souls,
            "summary": summary[:200],
            "indexed_at": time.time(),
        }

    def scan_and_index(self, full: bool = False, dry_run: bool = False,
                       max_files: int = 10000) -> dict:
        self._stats = {
            "total_files_scanned": 0,
            "new_files_indexed": 0,
            "skipped_files": 0,
            "errors": 0,
            "soul_entries": defaultdict(int),
        }
        indexed_files = []
        errors = []

        if not os.path.exists(self.data_root):
            logger.warning(f"Data root not found: {self.data_root}")
            return {"status": "no_data_root", "path": self.data_root}

        try:
            from soul_memory import write
        except ImportError:
            logger.error("soul_memory.write not available")
            return {"status": "error", "message": "soul_memory module not available"}

        for root, dirs, files in os.walk(self.data_root):
            dirs[:] = [d for d in dirs if d not in SKIP_PATTERNS]
            for filename in files:
                if self._stats["new_files_indexed"] >= max_files:
                    break

                filepath = os.path.join(root, filename)
                self._stats["total_files_scanned"] += 1

                if self._should_skip(filepath, filename):
                    self._stats["skipped_files"] += 1
                    continue

                if not full and self._is_indexed(filepath):
                    self._stats["skipped_files"] += 1
                    continue

                routing = _match_routing(filepath, filename.lower())
                if routing is None:
                    continue

                try:
                    summary = _summarize_file(filepath)
                    souls = routing["souls"]
                    category = routing["category"]
                    label = routing["label"]
                    max_entries = routing.get("max_entries", 10000)

                    topic = f"[{label}] {os.path.relpath(filepath, self.data_root)}"
                    text = (
                        f"Source: {filepath}\n"
                        f"Type: {category}\n"
                        f"Label: {label}\n"
                        f"Size: {os.path.getsize(filepath)} bytes\n"
                        f"Summary: {summary}"
                    )

                    if not dry_run:
                        for soul in souls:
                            try:
                                write(soul, category, {
                                    "topic": topic,
                                    "detail": text,
                                    "source_file": filepath,
                                    "indexed_by": "auto_index_engine",
                                }, importance=0.4, tags=["auto-indexed", label.lower().replace(" ", "-")])
                                self._stats["soul_entries"][soul] += 1
                            except Exception as e:
                                errors.append(f"{soul}:{filepath}: {e}")
                                self._stats["errors"] += 1

                        self._mark_indexed(filepath, souls, summary[:200])

                    self._stats["new_files_indexed"] += 1
                    indexed_files.append({
                        "file": os.path.relpath(filepath, self.data_root),
                        "souls": souls,
                        "summary": summary[:120],
                    })

                except Exception as e:
                    errors.append(f"{filepath}: {e}")
                    self._stats["errors"] += 1

        if not dry_run:
            self._save_journal()

        return {
            "status": "completed",
            "dry_run": dry_run,
            "stats": {
                "total_files_scanned": self._stats["total_files_scanned"],
                "new_files_indexed": self._stats["new_files_indexed"],
                "skipped_files": self._stats["skipped_files"],
                "errors": self._stats["errors"],
                "soul_entries": dict(self._stats["soul_entries"]),
            },
            "indexed_files": indexed_files[:50],
            "errors": errors[:20],
        }

    def get_status(self) -> dict:
        entries = self._journal.get("entries", {})
        return {
            "data_root": self.data_root,
            "journal_path": str(JOURNAL_PATH),
            "total_indexed": len(entries),
            "last_scan": self._journal.get("last_scan"),
            "souls_indexed": list(set(
                s for e in entries.values() for s in e.get("souls", [])
            )),
            "total_size_indexed": sum(
                e.get("size", 0) for e in entries.values()
            ),
        }

    def reset_journal(self):
        self._journal = {"version": 1, "entries": {}, "last_scan": None, "stats": {}}
        self._save_journal()
        logger.info("Index journal reset")


_global_engine: Optional[AutoIndexEngine] = None


def get_auto_index_engine(data_root: Optional[str] = None) -> AutoIndexEngine:
    global _global_engine
    if _global_engine is None:
        _global_engine = AutoIndexEngine(data_root=data_root)
    return _global_engine


def auto_index_all(dry_run: bool = False) -> dict:
    engine = get_auto_index_engine()
    return engine.scan_and_index(full=False, dry_run=dry_run)


_STARTUP_DONE = False


def startup_auto_index():
    global _STARTUP_DONE
    if _STARTUP_DONE:
        return
    _STARTUP_DONE = True
    import threading
    def _run():
        try:
            engine = get_auto_index_engine()
            report = engine.scan_and_index(full=False, dry_run=False, max_files=50)
            n = report["stats"]["new_files_indexed"]
            if n > 0:
                logger.info(f"Auto-index: {n} files indexed into {len(report['stats']['soul_entries'])} souls")
        except Exception as e:
            logger.debug(f"Auto-index startup scan skipped: {e}")
    t = threading.Thread(target=_run, daemon=True, name="auto_index_startup")
    t.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    engine = AutoIndexEngine()

    print("=== AutoIndexEngine Status ===")
    status = engine.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n=== Dry Run Scan ===")
    report = engine.scan_and_index(dry_run=True)
    s = report["stats"]
    print(f"  Scanned: {s['total_files_scanned']}")
    print(f"  Would index: {s['new_files_indexed']}")
    print(f"  Skipped: {s['skipped_files']}")
    print(f"  Soul entries: {dict(s['soul_entries'])}")
    for f in report["indexed_files"][:10]:
        print(f"    → {f['souls']} | {f['file'][:80]}")

    if not report["dry_run"]:
        print("\n=== Real Index ===")
        report2 = engine.scan_and_index(full=False, dry_run=False)
        print(f"  Indexed: {report2['stats']['new_files_indexed']} files")
        print(f"  Errors: {report2['stats']['errors']}")
