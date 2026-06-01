"""
Long Memory — Persistent Cross-KB Memory Adapter
===================================================
Provides stateful conversation memory across all 14 knowledge bases.
Each memory is scoped to a KB + thread_id, with optional cross-KB links.

Design principles:
- Minimal: SQLite-based, no embedding dependency (delegates to CPHYSJEPA if available)
- Adapter pattern: plug into existing soul_orchestrator without refactoring
- TTL-driven: auto-expire stale contexts, keep active threads warm
- Cross-KB: memories can link across knowledge bases for multi-expert reasoning

Memory lifecycle:
    create → append → retrieve → (optionally) link → archive → expire

Usage:
    memory = LongMemory()
    memory.start_thread("einstein", "session_001")
    memory.append("einstein", "session_001", "User asked about quantum entanglement")
    history = memory.retrieve("einstein", "session_001", last_n=10)
    memory.link("einstein", "session_001", "darwin", "Bio-quantum cross-reference")
"""

import sqlite3
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

KB_NAMES = [
    "einstein", "cezanne", "galileo", "darwin", "strategy",
    "montesquieu", "beethoven", "davinci", "humboldt", "guizhu",
    "herodotus", "yuanlongping", "monet", "vangogh",
]

DEFAULT_MEMORY_DIR = Path("D:/VORTEX_FLAME/.vf_memory")
DEFAULT_DB_PATH = DEFAULT_MEMORY_DIR / "long_memory.db"

TTL_MAP = {
    "active": timedelta(hours=24),
    "warm": timedelta(days=7),
    "cold": timedelta(days=30),
    "archive": timedelta(days=365),
}


@dataclass
class MemoryEntry:
    thread_id: str
    kb_name: str
    role: str
    content: str
    timestamp: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "kb_name": self.kb_name,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class CrossKBLink:
    id: str
    source_kb: str
    source_thread: str
    target_kb: str
    label: str
    timestamp: float


class LongMemory:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_threads (
                thread_id TEXT NOT NULL,
                kb_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata_json TEXT DEFAULT '{}',
                PRIMARY KEY (thread_id, timestamp)
            );
            CREATE INDEX IF NOT EXISTS idx_mt_kb
                ON memory_threads(kb_name, thread_id, timestamp);
            CREATE TABLE IF NOT EXISTS cross_kb_links (
                id TEXT PRIMARY KEY,
                source_kb TEXT NOT NULL,
                source_thread TEXT NOT NULL,
                target_kb TEXT NOT NULL,
                label TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cross_source
                ON cross_kb_links(source_kb, source_thread);
            CREATE INDEX IF NOT EXISTS idx_cross_target
                ON cross_kb_links(target_kb);
            CREATE TABLE IF NOT EXISTS memory_meta (
                kb_name TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at REAL NOT NULL,
                last_access REAL NOT NULL,
                message_count INTEGER DEFAULT 0,
                PRIMARY KEY (kb_name, thread_id)
            );
        """)
        conn.commit()
        conn.close()

    def start_thread(self, kb_name: str, thread_id: str = None) -> str:
        if thread_id is None:
            thread_id = f"{kb_name}_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT OR REPLACE INTO memory_meta (kb_name, thread_id, status, created_at, last_access, message_count)
               VALUES (?, ?, 'active', ?, ?, 0)""",
            (kb_name, thread_id, time.time(), time.time())
        )
        conn.commit()
        conn.close()
        return thread_id

    def append(self, kb_name: str, thread_id: str, content: str, role: str = "user",
               metadata: dict = None) -> None:
        conn = sqlite3.connect(str(self.db_path))
        now = time.time()
        conn.execute(
            """INSERT INTO memory_threads (thread_id, kb_name, role, content, timestamp, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (thread_id, kb_name, role, content, now, json.dumps(metadata or {}))
        )
        conn.execute(
            """INSERT OR REPLACE INTO memory_meta (kb_name, thread_id, status, created_at, last_access, message_count)
               VALUES (?, ?, 'active', COALESCE((SELECT created_at FROM memory_meta WHERE kb_name=? AND thread_id=?), ?), ?, 
               COALESCE((SELECT message_count FROM memory_meta WHERE kb_name=? AND thread_id=?), 0) + 1)""",
            (kb_name, thread_id, kb_name, thread_id, now, now, kb_name, thread_id)
        )
        conn.commit()
        conn.close()

    def retrieve(self, kb_name: str, thread_id: str, last_n: int = 20,
                 before_timestamp: float = None) -> List[MemoryEntry]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        if before_timestamp:
            rows = conn.execute(
                """SELECT * FROM memory_threads
                   WHERE kb_name=? AND thread_id=? AND timestamp < ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (kb_name, thread_id, before_timestamp, last_n)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM memory_threads
                   WHERE kb_name=? AND thread_id=?
                   ORDER BY timestamp DESC LIMIT ?""",
                (kb_name, thread_id, last_n)
            ).fetchall()
        conn.execute(
            "UPDATE memory_meta SET last_access=? WHERE kb_name=? AND thread_id=?",
            (time.time(), kb_name, thread_id)
        )
        conn.commit()
        conn.close()
        return [
            MemoryEntry(
                thread_id=r["thread_id"], kb_name=r["kb_name"],
                role=r["role"], content=r["content"],
                timestamp=r["timestamp"],
                metadata=json.loads(r["metadata_json"])
            ) for r in reversed(rows)
        ]

    def retrieve_context_window(self, kb_name: str, thread_id: str,
                                 max_tokens_estimate: int = 4000,
                                 chars_per_token: int = 3) -> str:
        entries = self.retrieve(kb_name, thread_id, last_n=50)
        max_chars = max_tokens_estimate * chars_per_token
        total = 0
        selected = []
        for entry in reversed(entries):
            chunk = f"[{entry.role.upper()}@{entry.kb_name}]: {entry.content}\n"
            if total + len(chunk) > max_chars:
                break
            selected.insert(0, chunk)
            total += len(chunk)
        return "".join(selected)

    def link(self, source_kb: str, source_thread: str, target_kb: str,
             label: str) -> str:
        link_id = hashlib.md5(
            f"{source_kb}:{source_thread}:{target_kb}:{label}:{time.time()}".encode()
        ).hexdigest()[:12]
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT OR REPLACE INTO cross_kb_links (id, source_kb, source_thread, target_kb, label, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (link_id, source_kb, source_thread, target_kb, label, time.time())
        )
        conn.commit()
        conn.close()
        return link_id

    def get_links_from(self, source_kb: str, source_thread: str = None) -> List[CrossKBLink]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        if source_thread:
            rows = conn.execute(
                "SELECT * FROM cross_kb_links WHERE source_kb=? AND source_thread=? ORDER BY timestamp DESC",
                (source_kb, source_thread)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cross_kb_links WHERE source_kb=? ORDER BY timestamp DESC",
                (source_kb,)
            ).fetchall()
        conn.close()
        return [
            CrossKBLink(id=r["id"], source_kb=r["source_kb"],
                        source_thread=r["source_thread"],
                        target_kb=r["target_kb"], label=r["label"],
                        timestamp=r["timestamp"])
            for r in rows
        ]

    def get_links_to(self, target_kb: str) -> List[CrossKBLink]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM cross_kb_links WHERE target_kb=? ORDER BY timestamp DESC",
            (target_kb,)
        ).fetchall()
        conn.close()
        return [
            CrossKBLink(id=r["id"], source_kb=r["source_kb"],
                        source_thread=r["source_thread"],
                        target_kb=r["target_kb"], label=r["label"],
                        timestamp=r["timestamp"])
            for r in rows
        ]

    def list_threads(self, kb_name: str, status: str = None) -> List[dict]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        if status:
            rows = conn.execute(
                "SELECT * FROM memory_meta WHERE kb_name=? AND status=? ORDER BY last_access DESC",
                (kb_name, status)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memory_meta WHERE kb_name=? ORDER BY last_access DESC",
                (kb_name,)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def prune_expired(self):
        now = datetime.now()
        conn = sqlite3.connect(str(self.db_path))
        for status, ttl in TTL_MAP.items():
            cutoff = (now - ttl).timestamp()
            if status == "active":
                continue
            conn.execute(
                "DELETE FROM memory_threads WHERE thread_id IN (SELECT thread_id FROM memory_meta WHERE status=? AND last_access < ?)",
                (status, cutoff)
            )
            conn.execute(
                "DELETE FROM memory_meta WHERE status=? AND last_access < ?",
                (status, cutoff)
            )
        conn.commit()
        conn.close()

    def thread_stats(self, kb_name: str = None) -> dict:
        conn = sqlite3.connect(str(self.db_path))
        if kb_name:
            row = conn.execute(
                "SELECT COUNT(*) as count, SUM(message_count) as total_msgs FROM memory_meta WHERE kb_name=?",
                (kb_name,)
            ).fetchone()
            mb = conn.execute(
                "SELECT SUM(LENGTH(content)) as bytes FROM memory_threads WHERE kb_name=?",
                (kb_name,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) as count, SUM(message_count) as total_msgs FROM memory_meta"
            ).fetchone()
            mb = conn.execute(
                "SELECT SUM(LENGTH(content)) as bytes FROM memory_threads"
            ).fetchone()
        conn.close()
        return {
            "threads": row[0] or 0,
            "total_messages": row[1] or 0,
            "storage_mb": round((mb[0] or 0) / (1024 * 1024), 2),
        }


class LongMemoryAdapter:
    def __init__(self, memory: LongMemory = None):
        self._memory = memory or LongMemory()

    def for_kb(self, kb_name: str) -> "_KBMemoryHandle":
        return _KBMemoryHandle(self._memory, kb_name)


class _KBMemoryHandle:
    def __init__(self, memory: LongMemory, kb_name: str):
        self._memory = memory
        self.kb_name = kb_name
        self._current_thread: Optional[str] = None

    def start_thread(self, thread_id: str = None) -> str:
        self._current_thread = self._memory.start_thread(self.kb_name, thread_id)
        return self._current_thread

    def remember(self, content: str, role: str = "assistant", metadata: dict = None):
        if not self._current_thread:
            self.start_thread()
        self._memory.append(self.kb_name, self._current_thread, content, role, metadata)

    def recall(self, last_n: int = 20) -> List[MemoryEntry]:
        if not self._current_thread:
            return []
        return self._memory.retrieve(self.kb_name, self._current_thread, last_n=last_n)

    def recall_context(self, max_tokens: int = 4000) -> str:
        if not self._current_thread:
            return ""
        return self._memory.retrieve_context_window(self.kb_name, self._current_thread, max_tokens_estimate=max_tokens)

    def link_to(self, target_kb: str, label: str) -> str:
        if not self._current_thread:
            self.start_thread()
        return self._memory.link(self.kb_name, self._current_thread, target_kb, label)


if __name__ == "__main__":
    memory = LongMemory()
    adapter = LongMemoryAdapter(memory)

    handle = adapter.for_kb("einstein")
    tid = handle.start_thread()
    print(f"Started thread: {tid}")

    handle.remember("F=ma是牛顿第二定律的核心表达式")
    handle.remember("能量守恒适用于封闭系统")

    entries = handle.recall(last_n=10)
    for e in entries:
        print(f"  [{e.role}] {e.content[:80]}")

    stats = memory.thread_stats("einstein")
    print(f"\nEinstein memory stats: {stats}")
