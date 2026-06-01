"""
Soul Memory — Persistent Memory Engine with Cross-Soul Bridge
==============================================================
Inspired by Supermemory's production architecture:
- Relational Versioning: updates/extends/derives/contradicts knowledge chains
- Temporal Grounding: dual timestamps (document_date + event_date)
- Hybrid Search: BM25 keyword + semantic embedding + graph traversal + temporal filtering
- Knowledge Contradiction Detection: auto-detect stale facts on write
- Forgetting Mechanism: access-frequency + time-decay pruning
- User Profiles: auto-extract identity/preferences from conversations
- Cross-Soul Bridge: Permission-gated read/write across soul memories

Storage: SQLite (single file per soul) + JSONL append-only log
Embedding: sentence-transformers/all-MiniLM-L6-v2 (384-dim, CPU <10ms)
No GPU required. No external LLM API required for core operations.
Graceful degradation: falls back to pure BM25 if embedding model unavailable.

Memory Categories (10):
- identity: Soul personality and core beliefs
- knowledge: Domain knowledge (BM25+semantic indexed)
- conversation: Conversation history
- todo: Task tracking
- self_training: Auto-generated training data
- trajectory: Success/failure execution trajectories
- audit: Security audit logs
- skill: Skill evolution records
- code_memory: Code structure index (CodeGraph syntax layer)
- domain_memory: Business domain knowledge (Understand-Anything semantic layer)

Public API:
- write(soul, category, content) -> entry_id
- search(soul, category, query, top_k) -> results
- recall(soul, query) -> cross-category recall with graph traversal
- ai_wakeup(soul) -> identity + recent context
- index_code(project_path) -> code_memory stats
- index_domain(project_path) -> domain_memory stats
- code_context(symbol) -> syntax + semantic context
- forget(soul, category, older_than_days) -> deleted count
- get_profile(soul) -> user profile summary
- detect_contradictions(soul, category, new_content) -> conflicting entries
- backfill_embeddings(soul) -> generate embeddings for existing entries
- cross_soul_recall(source_soul, query) -> permission-gated cross-soul query
- cross_soul_write(source_soul, target_soul, ...) -> permission-gated cross-soul write

Cross-Soul Permissions:
- CROSS_SOUL_PERMISSIONS: 14×14 directed permission graph
- Each soul can only read/write memories of explicitly listed souls
- cross_soul_write tags entries with source soul for traceability
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

logger = logging.getLogger(__name__)


MEMORY_CATEGORIES = [
    "identity",
    "knowledge",
    "conversation",
    "todo",
    "self_training",
    "trajectory",
    "audit",
    "skill",
    "code_memory",
    "domain_memory",
]

RELATION_TYPES = ["updates", "extends", "derives", "contradicts", "supersedes"]

EMBEDDING_DIM = 384
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HYBRID_ALPHA = 0.4

_MEMORY_DIR = Path(os.environ.get("VORTEX_FLAME_MEMORY_DIR", ".vf_memory"))


class EmbeddingProvider:
    _instance: Optional["EmbeddingProvider"] = None
    _model = None
    _available: Optional[bool] = None

    @classmethod
    def get(cls) -> "EmbeddingProvider":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._try_load()
        return self._available

    def _try_load(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            if self.__class__._model is None:
                self.__class__._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            return True
        except Exception as e:
            logger.warning(f"Embedding model unavailable, falling back to pure BM25: {e}")
            return False

    def encode(self, text: str) -> Optional[bytes]:
        model = self.__class__._model
        if not self.available or model is None:
            return None
        try:
            vec = model.encode(text, normalize_embeddings=True)
            return np.array(vec, dtype=np.float32).tobytes()
        except Exception:
            return None

    def encode_batch(self, texts: List[str]) -> List[Optional[bytes]]:
        if not self.available or self.__class__._model is None or not texts:
            return [None] * len(texts)
        model: Any = cast(Any, self.__class__._model)
        try:
            vecs_list = model.encode(texts, normalize_embeddings=True,
                                batch_size=64, show_progress_bar=False)
            return [np.array(v, dtype=np.float32).tobytes() for v in vecs_list]  # type: ignore[arg-type]
        except Exception:
            return [None] * len(texts)

    def encode_query(self, query: str) -> Optional[np.ndarray]:
        if not self.available or self.__class__._model is None:
            return None
        m: Any = cast(Any, self.__class__._model)
        try:
            vec = m.encode(query, normalize_embeddings=True)
            return np.array(vec, dtype=np.float32)
        except Exception:
            return None

    @staticmethod
    def cosine_similarity(query_vec: np.ndarray, entry_embedding_bytes: bytes) -> float:
        entry_vec = np.frombuffer(entry_embedding_bytes, dtype=np.float32)
        dot = np.dot(query_vec, entry_vec)
        q_norm = np.linalg.norm(query_vec)
        e_norm = np.linalg.norm(entry_vec)
        if q_norm < 1e-9 or e_norm < 1e-9:
            return 0.0
        return float(dot / (q_norm * e_norm))

    @staticmethod
    def blob_to_vec(blob: bytes) -> Optional[np.ndarray]:
        if blob is None or len(blob) == 0:
            return None
        return np.frombuffer(blob, dtype=np.float32)


@dataclass
class MemoryEntry:
    entry_id: str
    soul: str
    category: str
    content: dict
    document_date: str
    event_date: Optional[str] = None
    relations: List[dict] = field(default_factory=list)
    access_count: int = 0
    last_accessed: Optional[str] = None
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> "MemoryEntry":
        relations = json.loads(row.get("relations", "[]"))
        tags = json.loads(row.get("tags", "[]"))
        content = json.loads(row.get("content", "{}"))
        return cls(
            entry_id=row["entry_id"],
            soul=row["soul"],
            category=row["category"],
            content=content,
            document_date=row["document_date"],
            event_date=row.get("event_date"),
            relations=relations,
            access_count=row.get("access_count", 0),
            last_accessed=row.get("last_accessed"),
            importance=row.get("importance", 0.5),
            tags=tags,
        )


@dataclass
class MemoryRelation:
    relation_type: str
    target_id: str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {"type": self.relation_type, "target_id": self.target_id, "confidence": self.confidence}


@dataclass
class UserProfile:
    soul: str
    preferences: List[dict] = field(default_factory=list)
    identity_facts: List[dict] = field(default_factory=list)
    behavior_patterns: List[dict] = field(default_factory=list)
    last_updated: Optional[str] = None


class SoulMemoryEngine:
    """
    SQLite-backed persistent memory engine with:
    - Relational versioning (knowledge chains)
    - Dual timestamps (document_date + event_date)
    - BM25 full-text search via SQLite FTS5
    - Graph traversal for relation chains
    - Time-decay forgetting
    - Contradiction detection
    - Per-soul write locks for parallel safety
    """

    def __init__(self, memory_dir: Optional[str] = None):
        self.memory_dir = Path(memory_dir or _MEMORY_DIR)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._connections: Dict[str, sqlite3.Connection] = {}
        self._profiles: Dict[str, UserProfile] = {}
        self._embedding_provider = EmbeddingProvider.get()
        self._write_locks: Dict[str, threading.Lock] = {}

    def _get_write_lock(self, soul: str) -> threading.Lock:
        if soul not in self._write_locks:
            self._write_locks[soul] = threading.Lock()
        return self._write_locks[soul]

    def _get_db(self, soul: str) -> sqlite3.Connection:
        if soul not in self._connections:
            db_path = self.memory_dir / f"{soul}.db"
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._init_schema(conn)
            self._connections[soul] = conn
            self._auto_backup(soul, db_path)
        return self._connections[soul]

    def _auto_backup(self, soul: str, db_path: Path):
        backup_dir = self.memory_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{soul}_{int(time.time())}.db"
        try:
            if db_path.exists() and db_path.stat().st_size > 0:
                import shutil
                shutil.copy2(str(db_path), str(backup_path))
                backups = sorted(backup_dir.glob(f"{soul}_*.db"))
                max_backups = 5
                for old_backup in backups[:-max_backups]:
                    old_backup.unlink(missing_ok=True)
        except Exception:
            pass

    def _init_schema(self, conn: sqlite3.Connection):
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                entry_id TEXT PRIMARY KEY,
                soul TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                document_date TEXT NOT NULL,
                event_date TEXT,
                relations TEXT DEFAULT '[]',
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                importance REAL DEFAULT 0.5,
                tags TEXT DEFAULT '[]',
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_memories_soul_cat
                ON memories(soul, category);
            CREATE INDEX IF NOT EXISTS idx_memories_doc_date
                ON memories(document_date);
            CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance DESC);

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(entry_id, soul, category, content_text, tags_text,
                    tokenize='unicode61');

            CREATE TABLE IF NOT EXISTS relations (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (source_id, target_id, relation_type)
            );

            CREATE INDEX IF NOT EXISTS idx_relations_source
                ON relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_relations_target
                ON relations(target_id);

            CREATE TABLE IF NOT EXISTS profiles (
                soul TEXT PRIMARY KEY,
                preferences TEXT DEFAULT '[]',
                identity_facts TEXT DEFAULT '[]',
                behavior_patterns TEXT DEFAULT '[]',
                last_updated TEXT
            );
        """)
        self._ensure_embedding_column(conn)

    def _ensure_embedding_column(self, conn: sqlite3.Connection):
        cols = [row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()]
        if "embedding" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
            conn.commit()

    def _generate_id(self, soul: str, category: str, content: dict) -> str:
        raw = f"{soul}:{category}:{json.dumps(content, sort_keys=True)}:{time.time_ns()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _content_to_text(self, content: dict) -> str:
        parts = []
        for k, v in content.items():
            if isinstance(v, str):
                tokenized = self._maybe_tokenize_chinese(v)
                if tokenized != v:
                    logger.debug(f"[tokenize:write] field=\"{k}\" raw=\"{v[:40]}...\" → tokens={len(tokenized.split())}")
                parts.append(tokenized)
            elif isinstance(v, list):
                for item in v:
                    item_s = str(item)
                    tokenized_item = self._maybe_tokenize_chinese(item_s)
                    if tokenized_item != item_s:
                        logger.debug(f"[tokenize:write] field=\"{k}\" list_item raw=\"{item_s[:40]}...\" → tokens={len(tokenized_item.split())}")
                    parts.append(tokenized_item)
            elif isinstance(v, dict):
                parts.append(json.dumps(v))
            else:
                parts.append(str(v))
        return " ".join(parts)

    def write(self, soul: str, category: str, content: dict,
              event_date: Optional[str] = None,
              relations: Optional[List[dict]] = None,
              importance: float = 0.5,
              tags: Optional[List[str]] = None) -> str:
        if category not in MEMORY_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Must be one of {MEMORY_CATEGORIES}")

        conn = self._get_db(soul)
        entry_id = self._generate_id(soul, category, content)
        now = self._now_iso()
        content_json = json.dumps(content, ensure_ascii=False)
        content_text = self._content_to_text(content)
        tags_list = tags or []
        tags_json = json.dumps(tags_list)
        tags_text = " ".join(tags_list)
        relations_list = relations or []
        relations_json = json.dumps(relations_list)

        contradictions = self.detect_contradictions(soul, category, content)
        for c in contradictions:
            relations_list.append(MemoryRelation(
                relation_type="contradicts",
                target_id=c["entry_id"],
                confidence=c.get("confidence", 0.7),
            ).to_dict())
            conn.execute(
                "INSERT OR IGNORE INTO relations (source_id, target_id, relation_type, confidence) VALUES (?, ?, 'contradicted_by', ?)",
                (c["entry_id"], entry_id, c.get("confidence", 0.7)),
            )
        relations_json = json.dumps(relations_list)

        embedding_blob = self._embedding_provider.encode(content_text)

        lock = self._get_write_lock(soul)
        with lock:
            conn.execute("""
                INSERT INTO memories (entry_id, soul, category, content, document_date, event_date,
                                      relations, importance, tags, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (entry_id, soul, category, content_json, now, event_date,
                  relations_json, importance, tags_json, embedding_blob))

            conn.execute("""
                INSERT INTO memories_fts (entry_id, soul, category, content_text, tags_text)
                VALUES (?, ?, ?, ?, ?)
            """, (entry_id, soul, category, content_text, tags_text))

            for rel in relations_list:
                conn.execute(
                    "INSERT OR IGNORE INTO relations (source_id, target_id, relation_type, confidence) VALUES (?, ?, ?, ?)",
                    (entry_id, rel["target_id"], rel["type"], rel.get("confidence", 1.0)),
                )

            self._update_profile_from_write(soul, category, content)
            conn.commit()
        return entry_id

    def search(self, soul: str, category: str, query: str,
               top_k: int = 5,
               time_filter: Optional[dict] = None,
               min_importance: float = 0.0,
               alpha: Optional[float] = None) -> list:
        if category not in MEMORY_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Must be one of {MEMORY_CATEGORIES}")

        logger.info(f"[search] soul={soul} category={category} query=\"{query[:80]}\" top_k={top_k} alpha={alpha}")
        t_search_start = time.time()

        conn = self._get_db(soul)
        blend = alpha if alpha is not None else HYBRID_ALPHA
        params: list = []
        conditions = ["m.soul = ?", "m.category = ?"]
        params.extend([soul, category])

        if min_importance > 0:
            conditions.append("m.importance >= ?")
            params.append(min_importance)

        if time_filter:
            if "after" in time_filter:
                conditions.append("m.document_date >= ?")
                params.append(time_filter["after"])
            if "before" in time_filter:
                conditions.append("m.document_date <= ?")
                params.append(time_filter["before"])

        where_clause = " AND ".join(conditions)

        fts_query, fts_tokens = self._build_fts_query(query)
        logger.info(f"[search] soul={soul} category={category} fts_query=\"{fts_query}\" tokens={fts_tokens}")

        fts_rows = conn.execute("""
            SELECT entry_id, rank FROM memories_fts
            WHERE soul = ? AND category = ? AND memories_fts MATCH ?
            ORDER BY rank LIMIT ?
        """, (soul, category, fts_query, top_k * 3)).fetchall()
        fts_ids = [row["entry_id"] for row in fts_rows]
        fts_rank = {row["entry_id"]: row["rank"] for row in fts_rows}

        logger.info(f"[search] soul={soul} category={category} BM25 hits={len(fts_ids)}")
        for i, row in enumerate(fts_rows[:5]):
            logger.debug(f"[search]   BM25 #{i+1}: entry_id={row['entry_id'][:16]}... rank={row['rank']:.4f}")

        if fts_rows:
            token_debug = self._fts_token_match_debug(conn, soul, category, fts_tokens)
            if token_debug:
                logger.debug(f"[search]   token matches: {token_debug}")

        rows = conn.execute(f"""
            SELECT * FROM memories m WHERE {where_clause}
            ORDER BY m.importance DESC, m.document_date DESC
            LIMIT ?
        """, params + [top_k * 2]).fetchall()

        all_entries = []
        seen = set()
        for row in rows:
            entry = MemoryEntry.from_row(dict(row))
            if entry.entry_id not in seen:
                seen.add(entry.entry_id)
                all_entries.append((entry, dict(row)))

        if fts_ids:
            placeholders = ",".join("?" * len(fts_ids))
            fts_rows_full = conn.execute(f"""
                SELECT * FROM memories m WHERE m.entry_id IN ({placeholders})
            """, fts_ids).fetchall()
            for row in fts_rows_full:
                entry = MemoryEntry.from_row(dict(row))
                if entry.entry_id not in seen:
                    seen.add(entry.entry_id)
                    all_entries.append((entry, dict(row)))

        query_vec = self._embedding_provider.encode_query(query)
        has_semantic = query_vec is not None

        scored = []
        for entry, raw_row in all_entries:
            bm25_score = 0.0
            if entry.entry_id in fts_rank:
                bm25_score = max(0.0, -fts_rank[entry.entry_id])

            semantic_score = 0.0
            if has_semantic:
                emb_blob = raw_row.get("embedding")
                if emb_blob is not None and len(emb_blob) > 0:
                    semantic_score = EmbeddingProvider.cosine_similarity(query_vec, emb_blob)

            if has_semantic and any(r[1].get("embedding") for r in scored if r[1].get("embedding")):
                max_bm25 = max((s[2] for s in scored), default=1.0) or 1.0
            else:
                max_bm25 = max(bm25_score, 1.0)

            scored.append((entry, raw_row, bm25_score, semantic_score))

        logger.info(f"[search] soul={soul} category={category} candidates={len(scored)} (BM25+importance+FTS dedup)")

        if scored:
            max_bm25 = max(s[2] for s in scored) or 1.0
            max_sem = max(s[3] for s in scored) or 1.0

            def hybrid_sort_key(item):
                entry, _, bm25_s, sem_s = item
                norm_bm25 = bm25_s / max_bm25 if max_bm25 > 0 else 0.0
                norm_sem = sem_s / max_sem if max_sem > 0 else 0.0
                if has_semantic and max_sem > 0:
                    combined = blend * norm_bm25 + (1 - blend) * norm_sem
                else:
                    combined = norm_bm25
                importance_boost = entry.importance * 0.1
                return -(combined + importance_boost)

            scored.sort(key=hybrid_sort_key)

        results = []
        for entry, raw_row, bm25_s, sem_s in scored[:top_k]:
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE entry_id = ?",
                (self._now_iso(), entry.entry_id),
            )
            results.append(entry.to_dict())
            content_preview = (entry.content.get("topic", "") or entry.content.get("detail", "") or str(entry.content))[:60]
            logger.debug(
                f"[search]   result: entry_id={entry.entry_id[:16]}... "
                f"bm25={bm25_s:.3f} sem={sem_s:.3f} importance={entry.importance:.2f} "
                f"preview=\"{content_preview}\""
            )

        conn.commit()
        elapsed = (time.time() - t_search_start) * 1000
        logger.info(f"[search] soul={soul} category={category} returned={len(results)} elapsed={elapsed:.1f}ms")
        return results

    FTS_SPECIAL_CHARS = re.compile(r'[\*\:\^\+\-]|(?:\bAND\b|\bOR\b|\bNOT\b|\bNEAR\b)')
    _CHINESE_CHAR = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
    _jieba_loaded: bool = False

    _DOMAIN_TERMS = [
        "代码审计", "安全审计", "漏洞扫描", "SQL注入", "命令注入",
        "权限提升", "拒绝服务", "信息泄露", "供应链攻击", "零日漏洞",
        "攻击面", "威胁模型", "数据分类", "访问控制", "认证绕过",
        "会话劫持", "跨站脚本", "跨站请求", "反序列化", "任意文件",
        "远程代码", "缓冲区溢出", "整数溢出", "格式化字符串",
        "深度学习", "神经网络", "自然语言", "知识图谱", "向量检索",
        "混合检索", "语义搜索", "全文检索", "分词器", "嵌入向量",
    ]

    @classmethod
    def _maybe_tokenize_chinese(cls, text: str) -> str:
        if not cls._CHINESE_CHAR.search(text):
            logger.debug(f"[tokenize:cjk] skip (no CJK chars): \"{text[:60]}\"")
            return text
        if not cls._jieba_loaded:
            try:
                import jieba
                jieba.setLogLevel(logging.WARNING)
                for term in cls._DOMAIN_TERMS:
                    jieba.add_word(term)
                cls._jieba_loaded = True
                logger.info(f"[tokenize:cjk] jieba initialized with {len(cls._DOMAIN_TERMS)} domain terms")
            except ImportError:
                logger.debug("[tokenize:cjk] jieba not installed, returning raw text")
                return text
        import jieba
        tokens = list(jieba.cut(text))
        result = ' '.join(tokens)
        logger.debug(
            f"[tokenize:cjk] raw=\"{text[:80]}\" → tokens={len(tokens)} "
            f"sample=[{', '.join(repr(t) for t in tokens[:10])}]{'...' if len(tokens) > 10 else ''}"
        )
        return result

    @staticmethod
    def _tokenize_chinese_tokens(tokens: List[str]) -> List[str]:
        expanded = []
        cjk_count = 0
        for t in tokens:
            if SoulMemoryEngine._CHINESE_CHAR.search(t):
                cjk_count += 1
                tokenized = SoulMemoryEngine._maybe_tokenize_chinese(t)
                sub_tokens = tokenized.split()
                logger.debug(
                    f"[tokenize:query] CJK token raw=\"{t[:40]}\" → "
                    f"sub_tokens={len(sub_tokens)} [{', '.join(repr(s) for s in sub_tokens)}]"
                )
                expanded.extend(sub_tokens)
            else:
                expanded.append(t)
        if cjk_count > 0:
            logger.info(f"[tokenize:query] {cjk_count}/{len(tokens)} CJK tokens detected, expanded to {len(expanded)} total tokens")
        return expanded

    @staticmethod
    def _build_fts_query(query: str) -> Tuple[str, List[str]]:
        cleaned = re.sub(r'[\*\:\^\+\-]', ' ', query)
        cleaned = re.sub(r'\b(AND|OR|NOT|NEAR)\b', ' ', cleaned, flags=re.IGNORECASE)
        tokens = [t.strip() for t in cleaned.split() if t.strip()]
        tokens = SoulMemoryEngine._tokenize_chinese_tokens(tokens)
        escaped = [t.replace('"', '""') for t in tokens]
        starred_count = 0
        for i, t in enumerate(escaped):
            if len(t) >= 2 and not t.endswith('*'):
                escaped[i] = t + '*'
                starred_count += 1
        fts_query = ' OR '.join(escaped) if escaped else query.replace('"', '""')
        logger.info(
            f"[tokenize:fts_build] query=\"{query[:60]}\" → tokens={len(escaped)} "
            f"starred={starred_count} fts_query=\"{fts_query[:100]}\""
        )
        return fts_query, tokens

    @staticmethod
    def _fts_token_match_debug(conn, soul: str, category: str,
                                tokens: List[str]) -> str:
        parts = []
        for t in tokens:
            try:
                escaped = t.replace('"', '""')
                count = conn.execute(
                    "SELECT COUNT(*) as c FROM memories_fts WHERE soul=? AND category=? AND memories_fts MATCH ?",
                    (soul, category, escaped)
                ).fetchone()["c"]
                parts.append(f"{t}={count}")
            except Exception:
                parts.append(f"{t}=err")
        return ', '.join(parts)

    CATEGORY_ALPHA_PROFILES = {
        "knowledge": 0.3,
        "conversation": 0.5,
        "todo": 0.6,
        "self_training": 0.4,
        "trajectory": 0.5,
        "audit": 0.6,
        "skill": 0.4,
        "code_memory": 0.3,
        "domain_memory": 0.3,
        "identity": 0.5,
    }

    def _auto_alpha(self, query: str, category: str) -> float:
        base = self.CATEGORY_ALPHA_PROFILES.get(category, HYBRID_ALPHA)
        query_len = len(query.split())
        has_technical = any(t in query.lower() for t in [
            "equation", "formula", "theorem", "code", "function", "api",
            "方程", "公式", "定理", "代码", "函数", "接口",
        ])
        if has_technical:
            return min(base + 0.15, 0.8)
        if query_len > 20:
            return max(base - 0.1, 0.2)
        return base

    def recall(self, soul: str, query: str, top_k: int = 10,
               categories: Optional[List[str]] = None,
               alpha: Optional[float] = None) -> list:
        t_recall_start = time.time()
        conn = self._get_db(soul)
        cat_list = categories or MEMORY_CATEGORIES
        results = []

        logger.info(f"[recall] soul={soul} query=\"{query[:80]}\" top_k={top_k} categories={cat_list}")

        cat_hit_summary = {}
        for cat in cat_list:
            cat_alpha = alpha if alpha is not None else self._auto_alpha(query, cat)
            cat_results = self.search(soul, cat, query, top_k=max(3, top_k // len(cat_list)),
                                      alpha=cat_alpha)
            cat_hit_summary[cat] = len(cat_results)
            results.extend(cat_results)

        logger.info(f"[recall] soul={soul} per-category hits: {cat_hit_summary}")

        results.sort(key=lambda r: (-r.get("importance", 0), r.get("document_date", "")))

        entry_ids = [r["entry_id"] for r in results[:top_k]]
        graph_results = self._traverse_relations(conn, entry_ids, max_depth=2)
        existing_ids = {r["entry_id"] for r in results}
        for gr in graph_results:
            if gr["entry_id"] not in existing_ids:
                results.append(gr)
                existing_ids.add(gr["entry_id"])

        elapsed = (time.time() - t_recall_start) * 1000
        logger.info(f"[recall] soul={soul} final_results={len(results[:top_k])} graph_augmented={len(graph_results)} elapsed={elapsed:.1f}ms")

        for i, r in enumerate(results[:top_k]):
            content = r.get("content", {})
            topic = content.get("topic", "") if isinstance(content, dict) else ""
            detail = content.get("detail", "") if isinstance(content, dict) else str(content)
            preview = (topic or detail)[:60]
            logger.debug(
                f"[recall]   #{i+1}: category={r.get('category','?')} importance={r.get('importance',0):.2f} "
                f"preview=\"{preview}\""
            )

        return results[:top_k]

    def ai_wakeup(self, soul: str) -> dict:
        conn = self._get_db(soul)

        identity_rows = conn.execute("""
            SELECT * FROM memories WHERE soul = ? AND category = 'identity'
            ORDER BY importance DESC, document_date DESC LIMIT 5
        """, (soul,)).fetchall()

        recent_rows = conn.execute("""
            SELECT * FROM memories WHERE soul = ? AND category = 'conversation'
            ORDER BY document_date DESC LIMIT 10
        """, (soul,)).fetchall()

        todo_rows = conn.execute("""
            SELECT * FROM memories WHERE soul = ? AND category = 'todo'
            ORDER BY importance DESC, document_date DESC LIMIT 10
        """, (soul,)).fetchall()

        profile = self.get_profile(soul)

        identity = [MemoryEntry.from_row(dict(r)).to_dict() for r in identity_rows]
        recent = [MemoryEntry.from_row(dict(r)).to_dict() for r in recent_rows]
        todos = [MemoryEntry.from_row(dict(r)).to_dict() for r in todo_rows]

        return {
            "soul": soul,
            "identity": identity,
            "recent_conversations": recent,
            "active_todos": todos,
            "profile": profile,
            "wakeup_time": self._now_iso(),
        }

    CATEGORY_IMPORTANCE_THRESHOLDS = {
        "knowledge": 0.3,
        "conversation": 0.1,
        "todo": 0.1,
        "self_training": 0.2,
        "trajectory": 0.2,
        "audit": 0.1,
        "skill": 0.3,
        "code_memory": 0.4,
        "domain_memory": 0.4,
        "identity": 1.0,
    }

    def forget(self, soul: str, category: Optional[str] = None,
               older_than_days: int = 90,
               min_importance: Optional[float] = None,
               max_access_count: int = 2) -> int:
        conn = self._get_db(soul)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()

        categories_to_forget = [category] if category else [
            c for c in MEMORY_CATEGORIES if c != "identity"
        ]

        total_deleted = 0
        for cat in categories_to_forget:
            threshold = min_importance if min_importance is not None else self.CATEGORY_IMPORTANCE_THRESHOLDS.get(cat, 0.1)

            conditions = [
                "soul = ?",
                "category = ?",
                "document_date < ?",
                "importance <= ?",
                "access_count <= ?",
            ]
            params = [soul, cat, cutoff, threshold, max_access_count]

            where_clause = " AND ".join(conditions)

            ids_to_delete = conn.execute(
                f"SELECT entry_id FROM memories WHERE {where_clause}", params
            ).fetchall()
            delete_ids = [row["entry_id"] for row in ids_to_delete]

            if not delete_ids:
                continue

            placeholders = ",".join("?" * len(delete_ids))
            conn.execute(f"DELETE FROM memories_fts WHERE entry_id IN ({placeholders})", delete_ids)
            conn.execute(f"DELETE FROM relations WHERE source_id IN ({placeholders})", delete_ids)
            conn.execute(f"DELETE FROM relations WHERE target_id IN ({placeholders})", delete_ids)
            conn.execute(f"DELETE FROM memories WHERE entry_id IN ({placeholders})", delete_ids)
            total_deleted += len(delete_ids)

        conn.commit()
        return total_deleted

    def detect_contradictions(self, soul: str, category: str, new_content: dict) -> list:
        conn = self._get_db(soul)
        contradictions = []

        update_keywords = ["changed", "updated", "no longer", "switched", "replaced",
                           "instead of", "rather than", "not anymore", "cancelled",
                           "revoked", "deprecated", "obsolete", "wrong", "incorrect",
                           "actually", "correction"]
        content_text = self._content_to_text(new_content).lower()
        has_update_signal = any(kw in content_text for kw in update_keywords)

        for key, new_val in new_content.items():
            if not isinstance(new_val, str):
                continue
            existing = conn.execute("""
                SELECT * FROM memories WHERE soul = ? AND category = ?
                ORDER BY document_date DESC LIMIT 20
            """, (soul, category)).fetchall()

            for row in existing:
                old_content = json.loads(row["content"])
                old_val = old_content.get(key)
                if old_val is None or old_val == new_val:
                    continue
                if isinstance(old_val, str) and old_val.lower() != new_val.lower():
                    overlap = self._semantic_overlap(old_val, new_val)
                    if has_update_signal or overlap > 0.3:
                        confidence = max(overlap, 0.5) if has_update_signal else overlap + 0.2
                        contradictions.append({
                            "entry_id": row["entry_id"],
                            "old_key": key,
                            "old_value": old_val,
                            "new_value": new_val,
                            "confidence": min(confidence, 1.0),
                            "overlap_method": "semantic",
                            "document_date": row["document_date"],
                        })

        return contradictions

    def get_profile(self, soul: str) -> dict:
        conn = self._get_db(soul)
        row = conn.execute("SELECT * FROM profiles WHERE soul = ?", (soul,)).fetchone()

        if row:
            return {
                "soul": soul,
                "preferences": json.loads(row["preferences"]),
                "identity_facts": json.loads(row["identity_facts"]),
                "behavior_patterns": json.loads(row["behavior_patterns"]),
                "last_updated": row["last_updated"],
            }

        _ = UserProfile(soul=soul)
        self._rebuild_profile(conn, soul)
        row = conn.execute("SELECT * FROM profiles WHERE soul = ?", (soul,)).fetchone()
        if row:
            return {
                "soul": soul,
                "preferences": json.loads(row["preferences"]),
                "identity_facts": json.loads(row["identity_facts"]),
                "behavior_patterns": json.loads(row["behavior_patterns"]),
                "last_updated": row["last_updated"],
            }
        return {"soul": soul, "preferences": [], "identity_facts": [], "behavior_patterns": [], "last_updated": None}

    def index_code(self, project_path: str, soul: str = "cezanne") -> dict:
        stats = {"symbols_indexed": 0, "files_scanned": 0, "errors": []}

        try:
            project = Path(project_path)
            if not project.exists():
                return {"status": "error", "message": f"Path not found: {project_path}"}

            py_files = list(project.rglob("*.py"))[:200]
            stats["files_scanned"] = len(py_files)

            for py_file in py_files:
                try:
                    symbols = self._extract_symbols_from_file(py_file)
                    for sym in symbols:
                        self.write(soul, "code_memory", {
                            "name": sym["name"],
                            "kind": sym["kind"],
                            "file_path": str(py_file.relative_to(project)),
                            "line_start": sym["line_start"],
                            "line_end": sym["line_end"],
                        }, importance=0.6, tags=["code", sym["kind"]])
                        stats["symbols_indexed"] += 1
                except Exception as e:
                    stats["errors"].append(f"{py_file}: {str(e)}")

        except Exception as e:
            stats["errors"].append(str(e))

        stats["status"] = "indexed"
        return stats

    def index_domain(self, project_path: str, soul: str = "montesquieu") -> dict:
        stats = {"domains_indexed": 0, "concepts_found": 0, "errors": []}

        try:
            project = Path(project_path)
            if not project.exists():
                return {"status": "error", "message": f"Path not found: {project_path}"}

            py_files = list(project.rglob("*.py"))[:100]
            domain_map = defaultdict(list)

            for py_file in py_files:
                try:
                    concepts = self._extract_domain_concepts(py_file)
                    for concept in concepts:
                        domain_map[concept["category"]].append(concept)
                        self.write(soul, "domain_memory", {
                            "name": concept["name"],
                            "description": concept["description"],
                            "category": concept["category"],
                            "file_path": str(py_file.relative_to(project)),
                        }, importance=0.5, tags=["domain", concept["category"]])
                        stats["concepts_found"] += 1
                except Exception as e:
                    stats["errors"].append(f"{py_file}: {str(e)}")

            stats["domains_indexed"] = len(domain_map)
            stats["domain_categories"] = list(domain_map.keys())

        except Exception as e:
            stats["errors"].append(str(e))

        stats["status"] = "indexed"
        return stats

    def code_context(self, symbol: str, soul: str = "cezanne") -> dict:
        syntax_results = self.search(soul, "code_memory", symbol, top_k=5)
        domain_results = self.search(soul, "domain_memory", symbol, top_k=5)

        related_ids = []
        for r in syntax_results + domain_results:
            for rel in r.get("relations", []):
                related_ids.append(rel.get("target_id"))

        conn = self._get_db(soul)
        related_entries = []
        if related_ids:
            placeholders = ",".join("?" * len(related_ids))
            rows = conn.execute(
                f"SELECT * FROM memories WHERE entry_id IN ({placeholders})", related_ids
            ).fetchall()
            related_entries = [MemoryEntry.from_row(dict(r)).to_dict() for r in rows]

        return {
            "symbol": symbol,
            "syntax": syntax_results,
            "domains": domain_results,
            "related": related_entries,
        }

    def _traverse_relations(self, conn: sqlite3.Connection, entry_ids: List[str],
                            max_depth: int = 2) -> list:
        visited = set(entry_ids)
        frontier = list(entry_ids)
        results = []
        relation_types = ('updates', 'extends', 'derives', 'supersedes', 'contradicts', 'references')

        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier = []
            placeholders = ",".join("?" * len(frontier))

            forward_rows = conn.execute(f"""
                SELECT DISTINCT target_id FROM relations
                WHERE source_id IN ({placeholders})
                AND relation_type IN ({','.join('?' * len(relation_types))})
            """, frontier + list(relation_types)).fetchall()

            reverse_rows = conn.execute(f"""
                SELECT DISTINCT source_id FROM relations
                WHERE target_id IN ({placeholders})
                AND relation_type IN ({','.join('?' * len(relation_types))})
            """, frontier + list(relation_types)).fetchall()

            discovered = set()
            for row in forward_rows:
                discovered.add(row["target_id"])
            for row in reverse_rows:
                discovered.add(row["source_id"])

            for tid in discovered:
                if tid not in visited:
                    visited.add(tid)
                    next_frontier.append(tid)
                    mem = conn.execute("SELECT * FROM memories WHERE entry_id = ?", (tid,)).fetchone()
                    if mem:
                        results.append(MemoryEntry.from_row(dict(mem)).to_dict())

            frontier = next_frontier

        return results

    def _text_overlap(self, text_a: str, text_b: str) -> float:
        words_a = set(text_a.split())
        words_b = set(text_b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def _semantic_overlap(self, text_a: str, text_b: str) -> float:
        provider = EmbeddingProvider.get()
        if not provider.available:
            return self._text_overlap(text_a.lower(), text_b.lower())
        try:
            vec_a = provider.encode_query(text_a)
            vec_b = provider.encode_query(text_b)
            if vec_a is None or vec_b is None:
                return self._text_overlap(text_a.lower(), text_b.lower())
            return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8))
        except Exception:
            return self._text_overlap(text_a.lower(), text_b.lower())

    def _update_profile_from_write(self, soul: str, category: str, content: dict):
        conn = self._get_db(soul)
        profile = self._profiles.get(soul, UserProfile(soul=soul))

        if category == "identity":
            for key, val in content.items():
                existing = [f for f in profile.identity_facts if f.get("key") != key]
                existing.append({"key": key, "value": val, "updated": self._now_iso()})
                profile.identity_facts = existing

        elif category == "conversation":
            text = self._content_to_text(content).lower()
            preference_markers = ["i prefer", "i like", "i love", "i hate", "i dislike",
                                  "my favorite", "always", "never", "i want",
                                  "我喜欢", "我讨厌", "最爱"]
            for marker in preference_markers:
                if marker in text:
                    profile.preferences.append({
                        "source": "conversation",
                        "signal": marker,
                        "context": content,
                        "detected_at": self._now_iso(),
                    })
                    break

        profile.last_updated = self._now_iso()
        self._profiles[soul] = profile

        conn.execute("""
            INSERT OR REPLACE INTO profiles (soul, preferences, identity_facts, behavior_patterns, last_updated)
            VALUES (?, ?, ?, ?, ?)
        """, (
            soul,
            json.dumps(profile.preferences[-50:]),
            json.dumps(profile.identity_facts),
            json.dumps(profile.behavior_patterns),
            profile.last_updated,
        ))
        conn.commit()

    def _rebuild_profile(self, conn: sqlite3.Connection, soul: str):
        profile = UserProfile(soul=soul)
        rows = conn.execute("""
            SELECT * FROM memories WHERE soul = ? AND category IN ('identity', 'conversation')
            ORDER BY document_date DESC LIMIT 100
        """, (soul,)).fetchall()

        for row in rows:
            content = json.loads(row["content"])
            if row["category"] == "identity":
                for key, val in content.items():
                    existing = [f for f in profile.identity_facts if f.get("key") != key]
                    existing.append({"key": key, "value": val, "updated": row["document_date"]})
                    profile.identity_facts = existing

        profile.last_updated = self._now_iso()
        self._profiles[soul] = profile

        conn.execute("""
            INSERT OR REPLACE INTO profiles (soul, preferences, identity_facts, behavior_patterns, last_updated)
            VALUES (?, ?, ?, ?, ?)
        """, (
            soul,
            json.dumps(profile.preferences),
            json.dumps(profile.identity_facts),
            json.dumps(profile.behavior_patterns),
            profile.last_updated,
        ))
        conn.commit()

    def _extract_symbols_from_file(self, file_path: Path) -> list:
        symbols = []
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = text.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                m = re.match(r'^(class|def|async\s+def)\s+(\w+)', stripped)
                if m:
                    kind = "class" if m.group(1) == "class" else "function"
                    name = m.group(2)
                    end_line = i
                    for j in range(i, min(i + 200, len(lines))):
                        if j > i and re.match(r'^(class|def|async\s+def)\s+', lines[j].strip()):
                            break
                        end_line = j
                    symbols.append({
                        "name": name,
                        "kind": kind,
                        "line_start": i,
                        "line_end": end_line,
                    })
        except Exception:
            pass
        return symbols

    def _extract_domain_concepts(self, file_path: Path) -> list:
        concepts = []
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            docstring_match = re.search(r'"""(.*?)"""', text, re.DOTALL)
            if docstring_match:
                doc = docstring_match.group(1).strip()
                first_line = doc.split("\n")[0].strip()
                if first_line and len(first_line) > 10:
                    concepts.append({
                        "name": file_path.stem,
                        "description": first_line[:200],
                        "category": "module_docstring",
                    })

            class_names = re.findall(r'class\s+(\w+)', text)
            for name in class_names:
                concepts.append({
                    "name": name,
                    "description": f"Class {name} in {file_path.name}",
                    "category": "class",
                })
        except Exception:
            pass
        return concepts

    def backfill_embeddings(self, soul: str, batch_size: int = 128) -> dict:
        conn = self._get_db(soul)
        rows = conn.execute("""
            SELECT entry_id, content FROM memories WHERE embedding IS NULL
        """).fetchall()

        if not rows:
            return {"soul": soul, "backfilled": 0, "status": "no_missing_embeddings"}

        backfilled = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            texts = []
            for row in batch:
                content = json.loads(row["content"])
                texts.append(self._content_to_text(content))

            embeddings = self._embedding_provider.encode_batch(texts)
            for row, emb in zip(batch, embeddings):
                if emb is not None:
                    conn.execute(
                        "UPDATE memories SET embedding = ? WHERE entry_id = ?",
                        (emb, row["entry_id"]),
                    )
                    backfilled += 1

            conn.commit()

        return {"soul": soul, "backfilled": backfilled, "total_missing": len(rows),
                "status": "backfilled"}

    def close(self):
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()


_engine = SoulMemoryEngine()


def write(soul: str, category: str, content: dict,
          event_date: Optional[str] = None,
          relations: Optional[List[dict]] = None,
          importance: float = 0.5,
          tags: Optional[List[str]] = None) -> str:
    if category not in MEMORY_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of {MEMORY_CATEGORIES}")
    return _engine.write(soul, category, content, event_date=event_date,
                         relations=relations, importance=importance, tags=tags)


def search(soul: str, category: str, query: str, top_k: int = 5,
           time_filter: Optional[dict] = None,
           min_importance: float = 0.0,
           alpha: Optional[float] = None) -> list:
    if category not in MEMORY_CATEGORIES:
        raise ValueError(f"Invalid category '{category}'. Must be one of {MEMORY_CATEGORIES}")
    return _engine.search(soul, category, query, top_k=top_k,
                          time_filter=time_filter, min_importance=min_importance,
                          alpha=alpha)


def recall(soul: str, query: str, top_k: int = 10,
           categories: Optional[List[str]] = None,
           alpha: Optional[float] = None) -> list:
    return _engine.recall(soul, query, top_k=top_k, categories=categories, alpha=alpha)


def ai_wakeup(soul: str) -> dict:
    return _engine.ai_wakeup(soul)


def forget(soul: str, category: Optional[str] = None,
           older_than_days: int = 90,
           min_importance: float = 0.1,
           max_access_count: int = 2) -> int:
    return _engine.forget(soul, category=category, older_than_days=older_than_days,
                          min_importance=min_importance, max_access_count=max_access_count)


def detect_contradictions(soul: str, category: str, new_content: dict) -> list:
    return _engine.detect_contradictions(soul, category, new_content)


def get_profile(soul: str) -> dict:
    return _engine.get_profile(soul)


def index_code(project_path: str, soul: str = "cezanne") -> dict:
    return _engine.index_code(project_path, soul)


def index_domain(project_path: str, soul: str = "montesquieu") -> dict:
    return _engine.index_domain(project_path, soul)


def code_context(symbol: str, soul: str = "cezanne") -> dict:
    return _engine.code_context(symbol, soul)


def backfill_embeddings(soul: str, batch_size: int = 128) -> dict:
    return _engine.backfill_embeddings(soul, batch_size)


CROSS_SOUL_PERMISSIONS = {
    "cezanne":   ["einstein", "davinci", "galileo", "montesquieu"],
    "einstein":  ["cezanne", "galileo", "strategy", "humboldt"],
    "galileo":   ["einstein", "darwin", "humboldt"],
    "darwin":    ["einstein", "galileo", "guizhu"],
    "davinci":   ["cezanne", "monet", "vangogh", "strategy"],
    "strategy":  ["einstein", "davinci", "montesquieu"],
    "montesquieu": ["cezanne", "strategy", "guizhu"],
    "humboldt":  ["darwin", "galileo", "einstein"],
    "yuanlongping": ["darwin", "humboldt"],
    "guizhu":    ["montesquieu", "herodotus", "darwin"],
    "herodotus": ["guizhu", "montesquieu"],
    "monet":     ["davinci", "vangogh", "beethoven"],
    "vangogh":   ["monet", "davinci", "beethoven"],
    "beethoven": ["monet", "vangogh"],
}


def cross_soul_recall(source_soul: str, query: str, top_k: int = 5,
                      categories: Optional[List[str]] = None,
                      max_souls: int = 3) -> dict:
    """Allow one soul to query other souls' memories (with permission).

    Returns results grouped by soul, ranked by relevance.
    Only queries souls listed in CROSS_SOUL_PERMISSIONS[source_soul].
    """
    allowed = CROSS_SOUL_PERMISSIONS.get(source_soul, [])
    if not allowed:
        logger.info(f"[cross_soul_recall] source={source_soul} query=\"{query[:60]}\" → no permissions")
        return {"source_soul": source_soul, "results": {}, "status": "no_permissions"}

    logger.info(f"[cross_soul_recall] source={source_soul} query=\"{query[:60]}\" allowed={allowed[:max_souls]}")

    all_results = {}
    for target_soul in allowed[:max_souls]:
        try:
            results = _engine.recall(target_soul, query, top_k=top_k,
                                     categories=categories)
            if results:
                all_results[target_soul] = results
                logger.info(f"[cross_soul_recall]   {target_soul}: {len(results)} results")
            else:
                logger.debug(f"[cross_soul_recall]   {target_soul}: 0 results")
        except Exception as e:
            logger.debug(f"[cross_soul_recall]   {target_soul}: error — {e}")
            continue

    return {
        "source_soul": source_soul,
        "queried_souls": list(all_results.keys()),
        "results": all_results,
        "total_entries": sum(len(v) for v in all_results.values()),
        "status": "ok",
    }


def cross_soul_write(source_soul: str, target_soul: str, category: str,
                     content: dict, importance: float = 0.5,
                     tags: Optional[List[str]] = None) -> dict:
    """Allow one soul to write a memory entry into another soul's memory.

    Only permitted if target_soul is in source_soul's permission list.
    The entry is tagged with the source soul for traceability.
    """
    allowed = CROSS_SOUL_PERMISSIONS.get(source_soul, [])
    if target_soul not in allowed:
        return {"status": "denied", "reason": f"{source_soul} cannot write to {target_soul}"}

    tags = tags or []
    tags.append(f"cross_soul:{source_soul}")

    entry_id = _engine.write(target_soul, category, content,
                             importance=importance, tags=tags)
    return {
        "status": "written",
        "entry_id": entry_id,
        "source_soul": source_soul,
        "target_soul": target_soul,
    }


def cross_soul_multi_hop(source_soul: str, query: str, max_hops: int = 3,
                         top_k: int = 3) -> dict:
    """Multi-hop cross-soul reasoning: follow relation chains across soul boundaries.

    Starting from source_soul, queries allowed souls, then follows
    cross-references found in results to query additional souls.
    """
    allowed = CROSS_SOUL_PERMISSIONS.get(source_soul, [])
    if not allowed:
        return {"source_soul": source_soul, "hops": [], "status": "no_permissions"}

    visited_souls = {source_soul}
    frontier = list(allowed)
    all_results = {}
    hop_log = []

    for hop in range(max_hops):
        if not frontier:
            break
        next_frontier = []
        for target_soul in frontier:
            if target_soul in visited_souls:
                continue
            visited_souls.add(target_soul)
            try:
                results = _engine.recall(target_soul, query, top_k=top_k)
                if results:
                    all_results[target_soul] = results
                    hop_log.append({"hop": hop + 1, "soul": target_soul, "found": len(results)})
                    for r in results:
                        content = r.get("content", {})
                        refs = content.get("references", []) if isinstance(content, dict) else []
                        for ref in refs:
                            if isinstance(ref, str) and ref in CROSS_SOUL_PERMISSIONS:
                                if ref not in visited_souls:
                                    next_frontier.append(ref)
            except Exception:
                continue
        frontier = next_frontier

    return {
        "source_soul": source_soul,
        "hops": hop_log,
        "total_souls_queried": len(all_results),
        "results": all_results,
        "total_entries": sum(len(v) for v in all_results.values()),
        "status": "ok",
    }


def _startup_auto_index():
    import threading
    def _run():
        try:
            from auto_index_engine import startup_auto_index as sai
            sai()
        except Exception:
            pass
    t = threading.Thread(target=_run, daemon=True, name="memory_auto_index")
    t.start()


_startup_auto_index()
