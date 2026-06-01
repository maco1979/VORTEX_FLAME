"""
World-Embedding Cache — Offline Pre-encoded Causal Representations
==================================================================
Stores and retrieves C-JEPA World-Embeddings for knowledge base entries.

In the dual-pathway architecture:
  Path A (RAG): text → text embedding → vector DB (existing soul_memory)
  Path B (C-JEPA): text → causal structure → C-JEPA encode → World-Embedding → this cache

This cache provides:
  1. Offline pre-encoding: encode entire KB once, cache World-Embeddings
  2. Fast retrieval: query → semantic match → return World-Embedding vectors
  3. Incremental update: add new entries without re-encoding everything
  4. Version tracking: track which C-JEPA model version produced each embedding

Storage: SQLite + numpy binary blobs (same pattern as soul_memory)
No GPU required for retrieval. Encoding requires GPU (deferred).
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

WORLD_EMBEDDING_DIM = 128
CACHE_DIR = Path(os.environ.get("VORTEX_FLAME_CACHE_DIR", ".vf_world_cache"))


@dataclass
class WorldEmbeddingEntry:
    entry_id: str
    source_id: str
    soul: str
    category: str
    objects: List[str]
    causal_summary: str
    embedding: Optional[np.ndarray] = None
    embedding_dim: int = WORLD_EMBEDDING_DIM
    model_version: str = "placeholder_v0"
    source_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    access_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["embedding"] = self.embedding.tolist() if self.embedding is not None else None
        return d


class WorldEmbeddingCache:
    """
    SQLite-backed cache for C-JEPA World-Embeddings.

    Two-phase operation:
      Phase 1 (CPU, now): Store causal structure + placeholder embeddings
      Phase 2 (GPU, later): Replace placeholders with actual C-JEPA encoded embeddings

    The cache is designed so that:
    - Structure extraction and indexing happens immediately (CPU)
    - Actual embedding computation is deferred until GPU is available
    - Queries work with both placeholder and real embeddings
    - When real embeddings become available, they replace placeholders in-place
    """

    def __init__(self, cache_dir: Optional[str] = None, embedding_dim: int = WORLD_EMBEDDING_DIM):
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._connections: Dict[str, sqlite3.Connection] = {}
        self._write_locks: Dict[str, threading.Lock] = {}
        self._embedding_provider = None
        self._init_embedding_provider()

    def _init_embedding_provider(self):
        try:
            from soul_memory import EmbeddingProvider
            self._embedding_provider = EmbeddingProvider.get()
        except Exception:
            logger.warning("Embedding provider unavailable, using random projections for retrieval")

    def _get_db(self, soul: str) -> sqlite3.Connection:
        if soul not in self._connections:
            db_path = self.cache_dir / f"world_emb_{soul}.db"
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._init_schema(conn)
            self._connections[soul] = conn
        return self._connections[soul]

    def _get_lock(self, soul: str) -> threading.Lock:
        if soul not in self._write_locks:
            self._write_locks[soul] = threading.Lock()
        return self._write_locks[soul]

    def _init_schema(self, conn: sqlite3.Connection):
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS world_embeddings (
                entry_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                soul TEXT NOT NULL,
                category TEXT NOT NULL,
                objects TEXT NOT NULL,
                causal_summary TEXT NOT NULL DEFAULT '',
                embedding BLOB,
                embedding_dim INTEGER NOT NULL DEFAULT 128,
                model_version TEXT NOT NULL DEFAULT 'placeholder_v0',
                source_text TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT '',
                access_count INTEGER NOT NULL DEFAULT 0,
                is_real_embedding INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_source_id ON world_embeddings(source_id);
            CREATE INDEX IF NOT EXISTS idx_category ON world_embeddings(category);
            CREATE INDEX IF NOT EXISTS idx_model_version ON world_embeddings(model_version);
            CREATE INDEX IF NOT EXISTS idx_is_real ON world_embeddings(is_real_embedding);

            CREATE VIRTUAL TABLE IF NOT EXISTS world_emb_fts USING fts5(
                entry_id,
                objects,
                causal_summary,
                source_text,
                content=world_embeddings,
                content_rowid=rowid
            );

            CREATE TRIGGER IF NOT EXISTS world_emb_ai AFTER INSERT ON world_embeddings BEGIN
                INSERT INTO world_emb_fts(rowid, entry_id, objects, causal_summary, source_text)
                VALUES (new.rowid, new.entry_id, new.objects, new.causal_summary, new.source_text);
            END;

            CREATE TRIGGER IF NOT EXISTS world_emb_ad AFTER DELETE ON world_embeddings BEGIN
                INSERT INTO world_emb_fts(world_emb_fts, rowid, entry_id, objects, causal_summary, source_text)
                VALUES ('delete', old.rowid, old.entry_id, old.objects, old.causal_summary, old.source_text);
            END;
        """)
        conn.commit()

    def store(
        self,
        soul: str,
        source_id: str,
        category: str,
        objects: List[str],
        causal_summary: str,
        embedding: Optional[np.ndarray] = None,
        model_version: str = "placeholder_v0",
        source_text: str = "",
        metadata: Optional[Dict] = None,
    ) -> str:
        entry_id = hashlib.md5(
            f"{soul}:{source_id}:{category}:{':'.join(objects)}".encode()
        ).hexdigest()[:16]

        if embedding is None:
            embedding = self._generate_placeholder_embedding(objects, causal_summary)

        is_real = 1 if model_version != "placeholder_v0" else 0
        emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None

        conn = self._get_db(soul)
        lock = self._get_lock(soul)

        with lock:
            conn.execute("""
                INSERT OR REPLACE INTO world_embeddings
                (entry_id, source_id, soul, category, objects, causal_summary,
                 embedding, embedding_dim, model_version, source_text, metadata,
                 created_at, access_count, is_real_embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """, (
                entry_id, source_id, soul, category,
                json.dumps(objects, ensure_ascii=False),
                causal_summary,
                emb_blob,
                self.embedding_dim,
                model_version,
                source_text[:2000],
                json.dumps(metadata or {}, ensure_ascii=False),
                time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                is_real,
            ))
            conn.commit()

        return entry_id

    def store_from_object_graph(self, soul: str, graph_dict: Dict[str, Any], category: str = "knowledge") -> str:
        objects = graph_dict.get("objects", [])
        causal_chains = graph_dict.get("causal_chains", [])
        causal_summary = "; ".join(
            f"{c.get('cause', '')}→{c.get('effect', '')}" for c in causal_chains
        ) if causal_chains else ""
        source_id = graph_dict.get("graph_id", "unknown")
        source_text = graph_dict.get("source_text", "")
        metadata = {
            "num_entities": len(graph_dict.get("entities", [])),
            "num_relations": len(graph_dict.get("relations", [])),
            "num_causal_chains": len(causal_chains),
            "num_temporal_steps": len(graph_dict.get("temporal_sequence", [])),
        }
        return self.store(
            soul=soul,
            source_id=source_id,
            category=category,
            objects=objects,
            causal_summary=causal_summary,
            source_text=source_text,
            metadata=metadata,
        )

    def search(
        self,
        soul: str,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        require_real_embedding: bool = False,
    ) -> List[WorldEmbeddingEntry]:
        conn = self._get_db(soul)

        fts_query = query.replace('"', '""')
        sql = """
            SELECT we.* FROM world_embeddings we
            WHERE we.entry_id IN (
                SELECT entry_id FROM world_emb_fts WHERE world_emb_fts MATCH ?
            )
        """
        params: list = [fts_query]

        if category:
            sql += " AND we.category = ?"
            params.append(category)
        if require_real_embedding:
            sql += " AND we.is_real_embedding = 1"

        sql += " ORDER BY we.access_count DESC LIMIT ?"
        params.append(top_k)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            rows = conn.execute(
                "SELECT * FROM world_embeddings WHERE objects LIKE ? LIMIT ?",
                (f"%{query}%", top_k),
            ).fetchall()

        results = []
        for row in rows:
            entry = self._row_to_entry(row)
            results.append(entry)
            self._increment_access(conn, entry.entry_id)

        if len(results) < top_k and self._embedding_provider and self._embedding_provider.available:
            semantic_results = self._semantic_search(soul, query, top_k - len(results), category)
            existing_ids = {r.entry_id for r in results}
            for r in semantic_results:
                if r.entry_id not in existing_ids:
                    results.append(r)

        return results

    def search_by_objects(
        self,
        soul: str,
        objects: List[str],
        top_k: int = 5,
    ) -> List[WorldEmbeddingEntry]:
        conn = self._get_db(soul)
        results = []
        for obj in objects:
            rows = conn.execute(
                "SELECT * FROM world_embeddings WHERE objects LIKE ? LIMIT ?",
                (f'%"{obj}"%', top_k),
            ).fetchall()
            for row in rows:
                entry = self._row_to_entry(row)
                if not any(r.entry_id == entry.entry_id for r in results):
                    results.append(entry)

        return results[:top_k]

    def get_embedding(self, soul: str, entry_id: str) -> Optional[np.ndarray]:
        conn = self._get_db(soul)
        row = conn.execute(
            "SELECT embedding, embedding_dim FROM world_embeddings WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
        if row is None or row["embedding"] is None:
            return None
        return np.frombuffer(row["embedding"], dtype=np.float32)

    def get_embeddings_batch(self, soul: str, entry_ids: List[str]) -> Dict[str, np.ndarray]:
        conn = self._get_db(soul)
        placeholders = ",".join("?" * len(entry_ids))
        rows = conn.execute(
            f"SELECT entry_id, embedding, embedding_dim FROM world_embeddings WHERE entry_id IN ({placeholders})",
            entry_ids,
        ).fetchall()
        result = {}
        for row in rows:
            if row["embedding"] is not None:
                result[row["entry_id"]] = np.frombuffer(row["embedding"], dtype=np.float32)
        return result

    def upgrade_embedding(
        self,
        soul: str,
        entry_id: str,
        real_embedding: np.ndarray,
        model_version: str,
    ) -> bool:
        conn = self._get_db(soul)
        lock = self._get_lock(soul)
        emb_blob = real_embedding.astype(np.float32).tobytes()

        with lock:
            cursor = conn.execute("""
                UPDATE world_embeddings
                SET embedding = ?, model_version = ?, is_real_embedding = 1
                WHERE entry_id = ?
            """, (emb_blob, model_version, entry_id))
            conn.commit()

        return cursor.rowcount > 0

    def batch_upgrade_embeddings(
        self,
        soul: str,
        embeddings: Dict[str, np.ndarray],
        model_version: str,
    ) -> int:
        conn = self._get_db(soul)
        lock = self._get_lock(soul)
        upgraded = 0

        with lock:
            for entry_id, emb in embeddings.items():
                emb_blob = emb.astype(np.float32).tobytes()
                cursor = conn.execute("""
                    UPDATE world_embeddings
                    SET embedding = ?, model_version = ?, is_real_embedding = 1
                    WHERE entry_id = ?
                """, (emb_blob, model_version, entry_id))
                upgraded += cursor.rowcount
            conn.commit()

        return upgraded

    def get_placeholder_count(self, soul: str) -> int:
        conn = self._get_db(soul)
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM world_embeddings WHERE is_real_embedding = 0"
        ).fetchone()
        return row["cnt"] if row else 0

    def get_real_count(self, soul: str) -> int:
        conn = self._get_db(soul)
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM world_embeddings WHERE is_real_embedding = 1"
        ).fetchone()
        return row["cnt"] if row else 0

    def get_stats(self, soul: str) -> Dict[str, Any]:
        conn = self._get_db(soul)
        total = conn.execute("SELECT COUNT(*) as cnt FROM world_embeddings").fetchone()["cnt"]
        real = self.get_real_count(soul)
        placeholder = self.get_placeholder_count(soul)
        categories = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM world_embeddings GROUP BY category"
        ).fetchall()
        return {
            "total": total,
            "real_embeddings": real,
            "placeholder_embeddings": placeholder,
            "categories": {r["category"]: r["cnt"] for r in categories},
            "soul": soul,
        }

    def _generate_placeholder_embedding(self, objects: List[str], causal_summary: str) -> np.ndarray:
        seed_text = " ".join(objects) + " " + causal_summary
        seed_hash = hashlib.md5(seed_text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(seed_hash[:4], "little"))
        emb = rng.randn(self.embedding_dim).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def _semantic_search(
        self,
        soul: str,
        query: str,
        top_k: int,
        category: Optional[str] = None,
    ) -> List[WorldEmbeddingEntry]:
        if not self._embedding_provider or not self._embedding_provider.available:
            return []

        query_vec = self._embedding_provider.encode_query(query)
        if query_vec is None:
            return []

        conn = self._get_db(soul)
        sql = "SELECT * FROM world_embeddings"
        if category:
            sql += f" WHERE category = '{category}'"
        rows = conn.execute(sql).fetchall()

        scored = []
        for row in rows:
            if row["embedding"] is not None:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                sim = float(np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb) + 1e-8))
                scored.append((sim, row))

        scored.sort(key=lambda x: -x[0])
        results = []
        for sim, row in scored[:top_k]:
            entry = self._row_to_entry(row)
            entry.metadata["search_score"] = sim
            results.append(entry)

        return results

    def _row_to_entry(self, row: sqlite3.Row) -> WorldEmbeddingEntry:
        embedding = None
        if row["embedding"] is not None:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()

        return WorldEmbeddingEntry(
            entry_id=row["entry_id"],
            source_id=row["source_id"],
            soul=row["soul"],
            category=row["category"],
            objects=json.loads(row["objects"]),
            causal_summary=row["causal_summary"],
            embedding=embedding,
            embedding_dim=row["embedding_dim"],
            model_version=row["model_version"],
            source_text=row["source_text"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            access_count=row["access_count"],
        )

    def _increment_access(self, conn: sqlite3.Connection, entry_id: str):
        try:
            conn.execute(
                "UPDATE world_embeddings SET access_count = access_count + 1 WHERE entry_id = ?",
                (entry_id,),
            )
        except Exception:
            pass


def batch_index_knowledge_to_world_cache(
    kb_dirs: Optional[List[Tuple[str, str]]] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function: index all KB directories into World-Embedding cache.

    Args:
        kb_dirs: List of (directory_path, soul) tuples
        cache_dir: Override cache directory

    Returns:
        Aggregated stats
    """
    from causal_knowledge_extractor import CausalKnowledgeExtractor

    if kb_dirs is None:
        kb_dirs = [
            ("D:/VORTEX_FLAME/kb_harness", "cezanne"),
            ("D:/VORTEX_FLAME/kb_mcp", "guizhu"),
            ("D:/VORTEX_FLAME/kb_skill", "beethoven"),
        ]

    extractor = CausalKnowledgeExtractor(use_embedding=True)
    cache = WorldEmbeddingCache(cache_dir)

    total_stats = {
        "total_graphs": 0,
        "total_entries": 0,
        "total_placeholder": 0,
        "per_soul": {},
    }

    for dirpath, soul in kb_dirs:
        if not os.path.exists(dirpath):
            continue

        graphs = extractor.extract_from_directory(dirpath, max_files=500)
        soul_stats = {"graphs": len(graphs), "stored": 0, "errors": 0}

        for graph in graphs:
            try:
                graph_dict = graph.to_dict()
                entry_id = cache.store_from_object_graph(soul, graph_dict)
                soul_stats["stored"] += 1
            except Exception as e:
                logger.warning(f"Failed to store graph: {e}")
                soul_stats["errors"] += 1

        total_stats["total_graphs"] += len(graphs)
        total_stats["per_soul"][soul] = soul_stats

    for soul in set(s for _, s in kb_dirs):
        stats = cache.get_stats(soul)
        total_stats["total_entries"] += stats["total"]
        total_stats["total_placeholder"] += stats["placeholder_embeddings"]
        total_stats["per_soul"][soul]["cache_stats"] = stats

    return total_stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cache = WorldEmbeddingCache()

    entry_id = cache.store(
        soul="cezanne",
        source_id="test_device_fault",
        category="knowledge",
        objects=["传感器", "告警信号", "设备", "操作员"],
        causal_summary="传感器数据超限→触发告警→设备异常→操作员确认→停机",
        source_text="传感器采集数据后，如果数据超限，则触发告警信号。",
    )
    print(f"Stored entry: {entry_id}")

    results = cache.search("cezanne", "传感器告警", top_k=3)
    print(f"\nSearch results for '传感器告警': {len(results)}")
    for r in results:
        print(f"  {r.entry_id}: objects={r.objects}, causal={r.causal_summary[:60]}")

    stats = cache.get_stats("cezanne")
    print(f"\nCache stats: {stats}")

    print("\n=== Batch indexing KB directories ===")
    batch_stats = batch_index_knowledge_to_world_cache()
    print(f"Total graphs: {batch_stats['total_graphs']}")
    print(f"Total entries: {batch_stats['total_entries']}")
    for soul, s in batch_stats["per_soul"].items():
        print(f"  {soul}: graphs={s['graphs']}, stored={s['stored']}")
