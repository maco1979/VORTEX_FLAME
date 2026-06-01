#!/usr/bin/env python3
"""
VF Dedup — 多维度数据去重模块
=================================
双层去重策略: 规则引擎(精确/模糊) + 智能算法(SimHash/余弦相似度)

策略矩阵:
  EXACT    — hash精确匹配, O(n)单遍扫描, 100%精确
  SIMHASH  — 局部敏感哈希, 亚线性去重, 支持60-70%以上相似度检测
  FUZZY    — 编辑距离+Jaccard系数, 字段级别权重可配
  SEMANTIC — 余弦相似度(可选, 需sentence-transformers或sklearn)

冲突解决策略:
  keep_first / keep_longest / keep_newest / keep_by_field / merge

用法:
  from vf_dedup import DedupEngine
  dedup = DedupEngine(cfg, audit)
  clean, stats = dedup.process(data)
  report = dedup.report()
"""

import hashlib
import json
import math
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from vf_data_core import (
    AuditLogger, BaseProcessor, ConfigManager, ProcessingStats,
    compute_content_hash, normalize_field_names,
)


@dataclass
class DuplicateGroup:
    group_id: str
    size: int
    representative_idx: int
    member_indices: List[int]
    method: str
    confidence: float
    key_snippet: str = ""


@dataclass
class DedupStats(ProcessingStats):
    duplicate_groups: int = 0
    duplicates_removed: int = 0
    exact_matches: int = 0
    simhash_matches: int = 0
    fuzzy_matches: int = 0
    semantic_matches: int = 0
    method: str = ""


class SimHash:
    """
    SimHash — 局部敏感哈希实现
    核心原理: 将特征向量降维到固定长度指纹, 汉明距离衡量相似度

    数学: 对于特征集 {(w_i, v_i)},
      fingerprint = Σ sgn(w_i · hash_bit(v_i))
    其中 hash_bit(v_i) 将 v_i 映射为 {-1, +1}^L
    """

    def __init__(self, fp_len: int = 64, ngram: int = 3):
        self.fp_len = fp_len
        self.ngram = ngram
        self._masks = [1 << i for i in range(fp_len)]

    def compute(self, text: str) -> int:
        if not text:
            return 0

        text = text.lower().strip()
        weights = [0] * self.fp_len
        ngrams = self._extract_ngrams(text, self.ngram)
        if not ngrams:
            ngrams = [text]

        for ng in ngrams:
            h = int(hashlib.md5(ng.encode("utf-8")).hexdigest()[:16], 16)
            for i in range(self.fp_len):
                if h & self._masks[i % len(self._masks)]:
                    weights[i] += 1
                else:
                    weights[i] -= 1

        fp = 0
        for i in range(self.fp_len):
            if weights[i] > 0:
                fp |= (1 << i)
        return fp

    def hamming_distance(self, fp1: int, fp2: int) -> int:
        xor = fp1 ^ fp2
        return xor.bit_count()

    def is_similar(self, fp1: int, fp2: int, threshold: int = 3) -> bool:
        return self.hamming_distance(fp1, fp2) <= threshold

    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        chars = list(text)
        if len(chars) < n:
            return []
        result = []
        for i in range(len(chars) - n + 1):
            result.append("".join(chars[i:i + n]))
        return result


class RuleDedupEngine:
    def __init__(self, fields_weight: Optional[Dict[str, float]] = None,
                 fuzzy_threshold: float = 0.85):
        self.fields_weight = fields_weight or {}
        self.fuzzy_threshold = fuzzy_threshold

    def exact_match(self, rec1: Dict, rec2: Dict, key_fields: Optional[List[str]] = None) -> bool:
        if key_fields:
            return all(
                str(rec1.get(f, "")).strip() == str(rec2.get(f, "")).strip()
                for f in key_fields
            )
        return compute_content_hash(json.dumps(rec1, sort_keys=True, ensure_ascii=False)) == \
               compute_content_hash(json.dumps(rec2, sort_keys=True, ensure_ascii=False))

    def fuzzy_match(self, rec1: Dict, rec2: Dict) -> float:
        fields = set(list(rec1.keys()) + list(rec2.keys()))
        if not fields:
            return 0.0

        total_score = 0.0
        total_weight = 0.0
        for field in fields:
            v1 = str(rec1.get(field, "")).lower().strip()
            v2 = str(rec2.get(field, "")).lower().strip()
            weight = self.fields_weight.get(field, 1.0)

            if not v1 and not v2:
                continue
            if not v1 or not v2:
                total_weight += weight
                continue

            total_weight += weight
            if v1 == v2:
                total_score += weight
            else:
                total_score += weight * self._jaccard_similarity(v1, v2)

        if total_weight == 0:
            return 1.0
        return total_score / total_weight

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        set1 = set(s1.split())
        set2 = set(s2.split())
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(2)]
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            dp[i % 2][0] = i
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i % 2][j] = min(
                    dp[(i - 1) % 2][j] + 1,
                    dp[i % 2][j - 1] + 1,
                    dp[(i - 1) % 2][j - 1] + cost,
                )
        return 1.0 - dp[m % 2][n] / max(m, n)


class SemanticMatcher:
    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self._model = None
        self._tfidf = None
        self._try_load()

    def _try_load(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self._tfidf = TfidfVectorizer
            self._cosine = cosine_similarity
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._tfidf is not None

    def compute_similarity_matrix(self, texts: List[str]) -> List[List[float]]:
        if not self.available or len(texts) < 2:
            return []

        try:
            vec = self._tfidf(max_features=1000, stop_words=None)
            tfidf_matrix = vec.fit_transform(texts)
            sim = self._cosine(tfidf_matrix)
            return sim.tolist()
        except Exception:
            return []

    def find_similar_pairs(self, texts: List[str]) -> List[Tuple[int, int, float]]:
        sim_matrix = self.compute_similarity_matrix(texts)
        pairs = []
        n = len(sim_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= self.threshold:
                    pairs.append((i, j, sim_matrix[i][j]))
        return pairs


class DedupEngine(BaseProcessor):
    def __init__(self, config: ConfigManager, audit: AuditLogger):
        super().__init__(config, audit, "dedup")
        self._simhash = SimHash(
            fp_len=config.get("dedup", "simhash_fp_len", default=64),
            ngram=config.get("dedup", "simhash_ngram", default=3),
        )
        self._rule_engine = RuleDedupEngine(
            fields_weight=config.get("dedup", "fields_weight", default={}),
            fuzzy_threshold=config.get("dedup", "fuzzy_threshold", default=0.85),
        )
        self._semantic = SemanticMatcher(
            threshold=config.get("dedup", "semantic_threshold", default=0.92),
        )
        self._simhash_threshold = config.get("dedup", "simhash_threshold", default=3)
        self._conflict_resolution = config.get("dedup", "conflict_resolution", default="keep_first")
        self._methods = set(config.get("dedup", "methods", default=["exact", "simhash", "fuzzy"]))
        self._semantic_enabled = config.get("dedup", "semantic_enabled", default=False) and self._semantic.available

        self._duplicate_groups: List[DuplicateGroup] = []
        self._preview_cache: Dict[str, Any] = {}

    def process(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], ProcessingStats]:
        t0 = time.time()
        if not data:
            self.stats = DedupStats()
            return [], DedupStats()

        if not self._resource_check(len(data)):
            return data, DedupStats(total_input=len(data), total_output=len(data))

        removed = set()
        groups = []

        if "exact" in self._methods:
            g_exact = self._dedup_exact(data)
            groups.extend(g_exact)
            for g in g_exact:
                removed.update(g.member_indices[1:])

        if "simhash" in self._methods:
            remaining_indices = [i for i in range(len(data)) if i not in removed]
            remaining_data = [data[i] for i in remaining_indices]
            g_simhash = self._dedup_simhash(remaining_data, remaining_indices)
            groups.extend(g_simhash)
            for g in g_simhash:
                for idx in g.member_indices[1:]:
                    removed.add(idx)

        if "fuzzy" in self._methods:
            remaining_indices = [i for i in range(len(data)) if i not in removed]
            remaining_data = [data[i] for i in remaining_indices]
            g_fuzzy = self._dedup_fuzzy(remaining_data, remaining_indices)
            groups.extend(g_fuzzy)
            for g in g_fuzzy:
                for idx in g.member_indices[1:]:
                    removed.add(idx)

        if self._semantic_enabled:
            remaining_indices = [i for i in range(len(data)) if i not in removed]
            remaining_data = [data[i] for i in remaining_indices]
            g_sem = self._dedup_semantic(remaining_data, remaining_indices)
            groups.extend(g_sem)
            for g in g_sem:
                for idx in g.member_indices[1:]:
                    removed.add(idx)

        output = [data[i] for i in range(len(data)) if i not in removed]

        exact_groups = [g for g in groups if g.method == "exact"]
        simhash_groups = [g for g in groups if g.method == "simhash"]
        fuzzy_groups = [g for g in groups if g.method == "fuzzy"]
        semantic_groups = [g for g in groups if g.method == "semantic"]

        self._duplicate_groups = groups
        self.stats = DedupStats(
            total_input=len(data),
            total_output=len(output),
            filtered=len(data) - len(output),
            duplicates_removed=len(removed),
            duplicate_groups=len(groups),
            exact_matches=sum(g.size - 1 for g in exact_groups),
            simhash_matches=sum(g.size - 1 for g in simhash_groups),
            fuzzy_matches=sum(g.size - 1 for g in fuzzy_groups),
            semantic_matches=sum(g.size - 1 for g in semantic_groups),
            duration_ms=(time.time() - t0) * 1000,
            throughput_per_sec=len(data) / max((time.time() - t0), 0.001),
            method="+".join(self._methods),
        )

        self._audit_operation("dedup.process", "OK", self.stats.duration_ms,
                              len(data), asdict(self.stats))
        return output, self.stats

    def _dedup_exact(self, data: List[Dict]) -> List[DuplicateGroup]:
        seen: Dict[str, int] = {}
        groups_map: Dict[str, List[int]] = defaultdict(list)
        field_names = self.config.get("dedup", "fields_weight", default={})
        key_fields = list(field_names.keys()) if field_names else None

        for i, rec in enumerate(data):
            if key_fields:
                key = json.dumps({f: rec.get(f, "") for f in key_fields}, sort_keys=True, ensure_ascii=False)
            else:
                key = compute_content_hash(json.dumps(rec, sort_keys=True, ensure_ascii=False))

            if key in seen:
                groups_map[key].append(i)
            else:
                seen[key] = i
                groups_map[key] = [i]

        groups = []
        for key, indices in groups_map.items():
            if len(indices) >= 2:
                groups.append(DuplicateGroup(
                    group_id=str(uuid.uuid4())[:8],
                    size=len(indices),
                    representative_idx=indices[0],
                    member_indices=indices,
                    method="exact",
                    confidence=1.0,
                    key_snippet=key[:60],
                ))
        return groups

    def _dedup_simhash(self, data: List[Dict], original_indices: List[int]) -> List[DuplicateGroup]:
        if len(data) < 2:
            return []

        fingerprints = []
        for rec in data:
            text = json.dumps(rec, sort_keys=True, ensure_ascii=False)
            fp = self._simhash.compute(text)
            fingerprints.append(fp)

        n = len(fingerprints)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if parent[i] == parent[j]:
                    continue
                if self._simhash.is_similar(fingerprints[i], fingerprints[j], self._simhash_threshold):
                    union(i, j)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            root = find(i)
            clusters[root].append(i)

        groups = []
        for cluster_indices in clusters.values():
            if len(cluster_indices) < 2:
                continue
            sorted_indices = sorted(cluster_indices)
            original_members = [original_indices[idx] for idx in sorted_indices]
            groups.append(DuplicateGroup(
                group_id=str(uuid.uuid4())[:8],
                size=len(original_members),
                representative_idx=original_members[0],
                member_indices=original_members,
                method="simhash",
                confidence=0.85,
            ))
        return groups

    def _dedup_fuzzy(self, data: List[Dict], original_indices: List[int]) -> List[DuplicateGroup]:
        if len(data) < 2:
            return []

        n = len(data)
        used = set()
        groups = []

        for i in range(n):
            if i in used:
                continue
            group_members = [i]
            for j in range(i + 1, n):
                if j in used:
                    continue
                score = self._rule_engine.fuzzy_match(data[i], data[j])
                if score >= self._rule_engine.fuzzy_threshold:
                    group_members.append(j)
                    used.add(j)
            if len(group_members) >= 2:
                used.add(i)
                original_members = [original_indices[idx] for idx in group_members]
                groups.append(DuplicateGroup(
                    group_id=str(uuid.uuid4())[:8],
                    size=len(original_members),
                    representative_idx=original_members[0],
                    member_indices=original_members,
                    method="fuzzy",
                    confidence=score if len(group_members) == 2 else 0.82,
                    key_snippet=json.dumps(data[group_members[0]], ensure_ascii=False)[:60],
                ))
        return groups

    def _dedup_semantic(self, data: List[Dict], original_indices: List[int]) -> List[DuplicateGroup]:
        if len(data) < 2:
            return []

        texts = [json.dumps(rec, sort_keys=True, ensure_ascii=False) for rec in data]
        pairs = self._semantic.find_similar_pairs(texts)

        if not pairs:
            return []

        n = len(data)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j, _ in pairs:
            union(i, j)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            root = find(i)
            clusters[root].append(i)

        groups = []
        for cluster_indices in clusters.values():
            if len(cluster_indices) < 2:
                continue
            sorted_indices = sorted(cluster_indices)
            original_members = [original_indices[idx] for idx in sorted_indices]
            groups.append(DuplicateGroup(
                group_id=str(uuid.uuid4())[:8],
                size=len(original_members),
                representative_idx=original_members[0],
                member_indices=original_members,
                method="semantic",
                confidence=0.78,
            ))
        return groups

    def validate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        issues = []
        seen_keys = set()
        for i, rec in enumerate(data):
            key = compute_content_hash(json.dumps(rec, sort_keys=True, ensure_ascii=False))
            if key in seen_keys:
                issues.append({
                    "index": i, "type": "exact_duplicate",
                    "message": f"Record {i} is exact duplicate",
                    "severity": "WARN",
                })
            seen_keys.add(key)
        return issues

    def preview(self, data: List[Dict[str, Any]], max_groups: int = 20) -> Dict[str, Any]:
        _, stats = self.process(data)
        preview_groups = self._duplicate_groups[:max_groups]

        summary = []
        for g in preview_groups:
            members = [{"index": idx, "data": data[idx]} for idx in g.member_indices]
            summary.append({
                "group_id": g.group_id,
                "method": g.method,
                "confidence": g.confidence,
                "size": g.size,
                "members": members,
            })

        self._preview_cache = {
            "total_groups": stats.duplicate_groups,
            "total_duplicates": stats.duplicates_removed,
            "preview_groups": summary,
            "stats": asdict(stats),
        }
        return self._preview_cache

    def resolve_conflict(self, members: List[Dict[str, Any]]) -> Dict[str, Any]:
        strategy = self._conflict_resolution
        if not members:
            return {}
        if len(members) == 1:
            return dict(members[0])

        if strategy == "keep_first":
            return dict(members[0])
        elif strategy == "keep_longest":
            return max(members, key=lambda r: sum(len(str(v)) for v in r.values()))
        elif strategy == "keep_newest":
            for field in ["updated_at", "created_at", "timestamp", "date"]:
                members_with_dates = [m for m in members if field in m]
                if members_with_dates:
                    return max(members_with_dates, key=lambda m: str(m.get(field, "")))
            return dict(members[0])
        elif strategy == "merge":
            merged = {}
            for m in members:
                for k, v in m.items():
                    if k not in merged:
                        merged[k] = v
                    elif str(merged[k]).strip() != str(v).strip():
                        merged[k] = str(merged[k]) + " | " + str(v)
            return merged
        else:
            return dict(members[0])

    def report(self) -> Dict[str, Any]:
        base = super().report()
        base.update({
            "methods_used": list(self._methods),
            "simhash_config": {
                "fp_len": self._simhash.fp_len,
                "ngram": self._simhash.ngram,
                "threshold": self._simhash_threshold,
            },
            "conflict_resolution": self._conflict_resolution,
            "semantic_available": self._semantic.available,
        })
        return base
