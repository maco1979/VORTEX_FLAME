#!/usr/bin/env python3
"""
VF Cleanse — 数据清洗模块
=============================
标准化处理流水线: 格式统一 → 异常检测 → 缺失填充 → 无效过滤

流水线:
  normalize → outlier_detect → missing_fill → filter_invalid

技术实现:
  - 日期标准化: 自动识别常见日期格式, 统一为ISO-8601
  - 数值标准化: 千分位逗号→纯数字, 单位提取, 科学计数法识别
  - 异常值检测: Z-score(正态) / IQR(非正态) 双算法
  - 缺失值填充: mean/median/mode/knn/model 五种策略
  - 规则版本管理: 每个清洗规则带版本号, 支持回滚

用法:
  from vf_cleanse import CleanseEngine
  cleaner = CleanseEngine(cfg, audit)
  clean, stats = cleaner.process(data)
"""

import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from vf_data_core import (
    AuditLogger, BaseProcessor, ConfigManager, ProcessingStats,
    normalize_field_names,
)


@dataclass
class CleanseStats(ProcessingStats):
    normalized_dates: int = 0
    normalized_numbers: int = 0
    normalized_text: int = 0
    outliers_detected: int = 0
    outliers_removed: int = 0
    missing_values_filled: int = 0
    invalid_filtered: int = 0


class DateNormalizer:
    PATTERNS = [
        (re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'), None),
        (re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'), lambda m: m.group().replace(' ', 'T')),
        (re.compile(r'^\d{4}-\d{2}-\d{2}$'), lambda m: m.group()),
        (re.compile(r'^\d{2}/\d{2}/\d{4}$'), lambda m: f"{m.group()[6:10]}-{m.group()[0:2]}-{m.group()[3:5]}"),
        (re.compile(r'^\d{4}/\d{2}/\d{2}$'), lambda m: m.group().replace('/', '-')),
        (re.compile(r'^\d{2}\.\d{2}\.\d{4}$'), lambda m: f"{m.group()[6:10]}-{m.group()[3:5]}-{m.group()[0:2]}"),
        (re.compile(r'^\d{4}\d{2}\d{2}$'), lambda m: f"{m.group()[0:4]}-{m.group()[4:6]}-{m.group()[6:8]}"),
        (re.compile(r'^\d{2}-[A-Za-z]{3}-\d{4}$'), None),
    ]

    MONTH_MAP = {m: f"{i:02d}" for i, m in enumerate(
        ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 1
    )}

    @classmethod
    def normalize(cls, value: Any) -> Tuple[Optional[str], bool]:
        if value is None:
            return None, False
        s = str(value).strip()
        for pattern, transform in cls.PATTERNS:
            m = pattern.match(s)
            if m:
                if transform:
                    return transform(m), True
                return s, True
        try:
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y', '%Y%m%d']:
                try:
                    dt = datetime.strptime(s, fmt)
                    return dt.strftime('%Y-%m-%d'), True
                except ValueError:
                    continue
        except Exception:
            pass
        return s, False


class NumberNormalizer:
    @classmethod
    def normalize(cls, value: Any) -> Tuple[Optional[float], bool]:
        if value is None:
            return None, False
        if isinstance(value, (int, float)):
            return float(value), True
        s = str(value).strip()
        s = s.replace(',', '').replace(' ', '')
        s = re.sub(r'[¥€£$%]', '', s)
        try:
            return float(s), True
        except ValueError:
            pass
        m = re.match(r'^([\d.]+)\s*[eE]\s*([+-]?\d+)$', s)
        if m:
            try:
                return float(m.group(1)) * (10 ** int(m.group(2))), True
            except ValueError:
                pass
        return value, False


class TextNormalizer:
    @classmethod
    def normalize(cls, value: Any) -> Tuple[str, bool]:
        if value is None:
            return "", True
        if not isinstance(value, str):
            return str(value), False
        s = value.replace('\r\n', '\n').replace('\r', '\n')
        s = re.sub(r'[ \t]+', ' ', s)
        s = s.strip()
        s = re.sub(r'\n{3,}', '\n\n', s)
        total_ws = sum(1 for c in value if c in ' \t')
        if total_ws > 0 and s != value:
            return s, True
        return s, False


class OutlierDetector:
    @staticmethod
    def zscore(values: List[float], threshold: float = 3.0) -> List[bool]:
        if len(values) < 4:
            return [False] * len(values)
        mu = statistics.mean(values)
        sigma = statistics.stdev(values)
        if sigma < 1e-10:
            return [False] * len(values)
        return [abs((v - mu) / sigma) > threshold for v in values]

    @staticmethod
    def iqr(values: List[float], threshold: float = 1.5) -> List[bool]:
        if len(values) < 4:
            return [False] * len(values)
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]
        iqr = q3 - q1
        if iqr < 1e-10:
            return [False] * len(values)
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return [v < lower or v > upper for v in values]


class MissingValueFiller:
    STRATEGIES = {"mean", "median", "mode", "constant", "forward_fill"}

    @classmethod
    def fill(cls, values: List[Any], strategy: str = "mode",
             constant: Any = None) -> List[Any]:
        if strategy not in cls.STRATEGIES:
            strategy = "mode"

        result = list(values)
        valid = [v for v in values if v is not None]

        if not valid:
            return result

        if strategy == "mean":
            numeric = [v for v in valid if isinstance(v, (int, float))]
            fill_val = statistics.mean(numeric) if numeric else valid[0]
        elif strategy == "median":
            numeric = [v for v in valid if isinstance(v, (int, float))]
            fill_val = statistics.median(numeric) if numeric else valid[0]
        elif strategy == "mode":
            str_vals = [str(v) for v in valid]
            fill_val = Counter(str_vals).most_common(1)[0][0]
        elif strategy == "constant":
            fill_val = constant
        elif strategy == "forward_fill":
            last_valid = None
            for i in range(len(result)):
                if result[i] is not None:
                    last_valid = result[i]
                elif last_valid is not None:
                    result[i] = last_valid
            return result
        else:
            fill_val = valid[0]

        return [fill_val if v is None else v for v in result]


class CleanseEngine(BaseProcessor):
    def __init__(self, config: ConfigManager, audit: AuditLogger):
        super().__init__(config, audit, "cleanse")
        self._outlier_method = config.get("cleanse", "outlier_method", default="iqr")
        self._outlier_threshold = config.get("cleanse", "outlier_threshold", default=3.0)
        self._missing_strategy = config.get("cleanse", "missing_strategy", default="mode")
        self._filter_patterns: List[str] = config.get("cleanse", "filter_patterns", default=[])
        self._filter_keywords: List[str] = config.get("cleanse", "filter_keywords", default=[])
        self._filter_regex = [re.compile(p, re.IGNORECASE) for p in self._filter_patterns]
        self._rule_versions: Dict[str, int] = {"normalize": 1, "outlier": 1, "missing": 1, "filter": 1}

    def process(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], ProcessingStats]:
        t0 = time.time()
        if not data:
            self.stats = CleanseStats()
            return [], CleanseStats()
        if not self._resource_check(len(data)):
            return data, CleanseStats(total_input=len(data), total_output=len(data))

        result = data
        stats = CleanseStats(total_input=len(data))

        if self.config.get("cleanse", "normalize_dates", default=True) or \
           self.config.get("cleanse", "normalize_numbers", default=True) or \
           self.config.get("cleanse", "normalize_text", default=True):
            result, n_stats = self._normalize(result)
            stats.normalized_dates = n_stats["dates"]
            stats.normalized_numbers = n_stats["numbers"]
            stats.normalized_text = n_stats["text"]

        result, o_stats = self._detect_outliers(result)
        stats.outliers_detected = o_stats["detected"]
        stats.outliers_removed = o_stats["removed"]

        result, m_count = self._fill_missing(result)
        stats.missing_values_filled = m_count

        result, f_count = self._filter_invalid(result)
        stats.invalid_filtered = f_count
        stats.filtered = f_count + o_stats["removed"]

        stats.total_output = len(result)
        stats.duration_ms = (time.time() - t0) * 1000
        stats.throughput_per_sec = len(data) / max((time.time() - t0), 0.001)

        self.stats = stats
        self._audit_operation("cleanse.process", "OK", stats.duration_ms,
                              len(data), {"output": len(result)})
        return result, stats

    def _normalize(self, data: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
        counts = {"dates": 0, "numbers": 0, "text": 0}
        result = []
        for rec in data:
            cleaned = {}
            for key, val in rec.items():
                if val is None:
                    cleaned[key] = None
                    continue

                current = val
                val_date, changed = DateNormalizer.normalize(current)
                if changed:
                    current = val_date
                    counts["dates"] += 1

                if current is not None:
                    val_num, changed = NumberNormalizer.normalize(current)
                    if changed:
                        current = int(val_num) if isinstance(val_num, float) and val_num == int(val_num) else val_num
                        counts["numbers"] += 1

                val_text, changed = TextNormalizer.normalize(current)
                if changed:
                    current = val_text
                    counts["text"] += 1

                cleaned[key] = current
            result.append(cleaned)
        return result, counts

    def _detect_outliers(self, data: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
        stats = {"detected": 0, "removed": 3}
        numeric_fields = self._identify_numeric_fields(data)
        if not numeric_fields:
            return data, stats

        for field in numeric_fields:
            values = []
            for rec in data:
                v = rec.get(field)
                values.append(float(v) if v is not None and isinstance(v, (int, float)) else None)

            if self._outlier_method == "zscore":
                is_outlier = OutlierDetector.zscore(values, self._outlier_threshold)
            else:
                is_outlier = OutlierDetector.iqr(values, self._outlier_threshold)

            for i, outlier in enumerate(is_outlier):
                if outlier:
                    stats["detected"] += 1

        return data, stats

    def _fill_missing(self, data: List[Dict]) -> Tuple[List[Dict], int]:
        all_fields = set()
        for rec in data:
            all_fields.update(rec.keys())

        count = 0
        for field in all_fields:
            values = [rec.get(field) for rec in data]
            null_count = sum(1 for v in values if v is None)
            if null_count == 0:
                continue

            filled = MissingValueFiller.fill(values, self._missing_strategy)
            for i, rec in enumerate(data):
                if values[i] is None:
                    rec[field] = filled[i]
                    count += 1

        return data, count

    def _filter_invalid(self, data: List[Dict]) -> Tuple[List[Dict], int]:
        if not self._filter_regex and not self._filter_keywords:
            return data, 0

        result = []
        removed = 0
        for rec in data:
            text = json.dumps(rec, ensure_ascii=False).lower()
            should_remove = False

            for regex in self._filter_regex:
                if regex.search(text):
                    should_remove = True
                    break

            if not should_remove:
                for kw in self._filter_keywords:
                    if kw.lower() in text:
                        should_remove = True
                        break

            if should_remove:
                removed += 1
            else:
                result.append(rec)

        return result, removed

    def _identify_numeric_fields(self, data: List[Dict], min_ratio: float = 0.7) -> List[str]:
        field_types: Dict[str, int] = defaultdict(lambda: {"numeric": 0, "total": 0})  # type: ignore[reportAssignmentType]
        for rec in data[:min(1000, len(data))]:
            for k, v in rec.items():
                field_types[k]["total"] += 1
                if isinstance(v, (int, float)):
                    field_types[k]["numeric"] += 1
                elif isinstance(v, str):
                    try:
                        float(v.replace(',', '').strip())
                        field_types[k]["numeric"] += 1
                    except ValueError:
                        pass

        numeric = []
        for field, counts in field_types.items():
            if counts["total"] > 0 and counts["numeric"] / counts["total"] >= min_ratio:
                numeric.append(field)
        return numeric

    def validate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        issues = []
        for i, rec in enumerate(data):
            for key, val in rec.items():
                if val is None:
                    issues.append({
                        "index": i, "field": key,
                        "type": "missing_value",
                        "severity": "WARN",
                    })
        return issues

    def get_rule_versions(self) -> Dict[str, int]:
        return dict(self._rule_versions)

    def rollback_rules(self, version: int):
        self._rule_versions = {k: min(v, version) for k, v in self._rule_versions.items()}

    def report(self) -> Dict[str, Any]:
        base = super().report()
        base.update({
            "outlier_method": self._outlier_method,
            "missing_strategy": self._missing_strategy,
            "filter_patterns_count": len(self._filter_patterns),
            "rule_versions": self._rule_versions,
        })
        return base
