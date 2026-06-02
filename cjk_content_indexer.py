#!/usr/bin/env python3
"""
CJK Content Indexer — Auto-scan E:/AI_Data/ for Chinese text and inject into soul memory
============================================================================================

Scans text files (.txt, .md, .json, .jsonl, .csv, .yaml, .py, .html) in E:\AI_Data\
Extracts paragraphs/lines containing CJK characters
Routes to appropriate souls based on content domain + auto_index routing rules
Deduplicates via content hash

Usage:
  python cjk_content_indexer.py           # full scan
  python cjk_content_indexer.py --dry-run  # scan only, no write
  python cjk_content_indexer.py --max 500  # cap entries per run
  python cjk_content_indexer.py --soul cezanne  # force target soul
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from soul_memory import write

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger("cjk_indexer")

SCAN_ROOT = os.getenv("AI_DATA", r"E:\AI_Data")
SCAN_EXTS = {".txt", ".md", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".py", ".html", ".rst"}
SKIP_DIRS = {".cache", ".git", "__pycache__", ".huggingface", "venv", "node_modules"}
SKIP_FILES = {"CACHEDIR.TAG", ".gitattributes", ".gitignore", "config.json"}
MAX_FILE_SIZE_MB = 200
MAX_LINES_PER_FILE = 50000

CJK_CHAR = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
MIN_CJK_CHARS_PER_CHUNK = 8
MAX_CHUNK_CHARS = 2000

DOMAIN_CLASSIFIERS = [
    (re.compile(r"安全|漏洞|注入|攻击|防御|审计|威胁|渗透|加密|认证|授权|隐私|合规|GDPR|OWASP|XSS|SQL|RCE|DoS|STRIDE|CVE"), "cezanne"),
    (re.compile(r"数学|物理|量子|方程|定理|公式|概率|统计|优化|梯度|收敛"), "einstein"),
    (re.compile(r"博弈|策略|决策|风险|收益|投资|市场|经济|竞争|纳什|均衡|金融"), "strategy"),
    (re.compile(r"法律|法规|宪法|刑法|民法|合同|条例|合规|政策|治理|制度"), "montesquieu"),
    (re.compile(r"对话|哲学|伦理|道德|思维|认知|意识|心理学"), "guizhu"),
    (re.compile(r"生物|进化|基因|生态|细胞|DNA|物种|自然选择"), "darwin"),
    (re.compile(r"历史|文明|考古|古代|帝国|战争|朝代|文献"), "herodotus"),
    (re.compile(r"音乐|和声|旋律|节奏|乐章|交响|音符|乐器|作曲|声学"), "beethoven"),
    (re.compile(r"艺术|绘画|色彩|构图|风格|美学|印象|表现|雕塑|设计"), "monet"),
    (re.compile(r"天文|宇宙|星系|行星|恒星|黑洞|引力|暗物质|太空"), "galileo"),
    (re.compile(r"代码|编程|算法|函数|类|模块|API|接口|编译|调试|框架|Python|Java|C\+\+"), "cezanne"),
]

FALLBACK_SOUL = "cezanne"


def classify_soul(text: str) -> str:
    scores = {}
    for pattern, soul in DOMAIN_CLASSIFIERS:
        matches = len(pattern.findall(text))
        if matches > 0:
            scores[soul] = scores.get(soul, 0) + matches
    if not scores:
        return FALLBACK_SOUL
    return max(scores, key=scores.get)  # type: ignore[reportArgumentType]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


KNOWN_HASHES: set = set()


def load_known_hashes():
    from soul_memory import _engine
    try:
        conn = _engine._get_db("cezanne")
        rows = conn.execute(
            "SELECT content FROM memories WHERE content LIKE '%cjk_indexer%'"
        ).fetchall()
        for row in rows:
            try:
                c = json.loads(row[0])
                h = c.get("_cjk_hash", "")
                if h:
                    KNOWN_HASHES.add(h)
            except Exception:
                pass
    except Exception:
        pass
    logger.info("Loaded {} known CJK hashes".format(len(KNOWN_HASHES)))


def extract_cjk_chunks(filepath: str) -> List[Tuple[str, str]]:
    chunks = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_FILE_SIZE_MB * 1024 * 1024)
    except Exception as e:
        logger.debug("Cannot read {}: {}".format(filepath, str(e)[:60]))
        return chunks

    lines = content.split("\n")
    current_paragraph = []
    current_cjk_count = 0

    for line in lines[:MAX_LINES_PER_FILE]:
        line = line.strip()
        if not line:
            if current_paragraph and current_cjk_count >= MIN_CJK_CHARS_PER_CHUNK:
                text = " ".join(current_paragraph)
                if len(text) <= MAX_CHUNK_CHARS:
                    chunks.append((text[:MAX_CHUNK_CHARS], "paragraph"))
                else:
                    sentences = re.split(r"[。！？.!?\n]", text)
                    for sent in sentences:
                        sent = sent.strip()
                        if CJK_CHAR.search(sent) and len(sent) >= MIN_CJK_CHARS_PER_CHUNK:
                            chunks.append((sent[:MAX_CHUNK_CHARS], "sentence"))
            current_paragraph = []
            current_cjk_count = 0
            continue

        cjk_in_line = len(CJK_CHAR.findall(line))
        if cjk_in_line >= 3:
            current_paragraph.append(line)
            current_cjk_count += cjk_in_line
        elif current_paragraph:
            if current_cjk_count >= MIN_CJK_CHARS_PER_CHUNK:
                text = " ".join(current_paragraph)
                chunks.append((text[:MAX_CHUNK_CHARS], "paragraph"))
            current_paragraph = []
            current_cjk_count = 0

    if current_paragraph and current_cjk_count >= MIN_CJK_CHARS_PER_CHUNK:
        text = " ".join(current_paragraph)
        chunks.append((text[:MAX_CHUNK_CHARS], "paragraph"))

    return chunks


def scan_and_index(max_entries: int = 500, dry_run: bool = False,
                   force_soul: Optional[str] = None):
    load_known_hashes()

    total_files = 0
    cjk_files = 0
    total_chunks = 0
    written = 0

    for root, dirs, files in os.walk(SCAN_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        files = [f for f in files if f not in SKIP_FILES]

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SCAN_EXTS:
                continue

            total_files += 1
            filepath = os.path.join(root, fname)
            try:
                fsize = os.path.getsize(filepath)
                if fsize > MAX_FILE_SIZE_MB * 1024 * 1024:
                    continue
            except OSError:
                continue

            chunks = extract_cjk_chunks(filepath)
            if not chunks:
                continue

            cjk_files += 1
            relpath = os.path.relpath(filepath, SCAN_ROOT)
            logger.debug("CJK file: {} ({} chunks)".format(relpath, len(chunks)))

            for text, chunk_type in chunks:
                total_chunks += 1
                h = content_hash(text)
                if h in KNOWN_HASHES:
                    continue
                KNOWN_HASHES.add(h)

                if written >= max_entries:
                    logger.info("Reached max_entries={}".format(max_entries))
                    return {
                        "total_files": total_files,
                        "cjk_files": cjk_files,
                        "total_chunks": total_chunks,
                        "written": written,
                        "truncated": True,
                    }

                if dry_run:
                    soul = force_soul or classify_soul(text)
                    logger.info("[DRY] {} → {} | {} chars | {}"
                                .format(relpath[:50], soul, len(text), text[:60]))
                    written += 1
                    continue

                soul = force_soul or classify_soul(text)
                source_path = os.path.join(os.getenv("AI_DATA", "E:/AI_Data"), relpath.replace("\\", "/"))
                try:
                    write(soul, "knowledge", {
                        "topic": "CJK Content: {}".format(text[:80]),
                        "detail": text,
                        "source": source_path,
                        "_cjk_hash": h,
                        "_source_file": relpath,
                        "_chunk_type": chunk_type,
                    }, importance=0.7, tags=["cjk_indexer", "auto"])
                    written += 1
                    if written % 20 == 0:
                        logger.info("Written {}/{} chunks...".format(written, min(total_chunks, max_entries)))
                except Exception as e:
                    logger.debug("Write failed for {}: {}".format(relpath, str(e)[:60]))

    return {
        "total_files": total_files,
        "cjk_files": cjk_files,
        "total_chunks": total_chunks,
        "written": written,
        "truncated": False,
    }


def main():
    parser = argparse.ArgumentParser(description="CJK Content Indexer")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, no writes")
    parser.add_argument("--max", type=int, default=500, help="Max entries per run")
    parser.add_argument("--soul", type=str, default=None, help="Force target soul")
    args = parser.parse_args()

    logger.info("CJK Content Indexer starting")
    logger.info("  root: {}".format(SCAN_ROOT))
    logger.info("  dry_run: {}".format(args.dry_run))
    logger.info("  max_entries: {}".format(args.max))
    logger.info("  force_soul: {}".format(args.soul or "auto-classify"))

    t0 = time.time()
    stats = scan_and_index(
        max_entries=args.max,
        dry_run=args.dry_run,
        force_soul=args.soul,
    )
    elapsed = time.time() - t0

    logger.info("=" * 50)
    logger.info("Scan complete in {:.1f}s".format(elapsed))
    logger.info("  Files scanned:    {}".format(stats["total_files"]))
    logger.info("  CJK files found:  {}".format(stats["cjk_files"]))
    logger.info("  CJK chunks:       {}".format(stats["total_chunks"]))
    logger.info("  Written:          {}".format(stats["written"]))
    if stats.get("truncated"):
        logger.info("  ⚠ Truncated at max_entries={}".format(args.max))
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
