#!/usr/bin/env python3
"""
Populate FAISS Knowledge Store with ALL rStar-Coder + Opus data
Uses JSONL append-only format for massive data (590K+ entries)
No capacity limit - FAISS handles millions of entries
"""
import os, sys, json, time, glob, hashlib

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

RSTAR_DIR = "/mnt/e/AI_Data/rStar-Coder"
OPUS_DIR = "/mnt/e/AI_Data/Opus-4.6-Reasoning"

sys.path.insert(0, "/mnt/d/VORTEX_FLAME")
from long_memory import (
    write_knowledge, build_knowledge_index, close_knowledge_handles,
    _load_mem, _save_mem, _load_knowledge, _knowledge_data_path, get_stats
)


def clean_old_entries(soul="cezanne"):
    data = _load_mem(soul, use_cache=False)
    keep = []
    removed = 0
    for e in data["entries"]:
        cat = e.get("category", "")
        if cat in ("code",):
            removed += 1
            continue
        content = e.get("content", "")
        if len(content) > 5000:
            removed += 1
            continue
        keep.append(e)
    data["entries"] = keep
    _save_mem(data, soul)
    print(f"  Cleaned {soul}: removed {removed} bloated entries, kept {len(keep)}", flush=True)
    return removed


def _entry_id(source, fpath, row):
    return hashlib.md5(f"{source}:{fpath}:{row}".encode()).hexdigest()[:12]


_ID_INDEX_PATH = None


def _id_index_path(soul="cezanne"):
    global _ID_INDEX_PATH
    if _ID_INDEX_PATH is None:
        from long_memory import _knowledge_dir
        _ID_INDEX_PATH = os.path.join(_knowledge_dir(soul), "ids.txt")
    return _ID_INDEX_PATH


def _load_existing_ids(soul="cezanne"):
    id_path = _id_index_path(soul)
    if os.path.exists(id_path):
        with open(id_path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    entries = _load_knowledge(soul)
    ids = {e["id"] for e in entries}
    _save_id_index(ids, soul)
    return ids


def _save_id_index(ids, soul="cezanne"):
    id_path = _id_index_path(soul)
    with open(id_path, "w") as f:
        for eid in ids:
            f.write(eid + "\n")


def _append_id_index(eid, soul="cezanne"):
    id_path = _id_index_path(soul)
    with open(id_path, "a") as f:
        f.write(eid + "\n")


def populate_rstar(soul="cezanne"):
    print(f"\n  [1/2] Populating rStar-Coder...", flush=True)
    import pyarrow.parquet as pq

    sft_files = sorted(glob.glob(os.path.join(RSTAR_DIR, "seed_sft", "*.parquet")))
    test_files = sorted(glob.glob(os.path.join(RSTAR_DIR, "seed_testcase", "*.parquet")))
    all_files = sft_files + test_files
    print(f"  Found {len(sft_files)} seed_sft + {len(test_files)} seed_testcase = {len(all_files)} files", flush=True)

    existing_ids = _load_existing_ids(soul)
    total = 0
    added = 0

    for fi, fpath in enumerate(all_files):
        try:
            pf = pq.ParquetFile(fpath)
            num_groups = pf.metadata.num_row_groups
            for rg in range(num_groups):
                table = pf.read_row_group(rg)
                nrows = table.num_rows

                questions = table.column("question").to_pylist() if "question" in table.column_names else [None]*nrows
                codes = table.column("code").to_pylist() if "code" in table.column_names else [None]*nrows
                responses = table.column("response").to_pylist() if "response" in table.column_names else [None]*nrows
                starters = table.column("starter_code").to_pylist() if "starter_code" in table.column_names else [None]*nrows

                for i in range(nrows):
                    total += 1
                    question = str(questions[i] or "")[:800]
                    code = str(codes[i] or "")[:1500]
                    response = str(responses[i] or "")[:1500]
                    starter = str(starters[i] or "")[:500]

                    content_parts = []
                    if question and question != "nan" and question != "None":
                        content_parts.append(f"Q: {question}")
                    if starter and starter != "nan" and starter != "None":
                        content_parts.append(f"Starter: {starter}")
                    if code and code != "nan" and code != "None":
                        content_parts.append(f"Code: {code}")
                    if response and response != "nan" and response != "None":
                        content_parts.append(f"Solution: {response}")

                    content = "\n".join(content_parts)
                    if len(content) < 30:
                        continue

                    eid = _entry_id("rstar", fpath, total)
                    if eid in existing_ids:
                        continue

                    write_knowledge(soul, "rstar_coder", content, {
                        "source": "rStar-Coder",
                        "file": os.path.basename(fpath),
                        "row": total,
                    }, entry_id=eid)
                    existing_ids.add(eid)
                    _append_id_index(eid, soul)
                    added += 1

                    if added % 10000 == 0:
                        print(f"    [{fi+1}/{len(all_files)}] rg={rg}/{num_groups} {added} added, {total} scanned", flush=True)

                del table, questions, codes, responses, starters

        except Exception as e:
            print(f"    [WARN] {os.path.basename(fpath)}: {e}", flush=True)

    print(f"  rStar done: {added} added, {total} total scanned", flush=True)
    return added


def populate_opus(soul="cezanne"):
    print(f"\n  [2/2] Populating Opus-4.6-Reasoning...", flush=True)
    import pyarrow.parquet as pq

    all_files = glob.glob(os.path.join(OPUS_DIR, "*.parquet"))
    print(f"  Found {len(all_files)} files", flush=True)

    existing_ids = _load_existing_ids(soul)
    total = 0
    added = 0

    for fpath in all_files:
        try:
            pf = pq.ParquetFile(fpath)
            num_groups = pf.metadata.num_row_groups
            for rg in range(num_groups):
                table = pf.read_row_group(rg)
                nrows = table.num_rows
                col_data = {}
                for col in table.column_names:
                    col_data[col] = table.column(col).to_pylist()

                for i in range(nrows):
                    total += 1
                    values = {}
                    for col, data in col_data.items():
                        val = data[i]
                        if val is not None:
                            sv = str(val)
                            if sv and sv != "nan" and sv != "None":
                                values[col] = sv

                    content_parts = []
                    for k, v in values.items():
                        content_parts.append(f"{k}: {v[:800]}")

                    content = "\n".join(content_parts)
                    if len(content) < 30:
                        continue

                    eid = _entry_id("opus", fpath, total)
                    if eid in existing_ids:
                        continue

                    write_knowledge(soul, "opus_reasoning", content, {
                        "source": "Opus-4.6-Reasoning",
                        "file": os.path.basename(fpath),
                    }, entry_id=eid)
                    existing_ids.add(eid)
                    _append_id_index(eid, soul)
                    added += 1

                del table, col_data

        except Exception as e:
            print(f"    [WARN] {os.path.basename(fpath)}: {e}", flush=True)

    print(f"  Opus done: {added} added, {total} total scanned", flush=True)
    return added


def main():
    soul = "cezanne"
    print(f"\n{'#'*60}", flush=True)
    print(f"  POPULATING FAISS KNOWLEDGE STORE (FULL rStar + Opus)", flush=True)
    print(f"  Target: {soul} knowledge store (JSONL + FAISS)", flush=True)
    print(f"{'#'*60}", flush=True)

    t0 = time.time()

    print(f"\n  Step 0: Clean old bloated entries from mem.json...", flush=True)
    clean_old_entries(soul)

    print(f"\n  Step 1: Populate rStar-Coder (ALL data)...", flush=True)
    rstar_count = populate_rstar(soul)

    print(f"\n  Step 2: Populate Opus-4.6-Reasoning (ALL data)...", flush=True)
    opus_count = populate_opus(soul)

    print(f"\n  Step 2b: Closing file handles...", flush=True)
    close_knowledge_handles()

    print(f"\n  Step 3: Build FAISS knowledge index (this may take a while)...", flush=True)
    index_count = build_knowledge_index(soul)

    elapsed = (time.time() - t0) / 60

    stats = get_stats(soul)

    print(f"\n{'#'*60}", flush=True)
    print(f"  KNOWLEDGE STORE POPULATION COMPLETE", flush=True)
    print(f"  rStar added: {rstar_count}", flush=True)
    print(f"  Opus added: {opus_count}", flush=True)
    print(f"  Knowledge index: {index_count} entries", flush=True)
    print(f"  Main memory: {stats['total']} entries", flush=True)
    print(f"  Knowledge store: {stats.get('knowledge_total', 0)} entries", flush=True)
    print(f"  Time: {elapsed:.1f} minutes", flush=True)
    print(f"{'#'*60}", flush=True)


if __name__ == "__main__":
    main()
