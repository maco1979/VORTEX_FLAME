#!/usr/bin/env python3
"""
Shared Memory System for 7B + 8B Cezanne
- Stores rStar-Coder, Opus, and self-play trajectories
- Both models can read during inference (context learning)
- Only CS-filtered data used for training
- Adversarial loop: 7B answers → 8B judges → corrections stored
"""
import os, json, time, sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MEMORY_DIR = "/mnt/d/VORTEX_FLAME/shared_memory/cezanne"
LOG_DIR = "/mnt/d/VORTEX_FLAME/hermes_logs/cezanne"
RSTAR_DIR = "/mnt/e/AI_Data/rStar-Coder"
OPUS_DIR = "/mnt/e/AI_Data/Opus-4.6-Reasoning"

CS_KEYWORDS = [
    "algorithm", "sort", "search", "graph", "tree", "hash", "stack", "queue",
    "linked list", "binary", "recursion", "dynamic programming", "greedy",
    "debug", "bug", "error", "exception", "traceback", "fix",
    "complexity", "big-o", "time complexity", "space complexity",
    "memory", "cache", "cpu", "thread", "process", "deadlock", "mutex",
    "tcp", "udp", "socket", "dns", "http", "rest", "sql",
    "pointer", "reference", "malloc", "free", "heap", "stack overflow",
    "proof", "logic", "contradiction", "induction", "syllogism",
    "dijkstra", "bfs", "dfs", "quicksort", "mergesort",
    "off-by-one", "race condition", "base case", "boundary",
    "compiler", "interpreter", "virtual machine", "garbage collection",
    "concurrent", "parallel", "async", "synchronize",
    "data structure", "array", "vector", "map", "set", "dictionary",
    "function", "class", "object", "inheritance", "polymorphism",
    "loop", "while", "for", "if", "else", "switch",
    "python", "c++", "java", "rust", "go",
    "leetcode", "competitive programming",
]

NON_CS = [
    "recipe", "cooking", "baking", "restaurant", "food",
    "movie", "film", "actor", "actress", "director",
    "song", "music", "album", "artist", "band",
    "travel", "vacation", "hotel", "flight",
    "fashion", "clothing", "outfit", "dress",
    "pet", "dog", "cat", "animal",
    "relationship", "dating", "marriage", "wedding",
    "joke", "humor", "funny", "laugh",
]


def is_cs_related(text):
    text_lower = text.lower()
    cs_hits = sum(1 for kw in CS_KEYWORDS if kw in text_lower)
    non_cs_hits = sum(1 for kw in NON_CS if kw in text_lower)
    return cs_hits >= 2 and non_cs_hits == 0


def extract_rstar(max_items=8000):
    print(f"  Extracting rStar-Coder (target: {max_items} CS items)...", flush=True)
    import pyarrow.parquet as pq
    import glob

    sft_files = sorted(glob.glob(os.path.join(RSTAR_DIR, "seed_sft", "*.parquet")))
    items = []
    total_scanned = 0

    for fpath in sft_files:
        if len(items) >= max_items:
            break
        try:
            table = pq.read_table(fpath)
            df = table.to_pandas()
            for _, row in df.iterrows():
                if len(items) >= max_items:
                    break
                total_scanned += 1
                question = str(row.get("question", ""))
                code = str(row.get("code", ""))
                response = str(row.get("response", ""))
                starter = str(row.get("starter_code", ""))

                combined = f"{question} {code} {response}"
                if not is_cs_related(combined):
                    continue

                output = response if len(response) > 50 else code
                if len(output) < 30:
                    continue

                items.append({
                    "instruction": question[:500],
                    "input": starter[:200] if starter else "",
                    "output": output[:1000],
                    "source": "rStar-Coder",
                    "soul": "cezanne",
                })
        except Exception as e:
            print(f"    [WARN] {os.path.basename(fpath)}: {e}", flush=True)

        if total_scanned % 50000 == 0:
            print(f"    Scanned {total_scanned}, collected {len(items)} CS items", flush=True)

    print(f"    Total scanned: {total_scanned}, CS items: {len(items)}", flush=True)
    return items


def extract_opus(max_items=2000):
    print(f"  Extracting Opus-4.6-Reasoning (target: {max_items} CS items)...", flush=True)
    import glob

    items = []
    total_scanned = 0

    for fpath in glob.glob(os.path.join(OPUS_DIR, "*.parquet")):
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(fpath)
            df = table.to_pandas()
            for _, row in df.iterrows():
                total_scanned += 1
                if len(items) >= max_items:
                    break

                text_cols = [str(v) for v in row.values if isinstance(v, str)]
                combined = " ".join(text_cols)

                if not is_cs_related(combined):
                    continue

                instruction = str(row.get("prompt", row.get("question", row.iloc[0] if len(row) > 0 else "")))
                output = str(row.get("response", row.get("answer", row.iloc[1] if len(row) > 1 else "")))

                if len(output) < 50:
                    continue

                items.append({
                    "instruction": instruction[:500],
                    "input": "",
                    "output": output[:1500],
                    "source": "Opus-4.6-Reasoning",
                    "soul": "cezanne",
                })
        except Exception as e:
            print(f"    [WARN] {os.path.basename(fpath)}: {e}", flush=True)

    print(f"    Total scanned: {total_scanned}, CS items: {len(items)}", flush=True)
    return items


def build_shared_memory():
    os.makedirs(MEMORY_DIR, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  BUILDING SHARED MEMORY FOR 7B + 8B CEZANNE", flush=True)
    print(f"  Output: {MEMORY_DIR}", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. rStar-Coder
    rstar_items = extract_rstar(max_items=8000)
    rstar_path = os.path.join(MEMORY_DIR, "rstar_cs_8k.json")
    with open(rstar_path, "w", encoding="utf-8") as f:
        json.dump(rstar_items, f, ensure_ascii=False, indent=2)
    print(f"  Saved rStar: {len(rstar_items)} items → {rstar_path}", flush=True)

    # 2. Opus-4.6
    opus_items = extract_opus(max_items=2000)
    opus_path = os.path.join(MEMORY_DIR, "opus_cs_2k.json")
    with open(opus_path, "w", encoding="utf-8") as f:
        json.dump(opus_items, f, ensure_ascii=False, indent=2)
    print(f"  Saved Opus: {len(opus_items)} items → {opus_path}", flush=True)

    # 3. Self-play trajectories (from existing logs)
    sp_items = []
    for fname in os.listdir(LOG_DIR):
        if fname.startswith("selfplay_iter") and fname.endswith("_trajectory.json"):
            fpath = os.path.join(LOG_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                traj = json.load(f)
            for t in traj:
                if t.get("passed"):
                    sp_items.append({
                        "instruction": t["question"],
                        "input": "",
                        "output": t["answer"],
                        "source": f"selfplay_{t['topic']}",
                        "soul": "cezanne",
                    })
    sp_path = os.path.join(MEMORY_DIR, "selfplay_positive.json")
    with open(sp_path, "w", encoding="utf-8") as f:
        json.dump(sp_items, f, ensure_ascii=False, indent=2)
    print(f"  Saved Self-Play: {len(sp_items)} positive items → {sp_path}", flush=True)

    # 4. Index file
    index = {
        "rstar": {"path": rstar_path, "count": len(rstar_items), "type": "code_reasoning"},
        "opus": {"path": opus_path, "count": len(opus_items), "type": "deep_reasoning"},
        "selfplay": {"path": sp_path, "count": len(sp_items), "type": "self_play_positive"},
        "total_items": len(rstar_items) + len(opus_items) + len(sp_items),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    idx_path = os.path.join(MEMORY_DIR, "index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"  SHARED MEMORY BUILT", flush=True)
    print(f"  rStar: {len(rstar_items)} items", flush=True)
    print(f"  Opus:  {len(opus_items)} items", flush=True)
    print(f"  Self-Play: {len(sp_items)} items", flush=True)
    print(f"  Total: {index['total_items']} items", flush=True)
    print(f"  Index: {idx_path}", flush=True)
    print(f"{'='*60}", flush=True)

    return index


if __name__ == "__main__":
    build_shared_memory()
