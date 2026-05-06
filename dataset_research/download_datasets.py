import os
import sys
import json
import time
import hashlib
import argparse

sys.path.insert(0, r"D:\VORTEX_FLAME")
from soul_dataset_map import SOUL_DATASET_MAP, RECOMMENDED_DOWNLOAD_ORDER, ALL_SOUL_NAMES, SOULS_WITHOUT_DATASETS

DATA_ROOT = r"E:\VORTEX_FLAME_DATA"
HF_CACHE_DIR = r"E:\VORTEX_FLAME_DATA\hf_cache"
MEMORY_ROOT = r"E:\VORTEX_FLAME_DATA\long-memory"

os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HOME"] = HF_CACHE_DIR

def get_knowledge_dir(soul):
    return os.path.join(MEMORY_ROOT, soul, "knowledge")

def get_data_jsonl_path(soul):
    return os.path.join(get_knowledge_dir(soul), "data.jsonl")

def get_ids_path(soul):
    return os.path.join(get_knowledge_dir(soul), "ids.txt")

def load_existing_ids(soul):
    id_path = get_ids_path(soul)
    if os.path.exists(id_path):
        with open(id_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def append_id(eid, soul):
    id_path = get_ids_path(soul)
    os.makedirs(os.path.dirname(id_path), exist_ok=True)
    with open(id_path, "a", encoding="utf-8") as f:
        f.write(eid + "\n")

def write_entry(soul, category, content, metadata=None, entry_id=None):
    if entry_id is None:
        entry_id = hashlib.md5(f"{category}:{content[:200]}:{time.time()}".encode()).hexdigest()[:12]
    entry = {
        "id": entry_id,
        "category": category,
        "content": content,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": metadata or {},
    }
    data_path = get_data_jsonl_path(soul)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    append_id(entry_id, soul)
    return entry_id

def download_and_ingest(soul_name, dataset_key, max_rows=None, dry_run=False):
    soul_info = SOUL_DATASET_MAP.get(soul_name)
    if not soul_info:
        print(f"[ERROR] Unknown soul: {soul_name}")
        return 0
    ds_info = soul_info["datasets"].get(dataset_key)
    if not ds_info:
        print(f"[ERROR] Unknown dataset '{dataset_key}' for soul '{soul_name}'")
        return 0

    hf_id = ds_info["hf_id"]
    download_cmd = ds_info.get("download_cmd", "")
    
    if download_cmd == "LOCAL" or hf_id.startswith("E:/"):
        local_path = hf_id.replace("/", "\\")
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing LOCAL dataset for {soul_name}...")
        print(f"  Local path: {local_path}")
        print(f"  Knowledge store: {get_knowledge_dir(soul_name)}")
        
        if not os.path.exists(local_path):
            print(f"  [WARN] Local path does not exist: {local_path}")
            return 0
        
        if dry_run:
            print(f"  Would process local files from {local_path}")
            return 0
        
        existing_ids = load_existing_ids(soul_name)
        count = 0
        skipped = 0
        
        for fname in sorted(os.listdir(local_path)):
            fpath = os.path.join(local_path, fname)
            if not os.path.isfile(fpath):
                continue
            if fname.startswith(".") or fname == "README.md" or fname.endswith(".metadata"):
                continue
            
            try:
                if fname.endswith(".jsonl"):
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                row = json.loads(line)
                                content = json.dumps(row, ensure_ascii=False)[:2000]
                            except:
                                content = line.strip()[:2000]
                            
                            if len(content) < 20:
                                skipped += 1
                                continue
                            eid = hashlib.md5(f"{local_path}:{count}:{content[:200]}".encode()).hexdigest()[:12]
                            if eid in existing_ids:
                                skipped += 1
                                continue
                            write_entry(soul_name, f"dataset:{dataset_key}", content, {"source": local_path, "file": fname}, entry_id=eid)
                            count += 1
                            if count % 1000 == 0:
                                print(f"  Written {count} entries from {fname}...")
                
                elif fname.endswith(".json"):
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for i, row in enumerate(data):
                            content = json.dumps(row, ensure_ascii=False)[:2000]
                            if len(content) < 20:
                                skipped += 1
                                continue
                            eid = hashlib.md5(f"{local_path}:{i}:{content[:200]}".encode()).hexdigest()[:12]
                            if eid in existing_ids:
                                skipped += 1
                                continue
                            write_entry(soul_name, f"dataset:{dataset_key}", content, {"source": local_path, "file": fname}, entry_id=eid)
                            count += 1
                            if count % 1000 == 0:
                                print(f"  Written {count} entries from {fname}...")
                    elif isinstance(data, dict):
                        content = json.dumps(data, ensure_ascii=False)[:2000]
                        eid = hashlib.md5(f"{local_path}:0:{content[:200]}".encode()).hexdigest()[:12]
                        if eid not in existing_ids:
                            write_entry(soul_name, f"dataset:{dataset_key}", content, {"source": local_path, "file": fname}, entry_id=eid)
                            count += 1
                
                elif fname.endswith(".parquet"):
                    try:
                        import pyarrow.parquet as pq
                        table = pq.read_table(fpath)
                        for i in range(min(len(table), 10000)):
                            row = table.slice(i, 1).to_pydict()
                            content = json.dumps({k: str(v[0])[:500] for k, v in row.items()}, ensure_ascii=False)[:2000]
                            if len(content) < 20:
                                skipped += 1
                                continue
                            eid = hashlib.md5(f"{local_path}:{i}:{content[:200]}".encode()).hexdigest()[:12]
                            if eid in existing_ids:
                                skipped += 1
                                continue
                            write_entry(soul_name, f"dataset:{dataset_key}", content, {"source": local_path, "file": fname}, entry_id=eid)
                            count += 1
                            if count % 1000 == 0:
                                print(f"  Written {count} entries from {fname}...")
                    except Exception as e:
                        print(f"  [WARN] Failed to read parquet {fname}: {e}")
            except Exception as e:
                print(f"  [WARN] Failed to process {fname}: {e}")
        
        print(f"  Done: {count} entries written, {skipped} skipped from local files")
        return count

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Downloading {hf_id} for {soul_name}...")
    print(f"  HF cache: {HF_CACHE_DIR}")
    print(f"  Knowledge store: {get_knowledge_dir(soul_name)}")

    if dry_run:
        print(f"  Would download ~{ds_info['estimated_size_gb']}GB")
        print(f"  Priority: {ds_info['priority']}")
        print(f"  Reason: {ds_info['reason']}")
        return 0

    existing_ids = load_existing_ids(soul_name)
    print(f"  Existing IDs in {soul_name} knowledge: {len(existing_ids)}")

    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] `datasets` library not installed. Run: pip install datasets")
        return 0

    try:
        ds = load_dataset(hf_id, split="train", streaming=True, cache_dir=HF_CACHE_DIR)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return 0

    count = 0
    skipped = 0
    for row in ds:
        if max_rows and count >= max_rows:
            break

        content_parts = []
        for key, val in row.items():
            if key == "images":
                continue
            if isinstance(val, str) and len(val) > 10:
                content_parts.append(f"{key}: {val[:2000]}")
            elif isinstance(val, list) and len(val) > 0:
                if isinstance(val[0], str):
                    content_parts.append(f"{key}: {' | '.join(str(v)[:500] for v in val[:5])}")
                elif isinstance(val[0], dict):
                    content_parts.append(f"{key}: {json.dumps(val[:2], ensure_ascii=False)[:1000]}")

        content = "\n".join(content_parts)
        if len(content) < 20:
            skipped += 1
            continue

        eid = hashlib.md5(f"{hf_id}:{count}:{content[:200]}".encode()).hexdigest()[:12]
        if eid in existing_ids:
            skipped += 1
            continue

        metadata = {
            "source": hf_id,
            "dataset": dataset_key,
            "row_index": count,
        }

        write_entry(soul_name, f"dataset:{dataset_key}", content, metadata, entry_id=eid)
        count += 1

        if count % 1000 == 0:
            print(f"  Written {count} entries (skipped {skipped})...")

    print(f"  Done: {count} entries written, {skipped} skipped")
    return count

def batch_download(priority_filter=None, soul_filter=None, max_per_dataset=None, dry_run=False):
    total = 0
    for soul_name, dataset_key, priority, reason in RECOMMENDED_DOWNLOAD_ORDER:
        if priority_filter and priority != priority_filter:
            continue
        if soul_filter and soul_name != soul_filter:
            continue
        print(f"\n{'='*60}")
        print(f"[{priority}] {soul_name}/{dataset_key}: {reason}")
        n = download_and_ingest(soul_name, dataset_key, max_rows=max_per_dataset, dry_run=dry_run)
        total += n
    print(f"\n{'='*60}")
    print(f"Total entries written: {total}")
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets and ingest into soul knowledge stores")
    parser.add_argument("--soul", type=str, help=f"Filter by soul name ({'/'.join(ALL_SOUL_NAMES)})")
    parser.add_argument("--dataset", type=str, help="Specific dataset key to download")
    parser.add_argument("--priority", type=str, choices=["HIGH", "MEDIUM", "LOW"], help="Filter by priority")
    parser.add_argument("--max-rows", type=int, help="Max rows per dataset")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    parser.add_argument("--batch", action="store_true", help="Download all matching datasets in recommended order")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    args = parser.parse_args()

    if args.list:
        print(f"Data root: {DATA_ROOT}")
        print(f"HF cache:  {HF_CACHE_DIR}")
        print(f"Memory:    {MEMORY_ROOT}")
        print(f"\n14 Souls ({len(ALL_SOUL_NAMES)}):")
        for s in ALL_SOUL_NAMES:
            info = SOUL_DATASET_MAP[s]
            ds_count = len(info.get("datasets", {}))
            tier = info.get("tier", "?")
            print(f"  [{tier}] {s} ({info['name']}) - {info['domain']} - {ds_count} datasets")
        print(f"\nSouls without datasets ({len(SOULS_WITHOUT_DATASETS)}): {', '.join(SOULS_WITHOUT_DATASETS)}")
        print(f"\nAll datasets:")
        for soul_name, info in SOUL_DATASET_MAP.items():
            for ds_key, ds_info in info["datasets"].items():
                print(f"  {soul_name}/{ds_key} [{ds_info['priority']}] ~{ds_info['estimated_size_gb']}GB - {ds_info['reason'][:80]}")
    elif args.soul and args.dataset:
        download_and_ingest(args.soul, args.dataset, max_rows=args.max_rows, dry_run=args.dry_run)
    elif args.batch:
        batch_download(
            priority_filter=args.priority,
            soul_filter=args.soul,
            max_per_dataset=args.max_rows,
            dry_run=args.dry_run,
        )
    else:
        print("Usage:")
        print("  python download_datasets.py --list")
        print("  python download_datasets.py --soul einstein --dataset MathNet --dry-run")
        print("  python download_datasets.py --batch --priority HIGH --dry-run")
        print("  python download_datasets.py --batch --soul cezanne --max-rows 5000")
