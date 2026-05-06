#!/usr/bin/env python3
"""Build offset index for fast line lookup in data.jsonl"""
import sys, os, json
sys.path.insert(0, "/mnt/d/VORTEX_FLAME")
from long_memory import _knowledge_data_path, _knowledge_index_dir

soul = "cezanne"
data_path = _knowledge_data_path(soul)
idx_dir = _knowledge_index_dir(soul)
offsets_path = os.path.join(idx_dir, "offsets.json")

print("Building offset index...", flush=True)
offsets = []
with open(data_path, "r", encoding="utf-8") as f:
    while True:
        offset = f.tell()
        line = f.readline()
        if not line:
            break
        line = line.strip()
        if line:
            try:
                json.loads(line)
                offsets.append(offset)
            except json.JSONDecodeError:
                pass
        if len(offsets) % 100000 == 0 and len(offsets) > 0:
            print(f"  {len(offsets)} entries...", flush=True)

print(f"Total offsets: {len(offsets)}", flush=True)
with open(offsets_path, "w") as f:
    json.dump(offsets, f)
print(f"Saved to {offsets_path}", flush=True)
