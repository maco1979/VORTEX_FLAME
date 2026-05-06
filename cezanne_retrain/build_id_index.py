#!/usr/bin/env python3
import sys, json
sys.path.insert(0, "/mnt/d/VORTEX_FLAME")
from long_memory import _knowledge_data_path, _knowledge_dir
import os

soul = "cezanne"
data_path = _knowledge_data_path(soul)
id_path = os.path.join(_knowledge_dir(soul), "ids.txt")

print(f"Reading {data_path}...")
ids = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                entry = json.loads(line)
                ids.append(entry["id"])
            except json.JSONDecodeError:
                pass

print(f"Found {len(ids)} IDs, writing to {id_path}...")
with open(id_path, "w") as f:
    for eid in ids:
        f.write(eid + "\n")

print(f"Done! {len(ids)} IDs indexed.")
