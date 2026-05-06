#!/usr/bin/env python3
"""Build FAISS knowledge index from data.jsonl - memory-efficient streaming"""
import sys, os, json, gc
import numpy as np

sys.path.insert(0, "/mnt/d/VORTEX_FLAME")
from long_memory import _knowledge_data_path, _knowledge_index_dir, _get_encoder

soul = "cezanne"
data_path = _knowledge_data_path(soul)
idx_dir = _knowledge_index_dir(soul)

print("Step 1: Streaming read IDs and texts from JSONL...", flush=True)
ids = []
texts = []
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
                entry = json.loads(line)
                ids.append(entry["id"])
                texts.append(entry["content"])
                offsets.append(offset)
            except json.JSONDecodeError:
                pass
        if len(ids) % 100000 == 0 and len(ids) > 0:
            print(f"  Read {len(ids)} entries...", flush=True)

total = len(ids)
print(f"Total entries: {total}", flush=True)

print("Step 2: Encoding texts in batches...", flush=True)
encoder = _get_encoder()
dim = 384
all_vectors = np.zeros((total, dim), dtype="float32")
batch_size = 5000

for i in range(0, total, batch_size):
    batch = texts[i:i + batch_size]
    vecs = encoder.encode(batch, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    all_vectors[i:i + len(batch)] = vecs
    if (i + batch_size) % 50000 == 0 or i + batch_size >= total:
        print(f"  Encoded {min(i + batch_size, total)}/{total}", flush=True)
    del vecs

del texts
gc.collect()

print("Step 3: Building FAISS index...", flush=True)
import faiss
index = faiss.IndexFlatIP(dim)
index.add(all_vectors)
del all_vectors
gc.collect()

print("Step 4: Saving index, IDs and offsets...", flush=True)
faiss.write_index(index, os.path.join(idx_dir, "index.faiss"))
with open(os.path.join(idx_dir, "ids.json"), "w") as f:
    json.dump(ids, f)
with open(os.path.join(idx_dir, "offsets.json"), "w") as f:
    json.dump(offsets, f)

print(f"Done! Index: {total} entries, saved to {idx_dir}", flush=True)
