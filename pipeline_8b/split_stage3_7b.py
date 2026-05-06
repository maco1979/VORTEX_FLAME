#!/usr/bin/env python3
"""Split 7B Cezanne Stage3 data + add 37 targeted supplements + dedup"""
import json, os

S3_PATH = r"D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage3_fusion_8k_v7.json"
OUT_DIR = r"D:\VORTEX_FLAME\soul_training_data\cezanne"

print("Loading 7B Stage3 data...")
with open(S3_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Split
s3a = []  # CodeAlpaca
s3b = []  # Everything else
for item in data:
    if item.get("_cat") == "CodeAlpaca":
        s3a.append(item)
    else:
        s3b.append(item)

print(f"Stage3a (CodeAlpaca): {len(s3a)} items")
print(f"Stage3b (rest): {len(s3b)} items")

# Load same 37 supplements from 8B data and add to 7B
s8b_path = r"D:\VORTEX_FLAME\soul_training_data\cezanne_8b\cezanne_pro_8b_stage3b_supplement_v1.json"
print(f"\nLoading 8B supplements as template...")
with open(s8b_path, "r", encoding="utf-8") as f:
    s8b_data = json.load(f)

sups = [d for d in s8b_data if d.get("_cat", "") in (
    "DebugSupplement","LogicSupplement","MathSupplement","SortSupplement","GraphSupplement"
)]
print(f"Found {len(sups)} targeted supplements in 8B data")

# Dedup: track existing instruction fingerprints
fingerprints = set()
for item in s3b:
    fp = item.get("instruction", "")[:60].lower().strip()
    fingerprints.add(fp)

new_count = 0
for sup in sups:
    sup_copy = dict(sup)
    sup_copy["soul"] = "cezanne"  # Change to 7B
    fp = sup_copy.get("instruction", "")[:60].lower().strip()
    if fp not in fingerprints:
        s3b.append(sup_copy)
        fingerprints.add(fp)
        new_count += 1

print(f"Added {new_count} new supplements (dedup applied)")

# Full dedup check
all_fps = {}
dupes = 0
for i, item in enumerate(s3b):
    fp = item.get("instruction", "")[:50].lower().strip()
    if fp in all_fps:
        dupes += 1
    else:
        all_fps[fp] = i

if dupes == 0:
    print(f"Dedup: PASS (0 duplicates in {len(s3b)} items)")
else:
    print(f"Dedup: {dupes} potential dupes (instruction fingerprints)")

# Category summary
cats = {}
for item in s3b:
    c = item.get("_cat", "?")
    cats[c] = cats.get(c, 0) + 1
print(f"\nStage3b categories ({len(s3b)} total):")
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")

# Save
s3a_path = os.path.join(OUT_DIR, "cezanne_stage3a_code_v1.json")
s3b_path = os.path.join(OUT_DIR, "cezanne_stage3b_supplement_v1.json")

with open(s3a_path, "w", encoding="utf-8") as f:
    json.dump(s3a, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {s3a_path} ({len(s3a)} items, {os.path.getsize(s3a_path)/1024/1024:.1f}MB)")

with open(s3b_path, "w", encoding="utf-8") as f:
    json.dump(s3b, f, ensure_ascii=False, indent=2)
print(f"Saved: {s3b_path} ({len(s3b)} items, {os.path.getsize(s3b_path)/1024/1024:.1f}MB)")

# VRAM estimate
s3a_steps = len(s3a) * 2 // 4
s3b_steps = len(s3b) * 2 // 4
print(f"\n=== VRAM ESTIMATE ===")
print(f"7B Stage3a: {len(s3a)} items x 2 epochs / 4 = ~{s3a_steps} steps → VRAM ~7GB")
print(f"7B Stage3b: {len(s3b)} items x 2 epochs / 4 = ~{s3b_steps} steps → VRAM ~7GB")
print(f"7B is lighter: Mistral-7B ~4.5GB + LoRA + KV = ~7GB (safe on 12GB)")
