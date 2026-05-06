#!/usr/bin/env python3
"""
Stage3 v7 = v5 deduped + v6 parameterized
Best of both: v5's complete code + v6's diversity
"""
import json, os, random
from collections import Counter

OUT_7B = r"D:\VORTEX_FLAME\soul_training_data\cezanne"
OUT_8B = r"D:\VORTEX_FLAME\soul_training_data\cezanne_8b"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        data = data['data']
    return data

def dedup(data):
    seen = set()
    result = []
    for item in data:
        key = (item.get('instruction', '')[:80], item.get('output', '')[:200])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result

# Load v5 and dedup
print("Loading v5...")
v5 = load_json(os.path.join(OUT_7B, "cezanne_stage3_fusion_8k_v5.json"))
v5_deduped = dedup(v5)
print(f"  v5: {len(v5)} -> deduped: {len(v5_deduped)}")

# Load v6
print("Loading v6...")
v6 = load_json(os.path.join(OUT_7B, "cezanne_stage3_fusion_8k_v6.json"))
print(f"  v6: {len(v6)}")

# Merge and dedup again
merged = v5_deduped + v6
merged_deduped = dedup(merged)
print(f"  Merged: {len(merged)} -> deduped: {len(merged_deduped)}")

# Shuffle
random.seed(3407)
random.shuffle(merged_deduped)

# Stats
total = len(merged_deduped)
has_code = sum(1 for d in merged_deduped if '```' in d.get('output', ''))
cats = Counter(d.get('_cat', 'unknown') for d in merged_deduped)

print(f"\n=== Stage3 v7 Final ===")
print(f"  Total: {total}")
print(f"  Code blocks: {has_code} ({has_code/total*100:.1f}%)")
print(f"  Categories:")
for k, v in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"    {k}: {v} ({v/total*100:.1f}%)")

# Save 7B
s7_path = os.path.join(OUT_7B, "cezanne_stage3_fusion_8k_v7.json")
with open(s7_path, 'w', encoding='utf-8') as f:
    json.dump(merged_deduped, f, ensure_ascii=False, indent=2)
print(f"\n  Saved 7B: {s7_path}")

# Save 8B
s8_data = []
for d in merged_deduped:
    nd = dict(d)
    nd['soul'] = 'cezanne_pro'
    s8_data.append(nd)

ex_path = os.path.join(OUT_8B, "cezanne_pro_8b_exclusive_code_deep.json")
if os.path.exists(ex_path):
    exclusive = load_json(ex_path)
    # Dedup exclusive too
    ex_deduped = dedup(exclusive)
    print(f"  8B exclusive: {len(exclusive)} -> deduped: {len(ex_deduped)}")
    s8_data.extend(ex_deduped)

s8_path = os.path.join(OUT_8B, "cezanne_pro_8b_stage3_fusion_8k_v7.json")
with open(s8_path, 'w', encoding='utf-8') as f:
    json.dump(s8_data, f, ensure_ascii=False, indent=2)
print(f"  Saved 8B: {s8_path} ({len(s8_data)} items)")

# Also dedup Stage1/2 supplement data
print("\n=== Stage1/2 Supplement Dedup ===")
s1 = load_json(os.path.join(OUT_7B, "cezanne_stage1_math_8k_v2.json"))
s1_deduped = dedup(s1)
print(f"  Stage1 v2: {len(s1)} -> deduped: {len(s1_deduped)}")

s2 = load_json(os.path.join(OUT_7B, "cezanne_stage2_logic_8k_v2.json"))
s2_deduped = dedup(s2)
print(f"  Stage2 v2: {len(s2)} -> deduped: {len(s2_deduped)}")

# Save deduped versions
s1_path = os.path.join(OUT_7B, "cezanne_stage1_math_8k_v3.json")
with open(s1_path, 'w', encoding='utf-8') as f:
    json.dump(s1_deduped, f, ensure_ascii=False, indent=2)
print(f"  Saved Stage1 v3: {s1_path}")

s2_path = os.path.join(OUT_7B, "cezanne_stage2_logic_8k_v3.json")
with open(s2_path, 'w', encoding='utf-8') as f:
    json.dump(s2_deduped, f, ensure_ascii=False, indent=2)
print(f"  Saved Stage2 v3: {s2_path}")

# 8B versions
for src_name, src_data, dst_name in [
    ("Stage1", s1_deduped, "cezanne_pro_8b_stage1_math_8k_v2.json"),
    ("Stage2", s2_deduped, "cezanne_pro_8b_stage2_logic_8k_v2.json"),
]:
    d8 = []
    for d in src_data:
        nd = dict(d)
        nd['soul'] = 'cezanne_pro'
        d8.append(nd)
    p = os.path.join(OUT_8B, dst_name)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(d8, f, ensure_ascii=False, indent=2)
    print(f"  Saved 8B {src_name}: {p} ({len(d8)} items)")

print("\n  Done! All datasets deduped and saved.")
