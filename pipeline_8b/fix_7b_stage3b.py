#!/usr/bin/env python3
"""Fix 7B Stage3b: re-add all 37 supplements with instruction+input fingerprint"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load 7B stage3b
s3b_path = r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage3b_supplement_v1.json'
with open(s3b_path, 'r', encoding='utf-8') as f:
    s3b = json.load(f)

# Load 8B supplements as source
s8b_path = r'D:\VORTEX_FLAME\soul_training_data\cezanne_8b\cezanne_pro_8b_stage3b_supplement_v1.json'
with open(s8b_path, 'r', encoding='utf-8') as f:
    s8b_data = json.load(f)

# Get all 37 supplements from 8B
target_cats = {'DebugSupplement','LogicSupplement','MathSupplement','SortSupplement','GraphSupplement'}
s8b_sups = [d for d in s8b_data if d.get('_cat','') in target_cats]
print(f"8B has {len(s8b_sups)} supplements total")

# Build fingerprint: instruction[:30] + input[:50] (much more precise)
fingerprints = set()
for item in s3b:
    fp = item.get('instruction','')[:30].lower().strip() + '|||' + item.get('input','')[:50].lower().strip()
    fingerprints.add(fp)

added = 0
skipped = 0
for sup in s8b_sups:
    sup_copy = dict(sup)
    sup_copy['soul'] = 'cezanne'
    fp = sup_copy.get('instruction','')[:30].lower().strip() + '|||' + sup_copy.get('input','')[:50].lower().strip()
    if fp not in fingerprints:
        s3b.append(sup_copy)
        fingerprints.add(fp)
        added += 1
        print(f"  + [{sup_copy.get('_cat','')}] {sup_copy.get('instruction','')[:60]}...")
    else:
        skipped += 1

print(f"\nAdded: {added}, Skipped (true dupes): {skipped}")

# Category summary
cats = {}
for item in s3b:
    c = item.get('_cat', '?')
    cats[c] = cats.get(c, 0) + 1
print(f"\n7B Stage3b final: {len(s3b)} items")
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")

# Save
with open(s3b_path, 'w', encoding='utf-8') as f:
    json.dump(s3b, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {s3b_path}")

# Compare with 8B
s8b_final_cats = {}
for item in s8b_data:
    c = item.get('_cat', '?')
    s8b_final_cats[c] = s8b_final_cats.get(c, 0) + 1
print(f"\n=== FINAL 7B vs 8B Stage3b ===")
print(f"7B: {len(s3b)} items, 8B: {len(s8b_data)} items")
for c in sorted(target_cats):
    n7 = cats.get(c, 0)
    n8 = s8b_final_cats.get(c, 0)
    status = "OK" if n7 == n8 else f"GAP {n8-n7}"
    print(f"  {c:20s}: 7B={n7:3d}  8B={n8:3d}  [{status}]")

# VRAM recalc
steps = len(s3b) * 2 // 4
print(f"\nStage3b steps: ~{steps} (VRAM ~7GB, safe)")
