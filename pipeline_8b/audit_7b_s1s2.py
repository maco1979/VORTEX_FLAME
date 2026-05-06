#!/usr/bin/env python3
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

s1 = json.load(open(r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage1_math_8k_v3.json', 'r', encoding='utf-8'))
s2 = json.load(open(r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage2_logic_8k_v3.json', 'r', encoding='utf-8'))

s1_cats = {}
for x in s1:
    c = x.get('_cat', '?')
    s1_cats[c] = s1_cats.get(c, 0) + 1
s2_cats = {}
for x in s2:
    c = x.get('_cat', '?')
    s2_cats[c] = s2_cats.get(c, 0) + 1

print("=== 7B 训练内容全貌 ===")
print()
print(f"Stage1 (数学打底): {len(s1)} 条")
for c, n in sorted(s1_cats.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")

print(f"\nStage2 (逻辑突破): {len(s2)} 条")
for c, n in sorted(s2_cats.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")

# Get 3 sample from each
print("\n=== Stage1 样本 (3条) ===")
for i in [0, 100, 5000]:
    out = s1[i].get('output', '')
    inst = s1[i].get('instruction', '')[:80]
    print(f"[{i}] cat={s1[i].get('_cat','?')} | {inst}... | out_len={len(out)}")

print("\n=== Stage2 样本 (3条) ===")
for i in [0, 50, min(200, len(s2)-1)]:
    out = s2[i].get('output', '')
    inst = s2[i].get('instruction', '')[:80]
    print(f"[{i}] cat={s2[i].get('_cat','?')} | {inst}... | out_len={len(out)}")

# Stage3 existing (not split)
s3 = json.load(open(r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage3_fusion_8k_v7.json', 'r', encoding='utf-8'))
s3_cats = {}
for x in s3:
    c = x.get('_cat', '?')
    s3_cats[c] = s3_cats.get(c, 0) + 1
print(f"\nStage3 (现有原始): {len(s3)} 条")
for c, n in sorted(s3_cats.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")

# Split versions
s3a = json.load(open(r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage3a_code_v1.json', 'r', encoding='utf-8'))
s3b = json.load(open(r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage3b_supplement_v1.json', 'r', encoding='utf-8'))
s3b_cats = {}
for x in s3b:
    c = x.get('_cat', '?')
    s3b_cats[c] = s3b_cats.get(c, 0) + 1
print(f"\nStage3a (CodeAlpaca): {len(s3a)} 条")
print(f"Stage3b (补充+反遗忘): {len(s3b)} 条")
for c, n in sorted(s3b_cats.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")

# Summary comparison with 8B
s3b_8b = json.load(open(r'D:\VORTEX_FLAME\soul_training_data\cezanne_8b\cezanne_pro_8b_stage3b_supplement_v1.json', 'r', encoding='utf-8'))
s3b_8b_cats = {}
for x in s3b_8b:
    c = x.get('_cat', '?')
    s3b_8b_cats[c] = s3b_8b_cats.get(c, 0) + 1
print(f"\n=== 7B vs 8B Stage3b 对比 ===")
print(f"7B: {len(s3b)} 条, 8B: {len(s3b_8b)} 条")
all_cats = set(list(s3b_cats.keys()) + list(s3b_8b_cats.keys()))
for c in sorted(all_cats):
    n7 = s3b_cats.get(c, 0)
    n8 = s3b_8b_cats.get(c, 0)
    diff = n8 - n7
    marker = " <-- 8B多" if diff > 0 else ""
    print(f"  {c:20s}: 7B={n7:4d}  8B={n8:4d}{marker}")
