#!/usr/bin/env python3
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

f = r"D:\VORTEX_FLAME\soul_training_data\cezanne_8b\cezanne_pro_8b_stage3a_code_v1.json"
d = json.load(open(f, 'r', encoding='utf-8'))

cats = {}
bad = 0
empty_out = 0
no_code = 0
for x in d:
    c = x.get('_cat', '?')
    cats[c] = cats.get(c, 0) + 1
    out = x.get('output', '')
    inst = x.get('instruction', '')
    inp = x.get('input', '')
    if len(out) < 50:
        bad += 1
    if len(inst) < 5 and len(inp) < 5:
        empty_out += 1
    if '```' not in out and 'def ' not in out and 'class ' not in out:
        no_code += 1

print(f"Stage3a: {len(d)} items")
print(f"Short output (<50 chars): {bad}")
print(f"No instruction/input: {empty_out}")
print(f"No code block: {no_code}")
print(f"\nCategories:")
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")

print(f"\nSample 1 instruction: {d[0].get('instruction', '')[:100]}")
print(f"Sample 1 output ({len(d[0].get('output',''))} chars): {d[0].get('output','')[:150]}")
print(f"\nSample 500 instruction: {d[500].get('instruction','')[:100]}")
print(f"Sample 5000 instruction: {d[5000].get('instruction','')[:100]}")

# Check training config alignment
print(f"\n=== Training config check ===")
print(f"Items: {len(d)}")
print(f"Epochs: 2")
print(f"Batch: 1 x grad_accum=4")
print(f"Steps per epoch: {len(d)//4}")
print(f"Total steps (2 epochs): {len(d)*2//4}")
print(f"Expected: ~3981 steps, VRAM ~11.5GB")
