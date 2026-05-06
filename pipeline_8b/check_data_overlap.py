import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

s2_path = r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage2_logic_8k_v3.json'
s3_path = r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage3_cs_v1.json'
s1_path = r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage1_math_8k_v3.json'

for name, path in [("S1", s1_path), ("S2", s2_path), ("S3_CS", s3_path)]:
    try:
        d = json.load(open(path, 'r', encoding='utf-8'))
        print(f"\n{name}: {len(d)}条")
        print(f"  前5条instruction:")
        for i, s in enumerate(d[:5]):
            inst = s.get('instruction', '')[:100]
            print(f"    {i}: {inst}")
        if len(d) > 10:
            print(f"  中间5条:")
            for i, s in enumerate(d[len(d)//2:len(d)//2+5]):
                inst = s.get('instruction', '')[:100]
                print(f"    {len(d)//2+i}: {inst}")
    except Exception as e:
        print(f"\n{name}: ERROR - {e}")

s2 = json.load(open(s2_path, 'r', encoding='utf-8'))
s3 = json.load(open(s3_path, 'r', encoding='utf-8'))

s2_instructions = set(s.get('instruction', '')[:50] for s in s2)
s3_instructions = set(s.get('instruction', '')[:50] for s in s3)
overlap = s2_instructions & s3_instructions
print(f"\n去重检查: S2与S3_CS重叠 = {len(overlap)}条")
if overlap:
    for o in list(overlap)[:5]:
        print(f"  重叠: {o}")
