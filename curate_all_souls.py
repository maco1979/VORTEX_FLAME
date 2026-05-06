import json
import os

BASE = r"D:\VORTEX_FLAME\soul_training_data"
TARGET = 4000

souls_data = {
    "beethoven": "beethoven_55gb_v3.json",
    "cezanne": "cezanne_55gb_v3.json",
    "darwin": "darwin_55gb_v3.json",
    "davinci": "davinci_55gb_v3.json",
    "fkj": "fkj_55gb_v3.json",
    "galileo": "galileo_55gb_v3.json",
    "guizhu": "guizhu_55gb_v3.json",
    "herodotus": "herodotus_55gb_v3.json",
    "humboldt": "humboldt_55gb_v3.json",
    "monet": "monet_55gb_v3.json",
    "montesquieu": "montesquieu_55gb_v3.json",
    "strategy": "strategy_55gb_v3.json",
    "vangogh": "vangogh_55gb_v3.json",
    "yuanlongping": "yuanlongping_55gb_v3.json",
}

for soul, filename in souls_data.items():
    src = os.path.join(BASE, soul, filename)
    dst = os.path.join(BASE, soul, f"{soul}_4k.json")

    if not os.path.exists(src):
        print(f"[SKIP] {soul}: {filename} not found")
        continue

    if os.path.exists(dst):
        print(f"[EXISTS] {soul}: {soul}_4k.json already exists")
        continue

    print(f"[PROCESS] {soul}: loading {filename}...")
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"  Original: {total} samples")

    seen = set()
    deduped = []
    for s in data:
        key = s.get("instruction", "").strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(s)

    print(f"  After dedup: {len(deduped)}")

    filtered = [s for s in deduped if len(s.get("output", "")) >= 50]
    print(f"  After quality filter (output>=50 chars): {len(filtered)}")

    if len(filtered) <= TARGET:
        selected = filtered
    else:
        step = len(filtered) / TARGET
        selected = [filtered[int(i * step)] for i in range(TARGET)]

    print(f"  Selected: {len(selected)}")

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    dst_size = os.path.getsize(dst) / (1024 * 1024)
    print(f"  Saved: {soul}_4k.json ({dst_size:.1f}MB)")
    print()

print("DONE! All souls curated to 4k.")
