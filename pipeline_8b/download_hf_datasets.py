#!/usr/bin/env python3
"""
Download MathInstruct + CodeAlpaca-20K from HuggingFace
Filter for quality, convert to our format, merge into training data
"""
import json, os, random, time
from collections import Counter

OUT_7B = r"D:\VORTEX_FLAME\soul_training_data\cezanne"
OUT_8B = r"D:\VORTEX_FLAME\soul_training_data\cezanne_8b"
CACHE = r"D:\VORTEX_FLAME\soul_training_data\hf_cache"

os.makedirs(CACHE, exist_ok=True)


def download_mathinstruct():
    print("=" * 60)
    print("  [1/2] Downloading MathInstruct...")
    print("=" * 60)
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MathInstruct", split="train", cache_dir=CACHE)
    print(f"  Raw size: {len(ds)}")

    converted = []
    skipped_no_code = 0
    skipped_short = 0
    skipped_garbage = 0

    garbage_kw = ['电视噪音', '星际迷航', '电视剧', '电影剧情', '游戏攻略', '食谱', '旅游', '星座']

    for item in ds:
        inst = item.get("instruction", "") or ""
        inp = item.get("input", "") or ""
        out = item.get("output", "") or ""

        if len(out) < 80:
            skipped_short += 1
            continue

        if not inst.strip():
            continue

        if any(kw in (inst + out) for kw in garbage_kw):
            skipped_garbage += 1
            continue

        has_code = "```" in out or "def " in out or "class " in out or "import " in out

        converted.append({
            "instruction": inst,
            "input": inp,
            "output": out,
            "source": "MathInstruct",
            "soul": "cezanne",
            "_cat": "MathInstruct_code" if has_code else "MathInstruct_logic",
        })
        if not has_code:
            skipped_no_code += 1

    print(f"  Converted: {len(converted)}")
    print(f"  Skipped: short={skipped_short}, garbage={skipped_garbage}, no_code={skipped_no_code}")

    code_items = [d for d in converted if d["_cat"] == "MathInstruct_code"]
    logic_items = [d for d in converted if d["_cat"] == "MathInstruct_logic"]
    print(f"  With code: {len(code_items)}, Logic only: {len(logic_items)}")

    return converted


def download_codealpaca():
    print("\n" + "=" * 60)
    print("  [2/2] Downloading CodeAlpaca-20K...")
    print("=" * 60)
    from datasets import load_dataset
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", cache_dir=CACHE)
    print(f"  Raw size: {len(ds)}")

    converted = []
    skipped_no_code = 0
    skipped_short = 0
    skipped_garbage = 0

    garbage_kw = ['电视噪音', '星际迷航', '电视剧', '电影剧情', '游戏攻略', '食谱', '旅游', '星座']

    for item in ds:
        inst = item.get("instruction", "") or ""
        inp = item.get("input", "") or ""
        out = item.get("output", "") or ""

        if len(out) < 50:
            skipped_short += 1
            continue

        if not inst.strip():
            continue

        if any(kw in (inst + out) for kw in garbage_kw):
            skipped_garbage += 1
            continue

        has_code = "```" in out or "def " in out or "class " in out or "import " in out or "function " in out

        if not has_code:
            skipped_no_code += 1
            continue

        converted.append({
            "instruction": inst,
            "input": inp,
            "output": out,
            "source": "CodeAlpaca",
            "soul": "cezanne",
            "_cat": "CodeAlpaca",
        })

    print(f"  Converted: {len(converted)}")
    print(f"  Skipped: short={skipped_short}, garbage={skipped_garbage}, no_code={skipped_no_code}")

    return converted


def dedup(data):
    seen = set()
    result = []
    for item in data:
        key = (item.get("instruction", "")[:80], item.get("output", "")[:200])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def merge_into_existing(existing_path, new_items, label):
    if os.path.exists(existing_path):
        with open(existing_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict) and "data" in existing:
            existing = existing["data"]
    else:
        existing = []

    merged = existing + new_items
    merged_deduped = dedup(merged)
    random.shuffle(merged_deduped)

    with open(existing_path, "w", encoding="utf-8") as f:
        json.dump(merged_deduped, f, ensure_ascii=False, indent=2)

    print(f"  {label}: {len(existing)} existing + {len(new_items)} new -> {len(merged_deduped)} (deduped)")
    return merged_deduped


def main():
    math_data = download_mathinstruct()
    code_data = download_codealpaca()

    all_new = math_data + code_data
    all_new_deduped = dedup(all_new)

    print(f"\n{'='*60}")
    print(f"  Total new data: {len(all_new)} -> deduped: {len(all_new_deduped)}")

    has_code = sum(1 for d in all_new_deduped if "```" in d.get("output", "") or "def " in d.get("output", "") or "class " in d.get("output", ""))
    print(f"  Code rate: {has_code}/{len(all_new_deduped)} ({has_code/len(all_new_deduped)*100:.1f}%)")

    cats = Counter(d.get("_cat", "unknown") for d in all_new_deduped)
    print(f"  Categories: {dict(cats)}")

    # Save standalone
    standalone_path = os.path.join(OUT_7B, "hf_supplement.json")
    with open(standalone_path, "w", encoding="utf-8") as f:
        json.dump(all_new_deduped, f, ensure_ascii=False, indent=2)
    print(f"  Saved standalone: {standalone_path}")

    # Merge into Stage1 (math items)
    math_items = [d for d in all_new_deduped if d["_cat"].startswith("MathInstruct")]
    code_items = [d for d in all_new_deduped if d["_cat"] == "CodeAlpaca"]

    print(f"\n  MathInstruct items to merge into Stage1: {len(math_items)}")
    print(f"  CodeAlpaca items to merge into Stage3: {len(code_items)}")

    # Merge MathInstruct into Stage1
    print(f"\n  Merging MathInstruct into Stage1...")
    s1_7b = merge_into_existing(
        os.path.join(OUT_7B, "cezanne_stage1_math_8k_v3.json"),
        math_items, "7B Stage1"
    )

    # Merge CodeAlpaca into Stage3
    print(f"\n  Merging CodeAlpaca into Stage3...")
    s3_7b = merge_into_existing(
        os.path.join(OUT_7B, "cezanne_stage3_fusion_8k_v7.json"),
        code_items, "7B Stage3"
    )

    # 8B versions
    print(f"\n  Creating 8B versions...")
    math_8b = []
    for d in math_items:
        nd = dict(d)
        nd["soul"] = "cezanne_pro"
        math_8b.append(nd)

    code_8b = []
    for d in code_items:
        nd = dict(d)
        nd["soul"] = "cezanne_pro"
        code_8b.append(nd)

    s1_8b = merge_into_existing(
        os.path.join(OUT_8B, "cezanne_pro_8b_stage1_math_8k_v2.json"),
        math_8b, "8B Stage1"
    )

    s3_8b = merge_into_existing(
        os.path.join(OUT_8B, "cezanne_pro_8b_stage3_fusion_8k_v7.json"),
        code_8b, "8B Stage3"
    )

    # Final stats
    print(f"\n{'='*60}")
    print(f"  FINAL DATASET SIZES")
    print(f"  7B Stage1: {len(s1_7b)}")
    print(f"  7B Stage3: {len(s3_7b)}")
    print(f"  8B Stage1: {len(s1_8b)}")
    print(f"  8B Stage3: {len(s3_8b)}")
    print(f"{'='*60}")
    print("  Done!")


if __name__ == "__main__":
    main()
