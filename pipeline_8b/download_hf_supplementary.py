#!/usr/bin/env python3
"""
Download supplementary datasets from HuggingFace for Stage4 reserves:
  1. UltraChat - Chinese high-quality dialogue
  2. Anthropic-HH-RLHF - safety alignment + human preference
  3. Open-Orca/FLAN - smaller OpenOrca subset for reasoning

These are stored as standalone reserves for Stage4 targeted supplementation.
"""
import json, os, random, time
from collections import Counter

OUT_DIR = r"D:\VORTEX_FLAME\soul_training_data\hf_reserves"
CACHE = r"D:\VORTEX_FLAME\soul_training_data\hf_cache"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def download_ultrachat():
    print("=" * 60)
    print("  [1/3] Downloading UltraChat...")
    print("=" * 60)
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", cache_dir=CACHE)
    print(f"  Raw size: {len(ds)}")

    converted = []
    skipped = 0

    for item in ds:
        messages = item.get("messages", [])
        if len(messages) < 2:
            skipped += 1
            continue

        has_chinese = any('\u4e00' <= c <= '\u9fff' for m in messages for c in m.get("content", ""))

        inst = messages[0].get("content", "")
        out = messages[1].get("content", "") if len(messages) > 1 else ""

        if len(out) < 80:
            skipped += 1
            continue

        code_in_dialogue = any("```" in m.get("content", "") or "def " in m.get("content", "")
                              for m in messages)

        if has_chinese:
            cat = "UltraChat_cn_code" if code_in_dialogue else "UltraChat_cn"
        else:
            cat = "UltraChat_en_code" if code_in_dialogue else "UltraChat_en"

        converted.append({
            "instruction": inst,
            "input": "",
            "output": out,
            "source": "UltraChat",
            "soul": "cezanne",
            "_cat": cat,
        })

    print(f"  Total: {len(converted)}, Skipped: {skipped}")
    cats = Counter(d.get("_cat", "") for d in converted)
    print(f"  Categories: {dict(cats)}")

    path = os.path.join(OUT_DIR, "ultrachat_reserve.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path} ({os.path.getsize(path)/1024/1024:.1f}MB)")

    return converted


def download_hh_rlhf():
    print("\n" + "=" * 60)
    print("  [2/3] Downloading Anthropic-HH-RLHF...")
    print("=" * 60)
    from datasets import load_dataset

    converted = []

    for data_dir in ["helpful-base", "harmless-base"]:
        try:
            ds = load_dataset("Anthropic/hh-rlhf", split="train", cache_dir=CACHE,
                            data_dir=data_dir)
            print(f"  {data_dir}: {len(ds)} items")

            for item in ds:
                chosen = item.get("chosen", "")
                if len(chosen) < 100:
                    continue

                lines = chosen.strip().split("\n")
                human_msg = ""
                assistant_msg = ""
                for i, line in enumerate(lines):
                    if line.startswith("Human:"):
                        human_msg = line[6:].strip()
                    elif line.startswith("Assistant:"):
                        assistant_msg = line[10:].strip()
                        for j in range(i + 1, len(lines)):
                            if lines[j].startswith("Human:"):
                                break
                            assistant_msg += "\n" + lines[j].strip()

                if not human_msg or not assistant_msg or len(assistant_msg) < 50:
                    continue

                has_code = "```" in assistant_msg or "def " in assistant_msg or "class " in assistant_msg
                cat = "HH_RLHF_code" if has_code else "HH_RLHF_safe"

                converted.append({
                    "instruction": human_msg,
                    "input": "",
                    "output": assistant_msg,
                    "source": "Anthropic-HH-RLHF",
                    "soul": "cezanne",
                    "_cat": cat,
                })

        except Exception as e:
            print(f"  Error loading {data_dir}: {e}")

    print(f"  Total HH-RLHF: {len(converted)}")
    cats = Counter(d.get("_cat", "") for d in converted)
    print(f"  Categories: {dict(cats)}")

    path = os.path.join(OUT_DIR, "hh_rlhf_reserve.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path} ({os.path.getsize(path)/1024/1024:.1f}MB)")

    return converted


def download_openorca_flan():
    print("\n" + "=" * 60)
    print("  [3/3] Downloading OpenOrca-FLAN subset...")
    print("=" * 60)
    from datasets import load_dataset

    try:
        ds = load_dataset("Open-Orca/FLAN", split="train", cache_dir=CACHE)
        print(f"  Raw size: {len(ds)}")
    except Exception as e:
        print(f"  FLAN subset not available: {e}")
        print("  Trying 1M-GPT4-Augmented instead...")
        try:
            ds = load_dataset("Open-Orca/1M-GPT4-Augmented", split="train", cache_dir=CACHE)
            print(f"  Raw size: {len(ds)}")
        except Exception as e2:
            print(f"  1M-GPT4-Augmented also not available: {e2}")
            print("  Skipping OpenOrca - will use local parquet if available")
            return []

    converted = []
    max_items = 30000

    code_kw = ['def ', 'class ', 'import ', 'function ', '```', 'algorithm', 'sort',
               'search', 'tree', 'graph', 'recursion', 'binary', 'hash']
    math_kw = ['prove', 'theorem', 'equation', 'integral', 'derivative', 'matrix',
               '证明', '定理', '方程', '积分', '导数', '矩阵', '概率', 'logic', '推理']

    garbage_kw = ['电视噪音', '星际迷航', '电视剧', '电影剧情', '游戏攻略', '食谱', '旅游',
                  '星座', 'recipe', 'celebrity', 'gossip']

    for item in ds:
        inst = item.get("question", "") or item.get("instruction", "") or ""
        out = item.get("response", "") or item.get("output", "") or ""

        if len(out) < 100 or not inst.strip():
            continue

        if any(kw in (inst + out).lower() for kw in garbage_kw):
            continue

        text = inst + out
        has_code = any(kw in text for kw in code_kw)
        has_math = any(kw in text.lower() for kw in math_kw)

        if not has_code and not has_math:
            if len(converted) > max_items * 0.3:
                continue

        cat = "OpenOrca_code" if has_code else ("OpenOrca_math" if has_math else "OpenOrca_logic")

        converted.append({
            "instruction": inst,
            "input": "",
            "output": out,
            "source": "OpenOrca",
            "soul": "cezanne",
            "_cat": cat,
        })

        if len(converted) >= max_items:
            break

    print(f"  Collected: {len(converted)}")
    cats = Counter(d.get("_cat", "") for d in converted)
    print(f"  Categories: {dict(cats)}")

    path = os.path.join(OUT_DIR, "openorca_reserve.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {path} ({os.path.getsize(path)/1024/1024:.1f}MB)")

    return converted


def main():
    print("=" * 60)
    print("  HuggingFace Supplementary Dataset Download v2")
    print("  For Stage4 targeted supplementation reserves")
    print("=" * 60)

    ultra_data = download_ultrachat()
    hh_data = download_hh_rlhf()
    orca_data = download_openorca_flan()

    all_data = ultra_data + hh_data + orca_data

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  UltraChat: {len(ultra_data)} items")
    print(f"  HH-RLHF:   {len(hh_data)} items")
    print(f"  OpenOrca:  {len(orca_data)} items")
    print(f"  Total:     {len(all_data)} items")
    print(f"  Location:  {OUT_DIR}")
    print(f"{'='*60}")

    if all_data:
        cats = Counter(d.get("_cat", "") for d in all_data)
        print(f"  All categories: {dict(cats)}")

    print("\n  These are RESERVES for Stage4. Not merged into current training data.")
    print("  Use them when AB comparison + Arena reveal specific weaknesses.")


if __name__ == "__main__":
    main()
