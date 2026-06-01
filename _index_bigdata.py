import json
import os
import sys
import time

sys.path.insert(0, r"D:\VORTEX_FLAME")
from soul_memory import write

BASE = r"E:\大规模训练集"
BATCH_LOG = 500

STEM_KEYWORDS = {
    "物理": "einstein", "数学": "einstein", "化学": "einstein",
    "生物": "einstein", "天文": "einstein", "医学": "einstein",
    "编程": "cezanne", "代码": "cezanne", "算法": "cezanne",
    "计算机": "cezanne", "软件": "cezanne", "工程": "strategy",
    "哲学": "guizhu", "伦理": "guizhu", "逻辑": "einstein",
    "历史": "guizhu", "文化": "guizhu", "社会": "guizhu",
    "经济": "strategy", "金融": "strategy", "管理": "strategy",
    "法律": "guizhu", "政治": "guizhu", "教育": "guizhu",
    "心理": "guizhu", "艺术": "monet", "音乐": "beethoven",
    "设计": "davinci", "文学": "guizhu",
}

def classify_soul(instruction, output, domain=None):
    text = f"{instruction} {output}".lower()
    if domain:
        if isinstance(domain, list):
            text += " " + " ".join(str(d) for d in domain)
        else:
            text += " " + str(domain)
    best_soul = "guizhu"
    best_count = 0
    for kw, soul in STEM_KEYWORDS.items():
        c = text.count(kw)
        if c > best_count:
            best_count = c
            best_soul = soul
    return best_soul

def index_alpaca():
    print("=" * 60)
    print("  Indexing Alpaca中文指令 (48K)")
    print("=" * 60)
    path = os.path.join(BASE, "Alpaca中文指令", "alpaca_gpt4_data_zh.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total: {len(data)} records")
    count = 0
    for item in data:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        if not inst or not out:
            continue
        soul = classify_soul(inst, out)
        content = {
            "topic": f"Alpaca: {inst[:80]}",
            "source": "alpaca_gpt4_zh",
            "instruction": inst[:2000],
            "input": inp[:1000] if inp else "",
            "output": out[:3000],
        }
        tags = ["alpaca", "chinese_instruction", soul]
        write(soul, "knowledge", content, importance=0.5, tags=tags)
        count += 1
        if count % BATCH_LOG == 0:
            print(f"  Alpaca: {count}/{len(data)}")
    print(f"  Alpaca done: {count} indexed")

def index_coig():
    print("=" * 60)
    print("  Indexing COIG-CQIA (44K)")
    print("=" * 60)
    path = os.path.join(BASE, "COIG中文指令", "COIG-CQIA-full.jsonl")
    count = 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            item = json.loads(line)
            inst = item.get("instruction", "").strip()
            out = item.get("output", "").strip()
            if not inst or not out:
                continue
            domain = item.get("domain", None)
            soul = classify_soul(inst, out, domain)
            content = {
                "topic": f"COIG: {inst[:80]}",
                "source": "coig_cqia",
                "instruction": inst[:2000],
                "input": item.get("input", "")[:1000],
                "output": out[:3000],
                "task_type": str(item.get("task_type", ""))[:200],
                "domain": str(domain)[:200] if domain else "",
            }
            tags = ["coig", "chinese_instruction", soul]
            write(soul, "knowledge", content, importance=0.55, tags=tags)
            count += 1
            if count % BATCH_LOG == 0:
                print(f"  COIG: {count}/{total} (reading...)")
    print(f"  COIG done: {count}/{total} indexed")

def index_coig_subdirs():
    subdirs = {
        "chinese_traditional": {"soul": "guizhu", "importance": 0.7},
        "coig_pc": {"soul": "einstein", "importance": 0.6},
        "douban": {"soul": "guizhu", "importance": 0.5},
    }
    for subdir, cfg in subdirs.items():
        d = os.path.join(BASE, "COIG中文指令", subdir)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith(".jsonl"):
                continue
            path = os.path.join(d, fname)
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    inst = item.get("instruction", item.get("input", "")).strip()
                    out = item.get("output", "").strip()
                    if not inst or not out:
                        continue
                    content = {
                        "topic": f"COIG/{subdir}: {inst[:80]}",
                        "source": f"coig_{subdir}",
                        "instruction": inst[:2000],
                        "output": out[:3000],
                    }
                    tags = ["coig", subdir, cfg["soul"]]
                    write(cfg["soul"], "knowledge", content, importance=cfg["importance"], tags=tags)
                    count += 1
            print(f"  COIG/{subdir}/{fname}: {count} indexed")

def index_zhihu():
    print("=" * 60)
    print("  Indexing 知乎高质量")
    print("=" * 60)
    d = os.path.join(BASE, "知乎高质量", "data")
    if not os.path.isdir(d):
        print("  Directory not found")
        return
    import pyarrow.parquet as pq
    files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
    count = 0
    for fname in files:
        t = pq.read_table(os.path.join(d, fname))
        cols = t.column_names
        for i in range(len(t)):
            row = {}
            for c in cols:
                v = t.column(c)[i].as_py()
                row[c] = str(v)[:3000] if v else ""
            text = " ".join(str(row.get(c, "")) for c in cols)
            soul = classify_soul(text, "")
            content = {
                "topic": f"知乎: {text[:80]}",
                "source": "zhihu_hq",
            }
            for c in cols:
                content[c] = row[c]
            tags = ["zhihu", "chinese_qa", soul]
            write(soul, "knowledge", content, importance=0.55, tags=tags)
            count += 1
            if count % BATCH_LOG == 0:
                print(f"  知乎: {count}")
        print(f"  Shard {fname}: done")
    print(f"  知乎 done: {count} indexed")

def index_belle_sample():
    print("=" * 60)
    print("  Indexing Belle中文指令 (sampled 50K from 2M)")
    print("=" * 60)
    path = os.path.join(BASE, "Belle_中文指令", "train_2M_CN.json")
    count = 0
    sampled = 0
    SAMPLE_EVERY = 40
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % SAMPLE_EVERY != 0:
                continue
            item = json.loads(line)
            inst = item.get("instruction", "").strip()
            out = item.get("output", "").strip()
            if not inst or not out:
                continue
            soul = classify_soul(inst, out)
            content = {
                "topic": f"Belle: {inst[:80]}",
                "source": "belle_2m_cn_sampled",
                "instruction": inst[:2000],
                "input": item.get("input", "")[:1000],
                "output": out[:3000],
            }
            tags = ["belle", "chinese_instruction", soul]
            write(soul, "knowledge", content, importance=0.45, tags=tags)
            sampled += 1
            if sampled % BATCH_LOG == 0:
                print(f"  Belle: {sampled} sampled (at line {i})")
            if sampled >= 50000:
                break
    print(f"  Belle done: {sampled} indexed (sampled from {count}+)")

def index_fineweb_filtered():
    print("=" * 60)
    print("  Indexing FinewebEdu_filtered (pre-sorted by soul)")
    print("=" * 60)
    d = os.path.join(BASE, "FinewebEdu_filtered")
    souls = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
    total = 0
    for soul in souls:
        sd = os.path.join(d, soul)
        files = sorted(f for f in os.listdir(sd) if f.endswith(".json"))
        for fname in files:
            path = os.path.join(sd, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                inst = item.get("instruction", "").strip()
                out = item.get("output", "").strip()
                if not inst or not out:
                    continue
                content = {
                    "topic": f"FinewebEdu: {inst[:80]}",
                    "source": "fineweb_edu_filtered",
                    "instruction": inst[:2000],
                    "output": out[:3000],
                    "original_soul": item.get("soul", soul),
                    "matched_keyword": item.get("matched_keyword", ""),
                }
                tags = ["fineweb_edu", soul]
                write(soul, "knowledge", content, importance=0.5, tags=tags)
                total += 1
            print(f"  {soul}/{fname}: {len(data)} records")
    print(f"  FinewebEdu_filtered done: {total} indexed")

if __name__ == "__main__":
    t0 = time.time()
    index_alpaca()
    index_coig()
    index_coig_subdirs()
    index_zhihu()
    index_belle_sample()
    index_fineweb_filtered()
    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.0f}s")
