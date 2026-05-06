#!/usr/bin/env python3
"""
Step 1: Prepare Ministral-8B-Reasoning model on D: drive (SSD)
- Copy essential files from E: (HDD) to D: (SSD)
- Only copy consolidated.safetensors + config/tokenizer (skip 4-shard duplicates)
- Verify file integrity
"""
import os, shutil, json, time, hashlib

SRC = r"E:\models\Ministral-8B-Reasoning"
DST = r"D:\models\Ministral-8B-Reasoning"

ESSENTIAL_FILES = [
    "consolidated.safetensors",
    "config.json",
    "generation_config.json",
    "params.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tekken.json",
    "chat_template.jinja",
    "SYSTEM_PROMPT.txt",
]

SKIP_PATTERNS = [
    "model-0000",
    ".cache",
    ".metadata",
    ".gitignore",
    ".gitattributes",
    "CACHEDIR",
    "README",
]


def should_skip(filename):
    for pat in SKIP_PATTERNS:
        if pat in filename:
            return True
    return False


def get_file_md5(filepath, chunk_size=8*1024*1024):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 60)
    print("  8B Model Prep: E: (HDD) -> D: (SSD)")
    print("=" * 60)

    if not os.path.exists(SRC):
        print(f"[FATAL] Source not found: {SRC}")
        return

    os.makedirs(DST, exist_ok=True)

    src_files = os.listdir(SRC)
    copy_list = []
    skip_list = []

    for f in src_files:
        if should_skip(f):
            skip_list.append(f)
            continue
        if f in ESSENTIAL_FILES:
            copy_list.append(f)
        elif f not in ESSENTIAL_FILES and not should_skip(f):
            copy_list.append(f)

    print(f"\n  Source: {SRC}")
    print(f"  Dest:   {DST}")
    print(f"  Files to copy: {len(copy_list)}")
    print(f"  Files to skip: {len(skip_list)}")

    total_size = 0
    for f in copy_list:
        fp = os.path.join(SRC, f)
        if os.path.isfile(fp):
            total_size += os.path.getsize(fp)
    print(f"  Total size: {total_size/1024**3:.1f}GB")

    print(f"\n  Copying...")
    t0 = time.time()
    for i, f in enumerate(copy_list):
        src_fp = os.path.join(SRC, f)
        dst_fp = os.path.join(DST, f)
        if not os.path.isfile(src_fp):
            continue
        fsize_mb = os.path.getsize(src_fp) / 1024 / 1024
        if os.path.exists(dst_fp):
            src_size = os.path.getsize(src_fp)
            dst_size = os.path.getsize(dst_fp)
            if src_size == dst_size:
                print(f"  [{i+1}/{len(copy_list)}] SKIP (exists): {f} ({fsize_mb:.0f}MB)")
                continue
        print(f"  [{i+1}/{len(copy_list)}] Copying: {f} ({fsize_mb:.0f}MB)...", end="", flush=True)
        shutil.copy2(src_fp, dst_fp)
        print(" OK", flush=True)
    elapsed = time.time() - t0
    print(f"\n  Copy done in {elapsed:.0f}s")

    print(f"\n  Verifying integrity...")
    errors = 0
    for f in copy_list:
        src_fp = os.path.join(SRC, f)
        dst_fp = os.path.join(DST, f)
        if not os.path.isfile(src_fp) or not os.path.isfile(dst_fp):
            continue
        src_size = os.path.getsize(src_fp)
        dst_size = os.path.getsize(dst_fp)
        if src_size != dst_size:
            print(f"  [ERROR] Size mismatch: {f} (src={src_size}, dst={dst_size})")
            errors += 1

    if errors == 0:
        print(f"  All files verified OK")
    else:
        print(f"  {errors} errors found!")

    dst_total = sum(
        os.path.getsize(os.path.join(DST, f))
        for f in os.listdir(DST)
        if os.path.isfile(os.path.join(DST, f))
    )
    print(f"\n  D: drive model size: {dst_total/1024**3:.1f}GB")

    config_path = os.path.join(DST, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        text_cfg = cfg.get("text_config", {})
        print(f"\n  Model info:")
        print(f"    Architecture: {cfg.get('architectures', ['?'])[0]}")
        print(f"    Dtype: {cfg.get('dtype', '?')}")
        print(f"    Layers: {text_cfg.get('num_hidden_layers', '?')}")
        print(f"    Hidden: {text_cfg.get('hidden_size', '?')}")
        print(f"    Heads: {text_cfg.get('num_attention_heads', '?')}")
        print(f"    KV Heads: {text_cfg.get('num_key_value_heads', '?')}")
        print(f"    Vocab: {text_cfg.get('vocab_size', '?')}")
        print(f"    Max Pos: {text_cfg.get('max_position_embeddings', '?')}")
        vision_cfg = cfg.get("vision_config", {})
        if vision_cfg:
            print(f"    Vision Layers: {vision_cfg.get('num_hidden_layers', '?')}")
            print(f"    Vision Hidden: {vision_cfg.get('hidden_size', '?')}")

    print("\n" + "=" * 60)
    print("  PREP DONE - Ready for brain surgery analysis")
    print("  Next: python analyze_8b_layers.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
