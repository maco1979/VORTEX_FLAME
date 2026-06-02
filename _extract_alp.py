"OGG提取 — 全量64包 + MD5去重"
import os, struct, time, hashlib

ALP_DIR = os.getenv("ALP_DIR", "./alp_packages")
OUTPUT_DIR = os.getenv("ALP_OUTPUT", "./ableton_samples")
DEDUP_FILE = os.path.join(OUTPUT_DIR, "_dedup_hashes.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

seen_hashes = set()
if os.path.exists(DEDUP_FILE):
    with open(DEDUP_FILE, "r") as f:
        seen_hashes = set(line.strip() for line in f if line.strip())
    print(f"已加载 {len(seen_hashes)} 个已知hash")

alp_files = sorted([f for f in os.listdir(ALP_DIR) if f.lower().endswith(".alp")])
print(f"共 {len(alp_files)} 个包")

t0 = time.time()
total_new = 0
total_dup = 0
total_size_mb = 0

for i, alp_name in enumerate(alp_files):
    alp_path = os.path.join(ALP_DIR, alp_name)
    pack_name = os.path.splitext(alp_name)[0]
    size_mb = os.path.getsize(alp_path) / 1024 / 1024

    if size_mb < 2:
        continue

    t1 = time.time()
    with open(alp_path, "rb") as f:
        data = f.read()

    ogg_positions = []
    pos = 0
    while True:
        pos = data.find(b"OggS", pos)
        if pos == -1:
            break
        ogg_positions.append(pos)
        pos += 4

    streams = {}
    for p in ogg_positions:
        if p + 27 > len(data):
            continue
        n_seg = data[p + 26]
        if p + 27 + n_seg > len(data):
            continue
        seg_table = data[p + 27:p + 27 + n_seg]
        total = 27 + n_seg + sum(seg_table)
        if p + total > len(data) or total < 27:
            continue
        serial = struct.unpack_from("<I", data, p + 14)[0]
        if serial not in streams:
            streams[serial] = []
        streams[serial].append((p, total))

    new_here = 0
    dup_here = 0
    sz_here = 0
    for serial, pages in streams.items():
        if len(pages) < 2:
            continue
        pages.sort()

        chunk = b""
        for pg_start, pg_size in pages:
            chunk += data[pg_start:pg_start + pg_size]

        h = hashlib.md5(chunk).hexdigest()
        if h in seen_hashes:
            dup_here += 1
            continue

        seen_hashes.add(h)
        out_path = os.path.join(OUTPUT_DIR, f"abl_{h[:12]}.ogg")
        with open(out_path, "wb") as fout:
            fout.write(chunk)
        new_here += 1
        sz_here += len(chunk)

    del data
    dt = time.time() - t1
    sz_extracted_mb = sz_here / (1024 * 1024)
    total_new += new_here
    total_dup += dup_here
    total_size_mb += sz_extracted_mb

    if new_here > 0 or dup_here > 0:
        dup_str = f" (+{dup_here}重复)" if dup_here else ""
        print(f"  [{i+1}/{len(alp_files)}] {pack_name[:55]}: {new_here} new{dup_str} ({sz_extracted_mb:.1f}MB) {dt:.1f}s")
    else:
        print(f"  [{i+1}/{len(alp_files)}] {pack_name[:55]}: 无")

total_time = time.time() - t0
print(f"\n总计: {total_new} 新文件, {total_dup} 重复跳过, {total_size_mb:.1f}MB, {total_time:.0f}s -> {OUTPUT_DIR}")
