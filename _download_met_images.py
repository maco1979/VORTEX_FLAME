"""
Download curated high-resolution images from The Met Open Access for C-JEPA visual training.

Strategy:
  - Read MetObjects.csv, filter for CC0 paintings/drawings/sculptures with images
  - Download only high-priority artworks (paintings, drawings, prints, sculpture)
  - Target: ~10,000 images (~50GB at 5MB avg) - enough for C-JEPA fine-tuning
  - Save to E:\\AI_Data\\Met_Images\\ with structured subdirectories

This is for the C-JEPA visual encoder, NOT for soul_memory (that's _index_met.py).
"""

import csv
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

MET_CSV_PATH = r"E:\AI_Data\MetObjects.csv"
OUTPUT_ROOT = r"E:\AI_Data\Met_Images"

TARGET_CLASSIFICATIONS = {
    "Paintings",
    "Drawings",
    "Prints",
    "Sculpture",
}

MAX_IMAGES = 10000
DOWNLOAD_TIMEOUT = 30
MAX_WORKERS = 8
RETRY_COUNT = 2


def get_target_objects():
    if not os.path.exists(MET_CSV_PATH):
        print(f"ERROR: {MET_CSV_PATH} not found. Run _download_met_csv.py first.")
        sys.exit(1)

    targets = []
    with open(MET_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_public = row.get("Is Public Domain", "").strip() == "True"
            has_image = bool(row.get("Primary Image", "").strip())
            obj_class = row.get("Classification", "").strip()

            if not is_public or not has_image:
                continue
            if obj_class not in TARGET_CLASSIFICATIONS:
                continue

            targets.append({
                "object_id": row.get("Object ID", "").strip(),
                "title": row.get("Title", "").strip(),
                "artist": row.get("Artist Display Name", "").strip(),
                "classification": obj_class,
                "image_url": row.get("Primary Image", "").strip(),
                "date": row.get("Object Date", "").strip(),
            })

    return targets[:MAX_IMAGES]


def download_one(obj, output_dir):
    obj_id = obj["object_id"]
    class_dir = os.path.join(output_dir, obj["classification"].replace(" ", "_"))
    os.makedirs(class_dir, exist_ok=True)

    filename = f"{obj_id}.jpg"
    filepath = os.path.join(class_dir, filename)

    if os.path.exists(filepath):
        return obj_id, "exists", 0

    for attempt in range(RETRY_COUNT):
        try:
            urllib.request.urlretrieve(obj["image_url"], filepath)
            size_kb = os.path.getsize(filepath) / 1024
            if size_kb < 5:
                os.remove(filepath)
                return obj_id, "too_small", 0
            return obj_id, "ok", size_kb
        except Exception as e:
            if attempt < RETRY_COUNT - 1:
                time.sleep(1)
            else:
                return obj_id, f"error: {str(e)[:50]}", 0

    return obj_id, "failed", 0


def main():
    print("The Met Image Downloader for C-JEPA Training")
    print("=" * 60)

    targets = get_target_objects()
    print(f"Target objects: {len(targets)}")

    by_class = {}
    for t in targets:
        c = t["classification"]
        by_class[c] = by_class.get(c, 0) + 1
    for c, n in sorted(by_class.items()):
        print(f"  {c}: {n}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    stats = {"ok": 0, "exists": 0, "error": 0, "too_small": 0}
    total_bytes = 0
    completed = 0

    print(f"\nDownloading with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one, obj, OUTPUT_ROOT): obj for obj in targets}

        for future in as_completed(futures):
            obj_id, status, size_kb = future.result()
            stats[status] = stats.get(status, 0) + 1
            if status == "ok":
                total_bytes += size_kb * 1024

            completed += 1
            if completed % 100 == 0:
                print(f"  Progress: {completed}/{len(targets)} "
                      f"(ok={stats['ok']}, exists={stats['exists']}, "
                      f"errors={stats.get('error', 0)}, "
                      f"size={total_bytes/1024/1024:.0f}MB)")

    print("\n" + "=" * 60)
    print("Download Complete")
    print("=" * 60)
    print(f"  Downloaded: {stats['ok']}")
    print(f"  Already existed: {stats['exists']}")
    print(f"  Errors: {stats.get('error', 0)}")
    print(f"  Too small (removed): {stats.get('too_small', 0)}")
    print(f"  Total size: {total_bytes/1024/1024:.0f}MB")
    print(f"  Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
