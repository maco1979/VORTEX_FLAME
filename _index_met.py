"""
Index The Met Open Access collection into art soul knowledge bases.

Reads MetObjects.csv (317MB, 470K+ records) from E:\\AI_Data,
filters for CC0 public-domain artworks with images, and indexes
structured metadata into monet/vangogh/davinci domain_memory.

Strategy:
  - Only CC0 objects with image URLs (skip copyrighted/no-image)
  - Prioritize paintings, drawings, prints, sculptures
  - Batch write with rate limiting to avoid memory pressure
  - Smart sampling: full index for paintings/drawings, sampled for others
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soul_memory import write

MET_CSV_PATH = os.path.join(os.getenv("AI_DATA", r"E:\AI_Data"), "MetObjects.csv")
BATCH_SIZE = 500

TARGET_SOULS = {
    "davinci": {"importance": 0.8, "tag_prefix": "met_art"},
    "monet": {"importance": 0.75, "tag_prefix": "met_art"},
    "vangogh": {"importance": 0.75, "tag_prefix": "met_art"},
}

CATEGORY = "domain_memory"

PRIORITY_CLASSIFICATIONS = {
    "Paintings",
    "Drawings",
    "Prints",
    "Sculpture",
    "Photographs",
    "Ceramics",
    "Textiles",
    "Metalwork",
}

SECONDARY_CLASSIFICATIONS = {
    "Arms and Armor",
    "Costume",
    "Musical Instruments",
    "Glass",
    "Woodwork",
    "Ivory",
    "Manuscripts",
    "Illustrated Books",
}

SKIP_CLASSIFICATIONS = {
    "Coins and Medals",
    "Seals",
    "Stamps",
}


def check_csv():
    if os.path.exists(MET_CSV_PATH):
        size_mb = os.path.getsize(MET_CSV_PATH) / (1024 * 1024)
        print(f"MetObjects.csv found ({size_mb:.1f}MB)")
        return MET_CSV_PATH
    print(f"ERROR: MetObjects.csv not found at {MET_CSV_PATH}")
    print("Run _download_met_csv.py first")
    sys.exit(1)


def classify_priority(row):
    obj_class = row.get("Classification", "").strip()
    if obj_class in PRIORITY_CLASSIFICATIONS:
        return "high"
    if obj_class in SECONDARY_CLASSIFICATIONS:
        return "medium"
    if obj_class in SKIP_CLASSIFICATIONS:
        return "skip"
    return "low"


def build_content(row, priority):
    title = row.get("Title", "").strip()
    artist = row.get("Artist Display Name", "").strip()
    artist_nationality = row.get("Artist Nationality", "").strip()
    date = row.get("Object Date", "").strip()
    obj_class = row.get("Classification", "").strip()
    medium = row.get("Medium", "").strip()
    dimensions = row.get("Dimensions", "").strip()
    department = row.get("Department", "").strip()
    culture = row.get("Culture", "").strip()
    period = row.get("Period", "").strip()
    dynasty = row.get("Dynasty", "").strip()
    image_url = row.get("Primary Image", "").strip()
    object_url = row.get("Object URL", "").strip()
    object_id = row.get("Object ID", "").strip()
    gallery = row.get("Gallery Number", "").strip()
    credit = row.get("Credit Line", "").strip()

    description_parts = []
    if title:
        description_parts.append(f"Title: {title}")
    if artist:
        description_parts.append(f"Artist: {artist}")
    if artist_nationality:
        description_parts.append(f"Nationality: {artist_nationality}")
    if date:
        description_parts.append(f"Date: {date}")
    if obj_class:
        description_parts.append(f"Classification: {obj_class}")
    if medium:
        description_parts.append(f"Medium: {medium}")
    if dimensions:
        description_parts.append(f"Dimensions: {dimensions}")
    if culture:
        description_parts.append(f"Culture: {culture}")
    if period:
        description_parts.append(f"Period: {period}")
    if dynasty:
        description_parts.append(f"Dynasty: {dynasty}")
    if gallery:
        description_parts.append(f"Gallery: {gallery}")
    if credit:
        description_parts.append(f"Credit: {credit}")

    content = {
        "topic": f"Met: {title}" if title else f"Met Object {object_id}",
        "source": "met_open_access",
        "met_object_id": object_id,
        "title": title,
        "artist": artist,
        "date": date,
        "classification": obj_class,
        "medium": medium,
        "dimensions": dimensions,
        "department": department,
        "culture": culture,
        "period": period,
        "description": "\n".join(description_parts),
        "image_url": image_url,
        "object_url": object_url,
        "priority": priority,
    }

    return content


def main():
    csv_path = check_csv()

    print(f"\nParsing MetObjects.csv...")
    rows_by_priority = {"high": [], "medium": [], "low": [], "skip": []}
    total_rows = 0
    cc0_with_image = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            is_public = row.get("Is Public Domain", "").strip() == "True"
            has_image = bool(row.get("Primary Image", "").strip())

            if not is_public or not has_image:
                continue

            cc0_with_image += 1
            priority = classify_priority(row)
            rows_by_priority[priority].append(row)

    print(f"Total records: {total_rows}")
    print(f"CC0 with image: {cc0_with_image}")
    print(f"  High priority (paintings/drawings/prints/sculpture): {len(rows_by_priority['high'])}")
    print(f"  Medium priority (costume/instruments/glass): {len(rows_by_priority['medium'])}")
    print(f"  Low priority (other): {len(rows_by_priority['low'])}")
    print(f"  Skip (coins/stamps): {len(rows_by_priority['skip'])}")

    to_index = rows_by_priority["high"] + rows_by_priority["medium"]

    low_cap = 5000
    if len(rows_by_priority["low"]) > low_cap:
        step = len(rows_by_priority["low"]) // low_cap
        sampled_low = rows_by_priority["low"][::step][:low_cap]
        to_index.extend(sampled_low)
        print(f"  Low priority sampled: {len(sampled_low)} / {len(rows_by_priority['low'])}")
    else:
        to_index.extend(rows_by_priority["low"])

    print(f"\nTotal to index: {len(to_index)} objects -> {len(TARGET_SOULS)} souls")

    stats = {soul: {"indexed": 0, "errors": 0} for soul in TARGET_SOULS}
    batch_count = 0

    for idx, row in enumerate(to_index):
        priority = classify_priority(row)
        try:
            content = build_content(row, priority)
            obj_tags = ["met_museum", content["classification"].lower().replace(" ", "_") if content["classification"] else "unclassified"]

            if content.get("artist"):
                artist_tag = content["artist"].lower().replace(" ", "_")[:50]
                obj_tags.append(artist_tag)

            if content.get("culture"):
                obj_tags.append(content["culture"].lower().replace(" ", "_")[:30])

            for soul, cfg in TARGET_SOULS.items():
                try:
                    entry_id = write(
                        soul=soul,
                        category=CATEGORY,
                        content=content,
                        importance=cfg["importance"] if priority == "high" else cfg["importance"] - 0.1,
                        tags=[cfg["tag_prefix"]] + obj_tags,
                    )
                    stats[soul]["indexed"] += 1
                except Exception as e:
                    stats[soul]["errors"] += 1

        except Exception as e:
            for soul in TARGET_SOULS:
                stats[soul]["errors"] += 1

        batch_count += 1
        if batch_count % BATCH_SIZE == 0:
            print(f"  Progress: {batch_count}/{len(to_index)} objects processed")
            time.sleep(0.1)

        if batch_count % 5000 == 0:
            time.sleep(2)

    print("\n" + "=" * 60)
    print("The Met Open Access Indexing Complete")
    print("=" * 60)
    for soul, s in stats.items():
        print(f"  {soul}: {s['indexed']} indexed, {s['errors']} errors")
    total = sum(s["indexed"] for s in stats.values())
    print(f"  TOTAL: {total} entries across {len(TARGET_SOULS)} souls")
    print(f"  Source: {len(to_index)} Met objects (from {total_rows} total records)")


if __name__ == "__main__":
    main()
