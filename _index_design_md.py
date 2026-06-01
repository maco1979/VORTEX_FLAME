import os
import sys
import json
import time

sys.path.insert(0, r"D:\VORTEX_FLAME")
from soul_memory import write, SoulMemoryEngine

DESIGN_MD_ROOT = r"D:\VORTEX_FLAME\design_specs\awesome-design-md\design-md"

TARGET_SOULS = {
    "davinci": {"importance": 0.8, "tag_prefix": "design_system"},
    "monet": {"importance": 0.7, "tag_prefix": "visual_design"},
    "vangogh": {"importance": 0.7, "tag_prefix": "visual_design"},
}

CATEGORY = "domain_memory"

def parse_design_md(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    front_matter = {}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            body = parts[2].strip()
            for line in fm_text.split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    front_matter[k.strip()] = v.strip().strip('"')

    brand_name = front_matter.get("name", "").replace("-design-analysis", "")
    if not brand_name:
        brand_name = os.path.basename(os.path.dirname(filepath))

    colors = []
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("primary:") or line.startswith("  primary:"):
            if "#" in line:
                colors.append(line.strip())

    return {
        "brand": brand_name,
        "front_matter": front_matter,
        "body_length": len(body),
        "colors_sample": colors[:5],
    }, body

def main():
    brands = sorted(os.listdir(DESIGN_MD_ROOT))
    print(f"Found {len(brands)} brand directories")

    stats = {soul: {"indexed": 0, "errors": 0} for soul in TARGET_SOULS}

    for brand in brands:
        design_path = os.path.join(DESIGN_MD_ROOT, brand, "DESIGN.md")
        if not os.path.exists(design_path):
            continue

        try:
            meta, body = parse_design_md(design_path)
            brand_name = meta["brand"]

            content = {
                "topic": f"{brand_name} Design System",
                "source": "awesome-design-md",
                "brand": brand_name,
                "design_spec": body[:8000],
                "colors": meta["colors_sample"],
                "type": "design_tokens",
            }

            if len(body) > 8000:
                content["design_spec_extended"] = body[8000:16000]
            if len(body) > 16000:
                content["design_spec_remaining"] = body[16000:]

            for soul, cfg in TARGET_SOULS.items():
                try:
                    entry_id = write(
                        soul=soul,
                        category=CATEGORY,
                        content=content,
                        importance=cfg["importance"],
                        tags=[cfg["tag_prefix"], brand_name, "design_system", "ui_tokens"],
                    )
                    stats[soul]["indexed"] += 1
                except Exception as e:
                    stats[soul]["errors"] += 1
                    print(f"  ERROR [{soul}/{brand}]: {e}")

            print(f"  [{brand}] → 3 souls indexed ({meta['body_length']} chars)")

        except Exception as e:
            print(f"  PARSE ERROR [{brand}]: {e}")
            for soul in TARGET_SOULS:
                stats[soul]["errors"] += 1

    print("\n" + "=" * 60)
    print("DESIGN.md Indexing Complete")
    print("=" * 60)
    for soul, s in stats.items():
        print(f"  {soul}: {s['indexed']} indexed, {s['errors']} errors")
    total = sum(s["indexed"] for s in stats.values())
    print(f"  TOTAL: {total} entries across {len(TARGET_SOULS)} souls")

if __name__ == "__main__":
    main()
