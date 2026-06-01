#!/usr/bin/env python3
"""
VORTEX FLAME V3 Knowledge Indexer
Indexes ALL knowledge bases: Logic, Chemistry, Literature, Astronomy,
Earth Science, Biology, History, Philosophy, plus 5 missing soul rules.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from soul_memory import SoulMemoryEngine

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "extended_domain_knowledge_v3.json")

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: Config file not found: {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print("=" * 60)
    print("VORTEX FLAME V3 Knowledge Indexer — Full Science + Humanities")
    print("=" * 60)
    print(f"Loading {len(entries)} knowledge entries...")

    memory = SoulMemoryEngine()
    indexed = 0
    errors = 0
    soul_counts = {}
    module_counts = {}

    for entry in entries:
        try:
            memory.write(
                entry["soul"],
                entry.get("category", "knowledge"),
                {
                    "topic": entry["topic"],
                    "text": entry["text"],
                    "tags": entry["tags"],
                },
                tags=entry["tags"],
            )
            indexed += 1
            soul_counts[entry["soul"]] = soul_counts.get(entry["soul"], 0) + 1

            topic = entry["topic"]
            kb_tag = topic.split("]")[0] + "]" if "]" in topic else "?"
            module_counts[kb_tag] = module_counts.get(kb_tag, 0) + 1

            short = topic[:80]
            print(f"  OK [{entry['soul']:>15}] {short}")
        except Exception as e:
            errors += 1
            print(f"  ERR [{entry['soul']:>15}] {entry['topic'][:60]} — {e}")

    print("-" * 60)
    print(f"Total: {indexed} indexed, {errors} errors\n")
    print("By module:")
    for module, count in sorted(module_counts.items(), key=lambda x: -x[1]):
        print(f"  {module:<30} {count}")
    print("\nBy soul:")
    for soul, count in sorted(soul_counts.items(), key=lambda x: -x[1]):
        print(f"  {soul:>15}: {count}")
    print("=" * 60)
    print("V3 Knowledge base indexing complete.")
    print("Coverage: Logic, Chemistry, Literature, Astronomy, Earth Science,")
    print("          Biology, History, Philosophy + 10 soul domain rules.")
    print("Aligns with UNESCO GB/T13745 7 Basic Sciences + Humanities.")


if __name__ == "__main__":
    main()
