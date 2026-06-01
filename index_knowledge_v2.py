#!/usr/bin/env python3
"""
VORTEX FLAME V2 Knowledge Indexer
Indexes 6-module private KB rules, control loop, software catalog,
Audio-JEPA modules, hearing KB into soul_memory.
"""

import json
import os
import sys

sys.path.insert(0, str(os.path.dirname(__file__)))
from soul_memory import SoulMemoryEngine

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "extended_domain_knowledge_v2.json")


def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: Config file not found: {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print("=" * 60)
    print("VORTEX FLAME V2 Knowledge Indexer (Private KB Rules)")
    print("=" * 60)
    print(f"Loading {len(entries)} knowledge entries...")

    memory = SoulMemoryEngine()
    indexed = 0
    errors = 0
    soul_counts = {}

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
            print(f"  OK: {entry['topic']}")
        except Exception as e:
            errors += 1
            print(f"  ERR: {entry['topic']} - {e}")

    print("-" * 60)
    print(f"Total: {indexed} indexed, {errors} errors")
    for soul, count in sorted(soul_counts.items(), key=lambda x: -x[1]):
        print(f"  {soul}: {count} entries")
    print("=" * 60)
    print("Knowledge base indexing complete.")


if __name__ == "__main__":
    main()
