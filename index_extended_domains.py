#!/usr/bin/env python3
"""
VORTEX FLAME Extended Domain Knowledge Indexer v2
=================================================
Indexes domain-specific knowledge into soul_memory for dual-pathway retrieval.
Loads knowledge entries from JSON config file.

Categories:
  1. SCI_COMPUTE: Scientific computing, AI/ML, GPU, simulation
  2. VISUAL_VIDEO: Vision, cameras, video processing, streaming
  3. DATABASE_ETL: Databases, vector DBs, ETL pipelines
  4. NETWORK_OPS: Network, remote control, SSH, operations
  5. EMBEDDED_IOT: Embedded systems, microcontrollers, sensors
  6. VERTICAL_APPS: GIS, medical, finance, ERP, vertical software
  7. AUDIO_JEPA: Audio-JEPA auditory knowledge (speech, environment, music)
"""

import json
import os
import sys

sys.path.insert(0, str(os.path.dirname(__file__)))
from soul_memory import SoulMemoryEngine

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "extended_domain_knowledge.json")


def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: Config file not found: {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print("=" * 60)
    print("VORTEX FLAME Extended Domain Knowledge Indexer v2")
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
            print(f"  OK: {entry['topic'][:60]}")
        except Exception as e:
            errors += 1
            print(f"  ERR: {entry['topic'][:40]} -> {e}")

    print(f"\n{'=' * 60}")
    print(f"Indexed: {indexed} entries | Errors: {errors}")
    print(f"By soul: {dict(soul_counts)}")
    print("Done!")


if __name__ == "__main__":
    main()
