#!/usr/bin/env python3
"""Direct SQLite bulk importer for V3 knowledge entries — bypasses embedding bottleneck."""
import json, os, sys, sqlite3, time, uuid

CONFIG = "extended_domain_knowledge_v3.json"
if not os.path.exists(CONFIG):
    print(f"ERROR: {CONFIG} not found")
    sys.exit(1)

with open(CONFIG, "r", encoding="utf-8") as f:
    entries = json.load(f)

print(f"Loading {len(entries)} entries via direct SQLite...")

soul_counts = {}
indexed = 0
errors = 0
empty_embedding = b"\x00" * 1536  # placeholder 384-dim * float32

for i, entry in enumerate(entries):
    soul = entry["soul"]
    category = entry.get("category", "knowledge")
    topic = entry.get("topic", "")
    text = entry.get("text", "")
    tags = entry.get("tags", [])

    db_path = f".vf_memory/{soul}.db"
    try:
        conn = sqlite3.connect(db_path)
        entry_id = f"v3_{uuid.uuid4().hex[:12]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        content_json = json.dumps({"topic": topic, "text": text, "tags": tags}, ensure_ascii=False)
        tags_json = json.dumps(tags)
        tags_text = " ".join(tags)

        conn.execute("""
            INSERT OR REPLACE INTO memories (entry_id, soul, category, content, document_date,
                                            importance, tags, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (entry_id, soul, category, content_json, now, 0.5, tags_json, empty_embedding))

        conn.execute("""
            INSERT OR REPLACE INTO memories_fts (entry_id, soul, category, content_text, tags_text)
            VALUES (?, ?, ?, ?, ?)
        """, (entry_id, soul, category, text, tags_text))

        conn.commit()
        conn.close()

        indexed += 1
        soul_counts[soul] = soul_counts.get(soul, 0) + 1
        print(f"  [{i+1}/{len(entries)}] OK [{soul:>15}] {topic[:70]}")

    except Exception as e:
        errors += 1
        try:
            conn.close()
        except:
            pass
        print(f"  [{i+1}/{len(entries)}] ERR [{soul}] {topic[:50]} — {e}")

print("-" * 60)
print(f"Total: {indexed} indexed, {errors} errors\n")
print("By soul:")
for soul, count in sorted(soul_counts.items(), key=lambda x: -x[1]):
    print(f"  {soul:>15}: {count}")
print("=" * 60)
print("V3 Direct Import Complete. Note: embedding vectors are placeholders.")
print("Run 'python -c \"from soul_memory import SoulMemoryEngine; SoulMemoryEngine().rebuild_embeddings()\"' if needed.")
