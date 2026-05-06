import json, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

faiss_path = r'D:\VORTEX_FLAME\long-memory\global\faiss_index\cs_knowledge.faiss'
meta_path = r'D:\VORTEX_FLAME\long-memory\global\faiss_index\cs_knowledge_meta.json'
ids_path = r'D:\VORTEX_FLAME\long-memory\global\faiss_index\cs_knowledge_ids.json'
mem_path = r'D:\VORTEX_FLAME\long-memory\global\mem.json'

print("=== FAISS Index Files ===")
for p in [faiss_path, meta_path, ids_path]:
    if os.path.exists(p):
        size = os.path.getsize(p)
        print(f"  {os.path.basename(p)}: {size/1024:.1f} KB")
    else:
        print(f"  {os.path.basename(p)}: MISSING!")

meta = json.load(open(meta_path, 'r', encoding='utf-8'))
print(f"\n  Total chunks: {len(meta)}")
cats = {}
for c in meta:
    cats[c['category']] = cats.get(c['category'], 0) + 1
for k, v in sorted(cats.items()):
    print(f"    {k}: {v} chunks")

print(f"\n  Source files:")
sources = set()
for c in meta:
    sources.add(c['source'].split('/')[0] if '/' in c['source'] else c['source'])
for s in sorted(sources):
    count = sum(1 for c in meta if c['source'].startswith(s))
    print(f"    {s}: {count} chunks")

print(f"\n=== mem.json ===")
mem = json.load(open(mem_path, 'r', encoding='utf-8'))
cs_entries = [e for e in mem['entries'] if 'cs_knowledge' in e.get('category', '')]
print(f"  Total entries: {len(mem['entries'])}")
print(f"  CS knowledge entries: {len(cs_entries)}")

print(f"\n=== Sample CS entries ===")
for e in cs_entries[:3]:
    print(f"  [{e['category']}] {e.get('source', 'N/A')[:50]}")
    print(f"    {e['content'][:100]}...")
