import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

d = json.load(open(r'D:\VORTEX_FLAME\long-memory\global\faiss_index\cs_knowledge_meta.json', 'r', encoding='utf-8'))
print(f'Total chunks: {len(d)}')
cats = {}
for c in d:
    cats[c['category']] = cats.get(c['category'], 0) + 1
print('By category:')
for k, v in sorted(cats.items()):
    print(f'  {k}: {v}')

print('\nSample chunks:')
for c in d[:5]:
    print(f'  [{c["category"]}] {c["source"][:60]}... ({c["char_count"]} chars)')

print('\nSample content:')
for c in d[:2]:
    print(f'  [{c["category"]}] {c["content"][:200]}')
    print()

import os
faiss_path = r'D:\VORTEX_FLAME\long-memory\global\faiss_index\cs_knowledge.faiss'
print(f'FAISS file size: {os.path.getsize(faiss_path)/1024:.1f} KB')
