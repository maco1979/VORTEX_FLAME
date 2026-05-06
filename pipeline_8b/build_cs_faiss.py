#!/usr/bin/env python3
"""Build FAISS index for CS Knowledge Base
Reads all .md files from knowledge_base/cs_knowledge/
Chunks them, embeds with sentence-transformers, stores in FAISS
"""
import os, json, sys, hashlib
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

KNOWLEDGE_DIR = r"/mnt/d/VORTEX_FLAME/knowledge_base/cs_knowledge"
FAISS_DIR = r"/mnt/d/VORTEX_FLAME/long-memory/global/faiss_index"
MEM_FILE = r"/mnt/d/VORTEX_FLAME/long-memory/global/mem.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) + 2 > chunk_size and current:
            chunks.append(current.strip())
            if overlap > 0:
                words = current.split()
                overlap_text = ' '.join(words[-overlap // 4:]) if len(words) > overlap // 4 else ""
                current = overlap_text + "\n\n" + p if overlap_text else p
            else:
                current = p
        else:
            current = current + "\n\n" + p if current else p
    if current.strip():
        chunks.append(current.strip())
    return chunks


def load_all_md_files(base_dir):
    all_chunks = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if not f.endswith('.md'):
                continue
            fpath = os.path.join(root, f)
            rel_path = os.path.relpath(fpath, base_dir)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as fh:
                    text = fh.read()
            except Exception:
                continue
            if len(text) < 50:
                continue

            source = rel_path.replace('\\', '/')
            if 'CSAPP' in source:
                category = "CSAPP"
            elif 'CS149' in source:
                category = "CS149"
            elif '6.S081' in source or 'MIT6' in source:
                category = "MIT6.S081"
            else:
                category = "CS"

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.sha256(f"{source}:{i}".encode()).hexdigest()[:12]
                all_chunks.append({
                    "id": chunk_id,
                    "source": source,
                    "category": category,
                    "chunk_index": i,
                    "content": chunk,
                    "char_count": len(chunk),
                })
    return all_chunks


def build_faiss_with_sentence_transformers(chunks):
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        print("  Loading sentence-transformers model...", flush=True)
        model = SentenceTransformer('all-MiniLM-L6-v2')

        print(f"  Encoding {len(chunks)} chunks...", flush=True)
        texts = [c["content"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}", flush=True)
        return index, embeddings

    except ImportError:
        print("  sentence-transformers not available, using TF-IDF fallback...", flush=True)
        return build_faiss_tfidf_fallback(chunks)


def build_faiss_tfidf_fallback(chunks):
    import faiss
    import numpy as np
    from collections import Counter
    import math

    print("  Building TF-IDF vectors...", flush=True)
    all_words = set()
    doc_freqs = Counter()
    tokenized = []

    for c in chunks:
        words = c["content"].lower().split()
        tokenized.append(words)
        unique_words = set(words)
        all_words.update(unique_words)
        for w in unique_words:
            doc_freqs[w] += 1

    vocab = sorted(all_words)[:10000]
    word2idx = {w: i for i, w in enumerate(vocab)}
    dim = len(vocab)
    n_docs = len(chunks)

    embeddings = np.zeros((n_docs, dim), dtype=np.float32)
    for i, words in enumerate(tokenized):
        tf = Counter(words)
        for w, count in tf.items():
            if w in word2idx:
                j = word2idx[w]
                idf = math.log(n_docs / (1 + doc_freqs[w]))
                embeddings[i, j] = (count / len(words)) * idf if words else 0

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"  TF-IDF FAISS index built: {index.ntotal} vectors, dim={dim}", flush=True)
    return index, embeddings


def search(index, chunks, query, top_k=5):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        q_vec = model.encode([query], show_progress_bar=False)
    except ImportError:
        q_vec = None

    if q_vec is None:
        import numpy as np
        from collections import Counter
        import math
        words = query.lower().split()
        tf = Counter(words)
        dim = index.d
        q_vec = np.zeros((1, dim), dtype=np.float32)
        for w, count in tf.items():
            if hasattr(search, '_vocab') and w in search._vocab:
                j = search._vocab[w]
                idf = math.log(index.ntotal / (1 + 1))
                q_vec[0, j] = (count / len(words)) * idf if words else 0
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

    import numpy as np
    faiss = __import__('faiss')
    q_vec = np.array(q_vec, dtype=np.float32)
    faiss.normalize_L2(q_vec)
    scores, ids = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0 and idx < len(chunks):
            results.append({
                "score": float(score),
                "source": chunks[idx]["source"],
                "category": chunks[idx]["category"],
                "content": chunks[idx]["content"][:300],
            })
    return results


def main():
    print(f"{'='*60}", flush=True)
    print(f"  CS Knowledge Base FAISS Index Builder", flush=True)
    print(f"  Source: {KNOWLEDGE_DIR}", flush=True)
    print(f"  Target: {FAISS_DIR}", flush=True)
    print(f"{'='*60}", flush=True)

    print("\n  Step 1: Loading Markdown files...", flush=True)
    chunks = load_all_md_files(KNOWLEDGE_DIR)
    print(f"  Loaded: {len(chunks)} chunks from knowledge_base/cs_knowledge/", flush=True)

    cat_counts = {}
    for c in chunks:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count} chunks", flush=True)

    print("\n  Step 2: Building FAISS index...", flush=True)
    index, embeddings = build_faiss_with_sentence_transformers(chunks)

    print("\n  Step 3: Saving FAISS index...", flush=True)
    os.makedirs(FAISS_DIR, exist_ok=True)
    import faiss
    faiss.write_index(index, os.path.join(FAISS_DIR, "cs_knowledge.faiss"))

    ids_path = os.path.join(FAISS_DIR, "cs_knowledge_ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump([c["id"] for c in chunks], f)

    meta_path = os.path.join(FAISS_DIR, "cs_knowledge_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"  Saved: cs_knowledge.faiss ({index.ntotal} vectors)", flush=True)
    print(f"  Saved: cs_knowledge_ids.json", flush=True)
    print(f"  Saved: cs_knowledge_meta.json", flush=True)

    print("\n  Step 4: Updating global mem.json...", flush=True)
    if os.path.exists(MEM_FILE):
        with open(MEM_FILE, "r", encoding="utf-8") as f:
            mem = json.load(f)
    else:
        mem = {"entries": []}

    existing_ids = {e["id"] for e in mem.get("entries", [])}
    for c in chunks:
        if c["id"] not in existing_ids:
            mem["entries"].append({
                "id": c["id"],
                "category": f"cs_knowledge_{c['category']}",
                "content": c["content"][:500],
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "timestamp": __import__('time').strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {
                    "source_file": c["source"],
                    "category": c["category"],
                    "char_count": c["char_count"],
                    "faiss_index": "cs_knowledge.faiss",
                },
            })

    with open(MEM_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)
    print(f"  Updated: {MEM_FILE} ({len(mem['entries'])} total entries)", flush=True)

    print("\n  Step 5: Test search...", flush=True)
    test_queries = [
        "virtual memory page table TLB",
        "binary search bug off-by-one",
        "CUDA thread block grid",
        "malloc free heap memory",
        "TCP congestion control slow start",
    ]
    for q in test_queries:
        results = search(index, chunks, q, top_k=2)
        if results:
            print(f"  Q: '{q}'", flush=True)
            print(f"    Top: [{results[0]['category']}] {results[0]['source']} (score={results[0]['score']:.3f})", flush=True)
        else:
            print(f"  Q: '{q}' → No results", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  DONE! CS Knowledge Base indexed: {len(chunks)} chunks", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
