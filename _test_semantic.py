import sys, json, time
sys.path.insert(0, r"D:\VORTEX_FLAME")

from soul_memory import recall, search

print("=" * 80)
print("Semantic Retrieval Quality Test")
print("=" * 80)

tests = [
    ("einstein", "quantum entanglement and Bell inequalities", "Physics"),
    ("einstein", "black hole information paradox", "Physics"),
    ("beethoven", "harmonic analysis of musical composition", "Music"),
    ("beethoven", "audio signal processing and spectral features", "Audio"),
    ("cezanne", "graph neural network architecture design", "CS"),
    ("darwin", "evolutionary dynamics and population genetics", "Biology"),
    ("strategy", "Nash equilibrium in game theory", "Economics"),
    ("vangogh", "color theory and visual perception", "Art"),
    ("guizhu", "meditation and mindfulness philosophy", "Philosophy"),
    ("humboldt", "climate change and atmospheric physics", "Environment"),
]

for soul, query, domain in tests:
    t0 = time.time()
    
    results_bm25 = search(soul, "knowledge", query, top_k=3, alpha=1.0)
    
    results_hybrid = search(soul, "knowledge", query, top_k=3, alpha=0.6)
    
    results_semantic = search(soul, "knowledge", query, top_k=3, alpha=0.0)
    
    elapsed = time.time() - t0
    
    print(f"\n[{soul}] Query: '{query}' ({domain}) [{elapsed*1000:.0f}ms]")
    
    print(f"  BM25-only (alpha=1.0):")
    for r in results_bm25[:2]:
        c = json.loads(r["content"]) if isinstance(r["content"], str) else r["content"]
        print(f"    [{r.get('score', 0):.3f}] {c.get('topic', 'N/A')[:70]}")
    
    print(f"  Hybrid (alpha=0.6):")
    for r in results_hybrid[:2]:
        c = json.loads(r["content"]) if isinstance(r["content"], str) else r["content"]
        print(f"    [{r.get('score', 0):.3f}] {c.get('topic', 'N/A')[:70]}")
    
    print(f"  Semantic-only (alpha=0.0):")
    for r in results_semantic[:2]:
        c = json.loads(r["content"]) if isinstance(r["content"], str) else r["content"]
        print(f"    [{r.get('score', 0):.3f}] {c.get('topic', 'N/A')[:70]}")

print("\n" + "=" * 80)
print("Cross-soul recall test (same query, different souls)")
print("=" * 80)

cross_query = "neural network optimization"
for soul in ["cezanne", "einstein", "beethoven", "strategy"]:
    results = recall(soul, cross_query, top_k=2, alpha=0.6)
    print(f"\n[{soul}] '{cross_query}':")
    for r in results[:2]:
        c = json.loads(r["content"]) if isinstance(r["content"], str) else r["content"]
        print(f"  [{r.get('score', 0):.3f}] {c.get('topic', 'N/A')[:70]}")
