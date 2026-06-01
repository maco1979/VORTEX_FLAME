import sys, json, logging, time
sys.path.insert(0, r"D:\VORTEX_FLAME")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from soul_knowledge_alignment import align_soul

t0 = time.time()
result = align_soul("beethoven")
elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"Beethoven re-alignment result ({elapsed:.1f}s):")
print(json.dumps(result, indent=2, ensure_ascii=False))
