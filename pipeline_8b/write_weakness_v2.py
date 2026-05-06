import sys, os
sys.path.insert(0, r"D:\VORTEX_FLAME")
os.chdir(r"D:\VORTEX_FLAME")
import long_memory as lm

stats_before = lm.get_stats("cezanne")
print(f"Before: {stats_before['total']} entries")

lm.write("cezanne", "weakness_7b",
    "7B Cezanne S2 weaknesses (benchmark 2026-05-03): CRITICAL debug=0% (cannot find off-by-one/infinite loop/boundary bugs in binary search). MODERATE math_even_proof=75% (missing derivation label). MODERATE formal_logic=25% (same as 8B, locked at implies only). Stage3b has 20 debug+13 logic+13 math supplements. Expected after S3: debug 60%+, math 90%+.")

lm.write("cezanne", "weakness_8b",
    "8B Cezanne_PRO S3 weaknesses (benchmark 2026-05-03): MODERATE Dijkstra=80% (missing shortest keyword). MODERATE prime=75% (missing infinite). MODERATE security=80% (lost prepared). CRITICAL formal_logic=25% (locked across all 3 stages). Final avg 82%. Stage4: security+math fix, formal logic may need Coq/Lean data.")

lm.write("cezanne", "arena_strategy",
    "7B vs 8B Arena strategy (2026-05-03): 7B wins sort/graph/security (precise implementation 100%). 8B wins debug/math (reasoning depth). Both tied on logic=25%. Mode2 Attack-Defense is highest value: 8B debugs 7B code (7B at 0% debug learns from 8B fixes). 7B precise code feeds back to 8B quality (sort/graph were 80%). Goal: mutual improvement to 85%+.")

lm.write("cezanne", "cross_learning",
    "Cross-model learning: 7B and 8B share same cezanne memory bank. 7B needs 8B's debug reasoning. 8B needs 7B's implementation precision. Arena generates training data for both. Each reads their own weaknesses here before inference.")

lm.refresh_all()

stats = lm.get_stats("cezanne")
print(f"After: {stats['total']} entries")
for c, n in sorted(stats['categories'].items()):
    print(f"  {c}: {n}")
print("Done writing weakness profiles")
