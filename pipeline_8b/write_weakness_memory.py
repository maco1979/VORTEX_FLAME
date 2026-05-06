#!/usr/bin/env python3
"""Write 7B/8B weakness profiles to shared cezanne memory bank"""
import sys, os
sys.path.insert(0, r"D:\VORTEX_FLAME")
os.chdir(r"D:\VORTEX_FLAME")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import long_memory as lm

print("Writing weakness profiles to shared memory...")

# 1. 7B weaknesses
lm.write("cezanne", "weakness_7b",
    "7B Cezanne S2 weaknesses (benchmark 2026-05-03): "
    "[CRITICAL] debug=0% — cannot find ANY bugs in binary search (off-by-one, infinite loop, boundary). "
    "[MODERATE] math_even_proof=75% — missing 'derivation' label. "
    "[MODERATE] formal_logic=25% — same as 8B, locked at 'implies' only. "
    "Fix: Stage3b has 20 debug supplements + 13 logic + 13 math. "
    "Expected after Stage3: debug 60%+, math 90%+, logic may stay 25% (architecture limit).")

lm.write("cezanne", "weakness_7b_fix",
    "7B Stage3b supplements: 8 debug extras (race condition, ReDoS, concurrent cache, "
    "coin change recursion, matrix rotate, palindrome, majority element, search insert off-by-one). "
    "2 logic extras (truth table transitivity, De Morgan). "
    "2 math extras (odd square proof, derivative velocity). "
    "Total extra: 12 items on top of baseline 1456. "
    "Strategy: debug=0% is the #1 priority, 7B needs more debug training than 8B.")

# 2. 8B weaknesses
lm.write("cezanne", "weakness_8b",
    "8B Cezanne_PRO S3 weaknesses (benchmark 2026-05-03): "
    "[MODERATE] Dijkstra=80% — missing 'shortest' keyword. "
    "[MODERATE] prime_contradiction=75% — missing 'infinite'. "
    "[CRITICAL] formal_logic=25% — locked across all 3 stages. "
    "[MODERATE] security=80% — lost 'prepared' in S3. "
    "[OK] sort=100%, tree=100%, dp=100%, debug=80%, derivative=80%. "
    "Final avg: 82%. Stage4 should fix security+math without large retraining.")

lm.write("cezanne", "weakness_8b_fix",
    "8B Stage4 plan: security keyword supplements (~1000 items from hh-rlhf), "
    "math balance adjustment, long-context debug cases. "
    "Formal logic at 25% may be 8B architecture limit — try Coq/Lean proof data in Stage4 "
    "before concluding it's impossible.")

# 3. Mutual comparison
lm.write("cezanne", "arena_weakness_map",
    "7B vs 8B weakness map for Arena (2026-05-03): "
    "7B wins: sort=100>80, graph=100>80, security=100>80. "
    "8B wins: debug=20>0, math_even=100>75. "
    "Tied: logic=25%=25%, tree=100%=100%, dp=100%=100%. "
    "Arena Mode 2 (Attack-Defense) is highest value: 8B debugs 7B's code, 7B learns from fixes. "
    "7B's precise implementations feed back to 8B's code quality. "
    "Mutual weakness (logic 25%) needs external formal proof data, not Arena.")

lm.write("cezanne", "shared_learning_strategy",
    "Cross-model learning strategy: "
    "Both 7B and 8B read/write same memory bank (cezanne/). "
    "Training logs, benchmark results, weakness profiles are shared. "
    "When Arena runs, generated training data goes to both models. "
    "7B's bug fixes teach 8B edge case awareness. "
    "8B's debug reasoning teaches 7B systematic debugging. "
    "Goal: both models converge to 85%+ through mutual improvement.")

lm.refresh_all()

# Verify
stats = lm.get_stats("cezanne")
print(f"\nMemory bank updated. Cezanne now: {stats['total']} entries")
for cat, n in sorted(stats['categories'].items()):
    print(f"  {cat}: {n}")

# Test recall
result = lm.recall("cezanne", "what are my weaknesses?")
print(f"\nRecall test: 'what are my weaknesses?'")
for line in result.split("\n")[:5]:
    print(f"  {line[:120]}...")
