"""
7B vs 8B Training Data Strategy - CORRECTED
=============================================

WRONG approach (old):
  7B: short chains (3-5 steps) -> 7B gets handicapped -> unfair comparison
  8B: long chains (8-15 steps) -> 8B gets advantage -> proves nothing

CORRECT approach:
  7B: BEST data it can handle (5-8 steps, clear reasoning) -> 7B at full strength
  8B: SAME base data + EXTRA deep chains (8-15 steps) -> 8B pushes beyond 7B's limit

Key principle: Both models should be at their BEST for comparison.
Differentiation means 8B gets ADDITIONAL data that 7B can't handle,
NOT that 7B gets WORSE data.

Data structure:
  7B Stage3: existing data (avg 1489 chars, 5-8 step reasoning) = 7B's BEST
  8B Stage3: 7B's data (same) + 8B-exclusive ultra-deep data (15+ step proofs)
  
  The 8B-exclusive portion is what 7B CANNOT learn from,
  not what 7B is DENIED access to.

Benchmark fairness:
  - Same topics, same difficulty for shared portion
  - 8B has ADDITIONAL training on ultra-deep reasoning
  - If 8B still outperforms 7B, it's because 8B's architecture is better
  - If 7B matches 8B on shared topics, 7B is more efficient per parameter

VRAM constraint reality:
  - 7B can train with seq_len=512 comfortably
  - 8B can only train with seq_len=128-256 on 12GB
  - This means 8B actually sees LESS context per sample during training
  - But 8B has more parameters to model deeper reasoning patterns
  - The tradeoff: fewer tokens seen, but deeper understanding per token
"""
